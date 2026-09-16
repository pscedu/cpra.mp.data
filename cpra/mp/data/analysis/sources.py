"""A Polars IO plugin exposing windowed raster reads as a LazyFrame.

Row model is WIDE -- one row per pixel, one column per band -- which is the only
representation measured to hold a bounded memory profile at full raster extent (a
row-per-pixel-per-band layout peaked far higher on the same data).

The source owns the spatial part (reading the zone-ID crosswalk raster and dropping
pixels outside the summarized zones); everything downstream (grouping, aggregating) is
ordinary Polars, so it runs on the streaming engine and can spill.

See ``polars.io.plugins.register_io_source`` for the IO-plugin contract this implements:
https://docs.pola.rs/user-guide/plugins/io_plugins/
"""

import itertools
from typing import NamedTuple

import numpy as np
import polars as pl
import rasterio as rio
from polars.io.plugins import register_io_source
from rasterio import windows

from cpra.mp.data.analysis.expressions import (
    count_by_value_frame,
    parse_value_expr,
    rewrite_value_column,
    value_predicate_mask,
)
from cpra.mp.data.analysis.utils import raster_blocks_per_batch, raster_gdal_env

NUMPY_TO_POLARS = {
    "int8": pl.Int8,
    "uint8": pl.UInt8,
    "int16": pl.Int16,
    "uint16": pl.UInt16,
    "int32": pl.Int32,
    "uint32": pl.UInt32,
    "int64": pl.Int64,
    "float32": pl.Float32,
    "float64": pl.Float64,
}

# The zonal summarize aggregates. The Float64 cast on `sum`/`mean` is deliberate: Polars
# would otherwise accumulate a Float32 column in Float32, which measured ~1.5 absolute off
# from accumulating in float64 over 10M pixels in an earlier round. `min`/`max` need no
# cast -- they never accumulate.
#
# `count` exists only for the plugin-routed conditional-count query ("how many pixels in
# this zone satisfy X"): a value-where mask (see expressions.value_predicate_mask /
# masked_value_predicate_exprs below) nulls out non-matching pixels, and `count()` is
# Polars' own null-skipping row-count, so "count contributing pixels" and "count pixels
# the predicate kept" are the same operation with no extra code. It is deliberately absent
# from `HORIZONTAL_RASTER_AGGREGATES` below -- see that dict's docstring.
POLARS_RASTER_AGGREGATES = {
    "count": lambda c: c.count(),
    "sum": lambda c: c.cast(pl.Float64).sum(),
    "mean": lambda c: c.cast(pl.Float64).mean(),
    "min": lambda c: c.min(),
    "max": lambda c: c.max(),
}


# Horizontal counterparts to `POLARS_RASTER_AGGREGATES` above. These reduce ACROSS a row's
# selected band columns (dataset.py's per-band calculate + group_aggregate collapse), not
# DOWN a `group_by`'s rows the way the zonal summarize aggregates above do -- band-collapse and
# zonal summarize are different axes of the same per-band pipeline and are kept in a
# separate dict so the two are never reached for each other by mistake. Shared by
# `raster.summarize_zone_stats` (pre-collapsing a group before the vertical zonal
# `guarded_aggregate`) and `Dataset._write_raster_windowed` (collapsing a group into one
# output band) -- the one shared expression vocabulary both paths use for group_aggregate.
#
# Deliberately has no `count` key, unlike `POLARS_RASTER_AGGREGATES` above -- see
# `types.AggregateFunction.count`'s comment for why a flat-API raster `group_aggregate`
# rejects it outright (`dataset.py`'s own `group_aggregate == "count"` check) rather than
# giving it a meaning here. A PAIRED analysis pipeline's `group.aggregate == "count"`
# ("how many calendar steps satisfied the condition") is a DIFFERENT question that a
# `where` clause makes meaningful -- `pipeline_exec.PairedAnalysis` implements that locally
# (`_band_valid_expr`/`_collapse_expr`'s own `count` branch), never by adding a key here.
# Do not "unify" the two by adding `count` to this dict.
HORIZONTAL_RASTER_AGGREGATES = {
    "max": pl.max_horizontal,
    "mean": pl.mean_horizontal,
    "min": pl.min_horizontal,
    "sum": pl.sum_horizontal,
}


def horizontal_aggregate_expr(aggregate, exprs, no_contributor):
    """One `group_aggregate` collapse across `exprs`, with the all-null-group edge case
    resolved explicitly rather than left to each aggregate's own default.

    `exprs` are the (already nodata-masked-to-null, already-multiplied) column expressions
    for one output group's contributing bands -- masking must happen upstream of this call,
    against each band's own raw pixel value, since multiplying first would make a scaled
    nodata sentinel stop matching the sentinel it started as. Bare `*_horizontal` already
    skips nulls on its own (that is what makes a masked-out band drop out of the reduction
    for free), but the four aggregates disagree about what an all-null row should produce:
    `min`/`max`/`mean` give null, `sum` gives 0 (its additive identity) -- neither is
    necessarily the answer a caller wants for "no band contributed anything to this pixel
    at all". `no_contributor` is that answer, supplied by the caller as a Polars expression
    (typically `pl.lit(...)`) rather than a raw value, so the caller controls its dtype
    explicitly: `pl.lit(None)` reproduces the summarize path's desired null-propagation (so
    the downstream vertical `guarded_aggregate` sees an empty group rather than a
    fabricated 0), while `pl.lit(nodata, dtype=...)` matched to the raster's own dtype is
    what the output writer needs, since an integer output band cannot hold NaN the way the
    summarize path's Float64 result can.

    The guard asks whether *every* band is null rather than counting how many are not. The
    two are equivalent here, but counting is materially more expensive: it needs a boolean
    per band, a widening cast per band, and a running integer accumulator across all of
    them, only for the total to be compared against zero and discarded. `all_horizontal`
    over the null masks needs the booleans alone. Measured on the collapse expression at
    tile and half-tile widths, min of five interleaved rounds, the difference is 1.4-1.5x on
    the whole expression -- worth the wording because this runs once per output pixel per
    group, on the box that also has to serve tiles.
    """
    agg_fn = HORIZONTAL_RASTER_AGGREGATES[aggregate]
    no_contributor_at = pl.all_horizontal([e.is_null() for e in exprs])
    return pl.when(no_contributor_at).then(no_contributor).otherwise(agg_fn(exprs))


def band_calculate_passes(raw_names, plans, applicability):
    """The calculate stage's entries, applied per band, as sequential `with_columns`
    passes -- one pass per entry (a `list[pl.Expr]`, each aliased back onto its own band
    column), in entry order, so entry N+1 sees entry N's output: the documented
    sequential-composition contract, expressed structurally rather than by nesting
    expressions.

    `plans` are `expressions.CalculatePlan`s; `applicability[k][i]` says whether plan
    `k`'s dimension `where` matched the band-metadata row behind `raw_names[i]` (the
    caller evaluates that against its own rows -- see `raster.apply_calculate_band_wide`,
    the only caller). A plan's `value_where`, being per-pixel,
    is folded into each band's own rewrite condition here instead, rewritten per band by
    `rewrite_value_column` exactly like a filter value condition. A band this plan does
    not apply to is simply left out of that pass -- pass-through, never dropped or
    masked.

    The `is_not_null()` guard on every rewrite makes "NoData in, NoData out" a structural
    property rather than a per-expression obligation: a nodata pixel (already masked to
    null by the caller -- `scan_rasters(mask_nodata=True)`, or the nodata pass
    `raster.apply_band_ops` prepends) never matches, so even an expression that COULD
    backfill (`coalesce(value, 0)` is whitelisted) only ever runs on real values.
    """
    passes = []
    for plan, applies in zip(plans, applicability):
        exprs = []
        for name, applicable in zip(raw_names, applies):
            if not applicable:
                continue
            condition = pl.col(name).is_not_null()
            if plan.value_where is not None:
                condition = condition & value_predicate_mask(
                    rewrite_value_column(plan.value_where, name), allowed_columns=[name]
                )
            new_value = parse_value_expr(
                rewrite_value_column(plan.expression, name), allowed_columns=[name]
            )
            exprs.append(pl.when(condition).then(new_value).otherwise(pl.col(name)).alias(name))
        if exprs:
            passes.append(exprs)
    return passes


def guarded_aggregate(aggregate, name):
    """A `group_by().agg()` expression for one column, with the all-nodata edge case fixed.

    Bare Polars returns `null` for `min`/`max`/`mean` on an all-null group, and `0` (the
    additive identity) for `sum` -- neither is a meaningful answer for a (band, zone) with
    no contributing pixel at all, which `RasterGridAnalysis._apply_transform_params_impl` needs reported as
    `NaN` instead. `count()` already skips nulls, so it is exactly the contributing-pixel
    count on its own, but the guard below is still needed for all four aggregates, not
    just `sum`, to turn "no contributing pixel" into `NaN` consistently.

    `count` itself is deliberately exempt from the guard: unlike a mean or a sum, "zero
    contributing pixels" is `count`'s own correct answer, not an edge case standing in for
    an undefined one -- a zone with no pixels satisfying a value-where predicate really did
    have zero of them, and reporting that as `NaN` would turn a real, meaningful zero into
    a missing-value marker.
    """
    if aggregate == "count":
        return pl.col(name).count().alias(name)

    agg_fn = POLARS_RASTER_AGGREGATES[aggregate]
    c = pl.col(name)
    return (
        pl.when(c.count() == 0)
        .then(pl.lit(float("nan"), dtype=pl.Float64))
        .otherwise(agg_fn(c))
        .alias(name)
    )


def masked_value_predicate_exprs(raw_names, expr_str, extra_columns=()):
    """Per-band value-where masking: null out a band's own pixel wherever `expr_str`
    (written once, against the placeholder column `value`) evaluates false for it.

    Returns one expression per name in `raw_names`, each aliased back to that same name,
    meant for a `with_columns` call on the plugin's output frame -- so the caller gets
    back a frame with the identical shape and column names it started with, just with
    some values replaced by null. That is deliberate: masking to null (rather than
    filtering rows) is what lets a value-where predicate compose with the row-wide band
    layout (one row is one pixel, N bands) without a match on one band's predicate
    dropping its sibling bands' values too, and it is what lets a masked-out value fall
    into the SAME null-skipping aggregate machinery a masked-out nodata pixel already
    does (`guarded_aggregate`, `POLARS_RASTER_AGGREGATES`) -- no separate "predicate
    failed" code path.

    `expr_str` is rewritten per band via `rewrite_value_column` (the placeholder becomes
    that band's own synthetic column name), then parsed and whitelisted independently by
    `value_predicate_mask` -- each band's namespace is exactly `{that band, *extra_columns}`,
    never every band at once, so a predicate cannot accidentally (or intentionally) compare
    one band against another. `extra_columns` exists for `grid_cell`, when a caller also
    wants a value-where predicate to be allowed to reference it (e.g. "value > x" scoped by
    zone) -- empty by default, since most callers don't need it.
    """
    exprs = []
    for name in raw_names:
        mask = value_predicate_mask(rewrite_value_column(expr_str, name), allowed_columns=[name, *extra_columns])
        exprs.append(pl.when(mask).then(pl.col(name)).otherwise(None).alias(name))
    return exprs


def count_by_value(paths, row_indices, bands, crosswalk_path, zone_ids, value_where=None, max_distinct=None):
    """Per-zone categorical pixel count-by-value: one row per (zone, distinct pixel
    value), via `scan_rasters` + `expressions.count_by_value_frame`.

    Single-band only. `paths`/`row_indices`/`bands` must together describe exactly one
    selected band -- `RasterGridAnalysis._apply_transform_params_impl` enforces this before ever
    calling in (via its already-computed `n_bands`), since a multi-band request has no
    single natural (zone, value) shape: counting each band's values into the same `value`
    column would silently conflate unrelated variables/years, and counting them into
    separate columns would need a different output shape per band count. A silently wrong
    multi-band result is not acceptable, so this is refused rather than guessed at --
    re-asserted here defensively rather than only trusted from the caller.

    `value_where`, when given, is applied the same way `summarize_zone_stats` (the
    other summarize_aggregate path) applies it: masked to null against the (single) band's
    own raw pixel value, before counting -- so a pixel the predicate rejects is excluded
    from the category counts exactly like a masked nodata pixel already is (mask first,
    then count distinct survivors).
    """
    raw_names = [f"b{i}" for indices in row_indices for i in indices]
    if len(raw_names) != 1:
        raise ValueError(f"count_by_value supports exactly one raster band, got {len(raw_names)}")
    value_col = raw_names[0]

    lf = scan_rasters(paths, row_indices, bands, crosswalk_path, zone_ids, mask_nodata=True)
    if value_where is not None:
        lf = lf.with_columns(masked_value_predicate_exprs(raw_names, value_where, extra_columns=("grid_cell",)))

    counted = count_by_value_frame(lf, ["grid_cell"], value_col, max_distinct).collect(engine="streaming")
    return counted.rename({value_col: "value"})


def _raster_dtype(path):
    """Polars dtype matching a COG's stored band dtype.

    Long-format rasters can be integer (e.g. elevation, stored as Int32 millimetres --
    see cpra/mp/data/variables.py:38, where the mm-to-m conversion is presentation-only).
    The schema must follow the file rather than assume float, or Polars raises a
    SchemaError the moment a batch arrives with a narrower dtype than declared.
    """
    with rio.open(str(path)) as src:
        return NUMPY_TO_POLARS[src.dtypes[0]]


def _zone_keep_positions(zone_ids, zones):
    """Row positions in `zones` (a flat, per-pixel zone-id array read from a crosswalk
    window) that fall inside `zone_ids` -- the requested destination zone set.

    Shared by every source below that drops pixels outside a requested zone set
    (`scan_rasters`'s zoned mode and `scan_broadcast`'s output-zone filter): a pixel whose
    crosswalk value is not one of `zone_ids` -- including the crosswalk's nodata sentinel,
    which is never itself a requested zone id -- is dropped by construction, with no
    separate nodata branch needed here. `zone_ids` must be sorted (each call site's caller
    already sorts its own zone id array once, not per window).
    """
    positions = np.searchsorted(zone_ids, zones).clip(max=zone_ids.size - 1)
    return np.flatnonzero(zone_ids[positions] == zones)


def _dense_lookup_array(ids, values):
    """A dense NumPy array indexed by raw id, `np.nan` everywhere except `ids`.

    Shared by `scan_rasters`' crosswalk value lookup, by `scan_broadcast` (below), and by
    `paired.py`'s unported tile path. `ids` must be sorted for the caller's own
    bookkeeping (this function itself only needs `ids` and `values` aligned pairwise --
    sortedness is not used here, only by the callers that build these arrays from a
    sorted vector-grid selection). An id that is in-range but was never one of `ids`
    (absent from the vector frame) resolves to `nan`, indistinguishable from a lookup
    miss, with no separate branch. Duplicate ids resolve last-wins, NumPy's own rule for
    a fancy-index assignment.
    """
    max_id = int(ids.max()) if ids.size else -1
    lookup = np.full(max_id + 1, np.nan, dtype="float64")
    lookup[ids] = values
    return lookup


def _crosswalk_lookup_values(lookup_array, ids, valid):
    """Per-pixel values for one band's crosswalk value column, via that column's dense
    lookup array (see `_dense_lookup_array`).

    `valid` has already folded in the crosswalk raster's own nodata check (see
    `scan_rasters`' ``source``); this function adds the remaining two ways a pixel can miss:
    the id is out of the lookup array's range entirely (bigger than any vector row's id --
    `lookup_array is None` covers "no vector rows at all", the same treatment), or the id is
    in-range but was never one of the vector frame's own rows, which `_dense_lookup_array`
    already resolves to `nan`. Every miss kind converges on the same `nan` return value,
    which the caller turns into a true Polars null with one `fill_nan(None)` -- deliberately
    NOT dropped the way `scan_broadcast` drops an equivalent miss, since the row must
    survive for a predicate spanning both a band column and a value column to evaluate at
    all, and for a raster-output write to keep one row per pixel.
    """
    if lookup_array is None or lookup_array.size == 0:
        return np.full(ids.shape, np.nan)
    within = valid & (ids >= 0) & (ids < lookup_array.size)
    out = np.full(ids.shape, np.nan)
    if within.any():
        out[within] = lookup_array[ids[within]]
    return out


def _batched_windows(src, blocks_per_batch):
    """Window unions over one block row at a time, chunked to `blocks_per_batch` blocks.

    Mirrors `Dataset._batched_block_windows`: a block row can be tens of thousands of
    pixels wide regardless of how much data a request needs, so unioning a whole row per
    batch would make every batch's buffer scale with the raster's width and the number of
    bands selected rather than with `blocks_per_batch`. At full raster extent, whole-block-
    row batches measured a peak RSS high enough to exceed a 15 GB VM; chunking is a
    prerequisite for this source, not an optional tuning knob.
    """
    groups = itertools.groupby(src.block_windows(1), key=lambda bw: bw[0][0])
    for _row, group in groups:
        row_windows = [w for _ji, w in group]
        for i in range(0, len(row_windows), blocks_per_batch):
            yield windows.union(*row_windows[i : i + blocks_per_batch])


class GridCrosswalk(NamedTuple):
    """One vector grid's crosswalk raster, attached to a `scan_rasters` scan.

    `name` is the ZONE column this crosswalk emits -- ONE column, not one per band: a
    crosswalk raster answers "which zone of that vector grid is this pixel in", which is a
    property of the pixel alone and is therefore shared by every band column in the row.

    `path` is a crosswalk raster co-registered with the scanned rasters' own grid, so it is
    read with the same window and needs no resampling. The same file may back two
    `GridCrosswalk`s under different names; nothing here assumes distinct files.

    `values` is optional. When given it is a BAND-WIDE value table produced by
    `raster.pivot_vector_to_band_wide`: one row per zone, a key column named `name`, and
    one Float64 column named ``f"{name}{i}"`` per scanned band row-position ``i`` (the same
    ``i`` that names ``b<i>``). The scan then emits those columns too, each pixel taking
    its own zone's value. Splitting it this way keeps the two genuinely different jobs
    apart: turning a LONG vector frame (one row per zone per calendar step) into one column
    per band is an ordinary frame-level pivot with no raster in it, and is done once, up
    front, outside this module; what remains here is a per-pixel positional lookup, which
    is all a raster source can do.

    Passing the pivoted table INTO the scan, rather than joining it onto the scan's output,
    is forced rather than chosen -- see `scan_rasters`' own note on batch ordering. Without
    that constraint this would be a `join` on `name` downstream of the scan; with it, the
    lookup has to happen where row order is still guaranteed by construction.
    """

    name: str
    path: object
    values: object = None


def scan_rasters(
    paths,
    row_indices,
    bands,
    zones_path=None,
    zone_ids=None,
    value_dtype=None,
    mask_nodata=True,
    crosswalks=(),
    with_px=False,
    read_windows=None,
):
    """Return a LazyFrame of (grid_cell, <crosswalk columns>, <band columns>) over one or
    more multiband COGs.

    ``paths``, ``row_indices`` and ``bands`` are exactly `group_rows_by_path`'s return
    values -- one dataset per unique file backing a request, with the caller's own row
    positions and the (1-based) band numbers each file supplies. Column names are
    synthetic (``b<row index>``), tied to those row positions rather than to on-disk band
    descriptions, so a caller can map result columns straight back to its own dimension
    rows without depending on metadata a source file may not carry.

    ``zones_path``/``zone_ids`` are now two INDEPENDENT knobs rather than one coupled
    pair: scanning bands windowed is split from joining zone IDs, so the latter can be
    the band-wide lowering of `project` instead of being baked into the scan.

    - ``zones_path=None`` (`zone_ids` must then also be `None` -- see the guard below):
      the original dense, one-row-per-pixel shape -- every pixel, in raster order, no
      crosswalk read and no `grid_cell` column at all. Unchanged from before this split.
    - ``zones_path`` given, ``zone_ids=None``: NEW -- attach a `grid_cell` column
      (this window's own zone-crosswalk pixel value, read positionally alongside the
      band data) to EVERY pixel, dropping none. Still the dense, position-preserving
      shape (pixel count in == pixel count out), just carrying an extra column. This is
      the "join zone IDs" half of `project`'s band-wide lowering: `scan_rasters` only
      ATTACHES the raw zone id here; deciding which zones survive and collapsing them
      is an ordinary downstream `filter`+`group_by`, so `project` can run at whatever
      position a pipeline declares it, rather than always being folded into the initial
      scan. (`raster.apply_project_band_wide` used to be that downstream consumer; it was
      deleted as unused -- the raster interpreter reaches a summarizing `project` through
      `_apply_transform_params_impl`/`summarize_zone_stats` instead, which use the
      zoned/filtered mode below. This attach-only mode is still exercised directly by
      `tests/api/test_raster_ops.py::TestScanRastersAttachWithoutFilter`.)
    - ``zones_path`` and ``zone_ids`` both given: the original zoned/TABULAR shape --
      a `grid_cell` column holds the destination zone ID, and pixels whose zone is not
      in `zone_ids` (including crosswalk nodata) are DROPPED in the source. Unchanged:
      this is still what `raster.summarize_zone_stats`/`count_by_value` use,
      and dropping here (rather than in a downstream `filter`) remains correct for them
      specifically because their output is immediately `group_by`'d -- row COUNT was
      never part of their contract, only group membership -- unlike a raster-output
      query, whose pixel count is its contract.

    ``zone_ids`` without ``zones_path`` is rejected: there is no crosswalk to source a
    zone ID from.

    ``crosswalks`` is a sequence of `GridCrosswalk`, the SAME positional read generalized
    to any number of vector grids. Each emits one Int32 zone column named by its own
    `GridCrosswalk.name`, holding that crosswalk raster's own pixel value; nothing is
    filtered and no row is dropped. A crosswalk carrying a pivoted ``values`` table
    additionally emits one Float64 column ``f"{name}{i}"`` per band row-position ``i``,
    holding that pixel's zone's value for that band -- an O(1) dense-array lookup
    (`_dense_lookup_array`), not a hash join, for the same reason `scan_broadcast`
    documents: the key range is small and dense, so the array wins outright.

    Every emitted name -- zone columns and value columns alike -- must be unique across
    crosswalks and must not collide with `grid_cell`, `_px`, or a `b<i>` band column. These
    are silent column-overwrite risks in the per-window dict merge below, not hypothetical
    tidiness, so they are rejected at scan-construction time.

    WHY THE VALUE LOOKUP LIVES IN THE SOURCE. On its face this belongs downstream: attach
    the zone id here, `join` the pivoted table on it there, and the scan stays a scan. That
    is not available on this Polars version. `collect_batches(maintain_order=True)` over a
    `register_io_source` plugin does NOT preserve batch order once the plan contains a
    `join`, `replace_strict`, `is_in`, or `gather` -- verified on an arithmetic-only control
    (order preserved) against each of those four (order lost, batches cyclically rotated and
    not reproducibly so). Pure arithmetic and `when/then` are safe; `engine="in-memory"` is
    also safe but materializes the whole source before yielding its first batch, which
    defeats a bounded-memory raster write. A raster-output write is positional -- row `k` of
    the stream IS pixel `k` -- so losing batch order is a silent wrong answer, not a slow
    one. Doing the lookup here keeps the whole plan inside the arithmetic-and-`when/then`
    subset that does hold order. The PIVOT still happens outside, as an ordinary frame op
    (`raster.pivot_vector_to_band_wide`); only the per-pixel lookup is pulled in.

    A crosswalk's ZONE column nodata sentinel becomes NULL, unconditionally -- NOT gated on
    ``mask_nodata``, which governs the BAND columns' value sentinels. The two are different
    questions: a band's nodata means "no measurement here", while a crosswalk's means "this
    pixel is outside that vector grid's domain entirely". Null is what lets a consumer
    distinguish the two; it is also what the nodata-as-invalid test below keys the value
    lookup off, so a pixel outside the vector grid's domain gets a null value rather than
    whatever zone's value the raw sentinel happened to index.

    A VALUE column is null on any miss, and the row always SURVIVES: crosswalk nodata, a
    zone id out of the pivoted table's range, a zone id in range but absent from the table,
    and a band whose calendar step had no vector rows at all all converge on null. The row
    is deliberately kept -- unlike `scan_broadcast`, which drops an equivalent miss --
    because a predicate spanning a band column and a value column has to be able to see the
    band side of a pixel whose value side missed.

    ``with_px=True`` attaches ``_px``, a ROW-MAJOR pixel index over the FULL raster extent
    (``row * width + col``, not window-relative and not renumbered by a zone filter). It
    costs one `arange`-and-broadcast per window -- no file read -- so, like `grid_cell`, it
    is computed wholesale rather than gated on projection pushdown once requested.

    ``read_windows`` restricts the scan to a caller-supplied list of `rasterio.windows.
    Window`, in the order given, instead of tiling the whole raster into batched blocks.
    It is a SPATIAL restriction, not a row limit (`n_rows` is that): the rows emitted are
    exactly the pixels of those windows, concatenated, and every column still means what it
    means everywhere else -- ``_px`` in particular stays an index over the FULL raster
    extent, so a restricted scan's pixel indices line up with an unrestricted one's.

    This is what lets a MAP TILE be read through the same scan a full-raster write uses
    (`paired._paired_notebook_tile` passes the one window `utils.tile_source_window`
    computed for the tile's footprint) rather than through a parallel implementation.
    Windows are NOT re-chunked to the batch budget -- a caller that hands over a window
    bigger than one batch's worth of pixels gets it in one batch -- because a caller
    specific enough to name its own windows has already decided how much it wants in
    memory at once, and splitting them would break the "row `k` of the stream is pixel `k`
    of window `w`" correspondence that a positional consumer reassembles from.

    ``value_dtype=None`` infers the schema from `paths[0]` -- see `_raster_dtype`.

    ``mask_nodata=True`` maps each BAND column's stored nodata sentinel (read per-file, so
    multiple sources may declare different values) to null before any aggregate sees it;
    a file with no declared nodata (`src.nodata is None`) is left unmasked -- such a raster
    is treated as never having nodata at all.
    """
    if not paths:
        raise ValueError("scan_rasters requires at least one source path")
    if zone_ids is not None and zones_path is None:
        raise ValueError("scan_rasters: zone_ids requires zones_path")
    zones_path = str(zones_path) if zones_path is not None else None
    attach_zone = zones_path is not None
    filter_to_zones = zone_ids is not None

    names_by_path = [[f"b{i}" for i in indices] for indices in row_indices]
    bands_by_path = [
        list(path_bands) if path_bands is not None else list(range(1, len(names) + 1))
        for path_bands, names in zip(bands, names_by_path)
    ]
    all_names = [n for names in names_by_path for n in names]
    if value_dtype is None:
        value_dtype = _raster_dtype(paths[0])

    crosswalks = [cw._replace(path=str(cw.path)) for cw in crosswalks]
    crosswalk_names = [cw.name for cw in crosswalks]

    # Value column names, one row-position-indexed list per path, shaped exactly like
    # `names_by_path` above so projection pushdown can be computed the same way for either.
    # A crosswalk with no `values` table contributes none, and is then attach-only.
    value_names_by_crosswalk = {
        cw.name: [[f"{cw.name}{i}" for i in indices] for indices in row_indices] if cw.values is not None else []
        for cw in crosswalks
    }
    all_value_names = [n for per_path in value_names_by_crosswalk.values() for names in per_path for n in names]

    emitted = crosswalk_names + all_value_names
    if len(set(emitted)) != len(emitted):
        raise ValueError("scan_rasters: crosswalk names must be unique")
    colliding = set(emitted) & (set(all_names) | {"grid_cell", "_px"})
    if colliding:
        raise ValueError(f"scan_rasters: crosswalk name collides with a reserved column name: {sorted(colliding)}")
    crosswalk_name_set = set(emitted)

    # Dense lookup arrays are built ONCE here, at scan construction -- not per window --
    # the same reasoning `scan_broadcast` documents for why this is a lookup and not a
    # `.join()`. A value column whose pivoted source column is all-null (its calendar step
    # contributed no vector rows) still gets an array; every id then misses and resolves to
    # `nan`, so "no rows for this step" needs no branch of its own downstream.
    lookup_arrays = {}
    for cw in crosswalks:
        if cw.values is None:
            continue
        missing = [n for per_path in value_names_by_crosswalk[cw.name] for n in per_path if n not in cw.values.columns]
        if cw.name not in cw.values.columns or missing:
            raise ValueError(
                f"scan_rasters: crosswalk {cw.name!r} values table is missing columns: "
                f"{sorted(set(missing) | ({cw.name} if cw.name not in cw.values.columns else set()))}"
            )
        ids = cw.values[cw.name].to_numpy()
        for per_path in value_names_by_crosswalk[cw.name]:
            for n in per_path:
                lookup_arrays[n] = _dense_lookup_array(ids, cw.values[n].to_numpy())

    cols_schema = {name: value_dtype for name in all_names}
    schema = pl.Schema(
        {
            **({"grid_cell": pl.Int32} if attach_zone else {}),
            **({"_px": pl.Int64} if with_px else {}),
            **{name: pl.Int32 for name in crosswalk_names},
            **cols_schema,
            **{name: pl.Float64 for name in all_value_names},
        }
    )

    def source(with_columns, predicate, n_rows, batch_size):
        # --- projection pushdown: only decompress the bands, and only open the crosswalk
        # rasters, the query asks for. `grid_cell`/`_px` are excluded from THIS decision
        # (neither one costs a band decompression), and are gated separately below on
        # whether the projection actually named them. ---
        wanted = (
            set(all_names) | crosswalk_name_set
            if with_columns is None
            else {c for c in with_columns if c not in ("grid_cell", "_px")}
        )

        # A file none of whose bands are wanted is never opened at all.
        plan = []
        for path, names, path_bands in zip(paths, names_by_path, bands_by_path):
            picked = [(n, b) for n, b in zip(names, path_bands) if n in wanted]
            if picked:
                plan.append((str(path), [n for n, _ in picked], [b for _, b in picked]))

        # A crosswalk raster is opened if its zone column OR any of its value columns is
        # wanted -- one read per crosswalk per window serves all of them, so a query
        # selecting several value columns from one crosswalk never re-reads the file.
        crosswalk_plan = []
        for cw in crosswalks:
            specs = [n for per_path in value_names_by_crosswalk[cw.name] for n in per_path if n in wanted]
            if cw.name in wanted or specs:
                crosswalk_plan.append((cw, cw.name in wanted, specs))

        # `grid_cell`/`_px` alone are still rows: a query selecting only the attached zone
        # id, or only the pixel index, has to get one row per pixel rather than an empty
        # frame -- neither opens a band file, so the `plan`/`crosswalk_plan` emptiness test
        # above cannot stand in for "nothing was asked for".
        requested = set(schema.names()) if with_columns is None else set(with_columns)
        if not plan and not crosswalk_plan and not (requested & {"grid_cell", "_px"}):
            return

        # Emit EXACTLY the projected columns, never a superset. `grid_cell` and `_px` are
        # cheap enough to compute unconditionally, and it is tempting to hand them over
        # whether or not they were asked for -- but a frame carrying a column the
        # projection did not request makes `collect_batches(maintain_order=True)` return
        # this source's batches OUT OF RASTER ORDER on a multi-threaded pool (measured
        # with `pl.thread_pool_size() == 8`; `POLARS_MAX_THREADS=1` hides it). Every
        # raster-output write path reassembles windows positionally from those batches,
        # so an out-of-order batch writes real values to the wrong pixels -- silently,
        # with the right value distribution. The band and crosswalk columns are already
        # gated on `wanted` above; these two are the only ones that were not.
        attach_zone_col = attach_zone and "grid_cell" in requested
        attach_px_col = with_px and "_px" in requested

        remaining = n_rows
        n_selected_bands = sum(len(names) for _p, names, _b in plan)

        # The window list -- and the batch size it's chunked to -- are derived once, from
        # the first source surviving projection pushdown, so a narrow band selection gets
        # a correspondingly larger batch rather than inheriting a size tuned for the full
        # band count. Uses the shared `raster_blocks_per_batch` budget/knob (see
        # config.RASTER_BATCH_BUFFER_MB) rather than a plugin-only setting. A band file is
        # the reference when one is open (its tiling is what governs decompression cost);
        # failing that the zone crosswalk, failing that the first surviving
        # `GridCrosswalk` -- all co-registered, so any one gives the same block geometry.
        #
        # Each selected VALUE column's per-pixel footprint is a Float64 lookup result plus
        # a same-shape id/validity intermediate (mirroring `_BROADCAST_ITEMSIZE_INFLATION`'s
        # reasoning for `scan_broadcast`), charged as 2 units at an 8-byte floor per COLUMN
        # rather than per crosswalk raster opened -- an intentional over-estimate when
        # several columns share one crosswalk file, which only makes this heuristic more
        # conservative, never less. An attach-only crosswalk (no `values`) contributes no
        # units: its Int32 id array is the only thing it costs.
        if plan:
            ref_path = plan[0][0]
        elif attach_zone:
            ref_path = zones_path
        elif crosswalk_plan:
            ref_path = crosswalk_plan[0][0].path
        else:
            ref_path = str(paths[0])
        n_selected_values = sum(len(specs) for _cw, _want_zone, specs in crosswalk_plan)
        with raster_gdal_env(), rio.open(ref_path) as first_src:
            block_height, block_width = first_src.block_shapes[0]
            raster_width = first_src.width
            if read_windows is not None:
                # The caller named its own windows; the block/batch budget below has
                # nothing left to decide. `raster_width` above is still the FULL raster's,
                # which is what `_px` is indexed against.
                window_list = list(read_windows)
            else:
                itemsize = np.dtype(first_src.dtypes[0]).itemsize
                if n_selected_values:
                    itemsize = max(itemsize, 8)
                n_units = n_selected_bands + 2 * n_selected_values
                blocks_per_batch = raster_blocks_per_batch(max(n_units, 1), block_height, block_width, itemsize)
                window_list = list(_batched_windows(first_src, blocks_per_batch))

        for window in window_list:
            if remaining is not None and remaining <= 0:
                break

            # Everything GDAL-related for this batch -- entering the env, opening the
            # crosswalk and data files, and reading them -- happens inside this `with`
            # block and nothing crosses the `yield` below. Polars' streaming engine may
            # resume this generator from a different thread than the one that entered it
            # on each call (confirmed on this Polars version: holding a `rio.Env` and
            # dataset handles open across a `yield` here raises
            # `EnvError: No GDAL environment exists` the moment the engine resumes), and
            # both `rio.Env` and rasterio dataset handles are thread-local, so nothing
            # opened before a `yield` may be assumed valid after it. Re-entering the env
            # and reopening datasets per batch is a hard constraint of the streaming
            # engine, not an implementation shortcut -- the reopen cost is included in
            # every measured number for this source.
            #
            # Separately (see `raster_gdal_env`'s docstring): rasterio applies GDAL config
            # process-globally when entered from the main thread and thread-locally
            # otherwise. This generator runs on a Polars worker thread, never the main
            # thread, so entering the env here takes the thread-local path and cannot leak
            # into a concurrent request the way entering it on the calling coroutine could.
            with raster_gdal_env():
                if not attach_zone:
                    keep = None
                    zones = None
                else:
                    with rio.open(zones_path) as zsrc:
                        zones = zsrc.read(1, window=window).ravel()

                    if filter_to_zones:
                        # Source-level spatial filter: keep only pixels in a
                        # summarized zone (the original zoned-mode behavior, unchanged).
                        keep = _zone_keep_positions(zone_ids, zones)
                        if keep.size == 0:
                            continue
                    else:
                        # NEW: attach every pixel's own zone id, dropping nothing --
                        # the "join zone IDs" half of `project`'s band-wide lowering,
                        # split from the "filter to these specific destination zones"
                        # half so a downstream op can do the latter at its own
                        # declared position instead of it being baked into this scan.
                        keep = None

                # One read per attached crosswalk, with this batch's own window -- the
                # same positional read `zones_path` does above, just not privileged.
                # Subset by `keep` alongside everything else so a zoned scan and an
                # attached crosswalk stay row-aligned.
                crosswalk_cols = {}
                crosswalk_nodata = {}
                value_cols = {}
                for cw, want_zone, specs in crosswalk_plan:
                    with rio.open(cw.path) as csrc:
                        cw_flat = csrc.read(1, window=window).ravel()
                        cw_nodata = csrc.nodata
                    cw_ids = (cw_flat if keep is None else cw_flat[keep]).astype("int32")
                    if want_zone:
                        crosswalk_cols[cw.name] = cw_ids
                        crosswalk_nodata[cw.name] = cw_nodata
                    if specs:
                        valid = np.ones(cw_ids.shape, dtype=bool) if cw_nodata is None else (cw_ids != cw_nodata)
                        for name in specs:
                            value_cols[name] = _crosswalk_lookup_values(lookup_arrays[name], cw_ids, valid)
                    del cw_flat, cw_ids

                cols = {}
                nodata_by_col = {}
                for path, names, path_bands in plan:
                    with rio.open(path) as src:
                        data = src.read(path_bands, window=window)
                        nodata = src.nodata
                    for i, name in enumerate(names):
                        flat = data[i].ravel()
                        cols[name] = flat if keep is None else flat[keep]
                        nodata_by_col[name] = nodata
                    del data

            leading = {}
            if attach_zone_col:
                zone_col = zones if keep is None else zones[keep]
                leading["grid_cell"] = zone_col.astype("int32")
            if attach_px_col:
                # Row-major index over the FULL raster, not the window: `window.row_off +
                # local row` times the raster's own width. Derived from the window's own
                # offsets rather than accumulated across batches so it stays correct
                # whatever order or subset of windows this generator is driven through.
                rows = np.arange(window.row_off, window.row_off + window.height, dtype="int64")
                cols_idx = np.arange(window.col_off, window.col_off + window.width, dtype="int64")
                px = (rows[:, None] * raster_width + cols_idx[None, :]).ravel()
                leading["_px"] = px if keep is None else px[keep]
            df = pl.DataFrame({**leading, **crosswalk_cols, **cols, **value_cols})
            del cols, leading, crosswalk_cols

            # Nodata -> null, so every downstream aggregate skips it (including `mean`'s
            # denominator, via Polars' native null-skipping `count()`/`mean()`) with no
            # per-aggregate special-casing needed here.
            if mask_nodata:
                df = df.with_columns(
                    [
                        pl.when(pl.col(n) == nd).then(None).otherwise(pl.col(n)).alias(n)
                        for n, nd in nodata_by_col.items()
                        if nd is not None
                    ]
                )

            # An attached crosswalk's nodata is masked whatever `mask_nodata` says: see
            # the parameter's docstring above -- this column exists to be a join key, and
            # "outside this grid's domain" has to miss that join rather than match some
            # unrelated zone whose id happens to equal the sentinel.
            df = df.with_columns(
                [
                    pl.when(pl.col(n) == int(nd)).then(None).otherwise(pl.col(n)).alias(n)
                    for n, nd in crosswalk_nodata.items()
                    if nd is not None
                ]
            )

            # Value-column misses arrive as `nan` (see `_crosswalk_lookup_values`); turn
            # them into a true Polars null here, once, rather than at every miss branch.
            if value_cols:
                df = df.with_columns([pl.col(n).fill_nan(None) for n in value_cols])
            del value_cols

            # --- predicate pushdown: drop rows before they leave the source ---
            if predicate is not None:
                df = df.filter(predicate)

            if remaining is not None and df.height > remaining:
                df = df.head(remaining)
            if remaining is not None:
                remaining -= df.height

            yield df

    return register_io_source(source, schema=schema)


# See `scan_broadcast`'s window-sizing comment: a measured stand-in for how many
# same-length 8-byte temporary arrays are alive at once per window in that source's
# `source()` closure, used only to size its read window -- never a real dtype.
_BROADCAST_ITEMSIZE_INFLATION = 4


def scan_broadcast(broadcast_crosswalk_path, broadcast_ids, broadcast_values, output_zones_path, output_zone_ids):
    """Return a LazyFrame of (grid_cell, value) broadcasting a small vector-grid
    selection onto a raster grid's pixels via two co-registered crosswalk rasters.

    This currently has no API caller. It is retained as the executor for a planned
    pipeline grid-projection operation: projecting a vector (zone-keyed) dataset's
    per-zone values onto a raster grid's own pixels through a crosswalk raster, for
    a caller that wants to zonal-summarize those projected values by a second, possibly
    different zone set -- e.g. projecting a hydro-compartment salinity value onto
    vegetation pixels, then zonal-summarizing by ecoregion. Measured at 752 MB peak at
    full raster extent, and bit-identical to the reference formula
    (``np.where(ids == nodata, nan, vector_values[ids])``) it reproduces without holding
    the raster grid's full pixel grid in the vector dataset's own shape.

    This is the cross-grid counterpart to `scan_rasters` above: where `scan_rasters`
    reads actual band data from a variable's own files, this reads no variable data at
    all -- the vector dataset (a variable+selection on the vector grid) is assumed
    already collapsed, by the caller, to one row per vector zone (`broadcast_ids`/
    `broadcast_values`, aligned arrays), so every pixel's value is determined entirely by
    which vector zone it falls in. That is also why this can be a NumPy lookup array
    rather than a Polars join: the lookup array's size is the vector zone ID range, not
    the (unbounded) pixel count, so the whole "join" is O(1) per pixel and never grows
    with how much of the raster is read. A real Polars `.join()` against the small
    vector frame would reach the identical result -- the lookup is preferred here only
    because the join key range is already known to be small and dense enough that
    indexing beats hashing, not because a join wouldn't also fit in memory.

    `output_zones_path`/`output_zone_ids` name a SECOND crosswalk (the destination zone a
    caller wants results grouped by) -- independent of `broadcast_crosswalk_path`, which
    supplies the vector VALUE. The two may be the same file (a caller re-summarizing
    broadcast values back onto their own vector zones, a useful identity check) or
    different (zonal stats of broadcast values over an entirely different zone set than
    the one that supplied the vector value). Both rasters are read per window; a pixel is
    dropped -- not merely nulled -- when EITHER crosswalk's ID at that pixel is nodata or
    absent from its own ID set (`broadcast_ids` / `output_zone_ids`). Dropping instead of
    nulling reproduces the same result every supported aggregate would get from nulling,
    since every aggregate a caller can apply downstream is null-skipping, so a row that
    never existed and a row that existed with a null value are indistinguishable to
    `count`, `sum`, `mean`, `min`, `max`. Either way the pixel never contributes a vector
    value to any output group, matching the ``np.where(ids == nodata, nan, ...)`` masking
    this executor models.

    `broadcast_ids` must be sorted; `broadcast_values` is any numeric NumPy dtype aligned
    to it (typically Float64, so a null-shaped "no vector row for this id" default of NaN
    survives the lookup untouched -- see the guard in `source` below).
    """
    broadcast_crosswalk_path = str(broadcast_crosswalk_path)
    output_zones_path = str(output_zones_path)
    schema = pl.Schema({"grid_cell": pl.Int32, "value": pl.Float64})

    # A dense lookup array indexed by raw vector zone ID, built once here (not per
    # window): NaN everywhere except the caller's own vector rows, so a crosswalk ID that
    # is in-range but was never one of `broadcast_ids` (the "absent from the vector
    # frame" case) resolves to NaN exactly like a lookup miss, with no separate branch.
    # This array's size scales with the vector ID range, not the (unbounded) pixel count
    # -- a caller with a high-cardinality vector selection should bound its row count
    # before calling in, the same way `expressions.count_by_value_frame` bounds distinct
    # values before its own array-shaped operation.
    lookup = _dense_lookup_array(broadcast_ids, broadcast_values)
    max_id = int(broadcast_ids.max()) if broadcast_ids.size else -1

    def source(with_columns, predicate, n_rows, batch_size):
        remaining = n_rows

        with raster_gdal_env(), rio.open(broadcast_crosswalk_path) as first_src:
            block_height, block_width = first_src.block_shapes[0]
            # `raster_blocks_per_batch`'s budget is bytes-per-band-per-pixel
            # (`itemsize`), sized for `scan_rasters`' one-array-per-band shape. This
            # source's per-pixel footprint is different in kind: reading a window touches
            # both zone rasters and carries several same-length Int64/Float64 intermediate
            # arrays at once (search positions, keep masks, the looked-up values) that
            # `scan_rasters` never allocates, since it has no lookup step. A single-band
            # crosswalk's own itemsize (commonly Int16, 2 bytes) badly underestimates that
            # footprint -- passed through unchanged, the derived window can span an entire
            # block row (measured 1.7 GB peak on the real 1657-zone crosswalk at full
            # 20674x8265 extent, on data with no per-pixel data file to read at all).
            # `_BROADCAST_ITEMSIZE_INFLATION` is a measured stand-in for "how many
            # same-length 8-byte temporaries are alive at once per window", not the raster's
            # real dtype -- treating it as 4 such arrays brought the same measurement down
            # to well under a quarter of that ceiling with no correctness change (it only
            # feeds the batch-size heuristic, never the actual read or dtype).
            itemsize = 8 * _BROADCAST_ITEMSIZE_INFLATION
            blocks_per_batch = raster_blocks_per_batch(1, block_height, block_width, itemsize)
            window_list = list(_batched_windows(first_src, blocks_per_batch))

        for window in window_list:
            if remaining is not None and remaining <= 0:
                break

            # See `scan_rasters`'s identical comment: nothing GDAL-related may cross the
            # `yield` below, since the streaming engine can resume this generator on a
            # different thread than the one that entered the env.
            with raster_gdal_env():
                with rio.open(broadcast_crosswalk_path) as bsrc:
                    broadcast_zones = bsrc.read(1, window=window).ravel()
                    broadcast_nodata = bsrc.nodata
                if output_zones_path == broadcast_crosswalk_path:
                    output_zones = broadcast_zones
                    output_nodata = broadcast_nodata
                else:
                    with rio.open(output_zones_path) as osrc:
                        output_zones = osrc.read(1, window=window).ravel()
                        output_nodata = osrc.nodata

                # Source-level spatial filter, mirroring `scan_rasters`: keep only pixels
                # whose OUTPUT zone is one of the requested destinations.
                keep = _zone_keep_positions(output_zone_ids, output_zones)
                if output_nodata is not None:
                    keep = keep[output_zones[keep] != output_nodata]
                if keep.size == 0:
                    continue

                b_zones = broadcast_zones[keep]
                out_zones = output_zones[keep]

                # Second mask, same treatment: a broadcast-crosswalk nodata pixel, or one
                # whose id is out of the lookup's range entirely (bigger than any vector
                # row's id), is dropped here rather than left to look up out of bounds.
                valid = b_zones >= 0
                if broadcast_nodata is not None:
                    valid &= b_zones != broadcast_nodata
                valid &= b_zones <= max_id
                if not valid.all():
                    b_zones = b_zones[valid]
                    out_zones = out_zones[valid]
                if b_zones.size == 0:
                    continue

                values = lookup[b_zones]
                # A crosswalk ID inside the lookup's range but never one of the vector
                # frame's own rows resolves to NaN from the `np.full` default above --
                # drop those pixels too, rather than emit a fabricated NaN "value" a
                # caller's aggregate would otherwise have to special-case.
                present = ~np.isnan(values)
                if not present.all():
                    values = values[present]
                    out_zones = out_zones[present]

            if out_zones.size == 0:
                continue

            df = pl.DataFrame({"grid_cell": out_zones.astype("int32"), "value": values})

            if predicate is not None:
                df = df.filter(predicate)

            if remaining is not None and df.height > remaining:
                df = df.head(remaining)
            if remaining is not None:
                remaining -= df.height

            yield df

    return register_io_source(source, schema=schema)
