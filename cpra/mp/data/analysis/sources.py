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
# `Dataset._summarize_zone_stats_plugin` (pre-collapsing a group before the vertical zonal
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
    caller evaluates that against its own rows -- see
    `RasterGridAnalysis._band_applicability`). A plan's `value_where`, being per-pixel,
    is folded into each band's own rewrite condition here instead, rewritten per band by
    `rewrite_value_column` exactly like a filter value condition. A band this plan does
    not apply to is simply left out of that pass -- pass-through, never dropped or
    masked.

    The `is_not_null()` guard on every rewrite makes "NoData in, NoData out" a structural
    property rather than a per-expression obligation: a nodata pixel (already masked to
    null by the caller -- `scan_rasters(mask_nodata=True)`, or the nodata pass
    `band_collapse_exprs` prepends) never matches, so even an expression that COULD
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


def band_collapse_exprs(raw_names, calculate_passes, group_row_idxs, aggregate, dtype, nodata=None):
    """The whole calculate/group_aggregate step for a raster selection, as expressions.

    Operates on a DataFrame whose columns are `raw_names` -- one raw band per selected
    (path, band) row, in df row order. Returns `(out_names, passes, exprs)`: apply each
    list in `passes` as one `with_columns` call, in order (the nodata mask, then each
    calculate entry -- sequential passes, since entry N+1 must see entry N's output),
    then `exprs` as a final `select`, producing one output column per output band named
    by `out_names`. `calculate_passes` is `band_calculate_passes`' output (empty list for
    a request with no calculate entries) and `group_row_idxs` (the contributing row
    positions for each aggregated output band, or None for a 1:1 selection) is indexed by
    df row position too, so everything lines up. `dtype` is the raw bands' numpy dtype
    name, which is also the output dtype before any calculate expression promotes it.

    This is the single implementation of that step, shared by every caller that needs it
    (`Dataset._write_raster_windowed`, `Dataset._collapse_tile_bands`, and -- expressed over
    the plugin's own scan rather than a per-window frame -- `_summarize_zone_stats_plugin`),
    so the four aggregates and the calculate-before-collapse order are defined once.

    Passing `nodata` masks each band against its own raw pixel value BEFORE any calculate
    entry runs, since rewriting a sentinel would stop it matching the sentinel it started
    as (and would violate the calculate stage's NoData-in-NoData-out contract). A masked
    pixel then drops out of the collapse instead of contributing a sentinel to it, and the
    sentinel comes back only where nothing contributed. Pass None to leave pixels untouched,
    which is what a caller wants when its output has to preserve whatever was stored rather
    than reinterpret it -- only valid with no calculate passes, or for a source that
    declares no nodata at all.

    No null reaches the output columns: a masked-out band's null is filled back with the
    sentinel, by `horizontal_aggregate_expr`'s `no_contributor` branch for an aggregated
    selection and by an explicit `fill_null` for a 1:1 one. So a caller gets a column it can
    hand straight to numpy in the dtype family it would have had unmasked -- which the tile
    path depends on, since a null would arrive as NaN and land in whichever legend bin NaN
    happens to digitize into (the last one).
    """
    passes = []
    if nodata is not None:
        passes.append(
            [pl.when(pl.col(name) == nodata).then(None).otherwise(pl.col(name)).alias(name) for name in raw_names]
        )
    passes.extend(calculate_passes)

    cols = [pl.col(name) for name in raw_names]

    if group_row_idxs is None:
        out_names = [f"out{k}" for k in range(len(cols))]
        if nodata is not None:
            fill = pl.lit(nodata).cast(NUMPY_TO_POLARS[dtype])
            cols = [c.fill_null(fill) for c in cols]
        return out_names, passes, [c.alias(name) for c, name in zip(cols, out_names)]

    # No band contributed anything to this pixel (every band in the group was nodata
    # there): the raster's own nodata sentinel, not NaN. Unlike the summarize path's Float64
    # result, an integer output band cannot hold NaN -- this is a deliberate difference from
    # that path's NaN marker, not an oversight. With no masking there is never a null to
    # select it, so 0 (in the output dtype) is a harmless placeholder for that case.
    no_contributor = (
        pl.lit(nodata).cast(NUMPY_TO_POLARS[dtype])
        if nodata is not None
        else pl.lit(0, dtype=NUMPY_TO_POLARS[dtype])
    )
    out_names = [f"out{k}" for k in range(len(group_row_idxs))]
    exprs = [
        horizontal_aggregate_expr(aggregate, [cols[i] for i in row_idxs], no_contributor).alias(name)
        for name, row_idxs in zip(out_names, group_row_idxs)
    ]
    return out_names, passes, exprs


def guarded_aggregate(aggregate, name):
    """A `group_by().agg()` expression for one column, with the all-nodata edge case fixed.

    Bare Polars returns `null` for `min`/`max`/`mean` on an all-null group, and `0` (the
    additive identity) for `sum` -- neither is a meaningful answer for a (band, zone) with
    no contributing pixel at all, which `_apply_transform_params_raster` needs reported as
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
    selected band -- `Dataset._apply_transform_params_raster` enforces this before ever
    calling in (via its already-computed `n_bands`), since a multi-band request has no
    single natural (zone, value) shape: counting each band's values into the same `value`
    column would silently conflate unrelated variables/years, and counting them into
    separate columns would need a different output shape per band count. A silently wrong
    multi-band result is not acceptable, so this is refused rather than guessed at --
    re-asserted here defensively rather than only trusted from the caller.

    `value_where`, when given, is applied the same way `_summarize_zone_stats_plugin` (the
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
    (`scan_rasters`'s zoned mode, `scan_broadcast`'s output-zone filter, and
    `scan_paired`'s output-zone filter): a pixel whose crosswalk value is not one of
    `zone_ids` -- including the crosswalk's nodata sentinel, which is never itself a
    requested zone id -- is dropped by construction, with no separate nodata branch
    needed here. `zone_ids` must be sorted (each call site's caller already sorts its
    own zone id array once, not per window).
    """
    positions = np.searchsorted(zone_ids, zones).clip(max=zone_ids.size - 1)
    return np.flatnonzero(zone_ids[positions] == zones)


def _dense_lookup_array(ids, values):
    """A dense NumPy array indexed by raw id, `np.nan` everywhere except `ids`.

    Factored out of `scan_broadcast` so `scan_paired` (below) can build the identical
    lookup shape once per calendar step rather than duplicating the array-construction
    logic. `ids` must be sorted for the caller's own bookkeeping (this function itself
    only needs `ids` and `values` aligned pairwise -- sortedness is not used here, only
    by the callers that build these arrays from a sorted coarse selection). An id that
    is in-range but was never one of `ids` (absent from the coarse frame) resolves to
    `nan`, indistinguishable from a lookup miss, with no separate branch.
    """
    max_id = int(ids.max()) if ids.size else -1
    lookup = np.full(max_id + 1, np.nan, dtype="float64")
    lookup[ids] = values
    return lookup


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


def scan_rasters(paths, row_indices, bands, zones_path=None, zone_ids=None, value_dtype=None, mask_nodata=True):
    """Return a LazyFrame of (grid_cell, <band columns>) over one or more multiband COGs.

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
      deleted as unused once the raster interpreter landed and none of the raster-side
      rejections it would have served turned out to be relaxable. This attach-only mode
      is still exercised directly by
      `tests/api/test_raster_source.py::TestScanRastersAttachWithoutFilter`.)
    - ``zones_path`` and ``zone_ids`` both given: the original zoned/TABULAR shape --
      a `grid_cell` column holds the destination zone ID, and pixels whose zone is not
      in `zone_ids` (including crosswalk nodata) are DROPPED in the source. Unchanged:
      this is still what `Dataset._summarize_zone_stats_plugin`/`count_by_value` use,
      and dropping here (rather than in a downstream `filter`) remains correct for them
      specifically because their output is immediately `group_by`'d -- row COUNT was
      never part of their contract, only group membership -- unlike a raster-output
      query, whose pixel count is its contract.

    ``zone_ids`` without ``zones_path`` is rejected: there is no crosswalk to source a
    zone ID from.

    ``value_dtype=None`` infers the schema from `paths[0]` -- see `_raster_dtype`.

    ``mask_nodata=True`` maps each column's stored nodata sentinel (read per-file, so
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

    cols_schema = {name: value_dtype for name in all_names}
    schema = pl.Schema({"grid_cell": pl.Int32, **cols_schema} if attach_zone else cols_schema)

    def source(with_columns, predicate, n_rows, batch_size):
        # --- projection pushdown: only decompress the bands the query asks for ---
        wanted = set(all_names) if with_columns is None else {c for c in with_columns if c != "grid_cell"}

        # A file none of whose bands are wanted is never opened at all.
        plan = []
        for path, names, path_bands in zip(paths, names_by_path, bands_by_path):
            picked = [(n, b) for n, b in zip(names, path_bands) if n in wanted]
            if picked:
                plan.append((str(path), [n for n, _ in picked], [b for _, b in picked]))

        if not plan:
            return

        remaining = n_rows
        n_selected_bands = sum(len(names) for _p, names, _b in plan)

        # The window list -- and the batch size it's chunked to -- are derived once, from
        # the first source surviving projection pushdown, so a narrow band selection gets
        # a correspondingly larger batch rather than inheriting a size tuned for the full
        # band count. Uses the shared `raster_blocks_per_batch` budget/knob (see
        # config.RASTER_BATCH_BUFFER_MB) rather than a plugin-only setting.
        with raster_gdal_env(), rio.open(plan[0][0]) as first_src:
            block_height, block_width = first_src.block_shapes[0]
            itemsize = np.dtype(first_src.dtypes[0]).itemsize
            blocks_per_batch = raster_blocks_per_batch(n_selected_bands, block_height, block_width, itemsize)
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

            if not attach_zone:
                df = pl.DataFrame(cols)
            else:
                zone_col = zones if keep is None else zones[keep]
                df = pl.DataFrame({"grid_cell": zone_col.astype("int32"), **cols})
            del cols

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
    """Return a LazyFrame of (grid_cell, value) broadcasting a small coarse-grid
    selection onto a fine grid's pixels via two co-registered crosswalk rasters.

    This currently has no API caller. It is retained as the executor for a planned
    pipeline grid-projection operation: projecting a coarse (zone-keyed) dataset's
    per-zone values onto a fine raster grid's own pixels through a crosswalk raster, for
    a caller that wants to zonal-summarize those projected values by a second, possibly
    different zone set -- e.g. projecting a hydro-compartment salinity value onto
    vegetation pixels, then zonal-summarizing by ecoregion. Measured at 752 MB peak at
    full raster extent, and bit-identical to the reference formula
    (``np.where(ids == nodata, nan, coarse_values[ids])``) it reproduces without holding
    the fine grid's full pixel grid in the coarse dataset's own shape.

    This is the cross-grid counterpart to `scan_rasters` above: where `scan_rasters`
    reads actual band data from a variable's own files, this reads no variable data at
    all -- the coarse dataset (a variable+selection on the coarse grid) is assumed
    already collapsed, by the caller, to one row per coarse zone (`broadcast_ids`/
    `broadcast_values`, aligned arrays), so every pixel's value is determined entirely by
    which coarse zone it falls in. That is also why this can be a NumPy lookup array
    rather than a Polars join: the lookup array's size is the coarse zone ID range, not
    the (unbounded) pixel count, so the whole "join" is O(1) per pixel and never grows
    with how much of the fine raster is read. A real Polars `.join()` against the small
    coarse frame would reach the identical result -- the lookup is preferred here only
    because the join key range is already known to be small and dense enough that
    indexing beats hashing, not because a join wouldn't also fit in memory.

    `output_zones_path`/`output_zone_ids` name a SECOND crosswalk (the destination zone a
    caller wants results grouped by) -- independent of `broadcast_crosswalk_path`, which
    supplies the coarse VALUE. The two may be the same file (a caller re-summarizing
    broadcast values back onto their own coarse zones, a useful identity check) or
    different (zonal stats of broadcast values over an entirely different zone set than
    the one that supplied the coarse value). Both rasters are read per window; a pixel is
    dropped -- not merely nulled -- when EITHER crosswalk's ID at that pixel is nodata or
    absent from its own ID set (`broadcast_ids` / `output_zone_ids`). Dropping instead of
    nulling reproduces the same result every supported aggregate would get from nulling,
    since every aggregate a caller can apply downstream is null-skipping, so a row that
    never existed and a row that existed with a null value are indistinguishable to
    `count`, `sum`, `mean`, `min`, `max`. Either way the pixel never contributes a coarse
    value to any output group, matching the ``np.where(ids == nodata, nan, ...)`` masking
    this executor models.

    `broadcast_ids` must be sorted; `broadcast_values` is any numeric NumPy dtype aligned
    to it (typically Float64, so a null-shaped "no coarse row for this id" default of NaN
    survives the lookup untouched -- see the guard in `source` below).
    """
    broadcast_crosswalk_path = str(broadcast_crosswalk_path)
    output_zones_path = str(output_zones_path)
    schema = pl.Schema({"grid_cell": pl.Int32, "value": pl.Float64})

    # A dense lookup array indexed by raw coarse zone ID, built once here (not per
    # window): NaN everywhere except the caller's own coarse rows, so a crosswalk ID that
    # is in-range but was never one of `broadcast_ids` (the "absent from the coarse
    # frame" case) resolves to NaN exactly like a lookup miss, with no separate branch.
    # This array's size scales with the coarse ID range, not the (unbounded) pixel count
    # -- a caller with a high-cardinality coarse selection should bound its row count
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
                # whose id is out of the lookup's range entirely (bigger than any coarse
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
                # A crosswalk ID inside the lookup's range but never one of the coarse
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


def _paired_lookup_values(lookup_array, pairing_ids, pairing_valid):
    """Per-pixel paired-column values for one fine band, via one calendar step's dense
    lookup array (see `_dense_lookup_array`).

    `pairing_valid` has already folded in the pairing crosswalk's own nodata/absent
    check (see `scan_paired`'s ``source`` below); this function adds the remaining two
    ways a pixel can miss a paired value: the id is out of the lookup array's range
    entirely (bigger than any coarse row's id -- `lookup_array is None` covers "no
    coarse row for this calendar step at all", the same treatment), or the id is
    in-range but was never one of the coarse frame's own rows, which `_dense_lookup_array`
    already resolves to `nan`. Every miss kind converges on the same `nan` return value,
    which the caller (`scan_paired.source`) turns into a true Polars null with one
    `fill_nan(None)` -- deliberately NOT dropped the way `scan_broadcast` drops an
    equivalent miss, since a paired column's row must survive for a predicate spanning
    both the fine and paired columns to evaluate at all.
    """
    if lookup_array is None or lookup_array.size == 0:
        return np.full(pairing_ids.shape, np.nan)
    within = pairing_valid & (pairing_ids >= 0) & (pairing_ids < lookup_array.size)
    out = np.full(pairing_ids.shape, np.nan)
    if within.any():
        out[within] = lookup_array[pairing_ids[within]]
    return out


class PairedCrosswalk(NamedTuple):
    """One `project`/`join` step's coarse crosswalk, generalizing
    `scan_paired`'s former single ``pairing_crosswalk_path``/``paired_lookups`` pair so
    more than one coarse dataset can be paired onto the same fine scan in a single pass.

    `name` is the caller's own identifier for the join step producing this crosswalk --
    it becomes the prefix of every column this crosswalk emits (``f"{name}{i}"``, ``i``
    the same row-position index ``b<i>``/``_px`` share), rather than every join
    converging on one baked-in ``p<i>`` column the way the single-crosswalk shape did.
    A caller pairing exactly one coarse dataset, wanting the original naming back, gets
    it back by passing ``name="p"``.

    `path` is a pairing-crosswalk raster co-registered with the fine grid -- may be the
    SAME file across two different `PairedCrosswalk`s (e.g. two join steps both pairing
    against `hydro_compartment_v001` with different coarse selections); nothing here
    assumes distinct files. `lookups` is ``{calendar_step: (ids, values)}``, exactly the
    per-crosswalk shape the old `paired_lookups` parameter had -- one such mapping per
    crosswalk instead of one for the whole call.
    """

    name: str
    path: object
    lookups: dict


def scan_paired(
    paths,
    row_indices,
    bands,
    crosswalks,
    calendar_steps,
    zones_path=None,
    zone_ids=None,
    value_dtype=None,
    mask_nodata=True,
):
    """Return a LazyFrame of (``[grid_cell]``, ``_px``, ``<b<i> fine columns>``,
    ``<crosswalk paired columns>``) over one or more co-registered fine-grid COGs,
    generalizing `scan_rasters` so a fine dataset's own pixel values and one or more
    coarse datasets' PAIRED, per-calendar-step values coexist as aligned columns in one
    frame, per pixel, per shared calendar step.

    This is the executor for an analysis pipeline's spatial join over co-registered
    rasters: the key is grid_cell plus every shared dimension, executed positionally
    rather than as a hash join -- the fine dataset's pixels and a coarse dataset's
    projection onto those same pixels never need one at all, because a pairing
    crosswalk raster already gives, for every fine pixel, the coarse zone id whose
    value it should be paired with. That is the same insight `scan_broadcast` already
    exploits (an O(1) lookup array beats a join when the key range is small and dense)
    generalized across a calendar axis: `scan_broadcast` broadcasts ONE coarse
    selection onto every pixel; this broadcasts a DIFFERENT coarse selection per fine
    band, one per calendar step -- and potentially more than one coarse dataset at
    once, each keeping `_dense_lookup_array`'s O(1) lookup as the
    band-wide lowering of its own `join` rather than a hash join against a small dense
    key range (measured better in `scan_broadcast`'s own docstring; not replaced here).

    ``paths``, ``row_indices`` and ``bands`` are exactly `group_rows_by_path`'s return
    values, identically to `scan_rasters` -- one fine-grid dataset per unique file, the
    caller's own row positions, and the (1-based) band numbers each file supplies. Fine
    value columns are named ``b<row index>`` exactly as `scan_rasters` names them, so a
    caller can map columns back to its own dimension rows the same way.

    ``crosswalks`` is a sequence of `PairedCrosswalk`. Each fine band gets one PAIRED
    column PER crosswalk, named ``f"{crosswalk.name}{row index}"`` (same row index as
    its ``b<row index>`` sibling), holding that crosswalk's coarse value looked up for
    that band's calendar step. Crosswalk names must be unique (enforced below) -- they
    are the only thing keeping two joins' columns apart in one frame.

    ``zones_path``/``zone_ids`` name the OUTPUT zone crosswalk and are now two
    INDEPENDENT, OPTIONAL knobs, mirroring `scan_rasters`' own `attach_zone`/
    `filter_to_zones` split -- the output zone crosswalk is optional because a
    raster-output query has no output zones:

    - ``zones_path=None`` (`zone_ids` must then also be `None`): NO output zone at all
      -- no `grid_cell` column, nothing filtered. This is the raster-output/write shape:
      every fine pixel survives, in raster order, identified only by `_px` (see below).
    - ``zones_path`` given, ``zone_ids=None``: attach a `grid_cell` column (this
      window's own zone-crosswalk pixel value) to EVERY pixel, dropping none -- the
      position-preserving "join zone IDs without filtering" shape `scan_rasters` added
      for the same reason.
    - ``zones_path`` and ``zone_ids`` both given: the original zoned/TABULAR shape -- a
      `grid_cell` column holds the destination zone id, and a pixel outside `zone_ids`
      (including this crosswalk's own nodata) is DROPPED. This is what a zonal paired
      aggregate uses today.

    ``_px`` is a ROW-MAJOR pixel index over the FULL raster extent (``row * width +
    col``, not window-relative), carried on every row regardless of which of the three
    modes above is in effect. It costs one `arange`-and-broadcast per window -- no file
    read -- so, like `grid_cell` above, it is always computed and attached rather than
    gated on projection pushdown. This is
    the positional information `paired.py`'s module docstring says `scan_paired` lacked
    -- carrying it is what makes a dense-mode scan (no `zones_path`) usable to drive a
    raster WRITE or a tile render, putting each pixel's value back at its own window
    position, rather than only ever a TABULAR zonal aggregate.

    ``value_dtype=None`` infers the fine schema from ``paths[0]``, exactly `scan_rasters`.
    ``mask_nodata=True`` nulls each fine column's own nodata sentinel before anything
    downstream sees it, exactly `scan_rasters` -- paired columns have no sentinel of
    their own to mask; their null-or-not comes entirely from their own pairing lookup.

    Null semantics -- three distinct miss kinds, two different outcomes:

    - **Output zone miss** (only possible when ``zones_path``/``zone_ids`` filter --
      this pixel's zone crosswalk id is nodata or not in ``zone_ids``): the pixel is
      DROPPED entirely, matching `scan_rasters` and `scan_broadcast` -- there is no
      output zone to attribute it to, so no row of any shape can represent it.
    - **Fine nodata** (this pixel's raw value at a selected band equals that file's own
      nodata sentinel): that band's ``b<i>`` is NULL, matching `scan_rasters`; the row
      SURVIVES, since other bands/paired columns at this pixel may still be valid.
    - **Pairing miss** (this pixel's crosswalk id is nodata/absent, OR that id has no
      row in this band's calendar step's ``lookups`` entry, OR that calendar step is
      missing from ``lookups`` altogether): that band's paired column is NULL. The row
      SURVIVES -- this is the deliberate divergence from `scan_broadcast`'s drop-on-miss
      rule. `scan_broadcast` can drop, because dropping and nulling are indistinguishable
      to every aggregate a caller applies afterwards (`count`/`sum`/`mean`/`min`/`max`
      all skip a row the same way they skip a null). A paired scan has no such
      guarantee: the exit-criterion query (``marsh > 0.5 AND sal > 5.5``) is a predicate
      over BOTH columns at once, and Polars' `&` propagates null in a way that still
      lets the fine side's own condition matter to which branch of an `OR` fires, or to
      a later step that only reads ``b<i>``. Dropping the row here would silently
      discard information a downstream predicate over the fine column alone is entitled
      to see; keeping it null instead defers that decision to whatever operator
      consumes this frame next.

    GDAL/threading discipline is identical to `scan_rasters`/`scan_broadcast`: nothing
    GDAL-related (an entered env, an open dataset handle) may cross the `yield` below,
    since Polars' streaming engine may resume this generator on a different thread than
    the one that entered them -- see either sibling source's identical comment at its
    own `yield` for the mechanism. Window batching reuses the same
    `raster_blocks_per_batch` budget (`config.RASTER_BATCH_BUFFER_MB`), so this source's
    memory profile scales with the same knob `scan_rasters` does.
    """
    if not paths:
        raise ValueError("scan_paired requires at least one fine-grid source path")
    if zone_ids is not None and zones_path is None:
        raise ValueError("scan_paired: zone_ids requires zones_path")
    zones_path = str(zones_path) if zones_path is not None else None
    attach_zone = zones_path is not None
    filter_to_zones = zone_ids is not None

    crosswalks = [cw._replace(path=str(cw.path)) for cw in crosswalks]
    crosswalk_by_name = {cw.name: cw for cw in crosswalks}
    if len(crosswalk_by_name) != len(crosswalks):
        raise ValueError("scan_paired: crosswalk names must be unique")

    fine_names_by_path = [[f"b{i}" for i in indices] for indices in row_indices]
    calendar_keys_by_path = [[calendar_steps[i] for i in indices] for indices in row_indices]
    bands_by_path = [
        list(path_bands) if path_bands is not None else list(range(1, len(names) + 1))
        for path_bands, names in zip(bands, fine_names_by_path)
    ]
    all_fine_names = [n for names in fine_names_by_path for n in names]
    fine_name_set = set(all_fine_names)

    if value_dtype is None:
        value_dtype = _raster_dtype(paths[0])

    # Per-crosswalk column names, one row-position-indexed list per path, shaped exactly
    # like `fine_names_by_path` above so projection pushdown can be computed the same
    # way for either. Lookup arrays are built once here, up front, per crosswalk -- not
    # per window -- the same reasoning `scan_broadcast` documents for why this is a
    # lookup and not a `.join()`. A calendar step absent from a crosswalk's own
    # `lookups` simply has no entry; `_paired_lookup_values` treats a missing entry
    # (`.get(...)` returning `None`) the same as an existing but empty one.
    names_by_crosswalk = {cw.name: [[f"{cw.name}{i}" for i in indices] for indices in row_indices] for cw in crosswalks}
    lookup_arrays_by_crosswalk = {
        cw.name: {step: _dense_lookup_array(np.asarray(ids), np.asarray(values)) for step, (ids, values) in cw.lookups.items()}
        for cw in crosswalks
    }
    all_paired_names = [name for names in names_by_crosswalk.values() for path_names in names for name in path_names]
    paired_name_set = set(all_paired_names)

    # A crosswalk named e.g. "b" would emit "b0", "b1", ... -- indistinguishable from
    # the fine columns' own `f"b{i}"` names, and a "grid_cell"/"_px"-named crosswalk
    # would collide with those two reserved columns. Any of these would silently
    # overwrite a column in the `dict` merge below rather than raise, which is exactly
    # the kind of silent-wrong-answer failure the rest of this module tries hard to
    # avoid -- reject it here instead, at scan-construction time.
    reserved = fine_name_set | {"grid_cell", "_px"}
    colliding = paired_name_set & reserved
    if colliding:
        raise ValueError(
            f"scan_paired: crosswalk name collides with a reserved column name: {sorted(colliding)}"
        )

    schema = pl.Schema(
        {
            **({"grid_cell": pl.Int32} if attach_zone else {}),
            "_px": pl.Int64,
            **{name: value_dtype for name in all_fine_names},
            **{name: pl.Float64 for name in all_paired_names},
        }
    )

    def source(with_columns, predicate, n_rows, batch_size):
        remaining = n_rows

        # --- projection pushdown: only decompress a band, or look up a paired value,
        # the query actually asks for. A paired column needs no file read at all (its
        # value comes entirely from a pairing crosswalk plus its precomputed lookup
        # array), so requesting one crosswalk's column alone never opens the fine file,
        # nor any OTHER crosswalk's raster. `grid_cell` and `_px` are excluded from this
        # decision (like `scan_rasters`' `grid_cell`) -- both are cheap to attach
        # wholesale and are computed unconditionally below.
        wanted = (
            fine_name_set | paired_name_set
            if with_columns is None
            else {c for c in with_columns if c not in ("grid_cell", "_px")}
        )
        wanted_fine = wanted & fine_name_set

        plan = []
        for path, names, path_bands in zip(paths, fine_names_by_path, bands_by_path):
            picked = [(n, b) for n, b in zip(names, path_bands) if n in wanted_fine]
            if picked:
                plan.append((str(path), [n for n, _ in picked], [b for _, b in picked]))

        crosswalk_plans = []
        for cw in crosswalks:
            specs = [
                (name, lookup_arrays_by_crosswalk[cw.name].get(step))
                for names, keys in zip(names_by_crosswalk[cw.name], calendar_keys_by_path)
                for name, step in zip(names, keys)
                if name in wanted
            ]
            if specs:
                crosswalk_plans.append((cw, specs))

        if not plan and not crosswalk_plans:
            return

        n_selected_fine = sum(len(names) for _p, names, _b in plan)
        n_selected_paired = sum(len(specs) for _cw, specs in crosswalk_plans)

        # Window/batch sizing follows `scan_rasters`' shape: derived once, from the
        # first source surviving projection pushdown, using the shared
        # `raster_blocks_per_batch` budget rather than a source-specific knob. The
        # reference raster for block shape is a fine-grid file when one is open (its
        # tiling is what actually governs decompression cost); failing that, the output
        # zone crosswalk when attached; failing that, the first surviving pairing
        # crosswalk -- all co-registered with the fine grid by construction, so any one
        # of them gives the same block geometry. Each paired column's per-pixel
        # footprint is a Float64 lookup result plus a same-shape pairing-id/validity
        # intermediate (mirroring `scan_broadcast`'s own inflation reasoning for its
        # lookup arrays), approximated here as 2 "band units" at the reference itemsize
        # per COLUMN rather than per crosswalk raster opened -- an intentional
        # over-estimate when several columns share one crosswalk file (that file's own
        # id/validity arrays are read once per window, not once per column), which
        # only makes the batch-size heuristic more conservative, never less.
        if plan:
            ref_path = plan[0][0]
        elif zones_path is not None:
            ref_path = zones_path
        else:
            ref_path = crosswalk_plans[0][0].path
        with raster_gdal_env(), rio.open(ref_path) as ref_src:
            block_height, block_width = ref_src.block_shapes[0]
            ref_itemsize = max(np.dtype(ref_src.dtypes[0]).itemsize, 8)
            n_units = max(n_selected_fine + n_selected_paired * 2, 1)
            blocks_per_batch = raster_blocks_per_batch(n_units, block_height, block_width, ref_itemsize)
            window_list = list(_batched_windows(ref_src, blocks_per_batch))
            raster_width = ref_src.width

        for window in window_list:
            if remaining is not None and remaining <= 0:
                break

            # See `scan_rasters`'s identical comment at its own `yield`: nothing
            # GDAL-related may cross the `yield` below.
            with raster_gdal_env():
                if attach_zone:
                    with rio.open(zones_path) as zsrc:
                        zones = zsrc.read(1, window=window).ravel()
                    if filter_to_zones:
                        # Source-level spatial filter: keep only pixels in a
                        # summarized output zone (the original zoned-mode behavior).
                        keep = _zone_keep_positions(zone_ids, zones)
                        if keep.size == 0:
                            continue
                    else:
                        # Attach every pixel's own output zone id, dropping nothing --
                        # mirrors `scan_rasters`' identical split.
                        keep = None
                else:
                    # No output zone crosswalk at all: the raster-output/write shape.
                    # Nothing is read here and nothing is dropped; `_px` alone
                    # identifies each surviving row's position.
                    zones = None
                    keep = None

                # Row-major pixel index over the FULL raster extent (not window-
                # relative) -- see this function's own docstring for why this is
                # always computed, never gated on projection pushdown.
                rows = np.arange(window.row_off, window.row_off + window.height, dtype="int64")
                px_cols = np.arange(window.col_off, window.col_off + window.width, dtype="int64")
                px_full = (rows[:, None] * raster_width + px_cols[None, :]).ravel()
                px = px_full if keep is None else px_full[keep]

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

                paired_cols = {}
                for cw, specs in crosswalk_plans:
                    with rio.open(cw.path) as psrc:
                        pairing_ids_full = psrc.read(1, window=window).ravel()
                        pairing_nodata = psrc.nodata
                    pairing_ids = pairing_ids_full if keep is None else pairing_ids_full[keep]
                    if pairing_nodata is not None:
                        pairing_valid = pairing_ids != pairing_nodata
                    else:
                        pairing_valid = np.ones(pairing_ids.shape, dtype=bool)
                    for name, lookup_array in specs:
                        paired_cols[name] = _paired_lookup_values(lookup_array, pairing_ids, pairing_valid)

            data = {}
            if attach_zone:
                data["grid_cell"] = (zones if keep is None else zones[keep]).astype("int32")
            data["_px"] = px
            data.update(cols)
            data.update(paired_cols)
            df = pl.DataFrame(data)
            del cols, paired_cols, data

            # Fine-side nodata -> null, identical to `scan_rasters`.
            if mask_nodata:
                df = df.with_columns(
                    [
                        pl.when(pl.col(n) == nd).then(None).otherwise(pl.col(n)).alias(n)
                        for n, nd in nodata_by_col.items()
                        if nd is not None
                    ]
                )

            # Paired-column misses arrive as `nan` (see `_paired_lookup_values`); turn
            # them into a true Polars null here, once, rather than at every miss branch
            # above.
            if crosswalk_plans:
                fill_names = [name for _cw, specs in crosswalk_plans for name, _ in specs]
                df = df.with_columns([pl.col(name).fill_nan(None) for name in fill_names])

            # --- predicate pushdown: drop rows before they leave the source ---
            if predicate is not None:
                df = df.filter(predicate)

            if remaining is not None and df.height > remaining:
                df = df.head(remaining)
            if remaining is not None:
                remaining -= df.height

            yield df

    return register_io_source(source, schema=schema)
