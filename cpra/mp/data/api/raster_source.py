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

import numpy as np
import polars as pl
import rasterio as rio
from polars.io.plugins import register_io_source
from rasterio import windows

from cpra.mp.data.api.expressions import count_by_value_frame, rewrite_value_column, value_predicate_mask
from cpra.mp.data.api.utils import raster_blocks_per_batch, raster_gdal_env

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
# selected band columns (dataset.py's per-band multiplier + group_aggregate collapse), not
# DOWN a `group_by`'s rows the way the zonal summarize aggregates above do -- band-collapse and
# zonal summarize are different axes of the same per-band pipeline and are kept in a
# separate dict so the two are never reached for each other by mistake. Shared by
# `Dataset._summarize_zone_stats_plugin` (pre-collapsing a group before the vertical zonal
# `guarded_aggregate`) and `Dataset._write_raster_windowed` (collapsing a group into one
# output band) -- the one shared expression vocabulary both paths use for group_aggregate.
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


def band_collapse_exprs(raw_names, multipliers, group_row_idxs, aggregate, dtype, nodata=None):
    """The whole calculate/group_aggregate step for a raster selection, as expressions.

    Operates on a DataFrame whose columns are `raw_names` -- one raw band per selected
    (path, band) row, in df row order. Returns `(out_names, exprs)`: one output column name
    and expression per output band. `multipliers` (calculate parameters) and
    `group_row_idxs` (the contributing row positions for each aggregated output band, or
    None for a 1:1 selection) are indexed by df row position too, so all three line up.
    `dtype` is the raw bands' numpy dtype name, which is also the output dtype before any
    multiplier promotes it.

    This is the single implementation of that step, shared by every caller that needs it
    (`Dataset._write_raster_windowed`, `Dataset._collapse_tile_bands`, and -- expressed over
    the plugin's own scan rather than a per-window frame -- `_summarize_zone_stats_plugin`),
    so the four aggregates and the multiply-before-collapse order are defined once.

    Passing `nodata` masks each band against its own raw pixel value BEFORE the multiply,
    since scaling a sentinel would stop it matching the sentinel it started as. A masked
    pixel then drops out of the collapse instead of contributing a sentinel to it, and the
    sentinel comes back only where nothing contributed. Pass None to leave pixels untouched,
    which is what a caller wants when its output has to preserve whatever was stored rather
    than reinterpret it.

    No null reaches the output columns: a masked-out band's null is filled back with the
    sentinel, by `horizontal_aggregate_expr`'s `no_contributor` branch for an aggregated
    selection and by an explicit `fill_null` for a 1:1 one. So a caller gets a column it can
    hand straight to numpy in the dtype family it would have had unmasked -- which the tile
    path depends on, since a null would arrive as NaN and land in whichever legend bin NaN
    happens to digitize into (the last one).
    """
    if nodata is not None:
        cols = [pl.when(pl.col(name) == nodata).then(None).otherwise(pl.col(name)) for name in raw_names]
    else:
        cols = [pl.col(name) for name in raw_names]

    # Skipped entirely (rather than always multiplying by 1.0) when every multiplier is 1,
    # so a plain selection keeps its source dtype instead of being silently promoted to
    # float.
    if not all(m == 1 for m in multipliers):
        cols = [c * float(m) for c, m in zip(cols, multipliers)]

    if group_row_idxs is None:
        out_names = [f"out{k}" for k in range(len(cols))]
        if nodata is not None:
            fill = pl.lit(nodata).cast(NUMPY_TO_POLARS[dtype])
            cols = [c.fill_null(fill) for c in cols]
        return out_names, [c.alias(name) for c, name in zip(cols, out_names)]

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
    return out_names, exprs


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
    separate columns would need a different output shape per band count. Per the plan's
    "silently wrong multi-band results are not acceptable" rule, this is refused rather
    than guessed at -- re-asserted here defensively rather than only trusted from the
    caller.

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

    ``zone_ids``, when given, is the sorted array of destination zone IDs to summarize
    (as resolved in `Dataset._apply_transform_params_raster`): a `grid_cell` column holds
    the destination zone ID from the crosswalk raster at `zones_path`, and pixels whose
    zone is not in `zone_ids` (including crosswalk nodata) are dropped in the source.
    ``zone_ids=None`` is the dense, one-row-per-pixel shape instead: every pixel, in raster
    order, no crosswalk read and no `grid_cell` column -- not used by summarize, but the
    shape a future full-resolution consumer (e.g. downloads) would need.

    ``value_dtype=None`` infers the schema from `paths[0]` -- see `_raster_dtype`.

    ``mask_nodata=True`` maps each column's stored nodata sentinel (read per-file, so
    multiple sources may declare different values) to null before any aggregate sees it;
    a file with no declared nodata (`src.nodata is None`) is left unmasked -- such a raster
    is treated as never having nodata at all.
    """
    if not paths:
        raise ValueError("scan_rasters requires at least one source path")
    zones_path = str(zones_path) if zones_path is not None else None
    dense = zone_ids is None

    names_by_path = [[f"b{i}" for i in indices] for indices in row_indices]
    bands_by_path = [
        list(path_bands) if path_bands is not None else list(range(1, len(names) + 1))
        for path_bands, names in zip(bands, names_by_path)
    ]
    all_names = [n for names in names_by_path for n in names]
    if value_dtype is None:
        value_dtype = _raster_dtype(paths[0])

    cols_schema = {name: value_dtype for name in all_names}
    schema = pl.Schema(cols_schema if dense else {"grid_cell": pl.Int32, **cols_schema})

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
                if dense:
                    keep = None
                    zones = None
                else:
                    with rio.open(zones_path) as zsrc:
                        zones = zsrc.read(1, window=window).ravel()

                    # Source-level spatial filter: keep only pixels in a summarized zone.
                    positions = np.searchsorted(zone_ids, zones).clip(max=zone_ids.size - 1)
                    keep = np.flatnonzero(zone_ids[positions] == zones)
                    if keep.size == 0:
                        continue

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

            if dense:
                df = pl.DataFrame(cols)
            else:
                df = pl.DataFrame({"grid_cell": zones[keep].astype("int32"), **cols})
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
    max_id = int(broadcast_ids.max()) if broadcast_ids.size else -1
    lookup = np.full(max_id + 1, np.nan, dtype="float64")
    lookup[broadcast_ids] = broadcast_values

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
                out_positions = np.searchsorted(output_zone_ids, output_zones).clip(max=output_zone_ids.size - 1)
                keep = np.flatnonzero(output_zone_ids[out_positions] == output_zones)
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
