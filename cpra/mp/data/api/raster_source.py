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

# Polars equivalents of `RASTER_AGGREGATES` (dataset.py). The Float64 cast on `sum`/`mean`
# is deliberate: Polars would otherwise accumulate a Float32 column in Float32, which
# diverged from the numpy path's explicit float64 accumulation by ~1.5 absolute over 10M
# pixels in an earlier round. `min`/`max` need no cast -- they never accumulate.
POLARS_RASTER_AGGREGATES = {
    "sum": lambda c: c.cast(pl.Float64).sum(),
    "mean": lambda c: c.cast(pl.Float64).mean(),
    "min": lambda c: c.min(),
    "max": lambda c: c.max(),
}


def guarded_aggregate(aggregate, name):
    """A `group_by().agg()` expression for one column, with the all-nodata edge case fixed.

    Bare Polars returns `null` for `min`/`max`/`mean` on an all-null group, and `0` (the
    additive identity) for `sum` -- neither matches the numpy path's `NaN` for a (band,
    zone) with no contributing pixel at all. `count()` skips nulls, so it is exactly the
    contributing-pixel count (the numpy path's `valid`), and the guard below is needed for
    all four aggregates, not just `sum`, to make the two paths agree exactly.
    """
    agg_fn = POLARS_RASTER_AGGREGATES[aggregate]
    c = pl.col(name)
    return (
        pl.when(c.count() == 0)
        .then(pl.lit(float("nan"), dtype=pl.Float64))
        .otherwise(agg_fn(c))
        .alias(name)
    )


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
    zone is not in `zone_ids` (including crosswalk nodata) are dropped in the source,
    mirroring `_get_zone_segments`. ``zone_ids=None`` is the dense, one-row-per-pixel
    shape instead: every pixel, in raster order, no crosswalk read and no `grid_cell`
    column -- not used by summarize, but the shape a future full-resolution consumer
    (e.g. downloads) would need.

    ``value_dtype=None`` infers the schema from `paths[0]` -- see `_raster_dtype`.

    ``mask_nodata=True`` maps each column's stored nodata sentinel (read per-file, so
    multiple sources may declare different values) to null before any aggregate sees it;
    a file with no declared nodata (`src.nodata is None`) is left unmasked, matching the
    numpy path's treatment of such rasters as never having nodata.
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
        # band count. Reuses the same budget/knob the numpy path uses
        # (`raster_blocks_per_batch`) rather than a second, plugin-only setting.
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
            # per-aggregate special-casing -- unlike the numpy path, which has to mask
            # explicitly and correct `mean`'s denominator by hand (see
            # `_accumulate_zone_stats`).
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
