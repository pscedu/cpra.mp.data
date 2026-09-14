"""`RasterGridAnalysis`: the raster half of the former `api/dataset.py`'s `Dataset` class.

See `analysis/base.py`'s module docstring for the staged `apply_params` flow and the hook
contract every method here fills in; see `analysis/vector.py` for the vector half.
"""

import asyncio
import itertools
import tempfile
from contextlib import ExitStack
from functools import partial
from io import BytesIO
from pathlib import Path

import mercantile
import numpy as np
import polars as pl
import polars.selectors as cs
import rasterio as rio
from PIL import Image
from rasterio import windows
from rasterio.coords import BoundingBox, disjoint_bounds
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from cpra.mp.data import config
from cpra.mp.data.analysis.base import BaseAnalysis
from cpra.mp.data.analysis.expressions import ExpressionError, parse_value_expr
from cpra.mp.data.analysis.geography import parse_coordinate_pairs, project_coordinates
from cpra.mp.data.analysis.sources import (
    band_calculate_passes,
    band_collapse_exprs,
    guarded_aggregate,
    horizontal_aggregate_expr,
    masked_value_predicate_exprs,
    scan_rasters,
)
from cpra.mp.data.analysis.sources import (
    count_by_value as raster_count_by_value,
)
from cpra.mp.data.analysis.utils import (
    asynctiming,
    group_rows_by_path,
    parallel_read_rasters,
    parallel_sample_rasters,
    raster_blocks_per_batch,
    raster_copy_gdal_env,
    raster_gdal_env,
    unpack_values,
)
from cpra.mp.data.config import (
    COUNT_BY_VALUE_MAX_DISTINCT,
    RASTER_NUM_THREADS,
    RASTER_USE_VRT_COPY,
    RASTER_WINDOW_BUFFER_LIMIT_MB,
)
from cpra.mp.data.raster import set_vrt_band_descriptions, vrt

# Interleaving for downloaded rasters, set rather than inherited from the source. A source
# COG may be written INTERLEAVE=TILE, which is a COG-driver-only option that the GTiff driver
# used here rejects outright ("INTERLEAVE=tile unsupported, value must be PIXEL or BAND"), so
# inheriting it makes such a source undownloadable. PIXEL is also what keeps a multiband
# response small: the same 52-band elev selection wrote 1.65 GB pixel-interleaved versus
# 19.3 GB band-interleaved, since only pixel interleaving puts each pixel's bands adjacent
# for the compression predictor.
OUTPUT_INTERLEAVE = "PIXEL"


def _batched_block_windows(block_windows, blocks_per_batch):
    """Yield window unions over `block_windows`, chunked to at most `blocks_per_batch`
    blocks each rather than a whole block row at a time.

    `block_windows` (as returned by a rasterio dataset's own `block_windows(1)`) is grouped
    into block rows first, then each row's blocks are split into batches of
    `blocks_per_batch`. A block row can be tens of thousands of pixels wide regardless of
    how much data a request actually needs, so unioning a whole row per iteration -- the
    original behavior -- makes every iteration's read buffer scale with the raster's width
    and the request's band count rather than with `blocks_per_batch`. See
    `raster_blocks_per_batch` for how that batch size is chosen.
    """
    groups = itertools.groupby(block_windows, key=lambda bw: bw[0][0])
    for _row, group in groups:
        row_windows = [w for _ji, w in group]
        for start in range(0, len(row_windows), blocks_per_batch):
            yield windows.union(*row_windows[start : start + blocks_per_batch])


async def _read_window_columns(inputs, row_indices, bands, window, raw_names):
    """Read one window's raw per-(path, band) arrays into named, raveled columns.

    Column names follow the same `b<row position>` convention `raster_source.scan_rasters`
    uses for its band columns, keyed to the caller's own df row positions (not file band
    order) so the caller's expressions -- built once against `raw_names` -- line up with
    whichever (path, band) pair actually backs each row. Returns the columns dict (ready
    for `pl.DataFrame`) and the window's `(height, width)` shape, needed to reshape each
    output column back into a 2-D band after the Polars round-trip.
    """
    band_reads = await parallel_read_rasters(inputs, window, bands=bands, use_gdal_env=True)
    shape = band_reads[0].shape[1:]
    cols = {}
    for path_row_indices, read in zip(row_indices, band_reads):
        for i, row_idx in enumerate(path_row_indices):
            cols[raw_names[row_idx]] = read[i].ravel()
    return cols, shape


class RasterGridAnalysis(BaseAnalysis):
    async def _apply_grid_cell(self):
        """Apply the grid_cell filter to a raster-grid dataset."""
        lngs, lats = parse_coordinate_pairs(unpack_values(self.options.grid_cell, "grid_cell"))

        # Transform the coordinates to the correct CRS, and sample the rasters in parallel.
        # Rows sharing the same path (bands of the same multiband COG) are grouped so each
        # file is opened and sampled once, requesting all of its needed bands together.
        df = self.materialize()
        unique_paths, row_indices, bands = group_rows_by_path(df)
        with ExitStack() as stack:
            datasets = [stack.enter_context(rio.open(path)) for path in unique_paths]
            projected_pairs = list(zip(*project_coordinates(lngs, lats, datasets[0].crs)))
            group_samples = await parallel_sample_rasters(datasets, projected_pairs, bands=bands)
            dtype = datasets[0].dtypes[0]

        # Add the resulting values to the LazyFrame as one column per sampled coordinate,
        # then unpivot into long format (grid_cell/value) to match the long-format contract
        # used everywhere else in the API. Each grid_cell here is a "lng,lat" coordinate
        # pair label rather than an integer grid ID, since these are arbitrary sample points.
        columns = [np.empty(len(df), dtype=dtype) for _ in lngs]
        for path_row_indices, path_samples in zip(row_indices, group_samples):
            for column, values in zip(columns, path_samples):
                column[path_row_indices] = values
        column_labels = [f"{x!s},{y!s}" for x, y in zip(lngs, lats)]
        wide_df = df.with_columns(**dict(zip(column_labels, columns))).select(cs.exclude("path", "band"))
        index_cols = [col for col in wide_df.columns if col not in column_labels]
        self.lf = wide_df.unpivot(
            index=index_cols, on=column_labels, variable_name="grid_cell", value_name="value"
        ).lazy()

    def _collapse_group(self, group_by_cols):
        """Record, in `self.raster_groups`, which band rows belong to which output
        group -- no pixels are read here; a later stage (transform, tile, or write)
        collapses the recorded groups against actual band data.
        """
        if self.options.group_aggregate == "count":
            # group_aggregate is the raster band-COLLAPSE step
            # (`raster_source.HORIZONTAL_RASTER_AGGREGATES`), reducing ACROSS one
            # pixel's selected bands -- unlike the zonal summarize aggregate this
            # option shares its name and values with, "how many of these bands
            # contributed to this pixel" has no natural per-pixel scalar meaning the
            # way max/mean/min/sum do (see the `AggregateFunction.count` comment in
            # types.py). Rejected explicitly here rather than left to fail as a raw
            # `KeyError` out of `HORIZONTAL_RASTER_AGGREGATES` deep in a later stage.
            self._error("group_aggregate 'count' is not supported for raster grids")

        # agg_groups() is deprecated and panics when grouping on a literal, so track
        # row positions via with_row_index instead. `group_by_cols` is empty rather than
        # the pre-cleanup `[True]` sentinel for the group_aggregate-without-group_by
        # case (see `BaseAnalysis._get_group_by_columns`), and `group_by()` with no
        # columns collapses everything into a single group -- exactly like
        # `group_by(True)` did, minus the spurious "literal" column Polars names a bare
        # `True` group key.
        self.raster_groups = self.lf.with_row_index("row_idxs").group_by(*group_by_cols).agg(pl.col("row_idxs"))

    def _band_applicability(self, df, plan):
        """Which of `df`'s band-metadata rows a calculate plan's dimension `where`
        selects, as a list of booleans in df row order.

        The dimension half of a calculate `where` picks BANDS (e.g.
        `model_group_id=500`), so it is evaluated here against the metadata rows'
        dimension columns, never against pixels; the value half is per-pixel and is
        folded in later by `band_calculate_passes`. No dimension `where` at all means
        the entry applies to every band.
        """
        if plan.dimension_where is None:
            return [True] * len(df)
        try:
            mask = (
                df.lazy()
                .select(pl.sql_expr(plan.dimension_where).fill_null(False))
                .collect()
                .to_series()
            )
        except Exception as exc:
            self._error(
                f"{plan.label}: where {plan.dimension_where!r} cannot be evaluated "
                f"against this dataset's bands: {exc}"
            )
        return mask.to_list()

    def _band_calculate_passes(self, df, raw_names):
        """This request's calculate plans as per-band `with_columns` passes over
        `raw_names` (see `raster_source.band_calculate_passes`), with each plan's
        dimension `where` resolved against `df`'s own band rows first. Empty list when
        the request has no calculate entries.
        """
        if not self._calculate_plans:
            return []
        applicability = [self._band_applicability(df, plan) for plan in self._calculate_plans]
        return band_calculate_passes(raw_names, self._calculate_plans, applicability)

    def _get_raster_dimension_frame(self, df):
        """Get `(dim_df, n_bands)`: a DataFrame with one row per output raster band,
        aligned by row order with the `(n_bands, zone_ids.size)` array
        `_summarize_zone_stats_plugin` returns for the same `df`, and that band count as
        a separate value.

        A dimension column left out of the group by clause is simply absent from
        `dim_df` -- mirroring `VectorGridAnalysis._collapse_group`'s drop (see its
        comment for the full rationale: the old behavior null-filled every collapsed
        dimension back in, which is the artifact this drops). `n_bands` is returned
        alongside `dim_df` rather than left for a caller to compute as `len(dim_df)`
        because Polars cannot represent "zero columns, N rows": a group_aggregate with
        no group_by at all collapses to zero surviving dimension columns, and selecting
        a DataFrame down to zero columns always reports height 0 regardless of the
        source frame's real row count. `self.raster_groups` (which always carries
        `row_idxs`, a real column, even when no dimension survives) is the one frame
        whose height is reliable in that case, so it is read off `group_df` BEFORE
        `row_idxs` is dropped down to (possibly) zero remaining columns.
        """
        if self.raster_groups is None:
            dim_df = df.select(pl.exclude("path"))
            return dim_df, len(dim_df)

        group_df = self.raster_groups.collect()
        n_bands = group_df.height
        return group_df.select(pl.exclude("row_idxs")), n_bands

    async def _apply_transform_params_impl(self):
        """Apply transform and extract parameters to the image.

        Summarizes raster pixels by geographic unit (e.g. hydro compartment) using a
        pre-generated zone-ID crosswalk raster (see
        `python -m cpra.mp.data.cli crosswalk`), converting the result into a tabular
        LazyFrame in the same shape produced by `VectorGridAnalysis`'s transform hook
        (dimension columns, a `grid_cell` column holding the destination grid_cell ID, a
        `value` column, and a `geographic_unit` column).

        The actual per-zone reduction is computed by `_summarize_zone_stats_plugin`,
        returning a `(n_bands, zone_ids.size)` array.

        A `where` VALUE conjunct (a per-pixel value predicate -- see
        `_apply_filter_params`/`self._value_where`) is only meaningful here, at the
        summarize stage: it is what `_summarize_zone_stats_plugin` masks each band's raw
        pixel value against, before the multiplier/group_aggregate collapse, so "count
        pixels above X" and "sum of pixels above X" both fall out of the existing
        null-skipping zonal aggregates for free. A plain, non-summarized raster
        selection has no zonal aggregate to feed a mask into and no per-pixel frame the
        API exposes to a caller at all (`self.lf` is the band-metadata table until this
        method builds one) -- so a `where` value conjunct given without `summarize_by`/
        `summarize_aggregate` is rejected outright below, rather than silently ignored.
        """
        if self._value_where is not None:
            # Parsed here, against the placeholder namespace every value conjunct
            # writes against (`value`, `grid_cell`), before touching the crosswalk file,
            # `self.materialize()`, or any pixel data -- an aggregate-inside-predicate or
            # a whitelist violation is rejected before a single file is opened, not
            # discovered partway through a streaming collect. `_summarize_zone_stats_plugin`
            # re-parses the same string per selected band (each band's own namespace is
            # exactly {that band, grid_cell} -- see `masked_value_predicate_exprs`), so this
            # call is a validation pass, not the one whose result gets used.
            try:
                parse_value_expr(self._value_where, ["value", "grid_cell"])
            except ExpressionError as exc:
                self._error(str(exc))

        if not self._validate_summarize_params():
            if self._value_where is not None:
                self._error("where's value condition(s) require summarize_by and summarize_aggregate for raster grids")
            return

        crosswalk_name = f"{self.grid.name}__{self.options.summarize_by}.tif"
        crosswalk_path = config.CROSSWALKS_PATH / crosswalk_name
        if not crosswalk_path.exists():
            msg = f"No crosswalk from {self.grid.name} to {self.options.summarize_by}"
            self._error(msg)

        df = self.materialize()
        if df.is_empty():
            self._error("Dataset does not exist", status_code=404)

        dim_df, n_bands = self._get_raster_dimension_frame(df)
        unique_paths, row_indices, bands = group_rows_by_path(df)

        with rio.open(crosswalk_path) as crosswalk:
            if self.summarize_grid_cell_ids:
                zone_ids = np.array(sorted(self.summarize_grid_cell_ids), dtype="int64")
            else:
                dst_ids = crosswalk.tags()["dst_ids"].split(",")
                zone_ids = np.array(sorted(int(dst_id) for dst_id in dst_ids), dtype="int64")

        if self.options.summarize_aggregate == "count_by_value":
            # Single RAW band only: a request selecting more than one raw pixel band (e.g.
            # every calendar_year in a range, or several bands pre-collapsed by
            # group_by/group_aggregate into one group) has no single natural (zone, value)
            # shape to count into -- see raster_source.count_by_value's docstring for why
            # this is refused rather than guessed at. Checked against the raw band count
            # from `row_indices` (not `n_bands`, which reflects the post-group_aggregate
            # collapsed count and would wrongly admit a group_by that collapses several raw
            # bands down to one group) -- `group_by`/`group_aggregate`'s own collapse logic
            # is never applied by `raster_source.count_by_value`, so a caller combining it
            # with count_by_value must be rejected outright rather than have it silently
            # ignore every raw band but one. `row_indices` comes from `group_rows_by_path`
            # on `self.materialize()`, which only reads the band-metadata table (path/band/
            # dimensions), never pixel data -- so this check is still "before any file [of
            # pixel data] is read", not merely before the summarize starts.
            raw_band_count = sum(len(indices) for indices in row_indices)
            if raw_band_count != 1:
                self._error(
                    f"count_by_value is only supported for a single raster band per request, got {raw_band_count} -- "
                    "narrow the request (e.g. to one calendar_year) and don't combine it with group_by/group_aggregate"
                )
            self._reject_calculate_with_count_by_value()

            try:
                result_df = await asyncio.to_thread(
                    self._summarize_count_by_value_plugin, unique_paths, row_indices, bands, crosswalk_path, zone_ids
                )
            except ExpressionError as exc:
                self._error(str(exc))

            # `dim_df` can legitimately have zero columns (every dimension collapsed
            # away by group_aggregate with no group_by) -- `.row(0)` on a zero-width
            # frame raises, so there is nothing to read off it in that case.
            dim_row = dim_df.row(0, named=True) if dim_df.width else {}
            self.lf = (
                result_df.lazy()
                .with_columns(**{col: pl.lit(val) for col, val in dim_row.items()})
                .with_columns(
                    pl.col("grid_cell").cast(pl.Int32), geographic_unit=pl.lit(self.options.summarize_by)
                )
            )
            return

        # The plugin path (`scan_rasters`) applies the calculate/group-aggregate
        # pre-collapse via the shared `band_calculate_passes`/
        # `horizontal_aggregate_expr` vocabulary in raster_source.py -- see
        # `_summarize_zone_stats_plugin` -- so every summarize request, regardless of
        # which parameters are in play, reaches the same single implementation.
        result = await asyncio.to_thread(
            self._summarize_zone_stats_plugin,
            df,
            unique_paths,
            row_indices,
            bands,
            crosswalk_path,
            zone_ids,
            n_bands,
        )

        # Assemble in wide format (one column per zone) to line up with `result`'s shape,
        # then unpivot to the long-format grid_cell/value contract used everywhere else --
        # matching VectorGridAnalysis's join-based output shape.
        dim_cols = dim_df.columns
        result_frame = pl.from_numpy(result, schema=[str(zone_id) for zone_id in zone_ids], orient="row")
        # `dim_df` can legitimately have zero columns (every dimension collapsed away by
        # group_aggregate with no group_by) -- `result_frame` already has the right
        # height (`n_bands` rows) on its own in that case, so there is nothing to
        # hstack onto it, and `hstack` would fail on the height mismatch a zero-column
        # `dim_df` reports (see `_get_raster_dimension_frame`'s docstring).
        wide = dim_df.hstack(result_frame) if dim_cols else result_frame
        self.lf = (
            wide.unpivot(index=dim_cols, variable_name="grid_cell", value_name="value")
            .with_columns(pl.col("grid_cell").cast(pl.Int32), geographic_unit=pl.lit(self.options.summarize_by))
            .lazy()
        )

    def _summarize_zone_stats_plugin(self, df, unique_paths, row_indices, bands, crosswalk_path, zone_ids, n_bands):
        """Zonal aggregate via the Polars IO plugin (`raster_source.scan_rasters`).

        The single implementation of the zonal summarize step (see
        `_apply_transform_params_impl`): the calculate entries and a group_aggregate
        pre-collapse are applied here via the shared `band_calculate_passes`/
        `horizontal_aggregate_expr` vocabulary, in that order (calculate, then collapse).
        `scan_rasters` masks nodata to null against each band's own raw pixel value
        before either step sees it (`mask_nodata=True`), so a masked band's null survives
        the calculate passes untouched (their `is_not_null()` guard -- NoData in, NoData
        out) and drops out of the horizontal collapse and the final vertical
        `guarded_aggregate` alike, with no separate accumulation pass needed here.

        `where`'s value conjunct (`self._value_where`), when set, is masked in the same
        way and at the same point as nodata -- against each band's own raw pixel value,
        before any calculate entry -- via `raster_source.masked_value_predicate_exprs`.
        That ordering matters for the same reason the nodata mask has to run first: a
        FILTER value predicate is written against the stored pixel value, not a rewritten
        one (the staged flow runs filter before calculate), so masking after the
        calculate passes would test the wrong number. A pixel the predicate rejects then
        behaves exactly like a nodata pixel for everything downstream -- it drops out of
        the collapse, and out of `guarded_aggregate`'s denominator -- which is what turns
        "count contributing pixels" into "count pixels where value > x" with no separate
        code path. (A CALCULATE entry's own value condition, by contrast, tests the
        running value at that entry's position in the sequence, per the sequential
        contract -- that's `band_calculate_passes`' job, not this mask's.)

        Synchronous (called via `asyncio.to_thread`): the streaming `collect()` below is a
        long-running, CPU/IO-bound call, so it's offloaded to a worker thread rather than
        blocking the event loop for the whole summarize, matching how the numpy path's
        blocking `rio_copy` call is offloaded elsewhere (`_write_raster_vrt`).
        """
        raw_names = [f"b{i}" for i in range(len(df))]
        lf = scan_rasters(unique_paths, row_indices, bands, crosswalk_path, zone_ids, mask_nodata=True)

        if self._value_where is not None:
            lf = lf.with_columns(
                masked_value_predicate_exprs(raw_names, self._value_where, extra_columns=("grid_cell",))
            )

        for pass_exprs in self._band_calculate_passes(df, raw_names):
            lf = lf.with_columns(pass_exprs)

        if self.raster_groups is not None:
            group_row_idxs = self.raster_groups.collect()["row_idxs"].to_list()
            out_names = [f"g{k}" for k in range(len(group_row_idxs))]
            group_exprs = [
                horizontal_aggregate_expr(
                    self.options.group_aggregate,
                    [pl.col(raw_names[i]) for i in row_idxs],
                    pl.lit(None),
                ).alias(name)
                for name, row_idxs in zip(out_names, group_row_idxs)
            ]
            lf = lf.with_columns(group_exprs)
        else:
            out_names = raw_names

        grouped = (
            lf.group_by("grid_cell")
            .agg([guarded_aggregate(self.options.summarize_aggregate, name) for name in out_names])
            .collect(engine="streaming")
        )

        result = np.full((n_bands, zone_ids.size), np.nan)
        if grouped.height:
            positions = np.searchsorted(zone_ids, grouped["grid_cell"].to_numpy().astype("int64"))
            for i, name in enumerate(out_names):
                result[i, positions] = grouped[name].to_numpy()
        return result

    def _summarize_count_by_value_plugin(self, unique_paths, row_indices, bands, crosswalk_path, zone_ids):
        """count_by_value's raster-side worker: delegates to `raster_source.count_by_value`
        (the single-band scan + per-zone categorical count), on a worker thread the same
        way `_summarize_zone_stats_plugin` is -- see that method's docstring for why (the
        streaming `collect()` underneath is long-running and CPU/IO-bound, so it must not
        block the event loop for the whole summarize).
        """
        return raster_count_by_value(
            unique_paths,
            row_indices,
            bands,
            crosswalk_path,
            zone_ids,
            value_where=self._value_where,
            max_distinct=COUNT_BY_VALUE_MAX_DISTINCT,
        )

    async def _read_raster_bands(self, df, rasters, row_indices, bands, window, out_shape=None, use_gdal_env=False):
        """Read one window's raw bands into a single ``(len(df), h, w)`` array in df row order.

        ``rasters`` holds one dataset per unique file backing ``df`` (rows sharing a
        multiband COG are grouped onto the same raster/band-list by ``group_rows_by_path``),
        with ``row_indices``/``bands`` mapping each raster back to the df rows and band
        numbers it supplies. Row order is what lets the caller line the array up with
        ``df``'s own rows (e.g. each calculate plan's per-band applicability) and with
        ``raster_groups``' row positions.

        ``use_gdal_env`` is forwarded to ``parallel_read_rasters``/``_read_window``, which enter
        ``raster_gdal_env()`` on the worker thread performing each read rather than here. Default
        ``False`` so callers (e.g. ``get_tile``) that deliberately don't want it keep working
        unchanged; ``_apply_transform_params_impl`` and ``_write_raster_windowed`` pass
        ``True``.
        """
        band_reads = await parallel_read_rasters(rasters, window, out_shape, bands=bands, use_gdal_env=use_gdal_env)
        data = np.empty((len(df), *band_reads[0].shape[1:]), dtype=band_reads[0].dtype)
        for path_row_indices, read in zip(row_indices, band_reads):
            data[path_row_indices] = read
        return data

    def _collapse_tile_bands(self, df, data, nodata):
        """Apply a tile's calculate/group_aggregate step with Polars expressions.

        Shares one operator vocabulary with the write and summarize paths
        (``raster_source.band_collapse_exprs``/``band_calculate_passes``) rather than
        reducing the array with NumPy, which is what makes tiles treat nodata the way the
        rest of the API does: masked against each band's own raw pixel value before any
        calculate entry runs, dropped from the group_aggregate, and restored as the
        sentinel only where no band contributed anything. The NumPy reduce it replaces did
        none of that -- a sentinel was scaled by any calculate multiplier and averaged in
        as if it were a real value, so one nodata band in a group pulled the whole
        rendered pixel toward -9999.

        In and out stay array plumbing: ``data[i].ravel()`` is a view of the read buffer, and
        each output column converts straight back into the ``(bands, h, w)`` shape
        ``Legend.colorize_pixels`` expects.
        """
        raw_names = [f"b{i}" for i in range(len(df))]
        group_row_idxs = (
            self.raster_groups.collect()["row_idxs"].to_list() if self.raster_groups is not None else None
        )
        out_names, passes, exprs = band_collapse_exprs(
            raw_names,
            self._band_calculate_passes(df, raw_names),
            group_row_idxs,
            self.options.group_aggregate,
            data.dtype.name,
            nodata=nodata,
        )
        frame = pl.DataFrame({name: data[i].ravel() for i, name in enumerate(raw_names)})
        for pass_exprs in passes:
            frame = frame.with_columns(pass_exprs)
        result = frame.select(exprs)
        return np.stack([result[name].to_numpy().reshape(data.shape[1:]) for name in out_names])

    @asynctiming
    async def get_tile(self, x, y, z, format, tile_size=256, **kwargs):
        """Get a map tile image."""
        tile_bounds = BoundingBox(*mercantile.xy_bounds(x, y, z))
        if self.dataset_bounds and disjoint_bounds(self.dataset_bounds, tile_bounds):
            return None

        df = self.materialize()
        unique_paths, row_indices, bands = group_rows_by_path(df)
        with ExitStack() as stack:
            inputs = [stack.enter_context(rio.open(path)) for path in unique_paths]

            if not self.dataset_bounds:
                self.dataset_bounds = BoundingBox(
                    *transform_bounds(inputs[0].crs, "EPSG:3857", *inputs[0].bounds)
                )
                if disjoint_bounds(self.dataset_bounds, tile_bounds):
                    return None

            # All inputs backing one request share the same variable and were produced by the
            # same conversion pipeline, so they share one nodata sentinel; read it from the
            # file rather than hardcoding it, matching the write and summarize paths. A
            # source declaring none still needs *some* value here, because GDAL has to put
            # something in the pixels of a tile that reach past the source's extent, and
            # those pixels genuinely are absent data -- so -9999 stands in for that case and
            # is masked below like any other sentinel. (Hardcoding -9999 on both sides, as
            # this used to, silently mismatches a source that declares anything else: its
            # real nodata would go unmasked while the fill value became a distinct second
            # sentinel.)
            nodata = inputs[0].nodata if inputs[0].nodata is not None else -9999

            # Warp each source directly onto the tile's Web Mercator pixel grid so
            # the output registers exactly with the tile, with no intermediate grid
            # or window rounding to shift pixels off their true location.
            tile_transform = from_bounds(*tile_bounds, tile_size, tile_size)
            vrts = [
                WarpedVRT(
                    dataset,
                    crs="EPSG:3857",
                    transform=tile_transform,
                    width=tile_size,
                    height=tile_size,
                    nodata=nodata,
                    add_alpha=False,
                    src_nodata=nodata,
                    dtype=dataset.dtypes[0],
                    resampling=Resampling.nearest,
                )
                for dataset in inputs
            ]

            # Each VRT above is already warped directly onto the tile's own pixel grid
            # (tile_transform/tile_size), so no separate intersection window or output
            # shape needs to be computed here -- reading the whole VRT reads exactly the
            # tile, and GDAL fills anything outside the source's extent with nodata.
            data = await self._read_raster_bands(df, vrts, row_indices, bands, None)
            color = self.legend.colorize_pixels(self._collapse_tile_bands(df, data, nodata))
            # JPEG has no alpha channel, so drop it. fromarray infers RGB vs RGBA from the
            # array's shape, and passing mode= explicitly is deprecated as of Pillow 11.
            image = Image.fromarray(color[:, :, :3] if format == "jpg" else color)

            with BytesIO() as memfile:
                image.save(memfile, format=("JPEG" if format == "jpg" else "PNG"))
                return memfile.getvalue()

    @asynctiming
    async def write_raster(self, out_path, **kwargs):
        """Write the selected bands to a raster file.

        Wide selections are streamed by GDAL from a band-separate VRT
        (``_write_raster_vrt``); narrow ones read windows into NumPy
        (``_write_raster_windowed``).  The two are within noise of each other on time, so the
        choice is made on memory -- see ``_windowed_read_is_lighter``.

        Aggregated selections always take the windowed path. GDAL's builtin ``mean``/``sum``/
        ``min``/``max`` pixel functions can express the reduction and are more accurate (they
        accumulate in float64 and can propagate nodata, neither of which the NumPy reduce
        does), but collapsing bands onto one ``VRTDerivedRasterBand`` makes GDAL re-read
        every contributing band per output chunk -- for one PIXEL-interleaved multiband
        source that decompresses each tile once per band. Measured reducing 52 elev bands:
        983s vs 80s, at every swath size tried. The windowed path reads each tile once.

        Selections with calculate entries also always take the windowed path: a general
        expression (or a per-pixel value condition) has no VRT spelling, and the windowed
        path is also what enforces the calculate stage's NoData contract (mask the
        sentinel before any entry runs, restore it after -- see ``band_collapse_exprs``),
        which a VRT ScaleRatio could not.
        """

        df = self.materialize()
        if not len(df):
            return False

        if (
            self.raster_groups is not None
            or self._calculate_plans
            or not RASTER_USE_VRT_COPY
            or self._windowed_read_is_lighter(df)
        ):
            return await self._write_raster_windowed(df, out_path, **kwargs)
        return await self._write_raster_vrt(df, out_path, **kwargs)

    def _windowed_read_is_lighter(self, df):
        """Whether the windowed path would use less memory than the VRT copy for ``df``.

        The windowed path holds one ``(bands, block_height, width)`` buffer per window plus
        the read that fills it, so its peak grows with the number of selected bands; the VRT
        copy's peak is roughly flat, set by the swath and cache caps.  Measured on 52-band
        float32 elev (peak RSS, windowed vs VRT): 1 band 1027 vs 1425 MB, 5 bands 1470 vs
        1329 MB, 52 bands 7983 vs 2529 MB -- so the crossover is around 4-5 bands here, and
        expressing it as a buffer size rather than a band count keeps it right for grids and
        dtypes of other sizes.
        """

        with rio.open(df["path"][0]) as source:
            block_height = source.block_shapes[0][0]
            buffer_bytes = len(df) * block_height * source.width * np.dtype(source.dtypes[0]).itemsize
        return buffer_bytes <= RASTER_WINDOW_BUFFER_LIMIT_MB * 1024 * 1024

    def _calendar_column(self, df):
        """The calendar dimension present in ``df``, if any."""

        return next((col for col in ("calendar_year", "calendar_day") if col in df.columns), None)

    async def _write_raster_vrt(self, df, out_path, **kwargs):
        """Stream the selected bands into ``out_path`` via a band-separate VRT.

        The VRT names each selected ``(path, band)`` as one output band, so GDAL reads each
        source once and never materializes the stack -- see ``raster_copy_gdal_env`` for the
        swath/cache settings that make this pay off.  Band descriptions are applied to the
        VRT XML rather than to pixels in Python. Only plain selections reach this path --
        ``write_raster`` routes any request with calculate entries (including the legacy
        negate/multiply10/multiply100 sugar) to the windowed path, which can express a
        general elementwise rewrite and its NoData contract; a VRT ScaleRatio cannot.
        """

        sources = list(zip(df["path"].to_list(), df["band"].to_list()))
        vrt_text = vrt(sources)

        # Label output bands with their calendar value, as the windowed path does. Set
        # explicitly rather than inherited so legacy sources (no on-disk description) are
        # labeled too -- see set_vrt_band_descriptions. Skipped when aggregating, since an
        # aggregated band spans several calendar values.
        calendar_col = self._calendar_column(df)
        if calendar_col is not None:
            descriptions = [f"{calendar_col}={value}" for value in df[calendar_col].to_list()]
            vrt_text = set_vrt_band_descriptions(vrt_text, descriptions)

        with rio.open(sources[0][0]) as source:
            profile = source.profile
        creation_options = {
            "driver": "GTiff",
            "tiled": profile.get("tiled", True),
            "blockxsize": profile.get("blockxsize"),
            "blockysize": profile.get("blockysize"),
            "interleave": OUTPUT_INTERLEAVE,
            "num_threads": RASTER_NUM_THREADS,
            "bigtiff": "YES",
            **kwargs,
        }
        creation_options = {k: v for k, v in creation_options.items() if v is not None}

        # The VRT is a throwaway sidecar whose SimpleSource entries hold absolute paths, so
        # its location doesn't matter -- keep it out of the output directory.
        with tempfile.TemporaryDirectory() as tmp_dir:
            vrt_path = Path(tmp_dir) / "download.vrt"
            vrt_path.write_text(vrt_text)

            # CreateCopy is one long blocking call, unlike the windowed path's per-window
            # awaits, so it runs in the default executor to keep the event loop free for
            # other requests. GDAL config is process-global, so the env entered here
            # applies inside the worker thread.
            loop = asyncio.get_running_loop()
            with raster_copy_gdal_env():
                await loop.run_in_executor(
                    None,
                    partial(rio_copy, str(vrt_path), str(out_path), **creation_options),
                )

        return True

    async def _write_raster_windowed(self, df, out_path, **kwargs):
        """Write a raster dataset to a file using windowed reading and writing to conserve
        memory.

        The calculate/group_aggregate step is Polars expressions over a per-window
        DataFrame, built once outside the window loop below (`passes`/`exprs`) by the
        shared `raster_source.band_collapse_exprs`, so it uses the same operator
        vocabulary as the tile and summarize paths instead of a second implementation of
        the same four aggregates. Reading and writing stay array plumbing: one window's
        worth of raw bands in, one `output.write` per window out.
        """

        unique_paths, row_indices, bands = group_rows_by_path(df)
        raw_names = [f"b{i}" for i in range(len(df))]
        calculate_passes = self._band_calculate_passes(df, raw_names)

        with ExitStack() as stack:
            # GDAL_NUM_THREADS' decompression speedup only takes effect for a dataset that
            # was *opened* with it in effect (measured on real data -- see
            # _apply_transform_params_impl); entering it here, before the sources (and the
            # output, for the same reason on the write side) are opened, is what makes it
            # apply to them. Scoped narrowly to just the opens, not the whole loop below, so
            # this doesn't hold a process-global GDAL config open (and leaking into unrelated
            # concurrent requests -- see raster_gdal_env's docstring) for the entire write.
            with raster_gdal_env():
                inputs = [stack.enter_context(rio.open(path)) for path in unique_paths]

                # Each selected (path, band) row — or each aggregated group — becomes one
                # output band, so the source's own band count must be overridden.
                if self.raster_groups is not None:
                    group_row_idxs = self.raster_groups.collect()["row_idxs"].to_list()
                    out_count = len(group_row_idxs)
                else:
                    group_row_idxs = None
                    out_count = len(df)
                profile = inputs[0].profile.copy()
                # group_by/group_aggregate can collapse the matched bands into fewer output
                # bands, so the output band count can't just be copied from the (single-band)
                # source COGs' profile.
                profile.update(count=out_count, interleave=OUTPUT_INTERLEAVE, **kwargs)
                output = stack.enter_context(rio.open(out_path, "w", **profile))
                out_dtype = profile["dtype"]
                nodata = inputs[0].nodata

            # Label output bands with their calendar value when rows map 1:1 to bands.
            if self.raster_groups is None:
                calendar_col = self._calendar_column(df)
                if calendar_col is not None:
                    for band_idx, value in enumerate(df[calendar_col].to_list(), start=1):
                        output.set_band_description(band_idx, f"{calendar_col}={value}")

            # Built ONCE, outside the window loop -- the expressions are identical for
            # every window, so there is no reason to reconstruct them per iteration.
            #
            # `nodata=` is passed for an aggregated (group_aggregate) selection, so a
            # nodata pixel drops out of the collapse instead of contributing a sentinel to
            # it, and for any selection with calculate entries, so an entry's rewrite
            # never touches the sentinel (NoData in, NoData out -- the sentinel is masked
            # to null before the entries run and restored afterwards). A plain,
            # non-aggregated, non-calculated selection passes every pixel through
            # untouched -- see the pinned test_non_aggregated_output_preserves_nodata.
            # This is where the write path deliberately differs from get_tile, which masks
            # either way: a written file can carry the sentinel forward for its reader to
            # interpret, but a rendered pixel has to commit to a color.
            out_names, passes, exprs = band_collapse_exprs(
                raw_names,
                calculate_passes,
                group_row_idxs,
                self.options.group_aggregate,
                out_dtype,
                nodata=nodata if (group_row_idxs is not None or calculate_passes) else None,
            )

            # Batch size is capped independently of the block row's width -- see
            # raster_blocks_per_batch and _batched_block_windows -- so a wide raster or a
            # large band selection doesn't force every iteration to materialize a whole
            # block row's pixels regardless of how much data the request actually needs.
            # Sized off the raw (pre-collapse) band count (`len(df)`), since that is what
            # actually governs the read buffer materialized each iteration -- a
            # group_aggregate can shrink the *output* band count without shrinking how many
            # raw bands must be read to produce it.
            block_height, block_width = inputs[0].block_shapes[0]
            itemsize = np.dtype(inputs[0].dtypes[0]).itemsize
            blocks_per_batch = raster_blocks_per_batch(len(df), block_height, block_width, itemsize)

            for window in _batched_block_windows(inputs[0].block_windows(1), blocks_per_batch):
                cols, shape = await _read_window_columns(inputs, row_indices, bands, window, raw_names)
                frame = pl.DataFrame(cols)
                del cols
                for pass_exprs in passes:
                    frame = frame.with_columns(pass_exprs)
                result = frame.select(exprs)
                del frame

                # Assemble every output band and write ONCE. Writing band-at-a-time into a
                # PIXEL-interleaved tiled COG makes GDAL read-modify-write every tile once
                # per band it touches -- measured on real data: 27.8s and a 301 MB output
                # for a band-at-a-time write of one window versus 3.6s and 22.9 MB for a
                # single write of the same window's full band stack. That cost is a property
                # of the sink (GDAL's per-block path for a PIXEL-interleaved dataset), not of
                # Polars, but it is the trap a per-band port of this loop would fall straight
                # into.
                block = np.stack(
                    [result[name].to_numpy().astype(out_dtype, copy=False).reshape(shape) for name in out_names]
                )
                output.write(block, window=window)
                del result, block

        return True
