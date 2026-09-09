import asyncio
import itertools
import re
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
from fastapi import HTTPException
from PIL import Image
from rasterio import windows
from rasterio.coords import BoundingBox, disjoint_bounds
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from cpra.mp.data import config
from cpra.mp.data.api.geography import (
    parse_coordinate_pairs,
    project_coordinates,
    resolve_vector_grid_cell,
)
from cpra.mp.data.api.models import DatasetDimensions, FilterDimensions, Options
from cpra.mp.data.api.raster_source import (
    NUMPY_TO_POLARS,
    guarded_aggregate,
    horizontal_aggregate_expr,
    scan_rasters,
)
from cpra.mp.data.api.sources import LazyFrameSource
from cpra.mp.data.api.types import DataFormat, OutputType
from cpra.mp.data.api.utils import (
    _read_window,
    asynctiming,
    get_lazy_frame,
    group_rows_by_path,
    parallel_read_rasters,
    parallel_sample_rasters,
    raster_blocks_per_batch,
    raster_copy_gdal_env,
    raster_gdal_env,
    unpack_values,
)
from cpra.mp.data.config import (
    RASTER_NUM_THREADS,
    RASTER_USE_IO_PLUGIN,
    RASTER_USE_VRT_COPY,
    RASTER_WINDOW_BUFFER_LIMIT_MB,
)
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.legends import LEGENDS, Legend
from cpra.mp.data.raster import scale_vrt_bands, set_vrt_band_descriptions, vrt
from cpra.mp.data.utils import get_input_data_frame, required_columns
from cpra.mp.data.variables import VARIABLES, Variable

DIMENSION_CHOICES = list(DIMENSIONS.keys())
SORT_CHOICES = DIMENSION_CHOICES + [f"-{dim_name}" for dim_name in DIMENSION_CHOICES]
DIMENSION_COLUMNS = cs.matches(r"^[a-z]") - cs.by_name("grid_cell", "value")
VALUE_COLUMNS = cs.by_name("value")

# Interleaving for downloaded rasters, set rather than inherited from the source. A source
# COG may be written INTERLEAVE=TILE, which is a COG-driver-only option that the GTiff driver
# used here rejects outright ("INTERLEAVE=tile unsupported, value must be PIXEL or BAND"), so
# inheriting it makes such a source undownloadable. PIXEL is also what keeps a multiband
# response small: the same 52-band elev selection wrote 1.65 GB pixel-interleaved versus
# 19.3 GB band-interleaved, since only pixel interleaving puts each pixel's bands adjacent
# for the compression predictor.
OUTPUT_INTERLEAVE = "PIXEL"

# Zonal summarize aggregates, as (reducing ufunc, identity element) pairs. Every aggregate
# reduces the same way; the identity element is what a nodata pixel is substituted with so
# it drops out of the reduction (0 leaves sum/mean unchanged, +-inf leave min/max
# unchanged). `mean` additionally divides by the per-band, per-zone count of pixels that
# actually contributed -- see `_accumulate_zone_stats` -- not the raw pixel count that
# `_get_zone_segments` returns, which counts nodata pixels too.
RASTER_AGGREGATES = {
    "max": (np.maximum, -np.inf),
    "mean": (np.add, 0.0),
    "min": (np.minimum, np.inf),
    "sum": (np.add, 0.0),
}


def _get_zone_segments(zones, zone_ids):
    """Sort a block of crosswalk zone IDs into contiguous runs of same-zone pixels.

    Returns the positions in `zone_ids` of the zones present in the block, the block's
    flat pixel indices ordered by zone, the offset at which each zone's run of pixels
    begins, and the number of pixels in each run. Pixels belonging to a zone that isn't
    being summarized, including nodata pixels, are dropped.
    """
    flat_zones = zones.ravel()
    positions = np.searchsorted(zone_ids, flat_zones).clip(max=zone_ids.size - 1)
    pixels = np.flatnonzero(zone_ids[positions] == flat_zones)
    pixels = pixels[np.argsort(positions[pixels], kind="stable")]
    present, starts, counts = np.unique(positions[pixels], return_index=True, return_counts=True)
    return present, pixels, starts, counts


def _accumulate_zone_stats(data, zones, zone_ids, nodata, aggregate_fn, identity, accumulator, valid):
    """Fold one window's pixels into `accumulator`/`valid` for one zonal aggregate.

    Nodata pixels (`data == nodata`) are excluded from every aggregate: substituted with
    the aggregate's identity element so the `reduceat` reduction is unaffected by their
    presence, and left out of `valid`, the per-band, per-zone count of pixels that actually
    contributed. That count is `mean`'s correct denominator (not the raw pixel count, which
    is wrong the moment any pixel is excluded) and also marks an all-nodata (band, zone)
    cell, which `_apply_transform_params_raster` turns into `NaN`.

    `valid` is per-band, not per-zone, because one band can be entirely nodata in a zone
    where another band has values -- a per-zone marker would wrongly blank out the other
    band's real value.

    `nodata=None` (no sentinel declared for this raster) treats every matched pixel as
    contributing, i.e. no masking, matching the pre-fix behavior for such rasters.
    """
    positions, pixels, starts, _pixel_counts = _get_zone_segments(zones, zone_ids)
    if positions.size == 0:
        return

    for band_idx, band in enumerate(data):
        vals = band.reshape(-1)[pixels].astype("float64", copy=False)
        ok = np.ones(vals.shape, dtype=bool) if nodata is None else (vals != nodata)
        masked = np.where(ok, vals, identity)
        accumulator[band_idx, positions] = aggregate_fn(
            accumulator[band_idx, positions], aggregate_fn.reduceat(masked, starts)
        )
        valid[band_idx, positions] += np.add.reduceat(ok.astype("float64"), starts)


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


class Dataset(LazyFrameSource):
    def __init__(
        self,
        dataset_dimensions: DatasetDimensions,
        filter_dimensions: FilterDimensions,
        options: Options,
        base_path=None,
    ):
        self.dataset_dimensions = dataset_dimensions
        self.filter_dimensions = filter_dimensions
        self.options = options
        self.dataset_bounds = None
        self.grid_cell_ids = None
        self.summarize_grid_cell_ids = None
        self.raster_groups = None

        self.lf = get_lazy_frame(self.dataset_dimensions, self.filter_dimensions, base_path)
        self.grid = GRIDS[self.dataset_dimensions.grid]

    def _error(self, msg, status_code=400):
        """Raise an exception."""
        raise HTTPException(status_code=status_code, detail=msg)

    @property
    def legend(self) -> Legend:
        color_map = self.options.color_map or ""

        if ":" in color_map:
            map_type, color_str = color_map.split(":", 1)

            if map_type not in ["continuous", "discrete"]:
                self._error(f"invalid color map type: {map_type}")

            color_strs = color_str.split(",")
            if map_type == "continuous":
                colors = color_strs[1::2]
                value_strs = color_strs[::2]
            else:
                colors = color_strs[::2]
                value_strs = color_strs[1::2]

            # TODO: Check that the color strings are valid.

            try:
                values = [float(val) for val in value_strs]
            except ValueError:
                self._error("invalid color map values")

            # Add a null upper bound value to discrete color maps.
            if map_type == "discrete":
                values += [None]

            if len(values) != len(colors):
                self._error("mismatched colors and values in color map")

            return Legend("custom", pl.DataFrame(dict(color=colors, upper_bound=values)), type=map_type)

        return LEGENDS.get(self.options.color_map) or self.variable.legend or LEGENDS["default"]

    @property
    def output_type(self) -> OutputType:
        """Get the output type for this dataset based on its parameters."""
        if self.grid.grid_type == "vector" or self.options.grid_cell or self.options.summarize_by:
            return OutputType.vector
        return OutputType.raster

    @property
    def variable(self) -> Variable:
        return VARIABLES[self.dataset_dimensions.variable]

    def tabular_kwargs(self, format):
        """Round CSV output to the variable's `float_precision`, when it sets one.

        CSV only: it's the one writer polars gives a `float_precision`, and the only format
        here that renders floats as text at all -- the binary formats hand over the stored
        value and leave rounding to whatever reads them.

        This deliberately applies to every float column in the response, not just `value`.
        Nothing else in a tabular response is a float today (dimensions are integers,
        strings, or dates), and a bare `.round()` on `value` would be wrong anyway: it
        would change the number in the binary formats too, where the request asked for the
        stored value.
        """
        precision = self.variable.float_precision
        if format == DataFormat.csv and precision is not None:
            return {"float_precision": precision}
        return {}

    def _is_raster_metadata_table(self):
        """Whether self.lf still holds the raster band-metadata table (dimension columns
        and a path column), as opposed to a tabular result already derived from raster
        pixels (via grid_cell sampling or summarize_by). Unlike `output_type`, which
        reflects the dataset's final response shape and is stable for the whole request,
        this reflects self.lf's shape at the point it's called, since summarize_by's
        raster-to-tabular conversion doesn't happen until the transform stage.
        """
        return self.grid.grid_type == "raster" and "path" in self.lf.collect_schema().names()

    async def _apply_filter_params(self):
        """Apply filter parameters to the LazyFrame."""
        if self.options.where:
            self.lf = self.lf.sql(f"select * from self where {self.options.where}")

        if self.options.grid_cell:
            if self.grid.grid_type == "vector":
                await self._apply_grid_cell_vector()
            else:
                await self._apply_grid_cell_raster()

    async def _apply_grid_cell_vector(self):
        """Apply the grid_cell filter to a vector-grid dataset.

        Accepts either grid cell IDs or lat/lng coordinate pairs, matching the raster
        convention. A grid_cell value is treated as IDs only if every token is a plain
        (unsigned, non-decimal) digit string; otherwise the whole list is parsed as
        coordinate pairs and resolved to grid cell IDs via point-in-polygon lookup
        against the grid's geometry. Coordinates that don't fall inside any polygon are
        dropped rather than erroring.

        Either way the result is reported under grid cell IDs, never the coordinates that
        selected them: a vector grid_cell *is* its ID. Several coordinates landing in one
        cell therefore select it once.
        """
        values = unpack_values(self.options.grid_cell, "grid_cell")

        if all(value.isdigit() for value in values):
            # TODO: Validate that the given IDs fall within the grid min and max.
            grid_cell_ids = values
        else:
            lngs, lats = parse_coordinate_pairs(values)
            resolved = await resolve_vector_grid_cell(self.grid.name, lngs, lats)
            grid_cell_ids = [str(geo_id) for geo_id in resolved if geo_id is not None]

        self.grid_cell_ids = list(dict.fromkeys(grid_cell_ids))
        self.lf = self.lf.filter(pl.col("grid_cell").is_in([int(g) for g in self.grid_cell_ids]))

    async def _apply_grid_cell_raster(self):
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

    def _apply_calculate_params(self):
        """Apply calculate parameters to the LazyFrame."""

        if self._is_raster_metadata_table():
            self.lf = self.lf.with_columns(multiplier=pl.lit(1))
            value_cols = pl.col("multiplier")
        else:
            value_cols = VALUE_COLUMNS

        if self.options.negate is None and self.options.multiply10 is None and self.options.multiply100 is None:
            return

        sql_expressions = [
            self.options.negate,
            self.options.multiply10,
            self.options.multiply100,
        ]
        factors = [-1, 10, 100]

        when_expression = pl
        for sql_expression, factor in zip(sql_expressions, factors):
            if sql_expression is None:
                continue
            when_expression = when_expression.when(pl.sql_expr(sql_expression)).then(value_cols * factor)

        when_expression = when_expression.otherwise(value_cols)
        self.lf = self.lf.with_columns(when_expression)

    def _apply_sort_params(self):
        """Apply sort and select parameters to the LazyFrame."""
        if self.options.sort is not None:
            parts = unpack_values(self.options.sort, "sort", SORT_CHOICES)
            cols = [part.replace("-", "") for part in parts]
            desc = [part.startswith("-") for part in parts]
            self.lf = self.lf.sort(cols, descending=desc)

        if self.options.select is not None:
            required_cols = required_columns(
                get_input_data_frame(**dict(self.dataset_dimensions)),
                include_grid_cell=True,
                include_value=True,
            )
            select_cols = unpack_values(self.options.select, "select", required_cols)
            self.lf = self.lf.select(select_cols)

    def _get_group_by_columns(self):
        """Get a list of group_by column names."""
        if self.options.group_by is not None and self.options.group_aggregate is None:
            msg = "group_aggregate must be specified when group_by is used"
            raise HTTPException(status_code=400, detail=msg)

        if self.options.group_by is None:
            return [True]
        return unpack_values(self.options.group_by, "group_by", DIMENSION_CHOICES)

    def _apply_group_params(self):
        """Apply group and aggregate parameters to the LazyFrame or image."""
        if self.options.group_by is None and self.options.group_aggregate is None:
            return

        group_by_cols = self._get_group_by_columns()
        if self._is_raster_metadata_table():
            # agg_groups() is deprecated and panics when grouping on a literal (the
            # group_aggregate-without-group_by case, group_by_cols == [True]), so track
            # row positions via with_row_index instead.
            self.raster_groups = (
                self.lf.with_row_index("row_idxs").group_by(*group_by_cols).agg(pl.col("row_idxs"))
            )
        else:
            # Set dimension columns to null if they are not part of the group by clause.
            required_cols = required_columns(get_input_data_frame(**dict(self.dataset_dimensions)))
            null_cols = [col for col in required_cols if col not in group_by_cols]

            # grid_cell must always be part of the group key in long format since each
            # row represents a single grid_cell.
            self.lf = self.lf.group_by(*group_by_cols, "grid_cell").agg(
                getattr(pl.col("value"), self.options.group_aggregate)()
            )

            if null_cols:
                self.lf = self.lf.with_columns(*[pl.lit(None).alias(col) for col in null_cols])

    async def _apply_transform_params(self):
        """Apply transform and extract parameters to the LazyFrame or image."""
        if self._is_raster_metadata_table():
            await self._apply_transform_params_raster()
        else:
            self._apply_transform_params_vector()
        self._apply_unpivot_param()

    def _validate_summarize_params(self):
        """Validate summarize parameters and parse summary grid_cell IDs, if present.

        Returns True if summarize_by/summarize_aggregate were provided, False if neither was.
        """
        if self.options.summarize_by is None and self.options.summarize_aggregate is None:
            return False

        if self.options.summarize_by is None or self.options.summarize_aggregate is None:
            msg = "summarize_by and summarize_aggregate parameters must be used together"
            raise HTTPException(status_code=400, detail=msg)

        # Extract and validate summary grid_cell IDs, if present.
        if self.options.summarize_grid_cell:
            # summarize_grid = GRIDS[self.options.summarize_by]
            # TODO: Validate that the given IDs fall within the grid min and max.
            self.summarize_grid_cell_ids = [
                int(geo_id) for geo_id in unpack_values(self.options.summarize_grid_cell, "summarize_grid_cell")
            ]

        return True

    def _apply_transform_params_vector(self):
        """Apply transform and extract parameters to the LazyFrame."""

        if not self._validate_summarize_params():
            return

        # Generate column expressions.
        crosswalk_name = f"{self.grid.name}__{self.options.summarize_by}.parquet"
        crosswalk_path = config.CROSSWALKS_PATH / crosswalk_name

        # scan_parquet is lazy and won't raise for a missing file until collection,
        # so check up front to report the missing crosswalk as a 400.
        if not crosswalk_path.exists():
            msg = f"No crosswalk from {self.grid.name} to {self.options.summarize_by}"
            raise HTTPException(status_code=400, detail=msg)

        crosswalk_lf = pl.scan_parquet(str(crosswalk_path))

        if self.grid_cell_ids:
            crosswalk_lf = crosswalk_lf.filter(pl.col("src_id").is_in([int(g) for g in self.grid_cell_ids]))

        if self.summarize_grid_cell_ids:
            crosswalk_lf = crosswalk_lf.filter(pl.col("dst_id").is_in(self.summarize_grid_cell_ids))

        # Join the data on grid_cell == src_id, scale the values, then aggregate
        # per destination grid_cell. The join replaces the wide-format horizontal
        # aggregation that was used when each source grid_cell was a column.
        dim_cols = cs.expand_selector(self.lf, DIMENSION_COLUMNS)
        agg_fn = self.options.summarize_aggregate
        self.lf = (
            self.lf.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner")
            .with_columns(value=pl.col("value") * pl.col("scale_factor"))
            .group_by(*dim_cols, pl.col("dst_id").alias("grid_cell"))
            .agg(getattr(pl.col("value"), agg_fn)())
            .with_columns(geographic_unit=pl.lit(self.options.summarize_by))
        )

    def _get_raster_dimension_frame(self, df):
        """Get a DataFrame with one row per output raster band, aligned by row order
        with the pixel array `get_raster_array` returns for the same `df`."""
        if self.raster_groups is None:
            return df.select(pl.exclude("path", "multiplier"))

        # Mirror the vector group-by branch: dimension columns that aren't part of the
        # group by clause collapse into a single group, so they become null here too.
        required_cols = required_columns(get_input_data_frame(**dict(self.dataset_dimensions)))
        group_df = self.raster_groups.collect().select(pl.exclude("row_idxs"))
        null_cols = [col for col in required_cols if col not in group_df.columns]
        if null_cols:
            group_df = group_df.with_columns(*[pl.lit(None).alias(col) for col in null_cols])
        return group_df

    async def _apply_transform_params_raster(self):
        """Apply transform and extract parameters to the image.

        Summarizes raster pixels by geographic unit (e.g. hydro compartment) using a
        pre-generated zone-ID crosswalk raster (see
        `python -m cpra.mp.data.cli crosswalk`), converting the result into a tabular
        LazyFrame in the same shape produced by `_apply_transform_params_vector`
        (dimension columns, a `grid_cell` column holding the destination grid_cell ID, a
        `value` column, and a `geographic_unit` column).

        The actual per-zone reduction is computed by one of two interchangeable
        implementations -- see `_summarize_zone_stats_numpy` and
        `_summarize_zone_stats_plugin` -- both returning a `(n_bands, zone_ids.size)`
        array in the same shape, so the assembly below doesn't need to know which one ran.
        """
        if not self._validate_summarize_params():
            return

        crosswalk_name = f"{self.grid.name}__{self.options.summarize_by}.tif"
        crosswalk_path = config.CROSSWALKS_PATH / crosswalk_name
        if not crosswalk_path.exists():
            msg = f"No crosswalk from {self.grid.name} to {self.options.summarize_by}"
            raise HTTPException(status_code=400, detail=msg)

        df = self.materialize()
        if df.is_empty():
            raise HTTPException(status_code=404, detail="Dataset does not exist")

        dim_df = self._get_raster_dimension_frame(df)
        n_bands = len(dim_df)
        unique_paths, row_indices, bands = group_rows_by_path(df)

        with rio.open(crosswalk_path) as crosswalk:
            if self.summarize_grid_cell_ids:
                zone_ids = np.array(sorted(self.summarize_grid_cell_ids), dtype="int64")
            else:
                dst_ids = crosswalk.tags()["dst_ids"].split(",")
                zone_ids = np.array(sorted(int(dst_id) for dst_id in dst_ids), dtype="int64")

        # The plugin path (`scan_rasters`) now applies the same multiplier/group-aggregate
        # pre-collapse as the numpy path's `get_raster_array`, via the shared
        # `horizontal_aggregate_expr`/`HORIZONTAL_RASTER_AGGREGATES` vocabulary in
        # raster_source.py -- see `_summarize_zone_stats_plugin` -- so eligibility depends
        # only on the flag, not on which parameters are in play.
        plugin_eligible = RASTER_USE_IO_PLUGIN

        if plugin_eligible:
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
        else:
            result = await self._summarize_zone_stats_numpy(
                df, unique_paths, row_indices, bands, crosswalk_path, zone_ids, n_bands
            )

        # Assemble in wide format (one column per zone) to line up with `result`'s shape,
        # then unpivot to the long-format grid_cell/value contract used everywhere else --
        # matching _apply_transform_params_vector's join-based output shape.
        dim_cols = dim_df.columns
        self.lf = (
            dim_df.hstack(pl.from_numpy(result, schema=[str(zone_id) for zone_id in zone_ids], orient="row"))
            .unpivot(index=dim_cols, variable_name="grid_cell", value_name="value")
            .with_columns(pl.col("grid_cell").cast(pl.Int32), geographic_unit=pl.lit(self.options.summarize_by))
            .lazy()
        )

    async def _summarize_zone_stats_numpy(self, df, unique_paths, row_indices, bands, crosswalk_path, zone_ids, n_bands):
        """Zonal aggregate via the batched block-window / masked-`reduceat` loop.

        The default path, and the only one used whenever a group_aggregate or calculate
        (multiplier) parameter is in play -- see the caller, `_apply_transform_params_raster`.
        Returns a `(n_bands, zone_ids.size)` array, NaN where a (band, zone) had no
        contributing (non-nodata) pixel.
        """
        is_mean = self.options.summarize_aggregate == "mean"
        aggregate_fn, start_value = RASTER_AGGREGATES[self.options.summarize_aggregate]

        with ExitStack() as stack:
            # GDAL_NUM_THREADS only speeds up decompression for reads on a dataset that was
            # *opened* with it in effect -- measured on real data: opening under the env then
            # reading plainly took ~5s/10 bands vs ~15s with no env anywhere, while entering the
            # env only around the later `.read()` calls (as this function used to assume) barely
            # moved the needle (~12-14s). So the env must be live here, at open time, not just
            # around the per-window reads below. Entered narrowly (only around the opens, not
            # the whole loop) since it sets a process-global GDAL config while running on this
            # coroutine's thread -- see raster_gdal_env's docstring -- so keeping it open for the
            # whole (potentially 90s+) loop would leak GDAL_NUM_THREADS/GDAL_CACHEMAX into any
            # unrelated concurrent request (e.g. a tile read) sharing the same worker thread pool
            # for that entire window; confirmed empirically that a bare open() is fast enough
            # that this leak window is negligible by comparison.
            with raster_gdal_env():
                crosswalk = stack.enter_context(rio.open(crosswalk_path))
                inputs = [stack.enter_context(rio.open(path)) for path in unique_paths]

            accumulator = np.full((n_bands, zone_ids.size), start_value)
            # Per-band, per-zone count of contributing (non-nodata) pixels -- mean's
            # denominator and the all-nodata marker. Per band rather than per zone: see
            # _accumulate_zone_stats.
            valid = np.zeros((n_bands, zone_ids.size))

            # All inputs backing one request share the same variable and are produced by
            # the same conversion pipeline, so they share one nodata sentinel; read it from
            # the file rather than hardcoding it. (The elevation sentinel is currently
            # -9999, a known data-generation bug being fixed separately to -9999000 -- this
            # needs no change either way, since it reads whatever the file declares.)
            nodata = inputs[0].nodata

            # Batch size is capped independently of the block row's width -- see
            # raster_blocks_per_batch and _batched_block_windows -- so a wide raster or a
            # large band selection doesn't force every iteration to materialize a whole
            # block row's pixels regardless of how much data the request actually needs.
            block_height, block_width = inputs[0].block_shapes[0]
            itemsize = np.dtype(inputs[0].dtypes[0]).itemsize
            blocks_per_batch = raster_blocks_per_batch(n_bands, block_height, block_width, itemsize)

            for window in _batched_block_windows(inputs[0].block_windows(1), blocks_per_batch):
                # Neither read depends on the other, so overlap the crosswalk read with
                # the (already parallel) reads of the data COGs. use_gdal_env=True enters
                # raster_gdal_env() around each read on the worker thread that performs it,
                # which is what keeps GDAL_CACHEMAX's memory bound in effect for every read
                # (measured: it must be live *at read time* to matter -- confirmed on real
                # data that opening under a capped GDAL_CACHEMAX but reading without it made no
                # difference to peak RSS, while capping it here did). This is a separate lever
                # from the open-time env above, which is what GDAL_NUM_THREADS needs instead.
                data, zones = await asyncio.gather(
                    self.get_raster_array(df, inputs, row_indices, bands, window, use_gdal_env=True),
                    asyncio.to_thread(_read_window, crosswalk, window, indexes=1, use_gdal_env=True),
                )

                # Grouping the block's pixels by zone once lets each band be reduced over
                # every zone in a single pass, and shares that grouping across the bands.
                # Nodata pixels are excluded here -- see _accumulate_zone_stats.
                _accumulate_zone_stats(data, zones, zone_ids, nodata, aggregate_fn, start_value, accumulator, valid)

        result = (
            np.divide(accumulator, valid, out=np.full_like(accumulator, np.nan), where=valid > 0)
            if is_mean
            else accumulator.copy()
        )
        # A (band, zone) with no contributing pixel has no answer -- either the zone never
        # appeared in a scanned window, or every matched pixel there was nodata.
        result[valid == 0] = np.nan
        return result

    def _summarize_zone_stats_plugin(self, df, unique_paths, row_indices, bands, crosswalk_path, zone_ids, n_bands):
        """Zonal aggregate via the Polars IO plugin (`raster_source.scan_rasters`).

        Equivalent to `_summarize_zone_stats_numpy` for every case that reaches this method
        (see `_apply_transform_params_raster`): a calculate (multiplier) parameter and a
        group_aggregate pre-collapse are applied here too, via the same
        `horizontal_aggregate_expr`/`HORIZONTAL_RASTER_AGGREGATES` vocabulary
        `Dataset.get_raster_array` uses, in the same order (multiply, then collapse).
        `scan_rasters` masks nodata to null against each band's own raw pixel value before
        either step sees it (`mask_nodata=True`), so a masked band's null survives the
        multiply and drops out of the horizontal collapse and the final vertical
        `guarded_aggregate` alike, with no separate accumulation pass needed here.

        Synchronous (called via `asyncio.to_thread`): the streaming `collect()` below is a
        long-running, CPU/IO-bound call, so it's offloaded to a worker thread rather than
        blocking the event loop for the whole summarize, matching how the numpy path's
        blocking `rio_copy` call is offloaded elsewhere (`_write_raster_vrt`).
        """
        raw_names = [f"b{i}" for i in range(len(df))]
        lf = scan_rasters(unique_paths, row_indices, bands, crosswalk_path, zone_ids, mask_nodata=True)

        multipliers = df["multiplier"].to_list()
        if all(m == 1 for m in multipliers):
            cols = [pl.col(name) for name in raw_names]
        else:
            cols = [pl.col(name) * float(m) for name, m in zip(raw_names, multipliers)]

        if self.raster_groups is not None:
            group_row_idxs = self.raster_groups.collect()["row_idxs"].to_list()
            out_names = [f"g{k}" for k in range(len(group_row_idxs))]
            group_exprs = [
                horizontal_aggregate_expr(
                    self.options.group_aggregate, [cols[i] for i in row_idxs], pl.lit(None)
                ).alias(name)
                for name, row_idxs in zip(out_names, group_row_idxs)
            ]
            lf = lf.with_columns(group_exprs)
        else:
            out_names = raw_names
            lf = lf.with_columns([c.alias(name) for c, name in zip(cols, out_names)])

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

    def _apply_unpivot_param(self):
        """Apply the unpivot parameter."""
        # Data is already in long format on disk; nothing to do here. The parameter
        # is kept for backward API compatibility.

    async def apply_params(self):
        """Apply the query parameters to the LazyFrame."""
        await self._apply_filter_params()
        self._apply_calculate_params()
        self._apply_group_params()
        await self._apply_transform_params()
        self._apply_sort_params()

    def estimate_data_size(self):
        """Estimate the size, in bytes, of data. This does not include dimension columns."""
        required_cols = required_columns(
            get_input_data_frame(**dict(self.dataset_dimensions)), include_dataset=False, include_metadata=False
        )
        group_by_cols = self._get_group_by_columns()
        has_group_by = self.options.group_aggregate is not None

        # Estimate rows.
        rows = 1
        for col in required_cols:
            # If there is a group by clause, only the group by columns contribute
            # to the number of output rows.
            if has_group_by and col not in group_by_cols:
                continue

            dimension = DIMENSIONS[col]

            value_count = None
            values = getattr(self.filter_dimensions, col)
            if values is not None:
                value_count = len(values)
            # Compare against None explicitly: dimension.values is a DataFrame, and
            # Polars raises on bool() rather than reporting emptiness.
            elif dimension.values is not None:
                value_count = len(dimension.values)
            elif col == "calendar_year":
                if self.dataset_dimensions.time_unit == "decadal":
                    value_count = 6
                else:
                    value_count = 52
            elif col == "calendar_day":
                if self.dataset_dimensions.time_unit == "monthly":
                    value_count = 52 * 12
                else:
                    value_count = 52 * 365

            if value_count is None:
                return None

            rows *= value_count

        # Estimate columns.
        columns = None
        if self.grid_cell_ids is not None:
            columns = len(self.grid_cell_ids)
        else:
            grid = GRIDS[self.dataset_dimensions.grid]
            columns = getattr(self.grid, "id_count", None)

            if columns is None:
                height = getattr(grid, "height", None)
                width = getattr(grid, "width", None)
                if height and width:
                    columns = height * width

        if columns is None:
            return None

        # Calculate bytes per value.
        variable = VARIABLES[self.dataset_dimensions.variable]
        dtype_digits = re.findall(r"\d+", str(variable.dtype))
        if len(dtype_digits) != 1:
            return None

        return rows * columns * int(dtype_digits[0]) / 8

    def validate_tile(self):
        """Validate that the tile is raster and will have a single band."""

        if self.output_type == OutputType.vector:
            raise HTTPException(
                status_code=400,
                detail="invalid output type for tile: vector",
            )

        group_lf = self.raster_groups if self.raster_groups is not None else self.lf
        groups = group_lf.select(pl.len()).collect().item()
        if groups > 1:
            raise HTTPException(
                status_code=400,
                detail=f"tile contains more than one band: {groups} bands found",
            )

    def validate_vector_tile(self):
        """Validate that the tile is a vector grid with an available PMTiles file."""
        if self.output_type != OutputType.vector:
            raise HTTPException(status_code=400, detail="invalid output type for vector tile: raster")

        grid_name = self.options.summarize_by or self.grid.name
        if not (config.TILES_PATH / f"{grid_name}.pmtiles").exists():
            raise HTTPException(status_code=404, detail=f"no tile file for grid: {grid_name}")

    @asynctiming
    async def get_vector_tile(self, x: int, y: int, z: int) -> bytes | None:
        """Get an MVT tile with dataset values injected into pre-generated geometry."""
        from cpra.mp.data.api.vector import (
            _data_to_id_value,
            inject_and_serialize,
            parse_tile,
            read_pmtile,
        )

        grid_name = self.options.summarize_by or self.grid.name
        tile = mercantile.Tile(x, y, z)

        compressed = read_pmtile(grid_name, tile)
        if compressed is None:
            return None

        # Parse once: get the VectorTile object and boundary feature IDs together.
        # The parsed object is reused for injection, avoiding a second decompress+parse.
        vt, raw_ids = parse_tile(compressed)
        if not raw_ids:
            return None

        tile_id_strs = [str(fid) for fid in raw_ids]
        if self.summarize_grid_cell_ids is not None:
            valid = {str(gid) for gid in self.summarize_grid_cell_ids}
            val_cols = [c for c in tile_id_strs if c in valid]
        elif self.grid_cell_ids is not None:
            valid = {str(gid) for gid in self.grid_cell_ids}
            val_cols = [c for c in tile_id_strs if c in valid]
        else:
            val_cols = tile_id_strs

        if not val_cols:
            return None

        tile_ints = [int(c) for c in val_cols]
        df = self.lf.filter(pl.col("grid_cell").is_in(tile_ints)).collect()
        if df.is_empty():
            return None

        id_values = _data_to_id_value(df)
        if not id_values:
            return None

        return inject_and_serialize(vt, id_values)

    async def get_raster_array(self, df, rasters, row_indices, bands, window, out_shape=None, use_gdal_env=False):
        """Read a raster array and apply the operations specified in the index DataFrame.

        ``rasters`` holds one dataset per unique file backing ``df`` (rows sharing a
        multiband COG are grouped onto the same raster/band-list by ``group_rows_by_path``),
        with ``row_indices``/``bands`` mapping each raster back to the df rows and band
        numbers it supplies.

        ``use_gdal_env`` is forwarded to ``parallel_read_rasters``/``_read_window``, which enter
        ``raster_gdal_env()`` on the worker thread performing each read rather than here. Default
        ``False`` so callers (e.g. ``get_tile``) that deliberately don't want it keep working
        unchanged; ``_apply_transform_params_raster`` and ``_write_raster_windowed`` pass
        ``True``.
        """
        band_reads = await parallel_read_rasters(rasters, window, out_shape, bands=bands, use_gdal_env=use_gdal_env)
        data = np.empty((len(df), *band_reads[0].shape[1:]), dtype=band_reads[0].dtype)
        for path_row_indices, read in zip(row_indices, band_reads):
            data[path_row_indices] = read

        # Apply multipliers from calculate parameters.
        if not (df["multiplier"] == 1).all():
            data = data * df["multiplier"].to_numpy()[..., np.newaxis, np.newaxis]

        # Apply group and aggregate parameters.
        if self.raster_groups is not None:
            aggregate_fn = getattr(np, self.options.group_aggregate)
            data = np.vstack(
                [
                    aggregate_fn(data[row_idxs, :, :], axis=0, keepdims=True)
                    for row_idxs in self.raster_groups.collect()["row_idxs"]
                ]
            )

        return data

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
                    nodata=-9999,
                    add_alpha=False,
                    src_nodata=-9999,
                    dtype=dataset.dtypes[0],
                    resampling=Resampling.nearest,
                )
                for dataset in inputs
            ]

            # Each VRT above is already warped directly onto the tile's own pixel grid
            # (tile_transform/tile_size), so no separate intersection window or output
            # shape needs to be computed here -- reading the whole VRT reads exactly the
            # tile, and GDAL fills anything outside the source's extent with nodata.
            data = await self.get_raster_array(df, vrts, row_indices, bands, None)
            color = self.legend.colorize_pixels(data)
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
        """

        df = self.materialize()
        if not len(df):
            return False

        if (
            self.raster_groups is not None
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
        swath/cache settings that make this pay off.  Band descriptions and the multiply
        parameters are applied to the VRT XML rather than to pixels in Python.
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

        # Apply multipliers from the calculate parameters. Applied to each source before
        # any aggregation below, matching the windowed path's multiply-then-reduce order.
        factors = df["multiplier"].to_list()
        if not all(factor == 1 for factor in factors):
            vrt_text = scale_vrt_bands(vrt_text, factors)

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

        The multiplier/group_aggregate step is Polars expressions over a per-window
        DataFrame rather than `get_raster_array`'s NumPy reduction, built once outside the
        window loop below (`exprs`) -- see `raster_source.horizontal_aggregate_expr` and
        `HORIZONTAL_RASTER_AGGREGATES` -- so it shares one operator vocabulary with the
        summarize plugin path (`_summarize_zone_stats_plugin`) instead of maintaining a
        second implementation of the same four aggregates. Reading and writing stay array
        plumbing: one window's worth of raw bands in, one `output.write` per window out.
        """

        unique_paths, row_indices, bands = group_rows_by_path(df)
        raw_names = [f"b{i}" for i in range(len(df))]
        multipliers = df["multiplier"].to_list()

        with ExitStack() as stack:
            # GDAL_NUM_THREADS' decompression speedup only takes effect for a dataset that
            # was *opened* with it in effect (measured on real data -- see
            # _apply_transform_params_raster); entering it here, before the sources (and the
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

            # Everything below is built ONCE, outside the window loop -- the expression is
            # identical for every window, so there is no reason to reconstruct it per
            # iteration.
            #
            # Nodata is masked -- against each band's own raw pixel value, before the
            # multiply, since scaling a nodata sentinel would stop it matching itself -- only
            # for an aggregated (group_aggregate) selection. A plain, non-aggregated
            # selection passes every pixel through untouched, matching get_raster_array's
            # own (unmasked) behavior for that case and the pinned
            # test_non_aggregated_output_preserves_nodata.
            if group_row_idxs is not None and nodata is not None:
                masked = [
                    pl.when(pl.col(name) == nodata).then(None).otherwise(pl.col(name)) for name in raw_names
                ]
            else:
                masked = [pl.col(name) for name in raw_names]

            # Multiply-then-collapse, matching get_raster_array's order. Skipped entirely
            # (rather than always multiplying by 1.0) when every multiplier is 1, so a plain
            # selection keeps its source dtype instead of being silently promoted to float --
            # matching get_raster_array's own `if not (df["multiplier"] == 1).all()` guard.
            if all(m == 1 for m in multipliers):
                scaled = masked
            else:
                scaled = [c * float(m) for c, m in zip(masked, multipliers)]

            if group_row_idxs is not None:
                # No band contributed anything to this pixel (every band in the group was
                # nodata there): write the raster's own nodata sentinel, not NaN. Unlike the
                # summarize path's Float64 result, an integer output band cannot hold NaN --
                # this is a deliberate difference from that path's NaN marker, not an
                # oversight. When the source declares no nodata at all, the masking above
                # never produces a null, so this value can never actually be selected; 0 (in
                # the output dtype) is used as a harmless placeholder for that case.
                no_contributor = (
                    pl.lit(nodata).cast(NUMPY_TO_POLARS[out_dtype])
                    if nodata is not None
                    else pl.lit(0, dtype=NUMPY_TO_POLARS[out_dtype])
                )
                agg = self.options.group_aggregate
                out_names = [f"out{k}" for k in range(len(group_row_idxs))]
                exprs = [
                    horizontal_aggregate_expr(agg, [scaled[i] for i in row_idxs], no_contributor).alias(name)
                    for name, row_idxs in zip(out_names, group_row_idxs)
                ]
            else:
                out_names = [f"out{k}" for k in range(len(scaled))]
                exprs = [c.alias(name) for c, name in zip(scaled, out_names)]

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
                result = pl.DataFrame(cols).select(exprs)
                del cols

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
