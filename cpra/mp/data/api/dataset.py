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
from cpra.mp.data.api.sources import LazyFrameSource
from cpra.mp.data.api.types import DataFormat, OutputType
from cpra.mp.data.api.utils import (
    asynctiming,
    get_lazy_frame,
    group_rows_by_path,
    parallel_read_rasters,
    parallel_sample_rasters,
    raster_copy_gdal_env,
    raster_gdal_env,
    unpack_values,
)
from cpra.mp.data.config import (
    RASTER_NUM_THREADS,
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

# Zonal summarize aggregates, as (reducing ufunc, accumulator start value) pairs. Every
# aggregate reduces the same way; mean additionally divides by the pixel counts that
# `_get_zone_segments` returns.
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
        is_mean = self.options.summarize_aggregate == "mean"
        aggregate_fn, start_value = RASTER_AGGREGATES[self.options.summarize_aggregate]

        unique_paths, row_indices, bands = group_rows_by_path(df)

        with ExitStack() as stack:
            crosswalk = stack.enter_context(rio.open(crosswalk_path))
            inputs = [stack.enter_context(rio.open(path)) for path in unique_paths]
            if self.summarize_grid_cell_ids:
                zone_ids = np.array(sorted(self.summarize_grid_cell_ids), dtype="int64")
            else:
                dst_ids = crosswalk.tags()["dst_ids"].split(",")
                zone_ids = np.array(sorted(int(dst_id) for dst_id in dst_ids), dtype="int64")

            accumulator = np.full((n_bands, zone_ids.size), start_value)
            counts = np.zeros(zone_ids.size) if is_mean else None
            seen = np.zeros(zone_ids.size, dtype=bool)

            groups = itertools.groupby(inputs[0].block_windows(1), key=lambda bw: bw[0][0])
            for _group_row, group in groups:
                window = windows.union(*[w for ji, w in group])

                # Neither read depends on the other, so overlap the crosswalk read with
                # the (already parallel) reads of the data COGs.
                data, zones = await asyncio.gather(
                    self.get_raster_array(df, inputs, row_indices, bands, window),
                    asyncio.to_thread(crosswalk.read, 1, window=window),
                )

                # Grouping the block's pixels by zone once lets each band be reduced over
                # every zone in a single pass, and shares that grouping across the bands.
                positions, pixels, starts, pixel_counts = _get_zone_segments(zones, zone_ids)
                if positions.size == 0:
                    continue

                # Accumulate in float64, as the source pixels are often narrow integers.
                block_stats = np.vstack(
                    [
                        aggregate_fn.reduceat(band.reshape(-1)[pixels].astype("float64", copy=False), starts)
                        for band in data
                    ]
                )
                accumulator[:, positions] = aggregate_fn(accumulator[:, positions], block_stats)
                seen[positions] = True

                if is_mean:
                    counts[positions] += pixel_counts

        result = (
            np.divide(accumulator, counts, out=np.full_like(accumulator, np.nan), where=counts > 0)
            if is_mean
            else accumulator
        )
        result[:, ~seen] = np.nan

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

    async def get_raster_array(self, df, rasters, row_indices, bands, window, out_shape=None):
        """Read a raster array and apply the operations specified in the index DataFrame.

        ``rasters`` holds one dataset per unique file backing ``df`` (rows sharing a
        multiband COG are grouped onto the same raster/band-list by ``group_rows_by_path``),
        with ``row_indices``/``bands`` mapping each raster back to the df rows and band
        numbers it supplies.
        """
        band_reads = await parallel_read_rasters(rasters, window, out_shape, bands=bands)
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
        """Write a raster dataset to a file using windowed reading and writing
        to conserve memory."""

        unique_paths, row_indices, bands = group_rows_by_path(df)
        with ExitStack() as stack:
            # Entered before the sources are opened so the thread setting applies to them.
            stack.enter_context(raster_gdal_env())
            inputs = [stack.enter_context(rio.open(path)) for path in unique_paths]

            # Each selected (path, band) row — or each aggregated group — becomes one
            # output band, so the source's own band count must be overridden.
            if self.raster_groups is not None:
                out_count = self.raster_groups.select(pl.len()).collect().item()
            else:
                out_count = len(df)
            profile = inputs[0].profile.copy()
            # group_by/group_aggregate can collapse the matched bands into fewer output
            # bands, so the output band count can't just be copied from the (single-band)
            # source COGs' profile.
            profile.update(count=out_count, interleave=OUTPUT_INTERLEAVE, **kwargs)
            output = stack.enter_context(rio.open(out_path, "w", **profile))

            # Label output bands with their calendar value when rows map 1:1 to bands.
            if self.raster_groups is None:
                calendar_col = self._calendar_column(df)
                if calendar_col is not None:
                    for band_idx, value in enumerate(df[calendar_col].to_list(), start=1):
                        output.set_band_description(band_idx, f"{calendar_col}={value}")

            # Group raster blocks by row, and read an entire row of blocks at once.
            groups = itertools.groupby(inputs[0].block_windows(1), key=lambda bw: bw[0][0])
            for _group_row, group in groups:
                window = windows.union(*[w for ji, w in group])
                data = await self.get_raster_array(df, inputs, row_indices, bands, window)
                output.write(data, window=window)

        return True
