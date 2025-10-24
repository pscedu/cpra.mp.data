import itertools
import re
from contextlib import ExitStack
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
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform

from cpra.mp.data.api.models import DatasetDimensions, FilterDimensions, Options
from cpra.mp.data.api.sources import LazyFrameSource
from cpra.mp.data.api.types import OutputType
from cpra.mp.data.api.utils import (
    asynctiming,
    get_lazy_frame,
    parallel_read_rasters,
    parallel_sample_rasters,
    unpack_values,
)
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.legends import LEGENDS, Legend
from cpra.mp.data.utils import get_input_data_frame, required_columns
from cpra.mp.data.variables import VARIABLES, Variable

DIMENSION_CHOICES = list(DIMENSIONS.keys())
SORT_CHOICES = DIMENSION_CHOICES + [f"-{dim_name}" for dim_name in DIMENSION_CHOICES]
DIMENSION_COLUMNS = cs.matches(r"^[a-z]")
VALUE_COLUMNS = cs.matches(r"^[0-9\-]")


class Dataset(LazyFrameSource):
    def __init__(
        self,
        dataset_dimensions: DatasetDimensions,
        filter_dimensions: FilterDimensions,
        options: Options,
    ):
        self.dataset_dimensions = dataset_dimensions
        self.filter_dimensions = filter_dimensions
        self.options = options
        self.dataset_bounds = None
        self.geography_ids = None
        self.summarize_geography_ids = None
        self.raster_groups = None

        self.lf = get_lazy_frame(self.dataset_dimensions, self.filter_dimensions)
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

            try:
                values = [float(val) for val in color_strs[::2]]
            except ValueError:
                self._error("invalid color map values")

            # TODO: Check that these are valid color strings.
            colors = color_strs[1::2]

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
        if self.grid.grid_type == "vector" or self.options.geography:
            return OutputType.vector
        return OutputType.raster

    @property
    def variable(self) -> Variable:
        return VARIABLES[self.dataset_dimensions.variable]

    async def _apply_filter_params(self):
        """Apply filter parameters to the LazyFrame."""
        if self.options.where:
            self.lf = self.lf.sql(f"select * from self where {self.options.where}")

        if self.options.geography:
            if self.grid.grid_type == "vector":
                await self._apply_geography_vector()
            else:
                await self._apply_geography_raster()

    async def _apply_geography_vector(self):
        """Apply the geography filter to a vector-grid dataset."""
        # TODO: Validate that the given IDs fall within the grid min and max.
        self.geography_ids = unpack_values(self.options.geography, "geography")
        self.lf = self.lf.select(cs.matches(r"^[a-z]"), *self.geography_ids)

    async def _apply_geography_raster(self):
        """Apply the geography filter to a raster-grid dataset."""
        # Check that coordinates are numeric and that there are an even number.
        try:
            coords = [float(coord) for coord in unpack_values(self.options.geography, "geography")]
        except ValueError:
            raise HTTPException(status_code=400, detail="geography contains non-numeric coordinates")

        coords_len = len(coords)
        if coords_len % 2:
            raise HTTPException(
                status_code=400,
                detail="geography contain an even number of coordinates",
            )

        # Extract lng,lat pairs.
        lngs = []
        lats = []
        for start_idx in range(0, coords_len, 2):
            lng, lat = coords[start_idx : (start_idx + 2)]
            if abs(lng) > 180 or abs(lat) > 90:
                msg = f"geography contains invalid coordinate pair: {lng}, {lat}"
                raise HTTPException(status_code=400, detail=msg)
            lngs.append(lng)
            lats.append(lat)

        # Transform the coordinates to the correct CRS, and sample the rasters in parallel.
        df = self.materialize()
        with ExitStack() as stack:
            datasets = [stack.enter_context(rio.open(path)) for path in df["path"]]
            src_crs = CRS.from_epsg(4326)
            dst_crs = datasets[0].crs
            projected_pairs = list(zip(*transform(src_crs, dst_crs, lngs, lats)))
            samples = await parallel_sample_rasters(datasets, projected_pairs)

        # Add the resulting values to the LazyFrame, and remove the path column.
        columns = [np.concatenate(col) for col in zip(*samples)]
        column_labels = [f"{x!s},{y!s}" for x, y in zip(lngs, lats)]
        self.lf = df.with_columns(**dict(zip(column_labels, columns))).select(cs.exclude("path")).lazy()

    def _apply_calculate_params(self):
        """Apply calculate parameters to the LazyFrame."""

        if self.output_type == OutputType.raster:
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
                include_geography=True,
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
        if self.output_type == OutputType.vector:
            # Set dimension columns to null if they are not part of the group by clause.
            required_cols = required_columns(get_input_data_frame(**dict(self.dataset_dimensions)))
            null_cols = [col for col in required_cols if col not in group_by_cols]

            self.lf = getattr(
                self.lf.select(pl.exclude(null_cols)).group_by(*group_by_cols),
                self.options.group_aggregate,
            )()

            if null_cols:
                self.lf = self.lf.with_columns(*[pl.lit(None).alias(col) for col in null_cols])

        else:
            self.raster_groups = self.lf.group_by(*group_by_cols).agg(row_idxs=pl.col("path").agg_groups())

    def _apply_transform_params(self):
        """Apply transform and extract parameters to the LazyFrame or image."""
        if self.output_type == OutputType.vector:
            self._apply_transform_params_vector()
        else:
            self._apply_transform_params_raster()
        self._apply_unpivot_param()

    def _apply_transform_params_vector(self):
        """Apply transform and extract parameters to the LazyFrame."""

        if self.options.summarize_by is None and self.options.summarize_aggregate is None:
            return

        if self.options.summarize_by is None or self.options.summarize_aggregate is None:
            msg = "summarize_by and summarize_aggregate parameters must be used together"
            raise HTTPException(status_code=400, detail=msg)

        # Extract and validate summary geography IDs, if present.
        if self.options.summarize_geography:
            # summarize_grid = GRIDS[self.options.summarize_by]
            # TODO: Validate that the given IDs fall within the grid min and max.
            self.summarize_geography_ids = [
                int(geo_id) for geo_id in unpack_values(self.options.summarize_geography, "summarize_geography")
            ]

        # Generate column expressions.
        crosswalk_name = f"{self.dataset_dimensions.grid}__{self.options.summarize_by}.parquet"
        crosswalk_path = Path(__file__).parent.parent / "crosswalk" / crosswalk_name

        try:
            crosswalk_lf = pl.scan_parquet(str(crosswalk_path))
        except FileNotFoundError:
            msg = f"No crosswalk from {self.grid} to {self.options.summarize_by}"
            raise HTTPException(status_code=400, detail=msg)

        if self.geography_ids:
            crosswalk_lf = crosswalk_lf.filter(pl.col("src_id").is_in(self.geography_ids))

        if self.summarize_geography_ids:
            crosswalk_lf = crosswalk_lf.filter(pl.col("dst_id").is_in(self.summarize_geography_ids))

        crosswalk_df = (
            crosswalk_lf.select("dst_id", summarize_columns=pl.struct("src_id", "scale_factor"))
            .group_by(pl.col("dst_id"))
            .agg("summarize_columns")
            .sort("dst_id")
            .collect()
        )

        agg_function = getattr(pl, f"{self.options.summarize_aggregate}_horizontal")
        self.lf = self.lf.select(
            DIMENSION_COLUMNS,
            **{
                str(row["dst_id"]): agg_function(
                    *[pl.col(str(col["src_id"])) * col["scale_factor"] for col in row["summarize_columns"]]
                )
                for row in crosswalk_df.iter_rows(named=True)
            },
        ).with_columns(geographic_unit=pl.lit(self.options.summarize_by))

    def _apply_transform_params_raster(self):
        """Apply transform and extract parameters to the image."""

    def _apply_unpivot_param(self):
        """Apply the unpivot parameter."""

        if self.options.unpivot:
            self.lf = self.lf.unpivot(index=DIMENSION_COLUMNS, variable_name="geography")

    async def apply_params(self):
        """Apply the query parameters to the LazyFrame."""
        await self._apply_filter_params()
        self._apply_calculate_params()
        self._apply_group_params()
        self._apply_transform_params()
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
            elif dimension.values:
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
        if self.geography_ids is not None:
            columns = len(self.geography_ids)
        else:
            grid = GRIDS[self.dataset_dimensions.grid]
            columns = getattr(grid, "id_count", None)

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

    async def get_raster_array(self, df, rasters, window, out_shape=None):
        """Read a raster array and apply the operations specified
        in the index DataFrame."""
        data = np.vstack(await parallel_read_rasters(rasters, window, out_shape))

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
        with ExitStack() as stack:
            inputs = [stack.enter_context(rio.open(path)) for path in df["path"]]
            vrts = [
                WarpedVRT(
                    dataset,
                    crs="EPSG:3857",
                    nodata=-9999,
                    add_alpha=False,
                    src_nodata=-9999,
                    dtype=dataset.dtypes[0],
                    resampling=Resampling.nearest,
                )
                for dataset in inputs
            ]

            if not self.dataset_bounds:
                self.dataset_bounds = vrts[0].bounds
                if disjoint_bounds(self.dataset_bounds, tile_bounds):
                    return None

            read_bounds = BoundingBox(
                max(tile_bounds.left, self.dataset_bounds.left),
                max(tile_bounds.bottom, self.dataset_bounds.bottom),
                min(tile_bounds.right, self.dataset_bounds.right),
                min(tile_bounds.top, self.dataset_bounds.top),
            )
            scale = tile_size / abs(tile_bounds.right - tile_bounds.left)
            out_shape = (
                1,
                round(scale * abs(read_bounds.top - read_bounds.bottom)),
                round(scale * abs(read_bounds.right - read_bounds.left)),
            )
            window = vrts[0].window(*read_bounds)

            data = await self.get_raster_array(df, vrts, window, out_shape)
            color = self.legend.colorize_pixels(data)
            image = Image.fromarray(
                color[:, :, :3] if format == "jpg" else color,
                mode=("RGB" if format == "jpg" else "RGBA"),
            )

            # If the image is smaller than the tile size, pad it with empty pixels.
            if image.size != (tile_size, tile_size):
                left_offset = round(abs(read_bounds.left - tile_bounds.left) * scale)
                top_offset = round(abs(tile_bounds.top - read_bounds.top) * scale)
                padded_image = Image.new(image.mode, (tile_size, tile_size))
                padded_image.paste(image, (left_offset, top_offset), mask=image)
                image = padded_image

            with BytesIO() as memfile:
                image.save(memfile, format=("JPEG" if format == "jpg" else "PNG"))
                return memfile.getvalue()

    @asynctiming
    async def write_raster(self, out_path, **kwargs):
        """Write a raster dataset to a file using windowed reading and writing
        to conserve memory."""

        df = self.materialize()
        if not len(df):
            return False

        with ExitStack() as stack:
            inputs = [stack.enter_context(rio.open(path)) for path in df["path"]]
            profile = inputs[0].profile.copy()
            profile.update(kwargs)
            output = stack.enter_context(rio.open(out_path, "w", **profile))

            # Group raster blocks by row, and read an entire row of blocks at once.
            groups = itertools.groupby(inputs[0].block_windows(1), key=lambda bw: bw[0][0])
            for _group_row, group in groups:
                window = windows.union(*[w for ji, w in group])
                data = await self.get_raster_array(df, inputs, window)
                output.write(data, window=window)

        return True
