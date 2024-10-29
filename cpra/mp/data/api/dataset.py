import itertools
import mercantile
import numpy as np
import polars as pl
import polars.selectors as cs
import rasterio as rio
from io import BytesIO
from pathlib import Path
from PIL import Image
from rasterio import windows
from rasterio.coords import BoundingBox, disjoint_bounds
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform
from cpra.mp.data.api.models import DatasetDimensions, FilterDimensions, Options
from cpra.mp.data.api.sources import LazyFrameSource
from cpra.mp.data.api.utils import (
    asynctiming,
    get_lazy_frame,
    parallel_read_rasters,
    parallel_sample_rasters,
    unpack_values,
)
from cpra.mp.data.api.types import OutputType
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import get_grid
from cpra.mp.data.legends import Legend, LEGENDS
from cpra.mp.data.variables import Variable, VARIABLES
from contextlib import ExitStack
from fastapi import HTTPException

DIMENSION_CHOICES = list(DIMENSIONS.keys())
SORT_CHOICES = DIMENSION_CHOICES + [f"-{dim_name}" for dim_name in DIMENSION_CHOICES]
DIMENSION_COLUMNS = cs.matches(r"^[a-z]")
VALUE_COLUMNS = cs.matches(r"^[0-9]")


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

        self.lf = get_lazy_frame(self.dataset_dimensions, self.filter_dimensions)
        self.grid = get_grid(
            self.dataset_dimensions.geographic_unit,
            self.dataset_dimensions.plan_version,
        )

    @property
    def legend(self) -> Legend:
        return (
            LEGENDS.get(self.options.color_map)
            or self.variable.legend
            or LEGENDS["default"]
        )

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
        self.geography_ids = unpack_values(
            self.options.geography,
            "geography",
            choices=[str(geo_id) for geo_id in self.grid.geographies],
        )
        self.lf = self.lf.select(cs.matches(r"^[a-z]"), *self.geography_ids)

    async def _apply_geography_raster(self):
        """Apply the geography filter to a raster-grid dataset."""
        # Check that coordinates are numeric and that there are an even number.
        try:
            coords = [
                float(coord)
                for coord in unpack_values(self.options.geography, "geography")
            ]
        except ValueError:
            raise HTTPException(
                status_code=400, detail="geography contains non-numeric coordinates"
            )

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
        column_labels = [f"{str(x)},{str(y)}" for x, y in zip(lngs, lats)]
        self.lf = (
            df.with_columns(**dict(zip(column_labels, columns)))
            .select(cs.exclude("path"))
            .lazy()
        )

    def _apply_calculate_params(self):
        """Apply calculate parameters to the LazyFrame."""
        if (
            self.options.negate is None
            and self.options.multiply10 is None
            and self.options.multiply100 is None
        ):
            return

        if self.output_type == OutputType.vector:
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
                when_expression = when_expression.when(
                    pl.sql_expr(sql_expression)
                ).then(VALUE_COLUMNS * factor)

            when_expression = when_expression.otherwise(VALUE_COLUMNS)
            self.lf = self.lf.with_columns(when_expression)

    def _apply_sort_param(self, order: str, param_name: str):
        """Apply a sort parameter to the LazyFrame."""
        if order is None:
            return

        parts = unpack_values(order, param_name, SORT_CHOICES)
        cols = [part.replace("-", "") for part in parts]
        desc = [part.startswith("-") for part in parts]
        self.lf = self.lf.sort(cols, descending=desc)

    def _apply_group_params(self):
        """Apply group and aggregate parameters to the LazyFrame or image."""
        if self.options.group_by is None and self.options.group_aggregate is None:
            return

        if self.options.group_by is None or self.options.group_aggregate is None:
            msg = "group_by and group_aggregate parameters must be used together"
            raise HTTPException(status_code=400, detail=msg)

        if self.output_type == OutputType.vector:
            group_cols = unpack_values(
                self.options.group_by, "group_by", DIMENSION_CHOICES
            )
            self.lf = getattr(
                self.lf.group_by(*group_cols), self.options.group_aggregate
            )()

    def _apply_transform_params(self):
        """Apply transform and extract parameters to the LazyFrame or image."""
        if self.output_type == OutputType.vector:
            self._apply_transform_params_vector()
        else:
            self._apply_transform_params_raster()

    def _apply_transform_params_vector(self):
        """Apply transform and extract parameters to the LazyFrame."""

        if self.options.melt:
            self.lf = self.lf.unpivot(
                index=DIMENSION_COLUMNS, variable_name="geography"
            )

        if (
            self.options.summarize_by is None
            and self.options.summarize_aggregate is None
        ):
            return

        if (
            self.options.summarize_by is None
            or self.options.summarize_aggregate is None
        ):
            msg = (
                "summarize_by and summarize_aggregate parameters must be used together"
            )
            raise HTTPException(status_code=400, detail=msg)

        # Extract and validate summary geography IDs, if present.
        if self.options.summarize_geography:
            summarize_grid = get_grid(
                self.options.summarize_by, self.dataset_dimensions.plan_version
            )
            self.summarize_geography_ids = [
                int(geo_id)
                for geo_id in unpack_values(
                    self.options.summarize_geography,
                    "summarize_geography",
                    choices=[str(geo_id) for geo_id in summarize_grid.geographies],
                )
            ]

        # Generate column expressions.
        crosswalk_name = (
            f"{self.dataset_dimensions.plan_version}__{self.dataset_dimensions.geographic_unit}__"
            f"{self.dataset_dimensions.plan_version}__{self.options.summarize_by}.parquet"
        )
        crosswalk_path = Path(__file__).parent.parent / "crosswalk" / crosswalk_name

        try:
            crosswalk_lf = pl.scan_parquet(str(crosswalk_path))
        except FileNotFoundError:
            msg = (
                f"No crosswalk from {self.geographic_unit} to {self.options.summarize_by} "
                f"for plan version {self.plan_version}"
            )
            raise HTTPException(status_code=400, detail=msg)

        if self.geography_ids:
            crosswalk_lf = crosswalk_lf.filter(
                pl.col("src_id").is_in(self.geography_ids)
            )

        if self.summarize_geography_ids:
            crosswalk_lf = crosswalk_lf.filter(
                pl.col("dst_id").is_in(self.summarize_geography_ids)
            )

        crosswalk_df = (
            crosswalk_lf.select(
                "dst_id", summarize_columns=pl.struct("src_id", "scale_factor")
            )
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
                    *[
                        pl.col(str(col["src_id"])) * col["scale_factor"]
                        for col in row["summarize_columns"]
                    ]
                )
                for row in crosswalk_df.iter_rows(named=True)
            },
        ).with_columns(geographic_unit=pl.lit(self.options.summarize_by))

    def _apply_transform_params_raster(self):
        """Apply transform and extract parameters to the image."""

    async def apply_params(self):
        """Apply the query parameters to the LazyFrame."""
        await self._apply_filter_params()
        self._apply_calculate_params()
        self._apply_group_params()
        self._apply_transform_params()
        self._apply_sort_param(self.options.sort, "sort")

    @asynctiming
    async def get_tile(self, x, y, z, tile_size=256, **kwargs):
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

            data = np.vstack(await parallel_read_rasters(vrts, window, out_shape))
            # TODO: Handle multiple raster layers.
            # pixel_values = np.diff(data, axis=0)
            color = self.legend.colorize_pixels(data)
            image = Image.fromarray(color, mode="RGBA")

            # If the image is smaller than the tile size, pad it with empty pixels.
            if image.size != (tile_size, tile_size):
                left_offset = round(abs(read_bounds.left - tile_bounds.left) * scale)
                top_offset = round(abs(tile_bounds.top - read_bounds.top) * scale)
                padded_image = Image.new(image.mode, (tile_size, tile_size))
                padded_image.paste(image, (left_offset, top_offset), mask=image)
                image = padded_image

            with BytesIO() as memfile:
                image.save(memfile, format="PNG")
                return memfile.getvalue()

    @asynctiming
    async def write_raster(self, out_path, **kwargs):
        """Write a raster dataset to a file using windowed reading and writing
        to conserve memory."""

        df = self.materialize()
        with ExitStack() as stack:
            inputs = [stack.enter_context(rio.open(path)) for path in df["path"]]
            profile = inputs[0].profile.copy()
            profile.update(kwargs)
            output = stack.enter_context(rio.open(out_path, "w", **profile))

            # Group raster blocks by row, and read an entire row of blocks at once.
            groups = itertools.groupby(
                inputs[0].block_windows(1), key=lambda bw: bw[0][0]
            )
            for _group_row, group in groups:
                window = windows.union(*[w for ji, w in group])
                data = np.vstack(await parallel_read_rasters(inputs, window))
                output.write(np.diff(data, axis=0), window=window)
