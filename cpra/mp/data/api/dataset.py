import asyncio
import itertools
import mercantile
import numpy as np
import polars.selectors as cs
import rasterio as rio
from cachetools import cached, TTLCache
from io import BytesIO
from PIL import Image
from polars.exceptions import ColumnNotFoundError
from rasterio import windows
from rasterio.coords import BoundingBox, disjoint_bounds
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from cpra.mp.data import read_data
from cpra.mp.data.grids import get_grid
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.api.types import (
    AggregateFunction,
    ColorMap,
    DimensionsQuery,
    GeographicUnit,
    PlanVersion,
    TimeUnit,
    Variable,
    VectorGeographicUnit,
)
from cpra.mp.data.api.utils import asynctiming, enum_value, timing
from contextlib import ExitStack
from fastapi import Depends, HTTPException


@cached(cache=TTLCache(maxsize=128, ttl=600))
def read_data_cached(*args, **kwargs):
    return read_data(*args, **kwargs)


def read_window(dataset, window, out_shape=None, resampling=Resampling.nearest):
    return dataset.read(window=window, out_shape=out_shape, resampling=resampling)


async def parallel_read_rasters(
    datasets, window=None, out_shape=None, resampling=Resampling.nearest
):
    loop = asyncio.get_running_loop()
    return await asyncio.gather(
        *[
            loop.run_in_executor(
                None, read_window, dataset, window, out_shape, resampling
            )
            for dataset in datasets
        ]
    )


def unpack_values(values_str, param_name, choices=None, separator=","):
    """Unpack values from a comma-separated list."""
    values = [val.strip() for val in values_str.split(separator)]

    # Validate choices.
    if choices:
        invalid_values = [val for val in values if val not in choices]
        if invalid_values:
            msg = f"Invalid values for {param_name}: {", ".join(invalid_values)}"
            raise HTTPException(status_code=400, detail=msg)

    return values


class Dataset:
    def __init__(
        self,
        plan_version: PlanVersion,
        variable: Variable,
        geographic_unit: GeographicUnit,
        time_unit: TimeUnit,
        geography: str | None = None,
        where: str | None = None,
        presort: str | None = None,
        group_by: str | None = None,
        group_aggregate: AggregateFunction | None = None,
        color_map: ColorMap | None = None,
        melt: bool = False,
        summarize_by: VectorGeographicUnit | None = None,
        summarize_aggregate: AggregateFunction | None = None,
        summarize_geography: str | None = None,
        sort: str | None = None,
        dimensions: DimensionsQuery = Depends(),
    ):
        self.plan_version = plan_version.value
        self.variable = variable.value
        self.geographic_unit = geographic_unit.value
        self.time_unit = time_unit.value
        self.geography = geography
        self.where = where
        self.presort = presort
        self.group_by = group_by
        self.group_aggregate = group_aggregate
        self.color_map = enum_value(color_map)
        self.melt = melt
        self.summarize_by = enum_value(summarize_by)
        self.summarize_aggregate = enum_value(summarize_aggregate)
        self.summarize_geography = summarize_geography
        self.sort = sort
        self.dimensions = {k: enum_value(v) for k, v in dimensions if v is not None}
        self.dataset_bounds = None

        try:
            self.lf = read_data_cached(
                plan_version=self.plan_version,
                variable=self.variable,
                geographic_unit=self.geographic_unit,
                time_unit=self.time_unit,
                **self.dimensions,
            )
        except ValidationError as err:
            # TODO: Also handle directory does not exist errors.
            raise HTTPException(status_code=422, detail=str(err))

        self.grid = get_grid(self.geographic_unit, self.plan_version)

        self._apply_filter_params()

    def _apply_filter_params(self):
        """Apply filter parameters to the LazyFrame."""
        if self.where:
            self.lf = self.lf.sql(f"select * from self where {self.where}")

        if self.geography:
            if self.grid.grid_type == "vector":
                geography_ids = unpack_values(
                    self.geography,
                    "geography",
                    choices=[str(geo_id) for geo_id in self.grid.geographies],
                )
                self.lf = self.lf.select(cs.matches(r"^[a-z]"), *geography_ids)
            else:
                pass

    def _materialize(self, method="collect", *args, **kwargs):
        """Materialize the LazyFrame with error handling."""

        try:
            return getattr(self.lf, method)(*args, **kwargs)
        except ColumnNotFoundError as err:
            raise HTTPException(status_code=400, detail=str(err))

    @asynctiming
    async def get_tile(self, x, y, z, tile_size=256, **kwargs):
        """Get a map tile image."""

        tile_bounds = BoundingBox(*mercantile.xy_bounds(x, y, z))
        if self.dataset_bounds and disjoint_bounds(self.dataset_bounds, tile_bounds):
            return None

        df = self._materialize()
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
            pixel_values = np.diff(data, axis=0)

            breaks = (-2, -1, 0, 1, 2, 3)
            colors = (
                (0, 0, 0, 255),
                (40, 40, 40, 255),
                (80, 80, 80, 255),
                (120, 120, 120, 255),
                (160, 160, 160, 255),
                (200, 200, 200, 255),
                (240, 240, 240, 255),
            )
            color_idx = np.digitize(pixel_values, breaks)
            color = np.take(colors, color_idx, axis=0)[0].astype("uint8")
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

    @timing
    def stream_tabular(self, out_path, format):
        sink_method = {
            "arrow": "sink_ipc",
            "csv": "sink_csv",
            "ndjson": "sink_ndjson",
            "parquet": "sink_parquet",
        }[format]
        # TODO: Set parquet compression and row group parameters.
        self._materialize(sink_method, out_path)

    @asynctiming
    async def write_raster(self, out_path, **kwargs):
        """Write a raster dataset to a file using windowed reading and writing
        to conserve memory."""

        df = self._materialize()
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
