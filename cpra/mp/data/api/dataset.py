import asyncio
import itertools
import mercantile
import numpy as np
import rasterio as rio
from io import BytesIO
from PIL import Image
from rasterio import windows
from rasterio.coords import BoundingBox, disjoint_bounds
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from cpra.mp.data import read_data
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
from cpra.mp.data.api.utils import enum_value, timing
from contextlib import ExitStack

from fastapi import Depends, HTTPException


def read_window(input, window):
    return input.read(window=window)


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
        self.web_mercator_bounds = None

        try:
            self.lf = read_data(
                plan_version=self.plan_version,
                variable=self.variable,
                geographic_unit=self.geographic_unit,
                time_unit=self.time_unit,
                **self.dimensions,
            )
        except ValidationError as err:
            # TODO: Also handle directory does not exist errors.
            raise HTTPException(status_code=422, detail=str(err))

    @timing
    async def get_tile(self, x, y, z, **kwargs):
        """Get a map tile image."""

        tile_bounds = BoundingBox(*mercantile.xy_bounds(x, y, z))
        if self.web_mercator_bounds and disjoint_bounds(
            self.web_mercator_bounds, tile_bounds
        ):
            return None

        df = self.lf.collect()
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

            if not self.web_mercator_bounds:
                self.web_mercator_bounds = vrts[0].bounds
                if disjoint_bounds(self.web_mercator_bounds, tile_bounds):
                    return None

            window = vrts[0].window(*tile_bounds)
            data = np.vstack(
                [vrt.read(window=window, out_shape=(1, 256, 256)) for vrt in vrts]
            )
            diff = np.diff(data, axis=0)
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
            color_idx = np.digitize(diff, breaks)
            color = np.take(colors, color_idx, axis=0)[0].astype("uint8")
            image = Image.fromarray(color, mode="RGBA")
            with BytesIO() as memfile:
                image.save(memfile, format="PNG")
                return memfile.getvalue()

    @timing
    async def write_raster(self, out_path, **kwargs):
        """Write a raster dataset to a file using windowed reading and writing
        to conserve memory."""

        df = self.lf.collect()
        with ExitStack() as stack:
            inputs = [stack.enter_context(rio.open(path)) for path in df["path"]]
            profile = inputs[0].profile.copy()
            profile.update(kwargs)
            output = stack.enter_context(rio.open(out_path, "w", **profile))

            loop = asyncio.get_running_loop()
            # Group raster blocks by row, and read an entire row of blocks at once.
            groups = itertools.groupby(
                inputs[0].block_windows(1), key=lambda bw: bw[0][0]
            )
            for _group_row, group in groups:
                window = windows.union(*[w for ji, w in group])
                data = np.vstack(
                    await asyncio.gather(
                        *[
                            loop.run_in_executor(None, read_window, input, window)
                            for input in inputs
                        ]
                    )
                )
                output.write(np.diff(data, axis=0), window=window)
