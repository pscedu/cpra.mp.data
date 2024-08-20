import asyncio
import itertools
import numpy as np
import rasterio as rio
from rasterio import windows
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
