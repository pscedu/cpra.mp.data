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
from enum import Enum
from fastapi import Depends, HTTPException


def enum_value(inst):
    """Get the value from an enum."""
    return inst.value if isinstance(inst, Enum) else inst


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
