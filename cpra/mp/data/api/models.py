from cpra.mp.data.dimensions import DIMENSIONS, DATASET_DIMENSIONS
from cpra.mp.data.api.types import (
    dimension_type,
    AggregateFunction,
    ColorMap,
    CompressionAlgorithm,
    VectorGrid,
)
from pydantic import create_model, BaseModel, ConfigDict
from typing import Any

BaseDatasetDimensions = create_model(
    "BaseDatasetDimensions",
    model_config=ConfigDict(protected_namespaces=()),
    **{
        dim.name: (dimension_type(dim.name), None)
        for dim in DATASET_DIMENSIONS.values()
    },
)

BaseFilterDimensions = create_model(
    "BaseFilterDimensions",
    model_config=ConfigDict(protected_namespaces=()),
    **{
        dim.name: (
            list[dimension_type(dim.name)] | None,
            None,
        )
        for dim in DIMENSIONS.values()
        if dim.type not in ("dataset", "metadata")
    },
)


class DatasetDimensions(BaseDatasetDimensions):
    class Config:
        use_enum_values = True


class FilterDimensions(BaseFilterDimensions):
    class Config:
        use_enum_values = True


class Options(BaseModel):
    geography: str | None = None
    where: str | None = None
    negate: str | None = None
    multiply10: str | None = None
    multiply100: str | None = None
    group_by: str | None = None
    group_aggregate: AggregateFunction | None = None
    color_map: ColorMap | None = None
    melt: bool = False
    summarize_by: VectorGrid | None = None
    summarize_aggregate: AggregateFunction | None = None
    summarize_geography: str | None = None
    sort: str | None = None

    class Config:
        use_enum_values = True


class Compression(BaseModel):
    compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.zstd
    compression_level: int = 6


class TileQuery(Options, FilterDimensions):
    class Config:
        use_enum_values = True

    @property
    def filter_dimensions(self):
        return FilterDimensions(**dict(self))

    @property
    def options(self):
        return Options(**dict(self))


class DataQuery(TileQuery, Compression):
    class Config:
        use_enum_values = True

    @property
    def compression(self):
        return Compression(**dict(self))


class DimensionValue(BaseModel):
    value: Any
    label: str | None


class Dimension(BaseModel):
    name: str
    label: str
    type: str
    dtype: str
    required: bool
    null: bool
    values: list[DimensionValue] | None
    min_value: Any | None
    max_value: Any | None


class Variable(BaseModel):
    name: str
    label: str
    dtype: str
    model: str
    subroutine: str
    unit: str | None
    legend: str | None
    geographic_units: list[str] | None
    time_units: list[str] | None
    required_dimensions: list[str] | None


class Grid(BaseModel):
    name: str
    grid_type: str
    crs: str
    geographies: Any | None
    bounds: list[int] | None
    height: int | None
    width: int | None


class LegendBreakPoint(BaseModel):
    label: str
    color: str
    upper_bound: int | float | None


class Legend(BaseModel):
    name: str
    break_points: list[LegendBreakPoint]
