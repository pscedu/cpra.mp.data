from cpra.mp.data.dimensions import DIMENSIONS, DATASET_DIMENSIONS
from cpra.mp.data.api.types import (
    dimension_type,
    AggregateFunction,
    ColorMap,
    VectorGeographicUnit,
)
from pydantic import create_model, BaseModel, ConfigDict

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
        if dim.type != "dataset"
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
    summarize_by: VectorGeographicUnit | None = None
    summarize_aggregate: AggregateFunction | None = None
    summarize_geography: str | None = None
    sort: str | None = None

    class Config:
        use_enum_values = True


class FilterDimensionsAndOptions(Options, FilterDimensions):
    class Config:
        use_enum_values = True

    @property
    def filter_dimensions(self):
        return FilterDimensions(**dict(self))

    @property
    def options(self):
        return Options(**dict(self))
