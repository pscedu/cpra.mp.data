from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.api.types import (
    dimension_type,
    AggregateFunction,
    ColorMap,
    VectorGeographicUnit,
)
from pydantic import create_model, BaseModel, ConfigDict

BaseDimensions = create_model(
    "DimensionsQuery",
    model_config=ConfigDict(protected_namespaces=()),
    **{
        dim.name: (dimension_type(dim.name), None)
        if dim.type == "dataset"
        else (
            list[dimension_type(dim.name)] | None,
            None,
        )
        for dim in DIMENSIONS.values()
    },
)


class Dimensions(BaseDimensions):
    class Config:
        use_enum_values = True


class Options(BaseModel):
    geography: str | None = (None,)
    where: str | None = (None,)
    negate: str | None = (None,)
    multiply10: str | None = (None,)
    multiply100: str | None = (None,)
    group_by: str | None = (None,)
    group_aggregate: AggregateFunction | None = (None,)
    color_map: ColorMap | None = (None,)
    melt: bool = (False,)
    summarize_by: VectorGeographicUnit | None = (None,)
    summarize_aggregate: AggregateFunction | None = (None,)
    summarize_geography: str | None = (None,)
    sort: str | None = (None,)

    class Config:
        use_enum_values = True
