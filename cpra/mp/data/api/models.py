from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model
from sqlmodel import Field, SQLModel

from cpra.mp.data.api.types import (
    AggregateFunction,
    CompressionAlgorithm,
    VectorGrid,
    dimension_type,
)
from cpra.mp.data.api.types import Grid as GridName
from cpra.mp.data.dimensions import DATASET_DIMENSIONS, DIMENSIONS

BaseDatasetDimensions = create_model(
    "BaseDatasetDimensions",
    __config__=ConfigDict(protected_namespaces=()),
    **{dim.name: (dimension_type(dim.name), None) for dim in DATASET_DIMENSIONS.values()},
)

BaseFilterDimensions = create_model(
    "BaseFilterDimensions",
    __config__=ConfigDict(protected_namespaces=()),
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
    model_config = ConfigDict(use_enum_values=True)


class FilterDimensions(BaseFilterDimensions):
    model_config = ConfigDict(use_enum_values=True)


class Options(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    grid_cell: str | None = None
    where: str | None = None
    # A predicate over VALUES rather than dimensions, unlike `where` above -- e.g.
    # "value > 3" to keep only cells/pixels above a threshold. Written once against the
    # placeholder column `value`, for both grid types (`Dataset._apply_filter_params` for
    # vector; `Dataset._summarize_zone_stats_plugin`, one raster band at a time, via
    # `raster_source.masked_value_predicate_exprs`). Parsed through
    # `api.expressions.parse_value_expr`'s restricted whitelist rather than handed to
    # `lf.sql()` unrestricted the way `where` is, since it runs on the hot per-row path an
    # aggregate or a sort inside it can force off of streaming pushdown entirely -- see
    # `api/expressions.py`'s module docstring. A non-matching value is masked to null, not
    # dropped as a row, so it composes with every aggregate's existing null-skipping
    # behaviour (including `count`, which is what makes "count pixels where value > x"
    # possible) instead of adding a second "predicate failed" code path.
    value_where: str | None = None
    negate: str | None = None
    multiply10: str | None = None
    multiply100: str | None = None
    group_by: str | None = None
    group_aggregate: AggregateFunction | None = None
    color_map: str | None = None
    unpivot: bool = False
    summarize_by: VectorGrid | None = None
    summarize_aggregate: AggregateFunction | None = None
    summarize_grid_cell: str | None = None
    sort: str | None = None
    select: str | None = None


class Compression(BaseModel):
    compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.zstd
    compression_level: int = 6


class TileQuery(Options, FilterDimensions):
    model_config = ConfigDict(use_enum_values=True)

    @property
    def filter_dimensions(self):
        return FilterDimensions(**dict(self))

    @property
    def options(self):
        return Options(**dict(self))


class DataQuery(TileQuery, Compression):
    model_config = ConfigDict(use_enum_values=True)

    @property
    def compression(self):
        return Compression(**dict(self))


class CrosswalkQuery(Compression):
    model_config = ConfigDict(use_enum_values=True)

    grid_cell: str | None = None
    to_grid: list[GridName] | None = None

    @property
    def compression(self):
        return Compression(**dict(self))


class DimensionValue(BaseModel):
    value: Any
    label: str | None
    # Set only where a value has a color of its own wherever it is drawn, which is the
    # vegetation species; null for every other dimension. Values are listed in display
    # order, so a client colors and orders a series from this list alone.
    color: str | None = None


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
    subroutine: str | None
    unit: str | None
    display_unit: str | None
    display_scale_factor: float | None
    display_number_format: str | None
    float_precision: int | None
    legend: str | None
    geographic_units: list[str] | None
    time_units: list[str] | None
    required_dimensions: list[str] | None
    difference_legend_limit: float | None


class Grid(BaseModel):
    name: str
    label: str
    grid_type: str
    crs: str
    id_min: int | None
    id_max: int | None
    id_count: int | None
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


class Visualization(BaseModel):
    name: str
    label: str
    type: str
    max_grid_cells: int
    variables: list[str] | None = None


class VisualizationQuery(DataQuery):
    end_date: date | None = None
    height: int = 400
    start_date: date | None = None
    width: int = 600

    # Map-specific parameters
    show_projects: bool = True
    projects_model_group_id: list[int] | None = None
    show_water: bool = False
    show_grid_labels: bool = False
    show_grid_lines: bool = True
    is_overlay: bool = True


class MPDDimensionValue(SQLModel, table=True):
    """Model group from the MPD."""

    __tablename__ = "vw_a_portal_dimension_value"
    dimension: str = Field(primary_key=True)
    value_int: int | None = None
    value_str: str | None = None
    label: str = Field(primary_key=True)
