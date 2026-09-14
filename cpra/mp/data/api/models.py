from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlmodel import Field, SQLModel

from cpra.mp.data.analysis.parameters import (  # noqa: F401
    DatasetDimensions,
    FilterDimensions,
    Options,
)
from cpra.mp.data.analysis.types import (
    CompressionAlgorithm,
)
from cpra.mp.data.analysis.types import Grid as GridName


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
