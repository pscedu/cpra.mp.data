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
        if dim.type not in ("dataset",)
    },
)


class DatasetDimensions(BaseDatasetDimensions):
    class Config:
        use_enum_values = True


class FilterDimensions(BaseFilterDimensions):
    """Structured filter parameters — used by tile endpoints and the catalog."""

    class Config:
        use_enum_values = True


class Options(BaseModel):
    """
    Query options for the data download endpoint.

    The `sql` parameter is a full SQL SELECT statement that operates on a
    pre-registered view called `data`.  It replaces the structured filter and
    transformation parameters from the previous API version.  Example::

        SELECT grid_cell, AVG(value) AS value
        FROM data
        WHERE calendar_year IN (2025, 2035, 2045)
          AND scenario_id = 1
        GROUP BY grid_cell

    When `summarize_by` is specified the server registers the relevant crosswalk
    table as a view named ``xw`` (columns: src_id, dst_id, scale_factor) so
    that users can also write the join themselves.

    When `sql` is omitted the server executes ``SELECT * FROM data``.
    """

    # Coordinate-based point extraction (raster grids only).
    # Comma-separated lng,lat pairs, e.g. "−90.5,29.3,−90.6,29.4".
    grid_cell: str | None = None

    # Full SQL SELECT operating on the 'data' view (and optionally 'xw').
    sql: str | None = None

    # Geographic upsampling via pre-built crosswalk tables.
    summarize_by: VectorGrid | None = None
    summarize_aggregate: AggregateFunction | None = None
    summarize_grid_cell: str | None = None

    # Colour ramp for raster tile rendering.
    color_map: str | None = None

    class Config:
        use_enum_values = True


class Compression(BaseModel):
    compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.zstd
    compression_level: int = 6


class TileQuery(FilterDimensions):
    """Query parameters for raster and vector tile endpoints."""

    class Config:
        use_enum_values = True

    grid_cell: str | None = None
    where: str | None = None
    negate: str | None = None
    multiply10: str | None = None
    multiply100: str | None = None
    group_by: str | None = None
    group_aggregate: AggregateFunction | None = None
    color_map: str | None = None
    summarize_by: VectorGrid | None = None
    summarize_aggregate: AggregateFunction | None = None
    summarize_grid_cell: str | None = None
    sort: str | None = None
    select: str | None = None

    @property
    def filter_dimensions(self):
        return FilterDimensions(**{
            k: v for k, v in dict(self).items()
            if k in FilterDimensions.model_fields
        })

    @property
    def options(self):
        return self


class DataQuery(Options, Compression):
    """Query parameters for the data download endpoint."""

    class Config:
        use_enum_values = True

    @property
    def compression(self):
        return Compression(
            compression_algorithm=self.compression_algorithm,
            compression_level=self.compression_level,
        )


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
    subroutine: str | None
    unit: str | None
    number_format: str | None
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
    max_geographies: int
    variables: list[str] | None = None


class VisualizationQuery(DataQuery):
    end_date: date | None = None
    height: int = 400
    start_date: date | None = None
    width: int = 600

    # Map-specific parameters
    show_projects: bool = True
    show_water: bool = False
    show_grid_labels: bool = False
    show_grid_lines: bool = True
    is_overlay: bool = True


class MPDDimensionValue(SQLModel, table=True):
    __tablename__ = "vw_a_portal_dimension_value"
    dimension: str = Field(primary_key=True)
    value_int: int | None = None
    value_str: str | None = None
    label: str = Field(primary_key=True)
