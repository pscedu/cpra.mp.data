from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlmodel import Field, SQLModel

from cpra.mp.data.analysis.request.parameters import (
    DataOptions,
    DatasetDimensions,  # noqa: F401 -- re-exported; imported from here by cli/, visualizations/ and tests
    FilterDimensions,
)
from cpra.mp.data.analysis.request.types import (
    CompressionAlgorithm,
)
from cpra.mp.data.analysis.request.types import Grid as GridName


class Compression(BaseModel):
    compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.zstd
    compression_level: int = 6


class TileQuery(DataOptions, FilterDimensions):
    """The data/tile HTTP endpoints accept exactly dimension filters plus
    `grid_cell`/`where`/`sort`/`select`/`color_map` (and compression, on `DataQuery`
    below). The pipeline-execution parameters that used to sit alongside them
    (`group_by`/`group_aggregate`/`summarize_*`/`calculate`) are pipeline steps now, and
    sending one is rejected explicitly with the step to use instead
    (`api.__init__.reject_retired_params`) rather than silently ignored.

    The two properties below split a flat query back into the two halves it was assembled
    from, which is what `api.data_query.build_data_query` takes.
    """

    model_config = ConfigDict(use_enum_values=True)

    @property
    def filter_dimensions(self):
        return FilterDimensions(**dict(self))

    @property
    def options(self):
        return DataOptions(**dict(self))


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


class Analysis(BaseModel):
    """Catalog entry for a named analysis -- served by
    `/v1/analyses.json`, mirroring `Visualization`/`/v1/visualizations.json` exactly:
    built from each registered class's `Meta.__dict__` (see `api/__init__.py::
    analyses_catalog`), so a field this model doesn't recognize is silently dropped
    (pydantic's default `extra="ignore"`) and one Meta doesn't declare (e.g. `legend` for
    an analysis with none) falls back to its default here.
    """

    name: str
    label: str
    type: str
    kind: str | None = None
    variables: list[str] | None = None
    legend: str | None = None
    result_kind: str | None = None
    subsumed_dimensions: list[str] = []


class AnalysisQuery(DataOptions, FilterDimensions):
    """Accepted request parameters for the named-analysis routes: the
    dataset's dimension filters (from `FilterDimensions`) plus `DataOptions`'s
    `grid_cell`/`where`/`sort`/`select`/`color_map`. Nothing here describes a
    TRANSFORMATION, because a named analysis's transformation is entirely its own declared
    parameters plus its `Meta`-fixed pipeline shape.

    A named analysis's own declared parameters (`dimension`, `order`,
    `marsh_threshold`, ...) have no fixed field set here -- they vary per analysis and
    are read directly off the raw query string by the route (see
    `api.__init__._analysis_extra_params`) and handed to `resolve_analysis(params=...)`
    for the registry itself to validate.
    """

    model_config = ConfigDict(use_enum_values=True)

    @property
    def filter_dimensions(self):
        return FilterDimensions(**dict(self))


class AnalysisDataQuery(AnalysisQuery, Compression):
    model_config = ConfigDict(use_enum_values=True)

    @property
    def compression(self):
        return Compression(**dict(self))


class VisualizationQuery(DataQuery):
    # A named analysis (`cpra.mp.data.analyses.ANALYSES`), e.g. `difference` or
    # `land-change`. Deliberately the ONLY field this model declares for the analysis --
    # its own parameters (`dimension`/`order`/...) have no fixed field set (see
    # `AnalysisQuery`'s docstring above) and are instead read directly off the raw request
    # query string by the visualization layer itself (`DatasetVisualization.
    # resolved_analysis`, in `cpra.mp.data.visualizations.base`), the same way
    # `api.__init__._analysis_extra_params` does for the plain analysis routes -- so no
    # other change is needed here, or in `api/__init__.py`, to thread them through.
    analysis: str | None = None
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
