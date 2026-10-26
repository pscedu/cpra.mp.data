from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlmodel import Field, SQLModel

from cpra.mp.data.analysis.request.parameters import (
    Compression,  # also re-exported; the routes and api/utils.py bind it from here
    DataOptions,
    DatasetDimensions,  # noqa: F401 -- re-exported; imported from here by cli/, visualizations/ and tests
    FilterDimensions,
)
from cpra.mp.data.analysis.request.types import Grid as GridName


class DataQuery(DataOptions, FilterDimensions, Compression):
    """The BINDABLE half of a request: dimension filters, `grid_cell`/`where`/`sort`/
    `select`/`color_map`, and compression.

    No longer bound by any data route. `analysis.request.unified.partition_query` is what
    accepts and rejects a request now -- straight off the raw query string, identically on
    every route, because the categories it sorts into (an analysis's own parameters, dotted
    pipeline keys) have no fixed field set a pydantic model could declare. What this model
    is still for is two things that DO want a fixed field set: `VisualizationQuery` below,
    which binds it so a visualization class can read `self.params["grid_cell"]` as an
    ordinary attribute with its declared default; and `RENDER_OPTIONS` below, which
    subtracts it to name category 5 structurally rather than by repeating a list.

    `TileQuery` is gone with the split routes it named. It was this model minus
    `Compression`, which stopped a compression parameter BINDING on a tile route without
    rejecting it; `partition_query(allow_compression=False)` rejects it outright instead.
    The `filter_dimensions`/`options`/`compression` splitter properties went with
    `api.data_query.build_data_query`, the one caller that wanted the halves back apart.
    """

    model_config = ConfigDict(use_enum_values=True)


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
    # Whether this visualization plots the time dimension as an axis, and therefore cannot
    # present an analysis that collapses it (a `diff` over `calendar_year` returns one value
    # per grid cell with no year column to plot against). True for
    # every chart and False for the map alone -- the default is the one that holds for four
    # of the five, and `visualizations/map.py` is the one `Meta` that states it.
    # `DatasetVisualization.validate_time_axis` enforces it server-side; this field is what
    # lets a client not offer the combination in the first place.
    time_axis: bool = True
    # The named analysis this visualization IS the presentation of, put at the head of the
    # chain by `BaseVisualization.request_items` -- `sediment_dynamics` for the one chart
    # that declares it (`visualizations/sediment_dynamics.py`), null for every other.
    #
    # Published for the same reason as `time_axis` above: so a client can decline to offer
    # a combination the server would reject. A caller who names their own analysis gets it
    # appended after this one, and `resolve_analysis_chain` allows only a `kind="dimension"`
    # analysis in a non-leading stage -- so picking a `kind="variable"` analysis (which is
    # what the portal's variable dropdown holds) alongside such a chart is a 400. Without
    # this field the only way to know which chart implies an analysis is to un-hyphenate
    # its name and hope, which is a coincidence rather than a contract.
    analysis: str | None = None


class CatalogOutput(BaseModel):
    """What the request a `/v1/catalog.json` call describes would PRODUCE -- the output
    descriptor (`analysis.request.descriptor.OutputDescriptor`) serialized, with the
    resolved analysis's presentation overrides folded in.

    This is what replaces the dataset triple the catalog used to echo back. That triple was
    always a restatement of the caller's own path segments, and for a request with several
    inputs there is no single one to echo; what a client needs -- which grid to draw on,
    which time unit to step by, which variable's units and legend to label with -- is a fact
    about the OUTPUT, and `lnd_type_plus` is the proof that the two differ (it reads three
    variables off two grids and outputs on a third).

    `variables` is parallel to `value_columns` and holds variable NAMES rather than whole
    records: a client already has every variable's display metadata from
    `/v1/variables.json`, and serving it twice is two serializations of one table to keep in
    step. An entry is null where no variable could be resolved for that column.

    `folded_into` says how to read `value_columns`, and is null for every request whose
    pipeline has no `unpivot` in it. When it is a column name, the entries in
    `value_columns` are the VALUES that column takes rather than columns of the result --
    the result carries one `value` column and stacks the measurements, one row per name.
    A client keying display metadata per measurement does the same thing either way: zip
    `value_columns` with `variables`, then look the measurement up by column when
    `folded_into` is null and by that column's value when it is not.

    `label`/`legend`/`result_kind` are the overlay `descriptor.py` deliberately declines to
    apply (`analyses/` imports `request/`, not the reverse), resolved here the same way
    `visualizations.map` resolves it: the analysis's own `Meta` first, then the sole
    presentation variable, then the plain defaults.

    `collapsed_dimensions` names what the pipeline has already collapsed, and so explains
    the absences in `choices`/`results` -- a dimension listed here is absent because the
    caller cannot steer it, not because no data has it.

    Also carried by `Analysis` below, for an analysis whose result is a fact about the
    analysis rather than about any request -- see that model's `output`.
    """

    grid: str
    grid_type: str
    shape: str
    time_unit: str | None
    time_column: str | None
    value_columns: list[str]
    variables: list[str | None]
    folded_into: str | None = None
    label: str | None
    legend: str | None
    result_kind: str
    collapsed_dimensions: list[str]


class Analysis(BaseModel):
    """Catalog entry for a named analysis -- served by `/v1/analyses.json`, built from
    each registered class's `Meta.model_dump()` (see `api/__init__.py::analyses_catalog`).

    Every field below is declared by `analyses.base.AnalysisMeta`, which validates each
    `Meta` at import time, so this model is a serialization contract rather than a second
    line of defense: the dump it reads can no longer be missing a field or carrying an
    unrecognized one. That is why `kind` is required here and was `str | None` before --
    an entry that declared none (`sal_veg_investigation`) used to arrive as an absent key
    and land on this default, which made "internal" and "the author forgot" the same
    value. This does NOT mirror `Visualization` any more: those `Meta`s are still bare
    classes dumped through `__dict__`, and their model stays permissive for that reason.

    There is no `type` field: `AnalysisMeta` dropped it with the route split it described
    (see that model's comment above `kind`). Whether an analysis reads the caller's dataset
    is now answered by its own `pipeline()` at resolve time, not advertised here -- the
    portal never read it, and a catalog field nothing consumes is a field that drifts.

    `output` is what replaces it for the one question the portal does have to answer before
    it has sent anything: what does this analysis PRODUCE? It is non-null exactly for an
    analysis that resolves with no dataset, no selection and no parameters -- one that fixes
    its own inputs (`lnd_type_plus`, `sal_veg_investigation`), whose result is therefore a
    fact about the analysis rather than about any request. Every other entry's result
    depends on a dataset the caller has not named yet, so there is nothing honest to put
    here and it is null. See `api/__init__.py::analyses_catalog`.

    This is what retires the portal's `CANONICAL_DERIVED_DATASETS` outright. That table
    existed because an analysis with no caller-chosen dataset still has to be OFFERED in a
    variable picker, and the picker needed a grid and a time unit to key its options off;
    with no way to ask, the portal hardcoded a plausible triple and kept it in agreement
    with what the analysis actually does by hand. `output.grid`/`output.time_unit` are that
    agreement computed instead of maintained -- and they arrive with the catalog, before the
    first request, which is what makes them usable at picker-build time.
    """

    name: str
    label: str
    kind: str
    variables: list[str] | None = None
    legend: str | None = None
    result_kind: str | None = None
    subsumed_dimensions: list[str] = []
    output: CatalogOutput | None = None


class VisualizationQuery(DataQuery):
    """What the visualization route BINDS, which is no longer the same thing as what it
    ACCEPTS.

    The decoder (`analysis.request.unified.partition_query`) is what accepts and validates
    a visualization request now, straight off the raw query string; this model only puts
    the subset a visualization class reads as ordinary attributes into `self.params`, with
    their declared defaults (`height=400`, `show_grid_lines=True`, ...). A parameter this
    model does not declare -- a dotted `step.2.op`, an analysis's own `dimension=` -- still
    reaches the pipeline, through the decoder; it just isn't bound here.

    The fields below `analysis` are category 5, the RENDER options: per the plan they are
    per-visualization-class the way an analysis's `Parameters` are per-analysis, and are
    the only fields a follow-up data/tile request must NOT carry. `RENDER_OPTIONS` below
    derives that set structurally rather than repeating it.
    """

    # A named analysis (`cpra.mp.data.analyses.ANALYSES`), e.g. `diff` or `lnd_chng`, or a
    # comma-separated chain of them. Bound here, unlike the analysis's own parameters,
    # because a visualization reads it directly (`_analysis_name`); it is NOT a render
    # option, since the data the chart draws is the analysis's output.
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


# Category 5 of the decoder's partition: the render options, derived rather than listed.
# `VisualizationQuery` minus everything it inherits from `DataQuery` (categories 3 and 4,
# plus compression) minus `analysis` (category 1) is exactly the fields declared in the
# class body above, and only those are render options.
#
# Two consumers, and they are the same set seen from opposite ends. The visualization route
# passes it to `decode_request(render_fields=...)`, which is what keeps
# `projects_model_group_id` -- a dimension-NAMED parameter that is emphatically not a
# dimension filter -- out of `filters`. `DatasetVisualization.forward_query` subtracts it, which
# is what keeps `height=`/`show_water=` off the data and tile URLs a spec emits, where they
# would be leftovers with nothing to declare them (a 400 under `analysis=download`).
#
# Derived, not written out, because the two uses must not drift: a field added to the class
# body above and forgotten in one of the two lists would be silently misfiled in the other.
RENDER_OPTIONS = frozenset(VisualizationQuery.model_fields) - frozenset(DataQuery.model_fields) - {"analysis"}


class MPDDimensionValue(SQLModel, table=True):
    """Model group from the MPD."""

    __tablename__ = "vw_a_portal_dimension_value"
    dimension: str = Field(primary_key=True)
    value_int: int | None = None
    value_str: str | None = None
    label: str = Field(primary_key=True)
