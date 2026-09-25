"""Shared machinery for `cpra.mp.data.analyses`'s registry entries: the exception family
the registry (`cpra/mp/data/analyses/__init__.py`) raises, the `DatasetRef`/`ResolvedAnalysis`
value objects the interface contract with the API route layer names explicitly, the
`Analysis` base class every entry (`difference.py`, `land_change.py`,
`salinity_veg_investigation.py`) subclasses, and a handful of small pure helpers
(SQL-literal quoting, structural
`collapsed_dimensions` derivation) those entries share.

Mirrors `cpra.mp.data.visualizations.base`'s conventions where they overlap (a nested
`Meta` plain class per entry, never a pydantic model; `Parameters` IS a pydantic model,
since it is validated directly from raw request-shaped values) -- see that module and
`cpra/mp/data/visualizations/stacked_bar.py`'s `Meta` for the pattern being followed.

Like `cpra.mp.data.analysis.request`, nothing here imports FastAPI: every exception is a
plain `ValueError` subclass, and `cpra/mp/data/api/boundary.py` -- which owns the one
type-to-status table -- is responsible for translating it to a status code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from cpra.mp.data.analysis.request.data_query import grid_cell_conjunct
from cpra.mp.data.analysis.request.expressions import split_top_level_and
from cpra.mp.data.analysis.request.pipeline import GroupStep, Pipeline
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.utils import get_input_data_frame, required_columns

# The two keys `resolve_analysis`'s `filters` mapping may carry that are NOT dimension
# filters: a `grid_cell` selection and a caller-written `where`. Both are spatial/value
# conditions on the data rather than parts of the dataset selection, so both are composed
# into the leading `filter` step by `selection_filter_where` below rather than rejected as
# unrecognized dimension names.
RESERVED_FILTER_KEYS = frozenset({"grid_cell", "where"})


class AnalysisRegistryError(ValueError):
    """Base for every error `cpra.mp.data.analyses.resolve_analysis` raises -- an
    ordinary `ValueError` subclass, in the same family as
    `analysis.request.errors.RequestError`, never `HTTPException`. `resolve_analysis`
    itself never lets a bare `RequestError` escape (see its docstring): every rejection
    reaching a caller is one of this family's three subclasses below.
    """


class UnknownAnalysisError(AnalysisRegistryError):
    """`resolve_analysis` (or `api/utils.py`'s `get_analysis_class`) was asked for a name
    not in `ANALYSES` -- maps to a 404 at the route layer.
    """


class InvalidAnalysisParametersError(AnalysisRegistryError):
    """An analysis's `Parameters` model, its cardinality rule against `filters` (e.g.
    `difference`'s "exactly 2 selected values of `dimension`" rule, or `land-change`'s
    `calendar_year`/`model_group_id` cardinality rule -- see each module's
    `validate_selection`), or the `Pipeline` its `pipeline()` method builds from those
    values, was rejected. Maps to a 400 at the route layer.
    """


class AnalysisTypeMismatchError(AnalysisRegistryError):
    """An analysis was resolved against the wrong route kind: a `type="custom"` analysis
    given a `dataset`, or a `type="dataset"` analysis resolved with `dataset=None`. Maps
    to a 400 at the route layer -- see `api/utils.py`'s
    `get_analysis_class(name, type=...)`, which makes the identical distinction for
    visualizations already.
    """


@dataclass(frozen=True)
class DatasetRef:
    """The three dataset-selection dimensions a `type="dataset"` analysis is resolved
    against -- exactly the `variable`/`time_unit`/`grid` triple every flat data endpoint
    and every `analysis.request.pipeline.InputSelection` already takes. `None` for a
    `type="custom"` analysis (see `resolve_analysis`), since a custom analysis's inputs
    are hardcoded in its own `pipeline()` method, not supplied by the route.
    """

    variable: str
    time_unit: str
    grid: str


def required_dataset_dimensions(dataset: DatasetRef) -> list[str]:
    """Every real, non-dataset, non-metadata dimension `dataset`'s
    variable/time_unit/grid combination carries -- e.g. `model_group_id`, `scenario_id`,
    and any variable-specific `required_dimensions` (`aggregate_type`, `ffibs_type`, ...).
    Pure/in-memory: `get_input_data_frame` only cross-joins tiny literal `pl.DataFrame`s
    and never reads a file, so this needs no `base_path` -- the same pattern
    `visualizations.base.get_data_uri` already uses. Used to build a `difference`/
    `land-change` pipeline's trailing `group.by` (every required dimension except the one
    the analysis operates over), and by `default_collapsed_dimensions` below.
    """
    return required_columns(
        get_input_data_frame(variable=dataset.variable, time_unit=dataset.time_unit, grid=dataset.grid),
        include_dataset=False,
        include_metadata=False,
    )


def is_filterable_dimension(name: str) -> bool:
    """Is `name` a real dimension a pipeline's `filter.where`/`case` arm `when` may reference
    -- i.e. neither `dataset`-typed (`variable`/`time_unit`/`grid` themselves, which
    select the input rather than filter it) nor `metadata`-typed. Mirrors
    `request.parameters.BaseFilterDimensions`'s own field set, but reads `DIMENSIONS`
    directly since there is no per-dataset question yet (a `Parameters.dimension` field is
    validated before any `DatasetRef` filtering question even applies).
    """
    dim = DIMENSIONS.get(name)
    return dim is not None and dim.type not in ("dataset", "metadata")


def cast_dimension_value(dimension: str, value: Any) -> Any:
    """Cast `value` to `dimension`'s own storage dtype (`Dimension.cast`), so a raw
    request value -- which may arrive as a string regardless of the dimension's real type
    (see `analysis.request.pipeline.pipeline_from_query`'s "a query value is always a
    string" note) -- compares and sorts correctly (e.g. `calendar_year` values `["2040", "9"]`
    must sort as the integers `9 < 2040`, not lexically). Falls back to the raw value for
    a name `DIMENSIONS` doesn't recognize (`Parameters` field validation already rejects
    that case before this is ever called on it -- see `is_filterable_dimension`).
    """
    dim = DIMENSIONS.get(dimension)
    return value if dim is None else dim.cast(value)


def normalized_filter_values(dimension: str, raw: Any) -> list[Any]:
    """A `filters[dimension]` value -- a scalar or a list, straight off the request --
    normalized to a de-duplicated list of `dimension`-typed values, preserving first-seen
    order. Shared by every cardinality check (`difference`/`land-change`'s
    `validate_selection`) and every derivation (their `pipeline()` methods), so the two
    never disagree about how many distinct values were actually selected.
    """
    values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    casted = [cast_dimension_value(dimension, v) for v in values]
    return list(dict.fromkeys(casted))


def format_dimension_literal(dimension: str, value: Any) -> str:
    """Render `value` as a SQL literal for `dimension`, for interpolating into a pipeline
    step's `where`/`when` string -- the Python equivalent of the portal's own
    `sqlValue()` helper (`cpra-mp-ui/src/helpers/utils.ts`): a numeric dimension's value is
    unquoted, `calendar_day` gets a `'...'::date` cast, and every other (string) dimension
    is single-quoted (with `'` doubled, defensively, even though no real dimension value
    is expected to contain one). `Dimension.dtype` is a polars TYPE CLASS, not an instance
    (see `dimensions.Dimension.__init__`), so `.is_numeric()` is called on the class
    itself -- confirmed this still returns the right answer.
    """
    casted = cast_dimension_value(dimension, value)
    dim = DIMENSIONS.get(dimension)
    if dim is not None and dim.dtype.is_numeric():
        return str(casted)
    if dimension == "calendar_day":
        return f"'{casted}'::date"
    escaped = str(casted).replace("'", "''")
    return f"'{escaped}'"


def dataset_input_selection(dataset: DatasetRef) -> dict[str, Any]:
    """Build the `InputSelection`-shaped dict for `dataset`'s single pipeline input: the
    `variable`/`time_unit`/`grid` triple and nothing else. An `InputSelection` selects
    WHICH dataset to read and carries no filtering of any kind (it is `extra="forbid"`, so
    a dimension name folded in here is a hard rejection, not a silently-ignored key) --
    a caller's dimension filters become a `filter` STEP instead, via
    `selection_filter_where` below. Shared by the single-input (`type="dataset"`) entries.
    """
    return {
        "variable": dataset.variable,
        "time_unit": dataset.time_unit,
        "grid": dataset.grid,
    }


def dataset_output_format(dataset: DatasetRef) -> str:
    """The `output.format` a single-input analysis over `dataset` should declare: `tif` on
    a raster grid, `csv` on a vector one.

    `Output.format` is required, so every `pipeline()` has to name one -- but on the
    `data.{format}` and tile routes the value never survives: each of those overwrites it
    from its own URL segment before running anything, which is what lets one analysis be
    fetched as `csv`, `tif`, `png` or `.pbf` without declaring any of them. `pipeline.json`
    is the exception: it has no format segment to source an override from, so it publishes
    whatever `pipeline()` set, and a caller resubmitting that pipeline verbatim has to get
    one the request layer accepts.

    That makes the declared value a real contract rather than a placeholder, and
    `request.ops.validate_shapes` enforces it in both directions: a long (vector) final
    frame under a raster format is rejected, and so is a band-wide (raster) one under a
    tabular format. A `type="dataset"` analysis reads whichever dataset the route resolved
    for it, so the only value that is right for every reachable selection is one derived
    from that dataset's own grid kind -- these analyses neither project nor change shape,
    so the final frame's kind is the input grid's kind.
    """
    return "tif" if GRIDS[dataset.grid].grid_type == "raster" else "csv"


def selection_filter_where(
    filters: dict[str, Any], *, grid_cell_where: str | None = None
) -> str | None:
    """Compose `filters` into the single `where` string a leading `filter` step takes, or
    `None` when there is nothing to filter on. This is the named-analysis counterpart of
    `request.data_query.build_data_query_pipeline`'s own conjunct assembly, and composes
    the same three ingredients in the same order:

    1. one conjunct per DIMENSION key -- `dim = lit` for a single selected value,
       `dim IN (lit, ...)` for several, rendered by `format_dimension_literal` and
       normalized by `normalized_filter_values` so the two never disagree about how many
       distinct values were selected. Iterated in SORTED key order, so the composed string
       does not depend on (or silently change with) the caller's dict ordering;
    2. the spatial conjunct: `grid_cell_where` verbatim when given (a caller that had to
       resolve coordinates to IDs, which needs `io/`, passes the resolved conjunct in the
       same way `build_data_query_pipeline` takes it), otherwise
       `request.data_query.grid_cell_conjunct(filters["grid_cell"])` for the digit-ID case
       -- one definition of the digit-ID rule, shared rather than re-spelled here;
    3. each top-level conjunct of a caller's own `where`, split by
       `expressions.split_top_level_and` and parenthesized INDIVIDUALLY. Splitting first
       is what keeps the parens from hiding the top-level `AND` seam the engine classifies
       each conjunct's kind against -- see `build_data_query_pipeline`'s docstring for the
       full argument; the reasoning is identical here.

    Any key that is neither a filterable dimension (`is_filterable_dimension`) nor one of
    `RESERVED_FILTER_KEYS` raises `InvalidAnalysisParametersError` naming it, so a typo
    fails loudly instead of quietly filtering nothing.
    """
    conjuncts: list[str] = []

    unknown = [
        key
        for key in filters
        if key not in RESERVED_FILTER_KEYS and not is_filterable_dimension(key)
    ]
    if unknown:
        raise InvalidAnalysisParametersError(
            f"unrecognized filter name(s): {', '.join(sorted(unknown))} -- expected a "
            f"filterable dimension or one of {', '.join(sorted(RESERVED_FILTER_KEYS))}"
        )

    for name in sorted(key for key in filters if key not in RESERVED_FILTER_KEYS):
        values = normalized_filter_values(name, filters[name])
        if not values:
            continue
        literals = [format_dimension_literal(name, value) for value in values]
        if len(literals) == 1:
            conjuncts.append(f"{name} = {literals[0]}")
        else:
            conjuncts.append(f"{name} IN ({', '.join(literals)})")

    if grid_cell_where is not None:
        conjuncts.append(grid_cell_where)
    elif filters.get("grid_cell"):
        conjuncts.append(grid_cell_conjunct(str(filters["grid_cell"])))

    if filters.get("where"):
        conjuncts.extend(f"({conjunct})" for conjunct in split_top_level_and(filters["where"]))

    return " AND ".join(conjuncts) if conjuncts else None


def pipeline_input_dimensions(pipeline: Pipeline) -> list[str]:
    """Every real, non-dataset, non-metadata dimension any of `pipeline`'s inputs
    carries -- an ordered, deduplicated union (first-seen order over `pipeline.inputs`'s
    own order), rather than the `frozenset` `default_collapsed_dimensions` returns, so a
    caller deriving a `group.by` from it (e.g. `resolve_analysis_chain`) gets a
    deterministic column order matching this codebase's convention of deriving one from
    `required_dataset_dimensions`'s own list order.
    """
    seen: dict[str, None] = {}
    for selection in pipeline.inputs.values():
        for dim in required_columns(
            get_input_data_frame(
                variable=selection.variable, time_unit=selection.time_unit, grid=selection.grid
            ),
            include_dataset=False,
            include_metadata=False,
        ):
            seen.setdefault(dim, None)
    return list(seen)


def default_collapsed_dimensions(pipeline: Pipeline) -> frozenset[str]:
    """The dimensions a pipeline's result no longer varies over: every dimension any of
    its inputs carries, minus whatever the pipeline's OWN trailing `group.by` still keeps
    (derived structurally, rather than each analysis hand-declaring it) --
    `frozenset()` when the last step isn't a `group` at all (nothing has been collapsed).
    For `salinity-veg-investigation`, whose trailing `group.by` is always `[]`, this is
    simply every dimension either input carries (both are pinned single-valued dataset
    selections, so nothing meaningful varies in the output either way).
    """
    last_step = pipeline.steps[-1]
    if not isinstance(last_step, GroupStep):
        return frozenset()
    return frozenset(pipeline_input_dimensions(pipeline)) - set(last_step.by)


class AnalysisParameters(BaseModel):
    """Base for every analysis's nested `Parameters` model -- `extra="forbid"` so an
    unrecognized parameter name 400s explicitly rather than being silently ignored,
    matching this codebase's general pydantic convention (see e.g.
    `analysis.request.pipeline.InputSelection`/`Pipeline`, though those use `extra="allow"`/
    `"forbid"` for their own, different reasons; a `Parameters` model has no reason to
    accept a name it doesn't declare).
    """

    model_config = ConfigDict(extra="forbid")


class Analysis:
    """Base class for one `ANALYSES` registry entry. A concrete subclass declares:

    - a nested `Meta` (a plain class, not a pydantic model -- mirrors
      `visualizations.stacked_bar.StackedBar.Meta`): `name`, `label`, `type`
      (`"dataset"` or `"custom"`), `variables` (`None` = any), `legend`, `result_kind`
      (`"value"`, `"difference"`, or `"category"`);
    - a nested `Parameters` (a pydantic model subclassing `AnalysisParameters`) for its
      own request-specific parameters;
    - a `pipeline()` method building a real `analysis.request.pipeline.Pipeline` from
      `self.dataset`/`self.filters`/`self.grid_cell_where`/`self.parameters` -- the ONLY
      thing a resolved analysis exposes for actually running it (nothing may be reachable
      only through a named analysis, so this always funnels through the same
      `execution.plan.compile_plan` any hand-authored pipeline does);
    - optionally, a `validate_selection` classmethod for a cardinality/consistency rule
      that needs `filters` as well as `parameters` (e.g. "exactly 2 selected values of
      `dimension`") -- the default is a no-op.

    `legend`/`result_kind`/`title_suffix`/`collapsed_dimensions` are presentation hooks a
    subclass may override; the default `collapsed_dimensions` derives it structurally
    from `self.pipeline()` via `default_collapsed_dimensions` rather than needing a
    hand-declared value.
    """

    class Meta:
        name: ClassVar[str]
        label: ClassVar[str]
        type: ClassVar[str]
        variables: ClassVar[list[str] | None] = None
        legend: ClassVar[str | None] = None
        result_kind: ClassVar[str] = "value"
        # "variable": stands in for one (or several) input variable(s) in the portal's
        # variable dropdown, e.g. `land-type-confidence`. "dimension": operates across one
        # dimension of whatever it's given -- a plain dataset or the output of a
        # `kind="variable"` analysis it's chained after, e.g. `difference`. `None` (the
        # default) is neither -- `salinity-veg-investigation`'s reading, a `type="custom"`
        # analysis with no variable-dropdown role and nothing to chain onto it.
        kind: ClassVar[str | None] = None
        # The dimensions this analysis removes from the caller's view, for the portal's
        # `subsumedFor(dataset)` lookup -- advisory catalog metadata only. It plays no role
        # in resolution or chaining, both of which read the runtime `collapsed_dimensions`
        # instead (see `default_collapsed_dimensions` below and each `Analysis` subclass's
        # own `collapsed_dimensions` property/override).
        subsumed_dimensions: ClassVar[list[str]] = []

    class Parameters(AnalysisParameters):
        pass

    def __init__(
        self,
        *,
        dataset: DatasetRef | None,
        filters: dict[str, Any],
        parameters: "Analysis.Parameters",
        grid_cell_where: str | None = None,
    ):
        self.dataset = dataset
        self.filters = filters
        self.parameters = parameters
        # An already-resolved spatial conjunct (e.g. `"grid_cell IN (1, 2)"`), for a
        # caller that had to turn coordinates into IDs itself -- it entirely replaces
        # `filters["grid_cell"]`'s own digit-ID handling, exactly as the same-named
        # argument does for `request.data_query.build_data_query_pipeline`.
        self.grid_cell_where = grid_cell_where

    @classmethod
    def validate_selection(cls, parameters: "Analysis.Parameters", filters: dict[str, Any]) -> None:
        """A cardinality/consistency rule that needs `filters` alongside `parameters` --
        raise `InvalidAnalysisParametersError` to reject. The default performs no extra
        check (an analysis whose `Parameters` model alone is a sufficient validity check,
        e.g. `salinity-veg-investigation`, does not need to override this).
        """

    def pipeline(self) -> Pipeline:
        raise NotImplementedError

    def _dimension_steps(self, group_by: list[str]) -> list[dict]:
        """The steps a `kind="dimension"` analysis contributes when chained after
        another analysis's output (see `analyses.resolve_analysis_chain`). Only a
        `kind="dimension"` analysis needs to implement this -- every other analysis is
        chain-terminal and never reaches this call.
        """
        raise NotImplementedError

    @property
    def legend(self) -> str | None:
        return self.Meta.legend

    @property
    def result_kind(self) -> str:
        return self.Meta.result_kind

    @property
    def title_suffix(self) -> str:
        return ""

    @property
    def collapsed_dimensions(self) -> frozenset[str]:
        return default_collapsed_dimensions(self.pipeline())


@dataclass
class ResolvedAnalysis:
    """What `resolve_analysis` returns: the compiled request's real `Pipeline`, plus the
    presentation metadata the route layer (and, downstream, whatever renders a legend or a
    title) needs and would otherwise have to re-derive or hand-wire per analysis.
    """

    pipeline: Pipeline
    meta: type
    legend: str | None
    result_kind: str
    collapsed_dimensions: frozenset[str]
    title_suffix: str
