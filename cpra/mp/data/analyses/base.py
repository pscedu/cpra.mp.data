"""Shared machinery for `cpra.mp.data.analyses`'s registry entries: the exception family
the registry (`cpra/mp/data/analyses/__init__.py`) raises, the `DatasetRef`/`ResolvedAnalysis`
value objects the interface contract with the API route layer names explicitly, the
`Analysis` base class every entry (`difference.py`, `concatenate.py`, `exit_criterion.py`)
subclasses, and a handful of small pure helpers (SQL-literal quoting, structural
`collapsed_dimensions` derivation) those entries share.

Mirrors `cpra.mp.data.visualizations.base`'s conventions where they overlap (a nested
`Meta` plain class per entry, never a pydantic model; `Parameters` IS a pydantic model,
since it is validated directly from raw request-shaped values) -- see that module and
`cpra/mp/data/visualizations/stacked_bar.py`'s `Meta` for the pattern being followed.

Like `cpra.mp.data.analysis.pipeline`/`planner`, nothing here imports FastAPI: every
exception is a plain `ValueError` subclass, and the API layer (owned by a different,
concurrent change -- see `cpra/mp/data/api/utils.py`'s `get_analysis_class`) is
responsible for translating it to a status code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from cpra.mp.data.analysis.pipeline import GroupStep, Pipeline
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.utils import get_input_data_frame, required_columns


class AnalysisRegistryError(ValueError):
    """Base for every error `cpra.mp.data.analyses.resolve_analysis` raises -- an
    ordinary `ValueError` subclass, in the same family as `analysis.pipeline.PipelineError`
    and `analysis.base.AnalysisError`, never `HTTPException`. `resolve_analysis` itself
    never lets a bare `PipelineError` escape (see its docstring): every rejection reaching
    a caller is one of this family's three subclasses below, which is also the family the
    API route layer's `get_analysis_class` (see `api/utils.py`) is written to expect.
    """


class UnknownAnalysisError(AnalysisRegistryError):
    """`resolve_analysis` (or `api/utils.py`'s `get_analysis_class`) was asked for a name
    not in `ANALYSES` -- maps to a 404 at the route layer.
    """


class InvalidAnalysisParametersError(AnalysisRegistryError):
    """An analysis's `Parameters` model, its cardinality rule against `filters` (e.g.
    `difference`/`concatenate`'s "exactly 2 selected values of `dimension`" rule -- see
    each module's `validate_selection`), or the `Pipeline` its `pipeline()` method builds
    from those values, was rejected. Maps to a 400 at the route layer.
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
    and every `analysis.pipeline.InputSelection` already takes. `None` for a
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
    `analysis.planner._check_input_dimensions` and
    `visualizations.base.get_data_uri` already use. Used to build a `difference`/
    `concatenate` pipeline's trailing `group.by` (every required dimension except the one
    the analysis operates over), and by `default_collapsed_dimensions` below.
    """
    return required_columns(
        get_input_data_frame(variable=dataset.variable, time_unit=dataset.time_unit, grid=dataset.grid),
        include_dataset=False,
        include_metadata=False,
    )


def is_filterable_dimension(name: str) -> bool:
    """Is `name` a real dimension a pipeline's `InputSelection`/`filter`/`calculate` may
    reference -- i.e. neither `dataset`-typed (`variable`/`time_unit`/`grid` themselves,
    which are structural, not filters) nor `metadata`-typed. Mirrors
    `analysis.planner._check_input_dimensions`'s `FilterDimensions.model_fields` check,
    but only needs `DIMENSIONS` directly here since there is no per-dataset question yet
    (a `Parameters.dimension` field is validated before any `DatasetRef` filtering
    question even applies).
    """
    dim = DIMENSIONS.get(name)
    return dim is not None and dim.type not in ("dataset", "metadata")


def cast_dimension_value(dimension: str, value: Any) -> Any:
    """Cast `value` to `dimension`'s own storage dtype (`Dimension.cast`), so a raw
    request value -- which may arrive as a string regardless of the dimension's real type
    (see `analysis.pipeline.pipeline_from_query`'s "a query value is always a string"
    note) -- compares and sorts correctly (e.g. `calendar_year` values `["2040", "9"]`
    must sort as the integers `9 < 2040`, not lexically). Falls back to the raw value for
    a name `DIMENSIONS` doesn't recognize (`Parameters` field validation already rejects
    that case before this is ever called on it -- see `is_filterable_dimension`).
    """
    dim = DIMENSIONS.get(dimension)
    return value if dim is None else dim.cast(value)


def normalized_filter_values(dimension: str, raw: Any) -> list[Any]:
    """A `filters[dimension]` value -- a scalar or a list, straight off the request --
    normalized to a de-duplicated list of `dimension`-typed values, preserving first-seen
    order. Shared by every cardinality check (`difference`/`concatenate`'s
    `validate_selection`) and every derivation (their `pipeline()` methods), so the two
    never disagree about how many distinct values were actually selected.
    """
    values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    casted = [cast_dimension_value(dimension, v) for v in values]
    return list(dict.fromkeys(casted))


def format_dimension_literal(dimension: str, value: Any) -> str:
    """Render `value` as a SQL literal for `dimension`, for interpolating into a pipeline
    step's `where`/`calculate.where` string -- the Python equivalent of the portal's own
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


def dataset_input_selection(dataset: DatasetRef, filters: dict[str, Any]) -> dict[str, Any]:
    """Build the `InputSelection`-shaped dict for `dataset`'s single pipeline input,
    folding `filters` (dimension filters, plus an optional reserved `grid_cell` key -- see
    the `resolve_analysis` contract) onto it exactly as given. Shared by `difference.py`/
    `concatenate.py`, the two single-input (`type="dataset"`) entries.
    """
    selection: dict[str, Any] = {
        "variable": dataset.variable,
        "time_unit": dataset.time_unit,
        "grid": dataset.grid,
    }
    for key, value in filters.items():
        selection[key] = value
    return selection


def default_collapsed_dimensions(pipeline: Pipeline) -> frozenset[str]:
    """The dimensions a pipeline's result no longer varies over: every dimension any of
    its inputs carries, minus whatever the pipeline's OWN trailing `group.by` still keeps
    (derived structurally, rather than each analysis hand-declaring it) --
    `frozenset()` when the last step isn't a `group` at all (nothing has been collapsed).
    For `exit-criterion`, whose trailing `group.by` is always `[]`, this is simply every
    dimension either input carries (both are pinned single-valued dataset selections, so
    nothing meaningful varies in the output either way).
    """
    last_step = pipeline.steps[-1]
    if not isinstance(last_step, GroupStep):
        return frozenset()

    all_dims: set[str] = set()
    for selection in pipeline.inputs.values():
        all_dims.update(
            required_columns(
                get_input_data_frame(
                    variable=selection.variable, time_unit=selection.time_unit, grid=selection.grid
                ),
                include_dataset=False,
                include_metadata=False,
            )
        )
    return frozenset(all_dims) - set(last_step.by)


class AnalysisParameters(BaseModel):
    """Base for every analysis's nested `Parameters` model -- `extra="forbid"` so an
    unrecognized parameter name 400s explicitly rather than being silently ignored,
    matching this codebase's general pydantic convention (see e.g.
    `analysis.pipeline.InputSelection`/`Pipeline`, though those use `extra="allow"`/
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
    - a `pipeline()` method building a real `analysis.pipeline.Pipeline` from
      `self.dataset`/`self.filters`/`self.parameters` -- the ONLY thing a resolved
      analysis exposes for actually running it (nothing may be reachable only
      through a named analysis, so this always funnels through the same
      `compile_pipeline` any hand-authored pipeline does);
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

    class Parameters(AnalysisParameters):
        pass

    def __init__(self, *, dataset: DatasetRef | None, filters: dict[str, Any], parameters: "Analysis.Parameters"):
        self.dataset = dataset
        self.filters = filters
        self.parameters = parameters

    @classmethod
    def validate_selection(cls, parameters: "Analysis.Parameters", filters: dict[str, Any]) -> None:
        """A cardinality/consistency rule that needs `filters` alongside `parameters` --
        raise `InvalidAnalysisParametersError` to reject. The default performs no extra
        check (an analysis whose `Parameters` model alone is a sufficient validity check,
        e.g. `exit-criterion`, does not need to override this).
        """

    def pipeline(self) -> Pipeline:
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
