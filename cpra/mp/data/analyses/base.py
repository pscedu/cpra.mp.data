"""Shared machinery for `cpra.mp.data.analyses`'s registry entries: the exception family
the registry (`cpra/mp/data/analyses/__init__.py`) raises, the `DatasetRef`/`ResolvedAnalysis`
value objects the interface contract with the API route layer names explicitly, the
`Analysis` base class every entry (`diff.py`, `lnd_chng.py`,
`sal_veg_investigation.py`) subclasses, and a handful of small pure helpers
(SQL-literal quoting, structural
`collapsed_dimensions` derivation) those entries share.

Both per-entry declarations are validated pydantic models: `Meta` is an `AnalysisMeta`
instance built at import time, and `Parameters` is an `AnalysisParameters` subclass
validated per request. `cpra.mp.data.visualizations` still uses the older bare
`class Meta:` convention (see `cpra/mp/data/visualizations/stacked_bar.py`) and has not
been converted; nothing here depends on the two staying alike.

Like `cpra.mp.data.analysis.request`, nothing here imports FastAPI: every exception is a
plain `ValueError` subclass, and `cpra/mp/data/api/boundary.py` -- which owns the one
type-to-status table -- is responsible for translating it to a status code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from cpra.mp.data.analysis.request.data_query import grid_cell_conjunct
from cpra.mp.data.analysis.request.expressions import split_top_level_and
from cpra.mp.data.analysis.request.pipeline import FilterStep, GroupStep, Pipeline
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
    """`resolve_analysis` was asked for a name not in `ANALYSES` -- maps to a 404 at the
    route layer, via `api/__init__.py`'s `AnalysisRegistryError` handler.
    """


class InvalidAnalysisParametersError(AnalysisRegistryError):
    """An analysis's `Parameters` model, its cardinality rule against `filters` (e.g.
    `diff`'s "exactly 2 selected values of `dimension`" rule, or `lnd_chng`'s
    `model_group_id` cardinality rule -- see each module's
    `validate_selection`), or the `Pipeline` its `pipeline()` method builds from those
    values, was rejected. Maps to a 400 at the route layer.
    """


class AnalysisDatasetMismatchError(AnalysisRegistryError):
    """An analysis was resolved with the wrong answer to "did the caller name a dataset?":
    one that fixes its own inputs was given a `variable`/`time_unit`/`grid` triple, or one
    that reads the caller's dataset was given none. Maps to a 400 at the route layer.

    This was `AnalysisTypeMismatchError`, raised by `resolve_analysis` off `Meta.type`
    before it built the analysis at all. `Meta.type` said which ROUTE could resolve an
    analysis, and with one route there is no route kind left to mismatch -- so the check
    moved into `pipeline()`, where each analysis states its own requirement through
    `Analysis.require_dataset` / `Analysis.reject_dataset` below. The error survives the
    field because the mistake it names does: `diff` still cannot difference a dataset
    nobody chose, and `lnd_type_plus` still reads three fixed inputs that a caller's triple
    would only contradict.
    """


class AnalysisChainError(AnalysisRegistryError):
    """A comma-separated chain (`lnd_conf,diff`) put an analysis somewhere it cannot go --
    today only "a stage after the first is not `kind='dimension'`". Maps to a 400.

    Split out of the former `AnalysisTypeMismatchError`, which carried both this and the
    dataset question under one name while reporting a `kind` in its message and a `type` in
    its name. They are unrelated mistakes: this one is about chain POSITION and is decided
    before anything is resolved; `AnalysisDatasetMismatchError` above is about what the
    caller selected and is decided inside `pipeline()`.
    """


@dataclass(frozen=True)
class DatasetRef:
    """The three dataset-selection dimensions an analysis reads its input from -- exactly
    the `variable`/`time_unit`/`grid` triple the unified data route's own sugar names and
    every `analysis.request.pipeline.InputSelection` already takes. `None` when the caller
    named no dataset, which is the only correct call for an analysis whose inputs are
    hardcoded in its own `pipeline()` method (`lnd_type_plus`, `sal_veg_investigation`).
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
    `visualizations.base.get_data_uri` already uses. Used to build a `diff`/`lnd_chng`
    pipeline's `group.by` (every required dimension except the one that branch folds
    over), and by `default_collapsed_dimensions` below.
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
    order. Shared by every cardinality check (`diff`/`lnd_chng`'s
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
    `selection_filter_where` below. Shared by the single-input, caller-dataset entries.
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
    tabular format. Such an analysis reads whichever dataset the caller named
    (`Analysis.require_dataset`), so the only value that is right for every reachable selection is one derived
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
    `frozenset()` when nothing collapsing is found (nothing has been collapsed).
    For `sal_veg_investigation`, whose trailing `group.by` is always `[]`, this is
    simply every dimension either input carries (both are pinned single-valued dataset
    selections, so nothing meaningful varies in the output either way).

    Scans BACKWARD past trailing `filter` steps rather than reading `steps[-1]` outright,
    because a caller's dimension/`grid_cell`/`where` selection is appended to the resolved
    pipeline as a trailing `filter` (`request.unified`, via `selection_filter_where`). A
    filter narrows rows; it never collapses a dimension, so it must not hide the `group`
    that did. Reading `steps[-1]` bare would make every selected request report nothing
    collapsed, which `visualizations.base` then turns into a `select` naming a column the
    read cannot find.

    An empty `steps` is the `download` sentinel's shape -- one input, no ops -- and
    collapses nothing.
    """
    for step in reversed(pipeline.steps):
        if isinstance(step, FilterStep):
            continue
        if isinstance(step, GroupStep):
            return frozenset(pipeline_input_dimensions(pipeline)) - set(step.by)
        return frozenset()
    return frozenset()


class AnalysisMeta(BaseModel):
    """One registry entry's static metadata -- a VALIDATED model, instantiated at import
    time, not the bare `class Meta:` this used to be.

    That change is the point of the class. `/v1/analyses.json` serves these fields
    directly (`api.__init__.analyses_catalog`), and `resolve_analysis_chain`'s chain gate
    and `visualizations.map`'s title/legend hooks read them at request time -- so a field
    an entry simply forgot to declare used to surface as an `AttributeError` (a 500) from
    whichever of those ran first, at whatever point in the request it ran.
    `sal_veg_investigation` shipped with no `kind` for exactly that reason. Constructing
    each `Meta` through this model instead turns every omission and every misspelled value
    into an error at the import that registers it, where it cannot reach a caller.

    `frozen=True` because these are catalog facts about a class, not per-request state, and
    `extra="forbid"` so a field name that drifts (a `Meta` still declaring a field this
    model has dropped) fails loudly rather than being silently ignored the way
    `api.models.Analysis`'s permissive read of it would.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    label: str
    # There is no `type` field. It said which ROUTE kind could resolve this analysis
    # (`"dataset"` took the caller's `{variable}/{time_unit}/{grid}` path segments,
    # `"custom"` fixed its own inputs and required `dataset=None`) and had exactly two
    # enforcers, `resolve_analysis`'s `expected_type` check and `api.utils.
    # get_analysis_class(name, type)`. Both were route-layer discriminators, and there is
    # one route now. The underlying REQUIREMENT did not go away and is stated where it is
    # actually known -- in each `pipeline()`, via `Analysis.require_dataset` /
    # `Analysis.reject_dataset`.
    #
    # The analysis's PRESENTATION class, which is a different partition from the one
    # `type` drew and the one the portal actually renders from:
    #
    # - `"variable"` stands in for one (or several) input variable(s) in the portal's
    #   variable dropdown (`lnd_conf`, `lnd_chng`, `lnd_type_plus`);
    # - `"dimension"` operates across one dimension of whatever it is given -- a plain
    #   dataset, or the output of a `"variable"` analysis it is chained after (`diff`).
    #   Only this kind may appear as a non-leading stage of a chain (see
    #   `resolve_analysis_chain`);
    # - `"internal"` is neither: no variable-dropdown role and nothing to chain onto it
    #   (`sal_veg_investigation`, which exists to exercise the multi-input path).
    #
    # Required, and three-valued. It was an optional `str | None` whose `None` meant
    # "internal" by omission, which is indistinguishable from "the author forgot".
    kind: Literal["variable", "dimension", "internal"]
    # Which datasets an analysis that reads the caller's dataset is meaningful for; `None`
    # means "any", and is also what an analysis with fixed inputs declares, since it has no
    # caller-chosen dataset to qualify.
    variables: list[str] | None = None
    # A `structure/legends.csv` name that overrides the input variable's own legend, and the
    # kind of value the result carries. Both are presentation overrides consulted ahead of
    # anything derivable from the pipeline -- see `analysis.request.descriptor`, which
    # derives the rest and deliberately does not fold these in.
    legend: str | None = None
    result_kind: Literal["value", "difference", "category"] = "value"
    # The legend `legend` gives way to once a chained `kind="dimension"` stage has turned
    # this analysis's output into a DIFFERENCE (`analyses.resolve_analysis_chain`), for an
    # analysis whose differenced result has a reading of its own. `lnd_conf` is the case
    # it exists for: its own output is a weighted fraction on `[0, 1]` and `lnd_conf`
    # classifies that, while `lnd_conf,diff` is a signed `[-1, 1]` change in that fraction,
    # which `lnd_conf_diff` classifies as "likely land gain"/"high confidence no change"/...
    # A `Meta.legend` cannot say both, and neither can `diff`, which knows nothing about
    # what it is chained onto -- so the LEADING analysis names the reading of its own
    # difference here, and the chain applies it.
    #
    # `None` -- every entry but `lnd_conf` today -- means a difference of this analysis has
    # no named legend, and `visualizations.map` renders it as the synthesized diverging ramp
    # off the variable's `difference_legend_limit`, exactly as it always has.
    difference_legend: str | None = None
    # The dimensions this analysis removes from the caller's view, for the portal's
    # `subsumedFor(dataset)` lookup -- advisory catalog metadata only. It plays no role in
    # resolution or chaining, both of which read the runtime `collapsed_dimensions` instead
    # (see `default_collapsed_dimensions` above and each `Analysis` subclass's own
    # `collapsed_dimensions` property/override).
    subsumed_dimensions: list[str] = []


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

    - a `Meta` -- an `AnalysisMeta` INSTANCE, built at import time, so every required
      field's absence and every misspelled value is an import error rather than a 500 from
      whichever consumer reached for it first (see `AnalysisMeta`);
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

    # Declared, not defaulted: there is no sensible `name`/`label`/`type`/`kind` for the
    # abstract base, and a stand-in would only let a subclass inherit one silently. A
    # subclass that declares no `Meta` fails at `analyses.ANALYSES`'s own registration
    # check, which is import time.
    Meta: ClassVar[AnalysisMeta]

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
        e.g. `sal_veg_investigation`, does not need to override this).
        """

    def pipeline(self) -> Pipeline:
        raise NotImplementedError

    def require_dataset(self) -> DatasetRef:
        """`self.dataset`, or a 400 naming what to send -- for an analysis that reads
        whatever dataset the caller chose.

        This is the half of the retired `Meta.type` gate that still has to be enforced, and
        it is a `raise` rather than the `assert` it replaces because it is now reachable
        from an ordinary request. Under the old route layout the two cases were structural:
        `/v1/datasets/{variable}/{time_unit}/{grid}/analyses/...` could not be matched
        without a triple and `/v1/analyses/{name}/...` had nowhere to put one, so an
        `assert` only ever documented a route-layer invariant. On one route the triple is
        query sugar a caller may simply omit, and omitting it must answer 400, not 500.
        """
        if self.dataset is None:
            raise AnalysisDatasetMismatchError(
                f"analysis {self.Meta.name!r} reads the dataset you choose -- name one "
                "with variable=, time_unit= and grid="
            )
        return self.dataset

    def reject_dataset(self) -> None:
        """The mirror image, for an analysis whose inputs are fixed in `pipeline()`.

        Rejected rather than ignored: the triple is INPUT sugar (it is how the unified
        route spells "read this dataset"), and an analysis that reads three fixed inputs
        across two grids has nothing to read it as. Accepting it silently would leave a
        caller believing they had selected something. What a chart drawn from such an
        analysis renders on is a separate question, answered from the resolved pipeline by
        `analysis.request.descriptor.output_descriptor` rather than from a triple the
        caller had to invent to agree with it.
        """
        if self.dataset is not None:
            raise AnalysisDatasetMismatchError(
                f"analysis {self.Meta.name!r} reads its own fixed inputs and takes no "
                f"dataset -- drop variable=, time_unit= and grid= (got {self.dataset!r})"
            )

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
    meta: AnalysisMeta
    legend: str | None
    result_kind: str
    collapsed_dimensions: frozenset[str]
    title_suffix: str
