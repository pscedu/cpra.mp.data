"""The canonical request model for analysis pipelines: named inputs, an ordered list of
operations over them, and an output description -- plus the two encodings the request can
arrive in (a JSON body, and a dotted GET query string suitable for tile URLs) and the
plan-time validation both encodings share.

This module builds the REPRESENTATION and its VALIDATION only. Nothing here reads a file,
opens a dataset, or knows a crosswalk -- see the module docstring's "deferred to compile
time" notes throughout for the rules that need that knowledge and are therefore NOT here.
A sibling module executes an already-validated `Pipeline`; a later change wires an HTTP
endpoint on top of both. Following `api/expressions.py` and `api/fold.py`'s established
shape: every function here takes and returns plain values and raises `PipelineError` (an
ordinary `ValueError` subclass) -- never `HTTPException`, never a FastAPI import -- so this
module is usable outside the API later (`read_data()` in `cpra/mp/data/__init__.py`).

**The operations** (`project`, `join`, `filter`, `calculate`, `group` -- `inputs`/`output`
are structural, not operations): see `ProjectStep`, `JoinStep`, `FilterStep`,
`CalculateStep`, `GroupStep` below.

**Two encodings, one model.** `Pipeline` is parsed directly from a JSON-shaped dict (the
POST body) via `parse_pipeline`, or from a flat dotted query mapping (the GET/tile
encoding) via `pipeline_from_query` -- both funnel through the same `Pipeline` model and
the same `validate_pipeline` business rules, so a pipeline expressed either way is
accepted or rejected identically. `pipeline_to_query` is the inverse of
`pipeline_from_query`, canonicalized (sorted) so equivalent pipelines share one cache key.

**Why business validation is NOT wired into pydantic validators.** Pydantic v2 catches any
exception a `field_validator`/`model_validator` raises -- including a custom `ValueError`
subclass -- and re-raises it as `pydantic.ValidationError` (itself a `ValueError`
subclass, but not `PipelineError`), regardless of where in the model the raise happened.
Reusing `PipelineError` as a single, directly-catchable family (matching
`ExpressionError`/`FoldError`) therefore means most of this module's business rules run as
a plain function (`validate_pipeline`) called AFTER a `Pipeline` is already constructed,
not as a pydantic hook -- see `parse_pipeline`, which is the combined "parse the shape,
then validate the rules" entry point every caller should use. The one exception is
`InputSelection`'s scalar-vs-list normalization, which is a SHAPE coercion (not a business
rule) and stays a `model_validator` for that reason; raw `Pipeline(...)`/`model_validate`
without going through `parse_pipeline` still enforces it, just surfaced as
`pydantic.ValidationError` rather than `PipelineError` if used directly.
"""

import re
from collections.abc import Mapping
from datetime import date
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from cpra.mp.data.analysis.expressions import (
    ExpressionError,
    classify_where,
    parse_calculate_expression,
)
from cpra.mp.data.analysis.fold import FoldError, validate_decomposable
from cpra.mp.data.analysis.types import (
    AggregateFunction,
    DataFormat,
    Grid,
    TileFormat,
    TimeUnit,
    Variable,
)


class PipelineError(ValueError):
    """A pipeline request failed a plan-time rule -- malformed shape, an unresolvable
    step reference, a non-decomposable aggregate, or any other check in this module. An
    ordinary `ValueError` subclass, in the family of `api/expressions.ExpressionError`
    and `api/fold.FoldError` -- never `HTTPException`; a future translation layer maps
    it to a status code, the same split those two modules already establish.
    """


# Every name a user chooses -- an input's key in `Pipeline.inputs`, or a step's `as` --
# shares one shape rule: it has to be usable as an identifier/column name downstream, and
# it must not shadow a name the row model already gives fixed meaning (the raw value
# column, and the grid key every row carries).
_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
RESERVED_NAMES = frozenset({"value", "grid_cell"})

# Scalar types an `InputSelection` dimension filter value may take, bare or as a list of
# them -- see `InputSelection._normalize_dimension_filters`.
#
# `date` is here because it is a REAL dimension value type, not a convenience: `Date` is
# one of the five dtypes `structure/dimensions.csv` declares (`calendar_day`), and
# `types.dimension_type` maps it to `datetime.date`, so `FilterDimensions.calendar_day` is
# `list[date]` and FastAPI binds `?calendar_day=2019-01-01` to a real `date` object. A
# pipeline built from a request's own filters (every named analysis, via
# `analyses.resolve_analysis`) therefore receives `date`s, not strings. Omitting it made
# every date-dimension analysis -- e.g. a `difference` over `calendar_day` on a daily
# dataset -- fail validation outright.
#
# `datetime` needs no entry of its own: it subclasses `date`, so `isinstance` covers it.
# Serialization already round-trips correctly both ways out -- `pipeline_to_query` emits
# `str(date(2019, 1, 1))` == `"2019-01-01"`, and `pipeline.json` uses
# `model_dump(mode="json")`, which renders a `date` extra as the same ISO string.
_SCALAR_TYPES = (str, int, float, bool, date)

# Output formats that render to a single raster/image surface (a GeoTIFF band or a map
# tile pixel) rather than a table -- these can only ever carry ONE value column. Reused
# directly from `api/types` rather than re-listing format names, per the module's "no
# second selection grammar" principle applied to formats too.
_SINGLE_VALUE_COLUMN_FORMATS = frozenset({DataFormat.tif.value, TileFormat.png.value, TileFormat.jpg.value})


def _validate_name(name: str, kind: str) -> None:
    if not name:
        raise PipelineError(f"{kind} name must not be empty")
    if not _NAME_RE.match(name):
        raise PipelineError(
            f"{kind} name {name!r} must match [a-z_][a-z0-9_]* (lowercase letters, digits, "
            "underscore; must not start with a digit)"
        )
    if name in RESERVED_NAMES:
        raise PipelineError(
            f"{kind} name {name!r} collides with a reserved column name {sorted(RESERVED_NAMES)}"
        )


def _require_decomposable(aggregate: str, label: str) -> None:
    try:
        validate_decomposable(aggregate)
    except FoldError as exc:
        raise PipelineError(f"{label}: {exc}") from exc


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


class InputSelection(BaseModel):
    """One named dataset selection: `variable`/`time_unit`/`grid` (the SAME dataset
    dimensions the flat data endpoints require -- reused directly from `api.types` rather
    than a second selection grammar, per the spec's explicit "no second selection
    grammar" rule), plus any number of extra keys, each a DIMENSION FILTER whose value is
    a scalar or a list of scalars (`extra="allow"`; every non-dataset-dimension key
    lands in `model_extra` and is normalized to a list by `_normalize_dimension_filters`).

    `grid_cell` is a RESERVED field, not a dimension filter -- it is a Filter-group
    parameter (like `Options.grid_cell`, which is exactly where it lowers to; see
    `planner._dataset_for`), never an `extra` key, so it never lands in
    `dimension_filters`/`__pydantic_extra__` and `_check_input_dimensions`'s "unknown
    dimension" check never has to special-case it. Comma-separated IDs (or coordinate
    pairs, for a raster input) -- the same string `Options.grid_cell` itself accepts.

    This model validates SHAPE only. Whether an extra key names a real dimension (and
    whether its value(s) are the right type/in range for that dimension, and whether the
    dimension even applies to this `variable`/`grid` combination) all need the catalog --
    that check is deferred to compile time, which is a different task's machinery.
    """

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    variable: Variable
    time_unit: TimeUnit
    grid: Grid
    grid_cell: str | None = None

    @model_validator(mode="after")
    def _normalize_dimension_filters(self) -> "InputSelection":
        extra = self.__pydantic_extra__ or {}
        normalized: dict[str, list[Any]] = {}
        for key, value in extra.items():
            if isinstance(value, list):
                if not value:
                    raise PipelineError(f"dimension filter {key!r} must not be an empty list")
                values = value
            else:
                values = [value]
            for v in values:
                if not isinstance(v, _SCALAR_TYPES):
                    raise PipelineError(
                        f"dimension filter {key!r} must be a scalar or a list of scalars, "
                        f"got {type(v).__name__} in it"
                    )
            normalized[key] = list(values)
        self.__pydantic_extra__ = normalized
        return self

    @property
    def dimension_filters(self) -> dict[str, list[Any]]:
        """Every key besides `variable`/`time_unit`/`grid`, each normalized to a
        non-empty list of scalars -- what `pipeline_to_query` iterates to emit
        `input.<name>.<dimension>=v1,v2` parameters, and what a compile-time step will
        resolve against the catalog.
        """
        return dict(self.__pydantic_extra__ or {})


# --------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------


class ProjectStep(BaseModel):
    """Move a dataset across grids via a crosswalk. `input` is optional -- an omitted
    `input` chains from the previous step's output (see `validate_pipeline`'s chaining
    rule). `aggregate`, when given, must be one of `fold.DECOMPOSABLE_AGGREGATES` OR
    `count_by_value` (the one non-decomposable aggregate a `project` may still use --
    `count_by_value` is a ZONAL aggregate, meaningful for a many-to-one reaggregation
    exactly the way `sum`/`mean`/etc. are, even though it can't be folded across a
    crosswalk's row-group boundary the way those can; see `_validate_step_rules`'s
    `ProjectStep` branch, and contrast `GroupStep`, which still rejects it outright). The
    many-to-one-REQUIRES-aggregate / one-to-many-REJECTS-aggregate rule itself needs
    crosswalk knowledge to check and is deferred to compile time -- see
    `validate_pipeline`'s docstring).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, use_enum_values=True)

    op: Literal["project"] = "project"
    input: str | None = None
    to: Grid
    aggregate: AggregateFunction | None = None
    as_: str | None = Field(default=None, alias="as")


class JoinStep(BaseModel):
    """Combine two or more named inputs/step-outputs on shared dimensions + `grid_cell`.
    `type="attribute"` parses (it is a real member of the vocabulary) but is rejected by
    `validate_pipeline` as reserved/not-yet-implemented -- only `"spatial"` executes
    today. `inputs` must name at least two things and (checked here, since it needs no
    pipeline context) must not repeat a name; that each name actually resolves to a known
    input or a prior step's `as` is checked by `validate_pipeline`, which has the whole
    pipeline in view.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, use_enum_values=True)

    op: Literal["join"] = "join"
    type: Literal["spatial", "attribute"]
    inputs: list[str] = Field(min_length=2)
    as_: str = Field(alias="as")


class FilterStep(BaseModel):
    """A predicate over in-scope named columns, e.g. `marsh > 0.5 AND sal > 5.5`. `input`
    is optional -- an omitted `input` chains from the previous step's output, mirroring
    `ProjectStep.input` exactly (same optionality, same "explicit name or chain" meaning,
    same rule for who may use it: see `validate_pipeline`'s chaining rule). This is what
    lets a `filter` lead a pipeline: `planner._compile_single_input_chain` rejects a
    value-testing `filter` declared AFTER a many-to-one `project` outright, for both grid
    types (its SUMMARIZING-PROJECT ORDER GUARD -- a filter and a re-aggregation do not
    commute, and that shape's correctness has not been established for either engine),
    rather than risk returning a wrong answer. Before `input` existed there was no way to write the filter the other
    way around either, since `filter`/`calculate`/`group` all lacked any way to name an
    input and so could never be a pipeline's first step -- only `project`/`join` could
    satisfy the chaining rule's "first step must name its input" requirement. Naming
    `input` explicitly is what lets `filter` satisfy that same requirement now, exactly
    the way `project.input` already does, so the value-testing filter can be declared
    BEFORE the project instead; `calculate`/`group` still have no `input` field and so
    still can never lead a pipeline. `as` is not in the spec's table for this op but is
    accepted here anyway, generalizing the chaining rule's own "a prior step's `as`"
    phrasing, which does not restrict which op can produce a name a later step
    references.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    op: Literal["filter"] = "filter"
    input: str | None = None
    where: str
    as_: str | None = Field(default=None, alias="as")


class CalculateStep(BaseModel):
    """Conditionally rewrite the running value: one `expression`/`where` pair per step
    (several pairs = repeated steps, applied sequentially -- step N+1 sees step N's
    output). `expression` is arithmetic over the running value and numeric literals
    (referenced either by its in-scope column name -- the input/step name, exactly the
    way `filter.where` names columns -- or by the `value` placeholder); `where` uses the
    same dimension-and/or-value-condition grammar as `filter.where`, and an omitted
    `where` applies the expression to everything. Rows/bands NOT matching `where` pass
    through unchanged -- `calculate` never drops data; use `filter` for that. `input` is
    optional -- an omitted `input` chains from the previous step's output, mirroring
    `FilterStep.input` exactly and for the identical reason: a `calculate` declared as a
    leading step, ahead of the `project` it precedes, executes in exactly the position it
    is declared in: the interpreters run every op in declared order, so a leading
    `calculate` runs ahead of the `project` it precedes.
    There is no ordering question here the way there is for a POST-project
    `calculate`, which `planner._compile_single_input_chain` still rejects after a
    many-to-one `project` (the same unestablished-commutativity risk `FilterStep`'s
    docstring describes for a post-project value filter). `group` still has no `input` field (see `GroupStep`'s
    docstring for why that one is left alone) and so still can never lead a pipeline.
    `as` follows `FilterStep`'s reasoning: accepted, naming this step's output for a
    later step to reference.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    op: Literal["calculate"] = "calculate"
    input: str | None = None
    expression: str
    where: str | None = None
    as_: str | None = Field(default=None, alias="as")


class GroupStep(BaseModel):
    """Collapse `by` (may be empty, meaning "collapse to one row per `grid_cell`") with
    `aggregate` (required; must be in `fold.DECOMPOSABLE_AGGREGATES`) applied to `of`
    (defaults to `"value"`, generalizing the aggregand -- "first year a condition held"
    is `aggregate="min", of="calendar_year"`). Always consumes the previous step's
    output (there is no `input` field, unlike `filter`'s and `calculate`'s -- see
    `FilterStep`'s and `CalculateStep`'s docstrings), so it can never be a pipeline's
    first step.

    EXPLICIT DECISION: unlike `filter`
    and `calculate`, `group` deliberately did NOT gain an `input` field alongside them.
    `Options.group_by`/`Options.group_aggregate` are scalar fields, not a list like
    `Options.calculate` -- a leading (pre-project) `group` and a post-project `group`
    would have to share that one bucket. `calculate`'s equivalent bucket is a LIST that
    already accumulates leading and trailing entries side by side without collision (see
    `CalculateStep`'s docstring); `group`'s does not, so allowing a leading `group` would
    need either a second `Options` field (a real schema change) or a new compile-time
    rejection of "leading group + trailing group in the same pipeline" to keep that single
    bucket from being silently overwritten for the consumers that still read it
    (`estimate_data_size`, `legend`, `output_type`) -- genuinely separate design work, not a
    mechanical extension of the `filter`/`calculate` pattern. Left alone; a `group` step
    still cannot lead a pipeline.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, use_enum_values=True)

    op: Literal["group"] = "group"
    by: list[str]
    aggregate: AggregateFunction
    of: str = "value"
    as_: str | None = Field(default=None, alias="as")


Step = Annotated[
    Union[ProjectStep, JoinStep, FilterStep, CalculateStep, GroupStep],
    Field(discriminator="op"),
]


def _step_explicit_inputs(step: Any) -> list[str]:
    """The names a step references BY ITSELF, before any chaining default is applied --
    `[]` means "consumes the previous step's output" (`project`, `filter`, and
    `calculate` can also mean this via an omitted `input`; `group` always means this,
    having no field to be explicit with -- see `GroupStep`'s docstring for why it alone
    was left out of the `input`-field extension `calculate` received alongside
    `filter`'s).
    """
    if isinstance(step, (ProjectStep, FilterStep, CalculateStep)):
        return [step.input] if step.input is not None else []
    if isinstance(step, JoinStep):
        return list(step.inputs)
    return []


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------


class Output(BaseModel):
    """`format` reuses `DataFormat` (tabular + `tif`) and `TileFormat` (`png`/`jpg`)
    directly rather than a third format enum. `color_map` is only meaningful for a
    rendered format; not checked here (needs to know the output is single-valued first,
    which itself needs compile-time schema knowledge for the general case -- see
    `validate_pipeline`).

    `sort`/`select` are exactly `DataOptions.sort`/`.select`'s own flat semantics (a
    comma-separated `[-]column` list, and a comma-separated column subset, respectively)
    -- reused verbatim rather than a second grammar; see `planner._compile_single_input_
    chain`, which is the only shape that lowers them today (the paired shape rejects
    both -- see `_compile_paired_notebook_shape`'s docstring on why).
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    format: DataFormat | TileFormat
    color_map: str | None = None
    sort: str | None = None
    select: str | None = None


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


class Pipeline(BaseModel):
    """The canonical request: named `inputs`, an ordered non-empty list of `steps`, and
    an `output` description. Constructing this directly (or via `model_validate`) only
    enforces SHAPE (pydantic's own field/type checks, plus `InputSelection`'s scalar-list
    coercion). Use `parse_pipeline()` to also run `validate_pipeline()`'s business rules
    and get every rejection as `PipelineError` -- see the module docstring for why the
    two are not the same pydantic hook.
    """

    model_config = ConfigDict(extra="forbid")

    inputs: dict[str, InputSelection] = Field(min_length=1)
    steps: list[Step] = Field(min_length=1)
    output: Output


def validate_pipeline(pipeline: Pipeline) -> None:
    """Cross-step, plan-time validation over an already shape-valid `Pipeline`. Every
    check here can run without opening a file or knowing a crosswalk; anything that
    can't is named explicitly below, with the reason.

    Checks performed:

    - every input name matches the identifier/reserved-name rule (`_validate_name`);
    - **chaining**: the first step must name its input(s) explicitly (`project.input`,
      `filter.input`, `calculate.input`, or `join.inputs` -- `group` never can, so it may
      never be the first step, see `GroupStep`'s docstring for why it alone was left out
      of the `input`-field extension `calculate` received alongside `filter`'s); a later
      step with no explicit
      reference consumes the immediately preceding step's output; every explicit
      reference (an input name, or a prior -- not later -- step's `as`) must already be
      known at that point;
    - every step's `as`, when given, matches the identifier/reserved-name rule and does
      not collide with an input name or an earlier step's `as`;
    - `project`/`group` aggregates are in `fold.DECOMPOSABLE_AGGREGATES` (reusing
      `validate_decomposable` rather than a second list -- see that function);
    - `join.type == "attribute"` is rejected as reserved/not-yet-supported; `join.inputs`
      has no duplicate name;
    - `filter.where` is non-empty and, so far as is possible without knowing the column
      namespace, structurally sound: `expressions.classify_where` rejects an aggregate
      call or a subquery in any conjunct (every conjunct here names an input/step output,
      never the literal `value` placeholder, so every conjunct takes classify_where's
      "dimension" branch and gets that check; see the module docstring on why namespace
      checking itself is compile-time-only);
    - the output single-value-column rule, to the extent checkable without a schema: a
      `tif`/tile output whose LAST step is a `join` is rejected outright (a join always
      produces one column per joined input, and nothing narrows it after). A `join`
      earlier in the pipeline followed by `filter` (which drops no columns) or by an
      unrelated `project` is NOT caught here -- that needs to track column count through
      arbitrary op sequences, which this layer does not attempt; a `group` step DOES
      narrow to one column, by construction, wherever it appears.

    Deliberately NOT checked here (every one needs the executor's compile-time schema/
    crosswalk knowledge, named at the point in this docstring the spec called it out):
    the many-to-one/one-to-many `project.aggregate` requirement/rejection; whether a
    `join`'s shared dimensions actually align (or need to be pinned); the full
    column-namespace check on `filter.where` (which name is a real column here);
    dimension NAME/VALUE validity in `InputSelection`'s filters; and the general
    multi-column-output case described above.

    A step's `as` that no later step ever references is deliberately NOT an error --
    a pipeline may compute and name an intermediate it doesn't end up using further
    (readability, or reuse the next time the pipeline is edited). Not treated as
    warning-worthy in code (no logging call here); this paragraph is the record of that
    choice.
    """
    for name in pipeline.inputs:
        _validate_name(name, "input")

    known_names = set(pipeline.inputs)
    for i, step in enumerate(pipeline.steps, start=1):
        _validate_step_rules(i, step)

        explicit = _step_explicit_inputs(step)
        if explicit:
            for ref in explicit:
                if ref not in known_names:
                    raise PipelineError(
                        f"step {i} ({step.op}) references unknown input/step name {ref!r} "
                        f"-- known at this point: {sorted(known_names)}"
                    )
        elif i == 1:
            raise PipelineError(
                f"step 1 ({step.op}) has no input to chain from -- the first step must "
                "name its input explicitly (project.input, filter.input, "
                "calculate.input, or join.inputs)"
            )

        as_name = getattr(step, "as_", None)
        if as_name is not None:
            _validate_name(as_name, "step 'as'")
            if as_name in known_names:
                raise PipelineError(
                    f"step {i} 'as' name {as_name!r} collides with an existing input or "
                    "step name"
                )
            known_names.add(as_name)

    _validate_output_rule(pipeline)


def _validate_step_rules(i: int, step: Any) -> None:
    if isinstance(step, ProjectStep):
        # `count_by_value` is exempted here, specifically for `project` -- it is a ZONAL
        # aggregate (one categorical-count value per zone: the machinery
        # `expressions.count_by_value_frame` already implements, reused unchanged via
        # `Options.summarize_aggregate`), unlike `GroupStep`'s aggregate, which folds a
        # running per-row value across an offset-batched read and has no meaning for
        # `count_by_value` at all -- see `ProjectStep`'s own docstring and
        # `fold.DECOMPOSABLE_AGGREGATES`.
        if step.aggregate is not None and step.aggregate != "count_by_value":
            _require_decomposable(step.aggregate, f"step {i} (project) aggregate")
    elif isinstance(step, JoinStep):
        if step.type == "attribute":
            raise PipelineError(
                f"step {i} (join) type='attribute' is not yet supported -- reserved for a "
                "future release; only 'spatial' is implemented"
            )
        if len(set(step.inputs)) != len(step.inputs):
            raise PipelineError(f"step {i} (join) inputs contains duplicate name(s): {step.inputs}")
    elif isinstance(step, FilterStep):
        _validate_where_structurally(i, step.where)
    elif isinstance(step, CalculateStep):
        # Structural validation only: the whitelist/no-aggregate/no-comparison rules run
        # here (`allowed_columns=None` skips the namespace check, since which column name
        # is in scope -- the input/step name, or the `value` placeholder -- is a
        # compile-time question; the planner re-validates the rewritten expression against
        # the real namespace). `where` gets the same structural treatment `filter.where`
        # does.
        try:
            parse_calculate_expression(step.expression, allowed_columns=None)
        except ExpressionError as exc:
            raise PipelineError(
                f"step {i} (calculate) expression {step.expression!r} is invalid: {exc}"
            ) from exc
        if step.where is not None:
            _validate_where_structurally(i, step.where, op="calculate")
    elif isinstance(step, GroupStep):
        _require_decomposable(step.aggregate, f"step {i} (group) aggregate")
    else:  # pragma: no cover -- exhaustive over the Step union
        raise PipelineError(f"step {i} has unrecognized op {getattr(step, 'op', None)!r}")


def _validate_where_structurally(i: int, where: str, op: str = "filter") -> None:
    if not where or not where.strip():
        raise PipelineError(f"step {i} ({op}) where must be a non-empty string")
    try:
        classify_where(where)
    except ExpressionError as exc:
        raise PipelineError(f"step {i} ({op}) where={where!r} is invalid: {exc}") from exc


def _validate_output_rule(pipeline: Pipeline) -> None:
    if pipeline.output.format not in _SINGLE_VALUE_COLUMN_FORMATS:
        return
    last_step = pipeline.steps[-1]
    if isinstance(last_step, JoinStep):
        raise PipelineError(
            "tif/tile output requires exactly one value column, but the pipeline's last "
            "step is a join, which produces one named value column per joined input -- "
            "add a `group` (or narrow with `filter`/a column selection) after the join"
        )


def parse_pipeline(data: Mapping[str, Any]) -> Pipeline:
    """Parse `data` (the JSON-body shape: `{"inputs": ..., "steps": ..., "output": ...}`)
    into a fully-validated `Pipeline` -- pydantic's shape checks plus `validate_pipeline`'s
    business rules, both surfaced as `PipelineError`. The one entry point every caller
    (this module's own `pipeline_from_query`, tests, and the eventual endpoint) should use
    instead of `Pipeline.model_validate` directly, since that alone only enforces shape --
    see the module docstring for why the two cannot be merged into a single pydantic hook.
    """
    try:
        pipeline = Pipeline.model_validate(data)
    except ValidationError as exc:
        raise PipelineError(f"invalid pipeline: {exc}") from exc
    validate_pipeline(pipeline)
    return pipeline


# --------------------------------------------------------------------------------------
# The dotted GET/tile query codec
# --------------------------------------------------------------------------------------

# Field names on `InputSelection` that are scalar strings, never comma-split -- every
# other key under `input.<name>.*` is a dimension filter and IS comma-split into a list
# (even a single value: "input.marsh.scenario_id=7" decodes to `["7"]`, which
# `InputSelection._normalize_dimension_filters` would have produced anyway from a bare
# scalar, so the two are indistinguishable after decoding -- see `pipeline_from_query`'s
# docstring on why this loses no information the query encoding could represent).
# `grid_cell` belongs here, not in the dimension-filter branch: it's a declared
# `InputSelection` field (a reserved Filter-group parameter -- see that model's
# docstring), never split into a list, exactly like `Options.grid_cell`'s own flat query
# spelling elsewhere in this package.
_INPUT_SCALAR_FIELDS = frozenset({"variable", "time_unit", "grid", "grid_cell"})

# Per-op field whitelist and which of those fields are comma-list-valued, for the
# `step.<i>.<field>` namespace. `where` is deliberately never in a list-fields set: it is
# a single SQL-flavored string that may itself contain commas, spaces, and comparison
# operators verbatim (see the property tests) and must never be split.
_STEP_FIELDS: dict[str, frozenset[str]] = {
    "project": frozenset({"op", "input", "to", "aggregate", "as"}),
    "join": frozenset({"op", "type", "inputs", "as"}),
    "filter": frozenset({"op", "where", "as"}),
    "calculate": frozenset({"op", "expression", "where", "as"}),
    "group": frozenset({"op", "by", "aggregate", "of", "as"}),
}
_STEP_LIST_FIELDS: dict[str, frozenset[str]] = {
    "project": frozenset(),
    "join": frozenset({"inputs"}),
    "filter": frozenset(),
    # `expression` and `where` are single opaque strings that may themselves contain
    # commas verbatim -- never comma-split, same as `filter.where`.
    "calculate": frozenset(),
    "group": frozenset({"by"}),
}

_OUTPUT_FIELDS = frozenset({"format", "color_map", "sort", "select"})


def _insert_dotted(tree: dict[str, Any], segments: list[str], value: str, full_key: str) -> None:
    """Insert `value` into the nested-dict `tree` at `segments` (a dotted parameter
    name's parts, still strings -- list-index segments are only recognized as such
    later, by `_listify`). Raises `PipelineError` naming `full_key` the moment a leaf
    value and a nested value are asked to occupy the same path.
    """
    seg = segments[0]
    if len(segments) == 1:
        if isinstance(tree.get(seg), dict):
            raise PipelineError(
                f"parameter {full_key!r} conflicts with other parameter(s) nested under "
                "the same path"
            )
        tree[seg] = value
        return
    child = tree.get(seg)
    if child is None:
        child = {}
        tree[seg] = child
    elif not isinstance(child, dict):
        raise PipelineError(
            f"parameter {full_key!r} conflicts with a leaf value already set at a shorter "
            "path with the same prefix"
        )
    _insert_dotted(child, segments[1:], value, full_key)


def _listify(node: Any, path: str) -> Any:
    """Recursively turn every dict node whose keys are ALL 1-based, contiguous, digit
    strings into a list (0-based internally), leaving every other dict node (and every
    leaf) alone. `path` is the dotted prefix seen so far, purely for error messages.
    """
    if not isinstance(node, dict):
        return node

    keys = list(node)
    digit_keys = [k for k in keys if k.isdigit()]
    if digit_keys and len(digit_keys) != len(keys):
        raise PipelineError(
            f"parameter group {path or '<root>'!r} mixes list-index segments with named "
            f"segments: {sorted(set(keys) - set(digit_keys))}"
        )
    if not digit_keys:
        return {k: _listify(v, f"{path}.{k}" if path else k) for k, v in node.items()}

    for k in digit_keys:
        if k == "0" or (len(k) > 1 and k.startswith("0")):
            raise PipelineError(
                f"parameter group {path or '<root>'!r} has an invalid list index {k!r} "
                "-- indices are 1-based with no leading zero"
            )
    indices = sorted(int(k) for k in digit_keys)
    if indices != list(range(1, len(indices) + 1)):
        raise PipelineError(
            f"parameter group {path or '<root>'!r} has non-contiguous list indices "
            f"{indices} -- indices must run 1..N with no gaps"
        )
    return [_listify(node[str(i)], f"{path}.{i}" if path else str(i)) for i in indices]


def _decode_input(name: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PipelineError(
            f"input {name!r} must have named sub-keys, e.g. input.{name}.variable=..."
        )
    data: dict[str, Any] = {}
    for field, value in raw.items():
        if not isinstance(value, str):
            raise PipelineError(f"input {name!r} field {field!r} must be a scalar query value")
        data[field] = value if field in _INPUT_SCALAR_FIELDS else value.split(",")
    return data


def _decode_step(index: int, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PipelineError(f"step {index} must have named sub-keys, e.g. step.{index}.op=...")
    op = raw.get("op")
    if op is None:
        raise PipelineError(f"step {index} is missing required field 'op'")
    if op not in _STEP_FIELDS:
        raise PipelineError(f"step {index} has unknown op {op!r}")

    allowed = _STEP_FIELDS[op]
    list_fields = _STEP_LIST_FIELDS[op]
    unknown = set(raw) - allowed
    if unknown:
        raise PipelineError(f"step {index} ({op}) has unknown field(s): {sorted(unknown)}")

    data: dict[str, Any] = {"op": op}
    for field, value in raw.items():
        if field == "op":
            continue
        if not isinstance(value, str):
            raise PipelineError(f"step {index} ({op}) field {field!r} must be a scalar query value")
        if field in list_fields:
            data[field] = [] if value == "" else value.split(",")
        else:
            data[field] = value
    return data


def _decode_output(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PipelineError("'output' must have named sub-keys, e.g. output.format=...")
    unknown = set(raw) - _OUTPUT_FIELDS
    if unknown:
        raise PipelineError(f"output has unknown field(s): {sorted(unknown)}")
    return dict(raw)


def pipeline_from_query(params: Mapping[str, str]) -> Pipeline:
    """Decode the dotted GET/tile query encoding into a fully-validated `Pipeline`.

    Encoding rule (see `pipeline_to_query` for the inverse): dot = nesting, an integer
    path segment = a 1-based list position, comma = a list value (dimension filter
    arrays, and `join.inputs`) -- NEVER inside `filter.where`, which is one opaque
    string. `params` is the already-decoded (no percent-encoding) mapping an HTTP
    framework's query-parameter object would hand over.

    Malformed input -- a gap in list indices, a leaf/nested conflict at the same path, an
    unrecognized top-level group, a missing top-level group, an unknown step op or field
    -- raises `PipelineError` naming the offending parameter or parameter group. Once the
    dotted structure resolves cleanly, the resulting JSON-shaped dict is handed to
    `parse_pipeline`, so a query that's well-formed AS A QUERY but violates a business
    rule (e.g. a join with only one input) is rejected the same way parsing the
    equivalent JSON body would be.

    Note on fidelity: a query value is always a string, so a JSON dimension filter given
    as a bare scalar (`"model_group_id": 500`) and the same filter given as a query
    parameter (`input.marsh.model_group_id=500`) are indistinguishable after decoding --
    both become `["500"]` (a string, not the original int) once
    `InputSelection._normalize_dimension_filters` runs. This does not affect the
    encoding's bijection property (a query round-tripped through the model always
    produces the same query back -- see `pipeline_to_query`), and dimension VALUE typing
    is deferred to compile time regardless of which encoding produced the pipeline.
    """
    if not params:
        raise PipelineError("empty query: no pipeline parameters given")

    tree: dict[str, Any] = {}
    for key, value in params.items():
        if not isinstance(key, str) or not key:
            raise PipelineError(f"invalid parameter name {key!r}")
        if not isinstance(value, str):
            raise PipelineError(f"parameter {key!r} must have a string value")
        _insert_dotted(tree, key.split("."), value, key)

    top_level = {"input", "step", "output"}
    unknown_top = set(tree) - top_level
    if unknown_top:
        raise PipelineError(f"unknown top-level parameter group(s): {sorted(unknown_top)}")
    missing_top = top_level - set(tree)
    if missing_top:
        raise PipelineError(f"missing required parameter group(s): {sorted(missing_top)}")

    listified = _listify(tree, "")

    inputs_raw = listified["input"]
    if not isinstance(inputs_raw, dict):
        raise PipelineError("'input' must use named sub-keys, e.g. input.<name>.variable=...")
    inputs_json = {name: _decode_input(name, fields) for name, fields in inputs_raw.items()}

    steps_raw = listified["step"]
    if not isinstance(steps_raw, list):
        raise PipelineError("'step' must use 1-based list indices, e.g. step.1.op=...")
    steps_json = [_decode_step(i, s) for i, s in enumerate(steps_raw, start=1)]

    output_json = _decode_output(listified["output"])

    return parse_pipeline({"inputs": inputs_json, "steps": steps_json, "output": output_json})


def _query_sort_key(param_name: str) -> tuple:
    """Natural sort: a digit segment compares numerically (`step.2` before `step.10`),
    so the canonical order stays intuitive past 9 steps/inputs -- dict equality (what the
    bijection property test actually checks) does not depend on this, but the spec calls
    for a deterministic order as the cache-key normalization, which does.
    """
    return tuple((0, int(seg)) if seg.isdigit() else (1, seg) for seg in param_name.split("."))


def pipeline_to_query(pipeline: Pipeline) -> dict[str, str]:
    """Encode `pipeline` into the dotted GET/tile query form -- the inverse of
    `pipeline_from_query`. Returns a plain `dict[str, str]`, its keys in the canonical
    (natural-sorted) order described in the module docstring, so two `Pipeline`s that are
    equivalent after normalization serialize to the identical parameter set, regardless
    of the order fields happened to arrive in on either side.

    A field at its default (`GroupStep.of == "value"`) or unset (`None`) is omitted
    rather than written out explicitly -- `pipeline_from_query` reconstructs the same
    default/`None` on decode, which is what keeps this round-trip stable rather than
    merely reversible once.
    """
    params: dict[str, str] = {}

    for name, selection in pipeline.inputs.items():
        prefix = f"input.{name}"
        params[f"{prefix}.variable"] = str(selection.variable)
        params[f"{prefix}.time_unit"] = str(selection.time_unit)
        params[f"{prefix}.grid"] = str(selection.grid)
        if selection.grid_cell is not None:
            params[f"{prefix}.grid_cell"] = selection.grid_cell
        for dim_name, values in selection.dimension_filters.items():
            params[f"{prefix}.{dim_name}"] = ",".join(str(v) for v in values)

    for i, step in enumerate(pipeline.steps, start=1):
        prefix = f"step.{i}"
        params[f"{prefix}.op"] = step.op
        if isinstance(step, ProjectStep):
            if step.input is not None:
                params[f"{prefix}.input"] = step.input
            params[f"{prefix}.to"] = str(step.to)
            if step.aggregate is not None:
                params[f"{prefix}.aggregate"] = str(step.aggregate)
            if step.as_ is not None:
                params[f"{prefix}.as"] = step.as_
        elif isinstance(step, JoinStep):
            params[f"{prefix}.type"] = step.type
            params[f"{prefix}.inputs"] = ",".join(step.inputs)
            params[f"{prefix}.as"] = step.as_
        elif isinstance(step, FilterStep):
            params[f"{prefix}.where"] = step.where
            if step.as_ is not None:
                params[f"{prefix}.as"] = step.as_
        elif isinstance(step, CalculateStep):
            params[f"{prefix}.expression"] = step.expression
            if step.where is not None:
                params[f"{prefix}.where"] = step.where
            if step.as_ is not None:
                params[f"{prefix}.as"] = step.as_
        elif isinstance(step, GroupStep):
            params[f"{prefix}.by"] = ",".join(step.by)
            params[f"{prefix}.aggregate"] = str(step.aggregate)
            if step.of != "value":
                params[f"{prefix}.of"] = step.of
            if step.as_ is not None:
                params[f"{prefix}.as"] = step.as_
        else:  # pragma: no cover -- exhaustive over the Step union
            raise PipelineError(f"cannot encode unknown step op {step.op!r}")

    params["output.format"] = str(pipeline.output.format)
    if pipeline.output.color_map is not None:
        params["output.color_map"] = pipeline.output.color_map
    if pipeline.output.sort is not None:
        params["output.sort"] = pipeline.output.sort
    if pipeline.output.select is not None:
        params["output.select"] = pipeline.output.select

    return dict(sorted(params.items(), key=lambda item: _query_sort_key(item[0])))
