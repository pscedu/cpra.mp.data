"""The canonical request model for analysis pipelines: named inputs, an ordered list of
operations over them, and an output description -- plus the two encodings the request can
arrive in (a JSON body, and a dotted GET query string suitable for tile URLs) and the
structural, plan-time validation both encodings share.

This module builds the REPRESENTATION and its STRUCTURAL validation only: name shapes,
chaining, `as` collisions, per-step field rules that need no more than the step itself,
and everything about `where`/`expression` text that is checkable without knowing the
column namespace. It does NOT type shapes -- whether a `project` crosses grid kinds
legally, whether a `join`'s operands agree, whether the final frame fits the output
format -- because that needs each input's grid kind and a walk of the whole chain, which
`request/ops.py` performs (see that module's `validate_shapes`); `pipeline.py` cannot do
that walk itself without importing `ops.py`, which already imports the models defined
here for `compile_ops`, and this codebase does not do import cycles. `parse_pipeline`
returns a structurally-valid `Pipeline`; a caller that intends to compile or execute it
still needs `ops.validate_shapes` first.

An input names a dataset -- `variable`/`time_unit`/`grid` -- and nothing else. There is no
input-level or output-level filtering: every filter, spatial or otherwise, is a `filter`
step, so a request narrows what it selects exactly the same way regardless of when in the
chain the narrowing happens.

**Two encodings, one model.** `Pipeline` is parsed directly from a JSON-shaped dict (the
POST body) via `parse_pipeline`, or from a flat dotted query mapping (the GET/tile
encoding) via `pipeline_from_query` -- both funnel through the same `Pipeline` model and
the same `validate_pipeline` business rules, so a pipeline expressed either way is
accepted or rejected identically. `pipeline_to_query` is the inverse of
`pipeline_from_query`, canonicalized (sorted) so equivalent pipelines share one cache key.

**Why business validation is NOT wired into pydantic validators.** Pydantic v2 catches any
exception a `field_validator`/`model_validator` raises -- including a custom `ValueError`
subclass -- and re-raises it as `pydantic.ValidationError` (itself a `ValueError`
subclass, but not `RequestError`), regardless of where in the model the raise happened.
Reusing `RequestError` as a single, directly-catchable family therefore means this
module's business rules run as a plain function (`validate_pipeline`) called AFTER a
`Pipeline` is already constructed, not as a pydantic hook -- see `parse_pipeline`, the
combined "parse the shape, then validate the structural rules" entry point.

**A pipeline may have zero steps.** A plain data query -- no named analysis, nothing to
compute -- is a `Pipeline` with an empty `steps` list: the inputs (necessarily exactly
one; see `validate_pipeline`, since nothing selects among several with no step to combine
or choose between them) and an output description, nothing else.
"""

import re
from collections.abc import Mapping, Sequence
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cpra.mp.data.analysis.request import aggregates
from cpra.mp.data.analysis.request.errors import AggregateError, RequestError
from cpra.mp.data.analysis.request.expressions import (
    ExpressionError,
    classify_where_kinds,
    parse_calculate_expression,
)
from cpra.mp.data.analysis.request.types import AggregateFunction, DataFormat, Grid, TileFormat, TimeUnit, Variable
from cpra.mp.data.dimensions import DIMENSIONS

# Every name a user chooses -- an input's key in `Pipeline.inputs`, or a step's `as` --
# shares one shape rule: it has to be usable as an identifier/column name downstream, and
# it must not shadow a name the row model already gives fixed meaning (the value column,
# the grid key every row carries, the categorical dimension `count_by_value` introduces),
# a positional band name, or a dimension the data structure already declares.
_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
RESERVED_NAMES = frozenset({"value", "grid_cell", "value_category"})

# The positional band-name pattern (`b0`, `b1`, ...) and the reserved band-wide index
# column: both are recomputed at execution time and never stored, so a user-chosen name
# that merely matches one of these shapes would be silently shadowed by it later.
# Public (not underscore-prefixed) because `execution/frame.py` reuses both as the single
# source of truth for what counts as a positional name -- copying the pattern there would
# let the two definitions drift.
BAND_NAME_RE = re.compile(r"^b[0-9]+$")
RESERVED_BAND_INDEX_COLUMN = "_band"

# The two CARRIED per-pixel columns a band-wide frame can pick up alongside its `b<i>`
# bands -- `execution/frame.py`'s `extra_columns` contract. `ZONE_COLUMN` is a
# raster-to-vector crosswalk's zone ids, attached to a raster origin's own read;
# `DOMAIN_COLUMN` records whether a pixel is inside the crosswalk domain at all, attached
# by a vector-to-raster `project`. Each constant's full reason for existing is at its
# execution-layer use site (`execution/project.py` and `execution/frame.py` respectively,
# which import these rather than defining their own).
#
# They live here, in the request layer, for the same reason `BAND_NAME_RE` does: the
# shape-typing walk has to know these names before anything is opened. A `where` naming
# one of them is a PER-PIXEL predicate (a value conjunct, masked per band) rather than a
# dimension conjunct selecting bands out of metadata that carries no such column, and
# `ops.FrameType.extra_columns` is what tells `ops._type_filter` which frames carry them.
# Double-underscored so neither can collide with a dataset dimension or a user-chosen
# name (`_validate_name` rejects both shapes independently).
ZONE_COLUMN = "__zone__"
DOMAIN_COLUMN = "__domain__"

# Output formats that render to a single raster/image surface (a GeoTIFF band or a map
# tile pixel) rather than a table -- these need a band-wide (not long) final frame; see
# `ops.validate_shapes`'s sink-shape rule, which is what actually enforces this.
#
# `pbf` is deliberately NOT here. A vector tile is drawn, like a `png`, but it is drawn
# from ROWS: the geometry comes from the grid's PMTiles boundary archive and the pipeline
# contributes one value per `grid_cell`, so what it needs from the final frame is what a
# CSV needs, plus the stricter requirement that the frame be addressed against a vector
# grid at all. That extra half is `validate_shapes`'s own `pbf` rule, which is why `pbf`
# cannot simply be sorted into one of these two buckets.
RASTER_OUTPUT_FORMATS = frozenset({DataFormat.tif.value, TileFormat.png.value, TileFormat.jpg.value})


def _validate_name(name: str, kind: str, *, allow_dimension: bool = False) -> None:
    """The identifier rule for a name a request CHOOSES -- an input key, a step's `as`, an
    `unpivot`'s `into` column.

    `allow_dimension` opts out of the last check alone, and exists for exactly one caller:
    an `unpivot`'s label column, whose whole job is to hold variable names and whose
    default (`"variable"`) is therefore a dimension on purpose. For every other name the
    collision is a real hazard -- see that check's own message -- so the opt-out is stated
    at the one call site that needs it rather than being the default.
    """
    if not name:
        raise RequestError(f"{kind} name must not be empty")
    if not _NAME_RE.match(name):
        raise RequestError(
            f"{kind} name {name!r} must match [a-z_][a-z0-9_]* (lowercase letters, digits, "
            "underscore; must not start with a digit)"
        )
    if name in RESERVED_NAMES:
        raise RequestError(
            f"{kind} name {name!r} collides with a reserved column name {sorted(RESERVED_NAMES)}"
        )
    if name == RESERVED_BAND_INDEX_COLUMN or BAND_NAME_RE.match(name):
        raise RequestError(
            f"{kind} name {name!r} matches the positional band-name pattern (b0, b1, ...) "
            "or the reserved _band index column, and would be shadowed by it"
        )
    if name in DIMENSIONS and not allow_dimension:
        raise RequestError(
            f"{kind} name {name!r} collides with a dimension the data structure already "
            "declares -- this only bites once a join renames a value column to this "
            "name, which is exactly why it is rejected up front instead"
        )


def _validate_label_column_name(i: int, into: str) -> None:
    """An `unpivot`'s `into`: the identifier rule minus the dimension-collision check --
    see `UnpivotStep`, and `_validate_name`'s `allow_dimension`, for why that one does not
    apply to a column whose contents ARE dimension values.
    """
    _validate_name(into, f"step {i} (unpivot) 'into'", allow_dimension=True)


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


class InputSelection(BaseModel):
    """One named dataset selection: `variable`/`time_unit`/`grid`, nothing else. An input
    names a dataset; it does not filter one -- every filter (dimension, value, or
    spatial) is expressed as a `filter` step instead, so a request narrows its result the
    same way no matter where in the chain the narrowing happens.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    variable: Variable
    time_unit: TimeUnit
    grid: Grid


# The three columns that identify WHICH DATASET a row came from, rather than describing
# the observation itself. A vector scan materializes each of them as an ordinary literal
# column (every row of a `variable=marsh` read carries `variable = "marsh"`), which makes
# them indistinguishable from real dimension columns by schema alone -- so anything that
# reasons about "this frame's dimensions" has to name them explicitly to exclude them.
#
# Derived from `InputSelection`'s own fields rather than spelled out, so narrowing or
# widening what an input selects cannot leave a stale second list behind.
DATASET_SELECTION_COLUMNS = frozenset(InputSelection.model_fields)


# --------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------


class ProjectStep(BaseModel):
    """Move a dataset across grids via a crosswalk. `input` is optional -- an omitted
    `input` chains from the previous step's output (see `validate_pipeline`'s chaining
    rule). Whether `aggregate` is required, forbidden, or may be `count_by_value` depends
    on the project's direction (vector/raster on each side), which needs each operand's
    grid kind to know -- see `ops.validate_shapes`, which is what actually enforces
    aggregate presence/legality and the raster-to-raster rejection; this model only
    enforces the shape (`aggregate` is an optional `AggregateFunction`).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, use_enum_values=True)

    op: Literal["project"] = "project"
    input: str | None = None
    to: Grid
    aggregate: AggregateFunction | None = None
    as_: str | None = Field(default=None, alias="as")


class JoinStep(BaseModel):
    """Combine two or more named inputs/step-outputs on shared dimensions plus
    `grid_cell`. `type="attribute"` parses (it is a real member of the vocabulary) but is
    rejected by `validate_pipeline` as reserved/not-yet-implemented -- only `"spatial"`
    executes today. `inputs` must name at least two things and (checked here, since it
    needs no pipeline context) must not repeat a name; that each name actually resolves
    to a known input or a prior step's `as` is checked by `validate_pipeline`, and that
    the operands' SHAPES are compatible (both long, or both band-wide on the same grid)
    is checked by `ops.validate_shapes`, which has the shape-typing machinery this module
    does not.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, use_enum_values=True)

    op: Literal["join"] = "join"
    type: Literal["spatial", "attribute"]
    inputs: list[str] = Field(min_length=2)
    as_: str = Field(alias="as")


class FilterStep(BaseModel):
    """A predicate over in-scope named columns. `where` may combine dimension, value,
    and/or spatial (`grid_cell`) conjuncts with `AND` -- see
    `expressions.classify_where_kinds` for how each is told apart and what it lowers to.
    `input` is optional -- an omitted `input` chains from the previous step's output,
    mirroring `ProjectStep.input`. A pipeline step always runs at its own declared
    position, so a `filter` may lead a pipeline, sit between other steps, or follow one;
    there is no reordering to guard against the way a flat, order-free request surface
    would need.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    op: Literal["filter"] = "filter"
    input: str | None = None
    where: str
    as_: str | None = Field(default=None, alias="as")


class CalculateStep(BaseModel):
    """UNCONDITIONALLY rewrite the running value: `expression` (arithmetic over the running
    value and numeric literals, or over a `join`'s operand names) applied to every
    row/band. `input` is optional, mirroring `FilterStep.input`. `calculate` always produces
    exactly one value column, whatever it had going in -- the mechanism the
    single-value-column sink rule relies on to collapse a `join`'s several named value
    columns back to one; see `ops.validate_shapes`.

    **`calculate` is sugar for a `case` with only an `else`**, and carries no `where` of its
    own: `ops.compile_ops` lowers it to `CaseOp(cases=[], else_=expression)`, which is the
    same thing said in the op that already means it. A rewrite that applies only SOMETIMES
    is a `case` with an arm, and the `else` that case writes is the pass-through it wants --
    `else: "value"` to leave unmatched rows alone, or no `else` to null them. Spelling it
    that way is what makes "which rows does this touch" one question with one answer,
    instead of a `where` and an arm's `when` that would have to be read together.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    op: Literal["calculate"] = "calculate"
    input: str | None = None
    expression: str
    as_: str | None = Field(default=None, alias="as")


class GroupStep(BaseModel):
    """Collapse `by` (may be empty, meaning "collapse to one row per `grid_cell`") with
    `aggregate` applied to `of` (defaults to `"value"`, generalizing the aggregand --
    "first year a condition held" is `aggregate="min", of="calendar_year"`).
    `aggregate` must fold (`request.aggregates.DECOMPOSABLE_AGGREGATES`) and must not be
    `count_by_value` (a different output shape group's one-row-per-group contract cannot
    absorb; see `request.aggregates.validate_group_aggregate`); `value` must not be a
    member of `by` (`request.aggregates.validate_group_by`) since it names the measured
    quantity, not a dimension. `input` chains from the previous step when omitted, the
    same as `project`/`filter`/`calculate`; naming it is what lets a `group` lead a
    pipeline, so "read one dataset and collapse it" needs no filter in front of it purely
    to satisfy the chaining rule.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, use_enum_values=True)

    op: Literal["group"] = "group"
    input: str | None = None
    by: list[str]
    aggregate: AggregateFunction
    of: str = "value"
    as_: str | None = Field(default=None, alias="as")


class CaseWhen(BaseModel):
    """One arm of a `case`: rewrite the running value to `then` wherever `when` matches.
    `when` is a predicate in the same SQL-flavored language a `filter`'s `where` is written
    in, and `then` is a value expression in the same arithmetic language
    `calculate.expression` is -- so `then` may name a column (`"land_type"`) as readily as
    a constant (`"6"`), and needs nothing extra to do it.

    `when` may mix VALUE conjuncts (`"value > 0"`) with DIMENSION ones (`"year = 2070"`),
    and the two halves mean different things: a value conjunct is a per-pixel mask, while a
    dimension conjunct selects which BANDS (or which rows, in the long shape) this arm
    applies to at all. An arm whose dimension half excludes a band is simply not in that
    band's chain -- the band falls to whichever later arm does apply, or to the `else`. An
    arm whose `when` is PURELY dimensional therefore matches its applicable bands entirely.

    Structurally validated here (non-empty `when`, well-formed `then`). That the value half
    names only value columns the frame carries is checked when the arm is compiled; that
    the dimension half names real metadata columns is checked at execution, where the band
    metadata (or the long frame's schema) is in hand. `ops._type_case` rejects only a
    `grid_cell` conjunct, which would resample half the frame to points.
    """

    model_config = ConfigDict(extra="forbid")

    when: str
    then: str


class CaseStep(BaseModel):
    """Rewrite the running value by the FIRST matching arm of `cases`, falling through to
    `else` (default null) where none match -- one op for what a reclassification would
    otherwise need a chain of steps and a disjointness trick to say (see
    `analyses/lnd_type_plus.py`'s depth classes, and `analyses/lnd_conf.py`'s
    per-scenario weights).

    **Why this is not a chain of conditional rewrites.** A step that rewrites the RUNNING
    value sees whatever an earlier one already wrote, so a chain of them is only
    independent when each target code happens to be disjoint from every later step's source
    domain -- a property of the particular numbers chosen, not of the request language.
    `case` evaluates every arm's `when` against the value the frame arrived with and takes
    the first that matches, so the arms are independent by construction and their ORDER is
    the only thing that decides precedence.

    `input` is optional, mirroring `FilterStep.input`. `case` always produces exactly one
    value column, and accepts a PAIRED band-wide frame, reducing it to one (see
    `ops._type_case`). A row/band matching no arm and given no `else` becomes null, which is
    how `case` removes data; `else: "value"` is the pass-through to write when it should
    not.

    `calculate` is this op with no arms at all -- see `CalculateStep`, which lowers to
    exactly that. The surface `cases` here keeps `min_length=1` because a `case` written
    with no arms is a `calculate` the author should have spelled as one.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    op: Literal["case"] = "case"
    input: str | None = None
    cases: list[CaseWhen] = Field(min_length=1)
    # `else` is a Python keyword, so the field is `else_` and the alias is what a JSON body
    # and a dotted query both actually spell -- the same split `as_`/`as` has.
    else_: str | None = Field(default=None, alias="else")
    as_: str | None = Field(default=None, alias="as")


class UnpivotStep(BaseModel):
    """Fold a `join`'s several operand-named value columns back into ONE `value` column,
    plus a label column (`into`) carrying the operand name each folded row came from --
    the inverse of what a `join` does to the frame's width, and the op that lets a
    multi-variable result reach a sink at all.

    **Why this exists rather than a wider sink rule.** `ops.validate_shapes` requires
    exactly one value column at the sink, and the two ways to satisfy that from a join's
    output are to COLLAPSE the operands into one number (`case`/`calculate`, which is what
    `lnd_type_plus` does) or to keep all of them and stack them. Stacking is what a chart
    of five variables over one pixel needs, and it is the only one of the two that does
    not throw four of the five measurements away. Relaxing the sink rule instead would
    push the cost onto every consumer of `value` -- the shared chart base class, the
    vector-tile sink, and `_type_project`/`_type_group`, all of which name that column
    literally -- so the fold happens here, once, and everything downstream keeps seeing
    the one-value-column frame it already expects.

    **Long only.** A band-wide frame's bands are POSITIONAL (`b0`, `b1`, ...), and folding
    them means naming the band metadata each column stands for, which is the band-wide-to-
    long transition itself rather than a fold -- `ops._type_unpivot` rejects a band-wide
    input and says so. For a single-pixel chart that costs nothing: the `grid_cell` filter
    has already lowered the frame (`FILTER_LOWERING_TABLE`'s one shape-changing entry).

    `into` defaults to `"variable"`, which is a DIMENSION name on purpose -- a join names
    each operand after the variable it reads, so the folded labels are variable names and
    land in the column that already means that everywhere else. It is therefore checked
    against the reserved/band-name rules but NOT against `DIMENSIONS` the way an input or
    an `as` is (`_validate_label_column_name`): colliding with a dimension is the point.

    `input` is optional, chaining from the previous step, the same as
    `project`/`filter`/`calculate`/`case`/`group`.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    op: Literal["unpivot"] = "unpivot"
    input: str | None = None
    into: str = "variable"
    as_: str | None = Field(default=None, alias="as")


Step = Annotated[
    ProjectStep | JoinStep | FilterStep | CalculateStep | CaseStep | GroupStep | UnpivotStep,
    Field(discriminator="op"),
]


def _step_explicit_inputs(step: Any) -> list[str]:
    """The names a step references BY ITSELF, before any chaining default is applied --
    `[]` means "consumes the previous step's output", which every op but `join` can mean
    via an omitted `input`.
    """
    if isinstance(step, JoinStep):
        return list(step.inputs)
    return [step.input] if step.input is not None else []


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------


class Output(BaseModel):
    """`format` reuses `DataFormat` (tabular formats + `tif`) and `TileFormat`
    (`png`/`jpg`/`pbf`) directly rather than a third format enum. `color_map` is only
    meaningful for a rendered/raster format; not checked here (that needs to know the
    final frame's shape, which needs the shape-typing walk -- see `ops.validate_shapes`).
    `sort`/`select` are a comma-separated `[-]column` list and a comma-separated column
    subset, respectively; `validate_pipeline` checks their SYNTAX (via
    `_validate_output_sort_or_select`) but deliberately not whether a named column exists
    -- that half surfaces as Polars' `ColumnNotFoundError` at collect time (see
    `execution.sink.apply_output_shaping`).

    **No `compression` field.** Compression is a pair of QUERY parameters
    (`compression_algorithm`/`compression_level`, bound as `api.models.Compression` on
    the analysis routes), not part of the pipeline body. This model carried a
    `compression` field for a while that round-tripped through the dotted encoding and
    was never read by anything, so a caller who set it silently got the default instead;
    it is rejected outright now (`extra="forbid"`) rather than accepted and ignored.
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
    """The canonical request: named `inputs`, an ordered (possibly empty -- see the
    module docstring) list of `steps`, and an `output` description. Constructing this
    directly (or via `model_validate`) only enforces SHAPE (pydantic's own field/type
    checks). Use `parse_pipeline()` to also run `validate_pipeline()`'s structural rules,
    and `ops.validate_shapes()` after that for the shape-dependent rules this module
    cannot check on its own.
    """

    model_config = ConfigDict(extra="forbid")

    inputs: dict[str, InputSelection] = Field(min_length=1)
    steps: list[Step] = Field(default_factory=list)
    output: Output


def validate_pipeline(pipeline: Pipeline) -> None:
    """Structural, plan-time validation over an already shape-valid `Pipeline`. Every
    check here runs without opening a file, knowing a crosswalk, or knowing any input's
    grid kind; anything that needs one of those is `ops.validate_shapes`'s job instead
    (see that function's docstring for the full list it covers).

    Checks performed here:

    - every input name matches the identifier/reserved-name/dimension-collision rule
      (`_validate_name`);
    - a zero-step pipeline has exactly one input (nothing selects among several with no
      step to combine or choose between them);
    - **chaining**: the first step must name its input(s) explicitly (any op's `input`,
      or `join.inputs`); a later step with no explicit reference consumes the immediately
      preceding step's output; every explicit reference (an input name, or a prior -- not
      later -- step's `as`) must already be known at that point;
    - every step's `as`, when given, matches the identifier/reserved-name/dimension-
      collision rule and does not collide with an input name or an earlier step's `as`;
    - `group.aggregate` folds and is not `count_by_value`; `group.by` does not contain
      `value` (`request.aggregates`);
    - `join.type == "attribute"` is rejected as reserved/not-yet-supported; `join.inputs`
      has no duplicate name;
    - `filter.where` is non-empty and, so far as is possible without knowing the column
      namespace, structurally sound (`expressions.classify_where_kinds` rejects an
      aggregate call or a subquery in any dimension or spatial conjunct);
    - `calculate.expression` is well-formed arithmetic over the whitelist
      (`expressions.parse_calculate_expression`, column-namespace check deferred);
    - every `case` arm's `when` is a non-empty, structurally sound predicate and its
      `then` -- and the step's `else`, when given -- is well-formed arithmetic, both by
      the same two checks a `filter`'s `where` and a `calculate`'s `expression` get. That
      a `when`'s conjuncts name columns the frame actually carries is checked downstream --
      `ops._type_case` rejects a `grid_cell` one, and the rest resolve at execution, where
      the band metadata is in hand;
    - `unpivot.into` matches the identifier rule, minus the dimension-collision check a
      name a request chooses otherwise gets (`_validate_label_column_name`). That the
      frame actually has several value columns to fold, and that `into` does not collide
      with one of them, needs the typing walk and is `ops._type_unpivot`'s job;
    - `output.sort`/`output.select` are syntactically sound comma-separated `[-]column`
      lists -- no empty component, no bare `-`, no `-`-prefixed `select` entry, no name
      repeated within one list (`_validate_output_sort_or_select`). Whether a named
      column actually EXISTS on the final frame is deliberately not checked here -- see
      that function's docstring, and `execution.sink.apply_output_shaping`, for the other
      half of this split.

    A step's `as` that no later step ever references is deliberately NOT an error -- a
    pipeline may compute and name an intermediate it doesn't end up using further.
    """
    for name in pipeline.inputs:
        _validate_name(name, "input")

    if pipeline.output.sort is not None:
        _validate_output_sort_or_select(pipeline.output.sort, "sort")
    if pipeline.output.select is not None:
        _validate_output_sort_or_select(pipeline.output.select, "select")

    if not pipeline.steps and len(pipeline.inputs) != 1:
        raise RequestError(
            "a pipeline with no steps must have exactly one input -- add a step (at "
            "least a join) to combine multiple inputs, or drop to one"
        )

    known_names = set(pipeline.inputs)
    for i, step in enumerate(pipeline.steps, start=1):
        _validate_step_rules(i, step)

        explicit = _step_explicit_inputs(step)
        if explicit:
            for ref in explicit:
                if ref not in known_names:
                    raise RequestError(
                        f"step {i} ({step.op}) references unknown input/step name {ref!r} "
                        f"-- known at this point: {sorted(known_names)}"
                    )
        elif i == 1:
            raise RequestError(
                f"step 1 ({step.op}) has no input to chain from -- the first step must "
                f"name its input explicitly ({step.op}.input, or join.inputs)"
            )

        as_name = getattr(step, "as_", None)
        if as_name is not None:
            _validate_name(as_name, "step 'as'")
            if as_name in known_names:
                raise RequestError(
                    f"step {i} 'as' name {as_name!r} collides with an existing input or "
                    "step name"
                )
            known_names.add(as_name)


def _validate_step_rules(i: int, step: Any) -> None:
    if isinstance(step, ProjectStep):
        pass  # aggregate presence/legality needs grid kinds -- see ops.validate_shapes
    elif isinstance(step, JoinStep):
        if step.type == "attribute":
            raise RequestError(
                f"step {i} (join) type='attribute' is not yet supported -- reserved for a "
                "future release; only 'spatial' is implemented"
            )
        if len(set(step.inputs)) != len(step.inputs):
            raise RequestError(f"step {i} (join) inputs contains duplicate name(s): {step.inputs}")
    elif isinstance(step, FilterStep):
        _validate_where_structurally(i, step.where)
    elif isinstance(step, CalculateStep):
        try:
            parse_calculate_expression(step.expression, allowed_columns=None)
        except ExpressionError as exc:
            raise RequestError(
                f"step {i} (calculate) expression {step.expression!r} is invalid: {exc}"
            ) from exc
    elif isinstance(step, CaseStep):
        for position, case in enumerate(step.cases, start=1):
            _validate_where_structurally(i, case.when, op=f"case arm {position}")
            _validate_value_expression(i, case.then, f"arm {position} then")
        if step.else_ is not None:
            _validate_value_expression(i, step.else_, "else")
    elif isinstance(step, GroupStep):
        try:
            aggregates.validate_group_by(step.by)
            aggregates.validate_group_aggregate(step.aggregate)
        except AggregateError as exc:
            raise RequestError(f"step {i} (group): {exc}") from exc
    elif isinstance(step, UnpivotStep):
        _validate_label_column_name(i, step.into)
    else:  # pragma: no cover -- exhaustive over the Step union
        raise RequestError(f"step {i} has unrecognized op {getattr(step, 'op', None)!r}")


def _validate_output_sort_or_select(value: str, field: str) -> None:
    """Check the SYNTAX of an `output.sort`/`output.select` value -- a comma-separated
    `[-]column` list -- without knowing whether any named column actually exists on the
    final frame. That half is deliberately NOT this function's job: it needs the frame's
    real column namespace, which is only known once the pipeline is compiled and run, so
    an unknown name surfaces instead as Polars' own `ColumnNotFoundError` when the shaped
    frame is collected (see `execution.sink.apply_output_shaping`, which performs the
    split, no name check, and points back here for why). What IS checkable now, from the
    string alone: no empty component between commas, no bare `-` (a sort descending-
    marker with no column after it), no `-` prefix on a `select` entry (descending order
    is a `sort`-only concept), and no column named twice within one list.
    """
    parts = value.split(",")
    seen: set[str] = set()
    for part in parts:
        stripped = part.strip()
        if not stripped:
            raise RequestError(f"output.{field}={value!r} contains an empty component between commas")
        if field == "select" and stripped.startswith("-"):
            raise RequestError(
                f"output.select entry {stripped!r} must not start with '-' -- the "
                "descending-sort prefix is only meaningful for output.sort"
            )
        column = stripped.removeprefix("-")
        if not column:
            raise RequestError(f"output.{field}={value!r} contains a bare '-' with no column named")
        if column in seen:
            raise RequestError(f"output.{field}={value!r} names column {column!r} more than once")
        seen.add(column)


def _validate_value_expression(i: int, expression: str, what: str, op: str = "case") -> None:
    """The same check `calculate.expression` gets -- well-formed arithmetic over the
    whitelist, with the column-namespace half deferred (`allowed_columns=None`) because
    parse time does not know what columns the frame will carry -- named after which part
    of a multi-part step it applies to, since a `case` has several.
    """
    try:
        parse_calculate_expression(expression, allowed_columns=None)
    except ExpressionError as exc:
        raise RequestError(f"step {i} ({op}) {what}={expression!r} is invalid: {exc}") from exc


def _validate_where_structurally(i: int, where: str, op: str = "filter") -> None:
    if not where or not where.strip():
        raise RequestError(f"step {i} ({op}) where must be a non-empty string")
    try:
        classify_where_kinds(where)
    except ExpressionError as exc:
        raise RequestError(f"step {i} ({op}) where={where!r} is invalid: {exc}") from exc


def parse_pipeline(data: Mapping[str, Any]) -> Pipeline:
    """Parse `data` (the JSON-body shape: `{"inputs": ..., "steps": ..., "output": ...}`)
    into a structurally-validated `Pipeline` -- pydantic's shape checks plus
    `validate_pipeline`'s rules, both surfaced as `RequestError`. Does NOT run shape
    typing (`ops.validate_shapes`) -- a caller that intends to compile or execute the
    result still needs that call too; see the module docstring for why the two live in
    different modules.
    """
    try:
        pipeline = Pipeline.model_validate(data)
    except ValidationError as exc:
        raise RequestError(f"invalid pipeline: {exc}") from exc
    validate_pipeline(pipeline)
    return pipeline


def append_steps(pipeline: Pipeline, extra_steps: Sequence[Step], output: Output) -> Pipeline:
    """Combine a stored named-analysis `pipeline` with caller-supplied `extra_steps` and
    a caller-supplied `output` -- the mechanism a named-analysis request uses to narrow a
    stored result (to a set of cells, to a value range, ...) without a bespoke parameter
    per kind of narrowing. `extra_steps` follow the ordinary chaining rule against the
    COMBINED step list: the first one, naming no input, resolves against the stored
    pipeline's own last step's output (or, if the stored pipeline has no steps of its
    own, must name one explicitly, exactly like any pipeline's first step) -- no special
    case is needed beyond building the combined list and revalidating it as an ordinary
    `Pipeline`. `output` always replaces the stored pipeline's own and is applied after
    every appended step -- never between them and the stored ones -- because it is the
    single `Output` on the one combined `Pipeline` this function returns.

    Returns a structurally-validated `Pipeline`; a caller still needs `ops.validate_
    shapes` before compiling it, exactly like any other `Pipeline` `parse_pipeline`
    returns.
    """
    combined = Pipeline(inputs=pipeline.inputs, steps=[*pipeline.steps, *extra_steps], output=output)
    validate_pipeline(combined)
    return combined


# --------------------------------------------------------------------------------------
# The dotted GET/tile query codec
# --------------------------------------------------------------------------------------

def _query_fields(model: type[BaseModel]) -> frozenset[str]:
    """The names a dotted query may spell for `model`'s fields: the ALIAS where one is
    declared, the field name otherwise. The alias is what matters -- `as_` is a Python
    spelling forced by the keyword, and `as` is what a URL actually carries.
    """
    return frozenset(info.alias or name for name, info in model.model_fields.items())


def _query_list_fields(model: type[BaseModel]) -> frozenset[str]:
    """Those of `_query_fields(model)` whose value is a list of SCALARS, and which the
    decoder therefore splits on commas.

    Read off each field's annotation rather than named by hand, which is what keeps
    `where`/`expression` out of it structurally rather than by remembering to: each is a
    single SQL-flavored/arithmetic `str` that may contain commas, spaces, and operators
    verbatim, and splitting one would corrupt it. A field can only be split here if its
    model says it holds a list.

    A list of nested MODELS (`CaseStep.cases`) is excluded for the same reason from the
    other direction: its elements hold exactly those comma-bearing strings, so it is
    spelled positionally (`step.5.cases.2.when=...`) rather than comma-joined -- see
    `_query_model_list_fields`, which is where such a field goes instead.
    """
    return frozenset(
        info.alias or name
        for name, info in model.model_fields.items()
        if _is_list_annotation(info.annotation) and _list_element_model(info.annotation) is None
    )


def _query_model_list_fields(model: type[BaseModel]) -> dict[str, type[BaseModel]]:
    """Those of `_query_fields(model)` that hold a list of nested models, mapped to the
    ELEMENT model -- the decoder validates each element against its own field whitelist,
    and the encoder recurses into it under a 1-based index.

    One level deep, matching what the models declare: the only such field today is
    `CaseStep.cases`, whose element (`CaseWhen`) holds two plain strings. An element model
    that itself declared a model list would encode correctly (`_encode_model` recurses)
    but would not decode, so adding one means extending `_decode_model_list` to recurse too.
    """
    return {
        (info.alias or name): element
        for name, info in model.model_fields.items()
        if (element := _list_element_model(info.annotation)) is not None
    }


def _is_list_annotation(annotation: Any) -> bool:
    """Whether `annotation` is a list, including as one arm of a union (`list[str] | None`
    -- no field is declared that way today, but a future optional list must not silently
    stop being comma-split).
    """
    if get_origin(annotation) is list:
        return True
    if get_origin(annotation) in (Union, UnionType):
        return any(get_origin(arg) is list for arg in get_args(annotation))
    return False


def _list_element_model(annotation: Any) -> type[BaseModel] | None:
    """The `BaseModel` subclass `annotation`'s list elements are, or `None` if it is not a
    list or its elements are scalars. Looks through a union the same way
    `_is_list_annotation` does.
    """
    candidates = (
        get_args(annotation) if get_origin(annotation) in (Union, UnionType) else (annotation,)
    )
    for candidate in candidates:
        if get_origin(candidate) is not list:
            continue
        args = get_args(candidate)
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return args[0]
    return None


# The step models the `op` discriminator chooses between, read back off `Step` itself so a
# sixth op added to that union reaches the query codec without a second edit here.
_STEP_MODELS: tuple[type[BaseModel], ...] = get_args(get_args(Step)[0])

# The only fields an input may carry -- see `InputSelection`.
_INPUT_FIELDS = _query_fields(InputSelection)

# Per-op field whitelist and which of those fields are comma-list-valued, for the
# `step.<i>.<field>` namespace. Both are DERIVED from the step models rather than mirrored
# from them: a hand-kept copy has to be chased every time a field is added to a step, and
# the failure mode when it isn't chased is a legal request rejected as "unknown field".
# Keyed by each model's own `op` literal, which is the same value the discriminator reads.
_STEP_FIELDS: dict[str, frozenset[str]] = {
    model.model_fields["op"].default: _query_fields(model) for model in _STEP_MODELS
}
_STEP_LIST_FIELDS: dict[str, frozenset[str]] = {
    model.model_fields["op"].default: _query_list_fields(model) for model in _STEP_MODELS
}
_STEP_MODEL_LIST_FIELDS: dict[str, dict[str, type[BaseModel]]] = {
    model.model_fields["op"].default: _query_model_list_fields(model) for model in _STEP_MODELS
}

_OUTPUT_FIELDS = _query_fields(Output)


def _insert_dotted(tree: dict[str, Any], segments: list[str], value: str, full_key: str) -> None:
    """Insert `value` into the nested-dict `tree` at `segments` (a dotted parameter
    name's parts, still strings -- list-index segments are only recognized as such
    later, by `_listify`). Raises `RequestError` naming `full_key` the moment a leaf
    value and a nested value are asked to occupy the same path.
    """
    seg = segments[0]
    if len(segments) == 1:
        if isinstance(tree.get(seg), dict):
            raise RequestError(
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
        raise RequestError(
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
        raise RequestError(
            f"parameter group {path or '<root>'!r} mixes list-index segments with named "
            f"segments: {sorted(set(keys) - set(digit_keys))}"
        )
    if not digit_keys:
        return {k: _listify(v, f"{path}.{k}" if path else k) for k, v in node.items()}

    for k in digit_keys:
        if k == "0" or (len(k) > 1 and k.startswith("0")):
            raise RequestError(
                f"parameter group {path or '<root>'!r} has an invalid list index {k!r} "
                "-- indices are 1-based with no leading zero"
            )
    indices = sorted(int(k) for k in digit_keys)
    if indices != list(range(1, len(indices) + 1)):
        raise RequestError(
            f"parameter group {path or '<root>'!r} has non-contiguous list indices "
            f"{indices} -- indices must run 1..N with no gaps"
        )
    return [_listify(node[str(i)], f"{path}.{i}" if path else str(i)) for i in indices]


def _decode_input(name: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RequestError(
            f"input {name!r} must have named sub-keys, e.g. input.{name}.variable=..."
        )
    unknown = set(raw) - _INPUT_FIELDS
    if unknown:
        raise RequestError(f"input {name!r} has unknown field(s): {sorted(unknown)}")
    data: dict[str, Any] = {}
    for field, value in raw.items():
        if not isinstance(value, str):
            raise RequestError(f"input {name!r} field {field!r} must be a scalar query value")
        data[field] = value
    return data


def _decode_step(index: int, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RequestError(f"step {index} must have named sub-keys, e.g. step.{index}.op=...")
    op = raw.get("op")
    if op is None:
        raise RequestError(f"step {index} is missing required field 'op'")
    if op not in _STEP_FIELDS:
        raise RequestError(f"step {index} has unknown op {op!r}")

    allowed = _STEP_FIELDS[op]
    list_fields = _STEP_LIST_FIELDS[op]
    model_list_fields = _STEP_MODEL_LIST_FIELDS[op]
    unknown = set(raw) - allowed
    if unknown:
        raise RequestError(f"step {index} ({op}) has unknown field(s): {sorted(unknown)}")

    data: dict[str, Any] = {"op": op}
    for field, value in raw.items():
        if field == "op":
            continue
        if field in model_list_fields:
            data[field] = _decode_model_list(index, op, field, value, model_list_fields[field])
            continue
        if not isinstance(value, str):
            raise RequestError(f"step {index} ({op}) field {field!r} must be a scalar query value")
        if field in list_fields:
            data[field] = [] if value == "" else value.split(",")
        else:
            data[field] = value
    return data


def _decode_model_list(
    index: int, op: str, field: str, raw: Any, element_model: type[BaseModel]
) -> list[dict[str, Any]]:
    """Decode a `step.<i>.<field>.<n>.<subfield>=...` group into the list of dicts the
    nested model validates from. `_listify` has already turned the 1-based index segments
    into a real list (and rejected a gap or a leading zero in them); what is left is per-
    element: it must be a mapping of named sub-keys, those keys must be the element
    model's own, and each value must still be a scalar query value.

    This is where a nested model's strings escape comma-splitting. A `case` arm's `when`
    is an ordinary SQL-flavored predicate and may contain commas; spelling the arms
    positionally rather than as two parallel comma-joined lists is what lets it.
    """
    allowed = _query_fields(element_model)
    if not isinstance(raw, list):
        raise RequestError(
            f"step {index} ({op}) field {field!r} must use 1-based list indices with named "
            f"sub-keys, e.g. step.{index}.{field}.1.<{'|'.join(sorted(allowed))}>=..."
        )
    entries: list[dict[str, Any]] = []
    for position, entry in enumerate(raw, start=1):
        where = f"step {index} ({op}) {field}.{position}"
        if not isinstance(entry, dict):
            raise RequestError(
                f"{where} must have named sub-keys, e.g. "
                f"step.{index}.{field}.{position}.<{'|'.join(sorted(allowed))}>=..."
            )
        unknown = set(entry) - allowed
        if unknown:
            raise RequestError(f"{where} has unknown field(s): {sorted(unknown)}")
        for key, value in entry.items():
            if not isinstance(value, str):
                raise RequestError(f"{where} field {key!r} must be a scalar query value")
        entries.append(dict(entry))
    return entries


def _decode_output(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RequestError("'output' must have named sub-keys, e.g. output.format=...")
    unknown = set(raw) - _OUTPUT_FIELDS
    if unknown:
        raise RequestError(f"output has unknown field(s): {sorted(unknown)}")
    return dict(raw)


def pipeline_from_query(params: Mapping[str, str]) -> Pipeline:
    """Decode the dotted GET/tile query encoding into a structurally-validated
    `Pipeline` (see `parse_pipeline`'s docstring on the shape-typing step this still
    omits).

    Encoding rule (see `pipeline_to_query` for the inverse): dot = nesting, an integer
    path segment = a 1-based list position, comma = a list value (`join.inputs`,
    `group.by`) -- NEVER inside `filter.where`/`calculate.expression`, each one opaque
    string. A list whose elements are themselves models (`case.cases`) uses the index
    segments rather than commas, for exactly that reason: its elements hold predicates,
    which may contain commas (`step.5.cases.2.when=value > 0`). `params` is the
    already-decoded (no percent-encoding) mapping an HTTP framework's query-parameter
    object would hand over.

    Malformed input -- a gap in list indices, a leaf/nested conflict at the same path, an
    unrecognized top-level group, a missing top-level group, an unknown step op or field
    -- raises `RequestError` naming the offending parameter or parameter group. Once the
    dotted structure resolves cleanly, the resulting JSON-shaped dict is handed to
    `parse_pipeline`, so a query that's well-formed AS A QUERY but violates a structural
    rule (e.g. a join with only one input) is rejected the same way parsing the
    equivalent JSON body would be.
    """
    if not params:
        raise RequestError("empty query: no pipeline parameters given")

    tree: dict[str, Any] = {}
    for key, value in params.items():
        if not isinstance(key, str) or not key:
            raise RequestError(f"invalid parameter name {key!r}")
        if not isinstance(value, str):
            raise RequestError(f"parameter {key!r} must have a string value")
        _insert_dotted(tree, key.split("."), value, key)

    # `step` is the one top-level group a request may legitimately omit entirely -- a
    # zero-step (plain data query) pipeline has no `step.*` parameters to send at all.
    top_level = {"input", "step", "output"}
    required_top = {"input", "output"}
    unknown_top = set(tree) - top_level
    if unknown_top:
        raise RequestError(f"unknown top-level parameter group(s): {sorted(unknown_top)}")
    missing_top = required_top - set(tree)
    if missing_top:
        raise RequestError(f"missing required parameter group(s): {sorted(missing_top)}")

    listified = _listify(tree, "")

    inputs_raw = listified["input"]
    if not isinstance(inputs_raw, dict):
        raise RequestError("'input' must use named sub-keys, e.g. input.<name>.variable=...")
    inputs_json = {name: _decode_input(name, fields) for name, fields in inputs_raw.items()}

    steps_raw = listified.get("step", [])
    if steps_raw == {}:
        steps_raw = []
    if not isinstance(steps_raw, list):
        raise RequestError("'step' must use 1-based list indices, e.g. step.1.op=...")
    steps_json = [_decode_step(i, s) for i, s in enumerate(steps_raw, start=1)]

    output_json = _decode_output(listified["output"])

    return parse_pipeline({"inputs": inputs_json, "steps": steps_json, "output": output_json})


def _query_sort_key(param_name: str) -> tuple:
    """Natural sort: a digit segment compares numerically (`step.2` before `step.10`),
    so the canonical order stays intuitive past 9 steps/inputs.
    """
    return tuple((0, int(seg)) if seg.isdigit() else (1, seg) for seg in param_name.split("."))


def _encode_model(model: BaseModel, prefix: str, params: dict[str, str]) -> None:
    """Write `model`'s fields into `params` under `prefix`, spelled the way the dotted
    codec spells them.

    Read off `model_fields` -- the same source `_query_fields`/`_query_list_fields` read
    the DECODER's per-model whitelist off. That symmetry is the point. While the encoder
    was a hand-written branch per model and the decoder was derived, a field added to a
    step decoded the moment it was declared and encoded never: a request could be accepted
    and then re-serialize to something that no longer said what it said. Now both halves
    move when the model does.

    Four rules, each the generalization of what the per-model branches did by hand:

    - The NAME is the field's alias where it declares one, its own name otherwise. `as_` is
      a Python spelling forced by the keyword; `as` is what a URL carries.
    - A field AT ITS DEFAULT is dropped, which covers `None` (every optional field) and a
      real default alike (`GroupStep.of == "value"`). A required field has no default to
      equal, so it is always written. See `pipeline_to_query` for why dropping -- rather
      than writing the default out explicitly -- is what makes the round trip stable.
    - A list of nested MODELS (`CaseStep.cases`) recurses, one element per 1-based index
      segment, which is the spelling `_query_model_list_fields` tells the decoder to read
      back. Recursion rather than a second flattening rule: a nested model's own fields
      then get all four of these rules too, for free.
    - Any other LIST is comma-joined, which is the spelling `_query_list_fields` tells the
      decoder to split back apart.

    The discriminator falls out of the second rule rather than needing an exception to it:
    a step's `op` always equals its own default, so this drops it, and `pipeline_to_query`
    writes it itself.
    """
    for name, info in type(model).model_fields.items():
        value = getattr(model, name)
        if value == info.default:
            continue
        key = info.alias or name
        if _list_element_model(info.annotation) is not None:
            for position, element in enumerate(value, start=1):
                _encode_model(element, f"{prefix}.{key}.{position}", params)
            continue
        encoded = ",".join(str(item) for item in value) if isinstance(value, list) else str(value)
        params[f"{prefix}.{key}"] = encoded


def pipeline_to_query(pipeline: Pipeline) -> dict[str, str]:
    """Encode `pipeline` into the dotted GET/tile query form -- the inverse of
    `pipeline_from_query`. Returns a plain `dict[str, str]`, its keys in the canonical
    (natural-sorted) order described in the module docstring, so two `Pipeline`s that are
    equivalent after normalization serialize to the identical parameter set, regardless
    of the order fields happened to arrive in on either side.

    A field at its default (`GroupStep.of == "value"`) or unset (`None`) is omitted
    rather than written out explicitly -- `pipeline_from_query` reconstructs the same
    default/`None` on decode, which is what keeps this round-trip stable rather than
    merely reversible once. A `Pipeline` with no steps encodes no `step.*` parameters at
    all -- `pipeline_from_query` treats an entirely absent `step` group as the empty list.
    """
    params: dict[str, str] = {}

    for name, selection in pipeline.inputs.items():
        _encode_model(selection, f"input.{name}", params)

    for i, step in enumerate(pipeline.steps, start=1):
        prefix = f"step.{i}"
        # `op` is written here rather than by `_encode_model`, which drops every field
        # sitting at its default and would therefore drop exactly this one -- a step's `op`
        # is always its own model's default. It is also the one field the decoder cannot
        # do without: it chooses which model validates the rest.
        params[f"{prefix}.op"] = step.op
        _encode_model(step, prefix, params)

    _encode_model(pipeline.output, "output", params)

    return dict(sorted(params.items(), key=lambda item: _query_sort_key(item[0])))
