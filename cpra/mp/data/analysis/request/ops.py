"""The op IR: a small typed `Op` per pipeline step -- `ProjectOp`/`JoinOp`/`FilterOp`/
`CalculateOp`/`GroupOp`, exactly the five ops `pipeline.py`'s `Step` union already names
-- plus `compile_ops`, which turns an already-validated `Pipeline` into `list[Op]` **in
declared order**, and `validate_shapes`, which types every input/step's shape (`long` or
`band_wide`) and rejects the combinations spec'd as illegal.

**Shape-agnostic at the interface.** An `Op` carries only what the pipeline DECLARED: a
`project`'s target grid, a `filter`'s predicate, a `calculate`'s expression, a `group`'s
aggregate -- never whether the frame it will eventually run against is `long` (vector: one
row per (dimensions, `grid_cell`, `value`) observation) or `band_wide` (raster: one row per
pixel, one column per band). The `*_LOWERING_TABLE`s below are where shape is consulted,
held as DATA rather than as branching code, so which combination of op/predicate-kind/
grid-kind lowers to what is written down and checkable in one place.

**Shape typing needs only the declared ops and each input's grid kind** (`cpra.mp.data.
grids.GRIDS[name].grid_type`) -- no crosswalk, no catalog, no file. That is what lets
`validate_shapes` answer every shape-dependent rejection (raster-to-
raster `project`, a `join` mixing long and band-wide, a band-wide `join` across two
different grids, a long final frame under a raster output format, more than one value
column at the sink, a second `count_by_value` in one chain) with no IO at all.

**Why this lives here, and not in `pipeline.py`.** `compile_ops`/`validate_shapes` both
need `Pipeline`/`Step` from `pipeline.py`; `pipeline.py` cannot import this module back
without a cycle. `validate_pipeline` (`pipeline.py`) and `validate_shapes` (here) are
therefore two functions, in two modules, implementing what spec calls one VALIDATE stage:
a `Pipeline` needs both calls -- `pipeline.parse_pipeline(data)` then
`ops.validate_shapes(pipeline)` -- before `compile_ops` (which assumes both already ran)
is safe to call.
"""

import dataclasses
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from cpra.mp.data.analysis.request import aggregates
from cpra.mp.data.analysis.request.errors import AggregateError, RequestError, ShapeError
from cpra.mp.data.analysis.request.expressions import PredicateKind, classify_where_kinds
from cpra.mp.data.analysis.request.pipeline import (
    CalculateStep,
    FilterStep,
    GroupStep,
    InputSelection,
    JoinStep,
    Pipeline,
    ProjectStep,
    RASTER_OUTPUT_FORMATS,
    _step_explicit_inputs,
)
from cpra.mp.data.analysis.request.types import AggregateFunction, Grid
from cpra.mp.data.grids import GRIDS


class ProjectOp(BaseModel):
    """Move `input`'s rows onto `to`'s grid, aggregating with `aggregate` when the
    direction requires one -- see `ProjectStep`'s docstring and
    `request.aggregates.validate_project_aggregate` for exactly which directions do.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    op: Literal["project"] = "project"
    step_no: int
    input: str
    to: Grid
    aggregate: AggregateFunction | None = None
    output: str


class JoinOp(BaseModel):
    """Combine `inputs` (always exactly the names `JoinStep.inputs` gave, in that order --
    `join` is the one op with no implicit-chain default) on shared dimensions plus
    `grid_cell`, once both operands' shapes are confirmed compatible by `validate_shapes`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["join"] = "join"
    step_no: int
    type: Literal["spatial", "attribute"]
    inputs: list[str]
    output: str


class FilterOp(BaseModel):
    """A predicate (`where`) over `input`'s in-scope columns. The lowering depends on
    which of the three `PredicateKind`s `where` is (`expressions.classify_where_kinds`),
    not on a field this op carries -- see `FILTER_LOWERING_TABLE`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["filter"] = "filter"
    step_no: int
    input: str
    where: str
    output: str


class CalculateOp(BaseModel):
    """Conditionally (`where`, optional) rewrite `input`'s running value by
    `expression`. `label` is what an error raised while compiling `expression` calls
    this entry -- diagnostic only, defaults to `None`, meaning "name me after my step".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["calculate"] = "calculate"
    step_no: int
    input: str
    expression: str
    where: str | None = None
    output: str
    label: str | None = None


class GroupOp(BaseModel):
    """Collapse `by` (empty means "one row per `grid_cell`") with `aggregate` applied to
    `of`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    op: Literal["group"] = "group"
    step_no: int
    input: str
    by: list[str]
    aggregate: AggregateFunction
    of: str = "value"
    output: str


Op = Annotated[
    Union[ProjectOp, JoinOp, FilterOp, CalculateOp, GroupOp],
    Field(discriminator="op"),
]


def calculate_label(op: CalculateOp) -> str:
    """What an error raised while compiling `op`'s expression should call it: the
    explicit `CalculateOp.label`, else `step.<step_no>`, the pipeline step's own
    position.
    """
    return op.label if op.label is not None else f"step.{op.step_no}"


class Shape(str, Enum):
    """The two data shapes an op lowers against. `long` is the vector row model --
    dimension columns + `grid_cell` + `value`, one row per observation. `band_wide` is
    the raster row model -- one row per pixel, one column per band.
    """

    long = "long"
    band_wide = "band_wide"


# The filter lowering table: keyed by (predicate kind, input shape), it gives the output
# shape. Every combination is legal -- a filter never rejects a shape, only a `join` and
# a `project` can (see the tables below) -- except that a SPATIAL predicate is the one
# case that changes shape: sampling a band-wide frame at a set of `grid_cell`s turns it
# long, since the result is no longer positionally aligned to the raster's own window.
FILTER_LOWERING_TABLE: dict[tuple[PredicateKind, Shape], Shape] = {
    (PredicateKind.value, Shape.long): Shape.long,
    (PredicateKind.value, Shape.band_wide): Shape.band_wide,
    (PredicateKind.dimension, Shape.long): Shape.long,
    (PredicateKind.dimension, Shape.band_wide): Shape.band_wide,
    (PredicateKind.spatial, Shape.long): Shape.long,
    (PredicateKind.spatial, Shape.band_wide): Shape.long,
}

# The project lowering table: keyed by (source grid kind, target grid kind), it gives the
# output shape. Raster-to-raster is deliberately ABSENT -- there is no crosswalk between
# two raster lattices, so a project in that direction has no lowering at all; see
# `_type_project`, which turns the resulting `KeyError` into a `ShapeError`.
PROJECT_LOWERING_TABLE: dict[tuple[str, str], Shape] = {
    ("vector", "vector"): Shape.long,
    ("raster", "vector"): Shape.long,
    ("vector", "raster"): Shape.band_wide,
}

# `join` is not expressible as a single (op, shape) lowering key the way the other ops
# are -- it takes two (or more) operands, and its legality is a property of how their
# shapes relate to each other, not of one shape alone. Rather than force that into an
# artificial multi-shape key, `_type_join` below does the equivalent as an explicit
# pre-dispatch check: every operand must share one shape (this table says what that
# shared shape lowers to -- trivially itself), and a band-wide join additionally requires
# every operand to share one grid (checked separately, since it needs each operand's
# GRID, not just its shape).
JOIN_LOWERING_TABLE: dict[Shape, Shape] = {
    Shape.long: Shape.long,
    Shape.band_wide: Shape.band_wide,
}


def _default_output_name(step_no: int) -> str:
    """The output binding synthesized for a step that gave no `as_`. A bare digit string
    can never collide with a real `as_`/input name: both are required
    (`pipeline._validate_name`'s `_NAME_RE`) to start with a lowercase letter or
    underscore, so `str(step_no)` -- which starts with a digit -- is disjoint from every
    legal name by construction.
    """
    return str(step_no)


def compile_ops(pipeline: Pipeline) -> list[Op]:
    """Turn an already-validated `Pipeline` -- one that passed
    `pipeline.parse_pipeline`/`validate_pipeline` AND `validate_shapes`, whose chaining/
    `as`/shape rules this function assumes rather than re-checks -- into `list[Op]`, one
    entry per step, **in declared order**. Never opens a dataset and never decides a
    `project`'s cardinality direction; an `Op` records only what the pipeline DECLARED.

    Every `Op`'s `input` (or `inputs`, for `join`) is fully resolved here, never left as
    an implicit "chain from the previous step" default.
    """
    ops: list[Op] = []
    previous_output: str | None = None
    for step_no, step in enumerate(pipeline.steps, start=1):
        as_name = getattr(step, "as_", None)
        output = as_name if as_name is not None else _default_output_name(step_no)

        if isinstance(step, JoinStep):
            op: Op = JoinOp(step_no=step_no, type=step.type, inputs=list(step.inputs), output=output)
            ops.append(op)
            previous_output = output
            continue

        explicit = _step_explicit_inputs(step)
        input_name = explicit[0] if explicit else previous_output
        if input_name is None:
            # Unreachable for a `Pipeline` built through `parse_pipeline`: its chaining
            # rule already rejects a step 1 that names no explicit input. Guarded here
            # rather than silently emitting an `Op` with input=None, so a `Pipeline`
            # constructed directly (bypassing `parse_pipeline`) fails loudly instead of
            # producing an IR no lowering could ever run.
            raise RequestError(
                f"step {step_no} ({step.op}) has no input to chain from -- this Pipeline "
                "did not pass validate_pipeline's chaining rule"
            )

        if isinstance(step, ProjectStep):
            op = ProjectOp(
                step_no=step_no, input=input_name, to=step.to, aggregate=step.aggregate, output=output
            )
        elif isinstance(step, FilterStep):
            op = FilterOp(step_no=step_no, input=input_name, where=step.where, output=output)
        elif isinstance(step, CalculateStep):
            op = CalculateOp(
                step_no=step_no, input=input_name, expression=step.expression, where=step.where, output=output
            )
        elif isinstance(step, GroupStep):
            op = GroupOp(
                step_no=step_no,
                input=input_name,
                by=list(step.by),
                aggregate=step.aggregate,
                of=step.of,
                output=output,
            )
        else:  # pragma: no cover -- exhaustive over the Step union
            raise RequestError(f"step {step_no} has unrecognized op {getattr(step, 'op', None)!r}")

        ops.append(op)
        previous_output = output

    return ops


@dataclasses.dataclass(frozen=True)
class _FrameState:
    """What `validate_shapes`'s walk tracks for one named input/step output.

    `shape`/`grid` are what a later `project`/`join` needs to know this frame's own
    legality: `grid` is the grid this frame is CURRENTLY addressed against (an input's
    own grid, or a `project`'s `to`) -- tracked separately from `shape`, since a spatial
    filter can turn a band-wide frame long while it is still fundamentally addressed
    against that same raster grid (its `grid_cell` values are still that raster's pixel
    indices), and a later `project` FROM that frame is still raster-to-* for exactly that
    reason, regardless of its current (now-long) shape.

    `has_count_by_value`/`value_columns` are the two other facts the sink rules need:
    whether a `count_by_value` aggregate has already run somewhere in this frame's
    history (at most one per chain), and how many distinct value columns this frame
    currently carries (exactly one is required at the sink).
    """

    shape: Shape
    grid: str
    has_count_by_value: bool
    value_columns: frozenset[str]


def _initial_state(selection: InputSelection) -> _FrameState:
    grid_name = selection.grid
    kind = GRIDS[grid_name].grid_type
    shape = Shape.band_wide if kind == "raster" else Shape.long
    return _FrameState(shape=shape, grid=grid_name, has_count_by_value=False, value_columns=frozenset({"value"}))


def _type_project(step_no: int, step: ProjectStep, current: _FrameState) -> _FrameState:
    from_kind = GRIDS[current.grid].grid_type
    to_kind = GRIDS[step.to].grid_type
    try:
        out_shape = PROJECT_LOWERING_TABLE[(from_kind, to_kind)]
    except KeyError as exc:
        raise ShapeError(
            f"step {step_no} (project): no lowering from a {from_kind} grid to a raster "
            "grid -- raster-to-raster project is never legal, since there is no "
            "crosswalk between two raster lattices"
        ) from exc

    try:
        aggregates.validate_project_aggregate(
            step.aggregate, direction_takes_aggregate=not (from_kind == "vector" and to_kind == "raster")
        )
    except AggregateError as exc:
        raise ShapeError(f"step {step_no} (project): {exc}") from exc

    has_count_by_value = current.has_count_by_value
    if step.aggregate == "count_by_value":
        if has_count_by_value:
            raise ShapeError(
                f"step {step_no} (project): a second count_by_value in this chain has "
                "nowhere to put its categories -- value_category is already occupied by "
                "an earlier one"
            )
        has_count_by_value = True

    return _FrameState(shape=out_shape, grid=step.to, has_count_by_value=has_count_by_value, value_columns=frozenset({"value"}))


def _type_filter(step: FilterStep, current: _FrameState) -> _FrameState:
    kinds = classify_where_kinds(step.where)
    if kinds[PredicateKind.spatial]:
        kind = PredicateKind.spatial
    elif kinds[PredicateKind.value]:
        kind = PredicateKind.value
    else:
        kind = PredicateKind.dimension
    new_shape = FILTER_LOWERING_TABLE[(kind, current.shape)]
    return dataclasses.replace(current, shape=new_shape)


def _type_calculate(current: _FrameState) -> _FrameState:
    return dataclasses.replace(current, value_columns=frozenset({"value"}))


def _type_group(current: _FrameState) -> _FrameState:
    return dataclasses.replace(current, value_columns=frozenset({"value"}))


def _type_join(step_no: int, step: JoinStep, operand_states: list[_FrameState]) -> _FrameState:
    shapes = {s.shape for s in operand_states}
    if len(shapes) != 1:
        raise ShapeError(
            f"step {step_no} (join): cannot join a long frame with a band-wide one -- "
            "insert a project to bring every operand onto one shape first"
        )
    shape = shapes.pop()
    out_shape = JOIN_LOWERING_TABLE[shape]

    if shape == Shape.band_wide:
        grids = {s.grid for s in operand_states}
        if len(grids) != 1:
            raise ShapeError(
                f"step {step_no} (join): band-wide operands must share one grid, got "
                f"{sorted(grids)} -- insert a project to bring every operand onto one "
                "grid first"
            )
        grid = next(iter(grids))
    else:
        grid = operand_states[0].grid

    has_count_by_value = any(s.has_count_by_value for s in operand_states)
    value_columns = frozenset(step.inputs)
    return _FrameState(shape=out_shape, grid=grid, has_count_by_value=has_count_by_value, value_columns=value_columns)


def validate_shapes(pipeline: Pipeline) -> None:
    """Walk `pipeline` (assumed already structurally valid -- see `pipeline.
    validate_pipeline`) in declared order, typing each input/step output's shape and
    raising `ShapeError` for every shape-dependent rejection decidable from the declared
    ops and each input's grid kind alone:

    - raster-to-raster `project` (`_type_project`, via `PROJECT_LOWERING_TABLE`'s
      absence);
    - `project.aggregate` missing where required, or supplied where forbidden
      (`_type_project`, via `request.aggregates.validate_project_aggregate`);
    - a second `count_by_value` in one chain (`_type_project`);
    - a `join` mixing a long frame with a band-wide one (`_type_join`);
    - a band-wide `join` whose operands sit on two different grids (`_type_join`);
    - a long final frame under an output format that needs a raster surface (below);
    - more than one value column reaching the sink (below).

    Nothing here opens a file or knows a crosswalk.
    """
    states: dict[str, _FrameState] = {
        name: _initial_state(selection) for name, selection in pipeline.inputs.items()
    }
    previous_output: str | None = None

    if not pipeline.steps:
        final = next(iter(states.values()))
    else:
        for step_no, step in enumerate(pipeline.steps, start=1):
            if isinstance(step, JoinStep):
                operand_states = [states[name] for name in step.inputs]
                new_state = _type_join(step_no, step, operand_states)
            else:
                explicit = _step_explicit_inputs(step)
                input_name = explicit[0] if explicit else previous_output
                current = states[input_name]
                if isinstance(step, ProjectStep):
                    new_state = _type_project(step_no, step, current)
                elif isinstance(step, FilterStep):
                    new_state = _type_filter(step, current)
                elif isinstance(step, CalculateStep):
                    new_state = _type_calculate(current)
                elif isinstance(step, GroupStep):
                    new_state = _type_group(current)
                else:  # pragma: no cover -- exhaustive over the Step union
                    raise RequestError(f"step {step_no} has unrecognized op {getattr(step, 'op', None)!r}")

            as_name = getattr(step, "as_", None)
            output_name = as_name if as_name is not None else _default_output_name(step_no)
            states[output_name] = new_state
            previous_output = output_name

        final = states[previous_output]

    if len(final.value_columns) != 1:
        raise ShapeError(
            f"pipeline's final frame carries {len(final.value_columns)} value columns "
            f"{sorted(final.value_columns)} -- every output writes exactly one measured "
            "quantity; add a calculate to collapse them into one"
        )

    if final.shape == Shape.long and pipeline.output.format in RASTER_OUTPUT_FORMATS:
        raise ShapeError(
            f"pipeline's final frame is long (tabular), but output.format="
            f"{pipeline.output.format!r} needs a raster surface -- add a project to a "
            "raster grid before the sink"
        )
