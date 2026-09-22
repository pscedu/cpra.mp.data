"""The op IR: a small typed `Op` per pipeline step -- `ProjectOp`/`JoinOp`/`FilterOp`/
`CalculateOp`/`GroupOp`, exactly the five ops `pipeline.py`'s `Step` union already names
-- plus `compile_ops`, which turns an already-validated `Pipeline` into `list[Op]` **in
declared order**, and `validate_shapes`, which types every input/step's shape (`long` or
`band_wide`) and rejects the combinations that have no meaning (enumerated below).

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
therefore two halves of one validation step, split across two modules only to break that
cycle: a `Pipeline` needs both calls -- `pipeline.parse_pipeline(data)` then
`ops.validate_shapes(pipeline)` -- before anything executes it.

`validate_shapes` itself runs `compile_ops` first, so that the shape walk sees resolved
operand names rather than re-implementing the "chain from the previous step" default a
second time. That is not circular: `compile_ops` requires only STRUCTURAL validity (the
chaining and `as` rules `parse_pipeline` already enforced), never shape validity.
"""

import dataclasses
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from cpra.mp.data.analysis.request import aggregates
from cpra.mp.data.analysis.request.errors import AggregateError, RequestError, ShapeError
from cpra.mp.data.analysis.request.expressions import PredicateKind, classify_where_kinds
from cpra.mp.data.analysis.request.pipeline import (
    RASTER_OUTPUT_FORMATS,
    CalculateStep,
    FilterStep,
    GroupStep,
    InputSelection,
    JoinStep,
    Pipeline,
    ProjectStep,
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
    ProjectOp | JoinOp | FilterOp | CalculateOp | GroupOp,
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
    """Turn a structurally-valid `Pipeline` -- one that passed `pipeline.parse_pipeline`/
    `validate_pipeline`, whose chaining and `as` rules this function assumes rather than
    re-checks -- into `list[Op]`, one entry per step, **in declared order**. Never opens a
    dataset and never decides a `project`'s cardinality direction; an `Op` records only
    what the pipeline DECLARED.

    Shape validity is deliberately NOT a precondition: `validate_shapes` calls this first,
    then types the resulting ops, so requiring it here would be circular. Compiling a
    shape-invalid pipeline is harmless -- an `Op` list is just the declaration, and
    nothing executes one that has not been through `validate_shapes`.

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
class FrameType:
    """Everything the shape-typing walk knows about one named input/step output, and the
    whole of what a downstream consumer needs in order to pick a lowering for an op
    reading that frame: `annotate_frame_types` below returns one of these per name, and
    the execution walk dispatches on them rather than re-deriving any of it.

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


def project_direction(current: FrameType, to_grid: str) -> tuple[str, str]:
    """The `(from kind, to kind)` pair keying `PROJECT_LOWERING_TABLE` for a project
    carrying `current` onto `to_grid` -- "raster"/"vector", read off `GRIDS`.

    Public, and the one definition of what a project's DIRECTION is, because two callers
    need the same answer from opposite ends: `_type_project` below, to decide whether the
    direction has a lowering at all, and the execution walk, to decide WHICH of the three
    lowerings to dispatch to. A project is the one op whose lowering is not selected by
    the incoming shape alone -- a long frame projects to either vector (long out) or
    raster (band-wide out) -- so a shape-keyed dispatch cannot pick it and the grid kinds
    have to be consulted directly.

    Note `current.grid`, not the input's declared grid: a spatial filter can turn a
    band-wide frame long while leaving it addressed against the same raster lattice, and
    a project from there is still raster-to-something (see `FrameType`).
    """
    return GRIDS[current.grid].grid_type, GRIDS[to_grid].grid_type


def _initial_state(selection: InputSelection) -> FrameType:
    grid_name = selection.grid
    kind = GRIDS[grid_name].grid_type
    shape = Shape.band_wide if kind == "raster" else Shape.long
    return FrameType(shape=shape, grid=grid_name, has_count_by_value=False, value_columns=frozenset({"value"}))


def _type_project(step_no: int, step: ProjectStep | ProjectOp, current: FrameType) -> FrameType:
    from_kind, to_kind = project_direction(current, step.to)
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

    return FrameType(
        shape=out_shape, grid=step.to, has_count_by_value=has_count_by_value, value_columns=frozenset({"value"})
    )


def _type_filter(step: FilterStep | FilterOp, current: FrameType) -> FrameType:
    """`current.value_columns`, not the bare `value` default, is what the conjuncts are
    classified against -- on a PAIRED frame (one a `join` left carrying one operand-named
    value column per operand) `marsh > 0.5` is a value conjunct, and classifying it as a
    dimension one would send it to a lowering that evaluates it against band metadata,
    where no such column exists.

    A value conjunct over a paired BAND-WIDE frame also collapses the pairing: the
    lowering (`execution.band_ops.pair_and_mask_band_wide`) pairs each operand's band with
    its counterpart on the dimensions they share and emits ONE band per pair, so the
    result carries one value column again. Long shape does not collapse -- a long value
    filter is an ordinary row filter that leaves every operand column standing -- so the
    collapse is conditioned on the shape, not on the predicate kind alone.
    """
    kinds = classify_where_kinds(step.where, current.value_columns)
    if kinds[PredicateKind.spatial]:
        kind = PredicateKind.spatial
    elif kinds[PredicateKind.value]:
        kind = PredicateKind.value
    else:
        kind = PredicateKind.dimension
    new_shape = FILTER_LOWERING_TABLE[(kind, current.shape)]
    value_columns = current.value_columns
    if kind is PredicateKind.value and current.shape is Shape.band_wide and len(value_columns) > 1:
        value_columns = frozenset({"value"})
    return dataclasses.replace(current, shape=new_shape, value_columns=value_columns)


def _type_calculate(step_no: int, current: FrameType) -> FrameType:
    if current.shape is Shape.band_wide and len(current.value_columns) > 1:
        raise ShapeError(
            f"step {step_no} (calculate): a calculate over a paired band-wide frame "
            f"(value columns {sorted(current.value_columns)}) is not yet supported -- a "
            "band-wide frame's bands are positional, so there is no per-operand column "
            "for an expression to name; collapse the pairing with a value filter first"
        )
    return dataclasses.replace(current, value_columns=frozenset({"value"}))


def _type_group(current: FrameType) -> FrameType:
    return dataclasses.replace(current, value_columns=frozenset({"value"}))


def _type_join(step_no: int, step: JoinStep | JoinOp, operand_states: list[FrameType]) -> FrameType:
    """A join's operands must agree on both shape and grid, whatever that shape is.

    The grid requirement holds for a long join for the same reason it holds for a
    band-wide one: `grid_cell` identifies a cell WITHIN one grid, and nothing about the
    number 7 says which lattice it indexes. Two frames on different grids have no cell in
    common, so a join between them is asking for a correspondence that does not exist --
    and the right way to establish one is a `project` bringing both onto a single grid,
    which is exactly the op that owns cell-to-cell correspondence.

    Note `FrameType.grid`, not the input's declared grid: a spatial filter turns a
    band-wide frame long while leaving it addressed against the same raster lattice, so a
    long frame is not necessarily on a vector grid, and this check is about the lattice
    either way.

    **Each operand must carry exactly one value column**, which is what bounds the operand
    count in practice: a join names each operand's measurement after the operand itself
    (`value_columns = frozenset(step.inputs)` below), and an operand that already carries
    several -- because it is itself a join's output -- has no single measurement for that
    name to refer to. So `join(a, b)` is fine and `join(join(a, b), c)` is not, until
    something in between collapses the pairing back to one value column (a value `filter`
    over a band-wide frame does exactly that; see `_type_filter`). Rejecting here rather
    than at execution is the point: left alone, the chained form reaches the sink and
    fails on a `value` column that no longer exists, which tells the caller nothing about
    what they actually asked for.
    """
    shapes = {s.shape for s in operand_states}
    if len(shapes) != 1:
        raise ShapeError(
            f"step {step_no} (join): cannot join a long frame with a band-wide one -- "
            "insert a project to bring every operand onto one shape first"
        )
    shape = shapes.pop()
    out_shape = JOIN_LOWERING_TABLE[shape]

    grids = {s.grid for s in operand_states}
    if len(grids) != 1:
        raise ShapeError(
            f"step {step_no} (join): operands must share one grid, got {sorted(grids)} -- "
            "a `grid_cell` identifies a cell within one grid, so operands on different "
            "grids have no cell in common; insert a project to bring every operand onto "
            "one grid first"
        )
    grid = next(iter(grids))

    already_paired = [
        name for name, state in zip(step.inputs, operand_states) if len(state.value_columns) > 1
    ]
    if already_paired:
        raise ShapeError(
            f"step {step_no} (join): operand(s) {already_paired} already carry more than "
            "one value column, and joining one of those is not yet supported -- a join "
            "names each operand's measurement after the operand, so an operand with "
            "several measurements has no single one for that name to mean; collapse it "
            "to one value column first (a value filter over a band-wide frame does)"
        )

    has_count_by_value = any(s.has_count_by_value for s in operand_states)
    value_columns = frozenset(step.inputs)
    return FrameType(shape=out_shape, grid=grid, has_count_by_value=has_count_by_value, value_columns=value_columns)


def annotate_frame_types(pipeline: Pipeline, ops: list[Op]) -> dict[str, FrameType]:
    """Type every named frame in `pipeline` -- each declared input, and each op's output --
    returning name -> `FrameType`. `ops` is `compile_ops(pipeline)`'s output, so every
    operand name is already resolved and this walk never has to re-derive the "chain from
    the previous step" default.

    This is the ONE place the shape-typing rules run. `validate_shapes` calls it to reach
    the final frame it checks the sink rules against, and the execution layer's compile
    step calls it to record what the walk dispatches on -- neither carries its own copy of
    the rules (e.g. `_type_filter`'s spatial > value > dimension precedence), so the two
    cannot drift. It is exposed for that second caller: a consumer outside this module
    needs these answers and must not have to reach past the underscore to get them.

    Raises `ShapeError` for every shape-dependent rejection decidable from the declared
    ops and each input's grid kind alone; see `validate_shapes` for the list. Opens
    nothing.
    """
    frame_types: dict[str, FrameType] = {
        name: _initial_state(selection) for name, selection in pipeline.inputs.items()
    }
    for op in ops:
        if isinstance(op, JoinOp):
            new_type = _type_join(op.step_no, op, [frame_types[name] for name in op.inputs])
        elif isinstance(op, ProjectOp):
            new_type = _type_project(op.step_no, op, frame_types[op.input])
        elif isinstance(op, FilterOp):
            new_type = _type_filter(op, frame_types[op.input])
        elif isinstance(op, CalculateOp):
            new_type = _type_calculate(op.step_no, frame_types[op.input])
        elif isinstance(op, GroupOp):
            new_type = _type_group(frame_types[op.input])
        else:  # pragma: no cover -- exhaustive over the Op union
            raise RequestError(f"step {op.step_no} has unrecognized op {getattr(op, 'op', None)!r}")
        frame_types[op.output] = new_type
    return frame_types


def validate_shapes(pipeline: Pipeline) -> None:
    """Type `pipeline` (assumed already structurally valid -- see `pipeline.
    validate_pipeline`) via `annotate_frame_types`, then check the two rules that belong
    to the FINAL frame rather than to any one op. Raises `ShapeError` for every
    shape-dependent rejection decidable from the declared ops and each input's grid kind
    alone:

    - raster-to-raster `project` (`_type_project`, via `PROJECT_LOWERING_TABLE`'s
      absence);
    - `project.aggregate` missing where required, or supplied where forbidden
      (`_type_project`, via `request.aggregates.validate_project_aggregate`);
    - a second `count_by_value` in one chain (`_type_project`);
    - a `calculate` over a paired band-wide frame (`_type_calculate`);
    - a `join` mixing a long frame with a band-wide one (`_type_join`);
    - a band-wide `join` whose operands sit on two different grids (`_type_join`);
    - a `join` over an operand that is itself an uncollapsed join's output (`_type_join`);
    - a long final frame under an output format that needs a raster surface (below);
    - a band-wide final frame under an output format that needs a tabular sink (below --
      the converse of the previous rule; nothing past this function would otherwise
      catch it, since `execution.execute.run_to_path` picks its sink from the result's
      own shape, not from `output.format`);
    - more than one value column reaching the sink (below).

    Nothing here opens a file or knows a crosswalk.
    """
    ops = compile_ops(pipeline)
    frame_types = annotate_frame_types(pipeline, ops)
    final = frame_types[ops[-1].output] if ops else next(iter(frame_types.values()))

    if len(final.value_columns) != 1:
        raise ShapeError(
            f"pipeline's final frame carries {len(final.value_columns)} value columns "
            f"{sorted(final.value_columns)} -- every output writes exactly one measured "
            "quantity; collapse them into one first: a calculate over a long (vector) "
            "frame, or a value filter or a group over a band-wide (raster) one, where a "
            "calculate has no per-operand column to name (see _type_calculate)"
        )

    if final.shape == Shape.long and pipeline.output.format in RASTER_OUTPUT_FORMATS:
        raise ShapeError(
            f"pipeline's final frame is long (tabular), but output.format="
            f"{pipeline.output.format!r} needs a raster surface -- add a project to a "
            "raster grid before the sink"
        )

    if final.shape == Shape.band_wide and pipeline.output.format not in RASTER_OUTPUT_FORMATS:
        # The converse of the check above. A band-wide result is a `(LongFrame,
        # metadata, window)` triple, not a table -- `execution.execute.run_to_path`
        # picks its sink from the RESULT's shape, never from `pipeline.output.format`
        # (see that function's docstring), so nothing downstream of `validate_shapes`
        # would otherwise ever reject "band-wide result, output.format=csv": `run_to_
        # path` would silently call `sink.write_raster` and hand a caller who asked for
        # a table a GeoTIFF's bytes at a `.csv`-shaped path (or, from the tile routes,
        # a `png`/`jpg` URL would never even reach this pipeline shape, since a tabular
        # `output.format` is only reachable from the `.{format}` data routes, which
        # write whatever `run_to_path` returns without checking it matches the URL
        # extension). Rejecting here, at plan-time, is the only place that can catch it
        # without opening a file.
        raise ShapeError(
            f"pipeline's final frame is band-wide (raster), but output.format="
            f"{pipeline.output.format!r} needs a tabular sink -- add a project to a "
            "vector grid, or a spatial filter, before the sink"
        )
