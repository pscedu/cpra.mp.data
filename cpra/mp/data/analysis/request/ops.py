"""The op IR: a small typed `Op` per pipeline step -- `ProjectOp`/`JoinOp`/`FilterOp`/
`CaseOp`/`GroupOp`, five ops for the six `pipeline.py`'s `Step` union names, since a
`calculate` lowers to an arm-less `CaseOp` rather than getting one of its own -- plus
`compile_ops`, which turns an already-validated `Pipeline` into `list[Op]` **in declared
order**, and `validate_shapes`, which types every input/step's shape (`long` or
`band_wide`) and rejects the combinations that have no meaning (enumerated below).

**Shape-agnostic at the interface.** An `Op` carries only what the pipeline DECLARED: a
`project`'s target grid, a `filter`'s predicate, a `case`'s arms, a `group`'s
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
from cpra.mp.data.analysis.request.expressions import (
    PredicateKind,
    classify_predicate_kind,
    classify_where_kinds,
)
from cpra.mp.data.analysis.request.pipeline import (
    DOMAIN_COLUMN,
    RASTER_OUTPUT_FORMATS,
    ZONE_COLUMN,
    CalculateStep,
    CaseStep,
    CaseWhen,
    FilterStep,
    GroupStep,
    InputSelection,
    JoinStep,
    Pipeline,
    ProjectStep,
    UnpivotStep,
    _step_explicit_inputs,
)
from cpra.mp.data.analysis.request.types import AggregateFunction, Grid, TileFormat
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


class CaseOp(BaseModel):
    """Rewrite `input`'s running value by the first matching arm of `cases`, else `else_`
    (null when unset). `CaseWhen` is `pipeline.py`'s own model, reused rather than
    redeclared: an arm carries nothing an `Op` would need to resolve, so a second copy
    would only be a second thing to keep in step.

    **There is no `CalculateOp`.** A `calculate` compiles to this op with `cases == []` and
    `else_` its expression -- "the else, everywhere", which is what a `calculate` is once it
    no longer carries a `where` of its own. That is why `cases` takes no `min_length` here
    while `pipeline.CaseStep.cases` does: the arm-less form is a lowering target, not
    something a request spells `case`.

    `label` is what an error raised while compiling one of the expressions calls this entry
    -- diagnostic only, defaults to `None`, meaning "name me after my step".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["case"] = "case"
    step_no: int
    input: str
    cases: list[CaseWhen]
    else_: str | None = None
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


class UnpivotOp(BaseModel):
    """Fold `input`'s operand-named value columns into one `value` column plus the label
    column `into`. See `pipeline.UnpivotStep` for what the op means and why it is long-only;
    the op adds nothing the step did not already say.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["unpivot"] = "unpivot"
    step_no: int
    input: str
    into: str = "variable"
    output: str


Op = Annotated[
    ProjectOp | JoinOp | FilterOp | CaseOp | GroupOp | UnpivotOp,
    Field(discriminator="op"),
]


def calculate_label(op: CaseOp) -> str:
    """What an error raised while compiling `op`'s expression(s) should call it: the
    explicit `label`, else `step.<step_no>`, the pipeline step's own position.
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

# The unpivot lowering table, and the one table with a MISSING row on purpose:
# `Shape.band_wide` is absent, so `_type_unpivot`'s lookup raises `KeyError` and it turns
# that into a `ShapeError`, exactly as `_type_project` does for raster-to-raster. Saying
# "long only" as an absent key rather than as an `if` is what lets
# `tests/api/test_execute.py::TestLoweringTableAgreement` check the executor's registry
# against the same data the typing walk consults, instead of against a second hand-written
# list of which shapes this op accepts.
UNPIVOT_LOWERING_TABLE: dict[Shape, Shape] = {
    Shape.long: Shape.long,
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
            op = ProjectOp(step_no=step_no, input=input_name, to=step.to, aggregate=step.aggregate, output=output)
        elif isinstance(step, FilterStep):
            op = FilterOp(step_no=step_no, input=input_name, where=step.where, output=output)
        elif isinstance(step, CalculateStep):
            # The desugaring: a `calculate` is a `case` with no arms, whose `else` is the
            # expression. Nothing downstream of here knows the difference -- there is one
            # set of shape rules, one set of lowerings, and one expression semantics for
            # both spellings, which is the whole point of lowering rather than duplicating.
            op = CaseOp(step_no=step_no, input=input_name, cases=[], else_=step.expression, output=output)
        elif isinstance(step, CaseStep):
            op = CaseOp(
                step_no=step_no,
                input=input_name,
                cases=list(step.cases),
                else_=step.else_,
                output=output,
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
        elif isinstance(step, UnpivotStep):
            op = UnpivotOp(step_no=step_no, input=input_name, into=step.into, output=output)
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

    `extra_columns` is the CARRIED per-pixel columns a band-wide frame picks up alongside
    its `b<i>` bands -- `execution/frame.py`'s `extra_columns` contract, mirrored here so
    the typing walk knows about them before anything is opened. Only two names ever appear
    (`pipeline.ZONE_COLUMN`, `pipeline.DOMAIN_COLUMN`; see `_carried_columns` for where
    each is introduced), and it is empty for every long frame, since a carried column has
    no meaning once the frame is rows rather than pixels.

    It exists for `_type_filter`: a carried column is a real column of the DATA, one value
    per pixel, so a `where` naming it is a per-pixel mask -- the same lowering a value
    conjunct gets -- and not a dimension conjunct evaluated once per band against metadata
    that carries no such column. Classification is not cosmetic there: it picks the
    lowering AND the output shape (`FILTER_LOWERING_TABLE`), so getting it wrong is a
    typing error, not only a runtime one. `extra_columns` is deliberately kept separate
    from `value_columns` rather than folded into it: the two are unioned for the
    classification question alone, while `value_columns`' own count is what decides whether
    a band-wide frame has a PAIRING at all, and a carried column is not an operand's
    measurement.

    Defaults to empty, the answer for every long frame and every raster input no
    raster-to-vector `project` reaches -- which is to say, almost all of them.

    `folded_columns`/`folded_into` are what an `unpivot` leaves behind: the value-column
    names it folded away, and the label column they became values of. They exist because
    the fold is the one op that destroys information the RESULT is still about --
    `value_columns` afterwards is `{"value"}` like any other frame's, but the rows are five
    variables stacked, and `descriptor.output_descriptor` has to be able to say so (it
    resolves each value column's `Variable` by name, and after a fold there is no name left
    to resolve). Empty/`None` for every frame no `unpivot` reached, and carried forward
    unchanged by every op that `dataclasses.replace`s its input's type -- a later `filter`
    or `group` narrows a folded frame without un-folding it. A `join` builds a fresh
    `FrameType` and so resets both, which is correct: its operands' measurements are
    columns again.
    """

    shape: Shape
    grid: str
    has_count_by_value: bool
    value_columns: frozenset[str]
    extra_columns: frozenset[str] = frozenset()
    folded_columns: frozenset[str] = frozenset()
    folded_into: str | None = None


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


def _zone_carrying_inputs(pipeline: Pipeline, ops: list[Op]) -> frozenset[str]:
    """Which declared inputs OPEN with `pipeline.ZONE_COLUMN` already attached: the raster
    ones a raster-to-vector `project` is reachable from.

    `DOMAIN_COLUMN` needs no pre-pass -- the step that introduces it is the step whose
    output carries it, so the forward walk sees it the moment it appears. A zone id is the
    opposite: it is attached to an ORIGIN's own read because of a `project` declared
    somewhere later (`execution.execute._open_band_wide_input` passes `crosswalks=` for
    exactly the origins `execution.plan.ResolvedInput.straddling_vector_grid` names, and
    that field is itself a reachability fact), so a forward walk one op at a time cannot
    know it yet. This computes the same reachability `compile_plan` does -- origin sets
    carried forward, unioned at a fan-in, every raster origin reaching such a `project`
    marked -- and does it here so both layers agree about which frames carry the column.

    Reachability is over ALL ops, not just the chain a frame flows down to the sink: the
    column is attached at the READ, so a fan-out's other branches carry it whether or not
    they ever reach the `project` that asked for it (`execution/execute.py`'s closing
    paragraph makes the same point from the other end, about stripping it again).

    Grid kinds are recomputed here rather than read off `frame_types`, which does not exist
    yet: a frame's grid is its input's or the last `project`'s `to`, and nothing about
    carried columns can change that, so the two passes cannot disagree. Everything this
    pass would reject -- a project with no lowering, a mixed-shape join -- it simply
    carries past, leaving `annotate_frame_types`' own walk to raise the real error.
    """
    grids = {name: selection.grid for name, selection in pipeline.inputs.items()}
    origins = {name: frozenset({name}) for name in pipeline.inputs}
    carrying: set[str] = set()
    for op in ops:
        operands = list(op.inputs) if isinstance(op, JoinOp) else [op.input]
        reachable = frozenset().union(*(origins[name] for name in operands))
        origins[op.output] = reachable
        if isinstance(op, ProjectOp):
            grids[op.output] = op.to
            if GRIDS[grids[op.input]].grid_type == "raster" and GRIDS[op.to].grid_type == "vector":
                carrying |= {name for name in reachable if GRIDS[grids[name]].grid_type == "raster"}
        else:
            grids[op.output] = grids[operands[0]]
    return frozenset(carrying)


def _initial_state(selection: InputSelection, extra_columns: frozenset[str]) -> FrameType:
    grid_name = selection.grid
    kind = GRIDS[grid_name].grid_type
    shape = Shape.band_wide if kind == "raster" else Shape.long
    return FrameType(
        shape=shape,
        grid=grid_name,
        has_count_by_value=False,
        value_columns=frozenset({"value"}),
        extra_columns=extra_columns,
    )


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

    # A vector-to-raster project INTRODUCES `DOMAIN_COLUMN` and carries nothing in (its
    # input is long); every other direction leaves a LONG frame, which carries nothing at
    # all -- a zone id the input was carrying is consumed by the projection itself. Either
    # way the incoming frame's own set is never inherited, which is why this is the one
    # place that cannot spell the answer as "whatever came in".
    extra_columns = frozenset({DOMAIN_COLUMN}) if (from_kind, to_kind) == ("vector", "raster") else frozenset()
    return FrameType(
        shape=out_shape,
        grid=step.to,
        has_count_by_value=has_count_by_value,
        value_columns=frozenset({"value"}),
        extra_columns=extra_columns,
    )


def _type_filter(step_no: int, step: FilterStep | FilterOp, current: FrameType) -> FrameType:
    """`current.value_columns`, not the bare `value` default, is what the conjuncts are
    classified against -- on a PAIRED frame (one a `join` left carrying one operand-named
    value column per operand) `marsh > 0.5` is a value conjunct, and classifying it as a
    dimension one would send it to a lowering that evaluates it against band metadata,
    where no such column exists.

    Shape-preserving in `value_columns` as well as in `shape`: a `filter` RESTRICTS, in
    every one of its lowering positions, and restricting never changes how many operands a
    frame carries. That is why a value conjunct over a paired BAND-WIDE frame is rejected
    here rather than lowered. A band-wide frame's bands are positional, so a predicate
    naming two operands cannot be evaluated band by band; the only way to run it at all is
    to pair the bands up and emit one band per pair, which is a REDUCE, and a reduce has to
    pick which operand's value survives -- a choice nothing in the request states. `case`
    is the op that states it (`_type_case`), and a paired `case` is the same reduce with
    the answer written down. Long shape is unaffected: a long value filter is an ordinary
    row filter, every operand column left standing, nothing to pick between.

    `current.extra_columns` joins the value columns for the CLASSIFICATION only. A carried
    column is a real per-pixel column of the data (see `FrameType`), so `__zone__ = 7` is
    the per-pixel mask a value conjunct is, not a band selection evaluated against metadata
    that has no such column -- and `execution.band_ops.filter_band_wide` classifies the
    identical way, against the identical union, so the typed shape and the lowering that
    actually runs cannot disagree. It stays OUT of the pairing test below: that test asks
    how many operands a `join` left behind, and a carried column is not one of them.
    """
    kinds = classify_where_kinds(step.where, current.value_columns | current.extra_columns)
    if kinds[PredicateKind.spatial]:
        kind = PredicateKind.spatial
    elif kinds[PredicateKind.value]:
        kind = PredicateKind.value
    else:
        kind = PredicateKind.dimension
    if kind is PredicateKind.value and current.shape is Shape.band_wide and len(current.value_columns) > 1:
        # Re-classified against the OPERAND names alone: a value conjunct that comes back
        # `dimension` under that narrower namespace named only carried columns, and one of
        # those (`__domain__ > 0`) is a per-pixel mask every band takes independently --
        # nothing to pair, nothing to pick between. It is only an operand-named conjunct
        # that forces the pairing, so it is only that one this rejects.
        operand_named = [
            conjunct
            for conjunct in kinds[PredicateKind.value]
            if classify_predicate_kind(conjunct, current.value_columns) is PredicateKind.value
        ]
        if operand_named:
            raise ShapeError(
                f"step {step_no} (filter): conjunct(s) {operand_named} name a value column "
                f"of a paired band-wide frame (value columns "
                f"{sorted(current.value_columns)}), which is not something a filter can do "
                "-- a band-wide frame's bands are positional, so the only way to evaluate a "
                "predicate naming an operand is to pair the bands and emit one per pair, "
                "which reduces the frame and has to pick which operand's value survives. "
                "State it as a case instead, whose then= says which"
            )
    new_shape = FILTER_LOWERING_TABLE[(kind, current.shape)]
    # A spatial filter unpivots to long, and `execution.execute._run_filter_band_wide`
    # drops the carried columns from the result the moment it does -- there is no per-pixel
    # column once the frame is rows.
    extra_columns = current.extra_columns if new_shape is Shape.band_wide else frozenset()
    return dataclasses.replace(current, shape=new_shape, extra_columns=extra_columns)


def _type_case(step_no: int, step: CaseStep | CaseOp, current: FrameType) -> FrameType:
    """Shape-preserving, and always one value column out. It is the only op that collapses
    a band-wide pairing without also collapsing the bands: over a paired frame each
    operand's band is paired with its counterpart and the pair reduced to one band
    (`execution.band_ops.case_band_wide`), with the surviving value named by the matching
    arm's `then` -- which is exactly what `_type_filter` rejects a paired value predicate
    for not having. The pairing is something `case` consumes rather than something it
    cannot express. A `calculate` is this op with no arms (`compile_ops`), so it reaches
    that reduce through its `else` and needs no rule of its own.

    **An arm's `when` may be a value predicate, a dimension predicate, or both.** A
    dimension conjunct says WHICH ROWS/BANDS the arm applies to -- rows in long shape,
    where a dimension is an ordinary column, and bands in band-wide shape, where it is a
    row of `metadata_df`. Neither is a shape change: an arm that does not apply drops out
    of that row/band's chain and the remaining arms and the `else` still do, so the frame
    coming out has exactly the rows and bands that went in. That is what lets `case` need
    no lowering table at all -- it is unconditionally shape-preserving, so there is no
    (kind, shape) combination to tabulate.

    Whether the columns a dimension conjunct names EXIST is deliberately not checked here:
    `FrameType` records shapes and value columns, not dimension names, so this walk has
    nothing to check them against. `execution.long_ops.case_long` raises against the
    frame's real schema, and band-wide a conjunct naming an absent column is simply false
    for every band -- the same execution-time treatment a `filter`'s dimension half has
    always had.

    **A SPATIAL conjunct is still rejected.** It resamples the frame to points, which is a
    shape change, and an arm cannot do that to half its input. It is `filter`'s job, and
    writing it as a `filter` around or before the `case` says the same thing with the
    lowering it actually has.

    Classified against `value_columns | extra_columns`, the identical union `_type_filter`
    uses: a carried column (`__zone__`, `__domain__`) is a real per-pixel column, so an
    arm naming one is making a value predicate rather than a band selection, and the
    execution lowerings classify the same way.
    """
    columns = current.value_columns | current.extra_columns
    for position, case in enumerate(step.cases, start=1):
        spatial = classify_where_kinds(case.when, columns)[PredicateKind.spatial]
        if spatial:
            raise ShapeError(
                f"step {step_no} (case): arm {position}'s when={case.when!r} contains "
                f"grid_cell conjunct(s) {spatial} -- a spatial predicate resamples the "
                "frame to points, which is a shape change an arm cannot make to half its "
                "input; state that as a filter instead"
            )
    return dataclasses.replace(current, value_columns=frozenset({"value"}))


def _type_group(current: FrameType) -> FrameType:
    return dataclasses.replace(current, value_columns=frozenset({"value"}))


def _type_unpivot(step_no: int, step: UnpivotStep | UnpivotOp, current: FrameType) -> FrameType:
    """Fold several operand-named value columns into one `value` plus the label column
    `step.into`. Three rejections, in the order a caller hits them:

    **A band-wide input.** `UNPIVOT_LOWERING_TABLE` has no `band_wide` row, so the lookup
    below raises and this turns it into a `ShapeError`. A band-wide frame's columns are
    positional (`b0`, `b1`, ...) and what each one MEANS lives in band metadata, so folding
    them is not a fold at all -- it is the band-wide-to-long transition, which a spatial
    filter already is (`FILTER_LOWERING_TABLE`). Saying so beats letting the executor meet
    a frame it cannot name the columns of.

    **One value column.** Every frame that has not been through a `join` carries exactly
    `{"value"}` -- `_type_case`, `_type_group` and this function all collapse to that one
    name -- so folding one is never "fold the single measurement": it produces a label
    column whose every entry is the literal string `"value"`, which is a constant dressed
    up as a dimension. The caller meant to fold a join's output and put the `unpivot`
    somewhere that no longer sees one.

    **`into` naming one of the operands.** `unpivot(into="marsh")` over `{marsh, sal}` asks
    for a label column named after one of the things being labelled; polars would take the
    name and the result would carry `marsh` meaning "which operand" rather than "the marsh
    measurement". Structure alone cannot catch this (`pipeline._validate_label_column_name`
    does not know the frame's operands), which is why it is here.

    Everything else passes through: the fold drops no rows, changes no grid, and moves no
    dimension -- the frame is the same observations, stacked.
    """
    try:
        out_shape = UNPIVOT_LOWERING_TABLE[current.shape]
    except KeyError as exc:
        raise ShapeError(
            f"step {step_no} (unpivot): cannot unpivot a band-wide frame -- its columns "
            "are positional bands, not named measurements; a spatial filter (grid_cell) "
            "is what turns a band-wide frame long, and an unpivot goes after it"
        ) from exc

    if len(current.value_columns) < 2:
        raise ShapeError(
            f"step {step_no} (unpivot): input carries one value column "
            f"({sorted(current.value_columns)}), so there is nothing to fold -- unpivot "
            "stacks the operand-named value columns a join produces, and this frame has "
            "not been through one (or a later case/calculate/group already collapsed it)"
        )

    if step.into in current.value_columns:
        raise ShapeError(
            f"step {step_no} (unpivot): into={step.into!r} names one of the value columns "
            f"being folded ({sorted(current.value_columns)}) -- the label column has to be "
            "a name none of the operands already use"
        )

    return dataclasses.replace(
        current,
        shape=out_shape,
        value_columns=frozenset({"value"}),
        folded_columns=current.value_columns,
        folded_into=step.into,
    )


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
    something in between collapses the pairing back to one value column (a `case` or a
    `group` does exactly that; see `_type_case`). Rejecting here rather
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

    already_paired = [name for name, state in zip(step.inputs, operand_states) if len(state.value_columns) > 1]
    if already_paired:
        raise ShapeError(
            f"step {step_no} (join): operand(s) {already_paired} already carry more than "
            "one value column, and joining one of those is not yet supported -- a join "
            "names each operand's measurement after the operand, so an operand with "
            "several measurements has no single one for that name to mean; collapse it "
            "to one value column first (a case or a group over a band-wide frame does)"
        )

    has_count_by_value = any(s.has_count_by_value for s in operand_states)
    value_columns = frozenset(step.inputs)
    # UNIONED, matching `execution.band_ops.join_band_wide`, which takes each named carried
    # column from the first operand that has it: an operand can carry one the others never
    # had (a vector-to-raster `project` operand brings `DOMAIN_COLUMN` to a join with
    # rasters that have none). Long operands carry nothing, so a long join unions to empty
    # without needing to say so.
    extra_columns = frozenset().union(*(s.extra_columns for s in operand_states))
    return FrameType(
        shape=out_shape,
        grid=grid,
        has_count_by_value=has_count_by_value,
        value_columns=value_columns,
        extra_columns=extra_columns,
    )


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

    One pre-pass runs before the walk: `_zone_carrying_inputs`, which is the only part of
    `FrameType.extra_columns` a forward walk cannot see coming (see there). Everything else
    about carried columns is decided op by op, by the `_type_*` functions themselves.
    """
    zone_inputs = _zone_carrying_inputs(pipeline, ops)
    frame_types: dict[str, FrameType] = {
        name: _initial_state(selection, frozenset({ZONE_COLUMN}) if name in zone_inputs else frozenset())
        for name, selection in pipeline.inputs.items()
    }
    for op in ops:
        if isinstance(op, JoinOp):
            new_type = _type_join(op.step_no, op, [frame_types[name] for name in op.inputs])
        elif isinstance(op, ProjectOp):
            new_type = _type_project(op.step_no, op, frame_types[op.input])
        elif isinstance(op, FilterOp):
            new_type = _type_filter(op.step_no, op, frame_types[op.input])
        elif isinstance(op, CaseOp):
            new_type = _type_case(op.step_no, op, frame_types[op.input])
        elif isinstance(op, GroupOp):
            new_type = _type_group(frame_types[op.input])
        elif isinstance(op, UnpivotOp):
            new_type = _type_unpivot(op.step_no, op, frame_types[op.input])
        else:  # pragma: no cover -- exhaustive over the Op union
            raise RequestError(f"step {op.step_no} has unrecognized op {getattr(op, 'op', None)!r}")
        frame_types[op.output] = new_type
    return frame_types


def final_frame_name(pipeline: Pipeline, ops: list[Op]) -> str:
    """The name the pipeline's FINAL frame is bound to -- the last op's output, or, for a
    pipeline with no steps at all, its one input's own key.

    This is the same rule `execution.execute.execute` uses to decide which frame it hands
    back (see that function's docstring); the zero-step case is exact rather than a
    fallback, because `pipeline.validate_pipeline` already requires "zero steps implies
    exactly one input", so the unpacking below cannot see more than one name.

    Public, and takes `ops` rather than re-deriving them, because three callers need this
    answer against work they have already done: `validate_shapes` below (which has just
    compiled `ops`), `output_frame_type` (which compiles them for the same reason), and
    `api.pipeline`'s vector-tile renderer (which reads it off a compiled execution plan and
    must not compile a second time to ask). Before this was extracted each wrote the rule
    out again.
    """
    if ops:
        return ops[-1].output
    (name,) = pipeline.inputs
    return name


def output_frame_type(pipeline: Pipeline) -> FrameType:
    """The `FrameType` of `pipeline`'s final frame -- the shape, grid, and value columns of
    what it would WRITE.

    Opens nothing: `annotate_frame_types` needs only the declared ops and each input's grid
    kind, so this is available synchronously at spec-build time, before the pipeline is
    executed and before any crosswalk or catalog is consulted.

    `pipeline` must already be structurally valid (`pipeline.parse_pipeline`); this raises
    the same `ShapeError`s `validate_shapes` does for every per-op rule, since it runs the
    same walk. It does NOT run the sink rules -- a caller that wants those wants
    `validate_shapes`, which reaches the same frame by the same two lines.

    Callers outside the shape rules want this for the grid: `FrameType.grid` means "the
    grid this frame is currently addressed against", which is what decides whether a
    `grid_cell` is a vector id or a raster coordinate pair, and what a rendered result is
    drawn on. See `request.descriptor.output_descriptor`, which layers the presentation
    facts (time unit, per-column variable) on top of this one.
    """
    ops = compile_ops(pipeline)
    return annotate_frame_types(pipeline, ops)[final_frame_name(pipeline, ops)]


def _spatial_lowering(ops: list[Op], frame_types: dict[str, FrameType]) -> tuple[int, list[str]] | None:
    """The LAST `filter` op that turned a band-wide frame long, as `(step_no, the spatial
    conjuncts it did it with)` -- `None` when no op in `ops` did that.

    A filter is the only op whose lowering table changes shape (`FILTER_LOWERING_TABLE`),
    and within it band-wide-in/long-out is the single spatial entry, so the two frame types
    alone identify the cause with no re-classification needed. The conjuncts are re-derived
    only to QUOTE them back: `_type_filter` classified against
    `value_columns | extra_columns`, and this repeats that against the same input frame so
    the message names exactly the conjuncts that op treated as spatial.

    Used only to explain the long-final-frame rejection below. Nothing decides anything on
    the result -- a request that trips that rule is rejected either way, this only picks
    which message says so.
    """
    for op in reversed(ops):
        if not isinstance(op, FilterOp):
            continue
        current = frame_types[op.input]
        if current.shape is not Shape.band_wide or frame_types[op.output].shape is not Shape.long:
            continue
        kinds = classify_where_kinds(op.where, current.value_columns | current.extra_columns)
        return op.step_no, kinds[PredicateKind.spatial]
    return None


def validate_shapes(pipeline: Pipeline) -> None:
    """Type `pipeline` (assumed already structurally valid -- see `pipeline.
    validate_pipeline`) via `annotate_frame_types`, then check the three rules that belong
    to the FINAL frame rather than to any one op. Raises `ShapeError` for every
    shape-dependent rejection decidable from the declared ops and each input's grid kind
    alone:

    - raster-to-raster `project` (`_type_project`, via `PROJECT_LOWERING_TABLE`'s
      absence);
    - `project.aggregate` missing where required, or supplied where forbidden
      (`_type_project`, via `request.aggregates.validate_project_aggregate`);
    - a second `count_by_value` in one chain (`_type_project`);
    - a value `filter` over a paired band-wide frame (`_type_filter`);
    - a `case` arm whose `when` carries a spatial predicate (`_type_case`);
    - a `join` mixing a long frame with a band-wide one (`_type_join`);
    - a band-wide `join` whose operands sit on two different grids (`_type_join`);
    - a `join` over an operand that is itself an uncollapsed join's output (`_type_join`);
    - an `unpivot` over a band-wide frame, over a frame carrying one value column, or with
      an `into` naming one of the columns being folded (`_type_unpivot`);
    - a final frame that is not long on a VECTOR grid under `output.format='pbf'`
      (below);
    - a long final frame under an output format that needs a raster surface (below),
      reported against the `grid_cell` that caused it when a spatial `filter` is what
      lowered the frame (`_spatial_lowering`);
    - a band-wide final frame under an output format that needs a tabular sink (below --
      the converse of the previous rule; nothing past this function would otherwise
      catch it, since `execution.execute.run_to_path` picks its sink from the result's
      own shape, not from `output.format`);
    - more than one value column reaching the sink (below).

    Nothing here opens a file or knows a crosswalk.
    """
    ops = compile_ops(pipeline)
    frame_types = annotate_frame_types(pipeline, ops)
    final = frame_types[final_frame_name(pipeline, ops)]

    if len(final.value_columns) != 1:
        raise ShapeError(
            f"pipeline's final frame carries {len(final.value_columns)} value columns "
            f"{sorted(final.value_columns)} -- every output writes exactly one measured "
            "quantity; collapse them into one first with a case or a group, whose arms "
            "(or whose aggregate) say which operand the one surviving value comes from"
        )

    # The `pbf` rule, and the one sink rule that tests the final frame's GRID rather
    # than only its shape. A vector tile's geometry is the grid's own PMTiles boundary
    # archive (`execution.sink.sink_vector_tile`), into which this pipeline's values are
    # injected by `grid_cell` -- so the frame has to be long (rows to inject) AND
    # addressed against a grid that has polygons to inject them into. A raster point
    # query satisfies the first half and fails the second: its `grid_cell`s are pixel
    # ids of a grid with no boundary archive at all.
    #
    # Rejecting here, rather than letting `sink_vector_tile` return its `None`, is what
    # keeps "this pipeline could never have had a boundary archive" (a client error)
    # from collapsing into the 204 that means "no data at this location."
    #
    # Declared before the two rules below, which `pbf` would otherwise fall through
    # into: `pbf` is not a `RASTER_OUTPUT_FORMATS` member, so a band-wide final frame
    # would trip the "needs a tabular sink" rule and be told to add a spatial filter --
    # advice that produces a long frame still on a raster grid, which this rule rejects
    # anyway.
    if pipeline.output.format == TileFormat.pbf.value and (
        final.shape != Shape.long or GRIDS[final.grid].grid_type != "vector"
    ):
        raise ShapeError(
            f"pipeline's final frame (grid={final.grid!r}, shape={final.shape.value!r}) "
            "is not a long frame on a vector grid -- a .pbf vector tile needs the final "
            "result addressed against a vector grid's polygons, not a raster's pixels"
        )

    if final.shape == Shape.long and pipeline.output.format in RASTER_OUTPUT_FORMATS:
        # Same rejection either way; the two messages differ only in what they tell the
        # caller to change. A `grid_cell` on a raster grid means "sample these points", so
        # it lowers the frame to rows (`FILTER_LOWERING_TABLE`'s one shape-changing entry)
        # -- a deliberate, documented behaviour, not a missing `project`. Telling that
        # caller to "add a project to a raster grid" is advice that cannot work: projecting
        # the sampled points back onto a grid is not what they asked for, and the thing
        # they actually have to change is the `grid_cell` or the format. So when a spatial
        # filter is what did the lowering, name it.
        lowering = _spatial_lowering(ops, frame_types)
        if lowering is not None:
            step_no, conjuncts = lowering
            raise ShapeError(
                f"step {step_no} (filter): conjunct(s) {conjuncts} sample the raster at a "
                "set of grid_cell points, which makes the result tabular (long), but "
                f"output.format={pipeline.output.format!r} needs a raster surface. A "
                "grid_cell selection over a raster grid always returns the sampled points "
                "as rows -- ask for a tabular format to get them, or drop grid_cell to get "
                "the whole surface"
            )
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
