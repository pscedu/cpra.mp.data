"""`execute`: the walk. One loop over an `ExecutionPlan`'s `ops`, in declared order,
looking each op's lowering up in a single registry and applying it. No reordering: a
pipeline's declared order IS its semantics (see `tests/api/test_execute.py`'s
declared-order test for a concrete case where reordering gives a different answer).

**One registry, one loop, no per-op branching.** `_LOWERINGS` below is keyed by
`(op kind, dispatch key)` and every entry has the same signature -- take the frames bound
so far, the op, and the plan's frame types, and return the frame the op produces. The loop
therefore looks identical
for all five ops: compute the key, look up the lowering, call it, bind the result. What
differs between ops (a `join` reads several operands, a `filter` reads one and also a
predicate) is absorbed by that uniform signature inside each entry, rather than turning
the walk into a branch per op.

**Why the dispatch key is not always a shape.** For four of the five ops it is: the
incoming frame's shape selects the lowering. `project` is the exception, and not
arbitrarily -- it is the op that CHANGES shape, so the incoming shape does not determine
it. A long frame projects either to a vector grid (long out) or to a raster one (band-wide
out), and only the grid kinds say which. `project`'s key is therefore the
`(from kind, to kind)` pair, exactly the key `request.ops.PROJECT_LOWERING_TABLE` itself
uses, via the shared `request.ops.project_direction`.

**The lowering-table-vs-executor agreement.** `request/ops.py` holds
`FILTER_LOWERING_TABLE`/`PROJECT_LOWERING_TABLE`/`JOIN_LOWERING_TABLE` as declarative data:
every combination the request layer will type successfully. Until this module existed
nothing checked whether an executor exists for each of them. `NOT_YET_IMPLEMENTED` below
is the missing other half -- a named, deliberate record of exactly which combinations this
vertical slice does not execute yet (band-wide anything, and `project` in the two
directions that touch a raster grid) -- so `tests/api/test_execute.py` can assert, in both
directions, that the registered lowerings and the not-yet-implemented set together
partition exactly what the tables declare legal: no orphan executor for a combination no
table names, and no combination falling through neither set. Shrinking
`NOT_YET_IMPLEMENTED` is what adding a real band-wide lowering (or the remaining two
`project` directions) does, and the test is what forces whoever adds one to come back here
rather than leaving the new lowering un-wired by omission.

`calculate` and `group` have no lowering table of their own -- `request/ops.py` types both
as unconditionally shape-preserving, in code rather than as data -- so their entries below
are declared directly against `Shape`'s two members. There is no table for them to drift
from.
"""

from cpra.mp.data.analysis.execution import long_ops, project
from cpra.mp.data.analysis.execution.frame import LongFrame
from cpra.mp.data.analysis.execution.plan import ExecutionPlan
from cpra.mp.data.analysis.io.vector import scan_vector
from cpra.mp.data.analysis.request.ops import (
    PROJECT_LOWERING_TABLE,
    FrameType,
    JoinOp,
    Op,
    ProjectOp,
    Shape,
    project_direction,
)

__all__ = ["LoweringNotImplementedError", "NOT_YET_IMPLEMENTED", "execute"]


class LoweringNotImplementedError(NotImplementedError):
    """Raised by the walk when it reaches an `(op, key)` combination `request/ops.py`'s
    shape typing already accepts as legal, but for which no executor exists yet in this
    vertical slice.

    Deliberately NOT a member of the request layer's error family: the pipeline itself is
    well-formed -- it already passed `validate_shapes` -- so this is not a bad request, it
    is this engine's own, temporary, capability gap. A caller that wants "the engine cannot
    run this yet" catches this type; a caller that wants "the request was invalid" was
    never going to see it for a request that passed validation.
    """


# The named, deliberate not-yet-implemented set this vertical slice leaves for later
# stages -- see the module docstring. Each value holds the dispatch keys for which the
# walk raises `LoweringNotImplementedError` instead of dispatching: shapes for the four
# shape-keyed ops, and for `project` the `(from kind, to kind)` pairs
# `PROJECT_LOWERING_TABLE` declares MINUS `("vector", "vector")`, which has a real
# lowering below -- the one direction with no raster grid on either side of it, and so the
# one direction this vertical slice can execute without a raster scan.
NOT_YET_IMPLEMENTED: dict[str, frozenset] = {
    "filter": frozenset({Shape.band_wide}),
    "calculate": frozenset({Shape.band_wide}),
    "group": frozenset({Shape.band_wide}),
    "join": frozenset({Shape.band_wide}),
    "project": frozenset(PROJECT_LOWERING_TABLE) - {("vector", "vector")},
}


def _run_filter(frames: dict[str, LongFrame], op, frame_types) -> LongFrame:
    return long_ops.filter_long(frames[op.input], op.where)


def _run_calculate(frames: dict[str, LongFrame], op, frame_types) -> LongFrame:
    return long_ops.calculate_long(frames[op.input], op, frame_types[op.input].value_columns)


def _run_group(frames: dict[str, LongFrame], op, frame_types) -> LongFrame:
    return long_ops.group_long(frames[op.input], op)


def _run_join(frames: dict[str, LongFrame], op, frame_types) -> LongFrame:
    return long_ops.join_long([frames[name] for name in op.inputs], op)


def _run_project(frames: dict[str, LongFrame], op, frame_types) -> LongFrame:
    return project.project_vector_to_vector(frames, op, frame_types)


# The registry the walk dispatches through. One uniform signature per entry --
# `(frames, op, frame_types) -> LongFrame` -- so the loop below never has to know which op
# it is running. A filter's three predicate kinds share one entry because all three lower
# identically for a long frame (an ordinary row drop; see `long_ops.filter_long`); the
# band-wide half, where the three kinds genuinely differ, is later-stage work and will
# need the predicate kind in its key.
#
# `frame_types` is the plan's whole name -> `FrameType` map rather than just the incoming
# operand's, so that the signature stays the same for `join` (several operands) as for the
# rest. `_run_calculate` reads it for the incoming frame's value columns, which a long
# frame's schema cannot reveal on its own; `_run_project` reads it for the incoming
# frame's CURRENT grid (`current.grid`, not the pipeline input's own declared grid --
# `request.ops.FrameType`'s own docstring is where that distinction is spelled out), which
# is the other half of which crosswalk to open. Handing `frame_types` to every entry is
# what keeps this a registry rather than five differently-shaped callables the loop would
# have to tell apart.
_LOWERINGS: dict[tuple[str, object], object] = {
    ("filter", Shape.long): _run_filter,
    ("calculate", Shape.long): _run_calculate,
    ("group", Shape.long): _run_group,
    ("join", Shape.long): _run_join,
    ("project", ("vector", "vector")): _run_project,
}


def _operand_names(op: Op) -> list[str]:
    """The names of the frames `op` reads. `join` is the one op with several operands (and
    the one with no implicit "chain from the previous step" default), which is why this is
    a list rather than a single name.
    """
    return list(op.inputs) if isinstance(op, JoinOp) else [op.input]


def _dispatch_key(op: Op, frame_types: dict[str, FrameType]) -> object:
    """The key `op` dispatches on: its incoming frame's shape, or -- for `project`, the op
    whose lowering the incoming shape cannot determine -- the `(from kind, to kind)` pair.
    See the module docstring.

    A `join`'s operands are already known to share one shape (`validate_shapes` rejects a
    mixed join outright), so the first operand's shape speaks for all of them.
    """
    incoming = frame_types[_operand_names(op)[0]]
    if isinstance(op, ProjectOp):
        return project_direction(incoming, op.to)
    return incoming.shape


def _lowering_for(kind: str, key: object):
    """Look up `kind`'s lowering for `key`, raising `LoweringNotImplementedError` if `key`
    is in `NOT_YET_IMPLEMENTED[kind]` -- the one place that check happens, so every op
    shares an identical failure path instead of repeating it.
    """
    if key in NOT_YET_IMPLEMENTED[kind]:
        raise LoweringNotImplementedError(
            f"no {kind!r} lowering yet for {key!r} -- this pipeline's shape typing already "
            "accepted it, the walk just cannot run it yet"
        )
    return _LOWERINGS[(kind, key)]


def execute(plan: ExecutionPlan, *, base_path=None) -> LongFrame:
    """Run `plan` and return its final frame, lazily -- collecting is left to the caller,
    matching every other lowering in this package staying lazy until something downstream
    needs the data.

    Opens every long-shaped input up front via `io.vector.scan_vector`, with whatever
    pushdown the plan recorded for it. A band-wide input is deliberately left unopened:
    there is no raster scan in this vertical slice, and the first op to touch such a frame
    raises `LoweringNotImplementedError` before anything needs the frame a scan would have
    produced.

    Then walks `plan.ops` once, front to back, dispatching each through `_LOWERINGS` on
    `(op kind, dispatch key)` -- reading the frame types the plan already recorded at
    compile time rather than re-deriving any of them here.

    Returns the single input's frame for a zero-step pipeline: there is nothing to walk,
    and the request layer's own rule guarantees exactly one input in that case.
    """
    frames: dict[str, LongFrame] = {
        name: scan_vector(
            resolved.selection,
            base_path=base_path,
            columns=resolved.columns,
            predicate=resolved.predicate,
        )
        for name, resolved in plan.inputs.items()
        if plan.frame_types[name].shape is Shape.long
    }

    last_output: str | None = None
    for op in plan.ops:
        lowering = _lowering_for(op.op, _dispatch_key(op, plan.frame_types))
        frames[op.output] = lowering(frames, op, plan.frame_types)
        last_output = op.output

    if last_output is None:
        (last_output,) = frames.keys()
    return frames[last_output]
