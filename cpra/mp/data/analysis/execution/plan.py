"""`compile_plan`: turn an already-validated `Pipeline` into an `ExecutionPlan` -- the one
thing `execution/execute.py`'s walk actually consumes. Deliberately minimal: this vertical
slice (vector-only, no pushdown, no project, no raster) only needs the walk to know, for
every named input and step output, what it CARRIES and what to open it with; a later engine
milestone's `ExecutionPlan` will carry more (resolved windows, crosswalks, catalog
queries), but every field this one has already earns its place against something this
vertical slice actually executes.

**Compile opens no data files.** Everything here is derived from `pipeline` itself plus
`cpra.mp.data.grids.GRIDS` (a static table of grid metadata, not a scan) -- the same "no
IO at plan time" property `request.ops.validate_shapes` already has, which is what lets a
caller compile (and, later, plan pushdown against) a pipeline without touching a
filesystem.

**The pushdown seam.** `ResolvedInput.columns`/`.predicate` are exactly the projection/
predicate decision `io.vector.scan_vector` already knows how to accept -- always `None`
here, since nothing in this vertical slice computes a real one yet. Recording the decision
on the plan (rather than, say, a scan-time default the walk would have to invent) is
deliberate: whichever later stage adds real pushdown only has to fill in these two fields
differently per input; the walk itself (`execute.py`) never has to change to use them,
because it already reads them off the plan instead of deriving anything at run time.

**Where the frame types come from.** `ExecutionPlan.frame_types` maps every input/step-
output NAME to the `request.ops.FrameType` it carries -- its shape, the grid it is
currently addressed against, and the value columns it holds. That is computed once, here,
by calling `request.ops.annotate_frame_types`, which is the request layer's own public
answer to the question and the same walk `validate_shapes` runs. This module holds no copy
of those rules: reproducing, say, `_type_filter`'s spatial > value > dimension precedence
here would give the two layers separate definitions of one thing with no way to notice
them drifting apart.
"""

import dataclasses

from cpra.mp.data.analysis.request.ops import FrameType, Op, annotate_frame_types, compile_ops
from cpra.mp.data.analysis.request.pipeline import InputSelection, Pipeline

__all__ = ["ExecutionPlan", "ResolvedInput", "compile_plan"]


@dataclasses.dataclass(frozen=True)
class ResolvedInput:
    """One `Pipeline.inputs` entry, plus the pushdown decision `io.vector.scan_vector`
    accepts for it -- see the module docstring for why both `columns` and `predicate` are
    always `None` in this vertical slice.
    """

    selection: InputSelection
    columns: list[str] | None = None
    predicate: object | None = None  # `pl.Expr | None`; left untyped to avoid importing
    # polars into a module that never constructs one -- nothing here builds a predicate,
    # only carries whatever a later stage puts here.


@dataclasses.dataclass(frozen=True)
class ExecutionPlan:
    """`inputs`: every `Pipeline.inputs` entry, resolved (see `ResolvedInput`). `ops`:
    `request.ops.compile_ops`'s output, unchanged -- already fully resolved, in declared
    order. `frame_types`: every input/step-output name's `request.ops.FrameType`, computed
    once at compile time (see the module docstring) so `execute.py`'s walk never re-derives
    one.
    """

    inputs: dict[str, ResolvedInput]
    ops: list[Op]
    frame_types: dict[str, FrameType]


def compile_plan(pipeline: Pipeline) -> ExecutionPlan:
    """Compile `pipeline` (already validated -- `parse_pipeline` then `validate_shapes`)
    into the `ExecutionPlan` `execute.execute` consumes. Opens no data file; see the
    module docstring.
    """
    ops = compile_ops(pipeline)
    return ExecutionPlan(
        inputs={
            name: ResolvedInput(selection=selection, columns=None, predicate=None)
            for name, selection in pipeline.inputs.items()
        },
        ops=ops,
        frame_types=annotate_frame_types(pipeline, ops),
    )
