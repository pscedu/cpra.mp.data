"""`compile_plan`: turn an already-validated `Pipeline` into an `ExecutionPlan` -- the one
thing `execution/execute.py`'s walk consumes. Alongside the op list and per-name frame
types, this is where pushdown gets decided: for every named input, how much of what a
later op in the chain will do to it can be pulled back into the read itself, instead of
being applied only after the whole frame comes back.

**What "pushed" means for the two shapes.** A vector (long-shape) input is read through
`io.vector.scan_vector`, which accepts a column list and a row predicate -- ordinary
Parquet-scan pushdown, safe wherever it applies because it only ever changes WHEN a row or
column disappears, never WHETHER the final answer contains it. A raster (band-wide) input
is read through `io.raster.scan_raster`, where the position of a "column" is a band -- a
band a pipeline never actually reads is a file that VRT construction never has to touch at
all, which is why band selection (not a per-pixel row filter) is that shape's pushdown
target; see the module-level note below on why a row predicate does NOT get the same
treatment there.

**The forward walk.** `ops` (already fully resolved by `request.ops.compile_ops`, in
declared order) is walked once, carrying one `_Validity` state per line of computation
back to whichever named input it ultimately traces to (its "origin"). At each op, the
state says whether a `value` predicate, a dimension predicate on some still-open set of
dimension names, and a `grid_cell` (spatial) predicate could still be pushed all the way
back to the origin's own read without changing what any downstream consumer -- including
the final sink -- would see. This is the same table `request/ops.py`'s shape-typing walk
already commits to per op, restated here as a validity question rather than a shape
question:

* `filter` never changes what is still pushable -- it only *consumes* validity (deciding,
  per conjunct, whether THIS filter's own predicate can be pushed), it does not reduce it
  for whatever comes after.
* `calculate` invalidates `value` unconditionally (the running value is being replaced),
  leaving dimension/spatial validity untouched.
* `group` invalidates `value` unconditionally and narrows the open dimension set down to
  exactly `by` (grouping away a dimension makes it disappear from the frame, so nothing
  downstream can filter on it even in principle; a dimension that IS in `by` survives the
  group under its own name and remains legitimately pushable afterward -- a dimension
  filter declared after a `group` on a dimension that same `group` grouped by can still be
  pushed all the way back to the origin's own read, since every row the origin's read
  would drop is a row the `group` would have dropped from that dimension's bucket anyway).
  `grid_cell` survives only if `"grid_cell" in by`; band identity does not survive a group
  at all (a `group` collapses many pixels/rows into one), except that a dimension already
  named in `by` is, per the same reasoning above, still a legitimate thing to narrow a
  raster origin's band selection by afterward.
* `project` invalidates everything EXCEPT the one direction whose own semantics preserve
  `value` exactly: vector-to-raster (`project_direction` returns `("vector", "raster")`).
  That direction takes no aggregate (`request.ops._type_project`'s
  `direction_takes_aggregate` is `False` there), so a raster pixel's value after such a
  project is identically the value its containing vector zone already had -- meaning a
  `value` predicate declared AFTER this project is exactly as valid a filter on the
  original vector rows as it would be on the post-project pixels, and gets pushed all the
  way back to that vector origin's own scan. Every other project direction, and every
  other kind of validity (dimension, `grid_cell`), stops here regardless of direction:
  once rows have been redistributed onto a different grid, there is no single upstream row
  a downstream dimension/spatial predicate corresponds to.
* `join` stops everything, unconditionally, for both operands. Nothing in this pipeline
  model chains an op backward THROUGH a join -- pushing a predicate declared after a join
  back through only one operand would silently change what the other operand's rows get
  matched against, so it is simplest and always correct to just not attempt it. A
  band-wide `join` gets one piece of special handling despite this: since
  `execution.band_ops.join_band_wide` reads both operands through one shared VRT (see that
  module), the operands' resolved band metadata still needs to be sized as a UNION for the
  `ANALYSIS_BAND_CEILING` check below, even though no predicate pushes through the join
  itself.

**Which origins a frame came from is tracked separately from whether pushdown is still
valid, and the two must not be conflated.** They answer different questions, and the
things that invalidate pushdown mostly do not make the origin unknown. A frame read by
two consumers is no longer safe to push a predicate back through, but it still came from
exactly the same physical read it always did; a frame that has been through a `group` has
lost its `value` validity but not its provenance. `origin_sets` therefore carries, for
every name, the set of inputs feeding it, propagated through every op unconditionally --
a join's output is the union of its operands' sets, and everything else inherits its
input's. Only that set decides whether a band-wide `join` is recorded as a shared scan.
Deriving it instead from the pushdown state would silently drop the shared-scan
requirement for any join whose operand had been fanned out or nested under an earlier
join, which is not a pushdown optimization the planner is free to skip -- it is the
constraint that makes the join's two operands land on the same pixels at all.

**Fan-out is where a naive version of this walk breaks.** A name (an input or a step
output) can be read by more than one later op -- nothing in this pipeline model requires a
step chain to be linear. If two different ops both read the same upstream frame with two
different predicates, accumulating BOTH predicates onto that frame's origin's own physical
read would be wrong for whichever branch didn't ask for the other one's narrowing (the
shared read has only one result, and both branches see it). So `reference_count` is
computed once, up front, over every op's `input`/`inputs`; the moment a name has more than
one consumer, every op that reads it is treated exactly as if that name's validity state
were absent (nothing about it is used, and its own output stops tracking too). Everything
already accumulated onto an origin from BEFORE such a branch point remains valid --
that part of the chain really was exclusive up to there -- but nothing past it is ever
added.

**Column pushdown needs one more guard beyond validity/fan-out: a `group`.** A `filter`'s
or `calculate.where`'s referenced dimension names get folded into an origin's candidate
column set as they're encountered, but that set is only ever actually handed to
`io.vector.scan_vector` as a real narrowing (`ResolvedInput.columns`) once the origin's
chain has passed through at least one `group`. Before that point, narrowing the source
read's columns to "only what's been referenced so far" would be wrong even though every
individual predicate pushed so far is fine: a `filter`/`calculate`-only chain does not
touch which COLUMNS the frame carries (a filter drops rows, a calculate replaces `value`,
neither drops a dimension column), so if the chain reaches the sink without ever grouping,
the sink is still entitled to see every dimension column the raw scan has -- narrowing
would silently make some of them disappear. A `group`, by contrast, is a hard closure: its
own lowering (`execution.long_ops.group_long`) only ever emits `by + grid_cell + value`
columns, so whatever the source scan is narrowed to (every column already referenced,
unioned with `by` and `of`) reproduces an identical `group` result to reading everything
and letting the group discard the rest -- nothing downstream of a `group` can reach back
past it to notice the narrowing at all. There is no such gap for row-predicate pushdown:
dropping a row sooner or later gives the identical final row set, so validity/fan-out
alone govern it.

**Raster dimension-predicate pushdown needs no such gate.** Narrowing which bands survive
via a dimension predicate reproduces exactly what a real, executed dimension filter would
have done to that band-wide frame anyway (`execution.band_ops.select_dimension_band_wide`)
-- nothing about the surviving bands' own columns changes, so there is nothing analogous
to the vector "the sink still wants an untouched dimension column" risk to guard against.

**A raster value predicate is never pushed to the read, on purpose, and this is enforced
at both layers.** Per the table above, a `value` conjunct reaching a `filter` while its
origin is still band-wide never gets added to that origin's `ResolvedInput` at all --
masking a band-wide frame by value is always `execution.band_ops.mask_value_band_wide`'s
job, applied per pixel to the ALREADY-READ frame, never a row drop at the read. This
matters because `io.raster.scan_raster`'s IO-plugin `source()` now actually honors
whatever `predicate` it is given (see that module's docstring) -- Polars removes a
pushed-down `.filter()` from the query plan once an IO plugin accepts it and does not
re-apply it, so a value predicate that reached a band-wide read would silently drop pixel
ROWS, which drops that pixel from every band at once and breaks every band-wide invariant
downstream. The rule is therefore load-bearing at the `scan_raster` boundary, not just a
plan-shaping convenience -- which is also why it needs to be tested at the IO level, not
only by inspecting the plan.

**A spatial predicate reaching a band-wide read is a point sample.** Recorded as
`ResolvedInput.point_sample_step` (the step number of the `filter` that first samples that
origin) -- purely a marker for now. What point-sparse read mechanism actually consumes it,
and how per-pixel ops recorded before that point get replayed against the sampled rows,
is for the executor that eventually opens a band-wide input to decide; this module commits
only to WHERE that filter sits in the declared chain and to no longer tracking origin
validity for that origin past it (per the table above, a spatial filter turns a band-wide
frame long, so anything after it addresses a different, already-materialized frame).

**Band enumeration and its ceiling.** Every raster origin gets `io.catalog.
resolve_raster_bands` called on it exactly once, narrowed by whatever dimension predicates
survived pushdown, then checked against `cpra.mp.data.config.ANALYSIS_BAND_CEILING`
(`BandLimitError` if it's exceeded) -- except an origin feeding a recorded band-wide join,
which is exempted from its own check and instead checked as part of that join's operand
UNION (`ExecutionPlan.band_wide_join_scans`), matching how `execution.band_ops.
join_band_wide` actually reads both operands through one shared VRT. A dimension predicate
that legitimately narrows a raster origin to zero surviving bands is not an error here --
`resolve_raster_bands` itself never raises for an empty result, and naming a dimension
value that is legal but simply absent from the data is supposed to produce an empty
result rather than a failure.

There is one loose end there, recorded because it is easy to trip over from either side:
`io.raster.scan_raster` raises `ValueError` on zero-band metadata, since it derives the
read's window from the VRT it opens and a zero-band read opens nothing. That is a property
of where `scan_raster` happens to get its window, NOT a real obstacle -- a full-extent,
undecimated window is fully determined by the raster grid itself, which already carries
`bounds`, `height`, `width` and `crs` (`cpra.mp.data.grids.RasterGrid`); the transform
follows from `from_bounds(*bounds, width, height)`, and `batch_layout` is legitimately
empty when no batch was ever read. So an empty band-wide frame for a zero-band selection
is constructible without opening a file, and the grid is where its window should come
from. Nothing reaches this combination today only because `execute` does not yet dispatch
band-wide ops to `execution.band_ops` (which has implemented them); whoever wires that up
owns building the empty frame, and should take the window from the grid rather than
weakening the empty-result rule.

**Compile still opens no data file for a vector input** -- `io.vector.scan_vector` is
never called here, matching the request layer's own "no IO at plan time" property. A
raster input is the one exception: resolving its band list requires one catalog read
(`io.catalog.resolve_raster_bands`, which in turn calls `cpra.mp.data.raster.read_cog`) --
unavoidable, since which bands exist can only be answered by looking.
"""

import dataclasses
from collections import Counter

import polars as pl

from cpra.mp.data.analysis.io.catalog import resolve_raster_bands
from cpra.mp.data.analysis.request.errors import BandLimitError, RequestError
from cpra.mp.data.analysis.request.expressions import PredicateKind, classify_where_kinds
from cpra.mp.data.analysis.request.ops import (
    CalculateOp,
    FilterOp,
    FrameType,
    GroupOp,
    JoinOp,
    Op,
    ProjectOp,
    Shape,
    annotate_frame_types,
    compile_ops,
    project_direction,
)
from cpra.mp.data.analysis.request.pipeline import InputSelection, Pipeline
from cpra.mp.data.config import ANALYSIS_BAND_CEILING
from cpra.mp.data.grids import GRIDS

__all__ = ["ExecutionPlan", "ResolvedInput", "compile_plan"]


@dataclasses.dataclass(frozen=True)
class ResolvedInput:
    """One `Pipeline.inputs` entry, plus whatever pushdown `compile_plan` decided is safe
    for it.

    `columns`/`predicate` are what `io.vector.scan_vector` accepts, populated only for an
    input on a vector grid -- see the module docstring for exactly when each is non-`None`.
    `band_metadata`/`point_sample_step` are the raster-side counterpart, populated only for
    an input on a raster grid: `band_metadata` is `io.catalog.resolve_raster_bands`'s
    result, narrowed by every dimension predicate pushdown allowed through, and
    `point_sample_step` is the step number of the first spatial filter reached while this
    origin was still band-wide (`None` if none was). An input's grid kind decides which
    pair of fields is populated; the other pair is left at its default.
    """

    selection: InputSelection
    columns: list[str] | None = None
    predicate: object | None = None  # `pl.Expr | None`; left untyped to avoid importing
    # polars into every reader of this dataclass's annotations.
    band_metadata: object | None = None  # `pl.DataFrame | None`; same reasoning.
    point_sample_step: int | None = None


@dataclasses.dataclass(frozen=True)
class ExecutionPlan:
    """`inputs`: every `Pipeline.inputs` entry, resolved (see `ResolvedInput`). `ops`:
    `request.ops.compile_ops`'s output, unchanged. `frame_types`: every input/step-output
    name's `request.ops.FrameType`, computed once here so `execute.py`'s walk never
    re-derives one. `band_wide_join_scans`: every band-wide `join` step's output name
    mapped to the origins whose bands were read as one shared union for its ceiling check
    (see the module docstring) -- empty unless the pipeline has such a join.
    """

    inputs: dict[str, ResolvedInput]
    ops: list[Op]
    frame_types: dict[str, FrameType]
    band_wide_join_scans: dict[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class _Validity:
    """Per-origin pushdown state carried forward through one line of computation while
    walking `ops` -- see the module docstring's table for what each op does to it.

    `origin` is the named input this state still traces back to. `value`/`grid_cell` are
    plain booleans: still pushable, or not. `dimensions` is `None` (every dimension still
    open) or the exact `frozenset` of dimension names still open -- `group` is the only op
    that ever narrows this, to exactly its own `by`.
    """

    origin: str
    value: bool
    dimensions: frozenset[str] | None
    grid_cell: bool


def _referenced_names(conjunct: str) -> frozenset[str]:
    """The column names one already-validated `where` conjunct references -- reuses
    `pl.Expr.meta.root_names()`, the same mechanism `request.expressions.
    classify_predicate_kind` itself classifies conjuncts by.
    """
    return frozenset(pl.sql_expr(conjunct).meta.root_names())


def compile_plan(pipeline: Pipeline, *, base_path=None) -> ExecutionPlan:
    """Compile `pipeline` (already validated -- `parse_pipeline` then `validate_shapes`)
    into the `ExecutionPlan` `execute.execute` consumes, including every pushdown decision
    the module docstring describes. `base_path` is forwarded to `io.catalog.
    resolve_raster_bands` for each raster input; left `None` to use the configured default
    (tests always pass a sandboxed path instead -- see `tests/conftest.py`).
    """
    ops = compile_ops(pipeline)
    frame_types = annotate_frame_types(pipeline, ops)
    origin_kind = {name: GRIDS[selection.grid].grid_type for name, selection in pipeline.inputs.items()}

    # Fan-out detection (see module docstring): a name read by more than one op can never
    # have anything pushed back through it, since the two readers might not agree.
    reference_count: Counter[str] = Counter()
    for op in ops:
        if isinstance(op, JoinOp):
            for name in op.inputs:
                reference_count[name] += 1
        else:
            reference_count[op.input] += 1

    states: dict[str, _Validity | None] = {
        name: _Validity(origin=name, value=True, dimensions=None, grid_cell=True) for name in pipeline.inputs
    }

    vector_origins = [name for name, kind in origin_kind.items() if kind == "vector"]
    raster_origins = [name for name, kind in origin_kind.items() if kind == "raster"]
    vector_predicate_parts: dict[str, list] = {name: [] for name in vector_origins}
    vector_columns: dict[str, set[str]] = {name: {"grid_cell", "value"} for name in vector_origins}
    vector_columns_ok: dict[str, bool] = {name: False for name in vector_origins}
    raster_dimension_parts: dict[str, list] = {name: [] for name in raster_origins}
    point_sample_step: dict[str, int] = {}
    band_wide_join_origin_groups: dict[str, tuple[str, ...]] = {}

    # Provenance, tracked independently of `states` -- see the module docstring. A tuple
    # rather than a set so the recorded scan order is the order the operands were
    # declared in, which is deterministic and is the order a reader would expect.
    origin_sets: dict[str, tuple[str, ...]] = {name: (name,) for name in pipeline.inputs}

    def _incoming(name: str) -> "_Validity | None":
        state = states.get(name)
        if state is None or reference_count[name] > 1:
            return None
        return state

    for op in ops:
        if isinstance(op, JoinOp):
            # Indexed, not `.get`: every op below writes its own output's entry before it
            # continues, and `validate_pipeline` has already rejected a step naming a frame
            # that doesn't exist, so a miss here is a broken invariant and should say so.
            union = tuple(dict.fromkeys(name for operand in op.inputs for name in origin_sets[operand]))
            origin_sets[op.output] = union
            if (
                frame_types[op.output].shape is Shape.band_wide
                and all(origin_kind[name] == "raster" for name in union)
            ):
                band_wide_join_origin_groups[op.output] = union
            states[op.output] = None
            continue

        origin_sets[op.output] = origin_sets[op.input]

        incoming = _incoming(op.input)
        if incoming is None:
            states[op.output] = None
            continue

        if isinstance(op, ProjectOp):
            from_kind, to_kind = project_direction(frame_types[op.input], op.to)
            if from_kind == "vector" and to_kind == "raster":
                states[op.output] = _Validity(
                    origin=incoming.origin, value=incoming.value, dimensions=frozenset(), grid_cell=False
                )
            else:
                states[op.output] = None
            continue

        origin = incoming.origin
        kind = origin_kind[origin]

        if isinstance(op, FilterOp):
            kinds = classify_where_kinds(op.where)
            if kind == "vector":
                for bucket in kinds.values():
                    for conjunct in bucket:
                        vector_columns[origin] |= _referenced_names(conjunct)
                for conjunct in kinds[PredicateKind.dimension]:
                    names = _referenced_names(conjunct)
                    if incoming.dimensions is None or names <= incoming.dimensions:
                        vector_predicate_parts[origin].append(pl.sql_expr(conjunct))
                for conjunct in kinds[PredicateKind.value]:
                    if incoming.value:
                        vector_predicate_parts[origin].append(pl.sql_expr(conjunct))
                for conjunct in kinds[PredicateKind.spatial]:
                    if incoming.grid_cell:
                        vector_predicate_parts[origin].append(pl.sql_expr(conjunct))
                states[op.output] = incoming
            else:  # raster origin -- always band-wide here, since a spatial filter or a
                # non-value-preserving project (the only ways to leave band-wide shape for
                # a raster origin) already sets this state to `None` at the point it
                # happens, and nothing else changes a raster origin's shape.
                for conjunct in kinds[PredicateKind.dimension]:
                    names = _referenced_names(conjunct)
                    if incoming.dimensions is None or names <= incoming.dimensions:
                        raster_dimension_parts[origin].append(pl.sql_expr(conjunct))
                # A value conjunct is never pushed to a band-wide read regardless of
                # `incoming.value` -- see the module docstring.
                if kinds[PredicateKind.spatial]:
                    point_sample_step.setdefault(origin, op.step_no)
                    states[op.output] = None
                else:
                    states[op.output] = incoming
            continue

        if isinstance(op, CalculateOp):
            if kind == "vector" and op.where is not None:
                for bucket in classify_where_kinds(op.where).values():
                    for conjunct in bucket:
                        vector_columns[origin] |= _referenced_names(conjunct)
            states[op.output] = dataclasses.replace(incoming, value=False)
            continue

        if isinstance(op, GroupOp):
            by_set = frozenset(op.by)
            if kind == "vector":
                vector_columns[origin] |= by_set - {"grid_cell"}
                if op.of != "value":
                    vector_columns[origin].add(op.of)
                # A `group` is a hard closure on columns -- see the module docstring for
                # why it is safe to narrow the source read to exactly what has been
                # referenced so far, unioned with `by`/`of`, from this point on.
                vector_columns_ok[origin] = True
            new_dimensions = by_set if incoming.dimensions is None else (incoming.dimensions & by_set)
            new_grid_cell = incoming.grid_cell and ("grid_cell" in by_set)
            states[op.output] = _Validity(origin=origin, value=False, dimensions=new_dimensions, grid_cell=new_grid_cell)
            continue

        raise RequestError(f"step {op.step_no} has unrecognized op {getattr(op, 'op', None)!r}")  # pragma: no cover -- exhaustive over the Op union

    raster_band_metadata: dict[str, pl.DataFrame] = {}
    for name in raster_origins:
        metadata = resolve_raster_bands(pipeline.inputs[name], base_path=base_path)
        for expr in raster_dimension_parts[name]:
            metadata = metadata.filter(expr)
        raster_band_metadata[name] = metadata

    for join_output, origins in band_wide_join_origin_groups.items():
        total = sum(raster_band_metadata[name].height for name in origins)
        if total > ANALYSIS_BAND_CEILING:
            raise BandLimitError(
                f"step {join_output!r} (join): the shared union of {origins} resolves to "
                f"{total} bands, over the limit of {ANALYSIS_BAND_CEILING} -- see "
                "cpra.mp.data.config.ANALYSIS_BAND_CEILING"
            )

    joined_origins = {name for origins in band_wide_join_origin_groups.values() for name in origins}
    for name in raster_origins:
        if name in joined_origins:
            continue
        height = raster_band_metadata[name].height
        if height > ANALYSIS_BAND_CEILING:
            raise BandLimitError(
                f"input {name!r} resolves to {height} bands, over the limit of "
                f"{ANALYSIS_BAND_CEILING} -- see cpra.mp.data.config.ANALYSIS_BAND_CEILING"
            )

    inputs: dict[str, ResolvedInput] = {}
    for name, selection in pipeline.inputs.items():
        if origin_kind[name] == "raster":
            inputs[name] = ResolvedInput(
                selection=selection,
                band_metadata=raster_band_metadata[name],
                point_sample_step=point_sample_step.get(name),
            )
        else:
            predicate = None
            for part in vector_predicate_parts[name]:
                predicate = part if predicate is None else predicate & part
            columns = sorted(vector_columns[name]) if vector_columns_ok[name] else None
            inputs[name] = ResolvedInput(selection=selection, columns=columns, predicate=predicate)

    return ExecutionPlan(
        inputs=inputs,
        ops=ops,
        frame_types=frame_types,
        band_wide_join_scans=dict(band_wide_join_origin_groups),
    )
