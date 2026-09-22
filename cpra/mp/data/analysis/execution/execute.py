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
every combination the request layer will type successfully. `NOT_YET_IMPLEMENTED` below is
the missing other half -- a named, deliberate record of exactly which combinations this
walk does not execute -- so `tests/api/test_execute.py` can assert, in both directions,
that the registered lowerings and the not-yet-implemented set together partition exactly
what the tables declare legal: no orphan executor for a combination no table names, and no
combination falling through neither set. Every combination the tables declare legal now has
a registered lowering, so `NOT_YET_IMPLEMENTED` is empty; it stays here, rather than being
deleted, so the next lowering table that grows a new combination has somewhere to name the
gap instead of leaving it un-wired by omission.

`calculate` and `group` have no lowering table of their own -- `request/ops.py` types both
as unconditionally shape-preserving, in code rather than as data -- so their entries below
are declared directly against `Shape`'s two members. There is no table for them to drift
from.

**`project_raster_to_vector`'s registered lowering can still raise
`LoweringNotImplementedError`, for one narrower case `NOT_YET_IMPLEMENTED` cannot express.**
`project_direction` keys off the incoming frame's DECLARED grid, not its current actual
shape, so a raster-origin frame an earlier spatial filter already turned long dispatches to
the same `("raster", "vector")` entry a genuinely band-wide raster origin does. The two
cases need different data -- the ordinary one needs an already-open band-wide frame to
zonal-aggregate, the post-spatial-filter one has already been point-sampled into an
ordinary long frame with no band-wide frame left to give it -- so `_run_project_raster_to_
vector` below is a GUARD (not a signature adapter -- see `project.py`'s own module
docstring) that checks the incoming frame's actual shape itself and raises for the case it
cannot serve, before calling `project.project_raster_to_vector` with the same
`(frames, op, frame_types)` every other entry receives. See `tests/api/test_execute.py::
TestLoweringNotImplemented` for the concrete pipeline this covers.

**Opening a band-wide input.** A raster-shaped `ResolvedInput` is opened via
`io.raster.scan_raster` over its already-resolved `band_metadata`, except when a dimension
predicate pushed down at compile time narrowed that metadata to zero bands: `scan_raster`
raises on an empty band list (it derives its window from the VRT it opens, and a zero-band
read opens nothing), so that case is built directly with `execution.frame.empty_band_wide`
over a window taken from the raster grid's own geometry instead -- see
`_grid_read_window` and `execution.plan`'s module docstring, "loose end" paragraph, which
names this exact construction.

**A point sample reads a narrower window when the spatial filter says exactly which
pixels it needs -- and narrowing the window never changes what a `grid_cell` id means.**
`ResolvedInput.point_sample_step` names the step number of the first spatial `filter`
reached while an origin was still band-wide (`execution.plan`'s own module docstring);
before the ordinary input-opening loop runs, `execute` looks that step up in `plan.ops`
and tries `_derive_point_sample_window` against its `where`. When that `where` is simple
enough to name the exact absolute pixel ids before any file is opened (`grid_cell = N` or
`grid_cell IN (...)`, unquoted integers -- `_parse_point_sample_ids`), the origin is
opened over the SMALLEST window covering those pixels instead of its full extent. `where`
itself is never touched: `band_ops.sample_grid_cells_band_wide` always emits `grid_cell`
as the pixel's ABSOLUTE row-major position over the origin's full grid
(`execution.frame.row_major_grid_cell`'s `grid_width=` mode -- see `_run_filter_band_wide`
for where that width comes from), regardless of how much of the grid the read actually
covered, so the literal a caller wrote keeps naming the same pixel whether or not this
narrowing found anywhere to apply. Window narrowing is therefore a pure read-size
optimization here, not something a caller can observe in the result. Anything the `where`
cannot resolve into exact ids up front -- a negation, an `OR`, more than one spatial
conjunct, or a pixel outside the origin's own extent -- falls back to the full-window read
this walk has always done, which is always correct, merely more expensive than a window
`_derive_point_sample_window` could have derived. A tile-scoped origin (`resolved.
read_window is not None`) is never narrowed this way at all -- the tile's own window
already governs what gets read, and combining the two is a case no caller in this
codebase produces today.

**The raster-to-vector crosswalk is attached HERE, at open time, not inside
`project.project_raster_to_vector`.** `ResolvedInput.straddling_vector_grid` names the
destination vector grid a raster origin's pixels get redistributed onto, if any (see
`execution.plan`'s own module docstring); `_open_band_wide_input` resolves that origin's
crosswalk artifact and hands it to `scan_raster` as one more `crosswalks=` entry on the
SAME read the data bands come from, under `project.ZONE_COLUMN`. That is what lets
`project.project_raster_to_vector` read the crosswalk off the identical, already-processed
`lf` every other band-wide lowering between the origin and the `project` step has been
reading and writing, rather than re-scanning raw pixels from disk on its own and silently
discarding whatever an upstream `filter`/`calculate` did (`project.py`'s module docstring
gives the measured cost of combining two independently-chunked reads positionally, which
applies here just the same, plus the correctness cost of a second, independent read simply
not reflecting upstream work at all). `extra_columns`, threaded through every band-wide
lowering below, is what carries `ZONE_COLUMN` intact across that upstream work: see
`execution.frame.validate_extra_columns`'s own docstring for the contract these lowerings
already honor, and `_bind_band_wide_lowerings` below for how this walk supplies it without
turning the registry itself into a per-op branch. A `filter` whose predicate is spatial is
the one band-wide lowering that turns the frame LONG mid-walk (`_run_filter_band_wide`,
via `band_ops.sample_grid_cells_band_wide`); `ZONE_COLUMN` is stripped there, immediately,
rather than allowed to ride into a long frame as an unexplained extra dimension column a
later `group`/`join` could silently key on. Whatever, if anything, is still attached to
`extra_columns[last_output]` when the walk ends is stripped once more, defensively, before
`execute` returns -- a raster origin reachable from a `project` by one branch of a fan-out
can still be the frame ANOTHER branch's ops leave as the pipeline's actual last step (see
`tests/api/test_execution_plan.py::test_the_project_is_still_detected_behind_a_fan_out` for
a fan-out shape that is legal and already tested), and that other branch's own output
should never surface an internal-only crosswalk column just because the origin it started
from also happened to feed a `project` elsewhere.

**A band-wide `join`'s operands need ONE shared scan, not a scan apiece.**
`execution.band_ops.join_band_wide` aligns its operands positionally, which is only correct
if they share one chunking as well as one pixel grid (`execution.frame.
assert_same_pixel_grid` checks `batch_layout` for exactly this reason). Two separate
`scan_raster` calls over the same window can still chunk differently -- chunking depends on
band count, not only on the ground covered -- so `plan.band_wide_join_scans` names, per
band-wide join output, the raster origins that must come from one combined read.
`_open_band_wide_join_group` below performs that one scan over the concatenated band
metadata and slices the result back into one per-origin frame, before the ordinary
input-opening loop runs -- which is then skipped for any origin this prelude already
opened.

**An operand that entered the join as a VECTOR belongs to that same shared scan.** A
vector-to-raster `project` gets its pixel order from the crosswalk raster it broadcasts
across, so reading that crosswalk on its own puts the projected operand in a different
order from the data bands it is about to be aligned with -- the identical failure, for the
identical reason, as reading a raster-to-vector crosswalk separately. `plan.
band_wide_join_zone_crosswalks` names, per band-wide join output, one `plan.ZoneCrosswalk`
per such `project` upstream of any of that join's operands; `_open_band_wide_join_group`
attaches each to the SAME `scan_raster` call (one more `crosswalks=` entry, under its own
`f"__zone_step_{n}__"` column so two artifacts never collide) and hands the resulting
column back as a `project.ZoneSource` keyed by that project's step number.
`_bind_band_wide_lowerings` routes it to the one lowering that asked for it, exactly as it
routes `extra_columns`, leaving the walk's uniform call signature alone. A `project` with
no such entry -- nothing downstream joins its output against a separately-read raster --
reads the crosswalk itself, which is correct precisely because there is no other pixel
order for it to have to match.

The one shape still left out is a band-wide join with no raster origin at ALL (every
operand a separately-projected vector input): there are no data bands to build a shared
read around, and `scan_raster` has no zero-band answer to give, so `plan.
band_wide_join_scans` skips it and those operands are read however the walk produced them,
relying on `join_band_wide`'s own precondition to fail loudly if their chunkings do not
coincide.

Different operands in one shared join group can still straddle DIFFERENT destination
grids (one origin's `straddling_vector_grid` naming grid X, another's naming grid Y --
`plan.py`'s conflict check only rejects the SAME origin straddling two grids, never two
different origins straddling two different ones), so `_open_band_wide_join_group` cannot
attach a single crosswalk to the combined read the way `_open_band_wide_input` attaches
one to a single origin's read. It resolves one crosswalk per origin that needs one, each
under its own internal VRT column name (so two different crosswalk files never collide in
the one combined `scan_raster` call), and re-aliases each origin's own slice of the result
back to the canonical `project.ZONE_COLUMN` name -- so a lowering reading any one origin's
`BandWideFrame` out of `frames` afterward sees the same column name regardless of how many
crosswalks the underlying shared scan actually combined.
"""

import re

import polars as pl
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from cpra.mp.data.analysis.execution import band_ops, long_ops, project
from cpra.mp.data.analysis.execution.frame import (
    DOMAIN_COLUMN,
    BandWideFrame,
    LongFrame,
    ReadWindow,
    empty_band_wide,
    has_no_bands,
    released_band_metadata,
)
from cpra.mp.data.analysis.execution.plan import ExecutionPlan, ResolvedInput, ZoneCrosswalk, compile_plan
from cpra.mp.data.analysis.io.crosswalk_read import resolve_raster_crosswalk_path
from cpra.mp.data.analysis.io.geography import relabel_raster_grid_cell
from cpra.mp.data.analysis.io.raster import scan_raster
from cpra.mp.data.analysis.io.vector import scan_vector
from cpra.mp.data.analysis.request.expressions import PredicateKind, classify_where_kinds
from cpra.mp.data.analysis.request.ops import (
    FilterOp,
    FrameType,
    JoinOp,
    Op,
    ProjectOp,
    Shape,
    project_direction,
)
from cpra.mp.data.grids import GRIDS

__all__ = ["LoweringNotImplementedError", "NOT_YET_IMPLEMENTED", "execute", "run_to_path"]


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


# The named, deliberate not-yet-implemented set -- see the module docstring. Every
# combination each lowering table declares legal has a registered executor now, so every
# value here is empty; the keys stay present (rather than the dict being deleted) because
# `tests/api/test_execute.py::TestLoweringTableAgreement` reads this set unconditionally
# for all five ops, and because it is where the next lowering table to grow a combination
# this walk cannot yet run should name the gap.
NOT_YET_IMPLEMENTED: dict[str, frozenset] = {
    "filter": frozenset(),
    "calculate": frozenset(),
    "group": frozenset(),
    "join": frozenset(),
    "project": frozenset(),
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


def _run_filter_band_wide(
    frames: dict[str, BandWideFrame],
    op,
    frame_types,
    extra_columns: tuple[str, ...],
) -> BandWideFrame | LongFrame:
    """`extra_columns` is whatever non-band columns (a crosswalk zone id, notably --
    `ZONE_COLUMN`; see the module docstring) `op.input`'s own frame is carrying, supplied
    by `_bind_band_wide_lowerings` below rather than hardcoded, so this lowering can pass
    it straight through to `band_ops.filter_band_wide` (`execution.frame.
    validate_extra_columns`'s contract: every band-wide lowering names, explicitly, the
    non-band columns it should carry alongside the band columns it rewrites).

    A spatial predicate turns the result LONG (`band_ops.filter_band_wide`'s own dispatch,
    via `sample_grid_cells_band_wide`) -- and that lowering's own `extra_columns` handling
    puts any of them straight into the long output as ordinary columns
    (`sample_grid_cells_band_wide`'s unpivot `index=`). A crosswalk zone id has no meaning
    as a dimension column on a long frame (nothing downstream declared it, and a later
    `group`/`join` would silently key on it as if it were a real dimension), so it is
    stripped here, immediately, the one place this walk can tell the shape actually
    changed (`frame_types[op.output].shape` -- known statically, from the plan, not by
    inspecting the runtime result).

    `grid_width` -- `op.input`'s own grid's full pixel width (`GRIDS[...].width`) -- is
    passed to `band_ops.filter_band_wide` unconditionally, whether or not `op.where`
    actually turns out to have a spatial conjunct: this lowering has no cheaper way to know
    which case it is in than `filter_band_wide` itself already does (`classify_where_kinds`),
    and a `grid_width` `filter_band_wide` never needed costs nothing to have computed.
    `frame_types[op.input].grid` is the frame's CURRENTLY-addressed grid (`request.ops.
    FrameType`'s own docstring), which is what a still-band-wide frame's spatial conjunct is
    always written against.
    """
    grid_width = GRIDS[frame_types[op.input].grid].width
    result = band_ops.filter_band_wide(
        frames[op.input],
        op.where,
        extra_columns=extra_columns,
        grid_width=grid_width,
        value_columns=sorted(frame_types[op.input].value_columns),
    )
    if extra_columns and frame_types[op.output].shape is Shape.long:
        return result.drop(list(extra_columns))
    return result


def _run_calculate_band_wide(
    frames: dict[str, BandWideFrame], op, frame_types, extra_columns: tuple[str, ...]
) -> BandWideFrame:
    return band_ops.calculate_band_wide(frames[op.input], op, extra_columns=extra_columns)


def _run_group_band_wide(
    frames: dict[str, BandWideFrame], op, frame_types, extra_columns: tuple[str, ...]
) -> BandWideFrame:
    return band_ops.group_band_wide(frames[op.input], op, extra_columns=extra_columns)


def _run_join_band_wide(
    frames: dict[str, BandWideFrame], op, frame_types, extra_columns: tuple[str, ...]
) -> BandWideFrame:
    # Each operand is already sitting in `frames` under its own step name by the time the
    # walk reaches a `join` -- either opened directly, or opened as a shared scan by
    # `_open_band_wide_join_group` in `execute`'s prelude (see the module docstring). This
    # entry does not distinguish the two cases; `band_ops.join_band_wide`'s own
    # `assert_same_pixel_grid` precondition is what actually enforces that the operands
    # came from one shared read where that guarantee matters. `extra_columns` here is the
    # UNION over the operands (`_output_extra_columns`), matching `join_band_wide`'s own
    # documented "each column taken from the first operand that carries it" contract:
    # a vector-to-raster `project` operand brings `DOMAIN_COLUMN` that no other operand
    # has, and reading `op.inputs[0]`'s alone would drop it.
    return band_ops.join_band_wide(
        [frames[name] for name in op.inputs], list(op.inputs), extra_columns=extra_columns
    )


def _run_project_raster_to_vector(frames: dict[str, BandWideFrame], op, frame_types) -> LongFrame:
    """The registry entry for `("project", ("raster", "vector"))` -- a GUARD, not a
    signature adapter (see `project.py`'s own module docstring): it calls
    `project.project_raster_to_vector` with the identical `(frames, op, frame_types)`
    every other entry receives, unmodified.

    What it exists for is the one dispatch-key ambiguity `NOT_YET_IMPLEMENTED` cannot
    express (see the module docstring): `project_direction` keys off `op.input`'s
    DECLARED grid, so a raster-origin frame an earlier spatial filter already turned long
    reaches this same entry -- but `frames[op.input]` for that frame is an ordinary
    `LongFrame`, not the `BandWideFrame` triple `project.project_raster_to_vector` unpacks.
    Checked here, structurally, via `frame_types[op.input].shape`, and raised for
    explicitly, rather than letting that unpacking fail with a raw Python `TypeError` that
    gives no hint which pipeline step or which ambiguity caused it.
    """
    if frame_types[op.input].shape is not Shape.band_wide:
        raise LoweringNotImplementedError(
            f"no 'project' lowering for step {op.step_no} -- {op.input!r} is a "
            "raster-origin frame an earlier spatial filter already turned long (one row "
            "per sampled pixel), and project_raster_to_vector needs a band-wide frame to "
            "zonal-aggregate, which that frame no longer is"
        )
    return project.project_raster_to_vector(frames, op, frame_types)


# The registry the walk dispatches through. One uniform CALL signature per entry --
# `(frames, op, frame_types) -> LongFrame | BandWideFrame` -- so the loop below never has
# to know which op it is running. A filter's three predicate kinds share one entry per
# shape because all three lower to one function each (an ordinary row drop for long; a
# dispatch across the three predicate-kind lowerings, one of which can itself change shape,
# for band-wide -- see `band_ops.filter_band_wide`'s own docstring).
#
# `frame_types` is the plan's whole name -> `FrameType` map rather than just the incoming
# operand's, so that the signature stays the same for `join` (several operands) as for the
# rest. `_run_calculate`/`_run_calculate_band_wide` read it for the incoming frame's value
# columns (long) or to know which bands a `where` applies to (band-wide, via the frame
# itself rather than `frame_types`); the two `project` entries read it for the incoming
# frame's CURRENT grid (`current.grid`, not the pipeline input's own declared grid --
# `request.ops.FrameType`'s own docstring is where that distinction is spelled out), which
# is the other half of which crosswalk to open. Handing `frame_types` to every entry is
# what keeps this a registry rather than differently-shaped callables the loop would have
# to tell apart.
#
# The four `Shape.band_wide` non-project entries below are placeholders in THIS module-
# level dict -- `execute` never calls them directly. They also need `extra_columns` (the
# non-band columns, a crosswalk zone id notably, `op.input`'s frame is carrying; see the
# module docstring), which is per-EXECUTION state, not something a module-level constant
# dict can hold. `_bind_band_wide_lowerings` below builds the real, per-`execute()`-call
# registry by overriding just these four entries with closures bound over that call's own
# `extra_columns` dict. It overrides the vector-to-raster `project` entry too, over a
# second piece of the same per-execution state (that call's crosswalk reads, keyed by step
# number -- see that function), which is why the value registered here for it is likewise a
# placeholder. `tests/api/test_execute.py::TestLoweringTableAgreement` reads this dict's
# KEYS only, never calls its values, so the placeholder arity mismatch here is never itself
# exercised.
_LOWERINGS: dict[tuple[str, object], object] = {
    ("filter", Shape.long): _run_filter,
    ("filter", Shape.band_wide): _run_filter_band_wide,
    ("calculate", Shape.long): _run_calculate,
    ("calculate", Shape.band_wide): _run_calculate_band_wide,
    ("group", Shape.long): _run_group,
    ("group", Shape.band_wide): _run_group_band_wide,
    ("join", Shape.long): _run_join,
    ("join", Shape.band_wide): _run_join_band_wide,
    ("project", ("vector", "vector")): _run_project,
    ("project", ("raster", "vector")): _run_project_raster_to_vector,
    ("project", ("vector", "raster")): project.project_vector_to_raster,
}


def _bind_band_wide_lowerings(
    extra_columns: dict[str, tuple[str, ...]], zone_sources: dict[int, project.ZoneSource]
) -> dict[tuple[str, object], object]:
    """The real registry for one `execute()` call: `_LOWERINGS` with its four
    `Shape.band_wide` non-project entries overridden by closures bound over this call's own
    `extra_columns` -- see `_LOWERINGS`'s own comment for why they cannot simply be module-
    level constants the way every other entry is.

    Each closure reads `extra_columns.get(op.input, ())` (or, for `join`,
    `op.inputs[0]`'s -- `_operand_names(op)[0]` gives the right name either way) rather
    than being handed it as a call argument, so the CALL signature every entry in the
    returned dict is invoked with stays the uniform `(frames, op, frame_types)` the walk
    loop already uses for everything else.

    The vector-to-raster `project` entry is bound the same way and for the same reason,
    over `zone_sources` -- `_open_band_wide_join_group`'s crosswalk reads, keyed by the
    project step number that asked for one. `.get(op.step_no)` returns `None` for a step
    whose output is never joined against a separately-read raster, which is exactly the
    argument that tells `project.project_vector_to_raster` to read the crosswalk itself.
    """
    lowerings = dict(_LOWERINGS)
    lowerings[("project", ("vector", "raster"))] = (
        lambda frames, op, frame_types: project.project_vector_to_raster(
            frames, op, frame_types, zone_sources.get(op.step_no)
        )
    )
    lowerings[("filter", Shape.band_wide)] = lambda frames, op, frame_types: _run_filter_band_wide(
        frames, op, frame_types, extra_columns.get(op.input, ())
    )
    lowerings[("calculate", Shape.band_wide)] = lambda frames, op, frame_types: _run_calculate_band_wide(
        frames, op, frame_types, extra_columns.get(op.input, ())
    )
    lowerings[("group", Shape.band_wide)] = lambda frames, op, frame_types: _run_group_band_wide(
        frames, op, frame_types, extra_columns.get(op.input, ())
    )
    lowerings[("join", Shape.band_wide)] = lambda frames, op, frame_types: _run_join_band_wide(
        frames, op, frame_types, _output_extra_columns(op, frame_types, extra_columns)
    )
    return lowerings


def _operand_names(op: Op) -> list[str]:
    """The names of the frames `op` reads. `join` is the one op with several operands (and
    the one with no implicit "chain from the previous step" default), which is why this is
    a list rather than a single name.
    """
    return list(op.inputs) if isinstance(op, JoinOp) else [op.input]


def _output_extra_columns(
    op: Op, frame_types: dict[str, FrameType], extra_columns: dict[str, tuple[str, ...]]
) -> tuple[str, ...]:
    """The carried non-band columns `op`'s band-wide output has, given what its operands
    had. Three cases, in the order they are checked:

    A vector-to-raster `project` INTRODUCES one: `project.project_vector_to_raster`
    attaches `DOMAIN_COLUMN` to everything it returns (see that function, and the
    constant's own comment). Its input is a long frame that carried nothing, so this is
    the whole of its output's set -- and it is not inherited from the operand, which is
    why the plain `.get(operand)` below cannot produce it.

    A `join` UNIONS them: an operand can carry a column the others never had, which is
    exactly what the case above creates, and `band_ops.join_band_wide` is written to take
    each named column from the first operand that has it. Order follows `op.inputs` so the
    result is deterministic.

    Everything else INHERITS its first operand's, unchanged -- a `filter`/`calculate`/
    `group` passes its carried columns through, and none of them invents one.
    """
    if isinstance(op, ProjectOp) and project_direction(frame_types[op.input], op.to) == ("vector", "raster"):
        return (DOMAIN_COLUMN,)
    operands = _operand_names(op)
    if isinstance(op, JoinOp):
        return tuple(dict.fromkeys(column for name in operands for column in extra_columns.get(name, ())))
    return extra_columns.get(operands[0], ())


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


def _lowering_for(kind: str, key: object, registry: dict[tuple[str, object], object]):
    """Look up `kind`'s lowering for `key` in `registry`, raising
    `LoweringNotImplementedError` if `key` is in `NOT_YET_IMPLEMENTED[kind]` -- the one
    place that check happens, so every op shares an identical failure path instead of
    repeating it.

    `registry` is a parameter (rather than reading module-level `_LOWERINGS` directly) so
    the same lookup serves both the static table (whose keys `NOT_YET_IMPLEMENTED` and the
    lowering tables must agree with -- see `tests/api/test_execute.py::
    TestLoweringTableAgreement`) and `execute`'s own per-call registry from
    `_bind_band_wide_lowerings`, which shares every key with `_LOWERINGS` and only differs
    in four of the values.
    """
    if key in NOT_YET_IMPLEMENTED[kind]:
        raise LoweringNotImplementedError(
            f"no {kind!r} lowering yet for {key!r} -- this pipeline's shape typing already "
            "accepted it, the walk just cannot run it yet"
        )
    return registry[(kind, key)]


# The two `where` shapes `request.data_query.grid_cell_conjunct` (and any other caller
# resolving a coordinate to a literal id list before compiling a pipeline) ever produces
# for a raster spatial filter -- unquoted integers only, matching the convention every
# ABSOLUTE `grid_cell` id in this codebase already uses before any window has narrowed it.
# See `_parse_point_sample_ids` for what happens when a `where` does not match either one.
_POINT_SAMPLE_EQ_RE = re.compile(r"^grid_cell\s*=\s*(\d+)$", re.IGNORECASE)
_POINT_SAMPLE_IN_RE = re.compile(r"^grid_cell\s+in\s*\(\s*(\d+(?:\s*,\s*\d+)*)\s*\)$", re.IGNORECASE)


def _parse_point_sample_ids(where: str) -> list[int] | None:
    """Pull literal ABSOLUTE `grid_cell` ids out of `where`'s spatial conjunct, if `where`
    is simple enough to say exactly which pixels a point sample needs BEFORE any file is
    opened: exactly one top-level spatial conjunct (`request.expressions.
    classify_where_kinds`), and that one conjunct is exactly `grid_cell = N` or `grid_cell
    IN (N1, N2, ...)` with unquoted integer literals -- the two forms `request.data_query.
    grid_cell_conjunct` produces for a raster point query.

    Returns `None` -- not an error -- for anything else: two or more top-level spatial
    conjuncts, or one spatial conjunct that isn't exactly one of the two recognized forms
    (`grid_cell != 5`, `grid_cell = 5 OR grid_cell = 6`, a comparison this module was never
    taught to read). `None` means "cannot narrow the window with confidence", not "no
    pixels match" -- the caller (`_derive_point_sample_window`) falls back to reading the
    origin's full extent, which is always a CORRECT answer for any `where` this function
    cannot parse, only a more expensive one than a window this function could have derived.
    """
    kinds = classify_where_kinds(where)
    spatial = kinds[PredicateKind.spatial]
    if len(spatial) != 1:
        return None
    conjunct = spatial[0].strip()
    match = _POINT_SAMPLE_EQ_RE.match(conjunct)
    if match:
        return [int(match.group(1))]
    match = _POINT_SAMPLE_IN_RE.match(conjunct)
    if match:
        return [int(token) for token in match.group(1).split(",")]
    return None


def _derive_point_sample_window(grid_name: str, where: str) -> Window | None:
    """Size a point sample's own narrower read window from a spatial `filter`'s `where` --
    still written in `grid_name`'s full-grid, absolute-id convention, since every producer
    of a spatial conjunct reaching a still-band-wide raster origin uses that convention
    (`execution.plan`'s module docstring; no narrowing has happened yet at the point this
    `where` was written), and the convention `band_ops.sample_grid_cells_band_wide` emits
    its own `grid_cell` output in too -- see the module docstring for why that makes
    narrowing the window here a pure read-size optimization rather than something able to
    change what `where`'s literals mean.

    Returns `None` -- caller falls back to a full-window read -- when `_parse_point_sample_
    ids` cannot resolve `where` into exact ids, or when any resolved id falls outside
    `grid_name`'s own `(height, width)` extent (an out-of-range literal a caller supplied
    directly, with no coordinate resolution step in between to have already rejected it).

    `where` itself is never touched here or by any caller of this function: only the
    window is derived. The smallest window covering every resolved pixel, from the same
    `divmod(id, grid.width)` decode this module already uses to validate an id is in range.
    """
    ids = _parse_point_sample_ids(where)
    if ids is None:
        return None

    grid = GRIDS[grid_name]
    rows: list[int] = []
    cols: list[int] = []
    for cell_id in ids:
        row, col = divmod(cell_id, grid.width)
        if not (0 <= row < grid.height) or not (0 <= col < grid.width):
            return None
        rows.append(row)
        cols.append(col)

    row_off, col_off = min(rows), min(cols)
    height = max(rows) - row_off + 1
    width = max(cols) - col_off + 1
    return Window(col_off, row_off, width, height)


def _point_sample_overrides(plan: ExecutionPlan) -> dict[str, Window]:
    """For every raster origin in `plan.inputs` whose `point_sample_step` resolved to a
    narrower window, the window to open that origin's read over instead of its full
    extent -- see `_derive_point_sample_window` for how, and `execute`'s own docstring for
    how the result gets used.

    Deliberately skips an origin under either of two conditions, in both cases leaving
    that origin to the full-window read this walk has always done: `resolved.read_window
    is not None` (a tile-scoped origin -- narrowing further on top of a tile window is a
    combination no caller in this codebase produces, see the module docstring), or
    `has_no_bands(resolved.band_metadata)` (a zero-band origin already takes the
    grid-derived empty-frame path in `_open_band_wide_input`, which has no real read for a
    narrower window to change anything about). `resolved.tile_disjoint` origins are
    likewise skipped for the same "no real read happens" reason.

    Looks the step up in `plan.ops` by `step_no` rather than assuming position, since a
    `point_sample_step` is a step NUMBER (declared order), not necessarily that op's index
    in `plan.ops` (an op whose predicate does not survive to any origin's `ResolvedInput`
    can still occupy an earlier position in the list). `isinstance(op, FilterOp)` is
    asserted implicitly by construction: `plan.py` only ever records `point_sample_step`
    from a `FilterOp`'s own `step_no` (see `execution.plan`'s module docstring), so a step
    number recorded there always resolves back to a `FilterOp` here.
    """
    windows: dict[str, Window] = {}
    ops_by_step = {op.step_no: op for op in plan.ops}
    for name, resolved in plan.inputs.items():
        if resolved.point_sample_step is None:
            continue
        if resolved.read_window is not None or resolved.tile_disjoint:
            continue
        if has_no_bands(resolved.band_metadata):
            continue
        op = ops_by_step[resolved.point_sample_step]
        assert isinstance(op, FilterOp)
        window = _derive_point_sample_window(resolved.selection.grid, op.where)
        if window is not None:
            windows[name] = window
    return windows


def _grid_read_window(grid_name: str, window=None, decimation: tuple[int, int] = (1, 1)) -> ReadWindow:
    """A `ReadWindow` built from `grid_name`'s own geometry, with no file
    opened -- the window a zero-band raster origin gets, since `io.raster.scan_raster`
    cannot produce one itself (it raises on empty metadata; see the module docstring).
    `cpra.mp.data.grids.RasterGrid` already carries everything a full window needs
    (`bounds`, `height`, `width`, `crs`); the transform follows from
    `rasterio.transform.from_bounds`, exactly as `io.raster.scan_raster` derives it for a
    real, full-extent, undecimated read of the same grid (confirmed byte-identical against
    a real scan while building this). `source=()` names no file, matching the fact that
    none was opened; `batch_layout=()` is legitimately empty, meaning "the whole window,
    one batch" -- the same reading `execution.frame.row_major_grid_cell` already gives an
    empty `batch_layout` for a hand-built window.

    `window` (a `rasterio.windows.Window`, from `plan.ResolvedInput.read_window`) narrows
    it to that sub-window, which is what a zero-band origin under a tile-scoped plan needs:
    a `join` compares its operands' windows field by field (`frame.assert_same_pixel_grid`)
    and would reject a zero-band operand still describing the whole grid beside a real one
    describing the tile's window. `decimation` is carried for the same reason and behaves
    the same way: it is one more field that comparison reads, so a zero-band operand must
    claim whatever resolution its sibling was read at even though it has no rows to read.
    """
    grid = GRIDS[grid_name]
    grid_transform = from_bounds(*grid.bounds, width=grid.width, height=grid.height)
    if window is None:
        col_off, row_off, width, height = 0, 0, grid.width, grid.height
    else:
        col_off, row_off = int(window.col_off), int(window.row_off)
        width, height = int(window.width), int(window.height)
    return ReadWindow(
        source=(),
        col_off=col_off,
        row_off=row_off,
        width=width,
        height=height,
        transform=window_transform(Window(col_off, row_off, width, height), grid_transform),
        crs=CRS.from_user_input(grid.crs),
        decimation=decimation,
        batch_layout=(),
    )


def _open_band_wide_input(
    resolved: ResolvedInput, *, point_sample_window: Window | None = None
) -> tuple[BandWideFrame, tuple[str, ...]]:
    """Open one raster-shaped `ResolvedInput`, returning its frame alongside whatever
    `extra_columns` that frame's `lf` now carries beyond its `b<i>` band columns.

    Three cases: the grid-derived empty frame when dimension pushdown already narrowed the
    input to zero bands (see `_grid_read_window`) -- `extra_columns=()`, since there are no
    pixels to attach a crosswalk to and `project.project_raster_to_vector`'s own
    `has_no_bands` guard never looks for one; an ordinary `scan_raster` over the resolved
    band list, `extra_columns=()`, when `straddling_vector_grid` is `None` (no `project`
    reachable from this origin needs a crosswalk at all); and, when it is not `None`, that
    same `scan_raster` call with the resolved raster-to-vector crosswalk attached via
    `crosswalks=`, under `project.ZONE_COLUMN`, `extra_columns=(project.ZONE_COLUMN,)` --
    see the module docstring's "the raster-to-vector crosswalk is attached here" note for
    why this is where that attachment belongs.

    `resolved.read_window` rides along on every one of those `scan_raster` calls: `None`
    for an ordinary request (read the origin whole), or the window `compile_plan` sized for
    a tile -- UNLESS `point_sample_window` is given, which then takes its place. `execute`
    passes one only for an origin whose `point_sample_step` resolved to a narrower window
    (`_derive_point_sample_window`); `resolved.read_window` itself is left untouched by that
    decision (it is still `None` for the ordinary, non-tile request every point sample so
    far has been), so this is a call-site override, not a second copy of the same field.
    `resolved.decimation` rides along beside it -- the resolution `compile_plan` decided
    this origin's window may be read at, `(1, 1)` unless it was both tile-scoped and
    overview-eligible.
    A `tile_disjoint` origin -- the tile's footprint misses this raster's extent
    entirely -- has no window to read at all and becomes the same zero-band empty frame the
    first case produces, so "this tile is outside the data" arrives at the sink as the
    no-bands result every other "nothing to show" already arrives as, rather than as a
    second, parallel signal the sink would have to check separately.

    Not used for an origin `execute`'s band-wide-join prelude already opened as part of a
    shared scan -- see `_open_band_wide_join_group`.
    """
    if resolved.tile_disjoint:
        return empty_band_wide(_grid_read_window(resolved.selection.grid)), ()
    if point_sample_window is not None:
        # A point sample overrides the window, so it must also override the resolution: the
        # pixel under a coordinate is the whole answer, and a decimated grid can hand back a
        # neighbour's value instead. `compile_plan` already sets `overview_permitted` (and
        # therefore `decimation`) accordingly for any origin with a `point_sample_step`;
        # this keeps the override a single decision rather than two that must agree.
        window, decimation = point_sample_window, (1, 1)
    else:
        window, decimation = resolved.read_window, resolved.decimation
    if has_no_bands(resolved.band_metadata):
        return empty_band_wide(_grid_read_window(resolved.selection.grid, window, decimation)), ()
    if resolved.straddling_vector_grid is not None:
        crosswalk_path = resolve_raster_crosswalk_path(resolved.selection.grid, resolved.straddling_vector_grid)
        opened = scan_raster(
            resolved.band_metadata,
            crosswalks=[(project.ZONE_COLUMN, str(crosswalk_path))],
            window=window,
            decimation=decimation,
        )
        return opened, (project.ZONE_COLUMN,)
    return scan_raster(resolved.band_metadata, window=window, decimation=decimation), ()


#: `_shared_tile_window`'s answer for a join group whose tile misses the data entirely --
#: distinct from `None`, which already means "no tile, read whole".
_DISJOINT = object()


def _shared_tile_window(origins: tuple[str, ...], inputs: dict[str, ResolvedInput]):
    """The ONE window a band-wide join group's shared scan reads: `None` (read whole) when
    the plan is not tile-scoped, `_DISJOINT` when the tile misses the data, else the
    smallest window covering every origin's own.

    A union rather than any one origin's window, because two operands of one join can want
    different windows off the same grid: only an origin feeding a raster-to-vector
    `project` gets the straddling-feature widening, so a join between one that does and one
    that does not produces two different windows, and the shared scan has to cover both.
    Reading the union is always safe -- a wider window is more pixels, never different
    ones -- and the operands stay positionally aligned precisely BECAUSE they share it.

    Disjointness is a property of the tile and the grid alone, and `request.ops`'
    `validate_shapes` already requires a band-wide join's operands to share a grid, so the
    origins of one group are either all disjoint or none are; a mixture would mean that
    same-grid rule had been broken upstream, and is rejected here rather than silently
    unioned into a window covering only the operands that happened to overlap.
    """
    disjoint = {inputs[name].tile_disjoint for name in origins}
    if disjoint == {True}:
        return _DISJOINT
    if len(disjoint) != 1:
        raise ValueError(
            f"band-wide join group {list(origins)!r} mixes tile-disjoint and overlapping "
            "origins, which operands sharing one grid cannot do -- the group's origins do "
            "not all sit on the same grid"
        )

    windows = [inputs[name].read_window for name in origins if inputs[name].read_window is not None]
    if not windows:
        return None
    col_off = min(int(w.col_off) for w in windows)
    row_off = min(int(w.row_off) for w in windows)
    col_end = max(int(w.col_off) + int(w.width) for w in windows)
    row_end = max(int(w.row_off) + int(w.height) for w in windows)
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def _shared_tile_decimation(origins: tuple[str, ...], inputs: dict[str, ResolvedInput]) -> tuple[int, int]:
    """The ONE resolution a band-wide join group's shared scan reads at: the per-axis
    MINIMUM of its origins' own factors.

    One scan can only be decimated one way, and `frame.assert_same_pixel_grid` compares
    `decimation` precisely because two frames read at different resolutions do not address
    the same pixels -- so the group needs a single answer rather than an answer per operand.
    The minimum is the conservative one in both directions that matter. It is `(1, 1)`
    whenever any one origin is not overview-eligible (`compile_plan` gives that origin
    `(1, 1)`, and no factor is smaller), which is what keeps a `project`'s full-resolution
    requirement from being undone by the operand it is joined against. And where every
    origin IS eligible, the minimum reads at least as many pixels as the coarsest operand
    would have alone -- which matters because the shared window is the UNION of theirs
    (`_shared_tile_window`), wider than any single one and therefore able to bear a factor
    at least as coarse as any of them. Recomputing the factors against that union would be
    the tighter answer; it would also mean carrying the `TileRequest` down here to re-derive
    something `compile_plan` has already decided, to save pixels on a read that is already
    bounded.
    """
    factors = [inputs[name].decimation for name in origins]
    return (min(row for row, _col in factors), min(col for _row, col in factors))


def _empty_zone_sources(
    zone_crosswalks: tuple[ZoneCrosswalk, ...], window: ReadWindow
) -> dict[int, project.ZoneSource]:
    """A zero-row `ZoneSource` per entry, over `window` -- what a band-wide join group that
    never got to scan anything (tile-disjoint, or every origin narrowed to zero bands) owes
    its vector-to-raster `project` steps.

    Zero rows because there are no pixels: the join partners those steps will be aligned
    against are `execution.frame.empty_band_wide` frames over this same window, and a left
    join from an empty zone column produces the matching zero-row broadcast. The schema is
    still declared (rather than a bare `pl.LazyFrame()`) so the broadcast's own join on
    `ZONE_COLUMN` resolves a real column instead of failing on a missing one; `Int32`
    matches what a crosswalk raster's zone ids read back as, and is only ever joined
    against an equally empty frame here.
    """
    return {
        zone_crosswalk.step_no: project.ZoneSource(
            pl.LazyFrame(schema={project.ZONE_COLUMN: pl.Int32}), window
        )
        for zone_crosswalk in zone_crosswalks
    }


def _open_band_wide_join_group(
    origins: tuple[str, ...],
    inputs: dict[str, ResolvedInput],
    zone_crosswalks: tuple[ZoneCrosswalk, ...] = (),
) -> tuple[dict[str, tuple[BandWideFrame, tuple[str, ...]]], dict[int, project.ZoneSource]]:
    """One shared `scan_raster` call over the UNION of `origins`' band metadata, sliced
    back into one `(BandWideFrame, extra_columns)` pair per origin -- what
    `plan.band_wide_join_scans` records a band-wide join's raster operands as needing (see
    the module docstring for why one scan apiece is not equivalent).

    `zone_crosswalks` is `plan.band_wide_join_zone_crosswalks`'s entry for the same join:
    one `ZoneCrosswalk` per vector-to-raster `project` upstream of any of that join's
    operands. Each rides on this same scan as one more `crosswalks=` entry, and comes back
    in the second return value as a `project.ZoneSource` keyed by the project's own step
    number -- which is how `_bind_band_wide_lowerings` gets it to exactly the lowering
    that needs it. A projected operand's pixel order is the raster operands' because it
    came out of their read, not because the two happened to chunk alike.

    Band order in the combined read is the concatenation order, so origin *k*'s bands are
    exactly the metadata rows (and the resulting `b<i>` columns) at the offset its
    predecessors' heights sum to -- sliced and renamed back to that origin's own local
    `b0..b{height-1}`, matching what a standalone `scan_raster` of just that origin's
    metadata would have named them.

    Different origins in the group can straddle DIFFERENT destination grids (only one
    origin straddling two grids at once is rejected, at compile time -- see `plan.py`'s own
    conflict check), so this cannot attach one shared crosswalk the way
    `_open_band_wide_input` does for a single origin. Instead it resolves one crosswalk per
    origin that needs one, each under its own internal VRT column name
    (`f"__zone_{name}__"`, guaranteed distinct across origins so two different crosswalk
    files never collide in the one combined `scan_raster` call), and re-aliases each
    origin's own slice back to the canonical `project.ZONE_COLUMN` name -- so a lowering
    reading any one origin's `BandWideFrame` out of `frames` afterward sees the same column
    name regardless of how many crosswalks the underlying shared scan actually combined.

    All-zero case: if every origin in the group resolved to zero bands, no scan can be
    performed (`scan_raster` raises on empty metadata) and none is needed -- one
    grid-derived empty window, shared across every origin in the group, exactly like
    `_open_band_wide_input`'s own zero-band case, `extra_columns=()` throughout for the
    same reason given there. A tile-disjoint group takes the same exit for the same reason,
    one step earlier: there are no pixels under the tile to scan even where bands exist.
    Both exits still return a `ZoneSource` per `zone_crosswalks` entry -- an empty one over
    that same shared window, which is what keeps the projected operand's frame aligned
    with the empty frames its join partners get. Letting the lowering fall back to its own
    full-extent read there would hand `join_band_wide` a real, fully-populated operand to
    align against empty ones, which is the same mismatch this function exists to prevent.

    Under a tile-scoped plan the shared scan reads ONE window covering every origin's own
    (`_shared_tile_window`) -- which is not necessarily any single origin's, since only an
    origin feeding a raster-to-vector `project` gets the straddling-feature widening -- at
    ONE resolution covering every origin's own (`_shared_tile_decimation`).

    A PARTIAL zero-band mismatch within an otherwise-nonzero group (one operand resolves
    to zero bands, another does not) raises `ValueError` naming the mismatched origins,
    rather than letting it through to become a silent row-count mismatch downstream: the
    zero-band operand's slice would otherwise become a zero-column selection over the
    shared frame, which collapses to height 0 (the same Polars "zero columns loses the row
    count" gap `execution/frame.py`'s module docstring describes), while its sibling
    operand keeps the scan's real pixel count -- `join_band_wide`'s own precondition would
    reject that mismatch too, but only after producing a confusing shape-mismatch error
    with no indication which operand was actually the empty one.
    """
    read_window = _shared_tile_window(origins, inputs)
    if read_window is _DISJOINT:
        window = _grid_read_window(inputs[origins[0]].selection.grid)
        return (
            {name: (empty_band_wide(window), ()) for name in origins},
            _empty_zone_sources(zone_crosswalks, window),
        )

    decimation = _shared_tile_decimation(origins, inputs) if read_window is not None else (1, 1)

    band_metadata = [inputs[name].band_metadata for name in origins]
    heights = [df.height for df in band_metadata]
    if sum(heights) == 0:
        window = _grid_read_window(inputs[origins[0]].selection.grid, read_window, decimation)
        return (
            {name: (empty_band_wide(window), ()) for name in origins},
            _empty_zone_sources(zone_crosswalks, window),
        )

    zero_names = [name for name, height in zip(origins, heights) if height == 0]
    if zero_names:
        raise ValueError(
            f"band-wide join group {list(origins)!r} has a partial zero-band mismatch: "
            f"{zero_names!r} resolved to zero bands while the rest did not -- "
            "join_band_wide requires every operand to share one row count, which a mix of "
            "zero-band and non-zero-band operands from one shared scan cannot give"
        )

    combined = pl.concat(band_metadata, how="diagonal_relaxed")

    crosswalks: list[tuple[str, str]] = []
    zone_columns: dict[str, str] = {}
    for name in origins:
        target = inputs[name].straddling_vector_grid
        if target is None:
            continue
        crosswalk_path = resolve_raster_crosswalk_path(inputs[name].selection.grid, target)
        column = f"__zone_{name}__"
        crosswalks.append((column, str(crosswalk_path)))
        zone_columns[name] = column

    # Keyed by step number rather than by origin: a vector-to-raster `project`'s crosswalk
    # belongs to the STEP, not to any raster origin in this group (the group's origins are
    # the join's other operands), and two projects in one pipeline can legitimately want
    # the same artifact under two separate columns.
    project_zone_columns: dict[int, str] = {}
    for zone_crosswalk in zone_crosswalks:
        crosswalk_path = resolve_raster_crosswalk_path(zone_crosswalk.raster_grid, zone_crosswalk.vector_grid)
        column = f"__zone_step_{zone_crosswalk.step_no}__"
        crosswalks.append((column, str(crosswalk_path)))
        project_zone_columns[zone_crosswalk.step_no] = column

    lf, out_metadata, window = scan_raster(
        combined, crosswalks=crosswalks, window=read_window, decimation=decimation
    )

    zone_sources = {
        step_no: project.ZoneSource(lf.select(pl.col(column).alias(project.ZONE_COLUMN)), window)
        for step_no, column in project_zone_columns.items()
    }

    frames: dict[str, tuple[BandWideFrame, tuple[str, ...]]] = {}
    offset = 0
    for name, height in zip(origins, heights):
        renamed = [pl.col(f"b{offset + i}").alias(f"b{i}") for i in range(height)]
        extra_columns: tuple[str, ...] = ()
        if name in zone_columns:
            renamed.append(pl.col(zone_columns[name]).alias(project.ZONE_COLUMN))
            extra_columns = (project.ZONE_COLUMN,)
        frames[name] = (
            (lf.select(renamed), out_metadata.slice(offset, height), window),
            extra_columns,
        )
        offset += height
    return frames, zone_sources


def execute(plan: ExecutionPlan, *, base_path=None) -> LongFrame | BandWideFrame:
    """Run `plan` and return its final frame, lazily -- collecting is left to the caller,
    matching every other lowering in this package staying lazy until something downstream
    needs the data.

    Opens every input up front: a long-shaped one via `io.vector.scan_vector`, with
    whatever pushdown the plan recorded for it; a band-wide one via `_open_band_wide_input`
    (or, for a raster origin `plan.band_wide_join_scans` names, via one shared
    `_open_band_wide_join_group` scan performed before the per-input loop, which then skips
    every origin that prelude already opened) -- see the module docstring for both. Each of
    those also reports the `extra_columns` (non-band columns, a crosswalk zone id notably)
    its frame now carries, tracked in a plain `dict[str, tuple[str, ...]]` for the rest of
    this call -- see `_bind_band_wide_lowerings` for how the walk below reads it.

    That same prelude collects the crosswalk reads any vector-to-raster `project` feeding
    one of those joins needs (`plan.band_wide_join_zone_crosswalks`), keyed by step number
    in `zone_sources` and routed to the lowering the same way -- see the module docstring's
    paragraph on why a projected operand belongs to the join's shared scan.

    Then walks `plan.ops` once, front to back, dispatching each through this call's own
    registry (`_LOWERINGS` with its band-wide entries bound to `extra_columns`, via
    `_bind_band_wide_lowerings`) on `(op kind, dispatch key)` -- reading the frame types the
    plan already recorded at compile time rather than re-deriving any of them here. After
    each op that leaves a band-wide result, `extra_columns[op.output]` is set to whatever
    `op`'s own first operand (`_operand_names(op)[0]`) carried -- the same "operands[0]
    alone" rule `join_band_wide` itself already documents, which happens to also be correct
    for the other three ops (each has exactly one operand). An op whose output is long
    (a vector-to-vector or raster-to-vector `project`, or a spatial `filter`) needs no entry
    here: nothing downstream ever reads a long name out of `extra_columns` (`.get(...,  ())`
    on a name that was never set already returns the right, empty answer), and a spatial
    `filter`'s own `extra_columns` are stripped from the frame itself before this walk even
    gets here (`_run_filter_band_wide`).

    Before returning, releases a band-wide result's metadata
    (`execution.frame.released_band_metadata`), which drops the reserved band-index column
    either of two lowerings can leave standing (`band_ops.group_band_wide`'s `by=[]` case,
    `project.project_vector_to_raster`'s zero-pivot-dimension case) -- except where it is
    the only column left, since dropping it there would take the band count with it and
    report an empty result for a real single-band one; see that function. Also drops whatever
    `extra_columns` the final band-wide output still carries, defensively -- see the module
    docstring's closing paragraph for why a fan-out can leave a raster origin's crosswalk
    column attached to a frame that never actually ran through the `project` that needed it.

    Returns the single input's frame for a zero-step pipeline: there is nothing to walk,
    and the request layer's own rule guarantees exactly one input in that case.

    Before any of that: `_point_sample_overrides` looks for a raster origin whose
    `point_sample_step` resolved to a window narrower than its full extent, per-origin --
    see that function and `_derive_point_sample_window` for how, and the module docstring's
    "a point sample reads a narrower window" paragraph for why. Every origin it names gets
    opened over that narrower window (`_open_band_wide_input`'s own `point_sample_window`
    override) instead of the full-extent read `resolved.read_window` alone would give.
    `plan.ops` itself is walked unchanged below -- narrowing the window never requires
    rewriting any op, since `band_ops.sample_grid_cells_band_wide` emits `grid_cell` as an
    ABSOLUTE id regardless of how much of the grid its read actually covered, so the same
    `where` a caller wrote against the full grid keeps meaning the same pixels whether or
    not this narrowing found anywhere to apply.
    """
    point_sample_windows = _point_sample_overrides(plan)

    extra_columns: dict[str, tuple[str, ...]] = {}

    band_wide_join_frames: dict[str, BandWideFrame] = {}
    zone_sources: dict[int, project.ZoneSource] = {}
    for output, origins in plan.band_wide_join_scans.items():
        opened, group_zone_sources = _open_band_wide_join_group(
            origins, plan.inputs, plan.band_wide_join_zone_crosswalks.get(output, ())
        )
        for name, (frame, extra) in opened.items():
            band_wide_join_frames[name] = frame
            extra_columns[name] = extra
        # `setdefault`, not assignment: one `project` step can feed two band-wide joins
        # over DIFFERENT raster origins (a fan-out), and it runs once, so only one of those
        # two reads can be the one its pixels actually come from. Keeping the first makes
        # which one deterministic; the second join then fails on `join_band_wide`'s own
        # `assert_same_pixel_grid` precondition, which is the correct outcome -- those two
        # joins genuinely cannot both align against a single projection.
        for step_no, zone_source in group_zone_sources.items():
            zone_sources.setdefault(step_no, zone_source)

    frames: dict[str, LongFrame | BandWideFrame] = {}
    for name, resolved in plan.inputs.items():
        if plan.frame_types[name].shape is Shape.long:
            frames[name] = scan_vector(
                resolved.selection,
                base_path=base_path,
                columns=resolved.columns,
                predicate=resolved.predicate,
            )
        elif name in band_wide_join_frames:
            frames[name] = band_wide_join_frames[name]
        else:
            frames[name], extra_columns[name] = _open_band_wide_input(
                resolved, point_sample_window=point_sample_windows.get(name)
            )

    lowerings = _bind_band_wide_lowerings(extra_columns, zone_sources)

    last_output: str | None = None
    for op in plan.ops:
        lowering = _lowering_for(op.op, _dispatch_key(op, plan.frame_types), lowerings)
        frames[op.output] = lowering(frames, op, plan.frame_types)
        if plan.frame_types[op.output].shape is Shape.band_wide:
            extra_columns[op.output] = _output_extra_columns(op, plan.frame_types, extra_columns)
        last_output = op.output

    if last_output is None:
        (last_output,) = frames.keys()

    result = frames[last_output]
    if plan.frame_types[last_output].shape is Shape.band_wide:
        lf, metadata_df, window = result
        stray = extra_columns.get(last_output, ())
        if stray:
            lf = lf.drop(list(stray))
        result = (lf, released_band_metadata(metadata_df), window)
    return result


def run_to_path(
    pipeline, out_path, *, base_path=None, dtype=None, nodata=None, relabel=None, **profile_options
) -> bool:
    """Compile, run, and sink `pipeline` to `out_path` in one call -- `compile_plan` then
    `execute` then whichever of the two "a whole result, at a path" sinks matches the
    result's own shape (`execution.sink.sink_tabular` for a long frame,
    `execution.sink.write_raster` for a band-wide one).

    Only those two sinks fit this shape of call. `png`/`jpg`/`pbf` each need per-request
    parameters (a tile's `x`/`y`/`z`, its pixel size, a legend) that a generic
    `(pipeline, out_path)` signature has nowhere to put, so a caller producing one of those
    calls `execution.sink.render_tile`/`execution.sink.sink_vector_tile` directly instead
    of through this function -- this function is the composition helper for the two output
    kinds that need nothing beyond the pipeline and a destination.

    `execution.sink`'s `output.sort`/`output.select` shaping is applied to a long result
    before it is written, matching every other tabular path (`apply_output_shaping`, run
    once, last).

    `relabel`, when given, is an `io.geography.resolve_raster_point_labels` map (absolute
    `grid_cell` id -> the caller's own coordinate-string label(s) for it) applied to a long
    result by `io.geography.relabel_raster_grid_cell` -- the last thing a raster POINT query
    needs before its output is written, and the reason that query's `grid_cell` column
    comes back as `"lng,lat"` strings rather than the integer pixel ids the filter itself
    ran against. Applied BEFORE `apply_output_shaping`, not after, so `output.sort`/
    `output.select` see the column a caller actually receives: `sort=grid_cell` orders the
    label strings (what the caller asked to sort by), and `select=grid_cell` keeps a column
    that still exists. Ignored for a band-wide result -- a raster point query is long by
    construction (a spatial `filter` unpivots band-wide to long; see `execution.band_ops.
    filter_band_wide`), so a band-wide result and a `relabel` map cannot both be present
    unless a caller built one for a pipeline that never point-sampled at all.

    Returns `False` when the underlying sink does (a band-wide result with no bands to
    write -- see `execution.sink.write_raster`); `True` otherwise. A long/tabular write
    always returns `True` -- an empty long frame is still a well-formed, zero-row CSV/
    NDJSON/Parquet file, unlike a band-wide result with literally no bands to declare a
    raster's `count` from, so there is no analogous "nothing to write" case for that
    branch to report.

    `dtype`/`nodata` are forwarded to `write_raster`, left at THIS function's own `None`
    default when the caller does not set them -- `nodata=None` is not forwarded as a
    literal `None` (which `write_raster` would take as "declare no nodata at all", a
    different, real profile choice), it is left OUT of the call entirely so
    `write_raster`'s own `DEFAULT_NODATA` default takes over. `nodata=0`, a legitimate
    declared value some datasets use, is forwarded exactly as given -- not treated as
    "unset" the way a bare `nodata or DEFAULT_NODATA` would wrongly do. Both are ignored
    entirely for a long/tabular result, the same way neither has meaning for a CSV row.

    `**profile_options` is forwarded whole to whichever sink the result's shape picks --
    `write_raster`'s rasterio-profile kwargs (`compress`, `zstd_level`, etc.) for a
    band-wide result, `sink.sink_tabular`'s polars writer kwargs (`compression`,
    `compression_level`) for a long one. Only one branch runs per call, so the same dict
    is never split or reinterpreted -- whichever caller built it already built it in the
    vocabulary the result's own shape needs.

    `execution.sink` is imported inside the function body rather than at this module's top
    level: `sink.py` imports `LoweringNotImplementedError` from here at ITS top level, so
    importing it back at the top would close that into a cycle at import time.
    """
    from cpra.mp.data.analysis.execution import sink

    plan = compile_plan(pipeline, base_path=base_path)
    result = execute(plan, base_path=base_path)

    if isinstance(result, tuple):
        write_kwargs = dict(profile_options)
        if dtype is not None:
            write_kwargs["dtype"] = dtype
        if nodata is not None:
            write_kwargs["nodata"] = nodata
        return sink.write_raster(result, out_path, **write_kwargs)

    if relabel is not None:
        result = relabel_raster_grid_cell(result, relabel)

    lf = sink.apply_output_shaping(result, pipeline.output)
    sink.sink_tabular(lf, out_path, pipeline.output.format, **profile_options)
    return True
