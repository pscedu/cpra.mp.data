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
names this exact construction. `ResolvedInput.point_sample_step` is read from the plan but
not acted on here: it marks where a later spatial filter first samples an origin, for a
point-sparse read this walk does not implement -- every band-wide input is opened as a full
window regardless, and the spatial filter itself (`execution.band_ops.
sample_grid_cells_band_wide`) is what turns the already-opened frame long.

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
opened. A join whose operands are NOT all raster origins (e.g. one side is the output of a
vector-to-raster `project`) is deliberately absent from `plan.band_wide_join_scans` (see
`execution.plan`'s module docstring) and gets no shared-scan treatment here either; its
operands are read however the rest of the walk already produced them, relying on
`join_band_wide`'s own precondition to fail loudly if their windows do not coincide.

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

import polars as pl
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from cpra.mp.data.analysis.execution import band_ops, long_ops, project
from cpra.mp.data.analysis.execution.frame import (
    BandWideFrame,
    LongFrame,
    ReadWindow,
    drop_reserved_band_index,
    empty_band_wide,
    has_no_bands,
)
from cpra.mp.data.analysis.execution.plan import ExecutionPlan, ResolvedInput
from cpra.mp.data.analysis.io.crosswalk_read import resolve_raster_crosswalk_path
from cpra.mp.data.analysis.io.raster import scan_raster
from cpra.mp.data.analysis.io.vector import scan_vector
from cpra.mp.data.analysis.request.ops import (
    FrameType,
    JoinOp,
    Op,
    ProjectOp,
    Shape,
    project_direction,
)
from cpra.mp.data.grids import GRIDS

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
    """
    result = band_ops.filter_band_wide(frames[op.input], op.where, extra_columns=extra_columns)
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
    # came from one shared read where that guarantee matters. `extra_columns` here is
    # already `op.inputs[0]`'s own -- `_bind_band_wide_lowerings` supplies exactly that,
    # matching `join_band_wide`'s own documented "reads `extra_columns` from `operands[0]`
    # alone" contract.
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
# `extra_columns` dict, leaving every other entry (including both `project` directions,
# neither of which needs `extra_columns`) exactly as registered here. `tests/api/
# test_execute.py::TestLoweringTableAgreement` reads this dict's KEYS only, never calls its
# values, so the placeholder arity mismatch here is never itself exercised.
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


def _bind_band_wide_lowerings(extra_columns: dict[str, tuple[str, ...]]) -> dict[tuple[str, object], object]:
    """The real registry for one `execute()` call: `_LOWERINGS` with its four
    `Shape.band_wide` non-project entries overridden by closures bound over this call's own
    `extra_columns` -- see `_LOWERINGS`'s own comment for why they cannot simply be module-
    level constants the way every other entry is.

    Each closure reads `extra_columns.get(op.input, ())` (or, for `join`,
    `op.inputs[0]`'s -- `_operand_names(op)[0]` gives the right name either way) rather
    than being handed it as a call argument, so the CALL signature every entry in the
    returned dict is invoked with stays the uniform `(frames, op, frame_types)` the walk
    loop already uses for everything else.
    """
    lowerings = dict(_LOWERINGS)
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
        frames, op, frame_types, extra_columns.get(op.inputs[0], ())
    )
    return lowerings


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


def _grid_read_window(grid_name: str) -> ReadWindow:
    """A full-extent, undecimated `ReadWindow` built from `grid_name`'s own geometry, with
    no file opened -- the window a zero-band raster origin gets, since `io.raster.
    scan_raster` cannot produce one itself (it raises on empty metadata; see the module
    docstring). `cpra.mp.data.grids.RasterGrid` already carries everything a full window
    needs (`bounds`, `height`, `width`, `crs`); the transform follows from
    `rasterio.transform.from_bounds`, exactly as `io.raster.scan_raster` derives it for a
    real, full-extent, undecimated read of the same grid (confirmed byte-identical against
    a real scan while building this). `source=()` names no file, matching the fact that
    none was opened; `batch_layout=()` is legitimately empty, meaning "the whole window,
    one batch" -- the same reading `execution.frame.row_major_grid_cell` already gives an
    empty `batch_layout` for a hand-built window.
    """
    grid = GRIDS[grid_name]
    return ReadWindow(
        source=(),
        col_off=0,
        row_off=0,
        width=grid.width,
        height=grid.height,
        transform=from_bounds(*grid.bounds, width=grid.width, height=grid.height),
        crs=CRS.from_user_input(grid.crs),
        decimation=1,
        batch_layout=(),
    )


def _open_band_wide_input(resolved: ResolvedInput) -> tuple[BandWideFrame, tuple[str, ...]]:
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

    Not used for an origin `execute`'s band-wide-join prelude already opened as part of a
    shared scan -- see `_open_band_wide_join_group`.
    """
    if has_no_bands(resolved.band_metadata):
        return empty_band_wide(_grid_read_window(resolved.selection.grid)), ()
    if resolved.straddling_vector_grid is not None:
        crosswalk_path = resolve_raster_crosswalk_path(resolved.selection.grid, resolved.straddling_vector_grid)
        opened = scan_raster(resolved.band_metadata, crosswalks=[(project.ZONE_COLUMN, str(crosswalk_path))])
        return opened, (project.ZONE_COLUMN,)
    return scan_raster(resolved.band_metadata), ()


def _open_band_wide_join_group(
    origins: tuple[str, ...], inputs: dict[str, ResolvedInput]
) -> dict[str, tuple[BandWideFrame, tuple[str, ...]]]:
    """One shared `scan_raster` call over the UNION of `origins`' band metadata, sliced
    back into one `(BandWideFrame, extra_columns)` pair per origin -- what
    `plan.band_wide_join_scans` records a band-wide join's raster operands as needing (see
    the module docstring for why one scan apiece is not equivalent).

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
    same reason given there.

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
    band_metadata = [inputs[name].band_metadata for name in origins]
    heights = [df.height for df in band_metadata]
    if sum(heights) == 0:
        window = _grid_read_window(inputs[origins[0]].selection.grid)
        return {name: (empty_band_wide(window), ()) for name in origins}

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

    lf, out_metadata, window = scan_raster(combined, crosswalks=crosswalks)

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
    return frames


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

    Before returning, drops the reserved band-index column
    (`execution.frame.drop_reserved_band_index`) from a band-wide result's metadata if
    present -- unconditionally, since either of two lowerings can leave it standing
    (`band_ops.group_band_wide`'s `by=[]` case, `project.project_vector_to_raster`'s
    zero-pivot-dimension case) and both document that whatever finishes the walk owns
    removing it before metadata reaches anything outside. Also drops whatever
    `extra_columns` the final band-wide output still carries, defensively -- see the module
    docstring's closing paragraph for why a fan-out can leave a raster origin's crosswalk
    column attached to a frame that never actually ran through the `project` that needed it.

    Returns the single input's frame for a zero-step pipeline: there is nothing to walk,
    and the request layer's own rule guarantees exactly one input in that case.
    """
    extra_columns: dict[str, tuple[str, ...]] = {}

    band_wide_join_frames: dict[str, BandWideFrame] = {}
    for origins in plan.band_wide_join_scans.values():
        for name, (frame, extra) in _open_band_wide_join_group(origins, plan.inputs).items():
            band_wide_join_frames[name] = frame
            extra_columns[name] = extra

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
            frames[name], extra_columns[name] = _open_band_wide_input(resolved)

    lowerings = _bind_band_wide_lowerings(extra_columns)

    last_output: str | None = None
    for op in plan.ops:
        lowering = _lowering_for(op.op, _dispatch_key(op, plan.frame_types), lowerings)
        frames[op.output] = lowering(frames, op, plan.frame_types)
        if plan.frame_types[op.output].shape is Shape.band_wide:
            extra_columns[op.output] = extra_columns.get(_operand_names(op)[0], ())
        last_output = op.output

    if last_output is None:
        (last_output,) = frames.keys()

    result = frames[last_output]
    if plan.frame_types[last_output].shape is Shape.band_wide:
        lf, metadata_df, window = result
        stray = extra_columns.get(last_output, ())
        if stray:
            lf = lf.drop(list(stray))
        result = (lf, drop_reserved_band_index(metadata_df), window)
    return result
