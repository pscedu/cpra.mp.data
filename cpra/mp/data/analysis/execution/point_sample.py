"""`point_sample_reads`: which raster origins of a compiled plan the POINT reader
(`io.raster_points.sample_raster_points`) opens instead of the band-wide one, and at which
absolute `grid_cell` ids -- plus the two `where` readers that decision shares with
`execution.execute`'s narrower-window fallback (`parse_point_sample_ids`,
`derive_point_sample_window`).

Its own module because two others need the one answer: `execution.execute` picks the reader
from it, and `execution.plan.compile_plan` exempts exactly the origins it names from
`ANALYSIS_BAND_CEILING`. That ceiling bounds the VRT a band-wide read builds; the point
reader builds none (it opens each file once and reads one pixel per band), so an origin it
takes is not what the ceiling guards against -- but an origin it REJECTS falls back to a
band-wide read and is. The exemption is therefore only as correct as its agreement with the
reader choice, and sharing the function is what makes the two impossible to diverge. It
cannot live in either consumer: `execute` imports `plan`, so `plan` cannot import `execute`.
"""

import re
from typing import TYPE_CHECKING

from rasterio.windows import Window

from cpra.mp.data.analysis.execution.frame import has_no_bands
from cpra.mp.data.analysis.request.expressions import PredicateKind, classify_where_kinds
from cpra.mp.data.analysis.request.ops import FilterOp
from cpra.mp.data.grids import GRIDS

if TYPE_CHECKING:
    from cpra.mp.data.analysis.execution.plan import ExecutionPlan

__all__ = ["derive_point_sample_window", "parse_point_sample_ids", "point_sample_reads"]


# The two `where` shapes `request.data_query.grid_cell_conjunct` (and any other caller
# resolving a coordinate to a literal id list before compiling a pipeline) ever produces
# for a raster spatial filter -- unquoted integers only, matching the convention every
# ABSOLUTE `grid_cell` id in this codebase already uses before any window has narrowed it.
# See `parse_point_sample_ids` for what happens when a `where` does not match either one.
_POINT_SAMPLE_EQ_RE = re.compile(r"^grid_cell\s*=\s*(\d+)$", re.IGNORECASE)
_POINT_SAMPLE_IN_RE = re.compile(r"^grid_cell\s+in\s*\(\s*(\d+(?:\s*,\s*\d+)*)\s*\)$", re.IGNORECASE)


def parse_point_sample_ids(where: str) -> list[int] | None:
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
    pixels match" -- the caller (`derive_point_sample_window`) falls back to reading the
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


def derive_point_sample_window(grid_name: str, where: str) -> Window | None:
    """Size a point sample's own narrower read window from a spatial `filter`'s `where` --
    still written in `grid_name`'s full-grid, absolute-id convention, since every producer
    of a spatial conjunct reaching a still-band-wide raster origin uses that convention
    (`execution.plan`'s module docstring; no narrowing has happened yet at the point this
    `where` was written), and the convention `band_ops.sample_grid_cells_band_wide` emits
    its own `grid_cell` output in too -- see `execution.execute`'s module docstring for why that makes
    narrowing the window here a pure read-size optimization rather than something able to
    change what `where`'s literals mean.

    Returns `None` -- caller falls back to a full-window read -- when `parse_point_sample_ids`
    cannot resolve `where` into exact ids, or when any resolved id falls outside
    `grid_name`'s own `(height, width)` extent (an out-of-range literal a caller supplied
    directly, with no coordinate resolution step in between to have already rejected it).

    `where` itself is never touched here or by any caller of this function: only the
    window is derived. The smallest window covering every resolved pixel, from the same
    `divmod(id, grid.width)` decode that validates an id is in range.
    """
    ids = parse_point_sample_ids(where)
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


def point_sample_reads(plan: "ExecutionPlan") -> dict[str, list[int]]:
    """For every raster origin the POINT reader (`io.raster_points.sample_raster_points`)
    should open instead of `execution.execute._open_band_wide_input`, the absolute
    `grid_cell` ids to sample it at -- the gate `execution.execute`'s module docstring's
    "skips the band-wide reader entirely" paragraph describes. An origin this names is opened long; every other raster origin is
    opened band-wide exactly as before.

    Six conditions, all required, each rejecting into the bounding-box fallback rather than
    into an error:

    * `resolved.point_sample_step is not None` -- a spatial `filter` was reached while this
      origin was still band-wide (`execution.plan`'s own module docstring). Without one
      there is no set of pixels to sample and nothing to gate.
    * Not tile-scoped, and not already empty: `resolved.read_window is None`, not
      `resolved.tile_disjoint`, and `metadata_df` has bands. The same three skips
      `execution.execute._point_sample_overrides` makes, for the same reasons -- a tile's own window
      already governs the read, and the other two have no real read at all.
    * `resolved.straddling_vector_grid is None`. A crosswalk is attached to a band-wide
      read (`_open_band_wide_input`) and consumed positionally by a later `project`; this
      reader returns no pixel grid for one to ride along on. An origin needing one is not a
      point sample in any case -- it is a zonal aggregate that happens to share an origin.
    * The origin is not part of a band-wide join's shared scan
      (`plan.band_wide_join_scans`). Such an origin's read is owned by
      `_open_band_wide_join_group`, which aligns several origins into one window; a long
      frame cannot participate in that alignment. (Shape typing already makes this
      unreachable -- a spatial filter turns a frame long, and a long frame is not a
      band-wide join operand -- but the gate states it rather than relying on that.)
    * **The spatial filter reads the origin itself** (`op.input == name`), not something
      derived from it. This is the condition that makes the substitution exact rather than
      approximate. The long lowering the walk will dispatch to applies `op.where` to
      whatever frame it is handed; the point reader's frame holds only the named pixels.
      Those two agree only if nothing ran in between -- an intervening
      `case`/`group`/`join` would have run against every pixel of the bounding box on
      the old path and against three on this one, and for a `group` in particular those are
      different answers. `point_sample_step` is recorded through a chain of ops that left
      the origin's pixels alone, so it can point at a filter one or more steps downstream;
      this gate wants the first step, and nothing else.

      The fan-out case -- a second op reading the same origin, still expecting it to be
      band-wide -- needs no condition here, because `compile_plan` has already foreclosed
      it: an origin more than one op reads is invalidated on sight (its `reference_count`
      exceeds one, so `plan`'s `_incoming` returns nothing for it), and an invalidated
      origin never records a `point_sample_step` at all. The first condition above is
      therefore what rejects a fan-out, and the gate cannot widen past that without
      `compile_plan` widening first.
    * `parse_point_sample_ids` resolves `op.where` to literal ids AND
      `derive_point_sample_window` accepts them -- the same parse and the same range check
      the fallback already runs, so the two paths can never disagree about which `where`
      names exactly which pixels.

    Takes a finished `ExecutionPlan` because several conditions (`read_window`,
    `straddling_vector_grid`) are only settled once `compile_plan` has built one; that is
    why `compile_plan` checks the ceiling for raster origins against its own result, last
    (see the module docstring).
    """
    reads: dict[str, list[int]] = {}
    ops_by_step = {op.step_no: op for op in plan.ops}
    joined = {name for origins in plan.band_wide_join_scans.values() for name in origins}
    for name, resolved in plan.inputs.items():
        if resolved.point_sample_step is None:
            continue
        if resolved.read_window is not None or resolved.tile_disjoint:
            continue
        if has_no_bands(resolved.band_metadata):
            continue
        if resolved.straddling_vector_grid is not None or name in joined:
            continue
        op = ops_by_step[resolved.point_sample_step]
        assert isinstance(op, FilterOp)
        if op.input != name:
            continue
        ids = parse_point_sample_ids(op.where)
        if ids is None or derive_point_sample_window(resolved.selection.grid, op.where) is None:
            continue
        reads[name] = ids
    return reads
