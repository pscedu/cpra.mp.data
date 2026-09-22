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
origin). `execution.execute` reads it to attempt a NARROWER window for that origin's read,
built from the same full-grid, absolute-id convention every producer of a spatial conjunct
reaching a still-band-wide origin already uses (see `execution.execute._derive_point_
sample_window`) -- but only when the filter's own `where` is simple enough to name the
exact pixels before the read happens (`grid_cell = N` / `grid_cell IN (...)`, unquoted
integers); anything else -- a negation, an `OR`, a comparison other than equality --
falls back to reading the origin whole, which is still correct, only slower, and is a
choice this module leaves entirely to the executor. Narrowing the window this way changes
nothing about `where` itself, or about what a `grid_cell` id in the eventual result means:
`execution.band_ops.sample_grid_cells_band_wide` always emits an ABSOLUTE id (row-major
over the origin's full grid, not the narrowed window), so the literal `where` was written
against keeps naming the same pixel regardless of how much of the grid the read actually
covered. What this module commits to is unchanged by that choice: only WHERE the filter
sits in the declared chain, and to no longer tracking origin validity for that origin past
it (per the table above, a spatial filter turns a band-wide frame long, so anything after
it addresses a different, already-materialized frame).

**Overview eligibility is a per-raster-SOURCE property, decided here rather than wherever
the read happens.** A tile read is free to let GDAL serve a decimated window from a raster's
own pyramid (`io.raster_io.tile_read_shape`) whenever the read feeds nothing but per-pixel
operations -- every overview level is a real, precomputed downsample, so which one gets
picked changes only how coarse the answer is, never which source pixel a per-pixel result
traces to. Two things make that unsafe:

* A raster-to-vector `project` step redistributes pixels onto polygons that can be smaller
  than one overview pixel at a coarse level, or straddle a boundary an overview blurs
  across -- the join has to see the FULL-resolution pixels a polygon actually covers, not
  whichever overview happened to be near the tile's own resolution. `straddling_vector_grid`
  records, per raster origin, the OUTPUT vector grid such a step (reachable from that
  origin via ANY `ops` entry, not only ones pushdown validity survived to -- see the
  provenance paragraph above on why `origin_sets` rather than `states` is what this reads)
  targets, or `None` if none does; `overview_permitted` is unconditionally `False` whenever
  it is not `None`. A raster origin reachable from two raster-to-vector `project` steps
  that target DIFFERENT grids raises `request.errors.ShapeError` here, at compile time,
  rather than silently keeping whichever step this loop happened to see first: the crosswalk
  attached at open time (see `execution.execute`/`execution.project`'s own docstrings) is
  one artifact, resolved against one destination grid, so one origin cannot straddle two
  destinations at once the way `origin_sets`' fan-out otherwise freely allows for ops that
  do not carry that constraint (a fan-out into two `calculate`s, or into a `project` and an
  ordinary `join`, is unaffected -- only two conflicting `project` destinations off the same
  raster origin are rejected).
* A point sample (`point_sample_step`, above) reads exactly the pixel under a coordinate --
  an overview's decimated grid can hand back a neighbouring pixel's value instead of the
  one actually asked for. `overview_permitted` is `False` for any origin with a
  `point_sample_step`, for the same reason.

Every other raster origin -- including one whose pixels are later joined (not projected)
onto a vector chain, e.g. a vector-to-vector crosswalk aggregate joined with a raster for
display -- keeps `overview_permitted` at its default `True`: a `join` is not a `project`,
so it never appears in `straddling_vector_grid`, and nothing about attaching a raster's
already-resolved pixel values to a vector chain's rows demands full resolution the way
redistributing the pixels themselves onto new polygons does.

Both are read by `_resolve_tile_window` below. `straddling_vector_grid` is forwarded to
`io.raster_io.tile_source_window` as that function's own argument of the same name;
`overview_permitted` gates `io.raster_io.tile_decimation_factors`, whose answer becomes
`ResolvedInput.decimation` and travels to `io.raster.scan_raster` unchanged. A
non-permitted origin gets `(1, 1)` and reads every source pixel, which is what makes the
rule above an enforcement rather than an annotation.

A band-wide `join`'s operands are read as ONE shared scan, which can therefore carry only
ONE decimation; `execution.execute` takes the per-axis MINIMUM of the group's factors,
which is `(1, 1)` -- full resolution for the whole group -- the moment any one operand's
origin is not permitted. That is deliberately the conservative direction: the shared read
covers the UNION of the operands' windows, which is wider than any single one and could
bear a coarser factor than the minimum, so the group reads at worst a few times more pixels
than it strictly needs and never fewer.

**The tile read window is decided here, not at the read.** A tile is a property of the
REQUEST, and which pixels it needs off each raster origin depends on things only this
module knows: the grid each origin sits on, and whether any raster-to-vector `project`
reachable from it widens the window to whole straddling features. So `compile_plan` takes
an optional `TileRequest` and records one `read_window` per raster origin, which
`execution.execute` then hands to `io.raster.scan_raster`; the walk itself is identical
either way, and the sink warps whatever it gets onto the tile grid. The alternative -- a
sink that opens the raster again and runs its own op walk at a tile window -- was tried and
removed: it duplicated the walk, and it could only ever support the ops that keep a frame
band-wide, which excludes the raster-to-vector `project` the straddling-feature widening
exists for in the first place.

Sizing that window opens no file. `io.raster_io.raster_grid_geometry` builds
`tile_source_window`'s four inputs (crs, transform, width, height) from the grid, which a
raster on that grid is co-registered with exactly -- so this module keeps the "it opens no
data files" property that the rest of it already has, band resolution aside.

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

**A band-wide join's operands need one shared read even when one of them started life as a
vector.** The shared read exists because positional alignment is only correct when every
operand was chunked identically, and chunking follows BAND COUNT (`io.raster_io.
raster_blocks_per_batch`; `execution.project`'s module docstring gives the measured
mis-pairing for the two-read shape). That argument does not care where an operand's pixels
came from -- only that they came from the same read -- so it applies unchanged to a
vector input broadcast onto the raster grid by a vector-to-raster `project`, whose pixel
order is set by the crosswalk raster IT reads. `band_wide_join_scans` therefore records
only the RASTER origins (they are the ones contributing bands to the union, and the ones
the ceiling check sums), while `band_wide_join_zone_crosswalks` records, for the same join
output, one `ZoneCrosswalk` per vector-to-raster `project` upstream of ANY of that join's
operands -- the crosswalk artifacts `execution.execute` must hand `io.raster.scan_raster`
as `crosswalks=` entries on that one combined call, so the projected operand's pixel order
is the raster operands' by construction rather than by coincidence. Tracked in its own
forward pass (`zone_crosswalks_upstream_of`) rather than inside the main walk: unlike
everything `states` carries, it cannot be invalidated by a pushdown decision, and it has
to be complete before the first `JoinOp` consults it.

A band-wide join with NO raster origin at all -- every operand a separately-projected
vector input -- is the one shape left out, since there are no data bands for a shared read
to be built around and `scan_raster` has no zero-band answer to give (see the loose end
below). Those operands stay independently read, and `execution.band_ops.join_band_wide`'s
own `assert_same_pixel_grid` precondition is what decides whether their chunkings happened
to agree; single-band crosswalk reads over one grid usually do, but nothing here makes
them.

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

import mercantile
import polars as pl
from rasterio.enums import Resampling

from cpra.mp.data.analysis.io.catalog import resolve_raster_bands
from cpra.mp.data.analysis.io.raster_io import (
    raster_grid_geometry,
    resampling_buffer_px,
    tile_decimation_factors,
    tile_disjoint_from_source,
    tile_source_window,
)
from cpra.mp.data.analysis.request.errors import BandLimitError, RequestError, ShapeError
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

__all__ = ["DEFAULT_TILE_SIZE", "ExecutionPlan", "ResolvedInput", "TileRequest", "compile_plan"]

#: Pixels across for a map tile whose caller did not say. The web-map convention, and what
#: `visualizations/map.py` already declares as its layer's `tileSize`.
DEFAULT_TILE_SIZE = 256


@dataclasses.dataclass(frozen=True)
class TileRequest:
    """One map tile, as the compiler needs to see it: which tile, how many pixels across,
    and how its pixels will be resampled onto the tile grid.

    Passed to `compile_plan` by a caller rendering a tile; absent (`None`) for every other
    request, which reads each raster origin at full extent. It carries no legend and no
    image format -- those are the sink's, and nothing about the read window depends on
    them.

    `resampling` is here rather than at the sink because the read is what it changes: the
    buffer of neighbouring source pixels a window needs is a property of the resampling
    kernel (`io.raster_io.resampling_buffer_px`), so a window sized for one method is not
    valid for another.
    """

    x: int
    y: int
    z: int
    tile_size: int = DEFAULT_TILE_SIZE
    resampling: Resampling = Resampling.nearest

    @property
    def bounds(self):
        """This tile's bounds in `io.raster_io.TILE_CRS` (Web Mercator)."""
        return mercantile.xy_bounds(self.x, self.y, self.z)


@dataclasses.dataclass(frozen=True)
class ResolvedInput:
    """One `Pipeline.inputs` entry, plus whatever pushdown `compile_plan` decided is safe
    for it.

    `columns`/`predicate` are what `io.vector.scan_vector` accepts, populated only for an
    input on a vector grid -- see the module docstring for exactly when each is non-`None`.
    `band_metadata`/`point_sample_step`/`straddling_vector_grid`/`overview_permitted` are
    the raster-side counterpart, populated only for an input on a raster grid:
    `band_metadata` is `io.catalog.resolve_raster_bands`'s result, narrowed by every
    dimension predicate pushdown allowed through, and `point_sample_step` is the step
    number of the first spatial filter reached while this origin was still band-wide
    (`None` if none was). `straddling_vector_grid` is the name of the vector grid this
    origin's pixels get redistributed onto, if any `ops` entry ever runs a raster-to-vector
    `project` step reachable from this origin (`None` otherwise) -- see the module
    docstring's paragraph on overview eligibility for what it and `overview_permitted` are
    for and how they are derived. An input's grid kind decides which group of fields is
    populated; the other group is left at its default.

    `read_window`/`tile_disjoint`/`decimation` are the tile read decision for a raster
    input, and are the default triple (`None`, `False`, `(1, 1)`) -- meaning "read this
    origin at full extent and full resolution" -- unless `compile_plan` was given a
    `TileRequest`. Under one, `read_window` is the `rasterio.windows.Window` that tile needs
    off this origin, and `tile_disjoint` is `True` (with `read_window` still `None`) for the
    separate case where the tile's footprint misses this origin's extent altogether. The two
    cases are distinct fields rather than one nullable one because `None` already means
    "full extent" and a disjoint tile is the opposite of that -- see the module docstring's
    tile paragraph. `decimation` is the per-axis `(row_factor, col_factor)` that window may
    be read at, `(1, 1)` whenever `overview_permitted` is `False` -- it is what the
    eligibility rule above is FOR, and `io.raster.scan_raster` takes it under the same name.
    """

    selection: InputSelection
    columns: list[str] | None = None
    predicate: object | None = None  # `pl.Expr | None`; left untyped to avoid importing
    # polars into every reader of this dataclass's annotations.
    band_metadata: object | None = None  # `pl.DataFrame | None`; same reasoning.
    point_sample_step: int | None = None
    straddling_vector_grid: str | None = None
    overview_permitted: bool = True
    read_window: object | None = None  # `rasterio.windows.Window | None`; same reasoning.
    tile_disjoint: bool = False
    decimation: tuple[int, int] = (1, 1)


@dataclasses.dataclass(frozen=True)
class ZoneCrosswalk:
    """One vector-to-raster `project` step's crosswalk, named so that the shared read a
    band-wide `join` performs can carry it -- see the module docstring's paragraph on
    why that read, and not a second one inside the lowering, is where it belongs.

    `step_no` is the `ProjectOp`'s own step number, which is how `execution.execute` hands
    the resulting zone column back to exactly the lowering that asked for it.
    `raster_grid` is the step's `to` (the destination pixel lattice, and the grid the
    shared read is already addressing); `vector_grid` is the grid its input frame is
    currently on. The pair is what `io.crosswalk_read.resolve_raster_crosswalk_path`
    takes, in that order -- resolved at read time rather than here, matching
    `ResolvedInput.straddling_vector_grid`, which likewise records the grid and leaves the
    artifact lookup to `execution.execute`.
    """

    step_no: int
    raster_grid: str
    vector_grid: str


@dataclasses.dataclass(frozen=True)
class ExecutionPlan:
    """`inputs`: every `Pipeline.inputs` entry, resolved (see `ResolvedInput`). `ops`:
    `request.ops.compile_ops`'s output, unchanged. `frame_types`: every input/step-output
    name's `request.ops.FrameType`, computed once here so `execute.py`'s walk never
    re-derives one. `band_wide_join_scans`: every band-wide `join` step's output name
    mapped to the RASTER origins whose bands were read as one shared union for its ceiling
    check (see the module docstring) -- empty unless the pipeline has such a join.
    `band_wide_join_zone_crosswalks`: the same join outputs mapped to the `ZoneCrosswalk`s
    that shared read must also carry, one per vector-to-raster `project` upstream of any
    of that join's operands -- empty for a join whose operands are all raster origins.
    `tile`: the `TileRequest` this plan was compiled for, or `None` for a full-extent one;
    the sink needs it to warp the result onto the tile's own grid, and it is recorded here
    so that it comes from the same place the read windows were derived from rather than
    being passed to the sink a second time and trusted to match.
    """

    inputs: dict[str, ResolvedInput]
    ops: list[Op]
    frame_types: dict[str, FrameType]
    band_wide_join_scans: dict[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)
    band_wide_join_zone_crosswalks: dict[str, tuple[ZoneCrosswalk, ...]] = dataclasses.field(
        default_factory=dict
    )
    tile: TileRequest | None = None


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


def compile_plan(pipeline: Pipeline, *, base_path=None, tile: TileRequest | None = None) -> ExecutionPlan:
    """Compile `pipeline` (already validated -- `parse_pipeline` then `validate_shapes`)
    into the `ExecutionPlan` `execute.execute` consumes, including every pushdown decision
    the module docstring describes. `base_path` is forwarded to `io.catalog.
    resolve_raster_bands` for each raster input; left `None` to use the configured default
    (tests always pass a sandboxed path instead -- see `tests/conftest.py`).

    `tile`, when given, compiles the plan for one map tile: every raster origin gets a
    `read_window` sized to that tile instead of being read whole, derived here rather than
    at the read (see the module docstring's tile paragraph, and `_resolve_tile_window`).
    The ops, the pushdown decisions, and the band resolution are identical either way -- a
    tile changes how much of each raster is read, not what is computed over it.
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
    band_wide_join_zone_crosswalks: dict[str, tuple[ZoneCrosswalk, ...]] = {}

    # Which vector-to-raster `project` steps produced the pixels a given frame is made of
    # -- the vector-side counterpart of `origin_sets` below, and the input to the same
    # "one shared read" rule (see the module docstring). Derived in its own forward pass
    # rather than inside the main walk because it must be complete BEFORE the first
    # `JoinOp` reads it, and because it is a property of the ops alone: unlike everything
    # the main walk tracks, no pushdown decision can invalidate it.
    zone_crosswalks_upstream_of: dict[str, tuple[ZoneCrosswalk, ...]] = {name: () for name in pipeline.inputs}
    for op in ops:
        if isinstance(op, JoinOp):
            zone_crosswalks_upstream_of[op.output] = tuple(
                dict.fromkeys(
                    crosswalk for operand in op.inputs for crosswalk in zone_crosswalks_upstream_of[operand]
                )
            )
            continue
        if isinstance(op, ProjectOp):
            from_kind, to_kind = project_direction(frame_types[op.input], op.to)
            if (from_kind, to_kind) == ("vector", "raster"):
                # A vector-to-raster project CREATES this branch's pixel lattice, so
                # whatever came before it is irrelevant to how the branch is chunked; the
                # crosswalk it broadcasts across is the whole answer. (There can never be
                # a second one further along the same chain: this step's output is
                # band-wide, so any later `project` off it is a raster-to-something one.)
                zone_crosswalks_upstream_of[op.output] = (
                    ZoneCrosswalk(step_no=op.step_no, raster_grid=op.to, vector_grid=frame_types[op.input].grid),
                )
            else:
                # Either direction out of here is long, so there is no pixel order left
                # for a shared read to have to agree with.
                zone_crosswalks_upstream_of[op.output] = ()
            continue
        # A spatial `filter` unpivots band-wide to long (`execution.band_ops.
        # filter_band_wide`), which retires the pixel order along with the shape; every
        # other op leaves it exactly as it found it.
        zone_crosswalks_upstream_of[op.output] = (
            zone_crosswalks_upstream_of[op.input] if frame_types[op.output].shape is Shape.band_wide else ()
        )

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
            if frame_types[op.output].shape is Shape.band_wide:
                # Only the RASTER origins contribute bands to the shared read. A vector
                # origin reaches a band-wide join solely by way of a vector-to-raster
                # `project` (nothing else gives a vector input pixels), and that step's
                # crosswalk -- not the origin -- is what has to ride along on the same
                # read; `zone_crosswalks_upstream_of` is where it was recorded.
                raster_union = tuple(name for name in union if origin_kind[name] == "raster")
                if raster_union:
                    band_wide_join_origin_groups[op.output] = raster_union
                    band_wide_join_zone_crosswalks[op.output] = zone_crosswalks_upstream_of[op.output]
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

    # Overview eligibility (module docstring): which raster origins a `project` step ever
    # redistributes onto a vector grid. A second, narrower pass over `ops` rather than part
    # of the walk above -- it only needs `origin_sets`/`frame_types`, both already fully
    # populated by every op regardless of whether pushdown validity survived to reach it,
    # which is exactly what this has to ignore (a raster origin that fed a `project` before
    # its own validity was invalidated by an earlier fan-out still needs full-resolution
    # reads at that `project`).
    straddling_vector_grid: dict[str, str] = {}
    for op in ops:
        if not isinstance(op, ProjectOp):
            continue
        from_kind, to_kind = project_direction(frame_types[op.input], op.to)
        if from_kind != "raster" or to_kind != "vector":
            continue
        for origin in origin_sets[op.input]:
            if origin_kind[origin] != "raster":
                continue
            existing = straddling_vector_grid.get(origin)
            if existing is not None and existing != op.to:
                raise ShapeError(
                    f"raster input {origin!r} straddles two different destination grids "
                    f"({existing!r} and {op.to!r}) via separate 'project' steps reachable "
                    "from it -- a raster origin's full-resolution pixels can only be read "
                    "once per compiled plan, so a fan-out with two raster-to-vector "
                    "projects targeting different grids is not supported"
                )
            straddling_vector_grid[origin] = op.to

    inputs: dict[str, ResolvedInput] = {}
    for name, selection in pipeline.inputs.items():
        if origin_kind[name] == "raster":
            target = straddling_vector_grid.get(name)
            permitted = target is None and name not in point_sample_step
            read_window, tile_disjoint, decimation = _resolve_tile_window(
                tile, selection.grid, target, overview_permitted=permitted
            )
            inputs[name] = ResolvedInput(
                selection=selection,
                band_metadata=raster_band_metadata[name],
                point_sample_step=point_sample_step.get(name),
                straddling_vector_grid=target,
                overview_permitted=permitted,
                read_window=read_window,
                tile_disjoint=tile_disjoint,
                decimation=decimation,
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
        band_wide_join_zone_crosswalks={
            output: crosswalks for output, crosswalks in band_wide_join_zone_crosswalks.items() if crosswalks
        },
        tile=tile,
    )


def _resolve_tile_window(
    tile: TileRequest | None,
    grid_name: str,
    straddling_vector_grid: str | None,
    *,
    overview_permitted: bool = True,
):
    """The `(read_window, tile_disjoint, decimation)` triple one raster origin gets --
    `(None, False, (1, 1))`, "read it whole, at full resolution", whenever there is no tile
    to scope it to.

    Sized from the GRID's geometry (`io.raster_io.raster_grid_geometry`) rather than from
    an opened raster, which is what keeps this module's promise to open no data files while
    still producing the tile's read window here. A raster on grid *G* is co-registered with
    *G*, so the two answers are the same window; see `SourceGeometry` for why that holds.

    Both of `io.raster_io`'s tile-versus-source tests run, in this order, because neither
    subsumes the other: `tile_disjoint_from_source` carries the grid's extent out to the
    tile's CRS and is decisive for a tile far enough away that the inbound reprojection is
    not finite, while `tile_source_window` carries the tile's footprint inward and is the
    one that knows about buffering, clipping and widening. See the former for why that
    asymmetry is not redundancy.

    `straddling_vector_grid` is forwarded to `tile_source_window` unchanged: it is the
    widening that makes a tile whose pixels feed a raster-to-vector `project` read every
    pixel of each feature it touches, not just the part inside the tile. Deriving WHICH grid
    that is belongs here; applying it belongs there.

    `overview_permitted` is the caller's already-derived answer (see the module docstring's
    paragraph on overview eligibility) and is what turns `io.raster_io.
    tile_decimation_factors` on: permitted, the window is decimated to roughly
    `TILE_READ_OVERSAMPLE` times the tile's own resolution, which is what keeps a low-zoom
    tile's cost a function of the tile rather than of the zoom; not permitted, the factors
    stay `(1, 1)` and every source pixel is read. The buffer the factors are sized against
    is the same `resampling_buffer_px(tile.resampling)` that `tile_source_window` just
    widened the window by, rather than the bare `TILE_READ_BUFFER_PX` default, so the
    decimation target describes the window this function is actually returning.
    """
    if tile is None:
        return None, False, (1, 1)
    geometry = raster_grid_geometry(grid_name)
    if tile_disjoint_from_source(geometry, tile.bounds):
        return None, True, (1, 1)
    window = tile_source_window(
        geometry,
        tile.bounds,
        resampling=tile.resampling,
        straddling_vector_grid=straddling_vector_grid,
    )
    if window is None:
        return None, True, (1, 1)
    if not overview_permitted:
        return window, False, (1, 1)
    factors = tile_decimation_factors(
        window, tile.tile_size, buffer_px=resampling_buffer_px(tile.resampling)
    )
    return window, False, factors
