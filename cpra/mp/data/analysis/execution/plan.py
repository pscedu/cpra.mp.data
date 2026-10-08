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

**Dimension conjuncts are pushed by a different mechanism from value and spatial ones,
and the split is the main thing to hold onto about this module.** A `value` or `grid_cell`
conjunct is pushed only along a line of computation that has stayed exclusive and
value-preserving all the way back to one input -- the forward walk below tracks exactly
that, and gives up at a join or a fan-out. A DIMENSION conjunct is pushed from anywhere in
the pipeline to every origin it can possibly apply to, because it has two properties the
other two do not:

* `join_long` is an inner join, so `sigma_p(A join B) == sigma_p(A) join B` for a `p` over
  A's own columns. Narrowing one operand's read cannot change what the other operand's
  rows get matched against -- the rows it removes had no surviving match anyway.
* Every `filter` step still re-executes over whatever came back. A dimension narrowing is
  therefore ADDITIVE: getting it too wide costs time and nothing else, and the plan may
  freely decline to narrow whenever it is not certain. Only narrowing too FAR is a wrong
  answer, which is why every judgement below errs toward keeping a band.

Two passes over the frame graph do that pushing, after the walk, carrying the same answer
in two currencies -- see `compile_plan`'s own comments for each. The SYMBOLIC pass carries
a predicate per (frame, origin) and produces the dimension half of a vector origin's
`scan_vector` predicate. The BAND-IDENTITY pass carries surviving band positions per
(frame, origin), narrowing them at each conjunct against the unified metadata frame
(`execution.band_ops.unify_metadata_parts` -- the same function `join_band_wide`'s own
post-join dimension filter uses, so plan time and execution time cannot disagree), and
produces each raster origin's band list and each vector-to-raster projection's. The band
pass is the only one that can see a conjunct on ONE origin's dimension narrowing ANOTHER's
bands across a dimension they share; the symbolic one is the only one that reaches an
origin whose bands are not a plan-time fact at all. Where both apply they agree, and
applying both changes nothing.

In both passes a join MERGES its operands' entries and a fan-out's branches merge at the
terminal frames, in both cases by OR / set union: one physical read has to satisfy every
branch that shares it, and the superset is the safe direction.

**The forward walk.** `ops` (already fully resolved by `request.ops.compile_ops`, in
declared order) is walked once, carrying one `_Validity` state per line of computation
back to whichever named input it ultimately traces to (its "origin"). At each op, the
state says whether a `value` predicate and a `grid_cell` (spatial) predicate could still
be pushed all the way back to the origin's own read without changing what any downstream
consumer -- including the final sink -- would see. This is the same table `request/ops.py`'s
shape-typing walk already commits to per op, restated here as a validity question rather
than a shape question. It also records, unconditionally and regardless of that state, the
structure the two dimension passes run over: each frame's parents, the dimension conjuncts
declared on it, and which of each contributing ORIGIN's dimension columns it still carries.

* `filter` never changes what is still pushable -- it only *consumes* validity (deciding,
  per conjunct, whether THIS filter's own predicate can be pushed), it does not reduce it
  for whatever comes after.
* `case` invalidates `value` unconditionally (the running value is being replaced),
  leaving spatial validity untouched. A `calculate` is an arm-less `case`
  (`request.ops.compile_ops`), so it is the same entry rather than a second one.
* `group` invalidates `value` unconditionally. `grid_cell` survives only if
  `"grid_cell" in by`. Structurally, it is the only op that takes a dimension column AWAY
  from a frame, which is what narrows the recorded per-origin dimension sets to `by`: a
  conjunct naming a dimension this `group` collapsed has nothing to push back to, since
  the frame it was declared on does not carry that column at all. A dimension that IS in
  `by` survives under its own name and stays fully pushable -- every row the origin's read
  would drop for it is a row the `group` would have dropped from that dimension's bucket
  anyway.
* `project` invalidates `value` and `grid_cell` EXCEPT in the one direction whose own
  semantics preserve `value` exactly: vector-to-raster (`project_direction` returns
  `("vector", "raster")`). That direction takes no aggregate (`request.ops._type_project`'s
  `direction_takes_aggregate` is `False` there), so a raster pixel's value after such a
  project is identically the value its containing vector zone already had -- meaning a
  `value` predicate declared AFTER this project is exactly as valid a filter on the
  original vector rows as it would be on the post-project pixels, and gets pushed all the
  way back to that vector origin's own scan. Every other project direction stops both:
  once rows have been redistributed onto a different grid, there is no single upstream row
  a downstream value or spatial predicate corresponds to. Dimensions are untouched in
  EVERY direction -- a project changes which grid a row addresses, not which observation
  it is of -- so a dimension conjunct declared after one still narrows its origin.
* `join` stops `value` and `grid_cell` for both operands. What a value conjunct on a
  joined frame even names is an operand's renamed value column (`join_long`'s doing), not
  a source column, and a spatial one addresses a grid neither operand's read is indexed by
  on its own. A band-wide `join` gets one piece of special handling: since
  `execution.band_ops.join_band_wide` reads both operands through one shared VRT (see that
  module), the operands' resolved band metadata still needs to be sized as a UNION for the
  `ANALYSIS_BAND_CEILING` check below.

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

`reference_count` governs `_Validity` alone. The dimension passes answer the same question
without giving up, by taking the UNION of what the branches need rather than the
conjunction of it: a read narrowed to "2030 or 2040" satisfies the branch that asked for
2030 and the one that asked for 2040 at once, and each branch's own `filter` step still
runs. Branches that agree -- the same input read twice under the same selection, which is
the common case -- collapse back to that one predicate. The union is taken over the
TERMINAL frames, the ones nothing else reads, since every frame the plan computes is
reachable from one of those or is one.

**Column pushdown needs one more guard beyond validity/fan-out: a `group`.** A `filter`'s
or a `case` arm's referenced dimension names get folded into an origin's candidate
column set as they're encountered, but that set is only ever actually handed to
`io.vector.scan_vector` as a real narrowing (`ResolvedInput.columns`) once the origin's
chain has passed through at least one `group`. Before that point, narrowing the source
read's columns to "only what's been referenced so far" would be wrong even though every
individual predicate pushed so far is fine: a `filter`/`case`-only chain does not
touch which COLUMNS the frame carries (a filter drops rows, a case replaces `value`,
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

The one place the two dimension passes have to be reconciled into a single answer is a
vector origin feeding a vector-to-raster `project`, where the band pass may narrow the
projection's declared band list past what the symbolic pass narrowed the scan to. The scan
has to return the rows those bands are made of and no more, or the projection would
declare a band nothing fills, so the surviving band rows are turned back into a per-column
`IN`-list predicate (`_surviving_value_predicate`) and conjoined onto that origin's scan.

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

A spatial filter on a JOINED band-wide frame records the same thing for every raster origin
of that join at once. `_Validity` tracks one origin and a join has several, so the walk's
own `_incoming` gate retires them all before the per-op branches run -- but the origins are
named anyway, by `band_wide_join_origin_groups`, and they are read as one shared scan
(`execution.execute._open_band_wide_join_group`), which is exactly the unit a single window
can be sized for. So the gate records the sample rather than merely dropping it, subject to
both reference counts being one: the narrowing applies to the SHARED READ, so a second
consumer of the join's output, or an origin that also feeds some other branch, would be
handed one pixel where it asked for a grid. Only a filter sitting directly on the join is
recognized; an op in between is left unnarrowed rather than reasoned about, since not
narrowing is always correct.

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

**That shared read is per CONNECTED COMPONENT of origins, not per join.** A pipeline that
chains joins -- `join [a, b]`, then `join [<something derived from a>, c]` -- puts origin
`a` in two groups, and the two groups have different band counts, so reading `a` once per
group would chunk the same pixels two different ways and the earlier join's operands could
no longer be aligned positionally. `_merge_shared_origin_groups` therefore fuses (
transitively) every group sharing an origin before the ceiling check runs, so both joins
name one union and `execute` opens it once. The ceiling is checked against that union
because the union is what one read actually costs.

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
is constructible without opening a file, and the grid is where its window comes from:
`execution.execute._grid_read_window` is that construction, and is how a zero-BAND origin
already gets a window rather than a `ValueError`. What it does not settle is the shape
above -- a join group with no raster origin AT ALL has no shared read to build in the
first place, which is a different gap with a different answer.

**A vector-to-raster `project`'s bands are enumerated here too, for the same reasons.**
That step turns a long frame's distinct dimension combinations INTO the columns of a
band-wide one, so its width is settled by data, not by the request -- exactly the position
a raster input is in. `io.catalog.resolve_vector_bands` answers it, once per such step,
and the results land in `ExecutionPlan.vector_band_metadata` keyed by STEP NUMBER (a
projection, not an input, is what creates the bands, and one input projected twice is two
different widths). `execution.project.project_vector_to_raster` takes that frame as its
declared band universe instead of deriving one by collecting the data, and the ceiling
loops count those bands alongside the raster ones wherever the two meet in a band-wide
join -- projected bands are never READ, but they are columns of the joined frame all the
same.

The enumeration deliberately runs AFTER the walk rather than at the `project` step: a
value `filter` declared after a vector-to-raster project still pushes back to that vector
origin (per the table above), so the predicate the scan will use is only final once every
op has been seen -- and the enumeration must count the same rows the scan returns. A step
whose origin could not be traced (fan-out, or an operand of a join), or whose dataset has
nothing on disk yet, is simply absent from `vector_band_metadata`: its width is not a
plan-time fact, and nothing may treat the absence as zero.

**Compile opens no data file for a pipeline that stays LONG.** Anything resolving to BANDS
is the exception, in both directions: a raster input's band list requires one catalog read
(`io.catalog.resolve_raster_bands`, which in turn calls `cpra.mp.data.raster.read_cog`),
and a vector-to-raster `project` requires one `resolve_vector_bands` scan. Both are
unavoidable -- which bands exist can only be answered by looking -- and neither happens
for a pipeline with no raster input and no such projection.
"""

import dataclasses
from collections import Counter

import mercantile
import polars as pl
from rasterio.enums import Resampling

from cpra.mp.data.analysis.execution.band_ops import PAIRING_KEY_EXCLUDED_COLUMNS, unify_metadata_parts
from cpra.mp.data.analysis.io.catalog import (
    dataset_dimension_columns,
    resolve_raster_bands,
    resolve_vector_bands,
)
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
    CaseOp,
    FilterOp,
    FrameType,
    GroupOp,
    JoinOp,
    Op,
    ProjectOp,
    Shape,
    UnpivotOp,
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
    (`None` if none was) -- either one this origin carried its own validity to, or one
    sitting directly on a band-wide `join` this origin is an operand of, which samples
    every origin of that join at once (see the module docstring). `straddling_vector_grid`
    is the name of the vector grid this
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
    check (see the module docstring) -- empty unless the pipeline has such a join. Two
    joins sharing an origin carry the SAME tuple here, not their own: they are one read
    (`_merge_shared_origin_groups`), so `execute` opens the group once and every operand
    downstream of either join is chunked alike.
    `band_wide_join_zone_crosswalks`: the same join outputs mapped to the `ZoneCrosswalk`s
    that shared read must also carry, one per vector-to-raster `project` upstream of any
    of that join's operands -- empty for a join whose operands are all raster origins.
    `vector_band_metadata`: every vector-to-raster `project` step whose bands this compile
    could enumerate ahead of the read, mapped to that enumeration (`io.catalog.
    resolve_vector_bands`' frame -- one row per band, in band order). It is the vector
    counterpart of `ResolvedInput.band_metadata`, keyed by STEP rather than by input
    because a projection, not an input, is what turns vector rows into bands; it is what
    the ceiling is checked against for those bands, and `execution.execute` hands each
    entry to `execution.project.project_vector_to_raster` so the width is declared rather
    than discovered. A `project` missing from this dict is one whose input `compile_plan`
    could not trace back to a single origin -- a `join` or a fan-out upstream of it -- or
    whose dataset has nothing on disk to enumerate yet; either way that step enumerates its
    own bands at execution time instead.
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
    vector_band_metadata: dict[int, pl.DataFrame] = dataclasses.field(default_factory=dict)
    tile: TileRequest | None = None


@dataclasses.dataclass(frozen=True)
class _Validity:
    """Per-origin pushdown state carried forward through one line of computation while
    walking `ops` -- see the module docstring's table for what each op does to it.

    `origin` is the named input this state still traces back to; `value`/`grid_cell` are
    plain booleans: still pushable, or not.

    **Dimension validity is deliberately NOT here.** A dimension conjunct is pushed back
    through the band-identity pass below instead, which propagates per-origin surviving
    BAND SETS rather than a per-origin yes/no -- so it survives the two things that retire
    this state entirely, a join (both operands narrow, against the unified metadata frame)
    and a fan-out (the origin keeps the union of what its branches need). Which dimension
    names are still standing at a given frame is a structural fact about that frame, not a
    validity one, and lives in `compile_plan`'s `frame_origin_dimensions`.
    """

    origin: str
    value: bool
    grid_cell: bool


def _referenced_names(conjunct: str) -> frozenset[str]:
    """The column names one already-validated `where` conjunct references -- reuses
    `pl.Expr.meta.root_names()`, the same mechanism `request.expressions.
    classify_predicate_kind` itself classifies conjuncts by.
    """
    return frozenset(pl.sql_expr(conjunct).meta.root_names())


# One column per participating origin in a unified metadata frame, carrying that origin's
# own band positions. Named apart from anything a dataset can carry so it can never
# collide with a real dimension and never become a join key.
_ORIGIN_POSITION_PREFIX = "__origin_position_"


def _dimension_survivors(
    surviving: dict[str, frozenset[int]],
    metadata: dict[str, pl.DataFrame],
    standing: dict[str, frozenset[str]],
    conjunct: str,
) -> dict[str, frozenset[int]]:
    """Which of each origin's still-surviving bands a single dimension `conjunct` leaves,
    evaluated against the UNIFIED metadata frame of every origin feeding the frame the
    conjunct was declared on.

    `surviving` maps origin name to the band positions still standing there; `metadata`
    holds each enumerable origin's full band frame (raster bands, or a vector-to-raster
    projection's enumerated ones), positions indexing into it. `standing` says, per origin,
    which of ITS dimensions the frame still carries -- per origin rather than per frame so
    that a name one origin's upstream `group` collapsed cannot pair that origin's bands
    against another's that still has the column on disk.

    The unified frame is `execution.band_ops.unify_metadata_parts` -- the SAME function
    `join_band_wide`'s own post-join dimension filter uses at execution time, and
    deliberately not a local restatement of it. That is what makes narrowing here sound at
    all: the `filter` step still runs over whatever was read, so a plan-time answer that
    differs from the execution-time one can only be wrong. In particular it is a FULL outer
    join on the shared dimensions, never an inner one -- an inner join would drop the rows
    of an origin that shares no dimension with the conjunct's, making a trivially-true
    conjunct narrow the frame.

    Returns `surviving` unchanged -- too wide, which is always safe -- when the conjunct
    cannot be evaluated here at all: no origin in this frame is enumerable, or the conjunct
    names a column none of the enumerable ones carries (it belongs to a vector origin whose
    bands are not a plan-time fact, and that origin's own scan predicate is where it gets
    pushed instead).
    """
    parts: list[pl.DataFrame] = []
    positions: list[tuple[str, str]] = []
    available: set[str] = set()
    # Indexed, not `.get`: `surviving` is seeded from `metadata` and only ever unioned
    # afterward, so its keys are the enumerable origins and nothing else. A non-enumerable
    # one is absent from BOTH, which is how it ends up contributing no narrowing.
    for origin in sorted(surviving):
        frame = metadata[origin]
        carried = [
            name
            for name in frame.columns
            if name in standing[origin] and name not in PAIRING_KEY_EXCLUDED_COLUMNS
        ]
        column = f"{_ORIGIN_POSITION_PREFIX}{len(positions)}"
        positions.append((origin, column))
        parts.append(
            frame.with_row_index(column)
            .filter(pl.col(column).is_in(sorted(surviving[origin])))
            .select(column, *carried)
        )
        available |= set(carried)
    if not parts or not _referenced_names(conjunct) <= available:
        return surviving
    # `fill_null(False)` for the same reason `_joined_dimension_positions` needs it: the
    # full outer join leaves nulls wherever an origin has no row for a combination, and a
    # null comparison is "unknown", which must not be read as "keep".
    unified = unify_metadata_parts(parts).filter(pl.sql_expr(conjunct).fill_null(False))
    narrowed = dict(surviving)
    for origin, column in positions:
        narrowed[origin] = frozenset(unified[column].drop_nulls().to_list())
    return narrowed


def _surviving_value_predicate(
    metadata: pl.DataFrame, surviving: frozenset[int], dimension_columns: frozenset[str]
) -> list[pl.Expr]:
    """The scan predicate that narrows a vector origin's rows to the bands `surviving`
    leaves: one `IN`-list per dimension column whose surviving values are a PROPER subset
    of the ones the origin enumerated.

    Not the exact row set -- a per-column conjunction admits combinations no surviving band
    actually has -- but a superset of it, which is the safe direction, and one made of
    plain per-column `IN`s, which is what Parquet row-group statistics can actually skip on.
    The exact set is re-applied by the `filter` step itself over what comes back.

    A column whose values all survive contributes nothing: it would be a tautology on this
    scan, and leaving it out keeps the predicate readable on the plan.
    """
    parts: list[pl.Expr] = []
    kept = sorted(surviving)
    # `dimension_columns` is the origin's own dataset dimensions, which every column an
    # enumeration can carry is one of -- except `request.pipeline.
    # RESERVED_BAND_INDEX_COLUMN`, the stand-in for a projection that has no dimension at
    # all. That frame is one row high, so it can never be narrowed and never reaches here;
    # the guard is what says so rather than leaving it to that coincidence.
    for name in [name for name in metadata.columns if name in dimension_columns]:
        column = metadata[name]
        values = column.gather(kept).unique()
        if values.len() >= column.n_unique():
            continue
        parts.append(pl.col(name).is_in(values.sort().to_list()))
    return parts


def _merge_shared_origin_groups(
    groups: dict[str, tuple[str, ...]], crosswalks: dict[str, tuple[ZoneCrosswalk, ...]]
) -> None:
    """Fuse, in place, every band-wide join group that shares a raster origin with another
    into a single group carrying the union of their origins (and of their crosswalks).

    Per-join groups are what the walk can see; they are not yet what a READ can be. An
    origin appearing in two groups would be opened twice, once per group, and the two reads
    have different band counts whenever the groups differ -- so they chunk differently
    (`io.raster_io.raster_blocks_per_batch`) and their rows cannot be aligned positionally.
    `execute` keeps one frame per origin, so the second read silently replaces the first
    and the join that wanted the first fails `frame.assert_same_pixel_grid`. A chained
    pipeline provokes this on sight: `join [a, b]` then `join [<derived from a>, c]` puts
    `a` in both groups.

    Merging is transitive -- two groups sharing nothing directly still land together if a
    third bridges them -- because "same read" is an equivalence, not a pairwise favour.
    The union is what the ceiling check must then sum over (it is one read's band count),
    which is why this runs before it.

    Order is first-seen throughout, so the merged scan order stays a deterministic function
    of the pipeline rather than of set iteration.
    """
    parent: dict[str, str] = {}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for origins in groups.values():
        for name in origins:
            parent.setdefault(name, name)
        root = find(origins[0])
        for name in origins[1:]:
            parent[find(name)] = root

    # Accumulated only once every union above has settled, so `find` is stable and a
    # component's contents cannot be split across two roots.
    component_origins: dict[str, dict[str, None]] = {}
    component_crosswalks: dict[str, dict[ZoneCrosswalk, None]] = {}
    for output, origins in groups.items():
        root = find(origins[0])
        component_origins.setdefault(root, {}).update(dict.fromkeys(origins))
        component_crosswalks.setdefault(root, {}).update(dict.fromkeys(crosswalks.get(output, ())))

    for output, origins in groups.items():
        root = find(origins[0])
        groups[output] = tuple(component_origins[root])
        if component_crosswalks[root]:
            crosswalks[output] = tuple(component_crosswalks[root])


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
        name: _Validity(origin=name, value=True, grid_cell=True) for name in pipeline.inputs
    }

    vector_origins = [name for name, kind in origin_kind.items() if kind == "vector"]
    raster_origins = [name for name, kind in origin_kind.items() if kind == "raster"]
    vector_predicate_parts: dict[str, list] = {name: [] for name in vector_origins}
    vector_columns: dict[str, set[str]] = {name: {"grid_cell", "value"} for name in vector_origins}
    vector_columns_ok: dict[str, bool] = {name: False for name in vector_origins}
    point_sample_step: dict[str, int] = {}

    # The two dimension-pushdown passes' structural inputs, all recorded UNCONDITIONALLY
    # below -- before the `_incoming` gate, alongside `origin_sets` -- because none of them
    # is a pushdown-validity question. Which frames a frame was built from, which dimension
    # conjuncts were declared on it, and which of each contributing ORIGIN's dimensions are
    # still standing in it are facts about the ops; what a fan-out or a join costs is
    # answered later, by the passes themselves, as a union rather than as a refusal (see
    # the module docstring).
    #
    # Per origin, not per frame, and that distinction is load-bearing. `join [g, h]` where
    # `g` grouped `scenario_id` away and `h` kept it leaves the JOINED frame carrying
    # `scenario_id` -- from `h` alone. Reading the frame's set as one would then make a
    # `scenario_id` conjunct look pushable to `g`'s origin too, where it would narrow rows
    # the `group` was supposed to fold together and change that side's answer.
    origin_dimension_columns: dict[str, frozenset[str]] = {
        name: dataset_dimension_columns(selection) for name, selection in pipeline.inputs.items()
    }
    frame_origin_dimensions: dict[str, dict[str, frozenset[str]]] = {
        name: {name: columns} for name, columns in origin_dimension_columns.items()
    }
    frame_parents: dict[str, tuple[str, ...]] = {}
    frame_dimension_conjuncts: dict[str, tuple[str, ...]] = {}
    band_wide_join_origin_groups: dict[str, tuple[str, ...]] = {}
    band_wide_join_zone_crosswalks: dict[str, tuple[ZoneCrosswalk, ...]] = {}
    # Every vector-to-raster `project` whose bands can be enumerated before the read, mapped
    # to the origin they come from and the dimension names still standing at that point --
    # exactly what `io.catalog.resolve_vector_bands` needs. Recorded in the main walk rather
    # than a pass of its own because it is a PUSHDOWN-VALIDITY question, not a structural
    # one: `_incoming` returning `None` (a join or a fan-out upstream) is precisely the case
    # where no single origin's catalog can answer how wide this step's output will be.
    vector_band_projections: dict[int, tuple[str, frozenset[str] | None, str]] = {}
    # Which vector-to-raster `project` steps' bands land in the same band-wide frame as a
    # given join's -- the vector counterpart of `band_wide_join_origin_groups`. Recorded for
    # EVERY band-wide join, including one with no raster operand at all (that dict skips
    # those, having nothing to read), because two projections joined together still share
    # one frame's width and the ceiling is about that width.
    band_wide_join_vector_steps: dict[str, tuple[int, ...]] = {}

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
            frame_parents[op.output] = op.inputs
            # Intersected, not unioned, for an origin reaching this join down two operands
            # at once (a self-join, or a fan-out that rejoins): only a dimension BOTH
            # paths still carry can be narrowed at the one read they share.
            standing: dict[str, frozenset[str]] = {}
            for operand in op.inputs:
                for name, columns in frame_origin_dimensions[operand].items():
                    standing[name] = standing[name] & columns if name in standing else columns
            frame_origin_dimensions[op.output] = standing
            if frame_types[op.output].shape is Shape.band_wide:
                # Only the RASTER origins contribute bands to the shared read. A vector
                # origin reaches a band-wide join solely by way of a vector-to-raster
                # `project` (nothing else gives a vector input pixels), and that step's
                # crosswalk -- not the origin -- is what has to ride along on the same
                # read; `zone_crosswalks_upstream_of` is where it was recorded. Its bands
                # are not read at all (they are pivoted out of the long frame in memory),
                # but they are columns of the joined frame all the same, which is what
                # `band_wide_join_vector_steps` records them for.
                if zone_crosswalks_upstream_of[op.output]:
                    band_wide_join_vector_steps[op.output] = tuple(
                        crosswalk.step_no for crosswalk in zone_crosswalks_upstream_of[op.output]
                    )
                raster_union = tuple(name for name in union if origin_kind[name] == "raster")
                if raster_union:
                    band_wide_join_origin_groups[op.output] = raster_union
                    band_wide_join_zone_crosswalks[op.output] = zone_crosswalks_upstream_of[op.output]
            states[op.output] = None
            continue

        origin_sets[op.output] = origin_sets[op.input]
        frame_parents[op.output] = (op.input,)
        # A `group` is the only op that takes a dimension column AWAY from a frame; every
        # other one -- including a `project` in any direction, which changes which grid the
        # rows address, not which observation each row is of -- passes the sets through.
        frame_origin_dimensions[op.output] = (
            {
                name: columns & frozenset(op.by)
                for name, columns in frame_origin_dimensions[op.input].items()
            }
            if isinstance(op, GroupOp)
            else frame_origin_dimensions[op.input]
        )
        # Classified against the INCOMING frame's value columns, not the default
        # `("value",)`. On a paired frame an operand-named conjunct (`marsh > 0.5`) would
        # otherwise land in the dimension bucket and become a candidate for pushdown to the
        # reader -- where `marsh` is not a source column at all, but the name `join_long`
        # gave that operand's `value`.
        kinds = (
            classify_where_kinds(op.where, frame_types[op.input].value_columns)
            if isinstance(op, FilterOp)
            else None
        )
        if kinds is not None:
            frame_dimension_conjuncts[op.output] = tuple(kinds[PredicateKind.dimension])

        incoming = _incoming(op.input)
        if incoming is None:
            # A spatial filter sitting directly on a band-wide JOIN still point-samples
            # every raster origin that frame's pixels came from, even though no single
            # origin's `_Validity` survived the join to say so (`_Validity` tracks ONE
            # origin, and a join has several, so the gate above retires them all). The
            # origins are named anyway -- `band_wide_join_origin_groups[op.input]` is the
            # group `execution.execute._open_band_wide_join_group` opens as one shared
            # scan, and one shared scan takes one window, so recording the sample here is
            # what lets that window be the pixel's rather than the whole grid's. The frame
            # is band-wide by construction: only a band-wide join writes that dict.
            #
            # Both reference counts have to be one. The window is a property of the SHARED
            # READ, not of this filter's own frame, so anything else reading the group --
            # a second op off the join's output, or an origin feeding some other branch as
            # well as this join -- would silently be handed the narrowed read too, and see
            # one pixel where it asked for a grid. `_incoming` makes the same check for a
            # single origin; a joined group needs it made for every member.
            #
            # Only a filter DIRECTLY on the join output, deliberately. A `calculate` (or a
            # dimension-only `filter`) in between leaves the pixels alone and could in
            # principle carry the group forward, but a `group` or a `project` in between
            # does not, and distinguishing them is a second forward pass for a shape no
            # analysis here writes. Not narrowing is always correct, only slower.
            if (
                isinstance(op, FilterOp)
                and kinds[PredicateKind.spatial]
                and reference_count[op.input] == 1
            ):
                group = band_wide_join_origin_groups.get(op.input, ())
                if all(reference_count[origin] == 1 for origin in group):
                    for origin in group:
                        point_sample_step.setdefault(origin, op.step_no)
            states[op.output] = None
            continue

        if isinstance(op, ProjectOp):
            from_kind, to_kind = project_direction(frame_types[op.input], op.to)
            if from_kind == "vector" and to_kind == "raster":
                # The origin is necessarily a vector INPUT here, not merely a frame on a
                # vector grid: the only other way to be on one is a raster-to-vector
                # `project`, and that branch below has already retired this origin's state.
                # `None` rather than the frame's own set whenever nothing has narrowed it:
                # "every dimension this dataset has" is what `resolve_vector_bands` already
                # does by default, and saying it as a set would make the plan's record of
                # the enumeration depend on `dataset_dimension_columns` agreeing exactly
                # with the scan's schema, rather than merely on the two agreeing about what
                # a `group` took away.
                open_dimensions = frame_origin_dimensions[op.input][incoming.origin]
                vector_band_projections[op.step_no] = (
                    incoming.origin,
                    None if open_dimensions >= origin_dimension_columns[incoming.origin] else open_dimensions,
                    op.to,
                )
                states[op.output] = _Validity(
                    origin=incoming.origin, value=incoming.value, grid_cell=False
                )
            else:
                states[op.output] = None
            continue

        origin = incoming.origin
        kind = origin_kind[origin]

        if isinstance(op, FilterOp):
            if kind == "vector":
                for bucket in kinds.values():
                    for conjunct in bucket:
                        vector_columns[origin] |= _referenced_names(conjunct)
                # Nothing to do for a dimension conjunct: the symbolic pass below pushes it
                # for EVERY frame, including the ones this state does not survive to.
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
                #
                # Nothing to do for a dimension conjunct: band selection is the
                # band-identity pass's answer, recorded above and evaluated for every
                # frame, not only the ones this state survived to.
                #
                # A value conjunct is never pushed to a band-wide read regardless of
                # `incoming.value` -- see the module docstring.
                if kinds[PredicateKind.spatial]:
                    point_sample_step.setdefault(origin, op.step_no)
                    states[op.output] = None
                else:
                    states[op.output] = incoming
            continue

        if isinstance(op, CaseOp):
            # Arm by arm (a `calculate` is this op with no arms, so it takes the same
            # branch and simply has none to sweep). An arm's `when` may name a dimension
            # column as well as a value one, so the names it references are swept into the
            # origin's candidate column set exactly as a `filter`'s are -- what is NOT done
            # is pushing the conjunct itself down as a row predicate: an arm selects which
            # rows get REWRITTEN, not which rows exist, and pushing it to the reader would
            # drop the rows the arm does not apply to. What the op does do is rewrite the
            # value, which retires this origin's value validity.
            if kind == "vector":
                for arm in op.cases:
                    kinds_by_arm = classify_where_kinds(arm.when, frame_types[op.input].value_columns)
                    for bucket in kinds_by_arm.values():
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
            new_grid_cell = incoming.grid_cell and ("grid_cell" in by_set)
            states[op.output] = _Validity(origin=origin, value=False, grid_cell=new_grid_cell)
            continue

        if isinstance(op, UnpivotOp):  # pragma: no cover -- see below
            # Value validity retires, for the same reason a `case` retires it: after the
            # fold the running value is no longer the one column a pushed-down value
            # predicate would have been testing -- it is several stacked, discriminated by
            # `op.into`, and a predicate on `value` past here means "whichever operand this
            # row came from", which no single source read can answer. `grid_cell` validity
            # survives: the fold repeats each row's cell id rather than changing it.
            #
            # Nothing reaches this branch today, hence the pragma: an `unpivot` needs
            # several value columns, only a `join` produces those, and a join retires the
            # origin state outright -- so the `incoming is None` gate at the top of this
            # loop has already `continue`d for every unpivot there can currently be. It is
            # written out anyway rather than left to fall through to the unrecognized-op
            # raise below, so that the walk stays exhaustive over `Op` by construction
            # rather than by what today's pipelines happen to contain, and so that the day
            # some other op widens `value_columns` this says what an unpivot does to the
            # pushdown instead of erroring.
            states[op.output] = dataclasses.replace(incoming, value=False)
            continue

        # Unreachable: the branches above are exhaustive over the `Op` union, which is why
        # this line is excluded from coverage rather than exercised by a test.
        raise RequestError(f"step {op.step_no} has unrecognized op {getattr(op, 'op', None)!r}")  # pragma: no cover

    _merge_shared_origin_groups(band_wide_join_origin_groups, band_wide_join_zone_crosswalks)
    # A merge fuses two joins' crosswalks along with their origins, so the vector steps a
    # merged group covers are the merged tuple, not the one the walk recorded per join.
    for output, crosswalks in band_wide_join_zone_crosswalks.items():
        band_wide_join_vector_steps[output] = tuple(crosswalk.step_no for crosswalk in crosswalks)

    # **The symbolic dimension pass.** One forward sweep over the frame graph the walk
    # recorded, carrying per (frame, origin) the predicate that origin's own read may be
    # narrowed by. A conjunct declared on a frame ANDs into every origin of that frame
    # still carrying all of its columns; two paths meeting -- a join reached down both
    # sides, or a fan-out's branches recombined at the end -- OR, because one shared read
    # has to satisfy both and the superset is the safe direction.
    #
    # This is where a join and a fan-out stop being walls. Neither is a reason a DIMENSION
    # conjunct cannot reach an origin's read: `join_long` is an inner join, so
    # sigma_p(A join B) == sigma_p(A) join B for a `p` over A's own columns, and every
    # `filter` step still re-executes over whatever came back, so a narrowing that is
    # merely too WIDE costs time and nothing else. `_Validity` keeps its walls because
    # value and spatial pushdown do not have that second property in the same way (see
    # the module docstring).
    #
    # `None` is "nothing to narrow by" -- true -- which is why it absorbs an OR and is
    # replaced by an AND.
    narrowing: dict[str, dict[str, object]] = {name: {name: None} for name in pipeline.inputs}
    for op in ops:
        merged: dict[str, object] = {}
        for parent in frame_parents[op.output]:
            for origin, expr in narrowing[parent].items():
                if origin not in merged:
                    merged[origin] = expr
                elif merged[origin] is None or expr is None:
                    merged[origin] = None
                else:
                    merged[origin] = merged[origin] | expr
        for conjunct in frame_dimension_conjuncts.get(op.output, ()):
            names = _referenced_names(conjunct)
            for origin, standing in frame_origin_dimensions[op.output].items():
                # R2.9's "narrow only when certain": a conjunct naming anything this origin
                # does not still carry -- another operand's dimension, or one a `group`
                # collapsed on this origin's own side -- contributes nothing here rather
                # than being decomposed into the part that does apply.
                if origin in merged and names <= standing:
                    expr = pl.sql_expr(conjunct)
                    merged[origin] = expr if merged[origin] is None else merged[origin] & expr
        narrowing[op.output] = merged

    # Unioned over the TERMINAL frames -- the ones nothing else reads, so the answer leaves
    # through them. An origin that reaches none of them, or that reaches one having been
    # narrowed by nothing, keeps `None`.
    origin_dimension_predicate: dict[str, object] = {}
    for frame, per_origin in narrowing.items():
        if reference_count[frame]:
            continue
        for origin, expr in per_origin.items():
            if origin not in origin_dimension_predicate:
                origin_dimension_predicate[origin] = expr
            elif origin_dimension_predicate[origin] is None or expr is None:
                origin_dimension_predicate[origin] = None
            else:
                origin_dimension_predicate[origin] = origin_dimension_predicate[origin] | expr

    # The vector pushdown as the two passes so far leave it: the walk's value and spatial
    # conjuncts, plus the symbolic dimension predicate above. Resolved here rather than in
    # the `inputs` loop below because the band enumeration has to be made against exactly
    # what the scan will return -- the same predicate and the same projection, or it would
    # count combinations the read never produces. What is not final yet is the narrowing
    # one origin's bands take from ANOTHER origin's conjunct across a shared dimension;
    # that is the band-identity pass's answer, appended below once this enumeration has
    # given it something to answer with.
    for name in vector_origins:
        predicate = origin_dimension_predicate.get(name)
        if predicate is not None:
            vector_predicate_parts[name].append(predicate)
    vector_predicate_parts_final: dict[str, list] = {
        name: list(vector_predicate_parts[name]) for name in vector_origins
    }
    vector_projection: dict[str, list[str] | None] = {
        name: sorted(vector_columns[name]) if vector_columns_ok[name] else None for name in vector_origins
    }

    def _conjoined(parts: list) -> object:
        predicate = None
        for part in parts:
            predicate = part if predicate is None else predicate & part
        return predicate

    raster_band_metadata: dict[str, pl.DataFrame] = {
        name: resolve_raster_bands(pipeline.inputs[name], base_path=base_path) for name in raster_origins
    }

    # One query per projection that will actually become bands, and none at all for a
    # pipeline that stays long -- the gate is `vector_band_projections` itself, which is
    # only ever written at a vector-to-raster `project` step.
    vector_band_metadata: dict[int, pl.DataFrame] = {}
    projection_step_of: dict[str, int] = {}
    for step_no, (origin, dimensions, _to) in vector_band_projections.items():
        enumerated = resolve_vector_bands(
            pipeline.inputs[origin],
            base_path=base_path,
            columns=vector_projection[origin],
            predicate=_conjoined(vector_predicate_parts[origin]),
            dimensions=dimensions,
        )
        # `None` is "that dataset has nothing on disk to enumerate" -- see that function.
        # It joins the untraceable projections in meaning the same thing here: this step's
        # width is not a plan-time fact, so nothing below may treat it as one.
        if enumerated is not None:
            vector_band_metadata[step_no] = enumerated
            projection_step_of[origin] = step_no

    # **The band-identity pass.** The same sweep as the symbolic one, in the other
    # currency: per (frame, origin) BAND POSITIONS rather than a predicate, narrowed at
    # each dimension conjunct against the UNIFIED METADATA FRAME of every enumerable origin
    # feeding that frame. Two paths meeting union, for the same reason they OR above.
    #
    # It exists because the symbolic pass cannot express the thing this one is for: a
    # conjunct on ONE origin's dimension narrowing ANOTHER origin's bands, across a
    # dimension they share. `aggregate_type = 'mean'` names a column only `lnd_type` has,
    # yet the join pairs its bands with `elev`'s on `calendar_year`, so the `elev` bands
    # whose year no surviving `lnd_type` band matches are bands nothing will ever read.
    # That answer is only visible once both metadata frames are in hand, which is why this
    # runs after the enumeration and the symbolic one runs before it.
    origin_band_metadata: dict[str, pl.DataFrame] = {
        **raster_band_metadata,
        **{origin: vector_band_metadata[step] for origin, step in projection_step_of.items()},
    }
    surviving: dict[str, dict[str, frozenset[int]]] = {
        name: ({name: frozenset(range(origin_band_metadata[name].height))} if name in origin_band_metadata else {})
        for name in pipeline.inputs
    }
    for op in ops:
        united: dict[str, set[int]] = {}
        for parent in frame_parents[op.output]:
            for origin, positions in surviving[parent].items():
                united.setdefault(origin, set()).update(positions)
        carried = {origin: frozenset(positions) for origin, positions in united.items()}
        for conjunct in frame_dimension_conjuncts.get(op.output, ()):
            carried = _dimension_survivors(
                carried, origin_band_metadata, frame_origin_dimensions[op.output], conjunct
            )
        surviving[op.output] = carried

    # What each origin's read must actually cover: the union over every TERMINAL frame --
    # one nothing else reads, so the answer leaves through it. An origin that reaches no
    # terminal frame at all keeps every band it has; an origin that reaches one having
    # legitimately narrowed to nothing keeps nothing, which is an empty result, not an
    # error (`execution.band_ops.empty_band_wide` is what executes over it).
    required_bands: dict[str, frozenset[int]] = {}
    for frame, per_origin in surviving.items():
        if reference_count[frame]:
            continue
        for origin, positions in per_origin.items():
            required_bands[origin] = required_bands.get(origin, frozenset()) | positions
    for origin, metadata in origin_band_metadata.items():
        required_bands.setdefault(origin, frozenset(range(metadata.height)))

    # **The join-sibling floor.** `join_band_wide` aligns its operands POSITIONALLY, so
    # every operand has to hand it the same pixel row count -- and a zero-band operand
    # cannot, because "the original pixel count, zero bands" is the shape Polars has no
    # representation for (`execution/frame.py`'s module docstring). So an origin that
    # enters a band-wide join carrying bands, and would leave this pass carrying none, is
    # given one band back.
    #
    # That emptiness is this pass's own doing rather than the pipeline's. The conjunct
    # that caused it was declared AFTER the join, where it reads PAIRED metadata, and a
    # pairing can leave one operand with no surviving row while its sibling keeps some --
    # ask for a year only one operand's dataset has and that operand's rows are the only
    # ones the predicate keeps. Narrowing to exactly that is correct as an answer and
    # unreadable as a plan. The restored band costs one band of I/O and changes nothing
    # else: the `filter` step still re-executes over the joined frame and drops it there,
    # which is the same "too WIDE is slow and nothing worse" every other narrowing in this
    # pass rests on.
    #
    # An operand that reaches the join ALREADY empty -- a `filter` declared on its own
    # side, before the join -- is left alone. Execution re-derives that emptiness from
    # whatever is read, so restoring a band would only move the failure; `execute.
    # _open_band_wide_join_group`'s partial zero-band error is the right answer there and
    # says which operand was empty.
    for op in ops:
        if not isinstance(op, JoinOp) or frame_types[op.output].shape is not Shape.band_wide:
            continue
        for operand in op.inputs:
            for origin, positions in surviving[operand].items():
                if positions and not required_bands[origin]:
                    required_bands[origin] = frozenset({min(positions)})

    for name in raster_origins:
        raster_band_metadata[name] = raster_band_metadata[name][sorted(required_bands[name])]
    for origin, step_no in projection_step_of.items():
        metadata = vector_band_metadata[step_no]
        positions = required_bands[origin]
        if len(positions) < metadata.height:
            vector_band_metadata[step_no] = metadata[sorted(positions)]
            # The scan has to return the rows those bands are made of and no more, or the
            # projection would declare a band the data no longer fills. Derived from the
            # surviving positions rather than from the conjuncts, because the conjuncts
            # that got us here may have been declared past a join, where their own text is
            # not a predicate on this origin's rows at all.
            vector_predicate_parts_final[origin].extend(
                _surviving_value_predicate(metadata, positions, origin_dimension_columns[origin])
            )

    vector_predicate: dict[str, object] = {
        name: _conjoined(vector_predicate_parts_final[name]) for name in vector_origins
    }

    for join_output in dict.fromkeys([*band_wide_join_origin_groups, *band_wide_join_vector_steps]):
        origins = band_wide_join_origin_groups.get(join_output, ())
        # A step absent from `vector_band_metadata` contributes an unknown number of bands
        # rather than zero; nothing here can size it, which is the cost of the fan-out or
        # join that made it unenumerable, not something to pretend away with a guess.
        steps = tuple(
            step for step in band_wide_join_vector_steps.get(join_output, ()) if step in vector_band_metadata
        )
        total = sum(raster_band_metadata[name].height for name in origins) + sum(
            vector_band_metadata[step].height for step in steps
        )
        if total > ANALYSIS_BAND_CEILING:
            projected = f", plus the vector projection(s) at step(s) {steps}," if steps else ""
            raise BandLimitError(
                f"step {join_output!r} (join): the shared union of {origins}{projected} resolves to "
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

    joined_steps = {step for steps in band_wide_join_vector_steps.values() for step in steps}
    for step_no, metadata in vector_band_metadata.items():
        if step_no in joined_steps:
            continue
        if metadata.height > ANALYSIS_BAND_CEILING:
            origin, _dimensions, to = vector_band_projections[step_no]
            raise BandLimitError(
                f"step {step_no} (project): input {origin!r} resolves to {metadata.height} "
                f"bands when projected onto {to!r}, over the limit of "
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
            inputs[name] = ResolvedInput(
                selection=selection,
                columns=vector_projection[name],
                predicate=vector_predicate[name],
            )

    return ExecutionPlan(
        inputs=inputs,
        ops=ops,
        frame_types=frame_types,
        band_wide_join_scans=dict(band_wide_join_origin_groups),
        band_wide_join_zone_crosswalks={
            output: crosswalks for output, crosswalks in band_wide_join_zone_crosswalks.items() if crosswalks
        },
        vector_band_metadata=vector_band_metadata,
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
