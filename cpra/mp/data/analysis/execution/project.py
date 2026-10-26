"""The three lowerings `project` can dispatch to once its two operand grids are both
known: `project_vector_to_vector`, `project_raster_to_vector`, and
`project_vector_to_raster` -- keyed by `(from kind, to kind)` in `execute.py`'s registry
rather than by the incoming frame's shape, since `project` is the one op a shape alone
cannot route (see that module's docstring). Raster-to-raster is deliberately absent:
`request.ops.PROJECT_LOWERING_TABLE` does not name it, so no lowering for it exists here
either.

`project_vector_to_vector`, first: the direction with no raster on either side of it -- a
long frame's rows, re-addressed from `op.input`'s grid onto `op.to` via a crosswalk join,
then re-collapsed one row per (dimensions, destination cell) with `op.aggregate`.

**The join mechanic itself is the conventional one**, deliberately:
`lf.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner")`, and then
`pl.col("dst_id").alias("grid_cell")` to re-address the result. That mechanic carries no
numeric decision of its own -- it is just how a crosswalk join is expressed in Polars.

**Where this deliberately departs from how the crosswalk join was long done here: the
per-aggregate weighting.** The long-standing implementation multiplies `value` by
`scale_factor` before every aggregate except `count_by_value`, and groups afterward --
which is only correct for `sum`. `min`/`max`/`count` do not describe
a per-unit-area rate, so scaling the values first changes the ANSWER, not just the
mechanism: a source cell contributing `scale_factor=0.1` to a destination has one full
`value` in that destination's group, not `0.1 * value`, and `min`/`max`/`count` should see
it that way. This is the one place a fresh implementation was required rather than a
faithful copy, and it does change results for a project whose crosswalk has any row with
`scale_factor != 1.0` and an aggregate other than `sum`. The five treatments, by
aggregate:

- **`sum`**: weighted -- `sum(value * scale_factor)`, the allocated total.
- **`mean`**: weighted, but not the naive `sum(value * scale_factor) / sum(scale_factor)`
  -- see `_weighted_mean_exprs` below for why the denominator is restricted to rows where
  `value` is non-null, and `execution/fold.py::fold_group_by`'s `weight_column` parameter
  docstring for the same rule stated for the folded path.
- **`min`/`max`/`count`**: unweighted -- an ordinary `group_by` aggregate over the
  contributing rows' own `value`, `scale_factor` playing no part. `count` counts
  contributing source cells with a non-null `value`, once per destination they reach.
- **`count_by_value`**: unweighted, and reshaped -- see `_count_by_value` below.

Every one of the five folds through `execution/fold.py` (via `should_fold_group_by`,
exactly as `execution/long_ops.py::group_long` decides) when the destination cell key is
high-cardinality and nothing already narrowed the input -- `count_by_value` through
`fold.fold_count_by_value`, the other four through `fold.fold_group_by`. `sum`'s weight
enters through `chunk_transform` (its own `value` column is pre-multiplied by
`scale_factor` before an ordinary, unweighted `sum` folds it -- `sum(value *
scale_factor)` is already additive across chunks, with no numerator/denominator pair
needed); `mean`'s weight enters through `fold_group_by`'s own `weight_column` parameter
instead, since a mean is NOT simply additive and needs that pair carried through and
divided once at the end. `min`/`max`/`count` pass neither -- see `_join`'s call sites
below for exactly which of the two hooks each aggregate uses. `has_selective_filter=False`
is passed unconditionally, the same simplification `group_long` documents for the same
reason: nothing on the plan threads real filter history yet.

--------------------------------------------------------------------------------------

`project_raster_to_vector` and `project_vector_to_raster`: the two directions that touch
a raster grid on one side. Both read the raster-to-vector crosswalk artifact resolved by
`io.crosswalk_read.resolve_raster_crosswalk_path` -- a single-band GeoTIFF of destination
(raster-to-vector) or source (vector-to-raster) zone ids, co-registered with the raster
grid, ONE artifact serving both directions (see that function's docstring). Neither
direction weights a contributing raster pixel: a pixel carries the whole, implicit weight
of `1` (no area splitting the way a vector crosswalk's `scale_factor` expresses), so
neither reuses `_WEIGHTED_AGGREGATES`/`_weighted_mean_exprs` above -- those exist for a
crosswalk with a REAL per-row weight, which a raster-to-vector crosswalk simply does not
have.

**`project_raster_to_vector`: group-then-unpivot, not unpivot-then-group.** The band-wide
scan is per-PIXEL (potentially hundreds of thousands of rows) and per-band (one column per
surviving band); the destination is per-ZONE (typically far fewer rows) and per-band
(still one row per (band, zone) pair once reshaped to long). Aggregating first, while the
frame is still wide -- one `group_by("zone").agg([<aggregate>(name) for name in
band_names])` call, each band column reduced independently in the SAME group_by -- then
unpivoting the small, already-collapsed result, costs one reshape over `n_zones * n_bands`
rows. Unpivoting first would reshape `n_pixels * n_bands` rows before any reduction ever
ran, paying the full per-pixel cost a second time for no benefit -- the group_by would
still have to reduce the same total number of values either way. This is also why the
zone-identity join happens on the STILL-WIDE frame's own `group_by` key rather than folding
band dimension values into that key: a band's dimension identity (year, scenario, ...)
never needs to distinguish rows during the aggregate itself, since each band's own values
already live in its own column -- it is only needed once, when attaching it to the
now-long, now-small result (via a join on band identity, exactly `execution.band_ops.
sample_grid_cells_band_wide`'s technique, applied to a post-aggregation frame instead of
the raw pixel frame). Losing that attachment (or attaching it to the wrong band by
position) is exactly how bands differing only by dimension would end up silently merged;
`band_column_names(out_metadata_df)` and the aggregated column order are kept in lockstep
by construction (both indexed by the same `out_metadata_df` row order) rather than by
re-deriving one from the other.

**No fold for `project_raster_to_vector`'s zonal `group_by`.** `should_fold_group_by`/
`fold_group_by` (used by the vector-to-vector direction above) assume `lf_source()` can be
called fresh and `.slice(offset, step)`d per chunk, and are only a MEMORY fix (not also a
slowdown) when that slice pushes down to the underlying scan rather than re-reading
everything up to `offset` and discarding it (`fold.py`'s own docstring says so). A
`scan_raster`-backed `LazyFrame` cannot satisfy that: its IO-plugin `source(with_columns,
predicate, n_rows, batch_size)` generator (confirmed by reading `io/raster.py` directly)
honors only `n_rows`, never an arbitrary offset, so `.slice(offset, step)` on it can only
be satisfied by re-reading the raster from batch zero and discarding rows up to `offset`
every single chunk -- turning the fold into a slowdown, the opposite of what it exists for.
`scan_raster` already bounds READ-side memory on its own (`io/raster_io.py`'s
`batched_windows`/`raster_blocks_per_batch`), which is a different and already-solved
problem from the one folding addresses (bounding the GROUP-BY's accumulator memory for a
high-cardinality key). That second problem is real and NOT solved here: a destination
vector grid with enough distinct zones could still make this `group_by`'s own accumulator
large. Left as a known, deliberate limitation -- a raster-aware fold would need to
re-slice via bounded window re-scans (a `window=` argument to a fresh `scan_raster` call
per chunk), not `.slice(offset, step)` on an already-open one, and building that is
follow-on work this module does not attempt.

**The crosswalk rides on the data bands' OWN read (`scan_raster`'s `crosswalks=`), never a
second `scan_raster` call combined positionally.** This is a correctness requirement, not a
performance preference. A read large enough to be batched emits its pixels in an order that
depends on how it was chunked, and how it is chunked depends on the read's BAND COUNT: a
fixed buffer budget buys fewer blocks per batch the more bands each block carries
(`io/raster_io.py`'s `raster_blocks_per_batch`), and a batch covers a sub-rectangle of a
block row rather than a full-width strip. An N-band data read and a 1-band crosswalk read
of the identical window over the identical grid therefore emit the same pixels in two
DIFFERENT orders, and combining them positionally pairs each pixel's value with some other
pixel's zone -- while the result's shape, extent, row count and every geometric property of
its window all stay exactly right. Measured, at 1024x1024 with four data bands: 1044480 of
1048576 rows landed on the wrong pixel. One read has one chunking, so pixel identity is
shared by construction and no assertion is needed; `execution/frame.py`'s
`assert_same_pixel_grid` compares `batch_layout` precisely so that a future caller tempted
back into the two-read shape fails loudly instead of silently.

**Where that one read now happens: at open time, in `execute.py`, not in this module.**
`project_raster_to_vector` below does not call `scan_raster` itself and does not resolve a
crosswalk path -- `execute.py`'s own input-opening step does both, for any raster origin
`execution.plan.ResolvedInput.straddling_vector_grid` names, before the walk ever reaches a
`filter`/`calculate`/`group`/`join` between that origin and this lowering. That is what
makes this lowering safe to call on the SAME already-open, already-processed frame every
other band-wide lowering reads and writes: whatever a `calculate` rewrote or a `filter`
masked upstream is already sitting in `frames[op.input]`'s own `lf` by the time this
function runs, because it is the identical `lf` the rest of the walk has been threading
along -- there is no second, independent read left to silently discard that work. `
ZONE_COLUMN` (below) is the column that read's zone ids come back in, attached once, at the
origin, exactly like a `variable`/`time_unit` dimension column would be if this were a
vector scan.

The same rule binds the OTHER direction, for the same reason and by the same route.
`project_vector_to_raster` reads a crosswalk too -- it is the only raster read that
direction has -- and when its output is about to be joined band-wide against raster
operands read separately, an independent crosswalk scan puts it in a different pixel order
from theirs just as surely. So `execute.py` attaches that crosswalk to the join group's one
combined read as well (`execution.plan.ZoneCrosswalk` records which artifact; `ZoneSource`
below is how the resulting column arrives), and the lowering reads the crosswalk itself
only when there is no such join for its pixel order to have to agree with.

**`count_by_value` is the one aggregate that unpivots per pixel.** See
`_zonal_count_by_value` below: its accumulator is per (zone, distinct value), not the single
scalar per (zone, band) the other five collapse to while the frame is still wide, so the
group-then-unpivot order above has nothing to collapse it to. It reshapes first and groups
on a three-part key instead, and produces the same `value_category`/`value` output shape the
vector-to-vector direction's `_count_by_value` does.

**`project_vector_to_raster`: pivot to bands, then broadcast onto the crosswalk.** The
input is already a long vector frame (`frames[op.input]`, exactly like the vector-to-vector
direction's own input) -- there is no raster to scan for actual data, only the crosswalk,
which is why `ReadWindow.source` on this direction's output names the crosswalk file
itself rather than a data raster (see `scan_raster`'s own module docstring: a co-registered
crosswalk is an ordinary band of a VRT from the read's point of view, and here it is the
VRT's only band). Distinct combinations of the input's own dimension columns (excluding
`grid_cell`/`value`, and `request.pipeline.DATASET_SELECTION_COLUMNS` -- `variable`,
`time_unit`, `grid` -- the same three `execution.long_ops.join_long` already excludes
before computing a join's key columns, since they name WHICH DATASET a row came from, not
an observation dimension a band should be split on) become bands, sorted for a
deterministic, re-run-stable band order (Polars offers no ordering guarantee for a plain
`.unique()`/`.group_by()` otherwise -- `band_column_names`' positional-naming invariant
requires SOME fixed order, and "whatever order the source file happened to store rows in"
is not one). Each combination's per-source-cell values are pivoted into one column -- a
LAZY pivot, declaring the band list it already settled as `on_columns`, so the whole
direction is one unbroken `LazyFrame` chain rather than a Python round trip in the middle
of one. That is a continuity win and not a memory one: a pivot is still a pipeline breaker,
and the frame it materializes internally is the small per-source-cell table (one row per
vector cell, one column per band), never the pixel frame. Then that small,
per-source-cell-keyed wide table is joined onto the crosswalk's per-pixel zone column with
`maintain_order="left"` -- broadcasting each pixel to the value of the polygon
its zone id names, in one pass, with NO aggregation (`request.aggregates.
validate_project_aggregate` already requires `op.aggregate is None` for this direction,
enforced upstream; this function does not re-check it, matching this module's stance
elsewhere that legality already validated by the request layer is not re-validated here).
`maintain_order="left"` is load-bearing, not a style choice: a plain `.join()` does not
guarantee row order (`join_band_wide`'s own docstring makes the identical point for a
different join), and row order IS pixel identity for a band-wide frame -- an unordered join
here would silently scramble which pixel gets which broadcast value while leaving the
frame's shape looking perfectly fine. A pixel whose crosswalk entry is null (nodata) finds
no matching key in a left join by construction (Polars never matches a null join key), so
it comes back null on every band with no separate nodata branch needed.

--------------------------------------------------------------------------------------

**All three `project` lowerings share the one uniform registry shape,
`(frames, op, frame_types) -> <output frame>`, with no SIGNATURE adapter at the registry
boundary.** `project_raster_to_vector` takes `frames: dict[str, BandWideFrame]` exactly
like `project_vector_to_vector`/`project_vector_to_raster` do, and reads
`frames[op.input]` -- the SAME already-open, already-processed band-wide frame the walk
has been threading through every upstream `filter`/`calculate`/`group`/`join` -- rather
than raw, unopened band metadata. It does not call `scan_raster` and does not resolve a
crosswalk path itself; see "where that one read now happens" above for why that work
belongs to `execute.py`'s input-opening step instead, and what guarantees it gives this
function in return (the zone column, under `ZONE_COLUMN`, already sitting on
`frames[op.input]`'s own `lf`, attached at open time and carried through untouched by
every band-wide lowering in between). `execute.py` still wraps this one entry in a small
GUARD (not a signature adapter -- it calls this function with the identical arguments,
unmodified) for the one case below that is a dispatch-key ambiguity rather than an
argument-shape mismatch.

The one sub-case this function still cannot serve: a raster origin an earlier spatial
filter already sampled down to a long frame no longer has a band-wide `frames[op.input]`
to read at all (`ResolvedInput.band_metadata` describes the ORIGIN's own bands, not
whatever a later op did to them). `execute.py`'s dispatch keys on `frame_types[op.input].
shape` before ever reaching this function, and raises `LoweringNotImplementedError` for
that narrower case itself, rather than this function receiving a long frame with no zone
column to aggregate by and failing on a missing-column error instead.
"""

from typing import NamedTuple

import polars as pl

from cpra.mp.data.analysis.execution.fold import fold_count_by_value, fold_group_by, should_fold_group_by
from cpra.mp.data.analysis.execution.frame import (
    DOMAIN_COLUMN,
    RESERVED_BAND_INDEX_COLUMN,
    BandWideFrame,
    LongFrame,
    ReadWindow,
    add_reserved_band_index,
    band_column_names,
    drop_reserved_band_index,
    has_no_bands,
)
from cpra.mp.data.analysis.io.catalog import band_dimension_columns
from cpra.mp.data.analysis.io.crosswalk_read import read_vector_crosswalk, resolve_raster_crosswalk_path
from cpra.mp.data.analysis.io.raster import scan_raster
from cpra.mp.data.analysis.request.aggregates import validate_count_by_value_cardinality
from cpra.mp.data.analysis.request.ops import FrameType, ProjectOp
from cpra.mp.data.analysis.request.pipeline import ZONE_COLUMN
from cpra.mp.data.config import COUNT_BY_VALUE_MAX_DISTINCT

__all__ = [
    "ZONE_COLUMN",
    "ZoneSource",
    "project_raster_to_vector",
    "project_vector_to_raster",
    "project_vector_to_vector",
]

# Aggregates that read a `scale_factor`-weighted `value` -- see the module docstring for
# why the other three (`min`, `max`, `count`) do not, and `count_by_value` is handled
# entirely separately (`_count_by_value` below) because its output shape differs too.
_WEIGHTED_AGGREGATES = frozenset({"sum", "mean"})


def _dimension_columns(lf: pl.LazyFrame) -> list[str]:
    """Every column of `lf` except `grid_cell` and `value` -- the group key a `project`
    must preserve alongside the destination cell. `ProjectOp` has no field naming a
    different value column the way `GroupOp.of` does, so `"value"` is the only value
    column this lowering ever reads; a frame reaching here with more than one (an
    un-collapsed `join`'s output, addressed directly by a `project` with no intervening
    `calculate`) has no `"value"` column to find, and fails with Polars' own
    column-not-found error rather than a silent guess at which operand was meant.
    """
    return [name for name in lf.collect_schema().names() if name not in ("grid_cell", "value")]


def _readdress(lf: pl.LazyFrame, to_grid: str) -> pl.LazyFrame:
    """Rewrite the `grid` column, if the frame carries one, to the grid the rows are now
    addressed against.

    A vector scan materializes the dataset-selection fields (`variable`/`time_unit`/
    `grid`) as literal columns, so they ride along as ordinary columns and survive the
    group key below. `variable` and `time_unit` are still true after a project -- the same
    measurement, over the same time unit -- but `grid` is not: every `grid_cell` in the
    result is a DESTINATION cell id, and leaving the column naming the source grid makes
    the frame contradict itself in a way nothing downstream can detect. A reader
    (including whoever opens the written CSV) would resolve those cell ids against the
    wrong lattice and get a plausible, wrong answer rather than an error.

    Overwriting rather than dropping: the column is genuine provenance a caller can use,
    and the honest value for it after a re-addressing is the grid re-addressed onto. Note
    the frame TYPE already says exactly this (`request.ops`'s project typing sets the new
    grid), which is what makes the stale column a disagreement between the plan and the
    data rather than merely a missing update.
    """
    if "grid" not in lf.collect_schema().names():
        return lf
    return lf.with_columns(pl.lit(to_grid).alias("grid"))


def _weighted_mean_exprs() -> tuple[pl.Expr, pl.Expr]:
    """The (numerator, denominator) pair a weighted mean folds as -- divided once, at the
    very end, exactly like the unweighted mean `fold_group_by` already computes as (sum,
    count).

    The denominator is `sum(scale_factor)` restricted to rows where `value` is non-null,
    not a plain `sum(scale_factor)` over every contributing row. A source cell with a
    null `value` (nodata, or a value an earlier `calculate`/`filter` left unset) is
    skipped by the numerator already -- `value * scale_factor` is null there, and `.sum()`
    ignores it -- so counting its `scale_factor` in the denominator anyway would divide
    the same numerator by a LARGER weight than actually contributed to it, silently
    under-reporting the mean by exactly the null rows' share of the destination's total
    weight. A destination with every contributing `value` null therefore sums to `0.0` in
    both halves (Polars sums an empty/all-null selection to `0.0`, not null); dividing
    those directly gives `NaN`, not the "no data" answer this case should report, which is
    why both callers below guard the division rather than performing it directly.
    """
    numerator = (pl.col("value") * pl.col("scale_factor")).sum()
    denominator = pl.when(pl.col("value").is_not_null()).then(pl.col("scale_factor")).otherwise(None).sum()
    return numerator, denominator


def _finalize_mean(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Divide the `_weighted_mean_exprs` pair exactly once, guarding the all-null-group
    zero-over-zero case to null rather than `NaN` -- see `_weighted_mean_exprs`."""
    return lf.with_columns(
        pl.when(pl.col("__project_denom__") > 0)
        .then(pl.col("__project_num__") / pl.col("__project_denom__"))
        .otherwise(None)
        .alias("value")
    ).drop("__project_num__", "__project_denom__")


def _count_by_value(lf: pl.LazyFrame, crosswalk_lf: pl.LazyFrame, dim_cols: list[str]) -> pl.LazyFrame:
    """The unbatched `count_by_value` lowering: one row per (dimensions, destination
    cell, distinct `value`), with the distinct value in `value_category` and its count in
    `value`.

    Null is a category here, not a value to drop: neither the cardinality probe nor the
    grouped count filters nulls out, so a destination whose only contributing cells are
    nodata says so (one row, `value_category=null`) rather than vanishing from the
    result. `fold.fold_count_by_value` keeps null the same way on the folded path.

    The probe runs on `lf`'s OWN `value` column, before the crosswalk join -- the join can
    only drop rows (a source cell overlapping no destination) or duplicate a row's already-
    counted value across several destinations, never introduce a distinct value that was
    not already present -- so probing here can only over-count relative to the true
    post-join, per-destination cardinality, never under-count it. See
    `fold.fold_count_by_value`'s docstring for the identical argument on the folded path.

    The bound is `config.COUNT_BY_VALUE_MAX_DISTINCT`, read here rather than taken as a
    parameter: it is one deployment-wide configuration value, not a per-call decision, and
    there is nothing a caller could correctly pass but that. (`fold.fold_count_by_value`
    does take it, because it is a general fold utility with no config dependency of its
    own; this is a lowering, and reads the config it is governed by directly.)
    """
    n_unique = lf.select(pl.col("value").approx_n_unique()).collect(engine="streaming").item()
    validate_count_by_value_cardinality(n_unique, "value", COUNT_BY_VALUE_MAX_DISTINCT)
    joined = lf.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner").with_columns(
        pl.col("dst_id").alias("grid_cell")
    )
    return joined.group_by(*dim_cols, "grid_cell", pl.col("value").alias("value_category")).agg(pl.len().alias("value"))


def project_vector_to_vector(
    frames: dict[str, LongFrame], op: ProjectOp, frame_types: dict[str, FrameType]
) -> LongFrame:
    """The `("project", ("vector", "vector"))` lowering: re-aggregate `frames[op.input]`
    onto `op.to`'s cells via the crosswalk between `frame_types[op.input].grid` and
    `op.to`, with `op.aggregate` (never `None` here -- `request.aggregates.
    validate_project_aggregate` requires one for every direction but vector-to-raster,
    which broadcasts one value per target pixel rather than summarizing and so is required
    to carry none).

    Dimension columns (everything but `grid_cell`/`value`) are part of the group key, not
    just the destination cell -- a `calendar_year`/`scenario_id`/... column already on the
    frame stays a column after the project, and two rows that differ only in one of those
    stay in different groups. Dropping a dimension from the key here would silently merge
    across it (a year, a scenario, ...), which is never what a `project` step declares.

    The `grid` column a vector scan materializes is rewritten to `op.to` on the way out
    (`_readdress`), since after this op every `grid_cell` is a destination cell id.

    Raises `request.errors.NotFoundError` (via `io.crosswalk_read.read_vector_crosswalk`)
    if the two grids have no crosswalk built between them, and `request.errors.
    AggregateError` if `op.aggregate == "count_by_value"` and the data has more distinct
    values than `config.COUNT_BY_VALUE_MAX_DISTINCT` allows.
    """
    lf = frames[op.input]
    current = frame_types[op.input]
    crosswalk_lf = read_vector_crosswalk(current.grid, op.to)
    dim_cols = _dimension_columns(lf)
    key_names = [*dim_cols, "grid_cell"]
    aggregate = op.aggregate

    if aggregate == "count_by_value":
        if should_fold_group_by(key_names, has_selective_filter=False):
            source_lf = lf
            collapsed = fold_count_by_value(
                lambda: source_lf,
                [*dim_cols, pl.col("dst_id").alias("grid_cell")],
                "value",
                COUNT_BY_VALUE_MAX_DISTINCT,
                chunk_transform=lambda chunk: chunk.join(
                    crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner"
                ),
            )
            return _readdress(collapsed.lazy(), op.to)
        return _readdress(_count_by_value(lf, crosswalk_lf, dim_cols), op.to)

    def _join(chunk: pl.LazyFrame) -> pl.LazyFrame:
        return chunk.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner")

    # `fold_group_by`'s own `weight_column` parameter only changes the `mean` branch (it
    # carries the (numerator, denominator) pair that aggregate needs) -- it has nothing to
    # say about `sum`, whose weighting is instead folded into the `value` column itself,
    # inside `chunk_transform`, before the plain (unweighted) `sum` fold ever runs. Both
    # are correct decompositions of the same underlying rule (`sum(value * scale_factor)`
    # is additive across chunks on its own; the weighted mean is not, which is why it
    # needs the numerator/denominator carry instead), they just enter the fold through
    # different hooks.
    if aggregate == "sum":
        chunk_transform = lambda chunk: _join(chunk).with_columns(
            (pl.col("value") * pl.col("scale_factor")).alias("value")
        )
        weight_column = None
    elif aggregate == "mean":
        chunk_transform = _join
        weight_column = "scale_factor"
    else:
        chunk_transform = _join
        weight_column = None

    if should_fold_group_by(key_names, has_selective_filter=False):
        source_lf = lf
        collapsed = fold_group_by(
            lambda: source_lf,
            [*dim_cols, pl.col("dst_id").alias("grid_cell")],
            aggregate,
            weight_column=weight_column,
            chunk_transform=chunk_transform,
        )
        return _readdress(collapsed.lazy(), op.to)

    joined = _join(lf)
    group_key = [*dim_cols, pl.col("dst_id").alias("grid_cell")]

    if aggregate == "sum":
        return _readdress(
            joined.group_by(*group_key).agg((pl.col("value") * pl.col("scale_factor")).sum().alias("value")),
            op.to,
        )
    if aggregate == "mean":
        numerator, denominator = _weighted_mean_exprs()
        return _readdress(
            _finalize_mean(
                joined.group_by(*group_key).agg(
                    numerator.alias("__project_num__"), denominator.alias("__project_denom__")
                )
            ),
            op.to,
        )
    return _readdress(joined.group_by(*group_key).agg(getattr(pl.col("value"), aggregate)()), op.to)


#: The column the raster-to-vector crosswalk's zone ids come back in, named for
#: `scan_raster`'s `crosswalks=` parameter (which requires a name that is neither a `b<i>`
#: band name nor the reserved band-index column). Public (unlike `_BAND_IDENTITY_COLUMN`
#: below, which stays purely internal to this module's own reshape): `execute.py` attaches
#: this column at open time, on a raster origin `ResolvedInput.straddling_vector_grid`
#: names, so it needs the same name this module reads by. Double-underscored regardless,
#: so it still cannot collide with a dataset dimension riding along on `metadata_df`.
#:
#: Defined in `request/pipeline.py` and re-exported here, for the same reason
#: `frame.DOMAIN_COLUMN` is: `request.ops`' shape-typing walk records which frames carry it
#: (`FrameType.extra_columns`) so a `where` naming it classifies as a per-pixel predicate
#: rather than a band selection, and the request layer may not import this one.


class ZoneSource(NamedTuple):
    """An already-open crosswalk read for `project_vector_to_raster` to broadcast across,
    supplied by `execute.py` when this step's output is about to be joined band-wide with
    raster operands that were read separately -- see the module docstring's "the crosswalk
    rides on the data bands' OWN read" paragraph, which applies to this direction too.

    `lf` carries exactly one column, `ZONE_COLUMN`, one row per pixel in `window`'s own
    order; `window` is the `ReadWindow` that read produced, which is the whole point --
    passing it through unchanged is what makes the projected frame's `batch_layout`
    identical to its join partners' rather than merely similar.
    """

    lf: pl.LazyFrame
    window: ReadWindow


#: The column an unpivot puts a band's own `b<i>` name in, and the key the per-band
#: dimension values are then joined back on. Internal to the reshape: dropped before the
#: frame is returned.
_BAND_IDENTITY_COLUMN = "__band__"


def _zonal_count_by_value(in_zone: pl.LazyFrame, names: list[str]) -> pl.LazyFrame:
    """`count_by_value`'s zonal lowering: one row per (band, destination cell, distinct
    value), the distinct value in `value_category` and its count in `value` -- the same
    output shape `_count_by_value` gives the vector-to-vector direction, so a caller does
    not have to know which direction produced the frame to read it.

    This is the ONE aggregate that unpivots per PIXEL rather than per (zone, band). Every
    other aggregate collapses each band column to one scalar per zone while the frame is
    still wide, and only then reshapes the small result (see the module docstring's
    group-then-unpivot note). A per-(zone, distinct value) count has no such scalar to
    collapse to: its accumulator's width is the data's own cardinality, not one. Rather
    than build a second, differently-shaped wide path for it, this reshapes first and
    groups on a three-part key. The reshape is a streaming op and the group's accumulator
    is bounded by the probe below, so the cost is a larger intermediate row count, not a
    larger peak -- and this aggregate is for bounded categorical data (a land-type code)
    in the first place, which is exactly the case where the probe passes.

    Null is its own category, deliberately: a pixel whose value is nodata (or which an
    earlier step left unset) still lies in its zone, and a destination whose contributing
    pixels are all nodata should report that as one row with `value_category=null`, not
    vanish. Only pixels outside every zone are dropped, by the caller, before this runs.

    The cardinality probe reads the unpivoted `value` column across ALL bands at once,
    which can only over-count relative to any single band's own cardinality -- so a
    request that passes here would pass a per-band probe too, never the reverse. The bound
    it is checked against is `config.COUNT_BY_VALUE_MAX_DISTINCT`, read directly for the
    same reason `_count_by_value` reads it directly -- see that function's docstring.
    """
    per_pixel = in_zone.unpivot(index=[ZONE_COLUMN], on=names, variable_name=_BAND_IDENTITY_COLUMN, value_name="value")
    n_unique = per_pixel.select(pl.col("value").approx_n_unique()).collect(engine="streaming").item()
    validate_count_by_value_cardinality(n_unique, "value", COUNT_BY_VALUE_MAX_DISTINCT)
    return per_pixel.group_by(
        _BAND_IDENTITY_COLUMN,
        pl.col(ZONE_COLUMN).alias("grid_cell"),
        pl.col("value").alias("value_category"),
    ).agg(pl.len().alias("value"))


def _zonal_aggregate_expr(aggregate: str, name: str) -> pl.Expr:
    """The unweighted, per-zone reduction of one band column -- every contributing pixel
    counts once, never `scale_factor`-weighted (see the module docstring: a raster pixel's
    weight is the implicit, uniform `1` a vector crosswalk's real weight is not).

    `sum` alone needs an explicit no-contributor guard: Polars sums an empty/all-null
    selection to `0.0`, not null (confirmed directly, and already documented above for the
    vector-to-vector `mean` case), which would misreport "every contributing pixel was
    nodata" as a real zero total. `mean`/`min`/`max` already return null for an all-null
    group on their own (confirmed directly against this project's pinned Polars version),
    and `count` legitimately reports `0` for zero non-null contributors -- that is the
    right answer, not a stand-in for a missing one, exactly like the vector-to-vector
    `count` treatment above.
    """
    col = pl.col(name)
    if aggregate == "sum":
        return pl.when(col.count() == 0).then(None).otherwise(col.sum()).alias(name)
    if aggregate == "count":
        return col.count().alias(name)
    return getattr(col, aggregate)().alias(name)


def project_raster_to_vector(
    frames: dict[str, BandWideFrame], op: ProjectOp, frame_types: dict[str, FrameType]
) -> LongFrame:
    """The `("project", ("raster", "vector"))` lowering: zonal-aggregate a raster's own
    bands onto `op.to`'s cells via the raster-to-vector crosswalk between
    `frame_types[op.input].grid` (the raster grid) and `op.to`, with `op.aggregate` (never
    `None` here, for the same reason `project_vector_to_vector` names: `request.aggregates.
    validate_project_aggregate` requires one for every direction but vector-to-raster).

    `frames[op.input]` is the already-open `BandWideFrame` every upstream band-wide
    lowering has been reading and writing -- its `lf` already carries `ZONE_COLUMN`,
    attached once at open time by `execute.py` (see the module docstring's "where that one
    read now happens" note), and already reflects whatever `filter`/`calculate` ran
    upstream of this `project`. This function does not scan anything itself.

    Every one of `metadata_df`'s columns besides `path`/`band` and the reserved band-index
    column is carried onto the output long frame's dimension columns, attached per band via
    a join on band identity -- this is what keeps two bands differing only by, say,
    `calendar_year` from silently merging into one destination-cell group; the reserved
    index is excluded because it is a POSITION, not a dimension, and this is the last point
    it can be told apart from one (see `drop_reserved_band_index` at the call site below).
    See the module docstring's "group-then-unpivot" note
    for exactly how. `grid`, if present, is rewritten to `op.to` on the way out via the
    same `_readdress` the vector-to-vector direction uses -- true for the same reason: every
    `grid_cell` in the result addresses a destination (vector) cell now, and `variable`/
    `time_unit` remain true unchanged (still the same measurement, same time unit).

    Pixels whose crosswalk entry is nodata (masked at open time, the same unconditional
    masking `scan_raster`'s own module docstring describes) belong to no destination cell
    and are excluded before the zonal `group_by` runs, never grouped under a spurious
    null-zone key.

    A zero-band `metadata_df` (`has_no_bands`) returns an empty `pl.LazyFrame()` rather
    than running a `group_by`/`unpivot` over a frame with no band columns to reduce --
    the same "nothing left to describe" precedent `execution.frame.empty_band_wide` sets
    for the band-wide shape this input came from, carried over to this direction's own,
    differently-shaped (long) output.

    Raises `request.errors.AggregateError` if `op.aggregate == "count_by_value"` and the
    data has more distinct values than `config.COUNT_BY_VALUE_MAX_DISTINCT` allows (the
    same rule, and the same error, the vector-to-vector direction applies).
    """
    lf, metadata_df, _window = frames[op.input]
    if has_no_bands(metadata_df):
        return pl.LazyFrame()

    names = band_column_names(metadata_df)
    in_zone = lf.filter(pl.col(ZONE_COLUMN).is_not_null())

    if op.aggregate == "count_by_value":
        long_lf = _zonal_count_by_value(in_zone, names)
    else:
        zonal = in_zone.group_by(pl.col(ZONE_COLUMN).alias("grid_cell")).agg(
            [_zonal_aggregate_expr(op.aggregate, name) for name in names]
        )
        long_lf = zonal.unpivot(index=["grid_cell"], on=names, variable_name=_BAND_IDENTITY_COLUMN, value_name="value")

    # `drop_reserved_band_index` first: an upstream `group` with `by=[]` leaves the
    # reserved band-index column standing on `metadata_df` (that lowering's own docstring
    # says so), and every remaining column here becomes a DIMENSION column on the long
    # frame this function returns -- so without the drop, an internal positional index
    # rides out of the walk looking exactly like a dataset dimension a caller asked for.
    # `execute` cannot catch it afterward the way it does for a band-wide result: by then
    # the index is an ordinary column of a long frame, indistinguishable from a real one.
    carried = drop_reserved_band_index(metadata_df)
    dim_columns = [name for name in carried.columns if name not in ("path", "band")]
    band_identity = pl.DataFrame({_BAND_IDENTITY_COLUMN: pl.Series(names, dtype=pl.String)}).hstack(
        carried.select(dim_columns)
    )
    with_metadata = long_lf.join(band_identity.lazy(), on=_BAND_IDENTITY_COLUMN, how="left").drop(_BAND_IDENTITY_COLUMN)
    return _readdress(with_metadata, op.to)


def _enumerate_bands(lf: pl.LazyFrame, dim_cols: list[str]) -> pl.DataFrame:
    """`project_vector_to_raster`'s bands, derived from the DATA rather than from the plan
    -- the fallback for a projection `execution.plan.compile_plan` could not enumerate (see
    that function's `band_metadata` paragraph).

    One row per distinct combination of `dim_cols`, sorted by them, which is the band order
    the projection emits. `group_by` (never `select`) even in the `dim_cols == []` case, so
    a frame with no pivot dimension at all still reports a height of 1 rather than hitting
    the "zero columns loses the row count" Polars gap `execution/frame.py`'s module
    docstring describes -- the identical move `execution.band_ops.group_band_wide` makes for
    its own `by == []` branch, on a different frame.

    Executes `lf`. That is the cost this exists to avoid paying, and why the planner
    answers it instead wherever it can.
    """
    combo_df = lf.group_by(dim_cols, maintain_order=True).agg(pl.len().alias("__n__")).collect()
    if dim_cols:
        return combo_df.sort(dim_cols).select(dim_cols)
    return add_reserved_band_index(combo_df).drop("__n__")


def project_vector_to_raster(
    frames: dict[str, LongFrame],
    op: ProjectOp,
    frame_types: dict[str, FrameType],
    zone_source: ZoneSource | None = None,
    band_metadata: pl.DataFrame | None = None,
) -> BandWideFrame:
    """The `("project", ("vector", "raster"))` lowering: broadcast `frames[op.input]`'s
    rows onto `op.to`'s pixels via the raster-to-vector crosswalk between `op.to` (the
    raster grid) and `frame_types[op.input].grid` (the vector grid), with no aggregation --
    `request.aggregates.validate_project_aggregate` already requires `op.aggregate is None`
    for this direction and this function does not re-check it (aggregate legality is a
    request-layer concern, already enforced before a lowering ever runs).

    Distinct combinations of `frames[op.input]`'s own dimension columns (which columns
    those are is `io.catalog.band_dimension_columns`' rule, shared with the plan-time
    enumeration below) become bands, sorted for a deterministic, re-run-stable
    order; see the module docstring's "pivot to bands, then broadcast" note for the full
    reasoning, including why `DATASET_SELECTION_COLUMNS` is excluded and why the crosswalk
    join uses `maintain_order="left"`.

    **`band_metadata` is that combination list, decided at PLAN time.** `execution.plan.
    compile_plan` enumerates it with `io.catalog.resolve_vector_bands` -- the vector half of
    the same question `resolve_raster_bands` answers for a raster input -- and hands it down
    here, which is what lets this step's width be checked against `config.
    ANALYSIS_BAND_CEILING` before anything is read, and what removes the `collect()` that
    otherwise had to execute the frame just to learn how many columns it was about to have.
    The enumeration is made against the SAME pushdown the scan gets, so in the ordinary case
    it names exactly the combinations present; where a later step narrows the frame further
    than the reader was told (a `filter` that fan-out made unsafe to push, say), a declared
    combination can have no rows, and it comes back as an all-null band rather than silently
    vanishing and shifting every later band index. That is deliberate: a band's identity is
    a property of the request, as it already is on the raster side, not of which rows
    happened to survive.

    `band_metadata` is `None` for a direct call with no plan behind it, and is also ignored
    -- falling back to the execution-time enumeration -- if its columns do not match the
    incoming frame's pivot columns. The planner cannot always trace a projection back to a
    single origin (a `join` or a fan-out upstream ends that), and a mismatch is how it says
    so; the check is a schema comparison, so the fallback costs nothing to guard against.

    **Where the crosswalk read comes from: `zone_source`, when the walk has one to give.**
    There is no data raster on this direction's input side at all, so the crosswalk
    artifact is the only thing there is to read -- but if this step's output is about to
    be joined band-wide against raster operands, that read must be the SAME read those
    operands came from, for exactly the reason the module docstring's "the crosswalk rides
    on the data bands' OWN read" paragraph gives for the opposite direction: chunking
    follows band count, so an independent single-band crosswalk scan and an N-band data
    scan of the identical window emit the identical pixels in two different orders, and
    `execution.band_ops.join_band_wide` aligns its operands positionally. `execute.py`'s
    band-wide-join prelude therefore attaches this step's crosswalk to that join group's
    one combined `scan_raster` call (`execution.plan.ZoneCrosswalk` is what records which
    artifact, decided at compile time) and hands the resulting single-column frame down
    here as `zone_source` -- so the returned window IS the join partners' window, sharing
    their `batch_layout` by construction rather than by coincidence.

    `zone_source` is `None` for a step whose output is never joined with a separately-read
    raster -- a lone vector-to-raster projection sunk straight to a GeoTIFF, say. Nothing
    else's pixel order has to be matched there, so this function reads the crosswalk
    itself: an ordinary single-band `scan_raster` source (not via that function's own
    `crosswalks=` parameter, which requires an already-nonempty data-band `metadata_df`),
    since the crosswalk IS the sole band in that case, and `scan_raster`'s per-band nodata
    masking (honored by default, `mask_nodata=True`) already nulls it out correctly
    because the crosswalk GeoTIFF's own GDAL profile declares its nodata sentinel
    formally. Either way the returned `ReadWindow.source` names whatever the read actually
    opened -- the crosswalk alone for the standalone case, the join group's combined VRT
    sources for the shared one. `assert_same_pixel_grid` deliberately excludes `source`
    from its comparison, which is what lets the shared case pass it.

    **The returned frame carries one extra column, `frame.DOMAIN_COLUMN`.** Broadcasting
    across the crosswalk leaves every band null at a pixel the crosswalk does not map,
    which is indistinguishable -- from the bands alone -- from a pixel it does map onto a
    vector cell that simply has no data. The two are not the same answer, and
    `execution.band_ops.group_band_wide`'s `count` is where the difference shows, so the
    zone rides along as `DOMAIN_COLUMN` rather than being dropped here; see that constant's
    own comment. Callers report it in `extra_columns` like any other carried column.

    Raises `request.errors.NotFoundError` (via `io.crosswalk_read.
    resolve_raster_crosswalk_path`) if no raster-to-vector crosswalk artifact exists for
    `op.to` and this vector grid. Only reachable in the standalone case -- when
    `zone_source` is given, `execute.py` already resolved that path to perform the read.
    """
    lf = frames[op.input]
    current = frame_types[op.input]

    if zone_source is not None:
        zone_lf, window = zone_source
    else:
        crosswalk_path = resolve_raster_crosswalk_path(op.to, current.grid)
        crosswalk_metadata = pl.DataFrame({"path": [str(crosswalk_path)], "band": [1]})
        zone_band_wide_lf, _zone_metadata, window = scan_raster(crosswalk_metadata)
        zone_lf = zone_band_wide_lf.select(pl.col("b0").alias(ZONE_COLUMN))

    schema = lf.collect_schema()
    dim_cols = band_dimension_columns(schema.names())
    # `[RESERVED_BAND_INDEX_COLUMN]` is what a no-dimension enumeration looks like -- see
    # `_enumerate_bands`, and `execution/frame.py`'s module docstring for why the column
    # has to be there at all.
    if band_metadata is not None and band_metadata.columns == (dim_cols or [RESERVED_BAND_INDEX_COLUMN]):
        metadata_out = band_metadata
    else:
        metadata_out = _enumerate_bands(lf, dim_cols)
    band_count = metadata_out.height

    if dim_cols:
        band_index_df = metadata_out.with_row_index("__band_idx__")
        keyed = lf.join(band_index_df.lazy(), on=dim_cols, how="inner")
    else:
        keyed = lf.with_columns(pl.lit(0, dtype=pl.UInt32).alias("__band_idx__"))

    # `LazyFrame.pivot` takes `on_columns` as a REQUIRED argument where the eager form
    # discovers them from the data: a lazy pivot cannot learn its own output schema by
    # executing, so the caller has to declare it -- which is exactly the band list settled
    # above. Declaring it is also what makes the absent-band rule fall out rather than be
    # patched on afterwards: a declared index with no rows comes back as an all-null column
    # of `value`'s own dtype, in declared position, instead of vanishing and shifting every
    # later band (see the docstring's paragraph on `band_metadata`). `pivot` names its
    # columns after the `on` VALUES, hence the `str(k)` -> `b{k}` rename.
    #
    # `maintain_order=False` (the lazy default, stated here because the eager default is
    # the opposite): this frame is only ever the RIGHT side of the `maintain_order="left"`
    # join below, so its own row order never reaches the result and there is no reason to
    # pay for it.
    wide = keyed.select("grid_cell", "__band_idx__", "value").pivot(
        "__band_idx__",
        on_columns=list(range(band_count)),
        index="grid_cell",
        values="value",
        maintain_order=False,
    )
    wide = wide.rename({str(k): f"b{k}" for k in range(band_count)})

    broadcast = zone_lf.join(wide, left_on=ZONE_COLUMN, right_on="grid_cell", how="left", maintain_order="left")
    # The zone itself rides along as `DOMAIN_COLUMN` -- see that constant's own comment.
    # It is the zone COLUMN, not a derived boolean, because null-vs-not is the entire
    # question and the zone already answers it: a pixel the crosswalk raster's nodata
    # covers has no zone, and so was never in the domain this projection can speak about.
    # Kept distinct from `ZONE_COLUMN`, which the raster-to-vector direction carries for a
    # different purpose (a groupable key, not a domain mask) and which a single pipeline
    # can legitimately carry at the same time as this one.
    out_lf = broadcast.select([*(f"b{k}" for k in range(band_count)), pl.col(ZONE_COLUMN).alias(DOMAIN_COLUMN)])
    return out_lf, metadata_out, window
