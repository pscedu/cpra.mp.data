"""`project_vector_to_vector`: the one lowering this vertical slice gives `project`, for
the direction with no raster on either side of it -- a long frame's rows, re-addressed
from `op.input`'s grid onto `op.to` via a crosswalk join, then re-collapsed one row per
(dimensions, destination cell) with `op.aggregate`.

**Why the join mechanics below read like `analysis/vector.py::apply_project`.** They are
the same join, deliberately: `lf.join(crosswalk_lf, left_on="grid_cell",
right_on="src_id", how="inner")`, and then `pl.col("dst_id").alias("grid_cell")` to
re-address the result. That mechanic carries no numeric decision of its own -- it is just
how a crosswalk join is expressed in Polars -- and carrying it over rather than re-deriving it is the same
move `execution/long_ops.py` makes for its own four lowerings.

**What is NOT carried over, on purpose: the per-aggregate weighting.** The old engine
multiplies `value` by `scale_factor` before every aggregate except `count_by_value`, and
groups afterward -- which is only correct for `sum`. `min`/`max`/`count` do not describe
a per-unit-area rate, so scaling the values first changes the ANSWER, not just the
mechanism: a source cell contributing `scale_factor=0.1` to a destination has one full
`value` in that destination's group, not `0.1 * value`, and `min`/`max`/`count` should see
it that way. This is the one place a fresh implementation was required rather than a
faithful copy, and it does change results relative to the old engine for a project whose
crosswalk has any row with `scale_factor != 1.0` and an aggregate other than `sum`. The
five treatments, by aggregate:

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
is not one). Each combination's per-source-cell values are pivoted into one column, then
that small, per-source-cell-keyed wide table is joined onto the crosswalk's per-pixel zone
column with `maintain_order="left"` -- broadcasting each pixel to the value of the polygon
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

**Two deliberate gaps, recorded here so a later reader does not mistake either for an
oversight.**

*Wiring.* Neither direction is wired into `execution/execute.py`'s `_LOWERINGS`; both stay
in its `NOT_YET_IMPLEMENTED["project"]` set, matching `tests/api/test_execute.py`'s own
`TestLoweringTableAgreement` (a pre-existing, already-committed pin on that exact set) and
its `TestLoweringNotImplemented` (asserting both directions still raise
`LoweringNotImplementedError` through a real pipeline). The reason is structural, not just
that pinned test: `execute()`'s own frame-opening loop only ever populates `frames: dict[
str, LongFrame]` for `Shape.long`-typed inputs -- there is no existing mechanism in that
walk to open a band-wide input (a raster scan, or this module's own raster-touching
`project` output) into anything at all. Building that -- deciding when a raster is scanned,
what window it is scanned over, whether/how a chain of band-wide ops composes before
anything is materialized -- is a scan-policy decision that belongs to `execute()`, made
once at the point where frames are opened and shared by every band-wide lowering. Wiring
either direction from here would mean inventing that policy ad hoc, one lowering at a
time, rather than once, coherently, where it belongs.

*Signatures.* `project_vector_to_raster(frames, op, frame_types) -> BandWideFrame` takes
the identical input shape `project_vector_to_vector` already does (its input is a long
vector frame either way; only the OUTPUT shape differs by direction).
`project_raster_to_vector(metadata_df, op, frame_types) -> LongFrame` cannot take a
`frames` dict the same way -- nothing upstream of this module builds a dict of
already-opened band-wide inputs (see "wiring" above), so it takes the raw resolved raster
band metadata directly, the same input `io.raster.scan_raster` itself takes. This is a real
asymmetry, and whoever wires band-wide inputs into `execute()` has to reconcile it: that
work will need to decide how `execute()` obtains a `metadata_df` for a raster `op.input`
(presumably `io.catalog.resolve_raster_bands` against that input's own `InputSelection`)
and thread it to this function in place of a `frames[op.input]` lookup. Giving this
function the other shape now would only mean rewriting it once that policy exists.
"""

import polars as pl

from cpra.mp.data.analysis.execution.fold import fold_count_by_value, fold_group_by, should_fold_group_by
from cpra.mp.data.analysis.execution.frame import (
    BandWideFrame,
    LongFrame,
    add_reserved_band_index,
    band_column_names,
)
from cpra.mp.data.analysis.io.crosswalk_read import read_vector_crosswalk, resolve_raster_crosswalk_path
from cpra.mp.data.analysis.io.raster import scan_raster
from cpra.mp.data.analysis.request.errors import AggregateError
from cpra.mp.data.analysis.request.ops import FrameType, ProjectOp
from cpra.mp.data.analysis.request.pipeline import DATASET_SELECTION_COLUMNS
from cpra.mp.data.config import COUNT_BY_VALUE_MAX_DISTINCT

__all__ = ["project_vector_to_vector", "project_raster_to_vector", "project_vector_to_raster"]

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
    denominator = (
        pl.when(pl.col("value").is_not_null()).then(pl.col("scale_factor")).otherwise(None).sum()
    )
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


def _count_by_value(lf: pl.LazyFrame, crosswalk_lf: pl.LazyFrame, dim_cols: list[str], max_distinct: int) -> pl.LazyFrame:
    """The unbatched `count_by_value` lowering: one row per (dimensions, destination
    cell, distinct `value`), with the distinct value in `value_category` and its count in
    `value` -- a deliberately different schema from the old flat API's
    `request.expressions.count_by_value_frame` (which keeps the counted column's own name
    and calls the count `count`), because that schema drops null `value`s before both the
    cardinality probe and the grouped count, where this one keeps null as its own
    category: a destination whose only contributing cells are nodata should say so (one
    row, `value_category=null`), not vanish from the result. That old function stays
    exactly as it was -- it still serves the old engine's own `summarize` path, which
    expects its schema -- and is not reused or modified here.

    The probe runs on `lf`'s OWN `value` column, before the crosswalk join -- the join can
    only drop rows (a source cell overlapping no destination) or duplicate a row's already-
    counted value across several destinations, never introduce a distinct value that was
    not already present -- so probing here can only over-count relative to the true
    post-join, per-destination cardinality, never under-count it. See
    `fold.fold_count_by_value`'s docstring for the identical argument on the folded path.
    """
    n_unique = lf.select(pl.col("value").approx_n_unique()).collect(engine="streaming").item()
    if n_unique > max_distinct:
        raise AggregateError(
            f"count_by_value would produce approximately {n_unique} distinct values for "
            f"'value', over the limit of {max_distinct} -- count_by_value is meant for "
            "bounded categorical data (e.g. a land-type code), not continuous data"
        )
    joined = lf.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner").with_columns(
        pl.col("dst_id").alias("grid_cell")
    )
    return joined.group_by(*dim_cols, "grid_cell", pl.col("value").alias("value_category")).agg(
        pl.len().alias("value")
    )


def project_vector_to_vector(
    frames: dict[str, LongFrame], op: ProjectOp, frame_types: dict[str, FrameType]
) -> LongFrame:
    """The `("project", ("vector", "vector"))` lowering: re-aggregate `frames[op.input]`
    onto `op.to`'s cells via the crosswalk between `frame_types[op.input].grid` and
    `op.to`, with `op.aggregate` (never `None` here -- `request.aggregates.
    validate_project_aggregate` requires one for every direction but vector-to-raster,
    which has no lowering in this module at all).

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
        return _readdress(_count_by_value(lf, crosswalk_lf, dim_cols, COUNT_BY_VALUE_MAX_DISTINCT), op.to)

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
            joined.group_by(*group_key).agg(
                (pl.col("value") * pl.col("scale_factor")).sum().alias("value")
            ),
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
#: band name nor the reserved band-index column). Double-underscored so it cannot collide
#: with a dataset dimension riding along on `metadata_df`.
_ZONE_COLUMN = "__zone__"

#: The column an unpivot puts a band's own `b<i>` name in, and the key the per-band
#: dimension values are then joined back on. Internal to the reshape: dropped before the
#: frame is returned.
_BAND_IDENTITY_COLUMN = "__band__"


def _zonal_count_by_value(in_zone: pl.LazyFrame, names: list[str], max_distinct: int) -> pl.LazyFrame:
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
    request that passes here would pass a per-band probe too, never the reverse.
    """
    per_pixel = in_zone.unpivot(
        index=[_ZONE_COLUMN], on=names, variable_name=_BAND_IDENTITY_COLUMN, value_name="value"
    )
    n_unique = per_pixel.select(pl.col("value").approx_n_unique()).collect(engine="streaming").item()
    if n_unique > max_distinct:
        raise AggregateError(
            f"count_by_value would produce approximately {n_unique} distinct values for "
            f"'value', over the limit of {max_distinct} -- count_by_value is meant for "
            "bounded categorical data (e.g. a land-type code), not continuous data"
        )
    return per_pixel.group_by(
        _BAND_IDENTITY_COLUMN,
        pl.col(_ZONE_COLUMN).alias("grid_cell"),
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


def project_raster_to_vector(metadata_df: pl.DataFrame, op: ProjectOp, frame_types: dict[str, FrameType]) -> LongFrame:
    """The `("project", ("raster", "vector"))` lowering: zonal-aggregate a raster's own
    bands onto `op.to`'s cells via the raster-to-vector crosswalk between
    `frame_types[op.input].grid` (the raster grid) and `op.to`, with `op.aggregate` (never
    `None` here, for the same reason `project_vector_to_vector` names: `request.aggregates.
    validate_project_aggregate` requires one for every direction but vector-to-raster).

    `metadata_df` is the raster's already-resolved band list (`io.catalog.
    resolve_raster_bands`'s own return shape: one row per (file, band), `path`/`band` plus
    whatever dataset dimensions -- `variable`, `calendar_year`, `scenario_id`, ... -- ride
    along beside them) rather than an already-opened frame; see the module docstring's
    "signatures" note for why this direction's input cannot match the other three
    lowerings' `frames: dict[str, LongFrame]` shape.

    Every one of `metadata_df`'s columns besides `path`/`band` is carried onto the output
    long frame's dimension columns, attached per band via a join on band identity -- this
    is what keeps two bands differing only by, say, `calendar_year` from silently merging
    into one destination-cell group; see the module docstring's "group-then-unpivot" note
    for exactly how. `grid`, if present, is rewritten to `op.to` on the way out via the
    same `_readdress` the vector-to-vector direction uses -- true for the same reason: every
    `grid_cell` in the result addresses a destination (vector) cell now, and `variable`/
    `time_unit` remain true unchanged (still the same measurement, same time unit).

    Pixels whose crosswalk entry is nodata (masked by `scan_raster` itself, unconditionally
    -- see that function's own module docstring) belong to no destination cell and are
    excluded before the zonal `group_by` runs, never grouped under a spurious null-zone key.

    The crosswalk is read on the data bands' OWN read, as `scan_raster`'s `crosswalks=`
    parameter appends it -- one VRT, one chunking, so a pixel's zone and its values arrive
    on the same row by construction. Reading it with a second `scan_raster` call and
    combining the two positionally would be silently wrong for any raster big enough to
    batch; the module docstring gives the mechanism and the measurement.

    Raises `request.errors.NotFoundError` (via `io.crosswalk_read.
    resolve_raster_crosswalk_path`) if no raster-to-vector crosswalk artifact exists for
    this raster grid and `op.to`, and `request.errors.AggregateError` if
    `op.aggregate == "count_by_value"` and the data has more distinct values than
    `config.COUNT_BY_VALUE_MAX_DISTINCT` allows (the same rule, and the same error, the
    vector-to-vector direction applies).
    """
    current = frame_types[op.input]
    crosswalk_path = resolve_raster_crosswalk_path(current.grid, op.to)
    lf, out_metadata_df, _window = scan_raster(metadata_df, crosswalks=[(_ZONE_COLUMN, str(crosswalk_path))])
    names = band_column_names(out_metadata_df)
    in_zone = lf.filter(pl.col(_ZONE_COLUMN).is_not_null())

    if op.aggregate == "count_by_value":
        long_lf = _zonal_count_by_value(in_zone, names, COUNT_BY_VALUE_MAX_DISTINCT)
    else:
        zonal = in_zone.group_by(pl.col(_ZONE_COLUMN).alias("grid_cell")).agg(
            [_zonal_aggregate_expr(op.aggregate, name) for name in names]
        )
        long_lf = zonal.unpivot(index=["grid_cell"], on=names, variable_name=_BAND_IDENTITY_COLUMN, value_name="value")

    dim_columns = [name for name in out_metadata_df.columns if name not in ("path", "band")]
    band_identity = pl.DataFrame({_BAND_IDENTITY_COLUMN: pl.Series(names, dtype=pl.String)}).hstack(
        out_metadata_df.select(dim_columns)
    )
    with_metadata = long_lf.join(band_identity.lazy(), on=_BAND_IDENTITY_COLUMN, how="left").drop(
        _BAND_IDENTITY_COLUMN
    )
    return _readdress(with_metadata, op.to)


def _pivot_dimension_columns(lf: pl.LazyFrame) -> list[str]:
    """Every column of `lf` that should become a PIVOTED BAND dimension for
    `project_vector_to_raster` -- everything except `grid_cell`/`value` (same exclusion
    `_dimension_columns` above makes) and, in addition, `request.pipeline.
    DATASET_SELECTION_COLUMNS` (`variable`, `time_unit`, `grid`): those three name WHICH
    DATASET a row came from, not an observation dimension a band split should honor, the
    same reasoning `execution.long_ops.join_long` already applies before computing its own
    join key columns. Leaving them in would either pivot on a column that never varies
    (harmless but wasteful) or, worse, leave a stale source-grid value sitting in
    `metadata_out` once the frame addresses a raster grid instead of `lf`'s own vector one
    -- the same stale-column problem `_readdress` exists to fix on the long side, avoided
    here by never carrying the column onto a per-band identity in the first place.
    """
    return [
        name
        for name in lf.collect_schema().names()
        if name not in ("grid_cell", "value") and name not in DATASET_SELECTION_COLUMNS
    ]


def project_vector_to_raster(frames: dict[str, LongFrame], op: ProjectOp, frame_types: dict[str, FrameType]) -> BandWideFrame:
    """The `("project", ("vector", "raster"))` lowering: broadcast `frames[op.input]`'s
    rows onto `op.to`'s pixels via the raster-to-vector crosswalk between `op.to` (the
    raster grid) and `frame_types[op.input].grid` (the vector grid), with no aggregation --
    `request.aggregates.validate_project_aggregate` already requires `op.aggregate is None`
    for this direction and this function does not re-check it (aggregate legality is a
    request-layer concern, already enforced before a lowering ever runs).

    Distinct combinations of `frames[op.input]`'s own dimension columns (via
    `_pivot_dimension_columns`) become bands, sorted for a deterministic, re-run-stable
    order; see the module docstring's "pivot to bands, then broadcast" note for the full
    reasoning, including why `DATASET_SELECTION_COLUMNS` is excluded and why the crosswalk
    join uses `maintain_order="left"`.

    The one real raster read this direction performs is of the crosswalk artifact itself,
    not of any data raster -- there is no data raster on this direction's input side at
    all -- so the returned `ReadWindow.source` names the crosswalk file. Reads it as an
    ordinary single-band `scan_raster` source (not via that function's own `crosswalks=`
    parameter, which requires an already-nonempty data-band `metadata_df`): the crosswalk
    IS the sole band here, and `scan_raster`'s own per-band nodata masking (honored by
    default, `mask_nodata=True`) already nulls it out correctly, since the crosswalk
    GeoTIFF's own GDAL profile declares its nodata sentinel formally.

    Raises `request.errors.NotFoundError` (via `io.crosswalk_read.
    resolve_raster_crosswalk_path`) if no raster-to-vector crosswalk artifact exists for
    `op.to` and this vector grid.
    """
    lf = frames[op.input]
    current = frame_types[op.input]
    crosswalk_path = resolve_raster_crosswalk_path(op.to, current.grid)

    crosswalk_metadata = pl.DataFrame({"path": [str(crosswalk_path)], "band": [1]})
    zone_band_wide_lf, _zone_metadata, window = scan_raster(crosswalk_metadata)
    zone_lf = zone_band_wide_lf.select(pl.col("b0").alias("__zone__"))

    dim_cols = _pivot_dimension_columns(lf)
    # `group_by` (never `select`) even in the `dim_cols == []` case, so a frame with no
    # pivot dimension at all still reports a height of 1 rather than hitting the "zero
    # columns loses the row count" Polars gap `execution/frame.py`'s module docstring
    # describes -- the identical move `execution.band_ops.group_band_wide` makes for its
    # own `by == []` branch, on a different frame.
    combo_df = lf.group_by(dim_cols, maintain_order=True).agg(pl.len().alias("__n__")).collect()
    band_count = combo_df.height

    if dim_cols:
        combo_df = combo_df.sort(dim_cols)
        metadata_out = combo_df.select(dim_cols)
        band_index_df = combo_df.select(dim_cols).with_row_index("__band_idx__")
        keyed = lf.join(band_index_df.lazy(), on=dim_cols, how="inner")
    else:
        metadata_out = add_reserved_band_index(combo_df).drop("__n__")
        keyed = lf.with_columns(pl.lit(0, dtype=pl.UInt32).alias("__band_idx__"))

    wide = keyed.select("grid_cell", "__band_idx__", "value").collect().pivot(
        on="__band_idx__", index="grid_cell", values="value"
    )
    wide = wide.rename({str(k): f"b{k}" for k in range(band_count)})

    broadcast = zone_lf.join(
        wide.lazy(), left_on="__zone__", right_on="grid_cell", how="left", maintain_order="left"
    )
    out_lf = broadcast.select([f"b{k}" for k in range(band_count)])
    return out_lf, metadata_out, window
