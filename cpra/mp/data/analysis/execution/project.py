"""`project_vector_to_vector`: the one lowering this vertical slice gives `project`, for
the direction with no raster on either side of it -- a long frame's rows, re-addressed
from `op.input`'s grid onto `op.to` via a crosswalk join, then re-collapsed one row per
(dimensions, destination cell) with `op.aggregate`.

**Why the join mechanics below read like `analysis/vector.py::apply_project`.** They are
the same join, deliberately: `lf.join(crosswalk_lf, left_on="grid_cell",
right_on="src_id", how="inner")`, and then `pl.col("dst_id").alias("grid_cell")` to
re-address the result. That mechanic is not spec'd numeric behavior, just how a crosswalk
join is expressed in Polars, and carrying it over rather than re-deriving it is the same
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
"""

import polars as pl

from cpra.mp.data.analysis.execution.fold import fold_count_by_value, fold_group_by, should_fold_group_by
from cpra.mp.data.analysis.execution.frame import LongFrame
from cpra.mp.data.analysis.io.crosswalk_read import read_vector_crosswalk
from cpra.mp.data.analysis.request.errors import AggregateError
from cpra.mp.data.analysis.request.ops import FrameType, ProjectOp
from cpra.mp.data.config import COUNT_BY_VALUE_MAX_DISTINCT

__all__ = ["project_vector_to_vector"]

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
