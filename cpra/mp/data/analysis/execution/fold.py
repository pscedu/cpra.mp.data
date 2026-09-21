"""Bounded-memory grouped aggregation over a row source too large for one `group_by`.

Measured on the real damage file: `group_by("grid_cell").sum("value")` over 1.04B rows
(126,174 distinct `grid_cell` values) peaks at ~9.8 GB, on a 15 GB VM, even under the
streaming engine and regardless of `low_memory`/chunk-size tuning -- the cost is
in-flight input buffering inside the group-by operator, which scales with the group
key's cardinality, not with the hash table itself (~75 KB per group is far too much for
a `sum` accumulator). The same query with a selective dimension filter, or a low-
cardinality group key, is cheap already (0.1 s / 285 MB; 1.86B rows -> 1878 groups in
338 MB) -- so the fix bounds the CLIFF case without taxing the cheap one.

The fix: cut the input by row offset instead of asking one `group_by` to see it all.

    n_rows = lf_source().select(pl.len()).collect(engine="streaming").item()
    acc = None
    for offset in range(0, n_rows, step):
        chunk = lf_source().slice(offset, step)      # slice FIRST -- it pushes to row groups
        chunk = chunk_transform(chunk)               # any per-chunk filter/join, AFTER the slice
        part = chunk.group_by(keys).agg(<partial exprs>).collect(engine="streaming")
        acc = part if acc is None else pl.concat([acc, part]).group_by(keys).agg(<merge exprs>)
    <finalize (mean's division)>

`lf_source` is called fresh for every chunk and must be the bare, filter/join-free source
`.slice()` lands against -- see `fold_group_by`'s docstring for why the fold is only a
memory fix (not also a slowdown) when slice sits directly on the scan. A caller that needs
a filter or join to run too (the crosswalk reaggregation site does) supplies
`chunk_transform`, applied to each already-sliced chunk -- the same slice-then-filter
order `vector._scale_and_join` uses, and for the same reason: a join or
row-dropping filter placed BEFORE the slice would size the slice against the join's
(potentially larger, fanned-out) output instead of against the source rows this module's
`step` default was actually measured against. Every `.collect()` in this module forces
`engine="streaming"`: the in-memory engine measured 1.6-16x worse on every comparison in
the same investigation, including the cheap low-cardinality case.

Restricted to DECOMPOSABLE aggregates -- `sum`, `min`, `max`, `count`, and `mean` carried
as (sum, count) and divided once at the end, never as a mean of per-chunk means, which is
simply a different (wrong) number whenever chunks have unequal contributor counts. Anything
else (median, quantiles, `n_unique`, ...) has no correct partial/merge decomposition and is
refused before any file is read, by `validate_decomposable`/`FoldError` -- an ordinary
`ValueError` subclass, not `HTTPException`, matching `expressions.py`'s split between
library code (plain values in, ordinary exceptions out) and `op_error_boundary` (the
translation layer that maps a caught exception to a status code).

This module also owns the plan-time ROUTING decision (`should_fold_group_by`): folding is
worth its overhead (an extra `pl.len()` scan, plus re-grouping the accumulator after every
chunk) only for the request shapes that would otherwise hit the cliff. See that function's
docstring for the exact rule and the reasoning behind it.
"""

from collections.abc import Callable, Sequence
from typing import Any

import polars as pl

from cpra.mp.data.analysis.request.aggregates import DECOMPOSABLE_AGGREGATES
from cpra.mp.data.analysis.request.errors import AggregateError
from cpra.mp.data.config import TABULAR_FOLD_STEP

# Group-by columns whose presence, on its own, is treated as a high-cardinality signal --
# see `should_fold_group_by`. This is a structural signal (the column's IDENTITY), not a
# measurement of the actual data (Dataset cannot know a request's true group cardinality
# without a scan it does not want to pay for on the cheap path) -- `grid_cell` is exactly
# the column the measured cliff is keyed on (`clara_grid_cell_v001`: 126,174 distinct
# values), and every OTHER dimension column in this package is a small, catalog-bounded
# enum (`calendar_year`, `scenario_id`, ...), so it is currently also the only column that
# needs to be here. Extend this set if a future grid or dimension is added whose distinct
# value count can reach the same order of magnitude.
HIGH_CARDINALITY_GROUP_COLUMNS = frozenset({"grid_cell"})

# Internal accumulator column names for the `mean` carry. Prefixed/suffixed so they can
# never collide with a real dimension or value column name.
_SUM_ALIAS = "__fold_sum__"
_COUNT_ALIAS = "__fold_count__"


class FoldError(ValueError):
    """A fold was asked to do something it cannot do correctly -- a non-decomposable
    aggregate, or a non-positive step. An ordinary `ValueError` subclass, never
    `HTTPException`: `op_error_boundary` is the only place that translates this into a 400
    (see the module docstring, and `expressions.ExpressionError` for the established
    pattern).
    """


def validate_decomposable(aggregate: str) -> None:
    """Raise `FoldError` unless `aggregate` has a correct partial/merge decomposition.

    Checked before any file is read -- the same "reject at plan time, not deep inside a
    streaming collect" discipline `expressions.py` uses for value predicates.
    """
    if aggregate not in DECOMPOSABLE_AGGREGATES:
        raise FoldError(
            f"aggregate {aggregate!r} cannot be computed by an offset-batched fold -- only "
            f"{sorted(DECOMPOSABLE_AGGREGATES)} decompose into per-chunk partials that merge "
            "into the same result a single group_by would produce; median, quantiles, "
            "n_unique, and similar aggregates need a different algorithm, not this fold"
        )


def should_fold_group_by(
    group_by_keys: Sequence[Any],
    *,
    has_selective_filter: bool,
    high_cardinality_columns: frozenset[str] = HIGH_CARDINALITY_GROUP_COLUMNS,
) -> bool:
    """Decide whether a `group_by(*group_by_keys)` should run through the fold.

    Pure function over plain values -- no analysis object, no Polars schema inspection --
    so the rule is unit-testable on its own and reusable outside the API. Both arguments
    are exactly what a caller can know at plan time, before any file is read:

    - `group_by_keys`: the columns the result will actually be grouped by, as plain
      strings (a caller building an aliased Polars expression -- e.g.
      `pl.col("dst_id").alias("grid_cell")` for a crosswalk reaggregation -- passes the
      OUTPUT name it resolves to, "grid_cell" in that example, not the expression
      object; a non-string entry, such as the literal `True` Polars accepts as a
      "group everything into one group" key, is simply not a high-cardinality column
      name and is harmless to include if a future caller ever passes one -- see
      `fold_group_by`'s docstring for why one isn't produced by this package's own
      callers today).
    - `has_selective_filter`: whether the request already carries something that shrinks
      the row count reaching this group_by well below "the whole dataset" -- a pushed-down
      dimension filter (Hive-partition-level: `calendar_year=[...]`, `scenario_id=[...]`,
      ...) or a specific `summarize_grid_cell` destination selection (narrows the
      crosswalk before `apply_summarize`'s own group_by runs). `grid_cell` itself is
      deliberately NOT one of these any more: it narrows the OUTPUT, applied terminally
      after every group_by in the walk has already run, so by the time any group_by
      reaches this function it has not dropped a single row yet. Measured case: the identical high-cardinality group-by with a
      genuinely selective filter already runs in 0.1 s / 285 MB, so folding it too would
      only add overhead for no benefit.

    The rule: fold if and only if the key includes a high-cardinality column AND there is
    no selective filter already bounding the input. This is deliberately conservative in
    both directions the task cares about differently:

    - A request that doesn't need folding but gets it anyway (e.g. this package's own
      small fixture data, or a real request that happens to filter on something this
      function doesn't recognize as "selective") pays one extra `pl.len()` scan and a
      re-group per chunk -- cheap, and the fold is numerically exact regardless of input
      size (a single-chunk fold IS the plain group_by, just reached through one extra
      `.slice()`).
    - A request that needs folding but doesn't get it risks the full 9.8 GB cliff.

    Given that asymmetry, this function is intentionally simple rather than clever: it
    does not attempt to estimate actual cardinality (which would need a scan of its own,
    defeating the point of a cheap plan-time check) or account for every way a filter
    might shrink row count (e.g. `where`'s value conjunct, which MASKS values to null
    rather than dropping rows when the request is aggregating -- see
    `ops.FilterOp.mask_value` -- is deliberately not treated as selective here -- it
    doesn't change how many rows reach the group-by at all).
    """
    if has_selective_filter:
        return False
    return any(isinstance(key, str) and key in high_cardinality_columns for key in group_by_keys)


def fold_group_by(
    lf_source: Callable[[], pl.LazyFrame],
    keys: Sequence[Any],
    aggregate: str,
    value_column: str = "value",
    weight_column: str | None = None,
    step: int | None = None,
    chunk_transform: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
) -> pl.DataFrame:
    """Compute `<lf>.group_by(*keys).agg(<aggregate> of value_column)`, bounding peak
    memory by never handing the underlying `group_by` more than `step` rows at once.

    `weight_column`, when given, only changes what `aggregate == "mean"` carries through
    the fold: instead of (sum of `value_column`, count of non-null `value_column`) it
    carries (sum of `value_column * weight_column`, sum of `weight_column` restricted to
    rows where `value_column` is non-null) -- the area-weighted mean a vector-to-vector
    `project` needs (`execution/project.py`), where each contributing row's weight is the
    fraction of its source cell's area landing in this destination. The restriction to
    non-null-`value_column` rows on the SECOND half of that pair is not optional: a plain
    `sum(weight_column)` would count the weight of a null-valued row the numerator already
    skipped, understating the true per-unit-weight average by folding in weight that
    contributed nothing to the numerator. Both this restriction and the eventual
    zero-denominator guard below (`otherwise(None)`, not a `NaN` from dividing two zero
    sums) must hold per CHUNK, not just once at the end, or a chunk boundary landing
    inside an all-null-valued group would silently reproduce the bug this restriction
    exists to avoid. Ignored for every other aggregate -- `sum`/`min`/`max`/`count` on a
    project are unweighted by design (see that module's docstring for why only `sum` and
    `mean` take a weight), and unused entirely when `weight_column` is `None`, which
    reproduces this function's exact prior (unweighted-mean) behavior bit for bit.

    `lf_source` is called fresh for every chunk (it must return an independent,
    not-yet-collected `LazyFrame` each time -- a plain closure over an already-built
    `LazyFrame`, e.g. `lambda: base_lf`, is exactly right, since a `LazyFrame` is only a
    plan and re-collecting it re-runs that plan from its scan) and must be the BARE
    source, with no row-dropping `.filter()` or `.join()` already chained onto it --
    `.slice(offset, step)` is the first operation this function applies on top of it.
    This is what keeps the fold a pure memory fix rather than also a slowdown: a slice
    placed directly against a `scan_parquet` call pushes down to row-group level, so each
    chunk decodes roughly `step` rows regardless of how many chunks came before it. A
    slice placed AFTER a row-dropping filter or join cannot do this -- each `.collect()`
    would need to re-derive its offset from scratch -- and worse, it sizes each chunk
    against the FILTERED/JOINED row count rather than the source row count `step`'s
    default was measured against, which for a fan-out join can silently make chunks much
    larger than intended.

    `chunk_transform`, when given, is applied to each chunk immediately AFTER the slice
    and before `group_by` -- the place for a per-chunk filter or join that must run (the
    crosswalk reaggregation site's join to a destination zone, scaled by
    `scale_factor`), matching `vector._scale_and_join`'s slice-then-filter order.
    `n_rows` (and therefore the chunk boundaries) is always computed from `lf_source()`
    alone, before `chunk_transform` runs -- a caller whose transform can multiply row
    count (a many-to-one crosswalk join) still gets chunks sized off the SOURCE side,
    which is what actually bounds the underlying scan's memory. A caller building
    `lf_source`/`chunk_transform` around an upstream operation that is itself expensive
    regardless of chunking should only route through `should_fold_group_by` when that
    operation is known to be cheap (see `vector.py`'s two call sites).

    `keys` is passed to `group_by` as-is for the FIRST (partial) pass -- it may contain
    plain column names or Polars expressions (e.g. an aliased crosswalk destination
    column). The MERGE pass, on every chunk after the first, instead re-derives the
    group columns from the collected partial's own schema (everything except the
    value/accumulator columns) -- this sidesteps needing to know how a non-string key or
    an aliased expression resolves to a concrete column name (a bare Polars literal key,
    though this package's own callers no longer produce one -- see
    `stages.resolve_group_by_columns` -- would resolve to a column literally named
    "literal"), since after one `.collect()` there is no more ambiguity to resolve.

    Decomposable aggregates only -- `validate_decomposable` raises `FoldError` (an
    ordinary exception, before any file is read) for anything else. `count` is the one
    aggregate whose PARTIAL and MERGE steps differ: each chunk's partial count is a
    genuine per-key count of non-null contributors in that chunk (Polars' `Expr.count`
    semantics, matching the package's existing `getattr(pl.col("value"), "count")()` call
    sites), but merging two partial counts must SUM them -- calling `.count()` again on
    the merge would count how many CHUNKS contributed a row for a key, not how many
    source rows did. `mean` is carried as (sum, count) through every chunk and divided
    exactly once, in `finalize`, specifically to avoid a mean of per-chunk means (wrong
    the moment two chunks contribute unequal counts for the same key -- the final,
    possibly-short chunk is the common way this happens).

    No separate "no-contributor identity" substitution is needed here (contrast the
    band-wide horizontal fold, which must invent a null identity because every cell has a
    positional slot whether or not a band contributed to it): a vertical `group_by` simply
    omits a key from a chunk's partial output when that chunk has zero rows for it, so
    "absent from this chunk" and "present with a zero contribution" are never conflated by
    construction. What IS worth being deliberate about -- and is pinned by this module's
    tests -- is that Polars' own vertical aggregates already skip nulls exactly where the
    package's existing (unfolded) aggregation relies on that (`sum`/`min`/`max`/`count`
    of an all-null-in-this-chunk group), so merging via those same aggregate methods
    (never a raw `+`/`min`/`max` operator) reproduces the unfolded result exactly,
    including for a group that is entirely null-valued across the whole dataset.

    An empty chunk (every row filtered out of that slice) is not a special case: its
    partial `group_by` naturally produces zero rows with the correct schema, and
    `pl.concat` with an empty, schema-matching frame is a no-op.
    """
    validate_decomposable(aggregate)
    if step is None:
        step = TABULAR_FOLD_STEP
    if step <= 0:
        raise FoldError(f"fold step must be positive, got {step}")

    n_rows = lf_source().select(pl.len()).collect(engine="streaming").item()

    acc = None
    key_columns = None
    accumulator_columns = {_SUM_ALIAS, _COUNT_ALIAS} if aggregate == "mean" else {value_column}

    # `range(0, max(n_rows, 1), step)` rather than `range(0, n_rows, step)`: an empty
    # source (n_rows == 0) must still run one real (empty) group_by/agg so the result has
    # the correct schema, instead of `acc` staying `None` with nothing to return.
    for offset in range(0, max(n_rows, 1), step):
        chunk = lf_source().slice(offset, step)
        if chunk_transform is not None:
            chunk = chunk_transform(chunk)

        if aggregate == "mean":
            if weight_column is not None:
                sum_expr = (pl.col(value_column) * pl.col(weight_column)).sum()
                denom_expr = (
                    pl.when(pl.col(value_column).is_not_null())
                    .then(pl.col(weight_column))
                    .otherwise(None)
                    .sum()
                )
            else:
                sum_expr = pl.col(value_column).sum()
                denom_expr = pl.col(value_column).count()
            part = (
                chunk.group_by(*keys)
                .agg(sum_expr.alias(_SUM_ALIAS), denom_expr.alias(_COUNT_ALIAS))
                .collect(engine="streaming")
            )
        else:
            part = (
                chunk.group_by(*keys)
                .agg(getattr(pl.col(value_column), aggregate)().alias(value_column))
                .collect(engine="streaming")
            )

        if key_columns is None:
            key_columns = [c for c in part.columns if c not in accumulator_columns]

        if acc is None:
            acc = part
            continue

        merged = pl.concat([acc, part])
        if aggregate == "mean":
            acc = merged.group_by(*key_columns).agg(
                pl.col(_SUM_ALIAS).sum().alias(_SUM_ALIAS),
                pl.col(_COUNT_ALIAS).sum().alias(_COUNT_ALIAS),
            )
        elif aggregate == "count":
            acc = merged.group_by(*key_columns).agg(pl.col(value_column).sum().alias(value_column))
        else:
            acc = merged.group_by(*key_columns).agg(
                getattr(pl.col(value_column), aggregate)().alias(value_column)
            )

    if aggregate == "mean":
        acc = acc.with_columns(
            pl.when(pl.col(_COUNT_ALIAS) > 0)
            .then(pl.col(_SUM_ALIAS) / pl.col(_COUNT_ALIAS))
            .otherwise(None)
            .alias(value_column)
        ).drop(_SUM_ALIAS, _COUNT_ALIAS)

    return acc


def fold_count_by_value(
    lf_source: Callable[[], pl.LazyFrame],
    keys: Sequence[Any],
    value_column: str,
    max_distinct: int,
    step: int | None = None,
    chunk_transform: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
) -> pl.DataFrame:
    """The `count_by_value` counterpart of `fold_group_by`, kept as a separate function
    rather than one more `aggregate` branch there: `count_by_value`'s accumulator is not a
    scalar (or a scalar pair, like `mean`'s) but a whole COUNT TABLE, one row per (group,
    distinct value) -- a different accumulator shape, not a bigger one, so folding it does
    not decompose into `fold_group_by`'s existing merge branches. There is no aggregate in
    the vocabulary that does not fold; this is what makes that true for the one aggregate
    whose accumulator does not look like the other five's.

    The partial for one chunk is `chunk.group_by(*keys, value_column).agg(pl.len())`, and
    the merge for two partials is exactly `count`'s own merge: concatenate, re-group by
    every key INCLUDING the categorical one, and sum the counts. This is safe -- summing
    counts of matching (group, category) keys across chunks reproduces the single-pass
    answer exactly, the same associative-count argument `fold_group_by`'s own docstring
    makes for plain `count`, just with one more column riding along in the key.

    `value_column`'s null values are kept as their own category (never filtered out here
    or by a caller's `chunk_transform`) -- the categorical count is over whatever value a
    row actually carries, and a null measurement is a fact about that row, not the absence
    of one. Polars' own `group_by` already groups `None` as an ordinary key value, so
    nothing extra is needed to make that happen; it would take a deliberate filter to lose
    it, and this function does not add one.

    The cardinality probe (`approx_n_unique`, against `max_distinct`) runs once, up front,
    against `lf_source()` directly -- the bare, pre-`chunk_transform` source -- rather than
    per chunk or after transformation. This is deliberately conservative rather than
    exact for the general case, but for THIS package's one caller it is actually exact:
    the only `chunk_transform` `execution/project.py` supplies is an inner join to a
    crosswalk, which can only drop rows whose source cell overlaps no destination -- it
    can duplicate a row's existing value across several destination rows, but it can never
    manufacture a value that was not already in `value_column` before the join. A probe
    against the untransformed source therefore never UNDER-counts the true distinct-value
    total a caller's transform could produce, which is the direction that matters: this
    check exists to refuse a continuous column before the unbounded `group_by` it would
    otherwise cost, not to admit one a moment too early.

    Raises `request.errors.AggregateError` if the probe exceeds `max_distinct` -- the
    request layer's own "an aggregate can't do what was asked of it" family, since this is
    exactly that: `count_by_value` is for bounded categorical data, and a column that
    fails this probe is not that, discovered here because it can only be discovered by
    reading the data, never at plan time.
    """
    if step is None:
        step = TABULAR_FOLD_STEP
    if step <= 0:
        raise FoldError(f"fold step must be positive, got {step}")

    n_unique = lf_source().select(pl.col(value_column).approx_n_unique()).collect(engine="streaming").item()
    if n_unique > max_distinct:
        raise AggregateError(
            f"count_by_value would produce approximately {n_unique} distinct values for "
            f"'{value_column}', over the limit of {max_distinct} -- count_by_value is meant "
            "for bounded categorical data (e.g. a land-type code), not continuous data"
        )

    n_rows = lf_source().select(pl.len()).collect(engine="streaming").item()

    acc = None
    key_columns = None
    for offset in range(0, max(n_rows, 1), step):
        chunk = lf_source().slice(offset, step)
        if chunk_transform is not None:
            chunk = chunk_transform(chunk)

        part = (
            chunk.group_by(*keys, pl.col(value_column).alias("value_category"))
            .agg(pl.len().alias("value"))
            .collect(engine="streaming")
        )

        if key_columns is None:
            key_columns = [c for c in part.columns if c != "value"]

        if acc is None:
            acc = part
            continue

        merged = pl.concat([acc, part])
        acc = merged.group_by(*key_columns).agg(pl.col("value").sum().alias("value"))

    return acc
