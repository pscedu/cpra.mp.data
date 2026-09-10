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
order `band_wide_worker.run_long_fold` uses, and for the same reason: a join or
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
`ValueError` subclass, not `HTTPException`, matching `api/expressions.py`'s split between
library code (plain values in, ordinary exceptions out) and `Dataset` (the translation
layer that maps a caught exception to a status code).

This module also owns the plan-time ROUTING decision (`should_fold_group_by`): folding is
worth its overhead (an extra `pl.len()` scan, plus re-grouping the accumulator after every
chunk) only for the request shapes that would otherwise hit the cliff. See that function's
docstring for the exact rule and the reasoning behind it.
"""

from collections.abc import Callable, Sequence
from typing import Any

import polars as pl

from cpra.mp.data.config import TABULAR_FOLD_STEP

# `sum`, `min`, `max`, and `count` are directly associative: a partial computed the same
# way over any subset of rows merges into the whole with the identical aggregate applied
# again to the partials (`count`'s merge is the one exception spelled out in
# `fold_group_by` -- summing partial counts, not counting them). `mean` is decomposable
# only when carried as a (sum, count) pair and divided once at the very end; a mean of
# per-chunk means silently gives the wrong answer whenever chunks contribute unequal
# counts (a short final chunk, or a filter that empties some chunks unevenly), which is
# also the reason `fold_group_by` never exposes a "mean of means" code path at all.
DECOMPOSABLE_AGGREGATES = frozenset({"sum", "min", "max", "count", "mean"})

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
    `HTTPException`: `Dataset` is the only place that translates this into a 400 (see the
    module docstring, and `api/expressions.ExpressionError` for the established pattern).
    """


def validate_decomposable(aggregate: str) -> None:
    """Raise `FoldError` unless `aggregate` has a correct partial/merge decomposition.

    Checked before any file is read -- the same "reject at plan time, not deep inside a
    streaming collect" discipline `api/expressions.py` uses for value predicates.
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

    Pure function over plain values -- no `Dataset`, no Polars schema inspection -- so the
    rule is unit-testable on its own and reusable outside the API. Both arguments are
    exactly what a caller can know at plan time, before any file is read:

    - `group_by_keys`: the columns the result will actually be grouped by, as plain
      strings (a caller building an aliased or literal Polars expression -- e.g. `True`
      for "no group_by, collapse into grid_cell alone", or `pl.col("dst_id").alias(
      "grid_cell")` for a crosswalk reaggregation -- passes the OUTPUT name it resolves
      to, "grid_cell" in both examples, not the expression object).
    - `has_selective_filter`: whether the request already carries something that shrinks
      the row count reaching this group_by well below "the whole dataset" -- a pushed-down
      dimension filter (Hive-partition-level: `calendar_year=[...]`, `scenario_id=[...]`,
      ...) or a specific `grid_cell`/`summarize_grid_cell` selection. Measured case: the
      identical `grid_cell` group-by with such a filter already runs in 0.1 s / 285 MB, so
      folding it too would only add overhead for no benefit.

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
    might shrink row count (e.g. `value_where`, which MASKS values to null rather than
    dropping rows, is deliberately not treated as selective here -- it doesn't change how
    many rows reach the group-by at all).
    """
    if has_selective_filter:
        return False
    return any(isinstance(key, str) and key in high_cardinality_columns for key in group_by_keys)


def fold_group_by(
    lf_source: Callable[[], pl.LazyFrame],
    keys: Sequence[Any],
    aggregate: str,
    value_column: str = "value",
    step: int | None = None,
    chunk_transform: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
) -> pl.DataFrame:
    """Compute `<lf>.group_by(*keys).agg(<aggregate> of value_column)`, bounding peak
    memory by never handing the underlying `group_by` more than `step` rows at once.

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
    `scale_factor`), matching `band_wide_worker.run_long_fold`'s slice-then-filter order.
    `n_rows` (and therefore the chunk boundaries) is always computed from `lf_source()`
    alone, before `chunk_transform` runs -- a caller whose transform can multiply row
    count (a many-to-one crosswalk join) still gets chunks sized off the SOURCE side,
    which is what actually bounds the underlying scan's memory. A caller building
    `lf_source`/`chunk_transform` around an upstream operation that is itself expensive
    regardless of chunking should only route through `should_fold_group_by` when that
    operation is known to be cheap (see the two `Dataset` call sites).

    `keys` is passed to `group_by` as-is for the FIRST (partial) pass -- it may contain
    plain column names, Polars expressions, or the literal `True` (the vector "no
    group_by, collapse to grid_cell alone" case; see `Dataset._get_group_by_columns`).
    The MERGE pass, on every chunk after the first, instead re-derives the group columns
    from the collected partial's own schema (everything except the value/accumulator
    columns) -- this sidesteps needing to know how a non-string key like `True` or an
    aliased expression resolves to a concrete column name (Polars names a bare `True` key
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
            part = (
                chunk.group_by(*keys)
                .agg(
                    pl.col(value_column).sum().alias(_SUM_ALIAS),
                    pl.col(value_column).count().alias(_COUNT_ALIAS),
                )
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
