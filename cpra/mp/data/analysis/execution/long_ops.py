"""The long-shape lowerings: `filter_long`/`calculate_long`/`group_long`/`join_long`, one
free function per op this vertical slice executes, each taking what it reads as arguments
and returning a new `LongFrame` rather than mutating anything.

`execution/execute.py`'s walk looks each of these up by `(op kind, frame type)` and calls
it once per op, in declared order; none of them knows about the walk, the plan, or any
other op -- see that module's docstring for the dispatch itself.

**Why the Polars here is written out rather than delegated.** These bodies are
deliberately faithful carry-overs of numerically fussy leaf logic that predates this
package, rather than improvements made in passing; rewriting them would re-derive bugs
that were expensive to find the first time.

`project` has no long-shape entry here on purpose: `project` is the one op that CHANGES
shape (`request.ops.PROJECT_LOWERING_TABLE`), so it has no single-shape lowering the way
the other four do, and this vertical slice does not lower it at all yet.
"""

import polars as pl

from cpra.mp.data.analysis.execution.fold import fold_group_by, should_fold_group_by
from cpra.mp.data.analysis.execution.frame import LongFrame
from cpra.mp.data.analysis.request.expressions import (
    ExpressionError,
    classify_where,
    compile_calculate_entries,
    parse_value_expr,
    value_predicate_mask,
)
from cpra.mp.data.analysis.request.ops import CalculateOp, GroupOp, JoinOp, calculate_label
from cpra.mp.data.analysis.request.pipeline import DATASET_SELECTION_COLUMNS

__all__ = ["calculate_long", "filter_long", "group_long", "join_long"]


def filter_long(lf: LongFrame, where: str) -> LongFrame:
    """The long-shape lowering of a `filter`: a plain row drop, regardless of which of the
    three predicate kinds (value/dimension/spatial) `where` combines.

    Unlike band-wide -- where a value conjunct becomes a per-pixel MASK, because dropping
    a row there would remove that pixel from every band including the ones the predicate
    holds for -- a long frame has no positional alignment to preserve, so there is nothing
    a mask would buy. One row is one observation, and an observation the predicate rejects
    simply is not in the result.

    The dimension and spatial conjuncts go through `lf.sql()` as an unrestricted row
    filter (a spatial conjunct tests `grid_cell`, which in long shape is an ordinary
    column, so the two need no separate treatment here); the value conjuncts go through
    the checked expression grammar, which is what keeps an arbitrary expression from
    reaching Polars.

    Raises `request.expressions.ExpressionError` for a value conjunct that fails the
    expression whitelist, or a conjunct naming a column this frame does not carry.
    """
    if not where:
        return lf
    dimension_conjuncts, value_conjuncts = classify_where(where)
    if dimension_conjuncts:
        combined = " and ".join(f"({conjunct})" for conjunct in dimension_conjuncts)
        lf = lf.sql(f"select * from self where {combined}")
    if value_conjuncts:
        value_where = " and ".join(f"({conjunct})" for conjunct in value_conjuncts)
        lf = lf.filter(value_predicate_mask(value_where, lf.collect_schema().names()))
    return lf


def calculate_long(lf: LongFrame, op: CalculateOp, value_columns: frozenset[str]) -> LongFrame:
    """The long-shape lowering of a `calculate`: compute a new `value` from
    `op.expression` wherever `op.where` matches (everywhere, when `op.where` is `None`).

    `value_columns` is the incoming frame's value columns, straight off the frame type the
    plan recorded -- `{"value"}` for anything that has not been through a `join`, and the
    operand names (`{"a", "b"}`) for a frame that has. It is a parameter rather than
    something derived here because a long frame's schema cannot distinguish a value column
    from a dimension column by inspection; only the typing walk knows.

    It decides three things, all of which collapse to the old `value`-only behaviour when
    it is `{"value"}`:

    - **The expression namespace.** `op.expression` is validated against exactly these
      names, which is what lets the `a - b` a post-`join` `calculate` exists to run parse
      at all. A name outside them raises `ExpressionError` naming the namespace.
    - **The null guard.** The rewrite applies only where EVERY value column is non-null,
      so a null measurement stays null instead of silently becoming whatever the
      expression makes of it.
    - **What the result carries.** The computed column is always named `value` and the
      operand columns are dropped, because that is what `request.ops` types a `calculate`
      as producing -- one value column called `value`, whatever it read. Dropping them is
      what makes the joined frame legal at the sink.

    A row `op.where` does not match passes through UNCHANGED rather than being dropped --
    dropping is `filter`'s job, never `calculate`'s. "Unchanged" needs the incoming `value`
    to fall back to, though, and a joined frame has none: for that frame an unmatched row's
    `value` is null, which is the honest answer (nothing computed it) and the same thing
    every downstream aggregate already skips.

    `op.expression`/`op.where` go through the same `compile_calculate_entries` the request
    layer uses at parse time, so an expression that parses at validation is the one that
    runs here. Raises `request.expressions.ExpressionError`.
    """
    namespace = sorted(value_columns)
    (plan,) = compile_calculate_entries(
        [(calculate_label(op), op.expression, op.where)], namespace
    )
    schema_names = set(lf.collect_schema().names())
    condition = pl.all_horizontal([pl.col(column).is_not_null() for column in namespace])
    if plan.dimension_where is not None:
        dimension_expr = pl.sql_expr(plan.dimension_where)
        missing = set(dimension_expr.meta.root_names()) - schema_names
        if missing:
            raise ExpressionError(
                f"{plan.label}: where references column(s) not present in this dataset: "
                f"{sorted(missing)} -- available columns: {sorted(schema_names)}"
            )
        condition = condition & dimension_expr.fill_null(False)
    if plan.value_where is not None:
        condition = condition & value_predicate_mask(plan.value_where, namespace)
    new_value = parse_value_expr(plan.expression, namespace)
    unchanged = pl.col("value") if "value" in value_columns else pl.lit(None)
    return lf.with_columns(
        pl.when(condition).then(new_value).otherwise(unchanged).alias("value")
    ).drop([column for column in namespace if column != "value"])


def group_long(lf: LongFrame, op: GroupOp) -> LongFrame:
    """The long-shape lowering of a `group`: collapse `[*op.by, "grid_cell"]`, aggregating
    `op.of` with `op.aggregate`. `op.by == []` collapses to one row per `grid_cell`, per
    that field's own docstring.

    A dimension column left out of `op.by` is simply ABSENT from the result rather than
    returned as an all-null placeholder. "This dimension no longer has a value" is the
    honest shape for a collapsed dimension, and a later step naming one surfaces as a
    clean missing-column error instead of silently reading a column of nulls.

    Folds through `fold.fold_group_by` -- bounded memory, numerically exact -- whenever
    `should_fold_group_by` judges the collapse cliff-shaped, and otherwise uses Polars'
    own `group_by(...).agg(...)`. The `has_selective_filter=False` passed below is a
    deliberate, documented simplification: the fold decision wants to know whether some
    earlier step already shrank this frame enough that the cheap path is safe, and nothing
    on the plan records that yet. `False` takes the conservative side -- folding a frame
    that was already small costs a little redundant overhead, where skipping a fold that
    was needed costs the memory cliff the fold exists to avoid. Threading real filter
    history onto the plan is what would let this carry the true signal.

    May raise `fold.FoldError` for a non-decomposable aggregate reaching the fold path --
    unreachable from a validated request, since `request.aggregates` already restricts a
    `group`'s aggregate to the decomposable set, but `fold_group_by` guards it regardless.
    """
    key_names = [*op.by, "grid_cell"]
    if should_fold_group_by(key_names, has_selective_filter=False):
        source_lf = lf
        return fold_group_by(lambda: source_lf, key_names, op.aggregate, value_column=op.of).lazy()
    return lf.group_by(*op.by, "grid_cell").agg(getattr(pl.col(op.of), op.aggregate)())


def join_long(frames: list[LongFrame], op: JoinOp) -> LongFrame:
    """The long-shape lowering of a `join`: combine `frames` (already resolved, in
    `op.inputs` order) on `grid_cell` plus every shared dimension, inner, renaming each
    operand's `value` column to its own step name first.

    The rename is what makes a join usable: both operands carry `value`, so the result
    would otherwise have two columns of one name. `join(inputs: [a, b])` therefore yields
    value columns `a` and `b`, and the downstream `calculate` that collapses them back to
    a single `value` writes `a - b`.

    **The dataset-selection columns are excluded from the key, and dropped.** A vector scan
    materializes `variable`/`time_unit`/`grid` as literal columns, so two operands reading
    two different datasets disagree on them in every row. Left in the key, that makes the
    inner join match nothing at all and return an empty frame -- for the most ordinary
    join there is, two variables on one grid, compared cell by cell. They are not
    describing the observation, they are naming which dataset it came from, and after a
    join that question no longer has one answer for a row: the operand each measurement
    came from is carried by its renamed value column instead, which is both unambiguous
    and what a downstream expression actually refers to. Dropping `grid` is safe precisely
    because the request layer already rejects a join whose operands disagree on it
    (`request.ops`'s join typing): every row reaching here is addressed against one
    lattice, so the column carries no information the join needs.

    Every other column both operands carry joins as a key; a dimension column only one
    operand carries rides along unchanged rather than being dropped or forced into the key.
    The join is always inner -- outer semantics would push nulls into every downstream
    aggregate, and null handling is specified too precisely elsewhere for that to be a
    default.
    """
    renamed = [frame.rename({"value": name}) for name, frame in zip(op.inputs, frames)]
    carried = [
        frame.drop(
            [column for column in frame.collect_schema().names() if column in DATASET_SELECTION_COLUMNS]
        )
        for frame in renamed
    ]
    key_columns = [
        set(frame.collect_schema().names()) - {name} for name, frame in zip(op.inputs, carried)
    ]
    shared = sorted(set.intersection(*key_columns))
    result = carried[0]
    for other in carried[1:]:
        result = result.join(other, on=shared, how="inner")
    return result
