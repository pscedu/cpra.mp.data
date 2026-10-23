"""The long-shape lowerings: `filter_long`/`case_long`/`group_long`/`join_long`/
`unpivot_long`, one free function per op this vertical slice executes, each taking what it
reads as arguments and returning a new `LongFrame` rather than mutating anything. `calculate` has no lowering of
its own in either shape -- `request.ops.compile_ops` lowers it to an arm-less `case`, whose
`else` is the calculate's expression.

Plus `case_expression`, the one piece here that is NOT long-shaped: it builds the chained
`pl.when(...).then(...)` a `case` op compiles to, and both shapes' lowerings call it --
`case_long` below over the frame's value columns, `execution.band_ops.case_band_wide` over
one band's column (or one pair's). It lives in this module rather than a third because
`band_ops` may import `long_ops` and not the reverse (`tests/api/test_analysis_layering.
py`), so this is the lower of the two and the only one both can reach. Keeping a single
builder is what makes "first match wins, and an unmatched value takes the `else`" one
statement rather than two that have to be kept saying the same thing.

`execution/execute.py`'s walk looks each of these up by `(op kind, frame type)` and calls
it once per op, in declared order; none of them knows about the walk, the plan, or any
other op -- see that module's docstring for the dispatch itself.

**Why the Polars here is written out rather than delegated.** These bodies are
deliberately faithful carry-overs of numerically fussy leaf logic that predates this
package, rather than improvements made in passing; rewriting them would re-derive bugs
that were expensive to find the first time.

`project` has no entry here on purpose: `project` is the one op that CHANGES shape
(`request.ops.PROJECT_LOWERING_TABLE`), so it has no single-shape lowering the way the
others do. Its lowerings live in `execution/project.py`, one per direction. `unpivot` is
the converse case -- it has no BAND-WIDE lowering, so `unpivot_long` is the whole of it;
see `request.ops.UNPIVOT_LOWERING_TABLE` for why that is a rejection rather than a gap.
"""

from collections.abc import Callable, Sequence

import polars as pl

from cpra.mp.data.analysis.execution.fold import fold_group_by, should_fold_group_by
from cpra.mp.data.analysis.execution.frame import LongFrame
from cpra.mp.data.analysis.request.expressions import (
    ExpressionError,
    PredicateKind,
    classify_where_kinds,
    combine_conjuncts,
    parse_value_expr,
    split_where_halves,
    value_predicate_mask,
)
from cpra.mp.data.analysis.request.ops import CaseOp, GroupOp, UnpivotOp, calculate_label
from cpra.mp.data.analysis.request.pipeline import DATASET_SELECTION_COLUMNS

__all__ = ["case_expression", "case_long", "filter_long", "group_long", "join_long", "unpivot_long"]


def filter_long(lf: LongFrame, where: str, value_columns: Sequence[str] = ("value",)) -> LongFrame:
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

    `value_columns` is the incoming frame's value columns, straight off the frame type the
    plan recorded -- `{"value"}` for anything that has not been through a `join`, and the
    operand names (`{"marsh", "sal"}`) for a frame that has. It is what decides which
    bucket a conjunct lands in, and it is a parameter for the same reason
    `case_long`'s is: a long frame's schema cannot distinguish a value column from a
    dimension column by inspection, so only the typing walk knows (`case_long` takes it for
    the same reason). Without it this
    lowering classified by the literal name `value` and sorted a PAIRED frame's
    operand-named conjunct (`marsh > 0.5`) into the dimension bucket, where it took the
    unrestricted `lf.sql()` path rather than the whitelist -- and disagreed with
    `band_ops.filter_band_wide`, which has always had the parameter, about what the same
    `where` over the same data means.

    A value conjunct is checked against `value_columns` alone rather than against every
    column the frame happens to carry, which is what makes `value > calendar_year` a
    rejection here as it already is in band-wide shape: a single conjunct mixing a
    measurement and a dimension has no lowering, and admitting it in one shape only is
    how the two drift apart.

    Raises `request.expressions.ExpressionError` for a value conjunct that fails the
    expression whitelist, or a conjunct naming a column this frame does not carry.
    """
    if not where:
        return lf
    kinds = classify_where_kinds(where, value_columns)
    row_conjuncts = [*kinds[PredicateKind.dimension], *kinds[PredicateKind.spatial]]
    if row_conjuncts:
        lf = lf.sql(f"select * from self where {combine_conjuncts(row_conjuncts)}")
    if kinds[PredicateKind.value]:
        value_where = combine_conjuncts(kinds[PredicateKind.value])
        lf = lf.filter(value_predicate_mask(value_where, list(value_columns)))
    return lf


def _compile_case_part(
    what: str, compile_fn: Callable[[str, Sequence[str]], pl.Expr], text: str, allowed_columns: Sequence[str]
) -> pl.Expr:
    """Compile one of a `case`'s several expressions, re-raising with `what` (`step.5 arm
    2 when`) in front of whatever the grammar said. A `case` carries `2n + 1` of them and
    an unlabelled "invalid expression" would leave the caller counting arms.
    """
    try:
        return compile_fn(text, allowed_columns)
    except ExpressionError as exc:
        raise ExpressionError(f"{what}: {exc}") from exc


def case_expression(
    op: CaseOp,
    allowed_columns: Sequence[str],
    rewrite: Callable[[str], str] = lambda text: text,
    predicate_columns: Sequence[str] | None = None,
    resolve_dimension: Callable[[str], pl.Expr | bool] | None = None,
) -> pl.Expr:
    """The single `pl.Expr` a `case` op compiles to, in either shape: a chained
    `pl.when(...).then(...)` over `op.cases` in DECLARED ORDER, falling through to
    `op.else_` (null when unset). A `case` with NO arms is `op.else_` everywhere, which is
    exactly what a `calculate` is -- `request.ops.compile_ops` lowers one to the other, so
    this function is the whole of both ops' expression semantics.

    Two things this builds in, each the reason `case` exists as its own op rather than as
    a chain of value rewrites:

    - **First match wins, over the INCOMING value.** Polars evaluates a `when/then` chain
      top-down and every arm's condition reads the frame's own columns, not the chain's
      partial result -- so arm 2 sees what arrived, never what arm 1 wrote. A chain of
      independent rewrite steps cannot say that: each one rewrites the running value, so
      the next one's predicate sees the rewrite.
    - **No pass-through.** A row/pixel matching no arm takes `else_`, which defaults to
      null. Pass-through is something the AUTHOR asks for, by writing `else: "value"`, and
      is the shape every migrated `calculate` takes; it is not a default, so a total
      reclassification and a patch over one are told apart by what the request says rather
      than by which op it used.

    **Nodata is the author's to state, and nothing is added here.** This chain used to be
    wrapped in `pl.when(all operands non-null).then(...).otherwise(None)`, on the reasoning
    that an `else` would otherwise paint nodata with real data -- a null operand makes
    every arm's condition false (`value_predicate_mask` fills null with `False`), so nodata
    falls straight through to the `else`. That is a true description of an UNGUARDED
    `else`, and it is now what a pipeline gets, because the guard cost more than it bought:

    - It was `all_horizontal` over the whole namespace, not over what an arm reads, so on a
      paired frame one irrelevant operand's nodata nulled a result the matching arm had
      fully determined from the operands it did name. No `else` had to be involved.
    - It made null-testing arms dead code. `IS NULL` / `IS NOT NULL` parse (they are
      operators, so the whitelist never sees them) and `classify_where_kinds` buckets them
      as value conjuncts, so `_type_case` accepts an arm that means to handle nodata --
      and the guard then discarded the arm's answer.

    Both are expressible by the author in one conjunct, which is the trade: write
    `AND x IS NOT NULL` on an arm, or `coalesce(...)` (whitelisted) in a `then`/`else`, to
    say what nodata should do. An `else` now means what it says.

    `allowed_columns` is the namespace every arm's `then` and the `else` are checked
    against, and the namespace its `when`'s VALUE half is compiled against -- the frame's
    value columns in long shape, one band's column (plus any carried columns) in band-wide.
    `rewrite` is applied to each expression's TEXT first, which is how the band-wide
    lowerings substitute the placeholder `value` (or each operand name) for the actual band
    column; it defaults to the identity, which is what long shape needs since its columns
    are already named what the request calls them.

    **The dimension half of an arm, and why it takes two more parameters.** An arm's `when`
    is split by `request.expressions.split_where_halves`, whose classification has to run
    against the names the REQUEST wrote (`value`, or the operand names), not against the
    band columns `rewrite` maps them to -- classifying `value > 3` against `["b0"]` would
    call it a dimension conjunct. `predicate_columns` is those request-level names,
    defaulting to `allowed_columns` because long shape is already written in them.

    `resolve_dimension` turns a dimension half into a gate, and is what each shape's
    lowering supplies out of the band metadata only it has:

    - a `pl.Expr` (long shape: dimension columns are ordinary columns there, so the half
      compiles to a per-row mask) -- ANDed with the arm's value half;
    - `True`/`False` (band-wide: the half selects which BANDS the arm applies to, one
      answer per band or per pair, evaluated against `metadata_df`). `False` DROPS that arm
      from this band's chain rather than skipping the band -- the band still gets every
      other arm and the `else`, which is the difference between an arm not applying and a
      band not being touched.

    Omitting `resolve_dimension` means "this caller has no metadata to resolve against",
    and an arm carrying a dimension conjunct then raises rather than being silently read as
    a value predicate over a column the frame does not have.

    Raises `request.expressions.ExpressionError`, labelled with which arm and which half
    of it failed.
    """
    label = calculate_label(op)
    names = list(allowed_columns if predicate_columns is None else predicate_columns)
    chain = None
    for position, arm in enumerate(op.cases, start=1):
        what = f"{label} arm {position} when"
        dimension_where, value_where = split_where_halves(arm.when, names, what)
        mask: pl.Expr | None = None
        if dimension_where is not None:
            if resolve_dimension is None:
                raise ExpressionError(
                    f"{what}: conjunct(s) {dimension_where!r} name no value column "
                    f"({sorted(names)}), and this frame carries no band metadata for them "
                    "to select against"
                )
            gate = resolve_dimension(dimension_where)
            if gate is False:
                continue
            if gate is not True:
                mask = gate
        if value_where is not None:
            value_mask = _compile_case_part(what, value_predicate_mask, rewrite(value_where), allowed_columns)
            mask = value_mask if mask is None else mask & value_mask
        then = _compile_case_part(
            f"{label} arm {position} then", parse_value_expr, rewrite(arm.then), allowed_columns
        )
        # `mask is None` is an arm whose `when` was PURELY dimensional and applies to this
        # band: nothing per-pixel is left to test, so the arm matches the whole band -- and
        # every later arm is then unreachable, which is what first-match-wins means.
        condition = pl.lit(True) if mask is None else mask
        chain = pl.when(condition).then(then) if chain is None else chain.when(condition).then(then)
    otherwise = (
        pl.lit(None)
        if op.else_ is None
        # A `case` with no arms IS a `calculate` (see above), so the half that failed is
        # that request's `expression`, not an `else` it never wrote.
        else _compile_case_part(
            f"{label} {'else' if op.cases else 'expression'}",
            parse_value_expr,
            rewrite(op.else_),
            allowed_columns,
        )
    )
    # `chain is None` for a `case` with no arms -- a lowered `calculate` (`pipeline.
    # CaseStep` itself still requires `min_length=1`) -- and also for one whose every arm
    # this band's metadata dropped. Both mean the same thing: the else, everywhere.
    return otherwise if chain is None else chain.otherwise(otherwise)


def case_long(lf: LongFrame, op: CaseOp, value_columns: frozenset[str]) -> LongFrame:
    """The long-shape lowering of a `case`: one `with_columns`, writing the chained
    expression `case_expression` builds into `value` and dropping the operand columns.

    `value_columns` plays exactly the role it plays in `filter_long` -- the frame's own
    value columns off the recorded frame type, since a long frame's schema cannot tell a
    value column from a dimension column by inspection. It is the expression namespace, and
    the operand columns are dropped afterwards because `request.ops._type_case` types a
    `case` as producing exactly one value column called `value`, whatever it read.

    There is no `unchanged` fallback to thread: an unmatched row takes the `else`, not the
    value it arrived with, so the paired case (where there is no incoming `value` column to
    fall back TO) needs no special reading here. A request that wants pass-through says so
    with `else: "value"`, which a joined frame's namespace correctly rejects.

    **An arm's dimension half is an ordinary row mask here.** Long shape carries its
    dimension columns as real columns, so the half `split_where_halves` peels off compiles
    straight to a `pl.Expr` -- no band metadata, no per-band answer, just a narrower mask
    ANDed with the arm's value half. What it cannot know until the frame is in hand is
    whether the columns it names EXIST, so that is checked against the schema here and
    raised with the frame's own column list, the same execution-time error an unknown
    dimension column in a `where` has always given.
    """
    namespace = sorted(value_columns)
    schema_names = set(lf.collect_schema().names())

    def resolve_dimension(text: str) -> pl.Expr:
        expr = pl.sql_expr(text)
        missing = set(expr.meta.root_names()) - schema_names
        if missing:
            raise ExpressionError(
                f"{calculate_label(op)}: when references column(s) not present in this "
                f"dataset: {sorted(missing)} -- available columns: {sorted(schema_names)}"
            )
        # `fill_null(False)` so a null dimension value fails the arm rather than making its
        # condition null, which Polars treats as "not matched" anyway but only by accident.
        return expr.fill_null(False)

    value = case_expression(op, namespace, resolve_dimension=resolve_dimension)
    return lf.with_columns(value.alias("value")).drop(
        [column for column in namespace if column != "value"]
    )


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


def join_long(frames: list[LongFrame], names: Sequence[str]) -> LongFrame:
    """The long-shape lowering of a `join`: combine `frames` (already resolved, in
    `names` order) on `grid_cell` plus every shared dimension, inner, renaming each
    operand's `value` column to its own step name first.

    Takes the operand names rather than the `JoinOp` they came from -- nothing else about
    the op bears on the lowering, and `execution.band_ops.sample_grid_cells_band_wide`
    reaches this function from the band-wide side, where the operand names come off
    `JOIN_OPERAND_COLUMN` and there is no op in hand. Same signature shape as
    `band_ops.join_band_wide(frames, names, ...)`, its band-wide counterpart.

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
    renamed = [frame.rename({"value": name}) for name, frame in zip(names, frames)]
    carried = [
        frame.drop(
            [column for column in frame.collect_schema().names() if column in DATASET_SELECTION_COLUMNS]
        )
        for frame in renamed
    ]
    key_columns = [set(frame.collect_schema().names()) - {name} for name, frame in zip(names, carried)]
    shared = sorted(set.intersection(*key_columns))
    result = carried[0]
    for other in carried[1:]:
        result = result.join(other, on=shared, how="inner")
    return result


def unpivot_long(lf: LongFrame, op: UnpivotOp, value_columns: Sequence[str]) -> LongFrame:
    """The long-shape lowering of an `unpivot`: stack `value_columns` into one `value`
    column plus the label column `op.into`, which carries the name of the column each row's
    measurement came from. The exact inverse of the rename `join_long` above performs, and
    the op a multi-variable result reaches a one-value-column sink through.

    `value_columns` is the incoming frame's value columns off the typing walk, the same
    parameter and for the same reason `filter_long`/`case_long` take one: a long frame's
    schema cannot tell a value column from a dimension column by inspection, so only the
    walk knows which of them to fold. Everything NOT in it is an index column and rides
    through untouched, one copy per folded name -- `grid_cell` and every dimension
    included, which is exactly what makes the result a longer frame of the same
    observations rather than a different set of them.

    Sorted, so the stacking order is the operand names' own rather than whatever order a
    `frozenset` iterated in: the row order of the result is otherwise non-deterministic
    across runs, and `output.sort` is not always given.

    There is no band-wide counterpart, by construction -- `request.ops._type_unpivot`
    rejects a band-wide input outright (its columns are positional bands, so there are no
    names to fold), which is why `UNPIVOT_LOWERING_TABLE` has no `band_wide` row.
    """
    folded = sorted(value_columns)
    index = [column for column in lf.collect_schema().names() if column not in set(folded)]
    return lf.unpivot(index=index, on=folded, variable_name=op.into, value_name="value")
