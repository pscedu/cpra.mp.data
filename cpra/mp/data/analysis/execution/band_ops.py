"""The band-wide-shape lowerings: `mask_value_band_wide`/`select_dimension_band_wide`/
`sample_grid_cells_band_wide` (the three `filter` predicate kinds), `calculate_band_wide`,
`group_band_wide`, and `join_band_wide` -- the band-wide counterparts of
`execution/long_ops.py`'s four long-shape lowerings, plus `filter_band_wide`, the
dispatcher a `filter` op with a mixed-kind `where` runs through.

**Why a filter's three kinds are three functions, not one.** `request.expressions.
PredicateKind`'s own docstring already says what each kind means for a band-wide frame:
a value conjunct is always a per-cell mask (dropping the row would drop the pixel from
every OTHER band too); a dimension conjunct is a column selection (it decides which
bands survive, not which pixels); a spatial conjunct is the one case that changes the
frame's shape, from band-wide to long, because a `grid_cell` predicate addresses
individual pixels, and a band-wide frame's only handle on pixel identity is its row
position within `window` -- something only worth naming once the frame is no longer
band-wide. Three different operations naturally are three functions; `filter_band_wide`
exists only to apply `request.expressions.classify_where_kinds` and route to them, the
same partition `request.ops.FILTER_LOWERING_TABLE`/`_type_filter` already make at the
shape-typing level.

**Mixed-kind order (dimension before value): a decision, not a requirement.** Nothing in
the governing documents says which conjunct kind runs first when a `where` combines
several -- masking commutes with column subsetting (a band a dimension conjunct removes
never reaches the value mask either way), so the two orders are equivalent up to which
bands the value mask ends up allocating expressions for. Dimension-first is chosen since
it can only shrink that set, never grow it.

**Spatial: transition first, whole `where` after.** `sample_grid_cells_band_wide` does
not itself decide which pixels survive a mixed `where` -- it unconditionally unpivots
every pixel into one long row per (pixel, band) observation (attaching each band's
metadata via a join on band identity), then hands the FULL original `where` -- spatial,
dimension, and value conjuncts together -- to `execution.long_ops.filter_long`, which
already knows how to apply all three kinds to a long frame (a spatial conjunct is just an
ordinary `grid_cell` column there). This is what `execution/frame.py`'s
`assert_height_preserved` docstring means by "a spatial filter changes shape to long
instead of dropping band-wide rows": the band-wide-side operation is the transition
itself, not a row-dropping step, so the invariant that function checks (band-wide row
count unchanged) does not even apply here -- this is the second, unnamed departure from
it (`select_dimension_band_wide`'s empty-result case is the first, and the only one
`assert_height_preserved`'s own docstring names). This module treats that as a
documentation gap worth flagging rather than a contradiction to paper over: a lowering
whose contract is "return long" cannot be checked by a function whose signature compares
two band-wide row counts.

**`calculate_band_wide`'s per-band applicability.** A `CalculateOp.where`'s dimension half
selects which bands the entry applies to, evaluated once against `metadata_df` (one row
per band) rather than once per pixel -- the same
`(label, expression, where) -> CalculatePlan` compilation `execution.long_ops.
calculate_long` uses, applied per band the way `analysis/sources.py`'s
`band_calculate_passes` faithfully-carried-over logic already did for the old engine's
per-entry list. This module's version handles exactly one `CalculateOp` at a time (a
`calculate` step is always one entry here, never `sources.py`'s multi-entry list), and
extends the value mask's namespace to `extra_columns` in addition to the band's own
column -- `sources.py`'s original only ever allowed `[name]` -- matching
`masked_value_predicate_exprs`' own `extra_columns` convention (a value condition scoped
by, say, a crosswalk zone id) rather than the narrower one `band_calculate_passes` had no
occasion to need. The REWRITTEN VALUE gets the same namespace as the mask that gates it: a
carried column good enough to decide WHETHER a pixel is rewritten is good enough to appear
in WHAT it is rewritten to (`value * zone_factor`), and a namespace admitting a column on
one side of `pl.when(...).then(...)` but not the other would be an arbitrary line. That
namespace is passed to `compile_calculate_entries` as well, not only to the per-band
`parse_value_expr`; compilation validates the expression against its own
`allowed_columns` first, so widening only the later call would leave the wider namespace
unreachable.

**Known limitation: `extra_columns` are not part of the PREDICATE language.**
`request.expressions` classifies a conjunct by NAME -- `value` (and `grid_cell`) are the
only names that are not a dimension column -- so a `where` conjunct naming a carried column
(`zone = 7`) classifies as a DIMENSION conjunct and gets evaluated against `metadata_df`,
which has no such column, rather than against the pixel frame that does. Both the `filter`
dispatcher and a `calculate`'s `where` inherit this. The band-wide side cannot fix it
alone: whether a given name is a per-band dimension or a per-pixel carried column is
knowable only where the carried column is introduced. Until then, `extra_columns` means
"carry this column through unchanged", and the value namespaces above are reachable only by
calling a lowering directly with an already-classified value conjunct.

**`group_band_wide`'s horizontal `count`.** `sources.py`'s `HORIZONTAL_RASTER_AGGREGATES`
deliberately has no `count` key -- that dict serves band-COLLAPSE for the old engine's
raster output writer, where "how many calendar steps satisfied a condition" is a
different, already-served question. A band-wide `group.aggregate == "count"` here answers
a different question again: "how many of these bands contributed a non-null value at this
pixel", the per-pixel cross-band analogue of `execution.long_ops.group_long`'s row count.
It needs no `no_contributor` sentinel the other four aggregates need -- a pixel where
every contributing band is null naturally counts zero contributors, which is already the
right answer, not an edge case standing in for an undefined one.

**`group_band_wide(by=[])` and the reserved band index.** Collapsing every dimension
column out of `metadata_df` (nothing distinguishes the merged bands by dimension value
once every one of them is one group) hits the same "zero columns loses the row count"
Polars gap `execution/frame.py`'s docstring describes for the empty-dimension-filter
case, on the metadata side instead of the pixel side: this lowering's answer is
`add_reserved_band_index`, called on the still-populated group frame BEFORE the column
that carried its height is dropped. The reserved column is left standing in the frame this
function returns, matching `add_reserved_band_index`'s own docstring ("must never reach
anything outside the walk") rather than `drop_reserved_band_index`ing it here: this
module builds band-wide lowerings, not the walk itself (no band-wide walk exists yet --
see `execute.py`'s module docstring), so there is no "outside the walk" boundary for this
function to have already crossed. Whatever eventually finishes a band-wide walk owns that
call, the same way `execution/frame.py` already documents.

**`group_band_wide` ignores `GroupOp.of`.** `of` picks which column an aggregate reads in
long shape (default `"value"`, generalizing to e.g. "first year a condition held"); a
band-wide frame's per-band measurement IS the band column itself, with no second column a
band-wide `group` could plausibly read instead. This is a real gap between one field's
long-shape meaning and its (absent) band-wide one that the governing documents do not
resolve -- flagged here rather than silently accepting an `of` that would have no effect,
and recorded in the migration's running notes as a decision made without unambiguous
text.

**`join_band_wide`'s operand-provenance column.** A long-shape join
(`execution.long_ops.join_long`) renames each operand's `value` column to its own step
name, because two operands both carrying a column literally called `value` would collide.
Band-wide has no literal `value` column to rename -- every band IS a value, under a
purely positional name (`b0`, `b1`, ...) that already never collides across operands
(this lowering renumbers them itself; see below). The information a long join's rename
carries -- which operand a given value came from -- has to go somewhere else here, so it
goes onto `metadata_df` instead, in `JOIN_OPERAND_COLUMN`: one merged metadata row per
surviving band, each carrying which operand (step name) contributed that band. This is
the band-wide reading of "each operand's value renamed to its step name" worked out for
this module, not a literal rename, since band-wide has nothing named `value` to rename.

**`join_band_wide`'s window precondition, and why it is not `==`.** A band-wide join is a
positional row alignment: row *i* of every operand must be the same pixel. What makes that
true is that the operands share a pixel GRID -- same offsets, extent, transform, CRS,
decimation -- not that they came from the same file. The operands of a real join are two
different reads of two different rasters, brought onto one shared window by whatever
compiles the plan; comparing `ReadWindow` with a plain `==` would compare its opaque
`source` field too and reject precisely that case, leaving the lowering satisfiable only by
joining a frame to itself. So the precondition here is
`execution.frame.assert_same_pixel_grid`, which compares every field but `source`;
`assert_window_unchanged` (which does compare `source`) stays for its own, different
question -- one lowering's own window before and after. Bringing two operands onto a shared
window in the first place is a compile-time concern this module does not own; it only
checks the result.

**`join_band_wide` combines columns positionally, with no key.** Rows are already aligned
by the shared window, so there is nothing to join ON: the operands' band columns are
renamed to their final positions up front and combined with
`pl.concat(..., how="horizontal")`. The alternative -- materializing a row index and
inner-joining on it -- would compute and compare a key that carries no information, and
would also stake row order on a join whose output ordering Polars does not guarantee
(`maintain_order` defaults to no guarantee), when row order IS pixel identity for a
band-wide frame. A horizontal concat preserves it by construction.

**No lowering here calls `assert_height_preserved` on itself.** Its docstring once invited
that, but the function collects both frames, and a band-wide frame's input side is
typically an unmaterialized raster scan -- so an inline call is two full reads of every
pixel, per op, to check something already structural: every lowering below builds its
output with `select`/`with_columns`/a horizontal concat of equal-height frames, none of
which can change a row count. The invariant is checked from the outside, in the tests
(`assert_height_preserved`'s docstring now says so).
"""

from collections.abc import Sequence

import polars as pl

from cpra.mp.data.analysis.execution import long_ops
from cpra.mp.data.analysis.execution.frame import (
    BandWideFrame,
    LongFrame,
    ReadWindow,
    add_reserved_band_index,
    assert_same_pixel_grid,
    band_column_names,
    empty_band_wide,
    has_no_bands,
    validate_extra_columns,
)
from cpra.mp.data.analysis.request.expressions import (
    PredicateKind,
    classify_where_kinds,
    compile_calculate_entries,
    parse_value_expr,
    rewrite_value_column,
    value_predicate_mask,
)
from cpra.mp.data.analysis.request.ops import CalculateOp, GroupOp, calculate_label

__all__ = [
    "JOIN_OPERAND_COLUMN",
    "mask_value_band_wide",
    "select_dimension_band_wide",
    "sample_grid_cells_band_wide",
    "filter_band_wide",
    "calculate_band_wide",
    "group_band_wide",
    "join_band_wide",
]

# The metadata column `join_band_wide` adds recording which operand (step name)
# contributed each surviving band -- see the module docstring. Distinct from both
# `BAND_NAME_RE` and `RESERVED_BAND_INDEX_COLUMN` (`_band`), so it can never collide
# with either positional-name shape `execution/frame.py`'s invariants police.
JOIN_OPERAND_COLUMN = "_join_operand"

# Horizontal (within-row, across-band) reduce, carried over faithfully from
# `analysis/sources.py`'s `HORIZONTAL_RASTER_AGGREGATES` -- see that dict's own docstring
# for why it has no `count` key. `count` is added here as this module's own extension
# (see the module docstring) rather than by mutating the original, since the two dicts
# answer different questions and must not be reached for each other's by accident.
_HORIZONTAL_AGGREGATES = {
    "max": pl.max_horizontal,
    "mean": pl.mean_horizontal,
    "min": pl.min_horizontal,
    "sum": pl.sum_horizontal,
}


def _horizontal_aggregate_expr(aggregate: str, exprs: list[pl.Expr], no_contributor: pl.Expr) -> pl.Expr:
    """`analysis/sources.py`'s `horizontal_aggregate_expr`, faithfully carried over, plus
    this module's own `count` branch -- see the module docstring for why `count` needs no
    `no_contributor` sentinel the other four aggregates do.
    """
    if aggregate == "count":
        return pl.sum_horizontal([expr.is_not_null().cast(pl.UInt32) for expr in exprs])
    agg_fn = _HORIZONTAL_AGGREGATES[aggregate]
    no_contributor_at = pl.all_horizontal([expr.is_null() for expr in exprs])
    return pl.when(no_contributor_at).then(no_contributor).otherwise(agg_fn(exprs))


def _combine_conjuncts(conjuncts: list[str]) -> str:
    return " and ".join(f"({conjunct})" for conjunct in conjuncts)


def mask_value_band_wide(band_wide: BandWideFrame, value_where: str, extra_columns: Sequence[str]) -> BandWideFrame:
    """The band-wide lowering of a VALUE predicate conjunct: null out a band's own pixel
    wherever `value_where` (written once, against the placeholder `value`) evaluates false
    for it -- carrying over `analysis/sources.py`'s `masked_value_predicate_exprs` logic
    (a per-band `pl.when(mask).then(pl.col(name)).otherwise(None)`, `value_where` rewritten
    per band via `rewrite_value_column`), never a row drop: dropping a pixel's row would
    drop it from every OTHER band too, including the ones the predicate holds for.

    `extra_columns` is carried through unchanged and also added to each band's own value
    namespace (`[name, *extra_columns]`), matching `masked_value_predicate_exprs`'
    convention -- a value condition may reference a carried column (a crosswalk zone id,
    notably) in addition to the band it is masking.
    """
    validate_extra_columns(extra_columns)
    lf, metadata_df, window = band_wide
    names = band_column_names(metadata_df)
    exprs = []
    for name in names:
        mask = value_predicate_mask(rewrite_value_column(value_where, name), allowed_columns=[name, *extra_columns])
        exprs.append(pl.when(mask).then(pl.col(name)).otherwise(None).alias(name))
    out_lf = lf.select([*exprs, *extra_columns])
    return out_lf, metadata_df, window


def select_dimension_band_wide(
    band_wide: BandWideFrame, dimension_where: str, extra_columns: Sequence[str]
) -> BandWideFrame:
    """The band-wide lowering of a DIMENSION predicate conjunct: a column selection over
    `metadata_df`, not a per-pixel operation at all -- a dimension conjunct decides WHICH
    BANDS survive, evaluated once per band against its metadata row, never once per pixel.

    Renumbers the surviving bands positionally (`band_column_names` is always
    `b0..b{height-1}`; see that function's docstring), so the returned frame's band names
    are contiguous even though the metadata rows they describe may have come from anywhere
    in the original set.

    Returns `empty_band_wide(window)` when no band survives -- the one lowering this
    module's height-preservation invariant does not cover, named as the exception rather
    than special-cased inside `assert_height_preserved` (see that function's docstring).
    """
    validate_extra_columns(extra_columns)
    lf, metadata_df, window = band_wide
    names = band_column_names(metadata_df)
    indexed = metadata_df.with_row_index("__pos")
    kept = indexed.filter(pl.sql_expr(dimension_where).fill_null(False))
    if kept.height == 0:
        return empty_band_wide(window)
    positions = kept.get_column("__pos").to_list()
    kept_metadata = kept.drop("__pos")
    new_names = band_column_names(kept_metadata)
    select_exprs = [pl.col(names[position]).alias(new_name) for position, new_name in zip(positions, new_names)]
    out_lf = lf.select([*select_exprs, *extra_columns])
    return out_lf, kept_metadata, window


def sample_grid_cells_band_wide(band_wide: BandWideFrame, where: str, extra_columns: Sequence[str]) -> LongFrame:
    """The band-wide lowering of a `where` containing a SPATIAL (`grid_cell`) conjunct:
    the one case that changes the frame's shape, from band-wide to long -- see the module
    docstring for why the transition itself, not a row drop, is what "spatial filter"
    means on the band-wide side.

    Unconditionally unpivots every pixel into one long row per (pixel, band) observation:
    `grid_cell` is the pixel's row-major position within `window` (an ordinary
    `with_row_index`, since row order is the only positional information a band-wide frame
    keeps -- `execution/frame.py`'s module docstring), and each band's row picks up that
    band's own dimension values from `metadata_df` via a join on band identity. `where` in
    its ENTIRETY -- every conjunct, of every kind, not just the spatial one(s) -- is then
    applied by `execution.long_ops.filter_long`, which already knows how to run a mixed
    dimension/value/spatial `where` against a long frame (a spatial conjunct is just an
    ordinary `grid_cell` column there).
    """
    validate_extra_columns(extra_columns)
    lf, metadata_df, window = band_wide
    names = band_column_names(metadata_df)
    indexed = lf.with_row_index("grid_cell").with_columns(pl.col("grid_cell").cast(pl.Int64))
    long_lf = indexed.unpivot(index=["grid_cell", *extra_columns], on=names, variable_name="__band", value_name="value")
    band_identity = pl.DataFrame({"__band": pl.Series(names, dtype=pl.String)}).hstack(metadata_df)
    with_metadata = long_lf.join(band_identity.lazy(), on="__band", how="left").drop("__band")
    return long_ops.filter_long(with_metadata, where)


def filter_band_wide(band_wide: BandWideFrame, where: str, extra_columns: Sequence[str]) -> BandWideFrame | LongFrame:
    """The `filter` op's band-wide dispatcher: classify `where`'s conjuncts
    (`request.expressions.classify_where_kinds`) and route to the one of the three
    predicate-kind lowerings above that a spatial conjunct's presence selects -- exactly
    the precedence `request.ops._type_filter` already types this against (spatial, if
    present, wins outright; otherwise dimension runs before value -- see the module
    docstring for why that order is a non-observable choice rather than a requirement).

    A `where` combining a dimension and a value conjunct (no spatial one) stays band-wide
    the whole time: `select_dimension_band_wide` first, `mask_value_band_wide` second,
    short-circuiting to the empty result if the dimension half already leaves no band
    behind (nothing for a value mask to do to zero bands).
    """
    validate_extra_columns(extra_columns)
    kinds = classify_where_kinds(where)
    if kinds[PredicateKind.spatial]:
        return sample_grid_cells_band_wide(band_wide, where, extra_columns)

    result = band_wide
    if kinds[PredicateKind.dimension]:
        result = select_dimension_band_wide(result, _combine_conjuncts(kinds[PredicateKind.dimension]), extra_columns)
        if has_no_bands(result[1]):
            return result
    if kinds[PredicateKind.value]:
        result = mask_value_band_wide(result, _combine_conjuncts(kinds[PredicateKind.value]), extra_columns)
    return result


def calculate_band_wide(band_wide: BandWideFrame, op: CalculateOp, extra_columns: Sequence[str]) -> BandWideFrame:
    """The band-wide lowering of a `calculate`: per band, wherever `op.where` matches (the
    dimension half decides WHICH BANDS the entry applies to at all, evaluated once against
    `metadata_df`; the value half is folded into that band's own per-pixel mask), rewrite
    the pixel to `op.expression`; elsewhere -- a band the entry does not apply to, or a
    pixel the value half rejects -- pass through unchanged, never dropped (`calculate`
    never drops data; that is `filter`'s job).

    Carries over `analysis/sources.py`'s `band_calculate_passes` logic for exactly one
    entry (a `CalculateOp` is always one, never that function's multi-entry list -- see the
    module docstring), including its `is_not_null()` guard (nodata in, nodata out, as a
    structural property rather than a per-expression obligation).
    """
    validate_extra_columns(extra_columns)
    lf, metadata_df, window = band_wide
    names = band_column_names(metadata_df)
    (plan,) = compile_calculate_entries(
        [(calculate_label(op), op.expression, op.where)], allowed_columns=["value", *extra_columns]
    )

    if plan.dimension_where is not None:
        applicable = (
            metadata_df.select(pl.sql_expr(plan.dimension_where).fill_null(False).alias("__ok"))
            .get_column("__ok")
            .to_list()
        )
    else:
        applicable = [True] * len(names)

    exprs = []
    for name, applies in zip(names, applicable):
        if not applies:
            continue
        condition = pl.col(name).is_not_null()
        if plan.value_where is not None:
            condition = condition & value_predicate_mask(
                rewrite_value_column(plan.value_where, name), allowed_columns=[name, *extra_columns]
            )
        new_value = parse_value_expr(
            rewrite_value_column(plan.expression, name), allowed_columns=[name, *extra_columns]
        )
        exprs.append(pl.when(condition).then(new_value).otherwise(pl.col(name)).alias(name))

    out_lf = lf.with_columns(exprs).select([*names, *extra_columns]) if exprs else lf.select([*names, *extra_columns])
    return out_lf, metadata_df, window


def group_band_wide(band_wide: BandWideFrame, op: GroupOp, extra_columns: Sequence[str]) -> BandWideFrame:
    """The band-wide lowering of a `group`: a HORIZONTAL reduce across each output group's
    contributing bands, at every pixel -- the within-row axis, unrelated to
    `execution.long_ops.group_long`'s across-row collapse (`execution/fold.py`'s
    cliff-shaped-collapse fold rule has no counterpart here; see the module docstring of
    whatever caller reasons about that).

    `op.by` (may be empty) groups `metadata_df`'s rows exactly like a long `group_long`
    groups long rows, except the aggregate reduces ACROSS the bands one group collects
    rather than across the rows one grid_cell collects. `op.by == []` collapses to a
    single output band and hits the same "zero columns loses the row count" gap
    `empty_band_wide` exists for on the pixel side -- `add_reserved_band_index` is this
    lowering's answer on the metadata side; see the module docstring for why the reserved
    column is left standing in what this function returns rather than dropped here.

    `op.of` is ignored -- see the module docstring for why a band-wide frame has no second
    column a `group` could plausibly read instead of the band itself.
    """
    validate_extra_columns(extra_columns)
    lf, metadata_df, window = band_wide
    if has_no_bands(metadata_df):
        return empty_band_wide(window)

    names = band_column_names(metadata_df)
    indexed = metadata_df.with_row_index("__pos")
    groups_df = indexed.group_by(op.by, maintain_order=True).agg(pl.col("__pos"))

    if op.by:
        metadata_out = groups_df.select(op.by)
    else:
        metadata_out = add_reserved_band_index(groups_df).drop("__pos")

    no_contributor = pl.lit(None)
    band_exprs = []
    for k, positions in enumerate(groups_df.get_column("__pos").to_list()):
        contributing = [pl.col(names[position]) for position in positions]
        band_exprs.append(_horizontal_aggregate_expr(op.aggregate, contributing, no_contributor).alias(f"b{k}"))

    out_lf = lf.select([*band_exprs, *extra_columns])
    return out_lf, metadata_out, window


def join_band_wide(operands: list[BandWideFrame], names: list[str], extra_columns: Sequence[str]) -> BandWideFrame:
    """The band-wide lowering of a `join`: positional row alignment across `operands`
    (already resolved, in `op.inputs` order, `names` the matching step names) rather than
    a key-based join -- row *i* of every operand is already the same pixel once every
    operand's window addresses the same pixel grid (this function's precondition, asserted
    pairwise via `execution.frame.assert_same_pixel_grid`; see the module docstring for why
    that and not window equality). So there is no join KEY to compute and none is computed:
    the bands are renamed to their final positions and combined with
    `pl.concat(..., how="horizontal")`.

    Band columns concatenate in `names` order and renumber positionally (`b0..b{N-1}`
    across every operand's bands together, not per operand); `metadata_df`s concatenate
    the same way (`how="diagonal_relaxed"`, since two operands' dimension columns need not
    be identical), each row gaining `JOIN_OPERAND_COLUMN` naming which operand it came
    from -- the band-wide reading of "each operand's value renamed to its step name"; see
    the module docstring.

    `extra_columns` is read from `operands[0]` alone: a column both operands carry (a
    shared crosswalk, say) is guaranteed identical row-for-row once the pixel-grid
    precondition holds, so there is nothing a second copy from another operand would add.
    """
    validate_extra_columns(extra_columns)
    if len(operands) != len(names):
        raise ValueError(
            f"join_band_wide got {len(operands)} operand(s) but {len(names)} name(s) -- exactly one name per operand"
        )
    if len(operands) < 2:
        raise ValueError(f"join_band_wide requires at least two operands, got {len(operands)}")

    window: ReadWindow = operands[0][2]
    for _lf, _metadata_df, other_window in operands[1:]:
        assert_same_pixel_grid(window, other_window)

    # Each operand's bands are renamed straight to their final positions, so the frames
    # going into the horizontal concat already share no column name and the concat needs
    # no reconciliation of its own. `band_column_names`' naming is inlined here rather than
    # called: it names a frame's bands from ITS OWN metadata height, and the offset each
    # operand's bands start at is a property of the combination, not of that operand.
    aligned = []
    renamed_metadatas = []
    next_position = 0
    for k, ((lf, metadata_df, _window), name) in enumerate(zip(operands, names)):
        local_names = band_column_names(metadata_df)
        select_cols = [
            pl.col(local_name).alias(f"b{next_position + i}") for i, local_name in enumerate(local_names)
        ]
        next_position += len(local_names)
        if k == 0:
            select_cols += [pl.col(column) for column in extra_columns]
        aligned.append(lf.select(select_cols))
        renamed_metadatas.append(metadata_df.with_columns(pl.lit(name).alias(JOIN_OPERAND_COLUMN)))

    combined = pl.concat(aligned, how="horizontal")
    out_lf = combined.select([*(f"b{i}" for i in range(next_position)), *extra_columns])
    merged_metadata = pl.concat(renamed_metadatas, how="diagonal_relaxed")
    return out_lf, merged_metadata, window
