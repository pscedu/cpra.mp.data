"""The band-wide-shape lowerings: `mask_value_band_wide`/`select_dimension_band_wide`/
`sample_grid_cells_band_wide` (the three `filter` predicate kinds), `calculate_band_wide`,
`group_band_wide`, and `join_band_wide` -- the band-wide counterparts of
`execution/long_ops.py`'s four long-shape lowerings, plus `filter_band_wide`, the
dispatcher a `filter` op with a mixed-kind `where` runs through.

**A fourth filter lowering for PAIRED frames.** The three kinds above assume a frame with
one measured quantity. A frame a `join` left carrying one operand-named value column per
operand (`request.ops.FrameType.value_columns`) has several, and a `where` over it
(`marsh > 0.5 and sal > 5.5`) asks one question about a PAIR of bands rather than a
question about each band separately -- so the value branch routes to
`pair_and_mask_band_wide` instead, which reconstitutes the pairs from band metadata,
collapses each to one band, and masks it. `filter_band_wide` picks between the two on
`value_columns` alone; everything else about the dispatch is unchanged. See that
function for what pairs with what, and why an unpairable band is dropped rather than
raised on.

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
calculate_long` uses, applied per band. This module's version handles exactly one
`CalculateOp` at a time (a `calculate` step is always one entry here, never a list), and
extends the value mask's namespace to `extra_columns` in addition to the band's own
column, matching
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

**`group_band_wide`'s horizontal `count`.** A band-wide `group.aggregate == "count"`
answers "how many of these bands contributed a non-null value at this pixel", the
per-pixel cross-band analogue of `execution.long_ops.group_long`'s row count. The other
four horizontal aggregates are pure `pl.*_horizontal` reduces; `count` is not, which is
why `_HORIZONTAL_AGGREGATES` below does not carry it.
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

**`GroupOp.of`, band-wide: the bands say WHETHER, `of` says WHICH.** `of` picks which
column an aggregate reads in long shape (default `"value"`, generalizing to e.g. "first
year a condition held"). A band-wide frame has no second per-PIXEL column to read
instead -- but it does have `metadata_df`, one row of dimension values per band, and
that is where the long-shape reading lands: `of: calendar_year` makes each band
contribute its own calendar year, as a scalar, at every pixel where that band is
non-null. The two halves of "first year a condition held" then sit in the two places
band-wide shape keeps them: the condition is the per-pixel mask an upstream `filter`
already applied to each band, and the year is that band's metadata row.
`min of=calendar_year` over the survivors is the whole of it. An `of` naming neither
`value` nor a metadata column is rejected rather than ignored.

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
    DOMAIN_COLUMN,
    RESERVED_BAND_INDEX_COLUMN,
    BandWideFrame,
    LongFrame,
    ReadWindow,
    add_reserved_band_index,
    assert_same_pixel_grid,
    band_column_names,
    empty_band_wide,
    has_no_bands,
    row_major_grid_cell,
    validate_extra_columns,
)
from cpra.mp.data.analysis.request.expressions import (
    PredicateKind,
    classify_where_kinds,
    compile_calculate_entries,
    parse_value_expr,
    rewrite_value_column,
    rewrite_value_columns,
    value_predicate_mask,
)
from cpra.mp.data.analysis.request.ops import CalculateOp, GroupOp, calculate_label

__all__ = [
    "JOIN_OPERAND_COLUMN",
    "PAIRING_KEY_EXCLUDED_COLUMNS",
    "mask_value_band_wide",
    "pair_and_mask_band_wide",
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

# Metadata columns `pair_and_mask_band_wide` never pairs on -- see its docstring.
# `variable`/`time_unit`/`grid` identify WHICH DATASET a band came from, and a spatial
# join's whole premise is that its operands are different datasets; `path`/`band` name a
# physical file and are per-band by construction; `JOIN_OPERAND_COLUMN` is the axis being
# paired ACROSS, so pairing on it would put every operand in its own group and pair
# nothing with anything.
PAIRING_KEY_EXCLUDED_COLUMNS = frozenset({"variable", "time_unit", "grid", "path", "band", JOIN_OPERAND_COLUMN})

# Horizontal (within-row, across-band) reduce. No `count` key: `count` is not a
# `pl.*_horizontal` reduce and gets its own branch below -- see the module docstring.
_HORIZONTAL_AGGREGATES = {
    "max": pl.max_horizontal,
    "mean": pl.mean_horizontal,
    "min": pl.min_horizontal,
    "sum": pl.sum_horizontal,
}


def _horizontal_aggregate_expr(aggregate: str, exprs: list[pl.Expr], no_contributor: pl.Expr) -> pl.Expr:
    """The horizontal reduce for `aggregate`, plus the `count` branch -- see the module
    docstring for why `count` needs no `no_contributor` sentinel the other four
    aggregates do.
    """
    if aggregate == "count":
        # Signed, though a count never is: this lands in a GeoTIFF whose nodata sentinel
        # (`execution.sink.DEFAULT_NODATA`) is negative, and an unsigned band has nowhere
        # to put it. The width is the same either way.
        return pl.sum_horizontal([expr.is_not_null().cast(pl.Int32) for expr in exprs])
    agg_fn = _HORIZONTAL_AGGREGATES[aggregate]
    no_contributor_at = pl.all_horizontal([expr.is_null() for expr in exprs])
    return pl.when(no_contributor_at).then(no_contributor).otherwise(agg_fn(exprs))


def _combine_conjuncts(conjuncts: list[str]) -> str:
    return " and ".join(f"({conjunct})" for conjunct in conjuncts)


def mask_value_band_wide(band_wide: BandWideFrame, value_where: str, extra_columns: Sequence[str]) -> BandWideFrame:
    """The band-wide lowering of a VALUE predicate conjunct: null out a band's own pixel
    wherever `value_where` (written once, against the placeholder `value`) evaluates false
    for it -- a per-band `pl.when(mask).then(pl.col(name)).otherwise(None)`, with
    `value_where` rewritten per band via `rewrite_value_column`, never a row drop:
    dropping a pixel's row would
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


def _pairing_key_columns(metadata_df: pl.DataFrame) -> list[str]:
    """The metadata columns `pair_and_mask_band_wide` pairs bands on: every column that is
    non-null in EVERY band's row, minus `PAIRING_KEY_EXCLUDED_COLUMNS`.

    The non-null test is what makes this work without a schema: `join_band_wide` merges
    its operands' metadata with `how="diagonal_relaxed"`, so a column only one operand
    carries (a raster's `ffibs_type`, a vector's `aggregate_type`) is present but null on
    the other side, and pairing on it could only ever pair a band with nothing. A column
    every operand carries a real value for is exactly a dimension they can be compared
    along, which is exactly what a pairing key is.
    """
    return [
        name
        for name, nulls in zip(metadata_df.columns, metadata_df.null_count().row(0))
        if nulls == 0 and name not in PAIRING_KEY_EXCLUDED_COLUMNS
    ]


def pair_and_mask_band_wide(
    band_wide: BandWideFrame, value_where: str, extra_columns: Sequence[str], value_columns: Sequence[str]
) -> BandWideFrame:
    """The band-wide lowering of a value predicate over a PAIRED frame -- one a `join` left
    carrying one operand-named value column per operand (`request.ops.FrameType.
    value_columns`), where `value_where` names those operands rather than the placeholder
    `value`: `marsh > 0.5 and sal > 5.5`.

    A paired frame's bands are not comparable as they stand. `marsh`'s 2030 band and
    `sal`'s 2030 band are two measurements OF THE SAME THING, and the predicate is one
    question about that pair -- but band-wide shape has no column to put a pair in, only a
    flat positional list in which the two sit side by side with nothing saying they belong
    together. So this lowering does what `mask_value_band_wide` cannot: it reconstitutes
    the pairs, and COLLAPSES each one to a single band.

    * Bands pair by `_pairing_key_columns` -- the dimensions every operand carries a real
      value for (`calendar_year`, in practice, plus whatever scenario columns the operands
      agree on). A group with anything other than exactly one band per operand is dropped,
      not raised on: that is a pairing MISS (an operand has no counterpart at this
      dimension combination), the same non-event as a pixel whose crosswalk finds no
      vector cell, and an empty result for a request that pairs nothing is a legitimate
      all-nodata answer rather than an error.
    * Each surviving pair becomes ONE band, carrying the value of its LOWEST-POSITIONED
      band -- which is the operand the request named first, since `join_band_wide` lays
      its operands' bands out in `JoinOp.inputs` order. A pair has two measurements and
      an output band has room for one; nothing in the request says which, and the
      first-named operand is the only determinate choice available. (Read off the band
      positions rather than off `value_columns`, whose order is not meaningful: it comes
      from a `frozenset`.) It rarely matters -- band-wide `calculate` is rejected over a
      paired frame, so the collapsed value's only consumer is a `group`, and a `group`
      that means to reduce something else says so with `of` -- but it must be SOME value,
      and it must be the same one every time.
    * The predicate masks that band: `value_where` has every operand name rewritten to its
      own band column in this pair (`request.expressions.rewrite_value_columns`, one pass
      -- a conjunct may reference two operands at once, `marsh > sal`), then evaluates
      once per pair as a single `pl.when(...)`. Null where the pair fails, exactly the
      masking discipline `mask_value_band_wide` uses and for the same reason.

    The returned `metadata_df` carries the pairing key columns alone, one row per
    surviving pair -- the dimensions that still mean something once the operand axis is
    gone. `add_reserved_band_index` covers the no-shared-dimension case for the same
    "zero columns loses the row count" reason `group_band_wide`'s `by == []` branch does.
    """
    validate_extra_columns(extra_columns)
    lf, metadata_df, window = band_wide
    if has_no_bands(metadata_df):
        return empty_band_wide(window)
    names = band_column_names(metadata_df)
    operands = list(value_columns)
    key_columns = _pairing_key_columns(metadata_df)

    indexed = metadata_df.with_row_index("__pos")
    if key_columns:
        grouped = indexed.group_by(key_columns, maintain_order=True).agg(
            pl.col("__pos"), pl.col(JOIN_OPERAND_COLUMN).alias("__operands")
        )
    else:
        grouped = indexed.select(
            pl.col("__pos").implode(), pl.col(JOIN_OPERAND_COLUMN).implode().alias("__operands")
        )

    kept_rows = []
    band_exprs = []
    for row_index, (positions, row_operands) in enumerate(
        zip(grouped.get_column("__pos").to_list(), grouped.get_column("__operands").to_list())
    ):
        by_operand = dict(zip(row_operands, positions))
        if len(by_operand) != len(positions) or set(by_operand) != set(operands):
            continue
        substitutions = {operand: names[position] for operand, position in by_operand.items()}
        mask = value_predicate_mask(
            rewrite_value_columns(value_where, substitutions),
            allowed_columns=[*substitutions.values(), *extra_columns],
        )
        carried = pl.col(names[min(positions)])
        band_exprs.append(pl.when(mask).then(carried).otherwise(None).alias(f"b{len(kept_rows)}"))
        kept_rows.append(row_index)

    if not kept_rows:
        return empty_band_wide(window)

    kept = grouped[kept_rows]
    metadata_out = (
        kept.select(key_columns)
        if key_columns
        else add_reserved_band_index(kept).select(RESERVED_BAND_INDEX_COLUMN)
    )
    return lf.select([*band_exprs, *extra_columns]), metadata_out, window


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

    Also returns `empty_band_wide(window)` immediately, without evaluating
    `dimension_where` at all, when `band_wide` is ALREADY the empty triple: this is the
    walk's own redundant re-application of the predicate that caused an input to be
    opened empty in the first place (`ExecutionPlan`'s compile-time band-metadata pushdown
    narrows a raster origin before `execute()` ever runs, but the filter step that named
    the predicate still executes normally afterward, same as every other step). Zero bands
    means `metadata_df` carries none of its usual dimension columns either -- see
    `empty_band_wide`'s own docstring -- so attempting the SQL filter against it would fail
    on a missing column rather than correctly reporting "still nothing survives".
    """
    validate_extra_columns(extra_columns)
    lf, metadata_df, window = band_wide
    if has_no_bands(metadata_df):
        return empty_band_wide(window)
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


def sample_grid_cells_band_wide(
    band_wide: BandWideFrame, where: str, extra_columns: Sequence[str], *, grid_width: int
) -> LongFrame:
    """The band-wide lowering of a `where` containing a SPATIAL (`grid_cell`) conjunct:
    the one case that changes the frame's shape, from band-wide to long -- see the module
    docstring for why the transition itself, not a row drop, is what "spatial filter"
    means on the band-wide side.

    Unconditionally unpivots every pixel into one long row per (pixel, band) observation:
    `grid_cell` is the pixel's ABSOLUTE row-major position over the full `grid_width`-wide
    source grid, via `frame.row_major_grid_cell`'s absolute mode and NOT a plain
    `with_row_index` or a window-relative id -- row order is the only positional
    information a band-wide frame keeps, but that order is the order the READ produced,
    which is row-major only for a single-batch read (see that function, and
    `execution/frame.py`'s module docstring), and `window` may itself cover only part of
    the grid (`execution.execute`'s point-sample window narrowing). Using the ABSOLUTE id
    rather than a window-relative one is what keeps that narrowing a pure read-size
    optimization: a caller's `where: "grid_cell = N"` names the same pixel, and gets the
    same `N` back, whether or not the read that answered it happened to be narrowed.
    `grid_width` is therefore required here, not optional -- there is no meaningful
    fallback once a spatial conjunct is in play. Each band's row then picks up that band's
    own dimension values from `metadata_df` via a join on band identity. `where` in its
    ENTIRETY -- every conjunct, of every kind, not just the spatial one(s) -- is then
    applied by `execution.long_ops.filter_long`, which already knows how to run a mixed
    dimension/value/spatial `where` against a long frame (a spatial conjunct is just an
    ordinary `grid_cell` column there, and needs no further rewriting: the id it names is
    already the id `row_major_grid_cell` just produced).
    """
    validate_extra_columns(extra_columns)
    lf, metadata_df, window = band_wide
    names = band_column_names(metadata_df)
    indexed = row_major_grid_cell(lf, window, grid_width=grid_width)
    long_lf = indexed.unpivot(index=["grid_cell", *extra_columns], on=names, variable_name="__band", value_name="value")
    band_identity = pl.DataFrame({"__band": pl.Series(names, dtype=pl.String)}).hstack(metadata_df)
    with_metadata = long_lf.join(band_identity.lazy(), on="__band", how="left").drop("__band")
    return long_ops.filter_long(with_metadata, where)


def filter_band_wide(
    band_wide: BandWideFrame,
    where: str,
    extra_columns: Sequence[str],
    *,
    grid_width: int | None = None,
    value_columns: Sequence[str] = ("value",),
) -> BandWideFrame | LongFrame:
    """The `filter` op's band-wide dispatcher: classify `where`'s conjuncts
    (`request.expressions.classify_where_kinds`) and route to the one of the three
    predicate-kind lowerings above that a spatial conjunct's presence selects -- exactly
    the precedence `request.ops._type_filter` already types this against (spatial, if
    present, wins outright; otherwise dimension runs before value -- see the module
    docstring for why that order is a non-observable choice rather than a requirement).

    `value_columns` is the incoming frame's value columns (`request.ops.FrameType.
    value_columns`), and does two things. It is forwarded to `classify_where_kinds`, so a
    PAIRED frame's operand-named conjuncts (`marsh > 0.5`) classify as VALUE rather than
    falling through to dimension for not being spelled `value`. And when there is more
    than one of them, the value branch runs `pair_and_mask_band_wide` instead of
    `mask_value_band_wide`: a paired frame's bands have to be paired up before a predicate
    naming two operands means anything. It defaults to the unpaired `("value",)`, which is
    every frame that never went through a `join`.

    `grid_width` is only needed for the spatial branch (`sample_grid_cells_band_wide`
    requires it, to decode absolute pixel ids); left `None` for a `where` that turns out
    to have no spatial conjunct, since neither `select_dimension_band_wide` nor
    `mask_value_band_wide` needs it. A spatial conjunct with no `grid_width` supplied is a
    caller error, not a case to silently fall back on -- raised here rather than left to
    surface as a `TypeError` one call down.

    A `where` combining a dimension and a value conjunct (no spatial one) stays band-wide
    the whole time: `select_dimension_band_wide` first, `mask_value_band_wide` second,
    short-circuiting to the empty result if the dimension half already leaves no band
    behind (nothing for a value mask to do to zero bands).
    """
    validate_extra_columns(extra_columns)
    kinds = classify_where_kinds(where, value_columns)
    if kinds[PredicateKind.spatial]:
        if grid_width is None:
            raise ValueError(
                f"filter_band_wide needs grid_width to decode the spatial conjunct in {where!r} "
                "into absolute pixel ids"
            )
        return sample_grid_cells_band_wide(band_wide, where, extra_columns, grid_width=grid_width)

    result = band_wide
    if kinds[PredicateKind.dimension]:
        result = select_dimension_band_wide(result, _combine_conjuncts(kinds[PredicateKind.dimension]), extra_columns)
        if has_no_bands(result[1]):
            return result
    if kinds[PredicateKind.value]:
        value_where = _combine_conjuncts(kinds[PredicateKind.value])
        if len(value_columns) > 1:
            result = pair_and_mask_band_wide(result, value_where, extra_columns, value_columns)
        else:
            result = mask_value_band_wide(result, value_where, extra_columns)
    return result


def calculate_band_wide(band_wide: BandWideFrame, op: CalculateOp, extra_columns: Sequence[str]) -> BandWideFrame:
    """The band-wide lowering of a `calculate`: per band, wherever `op.where` matches (the
    dimension half decides WHICH BANDS the entry applies to at all, evaluated once against
    `metadata_df`; the value half is folded into that band's own per-pixel mask), rewrite
    the pixel to `op.expression`; elsewhere -- a band the entry does not apply to, or a
    pixel the value half rejects -- pass through unchanged, never dropped (`calculate`
    never drops data; that is `filter`'s job).

    Handles exactly one entry (a `CalculateOp` is always one, never a list -- see the
    module docstring), including the `is_not_null()` guard (nodata in, nodata out, as a
    structural property rather than a per-expression obligation).

    Returns `band_wide` unchanged, without evaluating `op.where`'s dimension half at all,
    when it is ALREADY the empty triple -- same redundant-re-application situation
    `select_dimension_band_wide` documents: zero bands means no dimension columns to
    evaluate a `where` against, and "nothing happens to zero bands" is already the correct
    answer without asking.
    """
    validate_extra_columns(extra_columns)
    lf, metadata_df, window = band_wide
    if has_no_bands(metadata_df):
        return band_wide
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

    `op.of` names the per-band quantity the aggregate reads. Left at its `"value"` default
    it is the band column itself, the only measurement a band-wide frame has. Set to a
    METADATA column (`of: calendar_year`) it is that band's own value for that column --
    a scalar per band, not a pixel array -- contributed wherever the band is non-null at
    this pixel and withheld where it is not. That is what makes "the first year this
    condition held" expressible in band-wide shape: the bands carry WHETHER, and `of`
    supplies WHICH, so `min of=calendar_year` reduces to the earliest year whose band
    survived. Any other `of` is rejected: it names neither a metadata column nor the
    band, so there is nothing for it to read.

    A frame carrying `frame.DOMAIN_COLUMN` (a vector-to-raster `project` is upstream) has
    every output band additionally nulled wherever that column is -- a pixel outside the
    crosswalk's domain gets nodata, never a real answer. This is invisible to the four
    reducing aggregates, whose answer there is already null; it exists for `count`, which
    would otherwise report a confident 0 for a pixel nobody ever asked about. See
    `DOMAIN_COLUMN`'s own comment.
    """
    validate_extra_columns(extra_columns)
    lf, metadata_df, window = band_wide
    if has_no_bands(metadata_df):
        return empty_band_wide(window)
    names = band_column_names(metadata_df)
    if op.of != "value" and op.of not in metadata_df.columns:
        raise ValueError(
            f"group of={op.of!r} names neither a band-wide frame's own measurement "
            f"('value') nor one of its band metadata columns ({sorted(metadata_df.columns)})"
        )
    of_values = metadata_df.get_column(op.of).to_list() if op.of != "value" else None
    indexed = metadata_df.with_row_index("__pos")
    groups_df = indexed.group_by(op.by, maintain_order=True).agg(pl.col("__pos"))

    if op.by:
        metadata_out = groups_df.select(op.by)
    else:
        metadata_out = add_reserved_band_index(groups_df).drop("__pos")

    no_contributor = pl.lit(None)
    in_domain = pl.col(DOMAIN_COLUMN).is_not_null() if DOMAIN_COLUMN in extra_columns else None
    band_exprs = []
    for k, positions in enumerate(groups_df.get_column("__pos").to_list()):
        if of_values is None:
            contributing = [pl.col(names[position]) for position in positions]
        else:
            # The band decides WHETHER it contributes; `of` decides WHAT. `pl.when` on the
            # band's own nullity rather than a bare literal, so a band masked out upstream
            # (a failed predicate, a nodata pixel) withholds its `of` value at that pixel
            # instead of contributing it everywhere the band merely exists.
            contributing = [
                pl.when(pl.col(names[position]).is_not_null()).then(pl.lit(of_values[position])).otherwise(None)
                for position in positions
            ]
        expr = _horizontal_aggregate_expr(op.aggregate, contributing, no_contributor)
        if in_domain is not None:
            expr = pl.when(in_domain).then(expr).otherwise(None)
        band_exprs.append(expr.alias(f"b{k}"))

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

    `extra_columns` names the carried columns the OUTPUT should have; each is taken from
    the first operand whose frame actually carries it. A column several operands carry (a
    shared crosswalk, say) is guaranteed identical row-for-row once the pixel-grid
    precondition holds, so which copy is taken cannot matter -- but WHETHER a column is
    taken does, because an operand can introduce one the others never had:
    `execution.project.project_vector_to_raster` attaches `frame.DOMAIN_COLUMN` to its own
    output, and a join pairing that projection against a plain raster read would drop it
    if only `operands[0]` were consulted.
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
    unclaimed = list(extra_columns)
    for (lf, metadata_df, _window), name in zip(operands, names):
        local_names = band_column_names(metadata_df)
        select_cols = [
            pl.col(local_name).alias(f"b{next_position + i}") for i, local_name in enumerate(local_names)
        ]
        next_position += len(local_names)
        # First operand that carries it wins, and it is then off the list -- taking a
        # second copy would put two columns of the same name into the horizontal concat.
        present = lf.collect_schema().names()
        claimed = [column for column in unclaimed if column in present]
        select_cols += [pl.col(column) for column in claimed]
        unclaimed = [column for column in unclaimed if column not in claimed]
        aligned.append(lf.select(select_cols))
        # Guarded on height: `with_columns` of a literal against a zero-ROW frame
        # broadcasts to length ONE rather than staying empty, so a zero-band operand would
        # otherwise contribute a metadata row describing a band that does not exist -- and
        # two of them would hand back metadata claiming two bands over a frame carrying
        # none, the inverse of the "zero columns loses the row count" gap this module's
        # docstring describes. A zero-band operand contributes zero band rows; that is the
        # whole of its contribution.
        renamed_metadatas.append(
            metadata_df.with_columns(pl.lit(name).alias(JOIN_OPERAND_COLUMN))
            if metadata_df.height
            else metadata_df
        )

    combined = pl.concat(aligned, how="horizontal")
    out_lf = combined.select([*(f"b{i}" for i in range(next_position)), *extra_columns])
    merged_metadata = pl.concat(renamed_metadatas, how="diagonal_relaxed")
    return out_lf, merged_metadata, window
