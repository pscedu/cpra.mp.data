"""The band-wide-shape lowerings: `mask_value_band_wide`/`select_dimension_band_wide`/
`sample_grid_cells_band_wide` (the three `filter` predicate kinds),
`case_band_wide`, `group_band_wide`, and `join_band_wide` -- the band-wide counterparts of
`execution/long_ops.py`'s long-shape lowerings, plus `filter_band_wide`, the
dispatcher a `filter` op with a mixed-kind `where` runs through.

**PAIRED frames are `case_band_wide`'s business alone.** The three filter kinds assume a
frame with one measured quantity. A frame a `join` left carrying one operand-named value
column per operand (`request.ops.FrameType.value_columns`) has several, and an expression
over it (`marsh > 0.5 and sal > 5.5`) asks one question about a PAIR of bands rather than
a question about each band separately. Answering it means reconstituting the pairs from
band metadata and reducing each to one band -- and a reduce has to say which operand's
value that band carries, which a predicate has no room to state and a `case` arm's `then`
does. So `case_band_wide` reads `value_columns` the way `filter_band_wide` merely
classifies against it: one value column makes it a per-band map, several make it a reduce
over reconstituted pairs (`_pair_band_groups`, `_paired_metadata`). A value predicate over
a paired frame never reaches `filter_band_wide` at all -- `request.ops._type_filter`
rejects it, pointing at `case`.

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

**PAIRED frames are `case_band_wide`'s business alone -- with one exception, the SPATIAL
filter.** A spatial conjunct does not evaluate anything against a paired frame's operands,
so it needs none of the pairing-and-reducing the paragraph above rules out; it changes the
frame's SHAPE, and the paired long shape it has to land in (one row per key, one
operand-named value column per operand) is not something a single unpivot produces. That
transition is `_sample_grid_cells_joined`, and it performs the operand combination by
calling `execution.long_ops.join_long` itself -- see its docstring.

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

**`case_band_wide`'s per-band arm applicability.** A `case` arm's `when` may carry a
dimension conjunct, and band-wide shape keeps its dimension values in `metadata_df` (one
row per band) rather than per pixel -- so that half of the `when` has a per-BAND answer,
evaluated once across the whole metadata frame by `_band_applicability` and indexed by
band, never once per pixel. What the answer decides is which ARMS enter a given band's
chain, not whether the band is touched at all: an arm a band fails drops out of that band's
`pl.when` chain and every other arm and the `else` still apply. The paired reading asks the
same question of a pair, against `_coalesced_pair_metadata`.

The value half of a `when`, the `then`, and the `else` all take the band's own column PLUS
`extra_columns` as their namespace, matching `masked_value_predicate_exprs`' own
`extra_columns` convention (a value condition scoped by, say, a crosswalk zone id). The
rewritten value gets the same namespace as the mask that gates it: a carried column good
enough to decide WHETHER a pixel is rewritten is good enough to appear in WHAT it is
rewritten to (`value * zone_factor`), and a namespace admitting a column on one side of
`pl.when(...).then(...)` but not the other would be an arbitrary line.

**`extra_columns` ARE part of the predicate language.** `request.expressions` classifies a
conjunct by NAME, so a carried column has to be named for it to classify as anything but a
dimension. `filter_band_wide` names them: it unions its own `extra_columns` into the
`value_columns` it hands `classify_where_kinds`, so `__zone__ = 7` is the per-pixel mask it
is rather than a band selection evaluated against a `metadata_df` that has no such column.
That union is only for the classification -- `value_columns`' own count still decides
whether a paired frame has a pairing to collapse, and a carried column is not an operand's
measurement.

Whether a name is a per-band dimension or a per-pixel carried column is knowable only where
the carried column is introduced, which is why the band-wide side could not decide it alone:
`request.ops.FrameType.extra_columns` is where it is decided, tracked forward by the
shape-typing walk from the two places a carried column appears (a raster origin's crosswalk
zone id, a vector-to-raster `project`'s domain column). `request.ops._type_filter`
classifies against the identical union, so a `filter`'s TYPED output shape and the lowering
that actually runs here always agree.

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
module builds band-wide lowerings, not the walk itself, so there is no "outside the walk"
boundary for this function to have already crossed. `execute` owns that call and makes it
as the walk returns, releasing the frame through `released_band_metadata`.

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

from collections.abc import Callable, Sequence

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
    ExpressionError,
    PredicateKind,
    classify_where_kinds,
    combine_conjuncts,
    rewrite_value_column,
    rewrite_value_columns,
    value_predicate_mask,
)
from cpra.mp.data.analysis.request.ops import CaseOp, GroupOp

__all__ = [
    "JOIN_OPERAND_COLUMN",
    "PAIRING_KEY_EXCLUDED_COLUMNS",
    "case_band_wide",
    "filter_band_wide",
    "group_band_wide",
    "join_band_wide",
    "mask_value_band_wide",
    "sample_grid_cells_band_wide",
    "select_dimension_band_wide",
    "unify_metadata_parts",
]

# The metadata column `join_band_wide` adds recording which operand (step name)
# contributed each surviving band -- see the module docstring. Distinct from both
# `BAND_NAME_RE` and `RESERVED_BAND_INDEX_COLUMN` (`_band`), so it can never collide
# with either positional-name shape `execution/frame.py`'s invariants police.
JOIN_OPERAND_COLUMN = "_join_operand"

# Metadata columns `_pairing_key_columns` never pairs on -- see its docstring.
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
    """The metadata columns `case_band_wide` pairs a joined frame's bands on: every column
    that is non-null in EVERY band's row, minus `PAIRING_KEY_EXCLUDED_COLUMNS`.

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


def _pair_band_groups(
    metadata_df: pl.DataFrame, value_columns: Sequence[str]
) -> tuple[pl.DataFrame, list[str], list[tuple[int, list[int], dict[str, str]]]]:
    """Reconstitute a PAIRED band-wide frame's pairs from its band metadata, returning
    `(grouped, key_columns, pairs)`:

    - `grouped` is `metadata_df` collapsed to one row per candidate pair, in band order;
    - `key_columns` is what it was grouped on (`_pairing_key_columns`, possibly empty);
    - `pairs` holds one `(row index into grouped, band positions, {operand -> band column
      name})` per SURVIVING pair.

    A group with anything other than exactly one band per operand is absent from `pairs`
    -- a pairing MISS (an operand has no counterpart at this dimension combination), the
    same non-event as a pixel whose crosswalk finds no vector cell, dropped rather than
    raised on.

    Split out from `case_band_wide` rather than inlined: the pairing is a fact about a
    joined frame's METADATA, answerable with no reference to what the reduce then does to
    the pixels. See `_paired_metadata` for the other half, the metadata the survivors leave
    behind.
    """
    names = band_column_names(metadata_df)
    operands = list(value_columns)
    key_columns = _pairing_key_columns(metadata_df)

    indexed = metadata_df.with_row_index("__pos")
    if key_columns:
        grouped = indexed.group_by(key_columns, maintain_order=True).agg(
            pl.col("__pos"), pl.col(JOIN_OPERAND_COLUMN).alias("__operands")
        )
    else:
        grouped = indexed.select(pl.col("__pos").implode(), pl.col(JOIN_OPERAND_COLUMN).implode().alias("__operands"))

    pairs: list[tuple[int, list[int], dict[str, str]]] = []
    for row_index, (positions, row_operands) in enumerate(
        zip(grouped.get_column("__pos").to_list(), grouped.get_column("__operands").to_list())
    ):
        by_operand = dict(zip(row_operands, positions))
        if len(by_operand) != len(positions) or set(by_operand) != set(operands):
            continue
        substitutions = {operand: names[position] for operand, position in by_operand.items()}
        pairs.append((row_index, list(positions), substitutions))
    return grouped, key_columns, pairs


def _paired_metadata(grouped: pl.DataFrame, kept_rows: list[int], coalesced: pl.DataFrame) -> pl.DataFrame:
    """The metadata a paired reduce's surviving pairs leave behind: the COALESCED pair rows
    (`_coalesced_pair_metadata`), one per surviving pair, minus any column no surviving pair
    carried a value for.

    The pairing KEY columns are the dimensions the pair was formed on; they are what this
    used to return alone. The rest are the one-sided ones -- a dimension only one operand
    carried, which pairing could not key on because the other operand has no value for it.
    Dropping those made band-wide disagree with long over the same pipeline: `long_ops.
    join_long` keys on the columns both operands share, so a column only one of them carries
    rides along unchanged, and the joined frame reports it. `lnd_type_plus` is the shipped
    proof in both directions -- its `aggregate_type=mean` is in the CSV a `grid_cell`
    request returns (that request lowers to long) and was absent from the GeoTIFF the same
    selection downloads. `lnd_chng` is the one a reader notices: its baseline branch groups
    `calendar_year` away so one baseline band can broadcast, which makes the comparison
    year one-sided, so the download said which model group and scenario it compared and
    not which year.

    Coalescing is what makes the value well-defined, and `_coalesced_pair_metadata` already
    computes it for the arms' own dimension halves: a key column is identical across the
    pair by construction, a one-sided column has exactly one real value in it, and the
    genuinely ambiguous case (three-plus operands, two carrying the same non-key column with
    different values) raises there rather than being picked between here. This reuses that
    frame rather than recomputing it, so the metadata a pair LEAVES and the metadata its
    arms were EVALUATED against cannot drift apart.

    An all-null column is dropped rather than carried: with no surviving pair holding a
    value, it says nothing about the output, and `sink._band_descriptions` would omit it
    from every band anyway. `add_reserved_band_index` covers what is left when that empties
    the frame -- operands sharing no dimension and carrying no one-sided one either -- for
    the same "zero columns loses the row count" reason `group_band_wide`'s `by == []` branch
    does. That fallback is built from `grouped` rather than from `coalesced`: a coalesced
    frame with no columns to coalesce has no height left to read either (Polars collapses a
    zero-column frame outright, which is the very hazard the reserved index exists for), so
    the row count has to come from a frame that still has columns.
    """
    carried = [name for name in coalesced.columns if coalesced.get_column(name).null_count() < coalesced.height]
    return (
        coalesced.select(carried)
        if carried
        else add_reserved_band_index(grouped[kept_rows]).select(RESERVED_BAND_INDEX_COLUMN)
    )


# The row index `select_dimension_band_wide` carries through its join reconstruction --
# one per operand, suffixed, so a joined frame can name them apart. Underscore-prefixed
# and so not a legal band name, which keeps it clear of `BAND_NAME_RE`'s positional shape.
_BAND_POSITION_COLUMN = "__band_position"


def _joined_metadata_parts(metadata_df: pl.DataFrame) -> list[pl.DataFrame]:
    """A joined `metadata_df` split back into one frame per operand, in the order
    `join_band_wide` laid the operands out, each carrying that operand's own band positions
    plus only the dimension columns it actually has values for.

    "Actually has values for" is the same non-null test `_pairing_key_columns` uses, and
    for the same reason: `join_band_wide` merges its operands' metadata with
    `how="diagonal_relaxed"`, so a column only one operand carries is present but NULL on
    the other side, and that null is the only record left of whose column it was.

    `PAIRING_KEY_EXCLUDED_COLUMNS` is dropped here as it is there.
    `variable`/`time_unit`/`grid`/`path`/`band` are non-null for every operand and differ
    BETWEEN them, so keeping them would make them join keys that match nothing: they are
    per-band identity, not a dimension two operands can be compared along.
    """
    indexed = metadata_df.with_row_index(_BAND_POSITION_COLUMN)
    operands = metadata_df.get_column(JOIN_OPERAND_COLUMN).unique(maintain_order=True).to_list()
    parts = []
    for position, operand in enumerate(operands):
        rows = indexed.filter(pl.col(JOIN_OPERAND_COLUMN) == operand)
        carried = [
            name
            for name, nulls in zip(rows.columns, rows.null_count().row(0))
            if nulls == 0 and name not in PAIRING_KEY_EXCLUDED_COLUMNS and name != _BAND_POSITION_COLUMN
        ]
        parts.append(
            rows.select([pl.col(_BAND_POSITION_COLUMN).alias(f"{_BAND_POSITION_COLUMN}_{position}"), *carried])
        )
    return parts


def unify_metadata_parts(parts: list[pl.DataFrame]) -> pl.DataFrame:
    """One metadata frame per operand, recombined into the frame the equivalent LONG
    pipeline would have filtered: one row per cross-operand combination, joined on the
    dimensions the operands share.

    This mirrors `long_ops.join_long`, which is the point -- metadata is the long frame
    minus `value` and `grid_cell`, so the combination that reproduces long semantics is
    the one long itself performs. The columns two parts have in common are exactly the
    shared dimensions (each part's identity column is uniquely suffixed by its caller, and
    a non-shared dimension by definition appears in one part only), so the join key needs
    no separate computation.

    Public, and deliberately not specific to this module's own `_joined_metadata_parts`:
    `execution.plan.compile_plan` builds the same frame at COMPILE time out of each
    origin's resolved band metadata, to decide which bands a dimension filter leaves worth
    reading at all. The two must agree exactly -- plan-time narrowing is sound only because
    the filter step re-executes over the narrowed read and reaches the same answer -- so
    they share one function rather than one description of a function.

    `how="full"` with coalesced keys, not `how="inner"`: an operand band with no
    counterpart keeps a row here, with the other operand's columns null. Inner would drop
    it, which would make a trivially-true `where` narrow the frame -- the filter quietly
    finishing a job `join_band_wide` did not do. A predicate that actually tests the
    missing operand's column still drops that row, since null fails `fill_null(False)`,
    which is the narrowing long would have performed at the join. (That band-wide `join`
    keeps unpaired bands where long's inner join does not is a property of
    `join_band_wide`, not of this filter.)

    No shared dimension at all means a cross join, which is the right degenerate answer:
    with nothing to pair on, every combination is a combination.
    """
    unified = parts[0]
    for right in parts[1:]:
        on = [name for name in unified.columns if name in right.columns]
        unified = unified.join(right, on=on, how="full", coalesce=True) if on else unified.join(right, how="cross")
    return unified


def _joined_dimension_positions(metadata_df: pl.DataFrame, dimension_where: str) -> list[int]:
    """The band positions of a JOINED frame that survive `dimension_where`: the predicate
    evaluated once against the unified frame, then read back per operand.

    A band survives if it appears in any surviving combination. That is what makes this
    agree with long: a conjunct on a dimension only one operand carries narrows THAT
    operand directly and every other operand THROUGH the join, rather than -- as filtering
    the `diagonal_relaxed` metadata directly does -- evaluating the conjunct against rows
    where the column is null for no reason other than that they came from a different
    operand, and discarding all of them.
    """
    referenced = set(pl.sql_expr(dimension_where).meta.root_names())
    identity = sorted(referenced & PAIRING_KEY_EXCLUDED_COLUMNS & set(metadata_df.columns))
    if identity:
        raise ExpressionError(
            f"where={dimension_where!r} references per-band identity column(s) {identity} over a "
            "JOINED band-wide frame -- these name which dataset a band came from and differ "
            "between operands, so they select no operand-independent set of bands; filter each "
            "input before the join instead"
        )

    parts = _joined_metadata_parts(metadata_df)
    unified = unify_metadata_parts(parts)
    survivors = unified.filter(pl.sql_expr(dimension_where).fill_null(False))
    kept: set[int] = set()
    for position in range(len(parts)):
        column = survivors.get_column(f"{_BAND_POSITION_COLUMN}_{position}")
        kept.update(value for value in column.to_list() if value is not None)
    return sorted(kept)


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

    Over a JOINED frame -- one whose metadata carries `JOIN_OPERAND_COLUMN` -- the
    predicate is not evaluated against the metadata as it stands. `join_band_wide` stacks
    its operands' metadata VERTICALLY (`how="diagonal_relaxed"`), one row per band, so a
    dimension only one operand carries is null on every other operand's rows; evaluating
    the conjunct there would fail `fill_null(False)` for reasons having nothing to do with
    the predicate and discard those operands entirely. Instead the join is reconstructed
    first (`_joined_metadata_parts`, `unify_metadata_parts`) and the predicate evaluated
    against THAT, so a conjunct on a non-shared dimension narrows the operand carrying it
    directly and the others through the join -- which is what the same pipeline means in
    long shape, where the join is a real key-based join and the dimension is an ordinary
    column of the joined frame. A `filter` must not mean two different things depending on
    a lowering the request never asked for.

    An unjoined frame takes the direct path unchanged: with one operand there is nothing
    to reconstruct and no null to misread.

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
    if JOIN_OPERAND_COLUMN in metadata_df.columns:
        positions = _joined_dimension_positions(metadata_df, dimension_where)
    else:
        indexed = metadata_df.with_row_index(_BAND_POSITION_COLUMN)
        kept = indexed.filter(pl.sql_expr(dimension_where).fill_null(False))
        positions = kept.get_column(_BAND_POSITION_COLUMN).to_list()
    if not positions:
        return empty_band_wide(window)
    kept_metadata = metadata_df[positions]
    new_names = band_column_names(kept_metadata)
    select_exprs = [pl.col(names[position]).alias(new_name) for position, new_name in zip(positions, new_names)]
    out_lf = lf.select([*select_exprs, *extra_columns])
    return out_lf, kept_metadata, window


def _sample_grid_cells_joined(
    band_wide: BandWideFrame,
    where: str,
    extra_columns: Sequence[str],
    *,
    grid_width: int,
    value_columns: Sequence[str],
) -> LongFrame:
    """`sample_grid_cells_band_wide` over a JOINED frame -- one whose metadata carries
    `JOIN_OPERAND_COLUMN`, and whose typed shape therefore has one operand-named value
    column per operand rather than a single `value`.

    The plain unpivot cannot serve this case. It puts every band's measurement into one
    `value` column and distinguishes them by the metadata each band's row carries -- which
    on a joined frame means the result is discriminated by `JOIN_OPERAND_COLUMN`, an
    internal tag, and carries no column named after any operand at all. `request.ops.
    _type_filter` types a spatial filter as shape-changing but `value_columns`-preserving
    (a filter RESTRICTS; it never changes how many operands a frame carries), so the next
    step goes looking for `marsh`/`sal` and finds `value`.

    So each operand is unpivoted SEPARATELY -- its own bands, its own metadata rows -- and
    the results handed to `long_ops.join_long`, the long-shape join itself. That is the
    strongest available statement of what this transition means: a band-wide `join`
    followed by a spatial filter produces exactly the frame the equivalent long pipeline
    would have, because the operand combination is performed by literally the same
    function. The rename to operand names, the shared-dimension key, and the dropping of
    `variable`/`time_unit`/`grid` all come from there rather than being restated here.

    **Inner, so a band with no counterpart is dropped.** That is `join_long`'s rule, and
    it is the right one here even though `unify_metadata_parts` deliberately takes the
    opposite one: that function serves the filter that stays BAND-WIDE, where an unpaired
    band is still a representable band and dropping it would have the filter quietly
    finish a job `join_band_wide` did not do. The paired LONG shape has no such cell -- a
    row carries one value per operand -- so the unpaired band has nowhere to go, and
    `case_band_wide`'s pairing (`_pair_band_groups`) already drops the same misses.

    **An operand left with no bands at all** (a preceding dimension filter can empty one;
    see `select_dimension_band_wide`) still gets a frame, an empty one, rather than being
    skipped. Skipping it would drop its column from the result and reintroduce the very
    missing-column failure above, on the path where the answer is simply "no rows".

    **The spatial conjuncts run once on the BAND-WIDE frame, before any operand is
    unpivoted, and the result is materialized there.** They are the one kind that can: a
    spatial conjunct names `grid_cell` alone, and `row_major_grid_cell` has already made
    that an ordinary column of the band-wide frame -- one whose value is a property of the
    PIXEL, identical across every band and therefore across every operand -- so dropping
    the rows it rejects while the pixels are still rows commutes with everything below.
    Both halves of that matter, and for different reasons:

    * *Before the unpivots*, because the operands do not share one unpivot here the way
      the plain path's bands do. Each is its own branch off `indexed`, the join stands
      between them and the filter, and joining three whole-raster frames on `grid_cell` to
      then keep two rows is minutes of work for an answer that is a handful of pixels.
    * *Materialized*, because those branches are the reason a lazy filter alone did not
      help: three unpivots reading one `LazyFrame` make three separate scans of the shared
      upstream work (the multi-raster read and, for `lnd_type_plus`, a vector-to-raster
      `project`'s rasterization), so the plan's expensive part ran once per operand.
      `.collect()` after the spatial filter costs a frame of a few rows and buys the read
      being paid for exactly once.

    The dimension and value conjuncts stay after the join, and `where` in its entirety is
    still applied there: a dimension conjunct may name a column only one operand carries
    (which is why `_joined_metadata_parts` keeps them apart in the first place), and a
    value conjunct names operand columns that do not exist until the join has made them.
    """
    lf, metadata_df, window = band_wide
    spatial = classify_where_kinds(where, value_columns)[PredicateKind.spatial]
    spatial_where = combine_conjuncts(spatial) if spatial else None
    names = band_column_names(metadata_df)
    present = metadata_df.get_column(JOIN_OPERAND_COLUMN).unique(maintain_order=True).to_list()
    operands = [*present, *sorted(set(value_columns) - set(present))]
    parts = dict(zip(present, _joined_metadata_parts(metadata_df)))
    schema = lf.collect_schema()
    value_dtype = schema[names[0]] if names else pl.Float64

    indexed = row_major_grid_cell(lf, window, grid_width=grid_width)
    if spatial_where is not None:
        indexed = indexed.filter(pl.sql_expr(spatial_where)).collect().lazy()
    frames: list[LongFrame] = []
    for position, operand in enumerate(operands):
        if operand not in parts:
            frames.append(pl.LazyFrame(schema={"grid_cell": pl.Int64, "value": value_dtype}))
            continue
        part = parts[operand]
        position_column = f"{_BAND_POSITION_COLUMN}_{present.index(operand)}"
        band_names = [names[band] for band in part.get_column(position_column).to_list()]
        # `extra_columns` ride on the first operand alone, so `join_long` carries them
        # along rather than keying on them -- the same rule `join_band_wide` applies when
        # it takes its carried columns from operand 0.
        carried = list(extra_columns) if position == 0 else []
        long_part = indexed.unpivot(
            index=["grid_cell", *carried], on=band_names, variable_name="__band", value_name="value"
        )
        identity = pl.DataFrame({"__band": pl.Series(band_names, dtype=pl.String)}).hstack(part.drop(position_column))
        frames.append(long_part.join(identity.lazy(), on="__band", how="left").drop("__band"))

    joined = long_ops.join_long(frames, operands)
    return long_ops.filter_long(joined, where, operands)


def sample_grid_cells_band_wide(
    band_wide: BandWideFrame,
    where: str,
    extra_columns: Sequence[str],
    *,
    grid_width: int,
    value_columns: Sequence[str] = ("value",),
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

    `value_columns` is the incoming frame's value columns, as `filter_band_wide` receives
    them; a JOINED frame (several operand-named ones) takes an entirely different route
    through `_sample_grid_cells_joined` -- see there for why one `value` column cannot
    carry a paired frame across this transition.
    """
    validate_extra_columns(extra_columns)
    lf, metadata_df, window = band_wide
    if JOIN_OPERAND_COLUMN in metadata_df.columns:
        return _sample_grid_cells_joined(
            band_wide, where, extra_columns, grid_width=grid_width, value_columns=value_columns
        )
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
    value_columns`), and is forwarded to `classify_where_kinds` so that a PAIRED frame's
    operand-named conjuncts (`marsh > 0.5`) classify as VALUE rather than falling through
    to dimension for not being spelled `value`. It defaults to the unpaired `("value",)`,
    which is every frame that never went through a `join`. A value conjunct over a frame
    that really is paired never reaches here: `request.ops._type_filter` rejects it, since
    evaluating it would mean pairing the bands and reducing each pair to one, and nothing
    in a predicate says which operand's value that one band should carry. `case_band_wide`
    is the lowering for that question, and its arms answer it.

    `extra_columns` joins `value_columns` for the classification, and only for that: a
    carried column is a real per-pixel column of `lf` (one value per pixel, no band axis),
    so a `where` naming one is the per-pixel mask a value conjunct is -- `mask_value_band_
    wide` already admits these names into its own namespace -- and not a band selection
    evaluated against `metadata_df`, which carries no such column and would fail on it.
    `request.ops._type_filter` classifies against the identical union, from
    `FrameType.extra_columns`, so the shape this step was TYPED as and the lowering that
    runs here cannot disagree.

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
    kinds = classify_where_kinds(where, [*value_columns, *extra_columns])
    if kinds[PredicateKind.spatial]:
        if grid_width is None:
            raise ValueError(
                f"filter_band_wide needs grid_width to decode the spatial conjunct in {where!r} into absolute pixel ids"
            )
        return sample_grid_cells_band_wide(
            band_wide, where, extra_columns, grid_width=grid_width, value_columns=value_columns
        )

    result = band_wide
    if kinds[PredicateKind.dimension]:
        result = select_dimension_band_wide(result, combine_conjuncts(kinds[PredicateKind.dimension]), extra_columns)
        if has_no_bands(result[1]):
            return result
    if kinds[PredicateKind.value]:
        result = mask_value_band_wide(result, combine_conjuncts(kinds[PredicateKind.value]), extra_columns)
    return result


def _band_applicability(metadata_df: pl.DataFrame) -> Callable[[str], list[bool]]:
    """A memoized "which bands does this dimension predicate hold for" over `metadata_df`
    -- one `list[bool]`, in band order, per distinct predicate TEXT.

    Memoized because `case_band_wide` asks the same question once per (arm, band) while
    the answer only ever varies by arm: evaluating each arm's dimension half once across
    the whole metadata frame, then indexing by band, is the same work `calculate`'s
    band-wide lowering used to do for its single `where`, generalized to n arms.

    `fill_null(False)` is what makes a band that simply does not CARRY the dimension (a
    joined frame merges its operands' metadata with `how="diagonal_relaxed"`, so one
    operand's column is null on the other's rows) fail the arm rather than match it.
    """
    cache: dict[str, list[bool]] = {}

    def applicable(text: str) -> list[bool]:
        if text not in cache:
            cache[text] = (
                metadata_df.select(pl.sql_expr(text).fill_null(False).alias("__ok")).get_column("__ok").to_list()
            )
        return cache[text]

    return applicable


def _coalesced_pair_metadata(metadata_df: pl.DataFrame, pairs: Sequence[tuple[int, list[int], dict]]) -> pl.DataFrame:
    """One metadata row per surviving pair, built by coalescing the pair's own operand rows
    column by column -- what an arm's dimension half is evaluated against on a PAIRED
    band-wide frame.

    A pair's operands do not share a metadata row, so "does `calendar_year = 2050` hold for
    this pair" has to be asked of the pair rather than of a band. Coalescing answers it
    without needing a combining rule for the common cases:

    - a PAIRING KEY column (`_pairing_key_columns`) is the same in every operand's row by
      construction -- the pair was formed by grouping on it -- so there is one value and
      nothing to choose between;
    - a non-key column is, by that function's own definition, null for at least one
      operand, which with two operands leaves exactly one real value.

    That leaves one genuinely ambiguous case: three or more operands, two of which both
    carry the same non-key column with DIFFERENT values. There is no answer to give there
    -- an arm naming that column means two things at once -- so it raises rather than
    picking. `PAIRING_KEY_EXCLUDED_COLUMNS` is dropped first: those columns (`variable`,
    `path`, the operand tag) differ per operand by design, so including them would make
    every pair a conflict.
    """
    columns = [name for name in metadata_df.columns if name not in PAIRING_KEY_EXCLUDED_COLUMNS]
    rows = []
    for _row_index, positions, substitutions in pairs:
        pair_rows = metadata_df[list(positions)].select(columns).rows(named=True)
        coalesced = {}
        for column in columns:
            values = {row[column] for row in pair_rows if row[column] is not None}
            if len(values) > 1:
                raise ExpressionError(
                    f"metadata column {column!r} has conflicting values {sorted(map(str, values))} "
                    f"across the operands {sorted(substitutions)} of one pair -- a case arm "
                    "naming it would mean two things at once; filter the operand branch that "
                    "should not carry it before the join"
                )
            coalesced[column] = next(iter(values)) if values else None
        rows.append(coalesced)
    # The original schema, not one inferred from `rows`: a column every pair coalesced to
    # null would otherwise come back as Polars' Null dtype, and a comparison against it is
    # a type error rather than the `False` `_band_applicability` means it to be.
    return pl.DataFrame(rows, schema={name: metadata_df.schema[name] for name in columns})


def _select_band_exprs(
    lf: pl.LazyFrame,
    exprs: Sequence[pl.Expr],
    names: Sequence[str],
    extra_columns: Sequence[str],
) -> pl.LazyFrame:
    """`lf.select(exprs + extra_columns)`, except via `with_columns` so that a band
    expression which happens to be a SCALAR still comes back one value per pixel.

    A `case` whose arms are all purely dimensional and whose `then`/`else` are constants
    compiles to an expression with no column reference in it at all -- `pl.when(lit(True)).
    then(lit(6))`, or just `pl.lit(None)` for a band every arm dropped out of with no
    `else`. `select` gives such an expression its literal length of ONE, and a select whose
    every column is unit-length yields a one-row frame: the window's row count, which the
    band-wide contract says this lowering preserves, silently gone. `with_columns`
    broadcasts a unit-length expression to the frame's height instead, and because it
    evaluates every expression against the INCOMING frame, a paired reduce writing `b0`
    from `b0` and `b2` still reads the originals.

    `names` is what to keep afterwards, in order -- for the paired reduce that is a PREFIX
    of the incoming band columns (N pairs out of 2N bands), so the trailing ones have to be
    dropped by the select rather than left standing.
    """
    return lf.with_columns(exprs).select([*names, *extra_columns])


def case_band_wide(
    band_wide: BandWideFrame,
    op: CaseOp,
    extra_columns: Sequence[str],
    value_columns: Sequence[str] = ("value",),
) -> BandWideFrame:
    """The band-wide lowering of a `case`, in the two readings `value_columns` picks
    between -- the same split `filter_band_wide` makes for a value predicate, and for the
    same reason.

    **Unpaired (one value column): a MAP.** One chained `pl.when(...)` per band, with the
    placeholder `value` rewritten to that band's own column (`rewrite_value_column`). Band
    count, band order and `metadata_df` are all unchanged -- this rewrites pixels, it never
    selects bands.

    **Paired (a `join` left one value column per operand): a REDUCE.** Each pair of bands
    becomes ONE band, with every operand name rewritten to its own band column within that
    pair (`rewrite_value_columns`). 2N bands in, N out, `metadata_df` collapsed to the
    pairing key. This is the ONLY way a band-wide pairing collapses without also collapsing
    the bands, and the reason it is `case` that does it rather than `filter`: a reduce has
    to pick which operand's value the one surviving band carries, and only an arm's `then`
    says.

    **An arm's dimension half selects which bands the ARM applies to.** Band-wide shape
    keeps its dimension values in `metadata_df`, one row per band, so a conjunct naming one
    has a per-band answer rather than a per-pixel one: `_band_applicability` evaluates each
    arm's dimension half once across the whole metadata frame, and the band-by-band result
    decides which arms enter that band's chain. A band an arm does not apply to still gets
    every OTHER arm and the `else` -- the arm drops out, the band does not. A band every arm
    drops out of is the `else` alone, which for a migrated `calculate` (`else: "value"`) is
    the identity, exactly the pass-through that lowering used to get by skipping the band.

    On the paired side the same question is asked of the PAIR, against
    `_coalesced_pair_metadata` -- the operands of one pair do not share a metadata row, so
    there is no single row to evaluate against until one is built.

    Neither reading adds a nodata guard of its own. See `case_expression` for why a `case`
    leaves nodata to the arms.

    A `case` over zero bands is `band_wide` unchanged in either reading -- there is nothing
    to rewrite and nothing to pair, and "nothing happens to zero bands" needs no expression
    compiled to say it. (A frame whose pairs all MISS is a different matter, and answers
    `empty_band_wide(window)`: there the pairing ran and left nothing.)
    """
    validate_extra_columns(extra_columns)
    lf, metadata_df, window = band_wide
    if has_no_bands(metadata_df):
        return band_wide
    names = band_column_names(metadata_df)

    if len(value_columns) > 1:
        grouped, _key_columns, pairs = _pair_band_groups(metadata_df, value_columns)
        if not pairs:
            return empty_band_wide(window)
        coalesced = _coalesced_pair_metadata(metadata_df, pairs)
        applicable = _band_applicability(coalesced)
        kept_rows = []
        band_exprs = []
        for pair_index, (row_index, _positions, substitutions) in enumerate(pairs):
            columns = list(substitutions.values())
            band_exprs.append(
                long_ops.case_expression(
                    op,
                    [*columns, *extra_columns],
                    lambda text, subs=substitutions: rewrite_value_columns(text, subs),
                    predicate_columns=[*substitutions, *extra_columns],
                    resolve_dimension=lambda text, i=pair_index: applicable(text)[i],
                ).alias(f"b{len(kept_rows)}")
            )
            kept_rows.append(row_index)
        return (
            _select_band_exprs(lf, band_exprs, [f"b{i}" for i in range(len(kept_rows))], extra_columns),
            _paired_metadata(grouped, kept_rows, coalesced),
            window,
        )

    applicable = _band_applicability(metadata_df)
    exprs = [
        long_ops.case_expression(
            op,
            [name, *extra_columns],
            lambda text, column=name: rewrite_value_column(text, column),
            predicate_columns=["value", *extra_columns],
            resolve_dimension=lambda text, i=position: applicable(text)[i],
        ).alias(name)
        for position, name in enumerate(names)
    ]
    return _select_band_exprs(lf, exprs, names, extra_columns), metadata_df, window


def _present_group_columns(metadata_df: pl.DataFrame, by: Sequence[str]) -> list[str]:
    """`by` narrowed to the metadata columns this frame actually carries.

    A band-wide `group`'s `by` names METADATA columns, one row per band. A pipeline's `by`
    is written in terms of DIMENSIONS the caller steers, and in long shape those coincide --
    every steerable dimension is a column of the frame. Band-wide they can come apart, since
    a band-wide frame's metadata carries only what its bands have values for.

    This is a BACKSTOP, not the fix for the one case that made it necessary. `lnd_chng,diff`
    over a raster grid used to arrive here with `calendar_year` in `by` and no such column:
    `lnd_chng`'s baseline branch groups the year away so one baseline band can broadcast,
    which made the year one-sided, and `_paired_metadata` used to keep the pairing key
    alone. It carries a one-sided column now -- see that function -- so the year is present
    and this narrowing no longer bites there. What is left for it is a `by` naming a column
    NO band has a value for, where there is genuinely nothing to group on.

    Dropping rather than raising is sound rather than lenient: a band-wide frame reaching a
    `group` holds one row per band, and a column absent from that frame took no more than
    one value across every band in it -- grouping on a constant cannot split anything, and
    grouping on a dimension no band carries cannot either.

    `op.of` keeps its hard error above for the opposite reason: `of` READS a band's value
    for a column, so an absent one has nothing to read and no harmless reading.
    """
    return [name for name in by if name in metadata_df.columns]


def group_band_wide(band_wide: BandWideFrame, op: GroupOp, extra_columns: Sequence[str]) -> BandWideFrame:
    """The band-wide lowering of a `group`: a HORIZONTAL reduce across each output group's
    contributing bands, at every pixel -- the within-row axis, unrelated to
    `execution.long_ops.group_long`'s across-row collapse. `execution/fold.py`'s
    cliff-shaped-collapse fold rule has no counterpart here: a horizontal reduce reads one
    row at a time, so there is no group whose accumulator could outgrow memory.

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
    by = _present_group_columns(metadata_df, op.by)
    indexed = metadata_df.with_row_index("__pos")
    groups_df = indexed.group_by(by, maintain_order=True).agg(pl.col("__pos"))

    if by:
        metadata_out = groups_df.select(by)
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
        select_cols = [pl.col(local_name).alias(f"b{next_position + i}") for i, local_name in enumerate(local_names)]
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
            metadata_df.with_columns(pl.lit(name).alias(JOIN_OPERAND_COLUMN)) if metadata_df.height else metadata_df
        )

    combined = pl.concat(aligned, how="horizontal")
    out_lf = combined.select([*(f"b{i}" for i in range(next_position)), *extra_columns])
    merged_metadata = pl.concat(renamed_metadatas, how="diagonal_relaxed")
    return out_lf, merged_metadata, window
