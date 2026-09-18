"""The two data shapes every op algebra lowering runs against, and the invariants that
make a band-wide frame composable across a chain of ops.

**Long** is the vector row model: dimension columns, plus `grid_cell`, plus `value`, one
row per observation. It needs no type of its own -- it is an ordinary `pl.LazyFrame`, and
`LongFrame` below is nothing but a name for that fact, so a lowering's signature can say
what shape it expects without a reader having to already know the convention.

**Band-wide** is the raster row model: one row per pixel, one column per band, carried as
the triple `BandWideFrame = (band_wide_lf, metadata_df, window)`:

- `band_wide_lf` -- the pixel data. Row *i* is pixel *i* of the read, in every band
  column, in the order the read produced them -- which is row-major within `window` only
  when the read came back in ONE batch. A read big enough to be chunked emits each batch's
  own sub-rectangle row-major, batch after batch, so its rows are a permutation of the
  row-major order and not that order itself. `window.batch_layout` is what names the
  permutation; `row_major_grid_cell` below is what converts a row index back into a
  row-major cell id, and is what any lowering that needs a pixel's POSITION (rather than
  merely its identity with the same pixel in another frame from the same read) has to go
  through.
- `metadata_df` -- the COLUMN dictionary: one row per band column, holding that band's
  dimension values (and, after a `group`, its group identity). `band_column_names` below
  is what ties a metadata row to the pixel column it describes; it is the only thing that
  makes band columns self-describing, since the column names themselves (`b0`, `b1`, ...)
  carry no information beyond position.
- `window` -- the FRAME dictionary: the source raster's identity, the integer pixel
  offset and size of the read, its affine transform, its CRS, and the decimation level if
  any. One `ReadWindow` per frame, never a column -- it is what reconstructs a pixel's
  position from its row index now that no positional column exists, and it is what a
  band-wide `join`'s precondition and a raster sink both check.

**No op may retain a row index across an op boundary.** `band_column_names` is
recomputed from whatever `metadata_df` an op was actually handed, never carried forward
from an earlier call -- a stored index goes stale the moment another op rewrites the
metadata, and the failure mode is a wrong answer, not an error. This module cannot make
that true by construction (nothing stops a future lowering from closing over an index
anyway); it is a convention every lowering has to keep, not a type this module enforces.
What IS enforced here, by assertion, are the two invariants cheap enough to check on
every call: that a band-wide lowering's output has the same row count as its input
(barring the explicit exception below), and that it returns the exact `window` it was
given, unchanged.

**Two degenerate shapes.** Polars cannot represent "zero columns, N rows" -- an
`.select()` (or `.drop()`) that leaves no columns behind reports height 0 regardless of
how many rows went in, silently discarding the row count along with the last column. Two
places in the op algebra can produce exactly zero columns and have to name the case
instead of falling into that hole by accident:

- A dimension filter whose predicate keeps no band leaves zero band columns. `band_wide`
  has nothing left to describe, so the answer is not "zero columns, however many rows
  `window` implies" -- it is `empty_band_wide`, a frame with nothing in it at all.
- A `group` with `by=[]` collapses every dimension column out of the *metadata* frame
  (there is no single value left to report for a dimension the group merged across), and
  a metadata frame with zero columns has the identical Polars problem: it cannot say how
  many bands survived. `RESERVED_BAND_INDEX_COLUMN` (`add_reserved_band_index`) is a
  column reserved for exactly this: it holds each surviving band's position, so the
  metadata frame's height keeps meaning something even when no dimension column does.
  It is the same kind of stored positional index the no-retained-row-index rule above
  forbids everywhere else, tolerated here because nothing else can carry the count.

  `released_band_metadata` removes it on the way out, and the rule it keeps is narrower
  than "it never escapes" -- that rule cannot be kept, because a metadata frame whose only
  column is this one loses its height the moment the column is dropped, which is the exact
  loss the column was added to prevent. What holds instead: the reserved column never
  reaches anything that would read it as a DIMENSION. `drop_reserved_band_index` is the
  unconditional drop, for a caller that wants the identity columns and carries its own row
  count some other way; see both functions.

Everything in this module is pure Polars: no file is opened, no raster is read, no
request is parsed. Three names it re-exports are defined elsewhere and imported rather
than redefined. `BAND_NAME_RE` and `RESERVED_BAND_INDEX_COLUMN` come from
`request/pipeline.py`, which checks a user's own names against the same two shapes --
recomputing that pattern here as a second regex is exactly the drift a single source of
truth is supposed to prevent. `ReadWindow` comes from `io/window.py`, because it
describes a read rather than an algebra: the scan constructs one and must be able to
return a complete band-wide triple without importing this module, which sits a layer
above it.
"""

import itertools
from collections.abc import Iterable

import polars as pl

from cpra.mp.data.analysis.io.window import ReadWindow
from cpra.mp.data.analysis.request.pipeline import BAND_NAME_RE, RESERVED_BAND_INDEX_COLUMN

# `BAND_NAME_RE`/`RESERVED_BAND_INDEX_COLUMN` come from the request layer (the name-shape
# rules a user's own names are checked against) and `ReadWindow` from the io layer (it
# describes a read, and the scan that performs one must be able to construct it without
# importing the algebra). All three are re-exported here so a lowering can import the
# whole band-wide contract from one place rather than tracking which layer owns each half.
__all__ = [
    "BAND_NAME_RE",
    "RESERVED_BAND_INDEX_COLUMN",
    "ReadWindow",
    "LongFrame",
    "BandWideFrame",
    "band_column_names",
    "validate_extra_columns",
    "has_no_bands",
    "empty_band_wide",
    "has_reserved_band_index",
    "add_reserved_band_index",
    "drop_reserved_band_index",
    "released_band_metadata",
    "assert_window_unchanged",
    "assert_same_pixel_grid",
    "assert_height_preserved",
    "row_major_grid_cell",
]


# The long shape needs no type of its own -- an ordinary `pl.LazyFrame`. This alias
# exists purely so a lowering's signature can say `LongFrame` and mean something, the
# same reason `BandWideFrame` exists below.
LongFrame = pl.LazyFrame

# The band-wide triple, spelled once as a type so every lowering's signature says the
# same three things in the same order instead of each writing out the tuple by hand.
BandWideFrame = tuple[pl.LazyFrame, pl.DataFrame, ReadWindow]


def band_column_names(metadata_df: pl.DataFrame) -> list[str]:
    """The band-wide frame's band column names for the metadata frame that currently
    describes it: `b0..b{metadata_df.height - 1}`.

    This one-line function IS the band-naming invariant: column `b<i>` is the band
    described by metadata row `i`. It is always recomputed from whatever `metadata_df`
    is in hand, never stored -- an op that changes the band set returns a metadata frame
    describing the bands it left behind, so recomputing this against that frame is
    already correct for whatever runs next. A cached or carried-forward list is exactly
    the stale-index failure mode this module's docstring warns about.
    """
    return [f"b{i}" for i in range(metadata_df.height)]


def validate_extra_columns(extra_columns: Iterable[str]) -> None:
    """Reject an `extra_columns` name that could be mistaken for a positional one.

    Every band-wide lowering takes `extra_columns` explicitly, naming the non-band
    columns (a crosswalk column attached by the scan, notably) it should carry through a
    `.select()` alongside the band columns it rewrites -- see this module's docstring.
    A name matching the band pattern (`b0`, `b1`, ...) or equal to the reserved band-
    index column would be silently shadowed the moment `band_column_names` or
    `add_reserved_band_index` produces the same name, for the identical reason
    `request/pipeline.py` forbids both shapes to a user-chosen input or step name.
    """
    for name in extra_columns:
        if name == RESERVED_BAND_INDEX_COLUMN or BAND_NAME_RE.match(name):
            raise ValueError(
                f"extra_columns name {name!r} matches the positional band-name pattern "
                f"(b0, b1, ...) or the reserved {RESERVED_BAND_INDEX_COLUMN!r} index "
                "column, and would be shadowed by it"
            )


def has_no_bands(metadata_df: pl.DataFrame) -> bool:
    """True once `metadata_df` describes zero bands -- `band_column_names(metadata_df)
    == []`. Named so a dimension-filter lowering can branch on it explicitly instead of
    discovering the Polars zero-column collapse by surprise; see `empty_band_wide`, the
    canonical result for this case.
    """
    return metadata_df.height == 0


def empty_band_wide(window: ReadWindow) -> BandWideFrame:
    """The canonical empty band-wide result: a dimension filter kept no band.

    Zero band columns means `band_wide` has nothing left to describe -- not "the
    original pixel count, zero bands", which Polars cannot represent (`.select([])`
    reports height 0 regardless of the input's height; see this module's docstring) and
    which would be a fabricated number even if it could: a frame with no bands carries no
    per-pixel information for any downstream op to preserve. This is deliberately
    unconditional on whatever `extra_columns` the caller may otherwise have carried --
    once every band is gone there is nothing left for an auxiliary column to describe
    either, so this returns a wholly empty triple rather than a partial one. `window` is
    returned unchanged, same as every other band-wide lowering (invariant: see
    `assert_window_unchanged`).
    """
    return pl.LazyFrame(), pl.DataFrame(), window


def has_reserved_band_index(metadata_df: pl.DataFrame) -> bool:
    """Whether `metadata_df` currently carries the reserved band-index column -- see
    `add_reserved_band_index`."""
    return RESERVED_BAND_INDEX_COLUMN in metadata_df.columns


def add_reserved_band_index(metadata_df: pl.DataFrame) -> pl.DataFrame:
    """Prepend the reserved band-index column, one row per band in `metadata_df`'s
    current order (`0..metadata_df.height - 1`).

    A `group` with `by=[]` collapses every dimension column out of the metadata frame --
    there is no single value left to report for a dimension the group merged across --
    which leaves a metadata frame with zero columns and the same Polars collapse
    `empty_band_wide` exists for on the pixel side. This column is what keeps the
    metadata frame's height meaningful in that case: it does not describe a band's
    identity the way a dimension column does, only its position, so it must never reach
    anything outside the walk -- `drop_reserved_band_index` is the matching call every
    caller of this function must make before metadata is exposed.

    Raises `ValueError` if `metadata_df` already carries the column: two nested band
    indices would mean one of them is stale, which is exactly the failure this module's
    invariants exist to prevent.

    Must be called BEFORE whatever reduction would otherwise collapse `metadata_df`'s
    columns away, not after -- once a `.select()`/`.group_by()` has already discarded
    every column, the row count it carried is already gone, and no column added
    afterward can recover a height Polars no longer remembers.
    """
    if has_reserved_band_index(metadata_df):
        raise ValueError(
            f"metadata_df already carries the reserved {RESERVED_BAND_INDEX_COLUMN!r} "
            "column -- adding a second one would make its meaning ambiguous"
        )
    # `DataFrame.insert_column` mutates its receiver in place (and returns it, not a
    # copy) -- a surprising exception to Polars' otherwise-uniform "operations return a
    # new frame" convention that would silently corrupt a caller's own `metadata_df` if
    # used here. `hstack` of two freshly built frames does not share that hazard: it
    # mutates neither operand (confirmed directly, not assumed).
    index_df = pl.DataFrame({RESERVED_BAND_INDEX_COLUMN: pl.Series(range(metadata_df.height), dtype=pl.UInt32)})
    return index_df.hstack(metadata_df)


def drop_reserved_band_index(metadata_df: pl.DataFrame) -> pl.DataFrame:
    """Remove the reserved band-index column if present; a no-op otherwise.

    Idempotent, and safe to call unconditionally wherever what is wanted is the set of
    columns that describe a band's IDENTITY -- a caller turning metadata columns into a
    long frame's dimension columns, notably, where a position must not be mistaken for a
    dimension.

    **Not, on its own, the call that releases metadata from the walk.** When the reserved
    column is the only column left, dropping it takes the row count with it (Polars cannot
    represent "zero columns, N rows"; see this module's docstring), which is precisely the
    loss `add_reserved_band_index` exists to prevent -- so a caller that needs the height
    to survive wants `released_band_metadata` below, not this. A caller that only reads
    `.columns` off the result is unaffected either way, because a column list does not
    carry a row count to lose.
    """
    if has_reserved_band_index(metadata_df):
        return metadata_df.drop(RESERVED_BAND_INDEX_COLUMN)
    return metadata_df


def released_band_metadata(metadata_df: pl.DataFrame) -> pl.DataFrame:
    """`metadata_df` as it should leave the walk: the reserved band-index column dropped,
    unless dropping it would take the frame's row count along with it.

    The reserved column is a position, not an identity, and everywhere it can be confused
    for a dimension it has to go. But a `group` with `by=[]` leaves a metadata frame whose
    ONLY column is that one -- every dimension column collapsed away, nothing else left to
    say how many bands survived -- and dropping it there reports zero bands for a result
    that genuinely has one. That is not a cosmetic difference: `has_no_bands` is what a
    raster sink asks to tell "this request matched no data" from "this request produced a
    raster", so an unconditional drop turns a correct single-band answer into an empty one.

    So in exactly that case the column stays, as the sole carrier of "there is one band
    here", and the band-wide contract's promise is the narrower one it can actually keep:
    the reserved column never reaches anything that would read it as a DIMENSION. Nothing
    consuming a released band-wide frame does -- a raster sink reads the metadata's height
    and its band columns, never its column names as dimensions -- and the one consumer that
    does turn metadata columns into dimensions, a raster-to-vector `project`, carries its
    own row count separately and so calls `drop_reserved_band_index` unconditionally
    instead.
    """
    if metadata_df.columns == [RESERVED_BAND_INDEX_COLUMN]:
        return metadata_df
    return drop_reserved_band_index(metadata_df)


def assert_window_unchanged(handed: ReadWindow, returned: ReadWindow) -> None:
    """No band-wide lowering may alter `window`: none of them drop rows, add rows, or
    change the pixel grid, so a lowering that returns a `window` different from the one
    it was handed is a bug, not a data-dependent outcome. Cheap to check -- `ReadWindow`
    is a small frozen dataclass, so this is a plain equality comparison, not a query --
    so every band-wide lowering should call this on its own output rather than trusting
    itself to have passed `window` through untouched.
    """
    assert returned == handed, (
        "band-wide lowering returned a different window than it was handed "
        f"({handed!r} -> {returned!r}) -- no band-wide lowering may alter the window"
    )


def assert_same_pixel_grid(one: ReadWindow, other: ReadWindow) -> None:
    """Two independently produced `ReadWindow`s address the same pixels in the same
    order -- the precondition a positional (row *i* is row *i*) combination of two
    band-wide frames needs before it may line their rows up.

    Every field BUT `source` is compared. `source` is deliberately excluded: the whole
    point of combining two band-wide frames positionally is that they came from
    different reads, so requiring identical `source` would reject exactly the case this
    check exists to admit. What makes row *i* the same pixel in both is the geometry --
    offsets, extent, transform, CRS, decimation -- not which file it was read out of.
    Contrast `assert_window_unchanged`, which compares `source` too because there the
    two windows are the SAME read, before and after one lowering, and a changed `source`
    there is unambiguously a bug.

    **`batch_layout` is compared, and is the field that most often differs.** Sharing a
    patch of ground is necessary but NOT sufficient: two reads of the identical window
    over the identical grid still emit their pixels in different ORDERS when they were
    chunked differently, and they are chunked differently whenever their band counts
    differ, since a fixed read-buffer budget buys fewer blocks per batch the more bands
    each block carries (`io/window.py`'s `batch_layout`, `io/raster_io.py`'s
    `raster_blocks_per_batch`). A batch covers a sub-rectangle of a block row, not a
    full-width strip, so two different chunkings are two different permutations of the
    same pixels -- every one present in both frames, at mismatched row indices. Without
    this field the two frames look interchangeable on every geometric measure and
    combining them yields a frame whose shape, extent and row count are all exactly right
    and whose values are paired with the wrong pixels: measured on a 1024x1024 four-band
    read against a co-registered one-band read, 1044480 of 1048576 rows landed on the
    wrong pixel, with nothing in the result to say so. A caller that trips this does not
    want the check relaxed; it wants ONE read (a single `scan_raster` over all the bands
    it needs, crosswalks included -- which is what that function's `crosswalks=` parameter
    is for), whose pixel order is then shared by construction rather than by assertion.
    """
    differing = [
        field
        for field in ("col_off", "row_off", "width", "height", "transform", "crs", "decimation", "batch_layout")
        if getattr(one, field) != getattr(other, field)
    ]
    assert not differing, (
        f"band-wide frames do not share a pixel grid -- {', '.join(differing)} differ "
        f"({one!r} vs {other!r}); their rows cannot be aligned positionally"
    )


def row_major_grid_cell(lf: pl.LazyFrame, window: ReadWindow, name: str = "grid_cell") -> pl.LazyFrame:
    """`lf` with a `name` column holding each row's ROW-MAJOR pixel position within
    `window` -- the cell id a raster grid addresses that pixel by, recovered from the row
    index the frame actually carries.

    A plain `with_row_index` is not that id. It is only that id when `window.batch_layout`
    describes a single batch covering the whole window, which is the case for a small read
    and for a windowed one, and NOT the case for a full read of a real grid: there the read
    is chunked to bound its buffer, each batch covers a sub-rectangle of a block row, and
    the frame's rows run row-major WITHIN each batch, batch after batch. Row 1 of such a
    frame is the pixel one column to the right, but row `batch_width` is the pixel at the
    start of the batch's second line -- not the pixel at the start of the raster's second
    line, which lives thousands of rows further on in another batch entirely. How many
    batches a read is cut into depends on its band count, so this is not even a fixed
    property of a grid: the same pixel sits at two different row indices in a one-band and
    a four-band read of the same window.

    Decoded with a backward `join_asof` on each batch's starting row index -- the frame's
    row index is ascending by construction, which is exactly what `join_asof` needs, and it
    keeps the whole decode lazy rather than materializing a per-pixel lookup table. The
    arithmetic per row is then the obvious one: which batch the row falls in, its offset
    within that batch, that offset split into a line and a column by the BATCH's width, and
    those placed back inside `window` by the batch's own offsets.

    `window.batch_layout` is empty for a `ReadWindow` not built by a batched scan (a
    hand-constructed one in a test, say). That is treated as the single-batch case -- the
    whole window, one batch -- so the id reduces to the plain row-major index, which is
    what such a window means.
    """
    layout = window.batch_layout or ((window.col_off, window.row_off, window.width, window.height),)
    starts = list(itertools.accumulate((w * h for _c, _r, w, h in layout), initial=0))[:-1]
    batches = pl.DataFrame(
        {
            "__batch_start": starts,
            "__batch_col": [col for col, _r, _w, _h in layout],
            "__batch_row": [row for _c, row, _w, _h in layout],
            "__batch_width": [width for _c, _r, width, _h in layout],
        },
        schema={
            "__batch_start": pl.Int64,
            "__batch_col": pl.Int64,
            "__batch_row": pl.Int64,
            "__batch_width": pl.Int64,
        },
    )
    offset = pl.col("__row_index") - pl.col("__batch_start")
    return (
        lf.with_row_index("__row_index")
        .with_columns(pl.col("__row_index").cast(pl.Int64))
        .join_asof(batches.lazy(), left_on="__row_index", right_on="__batch_start", strategy="backward")
        .with_columns(
            (
                (pl.col("__batch_row") + offset // pl.col("__batch_width") - window.row_off) * window.width
                + (pl.col("__batch_col") + offset % pl.col("__batch_width") - window.col_off)
            ).alias(name)
        )
        .drop("__row_index", "__batch_start", "__batch_col", "__batch_row", "__batch_width")
    )


def assert_height_preserved(before: pl.LazyFrame, after: pl.LazyFrame) -> None:
    """Every band-wide lowering preserves row count: a value filter masks instead of
    dropping, a spatial filter changes shape to long instead of dropping band-wide rows,
    a dimension filter selects columns, and a `group` reduces columns -- none of them
    touch which pixels are present. The one lowering that legitimately produces a
    different row count (a dimension filter matching no band) does not go through this
    function at all -- it returns `empty_band_wide` directly, named as the exception
    rather than an assertion this call would have to special-case.

    **This is a test-time assertion, not one a lowering calls on itself.** Both sides are
    collected, and a band-wide frame's `before` is typically an unmaterialized raster
    scan, so each call is a real read of every pixel -- twice per invocation, and a
    lowering calling it inline would pay that on every op in a walk. The lowerings
    themselves build `after` with `select`/`with_columns`/a horizontal concat of
    equal-height frames, none of which can change a row count, so the guarantee is
    structural there; this function is how the tests check that from the outside.
    """
    before_height = before.select(pl.len()).collect().item()
    after_height = after.select(pl.len()).collect().item()
    assert after_height == before_height, (
        f"band-wide lowering changed height from {before_height} to {after_height} -- "
        "no band-wide lowering may drop or add pixel rows (the sole exception, a "
        "dimension filter matching no band, returns empty_band_wide instead of going "
        "through this assertion)"
    )
