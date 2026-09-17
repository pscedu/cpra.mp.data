"""`RasterGridAnalysis`: the raster half of the former `api/dataset.py`'s `Dataset` class.

See `analysis/base.py`'s module docstring for the `apply_params` flow and the hook
contract every method here fills in; see `analysis/vector.py` for the vector half.
"""

import asyncio
import itertools
import tempfile
from contextlib import ExitStack
from functools import partial
from io import BytesIO
from pathlib import Path

import mercantile
import numpy as np
import polars as pl
import polars.selectors as cs
import rasterio as rio
from PIL import Image
from rasterio import windows
from rasterio.coords import BoundingBox, disjoint_bounds
from rasterio.shutil import copy as rio_copy
from rasterio.warp import transform_bounds

from cpra.mp.data import config
from cpra.mp.data.analysis import stages
from cpra.mp.data.analysis.base import AnalysisError, BaseAnalysis, op_error_boundary
from cpra.mp.data.analysis.io.geography import parse_coordinate_pairs, project_coordinates
from cpra.mp.data.analysis.io.raster_io import (
    asynctiming,
    group_rows_by_path,
    parallel_read_rasters,
    parallel_sample_rasters,
    raster_blocks_per_batch,
    raster_copy_gdal_env,
    raster_gdal_env,
    tile_read_shape,
    tile_source_window,
    unpack_values,
    warp_window_to_tile,
)
from cpra.mp.data.analysis.ops import CalculateOp, FilterOp, GroupOp, Op, ProjectOp, Shape, calculate_label
from cpra.mp.data.analysis.request.expressions import (
    ExpressionError,
    classify_where,
    compile_calculate_entries,
    parse_value_expr,
    rewrite_named_columns,
)
from cpra.mp.data.analysis.sources import (
    NUMPY_TO_POLARS,
    band_calculate_passes,
    guarded_aggregate,
    horizontal_aggregate_expr,
    masked_value_predicate_exprs,
    scan_rasters,
)
from cpra.mp.data.analysis.sources import (
    count_by_value as raster_count_by_value,
)
# The LONG-shape lowering of `filter`/`calculate`/`group`, shared with the vector
# interpreter rather than reimplemented here: a raster chain reaches that shape after a
# `project` raster-to-vector (a summarizing `project` step, or a `grid_cell` pointwise
# sample), and from that point on its frame is structurally a vector frame. It lives in
# `analysis/vector.py` because `apply_group_by`/`apply_project`, the two lowerings it calls
# that are not already in `analysis/base.py`, live there; `analysis/vector.py` imports
# nothing from this module, so this direction is the acyclic one.
from cpra.mp.data.analysis.vector import apply_grid_cell_selection, apply_long_op
from cpra.mp.data.config import (
    COUNT_BY_VALUE_MAX_DISTINCT,
    RASTER_NUM_THREADS,
    RASTER_USE_VRT_COPY,
    RASTER_WINDOW_BUFFER_LIMIT_MB,
)
from cpra.mp.data.raster import set_vrt_band_descriptions, vrt

# Interleaving for downloaded rasters, set rather than inherited from the source. A source
# COG may be written INTERLEAVE=TILE, which is a COG-driver-only option that the GTiff driver
# used here rejects outright ("INTERLEAVE=tile unsupported, value must be PIXEL or BAND"), so
# inheriting it makes such a source undownloadable. PIXEL is also what keeps a multiband
# response small: the same 52-band elev selection wrote 1.65 GB pixel-interleaved versus
# 19.3 GB band-interleaved, since only pixel interleaving puts each pixel's bands adjacent
# for the compression predictor.
OUTPUT_INTERLEAVE = "PIXEL"


# -- band-wide op lowerings --
#
# Re-entrant functions -- each takes its own arguments and returns new ones instead of
# reading `self.lf`/`self.options` -- that `RasterGridAnalysis.apply_steps`'s interpreter
# and its three pixel-read consumers (`summarize_zone_stats`,
# `_collapse_tile_bands`, `_write_raster_windowed`) use to apply a request's
# filter/calculate/group steps IN DECLARED ORDER against a real, per-op band-wide frame,
# rather than at one position fixed by the engine.
#
# Every op here has the SAME signature: `(band_wide_lf, metadata_df) -> (band_wide_lf,
# metadata_df)`. The band-wide frame is one row per pixel, one column per currently-live
# band; the metadata frame is one row per currently-live band (dimension columns only, no
# pixels -- tens to low hundreds of rows, never pixel-scale). The metadata frame is the
# band-wide frame's COLUMN DICTIONARY: it is what says what `b0..bN` currently mean, and
# the two are tied together by one invariant, `band_column_names` below --
#
#     the band columns are `b0..b{metadata_df.height - 1}`, recomputed at each step.
#
# An op therefore selects the columns it needs by QUERYING the metadata frame it was
# handed, never by trusting a row-index structure computed at scan time. That is the whole
# point of threading metadata through: an op that changes the band set (`filter`, `group`)
# returns a metadata frame describing the bands it left behind, so the NEXT op reads a
# dictionary that is already correct for the frame it actually receives. A stored index
# would have gone stale at exactly that moment.
#
# Which ops change the band set, i.e. which preserve BAND IDENTITY:
#
#   `apply_value_filter_band_wide`  per-cell mask       metadata unchanged
#   `apply_calculate_band_wide`     per-cell rewrite    metadata unchanged
#   `apply_dimension_filter_band_wide`  column drop     metadata narrowed (rows dropped)
#   `apply_group_band_wide`         horizontal reduce   metadata REPLACED (one row/group)
#
# That column is also the pushdown rule `apply_band_wide_op` follows: a
# dimension filter may be applied to the band-metadata scan up front (narrowing what is
# ever opened) only over a PREFIX of the chain in which every preceding op preserves band
# identity. `filter` and `calculate` do; `group` does not, so the first `group` ends the
# prefix and every later dimension filter runs at its own declared position instead, as a
# recorded band op replayed by `apply_band_ops`.

# The reserved column a COLLAPSED band-metadata frame carries so its HEIGHT keeps saying
# how many bands it describes. A `group` narrows the metadata to its own `by` columns, and
# `by` is legitimately allowed to be empty ("collapse every selected band into one"), which
# would otherwise leave a zero-COLUMN frame -- and Polars cannot represent "zero columns, N
# rows": such a frame reports height 0 no matter how many rows it was built from, which
# would silently break the `band_column_names` invariant every op depends on. This column
# holds each surviving band's position in the frame it belongs to -- the `i` of its own
# `b<i>` column -- so the invariant is written down rather than inferred. Dropped again by
# `RasterGridAnalysis._get_raster_dimension_frame` before any dimension frame is handed
# out, so it never reaches a response.
BAND_INDEX = "_band"


def band_column_names(metadata_df):
    """The band-wide frame's band column names, `b0..bN`, for the metadata frame that
    currently describes it -- `sources.scan_rasters`' own naming, extended to every later
    step by recomputing it from the CURRENT metadata frame rather than carrying the scan's
    original names forward.

    This one-line function is the band-wide invariant itself, in the one place every op
    and every pixel-read consumer reads it from: column `b<i>` is the band described by
    metadata row `i`. An op that drops or collapses bands renames what survives back down
    to a dense `b0..bK-1` precisely so this keeps holding for whatever runs next.
    """
    return [f"b{i}" for i in range(metadata_df.height)]


def band_dimension_keep(metadata_df, dimension_where):
    """Which of `metadata_df`'s rows (one row per live band column, in the SAME order the
    band-wide frame's `b<i>` columns are named in -- see `band_column_names`) satisfy a
    DIMENSION predicate.

    The shared "evaluate this dimension condition against the band-metadata table"
    primitive: the dimension-`FilterOp` lowering (`band_metadata_filter`, below) and the
    `CalculateOp` lowering (`apply_calculate_band_wide`, whose per-plan `where` selects
    which BANDS that plan rewrites) both go through it rather than each spelling the same
    `pl.sql_expr` evaluation.

    A falsy `dimension_where` keeps every band ("no dimension `where` at all" means the
    condition applies to every band). Raises `ExpressionError` if `dimension_where` cannot
    be evaluated against `metadata_df`'s columns -- a plain exception, since a library
    function has no `self._error` to call; each caller converts at its own boundary.
    """
    if not dimension_where:
        return [True] * metadata_df.height
    try:
        mask = metadata_df.lazy().select(pl.sql_expr(dimension_where).fill_null(False)).collect().to_series()
    except Exception as exc:
        raise ExpressionError(
            f"where {dimension_where!r} cannot be evaluated against this dataset's bands: {exc}"
        ) from exc
    return mask.to_list()


def band_metadata_filter(metadata_df, dimension_where):
    """The METADATA half of a dimension filter: `(metadata_df, keep)`, the surviving band
    rows plus the per-row boolean that says which ones they were.

    Split out from `apply_dimension_filter_band_wide` (below) because
    `apply_band_wide_op` needs exactly this half on its own, with no band-wide
    frame in sight: a dimension filter declared after a `group` has to narrow the running
    band metadata at its declared position, hours before any pixel is read. Sharing this
    function is what keeps that in-flight narrowing and the eventual pixel-read replay from
    being two implementations that can disagree.
    """
    keep = band_dimension_keep(metadata_df, dimension_where)
    return metadata_df.filter(pl.Series(keep)), keep


def band_metadata_scan_filter(lf, dimension_where):
    """The PUSHDOWN lowering of a dimension filter: narrow the band-metadata SCAN itself,
    so the bands the predicate rejects are never opened at all.

    Legal only over the prefix of a chain in which every preceding op preserves band
    identity (`filter`/`calculate` do, `group` does not) -- see
    `apply_band_wide_op`'s pushdown rule for the full argument. Once a `group`
    has run, `band_metadata_filter` above is the lowering instead.

    `lf.sql` rejects a SYNTACTICALLY malformed predicate eagerly, raising out of Polars'
    SQL layer; that becomes an `ExpressionError` here so the interpreter's one error
    boundary (`base.op_error_boundary`) sees the same typed exception every other lowering
    raises, with the message text unchanged.

    It does NOT cover an unknown COLUMN, and does not need to: `lf.sql` only PLANS, so name
    resolution is deferred to the eventual `collect()`, where `BaseAnalysis.materialize`'s
    own `ColumnNotFoundError` handler already turns it into the same 400. (That deferral is
    load-bearing rather than incidental -- `_validate_filter_where` deliberately does not
    namespace-restrict a dimension conjunct at plan time, matching the flat `where=`
    surface, so an unknown dimension name is EXPECTED to reach execution.) Both halves are
    pinned by `tests/api/test_raster_ops.py`'s `TestBandMetadataScanFilter`.
    """
    try:
        return lf.sql(f"select * from self where {dimension_where}")
    except Exception as exc:
        raise ExpressionError(str(exc)) from exc


def band_group_row_idxs(metadata_df, by):
    """Which of `metadata_df`'s band rows belong to each output group, keyed by `by`.

    An empty `by` means "collapse every row into a single group" -- Polars' own `group_by()`
    with no columns does the same, and this is spelled explicitly rather than relying on
    that so the single-group case is obvious at the call site.

    `maintain_order=True`: the output group order IS the output band order (group `k`
    becomes band `b<k>`), so it has to be a property of the data rather than of Polars'
    hash-partition scheduling. First-occurrence order also keeps a grouped selection's
    bands in the same relative order the ungrouped one would have had.
    """
    if not by:
        return [list(range(metadata_df.height))]
    grouped = metadata_df.with_row_index("row_idxs").group_by(*by, maintain_order=True).agg(pl.col("row_idxs"))
    return grouped["row_idxs"].to_list()


def band_metadata_group(metadata_df, by):
    """The METADATA half of a group: `(metadata_df, group_row_idxs)`, one row per surviving
    group plus the input row positions contributing to each.

    The new metadata frame is `BAND_INDEX` (see that constant for why it has to be there at
    all) hstacked with the `by` columns, taken from the FIRST input row of each group so
    the two stay aligned by construction rather than via a second, independently-ordered
    `group_by`. Every dimension column NOT in `by` is dropped, mirroring the long-shape
    lowering's own drop (`vector.apply_group_by`): a dimension the caller collapsed across
    has no single value left to report.

    Shared by `apply_group_band_wide` and by `collapse_band_group`, for the
    same reason `band_metadata_filter` is shared -- the metadata a group leaves behind must
    be identical whether it is being computed in flight or replayed at pixel-read time.
    """
    group_row_idxs = band_group_row_idxs(metadata_df, by)
    grouped = pl.DataFrame({BAND_INDEX: pl.Series(range(len(group_row_idxs)), dtype=pl.UInt32)})
    if by:
        first_idxs = [idxs[0] for idxs in group_row_idxs]
        grouped = grouped.hstack(metadata_df[first_idxs].select(by))
    return grouped, group_row_idxs


def apply_value_filter_band_wide(lf, metadata_df, value_where, extra_columns=()):
    """Band-wide lowering of a VALUE `FilterOp` conjunct (`ops.LOWERING_TABLE`'s
    `("filter (value)", Shape.band_wide)` row): a per-cell MASK via
    `sources.masked_value_predicate_exprs`, never a row filter/drop. Dropping the ROW would
    drop that pixel from every OTHER band column too, and a raster/tile-output query must
    return the same pixel count it read (pinned mechanically by
    `tests/api/test_raster_source.py::TestValueFilterPixelCountPreservation`, this
    function's own prerequisite).

    Preserves band identity: a mask nulls cells within the existing columns, so
    `metadata_df` is returned unchanged.

    A falsy `value_where` (`None`/`""`) is a no-op, matching `apply_value_filter`'s own
    (vector) guard. `extra_columns` is `masked_value_predicate_exprs`' identically-named
    parameter -- the columns a predicate may also reference (`grid_cell`, for a zoned scan).

    This is also the value-filter gate: a raster-OUTPUT query's value `where` reaches its
    `scan_rasters`-backed `lf` through THIS function, never `lf.filter(...)`, because the
    source plugin cannot know which shape it is scanning for, so the compiler has to
    decide. See `_write_raster_windowed`'s docstring for the ordering argument and
    `tests/api/test_write_raster.py::TestValueFilterGate` for the pixel-count/position
    -preservation assertion wired to that real path.
    """
    if not value_where:
        return lf, metadata_df
    names = band_column_names(metadata_df)
    return lf.with_columns(masked_value_predicate_exprs(names, value_where, extra_columns=extra_columns)), metadata_df


def apply_dimension_filter_band_wide(lf, metadata_df, dimension_where, extra_columns=()):
    """Band-wide lowering of a DIMENSION `FilterOp` conjunct (`ops.LOWERING_TABLE`'s
    `("filter (dimension)", Shape.band_wide)` row): a COLUMN SELECTION -- projection
    pushdown -- never a row filter. A dimension lives in column IDENTITY in band-wide shape
    (which band is which calendar step, model_group_id, ...), not in a per-pixel cell, so
    "keeping" a dimension condition means keeping the whole columns it selects and dropping
    the rest entirely.

    Does NOT preserve band identity, so it returns a narrowed `metadata_df` AND renames the
    surviving columns back down to a dense `b0..bK-1` in the same relative order -- which is
    what keeps `band_column_names(metadata_df)` true for whatever op runs next, instead of
    leaving a sparse `b0, b3, b7` naming that only a stored index could interpret. A single
    `.select()` (not `with_columns`) makes the rename safe even though the new names overlap
    the old ones: every expression is evaluated against the INPUT frame.

    `extra_columns` (e.g. a `grid_cell` column attached by a zoned scan) are always kept,
    exactly like `masked_value_predicate_exprs`' identically-named parameter.

    A falsy `dimension_where` is a no-op.
    """
    if not dimension_where:
        return lf, metadata_df
    names = band_column_names(metadata_df)
    new_metadata_df, keep = band_metadata_filter(metadata_df, dimension_where)
    kept = [name for name, keeping in zip(names, keep) if keeping]
    renamed = [pl.col(old).alias(new) for old, new in zip(kept, band_column_names(new_metadata_df))]
    return lf.select([*extra_columns, *renamed]), new_metadata_df


def apply_calculate_band_wide(lf, metadata_df, plans):
    """Band-wide lowering of `CalculateOp` (`ops.LOWERING_TABLE`'s `("calculate",
    Shape.band_wide)` row): `with_columns` across the current band columns, reusing
    `sources.band_calculate_passes` UNCHANGED -- the same shared calculate vocabulary every
    pixel-read consumer speaks, rather than a second implementation of it.

    `plans` are `expressions.CalculatePlan`s (`expressions.compile_calculate_entries`'s
    output), applied as sequential passes -- entry N+1 sees entry N's own rewrite, exactly
    `band_calculate_passes`' documented contract. An empty `plans` is a no-op.

    Each plan's per-band applicability is computed HERE, from the metadata frame this call
    was handed, so a plan declared after a `group` is resolved against the GROUPED bands it
    was actually written against rather than against the raw bands that no longer exist.
    The dimension half of a calculate `where` picks BANDS (e.g. `model_group_id = 500`); the
    value half is per-pixel and is folded into each band's own rewrite condition by
    `band_calculate_passes`.

    Preserves band identity: every rewrite is aliased back onto its own column, so
    `metadata_df` is returned unchanged.

    An unevaluable dimension `where` raises `ExpressionError` prefixed with the plan's own
    label, so the caller's 400 names the offending entry (`calculate.1`, `step.3`, ...)
    rather than reporting an anonymous failure.
    """
    if not plans:
        return lf, metadata_df
    applicability = []
    for plan in plans:
        try:
            applicability.append(band_dimension_keep(metadata_df, plan.dimension_where))
        except ExpressionError as exc:
            raise ExpressionError(f"{plan.label}: {exc}") from exc
    names = band_column_names(metadata_df)
    for pass_exprs in band_calculate_passes(names, plans, applicability):
        lf = lf.with_columns(pass_exprs)
    return lf, metadata_df


def apply_group_band_wide(lf, metadata_df, by, aggregate, extra_columns=()):
    """Band-wide lowering of `GroupOp` (`ops.LOWERING_TABLE`'s `("group", Shape.band_wide)`
    row): a HORIZONTAL reduce across the current band columns, collapsing each group keyed
    by `by` down to one output column, via the shared `sources.horizontal_aggregate_expr`.

    Does NOT preserve band identity -- it is the only op here that replaces the metadata
    frame outright (`band_metadata_group`), which is exactly why a dimension filter cannot
    be pushed past one. Output group `k` becomes column `b<k>`, keeping
    `band_column_names(metadata_df)` true for whatever runs next. A single `.select()` (not
    `with_columns`) makes that renaming safe even though the outputs reuse the inputs' own
    names: every expression is evaluated against the INPUT frame.

    "No band contributed anything to this pixel" stays NULL here, unconditionally, rather
    than being filled with a sentinel: a later op in the same chain (another `group`, or a
    `calculate`'s `is_not_null()` guard) has to be able to tell "nothing here" from a real
    value, and the one caller that does need a sentinel in its output columns
    (`apply_band_ops` with a `dtype`) fills it once, at the very end, after every op has run.

    Rejects `aggregate="count"` up front: `HORIZONTAL_RASTER_AGGREGATES` deliberately has no
    `count` entry, since "how many of these bands contributed to this pixel" has no natural
    per-pixel scalar meaning the way max/mean/min/sum do (see the `AggregateFunction.count`
    comment in types.py). Raised as an `ExpressionError` rather than left to surface as a
    bare `KeyError` from inside the aggregate lookup. `collapse_band_group` rejects the same
    value independently and earlier, with the identical message, on every live request
    path, so this is a second guard for a direct caller, not a new behavior.
    """
    if aggregate == "count":
        raise ExpressionError("group_aggregate 'count' is not supported for raster grids")
    names = band_column_names(metadata_df)
    new_metadata_df, group_row_idxs = band_metadata_group(metadata_df, by)
    exprs = [
        horizontal_aggregate_expr(aggregate, [pl.col(names[i]) for i in row_idxs], pl.lit(None)).alias(name)
        for name, row_idxs in zip(band_column_names(new_metadata_df), group_row_idxs)
    ]
    return lf.select([*extra_columns, *exprs]), new_metadata_df


def crosswalk_value_column_names(metadata_df, prefix):
    """The value column names `pivot_vector_to_band_wide` produces and
    `sources.scan_rasters` emits for a crosswalk named `prefix`: `<prefix>0..<prefix>N`.

    The same one-line invariant `band_column_names` is, for a crosswalk's value columns
    instead of the band columns: `<prefix><i>` is the vector value paired with band `b<i>`,
    so the two families stay index-aligned by construction and a caller writing an
    expression over both zips two lists computed from one `metadata_df` rather than tracking
    a stored mapping.
    """
    return [f"{prefix}{i}" for i in range(metadata_df.height)]


def pivot_vector_to_band_wide(
    metadata_df,
    vector_df,
    band_steps,
    *,
    step_column,
    vector_zone_column="grid_cell",
    value_column="value",
    prefix="v",
):
    """Turn a LONG vector frame -- one row per `(zone, step)` -- into the BAND-WIDE value
    table `sources.GridCrosswalk.values` takes: one row per zone, one column per band.

    This is the long-to-band-wide conversion, and it is the half of projecting a vector
    dataset onto a raster grid that has no raster in it at all. The other half -- which zone
    each PIXEL is in -- is a positional crosswalk read, and belongs to the scan
    (`scan_rasters(crosswalks=...)`), which also performs the final per-pixel lookup against
    the table this returns. See `GridCrosswalk`'s own docstring for why the lookup cannot
    currently sit between the two as an ordinary frame op.

    Pivoting first is not an optimization. `vector_df` is LONG, so looking it up by zone
    alone would match each pixel once PER STEP; expressed as a join that multiplies the
    frame's height by the number of steps, and expressed as a lookup array it is simply
    ill-defined -- which step's value did that zone get? A raster-output write requires one
    row per pixel in raster order (`_take_window_rows` surfaces any short read as a hard
    error, and a value filter has to be a mask for the same reason), so a height change is
    not a performance question, it is a wrong answer. Pivoting by the dimension that defines
    band identity makes the lookup single-valued: each band reads its own column.

    `band_steps` is one step value per band, positionally aligned with
    `band_column_names(metadata_df)`. Distinct steps are numbered and the vector frame is
    pivoted on that numbering rather than on the step values themselves, so a step of any
    dtype (Int32 year, `pl.Date`, string) works without the pivot's column names having to be
    round-trippable strings. Two bands sharing a step both select that one pivoted column,
    emitted twice under both names; a band whose step has no vector row at all gets an
    all-null column, the same outcome as a zone with no matching row.

    Duplicate zone ids resolve last-wins (`aggregate_function="last"`), which is NumPy's own
    rule for the fancy-index assignment in `sources._dense_lookup_array` that consumes this,
    and is kept identical deliberately. Rows with a null zone id or a step outside
    `band_steps` are dropped here: they can never be selected by any band, and
    `_dense_lookup_array` cannot index a null.

    Returns a `pl.DataFrame` whose key column is named `prefix` -- matching the
    `GridCrosswalk.name` it will be attached under, which is also the zone column that
    crosswalk emits -- plus one Float64 column per name in
    `crosswalk_value_column_names(metadata_df, prefix)`.
    """
    names = band_column_names(metadata_df)
    if len(band_steps) != len(names):
        raise ValueError(
            f"pivot_vector_to_band_wide: {len(band_steps)} steps for {len(names)} bands -- "
            "one step per band column is required"
        )
    out_names = crosswalk_value_column_names(metadata_df, prefix)

    # Distinct steps in first-appearance order -> 0..K-1. `dict.fromkeys` both dedupes and
    # fixes that order in one pass, and works for any hashable step value.
    step_index = {step: i for i, step in enumerate(dict.fromkeys(band_steps))}

    narrowed = (
        vector_df.lazy()
        .select(
            pl.col(vector_zone_column).cast(pl.Int64, strict=False),
            pl.col(step_column).replace_strict(step_index, default=None, return_dtype=pl.Int32).alias("_step_idx"),
            pl.col(value_column).cast(pl.Float64).alias("_value"),
        )
        .drop_nulls([vector_zone_column, "_step_idx"])
        .collect()
    )
    if narrowed.height == 0:
        # `DataFrame.pivot` on an empty frame produces no value columns at all, so the
        # select below would have nothing to reference. An empty table is a complete
        # answer -- every lookup misses and every band's column is null -- so build it
        # directly rather than special-casing the miss downstream.
        return pl.DataFrame(schema={prefix: pl.Int64, **{n: pl.Float64 for n in out_names}})

    wide = narrowed.pivot(
        on="_step_idx",
        index=vector_zone_column,
        values="_value",
        aggregate_function="last",
    )
    present = set(wide.columns)
    return wide.select(
        pl.col(vector_zone_column).alias(prefix),
        *[
            (pl.col(str(step_index[step])) if str(step_index[step]) in present else pl.lit(None, dtype=pl.Float64))
            .cast(pl.Float64)
            .fill_nan(None)
            .alias(out_name)
            for step, out_name in zip(band_steps, out_names)
        ],
    )


async def sample_band_ops_to_long(metadata_df, band_ops, grid_cell):
    """`project` raster-to-vector at arbitrary points, TERMINALLY: sample every band in
    `metadata_df` at each `"lng,lat"` pair in `grid_cell`, replay `band_ops` (`RasterGridAnalysis.
    _pixel_read_band_ops()` -- the same declared filter/calculate/group recipe every other
    pixel-read consumer replays) against the sampled points via `apply_band_ops`, then
    unpivot the surviving bands into a LONG LazyFrame (dimension columns, `grid_cell`,
    `value`) -- the other direction from `pivot_vector_to_band_wide` above.

    `metadata_df` is `RasterGridAnalysis._band_frame()`: the RAW per-band frame (`path`/
    `band` plus dimension columns), from before any `group` collapsed it -- a pixel-read
    consumer always starts there, since a collapsed metadata frame no longer names real
    files to sample. A `grid_cell` sample is a FOURTH pixel-read consumer here, alongside
    tile/write/summarize, not a separate code path with its own copy of filter/calculate/
    group -- see `RasterGridAnalysis.apply_steps`'s docstring for why the walk still
    records `band_ops` even when no `summarize_by` will ever consume them directly.

    The points sampled here are built as a genuine BAND-WIDE frame first (one row per
    requested point, standing in for "one row per pixel"; one column per band, named
    `band_column_names(metadata_df)`) -- exactly the shape `apply_band_ops` expects --
    and only unpivoted to long AFTER the replay, not before. Replaying against the raw
    per-point samples directly (rather than resampling from an already-long frame) is
    what lets a `group`'s horizontal collapse, or a post-group dimension filter's column
    drop, see the same band identity the band-wide walk itself tracked; unpivoting first
    would have already discarded it. `nodata=None` matches the summarize path's own
    choice: a value filter's mask (or an all-null `group`) propagates a genuine null into
    `value` rather than a fabricated sentinel, since this produces tabular output, not a
    rendered image.

    A declared value-`filter` conjunct is a ROW DROP once replayed here, matching the
    long-shape lowering's own contract (`vector.apply_long_op`'s docstring: "a `FilterOp`'s
    value conjunct is a ROW DROP unless the op itself asks for a mask") -- `apply_band_ops`
    itself cannot drop a row (band-wide shape must preserve every pixel's row identity, see
    `apply_value_filter_band_wide`), so the drop happens here instead, once, after the
    replay: any row a `("filter_value", ...)` entry masked to null is dropped. A null a
    `calculate`/`group` produces with NO value filter declared is left in place as an
    ordinary null `value`, the same as any other long-shape frame -- this only reaches for
    the row-drop the value-filter contract actually requires, not every null this replay
    could in principle produce.
    """
    lngs, lats = parse_coordinate_pairs(unpack_values(grid_cell, "grid_cell"))

    # Transform the coordinates to the correct CRS, and sample the rasters in parallel.
    # Rows sharing the same path (bands of the same multiband COG) are grouped so each
    # file is opened and sampled once, requesting all of its needed bands together.
    unique_paths, row_indices, bands = group_rows_by_path(metadata_df)
    with ExitStack() as stack:
        datasets = [stack.enter_context(rio.open(path)) for path in unique_paths]
        projected_pairs = list(zip(*project_coordinates(lngs, lats, datasets[0].crs)))
        group_samples = await parallel_sample_rasters(datasets, projected_pairs, bands=bands)
        dtype = datasets[0].dtypes[0]

    # `columns[p]` is one point's value across every band (length `len(metadata_df)`);
    # stacking them turns "one array per point" into a `(n_points, n_bands)` matrix, so
    # transposing column-vs-row gives the points-as-rows, bands-as-columns frame
    # `apply_band_ops` expects.
    columns = [np.empty(len(metadata_df), dtype=dtype) for _ in lngs]
    for path_row_indices, path_samples in zip(row_indices, group_samples):
        for column, values in zip(columns, path_samples):
            column[path_row_indices] = values
    band_names = band_column_names(metadata_df)
    points_arr = np.stack(columns, axis=0) if columns else np.empty((0, len(metadata_df)), dtype=dtype)
    points_df = pl.DataFrame({name: points_arr[:, i] for i, name in enumerate(band_names)})

    lf, out_metadata_df = apply_band_ops(points_df.lazy(), metadata_df, band_ops, nodata=None)
    column_labels = [f"{x!s},{y!s}" for x, y in zip(lngs, lats)]
    df = lf.collect().with_columns(pl.Series("grid_cell", column_labels))

    # Melt the surviving `b<i>` columns to long, then rejoin each row's dimension columns
    # off `out_metadata_df` by parsing `i` back out of the melted column's own name --
    # `out_metadata_df` row `i` is exactly what `b<i>` means, post-replay, by the same
    # `band_column_names` invariant every band-wide op preserves.
    value_cols = band_column_names(out_metadata_df)
    dim_df = out_metadata_df.select(cs.exclude("path", "band")).with_row_index("_sampled_band_idx")
    melted = df.unpivot(index=["grid_cell"], on=value_cols, variable_name="_band_col", value_name="value")
    melted = melted.with_columns(
        pl.col("_band_col").str.slice(1).cast(pl.UInt32).alias("_sampled_band_idx")
    )
    long_df = melted.join(dim_df, on="_sampled_band_idx", how="left").drop(["_sampled_band_idx", "_band_col"])

    if any(kind == "filter_value" for kind, *_ in band_ops):
        long_df = long_df.filter(pl.col("value").is_not_null())

    return long_df.lazy()


def band_ops_for_pixel_read(band_ops, value_where):
    """`band_ops` as a pixel-read consumer should replay them, with a standalone value
    `where` folded in.

    `RasterGridAnalysis.apply_steps` records a value conjunct as its own `("filter_value",
    ...)` entry at the position it was declared, so a value filter declared AFTER a
    calculate masks the REWRITTEN value, as declared. The flat (non-pipeline) request
    surface has no positions to record: `Options.where` is one string applied at stage one,
    before everything else, so its `self._value_where` is prepended here instead. Detected
    by looking for an existing `filter_value` entry rather than by asking which engine ran,
    since that is the actual question -- has this value predicate already been given a
    position, or not.
    """
    if value_where is None or any(kind == "filter_value" for kind, *_ in band_ops):
        return list(band_ops)
    return [("filter_value", value_where), *band_ops]


def apply_band_ops(lf, metadata_df, band_ops, *, extra_columns=(), dtype=None, nodata=None):
    """Replay `band_ops` (`RasterGridAnalysis._band_ops`: an ordered list of
    `("calculate", plans)` / `("filter", dimension_where)` / `("filter_value", value_where)`
    / `("group", by, aggregate)` entries, one per op actually declared, in DECLARED order)
    against a band-wide `lf` whose columns are `band_column_names(metadata_df)`.

    Every entry carries only what the caller DECLARED -- a predicate string, a plan list, a
    group key -- and never a row-index structure or a frozen copy of the metadata it was
    declared against. The metadata is recomputed as the replay walks the chain, by the same
    ops that transform the pixels, which is what makes an entry that changes the band set
    compose with everything after it instead of invalidating it.

    `dtype`/`nodata` given (a tile or write caller) means: mask the starting columns against
    `nodata` before the first entry replays (nodata interpretation is not itself a declared
    op, so it always runs first regardless of `band_ops`' contents), and cast+fill every
    surviving null in the FINAL columns back to `nodata` (or 0, absent one) once every entry
    has replayed, so the caller gets a column it can hand straight to NumPy. Left `None` (a
    summarize caller, whose `scan_rasters` call already masked nodata via `mask_nodata=True`
    and wants a genuine null propagated into the vertical `guarded_aggregate` rather than a
    fabricated value) means neither happens.

    Returns `(lf, metadata_df)`: both as the LAST entry left them, so a caller reads its
    output band names off `band_column_names(metadata_df)` and its output band count off
    `metadata_df.height`.
    """
    if dtype is not None and nodata is not None:
        lf = lf.with_columns(
            [
                pl.when(pl.col(name) == nodata).then(None).otherwise(pl.col(name)).alias(name)
                for name in band_column_names(metadata_df)
            ]
        )

    for kind, *rest in band_ops:
        if kind == "calculate":
            (plans,) = rest
            lf, metadata_df = apply_calculate_band_wide(lf, metadata_df, plans)
        elif kind == "filter":
            (dimension_where,) = rest
            lf, metadata_df = apply_dimension_filter_band_wide(
                lf, metadata_df, dimension_where, extra_columns=extra_columns
            )
        elif kind == "filter_value":
            (value_where,) = rest
            lf, metadata_df = apply_value_filter_band_wide(lf, metadata_df, value_where, extra_columns=extra_columns)
        elif kind == "group":
            by, aggregate = rest
            lf, metadata_df = apply_group_band_wide(lf, metadata_df, by, aggregate, extra_columns=extra_columns)
        else:  # pragma: no cover -- `_band_ops` only ever holds the four kinds above
            raise ValueError(f"unknown band op kind {kind!r}")

    if dtype is not None:
        # No null reaches the output columns: a masked-out band's null (nodata, a rejected
        # value predicate, or a group no band contributed to) is filled back with the
        # sentinel, in the output dtype, so the caller gets a column it can hand straight to
        # NumPy in the dtype family it would have had unmasked -- which the tile path depends
        # on, since a null would arrive as NaN and land in whichever legend bin NaN happens
        # to digitize into (the last one). With no sentinel declared there is never a null to
        # select, so 0 is a harmless placeholder for that case.
        fill = pl.lit(nodata if nodata is not None else 0).cast(NUMPY_TO_POLARS[dtype])
        lf = lf.with_columns([pl.col(name).fill_null(fill) for name in band_column_names(metadata_df)])
    return lf, metadata_df


# -- band-wide op units (plan time) ---------------------------------------------------
#
# `ops.LOWERING_TABLE`'s BAND-WIDE row for `filter`/`calculate`/`group`, as standalone
# functions rather than as branches of an interpreter method -- the band-wide counterpart
# of `vector.apply_long_op`, which is the long row of the same table. Neither reads or
# writes an instance attribute; every piece of state either arrives as an argument or
# leaves as a return value.
#
# The band-wide op's declared type is `(band-wide LF, metadata DF) -> (band-wide LF,
# metadata DF)`, and the `apply_*_band_wide` family above is exactly that. These three
# functions are the same ops at PLAN time, where no pixel has been read yet: the band-wide
# LF does not exist, so the op's effect on it is RECORDED as a `band_ops` entry for
# `apply_band_ops` to replay against the real frame later, while its effect on the METADATA
# half is applied immediately (the metadata frame is tens to low hundreds of rows, and
# every later op needs it to be correct). The recorded entry and the in-flight metadata
# narrowing go through the same primitives -- `band_metadata_filter`, `band_metadata_group`
# -- so the two halves cannot disagree about what a band is.


def flush_calculate_plans(band_ops, calculate_plans):
    """Move whatever `calculate_plans` has accumulated since the last flush into `band_ops`
    as a single `("calculate", plans)` entry, emptying `calculate_plans` IN PLACE (so a
    caller holding the same list object sees the drain).

    Called by `apply_band_wide_op`/`collapse_band_group` before either appends an entry of
    its own, and once more by each pixel-read consumer for a trailing calculate declared
    after the last filter/group with nothing following it -- so a calculate always lands at
    its declared position relative to the ops around it.

    The entry carries the plans alone. Which BANDS each plan applies to is not decided here:
    `apply_calculate_band_wide` answers that against whatever metadata the replay has
    reached by the time this entry runs, which is the only frame the plan's own dimension
    `where` is meaningful against.

    A no-op once `calculate_plans` is empty (already flushed, or never populated), so it is
    safe to call more than once for the same pending batch.
    """
    if not calculate_plans:
        return
    band_ops.append(("calculate", list(calculate_plans)))
    calculate_plans.clear()


def collapse_band_group(metadata_lf, by, aggregate, *, raw_band_frame, band_ops, calculate_plans):
    """The band-wide lowering of a `group`: collapse the CURRENT band metadata
    (`metadata_lf`: either the untouched raw per-band frame, on this chain's first `group`,
    or an earlier `group`/post-group dimension `filter`'s own output) into one row per `by`
    combination, and record the group as a band op for a pixel-read consumer to replay at
    this position.

    Returns `(metadata_lf, raw_band_frame)`. On the FIRST call `raw_band_frame` comes back
    as a snapshot of `metadata_lf` taken before anything was touched -- every pixel-read
    consumer needs the original (path, band) rows to build its scan, even once the running
    metadata has moved on to dimension-only rows. That snapshot doubles as the "a group has
    run" flag the pushdown rule tests (`apply_band_wide_op`) and `RasterGridAnalysis.
    _band_frame` returns. `band_ops`/`calculate_plans` are appended/drained in place.

    The recorded entry is `("group", by, aggregate)` -- the DECLARED key, not the row
    positions it happens to resolve to against today's metadata. `band_metadata_group`
    resolves those positions here, for the running metadata, and resolves them again at
    pixel-read time against the metadata a replay has reached by then; a stored index could
    not survive a second group or a post-group filter narrowing what it indexed into.

    `aggregate` is always explicit -- a pipeline may declare two group steps with different
    aggregates, and `Options.group_aggregate` holds only the last one, so `ops.GroupOp`
    carries its own and `apply_band_wide_op` passes that, rather than leaving this function
    to read an option that may not describe the step at hand.
    """
    if aggregate == "count":
        # group_aggregate is the raster band-COLLAPSE step
        # (`sources.HORIZONTAL_RASTER_AGGREGATES`), reducing ACROSS one pixel's selected
        # bands -- unlike the zonal summarize aggregate this option shares its name and
        # values with, "how many of these bands contributed to this pixel" has no natural
        # per-pixel scalar meaning the way max/mean/min/sum do (see the
        # `AggregateFunction.count` comment in types.py). Rejected explicitly here rather
        # than left to fail as a raw `KeyError` out of `HORIZONTAL_RASTER_AGGREGATES` deep
        # in a later stage.
        raise AnalysisError("group_aggregate 'count' is not supported for raster grids")

    metadata_df = metadata_lf.collect()
    if raw_band_frame is None:
        raw_band_frame = metadata_df

    flush_calculate_plans(band_ops, calculate_plans)
    band_ops.append(("group", list(by), aggregate))
    return band_metadata_group(metadata_df, by)[0].lazy(), raw_band_frame


def apply_band_wide_op(op, metadata_lf, *, raw_band_frame, band_ops, calculate_plans, value_where, rename_map):
    """The BAND-WIDE lowering of one `filter`/`calculate`/`group` op, at plan time.

    `ops.LOWERING_TABLE` gives each op two rows; this function is the band-wide row for the
    three ops whose lowering depends only on the SHAPE, exactly as `vector.apply_long_op` is
    the long row for the same three. Which of the two runs for a given declared op is the
    SHAPE dispatch in `RasterGridAnalysis.apply_steps` -- shape picks the implementation,
    never the order.

    Returns `(metadata_lf, raw_band_frame, value_where, has_dimension_conjunct)`;
    `band_ops` and `calculate_plans` are mutated in place. `has_dimension_conjunct` is the
    signal `BaseAnalysis._has_selective_dimension_filter` needs from a `FilterOp`, `False`
    for the ops that cannot produce one -- the same second return value `apply_long_op`
    gives, so an interpreter threads it identically whichever lowering ran.

    A `FilterOp`'s DIMENSION half either NARROWS THE SCAN or becomes a recorded band op, and
    which one is the pushdown rule:

        Before the first `group` (`raw_band_frame is None`), it is applied straight to
        `metadata_lf` (`band_metadata_scan_filter`, the same `lf.sql()` path
        `base.apply_dimension_filter` uses). That frame is still the table `_band_frame`
        hands a pixel-read consumer to build its scan from, so narrowing it here means the
        dropped bands are never opened at all -- real projection pushdown, not a column drop
        after the fact. This is legal only because every op that can precede it in that
        prefix (another dimension filter, a value filter, a calculate) PRESERVES BAND
        IDENTITY: none of them changes which band is which, so evaluating the predicate
        against the scan's metadata gives the same answer as evaluating it in place. The
        moment a `group` runs, that stops being true.

        After a `group`, it is recorded as `("filter", dimension_where)` at its own position
        and applied to the running metadata via `band_metadata_filter` -- the same primitive
        `apply_dimension_filter_band_wide` uses at replay time, so the in-flight narrowing
        and the eventual column drop cannot disagree. Pushing it back to the scan instead
        would be a REORDERING of the chain, not an optimization of a prefix: the predicate
        names the group's output dimensions, and "which raw bands would have satisfied it"
        is a different question with a different answer.

    A `FilterOp`'s VALUE half cannot run in either mode -- there is no pixel column yet --
    so it is recorded as `("filter_value", value_where)` at its declared position, for
    `apply_value_filter_band_wide` to mask once pixels are read, AND-combined onto the
    running `value_where` that comes back (which is what `RasterGridAnalysis.
    _apply_transform_params_impl` validates, and what it rejects a STANDALONE value filter
    from). Recording it at a position is what makes a value filter declared AFTER a
    calculate mask the REWRITTEN value, as declared, instead of the stored one.

    A `CalculateOp` is validated and appended to `calculate_plans`, which the next
    `group`/`filter`/pixel-read consumer flushes into `band_ops` at its own position
    (`flush_calculate_plans`) -- not applied immediately.

    Raises `ExpressionError` (from `band_metadata_scan_filter`/`compile_calculate_entries`)
    or `AnalysisError` (the `count` rejection in `collapse_band_group`); callers translate
    the former at `base.op_error_boundary`, which deliberately passes the latter through.
    """
    if isinstance(op, FilterOp):
        rewritten = rewrite_named_columns(op.where, rename_map)
        dimension_conjuncts, value_conjuncts = classify_where(rewritten)
        has_dimension_conjunct = bool(dimension_conjuncts)
        if has_dimension_conjunct:
            dimension_where = " and ".join(f"({conjunct})" for conjunct in dimension_conjuncts)
            if raw_band_frame is None:
                # Pushdown prefix: narrow the scan itself.
                metadata_lf = band_metadata_scan_filter(metadata_lf, dimension_where)
            else:
                # A `group` already ran, so this filter has to run at its own position
                # instead -- against the grouped bands it names.
                flush_calculate_plans(band_ops, calculate_plans)
                band_ops.append(("filter", dimension_where))
                metadata_lf = band_metadata_filter(metadata_lf.collect(), dimension_where)[0].lazy()
        if value_conjuncts:
            conjunct_where = " and ".join(f"({conjunct})" for conjunct in value_conjuncts)
            flush_calculate_plans(band_ops, calculate_plans)
            band_ops.append(("filter_value", conjunct_where))
            value_where = (
                conjunct_where if value_where is None else f"({value_where}) and ({conjunct_where})"
            )
        return metadata_lf, raw_band_frame, value_where, has_dimension_conjunct

    if isinstance(op, CalculateOp):
        expression = rewrite_named_columns(op.expression, rename_map)
        where = rewrite_named_columns(op.where, rename_map) if op.where is not None else None
        calculate_plans.extend(compile_calculate_entries([(calculate_label(op), expression, where)]))
        return metadata_lf, raw_band_frame, value_where, False

    if isinstance(op, GroupOp):
        metadata_lf, raw_band_frame = collapse_band_group(
            metadata_lf,
            list(op.by),
            op.aggregate,
            raw_band_frame=raw_band_frame,
            band_ops=band_ops,
            calculate_plans=calculate_plans,
        )
        return metadata_lf, raw_band_frame, value_where, False

    # pragma: no cover -- a `JoinOp` never reaches a single-input chain
    raise AnalysisError(f"unsupported op in the raster interpreter: {op.op!r}")  # pragma: no cover


# -- the `project` band-wide -> long lowering (zonal summarize) -----------------------
#
# `ops.LOWERING_TABLE`'s `project` raster-to-vector row, split into the validation half
# (`validate_raster_summarize`, which must run before a crosswalk or a pixel is touched),
# the reduction itself (`apply_zonal_summarize`), and the two synchronous workers each
# aggregate kind hands to `asyncio.to_thread`. None of them reads or writes an instance
# attribute; `RasterGridAnalysis._apply_transform_params_impl` is three lines of option-
# reading over them, and `apply_steps` reaches them through that method for a `ProjectOp`.


def validate_raster_summarize(value_where, *, grid_name, summarize_by, summarize_aggregate, summarize_grid_cell):
    """Everything a zonal summarize can reject before opening a file, returning
    `(has_summarize, summarize_grid_cell_ids, crosswalk_path)` -- `crosswalk_path` is
    `None` when the request declares no summarize at all.

    A `where` VALUE conjunct (a per-pixel value predicate -- see `base.apply_value_filter`
    and `apply_band_wide_op`'s `FilterOp` branch, which records it as a `("filter_value",
    ...)` band op) is only meaningful at the summarize stage: it is
    what `summarize_zone_stats` masks each band's raw pixel value against, before the
    multiplier/group_aggregate collapse, so "count pixels above X" and "sum of pixels
    above X" both fall out of the existing null-skipping zonal aggregates for free. A
    plain, non-summarized raster selection has no zonal aggregate to feed a mask into and
    no per-pixel frame the API exposes to a caller at all (a raster `self.lf` is the
    band-metadata table until the summarize builds one) -- so a `where` value conjunct
    given without `summarize_by`/`summarize_aggregate` is rejected outright here, rather
    than silently ignored.
    """
    if value_where is not None:
        # Parsed here, against the placeholder namespace every value conjunct
        # writes against (`value`, `grid_cell`), before touching the crosswalk file,
        # the band metadata, or any pixel data -- an aggregate-inside-predicate or
        # a whitelist violation is rejected before a single file is opened, not
        # discovered partway through a streaming collect. `summarize_zone_stats`
        # re-parses the same string per selected band (each band's own namespace is
        # exactly {that band, grid_cell} -- see `masked_value_predicate_exprs`), so this
        # call is a validation pass, not the one whose result gets used.
        try:
            parse_value_expr(value_where, ["value", "grid_cell"])
        except ExpressionError as exc:
            raise AnalysisError(str(exc))

    has_summarize, summarize_grid_cell_ids = stages.validate_summarize_options(
        summarize_by, summarize_aggregate, summarize_grid_cell
    )
    if not has_summarize:
        if value_where is not None:
            raise AnalysisError(
                "where's value condition(s) require summarize_by and summarize_aggregate for raster grids"
            )
        return False, summarize_grid_cell_ids, None

    crosswalk_path = config.CROSSWALKS_PATH / f"{grid_name}__{summarize_by}.tif"
    if not crosswalk_path.exists():
        raise AnalysisError(f"No crosswalk from {grid_name} to {summarize_by}")
    return True, summarize_grid_cell_ids, crosswalk_path


async def apply_zonal_summarize(
    df,
    dim_df,
    n_bands,
    *,
    crosswalk_path,
    summarize_by,
    summarize_aggregate,
    summarize_grid_cell_ids,
    calculate,
    value_where,
    band_ops,
):
    """Summarize raster pixels by geographic unit (e.g. hydro compartment) using a
    pre-generated zone-ID crosswalk raster (see `python -m cpra.mp.data.cli crosswalk`),
    returning a LONG LazyFrame in the same shape `vector.apply_summarize` produces
    (dimension columns, a `grid_cell` column holding the destination grid_cell ID, a
    `value` column, and a `geographic_unit` column).

    `df` is the full per-band metadata frame the read scans from; `dim_df`/`n_bands` come
    from `RasterGridAnalysis._get_raster_dimension_frame` and describe the OUTPUT bands,
    aligned row-for-row with the `(n_bands, zone_ids.size)` array `summarize_zone_stats`
    returns. The actual per-zone reduction is that worker's; this function resolves the
    zone list, routes to the right worker, and reshapes the result.
    """
    unique_paths, row_indices, bands = group_rows_by_path(df)

    with rio.open(crosswalk_path) as crosswalk:
        if summarize_grid_cell_ids:
            zone_ids = np.array(sorted(summarize_grid_cell_ids), dtype="int64")
        else:
            dst_ids = crosswalk.tags()["dst_ids"].split(",")
            zone_ids = np.array(sorted(int(dst_id) for dst_id in dst_ids), dtype="int64")

    if summarize_aggregate == "count_by_value":
        # Single RAW band only: a request selecting more than one raw pixel band (e.g.
        # every calendar_year in a range, or several bands pre-collapsed by
        # group_by/group_aggregate into one group) has no single natural (zone, value)
        # shape to count into -- see raster_source.count_by_value's docstring for why
        # this is refused rather than guessed at. Checked against the raw band count
        # from `row_indices` (not `n_bands`, which reflects the post-group_aggregate
        # collapsed count and would wrongly admit a group_by that collapses several raw
        # bands down to one group) -- `group_by`/`group_aggregate`'s own collapse logic
        # is never applied by `raster_source.count_by_value`, so a caller combining it
        # with count_by_value must be rejected outright rather than have it silently
        # ignore every raw band but one. `row_indices` comes from `group_rows_by_path`
        # on the band-metadata frame, which only reads path/band/dimension columns,
        # never pixel data -- so this check is still "before any file [of pixel data] is
        # read", not merely before the summarize starts.
        raw_band_count = sum(len(indices) for indices in row_indices)
        if raw_band_count != 1:
            raise AnalysisError(
                f"count_by_value is only supported for a single raster band per request, got {raw_band_count} -- "
                "narrow the request (e.g. to one calendar_year) and don't combine it with group_by/group_aggregate"
            )
        stages.reject_calculate_with_count_by_value(calculate)

        try:
            result_df = await asyncio.to_thread(
                summarize_count_by_value,
                unique_paths,
                row_indices,
                bands,
                crosswalk_path,
                zone_ids,
                value_where=value_where,
            )
        except ExpressionError as exc:
            raise AnalysisError(str(exc))

        # `dim_df` can legitimately have zero columns (every dimension collapsed
        # away by group_aggregate with no group_by) -- `.row(0)` on a zero-width
        # frame raises, so there is nothing to read off it in that case.
        dim_row = dim_df.row(0, named=True) if dim_df.width else {}
        return (
            result_df.lazy()
            .with_columns(**{col: pl.lit(val) for col, val in dim_row.items()})
            .with_columns(pl.col("grid_cell").cast(pl.Int32), geographic_unit=pl.lit(summarize_by))
        )

    # The plugin path (`scan_rasters`) applies the calculate/group-aggregate
    # pre-collapse via the shared `band_calculate_passes`/
    # `horizontal_aggregate_expr` vocabulary in raster_source.py -- see
    # `summarize_zone_stats` -- so every summarize request, regardless of
    # which parameters are in play, reaches the same single implementation.
    result = await asyncio.to_thread(
        summarize_zone_stats,
        df,
        unique_paths,
        row_indices,
        bands,
        crosswalk_path,
        zone_ids,
        n_bands,
        band_ops=band_ops,
        aggregate=summarize_aggregate,
    )

    # Assemble in wide format (one column per zone) to line up with `result`'s shape,
    # then unpivot to the long-format grid_cell/value contract used everywhere else --
    # matching `vector.apply_summarize`'s join-based output shape.
    dim_cols = dim_df.columns
    result_frame = pl.from_numpy(result, schema=[str(zone_id) for zone_id in zone_ids], orient="row")
    # `dim_df` can legitimately have zero columns (every dimension collapsed away by
    # group_aggregate with no group_by) -- `result_frame` already has the right
    # height (`n_bands` rows) on its own in that case, so there is nothing to
    # hstack onto it, and `hstack` would fail on the height mismatch a zero-column
    # `dim_df` reports (see `_get_raster_dimension_frame`'s docstring).
    wide = dim_df.hstack(result_frame) if dim_cols else result_frame
    return (
        wide.unpivot(index=dim_cols, variable_name="grid_cell", value_name="value")
        .with_columns(pl.col("grid_cell").cast(pl.Int32), geographic_unit=pl.lit(summarize_by))
        .lazy()
    )


def summarize_zone_stats(
    df, unique_paths, row_indices, bands, crosswalk_path, zone_ids, n_bands, *, band_ops, aggregate
):
    """Zonal aggregate via the Polars IO plugin (`sources.scan_rasters`), returning a
    `(n_bands, zone_ids.size)` array.

    The single implementation of the zonal summarize step (see `apply_zonal_summarize`):
    every filter/calculate/group step this request declared arrives as `band_ops` (from
    `RasterGridAnalysis._pixel_read_band_ops`) and is replayed here in DECLARED order by
    `apply_band_ops`, against the scan's own band-wide frame, whatever shape the chain has.

    `scan_rasters` masks nodata to null against each band's own raw pixel value before
    any op sees it (`mask_nodata=True`), so a masked band's null survives a calculate
    pass untouched (its `is_not_null()` guard -- NoData in, NoData out) and drops out of
    a horizontal collapse and the final vertical `guarded_aggregate` alike, with no
    separate accumulation pass needed here. `apply_band_ops` is therefore called with
    `dtype=None`: nodata is already handled, and a "no contributor" group here should
    stay NULL forever (feeding `guarded_aggregate`'s own NaN handling), never be filled
    to a sentinel the way a tile/write output needs.

    A value `where` conjunct is replayed at its own declared position like any other op
    (`_pixel_read_band_ops`), as a per-cell mask. On the flat request surface that
    position is first -- before any calculate entry -- which is what a FILTER value
    predicate means there: it is written against the stored pixel value, not one a
    calculate entry already rewrote. A pixel the predicate rejects then behaves exactly
    like a nodata pixel for everything downstream -- it drops out of the collapse, and
    out of `guarded_aggregate`'s denominator -- which is what turns "count contributing
    pixels" into "count pixels where value > x" with no separate code path. (A CALCULATE
    entry's own value condition, by contrast, tests the running value at that entry's
    position in the sequence, per the sequential contract -- that's
    `band_calculate_passes`' job, not this mask's.)

    Synchronous (called via `asyncio.to_thread`): the streaming `collect()` below is a
    long-running, CPU/IO-bound call, so it's offloaded to a worker thread rather than
    blocking the event loop for the whole summarize, matching how the numpy path's
    blocking `rio_copy` call is offloaded elsewhere (`_write_raster_vrt`).
    """
    lf = scan_rasters(unique_paths, row_indices, bands, crosswalk_path, zone_ids, mask_nodata=True)
    try:
        lf, out_metadata_df = apply_band_ops(lf, df, band_ops, extra_columns=("grid_cell",))
    except ExpressionError as exc:
        raise AnalysisError(str(exc))
    out_names = band_column_names(out_metadata_df)

    grouped = (
        lf.group_by("grid_cell")
        .agg([guarded_aggregate(aggregate, name) for name in out_names])
        .collect(engine="streaming")
    )

    result = np.full((n_bands, zone_ids.size), np.nan)
    if grouped.height:
        positions = np.searchsorted(zone_ids, grouped["grid_cell"].to_numpy().astype("int64"))
        for i, name in enumerate(out_names):
            result[i, positions] = grouped[name].to_numpy()
    return result


def summarize_count_by_value(unique_paths, row_indices, bands, crosswalk_path, zone_ids, *, value_where):
    """count_by_value's raster-side worker: delegates to `raster_source.count_by_value`
    (the single-band scan + per-zone categorical count), on a worker thread the same
    way `summarize_zone_stats` is -- see that function's docstring for why (the
    streaming `collect()` underneath is long-running and CPU/IO-bound, so it must not
    block the event loop for the whole summarize).
    """
    return raster_count_by_value(
        unique_paths,
        row_indices,
        bands,
        crosswalk_path,
        zone_ids,
        value_where=value_where,
        max_distinct=COUNT_BY_VALUE_MAX_DISTINCT,
    )


def _batched_block_windows(block_windows, blocks_per_batch):
    """Yield window unions over `block_windows`, chunked to at most `blocks_per_batch`
    blocks each rather than a whole block row at a time.

    `block_windows` (as returned by a rasterio dataset's own `block_windows(1)`) is grouped
    into block rows first, then each row's blocks are split into batches of
    `blocks_per_batch`. A block row can be tens of thousands of pixels wide regardless of
    how much data a request actually needs, so unioning a whole row per iteration -- the
    original behavior -- makes every iteration's read buffer scale with the raster's width
    and the request's band count rather than with `blocks_per_batch`. See
    `raster_blocks_per_batch` for how that batch size is chosen.
    """
    groups = itertools.groupby(block_windows, key=lambda bw: bw[0][0])
    for _row, group in groups:
        row_windows = [w for _ji, w in group]
        for start in range(0, len(row_windows), blocks_per_batch):
            yield windows.union(*row_windows[start : start + blocks_per_batch])


def _take_window_rows(batch_iter, buffer, needed):
    """Pull exactly `needed` rows, in order, out of `batch_iter` (a `collect_batches`
    iterator of `pl.DataFrame` chunks) plus whatever `buffer` already holds left over
    from the previous window -- the rolling flush `_write_raster_windowed` uses to turn a
    stream of Polars-sized batches back into raster-sized windows.

    `collect_batches`' own chunk boundaries are NOT the raster's block-row boundaries --
    the streaming engine buffers/splits independently of how `scan_rasters`' generator
    happened to yield its rows -- so a window's worth of rows can straddle several
    incoming chunks, or several windows can fit in one chunk. What IS guaranteed (measured,
    and true here by construction: dense mode, no `.filter()`/`.sort()`
    anywhere in this lf -- see the value-filter gate's own comment in
    `_write_raster_windowed`) is that rows arrive in strict raster-raveled pixel order with
    none dropped or duplicated, which is what lets a running "how many rows have I already
    consumed for this window" count stand in for a materialized pixel-index column: pulling
    the next `needed` rows off the front of (leftover buffer + next chunks) always yields
    exactly one window's pixels, in position, and never needs more than one short chunk's
    worth of buffering ahead of the window boundary it's about to satisfy.

    Returns `(rows, new_buffer)`: `rows` has exactly `needed` rows, and `new_buffer` is
    whatever's left (fed back in as `buffer` for the next window; `None` on the very first
    call). Raises `RuntimeError` if `batch_iter` runs out first -- which is the loud
    failure a `.filter()` accidentally reintroduced into this lf would produce (a genuine
    row-count deficit), rather than a silently misaligned grid.
    """
    while buffer is None or buffer.height < needed:
        try:
            chunk = next(batch_iter)
        except StopIteration:
            got = 0 if buffer is None else buffer.height
            raise RuntimeError(
                f"scan_rasters produced only {got} of {needed} rows needed for one output "
                "window -- the dense scan must yield exactly one row per pixel, in raster "
                "order, with none dropped; something in this lf's pipeline dropped rows "
                "(e.g. a `.filter()` reaching the source's predicate pushdown instead of "
                "the value-filter gate's mask)."
            ) from None
        buffer = chunk if buffer is None else pl.concat([buffer, chunk], how="vertical")
    return buffer.slice(0, needed), buffer.slice(needed, buffer.height - needed)


class RasterGridAnalysis(BaseAnalysis):
    def _band_frame(self):
        """The full per-band metadata (`path`/`band`/dimension columns) a pixel-read
        consumer needs to build its raw scan, and the metadata frame its replay starts from:
        `self.materialize()` if no `group` has run yet (in which case `self.lf` still IS
        that table), else `self._raw_band_frame` -- the snapshot the first `GroupOp`
        (`collapse_band_group`) took before `self.lf` moved on to collapsed,
        dimension-only metadata.

        Dimension filters declared BEFORE the first group are already baked into this frame
        (they narrowed the scan itself -- see `apply_band_wide_op`'s pushdown rule), which is why a
        replay starting here and `self.lf`'s own running state stay in agreement: both are
        the same raw frame carried through the same ops.
        """
        return self._raw_band_frame if self._raw_band_frame is not None else self.materialize()

    def _pixel_read_band_ops(self):
        """`self._band_ops` as a pixel-read consumer should replay them: any trailing
        calculate batch flushed into place first (`flush_calculate_plans`), then a
        standalone value `where` folded in (`band_ops_for_pixel_read`).

        Each of the three pixel reads gets its ops from exactly this and nothing else --
        `_collapse_tile_bands` and `_write_raster_windowed` call it themselves, and
        `_apply_transform_params_impl` calls it to hand `apply_zonal_summarize` the
        `band_ops` it forwards to `summarize_zone_stats` -- so they cannot drift apart on
        which ops a request actually has. The flush is repeated per consumer because a
        tile is served without a preceding `group` just as often as with one; it is a
        no-op once the pending batch has already moved.
        """
        flush_calculate_plans(self._band_ops, self._calculate_plans)
        return band_ops_for_pixel_read(self._band_ops, self._value_where)

    def _get_raster_dimension_frame(self, df):
        """Get `(dim_df, n_bands)`: a DataFrame with one row per output raster band, aligned
        by row order with the `(n_bands, zone_ids.size)` array `summarize_zone_stats`
        returns for the same `df`, and that band count as a separate value.

        A dimension column left out of the group by clause is simply absent from `dim_df` --
        mirroring the long-shape group lowering's drop (see `vector.apply_group_by`'s
        docstring for the full rationale).

        `n_bands` is read off the metadata frame's own height BEFORE `BAND_INDEX` is
        dropped, never off `len(dim_df)`: a group_aggregate with no group_by at all
        collapses to zero surviving dimension columns, and Polars cannot represent "zero
        columns, N rows" -- that is the entire reason `BAND_INDEX` exists.
        """
        if self._raw_band_frame is None:
            dim_df = df.select(pl.exclude("path"))
            return dim_df, len(dim_df)
        metadata_df = self.lf.collect()
        return metadata_df.drop(BAND_INDEX), metadata_df.height

    async def _apply_transform_params_impl(self):
        """The transform hook for a raster frame -- a thin caller over the free functions
        `validate_raster_summarize` and `apply_zonal_summarize`, which hold the whole
        `project` band-wide -> long lowering (see their docstrings). What stays here is
        what touches `self`: reading the options, taking the band-metadata frame
        (`_band_frame`) and its output-band description (`_get_raster_dimension_frame`),
        collecting the replay recipe (`_pixel_read_band_ops`), and storing the two results
        back.

        The empty-frame 404 stays here too, between the two calls, because it is about
        this dataset's own selection rather than about the summarize: it fires exactly
        where it used to, after the crosswalk check and before any pixel is read.
        """
        has_summarize, self.summarize_grid_cell_ids, crosswalk_path = validate_raster_summarize(
            self._value_where,
            grid_name=self.grid.name,
            summarize_by=self.options.summarize_by,
            summarize_aggregate=self.options.summarize_aggregate,
            summarize_grid_cell=self.options.summarize_grid_cell,
        )
        if not has_summarize:
            return

        df = self._band_frame()
        if df.is_empty():
            self._error("Dataset does not exist", status_code=404)

        dim_df, n_bands = self._get_raster_dimension_frame(df)
        self.lf = await apply_zonal_summarize(
            df,
            dim_df,
            n_bands,
            crosswalk_path=crosswalk_path,
            summarize_by=self.options.summarize_by,
            summarize_aggregate=self.options.summarize_aggregate,
            summarize_grid_cell_ids=self.summarize_grid_cell_ids,
            calculate=self.options.calculate,
            value_where=self._value_where,
            band_ops=self._pixel_read_band_ops(),
        )

    async def _read_raster_bands(self, df, rasters, row_indices, bands, window, out_shape=None, use_gdal_env=False):
        """Read one window's raw bands into a single ``(len(df), h, w)`` array in df row order.

        ``out_shape``, when given, is the ``(height, width)`` each dataset's window is
        DECIMATED to while reading -- rasterio hands it to GDAL, which serves the read from
        an overview when one matches, so the returned array is ``(len(df), *out_shape)``
        rather than the window's own shape. ``get_tile`` is the caller that uses it (see
        ``utils.tile_read_shape``); with None every read is at the source's full resolution.

        ``rasters`` holds one dataset per unique file backing ``df`` (rows sharing a
        multiband COG are grouped onto the same raster/band-list by ``group_rows_by_path``),
        with ``row_indices``/``bands`` mapping each raster back to the df rows and band
        numbers it supplies. Row order is what lets the caller line the array up with
        ``df``'s own rows, which is exactly what ``band_column_names`` assumes: array plane
        ``i`` becomes band column ``b<i>``, described by ``df`` row ``i``.

        ``use_gdal_env`` is forwarded to ``parallel_read_rasters``/``_read_window``, which enter
        ``raster_gdal_env()`` on the worker thread performing each read rather than here. Default
        ``False`` so callers (e.g. ``get_tile``) that deliberately don't want it keep working
        unchanged; ``_apply_transform_params_impl`` and ``_write_raster_windowed`` pass
        ``True``.
        """
        band_reads = await parallel_read_rasters(rasters, window, out_shape, bands=bands, use_gdal_env=use_gdal_env)
        data = np.empty((len(df), *band_reads[0].shape[1:]), dtype=band_reads[0].dtype)
        for path_row_indices, read in zip(row_indices, band_reads):
            data[path_row_indices] = read
        return data

    # -- ops interpreter ----------------------------------------------------------------

    async def apply_steps(self, ordered_ops: list[Op]):
        """The raster interpreter: walk `ordered_ops` and apply each op IN DECLARED
        ORDER, never reordering -- the raster half of what `vector.VectorGridAnalysis.
        apply_steps` already does for the long shape (see that method's docstring for the
        general argument against a fixed filter-then-calculate-then-group-then-transform
        order).

        No pixel is read here. What this method builds is the pair a pixel-read consumer
        later replays: the band METADATA (`self.lf`, narrowed and collapsed as it walks) and
        `self._band_ops`, the DECLARED ops in declared order. Every entry it records carries
        the caller's own predicate/plans/group key and nothing derived -- see
        `apply_band_ops`, which is the only thing that ever resolves those against real
        columns, and does so freshly at each step.

        Two lowerings, selected by an explicit `shape` variable (`ops.Shape`) that names
        which of the two frame shapes `self.lf` currently holds. `shape` is what picks the
        IMPLEMENTATION of each declared op; it never picks the ORDER, which comes from
        `ordered_ops` alone. It replaces an earlier `current_grid != self.grid.name` test
        that stood in for the same question -- a proxy that agrees with the shape because a
        summarizing `project` is the only thing in this walk that can reshape the frame:
        `shape` starts at `Shape.band_wide` unconditionally and the walk never leaves it any
        other way (`self.options.grid_cell` is a strictly TERMINAL filter now, applied only
        after this whole method's walk has finished -- see below -- so it never reshapes
        anything the walk itself sees).

        `Shape.band_wide` means one row per pixel and one column per band (`b0..bN`), with
        `self.lf` holding that frame's band METADATA -- and it additionally means the frame
        is POSITIONALLY ALIGNED to the raster: row `i` is pixel `i` of the output window,
        for every op in the chain. Shape alone would not say that, and raster output
        correctness depends on it, so every band-wide lowering here preserves row count by
        construction -- a value filter is a per-cell MASK, never a row drop (see
        `apply_value_filter_band_wide`), and `_take_window_rows` turns a short read into a
        hard `RuntimeError` rather than letting a silently-shortened frame reach a write.
        There is no band-wide frame in this package that is NOT raster-aligned (every one
        originates from `sources.scan_rasters`' dense per-pixel scan), so alignment is
        carried as an invariant of the lowerings -- pinned by `tests/api/test_raster_ops.py`'s
        `TestBandWideRowAlignment`, which asserts that each band-wide lowering leaves
        `height` unchanged -- rather than as a second type field whose only reachable value
        would be `True`. That class also records the one exception: a dimension filter
        matching no band leaves a frame with no columns, whose height is necessarily 0. It
        is unreachable as a write, because the same predicate empties the band metadata and
        `_apply_transform_params_impl` 404s on an empty band frame before any pixel is read.

        `Shape.long` means the ordinary (dimension columns, `grid_cell`, `value`) frame, one
        row per observation, with no alignment property at all: rows may be freely dropped.
        A raster chain reaches it exactly one way -- a summarizing `ProjectOp` at its own
        declared position, `project` raster-to-vector in `ops.LOWERING_TABLE`'s terms. Once
        there, every non-`project` op is lowered by the SAME `vector.apply_long_op` the
        vector interpreter calls, with the same arguments.

        - `Shape.band_wide`: `self.lf` is band metadata -- one
          row per selected band, or (once at least one `group` has run) one row per
          surviving group; no pixel data either way.

          Every `FilterOp`/`CalculateOp`/`GroupOp` is handed straight to the free function
          `apply_band_wide_op`, the single implementation of `ops.LOWERING_TABLE`'s
          band-wide row -- see its docstring for the pushdown rule, for why a value conjunct
          is recorded at a position rather than applied, and for what a `group` snapshots.
          Nothing about that lowering is a method: this loop reads the four pieces of
          running state off `self`, hands them over, and stores the results back.

          A `ProjectOp` that summarizes (`op.to != current_grid`) calls
          `self._apply_transform_params_impl()` unchanged, which both performs the transform
          AND flips `self.lf` to a long/tabular shape (it sets `geographic_unit` itself,
          unlike vector's `apply_project`, so no trailing "add geographic_unit" step is
          needed here the way vector's interpreter needs one). An identity `ProjectOp`
          (`op.to == current_grid`) does NOT call `_apply_transform_params_impl()` in place
          -- unlike a summarizing project, an identity one has no transform work to do at
          its own declared position, and this validation must still run exactly once, LAST,
          regardless of where in a pipeline an identity `project` happens to sit. Calling it
          here instead would let a value `where` conjunct declared by a LATER `FilterOp`
          step slip past the "standalone value filter needs summarize_by/summarize_aggregate"
          rejection entirely (nothing would ever re-check it) -- so this branch is a plain
          `continue`, and the trailing `if shape is Shape.band_wide: await
          self._apply_transform_params_impl()` after the whole loop (see below) reproduces
          that always-runs-last check for this shape instead.
        - `Shape.long`: `self.lf` is an ordinary long (grid_cell/value/dimensions) tabular
          frame -- structurally identical to vector's own `self.lf` -- so every
          `FilterOp`/`CalculateOp`/`GroupOp` is handed straight to `vector.apply_long_op`,
          the single shared implementation of `ops.LOWERING_TABLE`'s long row. A value
          conjunct is an unconditional ROW DROP there (nothing downstream is aligned to a
          raster any more), which is also why the band-wide branch cannot reuse it.

          A post-project literal `grid_cell = N` filter is an ordinary long-shape row filter
          here -- one zone's own aggregate does not depend on which OTHER zones were also
          requested, so post-hoc filtering the summarize's OUTPUT rows by `grid_cell` gives
          a numerically identical answer to narrowing the crosswalk's `zone_ids` before the
          summarize runs.

          A further `ProjectOp` from this shape can never be reached: the planner admits at
          most one `ProjectOp` per single-input chain, and `Shape.long` is entered ONLY by
          that loop's own (unique) `ProjectOp` branch above (there is no other entry any
          more -- see `self.options.grid_cell` below), so a second one to reject here is not
          a case this method's own loop can ever actually see.

        `self.options.grid_cell` (`InputSelection.grid_cell`, an INPUT-level sampling filter,
        never a pipeline step, so it never appears in `ordered_ops`) is a terminal, OUTPUT-side
        filter, applied once at the very end of this method against whichever grid `self.lf`
        ends up on -- see the block after the walk, below. It never reshapes anything the
        WALK above sees: `shape` starts and, absent a summarizing `project`, stays
        `Shape.band_wide` for the whole loop regardless of whether `grid_cell` is set, so a
        leading dimension filter still narrows the SCAN via the ordinary band-wide pushdown
        prefix (`apply_band_wide_op`'s pushdown rule) exactly as it would for a plain raster
        selection -- unlike the old up-front resample this replaced, which sampled every
        selected band before any filter could narrow that set. Getting `grid_cell` applied
        needs two different routes depending on whether the chain also summarizes:

        - `self.options.summarize_by` set: the chain's destination IS a real vector grid
          (`self.options.summarize_by` itself), and `self.lf` reaches `Shape.long` through
          the ordinary summarize -- the loop's `ProjectOp` branch for a pipeline, or the
          trailing `if shape is Shape.band_wide` block below for a flat `summarize_by=`
          request. Once there, `grid_cell` is applied terminally, reusing `vector.
          apply_grid_cell_selection` unchanged -- see the block immediately after the walk
          for why that reuse is exact, not just similar.

        - No `self.options.summarize_by`: there is no destination grid for `grid_cell` to
          resolve against other than the raster's own pixels, so `shape` never leaves
          `Shape.band_wide` and the walk above records `self._band_ops`/narrows
          `self.lf`'s band metadata exactly as it would for a plain (no `grid_cell`) raster
          selection -- no pixel read yet, same as always. The terminal block below is what
          finally reads pixels: it samples every SURVIVING band at the requested
          coordinates and replays `self._band_ops` (via `self._pixel_read_band_ops()`)
          against those samples with `apply_band_ops`, the identical replay recipe
          `_collapse_tile_bands`/`_write_raster_windowed`/`apply_zonal_summarize` each use
          for their own pixel reads -- see `sample_band_ops_to_long`'s own docstring.
          `self.grid_cell_ids` is not set by this route (unlike the terminal one above) --
          a sampled coordinate's label is not a cell ID of any grid, so there is nothing to
          report there.

        `rename_map`/`value_names`: identical construction to `VectorGridAnalysis.
        apply_steps` -- see that method's docstring for why a synthetic digit-string default
        output name (`ops._default_output_name`) is deliberately excluded.

        Errors: as in the vector interpreter, every lowering is a plain library function
        raising `ExpressionError`/`FoldError`, translated once by `base.op_error_boundary`
        around the whole walk. A rejection this method makes ITSELF (`self._error`) raises
        `AnalysisError`, which that boundary deliberately passes through untouched.
        """
        shape = Shape.band_wide
        # A `grid_cell` selection with no `summarize_by` (sub-case B, see above) has no
        # destination grid to reach `Shape.long` through, so the walk below never sees it
        # either -- it takes the ordinary band-wide lowerings throughout, exactly like a
        # plain (no `grid_cell`) raster selection, and the terminal block near the end of
        # this method is the only place that condition is ever checked directly.
        grid_cell_without_summarize = bool(self.options.grid_cell) and not self.options.summarize_by

        current_grid = self.grid.name
        # Empty `ordered_ops` (a selection that declared no `where`/`calculate`/`group` at
        # all) walks nothing and renames nothing -- see `VectorGridAnalysis.apply_steps`'
        # own comment on the same line.
        value_names = {op.input for op in ordered_ops[:1]}
        value_names.update(op.output for op in ordered_ops if not op.output[0].isdigit())
        rename_map = {name: "value" for name in value_names}

        with op_error_boundary():
            for op in ordered_ops:
                if isinstance(op, ProjectOp):
                    if op.to == current_grid:
                        continue
                    await self._apply_transform_params_impl()
                    shape = Shape.long
                    current_grid = op.to
                elif shape is Shape.long:
                    if isinstance(op, GroupOp) and op.aggregate == "count":
                        # Same rejection `collapse_band_group` makes for the band-wide shape,
                        # kept here so the answer to "is group_aggregate='count' legal on a raster
                        # selection" does not depend on which shape the frame happens to be in.
                        self._error("group_aggregate 'count' is not supported for raster grids")
                    self.lf, has_dimension_conjunct = apply_long_op(
                        self.lf,
                        op,
                        rename_map=rename_map,
                        has_selective_filter=self._has_selective_dimension_filter(),
                    )
                    self._where_has_dimension_conjunct = (
                        self._where_has_dimension_conjunct or has_dimension_conjunct
                    )
                else:
                    (
                        self.lf,
                        self._raw_band_frame,
                        self._value_where,
                        has_dimension_conjunct,
                    ) = apply_band_wide_op(
                        op,
                        self.lf,
                        raw_band_frame=self._raw_band_frame,
                        band_ops=self._band_ops,
                        calculate_plans=self._calculate_plans,
                        value_where=self._value_where,
                        rename_map=rename_map,
                    )
                    self._where_has_dimension_conjunct = self._where_has_dimension_conjunct or has_dimension_conjunct

        if shape is Shape.band_wide and not grid_cell_without_summarize:
            # No `ProjectOp` in this chain ever actually summarized (an identity project,
            # possibly followed by filter/calculate/group steps that ran against the
            # still-metadata-shaped `self.lf` above) -- `_apply_transform_params_impl` still
            # needs to run once, HERE, in what was always its fixed position (last) in the
            # old engine, regardless of where an identity `project` sits in DECLARED order:
            # it is what rejects a standalone value `where` conjunct given without
            # `summarize_by`/`summarize_aggregate` ("a plain, non-summarized raster selection
            # has no ... per-pixel frame ... at all" -- see that method's own docstring), and
            # every raster request that never summarizes -- flat or piped through this
            # interpreter alike -- must keep rejecting that shape, not just the ones where a
            # value filter happens to be declared before the (identity) project. A genuinely
            # summarizing `ProjectOp` above already called this once, at its own declared
            # position, and moved `shape` to `Shape.long` when it did -- so this is skipped
            # for that case, never a second summarize.
            #
            # Also skipped for `grid_cell_without_summarize` (sub-case B): that is a
            # deliberate admission, not an oversight, of the exact same shape this call
            # would otherwise reject -- a standalone value `where` with no `summarize_by`/
            # `summarize_aggregate`. The rejection exists because a plain raster selection
            # exposes no per-pixel frame for a value predicate to test; a `grid_cell`
            # selection is precisely the case that DOES, once the terminal block below
            # samples it, so a value filter recorded into `self._band_ops` here is replayed
            # against real samples there instead of being rejected as premature.
            await self._apply_transform_params_impl()

        if self.options.grid_cell and self.options.summarize_by:
            # Terminal `grid_cell`, for a chain whose destination grid is a summarized
            # vector zone (a summarizing `ProjectOp` above, or the flat `summarize_by`
            # spelling of the same thing -- either way `self.options.summarize_by` names
            # it). By this point `self.lf` is a genuine long (dimension columns,
            # `grid_cell`, `value`) frame on that destination grid -- structurally
            # identical to vector's own long frame at the same point -- so this reuses
            # `VectorGridAnalysis.apply_steps`' own terminal lowering unchanged:
            # `apply_grid_cell_selection` already accepts both a destination cell ID and a
            # lat/lng pair resolved via point-in-polygon against whichever grid it's given.
            self.lf, self.grid_cell_ids = await apply_grid_cell_selection(
                self.lf, self.options.summarize_by, self.options.grid_cell
            )
        elif grid_cell_without_summarize:
            # Terminal `grid_cell`, sub-case B: no destination vector grid, so this is a
            # genuine raster-to-vector point sample instead of a filter. `self.lf` is still
            # the band metadata the walk above narrowed/collapsed (never a real pixel read
            # yet); `sample_band_ops_to_long` is the pixel-read consumer that finally reads
            # one, replaying `self._band_ops` (via `self._pixel_read_band_ops()`) against
            # samples taken at the requested coordinates rather than against a raster
            # window -- see that function's own docstring for the replay/row-drop
            # contract. `self.grid_cell_ids` is deliberately left unset (see this method's
            # own docstring, in the `self.options.grid_cell` section above).
            self.lf = await sample_band_ops_to_long(
                self._band_frame(), self._pixel_read_band_ops(), self.options.grid_cell
            )

    def _collapse_tile_bands(self, df, data, nodata):
        """Apply a tile's declared filter/calculate/group steps with Polars expressions.

        Shares one operator vocabulary with the write and summarize paths (`apply_band_ops`)
        rather than reducing the array with NumPy, which is what makes tiles treat nodata
        the way the rest of the API does: masked against each band's own raw pixel value
        before any calculate entry runs, dropped from a group_aggregate, and restored as the
        sentinel only where no band contributed anything. The NumPy reduce it replaces did
        none of that -- a sentinel was scaled by any calculate multiplier and averaged in as
        if it were a real value, so one nodata band in a group pulled the whole rendered
        pixel toward -9999.

        In and out stay array plumbing: `data[i].ravel()` is a view of the read buffer, and
        each output column converts straight back into the `(bands, h, w)` shape
        `Legend.colorize_pixels` expects.

        `nodata=nodata` is passed UNCONDITIONALLY, unlike `_write_raster_windowed`'s
        conditional `nodata=nodata if band_ops else None` -- a deliberate, pre-existing
        divergence between the tile and write paths (see that method's own comment on it),
        preserved here exactly, not normalized: a written file can carry the sentinel forward
        for its reader to interpret, but a rendered pixel has to commit to a color, so a tile
        always masks nodata whether or not this request also aggregates/calculates/filters.
        """
        frame = pl.DataFrame({name: data[i].ravel() for i, name in enumerate(band_column_names(df))})
        try:
            frame, out_metadata_df = apply_band_ops(
                frame, df, self._pixel_read_band_ops(), dtype=data.dtype.name, nodata=nodata
            )
        except ExpressionError as exc:
            self._error(str(exc))
        return np.stack(
            [frame[name].to_numpy().reshape(data.shape[1:]) for name in band_column_names(out_metadata_df)]
        )

    @asynctiming
    async def get_tile(self, x, y, z, format, tile_size=256, **kwargs):
        """Get a map tile image."""
        tile_bounds = BoundingBox(*mercantile.xy_bounds(x, y, z))
        if self.dataset_bounds and disjoint_bounds(self.dataset_bounds, tile_bounds):
            return None

        df = self._band_frame()
        unique_paths, row_indices, bands = group_rows_by_path(df)
        with ExitStack() as stack:
            inputs = [stack.enter_context(rio.open(path)) for path in unique_paths]

            if not self.dataset_bounds:
                self.dataset_bounds = BoundingBox(
                    *transform_bounds(inputs[0].crs, "EPSG:3857", *inputs[0].bounds)
                )
                if disjoint_bounds(self.dataset_bounds, tile_bounds):
                    return None

            # All inputs backing one request share the same variable and were produced by the
            # same conversion pipeline, so they share one nodata sentinel; read it from the
            # file rather than hardcoding it, matching the write and summarize paths. A
            # source declaring none still needs *some* value here, because GDAL has to put
            # something in the pixels of a tile that reach past the source's extent, and
            # those pixels genuinely are absent data -- so -9999 stands in for that case and
            # is masked below like any other sentinel. (Hardcoding -9999 on both sides, as
            # this used to, silently mismatches a source that declares anything else: its
            # real nodata would go unmasked while the fill value became a distinct second
            # sentinel.)
            nodata = inputs[0].nodata if inputs[0].nodata is not None else -9999

            # Read the tile's footprint out of the sources in their OWN CRS, collapse the
            # request's bands over those pixels, and warp the collapsed result onto the
            # tile grid -- rather than warping each source first and collapsing on the tile.
            # Reprojection is a rendering step, not something the per-pixel band algebra
            # needs to know about, and keeping it last is what lets that algebra be the
            # same `apply_band_ops` call the write and summarize paths make.
            #
            # `Resampling.nearest` is what makes the two orders equivalent: every tile pixel
            # takes one whole source pixel's value, and the collapse is per-pixel across
            # bands, so collapsing before or after the warp reads the same numbers.
            # `tests/api/test_tile_warp.py` measures that equivalence against the warp-first
            # order directly, pixel by pixel.
            #
            # One window serves every input: all the files backing one request are the same
            # variable on the same grid, written by the same conversion pipeline, so they
            # are co-registered -- the same assumption `_write_raster_windowed` already
            # makes when it derives its block windows from one reference source.
            window = tile_source_window(inputs[0], tile_bounds)
            if window is None:
                # The tile's footprint misses the source's pixels entirely, even though it
                # overlapped the (rectangular, reprojected) dataset bounds tested above.
                # That is a rendered tile of nothing, not a missing tile.
                tile_data = np.full((1, tile_size, tile_size), nodata, dtype=inputs[0].dtypes[0])
            else:
                # A low-zoom tile's footprint is most of the raster -- 434 MB per band at
                # z=6 on morph_pixel -- for the 65,536 pixels it renders. `tile_read_shape`
                # decimates that read to about the tile's own resolution, which GDAL serves
                # from the source's overviews; `warp_window_to_tile` takes the same shape so
                # the georeferencing it derives describes the decimated array rather than
                # the window. None (a window already at tile resolution, z >~ 12) reads and
                # warps exactly as it did before.
                read_shape = tile_read_shape(window, tile_size)
                data = await self._read_raster_bands(
                    df, inputs, row_indices, bands, window, out_shape=read_shape
                )
                tile_data = warp_window_to_tile(
                    self._collapse_tile_bands(df, data, nodata),
                    window=window,
                    src_transform=inputs[0].transform,
                    src_crs=inputs[0].crs,
                    tile_bounds=tile_bounds,
                    tile_size=tile_size,
                    nodata=nodata,
                    out_shape=read_shape,
                )
            color = self.legend.colorize_pixels(tile_data)
            # JPEG has no alpha channel, so drop it. fromarray infers RGB vs RGBA from the
            # array's shape, and passing mode= explicitly is deprecated as of Pillow 11.
            image = Image.fromarray(color[:, :, :3] if format == "jpg" else color)

            with BytesIO() as memfile:
                image.save(memfile, format=("JPEG" if format == "jpg" else "PNG"))
                return memfile.getvalue()

    @asynctiming
    async def write_raster(self, out_path, **kwargs):
        """Write the selected bands to a raster file.

        Wide selections are streamed by GDAL from a band-separate VRT
        (``_write_raster_vrt``); narrow ones read windows into NumPy
        (``_write_raster_windowed``).  The two are within noise of each other on time, so the
        choice is made on memory -- see ``_windowed_read_is_lighter``.

        Aggregated selections always take the windowed path. GDAL's builtin ``mean``/``sum``/
        ``min``/``max`` pixel functions can express the reduction and are more accurate (they
        accumulate in float64 and can propagate nodata, neither of which the NumPy reduce
        does), but collapsing bands onto one ``VRTDerivedRasterBand`` makes GDAL re-read
        every contributing band per output chunk -- for one PIXEL-interleaved multiband
        source that decompresses each tile once per band. Measured reducing 52 elev bands:
        983s vs 80s, at every swath size tried. The windowed path reads each tile once.

        Selections with calculate entries also always take the windowed path: a general
        expression (or a per-pixel value condition) has no VRT spelling, and the windowed
        path is also what enforces the calculate stage's NoData contract (mask the
        sentinel before any entry runs, restore it after -- see ``apply_band_ops``),
        which a VRT ScaleRatio could not.

        A standalone value ``where`` (``self._value_where``) forces the windowed path for
        the same reason: the value-filter gate is implemented as a mask inside ``_write_raster_windowed``'s own
        ``scan_rasters``-backed pipeline, and the VRT path has no comparable hook at all --
        routing a value-filtered selection there would silently drop the filter rather
        than apply or reject it. No production caller reaches this branch today (a
        standalone value ``where`` on a raster grid is rejected earlier, in
        ``_apply_transform_params_impl``, unless ``summarize_by``/``summarize_aggregate``
        are also given); this is a defensive floor for any future or direct caller of this
        method, not a behavior change for an existing one.
        """

        df = self._band_frame()
        if not len(df):
            return False

        if (
            self._band_ops
            or self._calculate_plans
            or self._value_where is not None
            or not RASTER_USE_VRT_COPY
            or self._windowed_read_is_lighter(df)
        ):
            return await self._write_raster_windowed(df, out_path, **kwargs)
        return await self._write_raster_vrt(df, out_path, **kwargs)

    def _windowed_read_is_lighter(self, df):
        """Whether the windowed path would use less memory than the VRT copy for ``df``.

        The windowed path holds one ``(bands, block_height, width)`` buffer per window plus
        the read that fills it, so its peak grows with the number of selected bands; the VRT
        copy's peak is roughly flat, set by the swath and cache caps.  Measured on 52-band
        float32 elev (peak RSS, windowed vs VRT): 1 band 1027 vs 1425 MB, 5 bands 1470 vs
        1329 MB, 52 bands 7983 vs 2529 MB -- so the crossover is around 4-5 bands here, and
        expressing it as a buffer size rather than a band count keeps it right for grids and
        dtypes of other sizes.
        """

        with rio.open(df["path"][0]) as source:
            block_height = source.block_shapes[0][0]
            buffer_bytes = len(df) * block_height * source.width * np.dtype(source.dtypes[0]).itemsize
        return buffer_bytes <= RASTER_WINDOW_BUFFER_LIMIT_MB * 1024 * 1024

    def _calendar_column(self, df):
        """The calendar dimension present in ``df``, if any."""

        return next((col for col in ("calendar_year", "calendar_day") if col in df.columns), None)

    async def _write_raster_vrt(self, df, out_path, **kwargs):
        """Stream the selected bands into ``out_path`` via a band-separate VRT.

        The VRT names each selected ``(path, band)`` as one output band, so GDAL reads each
        source once and never materializes the stack -- see ``raster_copy_gdal_env`` for the
        swath/cache settings that make this pay off.  Band descriptions are applied to the
        VRT XML rather than to pixels in Python. Only plain selections reach this path --
        ``write_raster`` routes any request with calculate entries to the windowed path,
        which can express a general elementwise rewrite and its NoData contract; a VRT
        ScaleRatio cannot.
        """

        sources = list(zip(df["path"].to_list(), df["band"].to_list()))
        vrt_text = vrt(sources)

        # Label output bands with their calendar value, as the windowed path does. Set
        # explicitly rather than inherited so legacy sources (no on-disk description) are
        # labeled too -- see set_vrt_band_descriptions. Skipped when aggregating, since an
        # aggregated band spans several calendar values.
        calendar_col = self._calendar_column(df)
        if calendar_col is not None:
            descriptions = [f"{calendar_col}={value}" for value in df[calendar_col].to_list()]
            vrt_text = set_vrt_band_descriptions(vrt_text, descriptions)

        with rio.open(sources[0][0]) as source:
            profile = source.profile
        creation_options = {
            "driver": "GTiff",
            "tiled": profile.get("tiled", True),
            "blockxsize": profile.get("blockxsize"),
            "blockysize": profile.get("blockysize"),
            "interleave": OUTPUT_INTERLEAVE,
            "num_threads": RASTER_NUM_THREADS,
            "bigtiff": "YES",
            **kwargs,
        }
        creation_options = {k: v for k, v in creation_options.items() if v is not None}

        # The VRT is a throwaway sidecar whose SimpleSource entries hold absolute paths, so
        # its location doesn't matter -- keep it out of the output directory.
        with tempfile.TemporaryDirectory() as tmp_dir:
            vrt_path = Path(tmp_dir) / "download.vrt"
            vrt_path.write_text(vrt_text)

            # CreateCopy is one long blocking call, unlike the windowed path's per-window
            # awaits, so it runs in the default executor to keep the event loop free for
            # other requests. GDAL config is process-global, so the env entered here
            # applies inside the worker thread.
            loop = asyncio.get_running_loop()
            with raster_copy_gdal_env():
                await loop.run_in_executor(
                    None,
                    partial(rio_copy, str(vrt_path), str(out_path), **creation_options),
                )

        return True

    async def _write_raster_windowed(self, df, out_path, **kwargs):
        """Write a raster dataset to a file using windowed reading and writing to conserve
        memory.

        Reading is `raster_source.scan_rasters` (dense mode: no `zones_path`, so nothing
        is dropped and no `grid_cell` column is attached) driven by `collect_batches`,
        instead of a hand-managed `rasterio` window loop, onto the same Polars IO-plugin
        plumbing the tile and summarize paths already use
        (`_read_raster_bands`'s VRT/array path stays separate; only THIS writer moved).
        Every filter/calculate/group step this request declared is replayed by
        `apply_band_ops`, built once outside the window loop, so this writer uses the same
        operator vocabulary as the tile and summarize paths instead of a second
        implementation of it.

        Output geometry -- which pixels make up "one window" -- is still decided the same
        way as before the port: `_batched_block_windows(inputs[0].block_windows(1),
        blocks_per_batch)`, precomputed once into `window_list` before any row is
        consumed. What changed is where the pixels for each window COME from: rather than
        reading that exact window directly off disk, `_take_window_rows` pulls exactly
        `window.height * window.width` rows, in order, off the front of `lf.
        collect_batches(...)`'s output (buffering at most one short chunk's worth ahead of
        each boundary -- see that function's docstring for why one incomplete "block row"
        of buffer is enough: batches arrive in strict, gapless raster-pixel
        order). Reading and writing stay array plumbing at the boundary: one window's
        worth of raw bands in, one `output.write` per window out.

        The other two measured memory knobs are unaffected by this port, not made
        redundant by it: `raster_blocks_per_batch` still bounds `scan_rasters`' OWN per-batch read
        buffer (now applied inside that function, against the same
        `config.RASTER_BATCH_BUFFER_MB` budget, rather than only at this call site), and
        `raster_gdal_env`'s `GDAL_CACHEMAX` cap still applies -- `scan_rasters` enters it
        itself around each window's read (see that function's docstring on GDAL's thread-
        locality constraint across a generator `yield`), so this method no longer needs to
        enter it around a read loop of its own; it still enters it once, below, around
        opening `inputs`/`output`, for the GDAL_NUM_THREADS effect that only takes hold at
        open time.
        """

        unique_paths, row_indices, bands = group_rows_by_path(df)
        band_ops = self._pixel_read_band_ops()
        grouped_output = any(kind == "group" for kind, *_ in band_ops)

        with ExitStack() as stack:
            # GDAL_NUM_THREADS' decompression speedup only takes effect for a dataset that
            # was *opened* with it in effect (measured on real data -- see
            # _apply_transform_params_impl); entering it here, before the sources (and the
            # output, for the same reason on the write side) are opened, is what makes it
            # apply to them. Scoped narrowly to just the opens, not the whole write below, so
            # this doesn't hold a process-global GDAL config open (and leaking into unrelated
            # concurrent requests -- see raster_gdal_env's docstring) for the entire write.
            with raster_gdal_env():
                inputs = [stack.enter_context(rio.open(path)) for path in unique_paths]
                profile = inputs[0].profile.copy()
                out_dtype = profile["dtype"]
                nodata = inputs[0].nodata

                # `mask_nodata=False`: nodata handling is `apply_band_ops`' own job below
                # (conditioned on whether this request declared any op at all, which
                # `scan_rasters`' unconditional mask knows nothing about) -- masking here too
                # would be redundant at best and would break the plain-passthrough case at
                # worst (see the `nodata=` argument below).
                #
                # Built ONCE, outside the window loop -- the expressions are identical for
                # every window, so there is no reason to reconstruct them per iteration. The
                # scan itself is lazy, so building it inside this `raster_gdal_env()` block
                # opens nothing; it is here only to keep the lazy frame and the profile it is
                # sized against next to each other.
                #
                # `dtype=` -- which is what turns on the leading nodata mask and the
                # trailing sentinel refill -- is passed whenever this request declared ANY
                # band op: for a group, so a nodata pixel drops out of the collapse instead
                # of contributing a sentinel to it; for a calculate, so an entry's rewrite
                # never touches the sentinel (NoData in, NoData out -- masked to null before
                # the entries run and restored afterwards); and for a value filter, for the
                # same reason -- a masked-out pixel needs to come back as the sentinel, not a
                # Polars null a numpy int array can't hold. A plain, non-calculated,
                # non-filtered selection passes every pixel through untouched -- see the
                # pinned test_non_aggregated_output_preserves_nodata. This is where the write
                # path deliberately differs from get_tile, which masks either way: a written
                # file can carry the sentinel forward for its reader to interpret, but a
                # rendered pixel has to commit to a color.
                lf = scan_rasters(unique_paths, row_indices, bands, mask_nodata=False)
                try:
                    lf, out_metadata_df = apply_band_ops(
                        lf, df, band_ops, dtype=out_dtype if band_ops else None, nodata=nodata
                    )
                except ExpressionError as exc:
                    self._error(str(exc))
                out_names = band_column_names(out_metadata_df)

                # Each selected (path, band) row -- or each aggregated group -- becomes one
                # output band, so the source's own (single-band) band count must be
                # overridden with what the op chain actually left behind.
                profile.update(count=len(out_names), interleave=OUTPUT_INTERLEAVE, **kwargs)
                output = stack.enter_context(rio.open(out_path, "w", **profile))

                # Output window geometry, precomputed once, up front -- unchanged from
                # before the port. This list, not collect_batches' own chunk boundaries,
                # is what decides how many pixels make up "one window" for `output.write`.
                block_height, block_width = inputs[0].block_shapes[0]
                itemsize = np.dtype(inputs[0].dtypes[0]).itemsize
                blocks_per_batch = raster_blocks_per_batch(len(df), block_height, block_width, itemsize)
                window_list = list(_batched_block_windows(inputs[0].block_windows(1), blocks_per_batch))

            # Label output bands with their calendar value when rows map 1:1 to bands.
            if not grouped_output:
                calendar_col = self._calendar_column(df)
                if calendar_col is not None:
                    for band_idx, value in enumerate(df[calendar_col].to_list(), start=1):
                        output.set_band_description(band_idx, f"{calendar_col}={value}")

            # Sized off the same budget `blocks_per_batch` used for the output window
            # list, so the streaming engine's own buffering targets roughly one window's
            # worth of rows per chunk -- not load-bearing for correctness (`_take_window_rows`
            # tolerates any chunking), only for keeping this port's memory profile close to
            # the pre-port per-window granularity.
            chunk_size = blocks_per_batch * block_height * block_width
            batch_iter = iter(lf.collect_batches(chunk_size=chunk_size, maintain_order=True))
            buffer = None

            for window in window_list:
                needed = window.height * window.width
                result, buffer = _take_window_rows(batch_iter, buffer, needed)

                # Assemble every output band and write ONCE. Writing band-at-a-time into a
                # PIXEL-interleaved tiled COG makes GDAL read-modify-write every tile once
                # per band it touches -- measured on real data: 27.8s and a 301 MB output
                # for a band-at-a-time write of one window versus 3.6s and 22.9 MB for a
                # single write of the same window's full band stack. That cost is a property
                # of the sink (GDAL's per-block path for a PIXEL-interleaved dataset), not of
                # Polars, but it is the trap a per-band port of this loop would fall straight
                # into.
                block = np.stack(
                    [
                        result[name].to_numpy().astype(out_dtype, copy=False).reshape((window.height, window.width))
                        for name in out_names
                    ]
                )
                output.write(block, window=window)
                del result, block

        return True
