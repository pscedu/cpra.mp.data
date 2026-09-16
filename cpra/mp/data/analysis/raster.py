"""`RasterGridAnalysis`: the raster half of the former `api/dataset.py`'s `Dataset` class.

See `analysis/base.py`'s module docstring for the staged `apply_params` flow and the hook
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
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from cpra.mp.data import config
from cpra.mp.data.analysis.base import BaseAnalysis, apply_dimension_filter, apply_value_filter
from cpra.mp.data.analysis.expressions import (
    ExpressionError,
    classify_where,
    compile_calculate_entries,
    parse_value_expr,
    rewrite_named_columns,
)
from cpra.mp.data.analysis.geography import parse_coordinate_pairs, project_coordinates
from cpra.mp.data.analysis.ops import CalculateOp, FilterOp, GroupOp, Op, ProjectOp
from cpra.mp.data.analysis.sources import (
    NUMPY_TO_POLARS,
    band_calculate_passes,
    band_collapse_exprs,
    guarded_aggregate,
    horizontal_aggregate_expr,
    masked_value_predicate_exprs,
    scan_rasters,
)
from cpra.mp.data.analysis.sources import (
    count_by_value as raster_count_by_value,
)
from cpra.mp.data.analysis.utils import (
    asynctiming,
    group_rows_by_path,
    parallel_read_rasters,
    parallel_sample_rasters,
    raster_blocks_per_batch,
    raster_copy_gdal_env,
    raster_gdal_env,
    unpack_values,
)
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
# Re-entrant functions -- each takes its own `lf`/arguments and returns a new `lf` instead
# of reading `self.lf`/`self.options` -- that `RasterGridAnalysis.apply_steps`'s interpreter
# and its three pixel-read consumers (`_summarize_zone_stats_plugin`,
# `_collapse_tile_bands`, `_write_raster_windowed`) use to apply a request's
# filter/calculate/group steps IN DECLARED ORDER against a real, per-op band-wide frame,
# rather than the position a fixed five-stage engine would always run them at.
#
# Every function here operates on a "band-wide frame": one row per pixel, one column per
# selected raw band (`b<i>`, `sources.scan_rasters`' own naming), optionally with a
# `grid_cell` zone-id column attached (`scan_rasters`' widened `zones_path`-only mode --
# see that function's docstring). `band_metadata_df`, where a function needs it, is the
# one-row-per-band table (dimension columns only, no pixels) the columns were built
# against at the time that op was declared -- `RasterGridAnalysis._band_ops` freezes a
# copy for exactly this reason (see that attribute's own docstring in `base.py`), since
# `self.lf` itself may have moved on (further groups/filters) by the time a pixel-read
# consumer actually replays these ops.
#
# `apply_value_filter_band_wide`/`band_dimension_keep` are used directly by
# `_write_raster_windowed`/`_band_applicability`. `apply_dimension_filter_band_wide`,
# `apply_calculate_band_wide`, and `band_group_row_idxs` are used by
# `RasterGridAnalysis._collapse_group` (the group case) and `apply_band_ops` (the
# calculate/filter cases, for a request whose `_band_ops` holds more than the single
# calculate-then-one-group shape `sources.band_collapse_exprs` was built for -- see
# `RasterGridAnalysis._band_ops_is_simple`). `apply_group_band_wide` is used by the
# SIMPLE-shape branch of each of the three pixel-read consumers, in place of a direct
# `band_collapse_exprs` call, once a calculate has already run via
# `apply_calculate_band_wide` -- both together reproduce `band_collapse_exprs`'s own
# nodata-mask-then-calculate-then-collapse order exactly, for the one shape where that
# fixed order is also the declared one.
def apply_value_filter_band_wide(lf, band_names, value_where, extra_columns=()):
    """Band-wide lowering of a VALUE `FilterOp` conjunct (`ops.LOWERING_TABLE`'s
    `("filter (value)", Shape.band_wide)` row): a per-cell MASK via `sources.
    masked_value_predicate_exprs`, never a row filter/drop. Dropping the ROW would drop
    that pixel from every OTHER band column too, and a raster/tile-output query must
    return the same pixel count it read (pinned mechanically by
    `tests/api/test_raster_source.py::TestValueFilterPixelCountPreservation`, this
    function's own prerequisite). Reuses `masked_value_predicate_exprs` unchanged --
    the identical expression `RasterGridAnalysis._summarize_zone_stats_plugin` already
    applies against a ZONED scan; here it runs against ANY band-wide frame (dense or
    zone-tagged), at whatever position this `FilterOp` is declared, rather than always
    being pre-collapsed into the summarize stage.

    A falsy `value_where` (`None`/`""`) is a no-op, matching `apply_value_filter`'s own
    (vector) guard.

    This is also the value-filter gate: `_write_raster_windowed` (below) is the one
    place a raster-output query can carry a standalone value `where` (`self._value_where`)
    all the way to its `scan_rasters`-backed `lf`, and it calls THIS function, never
    `lf.filter(...)`, to apply it -- the source plugin cannot know which shape it is
    scanning for, so the compiler has to decide. See that method's docstring for the ordering argument
    and `tests/api/test_write_raster.py::TestValueFilterGate` for the pixel-count/position
    -preservation assertion wired to that real path, not just this primitive (which
    `tests/api/test_raster_ops.py::TestApplyValueFilterBandWide` already covers).
    """
    if not value_where:
        return lf
    return lf.with_columns(masked_value_predicate_exprs(band_names, value_where, extra_columns=extra_columns))


def band_dimension_keep(band_metadata_df, dimension_where):
    """Which of `band_metadata_df`'s rows (one row per raw band column, in the SAME
    order the band-wide frame's `b<i>` columns were built in) satisfy a `FilterOp`'s
    DIMENSION conjunct -- mirrors `RasterGridAnalysis._band_applicability`'s per-
    calculate-plan dimension evaluation, generalized here to a standalone predicate
    string so both the dimension-`FilterOp` lowering (`apply_dimension_filter_band_wide`,
    below) and the calculate lowering (`apply_calculate_band_wide`) can share one
    "evaluate this dimension condition against the band-metadata table" primitive
    instead of two.

    A falsy `dimension_where` keeps every band (mirrors `_band_applicability`'s "no
    dimension `where` at all means the entry applies to every band"). Raises
    `ExpressionError` if `dimension_where` cannot be evaluated against
    `band_metadata_df`'s columns -- the same failure `_band_applicability` reports via
    `self._error`, surfaced here as a plain exception for a caller with no `self` to
    call it on.
    """
    if not dimension_where:
        return [True] * band_metadata_df.height
    try:
        mask = (
            band_metadata_df.lazy()
            .select(pl.sql_expr(dimension_where).fill_null(False))
            .collect()
            .to_series()
        )
    except Exception as exc:
        raise ExpressionError(
            f"where {dimension_where!r} cannot be evaluated against this dataset's bands: {exc}"
        ) from exc
    return mask.to_list()


def apply_dimension_filter_band_wide(lf, band_metadata_df, band_names, dimension_where, extra_columns=()):
    """Band-wide lowering of a DIMENSION `FilterOp` conjunct (`ops.LOWERING_TABLE`'s
    `("filter (dimension)", Shape.band_wide)` row): a COLUMN SELECTION -- projection
    pushdown -- never a row filter. A dimension lives in column IDENTITY in band-wide
    shape (which raw band is which calendar step, model_group_id, ...), not in a per-
    pixel cell, so "keeping" a dimension condition means keeping the whole columns it
    selects and dropping the rest entirely -- the same narrowing `scan_rasters`'
    `with_columns` parameter provides, here expressed as an ordinary `.select()` on an
    already-scanned frame instead of the scan's own projection-pushdown parameter, since
    this op may run after other ops have already added/removed columns.

    Returns `(lf, kept_names)`: the narrowed frame plus the surviving `band_names`, in
    their original relative order, for a later op (`calculate`/`group`/`project`) to
    keep applying itself only to bands that are still present. `extra_columns` (e.g. a
    `grid_cell` column already attached by an earlier `project`) are always kept,
    exactly like `masked_value_predicate_exprs`'s identically-named parameter.

    A falsy `dimension_where` is a no-op: returns `(lf, list(band_names))` unchanged.
    """
    if not dimension_where:
        return lf, list(band_names)
    keep_mask = band_dimension_keep(band_metadata_df, dimension_where)
    kept = [name for name, keep in zip(band_names, keep_mask) if keep]
    return lf.select([*extra_columns, *kept]), kept


def apply_calculate_band_wide(lf, band_metadata_df, band_names, plans):
    """Band-wide lowering of `CalculateOp` (`ops.LOWERING_TABLE`'s `("calculate",
    Shape.band_wide)` row): `with_columns` across `band_names`, reusing `sources.
    band_calculate_passes` UNCHANGED -- the same shared calculate vocabulary
    `RasterGridAnalysis._write_raster_windowed`/`_collapse_tile_bands`/
    `_summarize_zone_stats_plugin` already apply, rather than a second implementation of
    the same vocabulary. Each plan's per-band applicability (`band_dimension_keep`,
    above) is computed here from `band_metadata_df`, mirroring
    `RasterGridAnalysis._simple_band_collapse_inputs` -- this function is that method's
    re-entrant core, generalized to explicit arguments instead of `self._band_ops`/
    `self._band_applicability`, for a `CalculateOp` to invoke at its own declared
    position instead of every calculate entry being pre-collapsed into one batch ahead
    of the (old engine's) fixed transform stage.

    `plans` are `expressions.CalculatePlan`s (`expressions.compile_calculate_entries`'s
    output) applied as sequential passes -- entry N+1 sees entry N's own rewrite,
    exactly `band_calculate_passes`' documented contract. An empty `plans` is a no-op.
    """
    if not plans:
        return lf
    applicability = [band_dimension_keep(band_metadata_df, plan.dimension_where) for plan in plans]
    for pass_exprs in band_calculate_passes(band_names, plans, applicability):
        lf = lf.with_columns(pass_exprs)
    return lf


def band_group_row_idxs(band_metadata_df, by):
    """Which raw band-column positions (df row order in `band_metadata_df`, the same
    order a band-wide frame's `b<i>` columns are built in) belong to each output
    `GroupOp` group, keyed by `by` -- the band-wide grouping bookkeeping
    `RasterGridAnalysis._collapse_group` already performs against `self.lf`'s band-
    metadata table, pulled out here as its own re-entrant function (mirroring
    `vector.py`'s `apply_group_by`/`apply_project`) so `apply_group_band_wide` (below)
    can call it directly instead of going through `self._band_groups`.

    An empty `by` means "collapse every row into a single group" -- matching
    `_collapse_group`'s own `group_by()` with no columns, which collapses everything
    into one group rather than the pre-cleanup `group_by(True)` sentinel it replaced.
    """
    if not by:
        return [list(range(band_metadata_df.height))]
    grouped = band_metadata_df.with_row_index("row_idxs").group_by(*by).agg(pl.col("row_idxs"))
    return grouped["row_idxs"].to_list()


def apply_group_band_wide(lf, band_names, group_row_idxs, aggregate, dtype, nodata=None):
    """Band-wide lowering of `GroupOp` (`ops.LOWERING_TABLE`'s `("group", Shape.
    band_wide)` row): a HORIZONTAL reduce across `band_names`, collapsing each group in
    `group_row_idxs` down to one output column -- reusing `sources.band_collapse_exprs`'s
    existing vocabulary rather than writing a second one, exactly the same expressions
    `RasterGridAnalysis._write_raster_windowed`/`_collapse_tile_bands`/
    `_summarize_zone_stats_plugin` already share via that function.

    Passes `calculate_passes=[]` to `band_collapse_exprs`: a `CalculateOp` ordered
    before this `GroupOp` already ran as its own op (`apply_calculate_band_wide`,
    above) -- `band_collapse_exprs`'s `calculate_passes` parameter exists to let the OLD
    (fixed-stage) engine pre-collapse calculate and group_aggregate into one call; this
    caller instead applies each op once, so there is nothing left for this call to also
    apply.

    Returns `(lf, out_names)`: `lf` narrowed to exactly the collapsed output columns
    (`band_collapse_exprs`'s own final `select`, which is why nothing but the group
    outputs survives -- any other column, e.g. a `grid_cell` attached by an earlier
    `project`, must be re-joined by the caller if it's still needed) plus the synthetic
    names (`out0`, `out1`, ...) `exprs` aliased them to.

    Rejects `aggregate="count"` up front, raising `ExpressionError` with the identical
    message `RasterGridAnalysis._collapse_group` already raises for it (via
    `self._error`, there translated to a 400 `AnalysisError`) -- `band_collapse_exprs`
    has no `count` entry in `HORIZONTAL_RASTER_AGGREGATES` (see `_collapse_group`'s own
    comment: "how many of these bands contributed to this pixel" has no natural
    per-pixel scalar meaning the way max/mean/min/sum do), so calling this function
    with it would otherwise surface as a bare `KeyError` from deep inside
    `band_collapse_exprs` rather than a caller-facing error. `_collapse_group`'s check
    already runs first, independently, on every live request path today, so this is a
    second, redundant guard for whichever caller invokes this function directly, not a
    behavior change for an existing one.
    """
    if aggregate == "count":
        raise ExpressionError("group_aggregate 'count' is not supported for raster grids")
    out_names, passes, exprs = band_collapse_exprs(band_names, [], group_row_idxs, aggregate, dtype, nodata=nodata)
    for pass_exprs in passes:
        lf = lf.with_columns(pass_exprs)
    return lf.select(exprs), out_names


def apply_band_ops(lf, raw_names, band_ops, *, extra_columns=(), dtype=None, nodata=None):
    """Replay `band_ops` (`RasterGridAnalysis._band_ops`: an ordered list of
    `("calc", band_metadata_df, plans)` / `("filter", band_metadata_df,
    dimension_where)` / `("group", group_row_idxs, aggregate)` entries, one per
    `CalculateOp`/post-group `FilterOp`/`GroupOp` actually declared, in DECLARED order)
    against `lf`, whose columns start as `raw_names` -- the general path a pixel-read
    consumer falls back to whenever `RasterGridAnalysis._band_ops_is_simple()` is False
    (a second `group`, a `calculate` declared after a `group`, or a post-group dimension
    `filter`), none of which `sources.band_collapse_exprs`'s fixed calculate-then-one-
    collapse order can represent correctly.

    `dtype`/`nodata` given (a tile or write caller) means: mask `raw_names` against
    `nodata` before the first entry replays (mirroring `band_collapse_exprs`'s own
    leading nodata pass -- nodata interpretation is not itself a declared op, so this
    always runs first regardless of `band_ops`' contents), and cast+fill every surviving
    null in the FINAL columns back to `nodata` (or 0, absent one) once every entry has
    replayed, so the caller gets a column it can hand straight to NumPy. Left `None` (a
    summarize caller, whose `scan_rasters` call already masked nodata via
    `mask_nodata=True` and wants a genuine null propagated into the vertical
    `guarded_aggregate` rather than a fabricated value) means neither happens: a `group`
    entry's own "no contributor" case stays NULL all the way through, which is what lets
    a LATER entry in the same chain (another `group`, or a `calculate`'s `is_not_null()`
    guard) correctly treat it as "nothing here" instead of a real zero or sentinel that
    would otherwise corrupt that later step's own aggregate.

    Returns `(lf, current_names)`: `current_names` is `raw_names` unless at least one
    `group` entry ran (then that group's synthetic `out<k>` names) or a `filter` entry
    narrowed it (a subset of whatever names were current then).
    """
    if dtype is not None and nodata is not None:
        lf = lf.with_columns(
            [pl.when(pl.col(name) == nodata).then(None).otherwise(pl.col(name)).alias(name) for name in raw_names]
        )

    current_names = list(raw_names)
    for kind, *rest in band_ops:
        if kind == "calc":
            metadata_df, plans = rest
            lf = apply_calculate_band_wide(lf, metadata_df, current_names, plans)
        elif kind == "filter":
            metadata_df, dimension_where = rest
            lf, current_names = apply_dimension_filter_band_wide(
                lf, metadata_df, current_names, dimension_where, extra_columns=extra_columns
            )
        elif kind == "group":
            group_row_idxs, aggregate = rest
            out_names = [f"out{k}" for k in range(len(group_row_idxs))]
            exprs = [
                horizontal_aggregate_expr(
                    aggregate, [pl.col(current_names[i]) for i in row_idxs], pl.lit(None)
                ).alias(name)
                for name, row_idxs in zip(out_names, group_row_idxs)
            ]
            lf = lf.with_columns(exprs).select([*extra_columns, *out_names])
            current_names = out_names
        else:  # pragma: no cover -- `_band_ops` only ever holds the three kinds above
            raise ValueError(f"unknown band op kind {kind!r}")

    if dtype is not None:
        fill = pl.lit(nodata if nodata is not None else 0).cast(NUMPY_TO_POLARS[dtype])
        lf = lf.with_columns(
            [pl.col(name).cast(NUMPY_TO_POLARS[dtype]).fill_null(fill).alias(name) for name in current_names]
        )
    return lf, current_names


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
    async def _apply_grid_cell(self):
        """Apply the grid_cell filter to a raster-grid dataset."""
        lngs, lats = parse_coordinate_pairs(unpack_values(self.options.grid_cell, "grid_cell"))

        # Transform the coordinates to the correct CRS, and sample the rasters in parallel.
        # Rows sharing the same path (bands of the same multiband COG) are grouped so each
        # file is opened and sampled once, requesting all of its needed bands together.
        df = self.materialize()
        unique_paths, row_indices, bands = group_rows_by_path(df)
        with ExitStack() as stack:
            datasets = [stack.enter_context(rio.open(path)) for path in unique_paths]
            projected_pairs = list(zip(*project_coordinates(lngs, lats, datasets[0].crs)))
            group_samples = await parallel_sample_rasters(datasets, projected_pairs, bands=bands)
            dtype = datasets[0].dtypes[0]

        # Add the resulting values to the LazyFrame as one column per sampled coordinate,
        # then unpivot into long format (grid_cell/value) to match the long-format contract
        # used everywhere else in the API. Each grid_cell here is a "lng,lat" coordinate
        # pair label rather than an integer grid ID, since these are arbitrary sample points.
        columns = [np.empty(len(df), dtype=dtype) for _ in lngs]
        for path_row_indices, path_samples in zip(row_indices, group_samples):
            for column, values in zip(columns, path_samples):
                column[path_row_indices] = values
        column_labels = [f"{x!s},{y!s}" for x, y in zip(lngs, lats)]
        wide_df = df.with_columns(**dict(zip(column_labels, columns))).select(cs.exclude("path", "band"))
        index_cols = [col for col in wide_df.columns if col not in column_labels]
        self.lf = wide_df.unpivot(
            index=index_cols, on=column_labels, variable_name="grid_cell", value_name="value"
        ).lazy()

    def _collapse_group(self, group_by_cols):
        """Collapse the CURRENT band-metadata state (`self.lf`: either the untouched raw
        per-band frame, on this request's first `group`, or an earlier `group`/post-group
        dimension `filter`'s own collapsed output) into one row per `group_by_cols`
        combination.

        On the FIRST call, snapshots `self.lf` into `self._raw_band_frame` before
        touching anything -- every pixel-read consumer needs the original (path, band)
        rows to build its scan, even once `self.lf` itself has moved on to dimension-only
        metadata. Any calculate entries accumulated so far (`self._calculate_plans`) are
        flushed into `self._band_ops` (`_flush_calculate_plans_to_band_ops`) BEFORE this
        group's own `("group", ...)` entry is appended, so a later pixel-read replay
        applies them at the right position relative to this group rather than always
        before every group regardless of where they were actually declared (see
        `apply_band_ops`).

        `group_row_idxs` (`band_group_row_idxs`, against the CURRENT `self.lf`) are
        positions into `self.lf`'s CURRENT rows -- correct for one group, but this method
        is also what makes a SECOND (or later) `group_by` compose instead of discarding
        the first: `self._band_groups` re-expresses each new group's row positions in
        terms of the ORIGINAL `self._raw_band_frame` rows by looking up the previous
        `self._band_groups` entry for each position, rather than being overwritten.
        `self.lf` is then reassigned to one row per surviving group (taken from the
        first raw row index in each group, so the two stay aligned by construction
        instead of via a second, independently-ordered `group_by`).
        """
        if self.options.group_aggregate == "count":
            # group_aggregate is the raster band-COLLAPSE step
            # (`raster_source.HORIZONTAL_RASTER_AGGREGATES`), reducing ACROSS one
            # pixel's selected bands -- unlike the zonal summarize aggregate this
            # option shares its name and values with, "how many of these bands
            # contributed to this pixel" has no natural per-pixel scalar meaning the
            # way max/mean/min/sum do (see the `AggregateFunction.count` comment in
            # types.py). Rejected explicitly here rather than left to fail as a raw
            # `KeyError` out of `HORIZONTAL_RASTER_AGGREGATES` deep in a later stage.
            self._error("group_aggregate 'count' is not supported for raster grids")

        metadata_df = self.lf.collect()
        previous_band_groups = self._band_groups
        if previous_band_groups is None:
            self._raw_band_frame = metadata_df

        self._flush_calculate_plans_to_band_ops(metadata_df)

        group_row_idxs = band_group_row_idxs(metadata_df, group_by_cols)
        self._band_ops.append(("group", group_row_idxs, self.options.group_aggregate))

        self._band_groups = [
            list(idxs)
            if previous_band_groups is None
            else [orig for p in idxs for orig in previous_band_groups[p]]
            for idxs in group_row_idxs
        ]

        first_idxs = [idxs[0] for idxs in group_row_idxs]
        self.lf = metadata_df[first_idxs].select(group_by_cols).lazy()

    def _flush_calculate_plans_to_band_ops(self, metadata_df):
        """Move whatever `self._calculate_plans` has accumulated since the last flush
        into `self._band_ops`, frozen against `metadata_df` -- the band-metadata state
        those plans' dimension `where`s must be evaluated against (`apply_band_ops`
        replays a `("calc", metadata_df, plans)` entry via `apply_calculate_band_wide`).

        Called by `_collapse_group` and by a post-group dimension `FilterOp`
        (`apply_steps`) before each appends its own `_band_ops` entry, and once more by
        each pixel-read consumer for a trailing calculate declared after the last
        group/filter with nothing following it -- so a calculate's applicability is
        always computed against the metadata state it was actually declared against,
        regardless of how many groups/filters surround it. A no-op once
        `self._calculate_plans` is empty (already flushed, or never populated), so it is
        safe to call more than once for the same pending batch.
        """
        if not self._calculate_plans:
            return
        self._band_ops.append(("calc", metadata_df, list(self._calculate_plans)))
        self._calculate_plans = []

    def _band_ops_is_simple(self):
        """Whether `self._band_ops` is the one shape whose collapse
        `_summarize_zone_stats_plugin` can build inline: a `group`, optionally preceded
        by a single calculate batch and nothing else. That is the only shape the inline
        form -- calculate passes, then one `horizontal_aggregate_expr` per group -- is
        correct for; a second `group`, a calculate declared AFTER a group, or a
        post-group dimension `filter` each break it, and so does a calculate with no
        group at all (there is no collapse to build, so there is nothing for the inline
        form to do). Everything else goes through `apply_band_ops`'s general,
        order-respecting replay.

        The write and tile paths don't make this choice: `sources.band_collapse_exprs`
        is the only band vocabulary they speak, so they read
        `_simple_band_collapse_inputs` directly and rely on the planner rejecting every
        shape it can't express for a raster input before it gets here.
        """
        kinds = [entry[0] for entry in self._band_ops]
        return kinds in ([], ["group"], ["calc", "group"])

    def _simple_band_collapse_inputs(self, raw_names):
        """`(calculate_passes, group_row_idxs)` for `_band_ops_is_simple()`'s shape, read
        from `self._band_ops` (already flushed there by `_collapse_group`/the caller's
        own trailing flush) instead of `self._calculate_plans`/a separately-tracked group
        result -- this is just unpacking the one calculate batch and the one group
        `_band_ops_is_simple()` guarantees are the only entries present, not a second
        computation.
        """
        calculate_passes = []
        group_row_idxs = None
        for kind, *rest in self._band_ops:
            if kind == "calc":
                metadata_df, plans = rest
                applicability = [self._band_applicability(metadata_df, plan) for plan in plans]
                calculate_passes = band_calculate_passes(raw_names, plans, applicability)
            elif kind == "group":
                group_row_idxs, _aggregate = rest
        return calculate_passes, group_row_idxs

    def _current_metadata_df(self, df):
        """The band-metadata frame a calculate/filter declared right now would apply
        against: `df` (the caller's own already-materialized frame -- `self.lf` itself,
        untouched) if no `group` has run yet, else `self.lf`'s CURRENT (already
        collapsed and/or post-group-filtered) state.
        """
        return df if self._band_groups is None else self.lf.collect()

    def _band_frame(self):
        """The full per-band metadata (`path`/`band`/dimension columns) a pixel-read
        consumer needs to build its raw scan: `self.materialize()` if no `group` has run
        yet (in which case `self.lf` still IS that table), else `self._raw_band_frame`
        -- the snapshot `_collapse_group` took before `self.lf` moved on to the
        collapsed (dimension-only) metadata a later dimension filter or second `group`
        needs instead.
        """
        return self._raw_band_frame if self._band_groups is not None else self.materialize()

    def _get_group_row_idxs(self):
        """The row-idx-per-group list, as plain Python lists of ORIGINAL
        `self._raw_band_frame` row positions -- `self._band_groups`, already maintained
        eagerly by `_collapse_group` (and by a post-group dimension `filter`; see
        `apply_steps`) rather than lazily collected, so this is a plain attribute read,
        not a query. Only ever called when `self._band_groups is not None`.
        """
        return self._band_groups

    def _band_applicability(self, df, plan):
        """Which of `df`'s band-metadata rows a calculate plan's dimension `where`
        selects, as a list of booleans in df row order.

        The dimension half of a calculate `where` picks BANDS (e.g.
        `model_group_id=500`), so it is evaluated here against the metadata rows'
        dimension columns, never against pixels; the value half is per-pixel and is
        folded in later by `band_calculate_passes`. No dimension `where` at all means
        the entry applies to every band.

        Delegates to the module-level `band_dimension_keep` -- the
        same per-band dimension evaluation `apply_calculate_band_wide` uses for a
        `CalculateOp`'s own applicability -- rather than duplicating the `pl.sql_expr`/
        `fill_null(False)` evaluation inline; this method is now that function's
        `self`-bound wrapper, adding only the `plan.label` prefix onto its
        `ExpressionError` so an existing pinned error string
        (`test_where_naming_a_dimension_this_dataset_lacks_is_400`) is unchanged.
        """
        try:
            return band_dimension_keep(df, plan.dimension_where)
        except ExpressionError as exc:
            self._error(f"{plan.label}: {exc}")

    def _get_raster_dimension_frame(self, df):
        """Get `(dim_df, n_bands)`: a DataFrame with one row per output raster band,
        aligned by row order with the `(n_bands, zone_ids.size)` array
        `_summarize_zone_stats_plugin` returns for the same `df`, and that band count as
        a separate value.

        A dimension column left out of the group by clause is simply absent from
        `dim_df` -- mirroring `VectorGridAnalysis._collapse_group`'s drop (see its
        comment for the full rationale: the old behavior null-filled every collapsed
        dimension back in, which is the artifact this drops). `n_bands` comes from
        `self._band_groups` (a plain Python list, not a frame's own row count) rather
        than `len(dim_df)`, because Polars cannot represent "zero columns, N rows": a
        group_aggregate with no group_by at all collapses to zero surviving dimension
        columns, and selecting a DataFrame down to zero columns always reports height 0
        regardless of the source frame's real row count.
        """
        if self._band_groups is None:
            dim_df = df.select(pl.exclude("path"))
            return dim_df, len(dim_df)
        return self.lf.collect(), len(self._band_groups)

    async def _apply_transform_params_impl(self):
        """Apply transform and extract parameters to the image.

        Summarizes raster pixels by geographic unit (e.g. hydro compartment) using a
        pre-generated zone-ID crosswalk raster (see
        `python -m cpra.mp.data.cli crosswalk`), converting the result into a tabular
        LazyFrame in the same shape produced by `VectorGridAnalysis`'s transform hook
        (dimension columns, a `grid_cell` column holding the destination grid_cell ID, a
        `value` column, and a `geographic_unit` column).

        The actual per-zone reduction is computed by `_summarize_zone_stats_plugin`,
        returning a `(n_bands, zone_ids.size)` array.

        A `where` VALUE conjunct (a per-pixel value predicate -- see
        `_apply_filter_params`/`self._value_where`) is only meaningful here, at the
        summarize stage: it is what `_summarize_zone_stats_plugin` masks each band's raw
        pixel value against, before the multiplier/group_aggregate collapse, so "count
        pixels above X" and "sum of pixels above X" both fall out of the existing
        null-skipping zonal aggregates for free. A plain, non-summarized raster
        selection has no zonal aggregate to feed a mask into and no per-pixel frame the
        API exposes to a caller at all (`self.lf` is the band-metadata table until this
        method builds one) -- so a `where` value conjunct given without `summarize_by`/
        `summarize_aggregate` is rejected outright below, rather than silently ignored.
        """
        if self._value_where is not None:
            # Parsed here, against the placeholder namespace every value conjunct
            # writes against (`value`, `grid_cell`), before touching the crosswalk file,
            # `self.materialize()`, or any pixel data -- an aggregate-inside-predicate or
            # a whitelist violation is rejected before a single file is opened, not
            # discovered partway through a streaming collect. `_summarize_zone_stats_plugin`
            # re-parses the same string per selected band (each band's own namespace is
            # exactly {that band, grid_cell} -- see `masked_value_predicate_exprs`), so this
            # call is a validation pass, not the one whose result gets used.
            try:
                parse_value_expr(self._value_where, ["value", "grid_cell"])
            except ExpressionError as exc:
                self._error(str(exc))

        if not self._validate_summarize_params():
            if self._value_where is not None:
                self._error("where's value condition(s) require summarize_by and summarize_aggregate for raster grids")
            return

        crosswalk_name = f"{self.grid.name}__{self.options.summarize_by}.tif"
        crosswalk_path = config.CROSSWALKS_PATH / crosswalk_name
        if not crosswalk_path.exists():
            msg = f"No crosswalk from {self.grid.name} to {self.options.summarize_by}"
            self._error(msg)

        df = self._band_frame()
        if df.is_empty():
            self._error("Dataset does not exist", status_code=404)

        dim_df, n_bands = self._get_raster_dimension_frame(df)
        unique_paths, row_indices, bands = group_rows_by_path(df)

        with rio.open(crosswalk_path) as crosswalk:
            if self.summarize_grid_cell_ids:
                zone_ids = np.array(sorted(self.summarize_grid_cell_ids), dtype="int64")
            else:
                dst_ids = crosswalk.tags()["dst_ids"].split(",")
                zone_ids = np.array(sorted(int(dst_id) for dst_id in dst_ids), dtype="int64")

        if self.options.summarize_aggregate == "count_by_value":
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
            # on `self._band_frame()`, which only reads the band-metadata table (path/
            # band/dimensions), never pixel data -- so this check is still "before any
            # file [of pixel data] is read", not merely before the summarize starts.
            raw_band_count = sum(len(indices) for indices in row_indices)
            if raw_band_count != 1:
                self._error(
                    f"count_by_value is only supported for a single raster band per request, got {raw_band_count} -- "
                    "narrow the request (e.g. to one calendar_year) and don't combine it with group_by/group_aggregate"
                )
            self._reject_calculate_with_count_by_value()

            try:
                result_df = await asyncio.to_thread(
                    self._summarize_count_by_value_plugin, unique_paths, row_indices, bands, crosswalk_path, zone_ids
                )
            except ExpressionError as exc:
                self._error(str(exc))

            # `dim_df` can legitimately have zero columns (every dimension collapsed
            # away by group_aggregate with no group_by) -- `.row(0)` on a zero-width
            # frame raises, so there is nothing to read off it in that case.
            dim_row = dim_df.row(0, named=True) if dim_df.width else {}
            self.lf = (
                result_df.lazy()
                .with_columns(**{col: pl.lit(val) for col, val in dim_row.items()})
                .with_columns(
                    pl.col("grid_cell").cast(pl.Int32), geographic_unit=pl.lit(self.options.summarize_by)
                )
            )
            return

        # The plugin path (`scan_rasters`) applies the calculate/group-aggregate
        # pre-collapse via the shared `band_calculate_passes`/
        # `horizontal_aggregate_expr` vocabulary in raster_source.py -- see
        # `_summarize_zone_stats_plugin` -- so every summarize request, regardless of
        # which parameters are in play, reaches the same single implementation.
        result = await asyncio.to_thread(
            self._summarize_zone_stats_plugin,
            df,
            unique_paths,
            row_indices,
            bands,
            crosswalk_path,
            zone_ids,
            n_bands,
        )

        # Assemble in wide format (one column per zone) to line up with `result`'s shape,
        # then unpivot to the long-format grid_cell/value contract used everywhere else --
        # matching VectorGridAnalysis's join-based output shape.
        dim_cols = dim_df.columns
        result_frame = pl.from_numpy(result, schema=[str(zone_id) for zone_id in zone_ids], orient="row")
        # `dim_df` can legitimately have zero columns (every dimension collapsed away by
        # group_aggregate with no group_by) -- `result_frame` already has the right
        # height (`n_bands` rows) on its own in that case, so there is nothing to
        # hstack onto it, and `hstack` would fail on the height mismatch a zero-column
        # `dim_df` reports (see `_get_raster_dimension_frame`'s docstring).
        wide = dim_df.hstack(result_frame) if dim_cols else result_frame
        self.lf = (
            wide.unpivot(index=dim_cols, variable_name="grid_cell", value_name="value")
            .with_columns(pl.col("grid_cell").cast(pl.Int32), geographic_unit=pl.lit(self.options.summarize_by))
            .lazy()
        )

    def _summarize_zone_stats_plugin(self, df, unique_paths, row_indices, bands, crosswalk_path, zone_ids, n_bands):
        """Zonal aggregate via the Polars IO plugin (`raster_source.scan_rasters`).

        The single implementation of the zonal summarize step (see
        `_apply_transform_params_impl`): the calculate/group_aggregate steps are applied
        here in DECLARED order -- via `apply_band_ops`'s general replay whenever
        `self._band_ops` holds more than one group, a calculate after a group, or a
        post-group filter (`_band_ops_is_simple()` is False), else via the same direct
        `band_calculate_passes`/`horizontal_aggregate_expr` calls this method has always
        made for that (overwhelmingly common) shape. `scan_rasters` masks nodata to null
        against each band's own raw pixel value before either sees it (`mask_nodata=True`),
        so a masked band's null survives a calculate pass untouched (its `is_not_null()`
        guard -- NoData in, NoData out) and drops out of a horizontal collapse and the
        final vertical `guarded_aggregate` alike, with no separate accumulation pass
        needed here -- `apply_band_ops` is called with `dtype=None` for exactly this
        reason: nodata is already handled, and a "no contributor" group here should stay
        NULL forever (feeding `guarded_aggregate`'s own NaN handling), never be filled to
        a sentinel the way a tile/write output needs.

        `where`'s value conjunct (`self._value_where`), when set, is masked in the same
        way and at the same point as nodata -- against each band's own raw pixel value,
        before any calculate entry -- via `raster_source.masked_value_predicate_exprs`.
        That ordering matters for the same reason the nodata mask has to run first: a
        FILTER value predicate is written against the stored pixel value, not a rewritten
        one (the staged flow runs filter before calculate), so masking after the
        calculate passes would test the wrong number. A pixel the predicate rejects then
        behaves exactly like a nodata pixel for everything downstream -- it drops out of
        the collapse, and out of `guarded_aggregate`'s denominator -- which is what turns
        "count contributing pixels" into "count pixels where value > x" with no separate
        code path. (A CALCULATE entry's own value condition, by contrast, tests the
        running value at that entry's position in the sequence, per the sequential
        contract -- that's `band_calculate_passes`' job, not this mask's.)

        Synchronous (called via `asyncio.to_thread`): the streaming `collect()` below is a
        long-running, CPU/IO-bound call, so it's offloaded to a worker thread rather than
        blocking the event loop for the whole summarize, matching how the numpy path's
        blocking `rio_copy` call is offloaded elsewhere (`_write_raster_vrt`).
        """
        raw_names = [f"b{i}" for i in range(len(df))]
        lf = scan_rasters(unique_paths, row_indices, bands, crosswalk_path, zone_ids, mask_nodata=True)

        if self._value_where is not None:
            lf = lf.with_columns(
                masked_value_predicate_exprs(raw_names, self._value_where, extra_columns=("grid_cell",))
            )

        self._flush_calculate_plans_to_band_ops(self._current_metadata_df(df))

        if self._band_ops_is_simple():
            calculate_passes, group_row_idxs = self._simple_band_collapse_inputs(raw_names)
            for pass_exprs in calculate_passes:
                lf = lf.with_columns(pass_exprs)
            if group_row_idxs is not None:
                out_names = [f"g{k}" for k in range(len(group_row_idxs))]
                group_exprs = [
                    horizontal_aggregate_expr(
                        self.options.group_aggregate,
                        [pl.col(raw_names[i]) for i in row_idxs],
                        pl.lit(None),
                    ).alias(name)
                    for name, row_idxs in zip(out_names, group_row_idxs)
                ]
                lf = lf.with_columns(group_exprs)
            else:
                out_names = raw_names
        else:
            lf, out_names = apply_band_ops(lf, raw_names, self._band_ops, extra_columns=("grid_cell",))

        grouped = (
            lf.group_by("grid_cell")
            .agg([guarded_aggregate(self.options.summarize_aggregate, name) for name in out_names])
            .collect(engine="streaming")
        )

        result = np.full((n_bands, zone_ids.size), np.nan)
        if grouped.height:
            positions = np.searchsorted(zone_ids, grouped["grid_cell"].to_numpy().astype("int64"))
            for i, name in enumerate(out_names):
                result[i, positions] = grouped[name].to_numpy()
        return result

    def _summarize_count_by_value_plugin(self, unique_paths, row_indices, bands, crosswalk_path, zone_ids):
        """count_by_value's raster-side worker: delegates to `raster_source.count_by_value`
        (the single-band scan + per-zone categorical count), on a worker thread the same
        way `_summarize_zone_stats_plugin` is -- see that method's docstring for why (the
        streaming `collect()` underneath is long-running and CPU/IO-bound, so it must not
        block the event loop for the whole summarize).
        """
        return raster_count_by_value(
            unique_paths,
            row_indices,
            bands,
            crosswalk_path,
            zone_ids,
            value_where=self._value_where,
            max_distinct=COUNT_BY_VALUE_MAX_DISTINCT,
        )

    async def _read_raster_bands(self, df, rasters, row_indices, bands, window, out_shape=None, use_gdal_env=False):
        """Read one window's raw bands into a single ``(len(df), h, w)`` array in df row order.

        ``rasters`` holds one dataset per unique file backing ``df`` (rows sharing a
        multiband COG are grouped onto the same raster/band-list by ``group_rows_by_path``),
        with ``row_indices``/``bands`` mapping each raster back to the df rows and band
        numbers it supplies. Row order is what lets the caller line the array up with
        ``df``'s own rows (e.g. each calculate plan's per-band applicability) and with
        ``_band_groups``' row positions.

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

    async def apply_params(self):
        """Dispatch to the ops interpreter (`apply_steps`) when `self._pipeline_ops` was
        attached -- mirrors `VectorGridAnalysis.apply_params` exactly; see that method's
        docstring for the shared reasoning (this override changes nothing for the flat,
        non-pipeline request surface, which never sets `_pipeline_ops`). `planner.
        _compile_single_input_chain`'s fast path attaches `_pipeline_ops` for a raster
        input too, as of this step, except when `selection.grid_cell is not None` (see
        `apply_steps`' own docstring for why that one shape is excluded) -- so a raster
        pipeline request with a `grid_cell` still falls through to `BaseAnalysis.
        apply_params`'s fixed five-stage flow below, unchanged.
        """
        if self._pipeline_ops is not None:
            await self.apply_steps(self._pipeline_ops)
            self._apply_sort_params()
            return
        await super().apply_params()

    async def apply_steps(self, ordered_ops: list[Op]):
        """The band-wide interpreter: walk `ordered_ops` and apply each op IN DECLARED
        ORDER, never reordering -- the raster half of what `vector.VectorGridAnalysis.
        apply_steps` already does for the long shape (see that method's docstring for
        the general argument against a fixed filter-then-calculate-then-group-then-
        transform order).

        `planner._compile_single_input_chain` currently only ever ADMITS a raster chain
        shaped `(filter|calculate)* project (filter|calculate)* group?` -- it still
        rejects a `group` followed by another `filter`/`calculate`/`group` for a raster
        input (`is_vector_input` gates those relaxations to vector chains only) -- so on
        every request that actually reaches this method through the planner, DECLARED
        order and this method's own bookkeeping happen to coincide with what a fixed
        five-stage engine would have done anyway. This method does not depend on that,
        though: each op is applied through the SAME re-entrant bookkeeping regardless of
        what came before it (`self._collapse_group`, `self._band_ops`, `apply_band_ops`),
        so a caller driving this method directly with a chain the planner does not yet
        admit -- a second `group`, a `calculate` after a `group`, or a `filter` after a
        `group` -- still gets the declared-order answer, not a silently wrong one.

        Two modes, split on `current_grid`, exactly mirroring the two shapes `self.lf`
        actually takes across a raster request's lifetime:

        - PRE-project (`current_grid == self.grid.name`): `self.lf` is still band
          metadata (one row per selected band, or -- once at least one `group` has run,
          see `_collapse_group` -- one row per surviving group; no pixel data either
          way). A `FilterOp`'s DIMENSION half, before any `group`, is an ordinary row
          drop against that table (`apply_dimension_filter`, the same call
          `_apply_filter_params` makes); its VALUE half cannot run yet (no pixel column
          exists), so it is AND-combined onto `self._value_where` for
          `apply_value_filter_band_wide` to mask once pixels are read -- combined the
          same way `_compile_single_input_chain` itself folds several filter steps'
          value conjuncts into one `Options.where` string. AFTER at least one `group`
          (`self._band_groups is not None`), a `FilterOp`'s dimension half instead
          narrows `self.lf` in lockstep with `self._band_groups`/`self._band_ops` (see
          below) rather than reassigning it directly the way the pre-group case does --
          reassigning directly here would silently desynchronize `self._band_groups`
          from `self.lf`'s new rows, corrupting every later group/pixel-read that reads
          either. A value half after a `group` is rejected outright (`self._error`): it
          would need to mask the ALREADY-COLLAPSED band values, a shape `apply_band_ops`
          does not yet replay a value-filter entry for. A `CalculateOp` is validated and
          appended to `self._calculate_plans`, which the next `group`/post-group
          `filter`/pixel-read consumer flushes into `self._band_ops` at its own position
          (`_flush_calculate_plans_to_band_ops`) -- not applied immediately, regardless
          of which engine populated it. A `GroupOp` calls `self._collapse_group(op.by)`
          directly -- the exact hook `_apply_group_params` calls, at this op's own
          declared position instead of always third. A `ProjectOp` that summarizes
          (`op.to != current_grid`) calls `self._apply_transform_params_impl()` unchanged,
          which both performs the transform AND flips `self.lf` to a long/tabular shape
          (it sets `geographic_unit` itself, unlike vector's `apply_project`, so no
          trailing "add geographic_unit" step is needed here the way vector's interpreter
          needs one). An identity `ProjectOp` (`op.to == current_grid`) does NOT call
          `_apply_transform_params_impl()` in place -- unlike a summarizing project, an
          identity one has no transform work to do at its own declared position, and this
          validation must still run exactly once, LAST, regardless of where in a pipeline
          an identity `project` happens to sit. Calling it here instead would let a value
          `where` conjunct declared by a LATER `FilterOp` step slip past the "standalone
          value filter needs summarize_by/summarize_aggregate" rejection entirely
          (nothing would ever re-check it) -- so this branch is a plain `continue`, and
          the trailing `if current_grid == self.grid.name: await
          self._apply_transform_params_impl()` after the whole loop (see below)
          reproduces that always-runs-last check for this shape instead.
        - POST-project (`current_grid != self.grid.name`, i.e. after a summarizing
          `project` ran): `self.lf` is now an ordinary long-shape (grid_cell/value/
          dimensions) tabular frame -- structurally identical to vector's own `self.lf` at
          the same point -- so a `FilterOp` here is lowered EXACTLY the way `vector.
          VectorGridAnalysis.apply_steps` lowers one: `apply_dimension_filter` then
          `apply_value_filter(..., mask=False)`, an unconditional row drop (see that
          method's docstring for why a drop, not a conditional mask-vs-drop, is the only
          coherent choice once the aggregate already ran at its own declared position).
          The planner never admits a `CalculateOp`/`GroupOp`/second `ProjectOp` once a
          `project` has summarized, so `FilterOp` is the only op this branch needs to
          handle. A post-project literal `grid_cell = N` filter is treated here as an
          ordinary long-shape row filter -- one zone's own aggregate does not depend on
          which OTHER zones were also requested, so post-hoc filtering the summarize's
          OUTPUT rows by `grid_cell` gives a numerically identical answer to narrowing
          the crosswalk's `zone_ids` before the summarize runs.

        `self.options.grid_cell` (`InputSelection.grid_cell`, an INPUT-level sampling
        filter, never a pipeline step) is deliberately EXCLUDED from the raster fast path
        by `planner._compile_single_input_chain` (this method is simply never reached with
        it set) rather than handled here: `_apply_grid_cell` reshapes `self.lf` from the
        metadata table straight into a long (grid_cell/value) frame via `.unpivot(...)`
        BEFORE any op could run, a shape change this method's PRE-project mode does not
        accommodate. The `if self.options.grid_cell:` guard below is kept anyway, purely
        for parity with `VectorGridAnalysis.apply_steps`'s own unconditional inclusion of
        the identical line -- it is dead code for every raster instance that actually
        reaches this method in practice (the planner gate guarantees `grid_cell is None`
        here), but costs nothing to leave in should that gate ever change.

        `rename_map`/`value_names`: identical construction to `VectorGridAnalysis.
        apply_steps` -- see that method's docstring for why a synthetic digit-string
        default output name (`ops._default_output_name`) is deliberately excluded.
        """
        if self.options.grid_cell:
            await self._apply_grid_cell()

        current_grid = self.grid.name
        value_names = {ordered_ops[0].input}
        value_names.update(op.output for op in ordered_ops if not op.output[0].isdigit())
        rename_map = {name: "value" for name in value_names}

        for op in ordered_ops:
            if isinstance(op, ProjectOp):
                if op.to == current_grid:
                    continue
                await self._apply_transform_params_impl()
                current_grid = op.to
            elif isinstance(op, FilterOp):
                rewritten = rewrite_named_columns(op.where, rename_map)
                if current_grid != self.grid.name:
                    # Post-project: self.lf is long-shape now -- see the docstring above.
                    try:
                        self.lf, value_where, has_dimension_conjunct = apply_dimension_filter(self.lf, rewritten)
                        self.lf = apply_value_filter(self.lf, value_where, mask=False)
                    except ExpressionError as exc:
                        self._error(str(exc))
                    self._where_has_dimension_conjunct = (
                        self._where_has_dimension_conjunct or has_dimension_conjunct
                    )
                elif self._band_groups is None:
                    # Pre-project, pre-group: self.lf is still the original per-band
                    # metadata table -- see the docstring above for why the dimension
                    # half runs now and the value half is stashed for a later pixel-read
                    # stage to mask.
                    try:
                        self.lf, value_where, has_dimension_conjunct = apply_dimension_filter(self.lf, rewritten)
                    except ExpressionError as exc:
                        self._error(str(exc))
                    if value_where is not None:
                        self._value_where = (
                            value_where
                            if self._value_where is None
                            else f"({self._value_where}) and ({value_where})"
                        )
                    self._where_has_dimension_conjunct = (
                        self._where_has_dimension_conjunct or has_dimension_conjunct
                    )
                else:
                    # Pre-project, POST-group: a `GroupOp` already ran, so `self.lf` now
                    # holds one row per surviving group (`group_by_cols` only -- see
                    # `_collapse_group`) instead of one row per raw band. Reassigning
                    # `self.lf` the same way the pre-group branch does would silently
                    # desynchronize it from `self._band_groups`/`self._band_ops`, which a
                    # later pixel-read consumer (`apply_band_ops`) indexes by POSITION --
                    # so this narrows all three together. A value conjunct has no
                    # band-wide replay entry that can mask an already-collapsed column,
                    # so it is rejected rather than silently mishandled.
                    metadata_df = self.lf.collect()
                    indexed_lf = metadata_df.with_row_index("__band_filter_idx").lazy()
                    try:
                        indexed_lf, value_where, has_dimension_conjunct = apply_dimension_filter(
                            indexed_lf, rewritten
                        )
                    except ExpressionError as exc:
                        self._error(str(exc))
                    if value_where is not None:
                        self._error(
                            "a value filter is not supported after a group_by step for a raster grid"
                        )
                    if has_dimension_conjunct:
                        self._flush_calculate_plans_to_band_ops(metadata_df)
                        dimension_conjuncts, _value_conjuncts = classify_where(rewritten)
                        dimension_where = " and ".join(f"({c})" for c in dimension_conjuncts)
                        filtered_df = indexed_lf.collect()
                        kept_positions = filtered_df["__band_filter_idx"].to_list()
                        self._band_ops.append(("filter", metadata_df, dimension_where))
                        self._band_groups = [self._band_groups[p] for p in kept_positions]
                        self.lf = filtered_df.drop("__band_filter_idx").lazy()
                    self._where_has_dimension_conjunct = (
                        self._where_has_dimension_conjunct or has_dimension_conjunct
                    )
            elif isinstance(op, CalculateOp):
                expression = rewrite_named_columns(op.expression, rename_map)
                where = rewrite_named_columns(op.where, rename_map) if op.where is not None else None
                try:
                    self._calculate_plans.extend(
                        compile_calculate_entries([(f"step.{op.step_no}", expression, where)])
                    )
                except ExpressionError as exc:
                    self._error(str(exc))
            elif isinstance(op, GroupOp):
                self._collapse_group(list(op.by))
            else:  # pragma: no cover -- a `JoinOp` never reaches a single-input chain
                self._error(f"unsupported op in the raster interpreter: {op.op!r}")

        if current_grid == self.grid.name:
            # No `ProjectOp` in this chain ever actually summarized (an identity project,
            # possibly followed by filter/calculate/group steps that ran against the
            # still-metadata-shaped `self.lf` above) -- `_apply_transform_params_impl`
            # still needs to run once, HERE, in what was always its fixed position (stage
            # 4, i.e. LAST) in the old engine, regardless of where an identity `project`
            # sits in DECLARED order: it is what rejects a standalone value `where`
            # conjunct given without `summarize_by`/`summarize_aggregate` ("a plain,
            # non-summarized raster selection has no ... per-pixel frame ... at all" --
            # see that method's own docstring), and every raster request that never
            # summarizes -- flat or piped through this interpreter alike -- must keep
            # rejecting that shape, not just the ones where a value filter happens to be
            # declared before the (identity) project. A genuinely summarizing `ProjectOp`
            # above already called this once, at its own declared position, and updated
            # `current_grid` away from `self.grid.name` when it did -- so this is a
            # plain no-op for that case, never a second summarize.
            await self._apply_transform_params_impl()

    def _collapse_tile_bands(self, df, data, nodata):
        """Apply a tile's calculate/group_aggregate step with Polars expressions.

        Shares one operator vocabulary with the write and summarize paths
        (``raster_source.band_collapse_exprs``/``band_calculate_passes``) rather than
        reducing the array with NumPy, which is what makes tiles treat nodata the way the
        rest of the API does: masked against each band's own raw pixel value before any
        calculate entry runs, dropped from the group_aggregate, and restored as the
        sentinel only where no band contributed anything. The NumPy reduce it replaces did
        none of that -- a sentinel was scaled by any calculate multiplier and averaged in
        as if it were a real value, so one nodata band in a group pulled the whole
        rendered pixel toward -9999.

        In and out stay array plumbing: ``data[i].ravel()`` is a view of the read buffer, and
        each output column converts straight back into the ``(bands, h, w)`` shape
        ``Legend.colorize_pixels`` expects.

        ``nodata=nodata`` is passed to ``band_collapse_exprs`` UNCONDITIONALLY, unlike
        ``_write_raster_windowed``'s conditional ``nodata=nodata if (...) else None`` --
        this is a deliberate, pre-existing divergence between the tile and write paths
        (see that method's own comment on it), preserved here exactly, not normalized:
        a written file can carry the sentinel forward for its reader to interpret, but a
        rendered pixel has to commit to a color, so a tile always masks nodata whether or
        not this request also aggregates/calculates/filters.

        ``calculate_passes``/``group_row_idxs`` come from ``self._band_ops`` (via
        ``_simple_band_collapse_inputs``) rather than from ``self._calculate_plans`` and
        ``self._band_groups`` directly, so a calculate declared before this request's
        ``group`` is applied at its own position rather than being lost to
        ``_collapse_group``'s flush. The flush is repeated here because a tile is served
        without a preceding ``group`` just as often as with one; it is a no-op once the
        pending batch has already moved, so calling it per tile is free.
        """
        raw_names = [f"b{i}" for i in range(len(df))]
        self._flush_calculate_plans_to_band_ops(self._current_metadata_df(df))
        calculate_passes, group_row_idxs = self._simple_band_collapse_inputs(raw_names)
        out_names, passes, exprs = band_collapse_exprs(
            raw_names,
            calculate_passes,
            group_row_idxs,
            self.options.group_aggregate,
            data.dtype.name,
            nodata=nodata,
        )
        frame = pl.DataFrame({name: data[i].ravel() for i, name in enumerate(raw_names)})
        for pass_exprs in passes:
            frame = frame.with_columns(pass_exprs)
        result = frame.select(exprs)
        return np.stack([result[name].to_numpy().reshape(data.shape[1:]) for name in out_names])

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

            # Warp each source directly onto the tile's Web Mercator pixel grid so
            # the output registers exactly with the tile, with no intermediate grid
            # or window rounding to shift pixels off their true location.
            tile_transform = from_bounds(*tile_bounds, tile_size, tile_size)
            vrts = [
                WarpedVRT(
                    dataset,
                    crs="EPSG:3857",
                    transform=tile_transform,
                    width=tile_size,
                    height=tile_size,
                    nodata=nodata,
                    add_alpha=False,
                    src_nodata=nodata,
                    dtype=dataset.dtypes[0],
                    resampling=Resampling.nearest,
                )
                for dataset in inputs
            ]

            # Each VRT above is already warped directly onto the tile's own pixel grid
            # (tile_transform/tile_size), so no separate intersection window or output
            # shape needs to be computed here -- reading the whole VRT reads exactly the
            # tile, and GDAL fills anything outside the source's extent with nodata.
            data = await self._read_raster_bands(df, vrts, row_indices, bands, None)
            color = self.legend.colorize_pixels(self._collapse_tile_bands(df, data, nodata))
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
        sentinel before any entry runs, restore it after -- see ``band_collapse_exprs``),
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
            self._band_groups is not None
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
        The calculate/group_aggregate step is unchanged: Polars expressions built once
        (`passes`/`exprs`) by the shared `raster_source.band_collapse_exprs`, so it uses
        the same operator vocabulary as the tile and summarize paths instead of a second
        implementation of the same four aggregates.

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
        raw_names = [f"b{i}" for i in range(len(df))]
        self._flush_calculate_plans_to_band_ops(self._current_metadata_df(df))
        calculate_passes, group_row_idxs = self._simple_band_collapse_inputs(raw_names)
        has_value_filter = self._value_where is not None

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

                # Each selected (path, band) row — or each aggregated group — becomes one
                # output band, so the source's own band count must be overridden.
                out_count = len(df) if group_row_idxs is None else len(group_row_idxs)
                profile = inputs[0].profile.copy()
                # group_by/group_aggregate can collapse the matched bands into fewer output
                # bands, so the output band count can't just be copied from the (single-band)
                # source COGs' profile.
                profile.update(count=out_count, interleave=OUTPUT_INTERLEAVE, **kwargs)
                output = stack.enter_context(rio.open(out_path, "w", **profile))
                out_dtype = profile["dtype"]
                nodata = inputs[0].nodata

                # Output window geometry, precomputed once, up front -- unchanged from
                # before the port. This list, not collect_batches' own chunk boundaries,
                # is what decides how many pixels make up "one window" for `output.write`.
                block_height, block_width = inputs[0].block_shapes[0]
                itemsize = np.dtype(inputs[0].dtypes[0]).itemsize
                blocks_per_batch = raster_blocks_per_batch(len(df), block_height, block_width, itemsize)
                window_list = list(_batched_block_windows(inputs[0].block_windows(1), blocks_per_batch))

            # Label output bands with their calendar value when rows map 1:1 to bands.
            if group_row_idxs is None:
                calendar_col = self._calendar_column(df)
                if calendar_col is not None:
                    for band_idx, value in enumerate(df[calendar_col].to_list(), start=1):
                        output.set_band_description(band_idx, f"{calendar_col}={value}")

            # Built ONCE, outside the window loop -- the expressions are identical for
            # every window, so there is no reason to reconstruct them per iteration.
            #
            # `nodata=` is passed for an aggregated (group_aggregate) selection, so a
            # nodata pixel drops out of the collapse instead of contributing a sentinel to
            # it; for any selection with calculate entries, so an entry's rewrite never
            # touches the sentinel (NoData in, NoData out -- the sentinel is masked to null
            # before the entries run and restored afterwards); and for a standalone value
            # `where` (`has_value_filter`), for the same reason -- a masked-out pixel needs
            # to come back as the sentinel, not a Polars null a numpy int array can't hold.
            # A plain, non-aggregated, non-calculated, non-filtered selection passes every
            # pixel through untouched -- see the pinned
            # test_non_aggregated_output_preserves_nodata. This is where the write path
            # deliberately differs from get_tile, which masks either way: a written file
            # can carry the sentinel forward for its reader to interpret, but a rendered
            # pixel has to commit to a color.
            out_names, passes, exprs = band_collapse_exprs(
                raw_names,
                calculate_passes,
                group_row_idxs,
                self.options.group_aggregate,
                out_dtype,
                nodata=nodata if (group_row_idxs is not None or calculate_passes or has_value_filter) else None,
            )

            # `mask_nodata=False`: nodata handling is `band_collapse_exprs`' own job above
            # (conditioned on aggregating/calculating/filtering, which `scan_rasters`'
            # unconditional mask knows nothing about) -- masking here too would be
            # redundant at best and would break the plain-passthrough case at worst (see
            # `nodata=` comment above).
            lf = scan_rasters(unique_paths, row_indices, bands, mask_nodata=False)

            # The value-filter gate: a standalone value `where`
            # on a raster-OUTPUT query lowers to a per-cell MASK, via
            # `apply_value_filter_band_wide`, and MUST NEVER be spelled as `lf.filter(...)`
            # here -- a `.filter()` on this lf would reach `scan_rasters`' own `predicate`
            # pushdown parameter and drop pixels outright, which `_take_window_rows` would
            # then surface as a hard `RuntimeError` (a short read) rather than a silently
            # misaligned grid, but the correct behavior is simply never to do it. Applied
            # before `passes` (the nodata mask, then calculate), matching
            # `_summarize_zone_stats_plugin`'s own ordering: a value predicate is written
            # against the stored pixel value, not one a calculate entry already rewrote.
            # Order relative to the nodata mask itself doesn't matter -- both are
            # idempotent null-outs of the same raw value -- only relative to calculate does.
            if has_value_filter:
                lf = apply_value_filter_band_wide(lf, raw_names, self._value_where)

            for pass_exprs in passes:
                lf = lf.with_columns(pass_exprs)
            lf = lf.select(exprs)

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
