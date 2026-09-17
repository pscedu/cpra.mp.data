"""Paired-shape execution for analysis pipelines (the former `api/pipeline_exec.py`,
execution half -- `analysis/planner.py`'s `_compile_paired_notebook_shape` is the compile
half that builds a `PairedAnalysis` in the first place).

The paired shape: a vector-grid input `project`-ed onto a raster grid, `join`-ed against
the raster input, then collapsed by a trailing per-pixel `group` -- exactly the notebook
exit-criterion query's shape. `PairedAnalysis` implements the duck-typed subset of
`RasterioStreamingResponse`/`RasterioTileResponse`'s interface that a raster output needs,
standing in for a `BaseAnalysis` instance (`BaseAnalysis` has no "join two datasets"
concept -- see that class's own docstring and `PairedAnalysis`'s, immediately below).

**There is no bespoke paired SOURCE.** `_write_paired_notebook_raster` is built from the
general raster scan plus one frame-level helper:

1. `raster.pivot_vector_to_band_wide` -- the vector dataset arrives LONG, one row per
   `(zone, calendar step)`. Pivoting it by step yields one row per zone and one column per
   band, which is the long-to-band-wide conversion and the only part of projecting a vector
   dataset onto a raster grid that is about the vector dataset alone. Pivoting FIRST is a
   correctness requirement, not a speed one: looked up by zone alone, a long frame has no
   single value per pixel to give, and a raster write reads rows back BY POSITION.
2. `sources.scan_rasters(..., crosswalks=[GridCrosswalk("p", pairing_path, wide)])` -- the
   raster bands, plus that pivoted table looked up positionally through the pairing
   crosswalk. ONE crosswalk read per window serves the zone column `p` (a property of the
   pixel, shared by every band in its row) and all of `p0..pN` (per band).

The lookup sits inside the scan rather than between the two as an ordinary frame op, which
was the intent; `sources.GridCrosswalk`'s docstring records the Polars batch-ordering
constraint that forced it and what would have to change for it to move out.

Collapsing those two into the `plan._collapse_expr()` `.select()` leaves the same
`collect_batches` + `_take_window_rows` drive `raster._write_raster_windowed` already
proved: one `output.write` per window, and nothing GDAL-related crossing a `yield` (that
constraint lives inside `scan_rasters`, not here). Because a paired plan's output is
unconditionally pixel-local (one row per pixel until the trailing per-pixel `group`
collapses across bands, never across pixels), only `tif`/`png`/`jpg` output is implemented
for this shape; a tabular paired output is deliberately left unsupported (see
`planner._compile_paired_notebook_shape`'s rejection) rather than materializing the full
joined frame, which is exactly the memory cliff this whole design avoids.

`_paired_notebook_tile` is built the same way, from the same `_paired_scan`. The two differ
only in which pixels they read and what they do with the collapsed rows: the write path
walks the whole raster in batched block windows and writes each one; the tile path restricts
the scan to the ONE window the tile's footprint covers (`utils.tile_source_window`), then
warps the collapsed result onto the tile grid (`utils.warp_window_to_tile`).

That reordering -- reproject LAST, not first -- is what made the tile path expressible as
the same scan at all. It used to warp every raster band AND the pairing crosswalk through a
`WarpedVRT` before collapsing anything, and no source has reprojection to offer, so it had
to hand-roll a parallel collapse over `sources._dense_lookup_array`/
`_crosswalk_lookup_values`. Reprojection is a rendering step; the per-pixel band algebra
never needed to know about it.
"""
from contextlib import ExitStack
from io import BytesIO

import mercantile
import numpy as np
import polars as pl
import rasterio as rio
from PIL import Image
from rasterio.coords import BoundingBox, disjoint_bounds
from rasterio.warp import transform_bounds

from cpra.mp.data import config
from cpra.mp.data.analysis.base import AnalysisError
from cpra.mp.data.analysis.io.raster_io import (
    group_rows_by_path,
    raster_blocks_per_batch,
    raster_gdal_env,
    tile_read_shape,
    tile_source_window,
    warp_window_to_tile,
)
from cpra.mp.data.analysis.pipeline import (
    PipelineError,
)
from cpra.mp.data.analysis.raster import (
    _take_window_rows,
    crosswalk_value_column_names,
    pivot_vector_to_band_wide,
)
from cpra.mp.data.analysis.request.expressions import (
    parse_value_expr,
    rewrite_named_columns,
)
from cpra.mp.data.analysis.sources import (
    GridCrosswalk,
    _batched_windows,
    horizontal_aggregate_expr,
    scan_rasters,
)
from cpra.mp.data.analysis.types import OutputType
from cpra.mp.data.legends import LEGENDS, Legend

# The column namespace a single-input pipeline's `filter.where` may reference once its own
# running value has been rewritten to `value` (see `_compile_single_input_chain`): every
# real filterable dimension, plus `grid_cell` and the placeholder itself. Built once at
# import time -- `FilterDimensions.model_fields` is itself fixed at import time already
# (built by `create_model` from `cpra.mp.data.dimensions.DIMENSIONS`).

def _legend_for(color_map):
    """Standalone reimplementation of `analysis.output.resolve_legend`'s custom color-map
    parsing, for a `PairedAnalysis` -- which has no backing `BaseAnalysis` instance to
    delegate to and must raise `PipelineError`, not the `AnalysisError` that function
    raises.
    """
    color_map = color_map or ""
    if ":" in color_map:
        map_type, color_str = color_map.split(":", 1)
        if map_type not in ("continuous", "discrete"):
            raise PipelineError(f"invalid color map type: {map_type}")
        color_strs = color_str.split(",")
        if map_type == "continuous":
            colors = color_strs[1::2]
            value_strs = color_strs[::2]
        else:
            colors = color_strs[::2]
            value_strs = color_strs[1::2]
        try:
            values = [float(v) for v in value_strs]
        except ValueError as exc:
            raise PipelineError("invalid color map values") from exc
        if map_type == "discrete":
            values += [None]
        if len(values) != len(colors):
            raise PipelineError("mismatched colors and values in color map")
        return Legend("custom", pl.DataFrame(dict(color=colors, upper_bound=values)), type=map_type)

    return LEGENDS.get(color_map) or LEGENDS["default"]


class PairedAnalysis:
    """The compiled paired shape: a vector-grid input broadcast onto a raster grid via
    `project`, joined against the raster-grid input via `join`, collapsed by a trailing `group`
    step -- the notebook query's exact shape (see the module docstring for why this
    has no bespoke source of its own). Implements the duck-typed subset of
    `LazyFrameSource`'s interface `RasterioStreamingResponse`/`RasterioTileResponse` need,
    standing in for a `BaseAnalysis` instance (which has no "join two datasets" concept).

    Deliberately a SIBLING of `BaseAnalysis`, not a subclass: `BaseAnalysis`'s entire
    reason to exist is the `self.lf` mutation walk (an ordered op list -- see that
    class's docstring) that `apply_params` runs. `PairedAnalysis` has no `self.lf` at all and no equivalent of that pipeline --
    its own `apply_params` is a single `_prepare()` call that builds numpy lookup arrays
    from two ALREADY-`apply_params`'d `raster_dataset`/`vector_dataset` instances (each a
    real `BaseAnalysis` subclass), and every other method it exposes
    (`get_tile`/`write_raster`/`validate_tile`/`estimate_data_size`/`output_type`) is a
    same-named, same-shaped, but independently-implemented method, not an override of
    one `BaseAnalysis` provides. Subclassing here would buy nothing but a shared base
    whose actual behavior (the op walk) `PairedAnalysis` doesn't use and can't
    honestly claim to specialize -- the honest relationship is "implements the same
    duck-typed surface", which is exactly what this class already documented itself as
    before the split, one paragraph up.
    """

    def __init__(self, *, raster_dataset, vector_dataset, raster_name, vector_alias, where, group_step, color_map):
        self.raster_dataset = raster_dataset
        self.vector_dataset = vector_dataset
        self.raster_name = raster_name
        self.vector_alias = vector_alias
        self.where = where
        self.group_step = group_step
        self.color_map = color_map
        self._prepared = False

    # -- LazyFrameSource-shaped surface -----------------------------------------------

    @property
    def output_type(self):
        return OutputType.raster

    @property
    def variable(self):
        return self.raster_dataset.variable

    async def apply_params(self):
        self._prepare()

    def estimate_data_size(self):
        return 0

    def validate_tile(self):
        self._prepare()
        if not self.band_sources:
            raise PipelineError("no data available to render a tile for this pipeline")

    def validate_vector_tile(self):
        """A paired pipeline's output is always raster (see `output_type` above) -- there
        is no vector-tile shape to validate further, so this always rejects, exactly like
        `BaseAnalysis.validate_vector_tile`'s own raster-output branch (same message, same
        status code), which is what a `.pbf` route call actually needs: an `AnalysisError`
        the route layer's registered exception handler turns into a graceful 400, not an
        unhandled exception. `PipelineError` (what `validate_tile` raises above) has no
        such handler registered and no call site wraps this method in a try/except, so it
        would surface as a raw 500 instead.
        """
        raise AnalysisError("invalid output type for vector tile: raster", status_code=400)

    async def write_raster(self, out_path, **kwargs):
        self._prepare()
        return _write_paired_notebook_raster(self, out_path, **kwargs)

    async def get_tile(self, x, y, z, format, tile_size=256, **kwargs):
        self._prepare()
        return await _paired_notebook_tile(self, x, y, z, format, tile_size=tile_size)

    # -- compile-time preparation -------------------------------------------------------

    def _prepare(self):
        if self._prepared:
            return

        raster_df = self.raster_dataset.materialize()
        vector_df = self.vector_dataset.materialize()

        of = self.group_step.of
        is_count = self.group_step.aggregate == "count"
        # `count` has no aggregand -- it counts BANDS satisfying `where` (or, with no
        # `where` at all, bands with real raster+paired data -- see `_band_valid_expr`),
        # never a value pulled from `of`. Requiring `of` to be left at its pydantic
        # default ("value") is simpler and safer than defining a meaning for a
        # non-default `of` alongside `count` (which would invite a caller to think
        # `count` somehow aggregates THAT column) -- "value" can never collide with a
        # real column name here regardless (`pipeline.RESERVED_NAMES` forbids any input/
        # step from being named "value"), so this is exactly "did the caller leave `of`
        # unset", not a coincidental match.
        if is_count and of != "value":
            raise PipelineError(
                f"group step's aggregate='count' counts bands satisfying `where` (or, "
                "with no `where`, bands with real data) -- it has no aggregand, so `of` "
                f"must be left at its default ('value'), not set to {of!r}"
            )
        if not is_count and of not in raster_df.columns and of not in (self.raster_name, self.vector_alias):
            raise PipelineError(
                f"group step's of={of!r} is not a column of input {self.raster_name!r}, "
                f"nor one of the joined names ({self.raster_name!r}, "
                f"{self.vector_alias!r})"
            )

        paths = raster_df["path"].to_list() if "path" in raster_df.columns else []
        has_bands = "band" in raster_df.columns
        row_bands = raster_df["band"].to_list() if has_bands else [1] * len(paths)
        self.band_sources = list(zip(paths, row_bands))
        self.raster_cols = [f"b{i}" for i in range(len(self.band_sources))]
        self.paired_cols = [f"p{i}" for i in range(len(self.band_sources))]

        if is_count:
            # Integer output; -1 is never a valid count (counts are >= 0), matching the
            # same sentinel `of_kind == "dimension"`'s integer branch below already uses
            # for the same reason (a real value, e.g. calendar_year 2024, could collide
            # with a small nodata sentinel, but a COUNT never goes negative, so -1 is
            # unambiguous here). See `_count_zone_gate` for when this sentinel is used
            # (outside the pairing crosswalk's domain) versus a real 0 (every band
            # nodata/pairing-missed/condition-failed, but still inside that domain).
            self.of_kind = "count"
            self.out_dtype = "int32"
            self.nodata_value = -1
        elif of in raster_df.columns:
            of_dtype = raster_df.schema[of]
            if not of_dtype.is_numeric():
                raise PipelineError(
                    f"group step's of={of!r} is a {of_dtype} dimension column -- only a "
                    "numeric dimension (e.g. calendar_year) is supported as a paired "
                    "pipeline's per-band aggregand; a temporal/string dimension has no "
                    "raster output dtype to write it as"
                )
            self.of_kind = "dimension"
            self.of_values = raster_df[of].to_list()
            self.out_dtype = "int32" if of_dtype.is_integer() else "float32"
            self.nodata_value = -1 if of_dtype.is_integer() else -9999.0
        else:
            self.of_kind = "named"
            self.of_name = of
            self.out_dtype = "float32"
            self.nodata_value = -9999.0

        raster_grid = self.raster_dataset.dataset_dimensions.grid
        vector_grid = self.vector_dataset.dataset_dimensions.grid
        # Raster grid first regardless of data-flow direction -- see
        # tests/factories.py's write_raster_crosswalk and the real crosswalk directory's
        # own naming (confirmed on real data).
        self.pairing_crosswalk_path = config.CROSSWALKS_PATH / f"{raster_grid}__{vector_grid}.tif"

        calendar_col = next(
            (c for c in ("calendar_year", "calendar_day") if c in raster_df.columns and c in vector_df.columns),
            None,
        )
        # ONE form of the vector-grid selection, serving both output paths (they run the
        # same `_paired_scan`; there used to be a second, dense-array form here that only
        # the tile path read).
        #
        # `self._vector_long` is the LONG frame -- `grid_cell | _step | value`, one row per
        # (zone, step) -- that `_paired_scan` pivots with `pivot_vector_to_band_wide`.
        # It is deliberately NOT pre-pivoted here, because which steps the pivot needs is a
        # property of the BANDS being read, which is the pivot's own input, not this
        # dataset's.
        #
        # `self._vector_band_steps` is one step per band, positionally aligned with
        # `band_sources`, in the same vocabulary `_vector_long._step` uses. With no shared
        # calendar dimension at all, both sides collapse onto the single sentinel step `0`
        # -- "one value per zone, the same for every band" -- rather than onto `None`, so
        # the pivot never has to key on a null.
        if calendar_col is not None:
            self._vector_band_steps = raster_df[calendar_col].to_list()
            self._vector_long = vector_df.select(
                pl.col("grid_cell"),
                pl.col(calendar_col).alias("_step"),
                pl.col("value"),
            )
        else:
            self._vector_band_steps = [0] * len(self.band_sources)
            self._vector_long = vector_df.select(
                pl.col("grid_cell"),
                pl.lit(0, dtype=pl.Int32).alias("_step"),
                pl.col("value"),
            )

        self._prepared = True

    # -- shared per-band collapse expression ---------------------------------------------

    def _band_value_expr(self, i):
        """Band `i`'s own contribution to the collapse -- its `of` value, `None`
        wherever `where` (rewritten onto this band's synthetic `raster_cols[i]`/
        `paired_cols[i]` names) doesn't hold. Factored out of `_collapse_expr` so that
        anything wanting ONE band's contribution -- `tests/api/test_paired_port.py`'s
        pre-port oracles evaluate it a band at a time over a standalone two-column frame --
        reuses the IDENTICAL expression rather than re-deriving the where-clause/nodata
        semantics in NumPy and risking drift from what this method produces as `where`
        grows more expressive than any one fixture.
        """
        if self.of_kind == "dimension":
            value_expr = pl.lit(self.of_values[i])
        elif self.of_name == self.raster_name:
            value_expr = pl.col(self.raster_cols[i])
        else:
            value_expr = pl.col(self.paired_cols[i])

        if self.where:
            rename_map = {self.raster_name: self.raster_cols[i], self.vector_alias: self.paired_cols[i]}
            rewritten = rewrite_named_columns(self.where, rename_map)
            cond = parse_value_expr(rewritten, allowed_columns=[self.raster_cols[i], self.paired_cols[i]])
            value_expr = pl.when(cond.fill_null(False)).then(value_expr).otherwise(None)
        return value_expr

    def _band_valid_expr(self, i):
        """Band `i`'s boolean CONTRIBUTES-to-`count` indicator: `where` holding for this
        band (rewritten onto its synthetic columns, exactly as `_band_value_expr` does),
        or -- with no `where` clause at all -- this band's own raster value and paired
        value both being present (not raster-nodata, not a pairing-miss). `count` has no
        `of`/value to select (`_prepare` rejects a non-default `of` alongside it), so
        this mirrors only the MASKING half of `_band_value_expr`, never its value-
        selection half -- deliberately a separate method rather than a branch inside
        `_band_value_expr`, since the two return different things (a boolean indicator
        vs. a masked value) for a fundamentally different question.
        """
        raster_col, paired_col = self.raster_cols[i], self.paired_cols[i]
        if self.where:
            rename_map = {self.raster_name: raster_col, self.vector_alias: paired_col}
            rewritten = rewrite_named_columns(self.where, rename_map)
            cond = parse_value_expr(rewritten, allowed_columns=[raster_col, paired_col])
            return cond.fill_null(False)
        return pl.col(raster_col).is_not_null() & pl.col(paired_col).is_not_null()

    def _collapse_expr(self):
        n = len(self.band_sources)
        if self.group_step.aggregate == "count":
            # `count`'s own LOCAL collapse form, deliberately NOT a
            # `HORIZONTAL_RASTER_AGGREGATES` entry: a plain horizontal sum of each band's
            # boolean validity indicator, never null (every summand is a concrete
            # True/False, never a Polars null), so "zero contributing bands" surfaces as a
            # real 0 here -- the "outside the pairing crosswalk's domain" nodata case is
            # NOT decided by this expression at all; it needs `pairing_valid`, which only
            # a caller holding the zone column has (see `_count_zone_gate`, applied by
            # `_collapse_paired_frame` after every use of this method).
            if n == 0:
                return pl.lit(0, dtype=pl.Int32).alias("value")
            valid_exprs = [self._band_valid_expr(i).cast(pl.Int32) for i in range(n)]
            return pl.sum_horizontal(valid_exprs).alias("value")

        value_exprs = [self._band_value_expr(i) for i in range(n)]
        collapsed = horizontal_aggregate_expr(self.group_step.aggregate, value_exprs, None)
        return collapsed.fill_null(self.nodata_value).alias("value")


#: Which `group` aggregates `PairedAnalysis` implements at all -- see
#: `_validate_paired_group_aggregate` for the defense-in-depth check that used to be
#: missing here (`aggregate == "count"` used to reach the collapse only because nothing
#: rejected it for this shape, and `horizontal_aggregate_expr` raised a bare `KeyError`
#: for it).
_SUPPORTED_PAIRED_AGGREGATES = frozenset({"min", "max", "sum", "mean", "count"})


def _validate_paired_group_aggregate(group_step):
    """Defense-in-depth: reject any `group_step.aggregate` outside
    `_SUPPORTED_PAIRED_AGGREGATES` for THIS shape, before `PairedAnalysis` ever dispatches on
    it.

    Today this is unreachable through the sanctioned `parse_pipeline` entry point --
    `_SUPPORTED_PAIRED_AGGREGATES` is now exactly `fold.DECOMPOSABLE_AGGREGATES`
    (`{"min", "max", "sum", "mean", "count"}`), and `validate_pipeline`'s own
    `_require_decomposable` check already rejects everything else (`count_by_value`,
    or any future non-decomposable value) before a `GroupStep` is even constructed. It
    guards two things a "GroupStep.aggregate is always decomposable" invariant does NOT:
    a caller that builds a `Pipeline`/`GroupStep` directly and skips `validate_pipeline`
    (see `pipeline.py`'s own "Why business validation is NOT wired into pydantic
    validators" section), and -- more importantly -- any FUTURE aggregate added to
    `fold.DECOMPOSABLE_AGGREGATES` without matching support added here: without this
    check, that aggregate would compile clean and fall through to `_collapse_expr`, whose
    `horizontal_aggregate_expr` call raises a bare `KeyError` for anything
    `HORIZONTAL_RASTER_AGGREGATES` doesn't know -- exactly the latent defect `count` itself
    used to hit before this change. Converts that failure mode into a plan-time
    `PipelineError` (a 400) for anything that reaches it.
    """
    if group_step.aggregate not in _SUPPORTED_PAIRED_AGGREGATES:
        raise PipelineError(
            f"not yet supported in analysis pipelines: this paired pipeline shape's "
            f"`group` step aggregate {group_step.aggregate!r} -- only "
            f"{sorted(_SUPPORTED_PAIRED_AGGREGATES)} are implemented for a paired "
            "(joined) pipeline's per-pixel band collapse"
        )


def _count_zone_gate(plan, flat_values, pairing_valid):
    """`count`'s zero-vs-nodata split, applied by `_collapse_paired_frame` after either
    output path's collapse computes a flat per-pixel band tally -- a no-op for every other
    aggregate.

    `pairing_valid` (this pixel's own membership in the pairing crosswalk -- the only
    "is this pixel even in this plan's output domain at all" signal a `PairedAnalysis` has;
    see the module docstring's discussion of why this shape has no separate output-zone
    crosswalk the way a zonal scan does) decides nodata; every in-zone pixel's own
    tally, INCLUDING a genuine 0 (every band nodata, pairing-missed, or condition-failed),
    is left untouched. This is the opposite of every OTHER aggregate's `no_contributor`
    fallback (`horizontal_aggregate_expr`'s `no_contributor_at`), which cannot distinguish
    "out of domain" from "in domain but nothing contributed" from column nullity alone
    and so must treat both the same (nodata) -- `count` can tell the two apart because it
    has `pairing_valid` in hand, and telling them apart matters: "zero contributing bands"
    is `count`'s own correct answer for an in-domain pixel, not an edge case standing in
    for an undefined one (see `sources.guarded_aggregate`'s identical `count`
    bypass for the flat API's vertical/zonal count, and `PairedAnalysis._band_valid_expr`).
    """
    if plan.group_step.aggregate != "count":
        return flat_values
    return np.where(pairing_valid, flat_values, plan.nodata_value)


#: The pairing `GridCrosswalk`'s name: both the zone-id column it emits and the prefix of
#: the per-band value columns `p0..pN`. `"p"` because `PairedAnalysis.paired_cols` -- which
#: `_collapse_expr` and both retained reference implementations all write expressions
#: against -- is already `p0..pN`, and `crosswalk_value_column_names` reproduces exactly
#: those names from the same prefix, so the two families cannot drift. It also has to avoid
#: `scan_rasters`' own reserved names (`grid_cell`, `_px`, `b<i>`), which that source
#: rejects rather than silently overwriting a column.
PAIRED_CROSSWALK_NAME = "p"


def _write_paired_notebook_raster(plan: PairedAnalysis, out_path, **profile_kwargs):
    """`_paired_scan` + `collect_batches`, driven exactly like `raster._write_raster_windowed`.

    The scan (shared with `_paired_notebook_tile`, whose only difference is that it restricts
    it to one window) reads the pairing crosswalk once per window and looks the pivoted vector
    table up per pixel, producing one `p<i>` beside each `b<i>` plus the raw zone column `p`.
    `plan._collapse_expr()` collapses each row through one `.select()`, and `_take_window_rows`
    pulls the result back out window-at-a-time to be written once per window. Nothing
    GDAL-related crosses a `yield`; that constraint lives inside `scan_rasters`, not here.

    `count`'s own zero-vs-nodata split (`_count_zone_gate`, applied by `_collapse_paired_frame`)
    needs `pairing_valid` -- whether each pixel is even inside the pairing crosswalk's domain,
    independent of any band's own lookup miss. That is the SAME crosswalk read the value
    columns come from, so `count` just keeps the zone column in the `.select()` and tests it
    for nullity; no second read of the crosswalk, and no second mode of the scan, is involved.
    (`scan_rasters` masks a crosswalk zone column's nodata sentinel to null unconditionally --
    see its `crosswalks` parameter -- which is what makes `is_not_null()` the whole of the
    domain test here.)

    One pipeline serves every supported aggregate: there is no wide-vs-streaming dispatch on
    this path, and none on the tile path either.
    """
    if not plan.band_sources:
        return False

    with ExitStack() as stack:
        with raster_gdal_env():
            ref_path, _ref_band = plan.band_sources[0]
            ref_src = stack.enter_context(rio.open(ref_path))
            profile = ref_src.profile.copy()
            block_height, block_width = ref_src.block_shapes[0]
            itemsize = np.dtype(plan.out_dtype).itemsize
            n_units = max(len(plan.band_sources) * 2, 1)
            blocks_per_batch = raster_blocks_per_batch(n_units, block_height, block_width, itemsize)
            window_list = list(_batched_windows(ref_src, blocks_per_batch))

            profile.update(count=1, dtype=plan.out_dtype, nodata=plan.nodata_value, compress="deflate")
            profile.update(profile_kwargs)
            output = stack.enter_context(rio.open(str(out_path), "w", **profile))

        lf, _metadata_df = _paired_scan(plan)
        lf = lf.select(_paired_select_columns(plan, plan._collapse_expr()))

        chunk_size = blocks_per_batch * block_height * block_width
        batch_iter = iter(lf.collect_batches(chunk_size=chunk_size, maintain_order=True))
        buffer = None

        for window in window_list:
            needed = window.height * window.width
            result, buffer = _take_window_rows(batch_iter, buffer, needed)

            values = _collapse_paired_frame(plan, result)
            block = values.astype(plan.out_dtype, copy=False).reshape((1, window.height, window.width))
            output.write(block, window=window)
            del result, block

    return True


def _paired_scan(plan, read_windows=None, read_shapes=None):
    """The scan both output paths share: every raster band, plus the pivoted vector table
    looked up per pixel through the pairing crosswalk.

    A free function taking `plan` rather than a method, because what it builds belongs to
    the REQUEST (these bands, this pairing crosswalk, this pivot) and not to the plan
    object, and because having the write path and the tile path call one function is the
    whole point -- the two differ only in `read_windows`/`read_shapes` (the whole raster,
    batched, at full resolution, versus one tile's source footprint decimated to roughly the
    tile's own resolution) and in what they do with the collapsed rows.

    Returns `(lf, metadata_df)`; the caller selects `plan._collapse_expr()` off `lf` and,
    for `count`, keeps `PAIRED_CROSSWALK_NAME` beside it for `_count_zone_gate`.
    """
    band_df = pl.DataFrame(
        {
            "path": [path for path, _ in plan.band_sources],
            "band": [band for _, band in plan.band_sources],
        }
    )
    paths, row_indices, bands = group_rows_by_path(band_df)

    # One row per band, in `band_sources` order -- the band-wide metadata frame's only
    # load-bearing property here is its HEIGHT, which is what `band_column_names`/
    # `crosswalk_value_column_names` count `b<i>`/`p<i>` off. `PairedAnalysis` has no
    # dimension frame of its own to hand over (that is the two-shape dispatch this class
    # still predates), so the minimal frame carrying that height is built here.
    metadata_df = pl.DataFrame({"_band": pl.Series(range(len(plan.band_sources)), dtype=pl.UInt32)})

    vector_wide = pivot_vector_to_band_wide(
        metadata_df,
        plan._vector_long,
        plan._vector_band_steps,
        step_column="_step",
        prefix=PAIRED_CROSSWALK_NAME,
    )
    assert crosswalk_value_column_names(metadata_df, PAIRED_CROSSWALK_NAME) == plan.paired_cols

    lf = scan_rasters(
        paths,
        row_indices,
        bands,
        crosswalks=[
            GridCrosswalk(
                name=PAIRED_CROSSWALK_NAME,
                path=str(plan.pairing_crosswalk_path),
                values=vector_wide,
            )
        ],
        read_windows=read_windows,
        read_shapes=read_shapes,
    )
    return lf, metadata_df


def _collapse_paired_frame(plan, frame):
    """`plan._collapse_expr()`'s result as a flat NumPy array, with `count`'s zone gate
    already applied -- the two steps that always go together, wherever the rows came from.

    `frame` is the already-collected result of selecting `_paired_scan`'s collapse columns
    (the zone column first, for `count`).
    """
    values = frame["value"].to_numpy()
    if plan.group_step.aggregate == "count":
        pairing_valid = frame[PAIRED_CROSSWALK_NAME].is_not_null().to_numpy()
        values = _count_zone_gate(plan, values, pairing_valid)
    return values


def _paired_select_columns(plan, collapse_expr):
    """Never `.filter()` a paired scan: `_take_window_rows` requires strict, gapless
    raster-pixel order (see its own docstring), and the tile path reshapes the rows into a
    window just as positionally. `count`'s zone column is carried along for
    `_count_zone_gate` to MASK afterward, never to drop rows.
    """
    if plan.group_step.aggregate == "count":
        return [pl.col(PAIRED_CROSSWALK_NAME), collapse_expr]
    return [collapse_expr]


async def _paired_notebook_tile(plan: PairedAnalysis, x, y, z, format, tile_size=256):
    """The same `_paired_scan` the write path runs, restricted to the one source window the
    tile's footprint covers -- and decimated to roughly the tile's own resolution while
    reading it -- collapsed the same way, and warped onto the tile grid LAST.

    The tile is rendered in the sources' OWN CRS and reprojected only once the per-pixel
    band algebra is done, which is what lets that algebra be the identical scan-and-collapse
    `_write_paired_notebook_raster` runs. Warping first -- one `WarpedVRT` per raster band
    plus one for the pairing crosswalk, which is what this did -- forced a hand-rolled
    parallel collapse instead, because no source has reprojection to offer.

    Reprojecting the PAIRING CROSSWALK is the part that most needed to stop: its pixels are
    zone IDs, and a zone ID resampled onto a foreign grid is only meaningful under nearest
    neighbour. Here it is never resampled at all -- it is read on its own grid, positionally
    beside the band pixels, and only the collapsed RESULT crosses CRSs.
    `tests/api/test_paired_port.py` pins that the ported tile and the warp-first one agree
    pixel for pixel, and `tests/api/test_tile_warp.py` pins the warp geometry itself.
    """
    if not plan.band_sources:
        return None

    tile_bounds = BoundingBox(*mercantile.xy_bounds(x, y, z))

    with raster_gdal_env(), rio.open(plan.band_sources[0][0]) as ref:
        dataset_bounds = BoundingBox(*transform_bounds(ref.crs, "EPSG:3857", *ref.bounds))
        if disjoint_bounds(dataset_bounds, tile_bounds):
            return None
        window = tile_source_window(ref, tile_bounds)
        src_transform, src_crs = ref.transform, ref.crs

    if window is None:
        # The tile's footprint misses the source's pixels entirely, even though it
        # overlapped the (rectangular, reprojected) dataset bounds tested above. That is a
        # rendered tile of nothing, not a missing tile.
        tile_values = np.full((1, tile_size, tile_size), plan.nodata_value, dtype=plan.out_dtype)
    else:
        # The scan is charged per PIXEL of the window, and a low-zoom tile's window is most
        # of the raster: at z=6 on morph_pixel that is ~108M pixels through the collapse to
        # render 65,536. `tile_read_shape` decimates the read to about the tile's own
        # resolution (GDAL serves it from the source's overviews), so the row count follows
        # the tile rather than the zoom. None -- a window already at tile resolution,
        # z >~ 12 -- scans and warps exactly as it did before.
        read_shape = tile_read_shape(window, tile_size)
        lf, _metadata_df = _paired_scan(plan, read_windows=[window], read_shapes=[read_shape])
        frame = lf.select(_paired_select_columns(plan, plan._collapse_expr())).collect()
        values = _collapse_paired_frame(plan, frame)
        rows, cols = read_shape if read_shape is not None else (window.height, window.width)
        tile_values = warp_window_to_tile(
            values.astype(plan.out_dtype, copy=False).reshape((1, rows, cols)),
            window=window,
            src_transform=src_transform,
            src_crs=src_crs,
            tile_bounds=tile_bounds,
            tile_size=tile_size,
            nodata=plan.nodata_value,
            out_shape=read_shape,
        )

    legend = _legend_for(plan.color_map)
    color = legend.colorize_pixels(tile_values)
    image = Image.fromarray(color[:, :, :3] if format == "jpg" else color)
    with BytesIO() as memfile:
        image.save(memfile, format=("JPEG" if format == "jpg" else "PNG"))
        return memfile.getvalue()


# ------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------


