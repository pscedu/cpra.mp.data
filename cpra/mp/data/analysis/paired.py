"""Paired-shape execution for analysis pipelines (the former api/pipeline_exec.py, execution half)."""
from contextlib import ExitStack
from io import BytesIO

import mercantile
import numpy as np
import polars as pl
import rasterio as rio
from PIL import Image
from rasterio.coords import BoundingBox, disjoint_bounds
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from cpra.mp.data import config
from cpra.mp.data.analysis.base import AnalysisError
from cpra.mp.data.analysis.expressions import (
    parse_value_expr,
    rewrite_named_columns,
)
from cpra.mp.data.analysis.pipeline import (
    PipelineError,
)
from cpra.mp.data.analysis.sources import (
    _batched_windows,
    _dense_lookup_array,
    _paired_lookup_values,
    horizontal_aggregate_expr,
)
from cpra.mp.data.analysis.types import OutputType
from cpra.mp.data.analysis.utils import (
    raster_blocks_per_batch,
    raster_gdal_env,
)
from cpra.mp.data.legends import LEGENDS, Legend

# The column namespace a single-input pipeline's `filter.where` may reference once its own
# running value has been rewritten to `value` (see `_compile_single_input_chain`): every
# real filterable dimension, plus `grid_cell` and the placeholder itself. Built once at
# import time -- `FilterDimensions.model_fields` is itself fixed at import time already
# (built by `create_model` from `cpra.mp.data.dimensions.DIMENSIONS`).

def _legend_for(color_map):
    """Standalone reimplementation of `BaseAnalysis.legend`'s custom color-map parsing,
    for a `PairedAnalysis` -- which has no backing `BaseAnalysis` instance to delegate
    to and must raise `PipelineError`, not `BaseAnalysis._error`'s `AnalysisError`.
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
    """The compiled paired shape: a coarse input broadcast onto a finer raster grid via
    `project`, joined against the finer input via `join`, collapsed by a trailing `group`
    step -- the notebook query's exact shape (see the module docstring for why this
    doesn't use `raster_source.scan_paired`). Implements the duck-typed subset of
    `LazyFrameSource`'s interface `RasterioStreamingResponse`/`RasterioTileResponse` need,
    standing in for a `BaseAnalysis` instance (which has no "join two datasets" concept).

    Deliberately a SIBLING of `BaseAnalysis`, not a subclass: `BaseAnalysis`'s entire
    reason to exist is the staged `self.lf` mutation pipeline (filter/calculate/group/
    transform/sort-select -- see that class's docstring) that `apply_params` runs in
    order. `PairedAnalysis` has no `self.lf` at all and no equivalent of that pipeline --
    its own `apply_params` is a single `_prepare()` call that builds numpy lookup arrays
    from two ALREADY-`apply_params`'d `fine_dataset`/`coarse_dataset` instances (each a
    real `BaseAnalysis` subclass), and every other method it exposes
    (`get_tile`/`write_raster`/`validate_tile`/`estimate_data_size`/`output_type`) is a
    same-named, same-shaped, but independently-implemented method, not an override of
    one `BaseAnalysis` provides. Subclassing here would buy nothing but a shared base
    whose actual behavior (the staged flow) `PairedAnalysis` doesn't use and can't
    honestly claim to specialize -- the honest relationship is "implements the same
    duck-typed surface", which is exactly what this class already documented itself as
    before the split, one paragraph up.
    """

    def __init__(self, *, fine_dataset, coarse_dataset, fine_name, coarse_alias, where, group_step, color_map):
        self.fine_dataset = fine_dataset
        self.coarse_dataset = coarse_dataset
        self.fine_name = fine_name
        self.coarse_alias = coarse_alias
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
        return self.fine_dataset.variable

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

        fine_df = self.fine_dataset.materialize()
        coarse_df = self.coarse_dataset.materialize()

        of = self.group_step.of
        is_count = self.group_step.aggregate == "count"
        # `count` has no aggregand -- it counts BANDS satisfying `where` (or, with no
        # `where` at all, bands with real fine+paired data -- see `_band_valid_expr`),
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
        if not is_count and of not in fine_df.columns and of not in (self.fine_name, self.coarse_alias):
            raise PipelineError(
                f"group step's of={of!r} is not a column of input {self.fine_name!r}, "
                f"nor one of the joined names ({self.fine_name!r}, "
                f"{self.coarse_alias!r})"
            )

        paths = fine_df["path"].to_list() if "path" in fine_df.columns else []
        has_bands = "band" in fine_df.columns
        row_bands = fine_df["band"].to_list() if has_bands else [1] * len(paths)
        self.band_sources = list(zip(paths, row_bands))
        self.fine_cols = [f"b{i}" for i in range(len(self.band_sources))]
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
        elif of in fine_df.columns:
            of_dtype = fine_df.schema[of]
            if not of_dtype.is_numeric():
                raise PipelineError(
                    f"group step's of={of!r} is a {of_dtype} dimension column -- only a "
                    "numeric dimension (e.g. calendar_year) is supported as a paired "
                    "pipeline's per-band aggregand; a temporal/string dimension has no "
                    "raster output dtype to write it as"
                )
            self.of_kind = "dimension"
            self.of_values = fine_df[of].to_list()
            self.out_dtype = "int32" if of_dtype.is_integer() else "float32"
            self.nodata_value = -1 if of_dtype.is_integer() else -9999.0
        else:
            self.of_kind = "named"
            self.of_name = of
            self.out_dtype = "float32"
            self.nodata_value = -9999.0

        fine_grid = self.fine_dataset.dataset_dimensions.grid
        coarse_grid = self.coarse_dataset.dataset_dimensions.grid
        # Raster grid first regardless of data-flow direction -- see
        # tests/factories.py's write_raster_crosswalk and the real crosswalk directory's
        # own naming (confirmed on real data).
        self.pairing_crosswalk_path = config.CROSSWALKS_PATH / f"{fine_grid}__{coarse_grid}.tif"

        calendar_col = next(
            (c for c in ("calendar_year", "calendar_day") if c in fine_df.columns and c in coarse_df.columns),
            None,
        )
        self.lookup_arrays = {}
        if calendar_col is not None:
            self.calendar_values = fine_df[calendar_col].to_list()
            for key, group in coarse_df.group_by(calendar_col):
                self.lookup_arrays[key[0]] = _dense_lookup_array(
                    group["grid_cell"].to_numpy(), group["value"].to_numpy().astype("float64")
                )
        else:
            self.calendar_values = [None] * len(self.band_sources)
            self.lookup_arrays[None] = _dense_lookup_array(
                coarse_df["grid_cell"].to_numpy(), coarse_df["value"].to_numpy().astype("float64")
            )

        self._prepared = True

    # -- shared per-band collapse expression ---------------------------------------------

    def _band_value_expr(self, i):
        """Band `i`'s own contribution to the collapse -- its `of` value, `None`
        wherever `where` (rewritten onto this band's synthetic `fine_cols[i]`/
        `paired_cols[i]` names) doesn't hold. Factored out of `_collapse_expr` so
        `_band_masked_value` (the per-band streaming accumulator's evaluator -- see
        `_accumulate_paired_window`) can reuse the IDENTICAL expression over a
        standalone one-band frame, rather than re-deriving the where-clause/nodata
        semantics for that path in numpy and risking drift from what this method
        produces as `where` grows more expressive than any one fixture.
        """
        if self.of_kind == "dimension":
            value_expr = pl.lit(self.of_values[i])
        elif self.of_name == self.fine_name:
            value_expr = pl.col(self.fine_cols[i])
        else:
            value_expr = pl.col(self.paired_cols[i])

        if self.where:
            rename_map = {self.fine_name: self.fine_cols[i], self.coarse_alias: self.paired_cols[i]}
            rewritten = rewrite_named_columns(self.where, rename_map)
            cond = parse_value_expr(rewritten, allowed_columns=[self.fine_cols[i], self.paired_cols[i]])
            value_expr = pl.when(cond.fill_null(False)).then(value_expr).otherwise(None)
        return value_expr

    def _band_valid_expr(self, i):
        """Band `i`'s boolean CONTRIBUTES-to-`count` indicator: `where` holding for this
        band (rewritten onto its synthetic columns, exactly as `_band_value_expr` does),
        or -- with no `where` clause at all -- this band's own fine value and paired
        value both being present (not fine-nodata, not a pairing-miss). `count` has no
        `of`/value to select (`_prepare` rejects a non-default `of` alongside it), so
        this mirrors only the MASKING half of `_band_value_expr`, never its value-
        selection half -- deliberately a separate method rather than a branch inside
        `_band_value_expr`, since the two return different things (a boolean indicator
        vs. a masked value) for a fundamentally different question.
        """
        fine_col, paired_col = self.fine_cols[i], self.paired_cols[i]
        if self.where:
            rename_map = {self.fine_name: fine_col, self.coarse_alias: paired_col}
            rewritten = rewrite_named_columns(self.where, rename_map)
            cond = parse_value_expr(rewritten, allowed_columns=[fine_col, paired_col])
            return cond.fill_null(False)
        return pl.col(fine_col).is_not_null() & pl.col(paired_col).is_not_null()

    def _collapse_expr(self):
        n = len(self.band_sources)
        if self.group_step.aggregate == "count":
            # `count`'s own LOCAL collapse form -- see `_STREAMING_AGGREGATES`'s
            # docstring for why this is deliberately NOT a `HORIZONTAL_RASTER_AGGREGATES`
            # entry: a plain horizontal sum of each band's boolean validity indicator,
            # never null (every summand is a concrete True/False, never a Polars null),
            # so "zero contributing bands" surfaces as a real 0 here -- the "outside the
            # pairing crosswalk's domain" nodata case is NOT decided by this expression
            # at all; it needs `pairing_valid`, which only the per-window callers have
            # (see `_count_zone_gate`, applied by every caller of this method).
            if n == 0:
                return pl.lit(0, dtype=pl.Int32).alias("value")
            valid_exprs = [self._band_valid_expr(i).cast(pl.Int32) for i in range(n)]
            return pl.sum_horizontal(valid_exprs).alias("value")

        value_exprs = [self._band_value_expr(i) for i in range(n)]
        collapsed = horizontal_aggregate_expr(self.group_step.aggregate, value_exprs, None)
        return collapsed.fill_null(self.nodata_value).alias("value")

    def _band_valid_mask(self, i, fine_flat, paired_flat):
        """Band `i`'s boolean valid mask, as a flat NumPy array -- the numpy-accumulator
        counterpart to `_band_valid_expr` (mirrors `_band_masked_value`'s own relationship
        to `_band_value_expr`: evaluated over a throwaway two-column frame instead of the
        full wide one, so `_accumulate_paired_window` never holds more than one band's
        arrays alive at once). See `_band_masked_value`'s docstring for why `fill_nan(None)`
        is needed before evaluating either expression over `fine_flat`/`paired_flat`.
        """
        fine_col, paired_col = self.fine_cols[i], self.paired_cols[i]
        df = pl.DataFrame({fine_col: fine_flat, paired_col: paired_flat})
        df = df.with_columns([pl.col(c).fill_nan(None) for c in (fine_col, paired_col)])
        result = df.select(self._band_valid_expr(i).alias("v"))
        return result["v"].to_numpy()

    def _band_masked_value(self, i, fine_flat, paired_flat):
        """Band `i`'s masked collapse value ALONE, as a flat float64 NumPy array (`NaN`
        standing in for Polars null -- the same substitution `fill_nan(None)` undoes on
        the way in and `to_numpy()` reintroduces on the way out). This is exactly what
        band `i` contributes to `_collapse_expr`'s wide `horizontal_aggregate_expr` call,
        evaluated over a throwaway two-column, single-band frame instead of the full
        `2 * len(band_sources)`-column one, so `_accumulate_paired_window` never holds
        more than one band's worth of source arrays alive at once. `fine_flat`/
        `paired_flat` carry `NaN` for nodata/pairing-miss on entry (the same convention
        `_write_paired_notebook_raster`'s window loop already reads them in); converting
        those to true Polars nulls with `fill_nan(None)` before evaluating the expression
        matters because Polars does NOT treat float `NaN` as null on its own (`NaN > x`
        is `True`, `NaN.is_null()` is `False`) -- skipping this step would silently change
        which pixels the where-clause and the nodata-mask treat as missing.
        """
        fine_col, paired_col = self.fine_cols[i], self.paired_cols[i]
        df = pl.DataFrame({fine_col: fine_flat, paired_col: paired_flat})
        df = df.with_columns([pl.col(c).fill_nan(None) for c in (fine_col, paired_col)])
        result = df.select(self._band_value_expr(i).alias("v"))
        return result["v"].to_numpy().astype("float64")


# `_validate_paired_group_aggregate`'s own docstring for the defense-in-depth check that
# used to be missing here -- `group_step.aggregate == "count"` used to reach the wide
# path today only because nothing rejected it for this shape, and `horizontal_aggregate_
# expr` raised a bare `KeyError` for it; that gap is what this whole change closes).
#
# Deliberately a SEPARATE frozenset from `_STREAMING_AGGREGATES` just below, even though
# the two happen to hold identical values today -- they answer different questions.
# `_SUPPORTED_PAIRED_AGGREGATES` is the durable "does `PairedAnalysis` implement this
# aggregate AT ALL" invariant `_validate_paired_group_aggregate` enforces; `_STREAMING_
# AGGREGATES` is a dispatch knob a caller (or a test) can legitimately flip per-run to
# force everything onto `_collapse_paired_window_wide` (see that set's own docstring and
# `TestPairedAnalysisCountConsistency.test_accumulator_matches_wide_fallback`). Validating
# against `_STREAMING_AGGREGATES` directly would make that monkeypatch ALSO disable
# support for every aggregate for the duration of the patch, rather than merely routing
# it onto the wide path -- exactly the trap that test hit during development.
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
    check, that aggregate would compile clean and fall through to
    `_collapse_paired_window_wide`, whose `horizontal_aggregate_expr` call raises a bare
    `KeyError` for anything `HORIZONTAL_RASTER_AGGREGATES` doesn't know -- exactly the
    latent defect `count` itself used to hit before this change. Converts that failure
    mode into a plan-time `PipelineError` (a 400) for anything that reaches it.
    """
    if group_step.aggregate not in _SUPPORTED_PAIRED_AGGREGATES:
        raise PipelineError(
            f"not yet supported in analysis pipelines: this paired pipeline shape's "
            f"`group` step aggregate {group_step.aggregate!r} -- only "
            f"{sorted(_SUPPORTED_PAIRED_AGGREGATES)} are implemented for a paired "
            "(joined) pipeline's per-pixel band collapse"
        )


# The dispatch knob `_write_paired_notebook_raster` reads to pick `_accumulate_paired_
# window` (streaming) vs. `_collapse_paired_window_wide` (wide-frame) for a given
# `group_step.aggregate` -- equal to `_SUPPORTED_PAIRED_AGGREGATES` today (every
# supported aggregate has a streaming implementation), but kept as its own name/frozenset
# rather than reusing that one directly, precisely so a caller (or a test -- see
# `TestPairedAnalysisCountConsistency.test_accumulator_matches_wide_fallback`, and `/tmp/
# pipetest/scale_probe.py`'s identical trick for scale measurement) can monkeypatch THIS
# set alone to force every aggregate onto the wide path without also disabling
# `_validate_paired_group_aggregate`'s support check for the duration of the patch.

_STREAMING_AGGREGATES = _SUPPORTED_PAIRED_AGGREGATES


def _count_zone_gate(plan, flat_values, pairing_valid):
    """`count`'s zero-vs-nodata split, applied identically after EITHER collapse path
    (`_accumulate_paired_window`/`_collapse_paired_window_wide`/`_paired_notebook_tile`)
    computes a flat per-pixel band tally -- a no-op for every other aggregate.

    `pairing_valid` (this pixel's own membership in the pairing crosswalk -- the only
    "is this pixel even in this plan's output domain at all" signal a `PairedAnalysis` has;
    see the module docstring's discussion of why this shape has no separate output-zone
    crosswalk the way `scan_paired` does) decides nodata; every in-zone pixel's own
    tally, INCLUDING a genuine 0 (every band nodata, pairing-missed, or condition-failed),
    is left untouched. This is the opposite of every OTHER aggregate's `no_contributor`
    fallback (`horizontal_aggregate_expr`'s `no_contributor_at`), which cannot distinguish
    "out of domain" from "in domain but nothing contributed" from column nullity alone
    and so must treat both the same (nodata) -- `count` can tell the two apart because it
    has `pairing_valid` in hand, and telling them apart matters: "zero contributing bands"
    is `count`'s own correct answer for an in-domain pixel, not an edge case standing in
    for an undefined one (see `raster_source.guarded_aggregate`'s identical `count`
    bypass for the flat API's vertical/zonal count, and `PairedAnalysis._band_valid_expr`).
    """
    if plan.group_step.aggregate != "count":
        return flat_values
    return np.where(pairing_valid, flat_values, plan.nodata_value)


def _collapse_paired_window_wide(plan, fine_srcs, window, pairing_ids, pairing_valid, shape, collapse_expr):
    """The original per-window collapse: every fine band and its paired lookup read
    into one `2 * len(band_sources)`-column frame, then `plan._collapse_expr()`
    evaluated once across all of it. Kept verbatim (moved, not rewritten) as the
    fallback for any `group_step.aggregate` outside `_STREAMING_AGGREGATES` -- see that
    set's own docstring. Also the path the TILE renderer always uses (`_paired_notebook_
    tile` evaluates `plan._collapse_expr()` the same way, inline, over a full tile frame
    rather than calling this function directly) and the one a test can force for every
    aggregate, `count` included, by monkeypatching `_STREAMING_AGGREGATES` to empty --
    see `TestPairedAnalysisCountConsistency` for why that matters: `count`'s wide form here
    and its streaming form in `_accumulate_paired_window` must produce bit-identical
    output, and `_count_zone_gate`'s call at the end of each is what that proof rests on.
    """
    cols = {}
    for i, (path, band) in enumerate(plan.band_sources):
        src = fine_srcs[path]
        band_data = src.read(band, window=window)
        flat = band_data.ravel().astype("float64")
        nodata = src.nodata
        if nodata is not None:
            flat = np.where(flat == nodata, np.nan, flat)
        cols[plan.fine_cols[i]] = flat

        lookup = plan.lookup_arrays.get(plan.calendar_values[i], plan.lookup_arrays.get(None))
        cols[plan.paired_cols[i]] = _paired_lookup_values(lookup, pairing_ids, pairing_valid)

    df = pl.DataFrame(cols)
    df = df.with_columns([pl.col(c).fill_nan(None) for c in plan.fine_cols + plan.paired_cols])
    result = df.select(collapse_expr)
    values = _count_zone_gate(plan, result["value"].to_numpy(), pairing_valid)
    return values.reshape(shape)


def _accumulate_paired_window(plan, fine_srcs, window, pairing_ids, pairing_valid, shape):
    """Per-band streaming counterpart to `_collapse_paired_window_wide`: folds one
    band's masked value (`plan._band_masked_value` -- the exact per-band piece
    `_collapse_expr` itself would compute) into a running accumulator, so at most one
    band's fine/paired arrays are ever alive at once instead of all
    `len(band_sources)` of them plus the wide Polars frame built from them. Bit-
    identical to `horizontal_aggregate_expr`'s own null-skipping reduction (see
    raster_source.py): `min`/`max` seed their running accumulator at +-inf so a pixel no
    band ever validates for never updates it; `sum` seeds at 0 and only adds a band's
    value where that band is valid (Polars' own `sum_horizontal` treats a null band the
    same way -- as a zero contribution, not as excluding the row); `mean` accumulates a
    running sum and a running per-pixel valid-band COUNT and divides at the end, which is
    what makes it differ from `sum` (dividing by contributor count, not by
    `len(band_sources)`). `any_valid` -- OR'd across every band -- is this function's
    counterpart to `horizontal_aggregate_expr`'s `no_contributor_at` (`all_horizontal`
    over the per-band null masks): wherever no band ever validated, the pixel gets the
    plan's own `nodata_value` instead of whatever the (otherwise untouched, +-inf/0)
    accumulator seed happens to hold.

    `count` is handled separately from the `any_valid`-gated aggregates above: it folds
    `plan._band_valid_mask` (the exact per-band piece `_collapse_expr`'s `count` branch
    itself would compute, mirroring how `masked`/`_band_masked_value` works for the other
    four) into a plain running integer tally rather than a `min`/`max`/`sum`-style
    accumulator, and is gated by `_count_zone_gate`'s `pairing_valid` check, NOT by
    `any_valid` -- see that function's own docstring for why those two gates disagree on
    purpose (a real 0 tally is `count`'s correct answer for an in-domain pixel; the
    `any_valid`-gated aggregates have no equivalent "real 0" to protect).
    """
    aggregate = plan.group_step.aggregate
    flat_shape = pairing_ids.shape

    if aggregate == "count":
        tally = np.zeros(flat_shape, dtype="int64")
        for i, (path, band) in enumerate(plan.band_sources):
            src = fine_srcs[path]
            band_data = src.read(band, window=window)
            fine_flat = band_data.ravel().astype("float64")
            nodata = src.nodata
            if nodata is not None:
                fine_flat = np.where(fine_flat == nodata, np.nan, fine_flat)

            lookup = plan.lookup_arrays.get(plan.calendar_values[i], plan.lookup_arrays.get(None))
            paired_flat = _paired_lookup_values(lookup, pairing_ids, pairing_valid)

            tally += plan._band_valid_mask(i, fine_flat, paired_flat).astype("int64")

        return _count_zone_gate(plan, tally, pairing_valid).reshape(shape)

    if aggregate == "min":
        acc = np.full(flat_shape, np.inf, dtype="float64")
    elif aggregate == "max":
        acc = np.full(flat_shape, -np.inf, dtype="float64")
    else:  # sum / mean
        acc = np.zeros(flat_shape, dtype="float64")
    count = np.zeros(flat_shape, dtype="int64") if aggregate == "mean" else None
    any_valid = np.zeros(flat_shape, dtype=bool)

    for i, (path, band) in enumerate(plan.band_sources):
        src = fine_srcs[path]
        band_data = src.read(band, window=window)
        fine_flat = band_data.ravel().astype("float64")
        nodata = src.nodata
        if nodata is not None:
            fine_flat = np.where(fine_flat == nodata, np.nan, fine_flat)

        lookup = plan.lookup_arrays.get(plan.calendar_values[i], plan.lookup_arrays.get(None))
        paired_flat = _paired_lookup_values(lookup, pairing_ids, pairing_valid)

        masked = plan._band_masked_value(i, fine_flat, paired_flat)
        valid = ~np.isnan(masked)
        any_valid |= valid

        if aggregate == "min":
            acc = np.minimum(acc, np.where(valid, masked, np.inf))
        elif aggregate == "max":
            acc = np.maximum(acc, np.where(valid, masked, -np.inf))
        else:
            acc = acc + np.where(valid, masked, 0.0)
            if aggregate == "mean":
                count += valid

    if aggregate == "mean":
        safe_count = np.where(count == 0, 1, count)
        acc = acc / safe_count

    result = np.where(any_valid, acc, plan.nodata_value)
    return result.reshape(shape)


def _write_paired_notebook_raster(plan: PairedAnalysis, out_path, **profile_kwargs):
    """Direct rasterio block-window loop: read every fine band and the pairing
    crosswalk over the SAME window, look up each pixel's paired coarse value, collapse
    per pixel, and write the collapsed window straight back at its own position --
    never materializing a full-extent array of any kind. See the module docstring for
    why `scan_paired` cannot drive this instead.

    The per-window collapse itself dispatches on `group_step.aggregate`:
    `_accumulate_paired_window` (a per-band streaming accumulator -- see its own
    docstring) for every aggregate it covers, `_collapse_paired_window_wide` (the
    original wide-frame collapse, unchanged) for anything it doesn't -- see
    `_STREAMING_AGGREGATES`.
    """
    if not plan.band_sources:
        return False

    with ExitStack() as stack, raster_gdal_env():
        ref_path, _ref_band = plan.band_sources[0]
        ref_src = stack.enter_context(rio.open(ref_path))
        profile = ref_src.profile.copy()
        block_height, block_width = ref_src.block_shapes[0]
        itemsize = np.dtype(plan.out_dtype).itemsize
        n_units = max(len(plan.band_sources) * 2, 1)
        blocks_per_batch = raster_blocks_per_batch(n_units, block_height, block_width, itemsize)
        window_list = list(_batched_windows(ref_src, blocks_per_batch))

        fine_srcs = {path: stack.enter_context(rio.open(path)) for path in {p for p, _ in plan.band_sources}}
        pairing_src = stack.enter_context(rio.open(str(plan.pairing_crosswalk_path)))

        profile.update(count=1, dtype=plan.out_dtype, nodata=plan.nodata_value, compress="deflate")
        profile.update(profile_kwargs)
        dst = stack.enter_context(rio.open(str(out_path), "w", **profile))

        streaming = plan.group_step.aggregate in _STREAMING_AGGREGATES
        collapse_expr = None if streaming else plan._collapse_expr()

        for window in window_list:
            pairing_raw = pairing_src.read(1, window=window)
            shape = pairing_raw.shape
            pairing_ids = pairing_raw.ravel().astype("int64")
            pairing_nodata = pairing_src.nodata
            if pairing_nodata is not None:
                pairing_valid = pairing_ids != int(pairing_nodata)
            else:
                pairing_valid = np.ones(pairing_ids.shape, dtype=bool)

            if streaming:
                values = _accumulate_paired_window(plan, fine_srcs, window, pairing_ids, pairing_valid, shape)
            else:
                values = _collapse_paired_window_wide(
                    plan, fine_srcs, window, pairing_ids, pairing_valid, shape, collapse_expr
                )
            dst.write(values.astype(plan.out_dtype), 1, window=window)

    return True


async def _paired_notebook_tile(plan: PairedAnalysis, x, y, z, format, tile_size=256):
    """Mirrors `RasterGridAnalysis.get_tile`'s `WarpedVRT`-based shape, over every fine band plus
    the pairing crosswalk warped onto the requested tile grid, then the same per-pixel
    collapse `_write_paired_notebook_raster` uses.
    """
    if not plan.band_sources:
        return None

    tile_bounds = BoundingBox(*mercantile.xy_bounds(x, y, z))

    with ExitStack() as stack:
        fine_srcs = [stack.enter_context(rio.open(path)) for path, _ in plan.band_sources]
        pairing_src = stack.enter_context(rio.open(str(plan.pairing_crosswalk_path)))

        ref = fine_srcs[0] if fine_srcs else pairing_src
        dataset_bounds = BoundingBox(*transform_bounds(ref.crs, "EPSG:3857", *ref.bounds))
        if disjoint_bounds(dataset_bounds, tile_bounds):
            return None

        tile_transform = from_bounds(*tile_bounds, tile_size, tile_size)

        cols = {}
        for i, ((path, band), src) in enumerate(zip(plan.band_sources, fine_srcs)):
            fine_nodata = src.nodata if src.nodata is not None else -9999
            vrt = stack.enter_context(
                WarpedVRT(
                    src,
                    crs="EPSG:3857",
                    transform=tile_transform,
                    width=tile_size,
                    height=tile_size,
                    nodata=fine_nodata,
                    add_alpha=False,
                )
            )
            data = vrt.read(band).ravel().astype("float64")
            data = np.where(data == fine_nodata, np.nan, data)
            cols[plan.fine_cols[i]] = data

        pairing_nodata = pairing_src.nodata if pairing_src.nodata is not None else -1
        pairing_vrt = stack.enter_context(
            WarpedVRT(
                pairing_src,
                crs="EPSG:3857",
                transform=tile_transform,
                width=tile_size,
                height=tile_size,
                nodata=pairing_nodata,
                add_alpha=False,
            )
        )
        pairing_raw = pairing_vrt.read(1)
        pairing_ids = pairing_raw.ravel().astype("int64")
        pairing_valid = pairing_ids != int(pairing_nodata)

        for i in range(len(plan.band_sources)):
            lookup = plan.lookup_arrays.get(plan.calendar_values[i], plan.lookup_arrays.get(None))
            cols[plan.paired_cols[i]] = _paired_lookup_values(lookup, pairing_ids, pairing_valid)

    df = pl.DataFrame(cols)
    df = df.with_columns([pl.col(c).fill_nan(None) for c in plan.fine_cols + plan.paired_cols])
    result = df.select(plan._collapse_expr())
    values = _count_zone_gate(plan, result["value"].to_numpy(), pairing_valid)
    values = values.reshape(tile_size, tile_size)

    legend = _legend_for(plan.color_map)
    color = legend.colorize_pixels(values[np.newaxis, :, :])
    image = Image.fromarray(color[:, :, :3] if format == "jpg" else color)
    with BytesIO() as memfile:
        image.save(memfile, format=("JPEG" if format == "jpg" else "PNG"))
        return memfile.getvalue()


# ------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------


