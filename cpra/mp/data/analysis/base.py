"""`BaseAnalysis`: the template-method base class for a data-API query against one
dataset (the former `api/dataset.py`, split by grid type -- see `analysis/vector.py`'s
`VectorGridAnalysis` and `analysis/raster.py`'s `RasterGridAnalysis`, and the
`create_analysis` factory in `analysis/__init__.py` that picks between them).

Nothing in `cpra.mp.data.analysis` may import FastAPI (see
`tests/api/test_analysis_boundary.py`), so this module raises `AnalysisError` wherever
`Dataset` used to raise `HTTPException` directly. `api/__init__.py` registers a single
FastAPI exception handler that translates an uncaught `AnalysisError` into the same
status code/detail an `HTTPException` would have produced; `api/tabular.py`'s
`PolarsStreamingResponse` (whose write runs on a background thread, entirely outside
FastAPI's own exception handling) reads `.status_code`/`.detail` off the caught exception
directly, which is why `AnalysisError` carries them as plain attributes rather than
relying on the translation point.

**The staged `apply_params` flow.** A `BaseAnalysis` instance mutates `self.lf` (a Polars
LazyFrame) through five ordered stages, run once by `apply_params`:

1. **filter** (`_apply_filter_params`) -- `where`'s dimension conjuncts become a
   `.sql()`/`.filter()` on `self.lf`; its value conjunct is stashed on
   `self._value_where` for later stages to apply (vector: immediately, as a mask or a
   filter; raster: per band, once pixels are read). `options.grid_cell`, if given, is
   applied here too, via the `_apply_grid_cell` hook.
2. **calculate** (`_apply_calculate_params`) -- the generic `calculate` entries
   (expression/where pairs) conditionally rewrite `value` in place (vector), or
   are validated and stashed on `self._calculate_plans` for the pixel-read paths to
   apply per band (raster, whose `self.lf` at this point is still the band-metadata
   table, not pixels -- see `RasterGridAnalysis._simple_band_collapse_inputs`).
3. **group** (`_apply_group_params`) -- `group_by`/`group_aggregate` collapse rows via the
   `_collapse_group` hook (vector: dimension-collapse across rows per grid_cell, folding
   through `analysis.fold` when the collapse is cliff-shaped; raster: records which band
   rows belong to which output group in `self._band_groups`, collapsed later against
   actual pixels).
4. **transform** (`_apply_transform_params`) -- `summarize_by`/`summarize_aggregate`
   convert to another grid's cells via the `_apply_transform_params_impl` hook (vector: a
   crosswalk-table join; raster: a zonal reduction against a crosswalk raster, which is
   also where a raster's `self.lf` first becomes a real per-pixel-derived tabular frame).
5. **sort/select** (`_apply_sort_params`) -- `sort`/`select` order and narrow the final
   columns.

**Hook contract.** Each hook below is declared abstract here (raising
`NotImplementedError`) and overridden once per subclass; every override must leave
`self.lf` in the same SHAPE the base flow expects next (long format: dimension columns,
`grid_cell`, `value`), since the following stage runs unconditionally regardless of which
subclass produced it.

- `async _apply_grid_cell(self)` -- apply `options.grid_cell` (IDs or coordinates) to
  `self.lf`. Vector: filters to matching `grid_cell` IDs, resolving coordinates to IDs
  first if needed, and records them on `self.grid_cell_ids`. Raster: samples the raster at
  each coordinate and rebuilds `self.lf` as a long-format (grid_cell/value) frame labeled
  by coordinate, since a raster has no cell IDs of its own.
- `_collapse_group(self, group_by_cols)` -- apply `group_by`/`group_aggregate` (already
  validated not to be `count_by_value`, and, on input, `group_by_cols` is a list of
  dimension names, empty meaning "aggregate with no grouping" -- see
  `_get_group_by_columns`). Vector: a real `group_by(...).agg(...)` (or
  `analysis.fold.fold_group_by` when cliff-shaped), DROPPING dimension columns left out
  of the group (they no longer have a value, so they no longer have a column -- see the
  method's own comment). Raster: no pixels are read here -- this only records the row
  groups (`self._band_groups`) that a later stage (transform, tile, or write) collapses
  against actual band data.
- `async _apply_transform_params_impl(self)` -- apply `summarize_by`/`summarize_aggregate`,
  converting to another grid's cells. Called only after `_validate_summarize_params`
  confirms both parameters (or neither) were given; a hook implementation returns
  immediately if neither was. Must leave `self.lf` in the shared summarize output shape:
  dimension columns, a `grid_cell` column holding the destination cell ID, a `value`
  column, and a `geographic_unit` column naming the destination grid.

**Output surface.** Besides `apply_params`, every subclass exposes: `materialize` (this
class's own duck-typed reimplementation of `api.sources.LazyFrameSource.materialize` --
`BaseAnalysis` cannot subclass `LazyFrameSource` itself, since that lives in `api/` and
raises `HTTPException`; `stream_tabular`/`write_tabular`/`tabular_kwargs` are the same
story), `output_type`, `estimate_data_size`, `legend`, and grid-specific tile/write
methods (`get_tile`/`write_raster`/`validate_tile` for a raster response,
`get_vector_tile`/`validate_vector_tile` for a vector tile response) -- present on
whichever subclass actually supports them; `validate_tile`/`validate_vector_tile`
themselves need no per-subclass override, since both check `self.output_type` (which can
diverge from `self.grid.grid_type` -- see `output_type`'s own docstring) rather than which
subclass `self` is.
"""

import re

import polars as pl
from polars.exceptions import ColumnNotFoundError

from cpra.mp.data import config
from cpra.mp.data.analysis.expressions import (
    ExpressionError,
    classify_where,
    compile_calculate_entries,
    parse_value_expr,
    value_predicate_mask,
)
from cpra.mp.data.analysis.types import DataFormat, OutputType
from cpra.mp.data.analysis.utils import get_lazy_frame, timing, unpack_values
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.legends import LEGENDS, Legend
from cpra.mp.data.utils import get_input_data_frame, required_columns
from cpra.mp.data.variables import VARIABLES, Variable

DIMENSION_CHOICES = list(DIMENSIONS.keys())
SORT_CHOICES = DIMENSION_CHOICES + [f"-{dim_name}" for dim_name in DIMENSION_CHOICES]


# -- re-entrant stage cores ---------------------------------------------------------
#
# Each function below is the same Polars logic `_apply_filter_params`/`_apply_calculate_
# params` used to run only against `self.lf`/`self.options`, pulled out to take its own
# arguments and return a new `lf` instead of mutating an instance -- so a caller walking a
# `list[ops.Op]` in DECLARED order (`vector.VectorGridAnalysis.apply_steps`, the ops
# interpreter) can invoke exactly the same stage logic once per op, in whatever order the
# pipeline actually declared, rather than the fixed five-stage order `apply_params` always
# runs in. Each one raises the plain `ExpressionError`/`FoldError` its Polars-adjacent
# helpers already raise -- never `AnalysisError` -- since these are library functions with
# no `self._error` to call; every caller (a `BaseAnalysis` stage method, or the
# interpreter) catches and converts at its own boundary, exactly like `_apply_filter_
# params`/`_apply_calculate_params` already did for the logic now living here.
def apply_dimension_filter(lf, where):
    """Split `where` into DIMENSION and VALUE conjuncts (`expressions.classify_where`) and
    apply the dimension half to `lf` via the same unrestricted `lf.sql()` path `where` has
    always used. Returns `(lf, value_where, has_dimension_conjunct)`: `value_where` is the
    combined value-conjunct string (`None` if `where` had none), left for the caller to
    apply separately, since MASK-vs-DROP (see `apply_value_filter`) is a policy decision
    this function -- which only ever handles the columnar dimension half -- does not make.
    `has_dimension_conjunct` is `bool(dimension_conjuncts)`, the signal `_has_selective_
    dimension_filter` needs to decide whether a downstream `group_by`/`project` is
    cliff-shaped (see that method's docstring).

    A falsy `where` (`None` or `""`) is a no-op: returns `(lf, None, False)` unchanged,
    matching `_apply_filter_params`'s own `if self.options.where:` guard.
    """
    if not where:
        return lf, None, False
    dimension_conjuncts, value_conjuncts = classify_where(where)
    has_dimension_conjunct = bool(dimension_conjuncts)
    if dimension_conjuncts:
        combined = " and ".join(f"({conjunct})" for conjunct in dimension_conjuncts)
        lf = lf.sql(f"select * from self where {combined}")
    value_where = None
    if value_conjuncts:
        value_where = " and ".join(f"({conjunct})" for conjunct in value_conjuncts)
    return lf, value_where, has_dimension_conjunct


def apply_value_filter(lf, value_where, *, mask):
    """Apply a combined value-conjunct predicate (`value_where`, already reduced to one
    string by `apply_dimension_filter` or an equivalent split) to `lf`: as a MASK
    (`mask=True`, `pl.when(...).then(value).otherwise(None)` -- nulls a non-matching value
    rather than dropping its row, composing with a downstream aggregate's own null-
    skipping behavior) or a row DROP (`mask=False`, `lf.filter(...)`). `_apply_filter_
    params`'s docstring explains exactly when the OLD (fixed five-stage) engine picks
    which: mask for an aggregating request (`_is_aggregating()`), drop otherwise. The
    ops interpreter (`vector.apply_steps`) always passes `mask=False` -- see that
    method's own docstring for why a straight drop, not that conditional, is the only
    semantically coherent choice once filter/group/project run in their own declared
    positions instead of a fixed order.

    A `None` `value_where` (no value conjunct) is a no-op: returns `lf` unchanged.
    """
    if value_where is None:
        return lf
    mask_expr = value_predicate_mask(value_where, lf.collect_schema().names())
    if mask:
        return lf.with_columns(pl.when(mask_expr).then(pl.col("value")).otherwise(None).alias("value"))
    return lf.filter(mask_expr)


def apply_calculate_plans(lf, calculate_plans):
    """Apply already-validated `expressions.CalculatePlan`s to a vector/tabular `lf`,
    sequentially in list order (entry N+1 sees entry N's output) -- the re-entrant core of
    `_apply_calculate_params`'s vector-side rewrite (the raster band-metadata early return
    there has no equivalent here: this function always assumes `lf` already carries a real
    `value` column, which is exactly what `_apply_calculate_params` checks with `_is_
    raster_metadata_table()` before ever calling the logic now living here). A missing
    value stays missing (`is_not_null()` guards every rewrite) and a row not matching a
    plan's own `where` passes through unchanged, for the same reasons `_apply_calculate_
    params`'s docstring gives.

    An empty `calculate_plans` is a no-op: returns `lf` unchanged.
    """
    if not calculate_plans:
        return lf
    schema_names = set(lf.collect_schema().names())
    for plan in calculate_plans:
        condition = pl.col("value").is_not_null()
        if plan.dimension_where is not None:
            dimension_expr = pl.sql_expr(plan.dimension_where)
            missing = set(dimension_expr.meta.root_names()) - schema_names
            if missing:
                raise ExpressionError(
                    f"{plan.label}: where references column(s) not present in this "
                    f"dataset: {sorted(missing)} -- available columns: {sorted(schema_names)}"
                )
            condition = condition & dimension_expr.fill_null(False)
        if plan.value_where is not None:
            condition = condition & value_predicate_mask(plan.value_where, ["value"])
        new_value = parse_value_expr(plan.expression, ["value"])
        lf = lf.with_columns(pl.when(condition).then(new_value).otherwise(pl.col("value")).alias("value"))
    return lf


class AnalysisError(Exception):
    """Raised throughout `cpra.mp.data.analysis` in place of FastAPI's `HTTPException`
    (see this module's docstring for why). Carries the same `status_code`/`detail` shape.
    """

    def __init__(self, detail, status_code=400):
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


class BaseAnalysis:
    def __init__(
        self,
        dataset_dimensions,
        filter_dimensions,
        options,
        base_path=None,
    ):
        self.dataset_dimensions = dataset_dimensions
        self.filter_dimensions = filter_dimensions
        self.options = options
        self.dataset_bounds = None
        self.grid_cell_ids = None
        self.summarize_grid_cell_ids = None
        # `RasterGridAnalysis`-only bookkeeping for a `group_by` step (a vector frame
        # never sets any of these three -- its own `_collapse_group` just reassigns
        # `self.lf` directly, since a row-per-grid_cell frame has no positional band
        # columns to keep aligned). `self.lf` for a raster analysis starts as the
        # per-band metadata table (one row per selected (path, band)); the FIRST
        # `_collapse_group` call snapshots that table into `_raw_band_frame` before
        # reassigning `self.lf` to the collapsed (dimension-only) metadata, since every
        # later pixel-read still needs the original (path, band) rows to build its scan
        # even once `self.lf` itself has moved on. `_band_groups` is, for each CURRENT
        # row of `self.lf`, the list of ORIGINAL `_raw_band_frame` row positions that
        # collapsed into it -- composed across however many `group`/post-group
        # dimension-filter steps have run so far, so it stays correct through more than
        # one of either (see `RasterGridAnalysis._collapse_group`). `_band_ops` is the
        # ordered pixel-VALUE replay recipe (calculate/group/filter entries, one per
        # step actually declared, in declared order) a pixel-read consumer replays
        # against the raw scan once at least one group has run -- see
        # `RasterGridAnalysis.apply_band_ops` and that class's three pixel-read
        # methods.
        self._raw_band_frame = None
        self._band_groups = None
        self._band_ops = []
        # Set by `_apply_filter_params`, the first step `apply_params` runs, from
        # `where`'s value conjuncts (see `expressions.classify_where`) -- the internal
        # successor to the retired `Options.value_where` field. `None` means `where`
        # contained no value conjunct; every other consumer of the old `value_where`
        # field (`RasterGridAnalysis._apply_transform_params_impl`,
        # `_summarize_zone_stats_plugin`, `_summarize_count_by_value_plugin`) reads this
        # attribute instead now.
        self._value_where = None
        # Set by `_apply_calculate_params`: the validated `CalculatePlan`s for this
        # request (the generic `calculate` entries), in application order. Empty when
        # the request has no calculate parameters at all. A vector frame consumes them
        # immediately inside `_apply_calculate_params`; the raster pixel-read paths
        # (tile/write/summarize) flush it into `_band_ops` and read it back, per band,
        # via `_simple_band_collapse_inputs`.
        self._calculate_plans = []
        # Also set by `_apply_filter_params`: whether `where` produced at least one
        # DIMENSION conjunct, i.e. whether it actually did anything that could shrink
        # the row count reaching a group_by -- see `_has_selective_dimension_filter`,
        # which used to just check `options.where is not None` and is now wrong to,
        # since `where` may hold only a value conjunct (a mask, not a row filter).
        self._where_has_dimension_conjunct = False
        # Set by `planner._compile_single_input_chain`'s vector fast path to
        # `ops.compile_ops(pipeline)`'s
        # `list[Op]` -- never by this constructor itself, and never by anything on the
        # flat (non-pipeline) request surface. `None` (the default here) means "no op IR
        # was attached", the signal `VectorGridAnalysis.apply_params`'s override reads to
        # decide between the interpreter (`apply_steps`) and this class's own fixed
        # five-stage flow -- see that override's docstring.
        self._pipeline_ops = None

        self.lf = get_lazy_frame(self.dataset_dimensions, self.filter_dimensions, base_path)
        self.grid = GRIDS[self.dataset_dimensions.grid]

    def _error(self, msg, status_code=400):
        """Raise an exception."""
        raise AnalysisError(msg, status_code=status_code)

    # -- output surface: materialize/tabular (LazyFrameSource-shaped, duck-typed) --------

    def materialize(self, method="collect", *args, **kwargs):
        """Materialize the LazyFrame with error handling."""
        try:
            return getattr(self.lf, method)(*args, **kwargs)
        except ColumnNotFoundError as err:
            raise AnalysisError(str(err), status_code=400)
        except FileNotFoundError:
            raise AnalysisError("Dataset does not exist", status_code=404)

    def tabular_kwargs(self, format):
        """Round CSV output to the variable's `float_precision`, when it sets one.

        CSV only: it's the one writer polars gives a `float_precision`, and the only format
        here that renders floats as text at all -- the binary formats hand over the stored
        value and leave rounding to whatever reads them.

        This deliberately applies to every float column in the response, not just `value`.
        Nothing else in a tabular response is a float today (dimensions are integers,
        strings, or dates), and a bare `.round()` on `value` would be wrong anyway: it
        would change the number in the binary formats too, where the request asked for the
        stored value.
        """
        precision = self.variable.float_precision
        if format == DataFormat.csv and precision is not None:
            return {"float_precision": precision}
        return {}

    @timing
    def stream_tabular(self, out_path, format, *args, **kwargs):
        sink_method = {
            "arrow": "sink_ipc",
            "csv": "sink_csv",
            "ndjson": "sink_ndjson",
            "parquet": "sink_parquet",
        }[format]
        # TODO: Set parquet compression and row group parameters.
        self.materialize(sink_method, out_path, *args, **kwargs)

    @timing
    def write_tabular(self, out_path, format, *args, **kwargs):
        write_method = {
            "arrow": "write_ipc",
            "csv": "write_csv",
            "ndjson": "write_ndjson",
            "parquet": "write_parquet",
        }[format]
        # TODO: Set parquet compression and row group parameters.
        df = self.materialize()
        getattr(df, write_method)(out_path, *args, **kwargs)

    # -- output surface: shared properties -----------------------------------------------

    @property
    def legend(self) -> Legend:
        color_map = self.options.color_map or ""

        if ":" in color_map:
            map_type, color_str = color_map.split(":", 1)

            if map_type not in ["continuous", "discrete"]:
                self._error(f"invalid color map type: {map_type}")

            color_strs = color_str.split(",")
            if map_type == "continuous":
                colors = color_strs[1::2]
                value_strs = color_strs[::2]
            else:
                colors = color_strs[::2]
                value_strs = color_strs[1::2]

            # TODO: Check that the color strings are valid.

            try:
                values = [float(val) for val in value_strs]
            except ValueError:
                self._error("invalid color map values")

            # Add a null upper bound value to discrete color maps.
            if map_type == "discrete":
                values += [None]

            if len(values) != len(colors):
                self._error("mismatched colors and values in color map")

            return Legend("custom", pl.DataFrame(dict(color=colors, upper_bound=values)), type=map_type)

        return LEGENDS.get(self.options.color_map) or self.variable.legend or LEGENDS["default"]

    @property
    def output_type(self) -> OutputType:
        """Get the output type for this dataset based on its parameters.

        Not simply `self.grid.grid_type`: a raster grid still answers as `vector` once
        `grid_cell` (pointwise sampling) or `summarize_by` (zonal summarize) narrows it
        to a tabular result, which is why `validate_tile`/`validate_vector_tile` check
        this property rather than which subclass `self` is.
        """
        if self.grid.grid_type == "vector" or self.options.grid_cell or self.options.summarize_by:
            return OutputType.vector
        return OutputType.raster

    @property
    def variable(self) -> Variable:
        return VARIABLES[self.dataset_dimensions.variable]

    def _is_raster_metadata_table(self):
        """Whether self.lf still holds the raster band-metadata table (dimension columns
        and a path column), as opposed to a tabular result already derived from raster
        pixels (via grid_cell sampling or summarize_by). Unlike `output_type`, which
        reflects the dataset's final response shape and is stable for the whole request,
        this reflects self.lf's shape at the point it's called, since summarize_by's
        raster-to-tabular conversion doesn't happen until the transform stage.
        """
        return self.grid.grid_type == "raster" and "path" in self.lf.collect_schema().names()

    def _is_aggregating(self):
        """Whether this request collapses rows via an aggregate -- `group_aggregate`
        (vector row-collapse) or `summarize_aggregate` (raster zonal / vector
        cross-grid summarize) -- as opposed to a PLAIN, non-aggregating tabular/tile
        response. Used by `_apply_filter_params` to decide whether a `where` value
        conjunct masks a non-matching value to null (composing with an aggregate's own
        null-skipping behavior) or drops the row outright (the only sensible behavior
        once there is no aggregate downstream to skip a null for -- see that method).
        """
        return self.options.group_aggregate is not None or self.options.summarize_aggregate is not None

    def validate_tile(self):
        """Validate that the tile is raster and will have a single band."""

        if self.output_type == OutputType.vector:
            self._error("invalid output type for tile: vector", status_code=400)

        if self._band_groups is not None:
            groups = len(self._band_groups)
        else:
            groups = self.lf.select(pl.len()).collect().item()
        if groups > 1:
            self._error(f"tile contains more than one band: {groups} bands found", status_code=400)

    def validate_vector_tile(self):
        """Validate that the tile is a vector grid with an available PMTiles file."""
        if self.output_type != OutputType.vector:
            self._error("invalid output type for vector tile: raster", status_code=400)

        grid_name = self.options.summarize_by or self.grid.name
        if not (config.TILES_PATH / f"{grid_name}.pmtiles").exists():
            self._error(f"no tile file for grid: {grid_name}", status_code=404)

    # -- stage 1: filter ------------------------------------------------------------------

    async def _apply_filter_params(self):
        """Apply filter parameters to the LazyFrame.

        `where` accepts both DIMENSION conditions and VALUE conditions (against the
        placeholder column `value`), AND-ed together in one string --
        `expressions.classify_where` splits it on top-level `AND` and sorts each
        conjunct by which kind it is (see that function's docstring for the full split/
        classify/reject contract, including why a conjunct referencing both kinds is
        rejected outright). Dimension conjuncts are recombined and handed to the same
        unrestricted `lf.sql()` path `where` has always used; value conjuncts are
        recombined into `self._value_where`, the internal successor to the retired
        `Options.value_where` field, for this method (vector) and every raster-side
        consumer (`RasterGridAnalysis._apply_transform_params_impl`,
        `_summarize_zone_stats_plugin`, `_summarize_count_by_value_plugin`) to read
        instead.

        A thin caller over `apply_dimension_filter`/`apply_value_filter`: the logic
        itself now lives in
        those two re-entrant, `self`-free functions so `vector.VectorGridAnalysis.
        apply_steps` (the ops interpreter) can invoke the identical Polars behavior once
        per `FilterOp`, in declared order, instead of once for the whole combined `where`
        string here. This method's OWN behavior is unchanged by that split -- it still
        runs once, against `self.options.where` as a whole, for every non-interpreter
        request (every flat request, and any pipeline the vector fast path doesn't
        select).
        """
        if self.options.where:
            try:
                self.lf, self._value_where, self._where_has_dimension_conjunct = apply_dimension_filter(
                    self.lf, self.options.where
                )
            except ExpressionError as exc:
                self._error(str(exc))

        # A vector frame already carries a real `value` column at this point, so a
        # `where` value conjunct can be applied directly here. A raster's `value`
        # doesn't exist yet -- `self.lf` is still the band-metadata table (path/band/
        # dimensions), one row per selected band, not one row per pixel -- so the
        # raster half of this same option is applied later, per band, once pixels are
        # actually read (see `RasterGridAnalysis._apply_transform_params_impl` /
        # `_summarize_zone_stats_plugin`).
        #
        # AGGREGATING requests (group_aggregate/summarize_aggregate, including
        # summarize_aggregate=count_by_value) mask a non-matching value to null rather
        # than dropping its row: that's what lets the predicate compose with whatever
        # aggregate the request goes on to apply via their EXISTING null-skipping
        # behavior (`guarded_aggregate`'s count()-based denominator, count_by_value's
        # own non-null filter, ...), rather than needing a second, predicate-aware code
        # path. Dropping the row instead would also erase the group/zone's siblings
        # that DO belong in the denominator for min/max/mean, and would make "count how
        # many values passed" indistinguishable from "count how many rows exist".
        #
        # PLAIN (non-aggregating) requests have no aggregate downstream to skip a null
        # for, so a masked-to-null value would just surface as a null in the response --
        # observably different from "this row failed the filter" and indistinguishable
        # from a genuine nodata value. There, the row is dropped outright instead (a
        # deliberate behavior change from the old value_where, which always masked).
        if self._value_where is not None and self.grid.grid_type == "vector":
            try:
                self.lf = apply_value_filter(self.lf, self._value_where, mask=self._is_aggregating())
            except ExpressionError as exc:
                self._error(str(exc))

        if self.options.grid_cell:
            await self._apply_grid_cell()

    async def _apply_grid_cell(self):
        """Apply the `grid_cell` filter to `self.lf`. See the module/class docstring's
        hook contract -- overridden per subclass; there is no grid-agnostic default.
        """
        raise NotImplementedError

    # -- stage 2: calculate -----------------------------------------------------------------

    def _resolve_calculate_entries(self):
        """The effective calculate entries for this request, as `(label, expression,
        where)` triples in application order: the generic `options.calculate` list,
        applied in list order (entry N+1 sees entry N's output).
        """
        generic = self.options.calculate or []
        return [
            (f"calculate.{i}", entry.expression, entry.where)
            for i, entry in enumerate(generic, start=1)
        ]

    def _apply_calculate_params(self):
        """Apply the calculate stage: expression/where entries that rewrite `value` for
        the rows/bands/pixels matching each entry's `where`, sequentially in entry order
        (entry N+1 sees entry N's output). Rows NOT matching a `where` pass through
        unchanged -- calculate never drops or masks data (that is Filter's job) -- and a
        missing value stays missing (`is_not_null()` guards every rewrite, so NoData in
        means NoData out regardless of what the expression could compute).

        Everything is validated here (`expressions.compile_calculate_entries`), before
        any pixel is read, for both grid types. A vector/tabular frame (including a
        raster already narrowed to a table by `grid_cell` sampling) is rewritten in
        place; for a raster whose `self.lf` is still the band-metadata table the
        validated plans are only stashed on `self._calculate_plans` -- the entry's
        dimension `where` selects BANDS (evaluated against the metadata rows by
        `RasterGridAnalysis._band_applicability`) and its expression/value-`where` are
        applied per band, against actual pixels, by every raster read path
        (`_summarize_zone_stats_plugin`, `_collapse_tile_bands`,
        `_write_raster_windowed`) via `_simple_band_collapse_inputs`.

        A thin caller over `apply_calculate_plans` for the vector/tabular rewrite
        itself -- pulled out so
        `vector.VectorGridAnalysis.apply_steps` (the ops interpreter) can invoke the
        identical Polars behavior once per `CalculateOp`, in declared order, instead of
        once for the whole `options.calculate` list here. Validation
        (`compile_calculate_entries`) and the raster-metadata-table early return stay here
        unchanged -- both are about THIS method's own entry source (`options.calculate`)
        and grid-type dispatch, neither of which the interpreter shares.
        """
        entries = self._resolve_calculate_entries()
        try:
            self._calculate_plans = compile_calculate_entries(entries)
        except ExpressionError as exc:
            self._error(str(exc))

        if not self._calculate_plans or self._is_raster_metadata_table():
            return

        try:
            self.lf = apply_calculate_plans(self.lf, self._calculate_plans)
        except ExpressionError as exc:
            self._error(str(exc))

    # -- stage 5: sort/select ---------------------------------------------------------------

    def _apply_sort_params(self):
        """Apply sort and select parameters to the LazyFrame."""
        if self.options.sort is not None:
            parts = unpack_values(self.options.sort, "sort", SORT_CHOICES)
            cols = [part.replace("-", "") for part in parts]
            desc = [part.startswith("-") for part in parts]
            self.lf = self.lf.sort(cols, descending=desc)

        if self.options.select is not None:
            required_cols = required_columns(
                get_input_data_frame(**dict(self.dataset_dimensions)),
                include_grid_cell=True,
                include_value=True,
            )
            select_cols = unpack_values(self.options.select, "select", required_cols)
            self.lf = self.lf.select(select_cols)

    # -- stage 3: group ---------------------------------------------------------------------

    def _get_group_by_columns(self):
        """Get a list of group_by column names.

        Empty, not a `[True]` literal sentinel (its pre-cleanup shape): a caller
        grouping by `True` gets a column literally named "literal" back from Polars
        (`df.group_by(True)` -- see `fold.py`'s `fold_group_by` docstring for the same
        naming quirk), which leaked into every collapse-to-cell/collapse-everything
        response as a meaningless `literal` boolean column. An empty list needs no such
        placeholder: `VectorGridAnalysis._collapse_group` always appends "grid_cell"
        itself before grouping, so `[]` still collapses to one row per grid_cell, and
        `RasterGridAnalysis._collapse_group`'s `group_by()` with no columns collapses
        to a single overall group -- both without inventing a key column.
        """
        if self.options.group_by is not None and self.options.group_aggregate is None:
            self._error("group_aggregate must be specified when group_by is used")

        if self.options.group_by is None:
            return []
        return unpack_values(self.options.group_by, "group_by", DIMENSION_CHOICES)

    def _has_selective_dimension_filter(self):
        """Whether this request already carries something that shrinks the row count
        reaching a vector group_by well below "the whole dataset" -- the signal
        `analysis.fold.should_fold_group_by` uses to skip the fold on an already-cheap
        query.

        `filter_dimensions` values are pushed down as Hive-partition filters inside
        `get_lazy_frame`/`read_parquet`, before `self.lf` exists at all -- the measured
        case where a `grid_cell` group-by drops from 9.8 GB to 285 MB. `options.grid_cell`
        (a real `.filter()` on specific cell IDs) and a `where` DIMENSION conjunct (an
        arbitrary SQL `.filter()`, applied to `self.lf` itself in `_apply_filter_params`)
        are the same kind of row-dropping filter, so they count too --
        `self._where_has_dimension_conjunct` (set by `_apply_filter_params`) is checked
        here rather than `self.options.where is not None`, since `where` may now hold
        only a VALUE conjunct, which never reaches this method as a row filter at all
        (see the next paragraph).

        A `where` VALUE conjunct and the `calculate` parameters are deliberately NOT
        included: wherever this method's caller
        (`_apply_group_params`, gating a vector `group_by`) applies, a value conjunct is
        always a `with_columns` mask, never a row-dropping `.filter()` -- masking is
        exactly what lets it compose with the group_by's own aggregate instead of
        changing its denominator (see `_apply_filter_params`) -- so it never changes how
        many rows reach the group-by, doesn't affect whether it's cliff-shaped, and
        (unlike a `.filter()`) doesn't block the fold's `.slice()` from reaching the scan
        either.
        """
        return (
            self.options.grid_cell is not None
            or self._where_has_dimension_conjunct
            or any(v is not None for v in dict(self.filter_dimensions).values())
        )

    def _apply_group_params(self):
        """Apply group and aggregate parameters to the LazyFrame or image.

        The shared preamble (`group_by_cols` resolution and the `count_by_value`
        rejection, neither of which differs by grid type) lives here; the actual
        collapse is the `_collapse_group` hook -- see the module/class docstring.
        """
        if self.options.group_by is None and self.options.group_aggregate is None:
            return

        if self.options.group_aggregate == "count_by_value":
            # count_by_value changes the OUTPUT SHAPE (one row per group per distinct
            # value) instead of collapsing a group to one row the way every other
            # group_aggregate value does -- on both grid types, `group_by`/
            # `group_aggregate` is a COLLAPSE step (raster band-collapse across selected
            # bands per pixel, vector dimension-collapse across rows per grid_cell) that
            # has no way to also multiply its own row count by the number of distinct
            # values without breaking that collapse's "one row per group" contract.
            # Rejected here, before any file is read, rather than left to fail deep in
            # `HORIZONTAL_RASTER_AGGREGATES`/`getattr(pl.col("value"), ...)` -- the same
            # pattern the raster-only `group_aggregate == "count"` rejection (in
            # `RasterGridAnalysis._collapse_group`) uses. summarize_by/
            # summarize_aggregate=count_by_value is the only way to reach it.
            self._error(
                "group_aggregate 'count_by_value' is not supported -- "
                "use summarize_by and summarize_aggregate=count_by_value instead"
            )

        group_by_cols = self._get_group_by_columns()
        self._collapse_group(group_by_cols)

    def _collapse_group(self, group_by_cols):
        """Collapse `self.lf` (a vector frame) or `self.lf`/`self._band_groups`/
        `self._band_ops` (a raster frame) per `group_by_cols`. See the module/class
        docstring's hook contract -- overridden per subclass; there is no grid-agnostic
        default.
        """
        raise NotImplementedError

    # -- stage 4: transform -----------------------------------------------------------------

    async def _apply_transform_params(self):
        """Apply transform and extract parameters to the LazyFrame or image."""
        await self._apply_transform_params_impl()

    async def _apply_transform_params_impl(self):
        """Apply `summarize_by`/`summarize_aggregate`. See the module/class docstring's
        hook contract -- overridden per subclass; there is no grid-agnostic default.
        """
        raise NotImplementedError

    def _validate_summarize_params(self):
        """Validate summarize parameters and parse summary grid_cell IDs, if present.

        Returns True if summarize_by/summarize_aggregate were provided, False if neither was.
        """
        if self.options.summarize_by is None and self.options.summarize_aggregate is None:
            return False

        if self.options.summarize_by is None or self.options.summarize_aggregate is None:
            self._error("summarize_by and summarize_aggregate parameters must be used together")

        # Extract and validate summary grid_cell IDs, if present.
        if self.options.summarize_grid_cell:
            # summarize_grid = GRIDS[self.options.summarize_by]
            # TODO: Validate that the given IDs fall within the grid min and max.
            self.summarize_grid_cell_ids = [
                int(geo_id) for geo_id in unpack_values(self.options.summarize_grid_cell, "summarize_grid_cell")
            ]

        return True

    def _reject_calculate_with_count_by_value(self):
        """count_by_value counts DISTINCT values; a calculate entry rewrites a value
        BEFORE it would be counted, which changes which raw values collapse into "the
        same" category without changing what the caller meant by "distinct" -- e.g. a
        `value * 10` entry turns land-type codes 1 and 2 into 10 and 20, not into two
        different categories, but does change what a downstream consumer sees as the
        reported "value" for each count. Every other aggregate folds a calculate entry
        through unremarkably (a rewritten number is still a meaningful sum/mean/min/max),
        so this is a rejection specific to count_by_value, not a general
        calculate/summarize incompatibility. Checked against `self.options` directly (not
        `self.lf`) so it fires before the eventual `group_by` even if
        `_apply_calculate_params` already ran earlier in the pipeline.
        """
        if self.options.calculate:
            self._error("calculate parameters cannot be combined with count_by_value")

    # -- orchestration ----------------------------------------------------------------------

    async def apply_params(self):
        """Apply the query parameters to the LazyFrame."""
        await self._apply_filter_params()
        self._apply_calculate_params()
        self._apply_group_params()
        await self._apply_transform_params()
        self._apply_sort_params()

    def estimate_data_size(self):
        """Estimate the size, in bytes, of data. This does not include dimension columns."""
        required_cols = required_columns(
            get_input_data_frame(**dict(self.dataset_dimensions)), include_dataset=False, include_metadata=False
        )
        group_by_cols = self._get_group_by_columns()
        has_group_by = self.options.group_aggregate is not None

        # Estimate rows.
        rows = 1
        for col in required_cols:
            # If there is a group by clause, only the group by columns contribute
            # to the number of output rows.
            if has_group_by and col not in group_by_cols:
                continue

            dimension = DIMENSIONS[col]

            value_count = None
            values = getattr(self.filter_dimensions, col)
            if values is not None:
                value_count = len(values)
            # Compare against None explicitly: dimension.values is a DataFrame, and
            # Polars raises on bool() rather than reporting emptiness.
            elif dimension.values is not None:
                value_count = len(dimension.values)
            elif col == "calendar_year":
                if self.dataset_dimensions.time_unit == "decadal":
                    value_count = 6
                else:
                    value_count = 52
            elif col == "calendar_day":
                if self.dataset_dimensions.time_unit == "monthly":
                    value_count = 52 * 12
                else:
                    value_count = 52 * 365

            if value_count is None:
                return None

            rows *= value_count

        # Estimate columns.
        columns = None
        if self.grid_cell_ids is not None:
            columns = len(self.grid_cell_ids)
        else:
            grid = GRIDS[self.dataset_dimensions.grid]
            columns = getattr(self.grid, "id_count", None)

            if columns is None:
                height = getattr(grid, "height", None)
                width = getattr(grid, "width", None)
                if height and width:
                    columns = height * width

        if columns is None:
            return None

        # Calculate bytes per value.
        variable = VARIABLES[self.dataset_dimensions.variable]
        dtype_digits = re.findall(r"\d+", str(variable.dtype))
        if len(dtype_digits) != 1:
            return None

        return rows * columns * int(dtype_digits[0]) / 8
