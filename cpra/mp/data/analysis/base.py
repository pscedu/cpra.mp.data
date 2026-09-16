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
   table, not pixels -- see `raster.apply_calculate_band_wide`).
3. **group** (`_apply_group_params`) -- `group_by`/`group_aggregate` collapse rows via the
   `_collapse_group` hook (vector: dimension-collapse across rows per grid_cell, folding
   through `analysis.fold` when the collapse is cliff-shaped; raster: collapses the band
   METADATA and records the group as a `self._band_ops` entry, for a later stage to
   replay against actual pixels).
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
  method's own comment). Raster: no pixels are read here -- it collapses the band
  metadata and appends a `self._band_ops` entry that a later stage (transform, tile, or
  write) replays against actual band data.
- `async _apply_transform_params_impl(self)` -- apply `summarize_by`/`summarize_aggregate`,
  converting to another grid's cells. Each implementation starts by calling
  `stages.validate_summarize_options`, which confirms both parameters (or neither) were
  given and parses `summarize_grid_cell`; the implementation returns immediately if
  neither was. Must leave `self.lf` in the shared summarize output shape:
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
diverge from `self.grid.grid_type` -- see `analysis/output.py`'s `resolve_output_type`)
rather than which subclass `self` is.

**Where the LOGIC of that surface lives.** The rendering half is `analysis/output.py`
(`tabular_kwargs`/`stream_tabular`/`write_tabular`/`resolve_legend`/`resolve_output_type`)
and the tile-request validation half is `analysis/tiles.py`
(`validate_raster_tile`/`validate_vector_tile`), both modules of plain free functions. The
members named above are delegations to them, kept on the class only because
`api/tabular.py` and `api/__init__.py` reach for them by duck typing. None of that is op
algebra, and none of it depends on which ops ran. The catalog and frame-shape lookups
(`variable`, `estimate_data_size`) are the same story, in `analysis/metadata.py`.

**Where the LOGIC of the five stages lives.** Two layers, split by what they depend on.
The ALGEBRA -- `apply_dimension_filter`/`apply_value_filter`/`apply_calculate_plans`
below, plus `vector.apply_group_by`/`apply_project`/`apply_long_op` and `raster`'s
`apply_*_band_wide` family -- is re-entrant and op-shaped: each takes an `lf` and returns
one, so an interpreter walking a `list[ops.Op]` can invoke it once per op in whatever
order the request declared. The OPTION HANDLING that reads `Options` and decides what to
hand the algebra is `analysis/stages.py`, another module of plain free functions. The four
`_apply_*_params` methods below are what is left once both layers are factored out: they
read `self.options`, store results back on `self`, dispatch the subclass hook, and convert
a typed library error to a 400 via `op_error_boundary` -- and nothing else.
"""

from contextlib import contextmanager

import polars as pl
from polars.exceptions import ColumnNotFoundError

from cpra.mp.data.analysis import metadata, output, stages, tiles
from cpra.mp.data.analysis.expressions import (
    ExpressionError,
    classify_where,
    parse_value_expr,
    value_predicate_mask,
)
from cpra.mp.data.analysis.fold import FoldError
from cpra.mp.data.analysis.types import OutputType
from cpra.mp.data.analysis.utils import get_lazy_frame, unpack_values
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.legends import Legend
from cpra.mp.data.variables import Variable

# The accepted `group_by` values. (`sort`'s own choice list is `stages.SORT_CHOICES`,
# next to the sort/select stage logic that validates against it.)
DIMENSION_CHOICES = list(DIMENSIONS.keys())


# -- re-entrant stage cores ---------------------------------------------------------
#
# Each function below is the same Polars logic `_apply_filter_params`/`_apply_calculate_
# params` used to run only against `self.lf`/`self.options`, pulled out to take its own
# arguments and return a new `lf` instead of mutating an instance -- so a caller walking a
# `list[ops.Op]` in DECLARED order (either interpreter's `apply_steps`, via the shared
# long lowering `vector.apply_long_op`) can invoke exactly the same stage logic once per
# op, in whatever order the
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
    params`'s docstring explains exactly when the OLD (fixed five-stage) flow picks
    which: mask for an aggregating request (`_is_aggregating()`), drop otherwise. The
    LONG-shape op lowering (`vector.apply_long_op`, shared by both interpreters) always
    passes `mask=False` -- see that function's own docstring for why a straight drop, not
    that conditional, is the only semantically coherent choice once filter/group/project
    run in their own declared positions instead of a fixed order. (The BAND-WIDE lowering
    does not reach this function at all: it must stay positionally aligned to the raster,
    so its value filter is always a per-cell mask -- `raster.apply_value_filter_band_wide`.)

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


@contextmanager
def op_error_boundary():
    """The single translation point between the op ALGEBRA and the request LIFECYCLE.

    Every re-entrant op core in this package (`apply_dimension_filter`/`apply_value_filter`/
    `apply_calculate_plans` above, `vector.apply_group_by`/`apply_project`/`apply_long_op`,
    `raster.band_metadata_scan_filter` and the `apply_*_band_wide` family) is a plain
    library function: it has no `self`, no `status_code`, and no opinion about HTTP, so it
    signals failure with the typed library exceptions its Polars-adjacent helpers already
    raise -- `ExpressionError` for anything the expression layer refuses to parse or
    whitelist, `FoldError` for a chunked collapse that cannot proceed. Wrapping an
    interpreter's whole op walk in this boundary converts either one, once, into the
    `AnalysisError` the API layer already knows how to render, with the message text
    unchanged.

    This replaces a per-op `try/except ...: self._error(str(exc))` around nearly every
    lowering call in `vector.VectorGridAnalysis.apply_steps`/`raster.RasterGridAnalysis.
    apply_steps`, which made the mapping "typed library error -> 400" a thing each call
    site had to remember rather than a property of the interpreter.

    Deliberately does NOT catch `AnalysisError`: an op core that already knows the
    request-level answer (a missing crosswalk file, a rejected aggregate) raises that
    directly, including its own `status_code`, and re-wrapping it here would flatten a 404
    into a 400.
    """
    try:
        yield
    except (ExpressionError, FoldError) as exc:
        raise AnalysisError(str(exc)) from exc


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
        # `RasterGridAnalysis`-only band bookkeeping (a vector frame never sets either
        # -- its own `_collapse_group` just reassigns `self.lf` directly, since a
        # row-per-grid_cell frame has no band columns to keep a dictionary for).
        #
        # `self.lf` for a raster analysis starts as the per-band metadata table (one row
        # per selected (path, band)) and stays band metadata until a summarize converts
        # it to a tabular frame: it is the COLUMN DICTIONARY for the band-wide frame a
        # pixel read will produce, narrowed and collapsed in step with the ops declared
        # against it. `_raw_band_frame` is the snapshot the FIRST `_collapse_group` call
        # takes before `self.lf` moves on to collapsed (dimension-only) metadata, since
        # every pixel read still needs the original (path, band) rows to build its scan;
        # it is also the "a group has run" flag, which is what
        # `RasterGridAnalysis._band_frame` and the dimension-filter pushdown rule in
        # `RasterGridAnalysis.apply_steps` both test.
        #
        # `_band_ops` is the ordered replay recipe: one entry per filter/calculate/group
        # step actually declared, in declared order, each carrying only what the caller
        # declared (a predicate string, a plan list, a group key) and never a row index
        # or a frozen metadata copy. `raster.apply_band_ops` resolves each entry against
        # the band-wide frame and the metadata the replay has reached by then -- see
        # that function and `RasterGridAnalysis`'s three pixel-read methods.
        self._raw_band_frame = None
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
        # (tile/write/summarize) flush it into `_band_ops` (as a `("calculate", plans)`
        # entry) and apply it per band via `raster.apply_calculate_band_wide`.
        self._calculate_plans = []
        # Also set by `_apply_filter_params`: whether `where` produced at least one
        # DIMENSION conjunct, i.e. whether it actually did anything that could shrink
        # the row count reaching a group_by -- see `_has_selective_dimension_filter`,
        # which used to just check `options.where is not None` and is now wrong to,
        # since `where` may hold only a value conjunct (a mask, not a row filter).
        self._where_has_dimension_conjunct = False
        # Set by `planner._compile_single_input_chain`'s fast path to
        # `ops.compile_ops(pipeline)`'s
        # `list[Op]` -- never by this constructor itself, and never by anything on the
        # flat (non-pipeline) request surface. `None` (the default here) means "no op IR
        # was attached", the signal both `apply_params` overrides (`VectorGridAnalysis`'
        # and `RasterGridAnalysis`') read to decide between their own interpreter
        # (`apply_steps`) and this class's fixed five-stage flow -- see those overrides'
        # docstrings.
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
        """Writer arguments this dataset needs for `format` -- see `output.tabular_kwargs`."""
        return output.tabular_kwargs(format, self.variable.float_precision)

    def stream_tabular(self, out_path, format, *args, **kwargs):
        """Sink the frame to `out_path` -- see `output.stream_tabular`."""
        output.stream_tabular(self.materialize, out_path, format, *args, **kwargs)

    def write_tabular(self, out_path, format, *args, **kwargs):
        """Collect and write the frame to `out_path` -- see `output.write_tabular`."""
        output.write_tabular(self.materialize, out_path, format, *args, **kwargs)

    # -- output surface: shared properties -----------------------------------------------

    @property
    def legend(self) -> Legend:
        """The `Legend` this request renders with -- see `output.resolve_legend`."""
        return output.resolve_legend(self.options.color_map, self.variable.legend)

    @property
    def output_type(self) -> OutputType:
        """The output type for this dataset given its parameters -- see
        `output.resolve_output_type`, which explains why this is not simply
        `self.grid.grid_type`.
        """
        return output.resolve_output_type(
            self.grid.grid_type,
            grid_cell=self.options.grid_cell,
            summarize_by=self.options.summarize_by,
        )

    @property
    def variable(self) -> Variable:
        """The catalog `Variable` this dataset selects -- see `metadata.variable_for`."""
        return metadata.variable_for(self.dataset_dimensions)

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
        """Validate that the tile is raster and will have a single band -- see
        `tiles.validate_raster_tile`.
        """
        tiles.validate_raster_tile(self.lf, self.output_type)

    def validate_vector_tile(self):
        """Validate that the tile is a vector grid with an available PMTiles file -- see
        `tiles.validate_vector_tile`.
        """
        tiles.validate_vector_tile(self.output_type, self.options.summarize_by or self.grid.name)

    # -- stage 1: filter ------------------------------------------------------------------

    async def _apply_filter_params(self):
        """Apply filter parameters to the LazyFrame.

        The `where` half is `stages.apply_filter_options`, which reads the option, splits
        it into dimension and value conjuncts, and decides mask-vs-drop -- see that
        function for the whole contract. What stays here is what touches `self`: storing
        its three results (`self.lf`, `self._value_where` -- the internal successor to the
        retired `Options.value_where` field, which every raster-side consumer reads -- and
        `self._where_has_dimension_conjunct`), converting its typed `ExpressionError` into
        a 400 at this stage's own boundary, and dispatching the `_apply_grid_cell` hook the
        subclass supplies.

        The option is read as a whole, once, exactly as before: an interpreter walking a
        `list[ops.Op]` instead calls the underlying re-entrant algebra
        (`apply_dimension_filter`/`apply_value_filter`) once per `FilterOp` in declared
        order, via `vector.apply_long_op`, and never reaches this method at all.
        """
        with op_error_boundary():
            self.lf, self._value_where, self._where_has_dimension_conjunct = stages.apply_filter_options(
                self.lf,
                self.options.where,
                grid_type=self.grid.grid_type,
                is_aggregating=self._is_aggregating(),
            )

        if self.options.grid_cell:
            await self._apply_grid_cell()

    async def _apply_grid_cell(self):
        """Apply the `grid_cell` filter to `self.lf`. See the module/class docstring's
        hook contract -- overridden per subclass; there is no grid-agnostic default.
        """
        raise NotImplementedError

    # -- stage 2: calculate -----------------------------------------------------------------

    def _apply_calculate_params(self):
        """Apply the calculate stage: expression/where entries that rewrite `value` for
        the rows/bands/pixels matching each entry's `where`, sequentially in entry order.

        Validation and application are `stages.resolve_calculate_plans` and
        `stages.apply_calculate_options` -- see those two for what each entry means and
        why a raster's band-metadata table is left for the pixel-read paths to rewrite.
        What stays here is what touches `self`: stashing the validated plans on
        `self._calculate_plans` (which every raster read path replays per band), storing
        the rewritten frame back on `self.lf`, and converting the typed `ExpressionError`
        either one raises into a 400 at this stage's own boundary.

        The option list is read as a whole, once, exactly as before: an interpreter
        instead calls the underlying re-entrant algebra (`apply_calculate_plans`) once per
        `CalculateOp`, in declared order, and never reaches this method at all.
        """
        with op_error_boundary():
            self._calculate_plans = stages.resolve_calculate_plans(self.options.calculate)
            self.lf = stages.apply_calculate_options(self.lf, self._calculate_plans, self.grid.grid_type)

    # -- stage 5: sort/select ---------------------------------------------------------------

    def _apply_sort_params(self):
        """Apply sort and select parameters to the LazyFrame -- see
        `stages.apply_sort_options`.

        Both interpreters call this at the end of their own op walk (see their
        `apply_params` overrides), because `sort`/`select` are response SHAPING options
        that have no op form and so always run last, whatever order the ops themselves
        declared.
        """
        self.lf = stages.apply_sort_options(
            self.lf, self.options.sort, self.options.select, self.dataset_dimensions
        )

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

        The shared preamble (the `count_by_value` rejection, which differs by neither grid
        type nor instance and so lives in `stages.reject_count_by_value_group`, and
        `group_by_cols` resolution) runs here; the actual collapse is the
        `_collapse_group` hook -- see the module/class docstring.
        """
        if self.options.group_by is None and self.options.group_aggregate is None:
            return

        stages.reject_count_by_value_group(self.options.group_aggregate)

        group_by_cols = self._get_group_by_columns()
        self._collapse_group(group_by_cols)

    def _collapse_group(self, group_by_cols):
        """Collapse `self.lf` (a vector frame) or `self.lf`/`self._band_ops` (a raster
        frame) per `group_by_cols`. See the module/class
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

    # -- orchestration ----------------------------------------------------------------------

    async def apply_params(self):
        """Apply the query parameters to the LazyFrame, in the fixed five-stage order.

        The one member of this flow that is still a method rather than a free function in
        `analysis/stages.py`, and deliberately so: every call below reaches a subclass
        hook (`_apply_filter_params` -> `_apply_grid_cell`, `_apply_group_params` ->
        `_collapse_group`, `_apply_transform_params` -> `_apply_transform_params_impl`),
        so a free-function version would take the instance under another name and gain
        nothing. It becomes extractable when those hooks stop being hooks, not before.
        """
        await self._apply_filter_params()
        self._apply_calculate_params()
        self._apply_group_params()
        await self._apply_transform_params()
        self._apply_sort_params()

    def estimate_data_size(self):
        """Estimate the size, in bytes, of data -- see `metadata.estimate_data_size`.

        `self.grid` is `GRIDS[self.dataset_dimensions.grid]` (set in `__init__` and never
        reassigned), which is the single grid object the estimate needs.
        """
        return metadata.estimate_data_size(
            self.dataset_dimensions,
            self.filter_dimensions,
            self.grid,
            self._get_group_by_columns(),
            has_group_by=self.options.group_aggregate is not None,
            grid_cell_ids=self.grid_cell_ids,
        )
