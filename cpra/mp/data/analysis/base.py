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

**The `apply_params` flow.** A `BaseAnalysis` instance mutates `self.lf` (a Polars
LazyFrame) by walking ONE ordered list of ops, run once by `apply_params`. There is no
second execution path: a request compiled from a declared pipeline arrives with its op
list already attached as `self._pipeline_ops`, and a flat request (`where`, `calculate`,
`group_by`/`group_aggregate`) is LOWERED into the equivalent list by
`stages.lower_flat_options` at the top of `apply_params`. Both then go to `apply_steps`,
the per-shape interpreter each subclass implements (`vector.VectorGridAnalysis.apply_steps`
for long frames, `raster.RasterGridAnalysis.apply_steps` for band-wide ones).

Two options have no op form and so run around the walk rather than inside it:

- `options.grid_cell` narrows the OUTPUT, after every op has run -- it is not a step in
  the chain, it is which cells the chain's result is narrowed to. A caller who wants an
  INPUT-side cell filter instead can express one directly as a leading `filter` step. Each
  `apply_steps` applies it itself, terminally, as a direct call to a free function
  (`vector.apply_grid_cell_selection`; `raster`'s own terminal lowering) -- there is no
  shared hook for it, since what "narrow the output to these cells" means depends on the
  OUTPUT's shape (a vector grid's real cell IDs vs. a raster's point samples), not just on
  grid type at the start.
- `sort`/`select` (`_apply_sort_params`) order and narrow the FINAL columns, after the
  walk, since they describe the response rather than the data.

`summarize_by`/`summarize_aggregate` sit between the two: they are a `project` in op form,
but the flat surface deliberately does not lower them, because the flat surface has no
declared order and a summarize is always last. Each interpreter calls
`_apply_transform_params_impl` at the end of its walk instead (see
`stages.lower_flat_options`).

**Hook contract.** The hook below is declared abstract here (raising `NotImplementedError`)
and overridden once per subclass.

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

**Where the LOGIC of an op lives.** Two layers, split by what they depend on. The ALGEBRA
-- `apply_dimension_filter`/`apply_value_filter`/`apply_calculate_plans` below, plus
`vector.apply_group_by`/`apply_project`/`apply_long_op` and `raster`'s
`apply_*_band_wide` family -- is re-entrant and op-shaped: each takes an `lf` and returns
one, so an interpreter walking a `list[ops.Op]` can invoke it once per op in whatever
order the request declared. The OPTION HANDLING that reads `Options`, lowers it into that
list, and makes the validations the list cannot express is `analysis/stages.py`, another
module of plain free functions. What is left on this class once both layers are factored
out is `apply_params` (five lines of "get the op list, walk it, sort"), the two hooks, and
`op_error_boundary`, the single point where a typed library error becomes a 400.
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
from cpra.mp.data.analysis.utils import get_lazy_frame
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.legends import Legend
from cpra.mp.data.variables import Variable


# -- re-entrant op cores -------------------------------------------------------------
#
# Each function below is Polars logic that once ran only against `self.lf`/`self.options`,
# taking its own arguments and returning a new `lf` instead of mutating an instance -- so a
# caller walking a `list[ops.Op]` in DECLARED order (either interpreter's `apply_steps`,
# via the shared long lowering `vector.apply_long_op`) can invoke it once per op, in
# whatever order the request actually declared. Each one raises the plain
# `ExpressionError`/`FoldError` its Polars-adjacent helpers already raise -- never
# `AnalysisError` -- since these are library functions with no `self._error` to call; the
# interpreter catches and converts at its own boundary (`op_error_boundary`, below).
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

    A falsy `where` (`None` or `""`) is a no-op: returns `(lf, None, False)` unchanged.
    (`stages.lower_flat_options` emits no `FilterOp` at all for a request with no `where`,
    and `ops.FilterOp.where` is required, so this guard is defensive rather than a path
    any request surface reaches.)
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
    skipping behavior) or a row DROP (`mask=False`, `lf.filter(...)`). The LONG-shape op
    lowering (`vector.apply_long_op`, shared by both interpreters) passes
    `ops.FilterOp.mask_value`, which is `True` only for the filter
    `stages.lower_flat_options` emits for an AGGREGATING flat request -- see
    `FilterOp.mask_value`'s docstring for why a straight drop is the only semantically
    coherent choice once filter/group/project run in their own declared positions, and why
    the orderless flat surface is the one case that needs the mask. (The BAND-WIDE lowering
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
    sequentially in list order (entry N+1 sees entry N's output) -- the LONG-shape half of
    `calculate`'s lowering, reached from `vector.apply_long_op`. It assumes `lf` already
    carries a real `value` column; a band-wide frame is band METADATA and has none, so the
    band-wide half (`raster.apply_band_wide_op`) records the compiled plan for the pixel
    read instead of calling this. A missing value stays missing (`is_not_null()` guards
    every rewrite) and a row not matching a plan's own `where` passes through unchanged, so
    a conditional `calculate` rewrites only where it applies and leaves the rest of the
    column alone.

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
        # -- its interpreter just reassigns `self.lf` directly, since a
        # row-per-grid_cell frame has no band columns to keep a dictionary for).
        #
        # `self.lf` for a raster analysis starts as the per-band metadata table (one row
        # per selected (path, band)) and stays band metadata until a summarize converts
        # it to a tabular frame: it is the COLUMN DICTIONARY for the band-wide frame a
        # pixel read will produce, narrowed and collapsed in step with the ops declared
        # against it. `_raw_band_frame` is the snapshot the FIRST `GroupOp`
        # (`raster.collapse_band_group`) takes before `self.lf` moves on to collapsed
        # (dimension-only) metadata, since
        # every pixel read still needs the original (path, band) rows to build its scan;
        # it is also the "a group has run" flag, which is what
        # `RasterGridAnalysis._band_frame` and the dimension-filter pushdown rule in
        # `raster.apply_band_wide_op` both test.
        #
        # `_band_ops` is the ordered replay recipe: one entry per filter/calculate/group
        # step actually declared, in declared order, each carrying only what the caller
        # declared (a predicate string, a plan list, a group key) and never a row index
        # or a frozen metadata copy. `raster.apply_band_ops` resolves each entry against
        # the band-wide frame and the metadata the replay has reached by then -- see
        # that function and `RasterGridAnalysis`'s three pixel-read methods.
        self._raw_band_frame = None
        self._band_ops = []
        # Set by whichever `apply_steps` ran, from each `FilterOp`'s value conjuncts (see
        # `expressions.classify_where`) -- the internal successor to the retired
        # `Options.value_where` field. `None` means no filter contributed a value
        # conjunct; every other consumer of the old `value_where` field
        # (`RasterGridAnalysis._apply_transform_params_impl`, `summarize_zone_stats`,
        # `summarize_count_by_value`) reads this attribute instead now.
        self._value_where = None
        # Set by whichever `apply_steps` ran: the validated `CalculatePlan`s for this
        # request, in application order. Empty when the request declared no `calculate`
        # at all. A long frame consumes them immediately (`apply_calculate_plans`, from
        # `vector.apply_long_op`); a band-wide one accumulates them here and the raster
        # pixel-read paths (tile/write/summarize) flush them into `_band_ops` (as a
        # `("calculate", plans)` entry) to apply per band via
        # `raster.apply_calculate_band_wide`.
        self._calculate_plans = []
        # Also set by whichever `apply_steps` ran: whether any `FilterOp` produced at
        # least one DIMENSION conjunct, i.e. whether the request actually did anything
        # that could shrink the row count reaching a group -- see
        # `_has_selective_dimension_filter`, which cannot just check
        # `options.where is not None`, since a `where` may hold only a value conjunct
        # (a mask, not a row filter).
        self._where_has_dimension_conjunct = False
        # Set by `planner._compile_single_input_chain` to `ops.compile_ops(pipeline)`'s
        # `list[Op]` -- never by this constructor itself, and never by anything on the
        # flat (non-pipeline) request surface. `None` (the default here) means "no op IR
        # was attached", the signal `apply_params` reads to lower `self.options` into the
        # equivalent list itself (`stages.lower_flat_options`). Either way exactly one
        # list reaches `apply_steps`; this attribute only says who built it.
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
        response. Read by `apply_params` and passed to `stages.lower_flat_options` as
        `is_aggregating`, which is what decides whether the `FilterOp` lowered from a flat
        `where` masks a non-matching value to null (composing with the aggregate's own
        null-skipping behavior) or drops the row outright (the only sensible behavior when
        there is no aggregate downstream to skip a null for). See
        `ops.FilterOp.mask_value`.
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

    # -- sort/select: the two options with no op form ---------------------------------------

    def _apply_sort_params(self):
        """Apply sort and select parameters to the LazyFrame -- see
        `stages.apply_sort_options`.

        Called by `apply_params` once, after the op walk, because `sort`/`select` are
        response SHAPING options that have no op form and so always run last, whatever
        order the ops themselves declared.
        """
        self.lf = stages.apply_sort_options(
            self.lf, self.options.sort, self.options.select, self.dataset_dimensions
        )

    def _has_selective_dimension_filter(self):
        """Whether this request already carries something that shrinks the row count
        reaching a vector group_by well below "the whole dataset" -- the signal
        `analysis.fold.should_fold_group_by` uses to skip the fold on an already-cheap
        query.

        `filter_dimensions` values are pushed down as Hive-partition filters inside
        `get_lazy_frame`/`read_parquet`, before `self.lf` exists at all -- the measured
        case where a partition filter drops a scan from 9.8 GB to 285 MB. A `where`
        DIMENSION conjunct (an arbitrary SQL `.filter()`, applied by
        `apply_dimension_filter` when the interpreter lowers a `FilterOp`) is the same kind
        of row-dropping filter, so it counts too -- `self._where_has_dimension_conjunct`
        (set by whichever `apply_steps` ran, from the second value each lowering returns)
        is checked here rather than `self.options.where is not None`, since a `where` may
        hold only a VALUE conjunct, which never reaches this method as a row filter at all
        (see the next paragraph).

        `options.grid_cell` is deliberately NOT included: it narrows the OUTPUT, applied
        once, terminally, after every op in the walk (including any `GroupOp`) has already
        run -- see `analysis/base.py`'s module docstring. By the time a `GroupOp` reaches
        this method, `grid_cell` has not yet dropped a single row, so it is not a selective
        filter for a `GroupOp`'s purposes, whatever it is eventually going to narrow at the
        end.

        A `where` VALUE conjunct and the `calculate` parameters are deliberately NOT
        included: wherever this method's callers (the two interpreters, gating a vector
        `GroupOp`) apply, a value conjunct is a `with_columns` mask rather than a
        row-dropping `.filter()` whenever it could change a group's denominator at all --
        masking is exactly what lets a flat `where` compose with the group_by's own
        aggregate (see `ops.FilterOp.mask_value`) -- so it never changes how many rows
        reach the group-by, doesn't affect whether it's cliff-shaped, and (unlike a
        `.filter()`) doesn't block the fold's `.slice()` from reaching the scan either.
        """
        return (
            self._where_has_dimension_conjunct
            or any(v is not None for v in dict(self.filter_dimensions).values())
        )

    # -- the summarize hook -----------------------------------------------------------------

    async def _apply_transform_params_impl(self):
        """Apply `summarize_by`/`summarize_aggregate`. See the module/class docstring's
        hook contract -- overridden per subclass; there is no grid-agnostic default.
        """
        raise NotImplementedError

    # -- orchestration ----------------------------------------------------------------------

    async def apply_params(self):
        """Apply the query parameters to the LazyFrame: hand an op list to the interpreter
        (`apply_steps`), then apply `sort`/`select`.

        **There is one execution engine.** Every request -- flat or piped -- runs through
        `apply_steps`. The only difference is where its `list[ops.Op]` came from:

        - A PIPELINE request arrives with `self._pipeline_ops` already attached, by
          `planner._compile_single_input_chain` (the only thing that ever sets
          it), holding `ops.compile_ops(pipeline)` -- the steps the caller declared, in
          DECLARED order.
        - A FLAT request still has `_pipeline_ops` at its `__init__` default of `None`, and
          is lowered here, by `stages.lower_flat_options`, into the same IR: `filter` from
          `where`, `calculate` from each `calculate` entry, `group` from
          `group_by`/`group_aggregate`, emitted in exactly one fixed order: filter, then
          calculate, then group. `Options` has no declared order of its own, so that
          lowering is order-preserving by construction -- see that function's docstring,
          including why `summarize_by`/`summarize_aggregate` and `sort`/`select` are
          deliberately not lowered.

        Lowered HERE, not in `__init__` or in `factory.create_analysis`. Construction stays
        free of validation and of `Options` interpretation: `lower_flat_options` raises
        `AnalysisError` for `group_by` without `group_aggregate` and for
        `group_aggregate='count_by_value'`, and those rejections belong where they have
        always fired -- inside the call that executes the request, not at the point an
        analysis object is built. This is also the ONE place every consumer passes through
        (`api/tabular.py`, `api/vector.py`, the tile routes, `visualizations/map.py`,
        `analysis.analyze_data`), whereas the construction paths are several.

        `apply_steps` is per-grid-type (`vector.VectorGridAnalysis.apply_steps` for the long
        shape, `raster.RasterGridAnalysis.apply_steps` for the band-wide/long pair), so this
        method is only ever reached on an instance that defines one -- `factory.
        create_analysis` returns one of those two subclasses and nothing else.

        `sort`/`select` run after the walk because they are response SHAPING options with no
        op form at all -- see `_apply_sort_params`.
        """
        ordered_ops = self._pipeline_ops
        if ordered_ops is None:
            ordered_ops = stages.lower_flat_options(
                self.options, is_aggregating=self._is_aggregating()
            )
        await self.apply_steps(ordered_ops)
        self._apply_sort_params()

    def estimate_data_size(self):
        """Estimate the size, in bytes, of data -- see `metadata.estimate_data_size`.

        `self.grid` is `GRIDS[self.dataset_dimensions.grid]` (set in `__init__` and never
        reassigned), which is the single grid object the estimate needs.

        The group-by columns come from `stages.resolve_group_by_columns` -- the same
        function `stages.lower_flat_options` builds `GroupOp.by` with -- rather than from
        the op list, because this method is routinely called WITHOUT `apply_params` having
        run (`api/__init__.py`'s size pre-check answers before any frame is walked).
        """
        return metadata.estimate_data_size(
            self.dataset_dimensions,
            self.filter_dimensions,
            self.grid,
            stages.resolve_group_by_columns(self.options.group_by, self.options.group_aggregate),
            has_group_by=self.options.group_aggregate is not None,
            grid_cell_ids=self.grid_cell_ids,
        )
