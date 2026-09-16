"""`VectorGridAnalysis`: the vector half of the former `api/dataset.py`'s `Dataset` class,
plus the pure MVT (Mapbox Vector Tile) helper functions formerly in `api/vector.py` (moved
here because `get_vector_tile` below is their only caller besides tests, and
`cpra.mp.data.analysis` may not import FastAPI -- see `analysis/base.py`'s docstring).
`api/vector.py` keeps only `VectorTileResponse`, which calls `get_vector_tile` on whatever
`BaseAnalysis` subclass it's given rather than these helpers directly.

See `analysis/base.py`'s module docstring for the staged `apply_params` flow and the hook
contract every method here fills in; see `analysis/raster.py` for the raster half.
"""

import gzip

import mercantile
import polars as pl
import polars.selectors as cs
from mapbox_vector_tile.Mapbox.vector_tile_pb2 import tile as VectorTile
from pmtiles.reader import MmapSource, Reader

from cpra.mp.data import config
from cpra.mp.data.analysis import stages
from cpra.mp.data.analysis.base import (
    AnalysisError,
    BaseAnalysis,
    apply_calculate_plans,
    apply_dimension_filter,
    apply_value_filter,
    op_error_boundary,
)
from cpra.mp.data.analysis.expressions import (
    ExpressionError,
    compile_calculate_entries,
    count_by_value_frame,
    rewrite_named_columns,
)
from cpra.mp.data.analysis.fold import FoldError, fold_group_by, should_fold_group_by
from cpra.mp.data.analysis.geography import (
    parse_coordinate_pairs,
    resolve_vector_grid_cell,
)
from cpra.mp.data.analysis.ops import CalculateOp, FilterOp, GroupOp, Op, ProjectOp
from cpra.mp.data.analysis.utils import asynctiming, unpack_values
from cpra.mp.data.config import COUNT_BY_VALUE_MAX_DISTINCT

DIMENSION_COLUMNS = cs.matches(r"^[a-z]") - cs.by_name("grid_cell", "value")


def read_pmtile(grid_name: str, tile: mercantile.Tile) -> bytes | None:
    """Return the raw (still gzip-compressed) MVT bytes for a tile, or None."""
    path = config.TILES_PATH / f"{grid_name}.pmtiles"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = Reader(MmapSource(f)).get(tile.z, tile.x, tile.y)
    return data  # may be None if tile not in archive


def _data_to_id_value(df: pl.DataFrame) -> dict[int, float]:
    """Extract {grid_cell_id: value} from a long-format DataFrame."""
    if df.is_empty() or "grid_cell" not in df.columns:
        return {}
    return {
        cell: val
        for cell, val in zip(df["grid_cell"].to_list(), df["value"].to_list())
        if val is not None
    }


def parse_tile(compressed: bytes) -> tuple[VectorTile, list[int]]:
    """Decompress and parse a gzip-compressed MVT tile once.

    Returns the parsed VectorTile object and the feature IDs from the boundary
    layer. Callers should pass the VectorTile to inject_and_serialize rather than
    re-parsing from bytes.
    """
    vt = VectorTile()
    vt.ParseFromString(gzip.decompress(compressed))
    ids = []
    for layer in vt.layers:
        if layer.name == "boundary":
            ids = [f.id for f in layer.features]
            break
    return vt, ids


def inject_and_serialize(vt: VectorTile, id_values: dict[int, float]) -> bytes:
    """Inject a 'value' property into each boundary feature and serialize.

    Geometry bytes are never decoded — only keys/values/tags are modified — so
    this is O(features) string/int work rather than O(coordinates) float work.
    """
    for layer in vt.layers:
        if layer.name != "boundary":
            continue
        key_idx = len(layer.keys)
        layer.keys.append("value")
        for feature in layer.features:
            v = id_values.get(feature.id)
            if v is not None:
                val = layer.values.add()
                val.double_value = v
                feature.tags.extend([key_idx, len(layer.values) - 1])

    return vt.SerializeToString()


# -- re-entrant stage cores, shared by the fixed-stage flow and the ops interpreter --
#
# `apply_group_by`/`apply_project` are the same two Polars logic blocks
# `_collapse_group`/`_apply_transform_params_impl` below run against `self.lf`/
# `self.options`, pulled out to take their own arguments and return a new `lf` instead of
# mutating an instance -- exactly the `apply_dimension_filter`/`apply_value_filter`/
# `apply_calculate_plans` pattern `analysis/base.py` establishes for filter/calculate; see
# that module's docstring for why each one raises a plain `FoldError`/`AnalysisError`
# rather than calling `self._error`. `apply_steps` (this module, below) is their one
# caller besides the `VectorGridAnalysis` methods themselves.
def apply_group_by(lf, by, aggregate, *, of="value", has_selective_filter):
    """Collapse `lf` by `[*by, "grid_cell"]`, aggregating `of` with `aggregate` -- the
    re-entrant core of `_collapse_group`, generalized to an explicit value column (`of`)
    instead of assuming `"value"`, so `GroupOp.of` (meaningful for the paired/join shape's
    own synthetic value columns -- see that op's docstring; a single-input pipeline's
    `GroupOp.of` is always `"value"`, per `planner._compile_single_input_chain`'s own
    `of != "value"` rejection) can be honored uniformly if a future join-shaped
    interpreter reuses this function. Folds through `analysis.fold.fold_group_by` when
    `should_fold_group_by` says the collapse is cliff-shaped -- identical routing to
    `_collapse_group`, just over an explicit `lf`/`has_selective_filter` rather than
    `self.lf`/`self._has_selective_dimension_filter()`. May raise `FoldError` (see
    `fold_group_by`); callers convert to `AnalysisError`.
    """
    key_names = [*by, "grid_cell"]
    if should_fold_group_by(key_names, has_selective_filter=has_selective_filter):
        source_lf = lf
        collapsed = fold_group_by(lambda: source_lf, key_names, aggregate, value_column=of)
        return collapsed.lazy()
    return lf.group_by(*by, "grid_cell").agg(getattr(pl.col(of), aggregate)())


def apply_project(lf, from_grid, to_grid, aggregate, *, has_selective_filter):
    """Move `lf`'s rows from `from_grid`'s cells onto `to_grid`'s cells via a crosswalk-
    table join, aggregating with `aggregate` -- the re-entrant core of
    `_apply_transform_params_impl`'s join/reaggregate logic, generalized to explicit grid
    names/aggregate instead of `self.grid`/`self.options.summarize_by`/`self.options.
    summarize_aggregate`, for `apply_steps` (below) to call directly against a
    `ProjectOp`.

    Deliberately does NOT pre-filter the crosswalk by `grid_cell_ids`/
    `summarize_grid_cell_ids` the way `_apply_transform_params_impl` does -- those are a
    performance narrowing specific to the OLD engine's `Options.grid_cell`/
    `summarize_grid_cell`, not a correctness requirement: a post-project `grid_cell`
    filter reaches this interpreter as an ordinary `FilterOp` on the joined result's own
    `grid_cell` column (compile_ops copies the raw `Pipeline`'s steps verbatim -- it has
    no notion of `_extract_post_project_grid_cell_filter`'s Options-only special case),
    and filtering the DESTINATION rows after the join is mathematically identical to
    filtering the crosswalk's `dst_id` before it, just less optimized. Also, unlike
    `_apply_transform_params_impl`, never appends `geographic_unit` itself -- a pipeline's
    `ProjectOp` may not be the request's LAST grid change (though a single-input chain
    without a `join` step can contain at most one -- see `apply_steps`'s docstring), so
    `apply_steps` adds that column once, itself, from the final grid.

    May raise `AnalysisError` (missing crosswalk file) or `FoldError`/`ExpressionError`
    (via `fold_group_by`/`count_by_value_frame`); callers convert the latter two.
    """
    crosswalk_path = config.CROSSWALKS_PATH / f"{from_grid}__{to_grid}.parquet"
    if not crosswalk_path.exists():
        raise AnalysisError(f"No crosswalk from {from_grid} to {to_grid}")
    crosswalk_lf = pl.scan_parquet(str(crosswalk_path))

    dim_cols = cs.expand_selector(lf, DIMENSION_COLUMNS)

    if aggregate == "count_by_value":
        joined = lf.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner").with_columns(
            pl.col("dst_id").alias("grid_cell")
        )
        return count_by_value_frame(joined, [*dim_cols, "grid_cell"], "value", COUNT_BY_VALUE_MAX_DISTINCT)

    key_names = [*dim_cols, "grid_cell"]

    def _scale_and_join(chunk):
        return chunk.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner").with_columns(
            value=pl.col("value") * pl.col("scale_factor")
        )

    if should_fold_group_by(key_names, has_selective_filter=has_selective_filter):
        source_lf = lf
        collapsed = fold_group_by(
            lambda: source_lf,
            [*dim_cols, pl.col("dst_id").alias("grid_cell")],
            aggregate,
            chunk_transform=_scale_and_join,
        )
        return collapsed.lazy()

    joined = lf.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner").with_columns(
        value=pl.col("value") * pl.col("scale_factor")
    )
    return joined.group_by(*dim_cols, pl.col("dst_id").alias("grid_cell")).agg(getattr(pl.col("value"), aggregate)())


def apply_long_op(lf, op, *, rename_map, has_selective_filter):
    """The LONG-shape lowering of one `filter`/`calculate`/`group` op: `lf` in, `lf` out,
    with no instance state read or written.

    `ops.LOWERING_TABLE` gives each op two rows -- one for a long frame (dimension columns,
    `grid_cell`, `value`; one row per observation) and one for a band-wide frame (one row
    per pixel, one column per band). This function is the long row for the three ops whose
    lowering depends only on the SHAPE, never on which grid the shape came from, so both
    interpreters can share one implementation instead of keeping two copies in step.
    `project` is deliberately not handled here: it is the one op that CHANGES shape, so its
    lowering differs per direction (see `apply_project` for vector-to-vector, and
    `raster.RasterGridAnalysis.apply_steps` for the raster-to-vector pair), and each
    interpreter applies its own.

    Returns `(lf, has_dimension_conjunct)` -- the second value is the signal
    `BaseAnalysis._has_selective_dimension_filter` needs from a `FilterOp` (see
    `apply_dimension_filter`); it is `False` for the ops that cannot produce one.

    Raises only `ExpressionError`/`FoldError`; callers translate at
    `base.op_error_boundary`.

    Filter semantics: a `FilterOp`'s value conjunct is always a ROW DROP here
    (`apply_value_filter(..., mask=False)`), never `_apply_filter_params`'s conditional
    mask-vs-drop (`_is_aggregating()`). `ops.LOWERING_TABLE` states the long-shape filter
    lowering as a literal, unconditional "row filter" (see `ops.FilterOp`'s docstring), and
    under TRUE declared-order execution a drop is the only semantically coherent choice: a
    mask exists solely to compose with a downstream aggregate's own null-skipping behavior
    when the filter and the aggregate are applied out of their declared order (the old
    engine's Filter-always-before-Group/Transform flow) -- here the aggregate already runs
    in its own declared position, so there is no "downstream" null left to skip around.
    (The band-wide lowering is the opposite: a mask, never a drop, because that shape must
    stay positionally aligned to the raster -- see `apply_value_filter_band_wide`.)
    """
    if isinstance(op, FilterOp):
        rewritten = rewrite_named_columns(op.where, rename_map)
        lf, value_where, has_dimension_conjunct = apply_dimension_filter(lf, rewritten)
        return apply_value_filter(lf, value_where, mask=False), has_dimension_conjunct
    if isinstance(op, CalculateOp):
        expression = rewrite_named_columns(op.expression, rename_map)
        where = rewrite_named_columns(op.where, rename_map) if op.where is not None else None
        plans = compile_calculate_entries([(f"step.{op.step_no}", expression, where)])
        return apply_calculate_plans(lf, plans), False
    if isinstance(op, GroupOp):
        return (
            apply_group_by(lf, op.by, op.aggregate, of=op.of, has_selective_filter=has_selective_filter),
            False,
        )
    raise ExpressionError(f"{op.op!r} has no long-shape lowering")  # pragma: no cover


class VectorGridAnalysis(BaseAnalysis):
    async def _apply_grid_cell(self):
        """Apply the grid_cell filter to a vector-grid dataset.

        Accepts either grid cell IDs or lat/lng coordinate pairs, matching the raster
        convention. A grid_cell value is treated as IDs only if every token is a plain
        (unsigned, non-decimal) digit string; otherwise the whole list is parsed as
        coordinate pairs and resolved to grid cell IDs via point-in-polygon lookup
        against the grid's geometry. Coordinates that don't fall inside any polygon are
        dropped rather than erroring.

        Either way the result is reported under grid cell IDs, never the coordinates that
        selected them: a vector grid_cell *is* its ID. Several coordinates landing in one
        cell therefore select it once.
        """
        values = unpack_values(self.options.grid_cell, "grid_cell")

        if all(value.isdigit() for value in values):
            # TODO: Validate that the given IDs fall within the grid min and max.
            grid_cell_ids = values
        else:
            lngs, lats = parse_coordinate_pairs(values)
            resolved = await resolve_vector_grid_cell(self.grid.name, lngs, lats)
            grid_cell_ids = [str(geo_id) for geo_id in resolved if geo_id is not None]

        self.grid_cell_ids = list(dict.fromkeys(grid_cell_ids))
        self.lf = self.lf.filter(pl.col("grid_cell").is_in([int(g) for g in self.grid_cell_ids]))

    def _collapse_group(self, group_by_cols):
        """Collapse `self.lf` per `group_by_cols`, a real vector dimension-collapse
        across rows per grid_cell (via `group_by(...).agg(...)`, or
        `analysis.fold.fold_group_by` when cliff-shaped -- see `should_fold_group_by`).

        A dimension column left out of `group_by_cols` is simply absent from the result
        -- `group_by(...).agg(...)` (and `fold_group_by`, which delegates to the same
        Polars primitive per chunk) already drops any column that is neither a group key
        nor an aggregated expression, so no extra step is needed to make that happen.
        This used to be undone right after: every such column was added straight back as
        an all-null, Null-dtype placeholder, so a caller saw a schema-stable but
        meaningless column for every dimension the request didn't group by (e.g.
        `model_group_id`, `scenario_id`, `calendar_day` all appearing, entirely null, in
        a plain `group_aggregate='count'` response). Dropping instead of null-filling is
        the honest shape for "this dimension no longer has a value" and matches
        `VectorGridAnalysis`'s own crosswalk-reaggregation branch below (`summarize_by`)
        and `PairedAnalysis`/the pipeline shape, neither of which ever reintroduces a
        collapsed dimension as null -- see the README's `analyze_data` example, whose
        documented output columns (`grid_cell, calendar_year, value`) already assumed
        the collapsed dimensions were simply gone. A `select` naming a since-collapsed
        dimension now surfaces as a clean `ColumnNotFoundError` -> 400 out of
        `materialize`, rather than silently returning an all-null column.

        A thin caller over the module-level `apply_group_by` -- the fold-vs-plain routing
        (`should_fold_group_by`, the "9.8 GB cliff" comment) now lives there, generalized
        to an explicit `lf`/`of`/
        `has_selective_filter` instead of `self.lf`/`self.options.group_aggregate`/`self.
        _has_selective_dimension_filter()`, so `apply_steps` (below) can invoke the
        identical logic once per `GroupOp`.
        """
        try:
            self.lf = apply_group_by(
                self.lf,
                group_by_cols,
                self.options.group_aggregate,
                has_selective_filter=self._has_selective_dimension_filter(),
            )
        except FoldError as exc:
            self._error(str(exc))

    async def _apply_transform_params_impl(self):
        """Apply transform and extract parameters to the LazyFrame.

        Declared `async` (with no internal `await`) purely to match the one hook name
        `BaseAnalysis._apply_transform_params` calls on both subclasses -- see that
        module's docstring for why the vector/raster pair collapses to a single hook.
        """

        has_summarize, self.summarize_grid_cell_ids = stages.validate_summarize_options(
            self.options.summarize_by, self.options.summarize_aggregate, self.options.summarize_grid_cell
        )
        if not has_summarize:
            return

        # Generate column expressions.
        crosswalk_name = f"{self.grid.name}__{self.options.summarize_by}.parquet"
        crosswalk_path = config.CROSSWALKS_PATH / crosswalk_name

        # scan_parquet is lazy and won't raise for a missing file until collection,
        # so check up front to report the missing crosswalk as a 400.
        if not crosswalk_path.exists():
            msg = f"No crosswalk from {self.grid.name} to {self.options.summarize_by}"
            self._error(msg)

        crosswalk_lf = pl.scan_parquet(str(crosswalk_path))

        if self.grid_cell_ids:
            crosswalk_lf = crosswalk_lf.filter(pl.col("src_id").is_in([int(g) for g in self.grid_cell_ids]))

        if self.summarize_grid_cell_ids:
            crosswalk_lf = crosswalk_lf.filter(pl.col("dst_id").is_in(self.summarize_grid_cell_ids))

        dim_cols = cs.expand_selector(self.lf, DIMENSION_COLUMNS)

        if self.options.summarize_aggregate == "count_by_value":
            # No `scale_factor` multiply here (see `stages.reject_calculate_with_count_by_value`'s
            # docstring for the same reasoning applied to the crosswalk's own scale
            # factor: rescaling a categorical code before counting it would change which
            # raw values collapse into "the same" category). Group by the destination
            # zone (aliased to `grid_cell`, matching every other summarize output) plus
            # the value itself -- `count_by_value_frame` bounds this group key's
            # cardinality; see its docstring for the math.
            stages.reject_calculate_with_count_by_value(self.options.calculate)
            joined = self.lf.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner").with_columns(
                pl.col("dst_id").alias("grid_cell")
            )
            try:
                self.lf = count_by_value_frame(
                    joined, [*dim_cols, "grid_cell"], "value", COUNT_BY_VALUE_MAX_DISTINCT
                ).with_columns(geographic_unit=pl.lit(self.options.summarize_by))
            except ExpressionError as exc:
                self._error(str(exc))
            return

        # Join the data on grid_cell == src_id, scale the values, then aggregate
        # per destination grid_cell. The join replaces the wide-format horizontal
        # aggregation that was used when each source grid_cell was a column.
        agg_fn = self.options.summarize_aggregate

        def _scale_and_join(chunk):
            return chunk.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner").with_columns(
                value=pl.col("value") * pl.col("scale_factor")
            )

        # The destination zone is aliased to `grid_cell` here too, so the same
        # high-cardinality signal `_collapse_group` uses applies -- and it has to:
        # some crosswalks land on a FINER destination than the source grid (e.g.
        # hydro_compartment_v001 -> clara_grid_cell_v001, 109,854 distinct destination
        # cells), so this reaggregation is not uniformly safe just because most
        # `summarize_by` targets (ecoregion, parish, hydro_compartment, ...) are small.
        # `summarize_grid_cell` narrows destinations the same way `grid_cell` narrows
        # sources, so it's added to the same selective-filter signal here.
        key_names = [*dim_cols, "grid_cell"]
        has_selective_filter = (
            self._has_selective_dimension_filter() or self.options.summarize_grid_cell is not None
        )
        if should_fold_group_by(key_names, has_selective_filter=has_selective_filter):
            source_lf = self.lf
            try:
                collapsed = fold_group_by(
                    lambda: source_lf,
                    [*dim_cols, pl.col("dst_id").alias("grid_cell")],
                    agg_fn,
                    chunk_transform=_scale_and_join,
                )
            except FoldError as exc:
                self._error(str(exc))
            self.lf = collapsed.lazy().with_columns(geographic_unit=pl.lit(self.options.summarize_by))
        else:
            joined = self.lf.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner").with_columns(
                value=pl.col("value") * pl.col("scale_factor")
            )
            self.lf = (
                joined.group_by(*dim_cols, pl.col("dst_id").alias("grid_cell"))
                .agg(getattr(pl.col("value"), agg_fn)())
                .with_columns(geographic_unit=pl.lit(self.options.summarize_by))
            )

    # -- ops interpreter: applies a pipeline's ops in declared order, vector shape -------

    async def apply_params(self):
        """Dispatch to the ops interpreter (`apply_steps`) when `self._pipeline_ops` was
        attached -- `planner._compile_single_input_chain`'s fast path, which attaches it
        unconditionally -- else fall through to `BaseAnalysis.apply_params`'s fixed
        five-stage flow unchanged. Every flat (non-pipeline) request reaches this dataset
        with `self._pipeline_ops` still at its `None` default (only
        `_compile_single_input_chain` ever sets it), so this override changes nothing for
        the entire existing flat surface -- only a pipeline request that both (a) has a
        single vector input and no `join` and (b) actually compiled (passed every existing
        check in `_compile_single_input_chain`, the SUMMARIZING-PROJECT ORDER GUARD
        included -- this override runs strictly after that function returns) takes the new
        path.
        """
        if self._pipeline_ops is not None:
            await self.apply_steps(self._pipeline_ops)
            self._apply_sort_params()
            return
        await super().apply_params()

    async def apply_steps(self, ordered_ops: list[Op]):
        """The vector ("long"-shape) interpreter: walk `ordered_ops`
        (`ops.compile_ops`'s output) and apply each op's `ops.LOWERING_TABLE` row against
        `self.lf`, IN DECLARED ORDER -- unlike `BaseAnalysis.apply_params`'s fixed five-
        stage flow (filter always before calculate always before group always before
        transform), which silently produces the wrong answer whenever a pipeline's
        declared order differs from that fixed order (see `planner.py`'s SUMMARIZING-
        PROJECT ORDER GUARD comment for the measured reproductions this interpreter makes
        unnecessary for the shapes it covers). Nothing here ever reorders `ordered_ops`; it
        is walked once, front to back, and every op's own re-entrant core
        (`apply_dimension_filter`/`apply_value_filter`/
        `apply_calculate_plans` from `analysis/base.py`, `apply_group_by`/`apply_project`
        above) is exactly what the corresponding fixed-stage method already used, called
        once per op instead of once per whole-request option.

        `self.options.grid_cell` (an INPUT-level filter -- `InputSelection.grid_cell`,
        never a pipeline step, so it never appears in `ordered_ops` at all) is applied
        once, up front, via the existing `_apply_grid_cell` hook -- the same relative
        timing `BaseAnalysis.apply_params`'s stage 1 gives it (before any calculate/group/
        transform), and the only sensible spot here too, since it is a SOURCE-grid cell
        filter and must run before a `ProjectOp` ever changes what `grid_cell` means.

        `rename_map`: every name a `FilterOp.where`/`CalculateOp.expression` may reference
        the running value BY -- the pipeline's own input name, plus every step's own
        EXPLICIT `as_` alias -- rewritten to the literal `value` placeholder before
        parsing, via `expressions.rewrite_named_columns`. Required because `ops.
        compile_ops` copies `where`/`expression` VERBATIM from the validated `Pipeline`
        (see that module's own docstring), the same rewrite `planner._compile_single_
        input_chain` already applies (its own `rename_map`, built the identical way --
        `input_name` plus each step's own `.as_` when not `None`) before lowering onto
        `Options.where`/`Options.calculate`.

        Deliberately EXCLUDES an op's SYNTHETIC default output name (`ops.
        _default_output_name`'s `str(step_no)`, used for any step with no explicit `as_`)
        -- including it here was tried and is actively wrong, not merely redundant: a
        synthetic name is a bare digit string (`"1"`, `"2"`, ...), and `rewrite_named_
        columns`'s regex has no way to tell "the token `2` because a caller's `where`
        legitimately compares against the number 2" from "the token `2` because it
        happens to equal step 2's made-up output name" -- both look identical to a plain
        word-boundary match. Concretely: a leading `filter(where="eco >= 2")` followed by
        a `project` step (no `as_` on either) synthesizes output names `"1"`/`"2"`; if
        `"2"` were in `rename_map`, `"eco >= 2"` would rewrite to `"value >= value"` (a
        tautology, filtering nothing) instead of `"value >= 2"`. `ops._default_output_
        name`'s own docstring proves the exact filter used below is safe: a real `as_`/
        input name is required (`pipeline._validate_name`'s `_NAME_RE`) to start with a
        lowercase letter or underscore, so `op.output[0].isdigit()` is `True` if and only
        if `output` is synthetic.

        Every op other than `project` is lowered by the shared `apply_long_op` above (the
        long row of `ops.LOWERING_TABLE` for `filter`/`calculate`/`group`), which
        `raster.RasterGridAnalysis.apply_steps` calls with the identical arguments once its
        own frame has reached the long shape -- one implementation of the long lowering
        rather than two that have to be kept in agreement. `project` stays here because it
        is the op that CHANGES shape, so its lowering differs per direction.

        Errors: every lowering called below is a plain library function raising
        `ExpressionError`/`FoldError`, translated into an `AnalysisError` exactly once by
        `base.op_error_boundary` around the whole walk rather than by a `try/except` at each
        call site -- see that context manager's docstring.

        `ordered_ops[0].input` is always the pipeline's own (single, since this method is
        only ever reached for a single-input, no-`join` chain) input name -- `compile_ops`
        resolves every implicit chain, and a chain's first step always names its input
        explicitly (`pipeline.validate_pipeline`'s chaining rule).

        `current_grid` tracks which grid `self.lf`'s `grid_cell` column actually indexes:
        starts at `self.grid.name` (the input's own grid) and becomes a `ProjectOp.to`
        once one runs. A single-input, no-`join` pipeline can contain at most one
        `ProjectOp` -- `_compile_single_input_chain`'s post-project step loop has no
        `ProjectStep` branch at all, so a second `project` falls into that loop's
        exhaustive `else` and is rejected with a `PipelineError` well before this method
        is ever reached -- so this is a plain reassignment, never a stack, and
        `geographic_unit` (added once, at the very end, exactly like
        `_apply_transform_params_impl` adds it once) is added only if `current_grid` ever
        actually changed -- an identity project (`op.to == current_grid`, skipped
        entirely, matching `_apply_transform_params_impl`'s own `stages.validate_
        summarize_options` early return for a project with no `summarize_by`) leaves it at
        `self.grid.name` and adds nothing, the same "no-op transform stage" behavior the old
        engine gives an identity project.

        `count_by_value` + `calculate`: mirrors `stages.reject_calculate_with_count_by_value`'s
        rejection (checked against `self.options.calculate`, which the planner still
        populates identically regardless of which engine executes -- see `_dataset_for`)
        rather than re-deriving it from `ordered_ops`, since `Options.calculate` is already
        the exact signal that method's caller (`_apply_transform_params_impl`) checks.

        Filter semantics (a value conjunct is a row DROP, never a mask): see
        `apply_long_op`'s docstring, which now owns that lowering.
        """
        if self.options.grid_cell:
            await self._apply_grid_cell()

        lf = self.lf
        current_grid = self.grid.name
        value_names = {ordered_ops[0].input}
        value_names.update(op.output for op in ordered_ops if not op.output[0].isdigit())
        rename_map = {name: "value" for name in value_names}

        with op_error_boundary():
            for op in ordered_ops:
                if isinstance(op, ProjectOp):
                    if op.to == current_grid:
                        continue
                    if op.aggregate == "count_by_value":
                        stages.reject_calculate_with_count_by_value(self.options.calculate)
                    lf = apply_project(
                        lf,
                        current_grid,
                        op.to,
                        op.aggregate,
                        has_selective_filter=self._has_selective_dimension_filter(),
                    )
                    current_grid = op.to
                else:
                    lf, has_dimension_conjunct = apply_long_op(
                        lf,
                        op,
                        rename_map=rename_map,
                        has_selective_filter=self._has_selective_dimension_filter(),
                    )
                    self._where_has_dimension_conjunct = (
                        self._where_has_dimension_conjunct or has_dimension_conjunct
                    )

        if current_grid != self.grid.name:
            lf = lf.with_columns(geographic_unit=pl.lit(current_grid))

        self.lf = lf

    @asynctiming
    async def get_vector_tile(self, x: int, y: int, z: int) -> bytes | None:
        """Get an MVT tile with dataset values injected into pre-generated geometry."""
        grid_name = self.options.summarize_by or self.grid.name
        tile = mercantile.Tile(x, y, z)

        compressed = read_pmtile(grid_name, tile)
        if compressed is None:
            return None

        # Parse once: get the VectorTile object and boundary feature IDs together.
        # The parsed object is reused for injection, avoiding a second decompress+parse.
        vt, raw_ids = parse_tile(compressed)
        if not raw_ids:
            return None

        tile_id_strs = [str(fid) for fid in raw_ids]
        if self.summarize_grid_cell_ids is not None:
            valid = {str(gid) for gid in self.summarize_grid_cell_ids}
            val_cols = [c for c in tile_id_strs if c in valid]
        elif self.grid_cell_ids is not None:
            valid = {str(gid) for gid in self.grid_cell_ids}
            val_cols = [c for c in tile_id_strs if c in valid]
        else:
            val_cols = tile_id_strs

        if not val_cols:
            return None

        tile_ints = [int(c) for c in val_cols]
        df = self.lf.filter(pl.col("grid_cell").is_in(tile_ints)).collect()
        if df.is_empty():
            return None

        id_values = _data_to_id_value(df)
        if not id_values:
            return None

        return inject_and_serialize(vt, id_values)
