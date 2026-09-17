"""`VectorGridAnalysis`: the vector half of the former `api/dataset.py`'s `Dataset` class,
plus the pure MVT (Mapbox Vector Tile) helper functions formerly in `api/vector.py` (moved
here because `get_vector_tile` below is their only caller besides tests, and
`cpra.mp.data.analysis` may not import FastAPI -- see `analysis/base.py`'s docstring).
`api/vector.py` keeps only `VectorTileResponse`, which calls `get_vector_tile` on whatever
`BaseAnalysis` subclass it's given rather than these helpers directly.

See `analysis/base.py`'s module docstring for the `apply_params` flow and the hook
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
from cpra.mp.data.analysis.execution.fold import FoldError, fold_group_by, should_fold_group_by
from cpra.mp.data.analysis.io.geography import (
    parse_coordinate_pairs,
    resolve_vector_grid_cell,
)
from cpra.mp.data.analysis.io.raster_io import asynctiming, unpack_values
from cpra.mp.data.analysis.ops import CalculateOp, FilterOp, GroupOp, Op, ProjectOp, calculate_label
from cpra.mp.data.analysis.request.expressions import (
    ExpressionError,
    compile_calculate_entries,
    count_by_value_frame,
    rewrite_named_columns,
)
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


# -- re-entrant op cores, the long-shape half of the lowering table -------------------
#
# Every long-shape lowering this module has, as a free function taking its own arguments
# and returning a new `lf` instead of mutating an instance -- exactly the
# `apply_dimension_filter`/`apply_value_filter`/`apply_calculate_plans` pattern
# `analysis/base.py` establishes for filter/calculate; see that module's docstring for why
# each one raises a plain `FoldError`/`AnalysisError` rather than calling `self._error`.
#
# `apply_summarize` holds what the `_apply_transform_params_impl` hook used to run
# inline, so that method is now three lines of option-reading. `apply_grid_cell_selection`
# is called directly by `apply_steps` itself, terminally (see that method's docstring) --
# there is no per-instance hook for `grid_cell` at all, since narrowing the OUTPUT is not a
# per-subclass concept the way the summarize transform is. `apply_group_by` and
# `apply_project` are reached only from `apply_steps` (this module, below), per
# `GroupOp`/`ProjectOp` -- a flat request's `group_by`/`group_aggregate` gets there too, as
# the `GroupOp`
# `stages.lower_flat_options` emits for it.
def apply_group_by(lf, by, aggregate, *, of="value", has_selective_filter):
    """Collapse `lf` by `[*by, "grid_cell"]`, aggregating `of` with `aggregate` -- the
    LONG-shape lowering of `ops.GroupOp`, and the whole of it. `of` is explicit rather than
    assumed to be `"value"` so `GroupOp.of` (meaningful for the paired/join shape's own
    synthetic value columns -- see that op's docstring; a single-input pipeline's
    `GroupOp.of` is always `"value"`, per `planner._compile_single_input_chain`'s own
    `of != "value"` rejection) can be honored uniformly if a future join-shaped
    interpreter reuses this function. Folds through `analysis.fold.fold_group_by` when
    `should_fold_group_by` says the collapse is cliff-shaped. May raise `FoldError` (see
    `fold_group_by`); callers convert to `AnalysisError`.

    A dimension column left out of `by` is simply absent from the result --
    `group_by(...).agg(...)` (and `fold_group_by`, which delegates to the same Polars
    primitive per chunk) already drops any column that is neither a group key nor an
    aggregated expression, so no extra step is needed to make that happen. This used to be
    undone right after: every such column was added straight back as an all-null,
    Null-dtype placeholder, so a caller saw a schema-stable but meaningless column for
    every dimension the request didn't group by (e.g. `model_group_id`, `scenario_id`,
    `calendar_day` all appearing, entirely null, in a plain `group_aggregate='count'`
    response). Dropping instead of null-filling is the honest shape for "this dimension no
    longer has a value" and matches this module's own crosswalk-reaggregation lowering
    (`apply_summarize`) and `PairedAnalysis`/the pipeline shape, neither of which ever
    reintroduces a collapsed dimension as null -- see the README's `analyze_data` example,
    whose documented output columns (`grid_cell, calendar_year, value`) already assumed the
    collapsed dimensions were simply gone. A `select` naming a since-collapsed dimension
    surfaces as a clean `ColumnNotFoundError` -> 400 out of `materialize`, rather than
    silently returning an all-null column.
    """
    key_names = [*by, "grid_cell"]
    if should_fold_group_by(key_names, has_selective_filter=has_selective_filter):
        source_lf = lf
        collapsed = fold_group_by(lambda: source_lf, key_names, aggregate, value_column=of)
        return collapsed.lazy()
    return lf.group_by(*by, "grid_cell").agg(getattr(pl.col(of), aggregate)())


async def apply_grid_cell_selection(lf, grid_name, grid_cell):
    """Narrow `lf` (a long frame) to the cells `grid_cell` names, returning
    `(lf, grid_cell_ids)`.

    Accepts either grid cell IDs or lat/lng coordinate pairs, matching the raster
    convention. A grid_cell value is treated as IDs only if every token is a plain
    (unsigned, non-decimal) digit string; otherwise the whole list is parsed as
    coordinate pairs and resolved to grid cell IDs via point-in-polygon lookup against
    the grid's geometry. Coordinates that don't fall inside any polygon are dropped
    rather than erroring.

    Either way the result is reported under grid cell IDs, never the coordinates that
    selected them: a vector grid_cell *is* its ID. Several coordinates landing in one
    cell therefore select it once, which is what the de-duplication is for -- and why the
    surviving IDs come back to the caller rather than staying local: `VectorGridAnalysis.
    apply_steps` stores them on `self.grid_cell_ids`, which nothing downstream of this
    call reads any more (it is applied last, so there is no "downstream" left in the walk)
    but is still surfaced for callers outside the walk that inspect it after the fact (see
    `get_vector_tile`).

    `grid_name` is whichever grid `lf`'s `grid_cell` column actually indexes AT THE POINT
    THIS RUNS -- not necessarily the analysis's own input grid -- since this is called
    terminally, against the pipeline's OUTPUT, by `VectorGridAnalysis.apply_steps` (see
    that method's docstring for how it resolves `final_grid`).
    """
    values = unpack_values(grid_cell, "grid_cell")

    if all(value.isdigit() for value in values):
        # TODO: Validate that the given IDs fall within the grid min and max.
        grid_cell_ids = values
    else:
        lngs, lats = parse_coordinate_pairs(values)
        resolved = await resolve_vector_grid_cell(grid_name, lngs, lats)
        grid_cell_ids = [str(geo_id) for geo_id in resolved if geo_id is not None]

    grid_cell_ids = list(dict.fromkeys(grid_cell_ids))
    return lf.filter(pl.col("grid_cell").is_in([int(g) for g in grid_cell_ids])), grid_cell_ids


def apply_project(lf, from_grid, to_grid, aggregate, *, has_selective_filter):
    """Move `lf`'s rows from `from_grid`'s cells onto `to_grid`'s cells via a crosswalk-
    table join, aggregating with `aggregate` -- the re-entrant core of `apply_summarize`'s
    join/reaggregate logic (below), generalized to explicit grid names/aggregate instead
    of `summarize_by`/`summarize_aggregate`, for `apply_steps` (below) to call directly
    against a `ProjectOp`.

    Deliberately does NOT pre-filter the crosswalk by `summarize_grid_cell_ids` the way
    `apply_summarize` does -- that is a performance narrowing specific to
    `Options.summarize_grid_cell`, not a correctness requirement: a post-project literal
    `grid_cell = N` filter reaches this interpreter as an ordinary `FilterOp` on the joined
    result's own `grid_cell` column (compile_ops copies the raw `Pipeline`'s steps
    verbatim), and filtering the DESTINATION rows after the join is mathematically
    identical to filtering the crosswalk's `dst_id` before it, just less optimized.
    `Options.grid_cell` itself never reaches this function at all any more -- it is
    applied once, terminally, by `apply_steps` after this function (and everything else in
    the walk) has already run -- so there is nothing of ITS to pre-narrow the crosswalk by
    here either. Also, unlike `apply_summarize`, never appends `geographic_unit` itself -- a
    pipeline's `ProjectOp` may not be the request's LAST grid change (though a single-input chain
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


def apply_summarize(
    lf,
    *,
    grid_name,
    summarize_by,
    summarize_aggregate,
    summarize_grid_cell,
    calculate,
    has_selective_filter,
):
    """The `summarize_by`/`summarize_aggregate` cross-grid reaggregation for a long frame,
    returning `(lf, summarize_grid_cell_ids)`. `lf` comes back unchanged (and the IDs as
    `stages.validate_summarize_options` resolved them) when the request declares no
    summarize at all.

    This is the OLD, `Options`-shaped entry point; `apply_project` above is the op-shaped
    one, and the two differ in three ways that are all `Options`-only:

    - the crosswalk is pre-narrowed by `summarize_grid_cell_ids` (destinations), a
      performance narrowing with no op equivalent -- see `apply_project`'s docstring for
      why the op form does not need it. There is no analogous SOURCE-side narrowing:
      `grid_cell` no longer selects source cells at all (it is applied terminally, against
      this function's OUTPUT, by `VectorGridAnalysis.apply_steps` -- see that method's
      docstring), so this function has nothing of the caller's to pre-narrow the crosswalk
      by on that side;
    - `geographic_unit` is appended here, because `summarize_by` is by construction the
      request's last grid change, whereas `apply_steps` adds that column once from the
      final grid instead;
    - `count_by_value` rejects a co-declared `calculate` (see
      `stages.reject_calculate_with_count_by_value`).

    Raises `AnalysisError` for a missing crosswalk and for the `FoldError`/
    `ExpressionError` the fold and the count-by-value bound may raise, so the caller has
    nothing left to convert.
    """
    has_summarize, summarize_grid_cell_ids = stages.validate_summarize_options(
        summarize_by, summarize_aggregate, summarize_grid_cell
    )
    if not has_summarize:
        return lf, summarize_grid_cell_ids

    # Generate column expressions.
    crosswalk_path = config.CROSSWALKS_PATH / f"{grid_name}__{summarize_by}.parquet"

    # scan_parquet is lazy and won't raise for a missing file until collection,
    # so check up front to report the missing crosswalk as a 400.
    if not crosswalk_path.exists():
        raise AnalysisError(f"No crosswalk from {grid_name} to {summarize_by}")

    crosswalk_lf = pl.scan_parquet(str(crosswalk_path))

    if summarize_grid_cell_ids:
        crosswalk_lf = crosswalk_lf.filter(pl.col("dst_id").is_in(summarize_grid_cell_ids))

    dim_cols = cs.expand_selector(lf, DIMENSION_COLUMNS)

    if summarize_aggregate == "count_by_value":
        # No `scale_factor` multiply here (see `stages.reject_calculate_with_count_by_value`'s
        # docstring for the same reasoning applied to the crosswalk's own scale
        # factor: rescaling a categorical code before counting it would change which
        # raw values collapse into "the same" category). Group by the destination
        # zone (aliased to `grid_cell`, matching every other summarize output) plus
        # the value itself -- `count_by_value_frame` bounds this group key's
        # cardinality; see its docstring for the math.
        stages.reject_calculate_with_count_by_value(calculate)
        joined = lf.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner").with_columns(
            pl.col("dst_id").alias("grid_cell")
        )
        try:
            counted = count_by_value_frame(joined, [*dim_cols, "grid_cell"], "value", COUNT_BY_VALUE_MAX_DISTINCT)
        except ExpressionError as exc:
            raise AnalysisError(str(exc))
        return counted.with_columns(geographic_unit=pl.lit(summarize_by)), summarize_grid_cell_ids

    # Join the data on grid_cell == src_id, scale the values, then aggregate
    # per destination grid_cell. The join replaces the wide-format horizontal
    # aggregation that was used when each source grid_cell was a column.
    def _scale_and_join(chunk):
        return chunk.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner").with_columns(
            value=pl.col("value") * pl.col("scale_factor")
        )

    # The destination zone is aliased to `grid_cell` here too, so the same
    # high-cardinality signal `apply_group_by` uses applies -- and it has to:
    # some crosswalks land on a FINER destination than the source grid (e.g.
    # hydro_compartment_v001 -> clara_grid_cell_v001, 109,854 distinct destination
    # cells), so this reaggregation is not uniformly safe just because most
    # `summarize_by` targets (ecoregion, parish, hydro_compartment, ...) are small.
    # `summarize_grid_cell` narrows destinations the same way a `WHERE grid_cell IN
    # (...)` would narrow a plain group's key, so it's added to the same
    # selective-filter signal here.
    key_names = [*dim_cols, "grid_cell"]
    has_selective_filter = has_selective_filter or summarize_grid_cell is not None
    if should_fold_group_by(key_names, has_selective_filter=has_selective_filter):
        source_lf = lf
        try:
            collapsed = fold_group_by(
                lambda: source_lf,
                [*dim_cols, pl.col("dst_id").alias("grid_cell")],
                summarize_aggregate,
                chunk_transform=_scale_and_join,
            )
        except FoldError as exc:
            raise AnalysisError(str(exc))
        summarized = collapsed.lazy()
    else:
        joined = lf.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner").with_columns(
            value=pl.col("value") * pl.col("scale_factor")
        )
        summarized = joined.group_by(*dim_cols, pl.col("dst_id").alias("grid_cell")).agg(
            getattr(pl.col("value"), summarize_aggregate)()
        )
    return summarized.with_columns(geographic_unit=pl.lit(summarize_by)), summarize_grid_cell_ids


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

    Filter semantics: a `FilterOp`'s value conjunct is a ROW DROP unless the op itself asks
    for a mask (`FilterOp.mask_value`, which only `stages.lower_flat_options` ever sets).
    `ops.LOWERING_TABLE` states the long-shape filter lowering as a literal "row filter",
    and under TRUE declared-order execution a drop is the only semantically coherent
    choice: a mask exists solely to compose with a downstream aggregate's own null-skipping
    behavior when the filter and the aggregate are applied out of their declared order --
    which is exactly, and only, what the FLAT request surface does, since `Options` has no
    declared order and its single `where` is always emitted ahead of its
    `group_by`/`summarize_by`. See `ops.FilterOp.mask_value`'s docstring for the full
    argument. (The band-wide lowering is neither: always a mask, regardless of this field,
    because that shape must stay positionally aligned to the raster -- see
    `apply_value_filter_band_wide`.)
    """
    if isinstance(op, FilterOp):
        rewritten = rewrite_named_columns(op.where, rename_map)
        lf, value_where, has_dimension_conjunct = apply_dimension_filter(lf, rewritten)
        return apply_value_filter(lf, value_where, mask=op.mask_value), has_dimension_conjunct
    if isinstance(op, CalculateOp):
        expression = rewrite_named_columns(op.expression, rename_map)
        where = rewrite_named_columns(op.where, rename_map) if op.where is not None else None
        plans = compile_calculate_entries([(calculate_label(op), expression, where)])
        return apply_calculate_plans(lf, plans), False
    if isinstance(op, GroupOp):
        return (
            apply_group_by(lf, op.by, op.aggregate, of=op.of, has_selective_filter=has_selective_filter),
            False,
        )
    raise ExpressionError(f"{op.op!r} has no long-shape lowering")  # pragma: no cover


class VectorGridAnalysis(BaseAnalysis):
    async def _apply_transform_params_impl(self):
        """The transform hook for a vector frame -- a thin caller over the free function
        `apply_summarize`, which holds the whole lowering (see its docstring). What stays
        here is what touches `self`: reading the four options it needs and storing the two
        results back.

        Declared `async` (with no internal `await`) purely to match the one hook name
        both interpreters call on both subclasses -- see `analysis/base.py`'s module
        docstring for why the vector/raster pair collapses to a single hook.
        """
        self.lf, self.summarize_grid_cell_ids = apply_summarize(
            self.lf,
            grid_name=self.grid.name,
            summarize_by=self.options.summarize_by,
            summarize_aggregate=self.options.summarize_aggregate,
            summarize_grid_cell=self.options.summarize_grid_cell,
            calculate=self.options.calculate,
            has_selective_filter=self._has_selective_dimension_filter(),
        )

    # -- ops interpreter: applies a pipeline's ops in declared order, vector shape -------

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
        above) is a plain free function, called once per op rather than once per
        whole-request option.

        `self.options.grid_cell` (never a pipeline step, so it never appears in
        `ordered_ops` at all) is applied once, terminally, AFTER the whole walk below --
        it narrows what comes OUT of this pipeline, not what goes in. A caller who wants
        an INPUT-side cell filter can express it directly as a leading `filter` step
        instead. Applied against `final_grid` (see the bottom of this method): whichever
        grid `self.lf`'s `grid_cell` column actually indexes by the time the walk (and any
        flat `summarize_by`) has finished, so a coordinate in `grid_cell` resolves against
        the OUTPUT grid, not necessarily `self.grid.name`. `apply_group_by`/`apply_project`/
        `apply_summarize` all keep `grid_cell` (or the destination zone aliased to it) as
        an implicit group key, so it is always present in `self.lf` by the time this runs,
        whatever ops the pipeline declared.

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
        lf = self.lf
        current_grid = self.grid.name
        # `ordered_ops` is empty for a request that declared nothing this interpreter can
        # execute -- a flat selection with no `where`/`calculate`/`group` at all, which
        # `stages.lower_flat_options` lowers to no ops (it may still carry `grid_cell`
        # above, and `summarize_by`/`sort`/`select`, none of which are ops). The walk below
        # is then a no-op and `rename_map` has nothing to rename.
        value_names = {op.input for op in ordered_ops[:1]}
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

        self.lf = lf

        if current_grid != self.grid.name:
            # A `ProjectOp` above actually reaggregated onto another grid, so `apply_project`
            # -- which, unlike `apply_summarize`, does not label its own output -- gets its
            # `geographic_unit` column here, once.
            self.lf = lf.with_columns(geographic_unit=pl.lit(current_grid))
        else:
            # Nothing reaggregated, so `summarize_by`/`summarize_aggregate` (which have no
            # op form -- see `stages.lower_flat_options`) still need their always-last
            # position, exactly the fixed spot the retired transform stage gave them. This
            # is the branch every FLAT request takes, and it is what makes a flat
            # `summarize_by=` work through this interpreter at all.
            #
            # The two branches are exclusive, not merely ordered: `current_grid` moves off
            # `self.grid.name` only for a summarizing `ProjectOp`, and that is exactly the
            # case where `planner._compile_single_input_chain` DID set
            # `Options.summarize_by` -- so calling the hook here as well would summarize a
            # second time. An IDENTITY project sets no `summarize_by` at all (see that
            # function's `first.to == selection.grid` branch), which is why taking this
            # branch for one is a no-op: `stages.validate_summarize_options` returns
            # "no summarize requested" and `apply_summarize` returns the frame untouched.
            await self._apply_transform_params_impl()

        if self.options.grid_cell:
            # Terminal `grid_cell`: filter the OUTPUT, on whichever grid it actually ended
            # up on. `current_grid` moves off `self.grid.name` only for a summarizing
            # `ProjectOp` (see the comment above); a flat `summarize_by` moves the grid
            # too, but through the OTHER branch, so `current_grid` alone would miss it --
            # the two are mutually exclusive (the same comment explains why), so checking
            # both covers every way `self.lf` can end up on a grid other than its own.
            final_grid = current_grid if current_grid != self.grid.name else (
                self.options.summarize_by or self.grid.name
            )
            self.lf, self.grid_cell_ids = await apply_grid_cell_selection(
                self.lf, final_grid, self.options.grid_cell
            )

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
