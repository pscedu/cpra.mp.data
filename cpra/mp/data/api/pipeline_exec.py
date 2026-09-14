"""Compile an already-validated `Pipeline` (api/pipeline.py) into something runnable, and
run it.

`api/pipeline.py` builds and validates the REPRESENTATION only -- everything it defers to
"compile time" (its own phrase, used throughout that module's docstrings) because it needs
catalog/crosswalk knowledge lands here: dimension-name validity, `project` cardinality
direction, `join` alignment/pinning, the `filter.where` column namespace, and the general
single-value-column output rule. Like `api/expressions.py` and `api/fold.py`, every function
here takes and returns plain values and raises `PipelineError` (or lets `pipeline.py`'s own
`PipelineError` propagate) -- never `HTTPException` -- so this module stays usable outside
the API. The ONE deliberate exception: a pipeline that "lowers" onto the existing `Dataset`
machinery (see `_compile_single_input_chain`) inherits `Dataset`'s own `HTTPException`-raising
code paths (`LazyFrameSource.materialize()`'s catch clauses, `Dataset._error`) at EXECUTION
time, since it reuses that code verbatim rather than reimplementing it -- only the compile-time
checks added in this module itself raise plain `PipelineError`.

**Two shapes compile today** (anything else is rejected with an informative "not yet
supported in analysis pipelines: <reason>" message -- v1 does not need to compile every
op sequence, per the spec):

1. **Single-input chains** (`_compile_single_input_chain`): one input, no `join`, opening
   with a `project` step (mandatory -- `pipeline.py`'s chaining rule makes `filter`/`group`
   impossible as a pipeline's first step, so even a pipeline that changes no grid at all
   must open with an IDENTITY `project`, `to` equal to its own input's grid), optionally
   followed by `filter` step(s) and then at most one `group` step. This compiles onto the
   EXISTING `Options`-driven `Dataset` machinery -- a many-to-one `project` becomes
   `Options.summarize_by`/`summarize_aggregate`; `filter.where` (written against the
   input's own name, per `pipeline.py`'s convention -- see `TestFilterStep` in
   `test_pipeline.py`) is rewritten to the `value` placeholder `Options.where` expects;
   `group` becomes `Options.group_by`/`group_aggregate`. The compiled result is the
   constructed `Dataset` INSTANCE ITSELF -- no wrapper class -- since `Dataset` already
   implements every method both `PolarsStreamingResponse` and
   `RasterioStreamingResponse`/`RasterioTileResponse` need.

2. **The paired (spatially-joined) shape** (`_compile_paired_notebook_shape`): exactly two
   inputs, a coarse one broadcast onto a finer RASTER grid via `project`, joined against the
   finer input via `join`, then any number of `filter` steps and exactly one trailing `group`
   step with `by=[]` (a per-pixel temporal collapse, not a zonal one) -- the exit-criterion
   notebook query's exact shape. Compiles to a `PairedPlan`, a small class implementing the
   same duck-typed raster subset `RasterioStreamingResponse`/`RasterioTileResponse` need,
   with NO backing `Dataset` (`Dataset` has no "join two datasets" concept).

   **Why not `raster_source.scan_paired`, which already does exactly this join?**
   `scan_paired` requires a mandatory OUTPUT zone crosswalk (`zones_path`/`zone_ids`) and
   returns one row per (surviving pixel, output zone) -- exactly right for a TABULAR/zonal
   paired aggregate, but its output frame carries no positional (row/col) information at
   all, only a zone id. That makes it structurally unusable for driving a raster WRITE or a
   tile render, which need to put each pixel's collapsed value back at that pixel's own
   window position -- not because `scan_paired`'s `zone_ids` array is large (it is a small,
   sorted array of destination zone ids, e.g. 1657 for `hydro_compartment_v001`, not a
   per-pixel array), but because the shape of what it returns has already thrown the
   position away. So `PairedPlan` instead runs its own direct rasterio block-window loop
   (`_write_paired_notebook_raster`/`_paired_notebook_tile`), reusing `scan_paired`'s and
   `scan_rasters`'s own low-level primitives (`_dense_lookup_array`, `_paired_lookup_values`,
   `_batched_windows`, `raster_blocks_per_batch`, `horizontal_aggregate_expr`) directly,
   never its Polars IO-plugin/LazyFrame machinery -- see `PairedPlan._prepare`. Because a
   paired plan's output is unconditionally pixel-local (one row per pixel until the trailing
   per-pixel `group` collapses across bands, never across pixels), only `tif`/`png`/`jpg`
   output is implemented for this shape; a tabular paired output is deliberately left
   unsupported (see `_compile_paired_notebook_shape`'s rejection) rather than materializing
   the full joined frame, which is exactly the memory cliff this whole design avoids.
"""

from contextlib import ExitStack
from io import BytesIO

import mercantile
import numpy as np
import polars as pl
import rasterio as rio
from pydantic import ValidationError
from rasterio.coords import BoundingBox, disjoint_bounds
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from PIL import Image

from cpra.mp.data import config
from cpra.mp.data.api.dataset import Dataset
from cpra.mp.data.api.expressions import (
    ExpressionError,
    parse_value_expr,
    rewrite_named_columns,
    split_top_level_and,
)
from cpra.mp.data.api.models import DatasetDimensions, FilterDimensions, Options
from cpra.mp.data.api.pipeline import (
    FilterStep,
    GroupStep,
    JoinStep,
    Pipeline,
    PipelineError,
    ProjectStep,
    _SINGLE_VALUE_COLUMN_FORMATS,
)
from cpra.mp.data.api.raster_source import (
    _batched_windows,
    _dense_lookup_array,
    _paired_lookup_values,
    horizontal_aggregate_expr,
)
from cpra.mp.data.api.types import DataFormat, OutputType, TileFormat
from cpra.mp.data.api.utils import group_rows_by_path, raster_blocks_per_batch, raster_gdal_env
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.legends import LEGENDS, Legend
from cpra.mp.data.utils import get_input_data_frame, required_columns

# The column namespace a single-input pipeline's `filter.where` may reference once its own
# running value has been rewritten to `value` (see `_compile_single_input_chain`): every
# real filterable dimension, plus `grid_cell` and the placeholder itself. Built once at
# import time -- `FilterDimensions.model_fields` is itself fixed at import time already
# (built by `create_model` from `cpra.mp.data.dimensions.DIMENSIONS`).
_FILTER_ALLOWED_DIMENSION_NAMES = frozenset(FilterDimensions.model_fields) | {"grid_cell"}

_RASTER_OUTPUT_FORMATS = frozenset({DataFormat.tif.value, TileFormat.png.value, TileFormat.jpg.value})


# ------------------------------------------------------------------------------------------
# Compile-time checks shared by both shapes
# ------------------------------------------------------------------------------------------


def _check_input_dimensions(name, selection):
    """Reject a dimension filter naming something that isn't a real dimension at all --
    the check `pipeline.py`'s own docstring explicitly defers ("dimension NAME/VALUE
    validity in `InputSelection`'s filters ... needs the executor's compile-time
    schema") -- and separately, one naming a dimension that's real SOMEWHERE but not
    carried by THIS input's own dataset (its `variable`/`time_unit`/`grid` combination).
    `FilterDimensions.model_fields` only knows the first: it's built from every dimension
    that isn't `dataset`/`metadata`-typed GLOBALLY, so e.g. `aggregate_type` passes it for
    ANY input even though it's only real for a variable that lists it in
    `VARIABLES[...].required_dimensions` (`dimensions.make_dimensions` turns that into a
    per-variable `required` expression). Left unchecked, a globally-real-but-not-this-
    dataset's dimension used to compile clean and only fail once the pipeline actually
    scanned the dataset, as a bare polars `ColumnNotFoundError` with no pipeline-level
    context (which input, which dimension, what IS valid) at all.
    `get_input_data_frame`/`required_columns` (`cpra.mp.data.utils` -- the same pair
    `tabular.read_parquet` uses to resolve a dataset's own Hive-partition columns, see its
    module docstring) reconstruct exactly that per-dataset dimension set, so the second
    loop below re-derives it once per input and checks every filter against it. Value
    validity (a real, dataset-carried dimension given a value it doesn't have) is still
    left to `Dataset`'s own construction, which already raises a 422 for it via
    `get_lazy_frame`.
    """
    for dim in selection.dimension_filters:
        if dim not in FilterDimensions.model_fields:
            raise PipelineError(
                f"input {name!r} filters on unknown dimension {dim!r} -- known "
                f"filterable dimensions: {sorted(FilterDimensions.model_fields)}"
            )

    if not selection.dimension_filters:
        return

    dataset_dims = set(
        required_columns(
            get_input_data_frame(variable=selection.variable, time_unit=selection.time_unit, grid=selection.grid),
            include_dataset=False,
            include_metadata=False,
        )
    )
    for dim in selection.dimension_filters:
        if dim not in dataset_dims:
            raise PipelineError(
                f"input {name!r} filters on dimension {dim!r} -- {dim!r} is a real "
                f"dimension, but not one this dataset (variable={selection.variable!r}, "
                f"time_unit={selection.time_unit!r}, grid={selection.grid!r}) carries; "
                f"this dataset's own dimensions: {sorted(dataset_dims)}"
            )


def _project_direction(from_grid, to_grid, aggregate):
    """The cardinality direction a `project` step needs to check its `aggregate`
    requirement -- deferred by `pipeline.py` since it needs grid-type knowledge.
    `from_grid == to_grid` (an identity project, used purely to satisfy the chaining
    rule -- see the module docstring) is its own direction, checked by the caller before
    this function is even reached.
    """
    from_type = GRIDS[from_grid].grid_type
    to_type = GRIDS[to_grid].grid_type
    if from_type == "raster" and to_type == "raster":
        raise PipelineError(
            f"not yet supported in analysis pipelines: project from raster grid "
            f"{from_grid!r} to raster grid {to_grid!r} -- no raster-to-raster crosswalk "
            "is supported"
        )
    if from_type == "raster" and to_type == "vector":
        return "many_to_one"
    if from_type == "vector" and to_type == "raster":
        return "one_to_many"
    # vector -> vector: direction is not derivable from grid type alone (either grid
    # could be the finer one); the caller's own stated intent -- whether `aggregate` was
    # given -- is the only signal available without opening the crosswalk file itself.
    return "many_to_one" if aggregate is not None else "one_to_many"


def _validate_project_cardinality(step_no, from_grid, to_grid, aggregate):
    direction = _project_direction(from_grid, to_grid, aggregate)
    if direction == "many_to_one" and aggregate is None:
        raise PipelineError(
            f"step {step_no} (project) from {from_grid!r} to {to_grid!r} is many-to-one "
            "(zonal reaggregation) and requires an `aggregate`"
        )
    if direction == "one_to_many" and aggregate is not None:
        raise PipelineError(
            f"step {step_no} (project) from {from_grid!r} to {to_grid!r} is one-to-many "
            "(broadcast) and must not specify `aggregate` -- there is nothing to "
            "aggregate when broadcasting a coarser value onto finer cells"
        )
    return direction


def _validate_join_alignment(step_no, names, selections):
    """A dimension present on only SOME of a join's sides must be pinned to a single
    value on every side that has it (there is no row on the other side(s) to align it
    against); a dimension present on ALL sides must resolve to the same value set on
    every side. Deferred by `pipeline.py` since it needs to look at every input's own
    dimension filters together, which is exactly this function's job.
    """
    all_dims = set()
    for sel in selections:
        all_dims.update(sel.dimension_filters)

    for dim in sorted(all_dims):
        present = [(n, s) for n, s in zip(names, selections) if dim in s.dimension_filters]
        if len(present) == len(selections):
            value_sets = {frozenset(s.dimension_filters[dim]) for _, s in present}
            if len(value_sets) > 1:
                raise PipelineError(
                    f"step {step_no} (join) dimension {dim!r} does not align across "
                    f"{[n for n, _ in present]} -- every joined input filters it, but not "
                    "to the same value(s)"
                )
        else:
            missing = [n for n in names if n not in {p[0] for p in present}]
            for n, s in present:
                if len(s.dimension_filters[dim]) != 1:
                    raise PipelineError(
                        f"step {step_no} (join) dimension {dim!r} is present only on "
                        f"input {n!r} (not on {missing}) and must be pinned to a single "
                        f"value there -- got {s.dimension_filters[dim]!r}"
                    )


def _validate_filter_where(step_no, where, extra_allowed_columns):
    allowed = _FILTER_ALLOWED_DIMENSION_NAMES | set(extra_allowed_columns)
    for conjunct in split_top_level_and(where):
        try:
            parse_value_expr(conjunct, allowed_columns=allowed)
        except ExpressionError as exc:
            raise PipelineError(
                f"step {step_no} (filter) where conjunct {conjunct!r} is invalid: {exc}"
            ) from exc


def _check_join_row_ceiling(step_no, name, lf):
    n_rows = lf.select(pl.len()).collect(engine="streaming").item()
    if n_rows > config.ANALYSIS_JOIN_MAX_ROWS:
        raise PipelineError(
            f"step {step_no} (join) input {name!r} has {n_rows} rows, over the "
            f"{config.ANALYSIS_JOIN_MAX_ROWS}-row ceiling for a pipeline's coarse/zone "
            "side (see config.ANALYSIS_JOIN_MAX_ROWS) -- narrow this input's dimension "
            "filters"
        )


def _dataset_for(name, selection, base_path, **option_kwargs):
    try:
        dataset_dimensions = DatasetDimensions(
            variable=selection.variable, time_unit=selection.time_unit, grid=selection.grid
        )
        filter_dimensions = FilterDimensions(**selection.dimension_filters)
        options = Options(**option_kwargs)
    except ValidationError as exc:
        raise PipelineError(f"input {name!r}: {exc}") from exc
    return Dataset(dataset_dimensions, filter_dimensions, options, base_path=base_path)


def _track_output_columns(pipeline: Pipeline) -> int:
    """The pipeline's own value-column count through its step chain, tracked generally
    (not just the narrow "last step is a join" case `pipeline.py`'s own
    `_validate_output_rule` already catches without needing schema knowledge). A `join`
    step's output carries one named value column per joined input; a `group` step
    collapses back to one column wherever it appears, by construction; `project`/`filter`
    never change the count.
    """
    n = 1
    for step in pipeline.steps:
        if isinstance(step, JoinStep):
            n = len(step.inputs)
        elif isinstance(step, GroupStep):
            n = 1
    return n


def _validate_output_columns(pipeline: Pipeline) -> None:
    n = _track_output_columns(pipeline)
    if pipeline.output.format in _SINGLE_VALUE_COLUMN_FORMATS and n != 1:
        raise PipelineError(
            f"output format {pipeline.output.format!r} requires exactly one value "
            f"column, but this pipeline's compiled plan produces {n} (from a `join` "
            "with no subsequent `group` to narrow it) -- add a `group` step before this "
            "output"
        )


# ------------------------------------------------------------------------------------------
# Shape 1: single-input chains (lowered onto `Dataset`/`Options`)
# ------------------------------------------------------------------------------------------


def _compile_single_input_chain(pipeline: Pipeline, base_path):
    """Compile a one-input, no-`join` pipeline onto the existing `Dataset`/`Options`
    machinery, returning the constructed `Dataset` instance directly. Returns `None`
    (rather than raising) when the pipeline has more than one input or contains a
    `join` -- the signal for `compile_pipeline` to try the paired shape instead; every
    other structural mismatch for a genuinely single-input pipeline raises `PipelineError`
    naming the reason (this branch owns single-input pipelines outright once selected).
    """
    if len(pipeline.inputs) != 1 or any(isinstance(s, JoinStep) for s in pipeline.steps):
        return None

    ((input_name, selection),) = pipeline.inputs.items()
    _check_input_dimensions(input_name, selection)

    steps = pipeline.steps
    first = steps[0]
    # INVARIANT relied on below, not a runtime check: `first` is always a `ProjectStep`
    # here. A pipeline reaching this function has one input and no `join` step anywhere
    # (the guard above), so its step 1 can only be `project`, `filter`, or `group` --
    # and `pipeline.py`'s own `validate_pipeline` chaining rule (see its docstring)
    # already rejects, for every pipeline built through the sanctioned `parse_pipeline`
    # entry point, a step 1 that isn't `project`/`join` naming its input explicitly
    # ("step 1 (...) has no input to chain from"). A `PipelineError`+direct-call unit
    # test used to guard this branch too; removed as genuinely unreachable dead code
    # (not merely shadowed by call order -- there is no legitimate, `parse_pipeline`-
    # validated pipeline that can ever reach here with a non-`project` step 1; the only
    # way to trigger it was constructing a `Pipeline` directly and skipping
    # `validate_pipeline`, which no real caller does -- see pipeline.py's "Why business
    # validation is NOT wired into pydantic validators" docstring section).
    value_names = {input_name}
    if first.as_ is not None:
        value_names.add(first.as_)

    option_kwargs = {}
    if first.to == selection.grid:
        if first.aggregate is not None:
            raise PipelineError(
                f"step 1 (project) targets its own input's grid ({selection.grid!r}) -- "
                "no aggregate is possible without an actual grid change; omit "
                "`aggregate`"
            )
    else:
        direction = _validate_project_cardinality(1, selection.grid, first.to, first.aggregate)
        if direction == "one_to_many":
            raise PipelineError(
                "not yet supported in analysis pipelines: a single-input pipeline's "
                f"`project` from {selection.grid!r} to {first.to!r} broadcasts onto a "
                "finer grid -- the `summarize_by` mechanism this compiles onto only "
                "aggregates TOWARD a grid, so a broadcast has no single-input output "
                "shape; pair it with a `join` instead"
            )
        option_kwargs["summarize_by"] = first.to
        option_kwargs["summarize_aggregate"] = first.aggregate

    filters = []
    group_step = None
    for i, step in enumerate(steps[1:], start=2):
        if isinstance(step, FilterStep):
            if group_step is not None:
                raise PipelineError(
                    f"not yet supported in analysis pipelines: step {i} (filter) comes "
                    "after a group step in a single-input pipeline -- a filter cannot "
                    "run after the collapse a group performs"
                )
            filters.append((i, step.where))
            if step.as_ is not None:
                value_names.add(step.as_)
        elif isinstance(step, GroupStep):
            if group_step is not None:
                raise PipelineError(
                    "not yet supported in analysis pipelines: more than one `group` "
                    "step for a single input"
                )
            if step.of != "value":
                raise PipelineError(
                    f"not yet supported in analysis pipelines: step {i} (group) "
                    f"of={step.of!r} -- a single-input pipeline (no join) can only "
                    "aggregate its own running value (`of` must be left at its default, "
                    "'value')"
                )
            group_step = step
            if step.as_ is not None:
                value_names.add(step.as_)
        else:  # pragma: no cover -- exhaustive over the Step union
            raise PipelineError(
                f"not yet supported in analysis pipelines: step {i} ({step.op}) cannot "
                "follow a single-input pipeline's leading `project` step (only "
                "`filter`/`group` may)"
            )

    rename_map = {name: "value" for name in value_names}
    rewritten_filters = []
    for i, where in filters:
        rewritten = rewrite_named_columns(where, rename_map)
        _validate_filter_where(i, rewritten, {"value"})
        rewritten_filters.append(rewritten)
    if rewritten_filters:
        option_kwargs["where"] = " AND ".join(f"({w})" for w in rewritten_filters)

    if group_step is not None:
        # `by=[]` means "collapse to one row per grid_cell" (see `GroupStep`'s
        # docstring) -- that is `Options`' OWN "group_aggregate with no group_by"
        # shape (`Dataset._get_group_by_columns`'s `[True]` sentinel branch), not
        # "group_by the empty string". Passing `group_by=""` through instead would
        # hit `unpack_values("", ...)`, which 400s ("Invalid values for group_by: ")
        # since "" is never a member of `DIMENSION_CHOICES` -- so the empty-list
        # case must omit `group_by` entirely rather than joining it into "".
        if group_step.by:
            option_kwargs["group_by"] = ",".join(group_step.by)
        option_kwargs["group_aggregate"] = group_step.aggregate

    return _dataset_for(input_name, selection, base_path, **option_kwargs)


# ------------------------------------------------------------------------------------------
# Shape 2: the paired (spatially-joined) shape -- PairedPlan
# ------------------------------------------------------------------------------------------


def _legend_for(color_map):
    """Standalone reimplementation of `Dataset.legend`'s custom color-map parsing, for a
    `PairedPlan` -- which has no backing `Dataset` to delegate to and must raise
    `PipelineError`, not `Dataset._error`'s `HTTPException`.
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


class PairedPlan:
    """The compiled paired shape: a coarse input broadcast onto a finer raster grid via
    `project`, joined against the finer input via `join`, collapsed by a trailing `group`
    step -- the notebook query's exact shape (see the module docstring for why this
    doesn't use `raster_source.scan_paired`). Implements the duck-typed subset of
    `LazyFrameSource`'s interface `RasterioStreamingResponse`/`RasterioTileResponse` need,
    standing in for a `Dataset` (which has no "join two datasets" concept).
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


def _find_paired_notebook_shape(pipeline: Pipeline):
    """Structural pre-check for the paired shape, used only to decide whether
    `compile_pipeline` should attempt it at all (as opposed to a single-input chain) --
    the real validation happens in `_compile_paired_notebook_shape`, which raises
    informative `PipelineError`s for every way a two-input pipeline can still not match.
    Returns `True`/`False` only.
    """
    return len(pipeline.inputs) == 2


def _compile_paired_notebook_shape(pipeline: Pipeline, base_path):
    if len(pipeline.inputs) != 2:
        raise PipelineError(
            "not yet supported in analysis pipelines: a pipeline with anything other "
            "than one or two inputs"
        )

    if pipeline.output.format not in _RASTER_OUTPUT_FORMATS:
        raise PipelineError(
            "not yet supported in analysis pipelines: a paired (joined) pipeline's "
            f"output format {pipeline.output.format!r} -- only tif/png/jpg (pixel-local "
            "raster output) is implemented for this shape; a tabular paired output "
            "would need materializing the full joined frame, which this compiler "
            "avoids for memory reasons"
        )

    steps = pipeline.steps
    if len(steps) < 2 or not isinstance(steps[0], ProjectStep) or not isinstance(steps[1], JoinStep):
        raise PipelineError(
            "not yet supported in analysis pipelines: a two-input pipeline must open "
            "with `project` (broadcasting the coarser input onto the finer grid) "
            f"immediately followed by `join` -- got {[s.op for s in steps[:2]]}"
        )

    project_step, join_step = steps[0], steps[1]
    if len(join_step.inputs) != 2:
        raise PipelineError(
            "not yet supported in analysis pipelines: a join step with other than two "
            "inputs"
        )

    coarse_name = project_step.input
    if coarse_name is None or coarse_name not in pipeline.inputs:
        raise PipelineError("step 1 (project) must explicitly name one of the pipeline's two inputs")
    coarse_selection = pipeline.inputs[coarse_name]

    projected_name = project_step.as_ or coarse_name
    if projected_name not in join_step.inputs:
        raise PipelineError(
            f"step 2 (join) inputs {join_step.inputs} must include step 1's own output "
            f"name ({projected_name!r})"
        )
    fine_name = next(n for n in join_step.inputs if n != projected_name)
    if fine_name not in pipeline.inputs:
        raise PipelineError(
            f"step 2 (join) input {fine_name!r} must be the pipeline's other (non-"
            "projected) input"
        )
    fine_selection = pipeline.inputs[fine_name]

    _check_input_dimensions(coarse_name, coarse_selection)
    _check_input_dimensions(fine_name, fine_selection)

    if GRIDS[fine_selection.grid].grid_type != "raster":
        raise PipelineError(
            "not yet supported in analysis pipelines: this pipeline shape requires the "
            f"joined (fine) input {fine_name!r} to be on a raster grid, got "
            f"{fine_selection.grid!r}"
        )
    if project_step.to != fine_selection.grid:
        raise PipelineError(
            f"step 1 (project) must target the joined input's own grid "
            f"({fine_selection.grid!r}), got {project_step.to!r}"
        )
    direction = _validate_project_cardinality(1, coarse_selection.grid, project_step.to, project_step.aggregate)
    if direction != "one_to_many":
        raise PipelineError(
            "not yet supported in analysis pipelines: step 1 (project) must broadcast "
            f"the coarser input {coarse_name!r} onto the finer grid -- "
            f"{coarse_selection.grid!r} -> {project_step.to!r} was not one-to-many"
        )

    _validate_join_alignment(2, [fine_name, coarse_name], [fine_selection, coarse_selection])

    where_parts = []
    group_step = None
    for i, step in enumerate(steps[2:], start=3):
        if isinstance(step, FilterStep):
            if group_step is not None:
                raise PipelineError(
                    f"not yet supported in analysis pipelines: step {i} (filter) comes "
                    "after a group step"
                )
            where_parts.append(step.where)
        elif isinstance(step, GroupStep):
            if group_step is not None:
                raise PipelineError(
                    "not yet supported in analysis pipelines: more than one `group` "
                    "step"
                )
            if step.by:
                raise PipelineError(
                    "not yet supported in analysis pipelines: this paired pipeline "
                    f"shape's `group` step has by={step.by!r} -- only by=[] (a "
                    "per-pixel temporal collapse) is supported; grouping by a real "
                    "dimension would produce a zonal/vector result, not a pixel-local "
                    "raster"
                )
            group_step = step
        else:  # pragma: no cover -- exhaustive over the Step union
            raise PipelineError(
                f"not yet supported in analysis pipelines: step {i} ({step.op}) cannot "
                "follow a paired join in this shape (only `filter`/`group` may)"
            )

    # INVARIANT relied on below, not a runtime check: `group_step` is always set here.
    # `_RASTER_OUTPUT_FORMATS` (this function's own format gate, above) is exactly
    # `_SINGLE_VALUE_COLUMN_FORMATS`, so any two-input, project+join pipeline reaching
    # this point with `group_step is None` was already rejected earlier -- before
    # `_find_paired_notebook_shape` is even consulted -- by `compile_pipeline`'s
    # unconditional `_validate_output_columns(pipeline)` call: `_track_output_columns`
    # reports 2 columns for a join nothing narrows afterward, which fails the "exactly
    # one value column" rule for a `_SINGLE_VALUE_COLUMN_FORMATS` output with an
    # equally-specific message (names the produced column count and says to add a
    # `group` step). A `PipelineError`+direct-call unit test used to guard this branch
    # too; kept as a test, but converted to exercise the real `compile_pipeline` entry
    # point and assert on the message that actually fires now (see
    # `TestCompilePairedNotebookShape.test_rejects_missing_trailing_group`) rather than
    # calling this function directly to reach dead code. REMOVED rather than reordered:
    # reordering would only change which of two equally-informative messages a caller
    # sees, not improve on either.
    where = " AND ".join(f"({w})" for w in where_parts) if where_parts else None
    if where:
        for conjunct in split_top_level_and(where):
            try:
                parse_value_expr(conjunct, allowed_columns=[fine_name, projected_name])
            except ExpressionError as exc:
                raise PipelineError(f"filter where conjunct {conjunct!r} is invalid: {exc}") from exc

    _validate_paired_group_aggregate(group_step)

    fine_dataset = _dataset_for(fine_name, fine_selection, base_path)
    coarse_dataset = _dataset_for(coarse_name, coarse_selection, base_path)

    _check_join_row_ceiling(2, coarse_name, coarse_dataset.lf)

    plan = PairedPlan(
        fine_dataset=fine_dataset,
        coarse_dataset=coarse_dataset,
        fine_name=fine_name,
        coarse_alias=projected_name,
        where=where,
        group_step=group_step,
        color_map=pipeline.output.color_map,
    )
    plan._prepare()
    return plan


# `group_step.aggregate` values `_accumulate_paired_window` (the streaming per-band
# accumulator) covers. Four of these (`min`/`max`/`sum`/`mean`) are exactly
# `HORIZONTAL_RASTER_AGGREGATES`'s keys -- the aggregates `horizontal_aggregate_expr`
# (what the wide-frame path ultimately calls, via `PairedPlan._collapse_expr`) knows how
# to reduce across bands. `count` is the fifth and is handled entirely differently: it
# is deliberately NOT a `HORIZONTAL_RASTER_AGGREGATES` key (see that dict's own docstring
# and `types.AggregateFunction.count`'s comment -- a flat-API raster `group_aggregate` has
# no per-pixel meaning for "how many bands contributed"), but a PAIRED pipeline's `where`
# clause gives that same tally a real, user-authored meaning ("how many calendar steps
# satisfied the condition"), so `PairedPlan` implements it LOCALLY (`_band_valid_expr`/
# `_band_valid_mask`, and the `count` branches of `_collapse_expr`/
# `_accumulate_paired_window` below) rather than teaching the shared vocabulary a second,
# incompatible meaning for the same name. Do not "unify" the two by adding `count` to
# `HORIZONTAL_RASTER_AGGREGATES` -- they answer different questions for different callers.
# `count_by_value` is excluded even earlier, by `fold.DECOMPOSABLE_AGGREGATES`, and never
# reaches this shape at all through the sanctioned `parse_pipeline` entry point (see
# `_validate_paired_group_aggregate`'s own docstring for the defense-in-depth check that
# used to be missing here -- `group_step.aggregate == "count"` used to reach the wide
# path today only because nothing rejected it for this shape, and `horizontal_aggregate_
# expr` raised a bare `KeyError` for it; that gap is what this whole change closes).
#
# Deliberately a SEPARATE frozenset from `_STREAMING_AGGREGATES` just below, even though
# the two happen to hold identical values today -- they answer different questions.
# `_SUPPORTED_PAIRED_AGGREGATES` is the durable "does `PairedPlan` implement this
# aggregate AT ALL" invariant `_validate_paired_group_aggregate` enforces; `_STREAMING_
# AGGREGATES` is a dispatch knob a caller (or a test) can legitimately flip per-run to
# force everything onto `_collapse_paired_window_wide` (see that set's own docstring and
# `TestPairedPlanCountConsistency.test_accumulator_matches_wide_fallback`). Validating
# against `_STREAMING_AGGREGATES` directly would make that monkeypatch ALSO disable
# support for every aggregate for the duration of the patch, rather than merely routing
# it onto the wide path -- exactly the trap that test hit during development.
_SUPPORTED_PAIRED_AGGREGATES = frozenset({"min", "max", "sum", "mean", "count"})


def _validate_paired_group_aggregate(group_step):
    """Defense-in-depth: reject any `group_step.aggregate` outside
    `_SUPPORTED_PAIRED_AGGREGATES` for THIS shape, before `PairedPlan` ever dispatches on
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
# `TestPairedPlanCountConsistency.test_accumulator_matches_wide_fallback`, and `/tmp/
# pipetest/scale_probe.py`'s identical trick for scale measurement) can monkeypatch THIS
# set alone to force every aggregate onto the wide path without also disabling
# `_validate_paired_group_aggregate`'s support check for the duration of the patch.
_STREAMING_AGGREGATES = _SUPPORTED_PAIRED_AGGREGATES


def _count_zone_gate(plan, flat_values, pairing_valid):
    """`count`'s zero-vs-nodata split, applied identically after EITHER collapse path
    (`_accumulate_paired_window`/`_collapse_paired_window_wide`/`_paired_notebook_tile`)
    computes a flat per-pixel band tally -- a no-op for every other aggregate.

    `pairing_valid` (this pixel's own membership in the pairing crosswalk -- the only
    "is this pixel even in this plan's output domain at all" signal a `PairedPlan` has;
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
    bypass for the flat API's vertical/zonal count, and `PairedPlan._band_valid_expr`).
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
    see `TestPairedPlanCountConsistency` for why that matters: `count`'s wide form here
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


def _write_paired_notebook_raster(plan: PairedPlan, out_path, **profile_kwargs):
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


async def _paired_notebook_tile(plan: PairedPlan, x, y, z, format, tile_size=256):
    """Mirrors `Dataset.get_tile`'s `WarpedVRT`-based shape, over every fine band plus
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


def compile_pipeline(pipeline: Pipeline, base_path=None):
    """Compile an already-validated `Pipeline` into a runnable plan: either a `Dataset`
    instance (single-input chains, shape 1) or a `PairedPlan` (the paired/notebook
    shape, shape 2). Raises `PipelineError` for anything else, naming the reason -- see
    the module docstring for exactly what compiles today.
    """
    _validate_output_columns(pipeline)

    single = _compile_single_input_chain(pipeline, base_path)
    if single is not None:
        return single

    if _find_paired_notebook_shape(pipeline):
        return _compile_paired_notebook_shape(pipeline, base_path)

    raise PipelineError(
        "not yet supported in analysis pipelines: a pipeline with more than one input "
        "that is not the paired (project + join) shape -- see the module docstring for "
        "the two shapes that compile today"
    )
