"""Compiles an already-validated `analysis.pipeline.Pipeline` (see that module for the
request representation, its two encodings, and the plan-time validation that needs no
catalog/crosswalk knowledge) into a runnable plan -- plus the compile-time checks that DO
need that knowledge, which `analysis.pipeline`'s own docstring explicitly defers to here:
dimension name/value validity, `project`/`group` cardinality direction, `join` alignment,
the join row ceiling, `filter.where`'s column namespace, and the general multi-column
output-count check.

Two shapes compile today, tried in order by `compile_pipeline`:

* **Shape 1: single-input chains** (`_compile_single_input_chain`) -- a pipeline with
  exactly one input and no `join` step anywhere, lowered onto the existing
  `BaseAnalysis`/`Options` machinery (`analysis.factory.create_analysis` builds the
  `VectorGridAnalysis`/`RasterGridAnalysis` instance).
* **Shape 2: the paired (spatially-joined) shape** (`_compile_paired_notebook_shape`) -- a
  coarse input `project`-broadcast onto a finer raster grid, `join`-ed against the finer
  input, and collapsed by a trailing `group` -- compiled into a `PairedAnalysis`
  (`analysis.paired`, the execution half of the former `api/pipeline_exec.py`; this module
  is the compile half).

Anything else raises `PipelineError` naming the reason -- see `compile_pipeline`.
"""


import polars as pl
from pydantic import ValidationError

from cpra.mp.data import config
from cpra.mp.data.analysis.expressions import (
    ExpressionError,
    parse_value_expr,
    rewrite_named_columns,
    split_top_level_and,
)
from cpra.mp.data.analysis.factory import create_analysis
from cpra.mp.data.analysis.paired import (
    PairedAnalysis,
    _validate_paired_group_aggregate,
)
from cpra.mp.data.analysis.parameters import (
    DatasetDimensions,
    FilterDimensions,
    Options,
)
from cpra.mp.data.analysis.pipeline import (
    _SINGLE_VALUE_COLUMN_FORMATS,
    FilterStep,
    GroupStep,
    JoinStep,
    Pipeline,
    PipelineError,
    ProjectStep,
)
from cpra.mp.data.analysis.types import DataFormat, TileFormat
from cpra.mp.data.grids import GRIDS
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
    left to the constructed `BaseAnalysis` subclass's own construction, which already
    raises a 422 for it via `get_lazy_frame`.
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
    return create_analysis(dataset_dimensions, filter_dimensions, options, base_path=base_path)


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
# Shape 1: single-input chains (lowered onto `BaseAnalysis`/`Options`)
# ------------------------------------------------------------------------------------------


def _compile_single_input_chain(pipeline: Pipeline, base_path):
    """Compile a one-input, no-`join` pipeline onto the existing `BaseAnalysis`/`Options`
    machinery, returning the constructed `BaseAnalysis` instance directly. Returns `None`
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
        # shape (`BaseAnalysis._get_group_by_columns`'s `group_by is None` branch,
        # returning `[]`), not "group_by the empty string". Passing `group_by=""`
        # through instead would hit `unpack_values("", ...)`, which 400s ("Invalid
        # values for group_by: ")
        # since "" is never a member of `DIMENSION_CHOICES` -- so the empty-list
        # case must omit `group_by` entirely rather than joining it into "".
        if group_step.by:
            option_kwargs["group_by"] = ",".join(group_step.by)
        option_kwargs["group_aggregate"] = group_step.aggregate

    return _dataset_for(input_name, selection, base_path, **option_kwargs)


# ------------------------------------------------------------------------------------------
# Shape 2: the paired (spatially-joined) shape -- PairedAnalysis
# ------------------------------------------------------------------------------------------


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

    plan = PairedAnalysis(
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
# (what the wide-frame path ultimately calls, via `PairedAnalysis._collapse_expr`) knows how
# to reduce across bands. `count` is the fifth and is handled entirely differently: it
# is deliberately NOT a `HORIZONTAL_RASTER_AGGREGATES` key (see that dict's own docstring
# and `types.AggregateFunction.count`'s comment -- a flat-API raster `group_aggregate` has
# no per-pixel meaning for "how many bands contributed"), but a PAIRED pipeline's `where`
# clause gives that same tally a real, user-authored meaning ("how many calendar steps
# satisfied the condition"), so `PairedAnalysis` implements it LOCALLY (`_band_valid_expr`/
# `_band_valid_mask`, and the `count` branches of `_collapse_expr`/
# `_accumulate_paired_window` below) rather than teaching the shared vocabulary a second,
# incompatible meaning for the same name. Do not "unify" the two by adding `count` to
# `HORIZONTAL_RASTER_AGGREGATES` -- they answer different questions for different callers.
# `count_by_value` is excluded even earlier, by `fold.DECOMPOSABLE_AGGREGATES`, and never
# reaches this shape at all through the sanctioned `parse_pipeline` entry point (see
def compile_pipeline(pipeline: Pipeline, base_path=None):
    """Compile an already-validated `Pipeline` into a runnable plan: either a `BaseAnalysis`
    instance (single-input chains, shape 1) or a `PairedAnalysis` (the paired/notebook
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
