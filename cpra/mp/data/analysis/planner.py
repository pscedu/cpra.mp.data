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
  `VectorGridAnalysis`/`RasterGridAnalysis` instance). `calculate` steps compile onto
  `Options.calculate`, the same generic entries the flat `calculate.N.*` parameters
  populate; `calculate` in the paired shape is rejected (see
  `_compile_paired_notebook_shape`).
* **Shape 2: the paired (spatially-joined) shape** (`_compile_paired_notebook_shape`) -- a
  coarse input `project`-broadcast onto a finer raster grid, `join`-ed against the finer
  input, and collapsed by a trailing `group` -- compiled into a `PairedAnalysis`
  (`analysis.paired`, the execution half of the former `api/pipeline_exec.py`; this module
  is the compile half).

Anything else raises `PipelineError` naming the reason -- see `compile_pipeline`.
"""


import re

import polars as pl
from pydantic import ValidationError

from cpra.mp.data import config
from cpra.mp.data.analysis.expressions import (
    ExpressionError,
    _check_no_dimension_subquery_or_aggregate,
    classify_where,
    parse_calculate_expression,
    parse_value_expr,
    rewrite_named_columns,
    split_top_level_and,
)
from cpra.mp.data.analysis.factory import create_analysis
from cpra.mp.data.analysis.ops import compile_ops
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
    CalculateStep,
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


def _validate_filter_where(step_no, where, extra_allowed_columns, op="filter"):
    """Validate a `filter`/`calculate` step's `where`, split into dimension and value
    conjuncts exactly the way `classify_where` splits the flat `where=` parameter (see
    that function's docstring for the full split/classify/reject contract) -- a
    dimension conjunct gets only the subquery/aggregate safety check
    (`_check_no_dimension_subquery_or_aggregate`), otherwise unrestricted (the same
    treatment `Dataset._apply_filter_params` gives it, via an unrestricted `lf.sql()`
    call); a value conjunct (referencing the running-value placeholder -- already
    rewritten to the literal column `value` by the caller, see `rewrite_named_columns`)
    gets the full `parse_value_expr` whitelist against `extra_allowed_columns` (always
    `{"value"}` from both call sites below), exactly as before.

    This used to run EVERY top-level
    conjunct through the value whitelist regardless of kind, which is narrower than what
    the flat `where=` parameter accepts for a dimension predicate -- e.g. `not (a = x and
    b = y)` (`not` is not in `ALLOWED_FUNCTIONS`), which is exactly the form the portal
    emits for a three-image `concatenate` and which `classify_where`'s own two-path split
    was built to allow for a dimension-only conjunct. Aligning this function with that
    split closes the gap: a dimension conjunct is no longer namespace-restricted to a
    known dimension name at pipeline-compile time (matching the flat surface, which has
    no such check either -- an unknown dimension name fails at execution, via `lf.sql()`,
    not here), only checked for the same aggregate/subquery shapes that would defeat
    streaming pushdown.
    """
    try:
        dimension_conjuncts, value_conjuncts = classify_where(where)
    except ExpressionError as exc:
        raise PipelineError(f"step {step_no} ({op}) where={where!r} is invalid: {exc}") from exc

    for conjunct in dimension_conjuncts:
        try:
            _check_no_dimension_subquery_or_aggregate(conjunct)
        except ExpressionError as exc:
            raise PipelineError(
                f"step {step_no} ({op}) where conjunct {conjunct!r} is invalid: {exc}"
            ) from exc

    for conjunct in value_conjuncts:
        try:
            parse_value_expr(conjunct, allowed_columns=extra_allowed_columns)
        except ExpressionError as exc:
            raise PipelineError(
                f"step {step_no} ({op}) where conjunct {conjunct!r} is invalid: {exc}"
            ) from exc


def _check_join_row_ceiling(step_no, name, lf):
    n_rows = lf.select(pl.len()).collect(engine="streaming").item()
    if n_rows > config.ANALYSIS_JOIN_MAX_ROWS:
        raise PipelineError(
            f"step {step_no} (join) input {name!r} has {n_rows} rows, over the "
            f"{config.ANALYSIS_JOIN_MAX_ROWS}-row ceiling for a pipeline's vector/zone "
            "side (see config.ANALYSIS_JOIN_MAX_ROWS) -- narrow this input's dimension "
            "filters"
        )


def _dataset_for(name, selection, base_path, **option_kwargs):
    """Build the `BaseAnalysis` for one `InputSelection`, forwarding its reserved
    `grid_cell` field (see `InputSelection`'s docstring) onto `Options.grid_cell`
    unchanged -- the same option a flat `grid_cell=` query parameter already sets. Only
    safe to call with a `selection.grid_cell` set for a shape whose dataset actually runs
    `apply_params()` (single-input chains do; the paired shape's `fine_dataset`/
    `coarse_dataset` do NOT, so `_compile_paired_notebook_shape` rejects a `grid_cell`
    selection outright, before ever reaching here, rather than silently no-op-ing it).
    """
    if selection.grid_cell is not None:
        option_kwargs["grid_cell"] = selection.grid_cell
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


# Accepted literal `grid_cell` filter shapes after a many-to-one `project` -- see
# `_extract_post_project_grid_cell_filter`. Case-insensitive on `IN`, matching this
# package's other case-insensitive SQL keyword handling (`split_top_level_and`'s "and").
_GRID_CELL_EQ_RE = re.compile(r"^\s*grid_cell\s*=\s*(-?\d+)\s*$", re.IGNORECASE)
_GRID_CELL_IN_RE = re.compile(r"^\s*grid_cell\s+in\s*\(\s*(-?\d+(?:\s*,\s*-?\d+)*)\s*\)\s*$", re.IGNORECASE)


def _extract_post_project_grid_cell_filter(step_no, where):
    """Recognize a `filter` step, occurring after a many-to-one (summarizing) `project`
    in a single-input chain, whose ENTIRE predicate is a literal `grid_cell` selection --
    and return it as the comma-separated ID string `Options.summarize_grid_cell` expects.
    Returns `None` when `where` does not reference `grid_cell` at all (an ordinary
    post-project filter, untouched by this rule, still folded into `Options.where` by the
    caller exactly as before).

    Why this exists: the executor's Filter stage (which `Options.where` feeds) runs
    against the SOURCE (pre-summarize) rows, before the Transform/summarize stage that a
    many-to-one `project` compiles onto ever narrows anything to the summarized grid --
    so a `filter` naming `grid_cell` after such a `project` would, if simply folded into
    `Options.where` like any other post-project filter, silently test the WRONG grid's
    cell IDs (the source dataset's, not the summarized zones'). `summarize_grid_cell` is
    the correct target (`BaseAnalysis._validate_summarize_params` applies it against the
    summarized grid), but it only accepts a literal comma-separated ID list, never a
    general SQL predicate -- so only the two literal forms below are recognized; ANY
    other way a filter's `where` mentions `grid_cell` after such a `project` (mixed with
    another column or another AND-ed conjunct, or a non-literal comparison like `grid_cell
    > 100`) is rejected outright, so the mis-execution this function exists to avoid can
    never be reached silently -- see `_compile_single_input_chain`'s `FilterStep` branch,
    the only caller.
    """
    try:
        conjuncts = split_top_level_and(where)
    except ExpressionError as exc:
        raise PipelineError(f"step {step_no} (filter) where={where!r} is invalid: {exc}") from exc

    references_grid_cell = False
    for conjunct in conjuncts:
        try:
            names = set(pl.sql_expr(conjunct).meta.root_names())
        except Exception as exc:  # pl.sql_expr's own exception types are not part of its stable API
            raise PipelineError(f"step {step_no} (filter) where conjunct {conjunct!r} is invalid: {exc}") from exc
        if "grid_cell" in names:
            references_grid_cell = True
            if len(conjuncts) != 1 or names != {"grid_cell"}:
                raise PipelineError(
                    f"step {step_no} (filter) where={where!r} mixes `grid_cell` with "
                    "another column or another AND-ed conjunct, after a many-to-one "
                    "`project` -- a post-project `grid_cell` filter must be that step's "
                    "ENTIRE predicate (e.g. `grid_cell = 5` or `grid_cell IN (5, 6)`), "
                    "since it targets the SUMMARIZED grid's cells, not a per-row "
                    "condition on the source dataset"
                )

    if not references_grid_cell:
        return None

    match = _GRID_CELL_EQ_RE.match(conjuncts[0]) or _GRID_CELL_IN_RE.match(conjuncts[0])
    if not match:
        raise PipelineError(
            f"step {step_no} (filter) where={where!r} names `grid_cell` after a "
            "many-to-one `project` but isn't a literal ID selection -- only `grid_cell = "
            "<id>` or `grid_cell IN (<id>, ...)` can be lowered onto the summarized "
            "grid's own cell filter; a comparison like `grid_cell > 100` has no such "
            "translation"
        )
    return ",".join(part.strip() for part in match.group(1).split(","))


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

    # Decided once, here, and reused throughout this function -- both to scope the
    # ordering relaxations below and, at the bottom of this function, to decide whether
    # to attach `_pipeline_ops` at all (see the ops-interpreter fast-path comment there,
    # which now just reads this variable instead of recomputing the same check). A
    # vector input's chain runs through `VectorGridAnalysis.apply_steps` (the ops
    # interpreter, over `ops.compile_ops(pipeline)`'s raw, declared-order IR) rather than
    # `BaseAnalysis.apply_params`'s fixed five-stage flow -- and several rejections below
    # exist ONLY because that old fixed-stage flow ran one bucket per stage (one
    # `Options.group_by`, one combined `Options.where`, ...) and so could not represent
    # shapes like "two independent group steps" or "a filter after a group step" at all,
    # regardless of whether the shape is meaningful. The interpreter has no buckets -- it
    # walks `compile_ops`'s ops list in declared order and applies each one as an
    # ordinary re-entrant transform -- so those specific rejections are now artifacts of
    # an engine a vector pipeline no longer runs through, not real limitations.
    #
    # A raster input ALSO routes through an interpreter (`RasterGridAnalysis.apply_steps`)
    # for every shape except a `grid_cell`-bearing selection (see the fast-path comment
    # below for why that one is excluded) -- but that interpreter does NOT yet make these
    # rejections safe to lift. It still defers the group collapse by recording absolute
    # row indices into `self._band_groups`, and band columns are named `b0..bN` BY
    # POSITION in the band-metadata frame, recomputed at read time. Any op that changes
    # that frame's membership or order after a group has been recorded silently
    # invalidates the stored indices. So the rejections below stay in force for raster
    # and are lifted only for vector.
    is_vector_input = GRIDS[selection.grid].grid_type == "vector"

    steps = pipeline.steps

    # Find the leading `project` step, consuming any leading run of `filter`/`calculate`
    # steps ahead of it. THIS REPLACES an older INVARIANT comment that used to sit here,
    # claiming `steps[0]` was always a `ProjectStep`, justified by `pipeline.py`'s
    # `validate_pipeline` chaining rule rejecting a step 1 that doesn't name its input
    # explicitly ("step 1 (...) has no input to chain from"). That invariant is FALSE as
    # of `FilterStep` and `CalculateStep`, each gaining its own `input` field: a
    # value-testing `filter` or a `calculate`
    # declared after a many-to-one `project` is rejected below with "move it before the
    # project step" advice, and before these fields existed that advice was impossible to
    # follow, since `filter`/`calculate`/`group` had no way to name an input and so could
    # never be a pipeline's first step -- only `project`/`join` could chain the required
    # way. Now step 1 can be a `filter` or a `calculate` naming its `input` explicitly, so
    # this function must find the `project` step itself rather than assume it sits at
    # index 0.
    #
    # `calculate` joins `filter` in the leading run for exactly the same reason `filter`
    # is let in -- `Options.calculate` lowers onto
    # `BaseAnalysis.apply_params`'s Calculate stage (stage 2), which runs unconditionally
    # before the Transform stage (stage 4) a `project`'s aggregation lowers onto, for
    # BOTH the raster and vector-interpreter engines (see `CalculateStep`'s own
    # docstring for the full argument). A leading `calculate` therefore executes in
    # exactly its declared position regardless of grid type, with no ordering question
    # the way a POST-project `calculate` has (still rejected a few lines below, after a
    # many-to-one `project` -- see the SUMMARIZING-PROJECT ORDER GUARD comment). `group` was deliberately
    # NOT extended the same way -- see `GroupStep`'s own docstring for why its scalar
    # `Options` bucket makes that a separate, larger change, left as future work.
    #
    # The chaining rule still guarantees SOME step 1 names its input explicitly, but it
    # does not guarantee a `project` exists anywhere in the pipeline at all -- a pipeline
    # consisting of nothing but `filter`/`calculate` steps chained off one explicit
    # `input` is perfectly valid per `validate_pipeline` (which is agnostic to which op
    # eventually binds the grid). So the bounds/type check just below is a REAL runtime
    # check, not a restatement of something `parse_pipeline` already guarantees, and it
    # is exercised by `TestPostProjectOrderingGuard`'s dedicated tests, not dead
    # code. Only `filter`/`calculate` steps are consumed into the leading run -- `group`
    # still has no `input` field (see `GroupStep`'s docstring) and so can never lead a
    # pipeline; a leading run that hits one before ever finding a `project` reports the
    # same "must contain a project step" error as running out of steps entirely.
    # A single list, in declared order, mixing both step types -- NOT two separate lists
    # keyed by type -- because relative order between a leading `filter` and a leading
    # `calculate` matters: a `filter` on `value` declared after a leading `calculate`
    # must still get the same `after_calculate` commutativity treatment a POST-project
    # filter-after-calculate gets below (see the unpacking loop a few lines below
    # `filters = []` for where that flag gets computed and reused).
    leading_steps: list[tuple[int, FilterStep | CalculateStep]] = []
    project_idx = 0
    while project_idx < len(steps) and isinstance(steps[project_idx], (FilterStep, CalculateStep)):
        leading_steps.append((project_idx + 1, steps[project_idx]))
        project_idx += 1

    if project_idx >= len(steps) or not isinstance(steps[project_idx], ProjectStep):
        found = steps[project_idx].op if project_idx < len(steps) else "no further steps"
        raise PipelineError(
            "not yet supported in analysis pipelines: a single-input pipeline must "
            "contain a `project` step, either as step 1 or after a leading run of "
            f"`filter`/`calculate` steps -- found {found} at step {project_idx + 1} instead"
        )

    first = steps[project_idx]
    project_step_no = project_idx + 1
    value_names = {input_name}
    if first.as_ is not None:
        value_names.add(first.as_)

    option_kwargs = {}
    if first.to == selection.grid:
        if first.aggregate is not None:
            raise PipelineError(
                f"step {project_step_no} (project) targets its own input's grid "
                f"({selection.grid!r}) -- no aggregate is possible without an actual "
                "grid change; omit `aggregate`"
            )
    else:
        direction = _validate_project_cardinality(
            project_step_no, selection.grid, first.to, first.aggregate
        )
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

    # SUMMARIZING-PROJECT ORDER GUARD: whether the pipeline's leading `project` step
    # (`first`, found above -- possibly after a run of leading `filter` steps, not
    # necessarily `steps[0]` itself) is many-to-one (summarizing) is exactly whether it
    # set `summarize_by` above -- and it is exactly this case where an ordering bug is
    # live. `summarize_by`/`summarize_aggregate` only take effect in `BaseAnalysis
    # .apply_params`'s TRANSFORM stage (stage 4), which runs after Filter (stage 1),
    # Calculate (stage 2), and Group (stage 3) UNCONDITIONALLY -- regardless of where in
    # the DECLARED pipeline a filter/calculate/group step sits relative to `project`. A
    # step declared after a many-to-one `project` therefore actually runs against the
    # PRE-project source rows, not the declared post-project result. That is verified
    # wrong today (not theoretical): `project(sum) -> filter(value > 2)` over ecoregions
    # 1-4 valued 1.0-4.0, crosswalked (1,2)->100/(3,4)->200, gives {100: 3.0, 200: 7.0}
    # declared vs {100: 0.0, 200: 7.0} actually executed (the filter silently removed
    # compartment 100's source cells before the crosswalk join summed an empty set -- no
    # error); `project(sum) -> calculate(value * value)` gives {100: 9.0, 200: 49.0}
    # declared vs {100: 5.0, 200: 25.0} actually executed (squared-then-summed instead of
    # summed-then-squared).
    #
    # An IDENTITY project (`to` equal to the input's own grid, the `if` branch above)
    # sets no `summarize_by` at all -- stage 4 is then a no-op, so there is nothing for
    # a later step to run "before", and no order to invert. That is why every registered
    # analysis (`difference`/`concatenate`/`exit_criterion`, all of which project to
    # their own dataset's grid) is unaffected by this guard: it only ever fires for a
    # `project` that actually changes grid, which none of them do.
    #
    # A `filter` testing only DIMENSION columns is the one case that still commutes with
    # a many-to-one `project` and stays accepted below: a projection rewrites
    # `grid_cell` and `value` (it re-keys rows onto a coarser zone and re-aggregates
    # their values), never a dimension column -- the identical argument the pre-existing
    # comment a few lines below already makes for a dimension filter ordered after a
    # `calculate` step. A `filter` on `grid_cell` alone is likewise still accepted, via
    # the pre-existing `_extract_post_project_grid_cell_filter` special-case immediately
    # below, unchanged by this guard.
    #
    # A `filter` in `leading_filters` (found above, BEFORE `first`) never needs any of
    # this reasoning at all: it is declared and executes in the same place -- the Filter
    # stage (stage 1), strictly before the project's Transform-stage (stage 4)
    # aggregation, by construction, regardless of whether that aggregation is
    # many-to-one. That is precisely why leading filters are the remediation this guard's
    # own error messages point authors at, below.
    is_summarizing_project = "summarize_by" in option_kwargs

    # EXPLICIT DECISION: the SUMMARIZING-PROJECT ORDER GUARD above -- and its three
    # enforcement sites below (the calculate-, group-, and
    # value-filter-after-a-summarizing-`project` rejections) -- is NOT relaxed for
    # either grid type, unlike the rejections deleted elsewhere in this function. This
    # guard is not a bucket-collision artifact of the old fixed-stage engine: it
    # protects against a genuine ORDERING question that is just as real for the
    # interpreter as for `BaseAnalysis.apply_params` -- whether a value-testing step
    # declared after a many-to-one `project` runs against the declared post-project
    # (aggregated) rows or the pre-project source rows. The interpreter answers this
    # correctly BY CONSTRUCTION (it applies `compile_ops`'s `ProjectOp` in declared
    # position, same as any other op, so a step declared after it genuinely does run
    # after it) -- so relaxing this guard would very likely be safe for both grid types.
    # But "very likely safe" is not the standard the other rejections in this function
    # were held to (each proven correct with a hand-verified test), and doing the same
    # here would mean re-deriving and hand-checking the interpreter's post-project
    # calculate/group/value-filter behavior against a crosswalk fixture -- real,
    # separate verification work, not a mechanical extension of what this change covers.
    # Left in force for both grid types; relaxing it is future work, not a gap
    # introduced by this change.
    #
    # Each entry is `(step_no, where, after_calculate, after_summarizing_project)`.
    # `after_summarizing_project` is always `False` here -- the leading run, by
    # construction, entirely precedes the `project` step found above, so the
    # after-`project` commutativity question the SUMMARIZING-PROJECT ORDER GUARD below asks never applies
    # to a leading filter. `after_calculate`, though, is NOT always `False`: a leading
    # `filter` declared AFTER a leading `calculate` (e.g.
    # `calculate, filter, project`) needs the identical after-`calculate` commutativity
    # check a POST-project filter-after-calculate gets in the main loop below -- both
    # ultimately combine into the same flat `Options.calculate` (a list, applied as ONE
    # Calculate stage, stage 2) and `Options.where` (a single combined string, applied as
    # ONE Filter stage, stage 1) regardless of whether their source step was leading or
    # post-project, so the OLD fixed-stage engine runs every filter (stage 1) before
    # every calculate (stage 2) either way -- a value-testing leading filter declared
    # after a leading calculate would, for a RASTER input, silently test the
    # PRE-calculate value despite being declared after the calculate that rewrites it,
    # exactly the same failure mode `after_calculate`'s existing check (below) already
    # guards against for the post-project case. Tracked here as `leading_after_calculate`,
    # flipped `True` the moment a leading `calculate` is seen, mirroring `bool
    # (calculate_steps)` in the main post-project loop below.
    filters = []
    calculate_steps = []
    leading_after_calculate = False
    for step_no, leading_step in leading_steps:
        if isinstance(leading_step, FilterStep):
            filters.append((step_no, leading_step.where, leading_after_calculate, False))
            if leading_step.as_ is not None:
                value_names.add(leading_step.as_)
        else:
            # A leading `calculate` -- see `CalculateStep`'s docstring for why this is
            # safe unconditionally, for both grid types.
            calculate_steps.append((step_no, leading_step))
            if leading_step.as_ is not None:
                value_names.add(leading_step.as_)
            leading_after_calculate = True

    group_step = None
    summarize_grid_cell = None
    for i, step in enumerate(steps[project_idx + 1 :], start=project_step_no + 1):
        if isinstance(step, FilterStep):
            # A filter declared after a group step used to be rejected outright: the old
            # fixed-stage engine has exactly one Filter stage (stage 1), running strictly
            # before its Group stage (3), so a filter ordered after a `group` step could
            # only ever silently apply to the PRE-group rows, never the declared post-group
            # result -- "keep only groups whose collapsed value passes some test" is a
            # legitimate, common query, not a meaningless shape, so that was a real engine
            # limitation. `vector.VectorGridAnalysis.apply_steps` walks `ops.compile_ops
            # (pipeline)` in DECLARED order and applies each op as an ordinary re-entrant
            # transform: a `FilterOp` declared after a `GroupOp` runs against whatever the
            # group already produced, because there is no separate fixed "Filter stage"
            # for it to have instead landed in. So this is lifted for vector.
            #
            # It is NOT lifted for raster. The raster interpreter still defers the
            # collapse as row indices in `self._band_groups`, and a dimension filter
            # after a group drops columns from the band-metadata frame -- which renames
            # the surviving bands, since `b0..bN` are positional. `band_group_row_idxs`
            # then indexes a band that no longer exists at that position (a recorded
            # `b2` against a frame narrowed to two columns). Lift this once the raster
            # side stops carrying positions across a metadata change.
            if not is_vector_input and group_step is not None:
                raise PipelineError(
                    f"not yet supported in analysis pipelines: step {i} (filter) comes "
                    "after a group step in a raster-input pipeline -- the raster engine "
                    "records the group's band positions before any bands are read, and a "
                    "later filter renames the surviving bands, so the recorded positions "
                    "would no longer refer to the bands they were captured for. Move this "
                    "filter step before the group step."
                )
            # Only a many-to-one (summarizing) `project` needs this special-casing --
            # see `_extract_post_project_grid_cell_filter`'s docstring. An identity
            # project (no `summarize_by` in `option_kwargs`) filters the same grid its
            # own rows already carry, so a `grid_cell` filter there is an ordinary
            # filter, folded into `Options.where` below exactly as before.
            grid_cell_ids = None
            if is_summarizing_project:
                grid_cell_ids = _extract_post_project_grid_cell_filter(i, step.where)
            if grid_cell_ids is not None:
                # `Options.summarize_grid_cell` is a single scalar field --
                # the old fixed-stage engine has exactly one slot to narrow the crosswalk
                # join by, so a SECOND post-project `grid_cell` filter had nowhere to go
                # but to silently overwrite (or be rejected ahead of overwriting, which is
                # what this did) the first one's value. This extraction still runs
                # (unchanged, both grid types -- it is what keeps a literal
                # `grid_cell = N` filter from being folded into `Options.where` and tested
                # against the wrong grid) but `Options.summarize_grid_cell` is no longer
                # what actually filters the materialized rows for EITHER grid type now:
                # both `VectorGridAnalysis.apply_steps` and `RasterGridAnalysis
                # .apply_steps` apply each post-`ProjectOp` `FilterOp` -- including a
                # `grid_cell` one -- directly against the already-projected `grid_cell`
                # column, in declared order, same as any other filter (vector: see
                # `TestVectorInterpreterDifferential
                # .test_summarizing_project_then_grid_cell_filter`; raster:
                # `RasterGridAnalysis.apply_steps`'s POST-project branch runs
                # `apply_dimension_filter`/`apply_value_filter` against `self.lf` once it
                # is a real materialized long-shape frame -- IDENTICAL to vector's
                # mechanism at that point, since a summarizing `ProjectOp` has already run
                # `_apply_transform_params_impl()` and replaced `self.lf` with a genuine
                # tabular result; nothing raster-specific is left to diverge on). A
                # second (or third, ...) post-project `grid_cell` filter is therefore just
                # an ordinary additional row-drop, ANDed together with whatever the first
                # one already dropped -- no bucket to collide over, for either grid type.
                # `summarize_grid_cell` below still gets the LAST one seen (an inert "last
                # write wins" simplification -- see this function's module docstring on
                # `Options` being vestigial for interpreter execution); this is safe
                # because it is used ONLY to narrow the initial
                # crosswalk scan for performance (an intersection of grid_cell sets is
                # always a SUBSET of any individual set, including the last one seen, so
                # narrowing by the last one never prematurely excludes a zone that a
                # correct narrowing-by-intersection would have kept -- the real
                # intersection is still enforced afterward by the sequential `FilterOp`
                # applications against the materialized frame).
                summarize_grid_cell = grid_cell_ids
            else:
                filters.append((i, step.where, bool(calculate_steps), is_summarizing_project))
            if step.as_ is not None:
                value_names.add(step.as_)
        elif isinstance(step, CalculateStep):
            # A calculate declared after a group step used to be rejected outright, for
            # the same reason as the filter-after-group case above: the old fixed-stage
            # engine's Calculate stage (2) always ran before its Group stage (3), so a
            # `calculate` ordered after a `group` step could only ever silently rewrite
            # the PRE-group rows, never the declared post-group (collapsed) result.
            # `VectorGridAnalysis.apply_steps` applies each op in declared order, so a
            # `CalculateOp` after a `GroupOp` runs against the already-collapsed rows --
            # lifted for vector.
            #
            # Still rejected for raster: `sources.band_collapse_exprs` hardcodes a
            # calculate-before-collapse pass order, so a calculate declared after a group
            # runs BEFORE it regardless. That is a silent wrong answer, not an error --
            # a declared `mean(x)**2` evaluates as `mean(x**2)`, which for bands [2, 4]
            # returns 10 instead of 9. Lift this once the pass order follows the declared
            # order instead of that fixed one.
            if not is_vector_input and group_step is not None:
                raise PipelineError(
                    f"not yet supported in analysis pipelines: step {i} (calculate) "
                    "comes after a group step in a raster-input pipeline -- the raster "
                    "engine applies every calculate before the group collapse regardless "
                    "of declared order, which would silently compute the aggregate of "
                    "the expression instead of the expression of the aggregate. Move "
                    "this calculate step before the group step."
                )
            if is_summarizing_project:
                # SUMMARIZING-PROJECT ORDER GUARD -- see the long comment above `is_summarizing_project`'s
                # assignment for the full argument and the measured numbers. Unlike a
                # `filter`, a `calculate` step has no dimension-only escape hatch: it
                # exists specifically to rewrite `value`, and `value` is exactly the
                # column a many-to-one `project` re-aggregates -- there is no shape of
                # `calculate` that commutes with that, so this rejection is unconditional
                # (not narrowed to "when it touches value", the way the filter guard
                # below is).
                raise PipelineError(
                    f"not yet supported in analysis pipelines: step {i} (calculate) "
                    "comes after a many-to-one (summarizing) `project` step in a "
                    "single-input pipeline -- `project`'s aggregation lowers onto "
                    "`Options.summarize_by`/`summarize_aggregate`, applied in "
                    "`BaseAnalysis.apply_params`'s Transform stage (stage 4), while "
                    "`calculate` lowers onto `Options.calculate`, applied in the "
                    "Calculate stage (stage 2) -- which runs BEFORE stage 4 "
                    "unconditionally, regardless of how the steps are declared. A "
                    "calculate step ordered after the project would therefore rewrite "
                    "the PRE-project source values, not the declared post-project "
                    "result, and the two do not commute in general (sum-then-square and "
                    "square-then-sum disagree for any non-uniform input) -- VERIFIED: "
                    "project(sum) -> calculate(value * value) over ecoregions valued "
                    "1.0-4.0 crosswalked (1,2)->100/(3,4)->200 gives {100: 9.0, 200: "
                    "49.0} declared vs {100: 5.0, 200: 25.0} actually executed. Move "
                    "this calculate step before the project step."
                )
            calculate_steps.append((i, step))
            if step.as_ is not None:
                value_names.add(step.as_)
        elif isinstance(step, GroupStep):
            # A second `group` step used to be rejected outright: `Options.group_by`/
            # `group_aggregate` are scalar fields, so a second `group` step had nowhere to
            # lower onto but to silently overwrite the first. "Collapse to one row per
            # grid_cell, then collapse further/differently" is a meaningful shape (e.g.
            # group by a dimension, then group again to a scalar), not a nonsensical one --
            # the rejection was purely an artifact of `Options` having one bucket, not of
            # the shape being unsupportable. Vector's `apply_group_by` runs a real
            # `.group_by().agg()` against the first group's OUTPUT rows, so a second
            # `GroupOp` composes against the first one's result -- lifted for vector.
            # `group_step` below is reassigned to the LAST `group` step seen (an inert
            # "last write wins" simplification for `Options.group_by`/`group_aggregate`,
            # which the vector interpreter's execution path does not consult).
            #
            # Still rejected for raster: `_collapse_group` re-derives row positions
            # against `self.lf`, which the first group never touched (it only recorded
            # indices), and then assigns over `self._band_groups`. The second group
            # therefore discards the first rather than composing with it -- structurally,
            # not as an edge case. Lift this once the collapse is applied to the frame
            # instead of recorded beside it.
            if not is_vector_input and group_step is not None:
                raise PipelineError(
                    "not yet supported in analysis pipelines: more than one `group` step "
                    f"in a raster-input pipeline (step {i}) -- the raster engine records "
                    "each group's band positions against the original, uncollapsed band "
                    "set, so a second group would silently replace the first instead of "
                    "collapsing its result."
                )
            if is_summarizing_project:
                # SUMMARIZING-PROJECT ORDER GUARD -- see the long comment above `is_summarizing_project`'s
                # assignment. Same reasoning as the `calculate` guard immediately above:
                # `group` lowers onto `Options.group_by`/`group_aggregate`, applied in
                # the Group stage (stage 3 of `apply_params`), which also runs before
                # the project's own Transform-stage aggregation (stage 4) regardless of
                # declared order -- so a group step ordered after the project would
                # collapse the PRE-project source rows into groups, not the declared
                # post-project zones, and a collapse does not commute with a subsequent
                # crosswalk re-aggregation in general (same class of failure verified
                # for `filter`/`calculate` above).
                raise PipelineError(
                    f"not yet supported in analysis pipelines: step {i} (group) comes "
                    "after a many-to-one (summarizing) `project` step in a single-input "
                    "pipeline -- `project`'s aggregation only takes effect in the "
                    "Transform stage (stage 4 of `BaseAnalysis.apply_params`), while "
                    "`group` runs in the Group stage (stage 3), unconditionally before "
                    "it -- a group declared after the project would collapse the "
                    "pre-project rows instead of the declared post-project zones. Move "
                    "this group step before the project step."
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
                "`filter`/`calculate`/`group` may)"
            )

    rename_map = {name: "value" for name in value_names}
    # Collects ATOMIC top-level conjuncts (not whole per-step `where` strings) across
    # every filter step, in the order encountered -- see the loop below for why the
    # granularity matters. This list is what finally becomes `Options.where`.
    rewritten_conjuncts = []
    for i, where, after_calculate, after_summarizing_project in filters:
        rewritten = rewrite_named_columns(where, rename_map)
        _validate_filter_where(i, rewritten, {"value"})
        # Classified ONCE here and reused for both policy checks below (a filter step
        # ordered after a `calculate`, and one ordered after a many-to-one `project`)
        # and for how this step's `where` gets folded into the combined `Options.where`
        # below -- see the module-level SUMMARIZING-PROJECT ORDER GUARD comment (above
        # `is_summarizing_project`'s assignment) and the pre-existing comment this
        # mirrors for the `after_calculate` case. `after_summarizing_project` is always
        # `False` for a leading filter (collected above, before the post-project loop) --
        # see that loop's own comment for why it never needs either check.
        try:
            dimension_conjuncts, value_conjuncts = classify_where(rewritten)
        except ExpressionError as exc:
            raise PipelineError(f"step {i} (filter) where={where!r} is invalid: {exc}") from exc
        # A value-testing filter declared after a calculate step used to be rejected:
        # `Options.calculate`/`Options.where` are each combinable buckets, but the flat
        # machinery's fixed stage order (Filter stage before Calculate stage) could only
        # honor a filter declared after a calculate if the two commuted, and a value
        # condition doesn't (it would test the pre-calculate value instead of the
        # post-calculate one the declared order promises). A dimension conjunct always
        # commuted and stayed accepted below.
        #
        # Lifted for vector: a vector input's `apply_steps` applies a `FilterOp` against
        # the already-recalculated `value` column, in declared order, with no separate
        # fixed "Filter stage" to instead run ahead of the calculate.
        #
        # Still rejected for raster: the band-wide write path runs every calculate pass
        # first (`for pass_exprs in passes: lf = lf.with_columns(pass_exprs)`) and only
        # then the collapse, a fixed order that a per-cell value mask declared after a
        # calculate cannot get in front of. The mask would test the raw pixel value, not
        # the calculated one the declared order promises. Lift this once those passes run
        # in declared order.
        if value_conjuncts and after_calculate and not is_vector_input:
            raise PipelineError(
                f"not yet supported in analysis pipelines: step {i} (filter) tests "
                f"`value` and comes after a calculate step in a raster-input pipeline "
                f"(where={where!r}) -- the raster engine applies calculates in a fixed "
                "order ahead of any value mask, so this filter would test the "
                "pre-calculate value instead of the post-calculate one. Move this filter "
                "step before the calculate step, or restrict it to dimension conditions."
            )
        if value_conjuncts and after_summarizing_project:
            # SUMMARIZING-PROJECT ORDER GUARD -- see the long comment above `is_summarizing_project`'s
            # assignment for the full argument and the measured numbers. A dimension
            # conjunct in THIS SAME `where` is unaffected and stays in
            # `dimension_conjuncts` below, folded into `Options.where` exactly as
            # before -- only a conjunct that tests `value` is rejected, since only
            # `value` is what the project's crosswalk aggregation rewrites.
            raise PipelineError(
                f"not yet supported in analysis pipelines: step {i} (filter) "
                f"where={where!r} tests the running value after a many-to-one "
                "(summarizing) `project` step -- the flat machinery this single-input "
                "pipeline compiles onto applies every filter's `where` in the Filter "
                "stage (stage 1 of `BaseAnalysis.apply_params`), which runs BEFORE the "
                "project's own crosswalk aggregation (stage 4, via "
                "`Options.summarize_by`/`summarize_aggregate`) regardless of where the "
                "filter is declared -- so a value condition ordered after the project "
                "would drop SOURCE rows before the crosswalk join ever sees them, not "
                "filter the declared post-project result. VERIFIED: project(sum) -> "
                "filter(value > 2) over ecoregions valued 1.0-4.0 crosswalked "
                "(1,2)->100/(3,4)->200 gives {100: 3.0, 200: 7.0} declared vs {100: "
                "0.0, 200: 7.0} actually executed -- no error, a plausible-looking "
                "wrong answer. A "
                "condition on a DIMENSION column (not `value`) is unaffected by this "
                "and stays legal here -- a projection rewrites `grid_cell` and "
                "`value`, never a dimension column -- so move a value condition "
                "before the project step by declaring it as a LEADING `filter` step "
                "instead, naming `input` explicitly the way this pipeline's own "
                "project step does (e.g. `{\"op\": \"filter\", \"input\": "
                f"{input_name!r}, \"where\": {where!r}}}` as step 1, ahead of the "
                "project step -- see `FilterStep.input`'s docstring), or filter on "
                "`grid_cell` instead if that's what it tests (see "
                "`_extract_post_project_grid_cell_filter`)."
            )
        # Fold this step's OWN conjuncts in individually, rather than keeping this
        # step's whole (possibly multi-conjunct) `where` as one un-split string -- this
        # is the fix for a pre-existing, adjacent bug. The combining line below wraps
        # each entry in `rewritten_conjuncts` in its own
        # paren before joining with `AND`, to keep an `OR` inside one entry from
        # leaking its precedence into a neighboring entry -- but wrapping a whole
        # multi-conjunct STEP (the previous shape: one `rewritten_filters` entry per
        # step, not per conjunct) hides that step's own top-level `AND` from anyone who
        # re-splits the combined string later, the same way `split_top_level_and`
        # itself would be hidden by any other unrelated parens. Concretely: a single
        # filter step `where="model_group_id = 500 AND value > 5"` used to combine to
        # `Options.where = "(model_group_id = 500 AND value > 5)"` -- and
        # `BaseAnalysis._apply_filter_params` re-runs `classify_where` on
        # `self.options.where` at EXECUTION time (base.py:311-327+), where the outer
        # parens keep the inner `AND` at paren-depth 1 the whole way through, so
        # `split_top_level_and` returns ONE conjunct spanning both columns --
        # and `classify_where` then rejects it outright ("references both a dimension
        # column and `value`"), a false rejection of a perfectly legal filter, and (had
        # it NOT been rejected) exactly the kind of misclassified-conjunct wrong ACCEPT
        # this phase exists to prevent, since this same `classify_where` split is what
        # the guard above relies on to tell a legal dimension conjunct from an illegal
        # value one. Splitting into atomic conjuncts HERE (already done, above, via
        # `classify_where`) and wrapping each atomic conjunct -- not each step -- keeps
        # the boundaries any later re-split will find identical to the ones just
        # classified right here, so nothing downstream can misclassify what this
        # function already classified correctly.
        rewritten_conjuncts.extend(dimension_conjuncts)
        rewritten_conjuncts.extend(value_conjuncts)
    if rewritten_conjuncts:
        option_kwargs["where"] = " AND ".join(f"({c})" for c in rewritten_conjuncts)

    if calculate_steps:
        # A calculate step compiles onto exactly the flat endpoints' generic calculate
        # entries (`Options.calculate`), in step order -- the same code path
        # `calculate.N.expression`/`calculate.N.where` take. Like `filter.where`, the
        # running value may be referenced by its in-scope name (the input/step names in
        # `rename_map`) or by the `value` placeholder; both rewrite to `value` here.
        entries = []
        for i, step in calculate_steps:
            expression = rewrite_named_columns(step.expression, rename_map)
            try:
                parse_calculate_expression(expression)
            except ExpressionError as exc:
                raise PipelineError(
                    f"step {i} (calculate) expression {step.expression!r} is invalid: {exc}"
                ) from exc
            where = None
            if step.where is not None:
                where = rewrite_named_columns(step.where, rename_map)
                _validate_filter_where(i, where, {"value"}, op="calculate")
            entries.append({"expression": expression, "where": where})
        option_kwargs["calculate"] = entries

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

    if summarize_grid_cell is not None:
        option_kwargs["summarize_grid_cell"] = summarize_grid_cell

    # `output.sort`/`output.select` lower onto exactly `Options.sort`/`.select` -- the
    # same flat semantics a `sort=`/`select=` query parameter already has (see
    # `Output`'s docstring) -- unconditionally here, so every single-input shape this
    # function compiles (including the project-only-reaggregation degenerate case, where
    # `steps[1:]` is empty and every other `option_kwargs` entry above is untouched)
    # picks them up, not just a chain that also has a `filter`/`group` step.
    if pipeline.output.sort is not None:
        option_kwargs["sort"] = pipeline.output.sort
    if pipeline.output.select is not None:
        option_kwargs["select"] = pipeline.output.select
    # `output.color_map` lowers onto exactly `Options.color_map` -- the same flat
    # semantics a `color_map=` query parameter already has (`BaseAnalysis.legend` reads
    # it directly) -- unconditionally here, alongside `sort`/`select` immediately above,
    # for the same reason: every single-input shape this function compiles, including
    # the project-only-reaggregation degenerate case, must pick it up, not just a chain
    # that also has a `filter`/`calculate`/`group` step. This was the one place `pipeline.
    # output.color_map` was NOT lowered:
    # `_compile_paired_notebook_shape` lowers it onto `PairedAnalysis.color_map` directly
    # (that shape has no `Options` at all), but this function -- the single-input chain,
    # which is what a named/dataset-scoped analysis's raster PNG/JPEG tile route actually
    # compiles onto -- silently dropped it, so `color_map` had no effect on any
    # single-input pipeline's rendered output.
    if pipeline.output.color_map is not None:
        option_kwargs["color_map"] = pipeline.output.color_map

    dataset = _dataset_for(input_name, selection, base_path, **option_kwargs)

    # OPS-INTERPRETER FAST PATH: attach the op IR so `apply_params` (see
    # `VectorGridAnalysis.apply_params`/`RasterGridAnalysis.apply_params`'s own docstrings)
    # executes this pipeline through the ops interpreter (`apply_steps`) IN DECLARED
    # ORDER, instead of `BaseAnalysis`'s fixed five-stage flow that the SUMMARIZING-PROJECT ORDER GUARD
    # above exists BECAUSE that fixed order and a pipeline's declared order can disagree.
    # `dataset` is built via the exact same `_dataset_for` call as before this change --
    # every `Options` field above is populated identically regardless of which engine
    # ends up executing it (`estimate_data_size`, `legend`, `output_type`, ... all still
    # read `Options` directly and are unaffected) -- so attaching `_pipeline_ops`
    # afterward changes nothing except what `apply_params()` (called by every consumer:
    # `api/tabular.py`, `api/vector.py`, tile routes, ...) does internally.
    #
    # `is_vector_input` (computed once, near the top of this function, from the INPUT's
    # own grid type -- not the output's) admits every vector-input shape unconditionally,
    # same as before this change: `selection.grid` is `raster` for a raster input
    # summarizing (many-to-one) onto a VECTOR `summarize_by` grid -- a real, existing shape
    # (the `direction == "many_to_one"` branch above) that is NOT `is_vector_input`, so it
    # falls to the `is_raster_input` branch below instead, not this one. A vector input can
    # never reach a raster `to` at all -- `_project_direction` rejects raster-to-raster
    # outright, and this function itself rejects a vector input's one-to-many broadcast a
    # few lines above ("broadcasts onto a finer grid ... pair it with a join instead") --
    # so checking the input's own grid type here is equivalent to "is `dataset` actually a
    # `VectorGridAnalysis`" without importing that class just to `isinstance`-check it.
    #
    # `is_raster_input` admits a raster-input shape EXCEPT when `selection.grid_cell` is
    # set (lowered a few lines above, in `_dataset_for`, onto `option_kwargs["grid_cell"]`
    # -> `Options.grid_cell`): `RasterGridAnalysis._apply_grid_cell` reshapes `self.lf`
    # from the band-metadata table straight into a long (grid_cell/value) frame via
    # `.unpivot(...)` before any pipeline op could run at all, a shape change the raster
    # interpreter's PRE-project mode (which assumes `self.lf` starts as the metadata
    # table) does not accommodate -- see `RasterGridAnalysis.apply_steps`'s own docstring.
    # A raster pipeline with `grid_cell` set therefore stays on the old fixed-stage engine
    # entirely, exactly as it did before this change (`RasterGridAnalysis.apply_params`'s
    # override still falls through to `BaseAnalysis.apply_params` whenever
    # `_pipeline_ops is None`).
    is_raster_input = GRIDS[selection.grid].grid_type == "raster"
    if is_vector_input or (is_raster_input and selection.grid_cell is None):
        dataset._pipeline_ops = compile_ops(pipeline)

    return dataset


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
    if pipeline.output.sort is not None or pipeline.output.select is not None:
        raise PipelineError(
            "not yet supported in analysis pipelines: `output.sort`/`output.select` in "
            "the paired (spatially-joined) pipeline shape -- both lower onto "
            "`Options.sort`/`.select`, which only take effect through "
            "`BaseAnalysis.apply_params()`'s sort/select stage, but this shape's "
            "`fine_dataset`/`coarse_dataset` never call `apply_params()` at all "
            "(`PairedAnalysis._prepare()` calls `.materialize()` on them directly) -- "
            "setting either here would silently no-op rather than sort/narrow anything"
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

    if coarse_selection.grid_cell is not None or fine_selection.grid_cell is not None:
        raise PipelineError(
            "not yet supported in analysis pipelines: an input `grid_cell` filter in "
            "the paired (spatially-joined) pipeline shape -- it lowers onto "
            "`Options.grid_cell`, which only takes effect via "
            "`BaseAnalysis._apply_grid_cell()`, itself only invoked from "
            "`apply_params()`; this shape's `fine_dataset`/`coarse_dataset` never call "
            "`apply_params()` (`PairedAnalysis._prepare()` calls `.materialize()` on "
            "them directly), so setting it here would silently no-op rather than "
            "filter anything"
        )

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
        elif isinstance(step, CalculateStep):
            raise PipelineError(
                f"not yet supported in analysis pipelines: step {i} (calculate) in the "
                "paired (spatially-joined) pipeline shape -- `calculate` compiles only "
                "for single-input pipelines today; apply the rewrite on the flat data "
                "endpoint (or a single-input pipeline) feeding this analysis instead"
            )
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
