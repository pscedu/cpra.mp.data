"""Option handling for the FLAT request surface: turning `Options`' fields into the op
list the one execution engine walks, plus the validations and the two options that have no
op form.

The distinction this module draws is between the ALGEBRA of an op and the OPTION HANDLING
around it. `base.apply_dimension_filter`/`apply_value_filter`/`apply_calculate_plans` are
the algebra -- re-entrant, shape-preserving, callable once per op by either interpreter
(`vector.apply_long_op` for a long frame, `raster.apply_band_wide_op` for a band-wide
one). What sat around them on `BaseAnalysis` was not: reading `options.where`, enumerating
`options.calculate` into labelled entries, resolving `group_by` into columns, rejecting a
`group_aggregate` this engine has no shape for, pairing `summarize_by` with
`summarize_aggregate`. That work is driven entirely by `Options`, never by which ops ran,
and it needs no instance -- so each function here takes exactly what it reads and returns
what the caller should store.

`lower_flat_options` is the join between the two request surfaces. A declared pipeline
arrives at `BaseAnalysis.apply_params` with its op list already compiled
(`self._pipeline_ops`); a flat request arrives with `Options` fields instead, and this is
what turns them into the SAME list, so there is exactly one thing left that executes a
request. Everything else here is either a validation that list cannot express
(`reject_count_by_value_group`, `reject_calculate_with_count_by_value`,
`validate_summarize_options`) or an option that is not a step at all
(`apply_sort_options`, which describes the response's columns rather than its data).

`AnalysisError` and the algebra functions are imported inside the functions that use them,
the same local-import idiom `analysis/utils.py` already uses: they are defined in
`analysis/base.py`, which imports this module at module scope, so a module-scope import
back would be a cycle.
"""

from cpra.mp.data.analysis.io.raster_io import unpack_values
from cpra.mp.data.analysis.ops import CalculateOp, FilterOp, GroupOp, _default_output_name
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.utils import get_input_data_frame, required_columns

# The accepted `sort` values: every dimension name, plus a `-`-prefixed descending
# spelling of each.
SORT_CHOICES = list(DIMENSIONS.keys()) + [f"-{dim_name}" for dim_name in DIMENSIONS]

# The accepted `group_by` values: every dimension name. Both lists are built from
# `DIMENSIONS` here, next to the option handling that validates against them, rather than
# in `analysis/base.py` (where the `group_by` list used to live, as `DIMENSION_CHOICES`) --
# `base` imports this module, so an import back would be a cycle.
GROUP_BY_CHOICES = list(DIMENSIONS.keys())

# The placeholder name a flat request's `where`/`calculate` expressions write the running
# value as -- the literal column `expressions.parse_value_expr` whitelists. A pipeline
# names its running value by the input's own name or a step's `as_` alias, which
# `apply_steps` rewrites to this same placeholder via `expressions.rewrite_named_columns`;
# a flat request has no such names, so the ops `lower_flat_options` emits declare their
# input AS the placeholder and the rewrite that both interpreters run is an identity.
FLAT_VALUE_NAME = "value"


def resolve_group_by_columns(group_by, group_aggregate):
    """The `GroupOp.by` a flat request's `group_by`/`group_aggregate` pair asks for, and
    the pairing rule between them.

    Empty, not a `[True]` literal sentinel (its pre-cleanup shape): a caller grouping by
    `True` gets a column literally named "literal" back from Polars (`df.group_by(True)` --
    see `fold.py`'s `fold_group_by` docstring for the same naming quirk), which leaked into
    every collapse-to-cell/collapse-everything response as a meaningless `literal` boolean
    column. An empty list needs no such placeholder: `vector.apply_group_by` always appends
    "grid_cell" itself before grouping, so `[]` still collapses to one row per grid_cell,
    and `raster.collapse_band_group`'s `group_by()` with no columns collapses to a single
    overall group -- both without inventing a key column.

    `group_by` without `group_aggregate` is rejected: `GroupOp.aggregate` is not optional,
    and "group these rows together but reduce them with nothing" has no output shape.
    """
    from cpra.mp.data.analysis.base import AnalysisError

    if group_by is not None and group_aggregate is None:
        raise AnalysisError("group_aggregate must be specified when group_by is used")

    if group_by is None:
        return []
    return unpack_values(group_by, "group_by", GROUP_BY_CHOICES)


def lower_flat_options(options, *, is_aggregating):
    """Lower a FLAT request's `Options` into the op IR (`list[ops.Op]`), so that every
    request -- flat or piped -- executes through the same interpreter
    (`vector.VectorGridAnalysis.apply_steps` / `raster.RasterGridAnalysis.apply_steps`)
    walking a `list[Op]`, and there is exactly one execution engine rather than two.

    **Order-preserving by construction.** The ops come out in one fixed order -- `filter`
    (`where`), then `calculate` (each entry, in list order), then `group`
    (`group_by`/`group_aggregate`) -- and the flat surface has no other order to preserve,
    because `Options` is a flat bag of keyword arguments with no declared sequence at all.
    `sort`/`select` and `summarize_by`/`summarize_aggregate` are deliberately NOT lowered:

    - `sort`/`select` are response SHAPING options with no op form; both interpreters'
      caller (`base.BaseAnalysis.apply_params`) applies `apply_sort_options` after the
      walk, since they describe the response's columns rather than its data.
    - `summarize_by`/`summarize_aggregate` are the flat spelling of a many-to-one
      `project`, but they carry two things `ProjectOp` does not: `summarize_grid_cell` (a
      narrowing of the DESTINATION grid's cells, which also feeds
      `summarize_grid_cell_ids` and through it `get_vector_tile`) and the raster
      zonal path's whole `validate_raster_summarize`/`apply_zonal_summarize` pair. Both
      interpreters already run `_apply_transform_params_impl` -- the method that reads
      exactly those options -- in a trailing, always-last position for a chain that never
      summarized (see each `apply_steps`), which is where an orderless flat summarize
      belongs anyway. Emitting a `ProjectOp` here instead would route a flat summarize
      through `vector.apply_project`, which knows nothing about `summarize_grid_cell`.

    `is_aggregating` (from `base.BaseAnalysis._is_aggregating`) becomes
    `FilterOp.mask_value` -- see that field's own docstring for why a flat `where` masks on
    an aggregating request where a pipeline `filter` step drops.

    Every op declares `input=FLAT_VALUE_NAME` and a synthetic digit-string `output`
    (`ops._default_output_name`), which together make both interpreters' `rename_map` an
    identity: `value` maps to `value`, and a digit-string output is excluded from the map
    by the same `output[0].isdigit()` filter that keeps a pipeline's synthetic names out of
    it (see `VectorGridAnalysis.apply_steps`' docstring for why that filter exists).

    Raises `AnalysisError` for the two rejections that belong to this pair of options
    rather than to any frame -- `group_by` without `group_aggregate`
    (`resolve_group_by_columns`) and `group_aggregate='count_by_value'`
    (`reject_count_by_value_group`). Both fire here, before a file is opened, exactly as
    they did from the group stage.
    """
    ops = []

    if options.where:
        ops.append(
            FilterOp(
                step_no=len(ops) + 1,
                input=FLAT_VALUE_NAME,
                where=options.where,
                output=_default_output_name(len(ops) + 1),
                mask_value=is_aggregating,
            )
        )

    for entry_no, entry in enumerate(options.calculate or [], start=1):
        ops.append(
            CalculateOp(
                step_no=len(ops) + 1,
                input=FLAT_VALUE_NAME,
                expression=entry.expression,
                where=entry.where,
                output=_default_output_name(len(ops) + 1),
                label=f"calculate.{entry_no}",
            )
        )

    if options.group_by is not None or options.group_aggregate is not None:
        reject_count_by_value_group(options.group_aggregate)
        ops.append(
            GroupOp(
                step_no=len(ops) + 1,
                input=FLAT_VALUE_NAME,
                by=resolve_group_by_columns(options.group_by, options.group_aggregate),
                aggregate=options.group_aggregate,
                output=_default_output_name(len(ops) + 1),
            )
        )

    return ops


def reject_count_by_value_group(group_aggregate):
    """Reject `group_aggregate=count_by_value`.

    count_by_value changes the OUTPUT SHAPE (one row per group per distinct value) instead
    of collapsing a group to one row the way every other `group_aggregate` value does -- on
    both grid types, `group_by`/`group_aggregate` is a COLLAPSE step (raster band-collapse
    across selected bands per pixel, vector dimension-collapse across rows per grid_cell)
    that has no way to also multiply its own row count by the number of distinct values
    without breaking that collapse's "one row per group" contract. Rejected here, before
    any file is read, rather than left to fail deep in
    `sources.HORIZONTAL_RASTER_AGGREGATES`/`getattr(pl.col("value"), ...)` -- the same
    pattern the raster-only `group_aggregate == "count"` rejection (in
    `raster.collapse_band_group`) uses. summarize_by/
    summarize_aggregate=count_by_value is the only way to reach count_by_value at all.
    """
    from cpra.mp.data.analysis.base import AnalysisError

    if group_aggregate == "count_by_value":
        raise AnalysisError(
            "group_aggregate 'count_by_value' is not supported -- "
            "use summarize_by and summarize_aggregate=count_by_value instead"
        )


def reject_calculate_with_count_by_value(calculate):
    """Reject a request that combines `calculate` with a count_by_value summarize.

    count_by_value counts DISTINCT values; a calculate entry rewrites a value BEFORE it
    would be counted, which changes which raw values collapse into "the same" category
    without changing what the caller meant by "distinct" -- e.g. a `value * 10` entry turns
    land-type codes 1 and 2 into 10 and 20, not into two different categories, but does
    change what a downstream consumer sees as the reported "value" for each count. Every
    other aggregate folds a calculate entry through unremarkably (a rewritten number is
    still a meaningful sum/mean/min/max), so this is a rejection specific to
    count_by_value, not a general calculate/summarize incompatibility.

    Checked against `calculate` (the request's declared entries) rather than against the
    frame, so it fires before the eventual group even if the calculate stage already ran
    earlier in the pipeline.
    """
    from cpra.mp.data.analysis.base import AnalysisError

    if calculate:
        raise AnalysisError("calculate parameters cannot be combined with count_by_value")


def validate_summarize_options(summarize_by, summarize_aggregate, summarize_grid_cell):
    """Validate the summarize options and parse `summarize_grid_cell` into integer IDs.

    Returns `(has_summarize, summarize_grid_cell_ids)`: `has_summarize` is True when
    `summarize_by`/`summarize_aggregate` were provided, False when NEITHER was (the signal
    both `_apply_transform_params_impl` implementations use to return without doing
    anything). Providing only one of the pair is an error.

    `summarize_grid_cell_ids` is `None` unless a summarize is actually happening AND
    `summarize_grid_cell` was given -- it names cells of the DESTINATION grid
    (`summarize_by`), so it is meaningless, and deliberately not parsed, without one.
    """
    from cpra.mp.data.analysis.base import AnalysisError

    if summarize_by is None and summarize_aggregate is None:
        return False, None

    if summarize_by is None or summarize_aggregate is None:
        raise AnalysisError("summarize_by and summarize_aggregate parameters must be used together")

    summarize_grid_cell_ids = None
    if summarize_grid_cell:
        # TODO: Validate that the given IDs fall within the grid min and max.
        summarize_grid_cell_ids = [
            int(geo_id) for geo_id in unpack_values(summarize_grid_cell, "summarize_grid_cell")
        ]

    return True, summarize_grid_cell_ids


def apply_sort_options(lf, sort, select, dataset_dimensions):
    """Apply the sort/select stage's two options to `lf`.

    `sort` accepts dimension names, optionally `-`-prefixed for descending (`SORT_CHOICES`).
    `select` is validated against the dataset's own required columns, `grid_cell` and
    `value` included, which is why this needs `dataset_dimensions` rather than just the
    frame.
    """
    if sort is not None:
        parts = unpack_values(sort, "sort", SORT_CHOICES)
        cols = [part.replace("-", "") for part in parts]
        desc = [part.startswith("-") for part in parts]
        lf = lf.sort(cols, descending=desc)

    if select is not None:
        required_cols = required_columns(
            get_input_data_frame(**dict(dataset_dimensions)),
            include_grid_cell=True,
            include_value=True,
        )
        select_cols = unpack_values(select, "select", required_cols)
        lf = lf.select(select_cols)

    return lf
