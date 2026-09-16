"""Option handling for the fixed five-stage flow: turning the request's `Options` fields
into the arguments each stage's algebra actually runs on.

The distinction this module draws is between the ALGEBRA of a stage and the OPTION
HANDLING around it. `base.apply_dimension_filter`/`apply_value_filter`/`apply_calculate_
plans` are the algebra -- re-entrant, shape-preserving, callable once per op by either
interpreter. What sat around them on `BaseAnalysis` was not: reading `options.where`,
deciding a raster's value conjunct waits for pixels, enumerating `options.calculate` into
labelled entries, rejecting a `group_aggregate` this engine has no shape for, pairing
`summarize_by` with `summarize_aggregate`. That work is driven entirely by `Options` and
the dataset's grid type, never by which ops ran, and it needs no instance -- so each
function here takes exactly what it reads and returns what the caller should store.

`BaseAnalysis` keeps `_apply_filter_params`/`_apply_calculate_params`/`_apply_group_
params`/`_apply_sort_params` as thin stage methods, because those four names are the fixed
flow's own vocabulary: `apply_params` calls all four in order, and both interpreters
(`vector.VectorGridAnalysis.apply_steps`, `raster.RasterGridAnalysis.apply_steps`) call
`_apply_sort_params` at the end of their own walk. What each of those methods still does
itself is exactly the part that touches `self`: read `self.options`, store the result back
on `self.lf`, and dispatch the one hook (`_apply_grid_cell`, `_collapse_group`) its
subclass supplies. The three members that did NOT need a name on the class at all --
`_resolve_calculate_entries`, `_reject_calculate_with_count_by_value` and
`_validate_summarize_params` -- are gone from it entirely; their call sites (in `base.py`,
`raster.py` and `vector.py`) name the functions here directly.

`AnalysisError` and the three algebra functions are imported inside the functions that use
them, the same local-import idiom `analysis/utils.py` already uses: they are defined in
`analysis/base.py`, which imports this module at module scope, so a module-scope import
back would be a cycle.
"""

from cpra.mp.data.analysis import metadata
from cpra.mp.data.analysis.expressions import compile_calculate_entries
from cpra.mp.data.analysis.utils import unpack_values
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.utils import get_input_data_frame, required_columns

# The accepted `sort` values: every dimension name, plus a `-`-prefixed descending
# spelling of each. Built from `DIMENSIONS` here rather than imported from
# `base.DIMENSION_CHOICES` (which `base._get_group_by_columns` still uses for `group_by`)
# because `base` imports this module, so an import back would be a cycle.
SORT_CHOICES = list(DIMENSIONS.keys()) + [f"-{dim_name}" for dim_name in DIMENSIONS]


def apply_filter_options(lf, where, *, grid_type, is_aggregating):
    """Apply the filter stage's `where` option to `lf`.

    Returns `(lf, value_where, has_dimension_conjunct)`. `where` accepts both DIMENSION
    conditions and VALUE conditions (against the placeholder column `value`), AND-ed
    together in one string; `base.apply_dimension_filter` splits them (see
    `expressions.classify_where` for the full split/classify/reject contract) and applies
    the dimension half. `value_where` is the combined value half, which the caller stores
    for every later consumer to read.

    Whether the value half is applied HERE depends on the grid type. A vector frame
    already carries a real `value` column, so it can be applied immediately. A raster's
    `value` does not exist yet -- the frame is still the band-metadata table (path/band/
    dimensions), one row per selected band, not one row per pixel -- so a raster's value
    conjunct is returned unapplied, for the raster read paths to apply per band once
    pixels are actually read (`RasterGridAnalysis._apply_transform_params_impl` /
    `_summarize_zone_stats_plugin` / `_summarize_count_by_value_plugin`).

    `is_aggregating` chooses MASK vs. DROP for the vector case. AGGREGATING requests
    (`group_aggregate`/`summarize_aggregate`, `count_by_value` included) mask a
    non-matching value to null rather than dropping its row: that is what lets the
    predicate compose with whatever aggregate the request goes on to apply via their
    EXISTING null-skipping behavior (`sources.guarded_aggregate`'s count()-based denominator,
    count_by_value's own non-null filter, ...), rather than needing a second,
    predicate-aware code path. Dropping the row instead would also erase the group/zone's
    siblings that DO belong in the denominator for min/max/mean, and would make "count how
    many values passed" indistinguishable from "count how many rows exist". PLAIN
    (non-aggregating) requests have no aggregate downstream to skip a null for, so a
    masked-to-null value would just surface as a null in the response -- observably
    different from "this row failed the filter" and indistinguishable from a genuine
    nodata value. There, the row is dropped outright.

    A falsy `where` is a no-op: returns `(lf, None, False)` unchanged.

    Raises the plain `ExpressionError` its helpers raise, never `AnalysisError` -- the
    caller wraps this in `base.op_error_boundary()`.
    """
    from cpra.mp.data.analysis.base import apply_dimension_filter, apply_value_filter

    lf, value_where, has_dimension_conjunct = apply_dimension_filter(lf, where)

    if value_where is not None and grid_type == "vector":
        lf = apply_value_filter(lf, value_where, mask=is_aggregating)

    return lf, value_where, has_dimension_conjunct


def resolve_calculate_entries(calculate):
    """The effective calculate entries for a request, as `(label, expression, where)`
    triples in application order: the `options.calculate` list, applied in list order
    (entry N+1 sees entry N's output). A falsy `calculate` yields `[]`.
    """
    return [
        (f"calculate.{i}", entry.expression, entry.where)
        for i, entry in enumerate(calculate or [], start=1)
    ]


def resolve_calculate_plans(calculate):
    """Validate a request's `calculate` entries into `expressions.CalculatePlan`s.

    Separate from `apply_calculate_options` because validation happens for BOTH grid types,
    before any pixel is read, while application happens only for a frame that already
    carries a real `value` column -- and because the caller stores the plans regardless
    (a raster's pixel-read paths replay them per band via
    `raster.apply_calculate_band_wide`).

    Raises the plain `ExpressionError` `compile_calculate_entries` raises, never
    `AnalysisError` -- the caller wraps this in `base.op_error_boundary()`.
    """
    return compile_calculate_entries(resolve_calculate_entries(calculate))


def apply_calculate_options(lf, calculate_plans, grid_type):
    """Apply already-validated `calculate_plans` to `lf`, when `lf` is something they can
    be applied to.

    Returns `lf` unchanged when there are no plans, or when `lf` is still a raster's
    band-metadata table (`metadata.is_raster_metadata_table`) -- there the entry's
    dimension `where` selects BANDS and its expression/value-`where` are applied per band,
    against actual pixels, by every raster read path (`_summarize_zone_stats_plugin`,
    `_collapse_tile_bands`, `_write_raster_windowed`) via
    `raster.apply_calculate_band_wide`. Otherwise the rewrite itself is
    `base.apply_calculate_plans` -- rows not matching an entry's `where` pass through
    unchanged (calculate never drops or masks data; that is Filter's job) and a missing
    value stays missing.

    Raises the plain `ExpressionError` its helpers raise, never `AnalysisError` -- the
    caller wraps this in `base.op_error_boundary()`.
    """
    from cpra.mp.data.analysis.base import apply_calculate_plans

    if not calculate_plans or metadata.is_raster_metadata_table(lf, grid_type):
        return lf
    return apply_calculate_plans(lf, calculate_plans)


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
    `RasterGridAnalysis._collapse_group`) uses. summarize_by/
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
