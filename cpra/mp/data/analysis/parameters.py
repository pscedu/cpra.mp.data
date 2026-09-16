
from pydantic import BaseModel, ConfigDict, create_model

from cpra.mp.data.analysis.types import (
    AggregateFunction,
    VectorGrid,
    dimension_type,
)
from cpra.mp.data.dimensions import DATASET_DIMENSIONS, DIMENSIONS

BaseDatasetDimensions = create_model(
    "BaseDatasetDimensions",
    __config__=ConfigDict(protected_namespaces=()),
    **{dim.name: (dimension_type(dim.name), None) for dim in DATASET_DIMENSIONS.values()},
)

BaseFilterDimensions = create_model(
    "BaseFilterDimensions",
    __config__=ConfigDict(protected_namespaces=()),
    **{
        dim.name: (
            list[dimension_type(dim.name)] | None,
            None,
        )
        for dim in DIMENSIONS.values()
        if dim.type not in ("dataset", "metadata")
    },
)


class DatasetDimensions(BaseDatasetDimensions):
    model_config = ConfigDict(use_enum_values=True)


class FilterDimensions(BaseFilterDimensions):
    model_config = ConfigDict(use_enum_values=True)


class CalculateEntry(BaseModel):
    """One generic calculate entry: `expression` (arithmetic over the placeholder column
    `value` and numeric literals -- see `expressions.parse_calculate_expression` for the
    exact grammar/whitelist) applied to the rows/bands/pixels matching `where` (the same
    dimension-and/or-value-condition grammar as `where` itself; omitted means
    the expression applies to everything). Rows not matching `where` pass through the
    calculate stage unchanged -- calculate never drops or masks data.

    This is the planner's lowering target -- `analyses/*.py` pipelines and the library
    surface (`create_analysis`/`analyze_data`) build these directly as
    `calculate=[{"expression": ..., "where": ...}, ...]`. Entries apply sequentially in
    list order -- entry N+1 sees entry N's output.
    """

    model_config = ConfigDict(extra="forbid")

    expression: str
    where: str | None = None


class DataOptions(BaseModel):
    """Plain data-access options: every field the HTTP data/tile endpoints
    (`api.models.TileQuery`/`DataQuery`) accept. `Options` (below) adds the
    pipeline-execution fields (`group_by`/`group_aggregate`/`summarize_*`/`calculate`)
    that only a compiled analysis pipeline sets -- those are no longer reachable as flat
    HTTP query parameters (see `api.__init__.reject_retired_params`).
    """

    model_config = ConfigDict(use_enum_values=True)

    grid_cell: str | None = None
    # Accepts both DIMENSION conditions (against the raster metadata table's path/band/
    # dimensions, or a vector frame's dimension columns -- historically the only kind
    # `where` accepted) and VALUE conditions (against actual pixel/cell VALUES, written
    # against the placeholder column `value` -- historically the separate, now-retired
    # `value_where` parameter), AND-ed together in one string, e.g.
    # `calendar_year > 2040 AND value > 3`. `api.expressions.classify_where` splits
    # `where` on top-level `AND` and sorts each conjunct by which kind it is: a dimension
    # conjunct is handed to `lf.sql()` largely unrestricted, the same as always; a value
    # conjunct is parsed through `api.expressions.parse_value_expr`'s restricted
    # whitelist instead, since it runs on the hot per-row path an aggregate or a subquery
    # inside it can force off of streaming pushdown entirely -- see `api/expressions.py`'s
    # module docstring. A single AND-term may not reference both kinds (rejected with an
    # error explaining the rule); write dimension and value conditions as separate AND-ed
    # conjuncts instead.
    #
    # For an AGGREGATING request (`group_aggregate`/`summarize_aggregate`), a value
    # conjunct's non-matching value is masked to null, not dropped as a row, so it
    # composes with the aggregate's existing null-skipping behaviour (including `count`,
    # which is what makes "count pixels where value > x" possible) instead of adding a
    # second "predicate failed" code path. For a PLAIN (non-aggregating) request, a row
    # failing a value conjunct is dropped instead, since there is no aggregate downstream
    # for a masked null to fall into unnoticed.
    where: str | None = None
    sort: str | None = None
    select: str | None = None
    color_map: str | None = None


class Options(DataOptions):
    """`DataOptions` plus the fields that describe how a compiled pipeline *executes*:
    `group_by`/`group_aggregate`/`summarize_*` collapse or transform rows, and
    `calculate` rewrites `value` -- all lowered onto this class by `planner.py` (see
    `analyses/*.py`) and consumed by `BaseAnalysis`. These are internal to the engine:
    not reachable as flat HTTP query parameters on the data endpoints (which
    accept only `DataOptions`'s fields) -- a named analysis's own request parameters are
    validated separately and lowered into these fields by the planner instead.
    """

    group_by: str | None = None
    group_aggregate: AggregateFunction | None = None
    summarize_by: VectorGrid | None = None
    summarize_aggregate: AggregateFunction | None = None
    summarize_grid_cell: str | None = None
    calculate: list[CalculateEntry] | None = None


def normalize_analysis_params(*, variable=None, time_unit=None, grid=None, **params):
    """Normalize `create_analysis`'s plain-value keyword surface into the validated
    `DatasetDimensions`/`FilterDimensions`/`Options` triple `BaseAnalysis` subclasses take
    internally. This split -- plain values in, three pydantic objects out -- is an
    internal implementation detail; the objects themselves are not the public contract
    (see `analysis/__init__.py`'s module docstring).

    `variable`/`time_unit`/`grid` select the dataset (all three required, the same as
    `DatasetDimensions`). Every other keyword in `params` is either a dimension filter
    (`scenario_id`, `calendar_year`, ...) -- accepted as a scalar (`scenario_id=7`) or a
    list (`scenario_id=[7, 8]`), a scalar being wrapped into a single-element list since
    `FilterDimensions` fields are always `list[...] | None` underneath -- or an `Options`
    field under its existing name (`where=`, `group_by=`, `group_aggregate=`,
    `grid_cell=`, ...), passed through as-is. An unrecognized keyword, or a missing
    `variable`/`time_unit`/`grid`, raises `AnalysisError` naming the problem and (for an
    unrecognized keyword) listing every valid dimension filter and option name, so a typo
    fails loudly instead of silently doing nothing.
    """
    from cpra.mp.data.analysis.base import AnalysisError

    missing = [name for name, value in (("variable", variable), ("time_unit", time_unit), ("grid", grid)) if value is None]
    if missing:
        raise AnalysisError(
            f"create_analysis() missing required argument(s): {', '.join(missing)}", status_code=400
        )

    filter_names = set(BaseFilterDimensions.model_fields)
    option_names = set(Options.model_fields)

    filter_kwargs = {}
    option_kwargs = {}
    unknown = []
    for name, value in params.items():
        if name in filter_names:
            filter_kwargs[name] = list(value) if isinstance(value, (list, tuple)) else [value]
        elif name in option_names:
            option_kwargs[name] = value
        else:
            unknown.append(name)

    if unknown:
        valid = ", ".join(sorted({"variable", "time_unit", "grid"} | filter_names | option_names))
        raise AnalysisError(
            f"create_analysis() got unexpected keyword argument(s): {', '.join(sorted(unknown))}. "
            f"Valid parameters are: {valid}.",
            status_code=400,
        )

    return (
        DatasetDimensions(variable=variable, time_unit=time_unit, grid=grid),
        FilterDimensions(**filter_kwargs),
        Options(**option_kwargs),
    )


