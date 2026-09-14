
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
    dimension-and/or-value-condition grammar as the Filter stage's `where`; omitted means
    the expression applies to everything). Rows not matching `where` pass through the
    calculate stage unchanged -- calculate never drops or masks data.

    On the HTTP data endpoints these arrive as the indexed `calculate.N.expression`/
    `calculate.N.where` query parameters (see `parse_indexed_calculate`); the library
    surface (`create_analysis`/`analyze_data`) takes `calculate=[{"expression": ...,
    "where": ...}, ...]` directly. Entries apply sequentially in list order -- entry N+1
    sees entry N's output.
    """

    model_config = ConfigDict(extra="forbid")

    expression: str
    where: str | None = None


class ScalarOptions(BaseModel):
    """The flat-scalar half of `Options` -- every field FastAPI can bind directly as a
    query parameter. The API's query models (`api.models.TileQuery`/`DataQuery`) inherit
    from THIS class rather than `Options`, because `Options.calculate` is a structured
    list (of `CalculateEntry`) that has no flat query-parameter spelling: it arrives as
    the dotted `calculate.N.*` parameters, which a route dependency parses off the raw
    query string (`parse_indexed_calculate`) and sets onto the built `Options` object.
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
    # DEPRECATED sugar for a generic calculate entry (see `Options.calculate` below and
    # `BaseAnalysis._resolve_calculate_entries`): `negate=X` is exactly
    # `calculate.N.expression=value * -1` + `calculate.N.where=X` (multiply10/multiply100
    # likewise, factors 10/100), applied in negate, multiply10, multiply100 order when
    # several are given. Mixing any of these with a generic `calculate` entry is rejected
    # (their relative order would be ambiguous). `negate` sign-flips `value` (composing
    # with `group_aggregate=sum` across two requests to produce a difference -- see
    # `AggregateFunction`/`group_aggregate` and the data portal's own use of this idiom).
    # LIMITATION of that idiom: a cell/pixel present in only ONE of the two merged
    # selections is indistinguishable, after the sum, from a cell/pixel present in both
    # with the other side's value at zero -- the "difference" silently degrades to just
    # the surviving side's value, with no flag that anything was missing. Every grid this
    # codebase currently serves is dense (no such ragged selection exists), which is the
    # only reason this has not needed a fix.
    negate: str | None = None
    multiply10: str | None = None
    multiply100: str | None = None
    group_by: str | None = None
    group_aggregate: AggregateFunction | None = None
    color_map: str | None = None
    unpivot: bool = False
    summarize_by: VectorGrid | None = None
    summarize_aggregate: AggregateFunction | None = None
    summarize_grid_cell: str | None = None
    sort: str | None = None
    select: str | None = None


class Options(ScalarOptions):
    """`ScalarOptions` plus the one structured option, the generic calculate entries --
    the full option surface `BaseAnalysis` consumes and the library entry points
    (`create_analysis`/`analyze_data`) accept. Entries apply sequentially in list order;
    see `CalculateEntry` and `BaseAnalysis._apply_calculate_params`.
    """

    calculate: list[CalculateEntry] | None = None


_CALCULATE_FIELDS = frozenset({"expression", "where"})


def parse_indexed_calculate(params):
    """Parse the flat data endpoints' indexed `calculate.<N>.expression` /
    `calculate.<N>.where` query parameters out of `params` (any string mapping -- the
    route dependency hands over the raw query params, since FastAPI's query models bind
    only flat scalar fields and dotted names never reach them).

    Returns an ordered `list[CalculateEntry]`, or `None` when no `calculate*` key is
    present at all. Raises `AnalysisError` (400) naming the exact rule violated for
    anything malformed: a key that isn't `calculate.<N>.<field>`, an unknown field, an
    index that isn't a plain 1-based integer, indices that aren't contiguous from 1, or
    an entry missing its required `expression`.
    """
    from cpra.mp.data.analysis.base import AnalysisError

    raw_entries = {}
    found = False
    for key, value in params.items():
        if key != "calculate" and not key.startswith("calculate."):
            continue
        found = True
        parts = key.split(".")
        if len(parts) != 3:
            raise AnalysisError(
                f"invalid calculate parameter {key!r} -- calculate entries are indexed "
                "expression/where pairs: calculate.<N>.expression=... and optionally "
                "calculate.<N>.where=... (e.g. calculate.1.expression=value * 0.6)",
                status_code=400,
            )
        _, index_str, field = parts
        if not index_str.isdigit() or index_str != str(int(index_str)) or int(index_str) < 1:
            raise AnalysisError(
                f"invalid calculate index {index_str!r} in {key!r} -- indices are 1-based "
                "integers with no leading zero",
                status_code=400,
            )
        if field not in _CALCULATE_FIELDS:
            raise AnalysisError(
                f"unknown calculate field {field!r} in {key!r} -- valid fields: "
                f"{', '.join(sorted(_CALCULATE_FIELDS))}",
                status_code=400,
            )
        raw_entries.setdefault(int(index_str), {})[field] = value

    if not found:
        return None

    indices = sorted(raw_entries)
    if indices != list(range(1, len(indices) + 1)):
        raise AnalysisError(
            f"calculate indices must be contiguous starting at 1 -- got {indices}",
            status_code=400,
        )

    entries = []
    for index in indices:
        raw = raw_entries[index]
        if "expression" not in raw:
            raise AnalysisError(
                f"calculate.{index} is missing its required 'expression' (only "
                f"calculate.{index}.where was given)",
                status_code=400,
            )
        entries.append(CalculateEntry(**raw))
    return entries


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


