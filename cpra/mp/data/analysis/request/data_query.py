"""Turn a plain data query -- the flat `DatasetDimensions`/`FilterDimensions`/`DataOptions`
triple the `/v1/datasets/{variable}/{time_unit}/{grid}` routes accept -- into a NEW-engine
`request.pipeline.Pipeline`, the one thing `api/pipeline.py`'s `respond_to_analysis*`
functions know how to answer.

This is deliberately the FIRST piece of the flat-route migration written, and the only one
tested in isolation before any route is rewired to call it (see this package's own
migration notes): every flat data/tile route's Pipeline-shaped request is "one input, at
most one `filter` step, an `Output`" -- there is no grouping, no join, no project, nothing
`Options` (as opposed to `DataOptions`) would add. Building that one shape correctly, and
proving it here against `request.pipeline.validate_pipeline`'s own rules, means every route
this feeds only has to worry about response plumbing, not pipeline construction.

**Why `analysis.parameters` (the legacy root) rather than a `request`-namespaced copy.**
`DatasetDimensions`/`FilterDimensions`/`DataOptions` already exist, are already the models
the flat routes' query parameters bind to today, and are structurally read off the same
live `cpra.mp.data.dimensions.DIMENSIONS` table `request.types.dimension_type` is. Building
a second copy under `request/` would just be the same fields typed out twice. Importing
from the legacy `analysis/` root is the explicit, temporary coexistence allowance
`tests/api/test_analysis_layering.py` documents (`_LEGACY_ROOT_IMPORT_ALLOWED`); it is not
a layering violation.

**What this module does NOT do.** It never opens a file, never knows a grid's kind
(vector/raster), and never resolves a coordinate-valued `grid_cell` to an ID -- all of that
needs `io/catalog.py`/`io/geography.py`, which is `io/`'s job, and `request/` may not import
`io/` (see the layering test again). Concretely:

- A digit-only `grid_cell` (`"1"`, `"1,2,3"`) is handled directly, the same
  `str.isdigit()`-per-token rule `analysis.vector.apply_grid_cell_selection` already uses
  for a vector grid, since deciding "these are literal IDs" needs nothing but the string
  itself.
- Anything else -- a coordinate pair, or any `grid_cell` value on a raster dataset (which
  is *always* coordinates; see `analysis.vector.apply_grid_cell_selection`'s docstring on
  the raster convention this engine has no equivalent for yet) -- is the caller's problem
  to resolve first. Pass the already-resolved spatial `where` conjunct as `grid_cell_where`
  (e.g. `"grid_cell IN (1, 2)"`, or `"grid_cell != grid_cell"` for "matched nothing") and
  this function uses it verbatim instead of trying to parse `options.grid_cell` itself.
  `grid_cell_where`, when given, entirely REPLACES this module's own `options.grid_cell`
  handling -- it does not additionally validate or re-derive it.
- The result is only structurally validated (`request.pipeline.validate_pipeline`, via
  `Pipeline`'s own construction path below). A caller still needs `request.ops.
  validate_shapes` -- which needs the catalog, i.e. `io/` -- before compiling or executing
  the result, exactly like any other `Pipeline`.
"""

from datetime import date

from cpra.mp.data.analysis.parameters import BaseFilterDimensions, DataOptions, DatasetDimensions, FilterDimensions
from cpra.mp.data.analysis.request.errors import RequestError
from cpra.mp.data.analysis.request.expressions import split_top_level_and
from cpra.mp.data.analysis.request.pipeline import FilterStep, InputSelection, Output, Pipeline, validate_pipeline
from cpra.mp.data.analysis.request.types import dimension_type

# The one input a flat data query ever names -- there is nothing else in scope to
# disambiguate it from, so a fixed name (rather than one derived from the request) is
# simplest and is never seen by a caller: `respond_to_analysis*` addresses steps/output by
# position, not by this name.
_INPUT_NAME = "data"


def _quote_value(value, python_type: type) -> str:
    """Render one filter-dimension value as a `pl.sql()`-safe literal, per the dimension's
    own declared type (`request.types.dimension_type`): unquoted for `int`/`float` (the
    only two numeric dtypes any dimension declares -- see `structure/dimensions.csv`), the
    `DATE 'YYYY-MM-DD'` literal form Polars SQL expects for `date` (`calendar_day` is the
    only dimension of this type today), and a single-quoted, single-quote-escaped string
    literal for everything else (a plain `str` dimension, or a dynamically-built string
    `Enum` -- `FilterDimensions`'s own `use_enum_values=True` means an enum member already
    arrives here as its plain string value, not the `Enum` instance, so both cases render
    identically).
    """
    if python_type is int or python_type is float:
        return str(value)
    if python_type is date:
        return f"DATE '{value.isoformat()}'"
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _dimension_conjuncts(filter_dimensions: FilterDimensions) -> list[str]:
    """One `where` conjunct per `FilterDimensions` field the caller actually set, each an
    `=` (single value) or `IN (...)` (more than one) test against that dimension's column
    -- the same shape `analyses/`'s planner already lowers a dimension filter to. Iterates
    `BaseFilterDimensions.model_fields` in SORTED order rather than declaration order, so
    the composed `where` string (and therefore this function's output) does not depend on,
    or silently change with, `structure/dimensions.csv`'s row order -- a property this
    module's own tests rely on to assert an exact string.
    """
    conjuncts = []
    for name in sorted(BaseFilterDimensions.model_fields):
        values = getattr(filter_dimensions, name)
        if not values:
            continue
        python_type = dimension_type(name)
        quoted = [_quote_value(value, python_type) for value in values]
        if len(quoted) == 1:
            conjuncts.append(f"{name} = {quoted[0]}")
        else:
            conjuncts.append(f"{name} IN ({', '.join(quoted)})")
    return conjuncts


def _grid_cell_conjunct(grid_cell: str) -> str:
    """The `where` conjunct for a digit-only `grid_cell` (`"1"` or `"1,2,3"`): `grid_cell
    IN (...)`, unquoted -- `grid_cell` is an ordinary integer column in long shape (see
    `execution.long_ops.filter_long`'s docstring; `tests/api/test_execution_plan.py` pins
    `where: "grid_cell = 7"`, unquoted, as the real usage). Every token must be a plain
    (unsigned, non-decimal) digit string, the same rule `analysis.vector.
    apply_grid_cell_selection` uses to tell a literal ID apart from a coordinate --
    anything else must be resolved by the caller and passed as this function's caller's
    own `grid_cell_where` instead (see the module docstring); reaching this function with
    a non-digit token is therefore a caller error, not a user-input error, and is reported
    as such.
    """
    tokens = [token.strip() for token in grid_cell.split(",")]
    if not tokens or not all(token.isdigit() for token in tokens):
        raise RequestError(
            f"grid_cell={grid_cell!r} is not a plain comma-separated list of digit IDs -- "
            "a coordinate-valued grid_cell must be resolved to IDs by the caller first "
            "and passed as build_data_query_pipeline(..., grid_cell_where=...) instead of "
            "left for this function to parse"
        )
    if len(tokens) == 1:
        return f"grid_cell = {tokens[0]}"
    return f"grid_cell IN ({', '.join(tokens)})"


def build_data_query_pipeline(
    dataset_dimensions: DatasetDimensions,
    filter_dimensions: FilterDimensions,
    options: DataOptions,
    format: str,
    *,
    grid_cell_where: str | None = None,
) -> Pipeline:
    """Build a structurally-valid, single-input, at-most-one-`filter`-step `Pipeline` for
    a flat data query: `dataset_dimensions` becomes the one input, every set
    `filter_dimensions` field and `options.grid_cell`/`options.where` become AND-ed
    conjuncts on one `filter` step (omitted entirely when there is nothing to filter on),
    and `options.sort`/`options.select`/`options.color_map` plus `format` become `output`.

    `options.where`, when given, is split on its own top-level `AND`s
    (`expressions.split_top_level_and`) and each resulting conjunct is parenthesized
    individually before being AND-ed in alongside the dimension/`grid_cell` conjuncts.

    Both halves of that matter, for opposite reasons. Parenthesizing keeps a top-level
    `OR` scoped to what the caller wrote, rather than letting it spill into an AND
    relationship with the dimension filters this function adds around it (`where="a OR b"`
    composed naively with a `scenario_id = 7` conjunct as `scenario_id = 7 AND a OR b`
    would test `a` unconditionally; `scenario_id = 7 AND (a OR b)` is what the caller
    meant). Splitting FIRST is what keeps that parenthesizing from destroying the AND seam
    the rest of the engine classifies against: `expressions.classify_where_kinds` sorts a
    `where` into dimension/value/spatial predicates one top-level conjunct at a time, so
    wrapping a whole multi-conjunct `where` in ONE pair of parens (`(calendar_year > 2035
    AND value > 2040.16)`) hides the seam and hands the engine a single conjunct that
    references both a dimension and `value` -- rejected as a mixed-kind predicate, even
    though the caller wrote exactly the two separate conjuncts that rule asks for. Each
    piece a top-level split produces contains no top-level `AND` by construction, so
    parenthesizing the pieces protects an `OR` without hiding anything.

    `grid_cell_where`, when given, is used as the spatial conjunct verbatim, entirely
    bypassing this function's own digit-ID-only handling of `options.grid_cell` -- see the
    module docstring for why (coordinate resolution needs `io/`, which `request/` may not
    import) and `_grid_cell_conjunct` for the digit-ID path this replaces.

    Only runs `request.pipeline.validate_pipeline`'s STRUCTURAL checks (via `Pipeline`'s
    own construction, below) -- a caller still needs `request.ops.validate_shapes` (which
    needs the catalog) before compiling or executing the result, exactly like any other
    `Pipeline` `parse_pipeline`/`pipeline_from_query` return.
    """
    inputs = {
        _INPUT_NAME: InputSelection(
            variable=dataset_dimensions.variable,
            time_unit=dataset_dimensions.time_unit,
            grid=dataset_dimensions.grid,
        )
    }

    conjuncts = _dimension_conjuncts(filter_dimensions)

    if grid_cell_where is not None:
        conjuncts.append(grid_cell_where)
    elif options.grid_cell:
        conjuncts.append(_grid_cell_conjunct(options.grid_cell))

    if options.where:
        conjuncts.extend(f"({conjunct})" for conjunct in split_top_level_and(options.where))

    steps = []
    if conjuncts:
        steps.append(FilterStep(input=_INPUT_NAME, where=" AND ".join(conjuncts)))

    output = Output(
        format=format,
        color_map=options.color_map,
        sort=options.sort,
        select=options.select,
    )

    pipeline = Pipeline(inputs=inputs, steps=steps, output=output)
    validate_pipeline(pipeline)
    return pipeline
