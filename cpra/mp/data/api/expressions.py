"""A restricted expression surface for value-level predicates and conditional aggregates.

Everything else the API accepts a SQL fragment for (`Options.where`, `Options.negate`,
`Options.multiply10`, ...) is handed to `pl.sql_expr` -- or even a full `lf.sql(...)`
call, for `where` -- unrestricted, because it only ever runs against DIMENSION columns:
the raster metadata table (path/band/dimensions) or a vector frame before its values are
touched. A value-level predicate is different in kind, not just in which columns it can
name: it runs against actual pixel/cell VALUES, on the hot path Polars can stream, so an
expression shaped wrong doesn't just give a bad answer -- it can force a full
materialization the query would otherwise never need. The measured example (see
raster_source.py's module docstring and the plan this module implements): two queries
returning the identical 126,174-group result set differ 94x in memory (285 MB vs 26859
MB) purely on whether one comparison in the predicate was `pl.col(x) == pl.col(x).min()`
instead of a literal. An aggregate inside a filter forces Polars to see every row before
it can evaluate what was supposed to be a per-row test, which blocks the pushdown this
whole plugin exists to make work.

So this module's job is narrower and stricter than "parse a SQL string": reject, at parse
time and before any file is opened, anything that is not an elementwise map, a filter
test, or an associative reduction over an explicit group key -- the operators closed
under streaming. Rejection is explicit (naming the offending token) rather than a
best-effort sanitizer, because a sanitizer that misses one shape is worse than a
whitelist that is honest about what it doesn't yet support.

Every function here takes and returns plain values (strings, sets, `polars.Expr`) and
raises `ExpressionError`, an ordinary `ValueError` subclass -- never `HTTPException` and
never a FastAPI import. `Dataset` (`api/dataset.py`) is the only place that translates an
`ExpressionError` into a 400 response; that split is what lets this module be reused
outside the API later (the `read_data()` entry point in `cpra/mp/data/__init__.py`).
"""

import re

import polars as pl


class ExpressionError(ValueError):
    """A user-supplied value expression failed the whitelist, the column-namespace
    check, or the plan-time streaming-safety rules below. A `ValueError` subclass so a
    caller that only wants to catch "this input was bad" doesn't need a second except
    clause, and so it plugs into the same `try/except ValueError` shape the rest of the
    library-first code in this package already uses.
    """


# Elementwise / scalar functions a value expression may call. Anything else -- including
# every Polars/SQL function not listed, not just the aggregates below -- is rejected:
# the point of a whitelist is that an *unlisted* function is refused by default, not
# admitted until proven dangerous.
ALLOWED_FUNCTIONS = frozenset({"abs", "round", "floor", "ceil", "sqrt", "coalesce", "cast"})

# Functions that reduce many rows to one. A value expression must never call one of
# these directly -- see the module docstring for the measured 94x memory cliff this
# exists to prevent. (Polars' own aggregate methods, e.g. `.min()`, are not reachable
# from `pl.sql_expr` at all -- this list guards the SQL spelling, `MIN(x)`.)
AGGREGATE_FUNCTION_NAMES = frozenset(
    {
        "min", "max", "sum", "avg", "mean", "count", "median", "stddev", "stddev_samp",
        "std", "var", "var_samp", "variance", "quantile", "first", "last", "mode",
        "array_agg", "string_agg",
    }
)

# Substrings that signal a non-streaming-closed construct: a sort (ORDER BY), a
# cross-row join, or a window function (OVER / PARTITION BY). None of these are
# elementwise maps, filters, or associative reductions over an explicit group key, so
# none of them are admitted at any point in this surface -- see the module docstring's
# "closed under streaming" framing.
_NON_STREAMING_KEYWORDS = ("order by", "group by", " join ", " over ", "partition by", "window")

_IDENTIFIER_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# The placeholder every value expression is written against, uniformly across both grid
# types -- see `rewrite_value_column`. Matched on a word boundary so an identifier that
# merely CONTAINS "value" (there are none in the current column namespaces, but nothing
# stops a future dimension from being named that way) is never partially rewritten.
_VALUE_TOKEN_RE = re.compile(r"\bvalue\b")


def _check_streaming_closed(expr_str):
    lowered = f" {expr_str.lower()} "
    for keyword in _NON_STREAMING_KEYWORDS:
        if keyword in lowered:
            raise ExpressionError(
                f"'{keyword.strip()}' is not allowed in a value expression (not closed under streaming)"
            )


def _check_no_aggregate_calls(expr_str):
    """The plan-time rule that must run before `pl.sql_expr` ever gets near the string:
    a match here means REJECT, unconditionally, before any file is opened -- there is no
    "aggregate is fine here but not there" exception inside one predicate.
    """
    for match in _IDENTIFIER_CALL_RE.finditer(expr_str):
        name = match.group(1).lower()
        if name in AGGREGATE_FUNCTION_NAMES:
            raise ExpressionError(
                f"aggregate function '{name}' is not allowed inside a value expression -- "
                "it forces a full pass before the predicate can be evaluated per row"
            )


def _check_allowed_calls(expr_str):
    for match in _IDENTIFIER_CALL_RE.finditer(expr_str):
        name = match.group(1).lower()
        if name in AGGREGATE_FUNCTION_NAMES:
            continue  # reported by _check_no_aggregate_calls with a more specific message
        if name not in ALLOWED_FUNCTIONS:
            raise ExpressionError(f"function '{name}' is not in the allowed whitelist {sorted(ALLOWED_FUNCTIONS)}")


def _check_column_namespace(expr, allowed_columns):
    referenced = set(expr.meta.root_names())
    disallowed = referenced - set(allowed_columns)
    if disallowed:
        token = sorted(disallowed)[0]
        raise ExpressionError(f"column '{token}' is not in the allowed namespace {sorted(allowed_columns)}")


def parse_value_expr(expr_str, allowed_columns, *, reject_aggregates=True):
    """Parse `expr_str` as a value-level SQL expression against a restricted whitelist.

    `allowed_columns` is the column namespace this particular caller's frame exposes --
    band names plus `grid_cell` for a raster row (see `raster_source.scan_rasters`'s
    `b<i>` convention), or `value`/`grid_cell`/dimension columns for a vector row. Every
    column `expr_str` references must be in it; the check runs by NAME
    (`Expr.meta.root_names()`), not by guessing at the frame's schema, so a caller that
    thinks a column is exposed but got the name wrong fails here rather than with a
    Polars `ColumnNotFoundError` deep inside a streaming collect.

    `reject_aggregates=False` exists only for a future caller that has already isolated
    an expression to run inside a `group_by().agg(...)` (where an aggregate function is
    exactly what's wanted) -- every current caller in this package leaves it at the
    default, since every current use is a per-row predicate or a per-row mask.

    Rejection order matters for the error a caller sees: the streaming-closed check and
    the aggregate-call check run on the raw STRING, before `pl.sql_expr` is asked to
    parse it at all, so a rejected expression never reaches the parser (cheap, and
    exactly what "before any file is opened" requires structurally, not just in
    aggregate). The column-namespace check runs after parsing, since it needs the parsed
    `Expr`'s own view of which columns it references rather than a regex guess.
    """
    if not expr_str or not expr_str.strip():
        raise ExpressionError("empty expression")

    _check_streaming_closed(expr_str)
    if reject_aggregates:
        _check_no_aggregate_calls(expr_str)
    _check_allowed_calls(expr_str)

    try:
        expr = pl.sql_expr(expr_str)
    except Exception as exc:  # pl.sql_expr's own exception types are not part of its stable API
        raise ExpressionError(f"invalid expression {expr_str!r}: {exc}") from exc

    _check_column_namespace(expr, allowed_columns)
    return expr


def rewrite_value_column(expr_str, column_name):
    """Rewrite the `value` placeholder in a value-predicate string to a concrete column
    name.

    Lets one user-supplied expression, written once against the placeholder `value` (the
    uniform surface both grid types present -- see the module docstring), be evaluated
    against a raster's actual per-band synthetic column (`raster_source.scan_rasters`'s
    `b<i>` names) once per selected band. A vector frame already has a real `value`
    column, so it never needs this rewrite -- it parses the placeholder string directly.

    A word-boundary regex substitution, not a bare `str.replace`: a plain substring swap
    would also corrupt any identifier that merely CONTAINS "value". Safe here because the
    result is re-parsed and re-whitelisted by `parse_value_expr` immediately after (this
    function performs no validation of its own), so a substitution that happened to
    produce nonsense is still caught before it reaches Polars.
    """
    return _VALUE_TOKEN_RE.sub(column_name, expr_str)


def value_predicate_mask(expr_str, allowed_columns):
    """Parse a value predicate and return it as a mask with `null` coerced to `False`.

    The shape every "value_where" call site in `dataset.py` wants: a boolean expression
    suitable for `pl.when(mask).then(pl.col(...)).otherwise(None)`, which nulls out a
    non-matching value rather than dropping its row. That is what lets a masked-out value
    fall into every aggregate's EXISTING null-skipping behaviour (`guarded_aggregate`'s
    `count()`-based denominator, `POLARS_RASTER_AGGREGATES`, the vector
    `getattr(pl.col("value"), aggregate)()` call sites) exactly like a nodata pixel does,
    with no separate "predicate failed" code path anywhere downstream.

    `fill_null(False)`: a predicate over an already-null (nodata) value evaluates to
    null, not `False` (`null > x` is `null`, matching SQL three-valued logic) -- coercing
    it here means a nodata value is masked out for the same reason a real value that
    fails the predicate is, rather than surviving the `pl.when` by accident because
    neither branch condition technically matched `False`.
    """
    return parse_value_expr(expr_str, allowed_columns).fill_null(False)


def count_by_value_frame(lf, group_cols, value_col, max_distinct):
    """Categorical count-by-value: one row per (group, distinct value) in `value_col`,
    with the count of rows sharing that value -- a fundamentally different OUTPUT SHAPE
    from every other aggregate in this package (which collapse a group to one row), so it
    is never reachable through `group_aggregate` (see the `AggregateFunction.count_by_value`
    comment in types.py and `Dataset._apply_group_params`'s rejection of it there); only
    `summarize_aggregate` can express it.

    `group_by(*group_cols, value_col).agg(pl.len())` is itself an ordinary streaming-closed
    reduction -- an associative count over an explicit composite group key, no different in
    kind from any other `group_by().agg(...)` -- so nothing about EVALUATING it needs
    special handling. What needs bounding is the key's OWN cardinality: unlike a `group_by`
    on a dimension column (whose distinct values are known in advance from the catalog),
    `value_col` can be arbitrary pixel/cell data, and grouping on it directly makes the
    result cardinality `n_groups * n_distinct(value_col)` -- unbounded unless something
    stops a caller from pointing this at a continuous column. (Restricting by dtype isn't
    enough on its own: elevation is stored as an int32 millimetre value with a huge range,
    so an integer dtype alone does not imply low cardinality.)

    So this runs one cheap probe first: `approx_n_unique()` on `value_col` ALONE, with no
    group key of its own -- a single streaming-closed reduction whose own memory is bounded
    by a small sketch (HyperLogLog-family), not by the input size or the (group, value)
    cardinality it exists to rule out. If the estimate exceeds `max_distinct`, the real
    `group_by(*group_cols, value_col)` -- which for a continuous column WOULD blow up -- is
    never attempted; `ExpressionError` is raised instead, before any grouped aggregate
    runs. This does mean `value_col` is scanned twice (once for the probe, once for the
    real aggregate); that is an accepted, deliberate cost -- a bounded single-column probe
    is cheap relative to the unbounded group_by it exists to prevent from ever starting.

    Nulls in `value_col` (a masked `value_where` non-match, or a masked nodata pixel) are
    excluded before either the probe or the real aggregate sees them -- they are not a
    valid "category", and `approx_n_unique()`/`group_by` would otherwise both count `None`
    as one more distinct value.
    """
    non_null = lf.filter(pl.col(value_col).is_not_null())
    n_unique = non_null.select(pl.col(value_col).approx_n_unique()).collect(engine="streaming").item()
    if n_unique > max_distinct:
        raise ExpressionError(
            f"count_by_value would produce approximately {n_unique} distinct values for "
            f"'{value_col}', over the limit of {max_distinct} -- count_by_value is meant "
            "for bounded categorical data (e.g. a land-type code), not continuous data"
        )
    return non_null.group_by(*group_cols, value_col).agg(pl.len().alias("count"))
