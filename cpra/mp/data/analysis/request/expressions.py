"""A restricted expression surface for value-level predicates and conditional aggregates.

`Options.where` accepts a single SQL-flavored boolean expression that may mix two kinds
of AND-ed conditions: DIMENSION conditions (against the raster metadata table --
path/band/dimensions -- or a vector frame's dimension columns) and VALUE conditions
(against actual pixel/cell VALUES, via the placeholder column `value`). Dimension
conjuncts are handed to `pl.sql_expr` -- or even a full `lf.sql(...)` call -- largely
unrestricted, the same as always. A value conjunct is different in kind, not just in
which columns it can name: it runs on the hot path Polars can stream, so an expression
shaped wrong doesn't just give a bad answer -- it can force a full materialization the
query would otherwise never need. The measured example (see `sources.py`'s module
docstring): two queries returning the identical
126,174-group result set differ 94x in memory (285 MB vs 26859 MB) purely on whether one
comparison in the predicate was `pl.col(x) == pl.col(x).min()` instead of a literal. An
aggregate inside a filter forces Polars to see every row before it can evaluate what was
supposed to be a per-row test, which blocks the pushdown this whole plugin exists to make
work -- and that hole is closed for BOTH conjunct kinds now, not just value conjuncts (see
`classify_where`).

So this module's job is narrower and stricter than "parse a SQL string": reject, at parse
time and before any file is opened, anything that is not an elementwise map, a filter
test, or an associative reduction over an explicit group key -- the operators closed
under streaming. Rejection is explicit (naming the offending token) rather than a
best-effort sanitizer, because a sanitizer that misses one shape is worse than a
whitelist that is honest about what it doesn't yet support.

Every function here takes and returns plain values (strings, sets, `polars.Expr`) and
raises `ExpressionError`, an ordinary `ValueError` subclass -- never `HTTPException` and
never a FastAPI import. `op_error_boundary` (`analysis/base.py`) is the only place that
translates an `ExpressionError` into a 400 response; that split is what lets this module be
reused outside the API later (the `read_data()` entry point in `cpra/mp/data/__init__.py`).
"""

import re
from enum import Enum
from typing import NamedTuple

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

# A subquery keyword inside a DIMENSION conjunct (see `classify_where`/
# `_check_no_dimension_subquery`). `pl.sql_expr` happens to reject a subquery that
# names a FROM relation ("... in (select max(x) from self)": nothing is registered
# under the name `self`), but NOT one that doesn't ("... in (select 1)" parses just
# fine) -- so a plain parse failure can't be relied on to close this hole; the keyword
# itself has to be checked for explicitly, the same way an aggregate call is.
_SELECT_KEYWORD_RE = re.compile(r"\bselect\b", re.IGNORECASE)

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
    "aggregate is fine here but not there" exception inside one predicate. Shared between
    `parse_value_expr` (a value expression) and `_check_no_dimension_subquery_or_aggregate`
    (a dimension `where` conjunct) -- the error message below is deliberately worded for
    either caller rather than naming just one.
    """
    for match in _IDENTIFIER_CALL_RE.finditer(expr_str):
        name = match.group(1).lower()
        if name in AGGREGATE_FUNCTION_NAMES:
            raise ExpressionError(
                f"aggregate function '{name}' is not allowed inside this expression -- "
                "it forces a full pass before the predicate can be evaluated per row"
            )


def _check_allowed_calls(expr_str):
    for match in _IDENTIFIER_CALL_RE.finditer(expr_str):
        name = match.group(1).lower()
        if name in AGGREGATE_FUNCTION_NAMES:
            continue  # reported by _check_no_aggregate_calls with a more specific message
        if name not in ALLOWED_FUNCTIONS:
            raise ExpressionError(f"function '{name}' is not in the allowed whitelist {sorted(ALLOWED_FUNCTIONS)}")


def _check_no_dimension_subquery_or_aggregate(conjunct):
    """The DIMENSION-conjunct counterpart of `_check_no_aggregate_calls` --  applied by
    `classify_where`, not `parse_value_expr`, since a dimension conjunct never reaches
    that function at all (it goes to `lf.sql()` unrestricted; see the module
    docstring). Without this check, a dimension conjunct is otherwise unrestricted, so
    nothing else stands between it and the same 94x/26.9GB pushdown-defeating
    materialization the value-expression whitelist exists to prevent -- an aggregate
    call (`calendar_year = max(calendar_year)`) or a subquery
    (`calendar_year in (select ...)`) inside a per-row filter both force a full pass
    before the comparison can be evaluated, exactly like the measured value-side
    example in the module docstring.
    """
    _check_no_aggregate_calls(conjunct)
    if _SELECT_KEYWORD_RE.search(conjunct):
        raise ExpressionError(
            f"subquery ('select') is not allowed in a where conjunct: {conjunct!r} -- it "
            "forces a full pass before the filter can be evaluated per row, the same "
            "pushdown-defeating shape an aggregate inside the filter would"
        )


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


def split_top_level_and(where_str):
    """Split a `where` string into its top-level AND-ed conjuncts.

    A plain character scanner over the raw string -- not a SQL parser, not Polars'
    serialized expression tree, and not a naive `re.split(r"(?i)\\band\\b", ...)` (which
    would also split inside a quoted string or a parenthesised subexpression). It walks
    `where_str` once, tracking single-quote state and paren depth, and only treats a
    case-insensitive, word-boundary "and" as a split point when both are at their
    baseline (outside any quote, at paren depth 0). Anything nested inside `OR`, `NOT`,
    parentheses, or a quoted literal stays part of the conjunct it's nested in and is
    handed whole to `pl.sql_expr` by `classify_where` -- this function only ever finds
    the TOP-level seam, which is the one `where`'s dimension/value split is defined
    against (see `classify_where`).

    Returns a list of conjunct strings (whitespace-stripped, not further validated --
    an empty or unparseable conjunct is `classify_where`'s problem, not this function's,
    since a would-be-reusable splitter shouldn't also own SQL semantics).
    """
    conjuncts = []
    depth = 0
    in_quote = False
    start = 0
    i = 0
    n = len(where_str)
    while i < n:
        ch = where_str[i]
        if in_quote:
            if ch == "'":
                in_quote = False
            i += 1
            continue
        if ch == "'":
            in_quote = True
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth -= 1
            i += 1
            continue
        if depth == 0 and where_str[i : i + 3].lower() == "and":
            before = where_str[i - 1] if i > 0 else " "
            after = where_str[i + 3] if i + 3 < n else " "
            is_word_char = lambda c: c.isalnum() or c == "_"  # noqa: E731
            if not is_word_char(before) and not is_word_char(after):
                conjuncts.append(where_str[start:i])
                i += 3
                start = i
                continue
        i += 1
    conjuncts.append(where_str[start:])
    return [c.strip() for c in conjuncts]


def classify_where(where_str):
    """Split `where` into top-level AND conjuncts (`split_top_level_and`) and sort each
    into a DIMENSION conjunct (references no `value` column) or a VALUE conjunct
    (references `value`, and nothing outside `value` plus literals/whitelisted
    functions) -- the seam this module's two existing surfaces are unified across: a
    dimension conjunct is recombined by the caller (`base.apply_dimension_filter`) into
    the existing unrestricted `lf.sql()` path, and a value conjunct is recombined and
    handed to `parse_value_expr`'s restricted whitelist, exactly the two mechanisms that
    already existed for `where` and the former `value_where` respectively.

    Classification is by NAME (`pl.sql_expr(conjunct).meta.root_names()`), not by
    guessing from the string: a conjunct is parsed once, here, purely to answer "which
    columns does this reference" -- the parsed `Expr` itself is discarded, since the
    caller needs the original conjunct STRING back (to recombine dimension conjuncts
    with `AND` for `lf.sql()`, or to hand a value conjunct to `parse_value_expr`, which
    re-parses it under its own restricted whitelist rather than trusting this parse).

    A conjunct referencing BOTH `value` and a dimension column is rejected outright --
    e.g. `value > calendar_year` -- with an error naming the conjunct and explaining
    the rule: dimension and value conditions must be separate AND-ed conjuncts (like
    `calendar_year > 2040 AND value > 3`), never combined inside one comparison, since
    there is no ambiguity-free way to classify a single conjunct that references both.

    A conjunct `pl.sql_expr` cannot parse at all raises `ExpressionError` here -- the
    same rejection an eventual `lf.sql()` or `parse_value_expr` failure would give for
    the same malformed input, just surfaced at the earliest possible point (before any
    file is opened) rather than once collection actually reaches that conjunct.

    A dimension conjunct is additionally checked for an aggregate call or a subquery
    (`_check_no_dimension_subquery_or_aggregate`) -- unlike `where` historically, a
    dimension conjunct is no longer assumed safe by construction just because it
    doesn't touch `value`; see that function and the module docstring for why.

    Returns `(dimension_conjuncts, value_conjuncts)`, two lists of ORIGINAL conjunct
    strings in the order they appeared in `where_str`.
    """
    dimension_conjuncts = []
    value_conjuncts = []
    for conjunct in split_top_level_and(where_str):
        if not conjunct:
            raise ExpressionError(f"empty conjunct in where={where_str!r}")
        try:
            expr = pl.sql_expr(conjunct)
        except Exception as exc:  # pl.sql_expr's own exception types are not part of its stable API
            raise ExpressionError(f"invalid where conjunct {conjunct!r}: {exc}") from exc

        referenced = set(expr.meta.root_names())
        references_value = "value" in referenced
        references_dimension = bool(referenced - {"value"})
        if references_value and references_dimension:
            raise ExpressionError(
                f"where conjunct {conjunct!r} references both a dimension column and "
                "`value` -- a single AND-term can only be one kind; write dimension and "
                "value conditions as separate AND-ed conjuncts (e.g. "
                "`calendar_year > 2040 AND value > 3`), not combined in one comparison"
            )

        if references_value:
            value_conjuncts.append(conjunct)
        else:
            _check_no_dimension_subquery_or_aggregate(conjunct)
            dimension_conjuncts.append(conjunct)

    return dimension_conjuncts, value_conjuncts


# Tokens that make a string a CONDITION rather than a value-producing expression: the SQL
# comparison operators and boolean/conditional keywords. A calculate `expression` computes
# the new value (arithmetic over `value`, numeric literals, and `ALLOWED_FUNCTIONS`);
# selecting WHICH rows/bands/pixels it applies to is its sibling `where`'s job, so any of
# these inside an `expression` is rejected by `parse_calculate_expression` with an error
# pointing at `where`. (`cast(value as float)`'s `as` is deliberately not in this list --
# `cast` is a whitelisted elementwise function, not a condition.)
_CONDITION_TOKEN_RE = re.compile(
    r"(<=|>=|!=|<>|<|>|=|!|\b(?:and|or|not|in|like|ilike|between|is|case|when|then|else|end)\b)",
    re.IGNORECASE,
)

# Everything a calculate `expression` may be spelled with, at the character level:
# identifiers (the `value` placeholder and whitelisted function names), numeric literals
# (digits, `.`, and the `e`/`E` an exponent uses are all identifier/digit/dot characters),
# the four arithmetic operators, parentheses, a comma (function arguments), and
# whitespace. Anything else -- `%`, `|`, `&`, `^`, a quoted string literal, ... -- is not
# in the documented grammar (`+ - * /`, parentheses, numeric literals, whitelisted
# functions) and is rejected by character before the parser ever sees it, so the
# whitelist stays honest about unlisted operators instead of admitting whatever
# `pl.sql_expr` happens to parse.
_CALCULATE_ALLOWED_CHARS_RE = re.compile(r"[^A-Za-z0-9_+\-*/(),.\s]")


def parse_calculate_expression(expr_str, allowed_columns=("value",)):
    """Parse a calculate `expression`: arithmetic (`+ - * /`, parentheses) over the
    placeholder column `value` and numeric literals, plus the same elementwise function
    whitelist (`ALLOWED_FUNCTIONS`) a `where` value condition may call -- built on
    `parse_value_expr`'s own validation machinery (streaming-closed check, aggregate-call
    rejection, function whitelist, parse, column-namespace check) rather than a second
    parser, with two additional rejections of its own:

    - a comparison/boolean token (`_CONDITION_TOKEN_RE`): an expression COMPUTES the new
      value; row/band selection is the entry's `where`'s job, and silently letting a
      boolean expression through would rewrite matched values to 0/1;
    - a subquery keyword, closing the same hole `_check_no_dimension_subquery_or_aggregate`
      closes for dimension conjuncts (`pl.sql_expr` does not reliably reject every
      subquery spelling on its own -- see `_SELECT_KEYWORD_RE`).

    `allowed_columns` defaults to the `value` placeholder alone -- no dimension columns
    inside an expression, per the calculate stage's documented grammar. `None` skips the
    namespace check entirely, for a caller (the pipeline layer's structural validation)
    that cannot know the in-scope column name yet and re-validates at compile time.
    """
    if not expr_str or not expr_str.strip():
        raise ExpressionError("empty calculate expression")

    match = _CONDITION_TOKEN_RE.search(expr_str)
    if match:
        raise ExpressionError(
            f"operator/keyword {match.group(0)!r} is not allowed in a calculate expression -- "
            "an expression computes the new value (arithmetic over `value`, numeric "
            f"literals, and the functions {sorted(ALLOWED_FUNCTIONS)}); selecting which "
            "rows/bands it applies to belongs in the entry's `where`"
        )
    stray = _CALCULATE_ALLOWED_CHARS_RE.search(expr_str)
    if stray:
        raise ExpressionError(
            f"character {stray.group(0)!r} is not allowed in a calculate expression -- "
            "only arithmetic (`+ - * /`, parentheses), numeric literals, `value`, and "
            f"the functions {sorted(ALLOWED_FUNCTIONS)} are"
        )
    if _SELECT_KEYWORD_RE.search(expr_str):
        raise ExpressionError(
            f"subquery ('select') is not allowed in a calculate expression: {expr_str!r}"
        )

    _check_streaming_closed(expr_str)
    _check_no_aggregate_calls(expr_str)
    _check_allowed_calls(expr_str)

    try:
        expr = pl.sql_expr(expr_str)
    except Exception as exc:  # pl.sql_expr's own exception types are not part of its stable API
        raise ExpressionError(f"invalid calculate expression {expr_str!r}: {exc}") from exc

    if allowed_columns is not None:
        try:
            _check_column_namespace(expr, allowed_columns)
        except ExpressionError as exc:
            raise ExpressionError(
                f"{exc} -- a calculate expression may only reference the value placeholder "
                "and numeric literals; dimension selection belongs in the entry's `where`"
            ) from exc
    return expr


class CalculatePlan(NamedTuple):
    """One validated calculate entry, split the way the two grid types consume it.

    `label` names the entry in errors (`calculate.1`, `calculate.2`, ...). `expression`
    is the validated arithmetic string,
    still written against the `value` placeholder -- kept as a STRING (not the parsed
    `Expr`) because the raster paths re-parse it once per band with `value` rewritten to
    that band's own synthetic column (`rewrite_value_column`), exactly like a `where`
    value condition. `dimension_where`/`value_where` are the entry's `where` split by
    `classify_where` into its two conjunct kinds and recombined (`None` when that kind
    was absent): the dimension half selects rows (vector) or bands (raster metadata); the
    value half is a per-cell/per-pixel elementwise mask.
    """

    label: str
    expression: str
    dimension_where: str | None
    value_where: str | None


def compile_calculate_entries(entries, allowed_columns=("value",)):
    """Validate calculate entries -- `(label, expression, where)` triples, in application
    order -- into `CalculatePlan`s, before any file is opened.

    `expression` goes through `parse_calculate_expression` against `allowed_columns`.
    `where` (optional) gets exactly a `FilterOp`'s treatment: split and
    classified by `classify_where` (so a conjunct mixing a dimension column and `value`,
    an aggregate call, or a subquery is rejected with the same messages `where` gives),
    dimension conjuncts recombined into `dimension_where` -- their column NAMES are
    grid-dependent and are checked where the plan is applied -- and value conjuncts
    recombined into `value_where` and validated here against the same namespace.

    `allowed_columns` defaults to the `value` placeholder alone, which is what every
    single-value-column caller wants and what this function accepted before the parameter
    existed. It is a parameter because a frame does not always carry its measurement in a
    column called `value`: a `join`'s output names each operand's measurement after the
    operand (`a`, `b`, ...), and the whole point of the `calculate` that follows is to
    compute one `value` from them, so that caller passes those names instead. Both uses
    below take the same namespace, since `expression` and `where`'s value conjuncts read
    the same columns of the same frame.

    Every rejection is an `ExpressionError` prefixed with the offending entry's label.
    """
    plans = []
    for label, expression, where in entries:
        try:
            parse_calculate_expression(expression, allowed_columns)
        except ExpressionError as exc:
            raise ExpressionError(f"{label}: {exc}") from exc

        dimension_where = None
        value_where = None
        if where is not None:
            try:
                dimension_conjuncts, value_conjuncts = classify_where(where)
            except ExpressionError as exc:
                raise ExpressionError(f"{label}: invalid where {where!r}: {exc}") from exc
            if dimension_conjuncts:
                dimension_where = " and ".join(f"({c})" for c in dimension_conjuncts)
            if value_conjuncts:
                value_where = " and ".join(f"({c})" for c in value_conjuncts)
                try:
                    parse_value_expr(value_where, allowed_columns)
                except ExpressionError as exc:
                    raise ExpressionError(f"{label}: invalid where {where!r}: {exc}") from exc

        plans.append(CalculatePlan(label, expression, dimension_where, value_where))
    return plans


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


def rewrite_named_columns(expr_str, column_map):
    """Generalization of `rewrite_value_column` to more than one placeholder, rewritten
    simultaneously rather than one at a time.

    `pipeline.py`'s analysis pipelines don't write their `filter.where` against the
    single `value` placeholder -- a step downstream of a spatial `join` (`paired
    ._write_paired_notebook_raster`) names each side by its own input/step name (e.g. `marsh > 0.5 AND sal >
    5.5`), and that same expression has to be re-evaluated once per selected calendar band,
    against that band's own synthetic columns (`scan_rasters`' `b<i>`/`p<i>` convention).
    `column_map` is `{name: replacement}` for every name in scope at that step, e.g.
    `{"marsh": "b0", "sal": "p0"}`.

    Built as ONE regex with alternation over every key, substituted in a single pass --
    not `column_map`'s items applied one at a time via repeated `rewrite_value_column`-
    style calls -- so that a replacement string which happens to collide with another
    key (or another replacement) is never re-substituted. Longer names are tried first
    in the alternation so a shorter key that is a prefix of a longer one (unlikely given
    the word-boundary anchor, but not ruled out) can't shadow it.

    Like `rewrite_value_column`, this performs no validation of its own -- the caller is
    expected to re-parse and re-whitelist the result with `parse_value_expr` immediately
    after, against the replacement columns' own namespace.
    """
    if not column_map:
        return expr_str
    names = sorted(column_map, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(name) for name in names) + r")\b")
    return pattern.sub(lambda m: column_map[m.group(1)], expr_str)


def value_predicate_mask(expr_str, allowed_columns):
    """Parse a value predicate and return it as a mask with `null` coerced to `False`.

    The shape every "value_where" call site in `base.py`/`sources.py` wants: a boolean
    expression suitable for `pl.when(mask).then(pl.col(...)).otherwise(None)`, which nulls
    out a non-matching value rather than dropping its row. That is what lets a masked-out
    value fall into every aggregate's EXISTING null-skipping behaviour
    (`guarded_aggregate`'s `count()`-based denominator, `POLARS_RASTER_AGGREGATES`, the
    vector `getattr(pl.col("value"), aggregate)()` call sites) exactly like a nodata pixel
    does, with no separate "predicate failed" code path anywhere downstream.

    `fill_null(False)`: a predicate over an already-null (nodata) value evaluates to
    null, not `False` (`null > x` is `null`, matching SQL three-valued logic) -- coercing
    it here means a nodata value is masked out for the same reason a real value that
    fails the predicate is, rather than surviving the `pl.when` by accident because
    neither branch condition technically matched `False`.
    """
    return parse_value_expr(expr_str, allowed_columns).fill_null(False)


class PredicateKind(str, Enum):
    """The three kinds a pipeline `filter`/`calculate.where` conjunct can be, by which
    reserved column it references: `value` (the measured quantity), `grid_cell` (a
    spatial predicate -- the one kind `classify_where` above does not distinguish, since
    it predates the pipeline request layer and only ever needed the dimension/value
    split), or neither (an ordinary dimension conjunct). Which kind a conjunct is decides
    its lowering: a dimension conjunct is a column selection in band-wide shape and a row
    filter in long shape either way; a value conjunct is a row filter in long shape and
    always a per-cell mask in band-wide shape (never a row drop, since dropping a pixel's
    row would drop it from every band); a spatial conjunct is a row filter in long shape,
    and a point sample in band-wide shape -- the one case that changes the frame's shape,
    from band-wide to long, rather than preserving it.
    """

    value = "value"
    dimension = "dimension"
    spatial = "spatial"


def classify_predicate_kind(conjunct: str) -> PredicateKind:
    """Classify one top-level AND conjunct (as `split_top_level_and` produces) into a
    `PredicateKind`, by whether it references the reserved `grid_cell` column, the
    reserved `value` column, or neither. A conjunct naming both is rejected outright, the
    same rule `classify_where` already applies to a conjunct naming both `value` and a
    dimension: there is no lowering for a single comparison that tests two kinds at once.
    """
    try:
        expr = pl.sql_expr(conjunct)
    except Exception as exc:  # pl.sql_expr's own exception types are not part of its stable API
        raise ExpressionError(f"invalid where conjunct {conjunct!r}: {exc}") from exc

    referenced = set(expr.meta.root_names())
    is_spatial = "grid_cell" in referenced
    is_value = "value" in referenced
    if is_spatial and is_value:
        raise ExpressionError(
            f"where conjunct {conjunct!r} references both `grid_cell` and `value` -- a "
            "single AND-term can only be one predicate kind; write them as separate "
            "AND-ed conjuncts"
        )
    if is_spatial:
        return PredicateKind.spatial
    if is_value:
        return PredicateKind.value
    return PredicateKind.dimension


def classify_where_kinds(where_str: str) -> dict[PredicateKind, list[str]]:
    """The three-way counterpart of `classify_where`: split `where_str` into its
    top-level AND conjuncts (`split_top_level_and`) and sort each into one of the three
    `PredicateKind`s (`classify_predicate_kind`), rather than the dimension/value split
    `classify_where` performs -- added alongside it, for the pipeline request layer's own
    shape-typing needs, without changing `classify_where` or anything that already calls
    it.

    Every dimension and spatial conjunct additionally gets the same structural safety
    check a dimension conjunct already gets from `classify_where`
    (`_check_no_dimension_subquery_or_aggregate`): neither is a value expression bound
    for `parse_value_expr`'s own whitelist, so without this check an aggregate call or a
    subquery inside either one would otherwise force the same pushdown-defeating full
    pass the module docstring describes. A value conjunct gets no additional check here,
    matching `classify_where`'s own behavior -- its caller re-parses and whitelists it
    with `parse_value_expr` when it actually needs to.

    Returns a dict with all three `PredicateKind` members always present (even when a
    kind's conjunct list is empty), each mapped to a list of the original conjunct
    strings in the order they appeared.
    """
    result: dict[PredicateKind, list[str]] = {kind: [] for kind in PredicateKind}
    for conjunct in split_top_level_and(where_str):
        if not conjunct:
            raise ExpressionError(f"empty conjunct in where={where_str!r}")
        kind = classify_predicate_kind(conjunct)
        if kind is not PredicateKind.value:
            _check_no_dimension_subquery_or_aggregate(conjunct)
        result[kind].append(conjunct)
    return result


def count_by_value_frame(lf, group_cols, value_col, max_distinct):
    """Categorical count-by-value: one row per (group, distinct value) in `value_col`,
    with the count of rows sharing that value -- a fundamentally different OUTPUT SHAPE
    from every other aggregate in this package (which collapse a group to one row), so it
    is never reachable through `group_aggregate` (see the `AggregateFunction.count_by_value`
    comment in types.py and `stages.reject_count_by_value_group`'s rejection); only
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
