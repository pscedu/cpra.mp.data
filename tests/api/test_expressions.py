"""Tests for the restricted value-expression surface (api/expressions.py).

Every function under test takes and returns plain values and raises `ExpressionError`
(an ordinary `ValueError`), never an `HTTPException` -- these tests exercise the library
layer directly, with no `Dataset`/FastAPI involved, matching the module's own "no
FastAPI below the translation layer" design constraint.
"""

import polars as pl
import pytest

from cpra.mp.data.analysis.request.expressions import (
    ExpressionError,
    PredicateKind,
    classify_predicate_kind,
    classify_where,
    classify_where_kinds,
    combine_conjuncts,
    compile_calculate_entries,
    parse_calculate_expression,
    parse_value_expr,
    rewrite_value_column,
    rewrite_value_columns,
    split_top_level_and,
    value_predicate_mask,
)


class TestParseValueExpr:
    def test_simple_comparison_parses(self):
        expr = parse_value_expr("value > 3", ["value"])
        df = pl.DataFrame({"value": [1, 5]})
        assert df.select(expr.alias("out"))["out"].to_list() == [False, True]

    def test_compound_and_parses(self):
        expr = parse_value_expr("value > 3 and value < 10", ["value"])
        df = pl.DataFrame({"value": [1, 5, 20]})
        assert df.select(expr.alias("out"))["out"].to_list() == [False, True, False]

    def test_allowed_function_call_parses(self):
        expr = parse_value_expr("abs(value) > 3", ["value"])
        df = pl.DataFrame({"value": [-5, 1]})
        assert df.select(expr.alias("out"))["out"].to_list() == [True, False]

    @pytest.mark.parametrize("word", ["and", "AND", "or", "OR"])
    def test_a_parenthesized_conjunct_is_not_read_as_a_function_call(self, word):
        """`(value > 1) and (value < 5)` puts a bare word immediately before a `(`, which
        is exactly the shape the function scan looks for -- so an ordinary parenthesized
        conjunction used to be rejected as a call to a nonexistent function named `and`.
        Note this is not a paired-frame concern: `where: "value > 1 AND value < 5"` on a
        single raster input has the same shape once `rewrite_value_column` has run.
        """
        expr = parse_value_expr(f"(value > 1) {word} (value < 5)", ["value"])
        df = pl.DataFrame({"value": [0, 3, 9]})
        expected = [False, True, False] if word.lower() == "and" else [True, True, True]
        assert df.select(expr.alias("out"))["out"].to_list() == expected

    def test_a_subquery_after_in_is_still_rejected(self):
        """The logical-operator exemption above is deliberately just `and`/`or`. Every
        other SQL word that can precede a `(` -- `in`, `not`, `exists` -- can precede a
        SUBQUERY, and the function scan is what currently stops one reaching the planner
        through a value conjunct.
        """
        with pytest.raises(ExpressionError):
            parse_value_expr("value in (select 1)", ["value"])

    def test_unknown_column_is_rejected(self):
        with pytest.raises(ExpressionError, match="not_a_column"):
            parse_value_expr("not_a_column > 3", ["value"])

    def test_column_outside_the_given_namespace_is_rejected(self):
        """`grid_cell` is a real column elsewhere, but not admitted unless the caller's
        own namespace says so -- the check is against the given allow-list, not against
        Polars' own knowledge of what columns exist somewhere in the schema."""
        with pytest.raises(ExpressionError, match="grid_cell"):
            parse_value_expr("grid_cell > 3", ["value"])

    def test_disallowed_function_is_rejected_by_name(self):
        with pytest.raises(ExpressionError, match="upper"):
            parse_value_expr("upper(value) = 'X'", ["value"])

    @pytest.mark.parametrize("fn", ["min", "max", "sum", "avg", "count", "median"])
    def test_aggregate_functions_are_rejected_by_name(self, fn):
        with pytest.raises(ExpressionError, match=fn):
            parse_value_expr(f"value = {fn}(value)", ["value"])

    def test_aggregate_rejection_runs_before_parsing(self):
        """The aggregate check runs on the raw string, so it never needs `pl.sql_expr` to
        succeed -- an aggregate expression is rejected even if it would otherwise be
        syntactically invalid for another reason too."""
        with pytest.raises(ExpressionError, match="min"):
            parse_value_expr("value = min(", ["value"])

    def test_reject_aggregates_false_allows_aggregate_calls(self):
        """Opt-out exists for a future caller building an expression to run inside its own
        group_by().agg(...) -- not used by any current call site in this package."""
        expr = parse_value_expr("min(value)", ["value"], reject_aggregates=False)
        df = pl.DataFrame({"value": [3, 1, 2]})
        assert df.select(expr.alias("out"))["out"].item() == 1

    @pytest.mark.parametrize(
        "expr_str",
        [
            "value order by value",
            "value in (select value from other group by value)",
            "value over (partition by value)",
        ],
    )
    def test_non_streaming_constructs_are_rejected(self, expr_str):
        with pytest.raises(ExpressionError):
            parse_value_expr(expr_str, ["value"])

    def test_empty_expression_is_rejected(self):
        with pytest.raises(ExpressionError, match="empty"):
            parse_value_expr("", ["value"])

    def test_invalid_syntax_is_reported_as_an_expression_error(self):
        with pytest.raises(ExpressionError):
            parse_value_expr("value >>> 3", ["value"])


class TestParseCalculateExpression:
    def test_arithmetic_over_value_parses_and_evaluates(self):
        expr = parse_calculate_expression("(value + 1) * 0.5")
        df = pl.DataFrame({"value": [1.0, 3.0]})
        assert df.select(expr.alias("out"))["out"].to_list() == [1.0, 2.0]

    def test_whitelisted_function_accepted(self):
        expr = parse_calculate_expression("abs(value) * -1")
        df = pl.DataFrame({"value": [-2.0, 2.0]})
        assert df.select(expr.alias("out"))["out"].to_list() == [-2.0, -2.0]

    @pytest.mark.parametrize(
        "bad",
        [
            "value > 3",  # comparison: a condition, not an expression
            "value = 3",
            "value * 2 AND value",  # boolean keyword
            "case when value > 1 then 2 else 3 end",  # conditional
            "value % 3",  # operator outside the + - * / whitelist
            "value * 'x'",  # string literal (quote character outside the whitelist)
        ],
    )
    def test_condition_and_non_whitelisted_operators_rejected(self, bad):
        with pytest.raises(ExpressionError):
            parse_calculate_expression(bad)

    def test_aggregate_call_rejected(self):
        with pytest.raises(ExpressionError, match="aggregate"):
            parse_calculate_expression("value - min(value)")

    def test_non_whitelisted_function_rejected(self):
        with pytest.raises(ExpressionError, match="whitelist"):
            parse_calculate_expression("exp(value)")

    def test_dimension_column_rejected(self):
        with pytest.raises(ExpressionError, match="scenario_id"):
            parse_calculate_expression("value * scenario_id")

    def test_empty_expression_rejected(self):
        with pytest.raises(ExpressionError, match="empty"):
            parse_calculate_expression("   ")

    def test_allowed_columns_none_skips_the_namespace_check(self):
        """The pipeline layer validates structure before it can know the in-scope column
        name -- the namespace check alone is deferred, everything else still runs."""
        parse_calculate_expression("marsh * 0.6", allowed_columns=None)
        with pytest.raises(ExpressionError):
            parse_calculate_expression("max(marsh)", allowed_columns=None)


class TestCompileCalculateEntries:
    def test_splits_where_into_dimension_and_value_halves(self):
        (plan,) = compile_calculate_entries(
            [("calculate.1", "value * 0.6", "model_group_id = 500 AND value > 2")]
        )
        assert plan.label == "calculate.1"
        assert plan.expression == "value * 0.6"
        assert plan.dimension_where == "(model_group_id = 500)"
        assert plan.value_where == "(value > 2)"

    def test_no_where_leaves_both_halves_none(self):
        (plan,) = compile_calculate_entries([("calculate.1", "value * 0.6", None)])
        assert plan.dimension_where is None
        assert plan.value_where is None

    def test_rejections_name_the_entry_label(self):
        with pytest.raises(ExpressionError, match="negate"):
            compile_calculate_entries([("negate", "value >< 1", None)])
        with pytest.raises(ExpressionError, match="calculate.2"):
            compile_calculate_entries(
                [
                    ("calculate.1", "value * 2", None),
                    ("calculate.2", "value * 2", "value > max(value)"),
                ]
            )

    def test_mixed_conjunct_in_where_rejected(self):
        with pytest.raises(ExpressionError, match="both a dimension column"):
            compile_calculate_entries([("calculate.1", "value * 2", "value > calendar_year")])


class TestRewriteValueColumn:
    def test_replaces_the_placeholder(self):
        assert rewrite_value_column("value > 3", "b0") == "b0 > 3"

    def test_replaces_every_occurrence(self):
        assert rewrite_value_column("value > 3 and value < 10", "b0") == "b0 > 3 and b0 < 10"

    def test_respects_word_boundaries(self):
        """An identifier that merely contains "value" must not be partially rewritten."""
        assert rewrite_value_column("multivalue > 3", "b0") == "multivalue > 3"

    def test_rewritten_string_is_reparsed_and_rewhitelisted(self):
        """rewrite_value_column performs no validation on its own -- an aggregate smuggled
        in under a rewritten name is still caught the moment the result is parsed."""
        rewritten = rewrite_value_column("value = min(value)", "b0")
        with pytest.raises(ExpressionError, match="min"):
            parse_value_expr(rewritten, ["b0"])


class TestRewriteValueColumns:
    def test_rewrites_every_named_column_in_one_pass(self):
        assert (
            rewrite_value_columns("marsh > 0.5 AND sal > 5.5", {"marsh": "b0", "sal": "b2"})
            == "b0 > 0.5 AND b2 > 5.5"
        )

    def test_one_pass_not_sequential(self):
        """Replacements are band names, so a sequential rewrite could feed one
        substitution's output into the next one's pattern -- rewriting `b1` to `b0` and
        then `b0` to `b1` would leave both operands naming the same band, which is a
        silently wrong answer rather than an error. One pass cannot do that.
        """
        assert rewrite_value_columns("b1 > 1 AND b0 > 2", {"b1": "b0", "b0": "b1"}) == "b0 > 1 AND b1 > 2"

    def test_respects_word_boundaries(self):
        assert rewrite_value_columns("marshland > 3", {"marsh": "b0"}) == "marshland > 3"

    def test_empty_mapping_is_the_identity(self):
        assert rewrite_value_columns("value > 3", {}) == "value > 3"


class TestValuePredicateMask:
    def test_true_and_false_pass_through(self):
        mask = value_predicate_mask("value > 3", ["value"])
        df = pl.DataFrame({"value": [1, 5]})
        assert df.select(mask.alias("m"))["m"].to_list() == [False, True]

    def test_null_input_becomes_false_not_null(self):
        """A predicate over an already-null value is null under SQL three-valued logic
        (`null > x` is `null`), not `False` -- value_predicate_mask coerces it so a
        nodata pixel is masked out for the same reason a failing real value is, rather
        than slipping through a `pl.when` that matches neither branch."""
        mask = value_predicate_mask("value > 3", ["value"])
        df = pl.DataFrame({"value": [None, 1, 5]}, schema={"value": pl.Float64})
        assert df.select(mask.alias("m"))["m"].to_list() == [False, False, True]


class TestSplitTopLevelAnd:
    def test_single_conjunct_with_no_and(self):
        assert split_top_level_and("calendar_year > 2040") == ["calendar_year > 2040"]

    def test_splits_two_conjuncts(self):
        assert split_top_level_and("calendar_year > 2040 and value > 3") == [
            "calendar_year > 2040",
            "value > 3",
        ]

    def test_splits_three_conjuncts(self):
        assert split_top_level_and("a > 1 and b > 2 and c > 3") == ["a > 1", "b > 2", "c > 3"]

    @pytest.mark.parametrize("connector", ["AND", "And", "aNd"])
    def test_case_insensitive(self, connector):
        assert split_top_level_and(f"a > 1 {connector} b > 2") == ["a > 1", "b > 2"]

    def test_does_not_split_inside_parens(self):
        """A nested AND inside a parenthesized group is not top-level -- it belongs to
        whatever function call or grouped expression contains it, not to the outer
        conjunct list."""
        assert split_top_level_and("f(a, b) > 1 and g > 2") == ["f(a, b) > 1", "g > 2"]

    def test_does_not_split_a_between_and(self):
        """`BETWEEN x AND y` uses the word AND as part of its own syntax, not as a
        top-level conjunct separator -- this splitter has no BETWEEN-awareness, so
        callers relying on BETWEEN would see it split in two; documented here as the
        behavior actually produced (not a bug the design asked to fix), matching the
        design's "library-first, not a full SQL grammar" scope.
        """
        assert split_top_level_and("value between 1 and 10") == ["value between 1", "10"]

    def test_does_not_split_inside_single_quoted_strings(self):
        assert split_top_level_and("name = 'foo and bar' and value > 1") == [
            "name = 'foo and bar'",
            "value > 1",
        ]

    def test_does_not_split_word_containing_and_substring(self):
        """`andes` and `sandy` both contain the letters "and" but not as a standalone
        word -- the word-boundary check must not treat them as the AND connector."""
        assert split_top_level_and("andes > 1 and sandy < 2") == ["andes > 1", "sandy < 2"]

    def test_or_only_clause_is_a_single_conjunct(self):
        assert split_top_level_and("a > 1 or b > 2") == ["a > 1 or b > 2"]

    def test_not_clause_is_a_single_conjunct(self):
        assert split_top_level_and("not (value > 1)") == ["not (value > 1)"]

    def test_strips_surrounding_whitespace_from_each_conjunct(self):
        assert split_top_level_and("  a > 1   and   b > 2  ") == ["a > 1", "b > 2"]


class TestCombineConjuncts:
    """`combine_conjuncts` is `split_top_level_and`'s inverse, and the parenthesisation is
    the whole of it -- see that function's docstring."""

    def test_joins_with_and(self):
        assert combine_conjuncts(["calendar_year > 2040", "scenario_id = 1"]) == (
            "(calendar_year > 2040) and (scenario_id = 1)"
        )

    def test_a_single_conjunct_is_still_parenthesised(self):
        assert combine_conjuncts(["value > 1"]) == "(value > 1)"

    def test_no_conjuncts_is_the_empty_string(self):
        # Every call site guards on a non-empty list first, so this is the boundary rather
        # than a live case -- pinned so the guard's removal would be a test failure and not
        # a silently empty `where` clause.
        assert combine_conjuncts([]) == ""

    def test_a_conjuncts_own_or_cannot_rebind_across_the_seam(self):
        """The reason the parens are not optional. `a = 1 and b = 2 or b = 3` splits into
        two conjuncts, the second carrying a lower-precedence `or`; joined bare, that `or`
        re-binds across the `and` and admits a row neither conjunct does. The
        parenthesised join selects exactly what applying the conjuncts as successive
        filters selects, which is what a conjunct list means everywhere it is used.
        """
        conjuncts = split_top_level_and("a = 1 and b = 2 or b = 3")
        assert conjuncts == ["a = 1", "b = 2 or b = 3"]

        df = pl.DataFrame({"a": [1, 1, 2, 2], "b": [2, 3, 2, 3]})
        successive = df.lazy()
        for conjunct in conjuncts:
            successive = successive.sql(f"select * from self where {conjunct}")

        combined = df.lazy().sql(f"select * from self where {combine_conjuncts(conjuncts)}")
        naive = df.lazy().sql(f"select * from self where {' and '.join(conjuncts)}")

        assert combined.collect().to_dicts() == successive.collect().to_dicts()
        assert naive.collect().to_dicts() != successive.collect().to_dicts()


class TestClassifyWhere:
    def test_dimension_only_conjunct(self):
        dimension, value = classify_where("calendar_year > 2040")
        assert dimension == ["calendar_year > 2040"]
        assert value == []

    def test_value_only_conjunct(self):
        dimension, value = classify_where("value > 3")
        assert dimension == []
        assert value == ["value > 3"]

    def test_multiple_dimension_and_value_conjuncts_are_sorted_and_kept_separate(self):
        dimension, value = classify_where("calendar_year > 2040 and value > 3 and aep = 0.2")
        assert dimension == ["calendar_year > 2040", "aep = 0.2"]
        assert value == ["value > 3"]

    def test_mixed_conjunct_is_rejected(self):
        """A single AND-term referencing both `value` and a dimension column has no
        unambiguous classification -- see `classify_where`'s docstring -- and is
        rejected outright rather than guessed at."""
        with pytest.raises(ExpressionError, match="calendar_year"):
            classify_where("value > calendar_year")

    def test_unparseable_conjunct_is_rejected(self):
        with pytest.raises(ExpressionError):
            classify_where("value >>> 3")

    def test_empty_conjunct_is_rejected(self):
        with pytest.raises(ExpressionError, match="empty"):
            classify_where("calendar_year > 2040 and")

    def test_dimension_conjunct_with_aggregate_call_is_rejected(self):
        with pytest.raises(ExpressionError, match="aggregate"):
            classify_where("calendar_year = min(calendar_year)")

    def test_dimension_conjunct_with_subquery_is_rejected(self):
        with pytest.raises(ExpressionError, match="subquery"):
            classify_where("calendar_year in (select 1)")

    def test_dimension_conjunct_with_subquery_referencing_self_fails_to_parse(self):
        """`select ... from self` has no registered `self` relation at this layer, so
        `pl.sql_expr` itself raises before the explicit subquery check ever runs -- still
        an `ExpressionError`, just via a different mechanism than the bare-`select` case
        above."""
        with pytest.raises(ExpressionError):
            classify_where("calendar_year in (select max(calendar_year) from self)")

    def test_value_conjunct_with_aggregate_call_is_not_rejected_at_classification(self):
        """Aggregate rejection for VALUE conjuncts happens downstream, in
        `parse_value_expr` (see `TestParseValueExpr.test_aggregate_functions_are_rejected_by_name`)
        -- `classify_where` itself only sorts a conjunct by which namespace it
        references, it does not run the value-expression whitelist.
        """
        dimension, value = classify_where("value = min(value)")
        assert dimension == []
        assert value == ["value = min(value)"]


class TestClassifyPredicateKind:
    def test_dimension_conjunct(self):
        assert classify_predicate_kind("calendar_year > 2040") is PredicateKind.dimension

    def test_value_conjunct(self):
        assert classify_predicate_kind("value > 3") is PredicateKind.value

    def test_spatial_conjunct(self):
        assert classify_predicate_kind("grid_cell = 'abc'") is PredicateKind.spatial

    def test_conjunct_referencing_both_grid_cell_and_value_is_rejected(self):
        """A single AND-term can only be one predicate kind -- exactly the same
        ambiguity `classify_where` rejects for a value/dimension mix (see
        `TestClassifyWhere.test_mixed_conjunct_is_rejected`), extended to the third
        kind."""
        with pytest.raises(ExpressionError, match="grid_cell"):
            classify_predicate_kind("grid_cell = 'abc' and value > 3")

    def test_unparseable_conjunct_is_rejected(self):
        with pytest.raises(ExpressionError):
            classify_predicate_kind("value >>> 3")


class TestClassifyWhereKinds:
    def test_single_dimension_conjunct(self):
        kinds = classify_where_kinds("calendar_year > 2040")
        assert kinds[PredicateKind.dimension] == ["calendar_year > 2040"]
        assert kinds[PredicateKind.value] == []
        assert kinds[PredicateKind.spatial] == []

    def test_single_value_conjunct(self):
        kinds = classify_where_kinds("value > 3")
        assert kinds[PredicateKind.value] == ["value > 3"]
        assert kinds[PredicateKind.dimension] == []
        assert kinds[PredicateKind.spatial] == []

    def test_single_spatial_conjunct(self):
        kinds = classify_where_kinds("grid_cell = 'abc'")
        assert kinds[PredicateKind.spatial] == ["grid_cell = 'abc'"]
        assert kinds[PredicateKind.value] == []
        assert kinds[PredicateKind.dimension] == []

    def test_all_three_kinds_together_are_sorted_and_kept_separate(self):
        kinds = classify_where_kinds("calendar_year > 2040 and value > 3 and grid_cell = 'abc'")
        assert kinds[PredicateKind.dimension] == ["calendar_year > 2040"]
        assert kinds[PredicateKind.value] == ["value > 3"]
        assert kinds[PredicateKind.spatial] == ["grid_cell = 'abc'"]

    def test_every_predicate_kind_is_always_a_key_even_when_empty(self):
        kinds = classify_where_kinds("value > 3")
        assert set(kinds) == set(PredicateKind)

    def test_empty_conjunct_is_rejected(self):
        with pytest.raises(ExpressionError, match="empty"):
            classify_where_kinds("calendar_year > 2040 and")

    def test_spatial_conjunct_with_aggregate_call_is_rejected(self):
        """Spatial conjuncts get the same structural (no aggregate, no subquery) check
        as dimension conjuncts do in `classify_where` -- only a VALUE conjunct is exempt,
        since the value-expression whitelist itself catches aggregates downstream."""
        with pytest.raises(ExpressionError, match="aggregate"):
            classify_where_kinds("grid_cell = min(grid_cell)")

    def test_spatial_conjunct_with_subquery_is_rejected(self):
        with pytest.raises(ExpressionError, match="subquery"):
            classify_where_kinds("grid_cell in (select 1)")

    def test_value_conjunct_with_aggregate_call_is_not_rejected_at_classification(self):
        kinds = classify_where_kinds("value = min(value)")
        assert kinds[PredicateKind.value] == ["value = min(value)"]
