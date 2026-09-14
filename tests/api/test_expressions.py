"""Tests for the restricted value-expression surface (api/expressions.py).

Every function under test takes and returns plain values and raises `ExpressionError`
(an ordinary `ValueError`), never an `HTTPException` -- these tests exercise the library
layer directly, with no `Dataset`/FastAPI involved, matching the module's own "no
FastAPI below the translation layer" design constraint.
"""

import polars as pl
import pytest

from cpra.mp.data.analysis.expressions import (
    ExpressionError,
    classify_where,
    compile_calculate_entries,
    count_by_value_frame,
    parse_calculate_expression,
    parse_value_expr,
    rewrite_value_column,
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


class TestCountByValueFrame:
    def test_counts_one_row_per_group_per_distinct_value(self):
        lf = pl.DataFrame(
            {
                "zone": [1, 1, 1, 2, 2],
                "value": [10, 10, 20, 10, 10],
            }
        ).lazy()

        out = count_by_value_frame(lf, ["zone"], "value", max_distinct=1000).collect().sort(["zone", "value"])

        assert out["zone"].to_list() == [1, 1, 2]
        assert out["value"].to_list() == [10, 20, 10]
        assert out["count"].to_list() == [2, 1, 2]

    def test_nulls_are_excluded_not_a_category(self):
        """A masked value_where non-match or a masked nodata pixel arrives here as null --
        it must not appear as its own category, nor be counted into any real one."""
        lf = pl.DataFrame({"zone": [1, 1, 1], "value": [10, None, 10]}, schema={"zone": pl.Int64, "value": pl.Int64}).lazy()

        out = count_by_value_frame(lf, ["zone"], "value", max_distinct=1000).collect()

        assert out["value"].to_list() == [10]
        assert out["count"].to_list() == [2]

    def test_cardinality_ceiling_rejects_before_grouping(self):
        """A continuous column is exactly what the ceiling exists to catch -- see the
        module's `count_by_value_frame` docstring for the bound math. `approx_n_unique` is
        an estimate, not exact, but with 5000 genuinely distinct values and a ceiling of
        10 the estimate is nowhere near the boundary."""
        lf = pl.DataFrame({"zone": [1] * 5000, "value": list(range(5000))}).lazy()

        with pytest.raises(ExpressionError, match="distinct"):
            count_by_value_frame(lf, ["zone"], "value", max_distinct=10)

    def test_low_cardinality_column_is_not_rejected(self):
        lf = pl.DataFrame({"zone": [1] * 100, "value": [i % 5 for i in range(100)]}).lazy()

        out = count_by_value_frame(lf, ["zone"], "value", max_distinct=10).collect()

        assert out.height == 5
        assert out["count"].sum() == 100
