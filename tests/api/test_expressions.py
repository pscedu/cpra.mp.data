"""Tests for the restricted value-expression surface (api/expressions.py).

Every function under test takes and returns plain values and raises `ExpressionError`
(an ordinary `ValueError`), never an `HTTPException` -- these tests exercise the library
layer directly, with no `Dataset`/FastAPI involved, matching the module's own "no
FastAPI below the translation layer" design constraint.
"""

import polars as pl
import pytest

from cpra.mp.data.api.expressions import (
    ExpressionError,
    count_by_value_frame,
    parse_value_expr,
    rewrite_value_column,
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
