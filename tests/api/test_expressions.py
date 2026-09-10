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
