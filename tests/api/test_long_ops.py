"""Tests for `cpra.mp.data.analysis.execution.long_ops` -- the long-shape lowerings for
`filter`/`calculate`/`group`/`join`.

Every fixture here is a small in-memory `pl.LazyFrame` built directly, not read off disk:
these functions are pure Polars transforms over whatever frame they are handed (see
`long_ops.py`'s own module docstring -- none of them know about the walk, the plan, a
`base_path`, or a real dataset), so a hand-built frame with hand-computed expected values
is a more direct test than routing through a fixture file would be.
"""

import polars as pl
import pytest

from cpra.mp.data.analysis.execution.long_ops import calculate_long, filter_long, group_long, join_long
from cpra.mp.data.analysis.request.expressions import ExpressionError
from cpra.mp.data.analysis.request.ops import CalculateOp, GroupOp, JoinOp


def _frame():
    # calendar_year=2030: grid_cell 1 -> 10.0, grid_cell 2 -> 20.0
    # calendar_year=2040: grid_cell 1 -> 30.0, grid_cell 2 -> 40.0
    return pl.LazyFrame(
        {
            "calendar_year": [2030, 2030, 2040, 2040],
            "grid_cell": [1, 2, 1, 2],
            "value": [10.0, 20.0, 30.0, 40.0],
        }
    )


class TestFilterLong:
    def test_dimension_predicate_drops_non_matching_rows(self):
        df = filter_long(_frame(), "calendar_year = 2030").collect()
        assert df.height == 2
        assert set(df["value"].to_list()) == {10.0, 20.0}

    def test_value_predicate_drops_non_matching_rows(self):
        df = filter_long(_frame(), "value > 15").collect()
        assert df.height == 3
        assert set(df["value"].to_list()) == {20.0, 30.0, 40.0}

    def test_spatial_predicate_drops_non_matching_rows(self):
        # `grid_cell` is the one column `classify_predicate_kind` calls SPATIAL --
        # `FILTER_LOWERING_TABLE` maps it to `Shape.long` too, so for this shape it is a
        # row drop exactly like the other two kinds (see `filter_long`'s own docstring).
        df = filter_long(_frame(), "grid_cell = 1").collect()
        assert df.height == 2
        assert set(df["value"].to_list()) == {10.0, 30.0}

    def test_combined_dimension_and_value_predicate(self):
        df = filter_long(_frame(), "calendar_year = 2030 AND value > 15").collect()
        assert df.height == 1
        assert df["value"].item() == 20.0

    def test_filter_never_masks_it_only_drops(self):
        # Unlike band-wide, a long filter has no `mask=True` path at all -- row count must
        # shrink, never merely gain nulls.
        before = _frame().collect().height
        after = filter_long(_frame(), "value > 1000").collect().height
        assert after == 0
        assert after < before


VALUE_ONLY = frozenset({"value"})


class TestCalculateLong:
    def test_rewrites_value_everywhere_when_where_is_none(self):
        op = CalculateOp(step_no=1, input="a", expression="value * 2", output="b")
        df = calculate_long(_frame(), op, VALUE_ONLY).collect().sort(["calendar_year", "grid_cell"])
        assert df["value"].to_list() == [20.0, 40.0, 60.0, 80.0]

    def test_rewrites_only_rows_matching_where(self):
        op = CalculateOp(
            step_no=1, input="a", expression="value * 2", where="calendar_year = 2030", output="b"
        )
        df = calculate_long(_frame(), op, VALUE_ONLY).collect().sort(["calendar_year", "grid_cell"])
        # 2030 rows doubled (10->20, 20->40); 2040 rows pass through unchanged (30, 40) --
        # a non-matching row is left alone, never dropped (dropping is `filter`'s job).
        assert df["value"].to_list() == [20.0, 40.0, 30.0, 40.0]
        assert df.height == 4

    def test_null_value_is_left_null(self):
        lf = pl.LazyFrame(
            {"calendar_year": [2030, 2040], "grid_cell": [1, 1], "value": [10.0, None]}
        )
        op = CalculateOp(step_no=1, input="a", expression="value * 2", output="b")
        df = calculate_long(lf, op, VALUE_ONLY).collect().sort("calendar_year")
        assert df["value"].to_list() == [20.0, None]


class TestCalculateLongAfterJoin:
    """A `calculate` reading a joined frame, whose value columns are named after the
    join's operands rather than `value`. This is the step the single-value-column sink
    rule relies on to make a join's output writable at all, so every assertion here is
    about the collapse: the operand names parse, the result is one column called `value`,
    and the operand columns are gone.
    """

    def _joined(self):
        # The shape `join_long` produces: two value columns named after the operands, and
        # no `value` column at all.
        return pl.LazyFrame(
            {
                "calendar_year": [2030, 2040],
                "grid_cell": [1, 1],
                "a": [10.0, 30.0],
                "b": [4.0, 5.0],
            }
        )

    def test_expression_over_operand_columns_collapses_to_one_value(self):
        op = CalculateOp(step_no=2, input="j", expression="a - b", output="c")
        df = (
            calculate_long(self._joined(), op, frozenset({"a", "b"}))
            .collect()
            .sort("calendar_year")
        )
        assert df["value"].to_list() == [6.0, 25.0]
        assert set(df.columns) == {"calendar_year", "grid_cell", "value"}

    def test_operand_columns_are_dropped_even_where_the_where_does_not_match(self):
        # The result's COLUMNS must not depend on which rows matched: `value_columns` is
        # what a downstream step and the sink rule were typed against, and a frame that
        # still carried `a`/`b` for the unmatched rows would carry three value columns.
        op = CalculateOp(
            step_no=2, input="j", expression="a - b", where="calendar_year = 2030", output="c"
        )
        df = (
            calculate_long(self._joined(), op, frozenset({"a", "b"}))
            .collect()
            .sort("calendar_year")
        )
        assert set(df.columns) == {"calendar_year", "grid_cell", "value"}
        # The unmatched row keeps its place (a `calculate` never drops rows) with a null
        # `value`: there is no incoming `value` for it to pass through unchanged.
        assert df["value"].to_list() == [6.0, None]
        assert df.height == 2

    def test_null_in_either_operand_leaves_value_null(self):
        lf = pl.LazyFrame(
            {"calendar_year": [2030, 2040], "grid_cell": [1, 1], "a": [10.0, None], "b": [4.0, 5.0]}
        )
        op = CalculateOp(step_no=2, input="j", expression="a - b", output="c")
        df = calculate_long(lf, op, frozenset({"a", "b"})).collect().sort("calendar_year")
        assert df["value"].to_list() == [6.0, None]

    def test_expression_naming_a_column_outside_the_value_namespace_is_rejected(self):
        # `value` is exactly the name that is NOT available on a joined frame, which makes
        # it the sharpest check that the namespace really is the frame's value columns and
        # not a superset that happens to include the old default.
        op = CalculateOp(step_no=2, input="j", expression="value * 2", output="c")
        with pytest.raises(ExpressionError, match="not in the allowed namespace"):
            calculate_long(self._joined(), op, frozenset({"a", "b"})).collect()

    def test_dimension_column_is_still_outside_the_expression_namespace(self):
        # Widening the namespace to the frame's value columns must not widen it to every
        # column the frame has -- a dimension inside an expression is rejected here for
        # the same reason it always was.
        op = CalculateOp(step_no=2, input="j", expression="a * calendar_year", output="c")
        with pytest.raises(ExpressionError, match="not in the allowed namespace"):
            calculate_long(self._joined(), op, frozenset({"a", "b"})).collect()


class TestGroupLong:
    def test_empty_by_collapses_to_one_row_per_grid_cell(self):
        op = GroupOp(step_no=1, input="a", by=[], aggregate="sum", output="b")
        df = group_long(_frame(), op).collect().sort("grid_cell")
        # grid_cell 1: 10.0 + 30.0 = 40.0; grid_cell 2: 20.0 + 40.0 = 60.0 -- hand-computed
        # from `_frame()`'s literal values, not captured from a run.
        assert df["grid_cell"].to_list() == [1, 2]
        assert df["value"].to_list() == [40.0, 60.0]

    def test_group_by_dimension_plus_grid_cell_is_the_identity_on_unique_keys(self):
        op = GroupOp(step_no=1, input="a", by=["calendar_year"], aggregate="sum", output="b")
        df = group_long(_frame(), op).collect().sort(["calendar_year", "grid_cell"])
        # Every (calendar_year, grid_cell) pair in `_frame()` is already unique, so
        # grouping by both changes nothing about the values, only confirms the key
        # composition (`[*by, "grid_cell"]`, per `GroupOp.by`'s docstring).
        assert df["value"].to_list() == [10.0, 20.0, 30.0, 40.0]

    def test_folded_result_matches_hand_computed_values(self):
        # `group_long` always calls `apply_group_by` with `has_selective_filter=False`,
        # and its key always includes `grid_cell` -- so `should_fold_group_by` is
        # unconditionally True here (see `long_ops.group_long`'s own docstring) and this
        # exercises the `fold_group_by` chunked path, not the plain `group_by(...).agg()`
        # branch. The fold is numerically exact regardless of chunking (`fold.py`'s own
        # docstring), so the expected values are the same hand-computed sums as above --
        # the point of this test is that the FOLDED path produces them, not merely that
        # some aggregation does.
        op = GroupOp(step_no=1, input="a", by=[], aggregate="mean", output="b")
        df = group_long(_frame(), op).collect().sort("grid_cell")
        # grid_cell 1: mean(10.0, 30.0) = 20.0; grid_cell 2: mean(20.0, 40.0) = 30.0
        assert df["value"].to_list() == pytest.approx([20.0, 30.0])


class TestJoinLong:
    def _pair(self):
        a = pl.LazyFrame(
            {
                "calendar_year": [2030, 2040],
                "grid_cell": [1, 1],
                "value": [5.0, 6.0],
            }
        )
        b = pl.LazyFrame(
            {
                "calendar_year": [2030, 2040],
                "grid_cell": [1, 1],
                "value": [100.0, 200.0],
            }
        )
        return a, b

    def test_joins_on_shared_dimensions_plus_grid_cell_and_renames_value_per_operand(self):
        a, b = self._pair()
        op = JoinOp(step_no=1, type="attribute", inputs=["a", "b"], output="j")
        df = join_long([a, b], op).collect().sort("calendar_year")
        assert df["a"].to_list() == [5.0, 6.0]
        assert df["b"].to_list() == [100.0, 200.0]
        assert "value" not in df.columns

    def test_only_shared_non_value_columns_form_the_key(self):
        # `a` carries an extra `region` column `b` does not have; the join key is the
        # intersection of non-value columns (`calendar_year`, `grid_cell`), not the union
        # -- `region` rides along in the result unchanged rather than being dropped or
        # forced into the key (see `join_long`'s own docstring).
        a = pl.LazyFrame(
            {
                "calendar_year": [2030, 2040],
                "grid_cell": [1, 1],
                "region": ["gulf", "gulf"],
                "value": [5.0, 6.0],
            }
        )
        b = pl.LazyFrame(
            {
                "calendar_year": [2030, 2040],
                "grid_cell": [1, 1],
                "value": [100.0, 200.0],
            }
        )
        op = JoinOp(step_no=1, type="attribute", inputs=["a", "b"], output="j")
        df = join_long([a, b], op).collect().sort("calendar_year")
        assert df["region"].to_list() == ["gulf", "gulf"]
        assert df["a"].to_list() == [5.0, 6.0]
        assert df["b"].to_list() == [100.0, 200.0]

    def _dataset_selection_pair(self, a_selection, b_selection):
        """Two operands shaped the way a real vector scan hands them over: every
        dataset-selection column materialized as a literal, one row per (year, cell).
        """
        a = pl.LazyFrame(
            {
                **{name: [value, value] for name, value in a_selection.items()},
                "calendar_year": [2030, 2040],
                "grid_cell": [1, 1],
                "value": [5.0, 6.0],
            }
        )
        b = pl.LazyFrame(
            {
                **{name: [value, value] for name, value in b_selection.items()},
                "calendar_year": [2030, 2040],
                "grid_cell": [1, 1],
                "value": [100.0, 200.0],
            }
        )
        return a, b

    def test_differing_dataset_selection_columns_do_not_kill_the_join(self):
        # The ordinary cross-variable join: two variables on one grid, compared cell by
        # cell. Both operands carry `variable`/`time_unit`/`grid` as literal columns, and
        # they disagree on `variable` in every row -- so treating those as shared key
        # columns would make this inner join match nothing and return an empty frame.
        # They are excluded from the key instead (see `join_long`'s own docstring), so the
        # join matches on `calendar_year` + `grid_cell` and both rows survive.
        a, b = self._dataset_selection_pair(
            {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"},
            {"variable": "omar", "time_unit": "annual", "grid": "ecoregion_v001"},
        )
        op = JoinOp(step_no=1, type="spatial", inputs=["a", "b"], output="j")
        df = join_long([a, b], op).collect().sort("calendar_year")
        assert df.height == 2
        assert df["a"].to_list() == [5.0, 6.0]
        assert df["b"].to_list() == [100.0, 200.0]

    def test_dataset_selection_columns_are_dropped_from_the_result(self):
        # Excluded from the key AND dropped, not merely suffixed: after a join "which
        # dataset did this row come from" has no single answer, and the operand each
        # measurement came from is carried by its renamed value column instead.
        a, b = self._dataset_selection_pair(
            {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"},
            {"variable": "omar", "time_unit": "annual", "grid": "ecoregion_v001"},
        )
        op = JoinOp(step_no=1, type="spatial", inputs=["a", "b"], output="j")
        df = join_long([a, b], op).collect()
        assert set(df.columns) == {"calendar_year", "grid_cell", "a", "b"}

    def test_dataset_selection_columns_are_dropped_even_when_both_operands_agree(self):
        # Two inputs selecting the SAME dataset (differing only in some dimension a
        # downstream filter narrows) agree on all three columns, so leaving them in the
        # key would not break the join -- they are dropped anyway, so a join's output
        # columns do not depend on whether the operands happened to match there.
        selection = {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"}
        a, b = self._dataset_selection_pair(selection, selection)
        op = JoinOp(step_no=1, type="spatial", inputs=["a", "b"], output="j")
        df = join_long([a, b], op).collect().sort("calendar_year")
        assert set(df.columns) == {"calendar_year", "grid_cell", "a", "b"}
        assert df["a"].to_list() == [5.0, 6.0]
        assert df["b"].to_list() == [100.0, 200.0]
