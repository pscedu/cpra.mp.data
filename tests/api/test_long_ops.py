"""Tests for `cpra.mp.data.analysis.execution.long_ops` -- the long-shape lowerings for
`filter`/`case`/`group`/`join`. There is no `calculate` lowering: `request.ops.compile_ops`
lowers a `calculate` to an arm-less `case`, so it is tested here as one.

Every fixture here is a small in-memory `pl.LazyFrame` built directly, not read off disk:
these functions are pure Polars transforms over whatever frame they are handed (see
`long_ops.py`'s own module docstring -- none of them know about the walk, the plan, a
`base_path`, or a real dataset), so a hand-built frame with hand-computed expected values
is a more direct test than routing through a fixture file would be.
"""

import polars as pl
import pytest

from cpra.mp.data.analysis.execution.long_ops import (
    case_long,
    filter_long,
    group_long,
    join_long,
)
from cpra.mp.data.analysis.request.expressions import ExpressionError
from cpra.mp.data.analysis.request.ops import CaseOp, GroupOp, JoinOp
from cpra.mp.data.analysis.request.pipeline import CaseWhen


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


class TestFilterLongValueColumns:
    """`value_columns` decides which bucket a conjunct lands in. Without it this lowering
    classified by the literal name `value`, so a PAIRED frame's operand-named conjunct
    fell through to the dimension bucket and took the unrestricted `lf.sql()` path --
    disagreeing with `band_ops.filter_band_wide`, which has always had the parameter.
    """

    @staticmethod
    def _paired():
        return pl.LazyFrame(
            {
                "grid_cell": [1, 2, 3],
                "calendar_year": [2030, 2030, 2030],
                "marsh": [0.9, None, 0.2],
                "sal": [6.0, 6.0, 6.0],
            }
        )

    def test_operand_named_conjunct_drops_rows(self):
        df = filter_long(self._paired(), "marsh > 0.5", ("marsh", "sal")).collect()
        assert df["grid_cell"].to_list() == [1]

    def test_operand_named_conjunct_goes_through_the_whitelist(self):
        """The point of the correction: an operand-named conjunct is a VALUE conjunct, so
        it is checked by `parse_value_expr` rather than handed whole to `lf.sql()`.
        `exp` is not in `ALLOWED_FUNCTIONS`, and ran before.
        """
        with pytest.raises(ExpressionError, match="not in the allowed whitelist"):
            filter_long(self._paired(), "exp(marsh) > 1", ("marsh", "sal")).collect()

    def test_conjunct_naming_two_operands_is_one_value_predicate(self):
        df = filter_long(self._paired(), "marsh < sal", ("marsh", "sal")).collect()
        assert df["grid_cell"].to_list() == [1, 3]

    def test_dimension_conjunct_still_sorts_apart_on_a_paired_frame(self):
        df = filter_long(self._paired(), "calendar_year = 2030 AND marsh > 0.5", ("marsh", "sal")).collect()
        assert df["grid_cell"].to_list() == [1]

    def test_mixed_value_and_dimension_conjunct_is_rejected_as_it_is_band_wide(self):
        """`value > calendar_year` has no lowering in either shape. It was rejected here
        by `classify_where`'s own mixed-conjunct rule and is now rejected by the value
        namespace, which is the same check band-wide shape already applied -- the two
        shapes agree rather than one admitting what the other refuses.
        """
        with pytest.raises(ExpressionError, match="not in the allowed namespace"):
            filter_long(_frame(), "value > calendar_year").collect()


VALUE_ONLY = frozenset({"value"})


class TestLoweredCalculateLong:
    """What `compile_ops` turns a `calculate` into: a `case` with NO arms, whose `else` is
    the calculate's expression. `case_expression` degrades to "the else, everywhere" for
    that shape, so the whole of `calculate`'s semantics is one corner of `case`'s.
    """

    def _calculate(self, expression):
        return CaseOp(step_no=1, input="a", cases=[], else_=expression, output="b")

    def test_rewrites_value_everywhere(self):
        df = (
            case_long(_frame(), self._calculate("value * 2"), VALUE_ONLY)
            .collect()
            .sort(["calendar_year", "grid_cell"])
        )
        assert df["value"].to_list() == [20.0, 40.0, 60.0, 80.0]
        assert df.height == 4

    def test_null_value_is_left_null(self):
        lf = pl.LazyFrame(
            {"calendar_year": [2030, 2040], "grid_cell": [1, 1], "value": [10.0, None]}
        )
        df = case_long(lf, self._calculate("value * 2"), VALUE_ONLY).collect().sort("calendar_year")
        assert df["value"].to_list() == [20.0, None]

    def test_the_failing_expression_is_named_as_an_expression_not_an_else(self):
        # An arm-less `case` is a `calculate`, so the message has to name the half the
        # AUTHOR wrote -- an `else` they never typed would send them looking for it.
        with pytest.raises(ExpressionError, match=r"^step\.1 expression: "):
            case_long(_frame(), self._calculate("exp(value)"), VALUE_ONLY).collect()


class TestLoweredCalculateLongAfterJoin:
    """The lowered `calculate` reading a joined frame, whose value columns are named after
    the join's operands rather than `value`. This is the step the single-value-column sink
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

    def _calculate(self, expression):
        return CaseOp(step_no=2, input="j", cases=[], else_=expression, output="c")

    def test_expression_over_operand_columns_collapses_to_one_value(self):
        df = (
            case_long(self._joined(), self._calculate("a - b"), frozenset({"a", "b"}))
            .collect()
            .sort("calendar_year")
        )
        assert df["value"].to_list() == [6.0, 25.0]
        assert set(df.columns) == {"calendar_year", "grid_cell", "value"}

    def test_null_in_either_operand_leaves_value_null(self):
        lf = pl.LazyFrame(
            {"calendar_year": [2030, 2040], "grid_cell": [1, 1], "a": [10.0, None], "b": [4.0, 5.0]}
        )
        df = (
            case_long(lf, self._calculate("a - b"), frozenset({"a", "b"}))
            .collect()
            .sort("calendar_year")
        )
        assert df["value"].to_list() == [6.0, None]

    def test_expression_naming_a_column_outside_the_value_namespace_is_rejected(self):
        # `value` is exactly the name that is NOT available on a joined frame, which makes
        # it the sharpest check that the namespace really is the frame's value columns and
        # not a superset that happens to include the old default.
        with pytest.raises(ExpressionError, match="not in the allowed namespace"):
            case_long(self._joined(), self._calculate("value * 2"), frozenset({"a", "b"})).collect()

    def test_dimension_column_is_still_outside_the_expression_namespace(self):
        # Widening the namespace to the frame's value columns must not widen it to every
        # column the frame has -- a dimension inside an expression is rejected here for
        # the same reason it always was.
        with pytest.raises(ExpressionError, match="not in the allowed namespace"):
            case_long(
                self._joined(), self._calculate("a * calendar_year"), frozenset({"a", "b"})
            ).collect()


def _case(*arms, else_=None):
    """A one-step `case` over `(when, then)` pairs. Always step 1, so the error-message
    test below can spell the position it expects.
    """
    return CaseOp(
        step_no=1,
        input="a",
        cases=[CaseWhen(when=when, then=then) for when, then in arms],
        else_=else_,
        output="b",
    )


class TestCaseLong:
    def test_first_matching_arm_wins(self):
        # The arms deliberately OVERLAP -- every value over 35 satisfies both -- so the
        # result can only be right if later arms never see a cell an earlier one claimed.
        op = _case(("value > 35", "1"), ("value > 15", "2"))
        df = case_long(_frame(), op, VALUE_ONLY).collect().sort(["calendar_year", "grid_cell"])
        assert df["value"].to_list() == [None, 2.0, 2.0, 1.0]

    def test_every_arm_reads_the_incoming_value_not_the_running_one(self):
        """The property `case` exists for. Written as a chain of `calculate`s these two
        entries collide -- the first writes `30`, which the second's `where` then matches
        -- so the 10.0 cell would come out `40`. Here each cell is written exactly once.
        """
        op = _case(("value = 10", "30"), ("value = 30", "40"))
        df = case_long(_frame(), op, VALUE_ONLY).collect().sort(["calendar_year", "grid_cell"])
        assert df["value"].to_list() == [30.0, None, 40.0, None]

    def test_an_unmatched_value_is_null_by_default(self):
        op = _case(("value > 1000", "1"))
        df = case_long(_frame(), op, VALUE_ONLY).collect()
        assert df["value"].to_list() == [None, None, None, None]
        # Masked, not dropped -- a `case` is shape-preserving, so the rows stay.
        assert df.height == 4

    def test_else_supplies_the_fall_through(self):
        op = _case(("value > 25", "1"), else_="0")
        df = case_long(_frame(), op, VALUE_ONLY).collect().sort(["calendar_year", "grid_cell"])
        assert df["value"].to_list() == [0.0, 0.0, 1.0, 1.0]

    def test_then_and_else_are_value_expressions_not_just_constants(self):
        op = _case(("value > 25", "value * 2"), else_="value + 1")
        df = case_long(_frame(), op, VALUE_ONLY).collect().sort(["calendar_year", "grid_cell"])
        assert df["value"].to_list() == [11.0, 21.0, 60.0, 80.0]

    def test_an_else_catches_nodata_and_the_author_says_what_to_do_about_it(self):
        """`case` adds no nodata guard of its own: a null matches no `when` (every
        predicate is `fill_null(False)`), so it falls straight through to the `else` and
        `else 0` really does paint every nodata cell `0`.

        The second half is the trade that makes the first half acceptable -- the author
        states the intent in one conjunct, and an arm that tests nullity WORKS, which it
        could not while a blanket guard discarded its answer.
        """
        lf = pl.LazyFrame(
            {"calendar_year": [2030, 2040], "grid_cell": [1, 1], "value": [10.0, None]}
        )
        op = _case(("value > 25", "1"), else_="0")
        df = case_long(lf, op, VALUE_ONLY).collect().sort("calendar_year")
        assert df["value"].to_list() == [0.0, 0.0]

        guarded = _case(("value > 25", "1"), ("value IS NULL", "value"), else_="0")
        df = case_long(lf, guarded, VALUE_ONLY).collect().sort("calendar_year")
        assert df["value"].to_list() == [0.0, None]

    def test_an_expression_outside_the_value_namespace_is_rejected(self):
        # A `then`/`else` is arithmetic over the frame's VALUE columns only -- unlike a
        # `when`, which may also name dimensions (see `TestCaseLongDimensionArms`).
        with pytest.raises(ExpressionError, match="not in the allowed namespace"):
            case_long(_frame(), _case(("value > 1", "calendar_year")), VALUE_ONLY).collect()

    def test_the_failing_part_of_a_multi_part_step_is_named(self):
        # A `case` has many expressions, so "step 1 (case) is invalid" would leave the
        # caller to find which one. Every part carries its own position and role.
        op = _case(("value > 1", "1"), ("exp(value) > 1", "2"))
        with pytest.raises(ExpressionError, match=r"^step\.1 arm 2 when: "):
            case_long(_frame(), op, VALUE_ONLY).collect()
        op = _case(("value > 1", "exp(value)"))
        with pytest.raises(ExpressionError, match=r"^step\.1 arm 1 then: "):
            case_long(_frame(), op, VALUE_ONLY).collect()
        op = _case(("value > 1", "1"), else_="exp(value)")
        with pytest.raises(ExpressionError, match=r"^step\.1 else: "):
            case_long(_frame(), op, VALUE_ONLY).collect()


class TestCaseLongDimensionArms:
    """An arm's `when` may mix VALUE conjuncts with DIMENSION ones. In the long shape a
    dimension is a real column, so both halves are per-row masks and the split is barely
    visible -- but it is the same split `case_band_wide` uses to decide which BANDS an arm
    applies to at all, and the two shapes have to agree on the answer.
    """

    def test_a_dimension_only_arm_selects_the_rows_it_rewrites(self):
        op = _case(("calendar_year = 2030", "value * 2"), else_="value")
        df = case_long(_frame(), op, VALUE_ONLY).collect().sort(["calendar_year", "grid_cell"])
        # 2030 doubled; 2040 handed back by the `else`, which is where the pass-through a
        # `calculate`'s `where` used to leave implicit now has to be written.
        assert df["value"].to_list() == [20.0, 40.0, 30.0, 40.0]
        assert df.height == 4

    def test_a_mixed_arm_ands_the_two_halves(self):
        op = _case(("calendar_year = 2040 AND value > 30", "1"), else_="0")
        df = case_long(_frame(), op, VALUE_ONLY).collect().sort(["calendar_year", "grid_cell"])
        assert df["value"].to_list() == [0.0, 0.0, 0.0, 1.0]

    def test_arms_partition_the_frame_by_dimension(self):
        """The ensemble-weighting shape (`analyses/lnd_conf.py`): one arm per
        member, each naming only its own dimension value, no `else` because the arms cover
        everything the upstream filter left.
        """
        op = _case(("calendar_year = 2030", "value * 0.25"), ("calendar_year = 2040", "value * 0.75"))
        df = case_long(_frame(), op, VALUE_ONLY).collect().sort(["calendar_year", "grid_cell"])
        assert df["value"].to_list() == [2.5, 5.0, 22.5, 30.0]

    def test_a_null_dimension_fails_the_arm_rather_than_nulling_the_condition(self):
        lf = pl.LazyFrame(
            {"calendar_year": [2030, None], "grid_cell": [1, 2], "value": [10.0, 20.0]}
        )
        op = _case(("calendar_year = 2030", "1"), else_="0")
        df = case_long(lf, op, VALUE_ONLY).collect().sort("grid_cell")
        assert df["value"].to_list() == [1.0, 0.0]

    def test_a_dimension_the_frame_does_not_carry_is_rejected_at_execution(self):
        """Deferred from shape typing on purpose: which dimension columns a frame carries
        is a property of the data, not of the request, so `request.ops._type_case` cannot
        know it and the check lands here instead. The message names the missing column and
        what was available.
        """
        with pytest.raises(ExpressionError, match=r"\['scenario_id'\].*available columns"):
            case_long(_frame(), _case(("scenario_id = 34", "1")), VALUE_ONLY).collect()

    def test_a_dimension_arm_on_a_joined_frame_names_the_shared_dimension(self):
        lf = pl.LazyFrame(
            {
                "calendar_year": [2030, 2040],
                "grid_cell": [1, 1],
                "a": [10.0, 30.0],
                "b": [4.0, 5.0],
            }
        )
        op = _case(("calendar_year = 2030", "a - b"), else_="a + b")
        df = case_long(lf, op, frozenset({"a", "b"})).collect().sort("calendar_year")
        assert df["value"].to_list() == [6.0, 35.0]
        assert set(df.columns) == {"calendar_year", "grid_cell", "value"}


class TestCaseLongAfterJoin:
    """A `case` reading a joined frame, whose value columns are named after the join's
    operands. Same collapse `calculate` makes: the arms may name any operand, and the
    result is one column called `value` with the operand columns gone.
    """

    def _joined(self):
        return pl.LazyFrame(
            {
                "calendar_year": [2030, 2040],
                "grid_cell": [1, 1],
                "a": [10.0, 30.0],
                "b": [4.0, 5.0],
            }
        )

    def test_arms_over_operand_columns_collapse_to_one_value(self):
        op = _case(("a > b", "a - b"), else_="b")
        df = case_long(self._joined(), op, frozenset({"a", "b"})).collect().sort("calendar_year")
        assert df["value"].to_list() == [6.0, 25.0]
        assert set(df.columns) == {"calendar_year", "grid_cell", "value"}

    def test_an_operand_an_arm_does_not_name_does_not_null_the_arms_answer(self):
        """The half-present pair, which is where the blanket guard was not merely a policy
        but wrong: row 2's `b` is null and `a` is 30, and the only arm reads `a` alone. It
        matched, on an operand that is fully present, and `30.0` is its answer -- the guard
        used to return null there, on the strength of a column no part of the `case`
        mentions.

        A pipeline that really does want both operands present says so, and the second
        half pins that it still can.
        """
        lf = pl.LazyFrame(
            {"calendar_year": [2030, 2040], "grid_cell": [1, 1], "a": [10.0, 30.0], "b": [4.0, None]}
        )
        op = _case(("a > 1", "a"), else_="0")
        df = case_long(lf, op, frozenset({"a", "b"})).collect().sort("calendar_year")
        assert df["value"].to_list() == [10.0, 30.0]

        both = _case(("a > 1 AND b IS NOT NULL", "a"))
        df = case_long(lf, both, frozenset({"a", "b"})).collect().sort("calendar_year")
        assert df["value"].to_list() == [10.0, None]

    def test_value_is_not_in_a_joined_frames_namespace(self):
        """A joined frame has no `value` column, so `value > 1` names nothing it carries.
        The rejection comes from the DIMENSION half, not the value one: with `value`
        outside the value namespace, `classify_where_kinds` can only read it as a
        dimension name, and a `when`'s dimension names are resolved against the frame's
        schema. The message still names `value` and lists what the frame does have, which
        is what the author needs either way.
        """
        op = _case(("value > 1", "1"))
        with pytest.raises(ExpressionError, match=r"\['value'\] -- available columns"):
            case_long(self._joined(), op, frozenset({"a", "b"})).collect()

    def test_then_and_else_still_reject_value_by_namespace(self):
        # The other half of the same point: a `then` is arithmetic, never a dimension
        # reference, so there is no second reading for it to fall into.
        with pytest.raises(ExpressionError, match="not in the allowed namespace"):
            case_long(self._joined(), _case(("a > 1", "value")), frozenset({"a", "b"})).collect()


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
