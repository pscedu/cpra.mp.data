"""Tests for `cpra.mp.data.analysis.execution.project` -- the vector-to-vector `project`
lowering, and the one place this rewrite introduces new numeric behavior rather than
carrying an existing one over (see the module's own docstring for exactly which
aggregates changed and why).

Every fixture below uses a crosswalk that splits a source cell across two destinations
with UNEQUAL weights, and includes a null-valued source cell -- an all-`1.0` crosswalk
cannot distinguish any of the five aggregate treatments from any other, and a fixture
with no null cannot catch the weighted mean's denominator restriction. All expected
values are hand-computed and written as literals, never captured from a run.
"""

import polars as pl
import pytest

from cpra.mp.data.analysis.execution import fold as fold_module
from cpra.mp.data.analysis.execution import project as project_module
from cpra.mp.data.analysis.execution.project import project_vector_to_vector
from cpra.mp.data.analysis.request.errors import AggregateError, NotFoundError
from cpra.mp.data.analysis.request.ops import FrameType, ProjectOp, Shape
from tests.factories import write_vector_crosswalk

SRC_GRID = "ecoregion_v001"
DST_GRID = "parish_v2020"


def _frame_types():
    """`project_vector_to_vector` only reads `frame_types[op.input].grid` -- `shape` is
    irrelevant to the lowering itself (the walk already consulted it to pick this
    lowering before ever calling in), but is filled in with a real value anyway rather
    than a placeholder, since `FrameType` is a plain, unvalidated dataclass that would
    otherwise happily carry a meaningless one."""
    return {
        "a": FrameType(shape=Shape.long, grid=SRC_GRID, has_count_by_value=False, value_columns=frozenset({"value"}))
    }


def _op(aggregate):
    return ProjectOp(step_no=1, input="a", to=DST_GRID, aggregate=aggregate, output="b")


# Split-cell fixture, unequal weights, one null source cell:
#   src 1 (value=10.0) -> dst 100 (w=0.75), dst 200 (w=0.25)
#   src 2 (value=20.0) -> dst 200 (w=1.0)
#   src 3 (value=None) -> dst 100 (w=1.0)
#
# Hand-computed, per destination:
#   dst 100: contributors are src 1 (10.0, w=0.75) and src 3 (None, w=1.0)
#     sum   = 10.0*0.75                       = 7.5
#     mean  = (10.0*0.75) / 0.75              = 10.0   (src 3's weight excluded: null value)
#     min   = 10.0   max = 10.0   count = 1           (src 3 doesn't count: null value)
#   dst 200: contributors are src 1 (10.0, w=0.25) and src 2 (20.0, w=1.0)
#     sum   = 10.0*0.25 + 20.0*1.0            = 22.5
#     mean  = (10.0*0.25 + 20.0*1.0) / (0.25 + 1.0) = 22.5 / 1.25 = 18.0
#     min   = 10.0   max = 20.0   count = 2
_MAPPING = [(1, 100, 0.75), (1, 200, 0.25), (2, 200, 1.0), (3, 100, 1.0)]


def _crosswalk_frame(crosswalk_path):
    write_vector_crosswalk(crosswalk_path, SRC_GRID, DST_GRID, _MAPPING)


def _source_lf(dim_value=None):
    columns = {"grid_cell": [1, 2, 3], "value": [10.0, 20.0, None]}
    if dim_value is not None:
        columns = {"calendar_year": [dim_value] * 3, **columns}
    return pl.LazyFrame(columns)


class TestPerAggregateSplitCell:
    """One test per aggregate treatment, all five sharing the same split-cell/unequal-
    weight/null fixture -- the resume contract's floor."""

    @pytest.mark.parametrize(
        "aggregate,expected_100,expected_200",
        [
            ("sum", 7.5, 22.5),
            ("mean", 10.0, 18.0),
            ("min", 10.0, 10.0),
            ("max", 10.0, 20.0),
            ("count", 1, 2),
        ],
    )
    def test_aggregate_matches_hand_computation(
        self, crosswalk_path, aggregate, expected_100, expected_200
    ):
        _crosswalk_frame(crosswalk_path)
        frames = {"a": _source_lf()}
        result = project_vector_to_vector(frames, _op(aggregate), _frame_types()).collect().sort(
            "grid_cell"
        )
        assert result["grid_cell"].to_list() == [100, 200]
        assert result["value"].to_list() == pytest.approx([expected_100, expected_200])

    def test_weighted_aggregates_differ_from_unweighted_ones(self, crosswalk_path):
        """A regression guard on the fixture itself: if `sum`/`mean` (weighted) ever
        collapsed to the same numbers as `min`/`max`/`count` (unweighted), the fixture
        would not actually be exercising the per-aggregate distinction
        `project_vector_to_vector` exists to get right."""
        _crosswalk_frame(crosswalk_path)
        frames = {"a": _source_lf()}
        sums = project_vector_to_vector(frames, _op("sum"), _frame_types()).collect().sort(
            "grid_cell"
        )["value"].to_list()
        maxes = project_vector_to_vector(frames, _op("max"), _frame_types()).collect().sort(
            "grid_cell"
        )["value"].to_list()
        assert sums != pytest.approx(maxes)


class TestBatchedMatchesUnbatched:
    def test_mean_agrees_at_different_fold_steps(self, crosswalk_path, monkeypatch):
        """`project_vector_to_vector` always routes through the fold in real usage
        (`grid_cell` is unconditionally part of the group key), so the batched-vs-
        unbatched equality the fold's resume contract demands is exercised here at the
        `TABULAR_FOLD_STEP` level, comparing a step smaller than the fixture's row count
        against one far larger -- both must agree exactly with the hand-computed values
        above."""
        _crosswalk_frame(crosswalk_path)
        frames = {"a": _source_lf()}
        for step in (1, 1000):
            monkeypatch.setattr(fold_module, "TABULAR_FOLD_STEP", step)
            result = (
                project_vector_to_vector(frames, _op("mean"), _frame_types())
                .collect()
                .sort("grid_cell")
            )
            assert result["value"].to_list() == pytest.approx([10.0, 18.0])


class TestDimensionColumnsSurviveAsGroupKeys:
    def test_two_years_do_not_merge(self, crosswalk_path):
        _crosswalk_frame(crosswalk_path)
        lf = pl.concat([_source_lf(2030), _source_lf(2040)])
        frames = {"a": lf}
        result = (
            project_vector_to_vector(frames, _op("sum"), _frame_types())
            .collect()
            .sort(["calendar_year", "grid_cell"])
        )
        assert result.height == 4
        # Same per-year answer as the single-year case, once for each year -- proof the
        # two years never merged into one group.
        expected = sorted(
            [(2030, 100, 7.5), (2030, 200, 22.5), (2040, 100, 7.5), (2040, 200, 22.5)]
        )
        actual = sorted(
            zip(result["calendar_year"].to_list(), result["grid_cell"].to_list(), result["value"].to_list())
        )
        assert actual == pytest.approx(expected)


class TestCountByValue:
    def test_new_schema_with_null_category(self, crosswalk_path):
        """dst 100's two contributors (src 1 -> 10.0, src 3 -> None) are two DISTINCT
        categories under `count_by_value` -- unlike the other four aggregates, a null
        `value` is not skipped here, it is counted as its own category."""
        _crosswalk_frame(crosswalk_path)
        frames = {"a": _source_lf()}
        result = (
            project_vector_to_vector(frames, _op("count_by_value"), _frame_types())
            .collect()
            .sort(["grid_cell", "value_category"])
        )
        assert set(result.columns) == {"grid_cell", "value_category", "value"}
        dst_100 = result.filter(pl.col("grid_cell") == 100)
        assert dst_100.height == 2
        rows = dict(zip(dst_100["value_category"].to_list(), dst_100["value"].to_list()))
        assert rows == {10.0: 1, None: 1}
        dst_200 = result.filter(pl.col("grid_cell") == 200)
        rows_200 = dict(zip(dst_200["value_category"].to_list(), dst_200["value"].to_list()))
        assert rows_200 == {10.0: 1, 20.0: 1}

    def test_cardinality_bound_rejects(self, crosswalk_path, monkeypatch):
        _crosswalk_frame(crosswalk_path)
        monkeypatch.setattr(project_module, "COUNT_BY_VALUE_MAX_DISTINCT", 1)
        frames = {"a": _source_lf()}
        with pytest.raises(AggregateError, match="distinct"):
            project_vector_to_vector(
                frames, _op("count_by_value"), _frame_types()
            ).collect()


class TestGridColumnIsReaddressed:
    """A vector scan materializes `variable`/`time_unit`/`grid` as literal columns, so
    they ride along as ordinary columns and survive `project`'s group key. `variable` and
    `time_unit` are still true afterwards; `grid` is not, and a frame whose `grid_cell`
    column holds destination ids while its `grid` column still names the source grid
    contradicts itself in a way no downstream step can detect -- a reader resolving those
    ids against the named lattice gets a plausible, wrong answer instead of an error.
    """

    @pytest.mark.parametrize("aggregate", ["sum", "mean", "min", "max", "count"])
    def test_grid_column_names_the_destination_grid(self, crosswalk_path, aggregate):
        _crosswalk_frame(crosswalk_path)
        lf = _source_lf().with_columns(
            pl.lit("ssf").alias("variable"),
            pl.lit("annual").alias("time_unit"),
            pl.lit(SRC_GRID).alias("grid"),
        )
        df = project_vector_to_vector({"a": lf}, _op(aggregate), _frame_types()).collect()
        assert set(df["grid"].to_list()) == {DST_GRID}
        # The two selection columns that a project does NOT invalidate are left alone --
        # the projected rows are still the same variable over the same time unit.
        assert set(df["variable"].to_list()) == {"ssf"}
        assert set(df["time_unit"].to_list()) == {"annual"}

    def test_count_by_value_readdresses_too(self, crosswalk_path):
        _crosswalk_frame(crosswalk_path)
        lf = _source_lf().with_columns(pl.lit(SRC_GRID).alias("grid"))
        df = project_vector_to_vector({"a": lf}, _op("count_by_value"), _frame_types()).collect()
        assert set(df["grid"].to_list()) == {DST_GRID}

    def test_a_frame_with_no_grid_column_is_left_alone(self, crosswalk_path):
        # `project_vector_to_vector` is also called on hand-built frames that never went
        # through a vector scan (every other test in this file), so the rewrite has to be
        # conditional on the column existing rather than unconditionally adding it.
        _crosswalk_frame(crosswalk_path)
        df = project_vector_to_vector({"a": _source_lf()}, _op("sum"), _frame_types()).collect()
        assert "grid" not in df.columns

    def test_direct_path_readdresses_as_well(self, crosswalk_path, monkeypatch):
        # The unfolded branch, forced the same way `TestDirectPathMatchesFoldedPath` does.
        monkeypatch.setattr(project_module, "should_fold_group_by", lambda *a, **k: False)
        _crosswalk_frame(crosswalk_path)
        lf = _source_lf().with_columns(pl.lit(SRC_GRID).alias("grid"))
        df = project_vector_to_vector({"a": lf}, _op("sum"), _frame_types()).collect()
        assert set(df["grid"].to_list()) == {DST_GRID}


class TestMissingCrosswalk:
    def test_raises_not_found_error(self, crosswalk_path):
        frames = {"a": _source_lf()}
        with pytest.raises(NotFoundError):
            project_vector_to_vector(frames, _op("sum"), _frame_types()).collect()


class TestDirectPathMatchesFoldedPath:
    """`should_fold_group_by` always returns `True` for a real `project` today --
    `grid_cell` is unconditionally part of its group key, and nothing on the plan threads
    real filter history yet (see the module docstring) -- so the direct, unfolded
    `group_by` branch at the bottom of `project_vector_to_vector` is not reachable through
    any pipeline this vertical slice can build, exactly the same situation
    `execution/long_ops.py::group_long` already documents and tolerates for its own
    equivalent fallback line.

    Forcing that branch here with a routing-function monkeypatch (not a contrived
    reach into private internals -- `should_fold_group_by` is the one, already-tested
    knob this module's own `if should_fold_group_by(...)` reads) is a cheap way to prove
    the direct implementation is not just present but CORRECT: it must reproduce the
    same hand-computed values the folded path does, over the same split-cell fixture.
    """

    @pytest.mark.parametrize(
        "aggregate,expected_100,expected_200",
        [
            ("sum", 7.5, 22.5),
            ("mean", 10.0, 18.0),
            ("min", 10.0, 10.0),
            ("max", 10.0, 20.0),
            ("count", 1, 2),
        ],
    )
    def test_direct_aggregate_matches_hand_computation(
        self, crosswalk_path, monkeypatch, aggregate, expected_100, expected_200
    ):
        monkeypatch.setattr(project_module, "should_fold_group_by", lambda *args, **kwargs: False)
        _crosswalk_frame(crosswalk_path)
        frames = {"a": _source_lf()}
        result = (
            project_vector_to_vector(frames, _op(aggregate), _frame_types()).collect().sort("grid_cell")
        )
        assert result["grid_cell"].to_list() == [100, 200]
        assert result["value"].to_list() == pytest.approx([expected_100, expected_200])

    def test_direct_count_by_value_matches_hand_computation(self, crosswalk_path, monkeypatch):
        monkeypatch.setattr(project_module, "should_fold_group_by", lambda *args, **kwargs: False)
        _crosswalk_frame(crosswalk_path)
        frames = {"a": _source_lf()}
        result = (
            project_vector_to_vector(frames, _op("count_by_value"), _frame_types())
            .collect()
            .sort(["grid_cell", "value_category"])
        )
        dst_100 = result.filter(pl.col("grid_cell") == 100)
        rows = dict(zip(dst_100["value_category"].to_list(), dst_100["value"].to_list()))
        assert rows == {10.0: 1, None: 1}
        dst_200 = result.filter(pl.col("grid_cell") == 200)
        rows_200 = dict(zip(dst_200["value_category"].to_list(), dst_200["value"].to_list()))
        assert rows_200 == {10.0: 1, 20.0: 1}

    def test_direct_count_by_value_cardinality_bound_rejects(self, crosswalk_path, monkeypatch):
        """The direct path's own cardinality probe (`_count_by_value`), not just the
        folded one -- both must refuse a too-wide column, since a real `project` could
        take either route depending on `should_fold_group_by`."""
        monkeypatch.setattr(project_module, "should_fold_group_by", lambda *args, **kwargs: False)
        monkeypatch.setattr(project_module, "COUNT_BY_VALUE_MAX_DISTINCT", 1)
        _crosswalk_frame(crosswalk_path)
        frames = {"a": _source_lf()}
        with pytest.raises(AggregateError, match="distinct"):
            project_vector_to_vector(frames, _op("count_by_value"), _frame_types()).collect()
