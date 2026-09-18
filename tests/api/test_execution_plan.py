"""Tests for `cpra.mp.data.analysis.execution.plan.compile_plan` -- turning an
already-validated `Pipeline` into the `ExecutionPlan` `execution.execute.execute`
consumes, including the pushdown decisions the module docstring describes.

Like `test_request_ops.py` (whose grid/variable fixtures this file reuses), a pipeline
with only vector inputs never opens a file or a crosswalk: `compile_plan` needs only the
declared pipeline and each input's grid kind (`cpra.mp.data.grids.GRIDS[...].grid_type`),
the same no-IO property `request.ops.validate_shapes` already has. A raster input is the
one exception -- resolving its band list takes one real call to `io.catalog.
resolve_raster_bands` -- so the raster-side tests below monkeypatch that function rather
than touch a filesystem, the same "no IO in this test file" spirit applied to the one
seam that necessarily has some.
"""

import dataclasses

import polars as pl
import pytest

from cpra.mp.data.analysis.execution import plan as plan_module
from cpra.mp.data.analysis.execution.plan import ExecutionPlan, ResolvedInput, compile_plan
from cpra.mp.data.analysis.request.errors import BandLimitError
from cpra.mp.data.analysis.request.ops import Shape, validate_shapes
from cpra.mp.data.analysis.request.pipeline import parse_pipeline
from cpra.mp.data.config import ANALYSIS_BAND_CEILING
from cpra.mp.data.grids import GRIDS

VECTOR_GRID = "hydro_compartment_v001"
RASTER_GRID = "veg_grid_cell_v001"

assert GRIDS[VECTOR_GRID].grid_type == "vector"
assert GRIDS[RASTER_GRID].grid_type == "raster"


def vector_input(grid=VECTOR_GRID, variable="sal"):
    return {"variable": variable, "time_unit": "annual", "grid": grid}


def raster_input(grid=RASTER_GRID, variable="ffibs_coverage_ratio"):
    return {"variable": variable, "time_unit": "annual", "grid": grid}


def _pipeline(data):
    """`parse_pipeline` then `validate_shapes` -- `compile_plan`'s own precondition,
    matching `compile_ops`'s (see that function's docstring)."""
    pipeline = parse_pipeline(data)
    validate_shapes(pipeline)
    return pipeline


class TestResolvedInputs:
    def test_every_pipeline_input_gets_a_resolved_input(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline)
        assert set(plan.inputs) == {"a"}
        assert isinstance(plan.inputs["a"], ResolvedInput)
        assert plan.inputs["a"].selection is pipeline.inputs["a"]

    def test_zero_steps_pushes_nothing(self):
        # With no op ever touching "a", there is nothing to narrow it by -- `columns`
        # stays `None` (nothing has closed the column set; see `TestVectorColumnPushdown`)
        # and `predicate` stays `None` (no filter to push).
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline)
        assert plan.inputs["a"].columns is None
        assert plan.inputs["a"].predicate is None


class TestVectorPredicatePushdown:
    """A `filter`'s conjunct pushed all the way back to its origin's own `ResolvedInput.
    predicate` is checked BEHAVIORALLY (apply it to a small hand-built frame and check
    which rows survive) rather than by comparing `pl.Expr` objects -- `Expr` does not
    define `==` as structural equality, and a behavioral check also demonstrates the
    predicate is something `io.vector.scan_vector` could actually use.
    """

    def test_a_value_conjunct_with_nothing_before_it_pushes(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [{"op": "filter", "input": "a", "where": "value > 1"}],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        df = pl.DataFrame({"value": [0, 2]})
        assert df.filter(resolved.predicate)["value"].to_list() == [2]
        assert resolved.columns is None  # no `group` has closed the column set

    def test_a_dimension_conjunct_pushes(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [{"op": "filter", "input": "a", "where": "calendar_year = 2035"}],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        df = pl.DataFrame({"calendar_year": [2030, 2035]})
        assert df.filter(resolved.predicate)["calendar_year"].to_list() == [2035]

    def test_a_spatial_conjunct_pushes(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [{"op": "filter", "input": "a", "where": "grid_cell = 7"}],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        df = pl.DataFrame({"grid_cell": [1, 7]})
        assert df.filter(resolved.predicate)["grid_cell"].to_list() == [7]

    def test_two_conjuncts_in_one_where_both_push_anded_together(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "filter", "input": "a", "where": "value > 1 and calendar_year = 2035"}
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        df = pl.DataFrame(
            {"value": [0, 2, 2], "calendar_year": [2035, 2030, 2035]}
        )
        assert df.filter(resolved.predicate)["calendar_year"].to_list() == [2035]

    def test_a_value_conjunct_after_a_calculate_does_not_push(self):
        # `calculate` invalidates `value` unconditionally: the running value has already
        # been replaced by the time this filter runs, so pushing its predicate onto the
        # ORIGIN's own scan would test a number the filter never actually sees.
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "calculate", "input": "a", "expression": "value * 2", "as": "c"},
                    {"op": "filter", "input": "c", "where": "value > 1"},
                ],
                "output": {"format": "csv"},
            }
        )
        assert compile_plan(pipeline).inputs["a"].predicate is None

    def test_a_dimension_conjunct_survives_a_calculate(self):
        # `calculate` only invalidates `value`, leaving dimension validity untouched.
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "calculate", "input": "a", "expression": "value * 2", "as": "c"},
                    {"op": "filter", "input": "c", "where": "calendar_year = 2035"},
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        df = pl.DataFrame({"calendar_year": [2030, 2035]})
        assert df.filter(resolved.predicate)["calendar_year"].to_list() == [2035]


class TestGroupSemanticValidity:
    """The one non-obvious clause in the invalidation table: a dimension a chain later
    `group`s by is not "used up" by the group -- a dimension filter naming it may still be
    pushed all the way back to the origin's own read, from either side of the `group`.
    """

    def test_a_dimension_filter_pushes_through_a_group_by_that_same_dimension(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "filter", "input": "a", "where": "calendar_year = 2035", "as": "f1"},
                    {"op": "group", "by": ["calendar_year"], "aggregate": "sum", "as": "g"},
                    {"op": "filter", "input": "g", "where": "calendar_year = 2035"},
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        df = pl.DataFrame({"calendar_year": [2030, 2035]})
        assert df.filter(resolved.predicate)["calendar_year"].to_list() == [2035]
        # The `group` closes the column set: everything referenced so far, plus `by`.
        assert resolved.columns == sorted({"grid_cell", "value", "calendar_year"})

    def test_a_dimension_filter_does_not_push_through_a_group_that_drops_it(self):
        # Same two filters, but `by=[]` grouped the dimension away -- `dimensions` narrows
        # to the empty set, so the SECOND filter's conjunct (naming a dimension no longer
        # open) does not join the first one at the origin's own read.
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "filter", "input": "a", "where": "calendar_year = 2035", "as": "f1"},
                    {"op": "group", "by": [], "aggregate": "sum", "as": "g"},
                    {"op": "filter", "input": "g", "where": "calendar_year = 2035"},
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        df = pl.DataFrame({"calendar_year": [2030, 2035]})
        assert df.filter(resolved.predicate)["calendar_year"].to_list() == [2035]


class TestVectorColumnPushdown:
    def test_no_group_means_no_column_narrowing(self):
        # A `filter`/`calculate`-only chain never drops a dimension column from the
        # frame it produces, so the source read must not narrow to less than everything
        # either -- the sink may still need a column no `where` ever referenced.
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "filter", "input": "a", "where": "calendar_year = 2035", "as": "f"},
                    {"op": "calculate", "expression": "value * 2"},
                ],
                "output": {"format": "csv"},
            }
        )
        assert compile_plan(pipeline).inputs["a"].columns is None

    def test_a_group_closes_the_column_set_to_by_plus_grid_cell_plus_of(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "calculate", "input": "a", "expression": "value", "as": "c"},
                    {"op": "group", "by": ["calendar_year"], "aggregate": "mean", "of": "value"},
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        assert resolved.columns == sorted({"grid_cell", "value", "calendar_year"})

    def test_a_group_of_a_non_value_column_keeps_that_column_in_the_read(self):
        # `of` names the column the aggregate consumes. Narrowing it away would leave the
        # group with nothing to aggregate, so it has to survive the closure alongside `by`.
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "calculate", "input": "a", "expression": "value", "as": "c"},
                    {"op": "group", "by": ["calendar_year"], "aggregate": "mean", "of": "grid_cell"},
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        assert "grid_cell" in resolved.columns

    def test_a_calculate_wheres_columns_survive_a_later_group(self):
        # The `where` is evaluated against the scanned frame, so every column it names has
        # to outlive the narrowing a later `group` closes -- even though `scenario_id` is
        # neither grouped by nor aggregated, and nothing else in the pipeline mentions it.
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {
                        "op": "calculate",
                        "input": "a",
                        "expression": "value * 2",
                        "where": "scenario_id = 7",
                        "as": "c",
                    },
                    {"op": "group", "by": ["calendar_year"], "aggregate": "mean", "of": "value"},
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        assert "scenario_id" in resolved.columns


class TestFanOutDisablesPushdown:
    def test_two_filters_reading_the_same_frame_get_nothing_pushed(self):
        # "a" is read directly by both filters below -- pushing either one's predicate
        # onto "a"'s own scan would silently narrow the OTHER filter's own output too,
        # so neither one's predicate (nor any column narrowing) is ever recorded for "a".
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "filter", "input": "a", "where": "value > 1", "as": "f1"},
                    {"op": "filter", "input": "a", "where": "value > 0", "as": "f2"},
                    {"op": "join", "type": "spatial", "inputs": ["f1", "f2"], "as": "j"},
                    {"op": "calculate", "expression": "f1 + f2"},
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        assert resolved.predicate is None
        assert resolved.columns is None


class TestProjectPushdown:
    def test_a_value_filter_after_a_vector_to_raster_project_pushes_to_the_vector_origin(self):
        # Vector-to-raster takes no aggregate (a pixel's value IS its containing zone's
        # value, unchanged), so a value predicate declared after this project is exactly
        # as valid against the origin's own pre-project rows as it is against the
        # post-project pixels.
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "project", "input": "a", "to": RASTER_GRID, "as": "p"},
                    {"op": "filter", "input": "p", "where": "value > 1"},
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        df = pl.DataFrame({"value": [0, 2]})
        assert df.filter(resolved.predicate)["value"].to_list() == [2]
        # Column pushdown is not attempted through a `project`, regardless of direction.
        assert resolved.columns is None

    def test_a_vector_to_vector_project_stops_pushdown_entirely(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "project", "input": "a", "to": "ecoregion_v001", "aggregate": "sum", "as": "p"},
                    {"op": "filter", "input": "p", "where": "value > 1"},
                ],
                "output": {"format": "csv"},
            }
        )
        assert compile_plan(pipeline).inputs["a"].predicate is None

    def test_a_raster_to_vector_project_stops_pushdown_entirely(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [
                    {"op": "project", "input": "a", "to": VECTOR_GRID, "aggregate": "sum", "as": "p"},
                    {"op": "filter", "input": "p", "where": "value > 1"},
                ],
                "output": {"format": "csv"},
            }
        )
        # A raster origin with no live state left has no `raster_dimension_parts`
        # contributions either -- confirmed by the plan compiling at all despite "a"'s
        # only dimension filter ever declared sitting after the project (there is none
        # here to conflict with, this simply confirms nothing raises).
        assert isinstance(compile_plan(pipeline), ExecutionPlan)


class TestShapeAnnotation:
    def test_a_vector_input_is_long(self):
        pipeline = _pipeline(
            {"inputs": {"a": vector_input()}, "steps": [], "output": {"format": "csv"}}
        )
        plan = compile_plan(pipeline)
        assert plan.frame_types["a"].shape == Shape.long

    def test_a_raster_input_is_band_wide(self):
        pipeline = _pipeline(
            {"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "csv"}}
        )
        plan = compile_plan(pipeline)
        assert plan.frame_types["a"].shape == Shape.band_wide

    def test_every_step_output_gets_its_own_shape_entry(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "filter", "input": "a", "where": "value > 1", "as": "f"},
                    {"op": "calculate", "expression": "value * 2", "as": "c"},
                    {"op": "group", "by": [], "aggregate": "sum", "as": "g"},
                ],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline)
        # `annotate_frame_types` records an entry for the input AND every step's own output
        # name -- one entry per `Op` in `plan.ops`, plus each of `plan.inputs`.
        assert set(plan.frame_types) == {"a", "f", "c", "g"}
        # None of filter/calculate/group changes shape (only `project` can -- see
        # `FILTER_LOWERING_TABLE`/`_type_calculate`/`_type_group`), so every one of them
        # stays long, the input's own shape.
        assert plan.frame_types["f"].shape == Shape.long
        assert plan.frame_types["c"].shape == Shape.long
        assert plan.frame_types["g"].shape == Shape.long

    def test_join_output_shape_matches_its_operands(self):
        # A bare join's output carries one value column per operand (`_type_join` sets
        # `value_columns` to `frozenset(op.inputs)`); `validate_shapes`'s own final-frame
        # rule requires exactly one, so the pipeline needs a `calculate` collapsing them --
        # unrelated to shape, but required for `_pipeline`'s `validate_shapes` call to pass
        # at all.
        pipeline = _pipeline(
            {
                # One grid, two variables: `_type_join` requires its operands to agree on
                # grid, so the two inputs differ in `variable` rather than in `grid`.
                "inputs": {"a": vector_input(), "b": vector_input(variable="stg")},
                "steps": [
                    {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                    {"op": "calculate", "expression": "a + b", "as": "c"},
                ],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline)
        assert plan.frame_types["a"].shape == Shape.long
        assert plan.frame_types["b"].shape == Shape.long
        assert plan.frame_types["j"].shape == Shape.long
        assert plan.frame_types["c"].shape == Shape.long

    def test_default_synthetic_output_names_get_shapes_too(self):
        # A step with no explicit `as` gets `str(step_no)` (`ops._default_output_name`) --
        # `annotate_frame_types` must key by `op.output`, whatever it resolved to, not assume
        # every step names itself.
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [{"op": "filter", "input": "a", "where": "value > 1"}],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline)
        assert plan.frame_types["1"].shape == Shape.long


class TestCompileOpensNoDataFiles:
    def test_a_vector_only_pipeline_never_touches_the_filesystem(self, monkeypatch):
        # True only for a pipeline with no raster input -- a raster input's band list can
        # only be answered by one real `io.catalog.resolve_raster_bands` call (see
        # `TestRasterBandPushdown` below, which monkeypatches exactly that seam). This
        # test pins the vector side: `io.vector.scan_vector` is never even imported by way
        # of being called at compile time.
        import cpra.mp.data.analysis.io.vector as io_vector

        def _boom(*args, **kwargs):
            raise AssertionError("compile_plan must never call scan_vector")

        monkeypatch.setattr(io_vector, "scan_vector", _boom)

        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [{"op": "filter", "input": "a", "where": "value > 1"}],
                "output": {"format": "csv"},
            }
        )
        compile_plan(pipeline)  # does not raise


class TestExecutionPlanShape:
    def test_ops_are_compile_ops_output_unchanged(self):
        from cpra.mp.data.analysis.request.ops import compile_ops

        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [{"op": "filter", "input": "a", "where": "value > 1"}],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline)
        assert plan.ops == compile_ops(pipeline)
        assert isinstance(plan, ExecutionPlan)


def _fake_resolve_raster_bands(rows):
    """A stand-in for `io.catalog.resolve_raster_bands`, monkeypatched onto
    `execution.plan`'s own reference to it (`plan_module.resolve_raster_bands`) so the
    tests below can characterize band-selection pushdown without touching a filesystem or
    depending on any real dataset's actual band count.

    `rows` is a list of `{dimension_name: value, ...}` dicts, one per fake band; `path`/
    `band` columns are added automatically (never inspected by anything under test here --
    only present because a real `resolve_raster_bands` result always carries them).
    """

    def _fake(selection, *, base_path=None):
        keys = rows[0].keys() if rows else []
        return pl.DataFrame(
            {
                "path": [f"/fake/{i}.tif" for i in range(len(rows))],
                "band": list(range(1, len(rows) + 1)),
                **{key: [row[key] for row in rows] for key in keys},
            }
        )

    return _fake


class TestRasterBandPushdown:
    def test_a_dimension_filter_narrows_band_metadata(self, monkeypatch):
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands(
                [{"calendar_year": 2030}, {"calendar_year": 2035}, {"calendar_year": 2040}]
            ),
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "filter", "input": "a", "where": "calendar_year >= 2035"}],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        assert resolved.band_metadata["calendar_year"].to_list() == [2035, 2040]

    def test_a_dimension_filter_narrows_bands_from_behind_a_calculate(self, monkeypatch):
        """`calculate` invalidates `value` and nothing else, so a dimension filter
        declared AFTER one still selects bands -- fewer files opened than the declared
        order suggests.
        """
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands(
                [{"calendar_year": 2030}, {"calendar_year": 2035}, {"calendar_year": 2040}]
            ),
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [
                    {"op": "calculate", "input": "a", "expression": "value * 2", "as": "c"},
                    {"op": "filter", "input": "c", "where": "calendar_year >= 2035"},
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        assert resolved.band_metadata["calendar_year"].to_list() == [2035, 2040]

    def test_a_dimension_filter_before_a_raster_to_vector_project_still_narrows_bands(self, monkeypatch):
        """The project stops pushdown for everything declared after it, but it does not
        undo what was already accumulated in front of it: the band set really was narrowed
        while the frame was still band-wide, and the `group` then runs on the long frame
        the project produced.
        """
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands(
                [{"calendar_year": 2030}, {"calendar_year": 2035}, {"calendar_year": 2040}]
            ),
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [
                    {"op": "filter", "input": "a", "where": "calendar_year >= 2035", "as": "f"},
                    {"op": "project", "input": "f", "to": "ecoregion_v001", "aggregate": "mean", "as": "p"},
                    {"op": "group", "by": ["grid_cell"], "aggregate": "sum"},
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        assert resolved.band_metadata["calendar_year"].to_list() == [2035, 2040]

    def test_a_value_conjunct_is_never_pushed_as_band_narrowing(self, monkeypatch):
        # A value predicate must never drop a band (nor, at the IO level, a pixel row --
        # see `tests/api/test_raster_scan.py`'s value-predicate test). At the plan level
        # the only observable effect is that `band_metadata` is unaffected by it -- there
        # is no other field a raster `ResolvedInput` could have recorded it into.
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2030}, {"calendar_year": 2035}]),
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "filter", "input": "a", "where": "value > 100"}],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        assert resolved.band_metadata.height == 2

    def test_a_spatial_filter_records_the_point_sample_step(self, monkeypatch):
        monkeypatch.setattr(
            plan_module, "resolve_raster_bands", _fake_resolve_raster_bands([{"calendar_year": 2030}])
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [
                    {"op": "calculate", "input": "a", "expression": "value * 2", "as": "c"},
                    {"op": "filter", "input": "c", "where": "grid_cell = 7"},
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        assert resolved.point_sample_step == 2  # the filter's own step_no

    def test_a_spatial_filter_after_a_calculate_and_a_group_still_records_the_step(self, monkeypatch):
        # Mirrors `tests/api/test_band_ops.py`'s
        # `TestSpatialFilterCommutesWithCalculateAndGroup`, which works out by hand why a
        # spatial conjunct sitting after a `calculate` AND a `group` in a raster chain is
        # still safe to mark this way: neither op can make one pixel's answer depend on
        # another pixel's input, so the frame this filter's own step sees is exactly what
        # restricting to one pixel before running `calculate`/`group` would have produced.
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2020}, {"calendar_year": 2021}]),
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [
                    {"op": "calculate", "input": "a", "expression": "value + 1", "as": "c"},
                    {"op": "group", "by": [], "aggregate": "sum", "as": "g"},
                    {"op": "filter", "input": "g", "where": "grid_cell = 1"},
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        assert resolved.point_sample_step == 3  # the filter's own step_no

    def test_no_spatial_filter_means_no_point_sample_step(self, monkeypatch):
        monkeypatch.setattr(
            plan_module, "resolve_raster_bands", _fake_resolve_raster_bands([{"calendar_year": 2030}])
        )
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "csv"}})
        assert compile_plan(pipeline).inputs["a"].point_sample_step is None


class TestBandLimitError:
    def test_a_single_raster_input_over_the_ceiling_raises(self, monkeypatch):
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2000 + i} for i in range(ANALYSIS_BAND_CEILING + 1)]),
        )
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "csv"}})
        with pytest.raises(BandLimitError):
            compile_plan(pipeline)

    def test_a_dimension_filter_that_narrows_under_the_ceiling_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2000 + i} for i in range(ANALYSIS_BAND_CEILING + 1)]),
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "filter", "input": "a", "where": "calendar_year = 2000"}],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        assert resolved.band_metadata.height == 1


class TestBandWideJoinCeiling:
    """A band-wide `join`'s two operands are read through one shared VRT
    (`execution.band_ops.join_band_wide`), so the ceiling is checked against the UNION of
    their resolved band counts, not against either one alone.
    """

    def _two_raster_inputs(self):
        return {
            "a": raster_input(variable="ffibs_coverage_ratio"),
            "b": raster_input(variable="vct"),
        }

    def _join_pipeline(self):
        return {
            "inputs": self._two_raster_inputs(),
            "steps": [
                {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                {"op": "calculate", "input": "j", "expression": "a + b"},
            ],
            "output": {"format": "csv"},
        }

    def test_operands_individually_under_the_ceiling_but_over_it_together_raises(self, monkeypatch):
        half_over = ANALYSIS_BAND_CEILING // 2 + 1
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2000 + i} for i in range(half_over)]),
        )
        pipeline = _pipeline(self._join_pipeline())
        with pytest.raises(BandLimitError):
            compile_plan(pipeline)

    def test_a_successful_band_wide_join_records_the_scan_union(self, monkeypatch):
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2030}, {"calendar_year": 2035}]),
        )
        pipeline = _pipeline(self._join_pipeline())
        plan = compile_plan(pipeline)
        assert plan.band_wide_join_scans == {"j": ("a", "b")}

    def test_an_operand_read_by_another_step_is_still_sized_as_a_union(self, monkeypatch):
        """Fan-out makes it unsafe to push a predicate back through `a`, but it does not
        make `a` a different raster. The shared-scan requirement, and the union sizing
        that follows from it, hold regardless of how many other steps also read `a`.
        """
        half_over = ANALYSIS_BAND_CEILING // 2 + 1
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2000 + i} for i in range(half_over)]),
        )
        request = self._join_pipeline()
        request["steps"].insert(0, {"op": "filter", "input": "a", "where": "calendar_year >= 2000", "as": "c"})
        with pytest.raises(BandLimitError):
            compile_plan(_pipeline(request))

    def test_a_join_of_a_join_unions_every_origin_underneath_it(self, monkeypatch):
        """The outer join reads the inner join's result, so one scan has to cover all
        three inputs -- the union is transitive, not just pairwise.
        """
        third_over = ANALYSIS_BAND_CEILING // 3 + 1
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2000 + i} for i in range(third_over)]),
        )
        inputs = self._two_raster_inputs()
        inputs["c"] = raster_input(variable="stg")
        request = {
            "inputs": inputs,
            "steps": [
                {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                {"op": "calculate", "input": "j", "expression": "a + b", "as": "m"},
                {"op": "join", "type": "spatial", "inputs": ["m", "c"], "as": "j2"},
                {"op": "calculate", "input": "j2", "expression": "value + c"},
            ],
            "output": {"format": "csv"},
        }
        # Pairwise, every join is under the ceiling; all three together are over it.
        assert 2 * third_over <= ANALYSIS_BAND_CEILING < 3 * third_over
        with pytest.raises(BandLimitError) as caught:
            compile_plan(_pipeline(request))
        assert "'j2'" in str(caught.value)

    def test_a_join_of_a_join_records_the_transitive_scan_union(self, monkeypatch):
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2030}, {"calendar_year": 2035}]),
        )
        inputs = self._two_raster_inputs()
        inputs["c"] = raster_input(variable="stg")
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": inputs,
                    "steps": [
                        {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                        {"op": "calculate", "input": "j", "expression": "a + b", "as": "m"},
                        {"op": "join", "type": "spatial", "inputs": ["m", "c"], "as": "j2"},
                        {"op": "calculate", "input": "j2", "expression": "value + c"},
                    ],
                    "output": {"format": "csv"},
                }
            )
        )
        assert plan.band_wide_join_scans == {"j": ("a", "b"), "j2": ("a", "b", "c")}

    def test_a_long_join_records_no_scan_union(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input(), "b": vector_input(variable="stg")},
                "steps": [
                    {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                    {"op": "calculate", "expression": "a + b"},
                ],
                "output": {"format": "csv"},
            }
        )
        assert compile_plan(pipeline).band_wide_join_scans == {}


class TestOverviewEligibility:
    """`ResolvedInput.overview_permitted`/`straddling_vector_grid` -- whether a raster
    origin's tile reads may be served from a decimated overview
    (`io.raster_io.tile_read_shape`) instead of always the full-resolution pixels.
    """

    def test_a_raster_to_vector_project_forbids_the_overview(self, monkeypatch):
        # A raster-to-vector project redistributes pixels onto polygons an overview level
        # can blur across or shrink below one pixel of, so the origin feeding it must never
        # be served a decimated read.
        monkeypatch.setattr(
            plan_module, "resolve_raster_bands", _fake_resolve_raster_bands([{"calendar_year": 2030}])
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [
                    {"op": "project", "input": "a", "to": VECTOR_GRID, "aggregate": "sum", "as": "p"},
                    {"op": "filter", "input": "p", "where": "value > 1"},
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        assert resolved.straddling_vector_grid == VECTOR_GRID
        assert resolved.overview_permitted is False

    def test_the_project_is_still_detected_behind_a_fan_out(self, monkeypatch):
        # `origin_sets` (unconditional provenance), not `states` (pushdown validity, which
        # a fan-out clears): "a" is read by two steps here, so its own pushdown state is
        # gone by the time the project runs, but it still IS the raster that project reads
        # from -- overview eligibility has to follow provenance, not validity.
        monkeypatch.setattr(
            plan_module, "resolve_raster_bands", _fake_resolve_raster_bands([{"calendar_year": 2030}])
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [
                    {"op": "calculate", "input": "a", "expression": "value * 2", "as": "c1"},
                    {"op": "calculate", "input": "a", "expression": "value * 3", "as": "c2"},
                    {"op": "project", "input": "c1", "to": VECTOR_GRID, "aggregate": "sum", "as": "p"},
                    {"op": "filter", "input": "p", "where": "value > 1"},
                ],
                "output": {"format": "csv"},
            }
        )
        # "c2" is a dead branch -- only present to make "a" read by two ops (the fan-out).
        resolved = compile_plan(pipeline).inputs["a"]
        assert resolved.straddling_vector_grid == VECTOR_GRID
        assert resolved.overview_permitted is False

    def test_a_point_sample_forbids_the_overview(self, monkeypatch):
        # An overview pixel is a real value, but not necessarily the one under the sampled
        # coordinate -- a point sample needs the exact pixel, never a decimated stand-in.
        monkeypatch.setattr(
            plan_module, "resolve_raster_bands", _fake_resolve_raster_bands([{"calendar_year": 2030}])
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "filter", "input": "a", "where": "grid_cell = 7"}],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        assert resolved.straddling_vector_grid is None  # a point sample, not a project
        assert resolved.overview_permitted is False

    def test_a_vector_to_vector_chain_joined_with_a_display_raster_still_permits_it(self, monkeypatch):
        # The accept case: "a" runs a v->v crosswalk-style project (its own, unrelated
        # branch -- never touching "b"/"c" at all), while "b" is only ever JOINED (not
        # projected) with another raster "c" for display. Attaching a raster's own,
        # already-resolved pixel values to a vector chain's rows via `join` is not the same
        # as redistributing the raster's pixels onto new polygons via `project`, so neither
        # "b" nor "c" is forbidden the overview by it.
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2030}, {"calendar_year": 2035}]),
        )
        pipeline = _pipeline(
            {
                "inputs": {
                    "a": vector_input(),
                    "b": raster_input(variable="ffibs_coverage_ratio"),
                    "c": raster_input(variable="vct"),
                },
                "steps": [
                    {"op": "project", "input": "a", "to": "ecoregion_v001", "aggregate": "sum", "as": "zp"},
                    {"op": "join", "type": "spatial", "inputs": ["b", "c"], "as": "j"},
                    {"op": "calculate", "input": "j", "expression": "b + c"},
                ],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline)
        assert plan.inputs["b"].straddling_vector_grid is None
        assert plan.inputs["b"].overview_permitted is True
        assert plan.inputs["c"].straddling_vector_grid is None
        assert plan.inputs["c"].overview_permitted is True

    def test_no_project_and_no_point_sample_means_the_default_permits_it(self, monkeypatch):
        monkeypatch.setattr(
            plan_module, "resolve_raster_bands", _fake_resolve_raster_bands([{"calendar_year": 2030}])
        )
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "csv"}})
        resolved = compile_plan(pipeline).inputs["a"]
        assert resolved.straddling_vector_grid is None
        assert resolved.overview_permitted is True

    def test_a_vector_origin_is_unaffected(self):
        # `overview_permitted`/`straddling_vector_grid` are a raster-only concept; a vector
        # `ResolvedInput` simply carries the dataclass defaults, which nothing reads.
        pipeline = _pipeline({"inputs": {"a": vector_input()}, "steps": [], "output": {"format": "csv"}})
        resolved = compile_plan(pipeline).inputs["a"]
        assert resolved.overview_permitted is True
        assert resolved.straddling_vector_grid is None
