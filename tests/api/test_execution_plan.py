"""Tests for `cpra.mp.data.analysis.execution.plan.compile_plan` -- turning an
already-validated `Pipeline` into the `ExecutionPlan` `execution.execute.execute`
consumes, including the pushdown decisions the module docstring describes.

Like `test_request_ops.py` (whose grid/variable fixtures this file reuses), a pipeline
that stays LONG never opens a file or a crosswalk: `compile_plan` needs only the declared
pipeline and each input's grid kind (`cpra.mp.data.grids.GRIDS[...].grid_type`), the same
no-IO property `request.ops.validate_shapes` already has. Anything that resolves to BANDS
is the exception, in both directions -- a raster input's band list takes one real call to
`io.catalog.resolve_raster_bands`, and a vector-to-raster `project` takes one to
`resolve_vector_bands` -- so the tests below monkeypatch those two functions rather than
touch a filesystem, the same "no IO in this test file" spirit applied to the seams that
necessarily have some.

`TestTileReadWindow` bends that once more, and only in its straddling-widening test: a
window widened to cover whole features is a union with real polygon bounds, so a geometry
file has to exist to read them from. Everything else about a tile-scoped compile -- the
window itself included -- comes off the grid registry with nothing opened.
"""

from typing import ClassVar

import mercantile
import polars as pl
import pytest
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

from cpra.mp.data.analysis.execution import plan as plan_module
from cpra.mp.data.analysis.execution.plan import (
    ExecutionPlan,
    ResolvedInput,
    TileRequest,
    compile_plan,
)
from cpra.mp.data.analysis.execution.point_sample import point_sample_reads
from cpra.mp.data.analysis.io.raster_io import TILE_CRS, resampling_buffer_px, tile_decimation_factors
from cpra.mp.data.analysis.request.errors import BandLimitError, ShapeError
from cpra.mp.data.analysis.request.ops import Shape, validate_shapes
from cpra.mp.data.analysis.request.pipeline import parse_pipeline
from cpra.mp.data.config import ANALYSIS_BAND_CEILING
from cpra.mp.data.grids import GRIDS
from tests.factories import write_grid_geometry

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


class TestCasePushdown:
    """A `case` rewrites the running value, so it retires this origin's value validity
    exactly as a `calculate` does -- and its arms' columns have to outlive a later
    `group`'s column closure for the same reason a `calculate`'s `where` does.
    """

    def test_a_value_conjunct_after_a_case_does_not_push(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {
                        "op": "case",
                        "input": "a",
                        "cases": [{"when": "value > 0", "then": "1"}],
                        "as": "c",
                    },
                    {"op": "filter", "input": "c", "where": "value > 1"},
                ],
                "output": {"format": "csv"},
            }
        )
        assert compile_plan(pipeline).inputs["a"].predicate is None

    def test_a_dimension_conjunct_survives_a_case(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {
                        "op": "case",
                        "input": "a",
                        "cases": [{"when": "value > 0", "then": "1"}],
                        "as": "c",
                    },
                    {"op": "filter", "input": "c", "where": "calendar_year = 2035"},
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        df = pl.DataFrame({"calendar_year": [2030, 2035]})
        assert df.filter(resolved.predicate)["calendar_year"].to_list() == [2035]

    def test_the_arms_add_nothing_to_a_later_groups_column_closure(self):
        """Unlike a `calculate`'s `where`, a `case` arm can only ever name a column the
        read already has: `request.ops._type_case` confines every `when` to the frame's
        VALUE namespace, so there is no `scenario_id = 7` arm to keep alive across the
        closure the way `test_a_calculate_wheres_columns_survive_a_later_group` has to.
        The plan sweeps the arms anyway -- widening a vector read is only ever slow, never
        wrong -- and this pins that the sweep does not somehow widen it here.
        """
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {
                        "op": "case",
                        "input": "a",
                        "cases": [
                            {"when": "value > 0", "then": "1"},
                            {"when": "value < 0", "then": "2"},
                        ],
                        "as": "c",
                    },
                    {"op": "group", "by": ["calendar_year"], "aggregate": "mean", "of": "value"},
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        assert resolved.columns == sorted({"grid_cell", "value", "calendar_year"})

    def test_a_dimension_arms_columns_are_kept_but_the_conjunct_is_not_pushed(self):
        """An arm's dimension conjunct decides which rows get REWRITTEN, not which exist,
        so pushing it to the read as a row predicate would delete the rows the `else` was
        supposed to hand back. What the plan takes from the arm is only its column names:
        `scenario_id` has to be read for the arm to be evaluable at all, even though
        nothing else in the pipeline mentions it.
        """
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {
                        "op": "case",
                        "input": "a",
                        "cases": [{"when": "scenario_id = 7", "then": "1"}],
                        "else": "value",
                        "as": "c",
                    },
                    {"op": "group", "by": ["calendar_year"], "aggregate": "mean", "of": "value"},
                ],
                "output": {"format": "csv"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        assert "scenario_id" in resolved.columns
        # Named, not pushed: nothing narrows the read, since every row still has to arrive
        # for the `else` to hand it back.
        assert resolved.predicate is None

    def test_a_spatial_arm_is_refused_before_the_plan_is_ever_built(self):
        with pytest.raises(ShapeError, match="state that as a filter instead"):
            _pipeline(
                {
                    "inputs": {"a": vector_input()},
                    "steps": [
                        {
                            "op": "case",
                            "input": "a",
                            "cases": [{"when": "grid_cell = 7", "then": "1"}],
                        }
                    ],
                    "output": {"format": "csv"},
                }
            )


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

    def test_a_case_arms_columns_survive_a_later_group(self):
        # The `when` is evaluated against the scanned frame, so every column it names has
        # to outlive the narrowing a later `group` closes -- even though `scenario_id` is
        # neither grouped by nor aggregated, and nothing else in the pipeline mentions it.
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {
                        "op": "case",
                        "input": "a",
                        "cases": [{"when": "scenario_id = 7", "then": "value * 2"}],
                        "else": "value",
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
                "output": {"format": "tif"},
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
        # A raster origin has no `predicate` to inspect at all -- its pushdown is band
        # selection, which `TestRasterBandPushdown` covers, and which a `project` in this
        # direction does NOT stop (a dimension conjunct declared after one still narrows
        # the bands, since projecting aggregates over `grid_cell` and leaves every
        # dimension alone). This just confirms the compile itself is unbothered.
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
            {"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}}
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
        # True only for a pipeline that stays LONG. A raster input's band list can only be
        # answered by one real `io.catalog.resolve_raster_bands` call (see
        # `TestRasterBandPushdown` below, which monkeypatches exactly that seam), and a
        # vector-to-raster `project` needs `resolve_vector_bands`, which does scan
        # (`TestVectorBandEnumeration`). Neither applies here, so nothing opens anything.
        import cpra.mp.data.analysis.io.catalog as io_catalog
        import cpra.mp.data.analysis.io.vector as io_vector

        def _boom(*args, **kwargs):
            raise AssertionError("compile_plan must never call scan_vector")

        monkeypatch.setattr(io_vector, "scan_vector", _boom)
        # `io/catalog.py` binds `scan_vector` at import, so patching the defining module
        # alone would not intercept the call that actually happens -- patch both.
        monkeypatch.setattr(io_catalog, "scan_vector", _boom)

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
    `path_predicate` is accepted and ignored: it only decides which files the real one
    opens, and returning every band is the superset it is allowed to return anyway.
    """

    def _fake(selection, *, base_path=None, path_predicate=None):
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
                "output": {"format": "tif"},
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
                "output": {"format": "tif"},
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
                "output": {"format": "tif"},
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
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
        assert compile_plan(pipeline).inputs["a"].point_sample_step is None


def _fake_raster_bands_per_variable(by_variable):
    """`_fake_resolve_raster_bands` for a pipeline with more than one raster input: `rows`
    per `selection.variable`, so two origins can be given genuinely different band lists.
    """

    def _fake(selection, *, base_path=None, path_predicate=None):
        rows = by_variable[selection.variable]
        keys = rows[0].keys() if rows else []
        return pl.DataFrame(
            {
                "path": [f"/fake/{selection.variable}-{i}.tif" for i in range(len(rows))],
                "band": list(range(1, len(rows) + 1)),
                **{key: [row[key] for row in rows] for key in keys},
            }
        )

    return _fake


class TestPointSampleThroughABandWideJoin:
    """A spatial `filter` sitting directly on a band-wide `join` records a
    `point_sample_step` for EVERY raster origin of that join -- the record
    `execution.execute` needs to read one pixel's window instead of every operand's whole
    raster (`analyses/sediment_dynamics.py` is the shape that pays for it).

    `_Validity` cannot say this on its own: it tracks a single origin, so a join retires
    every operand's state before the per-op branches run. What makes the record safe
    instead is that the operands are read as ONE scan
    (`execution.execute._open_band_wide_join_group`), and one scan takes one window -- so
    the group is exactly the unit a point sample can be sized for, provided nothing else
    is reading it.
    """

    OPERANDS: ClassVar[dict] = {"a": raster_input(variable="vct"), "b": raster_input(variable="ffibs_score")}
    JOIN: ClassVar[dict] = {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"}
    # The pixel and the fold that follows it -- `sediment_dynamics`' own shape, and the
    # only one that reaches a sink without throwing an operand away: a join's output
    # carries one value column per operand and every sink wants exactly one.
    SAMPLE: ClassVar[dict] = {"op": "filter", "input": "j", "where": "grid_cell = 7", "as": "s"}
    STACK: ClassVar[dict] = {"op": "unpivot", "input": "s"}

    @pytest.fixture(autouse=True)
    def _bands(self, monkeypatch):
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2030}, {"calendar_year": 2040}]),
        )

    def _compile(self, steps, fmt="csv"):
        # `fmt` follows the final frame's shape, which is what `validate_shapes` checks it
        # against: `csv` for a chain that ends folded and long, `tif` for one that never
        # leaves band-wide because nothing in it named a pixel.
        return compile_plan(
            _pipeline({"inputs": dict(self.OPERANDS), "steps": steps, "output": {"format": fmt}})
        )

    def _steps(self):
        return [dict(self.JOIN), dict(self.SAMPLE), dict(self.STACK)]

    def test_both_operands_record_the_filters_own_step(self):
        plan = self._compile(self._steps())
        assert plan.band_wide_join_scans == {"j": ("a", "b")}
        # 2 -- the `filter`'s own `step_no`, it being the second declared step.
        assert [plan.inputs[name].point_sample_step for name in ("a", "b")] == [2, 2]

    def test_the_sample_takes_both_operands_off_the_overviews(self):
        # The consequence that has to travel with the record, not a separate decision: an
        # overview's decimated grid can hand back a neighbouring pixel's value, which is
        # the wrong answer for a request that named one pixel. `compile_plan` reads
        # `point_sample_step` for this, so widening the record widens this too.
        plan = self._compile(self._steps())
        assert [plan.inputs[name].overview_permitted for name in ("a", "b")] == [False, False]

    def test_a_dimension_only_filter_on_the_join_records_nothing(self):
        # A dimension conjunct does not lower the frame and does not name a pixel; it is
        # band SELECTION, answered by the band-identity pass. Nothing to sample.
        plan = self._compile(
            [
                dict(self.JOIN),
                {"op": "filter", "input": "j", "where": "calendar_year >= 2030", "as": "narrowed"},
                {"op": "case", "input": "narrowed", "cases": [{"when": "a > 0.5", "then": "b"}]},
            ],
            fmt="tif",
        )
        assert plan.band_wide_join_scans == {"j": ("a", "b")}
        assert [plan.inputs[name].point_sample_step for name in ("a", "b")] == [None, None]

    def test_a_second_reader_of_the_joined_frame_blocks_it(self):
        # The narrowing applies to the shared READ, not to this filter's own frame, so a
        # second op off the same join would be handed one pixel where it asked for a
        # grid. A dangling second branch off `j` is legal (see
        # `TestBandWideJoinCeiling::test_an_operand_read_by_another_step_...` for the same
        # shape one level down) and is precisely the case that has to refuse.
        # Declared between the join and the sample rather than after the fold, only so the
        # LAST step stays the one the sink is typed against (`ops.final_frame_name`); where
        # in the chain it sits changes nothing about the reference count it adds.
        plan = self._compile(
            [
                dict(self.JOIN),
                {"op": "filter", "input": "j", "where": "calendar_year >= 2030", "as": "elsewhere"},
                dict(self.SAMPLE),
                dict(self.STACK),
            ]
        )
        assert [plan.inputs[name].point_sample_step for name in ("a", "b")] == [None, None]

    def test_an_operand_read_outside_the_join_blocks_it_too(self):
        # Same rule one level down: `a` feeds the join AND a branch of its own, and the
        # shared scan is where both of them get their pixels from.
        plan = self._compile(
            [
                {"op": "filter", "input": "a", "where": "calendar_year >= 2030", "as": "own_branch"},
                *self._steps(),
            ]
        )
        assert [plan.inputs[name].point_sample_step for name in ("a", "b")] == [None, None]

    def test_a_step_between_the_join_and_the_filter_is_left_unnarrowed(self):
        # Deliberate, and documented in the gate: a `calculate` here happens to leave the
        # pixels alone, but a `group` or a `project` would not, and telling them apart is
        # a forward pass no pipeline in this codebase needs. Reading whole is correct.
        plan = self._compile(
            [
                dict(self.JOIN),
                {"op": "calculate", "input": "j", "expression": "a + b", "as": "summed"},
                {"op": "filter", "input": "summed", "where": "grid_cell = 7"},
            ]
        )
        assert [plan.inputs[name].point_sample_step for name in ("a", "b")] == [None, None]


class TestDimensionPushdownAcrossJoinsAndFanOut:
    """A dimension conjunct reaches an origin's read no matter where in the pipeline it was
    declared -- across a join, across a fan-out, and from the other side of either.

    `_Validity` stops at both, and for value and spatial conjuncts that is the right
    answer. A dimension conjunct has two properties neither of those has: `join_long` is an
    inner join, so pushing one back through a single operand cannot change what the other
    operand's rows get matched against; and the `filter` step still re-executes over
    whatever came back, so a narrowing that is too WIDE is slow and nothing worse. The two
    passes these tests characterize are what turns that into band selection
    (`raster_band_metadata`) and into a `scan_vector` predicate.
    """

    _YEARS: ClassVar[list] = [{"calendar_year": 2030}, {"calendar_year": 2035}, {"calendar_year": 2040}]

    def _join_pipeline(self, steps):
        return {
            "inputs": {"a": raster_input(variable="ffibs_coverage_ratio"), "b": raster_input(variable="vct")},
            "steps": steps,
            "output": {"format": "tif"},
        }

    def test_a_conjunct_declared_after_a_join_narrows_both_operands(self, monkeypatch):
        monkeypatch.setattr(plan_module, "resolve_raster_bands", _fake_resolve_raster_bands(self._YEARS))
        plan = compile_plan(
            _pipeline(
                self._join_pipeline(
                    [
                        {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                        {"op": "filter", "input": "j", "where": "calendar_year >= 2035", "as": "d"},
                        {"op": "case", "input": "d", "cases": [{"when": "a > 0.5 AND b > 0.5", "then": "a"}]},
                    ]
                )
            )
        )
        for name in ("a", "b"):
            assert plan.inputs[name].band_metadata["calendar_year"].to_list() == [2035, 2040]

    def test_a_conjunct_on_one_operands_own_dimension_narrows_the_other_through_a_shared_one(
        self, monkeypatch
    ):
        # The case the symbolic pass cannot reach and the unified metadata frame exists
        # for: `ffibs_type` is a dimension only `a`'s dataset has, so nothing about it can
        # be handed to `b`'s read as a predicate. But the join pairs their bands on
        # `calendar_year`, so once `a` is down to 2030 the `b` bands for 2035 and 2040 are
        # bands nothing will ever look at.
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_raster_bands_per_variable(
                {
                    "ffibs_coverage_ratio": [
                        {"calendar_year": 2030, "ffibs_type": "fresh"},
                        {"calendar_year": 2035, "ffibs_type": "saline"},
                    ],
                    "vct": self._YEARS,
                }
            ),
        )
        plan = compile_plan(
            _pipeline(
                self._join_pipeline(
                    [
                        {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                        {"op": "filter", "input": "j", "where": "ffibs_type = 'fresh'", "as": "d"},
                        {"op": "case", "input": "d", "cases": [{"when": "a > 0.5 AND b > 0.5", "then": "a"}]},
                    ]
                )
            )
        )
        assert plan.inputs["a"].band_metadata["calendar_year"].to_list() == [2030]
        assert plan.inputs["b"].band_metadata["calendar_year"].to_list() == [2030]

    def test_a_band_with_no_counterpart_in_the_other_operand_is_not_dropped_by_the_pairing(
        self, monkeypatch
    ):
        # The unified frame is a FULL outer join, not an inner one, and this is where that
        # shows: `b`'s 2040 band pairs with no `a` band at all, so an inner join would drop
        # it for that reason alone -- making a trivially-true conjunct narrow the read.
        # `execution.band_ops.unify_metadata_parts` makes the same choice at execution, and
        # the two HAVE to agree: the `filter` step re-executes over whatever was read, so a
        # plan-time answer narrower than the execution-time one is simply wrong.
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_raster_bands_per_variable(
                {
                    "ffibs_coverage_ratio": [{"calendar_year": 2030}, {"calendar_year": 2035}],
                    "vct": self._YEARS,
                }
            ),
        )
        plan = compile_plan(
            _pipeline(
                self._join_pipeline(
                    [
                        {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                        {"op": "filter", "input": "j", "where": "calendar_year >= 2030", "as": "d"},
                        {"op": "case", "input": "d", "cases": [{"when": "a > 0.5 AND b > 0.5", "then": "a"}]},
                    ]
                )
            )
        )
        assert plan.inputs["a"].band_metadata["calendar_year"].to_list() == [2030, 2035]
        assert plan.inputs["b"].band_metadata["calendar_year"].to_list() == [2030, 2035, 2040]

    def test_a_fan_out_narrows_to_the_union_of_what_its_branches_need(self, monkeypatch):
        # One read, two consumers wanting different years: it has to carry both, and 2035
        # -- which neither asked for -- is still dropped. `_Validity` gives up here; a
        # union does not have to.
        monkeypatch.setattr(plan_module, "resolve_raster_bands", _fake_resolve_raster_bands(self._YEARS))
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {"a": raster_input()},
                    "steps": [
                        {"op": "filter", "input": "a", "where": "calendar_year = 2030", "as": "f1"},
                        {"op": "filter", "input": "a", "where": "calendar_year = 2040", "as": "f2"},
                        {"op": "join", "type": "spatial", "inputs": ["f1", "f2"], "as": "j"},
                        {"op": "case", "input": "j", "cases": [{"when": "f1 > 0 AND f2 > 0", "then": "f1"}]},
                    ],
                    "output": {"format": "tif"},
                }
            )
        )
        assert plan.inputs["a"].band_metadata["calendar_year"].to_list() == [2030, 2040]

    def test_branches_agreeing_on_a_predicate_narrow_to_that_predicate(self, monkeypatch):
        # The `lnd_type_plus` shape: the same input read twice under the same selection.
        # The union collapses to the one predicate rather than back to everything.
        monkeypatch.setattr(plan_module, "resolve_raster_bands", _fake_resolve_raster_bands(self._YEARS))
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {"a": raster_input()},
                    "steps": [
                        {"op": "filter", "input": "a", "where": "calendar_year = 2040", "as": "f1"},
                        {"op": "filter", "input": "a", "where": "calendar_year = 2040", "as": "f2"},
                        {"op": "join", "type": "spatial", "inputs": ["f1", "f2"], "as": "j"},
                        {"op": "case", "input": "j", "cases": [{"when": "f1 > 0 AND f2 > 0", "then": "f1"}]},
                    ],
                    "output": {"format": "tif"},
                }
            )
        )
        assert plan.inputs["a"].band_metadata["calendar_year"].to_list() == [2040]

    def test_a_conjunct_after_a_raster_to_vector_project_still_narrows_bands(self, monkeypatch):
        # Projecting aggregates over `grid_cell` and leaves every dimension alone, so band
        # *i* of the result is band *i* of the input -- a dimension conjunct declared after
        # one is as good a band selection as one declared before it.
        monkeypatch.setattr(plan_module, "resolve_raster_bands", _fake_resolve_raster_bands(self._YEARS))
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {"a": raster_input()},
                    "steps": [
                        {"op": "project", "input": "a", "to": "ecoregion_v001", "aggregate": "mean", "as": "p"},
                        {"op": "filter", "input": "p", "where": "calendar_year >= 2035"},
                    ],
                    "output": {"format": "csv"},
                }
            )
        )
        assert plan.inputs["a"].band_metadata["calendar_year"].to_list() == [2035, 2040]

    def test_a_dimension_a_group_collapsed_narrows_nothing(self, monkeypatch):
        # R2.4/R2.9: `by=[]` takes `calendar_year` out of the frame, so the conjunct below
        # names a column that is not there. Narrowing on it anyway would drop bands the
        # `group` was supposed to fold together -- a WRONG answer, not merely a slow one,
        # which is the one direction this may never err in.
        monkeypatch.setattr(plan_module, "resolve_raster_bands", _fake_resolve_raster_bands(self._YEARS))
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {"a": raster_input()},
                    "steps": [
                        {"op": "group", "input": "a", "by": [], "aggregate": "sum", "as": "g"},
                        {"op": "filter", "input": "g", "where": "calendar_year >= 2035"},
                    ],
                    "output": {"format": "tif"},
                }
            )
        )
        assert plan.inputs["a"].band_metadata["calendar_year"].to_list() == [2030, 2035, 2040]

    def test_a_dimension_one_operands_group_collapsed_narrows_only_the_other(self, monkeypatch):
        # Per-ORIGIN, not per-frame: the joined frame still carries `ffibs_type` -- from
        # `b` -- while `a`'s own side grouped it away, so the conjunct is `b`'s to answer
        # and `a` keeps every band.
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_raster_bands_per_variable(
                {
                    "vct": self._YEARS,
                    "ffibs_coverage_ratio": [
                        {"calendar_year": 2030, "ffibs_type": "fresh"},
                        {"calendar_year": 2035, "ffibs_type": "saline"},
                    ],
                }
            ),
        )
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {
                        "a": raster_input(variable="vct"),
                        "b": raster_input(variable="ffibs_coverage_ratio"),
                    },
                    "steps": [
                        {"op": "group", "input": "a", "by": ["grid_cell"], "aggregate": "sum", "as": "g"},
                        {"op": "join", "type": "spatial", "inputs": ["g", "b"], "as": "j"},
                        {"op": "filter", "input": "j", "where": "ffibs_type = 'fresh'", "as": "d"},
                        {"op": "case", "input": "d", "cases": [{"when": "g > 0.5 AND b > 0.5", "then": "g"}]},
                    ],
                    "output": {"format": "tif"},
                }
            )
        )
        assert plan.inputs["a"].band_metadata["calendar_year"].to_list() == [2030, 2035, 2040]
        assert plan.inputs["b"].band_metadata["calendar_year"].to_list() == [2030]

    def test_narrowing_to_no_bands_at_all_is_an_empty_result_not_an_error(self, monkeypatch):
        monkeypatch.setattr(plan_module, "resolve_raster_bands", _fake_resolve_raster_bands(self._YEARS))
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {"a": raster_input()},
                    "steps": [{"op": "filter", "input": "a", "where": "calendar_year = 1900"}],
                    "output": {"format": "tif"},
                }
            )
        )
        assert plan.inputs["a"].band_metadata.height == 0

    def test_a_pairing_that_leaves_one_operand_with_no_bands_still_keeps_it_one(self, monkeypatch):
        # The join-sibling floor. `a` has only 2030; `b` has 2030 and 2040; the conjunct
        # asks for 2040. Against the full outer pairing that is exactly right and entirely
        # unusable: `b`'s 2040 row pairs with no `a` row at all, so `a` survives with zero
        # bands while `b` keeps one -- and `join_band_wide` aligns POSITIONALLY, so a
        # zero-band operand beside a one-band one is the partial mismatch
        # `execute._open_band_wide_join_group` raises on.
        #
        # Reading nothing for `a` is not an option, so `a` keeps one band. Nothing about
        # the ANSWER changes: the `filter` step re-executes over the joined frame, where
        # the same pairing drops that band again. The e2e counterpart is
        # `test_execute.py::test_a_post_join_conjunct_that_empties_one_operand_still_runs`.
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_raster_bands_per_variable(
                {
                    "ffibs_coverage_ratio": [{"calendar_year": 2030}],
                    "vct": [{"calendar_year": 2030}, {"calendar_year": 2040}],
                }
            ),
        )
        plan = compile_plan(
            _pipeline(
                self._join_pipeline(
                    [
                        {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                        {"op": "filter", "input": "j", "where": "calendar_year = 2040", "as": "d"},
                        {"op": "case", "input": "d", "cases": [{"when": "a > 0.5 AND b > 0.5", "then": "a"}]},
                    ]
                )
            )
        )
        assert plan.inputs["a"].band_metadata["calendar_year"].to_list() == [2030]
        assert plan.inputs["b"].band_metadata["calendar_year"].to_list() == [2040]

    def test_an_operand_emptied_before_the_join_is_left_empty(self, monkeypatch):
        # The other side of that floor, and the reason it is conditioned on what the
        # operand carries INTO the join rather than on the final count: here the conjunct
        # is declared on `a`'s own side, so `a` genuinely enters the join with no bands.
        # Restoring one would not help -- the `filter` step re-derives the emptiness from
        # whatever was read -- and would only replace `_open_band_wide_join_group`'s named
        # error with `join_band_wide`'s anonymous shape mismatch. Pinned end-to-end by
        # `test_execute.py::test_a_partial_zero_band_join_operand_raises`.
        monkeypatch.setattr(plan_module, "resolve_raster_bands", _fake_resolve_raster_bands(self._YEARS))
        plan = compile_plan(
            _pipeline(
                self._join_pipeline(
                    [
                        {"op": "filter", "input": "a", "where": "calendar_year = 1900", "as": "fa"},
                        {"op": "join", "type": "spatial", "inputs": ["fa", "b"], "as": "j"},
                        {"op": "case", "input": "j", "cases": [{"when": "fa > 0.5 AND b > 0.5", "then": "fa"}]},
                    ]
                )
            )
        )
        assert plan.inputs["a"].band_metadata.height == 0
        assert plan.inputs["b"].band_metadata.height == 3

    def test_the_ceiling_is_checked_against_the_narrowed_count(self, monkeypatch):
        # R2.8: an input with more bands than the ceiling, filtered down under it, is a
        # plan that compiles -- the limit is on what gets READ, and the narrowing decided
        # what that is.
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2000 + i} for i in range(ANALYSIS_BAND_CEILING + 5)]),
        )
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {"a": raster_input()},
                    "steps": [{"op": "filter", "input": "a", "where": "calendar_year < 2010"}],
                    "output": {"format": "tif"},
                }
            )
        )
        assert plan.inputs["a"].band_metadata.height == 10


class TestPostJoinNarrowingAtProductionScale:
    """The size of the thing, on the shape `lnd_type_plus` actually has: an `elev` read and
    a `lnd_type` read on the same raster grid, joined, with the caller's selection declared
    ONCE after the join instead of repeated as a leading `filter` per input.

    Before the two dimension passes that selection reached neither read -- `_Validity` dies
    at the join -- so both origins were read whole and the narrowing happened in memory,
    after the I/O. These numbers are why the pushdown was worth building rather than a
    tidiness argument, and why an analysis no longer has to repeat its own selection per
    input to get a narrow read.
    """

    _GRID = "morph_pixel_v002"

    # 180 (model_group_id, scenario_id, calendar_year) combinations, sliced to the two
    # datasets' real band counts: `elev` 178, `lnd_type` 179, 357 together.
    _COMBINATIONS: ClassVar[list] = [
        {"model_group_id": model_group, "scenario_id": scenario, "calendar_year": year}
        for model_group in (1, 2)
        for scenario in (1, 2, 3)
        for year in range(2025, 2055)
    ]
    _SELECTION = "model_group_id = 1 AND scenario_id = 1 AND calendar_year = 2030"

    def _compile(self, monkeypatch, steps):
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_raster_bands_per_variable(
                {"elev": self._COMBINATIONS[:178], "lnd_type": self._COMBINATIONS[:179]}
            ),
        )
        return compile_plan(
            _pipeline(
                {
                    "inputs": {
                        "elev": raster_input(grid=self._GRID, variable="elev"),
                        "land": raster_input(grid=self._GRID, variable="lnd_type"),
                    },
                    "steps": steps,
                    "output": {"format": "tif"},
                }
            )
        )

    _JOIN: ClassVar[dict] = {"op": "join", "type": "spatial", "inputs": ["elev", "land"], "as": "j"}
    _SINK: ClassVar[dict] = {"op": "group", "by": [], "aggregate": "sum"}

    def test_with_nothing_declared_both_origins_are_read_whole(self, monkeypatch):
        plan = self._compile(monkeypatch, [self._JOIN, self._SINK])
        heights = {name: plan.inputs[name].band_metadata.height for name in ("elev", "land")}
        assert heights == {"elev": 178, "land": 179}
        assert sum(heights.values()) == 357

    def test_one_selection_declared_after_the_join_narrows_both_origins_to_one_band(self, monkeypatch):
        plan = self._compile(
            monkeypatch,
            [self._JOIN, {"op": "filter", "input": "j", "where": self._SELECTION, "as": "d"}, self._SINK],
        )
        heights = {name: plan.inputs[name].band_metadata.height for name in ("elev", "land")}
        assert heights == {"elev": 1, "land": 1}
        assert sum(heights.values()) == 2
        for name in ("elev", "land"):
            assert plan.inputs[name].band_metadata.row(0, named=True) == {
                "path": plan.inputs[name].band_metadata["path"][0],
                "band": 6,
                "model_group_id": 1,
                "scenario_id": 1,
                "calendar_year": 2030,
            }


class TestVectorDimensionPushdownAcrossJoins:
    """The same two rules on the vector side, where the pushdown is a `scan_vector`
    predicate rather than a band list. Checked behaviorally: `pl.Expr` has no structural
    equality, so what a predicate MEANS is the only thing there is to assert about it.
    """

    _YEARS = pl.DataFrame({"calendar_year": [2030, 2035, 2040]})

    def test_a_conjunct_declared_after_a_join_reaches_both_operands_scans(self):
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {"a": vector_input(variable="sal"), "b": vector_input(variable="stg")},
                    "steps": [
                        {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                        {"op": "filter", "input": "j", "where": "calendar_year >= 2035", "as": "d"},
                        {"op": "calculate", "input": "d", "expression": "a + b"},
                    ],
                    "output": {"format": "csv"},
                }
            )
        )
        for name in ("a", "b"):
            kept = self._YEARS.filter(plan.inputs[name].predicate)["calendar_year"].to_list()
            assert kept == [2035, 2040]

    def test_a_conjunct_naming_only_one_operands_dimension_reaches_only_that_one(self):
        # `aggregate_type` is a dimension of `sal`'s dataset and not of `vct`'s. Pushing it
        # to `b` would reach a column that is not in that scan at all.
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {
                        "a": vector_input(variable="sal"),
                        "b": vector_input(variable="vct", grid="veg_grid_cell_v001"),
                    },
                    "steps": [
                        {"op": "project", "input": "b", "to": VECTOR_GRID, "aggregate": "mean", "as": "p"},
                        {"op": "join", "type": "spatial", "inputs": ["a", "p"], "as": "j"},
                        {"op": "filter", "input": "j", "where": "aggregate_type = 'mean'", "as": "d"},
                        {"op": "calculate", "input": "d", "expression": "a + p"},
                    ],
                    "output": {"format": "csv"},
                }
            )
        )
        frame = pl.DataFrame({"aggregate_type": ["mean", "max"]})
        assert frame.filter(plan.inputs["a"].predicate)["aggregate_type"].to_list() == ["mean"]
        # And `b` got nothing at all -- not a predicate that happens to be satisfiable
        # without the column, which would still fail the moment its scan resolved it.
        assert plan.inputs["b"].predicate is None

    def test_a_fan_out_pushes_the_disjunction_of_what_its_branches_need(self):
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {"a": vector_input()},
                    "steps": [
                        {"op": "filter", "input": "a", "where": "calendar_year = 2030", "as": "f1"},
                        {"op": "filter", "input": "a", "where": "calendar_year = 2040", "as": "f2"},
                        {"op": "join", "type": "spatial", "inputs": ["f1", "f2"], "as": "j"},
                        {"op": "calculate", "input": "j", "expression": "f1 + f2"},
                    ],
                    "output": {"format": "csv"},
                }
            )
        )
        # Both branches' years survive and the year neither asked for does not -- the
        # disjunction, not "give up and read everything" and not either branch alone.
        kept = self._YEARS.filter(plan.inputs["a"].predicate)["calendar_year"].to_list()
        assert kept == [2030, 2040]

    def test_a_value_conjunct_declared_after_a_join_is_still_not_pushed(self):
        # R2.7: only DIMENSION conjuncts cross a join. A value conjunct on a joined frame
        # names an operand's renamed value column, which is not a source column at all.
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {"a": vector_input(variable="sal"), "b": vector_input(variable="stg")},
                    "steps": [
                        {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                        {"op": "case", "input": "j", "cases": [{"when": "a > 0.5 AND b > 0.5", "then": "a"}], "as": "d"},
                        {"op": "calculate", "input": "d", "expression": "a + b"},
                    ],
                    "output": {"format": "csv"},
                }
            )
        )
        assert plan.inputs["a"].predicate is None
        assert plan.inputs["b"].predicate is None

    def test_a_branch_nothing_reads_still_widens_the_read_it_shares(self):
        # `f2` feeds nothing -- but it is still a frame the plan will compute, off the one
        # scan `a` gets. Its year has to survive the read alongside the branch that does
        # reach the output; the union runs over every TERMINAL frame, not over the output's.
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {"a": vector_input()},
                    "steps": [
                        {"op": "filter", "input": "a", "where": "calendar_year = 2030", "as": "f1"},
                        {"op": "filter", "input": "a", "where": "calendar_year = 2040", "as": "f2"},
                        {"op": "calculate", "input": "f1", "expression": "value * 2"},
                    ],
                    "output": {"format": "csv"},
                }
            )
        )
        kept = self._YEARS.filter(plan.inputs["a"].predicate)["calendar_year"].to_list()
        assert kept == [2030, 2040]

    def test_a_projections_bands_narrow_from_the_raster_it_is_joined_with(self, monkeypatch):
        # The two passes meeting: `ffibs_type` is the raster's alone, so nothing about it
        # can be handed to the vector scan as a predicate -- but the unified frame pairs
        # the projection's bands with the raster's on `calendar_year`, and what survives
        # there becomes BOTH a narrower declared band list and an `IN`-list on the scan.
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands(
                [
                    {"calendar_year": 2030, "ffibs_type": "fresh"},
                    {"calendar_year": 2035, "ffibs_type": "saline"},
                ]
            ),
        )
        calls = []
        monkeypatch.setattr(
            plan_module,
            "resolve_vector_bands",
            _fake_resolve_vector_bands(
                [
                    {"calendar_year": 2030, "aggregate_type": "mean"},
                    {"calendar_year": 2035, "aggregate_type": "mean"},
                    {"calendar_year": 2040, "aggregate_type": "mean"},
                ],
                calls,
            ),
        )
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {"r": raster_input(), "v": vector_input()},
                    "steps": [
                        {"op": "project", "input": "v", "to": RASTER_GRID, "as": "p"},
                        {"op": "join", "type": "spatial", "inputs": ["r", "p"], "as": "j"},
                        {"op": "filter", "input": "j", "where": "ffibs_type = 'fresh'", "as": "d"},
                        {"op": "case", "input": "d", "cases": [{"when": "r > 0.5 AND p > 0.5", "then": "r"}]},
                    ],
                    "output": {"format": "tif"},
                }
            )
        )
        assert plan.inputs["r"].band_metadata["calendar_year"].to_list() == [2030]
        assert plan.vector_band_metadata[1]["calendar_year"].to_list() == [2030]
        # The enumeration itself ran BEFORE this narrowing was knowable -- it is what made
        # it knowable -- so the predicate it was made under carries none of it.
        assert calls[0]["predicate"] is None
        frame = pl.DataFrame({"calendar_year": [2030, 2035, 2040], "aggregate_type": ["mean"] * 3})
        narrowed = frame.filter(plan.inputs["v"].predicate)
        assert narrowed["calendar_year"].to_list() == [2030]
        # `aggregate_type` contributes no conjunct: every value it has survives, so an
        # `IN`-list over it would be a tautology on this scan.
        assert "aggregate_type" not in plan.inputs["v"].predicate.meta.root_names()

    def test_a_long_only_pipeline_with_no_join_is_unchanged(self):
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {"a": vector_input()},
                    "steps": [{"op": "filter", "input": "a", "where": "calendar_year = 2035"}],
                    "output": {"format": "csv"},
                }
            )
        )
        kept = self._YEARS.filter(plan.inputs["a"].predicate)["calendar_year"].to_list()
        assert kept == [2035]


class TestBandLimitError:
    def test_a_single_raster_input_over_the_ceiling_raises(self, monkeypatch):
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2000 + i} for i in range(ANALYSIS_BAND_CEILING + 1)]),
        )
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
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
                "output": {"format": "tif"},
            }
        )
        resolved = compile_plan(pipeline).inputs["a"]
        assert resolved.band_metadata.height == 1

    def _over_the_ceiling_sampled_at(self, monkeypatch, where):
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2000 + i} for i in range(ANALYSIS_BAND_CEILING + 1)]),
        )
        return _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "filter", "input": "a", "where": where}],
                "output": {"format": "csv"},
            }
        )

    def test_an_input_the_point_reader_takes_is_exempt(self, monkeypatch):
        """The ceiling bounds a band-wide read's VRT; the point reader builds none (the
        species stacked bar samples 47 species x 52 years of one cell)."""
        plan = compile_plan(self._over_the_ceiling_sampled_at(monkeypatch, "grid_cell IN (7, 8)"))
        assert point_sample_reads(plan) == {"a": [7, 8]}
        assert plan.inputs["a"].band_metadata.height == ANALYSIS_BAND_CEILING + 1

    def test_a_spatial_filter_the_point_reader_rejects_still_raises(self, monkeypatch):
        # An `OR` cannot be resolved to ids up front, so the read falls back to band-wide.
        pipeline = self._over_the_ceiling_sampled_at(monkeypatch, "grid_cell = 7 OR grid_cell = 8")
        with pytest.raises(BandLimitError):
            compile_plan(pipeline)


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
        # The value `filter` is what collapses the join's two operand-named value columns
        # back to one, which every output requires -- see `request.ops.validate_shapes`.
        # Nothing here depends on the predicate itself; these tests never execute.
        return {
            "inputs": self._two_raster_inputs(),
            "steps": [
                {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                {"op": "case", "input": "j", "cases": [{"when": "a > 0.5 AND b > 0.5", "then": "a"}]},
            ],
            "output": {"format": "tif"},
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
                {"op": "case", "input": "j", "cases": [{"when": "a > 0.5 AND b > 0.5", "then": "a"}], "as": "m"},
                {"op": "join", "type": "spatial", "inputs": ["m", "c"], "as": "j2"},
                {"op": "case", "input": "j2", "cases": [{"when": "m > 0.5 AND c > 0.5", "then": "m"}]},
            ],
            "output": {"format": "tif"},
        }
        # Pairwise, every join is under the ceiling; all three together are over it.
        assert 2 * third_over <= ANALYSIS_BAND_CEILING < 3 * third_over
        with pytest.raises(BandLimitError) as caught:
            compile_plan(_pipeline(request))
        # Both joins are sized against the same union, so which of the two the check
        # happens to reach first is not the point -- the three origins named are.
        assert "('a', 'b', 'c')" in str(caught.value)

    def test_a_join_of_a_join_records_the_transitive_scan_union(self, monkeypatch):
        """Both joins get the SAME union, not their own. `a` is an operand of each, and an
        origin cannot be read twice with two different band counts and still have its rows
        line up (`execution.frame.assert_same_pixel_grid`) -- so the inner join is read
        out of the outer join's scan too.
        """
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
                        {"op": "case", "input": "j", "cases": [{"when": "a > 0.5 AND b > 0.5", "then": "a"}], "as": "m"},
                        {"op": "join", "type": "spatial", "inputs": ["m", "c"], "as": "j2"},
                        {"op": "case", "input": "j2", "cases": [{"when": "m > 0.5 AND c > 0.5", "then": "m"}]},
                    ],
                    "output": {"format": "tif"},
                }
            )
        )
        assert plan.band_wide_join_scans == {"j": ("a", "b", "c"), "j2": ("a", "b", "c")}

    def test_two_joins_sharing_no_origin_stay_separate_scans(self, monkeypatch):
        """The fusion is per connected component, not global: joins over disjoint origins
        have nothing to align against each other, so each keeps its own read.
        """
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2030}, {"calendar_year": 2035}]),
        )
        inputs = self._two_raster_inputs()
        inputs["c"] = raster_input(variable="stg")
        inputs["d"] = raster_input(variable="sal")
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": inputs,
                    "steps": [
                        {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                        {"op": "case", "input": "j", "cases": [{"when": "a > 0.5 AND b > 0.5", "then": "a"}]},
                        {"op": "join", "type": "spatial", "inputs": ["c", "d"], "as": "j2"},
                        {"op": "case", "input": "j2", "cases": [{"when": "c > 0.5 AND d > 0.5", "then": "c"}]},
                    ],
                    "output": {"format": "tif"},
                }
            )
        )
        assert plan.band_wide_join_scans == {"j": ("a", "b"), "j2": ("c", "d")}

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


def _fake_resolve_vector_bands(rows, calls=None):
    """A stand-in for `io.catalog.resolve_vector_bands`, monkeypatched onto
    `plan_module.resolve_vector_bands` for exactly the reasons
    `_fake_resolve_raster_bands` exists -- the vector half of the same seam.

    `rows` is a list of `{dimension_name: value, ...}` dicts, one per enumerated band, or
    `None` for "that dataset has nothing on disk", which the real function answers with
    `None` too. `calls`, when given, is a list every call's keyword arguments are appended
    to, so a test can assert the enumeration was made under the SAME pushdown the scan
    gets -- the one property that cannot be checked from the returned frame.
    """

    def _fake(selection, *, base_path=None, columns=None, predicate=None, dimensions=None):
        if calls is not None:
            calls.append(
                {
                    "selection": selection,
                    "base_path": base_path,
                    "columns": columns,
                    "predicate": predicate,
                    "dimensions": dimensions,
                }
            )
        if rows is None:
            return None
        keys = rows[0].keys() if rows else []
        return pl.DataFrame({key: [row[key] for row in rows] for key in keys})

    return _fake


class TestVectorBandEnumeration:
    """`ExecutionPlan.vector_band_metadata` -- the bands a `("project", ("vector",
    "raster"))` step will produce, settled at compile time rather than discovered when the
    pivot runs (`execution.project.project_vector_to_raster`). A vector origin's rows
    become COLUMNS there, so its width is a plan-time fact for the same reason a raster
    input's is: nothing downstream can be sized, and no ceiling checked, by waiting.
    """

    def _project_pipeline(self, steps=None):
        return {
            "inputs": {"a": vector_input()},
            "steps": steps or [{"op": "project", "input": "a", "to": RASTER_GRID, "as": "p"}],
            "output": {"format": "tif"},
        }

    def test_a_projection_is_enumerated_and_keyed_by_its_step(self, monkeypatch):
        monkeypatch.setattr(
            plan_module,
            "resolve_vector_bands",
            _fake_resolve_vector_bands([{"calendar_year": 2030}, {"calendar_year": 2035}]),
        )
        plan = compile_plan(_pipeline(self._project_pipeline()))
        # Keyed by STEP, not by input: it is the `project` that turns rows into bands, and
        # the same input projected twice would resolve to two different widths.
        assert set(plan.vector_band_metadata) == {1}
        assert plan.vector_band_metadata[1]["calendar_year"].to_list() == [2030, 2035]

    def test_a_long_only_pipeline_issues_no_enumeration_query(self, monkeypatch):
        calls = []
        monkeypatch.setattr(plan_module, "resolve_vector_bands", _fake_resolve_vector_bands([], calls))
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "filter", "input": "a", "where": "value > 1", "as": "f"},
                    {"op": "group", "by": ["calendar_year"], "aggregate": "mean"},
                ],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline)
        # The gate is the presence of a vector-to-raster `project`, nothing subtler: a
        # pipeline that stays long pays nothing for a facility it never uses.
        assert calls == []
        assert plan.vector_band_metadata == {}

    def test_the_enumeration_is_made_under_the_scans_own_pushdown(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            plan_module, "resolve_vector_bands", _fake_resolve_vector_bands([{"calendar_year": 2030}], calls)
        )
        pipeline = _pipeline(
            self._project_pipeline(
                [
                    {"op": "filter", "input": "a", "where": "calendar_year = 2030", "as": "f"},
                    {"op": "project", "input": "f", "to": RASTER_GRID, "as": "p"},
                    {"op": "filter", "input": "p", "where": "value > 1"},
                ]
            )
        )
        plan = compile_plan(pipeline)
        resolved = plan.inputs["a"]

        assert len(calls) == 1
        assert calls[0]["selection"] is pipeline.inputs["a"]
        assert calls[0]["columns"] == resolved.columns
        # `pl.Expr` has no structural `==`, so the predicate is checked behaviorally, the
        # same way `TestVectorPredicatePushdown` checks the one that reaches the scan --
        # and against the same rows, since the point is that they are the same predicate.
        df = pl.DataFrame({"calendar_year": [2030, 2035], "value": [2, 2]})
        assert df.filter(calls[0]["predicate"])["calendar_year"].to_list() == [2030]
        assert df.filter(resolved.predicate)["calendar_year"].to_list() == [2030]

    def test_a_filter_declared_after_the_project_still_reaches_the_enumeration(self, monkeypatch):
        # A vector-to-raster project preserves `value` (a pixel's value IS its zone's), so
        # a value conjunct declared after it pushes back to the scan -- which is why the
        # enumeration runs once the walk is over, not at the `project` itself.
        calls = []
        monkeypatch.setattr(
            plan_module, "resolve_vector_bands", _fake_resolve_vector_bands([{"calendar_year": 2030}], calls)
        )
        compile_plan(
            _pipeline(
                self._project_pipeline(
                    [
                        {"op": "project", "input": "a", "to": RASTER_GRID, "as": "p"},
                        {"op": "filter", "input": "p", "where": "value > 1"},
                    ]
                )
            )
        )
        df = pl.DataFrame({"value": [0, 2]})
        assert df.filter(calls[0]["predicate"])["value"].to_list() == [2]

    def test_an_upstream_group_narrows_the_dimensions_passed(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            plan_module, "resolve_vector_bands", _fake_resolve_vector_bands([{"calendar_year": 2030}], calls)
        )
        compile_plan(
            _pipeline(
                self._project_pipeline(
                    [
                        {
                            "op": "group",
                            "input": "a",
                            "by": ["grid_cell", "calendar_year"],
                            "aggregate": "mean",
                            "as": "g",
                        },
                        {"op": "project", "input": "g", "to": RASTER_GRID, "as": "p"},
                    ]
                )
            )
        )
        # Everything the `group` collapsed is gone from the frame the pivot sees, so it
        # cannot split bands either -- `dimensions` is what tells the enumeration that.
        # `grid_cell` is not among them: what is passed is the frame's surviving DIMENSION
        # columns (`by` intersected with the dataset's own), and `grid_cell` is the row's
        # address, not one of them.
        assert calls[0]["dimensions"] == frozenset({"calendar_year"})

    def test_no_group_leaves_every_dimension_open(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            plan_module, "resolve_vector_bands", _fake_resolve_vector_bands([{"calendar_year": 2030}], calls)
        )
        compile_plan(_pipeline(self._project_pipeline()))
        # `None`, not "all of them spelled out": only a `group` narrows this, and the plan
        # does not otherwise enumerate what a dataset's dimensions are.
        assert calls[0]["dimensions"] is None

    def test_a_projection_off_a_join_is_not_enumerated(self, monkeypatch):
        calls = []
        monkeypatch.setattr(plan_module, "resolve_vector_bands", _fake_resolve_vector_bands([], calls))
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {"a": vector_input(), "b": vector_input(variable="stg")},
                    "steps": [
                        {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                        {"op": "calculate", "expression": "a + b", "as": "c"},
                        {"op": "project", "input": "c", "to": RASTER_GRID, "as": "p"},
                    ],
                    "output": {"format": "tif"},
                }
            )
        )
        # The join retires both origins' pushdown state, so there is no single input whose
        # rows this projection's bands correspond to -- and no honest query to ask.
        assert calls == []
        assert plan.vector_band_metadata == {}

    def test_a_dataset_with_nothing_on_disk_is_simply_absent(self, monkeypatch):
        monkeypatch.setattr(plan_module, "resolve_vector_bands", _fake_resolve_vector_bands(None))
        plan = compile_plan(_pipeline(self._project_pipeline()))
        # Compiling a plan against a dataset that has not been written yet stays possible;
        # the step is recorded as unenumerated rather than as zero bands wide.
        assert plan.vector_band_metadata == {}


class TestVectorBandCeiling:
    """The band ceiling applied to the vector half. Projected bands are never READ -- they
    are pivoted out of a long frame in memory -- but they are columns of the resulting
    band-wide frame exactly like a raster input's are, so they are counted the same way.
    """

    def test_an_over_wide_standalone_projection_raises(self, monkeypatch):
        monkeypatch.setattr(
            plan_module,
            "resolve_vector_bands",
            _fake_resolve_vector_bands([{"calendar_year": 2000 + i} for i in range(ANALYSIS_BAND_CEILING + 1)]),
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [{"op": "project", "input": "a", "to": RASTER_GRID, "as": "p"}],
                "output": {"format": "tif"},
            }
        )
        with pytest.raises(BandLimitError, match="when projected onto"):
            compile_plan(pipeline)

    def test_a_projection_at_the_ceiling_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(
            plan_module,
            "resolve_vector_bands",
            _fake_resolve_vector_bands([{"calendar_year": 2000 + i} for i in range(ANALYSIS_BAND_CEILING)]),
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [{"op": "project", "input": "a", "to": RASTER_GRID, "as": "p"}],
                "output": {"format": "tif"},
            }
        )
        assert compile_plan(pipeline).vector_band_metadata[1].height == ANALYSIS_BAND_CEILING

    def _mixed_join_pipeline(self):
        # The shape of `analyses/sal_veg_investigation.py`: a raster input joined
        # against a vector one projected onto the same grid.
        return {
            "inputs": {"r": raster_input(), "v": vector_input()},
            "steps": [
                {"op": "project", "input": "v", "to": RASTER_GRID, "as": "p"},
                {"op": "join", "type": "spatial", "inputs": ["r", "p"], "as": "j"},
                {"op": "case", "input": "j", "cases": [{"when": "r > 0.5 AND p > 0.5", "then": "r"}]},
            ],
            "output": {"format": "tif"},
        }

    def test_a_join_sums_the_raster_and_the_projected_halves(self, monkeypatch):
        half_over = ANALYSIS_BAND_CEILING // 2 + 1
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2000 + i} for i in range(half_over)]),
        )
        monkeypatch.setattr(
            plan_module,
            "resolve_vector_bands",
            _fake_resolve_vector_bands([{"calendar_year": 2000 + i} for i in range(half_over)]),
        )
        with pytest.raises(BandLimitError) as caught:
            compile_plan(_pipeline(self._mixed_join_pipeline()))
        message = str(caught.value)
        # Neither half is over the ceiling alone; the message has to name both, and name
        # the projection by the STEP that produces it, since it has no input name of its own.
        assert "('r',)" in message
        assert "step(s) (1,)" in message
        assert f"{2 * half_over} bands" in message

    def test_a_join_whose_halves_fit_together_does_not_raise(self, monkeypatch):
        half = ANALYSIS_BAND_CEILING // 2
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2000 + i} for i in range(half)]),
        )
        monkeypatch.setattr(
            plan_module,
            "resolve_vector_bands",
            _fake_resolve_vector_bands([{"calendar_year": 2000 + i} for i in range(half)]),
        )
        plan = compile_plan(_pipeline(self._mixed_join_pipeline()))
        assert plan.vector_band_metadata[1].height == half

    def test_a_projection_inside_a_join_is_not_also_checked_standalone(self, monkeypatch):
        """The join already counted this step, alongside the raster bands it shares a
        frame with. Checking it a second time on its own could only ever be redundant --
        or, worse, blame the projection for a width the join is responsible for.
        """
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2030}]),
        )
        monkeypatch.setattr(
            plan_module,
            "resolve_vector_bands",
            _fake_resolve_vector_bands([{"calendar_year": 2000 + i} for i in range(ANALYSIS_BAND_CEILING)]),
        )
        with pytest.raises(BandLimitError) as caught:
            compile_plan(_pipeline(self._mixed_join_pipeline()))
        assert "(join)" in str(caught.value)
        assert "when projected onto" not in str(caught.value)

    def test_an_unenumerated_projection_contributes_nothing_to_the_join_total(self, monkeypatch):
        """A step whose bands could not be resolved is an unknown, not a zero. Nothing
        here can size it, so the check does what it can with the halves it does know --
        and says so by naming no projection step at all.
        """
        monkeypatch.setattr(
            plan_module,
            "resolve_raster_bands",
            _fake_resolve_raster_bands([{"calendar_year": 2000 + i} for i in range(ANALYSIS_BAND_CEILING + 1)]),
        )
        monkeypatch.setattr(plan_module, "resolve_vector_bands", _fake_resolve_vector_bands(None))
        with pytest.raises(BandLimitError) as caught:
            compile_plan(_pipeline(self._mixed_join_pipeline()))
        assert "plus the vector projection" not in str(caught.value)


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

    def test_two_conflicting_project_targets_off_one_raster_origin_raises(self, monkeypatch):
        # The rejected counterpart to the fan-out test above: here BOTH branches off "a"
        # reach a raster-to-vector `project`, targeting two DIFFERENT destination grids
        # (`VECTOR_GRID` and `ecoregion_v001`). A raster origin's full-resolution pixels
        # can only be read once per compiled plan (one `scan_raster` call, one crosswalk
        # attached to it), so this has no single answer for "which crosswalk does 'a'
        # open with" -- `compile_plan` rejects it outright, at compile time, rather than
        # picking one target arbitrarily (`setdefault`-style) and silently ignoring the
        # other `project`'s destination.
        monkeypatch.setattr(
            plan_module, "resolve_raster_bands", _fake_resolve_raster_bands([{"calendar_year": 2030}])
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [
                    {"op": "calculate", "input": "a", "expression": "value * 2", "as": "c1"},
                    {"op": "calculate", "input": "a", "expression": "value * 3", "as": "c2"},
                    {"op": "project", "input": "c1", "to": VECTOR_GRID, "aggregate": "sum", "as": "p1"},
                    {"op": "project", "input": "c2", "to": "ecoregion_v001", "aggregate": "sum"},
                ],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(ShapeError, match="straddles two different destination grids"):
            compile_plan(pipeline)

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
                    {"op": "case", "input": "j", "cases": [{"when": "b > 0.5 AND c > 0.5", "then": "b"}]},
                ],
                "output": {"format": "tif"},
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
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
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


def _tile_over_the_raster_grid(zoom=12):
    """A tile covering the middle of `RASTER_GRID`'s extent, so every window below is a
    strict sub-rectangle of that grid rather than a clipped whole-extent read."""
    grid = GRIDS[RASTER_GRID]
    west, south, east, north = transform_bounds(grid.crs, "EPSG:4326", *grid.bounds)
    return mercantile.tile((west + east) / 2, (south + north) / 2, zoom)


class TestTileReadWindow:
    """`compile_plan(..., tile=...)` -- which pixels of each raster origin one map tile
    needs, decided here so the executor's one walk can run against them and the sink is left
    with nothing to do but warp.

    Sizing a window opens no raster: `io.raster_io.raster_grid_geometry` supplies the four
    properties off the grid registry instead, and `tests/api/test_tile_warp.py::
    TestRasterGridGeometryStandsInForAnOpenedRaster` is what pins those to a real file's.
    The straddling-widening case below is the one place this file's no-filesystem property
    genuinely bends -- the widened window is a union with real polygon bounds, so there has
    to be a geometry file to read them from -- and it uses the `geometry_path` fixture's
    sandbox, not any configured directory.
    """

    def test_no_tile_means_read_every_origin_whole(self, monkeypatch):
        monkeypatch.setattr(
            plan_module, "resolve_raster_bands", _fake_resolve_raster_bands([{"calendar_year": 2030}])
        )
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
        plan = compile_plan(pipeline)
        assert plan.tile is None
        assert plan.inputs["a"].read_window is None
        assert plan.inputs["a"].tile_disjoint is False

    def test_a_tile_narrows_the_window_to_a_sub_rectangle(self, monkeypatch):
        monkeypatch.setattr(
            plan_module, "resolve_raster_bands", _fake_resolve_raster_bands([{"calendar_year": 2030}])
        )
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
        tile = TileRequest(*_tile_over_the_raster_grid())
        plan = compile_plan(pipeline, tile=tile)

        grid = GRIDS[RASTER_GRID]
        window = plan.inputs["a"].read_window
        assert plan.tile is tile
        assert plan.inputs["a"].tile_disjoint is False
        assert int(window.width) < grid.width
        assert int(window.height) < grid.height
        assert 0 <= int(window.col_off) and int(window.col_off) + int(window.width) <= grid.width
        assert 0 <= int(window.row_off) and int(window.row_off) + int(window.height) <= grid.height

    def test_a_tile_that_misses_the_grid_is_recorded_as_disjoint(self, monkeypatch):
        # The opposite z=1 quadrant -- eastern/southern hemisphere, nowhere near this Gulf
        # Coast grid. `tile_disjoint` rather than a zero-sized window, because `None` on
        # `read_window` already means "read it whole" and a miss is the opposite of that.
        monkeypatch.setattr(
            plan_module, "resolve_raster_bands", _fake_resolve_raster_bands([{"calendar_year": 2030}])
        )
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
        plan = compile_plan(pipeline, tile=TileRequest(1, 1, 1))
        assert plan.inputs["a"].tile_disjoint is True
        assert plan.inputs["a"].read_window is None

    def test_an_origin_feeding_a_raster_to_vector_project_reads_a_wider_window(
        self, monkeypatch, geometry_path
    ):
        # The decision this whole arrangement exists to make reachable. "a" feeds a project
        # onto VECTOR_GRID, whose one feature here extends 3 pixels west of the tile's own
        # footprint; every pixel under that feature has to be read or the aggregate silently
        # loses the part on the far side of the tile edge. "b" is the control: the same grid,
        # the same tile, no project, so it gets the unwidened window and the comparison
        # below distinguishes "the widening ran" from "this tile is just large".
        from shapely.geometry import box

        monkeypatch.setattr(
            plan_module, "resolve_raster_bands", _fake_resolve_raster_bands([{"calendar_year": 2030}])
        )
        tile = TileRequest(*_tile_over_the_raster_grid())
        grid = GRIDS[RASTER_GRID]
        # veg_grid_cell_v001's pixels are exactly 480m square, so a feature extended by a
        # whole multiple of 480m widens the window by an exact, predictable pixel count.
        src_bounds = transform_bounds(TILE_CRS, grid.crs, *tile.bounds, densify_pts=21)
        extension_px = 3
        feature = box(src_bounds[0] - extension_px * 480, src_bounds[1], src_bounds[2], src_bounds[3])
        write_grid_geometry(geometry_path, VECTOR_GRID, [(1, feature)])

        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(), "b": raster_input(variable="sal")},
                "steps": [
                    {"op": "project", "input": "a", "to": VECTOR_GRID, "aggregate": "sum", "as": "p"},
                    {"op": "calculate", "input": "b", "expression": "value * 2", "as": "c"},
                ],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline, tile=tile)
        widened = plan.inputs["a"].read_window
        plain = plan.inputs["b"].read_window

        assert plan.inputs["a"].straddling_vector_grid == VECTOR_GRID
        assert plan.inputs["b"].straddling_vector_grid is None
        assert int(widened.col_off) == int(plain.col_off) - extension_px
        assert int(widened.width) == int(plain.width) + extension_px
        assert (int(widened.row_off), int(widened.height)) == (int(plain.row_off), int(plain.height))

    def test_a_vector_origin_gets_no_window(self):
        # Tile scoping is a raster read decision. A vector origin's rows are selected by
        # `grid_cell`, not by pixel offsets, so it carries the defaults untouched -- which
        # also means a tile-scoped plan mixing the two does not have to special-case it.
        pipeline = _pipeline({"inputs": {"a": vector_input()}, "steps": [], "output": {"format": "csv"}})
        plan = compile_plan(pipeline, tile=TileRequest(*_tile_over_the_raster_grid()))
        assert plan.inputs["a"].read_window is None
        assert plan.inputs["a"].tile_disjoint is False
        assert plan.inputs["a"].decimation == (1, 1)


#: A tile small enough that `veg_grid_cell_v001`'s windows clear the decimation threshold.
#:
#: The factors are sized against `oversample * (tile_size + 2 * buffer_px)` source pixels,
#: which at the production 256px tile is 528 -- more than this 1052x365 grid's short axis,
#: and only just under its long one, so nothing on it ever decimates at that size. A 16px
#: tile puts the same threshold at 48 and makes the decision visible on the fixture grid
#: these tests already use, rather than requiring a grid two orders of magnitude larger.
_DECIMATING_TILE_SIZE = 16


def _expected_factors(window, tile_size=_DECIMATING_TILE_SIZE):
    """What `_resolve_tile_window` should have chosen for `window`, derived the same way it
    does -- against the buffer `tile_source_window` actually widened by, not the bare
    default."""
    return tile_decimation_factors(window, tile_size, buffer_px=resampling_buffer_px(Resampling.nearest))


class TestTileDecimation:
    """`ResolvedInput.decimation` -- the resolution each raster origin's tile window may be
    read at, which is what `overview_permitted` has always been computed to decide.

    The window is unaffected either way: a decimated read covers the same ground, so only
    how many pixels come back changes.
    """

    def test_no_tile_means_full_resolution(self, monkeypatch):
        monkeypatch.setattr(
            plan_module, "resolve_raster_bands", _fake_resolve_raster_bands([{"calendar_year": 2030}])
        )
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
        assert compile_plan(pipeline).inputs["a"].decimation == (1, 1)

    def test_a_disjoint_tile_is_full_resolution(self, monkeypatch):
        # Nothing to read, so nothing to decimate -- and the zero-band frame this becomes
        # is compared field by field against its join partners, which must not see a factor
        # no read ever applied.
        monkeypatch.setattr(
            plan_module, "resolve_raster_bands", _fake_resolve_raster_bands([{"calendar_year": 2030}])
        )
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
        plan = compile_plan(pipeline, tile=TileRequest(1, 1, 1))
        assert plan.inputs["a"].tile_disjoint is True
        assert plan.inputs["a"].decimation == (1, 1)

    def test_an_eligible_origin_is_decimated_to_the_tile_read_rule(self, monkeypatch):
        monkeypatch.setattr(
            plan_module, "resolve_raster_bands", _fake_resolve_raster_bands([{"calendar_year": 2030}])
        )
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
        tile = TileRequest(*_tile_over_the_raster_grid(zoom=6), tile_size=_DECIMATING_TILE_SIZE)
        resolved = compile_plan(pipeline, tile=tile).inputs["a"]

        assert resolved.overview_permitted is True
        assert resolved.decimation == _expected_factors(resolved.read_window)
        # Not vacuously equal: this tile really is one the rule decimates.
        assert resolved.decimation != (1, 1)

    def test_a_raster_to_vector_project_origin_stays_full_resolution(self, monkeypatch, geometry_path):
        """The rule the whole eligibility computation exists for, now enforced rather than
        recorded: "a" feeds a project and reads every source pixel, while "b" -- same grid,
        same tile, no project -- is decimated. The control is what separates "the gate
        works" from "this tile was too small to decimate anyway".
        """
        from shapely.geometry import box

        monkeypatch.setattr(
            plan_module, "resolve_raster_bands", _fake_resolve_raster_bands([{"calendar_year": 2030}])
        )
        tile = TileRequest(*_tile_over_the_raster_grid(zoom=6), tile_size=_DECIMATING_TILE_SIZE)
        grid = GRIDS[RASTER_GRID]
        write_grid_geometry(
            geometry_path, VECTOR_GRID, [(1, box(*transform_bounds(TILE_CRS, grid.crs, *tile.bounds)))]
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(), "b": raster_input(variable="sal")},
                "steps": [
                    {"op": "project", "input": "a", "to": VECTOR_GRID, "aggregate": "sum", "as": "p"},
                    {"op": "calculate", "input": "b", "expression": "value * 2", "as": "c"},
                ],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline, tile=tile)

        assert plan.inputs["a"].decimation == (1, 1)
        assert plan.inputs["b"].decimation == _expected_factors(plan.inputs["b"].read_window)
        assert plan.inputs["b"].decimation != (1, 1)

    def test_a_point_sample_origin_stays_full_resolution(self, monkeypatch):
        # The other half of the gate: an overview pixel is a real value, but not necessarily
        # the one under the sampled coordinate.
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
        tile = TileRequest(*_tile_over_the_raster_grid(zoom=6), tile_size=_DECIMATING_TILE_SIZE)
        resolved = compile_plan(pipeline, tile=tile).inputs["a"]

        assert resolved.overview_permitted is False
        assert resolved.decimation == (1, 1)

    def test_a_production_sized_tile_over_this_grid_decimates_nothing(self, monkeypatch):
        # The threshold is real, not a formality: at 256px the target is 528 source pixels,
        # which this grid never exceeds by a factor of two on either axis. A rule that
        # decimated here would be decimating reads that are already tile-sized.
        monkeypatch.setattr(
            plan_module, "resolve_raster_bands", _fake_resolve_raster_bands([{"calendar_year": 2030}])
        )
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
        tile = TileRequest(*_tile_over_the_raster_grid(zoom=6))
        assert compile_plan(pipeline, tile=tile).inputs["a"].decimation == (1, 1)
