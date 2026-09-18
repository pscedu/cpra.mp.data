"""Tests for `cpra.mp.data.analysis.execution.plan.compile_plan` -- turning an
already-validated `Pipeline` into the `ExecutionPlan` `execution.execute.execute`
consumes.

Like `test_request_ops.py` (whose grid/variable fixtures this file reuses), these tests
never open a file or a crosswalk: `compile_plan` needs only the declared pipeline and each
input's grid kind (`cpra.mp.data.grids.GRIDS[...].grid_type`), the same no-IO property
`request.ops.validate_shapes` already has.
"""

import pytest

from cpra.mp.data.analysis.execution.plan import ExecutionPlan, ResolvedInput, compile_plan
from cpra.mp.data.analysis.request.ops import Shape, validate_shapes
from cpra.mp.data.analysis.request.pipeline import parse_pipeline
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

    def test_pushdown_is_not_computed_yet(self):
        # The seam: `columns`/`predicate` are always `None` in this vertical slice -- see
        # `execution/plan.py`'s module docstring for why that is deliberate, not an
        # oversight, and what fills them in later.
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [{"op": "filter", "input": "a", "where": "value > 1"}],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline)
        assert plan.inputs["a"].columns is None
        assert plan.inputs["a"].predicate is None


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
    def test_compile_plan_never_touches_the_filesystem(self, monkeypatch):
        # `compile_plan` takes no `base_path` at all -- there is structurally nothing it
        # could scan with. Pinning that `io.vector.scan_vector` is never even imported by
        # way of being called makes the "opens no data files" claim in the module
        # docstring a checked fact rather than an assertion resting on the signature
        # alone.
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
