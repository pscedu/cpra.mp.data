"""Tests for the request-layer op IR and shape typing
(`cpra.mp.data.analysis.request.ops`) -- `compile_ops`, `validate_shapes`, and the two
declarative lowering tables (`FILTER_LOWERING_TABLE`/`PROJECT_LOWERING_TABLE`) they read.

These tests never open a file or a crosswalk -- shape typing needs only the declared ops
and each input's grid kind (`cpra.mp.data.grids.GRIDS[...].grid_type`), which is exactly
what keeps shape typing a request-layer, no-IO concern. Real grids/variables from `cpra/mp/data/
structure/` are used throughout, mirroring `test_request_pipeline.py`'s convention.
"""

import pytest

from cpra.mp.data.analysis.request.errors import ShapeError
from cpra.mp.data.analysis.request.ops import (
    FILTER_LOWERING_TABLE,
    PROJECT_LOWERING_TABLE,
    CalculateOp,
    FilterOp,
    GroupOp,
    JoinOp,
    ProjectOp,
    Shape,
    compile_ops,
    validate_shapes,
)
from cpra.mp.data.analysis.request.expressions import PredicateKind
from cpra.mp.data.analysis.request.pipeline import parse_pipeline
from cpra.mp.data.grids import GRIDS

VECTOR_GRID = "hydro_compartment_v001"
VECTOR_GRID_2 = "ecoregion_v001"
VECTOR_GRID_3 = "block_group_v2020"
RASTER_GRID = "veg_grid_cell_v001"
RASTER_GRID_2 = "morph_pixel_v001"

assert GRIDS[VECTOR_GRID].grid_type == "vector"
assert GRIDS[VECTOR_GRID_2].grid_type == "vector"
assert GRIDS[VECTOR_GRID_3].grid_type == "vector"
assert GRIDS[RASTER_GRID].grid_type == "raster"
assert GRIDS[RASTER_GRID_2].grid_type == "raster"


def vector_input(grid=VECTOR_GRID, variable="sal"):
    return {"variable": variable, "time_unit": "annual", "grid": grid}


def raster_input(grid=RASTER_GRID, variable="ffibs_coverage_ratio"):
    return {"variable": variable, "time_unit": "annual", "grid": grid}


class TestProjectDirectionRules:
    def test_vector_to_raster_forbids_aggregate(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [{"op": "project", "input": "a", "to": RASTER_GRID, "aggregate": "sum"}],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(ShapeError, match="broadcasts"):
            validate_shapes(pipeline)

    def test_vector_to_raster_with_no_aggregate_accepted(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [{"op": "project", "input": "a", "to": RASTER_GRID}],
                "output": {"format": "tif"},
            }
        )
        validate_shapes(pipeline)  # does not raise

    def test_vector_to_vector_requires_aggregate(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [{"op": "project", "input": "a", "to": VECTOR_GRID_2}],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(ShapeError, match="requires an aggregate"):
            validate_shapes(pipeline)

    def test_raster_to_vector_requires_aggregate(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "project", "input": "a", "to": VECTOR_GRID}],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(ShapeError, match="requires an aggregate"):
            validate_shapes(pipeline)

    def test_raster_to_vector_with_aggregate_accepted(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "project", "input": "a", "to": VECTOR_GRID, "aggregate": "mean"}],
                "output": {"format": "csv"},
            }
        )
        validate_shapes(pipeline)

    def test_raster_to_raster_rejected(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "project", "input": "a", "to": RASTER_GRID_2, "aggregate": "sum"}],
                "output": {"format": "tif"},
            }
        )
        with pytest.raises(ShapeError, match="raster-to-raster"):
            validate_shapes(pipeline)

    def test_raster_to_raster_rejected_even_with_no_aggregate_given(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "project", "input": "a", "to": RASTER_GRID_2}],
                "output": {"format": "tif"},
            }
        )
        with pytest.raises(ShapeError, match="raster-to-raster"):
            validate_shapes(pipeline)


class TestSpatialFilterKeepsGridKind:
    """The correctness-critical case: a spatial `filter(grid_cell)` on a band-wide frame
    turns its SHAPE long, but must not change which grid (or grid KIND) the frame is
    still addressed against -- a later `project` off of it is still judged by that
    original grid's kind, not by the frame's current (now-long) shape.
    """

    def test_accepts_spatial_filter_then_project_vector_to_raster_to_tif(self):
        """The deliberate ACCEPT case: `filter(grid_cell)` on a vector
        input, then `project` vector-to-raster, sinks to `tif` -- must not be caught by
        the sink's long-shape-under-raster-format rule, since the spatial filter turns
        the frame long before the project turns it band-wide again."""
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "filter", "input": "a", "where": "grid_cell = 'abc'"},
                    {"op": "project", "to": RASTER_GRID},
                ],
                "output": {"format": "tif"},
            }
        )
        validate_shapes(pipeline)

    def test_rejects_raster_to_raster_project_after_a_spatial_filter_turned_it_long(self):
        """Without the grid-kind-not-shape fix, this would be wrongly accepted: after
        `filter(grid_cell)`, `current.shape == Shape.long`, but the frame is still tied
        to a RASTER grid -- a project to a second raster grid must still be rejected."""
        pipeline = parse_pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [
                    {"op": "filter", "input": "a", "where": "grid_cell = 'abc'"},
                    {"op": "project", "to": RASTER_GRID_2},
                ],
                "output": {"format": "parquet"},
            }
        )
        with pytest.raises(ShapeError, match="raster-to-raster"):
            validate_shapes(pipeline)


class TestCountByValue:
    def test_count_by_value_legal_via_project(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [{"op": "project", "input": "a", "to": VECTOR_GRID_2, "aggregate": "count_by_value"}],
                "output": {"format": "csv"},
            }
        )
        validate_shapes(pipeline)

    def test_second_count_by_value_in_one_chain_rejected(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "project", "input": "a", "to": VECTOR_GRID_2, "aggregate": "count_by_value", "as": "b"},
                    {"op": "project", "input": "b", "to": VECTOR_GRID_3, "aggregate": "count_by_value"},
                ],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(ShapeError, match="second count_by_value"):
            validate_shapes(pipeline)

    def test_second_count_by_value_across_a_join_rejected(self):
        """`has_count_by_value` OR-propagates through a join -- a count_by_value on one
        joined side still collides with a count_by_value on the other."""
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input(), "b": vector_input()},
                "steps": [
                    {"op": "project", "input": "a", "to": VECTOR_GRID_2, "aggregate": "count_by_value", "as": "ca"},
                    {"op": "project", "input": "b", "to": VECTOR_GRID_2, "aggregate": "count_by_value", "as": "cb"},
                    {"op": "join", "type": "spatial", "inputs": ["ca", "cb"], "as": "joined"},
                    {"op": "project", "input": "joined", "to": VECTOR_GRID_3, "aggregate": "count_by_value"},
                ],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(ShapeError, match="second count_by_value"):
            validate_shapes(pipeline)


class TestCountByValueCalculateOrdering:
    """There is deliberately NO rejection of `count_by_value` combined
    with `calculate` in either order -- both orderings must pass full shape typing, not
    merely structural parsing (see `test_request_pipeline.py::TestCalculateStep` for the
    structural-only counterparts of these two)."""

    def test_calculate_then_count_by_value_project_accepted(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "calculate", "input": "a", "expression": "value * 2", "as": "scaled"},
                    {"op": "project", "input": "scaled", "to": VECTOR_GRID_2, "aggregate": "count_by_value"},
                ],
                "output": {"format": "csv"},
            }
        )
        validate_shapes(pipeline)

    def test_count_by_value_project_then_calculate_accepted(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "project", "input": "a", "to": VECTOR_GRID_2, "aggregate": "count_by_value", "as": "counted"},
                    {"op": "calculate", "input": "counted", "expression": "value * 2"},
                ],
                "output": {"format": "csv"},
            }
        )
        validate_shapes(pipeline)


class TestJoinShapeRules:
    def test_long_and_band_wide_join_rejected(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input(), "b": raster_input()},
                "steps": [{"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"}],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(ShapeError, match="long frame with a band-wide"):
            validate_shapes(pipeline)

    def test_band_wide_join_across_two_different_grids_rejected(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": raster_input(grid=RASTER_GRID), "b": raster_input(grid=RASTER_GRID_2)},
                "steps": [{"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"}],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(ShapeError, match="share one grid"):
            validate_shapes(pipeline)

    def test_band_wide_join_on_the_same_grid_accepted(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": raster_input(grid=RASTER_GRID), "b": raster_input(grid=RASTER_GRID, variable="sal")},
                "steps": [{"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"}],
                "output": {"format": "csv", "select": "j"},
            }
        )
        # two value columns still reach the sink here (a, b) -- expected to fail the
        # single-value-column rule, not the join-compatibility rule this test targets.
        with pytest.raises(ShapeError, match="value column"):
            validate_shapes(pipeline)

    def test_long_join_across_two_different_grids_rejected(self):
        """A long join requires a shared grid just as a band-wide one does: `grid_cell`
        numbers a cell WITHIN a grid, so two vector grids' cell 7 are unrelated and the
        join has no correspondence to match on."""
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input(grid=VECTOR_GRID), "b": vector_input(grid=VECTOR_GRID_2)},
                "steps": [
                    {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                    {"op": "calculate", "expression": "a + b"},
                ],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(ShapeError, match="share one grid"):
            validate_shapes(pipeline)

    def test_long_join_on_one_grid_accepted(self):
        """The same two variables brought onto one grid -- the shape the rejection above
        asks for, and the one a real cross-variable comparison takes."""
        pipeline = parse_pipeline(
            {
                "inputs": {
                    "a": vector_input(grid=VECTOR_GRID),
                    "b": vector_input(grid=VECTOR_GRID, variable="stg"),
                },
                "steps": [
                    {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                    {"op": "calculate", "expression": "a - b"},
                ],
                "output": {"format": "csv"},
            }
        )
        validate_shapes(pipeline)

    def test_long_join_after_a_project_onto_a_common_grid_accepted(self):
        """The remedy the error message names, end to end: an operand on the wrong grid is
        brought onto the other's by a `project`, and the join then types cleanly. The
        check reads `FrameType.grid`, which a `project` updates, not the input's declared
        grid -- so inserting the project is genuinely what unblocks it."""
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input(grid=VECTOR_GRID), "b": vector_input(grid=VECTOR_GRID_2)},
                "steps": [
                    {"op": "project", "input": "b", "to": VECTOR_GRID, "aggregate": "mean", "as": "pb"},
                    {"op": "join", "type": "spatial", "inputs": ["a", "pb"], "as": "j"},
                    {"op": "calculate", "expression": "a + pb"},
                ],
                "output": {"format": "csv"},
            }
        )
        validate_shapes(pipeline)


class TestSinkRules:
    def test_more_than_one_value_column_at_sink_rejected(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input(), "b": vector_input()},
                "steps": [{"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"}],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(ShapeError, match="value column"):
            validate_shapes(pipeline)

    def test_calculate_after_join_collapses_to_one_value_column(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input(), "b": vector_input()},
                "steps": [
                    {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                    {"op": "calculate", "expression": "a + b"},
                ],
                "output": {"format": "csv"},
            }
        )
        validate_shapes(pipeline)

    def test_long_final_shape_under_raster_output_format_rejected(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [],
                "output": {"format": "tif"},
            }
        )
        with pytest.raises(ShapeError, match="raster surface"):
            validate_shapes(pipeline)

    def test_long_final_shape_under_png_tile_format_rejected(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [],
                "output": {"format": "png"},
            }
        )
        with pytest.raises(ShapeError, match="raster surface"):
            validate_shapes(pipeline)

    def test_long_final_shape_under_tabular_format_accepted(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [],
                "output": {"format": "csv"},
            }
        )
        validate_shapes(pipeline)

    def test_band_wide_final_shape_under_tif_format_accepted(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [],
                "output": {"format": "tif"},
            }
        )
        validate_shapes(pipeline)


class TestCompileOps:
    def test_compiles_one_op_per_step_in_order(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "project", "input": "a", "to": VECTOR_GRID_2, "aggregate": "sum", "as": "p"},
                    {"op": "filter", "where": "value > 1"},
                    {"op": "calculate", "expression": "value * 2"},
                    {"op": "group", "by": [], "aggregate": "min"},
                ],
                "output": {"format": "csv"},
            }
        )
        ops = compile_ops(pipeline)
        assert [type(op) for op in ops] == [ProjectOp, FilterOp, CalculateOp, GroupOp]
        assert ops[0].output == "p"
        assert ops[0].input == "a"
        # steps 2-4 gave no `as` -- default output names are the step's own position.
        assert ops[1].output == "2"
        assert ops[1].input == "p"  # chained from step 1's explicit `as`
        assert ops[2].input == "2"
        assert ops[3].input == "3"

    def test_compiles_join_op_with_explicit_inputs_list(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input(), "b": vector_input()},
                "steps": [{"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"}],
                "output": {"format": "csv"},
            }
        )
        ops = compile_ops(pipeline)
        assert isinstance(ops[0], JoinOp)
        assert ops[0].inputs == ["a", "b"]
        assert ops[0].output == "j"

    def test_default_output_names_never_collide_with_a_legal_as_name(self):
        """A bare digit string (`str(step_no)`) can never collide with a real `as`/input
        name: both are required to start with a lowercase letter or underscore."""
        from cpra.mp.data.analysis.request.pipeline import _NAME_RE

        for step_no in range(1, 50):
            assert not _NAME_RE.match(str(step_no))


class TestLoweringTableCoverage:
    """Self-consistency checks over the request layer's own declarative lowering tables
    -- that they are total over the kinds and shapes they claim to cover. Whether the
    executor's implementations actually agree with them is a separate question these
    tests do not reach, there being no executor to compare against yet."""

    def test_filter_lowering_table_covers_every_kind_and_shape(self):
        for kind in PredicateKind:
            for shape in Shape:
                assert (kind, shape) in FILTER_LOWERING_TABLE

    def test_only_spatial_predicates_change_shape(self):
        for kind in PredicateKind:
            for shape in Shape:
                out_shape = FILTER_LOWERING_TABLE[(kind, shape)]
                if kind is PredicateKind.spatial:
                    assert out_shape == Shape.long
                else:
                    assert out_shape == shape

    def test_project_lowering_table_omits_raster_to_raster(self):
        assert ("raster", "raster") not in PROJECT_LOWERING_TABLE

    def test_project_lowering_table_covers_exactly_the_three_legal_directions(self):
        assert set(PROJECT_LOWERING_TABLE) == {
            ("vector", "vector"),
            ("raster", "vector"),
            ("vector", "raster"),
        }

    def test_vector_to_raster_lowers_to_band_wide(self):
        assert PROJECT_LOWERING_TABLE[("vector", "raster")] == Shape.band_wide

    def test_other_two_directions_lower_to_long(self):
        assert PROJECT_LOWERING_TABLE[("vector", "vector")] == Shape.long
        assert PROJECT_LOWERING_TABLE[("raster", "vector")] == Shape.long
