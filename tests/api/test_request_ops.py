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
from cpra.mp.data.analysis.request.expressions import PredicateKind
from cpra.mp.data.analysis.request.ops import (
    FILTER_LOWERING_TABLE,
    PROJECT_LOWERING_TABLE,
    CalculateOp,
    FilterOp,
    GroupOp,
    JoinOp,
    ProjectOp,
    Shape,
    annotate_frame_types,
    compile_ops,
    validate_shapes,
)
from cpra.mp.data.analysis.request.pipeline import DOMAIN_COLUMN, ZONE_COLUMN, parse_pipeline
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

    def test_band_wide_final_shape_under_tabular_format_rejected(self):
        # The converse of `test_long_final_shape_under_raster_output_format_rejected`, and
        # the only thing standing between a caller and a wrong-content-type response:
        # `execution.execute.run_to_path` chooses its sink from the RESULT's shape, never
        # from `output.format`, so without this rule a `csv` request over a raster result
        # is answered 200 with a GeoTIFF's bytes rather than rejected.
        pipeline = parse_pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(ShapeError, match="tabular sink"):
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


class TestCarriedColumnTyping:
    """`FrameType.extra_columns` -- which frames carry a per-pixel column alongside their
    bands, and what that does to a `where` naming one.

    The two names are fixed (`ZONE_COLUMN`, `DOMAIN_COLUMN`) and neither is a dimension:
    they are columns of the DATA, one value per pixel. Typing them is what lets a
    `filter` naming one classify as the per-pixel mask it is instead of a band selection
    against metadata that has no such column.
    """

    @staticmethod
    def _types(spec):
        pipeline = parse_pipeline(spec)
        return annotate_frame_types(pipeline, compile_ops(pipeline))

    def test_an_ordinary_raster_chain_carries_nothing(self):
        types = self._types(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "filter", "input": "a", "where": "calendar_year = 2030"}],
                "output": {"format": "tif"},
            }
        )
        assert all(frame_type.extra_columns == frozenset() for frame_type in types.values())

    def test_a_raster_input_a_raster_to_vector_project_reaches_opens_with_the_zone_column(self):
        """The crosswalk zone id is attached at the READ, because of a `project` declared
        after it -- so the INPUT's frame type carries it, not the project's output (which
        is long, and where the projection has already consumed it).
        """
        types = self._types(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "project", "input": "a", "to": VECTOR_GRID, "aggregate": "mean"}],
                "output": {"format": "csv"},
            }
        )
        assert types["a"].extra_columns == frozenset({ZONE_COLUMN})
        assert types["1"].extra_columns == frozenset()

    def test_the_zone_column_reaches_a_fan_out_branch_that_never_projects(self):
        """Attached at the read, so every branch off that origin carries it -- including
        one that never reaches the `project` that asked for it.
        """
        types = self._types(
            {
                "inputs": {"a": raster_input()},
                "steps": [
                    {"op": "filter", "input": "a", "where": "calendar_year = 2030", "as": "f"},
                    {"op": "project", "input": "a", "to": VECTOR_GRID, "aggregate": "mean"},
                ],
                "output": {"format": "csv"},
            }
        )
        assert types["a"].extra_columns == frozenset({ZONE_COLUMN})
        assert types["f"].extra_columns == frozenset({ZONE_COLUMN})

    def test_a_vector_input_upstream_of_a_raster_to_vector_project_does_not_carry_it(self):
        """Only a RASTER origin's read gets a crosswalk attached. A vector input that
        reaches a raster-to-vector `project` only by being rasterized first is not one --
        it is the intervening `project`'s output that is read from a raster lattice, and
        that frame is the one the crosswalk would have to describe.
        """
        types = self._types(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "project", "input": "a", "to": RASTER_GRID},
                    {"op": "project", "to": VECTOR_GRID_2, "aggregate": "mean"},
                ],
                "output": {"format": "csv"},
            }
        )
        assert types["a"].extra_columns == frozenset()
        assert types["1"].extra_columns == frozenset({DOMAIN_COLUMN})
        assert types["2"].extra_columns == frozenset()

    def test_the_domain_column_is_inherited_by_every_band_wide_step_after_it(self):
        types = self._types(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "project", "input": "a", "to": RASTER_GRID},
                    {"op": "filter", "where": "calendar_year = 2030"},
                    {"op": "calculate", "expression": "value * 2"},
                    {"op": "group", "by": ["calendar_year"], "aggregate": "mean"},
                ],
                "output": {"format": "tif"},
            }
        )
        for name in ("1", "2", "3", "4"):
            assert types[name].extra_columns == frozenset({DOMAIN_COLUMN})

    def test_a_join_unions_what_its_operands_carry(self):
        types = self._types(
            {
                "inputs": {"v": vector_input(), "r": raster_input()},
                "steps": [
                    {"op": "project", "input": "v", "to": RASTER_GRID, "as": "p"},
                    {"op": "join", "type": "spatial", "inputs": ["p", "r"], "as": "j"},
                ],
                "output": {"format": "tif"},
            }
        )
        assert types["r"].extra_columns == frozenset()
        assert types["j"].extra_columns == frozenset({DOMAIN_COLUMN})

    def test_a_spatial_filter_drops_them_with_the_band_axis(self):
        """A spatial conjunct unpivots to long, and a per-pixel column has no meaning once
        the frame is rows -- `execution.execute._run_filter_band_wide` drops them there.
        """
        types = self._types(
            {
                "inputs": {"a": vector_input()},
                "steps": [
                    {"op": "project", "input": "a", "to": RASTER_GRID},
                    {"op": "filter", "where": "grid_cell = 7"},
                ],
                "output": {"format": "csv"},
            }
        )
        assert types["1"].extra_columns == frozenset({DOMAIN_COLUMN})
        assert types["2"].shape is Shape.long
        assert types["2"].extra_columns == frozenset()

    def test_a_where_naming_a_carried_column_types_as_a_value_filter(self):
        """The observable consequence, end to end. A value `filter` over a PAIRED band-wide
        frame collapses the pairing back to one value column; a dimension one does not. So
        typing `__domain__ > 0` correctly is what makes the `case` that follows legal at
        all: an operand-named conjunct over a paired band-wide frame is rejected outright
        (`_type_filter`), so a carried-column conjunct classified as a VALUE one has to be
        recognized as the per-band mask it is rather than lumped in with those.

        Classified as a dimension conjunct (the pre-fix behaviour) this same pipeline would
        have failed at execution on a `metadata_df` with no `__domain__` column.
        """
        spec = {
            "inputs": {"v": vector_input(), "r": raster_input()},
            "steps": [
                {"op": "project", "input": "v", "to": RASTER_GRID, "as": "p"},
                {"op": "join", "type": "spatial", "inputs": ["p", "r"], "as": "j"},
                {"op": "filter", "input": "j", "where": f"{DOMAIN_COLUMN} > 0", "as": "f"},
                {"op": "case", "input": "f", "cases": [{"when": "p > 0", "then": "r"}]},
            ],
            "output": {"format": "tif"},
        }
        types = self._types(spec)
        assert types["j"].value_columns == frozenset({"p", "r"})
        # A filter never collapses a pairing -- masking each band by a per-pixel column
        # leaves both operands exactly where they were.
        assert types["f"].value_columns == frozenset({"p", "r"})
        assert types["f"].shape is Shape.band_wide
        validate_shapes(parse_pipeline(spec))

    def test_an_operand_named_conjunct_over_a_paired_frame_is_rejected(self):
        """The other side of the same classification. A conjunct naming an OPERAND cannot
        be masked band by band -- it is a question about the pair -- so it is refused here
        rather than silently reducing the frame and picking a survivor.
        """
        spec = {
            "inputs": {"v": vector_input(), "r": raster_input()},
            "steps": [
                {"op": "project", "input": "v", "to": RASTER_GRID, "as": "p"},
                {"op": "join", "type": "spatial", "inputs": ["p", "r"], "as": "j"},
                {"op": "filter", "input": "j", "where": f"{DOMAIN_COLUMN} > 0 AND p > 0.5"},
            ],
            "output": {"format": "tif"},
        }
        with pytest.raises(ShapeError, match=r"step 3 \(filter\): conjunct\(s\) \['p > 0.5'\]"):
            self._types(spec)


class TestCaseShapeTyping:
    """`_type_case`. A `case` is unconditionally shape-preserving and always one value
    column out, which is what lets it have no lowering table at all -- but only because
    every arm's `when` is confined to the frame's VALUE namespace. A dimension or spatial
    arm would mean "some bands take this branch and others do not", or "some pixels do",
    and neither is a rewrite of the running value; both are `filter`'s job.
    """

    @staticmethod
    def _types(spec):
        pipeline = parse_pipeline(spec)
        return annotate_frame_types(pipeline, compile_ops(pipeline))

    @staticmethod
    def _spec(when, *, paired=False):
        steps = (
            [
                {"op": "project", "input": "v", "to": RASTER_GRID, "as": "p"},
                {"op": "join", "type": "spatial", "inputs": ["p", "r"], "as": "j"},
            ]
            if paired
            else [{"op": "filter", "input": "r", "where": "calendar_year = 2030", "as": "j"}]
        )
        return {
            "inputs": {"v": vector_input(), "r": raster_input()},
            "steps": [
                *steps,
                {"op": "case", "input": "j", "cases": [{"when": when, "then": "1"}], "as": "c"},
            ],
            "output": {"format": "tif"},
        }

    def test_a_value_arm_preserves_the_shape_and_yields_one_value_column(self):
        types = self._types(self._spec("value > 0"))
        assert types["c"].shape is types["j"].shape is Shape.band_wide
        assert types["c"].value_columns == frozenset({"value"})

    def test_a_paired_frame_collapses_to_one_value_column(self):
        # Typed like a paired FILTER, not like a `calculate`: `case` is the one op other
        # than a value `filter` and a `group` that can collapse a pairing, so the arms may
        # name the operands and what comes back carries `value` alone.
        types = self._types(self._spec("p > r", paired=True))
        assert types["j"].value_columns == frozenset({"p", "r"})
        assert types["c"].value_columns == frozenset({"value"})
        assert types["c"].shape is Shape.band_wide

    def test_a_carried_column_is_in_the_arms_namespace(self):
        # `extra_columns` is part of the value namespace for the same reason it is in a
        # `calculate`'s: it is a column of the data, one value per pixel.
        types = self._types(self._spec(f"{DOMAIN_COLUMN} > 0", paired=True))
        assert types["c"].value_columns == frozenset({"value"})

    def test_a_dimension_arm_is_rejected(self):
        with pytest.raises(ShapeError, match="state that as a filter instead"):
            self._types(self._spec("calendar_year = 2030"))

    def test_a_spatial_arm_is_rejected(self):
        with pytest.raises(ShapeError, match="state that as a filter instead"):
            self._types(self._spec("grid_cell = 7"))

    def test_the_rejection_names_the_arm_and_the_namespace_it_missed(self):
        spec = {
            "inputs": {"r": raster_input()},
            "steps": [
                {
                    "op": "case",
                    "input": "r",
                    "cases": [
                        {"when": "value > 1", "then": "1"},
                        {"when": "value > 0 AND calendar_year = 2030", "then": "2"},
                    ],
                }
            ],
            "output": {"format": "tif"},
        }
        # The second arm, and only its offending conjunct -- the `value > 0` half of the
        # same `when` is fine and is not reported.
        with pytest.raises(ShapeError, match=r"arm 2's when=.*\['calendar_year = 2030'\]"):
            self._types(spec)


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
