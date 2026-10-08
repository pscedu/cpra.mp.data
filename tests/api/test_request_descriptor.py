"""Tests for `cpra.mp.data.analysis.request.descriptor` -- the output descriptor a
resolved pipeline is rendered from.

Like `test_request_ops.py`, these open nothing: a descriptor is derived from the declared
ops, each input's grid kind, and each input's `variable`/`time_unit`, which is what makes
it available at spec-build time rather than only after execution.
"""

import pytest

from cpra.mp.data.analysis.request.descriptor import output_descriptor, time_column_for
from cpra.mp.data.analysis.request.errors import ShapeError
from cpra.mp.data.analysis.request.pipeline import parse_pipeline
from cpra.mp.data.grids import GRIDS

VECTOR_GRID = "hydro_compartment_v001"
VECTOR_GRID_2 = "ecoregion_v001"
RASTER_GRID = "veg_grid_cell_v001"

assert GRIDS[VECTOR_GRID].grid_type == "vector"
assert GRIDS[RASTER_GRID].grid_type == "raster"


def vector_input(grid=VECTOR_GRID, variable="sal", time_unit="annual"):
    return {"variable": variable, "time_unit": time_unit, "grid": grid}


def raster_input(grid=RASTER_GRID, variable="ffibs_coverage_ratio", time_unit="annual"):
    return {"variable": variable, "time_unit": time_unit, "grid": grid}


class TestGrid:
    def test_single_input_reports_that_inputs_own_grid(self):
        pipeline = parse_pipeline({"inputs": {"data": vector_input()}, "output": {"format": "csv"}})
        descriptor = output_descriptor(pipeline)
        assert descriptor.grid == VECTOR_GRID
        assert descriptor.grid_type == "vector"
        assert descriptor.shape == "long"

    def test_a_project_moves_the_reported_grid(self):
        """The whole reason the grid is derived rather than declared: a pipeline that
        projects renders on the grid it projected TO, and nothing in the request says so
        except the final frame."""
        pipeline = parse_pipeline(
            {
                "inputs": {"data": vector_input()},
                "steps": [{"op": "project", "input": "data", "to": RASTER_GRID}],
                "output": {"format": "tif"},
            }
        )
        descriptor = output_descriptor(pipeline)
        assert descriptor.grid == RASTER_GRID
        assert descriptor.grid_type == "raster"
        assert descriptor.shape == "band_wide"

    def test_frame_type_is_carried_through_whole(self):
        pipeline = parse_pipeline({"inputs": {"data": raster_input()}, "output": {"format": "tif"}})
        descriptor = output_descriptor(pipeline)
        assert descriptor.frame_type.grid == descriptor.grid
        assert descriptor.frame_type.shape.value == descriptor.shape
        assert descriptor.frame_type.value_columns == {"value"}


class TestTimeUnit:
    @pytest.mark.parametrize(
        ("time_unit", "expected_column"),
        [("annual", "calendar_year"), ("decadal", "calendar_year"), ("daily", "calendar_day")],
    )
    def test_time_column_follows_the_unit(self, time_unit, expected_column):
        pipeline = parse_pipeline(
            {"inputs": {"data": vector_input(time_unit=time_unit)}, "output": {"format": "csv"}}
        )
        descriptor = output_descriptor(pipeline)
        assert descriptor.time_unit == time_unit
        assert descriptor.time_column == expected_column

    def test_agreeing_inputs_report_the_shared_unit(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input(variable="sal"), "b": vector_input(variable="stg")},
                "steps": [{"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"}],
                "output": {"format": "csv"},
            }
        )
        assert output_descriptor(pipeline).time_unit == "annual"

    def test_disagreeing_inputs_report_no_unit_rather_than_a_guess(self):
        pipeline = parse_pipeline(
            {
                "inputs": {
                    "a": vector_input(variable="sal", time_unit="annual"),
                    "b": vector_input(variable="sal", time_unit="daily"),
                },
                "steps": [{"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"}],
                "output": {"format": "csv"},
            }
        )
        descriptor = output_descriptor(pipeline)
        assert descriptor.time_unit is None
        assert descriptor.time_column is None

    def test_time_column_for_is_the_same_mapping_standalone(self):
        assert time_column_for("annual") == "calendar_year"
        assert time_column_for("decadal") == "calendar_year"
        assert time_column_for("daily") == "calendar_day"
        assert time_column_for(None) is None


class TestVariables:
    def test_single_input_resolves_the_value_column_to_that_variable(self):
        pipeline = parse_pipeline({"inputs": {"data": vector_input(variable="sal")}, "output": {"format": "csv"}})
        descriptor = output_descriptor(pipeline)
        assert descriptor.value_columns == ("value",)
        assert descriptor.variable is not None
        assert descriptor.variable.name == "sal"
        assert descriptor.variable.display_unit == descriptor.variables[0].display_unit

    def test_a_group_does_not_lose_the_variable(self):
        """Every step between the input and the sink leaves the value column named
        `"value"`, so the input-variable fallback is what answers for a `diff`, a `group`,
        or any other single-input analysis -- not just a bare download."""
        pipeline = parse_pipeline(
            {
                "inputs": {"data": vector_input(variable="elev")},
                "steps": [{"op": "group", "input": "data", "by": ["grid_cell"], "aggregate": "mean"}],
                "output": {"format": "csv"},
            }
        )
        assert output_descriptor(pipeline).variable.name == "elev"

    def test_join_operand_columns_resolve_per_column_by_name(self):
        """Level 3 of the precedence: a join names each value column after its operand,
        and an operand is named for the variable it reads, so a multi-variable frame is
        answered per column with no single "the" variable."""
        pipeline = parse_pipeline(
            {
                "inputs": {"sal": vector_input(variable="sal"), "stg": vector_input(variable="stg")},
                "steps": [{"op": "join", "type": "spatial", "inputs": ["sal", "stg"], "as": "j"}],
                "output": {"format": "csv"},
            }
        )
        descriptor = output_descriptor(pipeline)
        assert descriptor.value_columns == ("sal", "stg")
        assert [variable.name for variable in descriptor.variables] == ["sal", "stg"]
        # More than one, so there is no single variable this output is "about".
        assert descriptor.variable is None

    def test_multi_input_collapsed_to_one_unnamed_column_resolves_to_nothing(self):
        """The case §3 sends to the analysis's own `Meta`: several input variables, one
        `"value"` column that names none of them. Guessing one would be wrong, so this
        reports `None` and the consumer falls back."""
        pipeline = parse_pipeline(
            {
                "inputs": {"a": vector_input(variable="sal"), "b": vector_input(variable="stg")},
                "steps": [
                    {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                    {"op": "calculate", "input": "j", "expression": "a - b"},
                ],
                "output": {"format": "csv"},
            }
        )
        descriptor = output_descriptor(pipeline)
        assert descriptor.value_columns == ("value",)
        assert descriptor.variables == (None,)
        assert descriptor.variable is None


class TestFoldedOutput:
    """An `unpivot`ed pipeline is described by its FOLDED names rather than by the `value`
    column it actually writes -- the one case where `value_columns` names row LABELS.
    """

    def _folded(self, into="variable"):
        return parse_pipeline(
            {
                "inputs": {"sal": vector_input(variable="sal"), "stg": vector_input(variable="stg")},
                "steps": [
                    {"op": "join", "type": "spatial", "inputs": ["sal", "stg"], "as": "j"},
                    {"op": "unpivot", "input": "j", "into": into, "as": "u"},
                ],
                "output": {"format": "csv"},
            }
        )

    def test_value_columns_reports_the_folded_names(self):
        # Identical to what the same pipeline reported BEFORE the unpivot (see
        # `TestVariables.test_join_operand_columns_resolve_per_column_by_name`), which is
        # the point: the fold changes where the names live, not which variables the result
        # is about. Reporting the literal `("value",)` would describe five stacked
        # variables as one nameless column.
        descriptor = output_descriptor(self._folded())
        assert descriptor.value_columns == ("sal", "stg")
        assert [variable.name for variable in descriptor.variables] == ["sal", "stg"]
        assert descriptor.frame_type.value_columns == frozenset({"value"})

    def test_folded_into_names_the_label_column(self):
        assert output_descriptor(self._folded()).folded_into == "variable"
        assert output_descriptor(self._folded(into="measurement")).folded_into == "measurement"

    def test_an_unfolded_pipeline_reports_no_folded_into(self):
        """Null for every pipeline without an `unpivot`, which is how a consumer tells the
        two readings of `value_columns` apart."""
        pipeline = parse_pipeline({"inputs": {"data": vector_input()}, "output": {"format": "csv"}})
        assert output_descriptor(pipeline).folded_into is None

    def test_there_is_no_single_variable(self):
        assert output_descriptor(self._folded()).variable is None


class TestErrors:
    def test_a_pipeline_that_cannot_type_raises_rather_than_describing_itself(self):
        pipeline = parse_pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "project", "input": "a", "to": "morph_pixel_v001"}],
                "output": {"format": "tif"},
            }
        )
        with pytest.raises(ShapeError, match="raster-to-raster"):
            output_descriptor(pipeline)
