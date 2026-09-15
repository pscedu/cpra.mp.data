"""Tests for the analysis-pipeline representation and its two encodings (api/pipeline.py).

Every function under test takes and returns plain values and raises `PipelineError` (an
ordinary `ValueError`), never an `HTTPException` -- these tests exercise the library layer
directly, with no `Dataset`/FastAPI involved, matching `api/expressions.py`'s and
`api/fold.py`'s "no FastAPI below the translation layer" design constraint. Real variables/
grids/dimensions from `cpra/mp/data/structure/` are used throughout (no fake dimensions),
following `tests/factories.py`'s own convention.
"""

import copy
from datetime import date

import pytest

from cpra.mp.data.analysis.pipeline import (
    Pipeline,
    PipelineError,
    parse_pipeline,
    pipeline_from_query,
    pipeline_to_query,
    validate_pipeline,
)

# The analysis-pipeline design's own exit-criterion example, verbatim:
# "first year each veg pixel has salinity > 5.5 AND freshwater-marsh coverage > 0.5".
NOTEBOOK_QUERY = {
    "inputs": {
        "marsh": {
            "variable": "ffibs_coverage_ratio",
            "time_unit": "annual",
            "grid": "veg_grid_cell_v001",
            "model_group_id": 500,
            "scenario_id": 7,
            "ffibs_type": "pl_fm",
        },
        "salinity": {
            "variable": "sal",
            "time_unit": "annual",
            "grid": "hydro_compartment_v001",
            "model_group_id": 500,
            "scenario_id": 7,
            "aggregate_type": "2_wk_max",
        },
    },
    "steps": [
        {"op": "project", "input": "salinity", "to": "veg_grid_cell_v001", "as": "sal"},
        {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
        {"op": "filter", "where": "marsh > 0.5 AND sal > 5.5"},
        {"op": "group", "by": [], "aggregate": "min", "of": "calendar_year"},
    ],
    "output": {"format": "tif"},
}


def simple_pipeline(**overrides):
    """A minimal one-input, one-step pipeline, valid unless overridden."""
    data = {
        "inputs": {
            "marsh": {
                "variable": "ffibs_coverage_ratio",
                "time_unit": "annual",
                "grid": "veg_grid_cell_v001",
            },
        },
        "steps": [{"op": "project", "input": "marsh", "to": "ecoregion_v001", "aggregate": "mean"}],
        "output": {"format": "csv"},
    }
    data.update(overrides)
    return data


class TestNotebookQuery:
    def test_parses_and_validates(self):
        pipeline = parse_pipeline(NOTEBOOK_QUERY)
        assert isinstance(pipeline, Pipeline)
        assert set(pipeline.inputs) == {"marsh", "salinity"}
        assert len(pipeline.steps) == 4
        assert pipeline.output.format == "tif"

    def test_round_trips_through_query_and_back(self):
        pipeline = parse_pipeline(NOTEBOOK_QUERY)
        query = pipeline_to_query(pipeline)
        pipeline2 = pipeline_from_query(query)
        assert pipeline_to_query(pipeline2) == query


class TestInputSelection:
    def test_scalar_dimension_filter_normalizes_to_list(self):
        pipeline = parse_pipeline(simple_pipeline())
        assert pipeline.inputs["marsh"].dimension_filters == {}

    def test_extra_scalar_filter_becomes_single_item_list(self):
        data = simple_pipeline()
        data["inputs"]["marsh"]["model_group_id"] = 500
        pipeline = parse_pipeline(data)
        assert pipeline.inputs["marsh"].dimension_filters == {"model_group_id": [500]}

    def test_extra_list_filter_stays_a_list(self):
        data = simple_pipeline()
        data["inputs"]["marsh"]["scenario_id"] = [7, 8]
        pipeline = parse_pipeline(data)
        assert pipeline.inputs["marsh"].dimension_filters == {"scenario_id": [7, 8]}

    def test_empty_list_filter_rejected(self):
        data = simple_pipeline()
        data["inputs"]["marsh"]["scenario_id"] = []
        with pytest.raises(ValueError):
            parse_pipeline(data)

    def test_non_scalar_filter_value_rejected(self):
        data = simple_pipeline()
        data["inputs"]["marsh"]["scenario_id"] = [{"nested": "dict"}]
        with pytest.raises(ValueError):
            parse_pipeline(data)

    def test_date_filter_value_accepted(self):
        """`date` is a real dimension value type, not an exotic one: `calendar_day` is
        declared `Date` in `structure/dimensions.csv`, so `types.dimension_type` types
        `FilterDimensions.calendar_day` as `list[date]` and FastAPI binds
        `?calendar_day=2019-01-01` to a `datetime.date`. Omitting it from the scalar
        whitelist rejected every request that filtered a daily dataset by day.
        """
        data = simple_pipeline()
        data["inputs"]["marsh"]["calendar_day"] = [date(2019, 1, 1), date(2070, 12, 31)]
        pipeline = parse_pipeline(data)
        assert pipeline.inputs["marsh"].dimension_filters == {
            "calendar_day": [date(2019, 1, 1), date(2070, 12, 31)]
        }

    def test_scalar_date_filter_value_accepted(self):
        data = simple_pipeline()
        data["inputs"]["marsh"]["calendar_day"] = date(2019, 1, 1)
        pipeline = parse_pipeline(data)
        assert pipeline.inputs["marsh"].dimension_filters == {"calendar_day": [date(2019, 1, 1)]}

    def test_missing_required_dataset_dimension_rejected(self):
        data = simple_pipeline()
        del data["inputs"]["marsh"]["grid"]
        with pytest.raises(ValueError):
            parse_pipeline(data)

    def test_unknown_variable_rejected(self):
        data = simple_pipeline()
        data["inputs"]["marsh"]["variable"] = "not_a_real_variable"
        with pytest.raises(ValueError):
            parse_pipeline(data)


class TestInputSelectionGridCell:
    """`grid_cell` on an input is a reserved Filter-group parameter (forwarded to
    `Options.grid_cell` in `planner.py::_dataset_for`), not a dimension filter -- it must
    never show up in `InputSelection.dimension_filters`, and it must round-trip through
    the dotted-query encoding like any other declared input field.
    """

    def test_declared_field_not_a_dimension_filter(self):
        data = simple_pipeline()
        data["inputs"]["marsh"]["grid_cell"] = "5,6"
        pipeline = parse_pipeline(data)
        assert pipeline.inputs["marsh"].grid_cell == "5,6"
        assert pipeline.inputs["marsh"].dimension_filters == {}

    def test_omitted_grid_cell_defaults_to_none(self):
        pipeline = parse_pipeline(simple_pipeline())
        assert pipeline.inputs["marsh"].grid_cell is None

    def test_round_trips_through_query(self):
        data = simple_pipeline()
        data["inputs"]["marsh"]["grid_cell"] = "5,6"
        pipeline = parse_pipeline(data)
        query = pipeline_to_query(pipeline)
        assert query["input.marsh.grid_cell"] == "5,6"
        pipeline2 = pipeline_from_query(query)
        assert pipeline2.inputs["marsh"].grid_cell == "5,6"

    def test_omitted_from_query_when_unset(self):
        pipeline = parse_pipeline(simple_pipeline())
        query = pipeline_to_query(pipeline)
        assert "input.marsh.grid_cell" not in query


class TestInputNaming:
    def test_input_name_must_be_snake_case(self):
        data = simple_pipeline()
        data["inputs"] = {"Marsh": data["inputs"]["marsh"]}
        data["steps"][0]["input"] = "Marsh"
        with pytest.raises(PipelineError, match="input"):
            parse_pipeline(data)

    def test_input_name_reserved_word_rejected(self):
        data = simple_pipeline()
        data["inputs"] = {"value": data["inputs"]["marsh"]}
        data["steps"][0]["input"] = "value"
        with pytest.raises(PipelineError, match="reserved"):
            parse_pipeline(data)

    def test_grid_cell_reserved_as_input_name(self):
        data = simple_pipeline()
        data["inputs"] = {"grid_cell": data["inputs"]["marsh"]}
        data["steps"][0]["input"] = "grid_cell"
        with pytest.raises(PipelineError, match="reserved"):
            parse_pipeline(data)


class TestProjectStep:
    def test_count_by_value_accepted_as_zonal_aggregate(self):
        # `count_by_value` is not in `fold.DECOMPOSABLE_AGGREGATES` (still correctly
        # rejected for a `group` step -- see `TestGroupStep.test_aggregate_must_be_
        # decomposable`), but it IS a valid ZONAL aggregate, so a `project` step permits
        # it specifically -- this is the documented "Categorical Count per Zone" query
        # (docs/data-analysis-api.md) expressed as a pipeline.
        data = simple_pipeline()
        data["steps"][0]["aggregate"] = "count_by_value"
        pipeline = parse_pipeline(data)
        assert pipeline.steps[0].aggregate == "count_by_value"

    def test_decomposable_aggregate_accepted(self):
        data = simple_pipeline()
        data["steps"][0]["aggregate"] = "sum"
        parse_pipeline(data)

    def test_no_aggregate_is_optional(self):
        data = simple_pipeline()
        del data["steps"][0]["aggregate"]
        parse_pipeline(data)

    def test_omitted_input_chains_from_previous_step(self):
        data = simple_pipeline()
        data["steps"].append({"op": "project", "to": "hydro_compartment_v001", "aggregate": "mean"})
        parse_pipeline(data)

    def test_omitted_input_on_first_step_rejected(self):
        data = simple_pipeline()
        del data["steps"][0]["input"]
        with pytest.raises(PipelineError, match="no input"):
            parse_pipeline(data)


class TestJoinStep:
    def test_attribute_join_rejected_as_reserved(self):
        data = simple_pipeline()
        data["inputs"]["sal"] = {
            "variable": "sal",
            "time_unit": "annual",
            "grid": "hydro_compartment_v001",
        }
        data["steps"] = [
            {"op": "join", "type": "attribute", "inputs": ["marsh", "sal"], "as": "paired"},
        ]
        with pytest.raises(PipelineError, match="attribute"):
            parse_pipeline(data)

    def test_spatial_join_with_two_inputs_accepted(self):
        data = simple_pipeline()
        data["inputs"]["sal"] = {
            "variable": "sal",
            "time_unit": "annual",
            "grid": "veg_grid_cell_v001",
        }
        data["steps"] = [
            {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
        ]
        parse_pipeline(data)

    def test_join_requires_at_least_two_inputs(self):
        data = simple_pipeline()
        data["steps"] = [
            {"op": "join", "type": "spatial", "inputs": ["marsh"], "as": "paired"},
        ]
        with pytest.raises(ValueError):
            parse_pipeline(data)

    def test_join_rejects_duplicate_input_name(self):
        data = simple_pipeline()
        data["steps"] = [
            {"op": "join", "type": "spatial", "inputs": ["marsh", "marsh"], "as": "paired"},
        ]
        with pytest.raises(PipelineError, match="duplicate"):
            parse_pipeline(data)

    def test_join_missing_as_rejected(self):
        data = simple_pipeline()
        data["inputs"]["sal"] = {
            "variable": "sal",
            "time_unit": "annual",
            "grid": "veg_grid_cell_v001",
        }
        data["steps"] = [
            {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"]},
        ]
        with pytest.raises(ValueError):
            parse_pipeline(data)

    def test_join_referencing_unknown_input_rejected(self):
        data = simple_pipeline()
        data["steps"] = [
            {"op": "join", "type": "spatial", "inputs": ["marsh", "nope"], "as": "paired"},
        ]
        with pytest.raises(PipelineError, match="nope"):
            parse_pipeline(data)


class TestFilterStep:
    def test_empty_where_rejected(self):
        data = simple_pipeline()
        data["steps"].append({"op": "filter", "where": ""})
        with pytest.raises(PipelineError, match="non-empty"):
            parse_pipeline(data)

    def test_filter_cannot_be_first_step(self):
        data = simple_pipeline()
        data["steps"] = [{"op": "filter", "where": "marsh > 0.5"}]
        with pytest.raises(PipelineError, match="no input"):
            parse_pipeline(data)

    def test_ordinary_predicate_accepted(self):
        data = simple_pipeline()
        data["steps"].append({"op": "filter", "where": "marsh > 0.5"})
        parse_pipeline(data)

    def test_compound_predicate_with_spaces_accepted(self):
        data = simple_pipeline()
        data["steps"].append({"op": "filter", "where": "marsh > 0.5 AND marsh < 10"})
        parse_pipeline(data)

    def test_aggregate_call_in_where_rejected(self):
        data = simple_pipeline()
        data["steps"].append({"op": "filter", "where": "marsh > max(marsh)"})
        with pytest.raises(PipelineError, match="aggregate"):
            parse_pipeline(data)

    def test_subquery_in_where_rejected(self):
        data = simple_pipeline()
        data["steps"].append({"op": "filter", "where": "marsh in (select 1)"})
        with pytest.raises(PipelineError, match="subquery"):
            parse_pipeline(data)


class TestCalculateStep:
    def test_ordinary_expression_accepted(self):
        data = simple_pipeline()
        data["steps"].append({"op": "calculate", "expression": "marsh * 0.6"})
        pipeline = parse_pipeline(data)
        assert pipeline.steps[-1].expression == "marsh * 0.6"
        assert pipeline.steps[-1].where is None

    def test_expression_with_where_accepted(self):
        data = simple_pipeline()
        data["steps"].append(
            {"op": "calculate", "expression": "marsh * 0.6", "where": "model_group_id = 500"}
        )
        pipeline = parse_pipeline(data)
        assert pipeline.steps[-1].where == "model_group_id = 500"

    def test_expression_is_required(self):
        data = simple_pipeline()
        data["steps"].append({"op": "calculate"})
        with pytest.raises(ValueError):
            parse_pipeline(data)

    def test_empty_expression_rejected(self):
        data = simple_pipeline()
        data["steps"].append({"op": "calculate", "expression": "  "})
        with pytest.raises(PipelineError, match="empty"):
            parse_pipeline(data)

    def test_comparison_operator_in_expression_rejected(self):
        data = simple_pipeline()
        data["steps"].append({"op": "calculate", "expression": "marsh > 0.5"})
        with pytest.raises(PipelineError, match="where"):
            parse_pipeline(data)

    def test_aggregate_call_in_expression_rejected(self):
        data = simple_pipeline()
        data["steps"].append({"op": "calculate", "expression": "max(marsh)"})
        with pytest.raises(PipelineError, match="aggregate"):
            parse_pipeline(data)

    def test_aggregate_call_in_where_rejected(self):
        data = simple_pipeline()
        data["steps"].append(
            {"op": "calculate", "expression": "marsh * 2", "where": "marsh > max(marsh)"}
        )
        with pytest.raises(PipelineError, match="aggregate"):
            parse_pipeline(data)

    def test_empty_where_rejected(self):
        data = simple_pipeline()
        data["steps"].append({"op": "calculate", "expression": "marsh * 2", "where": ""})
        with pytest.raises(PipelineError, match="non-empty"):
            parse_pipeline(data)

    def test_calculate_cannot_be_first_step(self):
        data = simple_pipeline()
        data["steps"] = [{"op": "calculate", "expression": "marsh * 0.6"}]
        with pytest.raises(PipelineError, match="no input"):
            parse_pipeline(data)

    def test_round_trips_through_query_encoding(self):
        data = simple_pipeline()
        data["steps"].append(
            {"op": "calculate", "expression": "marsh * 0.6", "where": "model_group_id = 500"}
        )
        pipeline = parse_pipeline(data)
        query = pipeline_to_query(pipeline)
        assert query["step.2.op"] == "calculate"
        assert query["step.2.expression"] == "marsh * 0.6"
        assert query["step.2.where"] == "model_group_id = 500"
        pipeline2 = pipeline_from_query(query)
        assert pipeline_to_query(pipeline2) == query

    def test_dotted_encoding_decodes_expression_and_where_verbatim(self):
        """`expression`/`where` are opaque strings -- never comma-split, so a function
        call with arguments survives the dotted GET encoding."""
        query = pipeline_to_query(parse_pipeline(simple_pipeline()))
        query.update(
            {
                "step.2.op": "calculate",
                "step.2.expression": "round(marsh * 0.6, 2)",
                "step.2.where": "model_group_id = 500 AND marsh > 0.5",
            }
        )
        pipeline = pipeline_from_query(query)
        assert pipeline.steps[1].expression == "round(marsh * 0.6, 2)"
        assert pipeline.steps[1].where == "model_group_id = 500 AND marsh > 0.5"

    def test_unknown_calculate_field_in_query_rejected(self):
        query = pipeline_to_query(parse_pipeline(simple_pipeline()))
        query.update({"step.2.op": "calculate", "step.2.expr": "marsh * 0.6"})
        with pytest.raises(PipelineError, match="unknown field"):
            pipeline_from_query(query)


class TestGroupStep:
    def test_aggregate_is_required(self):
        data = simple_pipeline()
        data["steps"].append({"op": "group", "by": []})
        with pytest.raises(ValueError):
            parse_pipeline(data)

    def test_aggregate_must_be_decomposable(self):
        data = simple_pipeline()
        data["steps"].append({"op": "group", "by": [], "aggregate": "count_by_value"})
        with pytest.raises(PipelineError):
            parse_pipeline(data)

    def test_empty_by_collapses_to_cell(self):
        data = simple_pipeline()
        data["steps"].append({"op": "group", "by": [], "aggregate": "sum"})
        pipeline = parse_pipeline(data)
        assert pipeline.steps[-1].by == []

    def test_of_defaults_to_value(self):
        data = simple_pipeline()
        data["steps"].append({"op": "group", "by": [], "aggregate": "min"})
        pipeline = parse_pipeline(data)
        assert pipeline.steps[-1].of == "value"

    def test_of_generalizes_the_aggregand(self):
        data = simple_pipeline()
        data["steps"].append(
            {"op": "group", "by": [], "aggregate": "min", "of": "calendar_year"}
        )
        pipeline = parse_pipeline(data)
        assert pipeline.steps[-1].of == "calendar_year"

    def test_group_cannot_be_first_step(self):
        data = simple_pipeline()
        data["steps"] = [{"op": "group", "by": [], "aggregate": "sum"}]
        with pytest.raises(PipelineError, match="no input"):
            parse_pipeline(data)


class TestChaining:
    def test_as_name_must_be_unique(self):
        data = simple_pipeline()
        data["steps"][0]["as"] = "marsh"  # collides with the input name
        with pytest.raises(PipelineError, match="collides"):
            parse_pipeline(data)

    def test_duplicate_as_names_across_steps_rejected(self):
        data = simple_pipeline()
        data["steps"][0]["as"] = "step_out"
        data["steps"].append({"op": "filter", "where": "marsh > 0", "as": "step_out"})
        with pytest.raises(PipelineError, match="collides"):
            parse_pipeline(data)

    def test_later_step_references_prior_as(self):
        data = simple_pipeline()
        data["inputs"]["sal"] = {
            "variable": "sal",
            "time_unit": "annual",
            "grid": "veg_grid_cell_v001",
        }
        data["steps"][0]["as"] = "projected"
        data["steps"].append(
            {"op": "join", "type": "spatial", "inputs": ["projected", "sal"], "as": "paired"}
        )
        pipeline = parse_pipeline(data)
        assert pipeline.steps[1].inputs == ["projected", "sal"]

    def test_forward_reference_to_later_as_rejected(self):
        data = simple_pipeline()
        data["inputs"]["sal"] = {
            "variable": "sal",
            "time_unit": "annual",
            "grid": "veg_grid_cell_v001",
        }
        data["steps"] = [
            {"op": "join", "type": "spatial", "inputs": ["marsh", "not_yet"], "as": "paired"},
            {"op": "project", "input": "sal", "to": "veg_grid_cell_v001", "as": "not_yet"},
        ]
        with pytest.raises(PipelineError, match="not_yet"):
            parse_pipeline(data)

    def test_explicit_input_override_on_non_first_step(self):
        """A later `project` step may still name `input` explicitly rather than
        chaining -- an override, not a requirement."""
        data = simple_pipeline()
        data["inputs"]["sal"] = {
            "variable": "sal",
            "time_unit": "annual",
            "grid": "hydro_compartment_v001",
        }
        data["steps"].append(
            {"op": "project", "input": "sal", "to": "veg_grid_cell_v001", "aggregate": "mean"}
        )
        pipeline = parse_pipeline(data)
        assert pipeline.steps[1].input == "sal"


class TestOutputRule:
    def test_tif_output_after_join_as_last_step_rejected(self):
        data = simple_pipeline()
        data["inputs"]["sal"] = {
            "variable": "sal",
            "time_unit": "annual",
            "grid": "veg_grid_cell_v001",
        }
        data["steps"] = [
            {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
        ]
        data["output"] = {"format": "tif"}
        with pytest.raises(PipelineError, match="join"):
            parse_pipeline(data)

    def test_tabular_output_after_join_accepted(self):
        data = simple_pipeline()
        data["inputs"]["sal"] = {
            "variable": "sal",
            "time_unit": "annual",
            "grid": "veg_grid_cell_v001",
        }
        data["steps"] = [
            {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
        ]
        data["output"] = {"format": "csv"}
        parse_pipeline(data)

    def test_tif_output_after_group_following_join_accepted(self):
        data = simple_pipeline()
        data["inputs"]["sal"] = {
            "variable": "sal",
            "time_unit": "annual",
            "grid": "veg_grid_cell_v001",
        }
        data["steps"] = [
            {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
            {"op": "group", "by": [], "aggregate": "sum"},
        ]
        data["output"] = {"format": "tif"}
        parse_pipeline(data)

    def test_png_tile_output_after_join_rejected(self):
        data = simple_pipeline()
        data["inputs"]["sal"] = {
            "variable": "sal",
            "time_unit": "annual",
            "grid": "veg_grid_cell_v001",
        }
        data["steps"] = [
            {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
        ]
        data["output"] = {"format": "png"}
        with pytest.raises(PipelineError, match="join"):
            parse_pipeline(data)


class TestOutputSortSelect:
    """`output.sort`/`output.select` mirror `ScalarOptions.sort`/`.select`'s flat
    semantics exactly (see `parameters.py`); at the pipeline-representation layer they
    are just optional strings, parsed and round-tripped like `color_map`. Only
    `_compile_single_input_chain` (planner.py) actually lowers them today -- see
    `TestOutputSortSelectCompile`/`TestPairedOutputSortSelectRejection` in
    test_pipeline_exec.py for the compile-time behavior these representation-level tests
    don't cover.
    """

    def test_sort_and_select_parse(self):
        data = simple_pipeline(output={"format": "csv", "sort": "-value", "select": "grid_cell,value"})
        pipeline = parse_pipeline(data)
        assert pipeline.output.sort == "-value"
        assert pipeline.output.select == "grid_cell,value"

    def test_omitted_sort_select_default_to_none(self):
        pipeline = parse_pipeline(simple_pipeline())
        assert pipeline.output.sort is None
        assert pipeline.output.select is None

    def test_sort_and_select_round_trip_through_query(self):
        data = simple_pipeline(output={"format": "csv", "sort": "-value", "select": "grid_cell,value"})
        pipeline = parse_pipeline(data)
        query = pipeline_to_query(pipeline)
        assert query["output.sort"] == "-value"
        assert query["output.select"] == "grid_cell,value"
        pipeline2 = pipeline_from_query(query)
        assert pipeline2.output.sort == "-value"
        assert pipeline2.output.select == "grid_cell,value"

    def test_omitted_sort_select_absent_from_query(self):
        pipeline = parse_pipeline(simple_pipeline())
        query = pipeline_to_query(pipeline)
        assert "output.sort" not in query
        assert "output.select" not in query


class TestPipelineShapeRules:
    def test_empty_inputs_rejected(self):
        data = simple_pipeline()
        data["inputs"] = {}
        with pytest.raises(ValueError):
            parse_pipeline(data)

    def test_empty_steps_rejected(self):
        data = simple_pipeline()
        data["steps"] = []
        with pytest.raises(ValueError):
            parse_pipeline(data)

    def test_unknown_top_level_key_rejected(self):
        data = simple_pipeline()
        data["bogus"] = 1
        with pytest.raises(ValueError):
            parse_pipeline(data)

    def test_unknown_step_field_rejected(self):
        data = simple_pipeline()
        data["steps"][0]["bogus"] = 1
        with pytest.raises(ValueError):
            parse_pipeline(data)

    def test_direct_model_validate_only_enforces_shape(self):
        """`Pipeline.model_validate` alone does not run `validate_pipeline`'s business
        rules -- e.g. a non-decomposable aggregate is shape-valid (a real
        `AggregateFunction` member) and so is accepted here, only to be rejected by
        `validate_pipeline` explicitly. `count_by_value` is used on a `group` step here
        (not `project`, which permits it -- see `TestProjectStep.
        test_count_by_value_accepted_as_zonal_aggregate` -- `group` still rejects it, the
        one `AggregateFunction` member `fold.DECOMPOSABLE_AGGREGATES` excludes).
        """
        data = simple_pipeline()
        data["steps"].append({"op": "group", "by": [], "aggregate": "count_by_value"})
        pipeline = Pipeline.model_validate(data)
        with pytest.raises(PipelineError):
            validate_pipeline(pipeline)

    def test_parse_pipeline_wraps_shape_errors_as_pipeline_error(self):
        data = simple_pipeline()
        del data["inputs"]["marsh"]["grid"]
        with pytest.raises(PipelineError):
            parse_pipeline(data)


class TestQueryCodecRoundTrip:
    """`pipeline_to_query`/`pipeline_from_query` bijection: JSON -> query -> model ->
    query must produce the identical query dict both times, for every representable
    pipeline shape (see the module docstring's fidelity note on why this is the right
    formulation of "lossless" for a string-only encoding).
    """

    def _round_trip(self, data):
        pipeline = parse_pipeline(data)
        query1 = pipeline_to_query(pipeline)
        pipeline2 = pipeline_from_query(query1)
        query2 = pipeline_to_query(pipeline2)
        assert query1 == query2
        return pipeline, query1

    def test_simple_pipeline_round_trips(self):
        self._round_trip(simple_pipeline())

    def test_notebook_query_round_trips(self):
        self._round_trip(copy.deepcopy(NOTEBOOK_QUERY))

    def test_dimension_filter_array_round_trips(self):
        data = simple_pipeline()
        data["inputs"]["marsh"]["scenario_id"] = [7, 8]
        _, query = self._round_trip(data)
        assert query["input.marsh.scenario_id"] == "7,8"

    def test_join_inputs_list_round_trips(self):
        data = simple_pipeline()
        data["inputs"]["sal"] = {
            "variable": "sal",
            "time_unit": "annual",
            "grid": "veg_grid_cell_v001",
        }
        data["steps"] = [
            {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
        ]
        data["output"] = {"format": "csv"}
        _, query = self._round_trip(data)
        assert query["step.1.inputs"] == "marsh,sal"

    def test_group_empty_by_round_trips(self):
        data = simple_pipeline()
        data["steps"].append({"op": "group", "by": [], "aggregate": "min"})
        _, query = self._round_trip(data)
        assert query["step.2.by"] == ""

    def test_where_with_spaces_and_operators_round_trips(self):
        data = simple_pipeline()
        data["steps"].append(
            {"op": "filter", "where": "marsh > 0.5 AND marsh < 10.0"}
        )
        _, query = self._round_trip(data)
        assert query["step.2.where"] == "marsh > 0.5 AND marsh < 10.0"

    def test_where_with_comma_inside_is_not_split(self):
        # A comma cannot legally appear in a real where-clause our whitelist accepts,
        # but the codec's job is to never split `where` regardless -- verified with a
        # function call form the expression whitelist does accept.
        data = simple_pipeline()
        data["steps"].append({"op": "filter", "where": "round(marsh, 1) > 0.5"})
        pipeline = parse_pipeline(data)
        query = pipeline_to_query(pipeline)
        assert query["step.2.where"] == "round(marsh, 1) > 0.5"
        pipeline2 = pipeline_from_query(query)
        assert pipeline2.steps[-1].where == "round(marsh, 1) > 0.5"

    def test_multi_step_pipeline_round_trips(self):
        data = simple_pipeline()
        data["inputs"]["sal"] = {
            "variable": "sal",
            "time_unit": "annual",
            "grid": "veg_grid_cell_v001",
        }
        data["steps"] = [
            {"op": "project", "input": "marsh", "to": "veg_grid_cell_v001"},
            {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
            {"op": "filter", "where": "marsh > 0.5"},
            {"op": "group", "by": [], "aggregate": "min", "of": "calendar_year"},
        ]
        data["output"] = {"format": "tif"}
        self._round_trip(data)

    def test_query_keys_are_canonically_sorted(self):
        pipeline = parse_pipeline(NOTEBOOK_QUERY)
        query = pipeline_to_query(pipeline)
        keys = list(query.keys())
        assert keys == sorted(keys)
        # Re-encoding is stable and deterministic, not merely alphabetically sorted by
        # accident of iteration order.
        assert pipeline_to_query(pipeline) == query

    def test_step_indices_sort_numerically_past_nine(self):
        data = simple_pipeline()
        data["steps"] = [{"op": "project", "input": "marsh", "to": "veg_grid_cell_v001"}]
        for _ in range(10):
            data["steps"].append({"op": "filter", "where": "marsh > 0"})
        pipeline = parse_pipeline(data)
        query = pipeline_to_query(pipeline)
        keys = [k for k in query if k.startswith("step.")]
        step10_index = keys.index("step.10.op")
        step2_index = keys.index("step.2.op")
        assert step2_index < step10_index

    def test_optional_fields_omitted_from_query(self):
        pipeline = parse_pipeline(simple_pipeline())
        query = pipeline_to_query(pipeline)
        assert "step.1.as" not in query
        assert "output.color_map" not in query

    def test_count_by_value_project_aggregate_round_trips(self):
        # `count_by_value` is only special-cased at the compile layer (planner.py) --
        # the representation/codec layer treats it like any other aggregate string.
        data = simple_pipeline()
        data["steps"][0]["aggregate"] = "count_by_value"
        _, query = self._round_trip(data)
        assert query["step.1.aggregate"] == "count_by_value"


class TestQueryCodecMalformed:
    def test_empty_query_rejected(self):
        with pytest.raises(PipelineError):
            pipeline_from_query({})

    def test_unknown_top_level_key_rejected(self):
        query = pipeline_to_query(parse_pipeline(simple_pipeline()))
        query["bogus.thing"] = "x"
        with pytest.raises(PipelineError, match="bogus"):
            pipeline_from_query(query)

    def test_missing_top_level_group_rejected(self):
        query = pipeline_to_query(parse_pipeline(simple_pipeline()))
        del query["output.format"]
        with pytest.raises(PipelineError, match="output"):
            pipeline_from_query(query)

    def test_gap_in_step_list_indices_rejected(self):
        query = {
            "input.marsh.variable": "ffibs_coverage_ratio",
            "input.marsh.time_unit": "annual",
            "input.marsh.grid": "veg_grid_cell_v001",
            "step.1.op": "project",
            "step.1.input": "marsh",
            "step.1.to": "ecoregion_v001",
            "step.1.aggregate": "mean",
            # step.2 skipped -- step.3 with no step.2 is a gap
            "step.3.op": "filter",
            "step.3.where": "marsh > 0",
            "output.format": "csv",
        }
        with pytest.raises(PipelineError, match="non-contiguous|gap"):
            pipeline_from_query(query)

    def test_zero_prefixed_list_index_rejected(self):
        query = {
            "input.marsh.variable": "ffibs_coverage_ratio",
            "input.marsh.time_unit": "annual",
            "input.marsh.grid": "veg_grid_cell_v001",
            "step.01.op": "project",
            "step.01.input": "marsh",
            "step.01.to": "ecoregion_v001",
            "output.format": "csv",
        }
        with pytest.raises(PipelineError):
            pipeline_from_query(query)

    def test_leaf_and_nested_conflict_at_same_path_rejected(self):
        query = {
            "input.marsh.variable": "ffibs_coverage_ratio",
            "input.marsh.time_unit": "annual",
            "input.marsh.grid": "veg_grid_cell_v001",
            "input.marsh.grid.extra": "conflict",
            "step.1.op": "project",
            "step.1.input": "marsh",
            "step.1.to": "ecoregion_v001",
            "output.format": "csv",
        }
        with pytest.raises(PipelineError):
            pipeline_from_query(query)

    def test_unknown_step_field_rejected(self):
        query = {
            "input.marsh.variable": "ffibs_coverage_ratio",
            "input.marsh.time_unit": "annual",
            "input.marsh.grid": "veg_grid_cell_v001",
            "step.1.op": "project",
            "step.1.input": "marsh",
            "step.1.to": "ecoregion_v001",
            "step.1.bogus": "x",
            "output.format": "csv",
        }
        with pytest.raises(PipelineError, match="bogus"):
            pipeline_from_query(query)

    def test_unknown_step_op_rejected(self):
        query = {
            "input.marsh.variable": "ffibs_coverage_ratio",
            "input.marsh.time_unit": "annual",
            "input.marsh.grid": "veg_grid_cell_v001",
            "step.1.op": "not_a_real_op",
            "output.format": "csv",
        }
        with pytest.raises(PipelineError, match="not_a_real_op"):
            pipeline_from_query(query)

    def test_input_missing_named_subkeys_rejected(self):
        query = {
            "input.marsh": "not_a_group",
            "step.1.op": "project",
            "step.1.input": "marsh",
            "step.1.to": "ecoregion_v001",
            "output.format": "csv",
        }
        with pytest.raises(PipelineError):
            pipeline_from_query(query)

    def test_step_group_missing_op_rejected(self):
        query = {
            "input.marsh.variable": "ffibs_coverage_ratio",
            "input.marsh.time_unit": "annual",
            "input.marsh.grid": "veg_grid_cell_v001",
            "step.1.input": "marsh",
            "step.1.to": "ecoregion_v001",
            "output.format": "csv",
        }
        with pytest.raises(PipelineError, match="op"):
            pipeline_from_query(query)
