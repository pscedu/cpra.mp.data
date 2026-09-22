"""Tests for the request-layer `Pipeline` representation and its structural, plan-time
validation (`cpra.mp.data.analysis.request.pipeline`) -- name shapes, chaining, `as`
collisions, per-step field rules that need no more than the step itself, and both
encodings (JSON body, dotted GET/tile query string). Shape-dependent rules (whether a
`project` crosses grid kinds legally, join operand compatibility, the output-format sink
rule) are `request.ops.validate_shapes`'s job and are exercised in
`test_request_ops.py` instead -- see `pipeline.py`'s own module docstring for why the two
live apart.

Real variables/grids/dimensions from `cpra/mp/data/structure/` are used throughout (no
fake dimensions), following the existing `test_pipeline.py`'s own convention.
"""

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from cpra.mp.data.analysis.request.errors import RequestError
from cpra.mp.data.analysis.request.pipeline import (
    _INPUT_FIELDS,
    _OUTPUT_FIELDS,
    _STEP_FIELDS,
    _STEP_LIST_FIELDS,
    _STEP_MODELS,
    _query_fields,
    _query_list_fields,
    InputSelection,
    Output,
    Pipeline,
    append_steps,
    parse_pipeline,
    pipeline_from_query,
    pipeline_to_query,
    validate_pipeline,
)

VECTOR_GRID = "hydro_compartment_v001"
VECTOR_GRID_2 = "ecoregion_v001"
RASTER_GRID = "veg_grid_cell_v001"


def simple_pipeline(**overrides):
    """A minimal one-input, one-step pipeline, valid unless overridden."""
    data = {
        "inputs": {
            "marsh": {
                "variable": "ffibs_coverage_ratio",
                "time_unit": "annual",
                "grid": RASTER_GRID,
            },
        },
        "steps": [{"op": "project", "input": "marsh", "to": VECTOR_GRID_2, "aggregate": "mean"}],
        "output": {"format": "csv"},
    }
    data.update(overrides)
    return data


def simple_query(**extra):
    """`simple_pipeline`'s equivalent in the dotted query encoding, as a plain mutable
    dict -- malformed-input tests start from a well-formed query and break exactly one
    thing, so what the resulting error is attributable to is never in doubt.
    """
    params = {
        "input.marsh.variable": "ffibs_coverage_ratio",
        "input.marsh.time_unit": "annual",
        "input.marsh.grid": RASTER_GRID,
        "step.1.op": "project",
        "step.1.input": "marsh",
        "step.1.to": VECTOR_GRID_2,
        "step.1.aggregate": "mean",
        "output.format": "csv",
    }
    params.update(extra)
    return params


class TestInputSelectionIsNarrowed:
    """`InputSelection` is deliberately narrowed to `variable`/`time_unit`/`grid`
    only -- an input NAMES a dataset, it does not filter one; every filter (dimension,
    value, or spatial) is a `filter` step instead, unlike the legacy `InputSelection`
    that also accepted arbitrary extra dimension-filter fields.
    """

    def test_bare_selection_parses(self):
        pipeline = parse_pipeline(simple_pipeline())
        assert set(pipeline.inputs["marsh"].model_dump()) == {"variable", "time_unit", "grid"}

    def test_extra_dimension_filter_field_is_rejected(self):
        data = simple_pipeline()
        data["inputs"]["marsh"]["scenario_id"] = 7
        with pytest.raises(ValueError):
            parse_pipeline(data)

    def test_grid_cell_field_is_rejected(self):
        """The legacy input carried an optional `grid_cell` field too -- narrowed away
        along with the dimension filters, since spatial narrowing is a `filter` step's
        job now, same as any other predicate kind."""
        data = simple_pipeline()
        data["inputs"]["marsh"]["grid_cell"] = "abc"
        with pytest.raises(ValueError):
            parse_pipeline(data)

    def test_unknown_variable_rejected(self):
        data = simple_pipeline()
        data["inputs"]["marsh"]["variable"] = "not_a_real_variable"
        with pytest.raises(ValueError):
            parse_pipeline(data)


class TestNameShapes:
    def test_input_name_must_be_snake_case(self):
        data = simple_pipeline()
        data["inputs"] = {"Marsh": data["inputs"]["marsh"]}
        data["steps"][0]["input"] = "Marsh"
        with pytest.raises(RequestError, match="must match"):
            parse_pipeline(data)

    def test_input_name_reserved_word_rejected(self):
        data = simple_pipeline()
        data["inputs"] = {"value": data["inputs"]["marsh"]}
        data["steps"][0]["input"] = "value"
        with pytest.raises(RequestError, match="reserved"):
            parse_pipeline(data)

    def test_grid_cell_reserved_as_input_name(self):
        data = simple_pipeline()
        data["inputs"] = {"grid_cell": data["inputs"]["marsh"]}
        data["steps"][0]["input"] = "grid_cell"
        with pytest.raises(RequestError, match="reserved"):
            parse_pipeline(data)

    def test_band_name_pattern_rejected(self):
        data = simple_pipeline()
        data["inputs"] = {"b0": data["inputs"]["marsh"]}
        data["steps"][0]["input"] = "b0"
        with pytest.raises(RequestError, match="positional band-name"):
            parse_pipeline(data)

    def test_input_name_colliding_with_a_dimension_name_rejected(self):
        """A step or input name that shadows a real dimension the data structure
        declares (e.g. `aep`) only bites once a `join` renames a value column to that
        name -- rejected up front instead of downstream."""
        data = simple_pipeline()
        data["inputs"] = {"aep": data["inputs"]["marsh"]}
        data["steps"][0]["input"] = "aep"
        with pytest.raises(RequestError, match="dimension"):
            parse_pipeline(data)

    def test_step_as_name_colliding_with_a_dimension_name_rejected(self):
        data = simple_pipeline()
        data["steps"][0]["as"] = "scenario_id"
        with pytest.raises(RequestError, match="dimension"):
            parse_pipeline(data)

    def test_step_as_name_must_be_snake_case(self):
        data = simple_pipeline()
        data["steps"][0]["as"] = "NotSnakeCase"
        with pytest.raises(RequestError, match="must match"):
            parse_pipeline(data)


class TestChaining:
    def test_omitted_input_chains_from_previous_step(self):
        data = simple_pipeline()
        data["steps"] = [
            {"op": "project", "input": "marsh", "to": VECTOR_GRID_2, "aggregate": "mean", "as": "a"},
            {"op": "filter", "where": "value > 0.1"},
        ]
        pipeline = parse_pipeline(data)
        assert pipeline.steps[1].input is None

    def test_first_step_with_no_explicit_input_rejected(self):
        data = simple_pipeline()
        data["steps"] = [{"op": "filter", "where": "value > 0.1"}]
        with pytest.raises(RequestError, match="no input to chain from"):
            parse_pipeline(data)

    def test_reference_to_unknown_name_rejected(self):
        data = simple_pipeline()
        data["steps"][0]["input"] = "does_not_exist"
        with pytest.raises(RequestError, match="unknown"):
            parse_pipeline(data)

    def test_forward_reference_to_later_as_rejected(self):
        data = simple_pipeline()
        data["steps"] = [
            {"op": "filter", "input": "later", "where": "value > 0.1"},
            {"op": "project", "input": "marsh", "to": VECTOR_GRID_2, "aggregate": "mean", "as": "later"},
        ]
        with pytest.raises(RequestError, match="unknown"):
            parse_pipeline(data)

    def test_group_without_an_input_cannot_be_first_step(self):
        """`group` chains like every other non-`join` op, so a leading one with no
        `input` fails the first-step rule for the same reason a leading `filter` does."""
        data = simple_pipeline()
        data["steps"] = [{"op": "group", "by": [], "aggregate": "sum"}]
        with pytest.raises(RequestError, match="no input to chain from"):
            parse_pipeline(data)

    def test_group_naming_its_input_can_be_first_step(self):
        """"Read one dataset and collapse it" is a whole pipeline, and naming the input
        on the `group` is what expresses it -- no filter in front purely to chain from.
        """
        data = simple_pipeline()
        data["steps"] = [{"op": "group", "input": "marsh", "by": [], "aggregate": "sum"}]
        pipeline = parse_pipeline(data)
        assert pipeline.steps[0].input == "marsh"

    def test_a_dangling_as_that_is_never_referenced_is_not_an_error(self):
        data = simple_pipeline()
        data["steps"][0]["as"] = "unused"
        pipeline = parse_pipeline(data)
        assert pipeline.steps[0].as_ == "unused"


class TestAsCollisions:
    def test_as_colliding_with_input_name_rejected(self):
        data = simple_pipeline()
        data["steps"][0]["as"] = "marsh"
        with pytest.raises(RequestError, match="collides"):
            parse_pipeline(data)

    def test_as_colliding_with_earlier_step_as_rejected(self):
        data = simple_pipeline()
        data["steps"] = [
            {"op": "project", "input": "marsh", "to": VECTOR_GRID_2, "aggregate": "mean", "as": "dup"},
            {"op": "filter", "where": "value > 0.1", "as": "dup"},
        ]
        with pytest.raises(RequestError, match="collides"):
            parse_pipeline(data)


class TestJoinStep:
    def test_attribute_join_rejected_as_reserved(self):
        data = {
            "inputs": {
                "a": {"variable": "sal", "time_unit": "annual", "grid": VECTOR_GRID},
                "b": {"variable": "sal", "time_unit": "annual", "grid": VECTOR_GRID},
            },
            "steps": [{"op": "join", "type": "attribute", "inputs": ["a", "b"], "as": "j"}],
            "output": {"format": "csv"},
        }
        with pytest.raises(RequestError, match="attribute"):
            parse_pipeline(data)

    def test_spatial_join_with_two_inputs_accepted(self):
        data = {
            "inputs": {
                "a": {"variable": "sal", "time_unit": "annual", "grid": VECTOR_GRID},
                "b": {"variable": "sal", "time_unit": "annual", "grid": VECTOR_GRID},
            },
            "steps": [{"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"}],
            "output": {"format": "csv"},
        }
        pipeline = parse_pipeline(data)
        assert pipeline.steps[0].inputs == ["a", "b"]

    def test_join_requires_at_least_two_inputs(self):
        data = {
            "inputs": {"a": {"variable": "sal", "time_unit": "annual", "grid": VECTOR_GRID}},
            "steps": [{"op": "join", "type": "spatial", "inputs": ["a"], "as": "j"}],
            "output": {"format": "csv"},
        }
        with pytest.raises(ValueError):
            parse_pipeline(data)

    def test_join_rejects_duplicate_input_name(self):
        data = {
            "inputs": {"a": {"variable": "sal", "time_unit": "annual", "grid": VECTOR_GRID}},
            "steps": [{"op": "join", "type": "spatial", "inputs": ["a", "a"], "as": "j"}],
            "output": {"format": "csv"},
        }
        with pytest.raises(RequestError, match="duplicate"):
            parse_pipeline(data)


class TestFilterStep:
    def test_empty_where_rejected(self):
        data = simple_pipeline()
        data["steps"].append({"op": "filter", "where": ""})
        with pytest.raises(RequestError, match="non-empty"):
            parse_pipeline(data)

    def test_dimension_predicate_accepted(self):
        data = simple_pipeline()
        data["steps"].append({"op": "filter", "where": "calendar_year > 2040"})
        pipeline = parse_pipeline(data)
        assert pipeline.steps[1].where == "calendar_year > 2040"

    def test_value_predicate_accepted(self):
        data = simple_pipeline()
        data["steps"].append({"op": "filter", "where": "value > 0.5"})
        parse_pipeline(data)

    def test_spatial_predicate_accepted(self):
        data = simple_pipeline()
        data["steps"].append({"op": "filter", "where": "grid_cell = 'abc'"})
        parse_pipeline(data)

    def test_aggregate_call_in_dimension_conjunct_rejected(self):
        data = simple_pipeline()
        data["steps"].append({"op": "filter", "where": "calendar_year = min(calendar_year)"})
        with pytest.raises(RequestError, match="aggregate"):
            parse_pipeline(data)


class TestCalculateStep:
    def test_ordinary_expression_accepted(self):
        data = simple_pipeline()
        data["steps"].append({"op": "calculate", "expression": "value * 2"})
        parse_pipeline(data)

    def test_invalid_expression_rejected(self):
        data = simple_pipeline()
        data["steps"].append({"op": "calculate", "expression": "value > 3"})
        with pytest.raises(RequestError):
            parse_pipeline(data)

    def test_calculate_before_count_by_value_accepted(self):
        """Neither ordering of `calculate` and a `count_by_value` project is rejected --
        see `test_request_ops.py::TestCountByValueCalculateOrdering` for both orders
        proven to pass full shape typing too, not merely structural parsing."""
        data = simple_pipeline()
        data["steps"] = [
            {"op": "calculate", "input": "marsh", "expression": "value * 2", "as": "scaled"},
            {"op": "project", "input": "scaled", "to": VECTOR_GRID_2, "aggregate": "count_by_value"},
        ]
        parse_pipeline(data)

    def test_count_by_value_before_calculate_accepted(self):
        data = simple_pipeline()
        data["steps"] = [
            {"op": "project", "input": "marsh", "to": VECTOR_GRID_2, "aggregate": "count_by_value", "as": "counted"},
            {"op": "calculate", "input": "counted", "expression": "value * 2"},
        ]
        parse_pipeline(data)


class TestGroupStep:
    def test_count_by_value_rejected_as_group_aggregate(self):
        data = simple_pipeline()
        data["steps"].append({"op": "group", "by": [], "aggregate": "count_by_value"})
        with pytest.raises(RequestError, match="count_by_value"):
            parse_pipeline(data)

    def test_non_decomposable_aggregate_rejected(self):
        data = simple_pipeline()
        data["steps"].append({"op": "group", "by": [], "aggregate": "not_a_real_aggregate"})
        with pytest.raises(ValueError):
            parse_pipeline(data)

    def test_value_in_group_by_rejected(self):
        data = simple_pipeline()
        data["steps"].append({"op": "group", "by": ["value"], "aggregate": "sum"})
        with pytest.raises(RequestError, match="value"):
            parse_pipeline(data)

    def test_ordinary_group_accepted(self):
        data = simple_pipeline()
        data["steps"].append({"op": "group", "by": [], "aggregate": "sum", "of": "calendar_year"})
        pipeline = parse_pipeline(data)
        assert pipeline.steps[-1].of == "calendar_year"


class TestZeroStepPipelines:
    def test_zero_step_pipeline_with_one_input_accepted(self):
        data = {
            "inputs": {"marsh": {"variable": "ffibs_coverage_ratio", "time_unit": "annual", "grid": RASTER_GRID}},
            "steps": [],
            "output": {"format": "csv"},
        }
        pipeline = parse_pipeline(data)
        assert pipeline.steps == []

    def test_zero_step_pipeline_omitting_steps_key_accepted(self):
        data = {
            "inputs": {"marsh": {"variable": "ffibs_coverage_ratio", "time_unit": "annual", "grid": RASTER_GRID}},
            "output": {"format": "csv"},
        }
        pipeline = parse_pipeline(data)
        assert pipeline.steps == []

    def test_zero_step_pipeline_with_two_inputs_rejected(self):
        """Nothing selects among several inputs with no step to combine or choose
        between them."""
        data = {
            "inputs": {
                "a": {"variable": "ffibs_coverage_ratio", "time_unit": "annual", "grid": RASTER_GRID},
                "b": {"variable": "ffibs_coverage_ratio", "time_unit": "annual", "grid": RASTER_GRID},
            },
            "steps": [],
            "output": {"format": "csv"},
        }
        with pytest.raises(RequestError, match="exactly one input"):
            parse_pipeline(data)

    def test_zero_step_pipeline_round_trips_through_query(self):
        data = {
            "inputs": {"marsh": {"variable": "ffibs_coverage_ratio", "time_unit": "annual", "grid": RASTER_GRID}},
            "steps": [],
            "output": {"format": "csv"},
        }
        pipeline = parse_pipeline(data)
        query = pipeline_to_query(pipeline)
        assert not any(key.startswith("step.") for key in query)
        pipeline2 = pipeline_from_query(query)
        assert pipeline2.steps == []
        assert pipeline_to_query(pipeline2) == query


class TestAppendSteps:
    def test_appended_step_chains_from_stored_pipelines_last_step(self):
        stored = parse_pipeline(simple_pipeline())
        combined = append_steps(
            stored,
            [{"op": "filter", "where": "value > 0.2"}],
            output={"format": "csv"},
        )
        assert isinstance(combined, Pipeline)
        assert len(combined.steps) == 2
        assert combined.steps[-1].where == "value > 0.2"

    def test_appended_step_on_a_zero_step_stored_pipeline_must_name_its_input(self):
        stored = parse_pipeline(
            {
                "inputs": {"marsh": {"variable": "ffibs_coverage_ratio", "time_unit": "annual", "grid": RASTER_GRID}},
                "steps": [],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(RequestError, match="no input to chain from"):
            append_steps(stored, [{"op": "filter", "where": "value > 0.2"}], output={"format": "csv"})

        combined = append_steps(
            stored,
            [{"op": "filter", "input": "marsh", "where": "value > 0.2"}],
            output={"format": "csv"},
        )
        assert len(combined.steps) == 1

    def test_appended_steps_output_replaces_stored_output(self):
        stored = parse_pipeline(simple_pipeline())
        combined = append_steps(stored, [], output={"format": "ndjson"})
        assert combined.output.format == "ndjson"
        assert stored.output.format == "csv"  # the stored pipeline itself is untouched


class TestQueryCodec:
    def test_round_trips_a_multi_step_pipeline(self):
        data = {
            "inputs": {
                "marsh": {
                    "variable": "ffibs_coverage_ratio",
                    "time_unit": "annual",
                    "grid": RASTER_GRID,
                },
                "salinity": {
                    "variable": "sal",
                    "time_unit": "annual",
                    "grid": VECTOR_GRID,
                },
            },
            "steps": [
                {"op": "project", "input": "salinity", "to": RASTER_GRID, "as": "sal_r"},
                {"op": "join", "type": "spatial", "inputs": ["marsh", "sal_r"], "as": "paired"},
                {"op": "filter", "where": "marsh > 0.5 AND sal_r > 5.5"},
                {"op": "group", "by": [], "aggregate": "min", "of": "calendar_year"},
            ],
            "output": {"format": "tif"},
        }
        pipeline = parse_pipeline(data)
        query = pipeline_to_query(pipeline)
        pipeline2 = pipeline_from_query(query)
        assert pipeline_to_query(pipeline2) == query

    def test_round_trips_a_group_that_names_its_input(self):
        data = simple_pipeline()
        data["steps"] = [{"op": "group", "input": "marsh", "by": ["calendar_year"], "aggregate": "sum"}]
        query = pipeline_to_query(parse_pipeline(data))
        assert query["step.1.input"] == "marsh"
        assert pipeline_from_query(query).steps[0].input == "marsh"

    def test_query_keys_are_naturally_sorted(self):
        data = simple_pipeline()
        data["steps"] = [
            {"op": "project", "input": "marsh", "to": VECTOR_GRID_2, "aggregate": "mean", "as": f"s{i}"}
            for i in range(1, 12)
        ]
        pipeline = parse_pipeline(data)
        query = pipeline_to_query(pipeline)
        step_keys = [k for k in query if k.startswith("step.")]
        assert step_keys == sorted(step_keys, key=lambda k: (int(k.split(".")[1]), k))

    def test_missing_required_top_level_group_rejected(self):
        with pytest.raises(RequestError, match="missing"):
            pipeline_from_query({"output.format": "csv"})

    def test_unknown_top_level_group_rejected(self):
        with pytest.raises(RequestError, match="unknown"):
            pipeline_from_query({"bogus.thing": "1", "input.a.variable": "sal", "output.format": "csv"})

    def test_empty_query_rejected(self):
        with pytest.raises(RequestError, match="empty"):
            pipeline_from_query({})

    def test_round_trips_every_optional_field_set(self):
        """The round-trip above exercises the fields a pipeline cannot omit. This one
        sets every field the encoder is allowed to DROP when it holds a default or
        `None` -- each step type's `as`, `calculate.where`, `group.of` at a non-default,
        and all three optional output fields -- because dropping a field and reviving the
        same default on decode is what makes the round-trip stable rather than merely
        reversible once, and a field that silently fails to survive would otherwise show
        up first as a cache key that changes on every re-encode.
        """
        data = {
            "inputs": {
                "marsh": {
                    "variable": "ffibs_coverage_ratio",
                    "time_unit": "annual",
                    "grid": RASTER_GRID,
                },
            },
            "steps": [
                {"op": "filter", "input": "marsh", "where": "calendar_year > 2040", "as": "recent"},
                {"op": "calculate", "expression": "value * 2", "where": "value > 0", "as": "doubled"},
                {"op": "project", "to": VECTOR_GRID_2, "aggregate": "mean", "as": "projected"},
                {"op": "group", "by": ["calendar_year"], "aggregate": "min", "of": "calendar_year", "as": "earliest"},
            ],
            "output": {
                "format": "csv",
                "color_map": "viridis",
                "sort": "-value",
                "select": "grid_cell,value",
            },
        }
        pipeline = parse_pipeline(data)
        query = pipeline_to_query(pipeline)
        assert query["step.1.as"] == "recent"
        assert query["step.2.where"] == "value > 0"
        assert query["step.4.of"] == "calendar_year"
        assert query["output.color_map"] == "viridis"
        assert query["output.select"] == "grid_cell,value"
        assert pipeline_to_query(pipeline_from_query(query)) == query

    def test_filter_and_calculate_carry_their_explicit_input_through_the_encoding(self):
        """`filter` and `calculate` both take an optional `input`, and the query encoding
        has to carry it like `project.input`. Dropping it does not fail loudly: a first
        step with nowhere to chain from raises, but a LATER step that named an earlier
        `as` silently re-binds to whatever step happens to precede it, computing a
        different pipeline than the one that was sent. Both are asserted here.
        """
        data = simple_pipeline()
        data["steps"] = [
            {"op": "filter", "input": "marsh", "where": "calendar_year > 2040", "as": "recent"},
            {"op": "calculate", "input": "marsh", "expression": "value * 2", "as": "doubled"},
            {"op": "filter", "input": "recent", "where": "value > 0"},
        ]
        query = pipeline_to_query(parse_pipeline(data))
        assert query["step.1.input"] == "marsh"
        assert query["step.2.input"] == "marsh"
        assert query["step.3.input"] == "recent"
        # The branch is what matters: step 3 reads `recent`, not the `doubled` that
        # immediately precedes it, and chaining-by-position would quietly swap the two.
        assert pipeline_from_query(query).steps[2].input == "recent"

    def test_default_of_is_omitted_from_the_encoding(self):
        data = simple_pipeline()
        data["steps"].append({"op": "group", "by": [], "aggregate": "sum", "of": "value"})
        query = pipeline_to_query(parse_pipeline(data))
        assert "step.2.of" not in query
        assert pipeline_from_query(query).steps[1].of == "value"


class TestQueryCodecMalformedInput:
    """The dotted encoding is the outermost surface of the request layer -- a tile URL
    is where malformed input actually arrives -- so every way the decoder can refuse a
    parameter set is pinned here. A branch that is unreachable in a JSON body (a leaf and
    a nested value fighting over one path, a gap in list indices) is reachable here,
    where the client writes the structure out one flat key at a time.
    """

    def test_empty_parameter_name_rejected(self):
        with pytest.raises(RequestError, match="invalid parameter name"):
            pipeline_from_query({"": "x"})

    def test_non_string_parameter_value_rejected(self):
        with pytest.raises(RequestError, match="must have a string value"):
            pipeline_from_query({"output.format": 5})

    def test_leaf_then_nested_at_the_same_path_rejected(self):
        params = simple_query()
        params["step.1.to.nested"] = "x"  # `step.1.to` is already a leaf
        with pytest.raises(RequestError, match="conflicts with a leaf value"):
            pipeline_from_query(params)

    def test_nested_then_leaf_at_the_same_path_rejected(self):
        params = {"input.marsh.variable.nested": "x", "input.marsh.variable": "sal"}
        with pytest.raises(RequestError, match="conflicts with other parameter"):
            pipeline_from_query(params)

    def test_mixed_list_index_and_named_segments_rejected(self):
        with pytest.raises(RequestError, match="mixes list-index segments"):
            pipeline_from_query(simple_query(**{"step.foo": "bar"}))

    def test_zero_list_index_rejected(self):
        """List indices are 1-based, matching how the encoder writes them out; a `step.0`
        would otherwise decode into a position no `pipeline_to_query` can ever produce.
        """
        params = {k: v for k, v in simple_query().items() if not k.startswith("step.1.")}
        params.update({"step.0.op": "project", "step.0.input": "marsh", "step.0.to": VECTOR_GRID_2})
        with pytest.raises(RequestError, match="invalid list index"):
            pipeline_from_query(params)

    def test_leading_zero_list_index_rejected(self):
        params = {k: v for k, v in simple_query().items() if not k.startswith("step.1.")}
        params.update({"step.01.op": "project", "step.01.input": "marsh", "step.01.to": VECTOR_GRID_2})
        with pytest.raises(RequestError, match="invalid list index"):
            pipeline_from_query(params)

    def test_non_contiguous_list_indices_rejected(self):
        """A gap means a step was dropped in transit. Decoding 1 and 3 as a two-step
        pipeline would silently renumber the client's step 3 into position 2 and run it
        against the wrong predecessor.
        """
        with pytest.raises(RequestError, match="non-contiguous"):
            pipeline_from_query(simple_query(**{"step.3.op": "group", "step.3.aggregate": "sum", "step.3.by": ""}))

    def test_input_group_as_a_leaf_rejected(self):
        params = {k: v for k, v in simple_query().items() if not k.startswith("input.")}
        params["input"] = "marsh"
        with pytest.raises(RequestError, match="named sub-keys"):
            pipeline_from_query(params)

    def test_one_input_as_a_leaf_rejected(self):
        params = {k: v for k, v in simple_query().items() if not k.startswith("input.")}
        params["input.marsh"] = "ffibs_coverage_ratio"
        with pytest.raises(RequestError, match="named sub-keys"):
            pipeline_from_query(params)

    def test_unknown_input_field_rejected(self):
        with pytest.raises(RequestError, match="unknown field"):
            pipeline_from_query(simple_query(**{"input.marsh.bogus": "x"}))

    def test_nested_value_under_an_input_field_rejected(self):
        params = {k: v for k, v in simple_query().items() if k != "input.marsh.variable"}
        params["input.marsh.variable.nested"] = "x"
        with pytest.raises(RequestError, match="scalar query value"):
            pipeline_from_query(params)

    def test_step_group_with_named_keys_rejected(self):
        params = {k: v for k, v in simple_query().items() if not k.startswith("step.")}
        params["step.op"] = "project"
        with pytest.raises(RequestError, match="1-based list indices"):
            pipeline_from_query(params)

    def test_step_as_a_leaf_rejected(self):
        params = {k: v for k, v in simple_query().items() if not k.startswith("step.")}
        params["step.1"] = "project"
        with pytest.raises(RequestError, match="named sub-keys"):
            pipeline_from_query(params)

    def test_step_missing_op_rejected(self):
        params = {k: v for k, v in simple_query().items() if k != "step.1.op"}
        with pytest.raises(RequestError, match="missing required field 'op'"):
            pipeline_from_query(params)

    def test_unknown_step_op_rejected(self):
        with pytest.raises(RequestError, match="unknown op"):
            pipeline_from_query(simple_query(**{"step.1.op": "frobnicate"}))

    def test_unknown_step_field_rejected(self):
        with pytest.raises(RequestError, match="unknown field"):
            pipeline_from_query(simple_query(**{"step.1.bogus": "x"}))

    def test_nested_value_under_a_step_field_rejected(self):
        params = {k: v for k, v in simple_query().items() if k != "step.1.to"}
        params["step.1.to.nested"] = "x"
        with pytest.raises(RequestError, match="scalar query value"):
            pipeline_from_query(params)

    def test_output_group_as_a_leaf_rejected(self):
        params = {k: v for k, v in simple_query().items() if not k.startswith("output.")}
        params["output"] = "csv"
        with pytest.raises(RequestError, match="named sub-keys"):
            pipeline_from_query(params)

    def test_unknown_output_field_rejected(self):
        with pytest.raises(RequestError, match="unknown field"):
            pipeline_from_query(simple_query(**{"output.bogus": "x"}))

    def test_output_compression_is_rejected_rather_than_ignored(self):
        """`output.compression` used to decode and round-trip here while nothing ever
        read it, so a caller who asked for gzip silently got the default. Both encodings
        reject it outright now; compression is the `compression_algorithm`/
        `compression_level` QUERY parameters (`api.models.Compression`), which are
        stripped from the query dict before it ever reaches this decoder.
        """
        with pytest.raises(RequestError, match="unknown field"):
            pipeline_from_query(simple_query(**{"output.compression": "gzip"}))

        data = simple_pipeline()
        data["output"]["compression"] = "gzip"
        with pytest.raises(RequestError):
            parse_pipeline(data)

    def test_a_well_formed_query_expressing_a_structurally_invalid_pipeline_is_still_rejected(self):
        """Once the dotted structure resolves, the result goes through the same
        `parse_pipeline` a JSON body does -- so a query that is well-formed AS A QUERY
        still answers to every structural rule, rather than reaching the planner because
        it arrived by URL instead of by POST.
        """
        with pytest.raises(RequestError, match="collides"):
            pipeline_from_query(simple_query(**{"step.1.as": "marsh"}))


def test_empty_input_name_rejected():
    with pytest.raises(RequestError, match="must not be empty"):
        parse_pipeline(simple_pipeline(inputs={"": {"variable": "sal", "time_unit": "annual", "grid": VECTOR_GRID}}))


def test_validate_pipeline_can_be_called_directly_on_an_already_shape_valid_pipeline():
    pipeline = Pipeline.model_validate(simple_pipeline())
    validate_pipeline(pipeline)  # does not raise


class TestTheQueryCodecsFieldTablesTrackTheModels:
    """The dotted query codec whitelists field names per op, and those whitelists are
    DERIVED from the step models rather than mirrored from them.

    The failure this guards against is concrete and has happened: adding `input` to
    `GroupStep` left the hand-kept table behind, so a legal `step.N.input` on a `group`
    was rejected as an unknown field until the table was chased by hand. These tests fail
    on the model side of that drift, not on a second copy of the answer.
    """

    def test_the_derived_tables_spell_out_what_each_op_accepts(self):
        """Written out rather than re-derived from `model_fields`: repeating the
        derivation here would assert only that one expression equals itself, and would
        pass just as happily if `_query_fields` dropped every aliased field.
        """
        assert _STEP_FIELDS == {
            "project": {"op", "input", "to", "aggregate", "as"},
            "join": {"op", "type", "inputs", "as"},
            "filter": {"op", "input", "where", "as"},
            "calculate": {"op", "input", "expression", "where", "as"},
            "group": {"op", "input", "by", "aggregate", "of", "as"},
        }

    def test_the_tables_cover_exactly_the_ops_the_discriminator_accepts(self):
        ops = {model.model_fields["op"].default for model in _STEP_MODELS}
        assert set(_STEP_FIELDS) == ops
        assert set(_STEP_LIST_FIELDS) == ops

    def test_only_list_valued_fields_are_comma_split(self):
        """A `str` field must never be split: `where`/`expression` carry commas verbatim
        inside a single SQL-flavored or arithmetic expression."""
        assert _STEP_LIST_FIELDS["join"] == {"inputs"}
        assert _STEP_LIST_FIELDS["group"] == {"by"}
        for op in ("project", "filter", "calculate"):
            assert _STEP_LIST_FIELDS[op] == frozenset(), op
        for op, list_fields in _STEP_LIST_FIELDS.items():
            assert list_fields <= _STEP_FIELDS[op], op

    def test_an_optional_list_field_would_still_be_comma_split(self):
        """No step declares `list[str] | None` today, so nothing in the real models
        exercises the union arm of `_is_list_annotation` -- but a future optional list
        that quietly stopped being split would be a wrong answer rather than an error,
        which is why the arm exists and why it is pinned here against a stand-in.
        """

        class OptionalListStep(BaseModel):
            op: Literal["stand_in"] = "stand_in"
            maybe: list[str] | None = None
            plain: str = ""
            renamed: str = Field(default="", alias="as")

        assert _query_list_fields(OptionalListStep) == {"maybe"}
        assert _query_fields(OptionalListStep) == {"op", "maybe", "plain", "as"}

    def test_the_tables_carry_the_alias_not_the_python_field_name(self):
        """`as_` exists only because `as` is a Python keyword; a URL spells `as`."""
        for op, fields in _STEP_FIELDS.items():
            assert "as" in fields, op
            assert "as_" not in fields, op

    def test_the_input_and_output_tables_are_derived_the_same_way(self):
        assert _INPUT_FIELDS == {"variable", "time_unit", "grid"}
        assert _OUTPUT_FIELDS == {"format", "color_map", "sort", "select"}
        assert _INPUT_FIELDS == set(InputSelection.model_fields)
        assert _OUTPUT_FIELDS == set(Output.model_fields)

    def test_a_group_steps_input_field_survives_the_round_trip(self):
        """The regression itself, end to end rather than through the table: the field
        that drifted is spellable, decodes, and comes back out of `pipeline_to_query`.
        """
        params = {
            "input.marsh.variable": "ffibs_coverage_ratio",
            "input.marsh.time_unit": "annual",
            "input.marsh.grid": VECTOR_GRID,
            "step.1.op": "group",
            "step.1.input": "marsh",
            "step.1.by": "calendar_year",
            "step.1.aggregate": "mean",
            "output.format": "csv",
        }
        pipeline = pipeline_from_query(params)
        assert pipeline.steps[0].input == "marsh"
        assert pipeline_to_query(pipeline)["step.1.input"] == "marsh"
