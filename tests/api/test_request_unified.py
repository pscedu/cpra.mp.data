"""`analysis.request.unified` -- the pure half of the unified decoder: the five-category
partition of a query string and the pipeline-assembly primitives.

Nothing here builds an app or opens a file. The orchestration half (resolving `analysis=`
against the registry, resolving a coordinate `grid_cell` against real geometry) lives in
`api/unified.py` and is tested in `tests/api/test_unified_decoder.py`.
"""

import pytest

from cpra.mp.data.analysis.request.errors import RequestError
from cpra.mp.data.analysis.request.pipeline import pipeline_from_query, pipeline_to_query
from cpra.mp.data.analysis.request.unified import (
    DOWNLOAD,
    RETIRED_PARAMS,
    append_selection,
    build_pipeline,
    download_pipeline,
    partition_query,
    retired_param_message,
    retired_params,
)

DATASET = {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"}


def _items(**kwargs):
    """A single-valued query string as the `(key, value)` pairs `partition_query` takes."""
    return list(kwargs.items())


class TestPartition:
    def test_each_category_lands_where_it_belongs(self):
        partition = partition_query(
            _items(
                analysis="diff",
                variable="ssf",
                time_unit="annual",
                grid="ecoregion_v001",
                calendar_year="2030",
                where="value > 3",
                grid_cell="1,2",
                sort="-value",
                compression_level="9",
                dimension="calendar_year",
            )
        )
        assert partition.analysis == ("diff",)
        assert partition.dataset == DATASET
        assert partition.filters == {
            "calendar_year": ["2030"],
            "where": "value > 3",
            "grid_cell": "1,2",
        }
        assert partition.output == {"sort": "-value"}
        assert partition.compression == {"compression_level": "9"}
        # `dimension`/`order` are `diff`'s own declared parameters; nothing here knows
        # that, which is the point -- they are simply what is left.
        assert partition.params == {"dimension": "calendar_year"}

    def test_a_repeated_dimension_accumulates(self):
        """The portal spells a multi-value selection as repeated keys (`getFilterParams`
        pushes one pair per value), so the partition has to consume `multi_items()` rather
        than a flattened dict -- a `dict(query)` would keep only the last year and answer
        200 with one year's data for a two-year `diff`."""
        partition = partition_query([("calendar_year", "2030"), ("calendar_year", "2040"), ("scenario_id", "7")])
        assert partition.filters == {"calendar_year": ["2030", "2040"], "scenario_id": ["7"]}

    def test_a_repeated_scalar_key_is_last_wins(self):
        """Every non-dimension category is scalar, and last-wins is what a duplicated
        query parameter has always meant through FastAPI's own binding."""
        partition = partition_query([("sort", "value"), ("sort", "-value")])
        assert partition.output == {"sort": "-value"}

    def test_dotted_pipeline_params_pass_through_untouched(self):
        partition = partition_query(
            _items(**{"input.data.variable": "ssf", "step.1.op": "filter", "output.select": "value"})
        )
        assert partition.pipeline_params == {
            "input.data.variable": "ssf",
            "step.1.op": "filter",
            "output.select": "value",
        }
        assert partition.params == {} and partition.output == {}

    def test_a_bare_group_name_is_not_a_pipeline_param(self):
        """`input`/`step`/`output` are group PREFIXES, not parameter names -- a bare
        `output=` has no dotted field after it and so is not something
        `pipeline_from_query` could decode. It falls through to the leftover check, which
        is what turns it into a 400 rather than a silently-dropped key."""
        with pytest.raises(RequestError, match="unknown query parameter"):
            partition_query(_items(output="csv", **DATASET))

    def test_analysis_download_is_the_empty_chain(self):
        assert partition_query(_items(analysis=DOWNLOAD)).analysis == ()
        assert partition_query(_items(analysis=DOWNLOAD)).is_download
        assert partition_query([]).is_download

    def test_a_comma_chain_splits_and_strips(self):
        assert partition_query(_items(analysis="lnd_conf, diff")).analysis == ("lnd_conf", "diff")

    def test_an_empty_chain_stage_is_rejected(self):
        with pytest.raises(RequestError, match="empty chain stage"):
            partition_query(_items(analysis="lnd_conf,"))

    def test_a_leftover_under_download_is_rejected(self):
        """The one place the old flat routes answered 200 with the wrong thing: a
        misspelled dimension was ignored by FastAPI's binding, so the response was the
        whole unfiltered dataset with no indication the filter had not applied."""
        with pytest.raises(RequestError, match=r"calender_year"):
            partition_query(_items(calender_year="2050", **DATASET))

    def test_a_leftover_under_a_named_analysis_is_left_for_the_registry(self):
        """Not an error here: which names are legal varies per analysis, and
        `resolve_analysis_chain` is what rejects one against the stage that would have had
        to declare it."""
        partition = partition_query(_items(analysis="diff", nonsense="1"))
        assert partition.params == {"nonsense": "1"}

    def test_render_fields_are_a_declared_category(self):
        """`projects_model_group_id` is the reason this is declared rather than inferred:
        it is dimension-NAMED and is emphatically not a dimension filter, so an inferring
        partition would silently filter the data by the map's project overlay setting."""
        partition = partition_query(
            [("projects_model_group_id", "500"), ("height", "400")],
            render_fields={"projects_model_group_id", "height"},
        )
        assert partition.render == {"projects_model_group_id": "500", "height": "400"}
        assert partition.filters == {} and partition.params == {}

    def test_a_render_field_not_declared_by_this_route_is_a_filter(self):
        """With no `render_fields` (every route but the visualization one),
        `projects_model_group_id` is not a render option and `model_group_id` is an
        ordinary dimension -- the two names are distinct, so nothing is ambiguous."""
        partition = partition_query([("model_group_id", "500")])
        assert partition.filters == {"model_group_id": ["500"]}

    def test_compression_is_rejected_on_a_tile_route(self):
        with pytest.raises(RequestError, match="not accepted on a tile route"):
            partition_query(_items(compression_algorithm="zstd"), allow_compression=False)


class TestRetiredParams:
    def test_every_retired_name_is_reported(self):
        found = retired_params(["group_by", "unpivot", "variable"])
        assert found == ["group_by", "unpivot"]
        assert "'group_by' has been retired" in retired_param_message(found)

    def test_the_calculate_family_is_matched_by_prefix(self):
        """The retired spelling was a whole family of indexed keys, not one name, so a
        plain dict lookup would miss `calculate.1.expression` entirely."""
        assert retired_params(["calculate.1.expression"]) == ["calculate"]
        assert retired_params(["calculate"]) == ["calculate"]

    def test_a_dotted_step_named_calculate_is_untouched(self):
        """`step.1.op=calculate` is the REPLACEMENT the message points at -- matching it
        as retired would reject the only remaining spelling."""
        assert retired_params(["step.1.op", "step.1.expression"]) == []

    def test_the_partition_rejects_one(self):
        with pytest.raises(RequestError, match="'group_by' has been retired"):
            partition_query(_items(group_by="calendar_year", **DATASET))

    def test_every_entry_names_a_replacement(self):
        assert all(hint for hint in RETIRED_PARAMS.values())


class TestDownloadPipeline:
    def test_the_sentinel_builds_one_input_and_no_steps(self):
        pipeline = download_pipeline(DATASET, "csv")
        assert list(pipeline.inputs) == ["data"]
        assert pipeline.inputs["data"].variable == "ssf"
        assert pipeline.steps == []
        assert pipeline.output.format == "csv"

    def test_the_input_is_named_data(self):
        """The name saved links and `pipeline.json` already carry -- so the sugar and the
        dotted spelling of the same request encode to the same query string."""
        assert pipeline_to_query(download_pipeline(DATASET, "csv"))["input.data.variable"] == "ssf"

    @pytest.mark.parametrize("missing", ["variable", "time_unit", "grid"])
    def test_a_partial_triple_names_what_is_missing(self, missing):
        partial = {k: v for k, v in DATASET.items() if k != missing}
        with pytest.raises(RequestError, match=f"missing \\['{missing}'\\]"):
            download_pipeline(partial, "csv")


class TestBuildPipeline:
    def test_the_url_format_always_wins(self):
        base = download_pipeline(DATASET, "csv")
        assert build_pipeline(base, {"output.format": "tif"}, format="png").output.format == "png"

    def test_a_dotted_input_field_merges_into_the_base_input(self):
        """Field-wise, not whole-input replacement -- which is what re-encoding the base
        and overlaying the caller's keys buys for free."""
        base = download_pipeline(DATASET, "csv")
        merged = build_pipeline(base, {"input.data.grid": "hydro_compartment_v001"}, format="csv")
        assert merged.inputs["data"].grid == "hydro_compartment_v001"
        assert merged.inputs["data"].variable == "ssf"

    def test_a_new_input_name_is_added(self):
        base = download_pipeline(DATASET, "csv")
        merged = build_pipeline(
            base,
            {
                "input.other.variable": "sal",
                "input.other.time_unit": "annual",
                "input.other.grid": "ecoregion_v001",
                "step.1.op": "join",
                "step.1.type": "spatial",
                "step.1.inputs": "data,other",
                "step.1.as": "joined",
            },
            format="csv",
        )
        assert sorted(merged.inputs) == ["data", "other"]

    def test_caller_steps_are_renumbered_after_the_base_steps(self):
        """A caller numbers from 1 because they are writing about their own request and
        have no way to know how many steps the analysis they named contributes -- which
        varies with the analysis's own parameters."""
        base = pipeline_from_query(
            {
                "input.data.variable": "ssf",
                "input.data.time_unit": "annual",
                "input.data.grid": "ecoregion_v001",
                "step.1.op": "filter",
                "step.1.input": "data",
                "step.1.where": "calendar_year = 2030",
                "output.format": "csv",
            }
        )
        merged = build_pipeline(base, {"step.1.op": "filter", "step.1.where": "value > 3"}, format="csv")
        assert [step.where for step in merged.steps] == ["calendar_year = 2030", "value > 3"]

    def test_flat_output_options_are_validated_not_assigned(self):
        """Written in as `output.sort` and decoded, so `validate_pipeline`'s syntax check
        runs -- the route-level `setattr` this replaces skipped it entirely."""
        base = download_pipeline(DATASET, "csv")
        assert build_pipeline(base, {}, {"sort": "-value"}, format="csv").output.sort == "-value"
        with pytest.raises(RequestError, match="bare '-'"):
            build_pipeline(base, {}, {"sort": "value,-"}, format="csv")

    def test_an_analysis_default_survives_an_unsent_option(self):
        base = build_pipeline(download_pipeline(DATASET, "csv"), {}, {"select": "value"}, format="csv")
        assert build_pipeline(base, {}, {}, format="csv").output.select == "value"

    def test_no_base_means_the_caller_s_dotted_params_are_the_whole_request(self):
        pipeline = build_pipeline(
            None,
            {
                "input.data.variable": "ssf",
                "input.data.time_unit": "annual",
                "input.data.grid": "ecoregion_v001",
            },
            format="csv",
        )
        assert list(pipeline.inputs) == ["data"] and pipeline.steps == []

    def test_a_malformed_step_index_is_reported(self):
        with pytest.raises(RequestError, match="is not a step parameter"):
            build_pipeline(download_pipeline(DATASET, "csv"), {"step.op": "filter"}, format="csv")


class TestAppendSelection:
    def test_nothing_selected_returns_the_pipeline_unchanged(self):
        pipeline = download_pipeline(DATASET, "csv")
        assert append_selection(pipeline, None) is pipeline

    def test_a_stepless_pipeline_names_its_input(self):
        """`validate_pipeline`'s ordinary rule: a pipeline's first step must name an input
        because there is no previous step to chain from."""
        appended = append_selection(download_pipeline(DATASET, "csv"), "calendar_year = 2030")
        assert appended.steps[0].input == "data"
        assert appended.steps[0].where == "calendar_year = 2030"

    def test_a_pipeline_with_steps_chains_from_the_last_one(self):
        base = append_selection(download_pipeline(DATASET, "csv"), "calendar_year = 2030")
        appended = append_selection(base, "value > 3")
        assert appended.steps[1].input is None
        assert [step.where for step in appended.steps] == ["calendar_year = 2030", "value > 3"]

    def test_the_output_is_carried_through(self):
        """A selection changes which rows are written, never what is written or how."""
        base = build_pipeline(download_pipeline(DATASET, "csv"), {}, {"sort": "-value"}, format="csv")
        assert append_selection(base, "calendar_year = 2030").output.sort == "-value"
