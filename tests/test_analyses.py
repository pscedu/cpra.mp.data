"""Unit tests for `cpra.mp.data.analyses`: the registry (`ANALYSES`), `resolve_analysis`'s
exception contract (unknown name / wrong route kind / invalid parameters), and each of the
three v1 entries' own parameter validation, cardinality rules, and `pipeline()`
derivation.

Pure/in-memory throughout -- `resolve_analysis`/`Analysis.pipeline()` never touch disk (see
`analyses/base.py`'s `required_dataset_dimensions` docstring), so nothing here needs
`base_path`/`tmp_path`. `tests/api/test_analyses_registry.py` covers the disk-touching half
(the plan §9 registry-drift test, and an exit-criterion end-to-end compile+materialize
check).
"""

import json
from datetime import date

import pytest
from pydantic import ValidationError

from cpra.mp.data.analyses import (
    ANALYSES,
    AnalysisTypeMismatchError,
    DatasetRef,
    InvalidAnalysisParametersError,
    ResolvedAnalysis,
    UnknownAnalysisError,
    resolve_analysis,
)
from cpra.mp.data.analyses.base import (
    cast_dimension_value,
    default_collapsed_dimensions,
    format_dimension_literal,
    is_filterable_dimension,
    normalized_filter_values,
)
from cpra.mp.data.analyses.concatenate import Concatenate
from cpra.mp.data.analyses.difference import Difference
from cpra.mp.data.analyses.exit_criterion import ExitCriterion
from cpra.mp.data.analysis.pipeline import (
    CalculateStep,
    FilterStep,
    GroupStep,
    ProjectStep,
    pipeline_to_query,
)

SSF_DATASET = DatasetRef(variable="ssf", time_unit="annual", grid="ecoregion_v001")
SAL_DATASET = DatasetRef(variable="sal", time_unit="daily", grid="hydro_compartment_v001")


# ------------------------------------------------------------------------------------------
# The registry itself
# ------------------------------------------------------------------------------------------


class TestRegistry:
    def test_contains_exactly_the_three_v1_entries(self):
        assert set(ANALYSES) == {"exit-criterion", "difference", "concatenate"}

    def test_keyed_by_meta_name(self):
        for name, cls in ANALYSES.items():
            assert cls.Meta.name == name

    def test_meta_shape(self):
        assert ANALYSES["difference"].Meta.type == "dataset"
        assert ANALYSES["difference"].Meta.variables is None
        assert ANALYSES["difference"].Meta.legend is None
        assert ANALYSES["difference"].Meta.result_kind == "difference"

        assert ANALYSES["concatenate"].Meta.type == "dataset"
        assert ANALYSES["concatenate"].Meta.variables == ["lnd_type"]
        assert ANALYSES["concatenate"].Meta.legend == "lnd_chng"
        assert ANALYSES["concatenate"].Meta.result_kind == "category"

        assert ANALYSES["exit-criterion"].Meta.type == "custom"
        assert ANALYSES["exit-criterion"].Meta.result_kind == "value"


# ------------------------------------------------------------------------------------------
# `resolve_analysis`'s own exception contract
# ------------------------------------------------------------------------------------------


class TestResolveAnalysisContract:
    def test_unknown_name_raises_unknown_analysis_error(self):
        with pytest.raises(UnknownAnalysisError):
            resolve_analysis("not-a-real-analysis", dataset=None, filters={}, params={})

    def test_dataset_analysis_without_dataset_raises_type_mismatch(self):
        with pytest.raises(AnalysisTypeMismatchError):
            resolve_analysis(
                "difference", dataset=None, filters={}, params={"dimension": "calendar_year", "order": "asc"}
            )

    def test_custom_analysis_with_dataset_raises_type_mismatch(self):
        with pytest.raises(AnalysisTypeMismatchError):
            resolve_analysis(
                "exit-criterion",
                dataset=SSF_DATASET,
                filters={},
                params={"model_group_id": 500, "scenario_id": 7},
            )

    def test_missing_required_parameter_raises_invalid_parameters(self):
        with pytest.raises(InvalidAnalysisParametersError):
            resolve_analysis(
                "difference",
                dataset=SSF_DATASET,
                filters={"calendar_year": [2030, 2040]},
                params={"dimension": "calendar_year"},  # missing `order`
            )

    def test_unknown_parameter_raises_invalid_parameters(self):
        with pytest.raises(InvalidAnalysisParametersError):
            resolve_analysis(
                "difference",
                dataset=SSF_DATASET,
                filters={"calendar_year": [2030, 2040]},
                params={"dimension": "calendar_year", "order": "asc", "bogus": 1},
            )

    def test_invalid_literal_raises_invalid_parameters(self):
        with pytest.raises(InvalidAnalysisParametersError):
            resolve_analysis(
                "difference",
                dataset=SSF_DATASET,
                filters={"calendar_year": [2030, 2040]},
                params={"dimension": "calendar_year", "order": "sideways"},
            )

    def test_successful_resolve_returns_resolved_analysis(self):
        result = resolve_analysis(
            "difference",
            dataset=SSF_DATASET,
            filters={"calendar_year": [2030, 2040], "model_group_id": 500, "scenario_id": 7, "percentile": 50},
            params={"dimension": "calendar_year", "order": "asc"},
        )
        assert isinstance(result, ResolvedAnalysis)
        assert result.meta is Difference.Meta
        assert result.legend is None
        assert result.result_kind == "difference"
        assert result.collapsed_dimensions == frozenset({"calendar_year"})
        assert result.title_suffix == " Difference"


class TestResolveOverADateDimension:
    """A `date`-valued dimension filter must survive all the way into the built pipeline.

    Regression: `calendar_day` is declared `Date` in `structure/dimensions.csv`, so
    `types.dimension_type` makes `FilterDimensions.calendar_day` a `list[date]` and FastAPI
    binds `?calendar_day=2019-01-01` to a real `datetime.date`. `InputSelection`'s scalar
    whitelist listed only `(str, int, float, bool)`, so every named analysis over a date
    dimension -- a `difference` on any daily dataset, which is exactly what the portal's
    difference map requests -- failed `Pipeline` validation outright with "must be a scalar
    or a list of scalars, got date in it".

    `format_dimension_literal` was already unit-tested against a `date`
    (`test_format_dimension_literal_calendar_day_gets_date_cast`), so the registry's own
    date handling was covered; what had no test was the SEAM, where that helper's output
    meets `InputSelection`'s validation. Hence this asserts on the resolved pipeline rather
    than on a helper in isolation.
    """

    FILTERS = {
        "calendar_day": [date(2019, 1, 1), date(2070, 12, 31)],
        "model_group_id": 500,
        "scenario_id": 7,
        "aggregate_type": "mean",
    }

    def test_difference_over_calendar_day_resolves(self):
        result = resolve_analysis(
            "difference",
            dataset=SAL_DATASET,
            filters=self.FILTERS,
            params={"dimension": "calendar_day", "order": "asc"},
        )
        assert result.result_kind == "difference"
        assert result.collapsed_dimensions == frozenset({"calendar_day"})

    def test_the_dates_survive_into_the_input_selection(self):
        result = resolve_analysis(
            "difference",
            dataset=SAL_DATASET,
            filters=self.FILTERS,
            params={"dimension": "calendar_day", "order": "asc"},
        )
        selection = result.pipeline.inputs["x"]
        assert selection.dimension_filters["calendar_day"] == [date(2019, 1, 1), date(2070, 12, 31)]

    def test_the_negated_side_is_quoted_and_cast_not_left_as_bare_arithmetic(self):
        """An unquoted date in a SQL predicate is arithmetic -- `2019-01-01` evaluates to
        2017 -- so the emitted literal must carry its quotes and its `::date` cast.
        """
        result = resolve_analysis(
            "difference",
            dataset=SAL_DATASET,
            filters=self.FILTERS,
            params={"dimension": "calendar_day", "order": "asc"},
        )
        calculate = next(step for step in result.pipeline.steps if isinstance(step, CalculateStep))
        assert calculate.where == "calendar_day = '2070-12-31'::date"

    def test_both_encodings_round_trip_the_dates(self):
        """The pipeline has to survive being written back out both ways: `pipeline_to_query`
        builds the dotted tile-URL form, and `pipeline.json` serializes via
        `model_dump(mode="json")` -- which must stay JSON-encodable.
        """
        result = resolve_analysis(
            "difference",
            dataset=SAL_DATASET,
            filters=self.FILTERS,
            params={"dimension": "calendar_day", "order": "asc"},
        )
        assert pipeline_to_query(result.pipeline)["input.x.calendar_day"] == "2019-01-01,2070-12-31"

        dumped = result.pipeline.model_dump(mode="json")
        json.dumps(dumped)  # must not raise -- a bare `date` is not JSON-encodable
        assert dumped["inputs"]["x"]["calendar_day"] == ["2019-01-01", "2070-12-31"]


# ------------------------------------------------------------------------------------------
# Shared helpers (base.py)
# ------------------------------------------------------------------------------------------


class TestSharedHelpers:
    def test_is_filterable_dimension(self):
        assert is_filterable_dimension("calendar_year") is True
        assert is_filterable_dimension("model_group_id") is True
        assert is_filterable_dimension("variable") is False  # dataset-typed
        assert is_filterable_dimension("grid") is False  # dataset-typed
        assert is_filterable_dimension("not_a_real_dimension") is False

    def test_cast_dimension_value_numeric(self):
        assert cast_dimension_value("calendar_year", "2030") == 2030
        assert isinstance(cast_dimension_value("calendar_year", "2030"), int)

    def test_normalized_filter_values_dedupes_and_casts(self):
        assert normalized_filter_values("calendar_year", ["2030", "2030", "2040"]) == [2030, 2040]
        assert normalized_filter_values("calendar_year", 2030) == [2030]

    def test_normalized_filter_values_sorts_numerically_not_lexically(self):
        # A lexical sort would put "10" before "9"; the dimension's own numeric dtype
        # must be applied before any ordering decision is made.
        values = normalized_filter_values("calendar_year", ["10", "9"])
        assert sorted(values) == [9, 10]

    def test_format_dimension_literal_numeric_is_unquoted(self):
        assert format_dimension_literal("calendar_year", "2030") == "2030"
        assert format_dimension_literal("model_group_id", 500) == "500"

    def test_format_dimension_literal_string_dimension_is_quoted(self):
        assert format_dimension_literal("aggregate_type", "2_wk_max") == "'2_wk_max'"

    def test_format_dimension_literal_calendar_day_gets_date_cast(self):
        import datetime

        literal = format_dimension_literal("calendar_day", datetime.date(2030, 1, 1))
        assert literal == "'2030-01-01'::date"

    def test_format_dimension_literal_escapes_embedded_quotes(self):
        literal = format_dimension_literal("aggregate_type", "o'brien")
        assert literal == "'o''brien'"

    def test_default_collapsed_dimensions_empty_without_trailing_group(self):
        pipeline = Difference(
            dataset=SSF_DATASET,
            filters={"calendar_year": [2030, 2040], "model_group_id": 500, "scenario_id": 7, "percentile": 50},
            parameters=Difference.Parameters(dimension="calendar_year", order="asc"),
        ).pipeline()
        # Sanity check on the fixture itself: difference's own pipeline DOES end with a
        # group, so collapsed_dimensions is non-empty -- see the dedicated Difference
        # tests below for the exact value. This test instead confirms the *general* rule
        # directly against a hand-built pipeline with no trailing group.
        pipeline.steps = pipeline.steps[:-1]  # drop the trailing group step
        assert default_collapsed_dimensions(pipeline) == frozenset()


# ------------------------------------------------------------------------------------------
# `difference`
# ------------------------------------------------------------------------------------------


class TestDifferenceParameters:
    def test_requires_order(self):
        with pytest.raises(ValidationError):
            Difference.Parameters(dimension="calendar_year")

    def test_rejects_invalid_order(self):
        with pytest.raises(ValidationError):
            Difference.Parameters(dimension="calendar_year", order="upward")

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            Difference.Parameters(dimension="calendar_year", order="asc", extra="nope")

    def test_rejects_non_filterable_dimension(self):
        with pytest.raises(ValidationError):
            Difference.Parameters(dimension="variable", order="asc")


class TestDifferenceCardinality:
    params = Difference.Parameters(dimension="calendar_year", order="asc")

    def test_rejects_missing_filter(self):
        with pytest.raises(InvalidAnalysisParametersError):
            Difference.validate_selection(self.params, {})

    def test_rejects_one_value(self):
        with pytest.raises(InvalidAnalysisParametersError):
            Difference.validate_selection(self.params, {"calendar_year": [2030]})

    def test_rejects_three_values(self):
        with pytest.raises(InvalidAnalysisParametersError):
            Difference.validate_selection(self.params, {"calendar_year": [2030, 2040, 2050]})

    def test_rejects_two_values_that_are_actually_the_same(self):
        with pytest.raises(InvalidAnalysisParametersError):
            Difference.validate_selection(self.params, {"calendar_year": [2030, "2030"]})

    def test_accepts_exactly_two_distinct_values(self):
        Difference.validate_selection(self.params, {"calendar_year": [2030, 2040]})  # no raise


class TestDifferencePipeline:
    def _build(self, order, filters=None):
        filters = filters or {"calendar_year": [2030, 2040], "model_group_id": 500, "scenario_id": 7, "percentile": 50}
        analysis = Difference(
            dataset=SSF_DATASET,
            filters=filters,
            parameters=Difference.Parameters(dimension="calendar_year", order=order),
        )
        return analysis, analysis.pipeline()

    def test_negates_the_later_value_for_order_asc(self):
        _, pipeline = self._build("asc")
        calc = pipeline.steps[1]
        assert isinstance(calc, CalculateStep)
        assert calc.expression == "value * -1"
        assert calc.where == "calendar_year = 2040"

    def test_negates_the_earlier_value_for_order_desc(self):
        _, pipeline = self._build("desc")
        calc = pipeline.steps[1]
        assert calc.where == "calendar_year = 2030"

    def test_step_shape(self):
        _, pipeline = self._build("asc")
        assert list(pipeline.inputs) == ["x"]
        assert isinstance(pipeline.steps[0], ProjectStep)
        assert pipeline.steps[0].to == "ecoregion_v001"
        assert pipeline.steps[0].input == "x"
        assert isinstance(pipeline.steps[1], CalculateStep)
        group = pipeline.steps[2]
        assert isinstance(group, GroupStep)
        assert sorted(group.by) == sorted(["model_group_id", "scenario_id", "percentile"])
        assert group.aggregate == "sum"

    def test_collapsed_dimensions_and_title_suffix(self):
        analysis, _ = self._build("asc")
        assert analysis.collapsed_dimensions == frozenset({"calendar_year"})
        assert analysis.title_suffix == " Difference"
        assert analysis.legend is None
        assert analysis.result_kind == "difference"

    def test_forwards_dimension_filters_and_grid_cell_onto_the_input(self):
        _, pipeline = self._build(
            "asc",
            filters={
                "calendar_year": [2030, 2040],
                "model_group_id": 500,
                "scenario_id": 7,
                "percentile": 50,
                "grid_cell": "1,2,3",
            },
        )
        selection = pipeline.inputs["x"]
        assert selection.grid_cell == "1,2,3"
        assert selection.dimension_filters["model_group_id"] == [500]


# ------------------------------------------------------------------------------------------
# `concatenate`
# ------------------------------------------------------------------------------------------


class TestConcatenateParameters:
    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            Concatenate.Parameters(dimension="calendar_year", extra="nope")

    def test_rejects_non_filterable_dimension(self):
        with pytest.raises(ValidationError):
            Concatenate.Parameters(dimension="grid")


class TestConcatenateCardinality:
    params = Concatenate.Parameters(dimension="calendar_year")

    def test_rejects_missing_dimension_filter(self):
        with pytest.raises(InvalidAnalysisParametersError):
            Concatenate.validate_selection(self.params, {})

    def test_rejects_one_dimension_value(self):
        with pytest.raises(InvalidAnalysisParametersError):
            Concatenate.validate_selection(self.params, {"calendar_year": [2030]})

    def test_rejects_three_dimension_values(self):
        with pytest.raises(InvalidAnalysisParametersError):
            Concatenate.validate_selection(self.params, {"calendar_year": [2030, 2040, 2050]})

    def test_accepts_two_dimension_values_with_no_model_group_filter(self):
        Concatenate.validate_selection(self.params, {"calendar_year": [2030, 2040]})  # no raise

    def test_accepts_two_dimension_values_with_one_model_group(self):
        Concatenate.validate_selection(
            self.params, {"calendar_year": [2030, 2040], "model_group_id": 500}
        )  # no raise

    def test_accepts_two_dimension_values_with_two_model_groups(self):
        Concatenate.validate_selection(
            self.params, {"calendar_year": [2030, 2040], "model_group_id": [500, 520]}
        )  # no raise

    def test_rejects_three_model_groups(self):
        with pytest.raises(InvalidAnalysisParametersError):
            Concatenate.validate_selection(
                self.params, {"calendar_year": [2030, 2040], "model_group_id": [500, 510, 520]}
            )


class TestConcatenatePipeline:
    def _build(self, filters):
        analysis = Concatenate(
            dataset=SSF_DATASET,
            filters=filters,
            parameters=Concatenate.Parameters(dimension="calendar_year"),
        )
        return analysis, analysis.pipeline()

    def test_two_image_mode(self):
        analysis, pipeline = self._build(
            {"calendar_year": [2030, 2040], "model_group_id": 500, "scenario_id": 7, "percentile": 50}
        )
        assert len(pipeline.steps) == 3
        assert isinstance(pipeline.steps[0], ProjectStep)
        calc = pipeline.steps[1]
        assert isinstance(calc, CalculateStep)
        assert calc.expression == "value * 10"
        assert calc.where == "calendar_year = 2030"
        group = pipeline.steps[2]
        assert isinstance(group, GroupStep)
        assert sorted(group.by) == sorted(["model_group_id", "scenario_id", "percentile"])
        assert analysis.collapsed_dimensions == frozenset({"calendar_year"})

    def test_two_image_mode_with_exactly_one_model_group_selected(self):
        # model_group_id present but pinned to a single value: still two-image.
        analysis, pipeline = self._build(
            {"calendar_year": [2030, 2040], "model_group_id": [500], "scenario_id": 7, "percentile": 50}
        )
        assert len(pipeline.steps) == 3
        assert pipeline.steps[1].expression == "value * 10"

    def test_three_image_mode(self):
        analysis, pipeline = self._build(
            {"calendar_year": [2030, 2040], "model_group_id": [520, 500], "scenario_id": 7, "percentile": 50}
        )
        assert len(pipeline.steps) == 5
        assert isinstance(pipeline.steps[0], ProjectStep)

        filter_step = pipeline.steps[1]
        assert isinstance(filter_step, FilterStep)
        assert filter_step.where == "model_group_id != 520 OR calendar_year != 2030"

        calc1 = pipeline.steps[2]
        assert isinstance(calc1, CalculateStep)
        assert calc1.expression == "value * 100"
        assert calc1.where == "calendar_year = 2030"

        calc2 = pipeline.steps[3]
        assert isinstance(calc2, CalculateStep)
        assert calc2.expression == "value * 10"
        assert calc2.where == "model_group_id = 500 AND calendar_year = 2040"

        group = pipeline.steps[4]
        assert isinstance(group, GroupStep)
        assert sorted(group.by) == sorted(["scenario_id", "percentile"])
        assert "model_group_id" not in group.by

        assert analysis.collapsed_dimensions == frozenset({"calendar_year", "model_group_id"})

    def test_meta_and_legend(self):
        analysis, _ = self._build({"calendar_year": [2030, 2040], "model_group_id": 500})
        assert analysis.legend == "lnd_chng"
        assert analysis.result_kind == "category"


# ------------------------------------------------------------------------------------------
# `exit-criterion`
# ------------------------------------------------------------------------------------------


class TestExitCriterionParameters:
    def test_requires_model_group_id_and_scenario_id(self):
        with pytest.raises(ValidationError):
            ExitCriterion.Parameters(scenario_id=7)
        with pytest.raises(ValidationError):
            ExitCriterion.Parameters(model_group_id=500)

    def test_defaults(self):
        params = ExitCriterion.Parameters(model_group_id=500, scenario_id=7)
        assert params.marsh_threshold == 0.5
        assert params.salinity_threshold == 5.5

    def test_marsh_threshold_bounds(self):
        with pytest.raises(ValidationError):
            ExitCriterion.Parameters(model_group_id=500, scenario_id=7, marsh_threshold=1.5)
        with pytest.raises(ValidationError):
            ExitCriterion.Parameters(model_group_id=500, scenario_id=7, marsh_threshold=-0.1)

    def test_salinity_threshold_lower_bound(self):
        with pytest.raises(ValidationError):
            ExitCriterion.Parameters(model_group_id=500, scenario_id=7, salinity_threshold=-1)

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            ExitCriterion.Parameters(model_group_id=500, scenario_id=7, bogus=1)


class TestExitCriterionPipeline:
    def _build(self, **overrides):
        params = {"model_group_id": 500, "scenario_id": 7, **overrides}
        analysis = ExitCriterion(dataset=None, filters={}, parameters=ExitCriterion.Parameters(**params))
        return analysis, analysis.pipeline()

    def test_matches_the_documented_exit_criterion_shape(self):
        """`docs/data-analysis-api.md`'s "The Exit-Criterion Query" example, reproduced
        via the named analysis instead of hand-authored JSON.
        """
        _, pipeline = self._build()

        assert set(pipeline.inputs) == {"marsh", "salinity"}
        marsh = pipeline.inputs["marsh"]
        assert marsh.variable == "ffibs_coverage_ratio"
        assert marsh.time_unit == "annual"
        assert marsh.grid == "veg_grid_cell_v001"
        assert marsh.dimension_filters["model_group_id"] == [500]
        assert marsh.dimension_filters["scenario_id"] == [7]
        assert marsh.dimension_filters["ffibs_type"] == ["pl_fm"]

        salinity = pipeline.inputs["salinity"]
        assert salinity.variable == "sal"
        assert salinity.time_unit == "annual"
        assert salinity.grid == "hydro_compartment_v001"
        assert salinity.dimension_filters["aggregate_type"] == ["2_wk_max"]

        project_step, join_step, filter_step, group_step = pipeline.steps
        assert isinstance(project_step, ProjectStep)
        assert project_step.input == "salinity"
        assert project_step.to == "veg_grid_cell_v001"
        assert project_step.as_ == "sal"

        assert join_step.op == "join"
        assert join_step.type == "spatial"
        assert set(join_step.inputs) == {"marsh", "sal"}

        assert isinstance(filter_step, FilterStep)
        assert filter_step.where == "marsh > 0.5 AND sal > 5.5"

        assert isinstance(group_step, GroupStep)
        assert group_step.by == []
        assert group_step.aggregate == "min"
        assert group_step.of == "calendar_year"

        assert pipeline.output.format == "tif"

    def test_custom_thresholds_are_interpolated(self):
        _, pipeline = self._build(marsh_threshold=0.75, salinity_threshold=3.0)
        filter_step = pipeline.steps[2]
        assert filter_step.where == "marsh > 0.75 AND sal > 3.0"

    def test_collapsed_dimensions_union_both_inputs(self):
        analysis, _ = self._build()
        assert analysis.collapsed_dimensions == frozenset(
            {"model_group_id", "scenario_id", "ffibs_type", "aggregate_type", "calendar_year"}
        )

    def test_meta_and_presentation(self):
        analysis, _ = self._build()
        assert analysis.result_kind == "value"
        assert analysis.legend is None
        assert analysis.title_suffix == ""
