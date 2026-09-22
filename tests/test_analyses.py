"""Unit tests for `cpra.mp.data.analyses`: the registry (`ANALYSES`), `resolve_analysis`'s
exception contract (unknown name / wrong route kind / invalid parameters), and each
registered entry's own parameter validation, cardinality rules, and `pipeline()`
derivation.

Pure/in-memory throughout -- `resolve_analysis`/`Analysis.pipeline()` never touch disk (see
`analyses/base.py`'s `required_dataset_dimensions` docstring), so nothing here needs
`base_path`/`tmp_path`. `tests/api/test_analyses_registry.py` covers the disk-touching
half (the registry-drift test, and one execution check per entry).

**Where a caller's filters end up.** An `InputSelection` names a dataset
(`variable`/`time_unit`/`grid`) and carries no filtering at all, so every dimension
filter, `grid_cell`, and caller-written `where` is composed into the pipeline's LEADING
`filter` STEP instead (`base.selection_filter_where`). Tests here therefore assert on
`pipeline.steps[0].where`, not on an input's fields.
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
    selection_filter_where,
)
from cpra.mp.data.analyses.concatenate import Concatenate
from cpra.mp.data.analyses.difference import Difference
from cpra.mp.data.analyses.land_type_confidence import (
    LAND_VALUES,
    SCENARIO_WEIGHTS,
    WATER_VALUES,
    LandTypeConfidence,
)
from cpra.mp.data.analyses.salinity_veg_investigation import SalinityVegInvestigation
from cpra.mp.data.analysis.request.pipeline import (
    CalculateStep,
    FilterStep,
    GroupStep,
    ProjectStep,
    pipeline_to_query,
)

SSF_DATASET = DatasetRef(variable="ssf", time_unit="annual", grid="ecoregion_v001")
SAL_DATASET = DatasetRef(variable="sal", time_unit="daily", grid="hydro_compartment_v001")
# `land-type-confidence`'s real dataset. Nothing here reads from disk -- these tests build
# pipelines and inspect them -- so the real variable/grid can be named directly rather than
# through the small stand-ins the execution tests need.
LND_TYPE_DATASET = DatasetRef(variable="lnd_type", time_unit="annual", grid="morph_pixel_v002")


# ------------------------------------------------------------------------------------------
# The registry itself
# ------------------------------------------------------------------------------------------


class TestRegistry:
    def test_contains_exactly_the_registered_entries(self):
        assert set(ANALYSES) == {
            "salinity-veg-investigation",
            "difference",
            "concatenate",
            "land-type-confidence",
        }

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

        assert ANALYSES["salinity-veg-investigation"].Meta.type == "custom"
        assert ANALYSES["salinity-veg-investigation"].Meta.result_kind == "value"

        assert ANALYSES["land-type-confidence"].Meta.type == "dataset"
        assert ANALYSES["land-type-confidence"].Meta.variables == ["lnd_type"]
        assert ANALYSES["land-type-confidence"].Meta.legend is None
        assert ANALYSES["land-type-confidence"].Meta.result_kind == "value"


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
                "salinity-veg-investigation",
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

    `calendar_day` is declared `Date` in `structure/dimensions.csv`, so
    `request.types.dimension_type` makes `FilterDimensions.calendar_day` a `list[date]`
    and FastAPI binds `?calendar_day=2019-01-01` to a real `datetime.date` -- a value that
    reaches `selection_filter_where` as a `date` object, not a string, and has to come out
    the other side as a quoted, `::date`-cast SQL literal.

    `format_dimension_literal` is already unit-tested against a `date`
    (`test_format_dimension_literal_calendar_day_gets_date_cast`); what this class covers
    is the SEAM, where that helper's output meets the composed `where` string and the
    pipeline's own serialization. Hence it asserts on the resolved pipeline rather than on
    a helper in isolation.
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

    def test_the_dates_survive_into_the_leading_filter_step(self):
        result = resolve_analysis(
            "difference",
            dataset=SAL_DATASET,
            filters=self.FILTERS,
            params={"dimension": "calendar_day", "order": "asc"},
        )
        step = result.pipeline.steps[0]
        assert isinstance(step, FilterStep)
        assert step.input == "x"
        assert "calendar_day IN ('2019-01-01'::date, '2070-12-31'::date)" in step.where

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
        `model_dump(mode="json")` -- which must stay JSON-encodable. Both now carry the
        dates inside the leading `filter` step's `where` string rather than as typed
        values on the input, so neither can hit a `date`-is-not-JSON-encodable failure;
        this pins that the composed literal itself round-trips intact.
        """
        result = resolve_analysis(
            "difference",
            dataset=SAL_DATASET,
            filters=self.FILTERS,
            params={"dimension": "calendar_day", "order": "asc"},
        )
        query = pipeline_to_query(result.pipeline)
        assert "calendar_day IN ('2019-01-01'::date, '2070-12-31'::date)" in query["step.1.where"]

        dumped = result.pipeline.model_dump(mode="json")
        json.dumps(dumped)  # must not raise -- a bare `date` is not JSON-encodable
        assert (
            "calendar_day IN ('2019-01-01'::date, '2070-12-31'::date)"
            in dumped["steps"][0]["where"]
        )


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

    def test_selection_filter_where_is_none_when_there_is_nothing_to_filter_on(self):
        assert selection_filter_where({}) is None
        assert selection_filter_where({"grid_cell": None, "where": None}) is None

    def test_selection_filter_where_orders_dimensions_by_name_not_by_dict_order(self):
        """The composed string must not depend on the order keys happened to arrive in --
        two callers passing the same filters get the same `where` either way.
        """
        one = selection_filter_where({"scenario_id": 7, "calendar_year": 2030})
        other = selection_filter_where({"calendar_year": 2030, "scenario_id": 7})
        assert one == other == "calendar_year = 2030 AND scenario_id = 7"

    def test_selection_filter_where_puts_the_spatial_conjunct_after_the_dimensions(self):
        assert selection_filter_where({"scenario_id": 7, "grid_cell": "4"}) == (
            "scenario_id = 7 AND grid_cell = 4"
        )

    def test_selection_filter_where_rejects_a_coordinate_grid_cell(self):
        """Resolving coordinates to ids needs the catalog, which this package cannot
        reach -- the caller resolves them and passes `grid_cell_where` instead.
        """
        from cpra.mp.data.analysis.request.errors import RequestError

        with pytest.raises(RequestError, match="digit IDs"):
            selection_filter_where({"grid_cell": "-90.1,29.5"})

    def test_selection_filter_where_accepts_the_reserved_keys(self):
        assert selection_filter_where({"grid_cell": "1", "where": "value > 3"}) == (
            "grid_cell = 1 AND (value > 3)"
        )

    def test_selection_filter_where_rejects_a_dataset_typed_dimension(self):
        # `variable`/`time_unit`/`grid` select the dataset; they are not filters.
        with pytest.raises(InvalidAnalysisParametersError, match="variable"):
            selection_filter_where({"variable": "ssf"})

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
        # The input names the dataset and nothing else -- every filter is a step.
        selection = pipeline.inputs["x"]
        assert (selection.variable, selection.time_unit, selection.grid) == (
            "ssf",
            "annual",
            "ecoregion_v001",
        )
        assert set(selection.model_dump(exclude_none=True)) == {"variable", "time_unit", "grid"}

        assert isinstance(pipeline.steps[0], FilterStep)
        assert pipeline.steps[0].input == "x"
        assert isinstance(pipeline.steps[1], CalculateStep)
        group = pipeline.steps[2]
        assert isinstance(group, GroupStep)
        assert sorted(group.by) == sorted(["model_group_id", "scenario_id", "percentile"])
        assert group.aggregate == "sum"

    def test_emits_no_identity_project_step(self):
        """A `project` to the dataset's own grid used to lead the chain purely to satisfy
        the "first step must name its input" rule. The leading `filter` carries that name
        now, and the identity project would be rejected outright on a raster dataset --
        `PROJECT_LOWERING_TABLE` has no raster-to-raster entry.
        """
        _, pipeline = self._build("asc")
        assert not any(isinstance(step, ProjectStep) for step in pipeline.steps)

    def test_collapsed_dimensions_and_title_suffix(self):
        analysis, _ = self._build("asc")
        assert analysis.collapsed_dimensions == frozenset({"calendar_year"})
        assert analysis.title_suffix == " Difference"
        assert analysis.legend is None
        assert analysis.result_kind == "difference"

    def test_composes_dimension_filters_and_grid_cell_into_the_leading_filter_step(self):
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
        step = pipeline.steps[0]
        assert isinstance(step, FilterStep)
        assert step.input == "x"
        # Dimensions in sorted name order, then the spatial conjunct last.
        assert step.where == (
            "calendar_year IN (2030, 2040) AND model_group_id = 500 AND percentile = 50 "
            "AND scenario_id = 7 AND grid_cell IN (1, 2, 3)"
        )

    def test_a_single_grid_cell_uses_equality_not_a_one_element_in_list(self):
        _, pipeline = self._build("asc", filters={"calendar_year": [2030, 2040], "grid_cell": "7"})
        assert pipeline.steps[0].where == "calendar_year IN (2030, 2040) AND grid_cell = 7"

    def test_a_resolved_grid_cell_where_replaces_the_grid_cell_key(self):
        """A coordinate-valued `grid_cell` needs the catalog to resolve, so the caller
        does it and passes the finished conjunct -- which wins over `filters["grid_cell"]`
        outright rather than being AND-ed alongside it.
        """
        analysis = Difference(
            dataset=SSF_DATASET,
            filters={"calendar_year": [2030, 2040], "grid_cell": "-90.1,29.5"},
            parameters=Difference.Parameters(dimension="calendar_year", order="asc"),
            grid_cell_where="grid_cell IN (11, 12)",
        )
        assert analysis.pipeline().steps[0].where == (
            "calendar_year IN (2030, 2040) AND grid_cell IN (11, 12)"
        )

    def test_a_callers_own_where_is_split_and_parenthesized_conjunct_by_conjunct(self):
        _, pipeline = self._build(
            "asc", filters={"calendar_year": [2030, 2040], "where": "value > 1 AND value < 9"}
        )
        # Split FIRST, then parenthesized: one paren pair per top-level conjunct, so the
        # AND seam the engine classifies each predicate's kind against survives intact.
        assert pipeline.steps[0].where == (
            "calendar_year IN (2030, 2040) AND (value > 1) AND (value < 9)"
        )

    def test_an_unrecognized_filter_name_is_rejected(self):
        analysis = Difference(
            dataset=SSF_DATASET,
            filters={"calendar_year": [2030, 2040], "nonsense": 1},
            parameters=Difference.Parameters(dimension="calendar_year", order="asc"),
        )
        with pytest.raises(InvalidAnalysisParametersError, match="nonsense"):
            analysis.pipeline()


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
        leading = pipeline.steps[0]
        assert isinstance(leading, FilterStep)
        assert leading.input == "x"
        assert leading.where == (
            "calendar_year IN (2030, 2040) AND model_group_id = 500 AND percentile = 50 "
            "AND scenario_id = 7"
        )
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
        assert len(pipeline.steps) == 4

        # The redundant corner is AND-ed onto the same leading `filter` the caller's own
        # dimension filters compose into -- both are dimension conditions at the same
        # point in the chain, so there is no reason for two consecutive filter steps.
        filter_step = pipeline.steps[0]
        assert isinstance(filter_step, FilterStep)
        assert filter_step.input == "x"
        # Dimension NAMES are sorted (so the string doesn't depend on dict order), but
        # each dimension's own VALUES keep the caller's order -- `[520, 500]` in, `(520,
        # 500)` out, matching `request.data_query`'s own conjunct assembly.
        assert filter_step.where == (
            "calendar_year IN (2030, 2040) AND model_group_id IN (520, 500) "
            "AND percentile = 50 AND scenario_id = 7 "
            "AND NOT (model_group_id = 520 AND calendar_year = 2030)"
        )

        calc1 = pipeline.steps[1]
        assert isinstance(calc1, CalculateStep)
        assert calc1.expression == "value * 100"
        assert calc1.where == "calendar_year = 2030"

        calc2 = pipeline.steps[2]
        assert isinstance(calc2, CalculateStep)
        assert calc2.expression == "value * 10"
        assert calc2.where == "model_group_id = 500 AND calendar_year = 2040"

        group = pipeline.steps[3]
        assert isinstance(group, GroupStep)
        assert sorted(group.by) == sorted(["scenario_id", "percentile"])
        assert "model_group_id" not in group.by

        assert analysis.collapsed_dimensions == frozenset({"calendar_year", "model_group_id"})

    def test_meta_and_legend(self):
        analysis, _ = self._build({"calendar_year": [2030, 2040], "model_group_id": 500})
        assert analysis.legend == "lnd_chng"
        assert analysis.result_kind == "category"


# ------------------------------------------------------------------------------------------
# `salinity-veg-investigation`
# ------------------------------------------------------------------------------------------


class TestSalinityVegInvestigationParameters:
    def test_requires_model_group_id_and_scenario_id(self):
        with pytest.raises(ValidationError):
            SalinityVegInvestigation.Parameters(scenario_id=7)
        with pytest.raises(ValidationError):
            SalinityVegInvestigation.Parameters(model_group_id=500)

    def test_defaults(self):
        params = SalinityVegInvestigation.Parameters(model_group_id=500, scenario_id=7)
        assert params.marsh_threshold == 0.5
        assert params.salinity_threshold == 5.5

    def test_marsh_threshold_bounds(self):
        with pytest.raises(ValidationError):
            SalinityVegInvestigation.Parameters(model_group_id=500, scenario_id=7, marsh_threshold=1.5)
        with pytest.raises(ValidationError):
            SalinityVegInvestigation.Parameters(model_group_id=500, scenario_id=7, marsh_threshold=-0.1)

    def test_salinity_threshold_lower_bound(self):
        with pytest.raises(ValidationError):
            SalinityVegInvestigation.Parameters(model_group_id=500, scenario_id=7, salinity_threshold=-1)

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            SalinityVegInvestigation.Parameters(model_group_id=500, scenario_id=7, bogus=1)


class TestSalinityVegInvestigationPipeline:
    def _build(self, **overrides):
        params = {"model_group_id": 500, "scenario_id": 7, **overrides}
        analysis = SalinityVegInvestigation(
            dataset=None, filters={}, parameters=SalinityVegInvestigation.Parameters(**params)
        )
        return analysis, analysis.pipeline()

    def test_matches_the_documented_salinity_veg_shape(self):
        """`docs/analysis-api.md`'s "The Salinity-Veg Investigation" example, reproduced via
        the named analysis instead of hand-authored JSON.
        """
        _, pipeline = self._build()

        assert set(pipeline.inputs) == {"coverage", "salinity"}
        coverage = pipeline.inputs["coverage"]
        assert (coverage.variable, coverage.time_unit, coverage.grid) == (
            "ffibs_coverage_ratio",
            "annual",
            "veg_grid_cell_v001",
        )
        salinity = pipeline.inputs["salinity"]
        assert (salinity.variable, salinity.time_unit, salinity.grid) == (
            "sal",
            "annual",
            "hydro_compartment_v001",
        )

        coverage_filter, salinity_filter, project_step, join_step, filter_step, group_step = (
            pipeline.steps
        )

        assert isinstance(coverage_filter, FilterStep)
        assert coverage_filter.input == "coverage"
        assert coverage_filter.where == (
            "ffibs_type = 'pl_fm' AND model_group_id = 500 AND scenario_id = 7"
        )
        assert coverage_filter.as_ == "marsh"

        assert isinstance(salinity_filter, FilterStep)
        assert salinity_filter.input == "salinity"
        assert salinity_filter.where == (
            "aggregate_type = '2_wk_max' AND model_group_id = 500 AND scenario_id = 7"
        )
        assert salinity_filter.as_ == "selected_salinity"

        assert isinstance(project_step, ProjectStep)
        assert project_step.input == "selected_salinity"
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

    def test_the_join_operands_are_named_for_what_the_threshold_filter_reads(self):
        """A join's output value columns are named after its OPERANDS, and an `as` may
        not collide with an input name -- so the two inputs take the longer names and the
        per-input steps rename down to the `marsh`/`sal` the trailing `where` references.
        """
        _, pipeline = self._build()
        join_step = next(step for step in pipeline.steps if step.op == "join")
        assert set(join_step.inputs).isdisjoint(pipeline.inputs)
        assert set(join_step.inputs) == {"marsh", "sal"}

    def test_custom_thresholds_are_interpolated(self):
        _, pipeline = self._build(marsh_threshold=0.75, salinity_threshold=3.0)
        filter_step = pipeline.steps[4]
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


# ------------------------------------------------------------------------------------------
# `land-type-confidence`
# ------------------------------------------------------------------------------------------


class TestLandTypeConfidenceParameters:
    def test_declares_no_parameters_of_its_own(self):
        LandTypeConfidence.Parameters()  # no raise

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            LandTypeConfidence.Parameters(dimension="calendar_year")


class TestLandTypeConfidenceSelection:
    params = LandTypeConfidence.Parameters()

    def test_accepts_filters_that_do_not_name_scenario_id(self):
        LandTypeConfidence.validate_selection(
            self.params, {"model_group_id": 705, "calendar_year": 2050}
        )  # no raise

    def test_rejects_a_caller_selected_scenario_id(self):
        with pytest.raises(InvalidAnalysisParametersError, match="fixed ensemble"):
            LandTypeConfidence.validate_selection(self.params, {"scenario_id": 34})

    def test_rejects_a_scenario_id_selecting_the_whole_ensemble_too(self):
        # Even the exact member set is rejected: the analysis supplies it itself, and
        # accepting it here would mean accepting a SUBSET next.
        with pytest.raises(InvalidAnalysisParametersError):
            LandTypeConfidence.validate_selection(
                self.params, {"scenario_id": sorted(SCENARIO_WEIGHTS)}
            )


class TestLandTypeConfidenceWeights:
    def test_the_weights_sum_to_one(self):
        # What makes the trailing `group`/`sum` a weighted MEAN rather than an
        # unnormalized total -- a result outside 0..1 would not be a confidence.
        assert sum(SCENARIO_WEIGHTS.values()) == pytest.approx(1.0)

    def test_every_land_type_code_is_classified_exactly_once(self):
        assert set(LAND_VALUES).isdisjoint(WATER_VALUES)
        assert set(LAND_VALUES) | set(WATER_VALUES) == {1, 2, 3, 4, 5}


class TestLandTypeConfidencePipeline:
    def _build(self, filters=None, dataset=LND_TYPE_DATASET):
        analysis = LandTypeConfidence(
            dataset=dataset,
            filters={"model_group_id": 705, "calendar_year": 2050} if filters is None else filters,
            parameters=LandTypeConfidence.Parameters(),
        )
        return analysis, analysis.pipeline()

    def test_the_leading_filter_supplies_the_fixed_ensemble(self):
        _, pipeline = self._build()
        leading = pipeline.steps[0]
        assert isinstance(leading, FilterStep)
        assert leading.input == "x"
        assert leading.where == (
            "calendar_year = 2050 AND model_group_id = 705 AND scenario_id IN (34, 35, 36, 37, 38)"
        )

    def test_a_second_filter_drops_everything_outside_the_mapping(self):
        _, pipeline = self._build()
        domain = pipeline.steps[1]
        assert isinstance(domain, FilterStep)
        assert domain.where == "value = 1 OR value = 2 OR value = 3 OR value = 4 OR value = 5"

    def test_the_mapping_calculates_come_before_the_weights(self):
        _, pipeline = self._build()
        land, water = pipeline.steps[2], pipeline.steps[3]
        assert isinstance(land, CalculateStep)
        assert land.expression == "1.0"
        assert land.where == "value = 1 OR value = 3 OR value = 4"
        assert isinstance(water, CalculateStep)
        assert water.expression == "0.0"
        assert water.where == "value = 2 OR value = 5"

    def test_one_weight_calculate_per_member_in_ascending_order(self):
        _, pipeline = self._build()
        weights = pipeline.steps[4:-1]
        assert [step.where for step in weights] == [
            f"scenario_id = {scenario_id}" for scenario_id in sorted(SCENARIO_WEIGHTS)
        ]
        assert [step.expression for step in weights] == [
            f"value * {SCENARIO_WEIGHTS[scenario_id]}" for scenario_id in sorted(SCENARIO_WEIGHTS)
        ]

    def test_the_group_sums_over_every_dimension_but_scenario_id(self):
        analysis, pipeline = self._build()
        group = pipeline.steps[-1]
        assert isinstance(group, GroupStep)
        assert group.aggregate == "sum"
        assert sorted(group.by) == ["calendar_year", "model_group_id"]
        assert "scenario_id" not in group.by
        assert analysis.collapsed_dimensions == frozenset({"scenario_id"})

    def test_the_step_count_is_the_mapping_plus_one_per_member(self):
        _, pipeline = self._build()
        assert len(pipeline.steps) == 4 + len(SCENARIO_WEIGHTS) + 1

    def test_the_output_format_is_the_one_the_pipeline_can_actually_run_under(self):
        # `Output.format` is required, so `pipeline()` has to set something; every
        # `data.{format}`/tile route then overwrites it from its own URL segment, leaving
        # `pipeline.json` -- which has no format segment -- as the only place this value is
        # observable. `lnd_type` is raster-only, so the pipeline always ends band-wide and a
        # tabular sink here would publish a pipeline that fails on resubmission.
        _, pipeline = self._build()
        assert pipeline.output.format == "tif"

    def test_the_ensemble_conjunct_is_present_even_with_no_caller_filters(self):
        # The fixed member set alone guarantees the leading `filter` exists, which is what
        # lets `pipeline()` name the input on `steps[0]` unconditionally.
        _, pipeline = self._build(filters={})
        assert pipeline.steps[0].input == "x"
        assert pipeline.steps[0].where == "scenario_id IN (34, 35, 36, 37, 38)"

    def test_a_grid_cell_filter_composes_with_the_ensemble_conjunct(self):
        _, pipeline = self._build(filters={"model_group_id": 705, "grid_cell": "3,5"})
        assert pipeline.steps[0].where == (
            "model_group_id = 705 AND scenario_id IN (34, 35, 36, 37, 38) AND grid_cell IN (3, 5)"
        )

    def test_meta_and_presentation(self):
        analysis, _ = self._build()
        assert analysis.legend is None
        assert analysis.result_kind == "value"
        assert analysis.title_suffix == " Confidence"

    def test_resolves_through_resolve_analysis(self):
        resolved = resolve_analysis(
            "land-type-confidence",
            dataset=LND_TYPE_DATASET,
            filters={"model_group_id": 705, "calendar_year": 2050},
            params={},
        )
        assert isinstance(resolved, ResolvedAnalysis)
        assert resolved.pipeline.steps[-1].aggregate == "sum"
        assert resolved.collapsed_dimensions == frozenset({"scenario_id"})

    def test_a_scenario_id_filter_is_rejected_by_resolve_analysis(self):
        with pytest.raises(InvalidAnalysisParametersError):
            resolve_analysis(
                "land-type-confidence",
                dataset=LND_TYPE_DATASET,
                filters={"model_group_id": 705, "scenario_id": [34, 35]},
                params={},
            )

    def test_the_get_encoding_numbers_all_ten_steps(self):
        # The ten steps are positional in the GET encoding (`step.<n>.` is 1-based), so a
        # caller can reproduce the analysis as a hand-authored pipeline URL.
        query = pipeline_to_query(self._build()[1])
        assert query["step.10.op"] == "group"
        assert query["step.5.expression"] == "value * 0.085"
