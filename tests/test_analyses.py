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
    dataset_output_format,
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
from cpra.mp.data.analyses.lnd_type_plus import WATER_LAND_TYPE, LndTypePlus
from cpra.mp.data.analyses.salinity_veg_investigation import SalinityVegInvestigation
from cpra.mp.data.analysis.request.pipeline import (
    CaseStep,
    FilterStep,
    GroupStep,
    ProjectStep,
    pipeline_to_query,
)
from cpra.mp.data.legends import LEGENDS

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
            "lnd-type-plus",
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

        assert ANALYSES["lnd-type-plus"].Meta.type == "custom"
        assert ANALYSES["lnd-type-plus"].Meta.variables is None
        assert ANALYSES["lnd-type-plus"].Meta.legend == "lnd_type_plus"
        assert ANALYSES["lnd-type-plus"].Meta.result_kind == "category"

    def test_every_declared_legend_is_a_real_one(self):
        """A `Meta.legend` is a plain string, so a typo (or a legend removed from
        `structure/legends.csv` out from under an analysis) would only surface as a catalog
        entry naming a legend the `/v1/legends.json` route never serves.
        """
        for name, cls in ANALYSES.items():
            if cls.Meta.legend is not None:
                assert cls.Meta.legend in LEGENDS, name


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
        case = next(step for step in result.pipeline.steps if isinstance(step, CaseStep))
        assert case.cases[0].when == "calendar_day = '2070-12-31'::date"

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

    def test_dataset_output_format_follows_the_grid_kind(self):
        """A `type="dataset"` analysis that never changes shape ends band-wide on a raster
        grid and tabular on a vector one, and `validate_shapes` rejects the mismatched sink
        in either direction -- so the declared format has to be derived, not fixed.
        """
        assert dataset_output_format(SSF_DATASET) == "csv"  # ecoregion_v001, vector
        assert dataset_output_format(LND_TYPE_DATASET) == "tif"  # morph_pixel_v002, raster

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
    def _build(self, order, filters=None, dataset=SSF_DATASET):
        filters = filters or {"calendar_year": [2030, 2040], "model_group_id": 500, "scenario_id": 7, "percentile": 50}
        analysis = Difference(
            dataset=dataset,
            filters=filters,
            parameters=Difference.Parameters(dimension="calendar_year", order=order),
        )
        return analysis, analysis.pipeline()

    def test_negates_the_later_value_for_order_asc(self):
        _, pipeline = self._build("asc")
        case = pipeline.steps[1]
        assert isinstance(case, CaseStep)
        assert [(arm.when, arm.then) for arm in case.cases] == [
            ("calendar_year = 2040", "value * -1")
        ]
        # The un-negated side has to reach the `sum` unchanged for the subtraction to come
        # out, so the pass-through is written rather than left implicit.
        assert case.else_ == "value"

    def test_negates_the_earlier_value_for_order_desc(self):
        _, pipeline = self._build("desc")
        case = pipeline.steps[1]
        assert case.cases[0].when == "calendar_year = 2030"
        assert case.else_ == "value"

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
        assert isinstance(pipeline.steps[1], CaseStep)
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

    def test_the_output_format_follows_the_selected_dataset(self):
        """`difference` declares `Meta.variables = None`, so it genuinely runs on both grid
        kinds, and it never changes shape -- its final frame is long on a vector grid and
        band-wide on a raster one. `pipeline.json` is the one route that publishes this
        value rather than overwriting it from a URL segment, so a fixed format would hand
        raster callers a pipeline `validate_shapes` rejects on resubmission.
        """
        _, vector = self._build("asc")
        assert vector.output.format == "csv"
        _, raster = self._build(
            "asc",
            filters={"calendar_year": [2030, 2040], "model_group_id": 500, "scenario_id": 7},
            dataset=LND_TYPE_DATASET,
        )
        assert raster.output.format == "tif"

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
    def _build(self, filters, dataset=SSF_DATASET):
        analysis = Concatenate(
            dataset=dataset,
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
        case = pipeline.steps[1]
        assert isinstance(case, CaseStep)
        assert [(arm.when, arm.then) for arm in case.cases] == [
            ("calendar_year = 2030", "value * 10")
        ]
        # The ones digit: the year no arm names keeps its value through the `sum`.
        assert case.else_ == "value"
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
        assert pipeline.steps[1].cases[0].then == "value * 10"

    def test_three_image_mode(self):
        analysis, pipeline = self._build(
            {"calendar_year": [2030, 2040], "model_group_id": [520, 500], "scenario_id": 7, "percentile": 50}
        )
        assert len(pipeline.steps) == 3

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

        # Both digits in ONE step, as arms: read against the value the frame arrived with,
        # so the hundreds arm cannot re-match what the tens arm wrote. As two successive
        # rewrites they were independent only because their conditions happened to be
        # disjoint.
        case = pipeline.steps[1]
        assert isinstance(case, CaseStep)
        assert [(arm.when, arm.then) for arm in case.cases] == [
            ("calendar_year = 2030", "value * 100"),
            ("model_group_id = 500 AND calendar_year = 2040", "value * 10"),
        ]
        assert case.else_ == "value"

        group = pipeline.steps[2]
        assert isinstance(group, GroupStep)
        assert sorted(group.by) == sorted(["scenario_id", "percentile"])
        assert "model_group_id" not in group.by

        assert analysis.collapsed_dimensions == frozenset({"calendar_year", "model_group_id"})

    def test_meta_and_legend(self):
        analysis, _ = self._build({"calendar_year": [2030, 2040], "model_group_id": 500})
        assert analysis.legend == "lnd_chng"
        assert analysis.result_kind == "category"

    def test_the_output_format_follows_the_selected_dataset(self):
        """`concatenate` neither projects nor changes shape, so its final frame's kind is
        the input grid's -- and `pipeline.json`, the only route that does not overwrite
        this value from a URL segment, has to publish a pipeline that would still validate
        on resubmission. `Meta.variables` is `lnd_type`, which lives only on a raster grid,
        so the real-world answer is `tif`.
        """
        filters = {"calendar_year": [2030, 2040], "model_group_id": 500}
        _, vector = self._build({**filters, "scenario_id": 7, "percentile": 50})
        assert vector.output.format == "csv"
        _, raster = self._build({**filters, "scenario_id": 7}, dataset=LND_TYPE_DATASET)
        assert raster.output.format == "tif"


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

        coverage_filter, salinity_filter, project_step, join_step, case_step, group_step = (
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

        # The threshold test collapses the pairing and says what survives it, which a
        # `filter` has no way to state.
        assert isinstance(case_step, CaseStep)
        assert [(arm.when, arm.then) for arm in case_step.cases] == [
            ("marsh > 0.5 AND sal > 5.5", "marsh")
        ]
        assert case_step.else_ is None

        assert isinstance(group_step, GroupStep)
        assert group_step.by == []
        assert group_step.aggregate == "min"
        assert group_step.of == "calendar_year"

        assert pipeline.output.format == "tif"

    def test_the_join_operands_are_named_for_what_the_threshold_case_reads(self):
        """A join's output value columns are named after its OPERANDS, and an `as` may
        not collide with an input name -- so the two inputs take the longer names and the
        per-input steps rename down to the `marsh`/`sal` the trailing arm references.
        """
        _, pipeline = self._build()
        join_step = next(step for step in pipeline.steps if step.op == "join")
        assert set(join_step.inputs).isdisjoint(pipeline.inputs)
        assert set(join_step.inputs) == {"marsh", "sal"}

    def test_custom_thresholds_are_interpolated(self):
        _, pipeline = self._build(marsh_threshold=0.75, salinity_threshold=3.0)
        case_step = pipeline.steps[4]
        assert case_step.cases[0].when == "marsh > 0.75 AND sal > 3.0"

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

    def test_the_mapping_is_one_case_before_the_weights(self):
        # Both arms read the INCOMING code, so unlike the two `calculate`s this replaced
        # there is no ordering question to get wrong -- and no `else`, so a code neither
        # arm names comes out null rather than as itself.
        _, pipeline = self._build()
        mapping = pipeline.steps[2]
        assert isinstance(mapping, CaseStep)
        assert [(arm.when, arm.then) for arm in mapping.cases] == [
            ("value = 1 OR value = 3 OR value = 4", "1.0"),
            ("value = 2 OR value = 5", "0.0"),
        ]
        assert mapping.else_ is None

    def test_one_weight_arm_per_member_in_ascending_order(self):
        # One `case` with an arm apiece, not N successive rewrites: each arm's `when` is
        # purely dimensional, so it picks out that member's bands and scales them, and the
        # five arms partition the frame the leading `filter` pinned to exactly these five.
        _, pipeline = self._build()
        weights = pipeline.steps[3]
        assert isinstance(weights, CaseStep)
        assert [(arm.when, arm.then) for arm in weights.cases] == [
            (f"scenario_id = {scenario_id}", f"value * {SCENARIO_WEIGHTS[scenario_id]}")
            for scenario_id in sorted(SCENARIO_WEIGHTS)
        ]
        # No pass-through: nothing reaches the `else`, and writing one would be claiming
        # there is an unweighted sixth member.
        assert weights.else_ is None

    def test_the_group_sums_over_every_dimension_but_scenario_id(self):
        analysis, pipeline = self._build()
        group = pipeline.steps[-1]
        assert isinstance(group, GroupStep)
        assert group.aggregate == "sum"
        assert sorted(group.by) == ["calendar_year", "model_group_id"]
        assert "scenario_id" not in group.by
        assert analysis.collapsed_dimensions == frozenset({"scenario_id"})

    def test_the_step_count_does_not_grow_with_the_member_count(self):
        # Two filters, the mapping `case`, the weighting `case`, the `group` -- five, no
        # matter how many members `SCENARIO_WEIGHTS` names, because the members are arms of
        # one step rather than steps of their own.
        _, pipeline = self._build()
        assert len(pipeline.steps) == 5

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

    def test_the_get_encoding_numbers_all_five_steps(self):
        # The five steps are positional in the GET encoding (`step.<n>.` is 1-based), so a
        # caller can reproduce the analysis as a hand-authored pipeline URL. Both `case`s'
        # arms nest one level further down, positionally in their own right.
        query = pipeline_to_query(self._build()[1])
        assert query["step.5.op"] == "group"
        assert query["step.3.cases.1.then"] == "1.0"
        assert query["step.3.cases.2.when"] == "value = 2 OR value = 5"
        assert query["step.4.cases.1.when"] == "scenario_id = 34"
        assert query["step.4.cases.1.then"] == "value * 0.085"


# ------------------------------------------------------------------------------------------
# `lnd-type-plus`
# ------------------------------------------------------------------------------------------


class TestLndTypePlusParameters:
    def test_requires_model_group_id_and_scenario_id(self):
        with pytest.raises(ValidationError):
            LndTypePlus.Parameters(model_group_id=705)

    def test_calendar_year_is_optional(self):
        assert LndTypePlus.Parameters(model_group_id=705, scenario_id=7).calendar_year is None

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            LndTypePlus.Parameters(model_group_id=705, scenario_id=7, dimension="calendar_year")


class TestLndTypePlusSelection:
    """It takes no `filters` at all -- see `LndTypePlus.validate_selection`: its three
    inputs span two grids, so a `grid_cell` names a cell in a lattice the request does not
    say which of, a bare `where` has no one operand to apply to, and a spatial predicate
    would turn the join long and invalidate the raster sink rather than simply narrowing
    the result.
    """

    params = LndTypePlus.Parameters(model_group_id=705, scenario_id=7)

    def test_accepts_an_empty_selection(self):
        LndTypePlus.validate_selection(self.params, {})  # no raise

    def test_rejects_a_grid_cell(self):
        with pytest.raises(InvalidAnalysisParametersError, match="takes no selection"):
            LndTypePlus.validate_selection(self.params, {"grid_cell": "3,5"})

    def test_rejects_a_where(self):
        with pytest.raises(InvalidAnalysisParametersError, match="takes no selection"):
            LndTypePlus.validate_selection(self.params, {"where": "value > 1"})


class TestLndTypePlusPipeline:
    def _build(self, **overrides):
        params = {"model_group_id": 705, "scenario_id": 7, "calendar_year": 2050, **overrides}
        analysis = LndTypePlus(
            dataset=None, filters={}, parameters=LndTypePlus.Parameters(**params)
        )
        return analysis, analysis.pipeline()

    def test_the_three_inputs_are_fixed(self):
        _, pipeline = self._build()
        assert set(pipeline.inputs) == {"stage", "elevation", "land_type"}
        stage = pipeline.inputs["stage"]
        assert (stage.variable, stage.time_unit, stage.grid) == (
            "stg",
            "annual",
            "hydro_compartment_v002",
        )
        elevation = pipeline.inputs["elevation"]
        assert (elevation.variable, elevation.time_unit, elevation.grid) == (
            "elev",
            "annual",
            "morph_pixel_v002",
        )
        land_type = pipeline.inputs["land_type"]
        assert (land_type.variable, land_type.time_unit, land_type.grid) == (
            "lnd_type",
            "annual",
            "morph_pixel_v002",
        )

    def test_the_whole_step_shape(self):
        """Five steps, and the last of them does all the work: stage onto the pixel grid,
        one join bringing all three inputs together, the caller's selection, and a single
        total `case` that subtracts, converts, restricts to water, reclassifies and passes
        the land type through -- with no reduce after it at all.
        """
        _, pipeline = self._build()
        assert [step.op for step in pipeline.steps] == ["filter", "project", "join", "filter", "case"]
        stage_filter, project_step, depth_join, selection, reclassify = pipeline.steps

        # The ONE leading filter, and it carries no part of the caller's selection: only
        # `aggregate_type`, which this analysis pins itself and which the `case`'s pairing
        # drops, so there is no later frame it could be stated on.
        assert isinstance(stage_filter, FilterStep)
        assert stage_filter.input == "stage"
        assert stage_filter.where == "aggregate_type = 'mean'"

        assert project_step.input == "selected_stage"
        assert project_step.to == "morph_pixel_v002"
        # Vector ONTO raster broadcasts rather than reduces, so no `aggregate`.
        assert project_step.aggregate is None
        assert project_step.as_ == "stage_m"

        # ONE join for all three inputs, not a chain of two: `join` is N-ary, and the
        # `case` below pairs N operands band-for-band just as readily as two. `elevation`
        # and `land_type` are the INPUTS -- nothing stands between either read and the
        # join, and no value-domain filter is wanted on the land type either, since
        # `lnd_type` tops out at `5` and anything outside its domain is the `-9999` nodata
        # sentinel, masked to null on read.
        assert depth_join.type == "spatial"
        assert depth_join.inputs == ["stage_m", "elevation", "land_type"]

        # The caller's selection, stated once, on the frame all three inputs have reached.
        assert isinstance(selection, FilterStep)
        assert selection.input is None
        assert selection.where == "calendar_year = 2050 AND model_group_id = 705 AND scenario_id = 7"
        assert selection.as_ == "selected"

        # Everything else, in one step, over the triple the join produced. Descending by
        # bound, because the first matching arm wins; the last one is the domain floor. The
        # millimetre-to-metre conversion is written into the shared depth rather than
        # standing as a `calculate` of its own -- there is no `neg_elevation_m` any more.
        #
        # The water restriction rides on every arm rather than standing as a mask of its
        # own, and the deepest class reports the open-water land type rather than a code of
        # its own, which is what makes the `else` sufficient and a trailing `max` needless.
        assert isinstance(reclassify, CaseStep)
        assert [(arm.when, arm.then) for arm in reclassify.cases] == [
            ("land_type = 2 AND stage_m - elevation / 1000 > 1.0", "2"),
            ("land_type = 2 AND stage_m - elevation / 1000 > 0.46", "7"),
            ("land_type = 2 AND stage_m - elevation / 1000 > 0", "6"),
        ]
        # The land type, unchanged, for everything no arm claims -- which is the whole
        # reason nothing follows this step.
        assert reclassify.else_ == "land_type"
        assert reclassify.as_ is None

        assert pipeline.output.format == "tif"

    def test_nothing_groups_on_the_dimension_only_stage_carries(self):
        # `aggregate_type` is required by `stg` and by neither raster, so pairing or
        # grouping on it would split every triple into one-band groups that never reduce
        # together. The `case`'s pairing key is the shared dimensions, which excludes it by
        # construction; this pins that no step reintroduces it.
        _, pipeline = self._build()
        for step in pipeline.steps:
            assert "aggregate_type" not in (getattr(step, "by", None) or [])

    def test_every_depth_lands_on_the_code_the_class_table_names(self):
        """Applying the `case` arms in order to any depth lands on the code
        `DEPTH_CLASSES` names, and everything at or below the domain floor falls through to
        the land type.

        Re-implemented here as a plain first-match-wins interpreter over the pipeline's
        own `when` strings rather than against a second copy of the class table, so an
        edit to the arms that gets an interval's open/closed end wrong fails this test.
        The `else` is interpreted too, since with no reduce after the `case` it is the only
        thing standing between an unclaimed pixel and nodata.

        Every arm also carries the water conjunct, which is interpreted here too -- so a
        land type that stops being tested, or stops being tested for water, fails below
        rather than being quietly dropped from the interpreter.
        """
        _, pipeline = self._build()
        case_step = pipeline.steps[-1]
        arms = [(arm.when, arm.then) for arm in case_step.cases]
        assert case_step.else_ == "land_type"

        def run(depth, land_type=WATER_LAND_TYPE):
            for when, then in arms:
                water, _, depth_test = when.partition(" AND ")
                assert water == "land_type = 2", f"unexpected water conjunct in {when!r}"
                if land_type == WATER_LAND_TYPE and _conjunct_holds(
                    depth, depth_test, measured="stage_m - elevation / 1000"
                ):
                    return float(then)
            return land_type  # the `else`

        # Every interval is closed at its upper bound, so each bound belongs to the class
        # BELOW it -- the pair either side of each is what pins that.
        assert run(0.01) == 6
        assert run(0.46) == 6
        assert run(0.47) == 7
        assert run(1.0) == 7
        # Deeper than 1.0 m is not a class of its own: it reports the open-water land type,
        # which is what the `lnd_type_plus` legend's relabelled `2` ("Water - Deep") means.
        assert run(1.01) == 2
        assert run(12.5) == 2
        # The domain floor. Zero itself is dry land standing at the water level, not a
        # depth of zero, and a negative is elevation above it -- neither is a class, so
        # both fall through to the land type they already were.
        assert run(0) == 2
        assert run(-3.2) == 2
        # ... and no depth is a class on a pixel that is not open water, however deeply the
        # water surface stands above its ground: the land type passes straight through.
        assert run(12.5, land_type=1) == 1
        assert run(0.47, land_type=5) == 5

    def test_every_step_name_a_join_reads_is_something_the_pipeline_defines(self):
        # A join operand is either an input or an earlier step's `as`, and the two
        # namespaces are disjoint (`parse_pipeline` rejects an `as` colliding with an input
        # name) -- which is what fixes the `stage`/`stage_m` naming: stage is filtered
        # before it is projected, so the projection needs a name of its own. `land_type`
        # needs no such split, because nothing stands between that read and the join, so
        # the joins name the input directly.
        _, pipeline = self._build()
        defined = set(pipeline.inputs)
        for step in pipeline.steps:
            for operand in getattr(step, "inputs", None) or []:
                assert operand in defined
            if step.as_ is not None:
                assert step.as_ not in pipeline.inputs
                defined.add(step.as_)
        assert "land_type" in pipeline.inputs

    def test_an_omitted_calendar_year_drops_only_that_conjunct(self):
        _, pipeline = self._build(calendar_year=None)
        # The stage filter never carried the selection at all, so nothing changes there.
        assert pipeline.steps[0].where == "aggregate_type = 'mean'"
        assert pipeline.steps[-2].where == "model_group_id = 705 AND scenario_id = 7"
        # ... and nothing reduces the years together: the `case`'s pairing is keyed on the
        # dimensions the inputs share, `calendar_year` among them, so an unfiltered request
        # returns one output band per year.
        analysis, _ = self._build(calendar_year=None)
        assert "calendar_year" not in analysis.collapsed_dimensions

    def test_only_stages_aggregate_type_is_collapsed(self):
        # The axis this analysis mainly collapses is the OPERAND axis (which input a band
        # came from), which is not a dimension. The one real dimension that does go is
        # `aggregate_type`: only `stg` carries it, the analysis pins it to `mean` itself,
        # and the `case`'s pairing key (the dimensions all three inputs share) cannot carry
        # it -- so a caller can neither see nor vary it, which is what
        # `collapsed_dimensions` exists to tell the visualization layer.
        analysis, _ = self._build()
        assert analysis.collapsed_dimensions == frozenset({"aggregate_type"})

    def test_meta_and_presentation(self):
        analysis, _ = self._build()
        assert analysis.legend == "lnd_type_plus"
        assert analysis.result_kind == "category"
        assert analysis.title_suffix == ""

    def test_resolves_through_resolve_analysis(self):
        resolved = resolve_analysis(
            "lnd-type-plus",
            dataset=None,
            filters={},
            params={"model_group_id": 705, "scenario_id": 7, "calendar_year": 2050},
        )
        assert isinstance(resolved, ResolvedAnalysis)
        assert resolved.pipeline.steps[-1].else_ == "land_type"
        assert resolved.collapsed_dimensions == frozenset({"aggregate_type"})

    def test_string_parameters_off_a_query_string_are_coerced(self):
        # Every parameter arrives as a string on the real routes (`_analysis_extra_params`
        # reads the raw query string), so the `int` fields have to coerce.
        resolved = resolve_analysis(
            "lnd-type-plus",
            dataset=None,
            filters={},
            params={"model_group_id": "705", "scenario_id": "7", "calendar_year": "2050"},
        )
        assert resolved.pipeline.steps[-2].where == (
            "calendar_year = 2050 AND model_group_id = 705 AND scenario_id = 7"
        )

    def test_a_selection_is_rejected_by_resolve_analysis(self):
        with pytest.raises(InvalidAnalysisParametersError, match="takes no selection"):
            resolve_analysis(
                "lnd-type-plus",
                dataset=None,
                filters={"grid_cell": "3,5"},
                params={"model_group_id": 705, "scenario_id": 7},
            )


def _conjunct_holds(value, conjunct, measured="value"):
    """Evaluate one `<measured> <op> <literal>` conjunct of a reclassification predicate --
    the only operators those steps use. `measured` is whatever the step writes on the left,
    the placeholder `value` unpaired and the operand arithmetic (`stage_m - elevation /
    1000`) over a joined frame; it is asserted rather than parsed, so an arm that
    starts measuring something else fails here instead of being silently interpreted.
    Deliberately narrow: it exists to interpret the pipeline this analysis really builds,
    not to be a general SQL evaluator.
    """
    for operator, test in (
        ("<=", lambda a, b: a <= b),
        (">", lambda a, b: a > b),
        ("<", lambda a, b: a < b),
    ):
        left, sep, right = conjunct.partition(f" {operator} ")
        if sep:
            assert left == measured, f"unexpected left-hand side in {conjunct!r}"
            return test(value, float(right))
    raise AssertionError(f"unrecognized conjunct {conjunct!r}")
