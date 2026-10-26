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
from typing import ClassVar

import numpy as np
import pytest
from pydantic import ValidationError

from cpra.mp.data.analyses import (
    ANALYSES,
    AnalysisDatasetMismatchError,
    AnalysisRegistryError,
    DatasetRef,
    InvalidAnalysisParametersError,
    ResolvedAnalysis,
    UnknownAnalysisError,
    _register,
    resolve_analysis,
    resolve_analysis_chain,
)
from cpra.mp.data.analyses.base import (
    Analysis,
    AnalysisMeta,
    cast_dimension_value,
    dataset_output_format,
    default_collapsed_dimensions,
    format_dimension_literal,
    is_filterable_dimension,
    normalized_filter_values,
    selection_filter_where,
)
from cpra.mp.data.analyses.diff import Difference
from cpra.mp.data.analyses.lnd_chng import (
    MP2023_BASELINE_YEAR,
    MP2029_BASELINE_YEAR,
    LandChange,
    baseline_year,
)
from cpra.mp.data.analyses.lnd_conf import (
    LAND_VALUES,
    SCENARIO_WEIGHTS,
    WATER_VALUES,
    LandConfidence,
)
from cpra.mp.data.analyses.lnd_conf_chng import CONFIDENCE_CLASSES
from cpra.mp.data.analyses.lnd_type_plus import WATER_LAND_TYPE, LndTypePlus
from cpra.mp.data.analyses.sal_veg_investigation import SalinityVegInvestigation
from cpra.mp.data.analysis.execution.sink import DEFAULT_NODATA
from cpra.mp.data.analysis.request.pipeline import (
    CalculateStep,
    CaseStep,
    FilterStep,
    GroupStep,
    JoinStep,
    ProjectStep,
    pipeline_to_query,
)
from cpra.mp.data.legends import LEGENDS

SSF_DATASET = DatasetRef(variable="ssf", time_unit="annual", grid="ecoregion_v001")
SAL_DATASET = DatasetRef(variable="sal", time_unit="daily", grid="hydro_compartment_v001")
# `lnd_conf`'s real dataset. Nothing here reads from disk -- these tests build
# pipelines and inspect them -- so the real variable/grid can be named directly rather than
# through the small stand-ins the execution tests need.
LND_TYPE_DATASET = DatasetRef(variable="lnd_type", time_unit="annual", grid="morph_pixel_v002")


# ------------------------------------------------------------------------------------------
# The registry itself
# ------------------------------------------------------------------------------------------


class TestRegistry:
    def test_contains_exactly_the_registered_entries(self):
        assert set(ANALYSES) == {
            "sal_veg_investigation",
            "diff",
            "lnd_chng",
            "lnd_conf",
            "lnd_conf_chng",
            "lnd_type_plus",
            "sediment_dynamics",
        }

    def test_keyed_by_meta_name(self):
        for name, cls in ANALYSES.items():
            assert cls.Meta.name == name

    def test_every_meta_is_a_validated_model(self):
        """Not a tautology: `cls.Meta.name` reads identically off a bare `class Meta:`, so
        an entry that kept the old form would key into `ANALYSES` correctly and only
        differ in having skipped every check `AnalysisMeta` performs. `_register` is what
        forbids it, and this is the assertion that `_register` is still wired in."""
        for cls in ANALYSES.values():
            assert isinstance(cls.Meta, AnalysisMeta)

    def test_a_bare_meta_is_refused_at_registration(self):
        class _BareMeta:
            name = "bare"
            label = "Bare"
            type = "dataset"
            kind = "dimension"

        class _BareAnalysis(Analysis):
            Meta = _BareMeta

        with pytest.raises(AnalysisRegistryError, match="must declare `Meta = AnalysisMeta"):
            _register([_BareAnalysis])

    def test_an_analysis_with_no_meta_of_its_own_is_refused(self):
        """`Analysis` declares `Meta` as a bare `ClassVar` annotation with no value, so a
        subclass that forgets one has no inherited stand-in to pick up -- the lookup fails
        here rather than at the first consumer to read a field off it."""

        class _NoMeta(Analysis):
            pass

        with pytest.raises(AnalysisRegistryError, match="no Meta of its own"):
            _register([_NoMeta])

    def test_a_dimension_named_parameter_is_refused_at_registration(self):
        """The rule the decoder's partition rests on: a dimension name is ALWAYS a filter.
        It is sound only while no analysis declares a dimension-named parameter --
        `lnd_type_plus` and `sal_veg_investigation` both did, back when a fixed-input
        analysis had no route that could carry a filter -- and a reintroduction is both the
        worst failure the design has and invisible to every response-level test: the
        caller's `model_group_id` lands in `filters`, `Parameters` takes its default, and
        the request 200s for the wrong model group. So it is refused at import.
        """

        class _DimensionNamed(Analysis):
            Meta = AnalysisMeta(name="dim_named", label="Dimension-named", kind="internal")

            class Parameters(Analysis.Parameters):
                model_group_id: int | None = None

        with pytest.raises(AnalysisRegistryError, match="always a FILTER"):
            _register([_DimensionNamed])

    def test_a_non_dimension_parameter_registers_fine(self):
        """The other side of the guard: `is_filterable_dimension` is what draws the line,
        so an ordinary parameter name -- and a DATASET-typed dimension, which selects an
        input rather than filtering one -- is untouched by it."""

        class _Ordinary(Analysis):
            Meta = AnalysisMeta(name="ordinary", label="Ordinary", kind="internal")

            class Parameters(Analysis.Parameters):
                threshold: float = 0.5

        assert _register([_Ordinary]) == {"ordinary": _Ordinary}

    def test_meta_shape(self):
        # No `Meta.type` assertions here any more -- the field is gone with the route split
        # that read it. `Meta.kind` is the surviving partition, and it is a DIFFERENT one:
        # `lnd_type_plus` was `type="custom"` and is `kind="variable"`, because "fixes its
        # own inputs" and "stands in for a variable in the portal dropdown" are unrelated
        # facts. Whether an entry takes the caller's dataset is asserted where it is now
        # stated -- by `pipeline()`, in `TestResolveAnalysisContract` below.
        assert ANALYSES["diff"].Meta.kind == "dimension"
        assert ANALYSES["diff"].Meta.variables is None
        assert ANALYSES["diff"].Meta.legend is None
        assert ANALYSES["diff"].Meta.result_kind == "difference"

        assert ANALYSES["lnd_chng"].Meta.kind == "variable"
        assert ANALYSES["lnd_chng"].Meta.variables == ["lnd_type"]
        assert ANALYSES["lnd_chng"].Meta.legend == "lnd_chng"
        assert ANALYSES["lnd_chng"].Meta.result_kind == "category"
        # Nothing: `lnd_chng` compares every selected year against one derived baseline and
        # keeps the year column, so `calendar_year` stays the caller's to choose.
        assert ANALYSES["lnd_chng"].Meta.subsumed_dimensions == []

        assert ANALYSES["sal_veg_investigation"].Meta.result_kind == "value"
        # Explicit, and explicitly `"internal"` -- neither a portal variable nor a
        # chainable stage. It declared no `kind` at all until `AnalysisMeta` made the
        # field required, which read as `None` at every consumer.
        assert ANALYSES["sal_veg_investigation"].Meta.kind == "internal"

        assert ANALYSES["lnd_conf"].Meta.kind == "variable"
        assert ANALYSES["lnd_conf"].Meta.variables == ["lnd_type"]
        assert ANALYSES["lnd_conf"].Meta.legend == "lnd_conf"
        assert ANALYSES["lnd_conf"].Meta.result_kind == "value"

        assert ANALYSES["lnd_type_plus"].Meta.kind == "variable"
        assert ANALYSES["lnd_type_plus"].Meta.variables is None
        assert ANALYSES["lnd_type_plus"].Meta.legend == "lnd_type_plus"
        assert ANALYSES["lnd_type_plus"].Meta.result_kind == "category"

    def test_every_declared_legend_is_a_real_one(self):
        """A `Meta.legend` is a plain string, so a typo (or a legend removed from
        `structure/legends.csv` out from under an analysis) would only surface as a catalog
        entry naming a legend the `/v1/legends.json` route never serves.
        """
        for name, cls in ANALYSES.items():
            if cls.Meta.legend is not None:
                assert cls.Meta.legend in LEGENDS, name

    def test_land_type_confidence_declares_the_undifferenced_legend(self):
        """`lnd_conf` outputs a weighted FRACTION on `[0, 1]`, so the legend it
        declares has to classify that domain. A wrong legend name is silently wrong colors,
        never an error, so the domain is asserted directly here.
        """
        declared = [
            b for b in LEGENDS[ANALYSES["lnd_conf"].Meta.legend].break_points["upper_bound"].to_list() if b is not None
        ]
        # `0.0` is the transparent `No Data` band's own upper bound, and the fraction's
        # floor: nothing this analysis outputs falls below it, so the band catches the
        # nodata sentinel alone (see the module docstring).
        assert all(0.0 <= b <= 1.0 for b in declared), declared

    def test_land_confidence_legend_bands_are_the_change_classes(self):
        """`lnd_conf`'s map shows the same four classes `lnd_conf_chng` pairs up, so a cell
        reads the same class in both -- including the float-slack fractions just off `0.5`
        and `1.0`."""
        legend = LEGENDS[ANALYSES["lnd_conf"].Meta.legend]
        weights = SCENARIO_WEIGHTS.values()
        fractions = [0.0, min(weights), 0.5 - 1e-7, 0.5, 0.5 + 1e-7, 1.0 - min(weights), 1.0 - 1e-7, 1.0]
        labels = legend.break_points["label"].to_list()
        bounds = [b for b in legend.break_points["upper_bound"].to_list() if b is not None]
        assert [labels[int(np.digitize(f, bounds))] for f in fractions] == [
            "0",
            "> 0 to < 0.5",
            "0.5 to < 1",
            "0.5 to < 1",
            "0.5 to < 1",
            "0.5 to < 1",
            "1",
            "1",
        ]
        # And the same classes `lnd_conf_chng`'s `case` assigns.
        thresholds = [float(when.split()[-1]) for when, _ in CONFIDENCE_CLASSES[:-1]]
        assert bounds[1:] == thresholds

    def test_land_confidence_change_legend_covers_every_transition_code(self):
        """Each of the 16 `initial * 10 + current` codes lands in a labelled, opaque band
        of its own color class -- not the `No Data` band, and not a neighbor's."""
        legend = LEGENDS[ANALYSES["lnd_conf_chng"].Meta.legend]
        rows = legend.break_points.to_dicts()
        codes = [i * 10 + c for i in range(1, 5) for c in range(1, 5)]
        rgba = legend.colorize_pixels(np.array([[codes]], dtype=float))[0]
        for code, color in zip(codes, rgba):
            assert color[3] == 255, code
        # Upper bounds are exclusive, so code N is in the row whose upper bound is the next
        # code -- the label for code 14 sits on the row bounded by 21.
        bounds = [r["upper_bound"] for r in rows]
        label_of = {c: rows[int(np.digitize(c, [b for b in bounds if b is not None]))]["label"] for c in codes}
        assert label_of[11] == "Stays Water, High Confidence"
        assert label_of[14] == "Land Gain, High Confidence"
        assert label_of[41] == "Land Loss, High Confidence"
        assert label_of[44] == "Stays Land, High Confidence"
        assert label_of[23] == "Land Gain, Low Confidence"
        assert label_of[32] == "Land Loss, Low Confidence"

    @pytest.mark.parametrize("legend_name", ["lnd_conf", "lnd_conf_chng"])
    def test_the_confidence_legends_render_nodata_transparent(self, legend_name):
        """The nodata sentinel has to fall in a transparent band.

        `Legend.colorize_pixels` digitizes EVERY pixel a tile carries, the nodata fill
        included (`execution.sink.render_tile` fills nulls with `DEFAULT_NODATA` before the
        warp and never masks them again), so a value below the lowest break lands in the
        first bucket and is drawn in that bucket's color. Without a leading `No Data` band
        that was the darkest water color for `lnd_conf` -- the whole area outside the model
        domain rendered as a confident all-water answer.

        Asserted through `colorize_pixels` rather than by reading the CSV, since the claim
        is about what a tile actually comes out as: alpha `0` in the RGBA it returns.
        """
        color = LEGENDS[legend_name].colorize_pixels(np.array([[[DEFAULT_NODATA]]]))
        assert color[0][0][3] == 0, color


# ------------------------------------------------------------------------------------------
# `resolve_analysis`'s own exception contract
# ------------------------------------------------------------------------------------------


class TestResolveAnalysisContract:
    def test_unknown_name_raises_unknown_analysis_error(self):
        with pytest.raises(UnknownAnalysisError):
            resolve_analysis("not-a-real-analysis", dataset=None, filters={}, params={})

    def test_an_analysis_that_needs_a_dataset_says_so_when_given_none(self):
        """`AnalysisDatasetMismatchError`, raised by `Analysis.require_dataset` from inside
        `pipeline()` -- not by a `Meta.type` check ahead of it. The retired field named the
        ROUTE a request had to arrive on, and there is one route now; what survives is the
        requirement itself, stated by the method that actually needs the dataset."""
        # A valid two-value selection, so `diff`'s own `validate_selection` passes and the
        # missing dataset is what the call actually fails on.
        with pytest.raises(AnalysisDatasetMismatchError):
            resolve_analysis(
                "diff",
                dataset=None,
                filters={"calendar_year": [2030, 2040]},
                params={"dimension": "calendar_year", "order": "asc"},
            )

    def test_an_analysis_with_fixed_inputs_says_so_when_given_a_dataset(self):
        """The other direction, from `Analysis.reject_dataset`. `sal_veg_investigation`
        hardcodes its two inputs, so a caller-named dataset is not a narrowing of anything
        -- it is a dataset the pipeline would never read."""
        with pytest.raises(AnalysisDatasetMismatchError):
            resolve_analysis(
                "sal_veg_investigation",
                dataset=SSF_DATASET,
                filters={},
                params={},
            )

    def test_missing_required_parameter_raises_invalid_parameters(self):
        with pytest.raises(InvalidAnalysisParametersError):
            resolve_analysis(
                "diff",
                dataset=SSF_DATASET,
                filters={"calendar_year": [2030, 2040]},
                params={"dimension": "calendar_year"},  # missing `order`
            )

    def test_unknown_parameter_raises_invalid_parameters(self):
        with pytest.raises(InvalidAnalysisParametersError):
            resolve_analysis(
                "diff",
                dataset=SSF_DATASET,
                filters={"calendar_year": [2030, 2040]},
                params={"dimension": "calendar_year", "order": "asc", "bogus": 1},
            )

    def test_invalid_literal_raises_invalid_parameters(self):
        with pytest.raises(InvalidAnalysisParametersError):
            resolve_analysis(
                "diff",
                dataset=SSF_DATASET,
                filters={"calendar_year": [2030, 2040]},
                params={"dimension": "calendar_year", "order": "sideways"},
            )

    def test_successful_resolve_returns_resolved_analysis(self):
        result = resolve_analysis(
            "diff",
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

    FILTERS: ClassVar[dict] = {
        "calendar_day": [date(2019, 1, 1), date(2070, 12, 31)],
        "model_group_id": 500,
        "scenario_id": 7,
        "aggregate_type": "mean",
    }

    def test_difference_over_calendar_day_resolves(self):
        result = resolve_analysis(
            "diff",
            dataset=SAL_DATASET,
            filters=self.FILTERS,
            params={"dimension": "calendar_day", "order": "asc"},
        )
        assert result.result_kind == "difference"
        assert result.collapsed_dimensions == frozenset({"calendar_day"})

    def test_the_dates_survive_into_the_leading_filter_step(self):
        result = resolve_analysis(
            "diff",
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
            "diff",
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
            "diff",
            dataset=SAL_DATASET,
            filters=self.FILTERS,
            params={"dimension": "calendar_day", "order": "asc"},
        )
        query = pipeline_to_query(result.pipeline)
        assert "calendar_day IN ('2019-01-01'::date, '2070-12-31'::date)" in query["step.1.where"]

        dumped = result.pipeline.model_dump(mode="json")
        json.dumps(dumped)  # must not raise -- a bare `date` is not JSON-encodable
        assert "calendar_day IN ('2019-01-01'::date, '2070-12-31'::date)" in dumped["steps"][0]["where"]


# ------------------------------------------------------------------------------------------
# `resolve_analysis_chain`'s own exception contract
# ------------------------------------------------------------------------------------------


class TestResolveAnalysisChainContract:
    """Pure/in-memory coverage of the chain-specific error paths --
    `tests/api/test_analyses_registry.py`'s chain tests cover the successful,
    disk-touching end-to-end case (`lnd_conf,diff`, §3.7 of the plan).
    """

    def test_differencing_a_dimension_the_leading_analysis_already_subsumes_is_caught_upstream(
        self,
    ):
        """`lnd_conf` subsumes `scenario_id` -- but that's not what fires here.
        Its OWN `validate_selection` rejects any caller-supplied `scenario_id` filter
        outright (it fixes the whole ensemble itself), and that runs before the chain
        mechanism's own "is this dimension already collapsed" guard ever gets a chance to.
        The `match` pins that it's the fixed-ensemble message, not a generic chain-level one.
        """
        with pytest.raises(InvalidAnalysisParametersError, match="fixed ensemble"):
            resolve_analysis_chain(
                ["lnd_conf", "diff"],
                dataset=LND_TYPE_DATASET,
                filters={"model_group_id": 705, "calendar_year": 2050, "scenario_id": [34, 35]},
                params={"dimension": "scenario_id", "order": "asc"},
            )

    def test_an_unknown_later_stage_raises_unknown_analysis_error(self):
        with pytest.raises(UnknownAnalysisError):
            resolve_analysis_chain(
                ["lnd_conf", "not-a-real-analysis"],
                dataset=LND_TYPE_DATASET,
                filters={"model_group_id": 705, "calendar_year": 2050},
                params={},
            )

    def test_a_three_stage_chain_with_a_stage_2_vs_3_collision_is_rejected(self):
        """The pairwise collision guard has to check every pair, not just stage 1 vs. 2:
        here stage 1 (`lnd_conf`) collides with neither later stage (it
        declares no parameters of its own), but stage 2 and stage 3 -- both `diff`
        -- collide with EACH OTHER on `dimension`/`order`.
        """
        with pytest.raises(InvalidAnalysisParametersError, match="diff"):
            resolve_analysis_chain(
                ["lnd_conf", "diff", "diff"],
                dataset=LND_TYPE_DATASET,
                filters={"model_group_id": 705, "calendar_year": [2040, 2050]},
                params={"dimension": "calendar_year", "order": "asc"},
            )

    def test_a_single_name_chain_delegates_straight_to_resolve_analysis(self):
        chained = resolve_analysis_chain(
            ["lnd_conf"],
            dataset=LND_TYPE_DATASET,
            filters={"model_group_id": 705, "calendar_year": 2050},
            params={},
        )
        plain = resolve_analysis(
            "lnd_conf",
            dataset=LND_TYPE_DATASET,
            filters={"model_group_id": 705, "calendar_year": 2050},
            params={},
        )
        assert chained.pipeline == plain.pipeline
        assert chained.collapsed_dimensions == plain.collapsed_dimensions


class TestChainedLegendResolution:
    """Which legend a chain publishes -- `resolve_analysis_chain`'s stage legend, else none
    for a differenced value result, else the running one.

    This is presentation metadata, so nothing about the compiled pipeline changes with it
    and no response-level test can notice it going wrong: `lnd_conf,diff` would still return
    the same numbers under the wrong legend, colored as an undifferenced `[0, 1]` fraction.
    """

    CHAIN_FILTERS: ClassVar[dict] = {"model_group_id": 705, "calendar_year": [2040, 2050]}
    CHAIN_PARAMS: ClassVar[dict] = {"dimension": "calendar_year", "order": "asc"}

    def test_differencing_land_confidence_drops_its_legend_for_the_ramp(self):
        """`lnd_conf,diff` publishes no legend, so the map draws the synthesized
        red-white-blue ramp: `lnd_conf`'s legend classifies an undifferenced fraction on
        `[0, 1]` and would misread a signed change in it.
        """
        resolved = resolve_analysis_chain(
            ["lnd_conf", "diff"],
            dataset=LND_TYPE_DATASET,
            filters=self.CHAIN_FILTERS,
            params=self.CHAIN_PARAMS,
        )
        assert resolved.legend is None
        assert resolved.result_kind == "difference"
        # Undifferenced, the same analysis still publishes the fraction's own legend.
        assert (
            resolve_analysis("lnd_conf", dataset=LND_TYPE_DATASET, filters={"model_group_id": 705}, params={}).legend
            == "lnd_conf"
        )

    def test_a_leading_analysis_with_no_differenced_legend_keeps_its_own(self):
        """`lnd_type_plus` declares `legend="lnd_type_plus"` and a `"category"` result, so
        a difference of it keeps the legend it had. The alternative -- falling through to
        `None` because the result is now a difference -- would silently drop a legend an
        analysis did declare.
        """
        resolved = resolve_analysis_chain(
            ["lnd_type_plus", "diff"],
            dataset=None,
            filters=self.CHAIN_FILTERS,
            params=self.CHAIN_PARAMS,
        )
        assert resolved.legend == "lnd_type_plus"
        assert resolved.result_kind == "difference"

    def test_a_plain_difference_still_publishes_no_legend(self):
        """`diff` alone over an ordinary dataset: no leading analysis means no legend, so
        `visualizations.map` draws the synthesized diverging ramp."""
        resolved = resolve_analysis(
            "diff",
            dataset=SSF_DATASET,
            filters={"model_group_id": 705, "calendar_year": [2040, 2050]},
            params=self.CHAIN_PARAMS,
        )
        assert resolved.legend is None
        assert resolved.result_kind == "difference"


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
        assert selection_filter_where({"scenario_id": 7, "grid_cell": "4"}) == ("scenario_id = 7 AND grid_cell = 4")

    def test_selection_filter_where_rejects_a_coordinate_grid_cell(self):
        """Resolving coordinates to ids needs the catalog, which this package cannot
        reach -- the caller resolves them and passes `grid_cell_where` instead.
        """
        from cpra.mp.data.analysis.request.errors import RequestError

        with pytest.raises(RequestError, match="digit IDs"):
            selection_filter_where({"grid_cell": "-90.1,29.5"})

    def test_selection_filter_where_accepts_the_reserved_keys(self):
        assert selection_filter_where({"grid_cell": "1", "where": "value > 3"}) == ("grid_cell = 1 AND (value > 3)")

    def test_selection_filter_where_rejects_a_dataset_typed_dimension(self):
        # `variable`/`time_unit`/`grid` select the dataset; they are not filters.
        with pytest.raises(InvalidAnalysisParametersError, match="variable"):
            selection_filter_where({"variable": "ssf"})

    def test_dataset_output_format_follows_the_grid_kind(self):
        """An analysis that reads the caller's dataset and never changes shape ends band-wide on a raster
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

    def test_default_collapsed_dimensions_sees_past_a_trailing_selection_filter(self):
        """A caller's dimension/`grid_cell`/`where` selection is appended as a trailing
        `filter`. A filter narrows rows but collapses nothing, so it must not hide the
        `group` that did -- otherwise every selected request reports nothing collapsed and
        `visualizations.base` builds a `select` naming a column the read cannot find.
        """
        analysis = Difference(
            dataset=SSF_DATASET,
            filters={"calendar_year": [2030, 2040], "model_group_id": 500, "scenario_id": 7, "percentile": 50},
            parameters=Difference.Parameters(dimension="calendar_year", order="asc"),
        )
        pipeline = analysis.pipeline()
        expected = default_collapsed_dimensions(pipeline)
        assert expected  # the fixture's own pipeline really does end in a group

        pipeline.steps.append(FilterStep(where="model_group_id = 500"))
        assert default_collapsed_dimensions(pipeline) == expected

        pipeline.steps.append(FilterStep(where="grid_cell = 5"))
        assert default_collapsed_dimensions(pipeline) == expected

    def test_default_collapsed_dimensions_empty_for_a_stepless_pipeline(self):
        """The `download` sentinel's shape -- one input, no ops. `steps[-1]` would raise."""
        pipeline = Difference(
            dataset=SSF_DATASET,
            filters={"calendar_year": [2030, 2040], "model_group_id": 500, "scenario_id": 7, "percentile": 50},
            parameters=Difference.Parameters(dimension="calendar_year", order="asc"),
        ).pipeline()
        pipeline.steps = []
        assert default_collapsed_dimensions(pipeline) == frozenset()


# ------------------------------------------------------------------------------------------
# `diff`
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
        assert [(arm.when, arm.then) for arm in case.cases] == [("calendar_year = 2040", "value * -1")]
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
        """`diff` declares `Meta.variables = None`, so it genuinely runs on both grid
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
        assert analysis.pipeline().steps[0].where == ("calendar_year IN (2030, 2040) AND grid_cell IN (11, 12)")

    def test_a_callers_own_where_is_split_and_parenthesized_conjunct_by_conjunct(self):
        _, pipeline = self._build("asc", filters={"calendar_year": [2030, 2040], "where": "value > 1 AND value < 9"})
        # Split FIRST, then parenthesized: one paren pair per top-level conjunct, so the
        # AND seam the engine classifies each predicate's kind against survives intact.
        assert pipeline.steps[0].where == ("calendar_year IN (2030, 2040) AND (value > 1) AND (value < 9)")

    def test_an_unrecognized_filter_name_is_rejected(self):
        analysis = Difference(
            dataset=SSF_DATASET,
            filters={"calendar_year": [2030, 2040], "nonsense": 1},
            parameters=Difference.Parameters(dimension="calendar_year", order="asc"),
        )
        with pytest.raises(InvalidAnalysisParametersError, match="nonsense"):
            analysis.pipeline()


# ------------------------------------------------------------------------------------------
# `lnd_chng`
# ------------------------------------------------------------------------------------------


class TestLandChangeParameters:
    def test_declares_no_parameters_of_its_own(self):
        LandChange.Parameters()  # no raise

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            LandChange.Parameters(dimension="calendar_year")


class TestLandChangeCardinality:
    params = LandChange.Parameters()

    def test_accepts_a_missing_calendar_year(self):
        """ "Every year against the derived baseline" -- the series a chart asks for, which
        is how the portal sends one (`getSpecUrl` sets `includeTimeDimension` false as soon
        as the request names a grid cell, so no `calendar_year` reaches the API at all).
        """
        LandChange.validate_selection(self.params, {"model_group_id": 500})  # no raise

    def test_accepts_several_calendar_years(self):
        LandChange.validate_selection(self.params, {"calendar_year": [2030, 2040], "model_group_id": 500})  # no raise

    def test_rejects_missing_model_group_id(self):
        with pytest.raises(InvalidAnalysisParametersError):
            LandChange.validate_selection(self.params, {"calendar_year": 2040})

    def test_accepts_one_calendar_year_with_one_model_group(self):
        LandChange.validate_selection(self.params, {"calendar_year": 2040, "model_group_id": 500})  # no raise

    def test_accepts_one_calendar_year_with_two_model_groups(self):
        LandChange.validate_selection(self.params, {"calendar_year": 2040, "model_group_id": [500, 520]})  # no raise

    def test_rejects_three_model_groups(self):
        with pytest.raises(InvalidAnalysisParametersError):
            LandChange.validate_selection(self.params, {"calendar_year": 2040, "model_group_id": [500, 510, 520]})


class TestLandChangeBaselineYear:
    """The grid-version business rule: a `_v001` grid is MP2023 data compared against
    `MP2023_BASELINE_YEAR`, a `_v002` grid is MP2029 data compared against
    `MP2029_BASELINE_YEAR`, and `model_group_id` has no say in it.
    """

    def _leading_filter(self, model_group_id, dataset=SSF_DATASET):
        """The BASELINE branch's own filter, which is `steps[0]` -- the derived year is the
        only year on it. The comparison branch's filter (a later step) carries the caller's
        own year selection and never the derived one.
        """
        analysis = LandChange(
            dataset=dataset,
            filters={
                "calendar_year": 2040,
                "model_group_id": model_group_id,
                "scenario_id": 7,
                "percentile": 50,
            },
            parameters=LandChange.Parameters(),
        )
        leading = analysis.pipeline().steps[0]
        assert isinstance(leading, FilterStep)
        assert "2040" not in leading.where  # the comparison year is the other branch's
        return leading

    def test_a_v001_grid_uses_the_mp2023_baseline(self):
        assert SSF_DATASET.grid.endswith("_v001")
        leading = self._leading_filter(500)
        assert f"calendar_year = {MP2023_BASELINE_YEAR}" in leading.where

    def test_a_v002_grid_uses_the_mp2029_baseline(self):
        dataset = DatasetRef(variable="ssf", time_unit="annual", grid="ecoregion_v002")
        leading = self._leading_filter(500, dataset=dataset)
        assert f"calendar_year = {MP2029_BASELINE_YEAR}" in leading.where

    def test_model_group_does_not_move_the_baseline(self):
        for model_group_id in (0, 500, 705, 900, [650, 900], [0, 650]):
            leading = self._leading_filter(model_group_id)
            assert f"calendar_year = {MP2023_BASELINE_YEAR}" in leading.where

    def test_an_unversioned_grid_is_rejected(self):
        with pytest.raises(InvalidAnalysisParametersError, match="parish_v2020"):
            baseline_year("parish_v2020")

    def test_baseline_year_is_asking_the_same_question_the_pipeline_does(self):
        """The helper exists so a caller can know which year a selection will read without
        building an `Analysis`; if it ever stopped agreeing with `pipeline()`, it would be
        worse than not having it.
        """
        leading = self._leading_filter(500)
        assert f"calendar_year = {baseline_year(SSF_DATASET.grid)}" in leading.where


class TestLandChangePipeline:
    def _build(self, filters, dataset=SSF_DATASET):
        analysis = LandChange(
            dataset=dataset,
            filters=filters,
            parameters=LandChange.Parameters(),
        )
        return analysis, analysis.pipeline()

    def test_two_image_mode(self):
        analysis, pipeline = self._build(
            {
                "calendar_year": [2040, 2050],
                "model_group_id": 705,
                "scenario_id": 7,
                "percentile": 50,
            }
        )
        baseline, group, comparison, join, calculate = pipeline.steps

        # The baseline branch: the derived year alone, read from its own input.
        assert isinstance(baseline, FilterStep)
        assert baseline.input == "base"
        assert baseline.where == (
            f"calendar_year = {MP2023_BASELINE_YEAR} AND model_group_id = 705 AND percentile = 50 AND scenario_id = 7"
        )
        # THE BROADCAST. `calendar_year` is missing from `by`, so the group drops the
        # column, and `join_long`/`_pair_band_groups` then have nothing to match it
        # against -- one baseline value per grid cell against every comparison year. The
        # `sum` itself is an identity here: one year, one model group, one row per cell.
        assert isinstance(group, GroupStep)
        assert sorted(group.by) == sorted(["model_group_id", "scenario_id", "percentile"])
        assert "calendar_year" not in group.by

        # The comparison branch: every selected year, unreduced, straight into the join.
        assert isinstance(comparison, FilterStep)
        assert comparison.input == "years"
        assert comparison.where == (
            "calendar_year IN (2040, 2050) AND model_group_id = 705 AND percentile = 50 AND scenario_id = 7"
        )
        assert comparison.as_ == "comparison"

        assert isinstance(join, JoinStep)
        assert join.inputs == ["baseline", "comparison"]
        # A join operand's name is its value column's name, so the tens digit is written
        # over the operands directly -- and a `calculate` leaves exactly one value column,
        # which is what the sink rule requires.
        assert isinstance(calculate, CalculateStep)
        assert calculate.expression == "baseline * 10 + comparison"

        # Nothing. The result carries one code per (grid cell, selected year).
        assert analysis.collapsed_dimensions == frozenset()

    def test_two_image_mode_with_exactly_one_model_group_selected(self):
        # model_group_id present but pinned to a single value: still two-image.
        _, pipeline = self._build({"calendar_year": 2040, "model_group_id": [705], "scenario_id": 7, "percentile": 50})
        assert len(pipeline.steps) == 5
        assert pipeline.steps[-1].expression == "baseline * 10 + comparison"

    def test_an_unselected_calendar_year_means_every_year(self):
        """The shape a chart request arrives in: no `calendar_year` at all. The comparison
        branch then writes no year conjunct and reads the lot, while the baseline branch
        still pins the derived year -- which is the whole series against one baseline.
        """
        _, pipeline = self._build({"model_group_id": 705, "scenario_id": 7, "percentile": 50})
        baseline, _, comparison, _, _ = pipeline.steps
        assert f"calendar_year = {MP2023_BASELINE_YEAR}" in baseline.where
        assert "calendar_year" not in comparison.where

    def test_three_image_mode(self):
        analysis, pipeline = self._build(
            {"calendar_year": 2040, "model_group_id": [720, 705], "scenario_id": 7, "percentile": 50}
        )
        baseline, baseline_group, comparison, case, group, join, calculate = pipeline.steps

        # The 2x2 (model group x year) grid's fourth corner -- `mg1` in the baseline year --
        # is redundant, and here it is simply never read: the baseline branch is `mg0`'s.
        # It used to be excluded by a `NOT (...)` conjunct, which is the same thing said
        # negatively because there was one shared read to say it on.
        assert isinstance(baseline, FilterStep)
        assert baseline.where == (
            f"calendar_year = {MP2023_BASELINE_YEAR} AND model_group_id = 705 AND percentile = 50 AND scenario_id = 7"
        )
        assert "NOT" not in baseline.where
        # The baseline drops `model_group_id` as well as `calendar_year`: the comparison
        # branch folds its two model groups away too, and the two operands have to agree
        # about which columns they still carry for the join to key on them.
        assert sorted(baseline_group.by) == sorted(["scenario_id", "percentile"])

        # Dimension NAMES are sorted (so the string doesn't depend on dict order), but
        # each dimension's own VALUES keep the caller's order -- `[720, 705]` in, `(720,
        # 705)` out, matching `request.data_query`'s own conjunct assembly.
        assert isinstance(comparison, FilterStep)
        assert comparison.where == (
            "calendar_year = 2040 AND model_group_id IN (720, 705) AND percentile = 50 AND scenario_id = 7"
        )

        # The tens and ones digits, packed on the comparison branch alone so the baseline
        # stays free to broadcast: the lower model group's land type is multiplied by 10
        # and the higher one's passes through, and the `group` sums the pair per year.
        assert isinstance(case, CaseStep)
        assert [(arm.when, arm.then) for arm in case.cases] == [("model_group_id = 705", "value * 10")]
        # The ones digit: the model group no arm names keeps its value through the `sum`.
        assert case.else_ == "value"

        assert isinstance(group, GroupStep)
        assert "model_group_id" not in group.by
        assert "calendar_year" in group.by  # unlike the baseline's -- the year survives here

        assert isinstance(join, JoinStep)
        assert join.inputs == ["baseline", "comparison"]
        assert isinstance(calculate, CalculateStep)
        assert calculate.expression == "baseline * 100 + comparison"

        # `model_group_id` only. `calendar_year` still varies the result.
        assert analysis.collapsed_dimensions == frozenset({"model_group_id"})

    def test_a_raster_selection_takes_exactly_one_comparison_year(self):
        """One image, one comparison year. Band-wide, the join pairs one baseline band
        against one comparison band, and N comparison bands land in the same pairing group
        (`calendar_year` is null on the baseline, so it is not in the pairing key) and
        MISS -- which drops silently. Said out loud instead.
        """
        with pytest.raises(InvalidAnalysisParametersError, match="exactly 1 calendar_year"):
            self._build(
                {"calendar_year": [2040, 2050], "model_group_id": 705, "scenario_id": 7},
                dataset=LND_TYPE_DATASET,
            )
        with pytest.raises(InvalidAnalysisParametersError, match="all of them"):
            self._build({"model_group_id": 705, "scenario_id": 7}, dataset=LND_TYPE_DATASET)

    def test_a_grid_cell_selection_lowers_a_raster_and_lifts_that_rule(self):
        """A spatial conjunct is the one thing that lowers band-wide to long
        (`ops.FILTER_LOWERING_TABLE`), and a per-grid-cell chart request always carries
        one -- so the series is reachable on the raster grid `lnd_type` actually lives on,
        not only on a vector dataset.
        """
        _, pipeline = self._build(
            {"calendar_year": [2040, 2050], "model_group_id": 705, "scenario_id": 7, "grid_cell": "3"},
            dataset=LND_TYPE_DATASET,
        )
        assert "calendar_year IN (2040, 2050)" in pipeline.steps[2].where
        # Both branches carry the spatial conjunct: the baseline is narrowed to the same
        # cells it will be broadcast over.
        assert all("grid_cell" in step.where for step in pipeline.steps[:3] if hasattr(step, "where"))

    def test_meta_and_legend(self):
        analysis, _ = self._build({"calendar_year": 2040, "model_group_id": 705})
        assert analysis.legend == "lnd_chng"
        assert analysis.result_kind == "category"

    def test_the_output_format_follows_the_selected_dataset(self):
        """`lnd_chng` neither projects nor changes shape, so its final frame's kind is
        the input grid's -- and `pipeline.json`, the only route that does not overwrite
        this value from a URL segment, has to publish a pipeline that would still validate
        on resubmission. `Meta.variables` is `lnd_type`, which lives only on a raster grid,
        so the real-world answer is `tif`.
        """
        filters = {"calendar_year": 2040, "model_group_id": 705}
        _, vector = self._build({**filters, "scenario_id": 7, "percentile": 50})
        assert vector.output.format == "csv"
        _, raster = self._build({**filters, "scenario_id": 7}, dataset=LND_TYPE_DATASET)
        assert raster.output.format == "tif"

    def test_resolves_through_resolve_analysis(self):
        resolved = resolve_analysis(
            "lnd_chng",
            dataset=SSF_DATASET,
            filters={"calendar_year": 2040, "model_group_id": 705, "scenario_id": 7, "percentile": 50},
            params={},
        )
        assert isinstance(resolved, ResolvedAnalysis)
        assert resolved.legend == "lnd_chng"
        assert resolved.result_kind == "category"
        assert resolved.collapsed_dimensions == frozenset()

    def test_missing_model_group_id_is_rejected_by_resolve_analysis(self):
        with pytest.raises(InvalidAnalysisParametersError):
            resolve_analysis("lnd_chng", dataset=SSF_DATASET, filters={"calendar_year": 2040}, params={})


# ------------------------------------------------------------------------------------------
# `sal_veg_investigation`
# ------------------------------------------------------------------------------------------


# What a caller pins in practice, and what used to be two required `Parameters` fields.
SELECTION = {"model_group_id": 500, "scenario_id": 7}


class TestSalinityVegInvestigationParameters:
    """The two thresholds are the whole of it. `model_group_id`/`scenario_id` used to be
    declared here too, the pair of them required -- they are ordinary FILTERS now, and
    `analyses._register` rejects a dimension-named `Parameters` field outright, so they
    could not be declared here again even by accident.
    """

    def test_the_field_set_is_exactly_the_two_thresholds(self):
        assert set(SalinityVegInvestigation.Parameters.model_fields) == {
            "marsh_threshold",
            "salinity_threshold",
        }

    def test_defaults(self):
        params = SalinityVegInvestigation.Parameters()
        assert params.marsh_threshold == 0.5
        assert params.salinity_threshold == 5.5

    def test_marsh_threshold_bounds(self):
        with pytest.raises(ValidationError):
            SalinityVegInvestigation.Parameters(marsh_threshold=1.5)
        with pytest.raises(ValidationError):
            SalinityVegInvestigation.Parameters(marsh_threshold=-0.1)

    def test_salinity_threshold_lower_bound(self):
        with pytest.raises(ValidationError):
            SalinityVegInvestigation.Parameters(salinity_threshold=-1)

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            SalinityVegInvestigation.Parameters(bogus=1)


class TestSalinityVegInvestigationPipeline:
    def _build(self, filters=SELECTION, **overrides):
        analysis = SalinityVegInvestigation(
            dataset=None,
            filters=dict(filters),
            parameters=SalinityVegInvestigation.Parameters(**overrides),
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

        (
            coverage_filter,
            salinity_filter,
            project_step,
            join_step,
            selection_filter,
            case_step,
            group_step,
        ) = pipeline.steps

        # The per-input `where`s are DEFINITIONAL only -- the one reading of each input this
        # investigation is about. The caller's `model_group_id`/`scenario_id` used to be
        # spelled into both of these as well; it is one conjunct past the join now
        # (`selection_filter` below), which `execution.plan` pushes back down to both reads.
        assert isinstance(coverage_filter, FilterStep)
        assert coverage_filter.input == "coverage"
        assert coverage_filter.where == "ffibs_type = 'pl_fm'"
        assert coverage_filter.as_ == "marsh"

        assert isinstance(salinity_filter, FilterStep)
        assert salinity_filter.input == "salinity"
        assert salinity_filter.where == "aggregate_type = '2_wk_max'"
        assert salinity_filter.as_ == "selected_salinity"

        assert isinstance(project_step, ProjectStep)
        assert project_step.input == "selected_salinity"
        assert project_step.to == "veg_grid_cell_v001"
        assert project_step.as_ == "sal"

        assert join_step.op == "join"
        assert join_step.type == "spatial"
        assert set(join_step.inputs) == {"marsh", "sal"}

        # The caller's selection, once, on the joined frame -- before the `case`, which is
        # what collapses the pairing a dimension conjunct still needs standing.
        assert isinstance(selection_filter, FilterStep)
        assert selection_filter.where == "model_group_id = 500 AND scenario_id = 7"

        # The threshold test collapses the pairing and says what survives it, which a
        # `filter` has no way to state.
        assert isinstance(case_step, CaseStep)
        assert [(arm.when, arm.then) for arm in case_step.cases] == [("marsh > 0.5 AND sal > 5.5", "marsh")]
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
        case_step = next(step for step in pipeline.steps if step.op == "case")
        assert case_step.cases[0].when == "marsh > 0.75 AND sal > 3.0"

    def test_an_unselected_request_states_no_selection_step_at_all(self):
        """Legal now, and the reason the step is conditional: with the dimension values no
        longer required `Parameters`, `selection_filter_where({})` is `None`, and a `filter`
        with no `where` is not a step. The `case` chains from the join directly.
        """
        _, pipeline = self._build(filters={})
        assert [step.op for step in pipeline.steps] == [
            "filter",
            "filter",
            "project",
            "join",
            "case",
            "group",
        ]

    def test_a_grid_cell_reaches_the_same_conjunct(self):
        """The blanket `validate_selection` used to reject this outright. It composes into
        the trailing conjunct like any other selection now; what makes it an error is the
        raster `output.format`, rejected by `validate_shapes` with a message naming
        `grid_cell` -- asserted at the decoder, in `test_unified_decoder.py`.
        """
        analysis = SalinityVegInvestigation(
            dataset=None,
            filters={"grid_cell": "1,2"},
            parameters=SalinityVegInvestigation.Parameters(),
        )
        selection = next(step for step in analysis.pipeline().steps if step.op == "filter" and step.input is None)
        assert selection.where == "grid_cell IN (1, 2)"

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
# `lnd_conf`
# ------------------------------------------------------------------------------------------


class TestLandConfidenceParameters:
    def test_declares_no_parameters_of_its_own(self):
        LandConfidence.Parameters()  # no raise

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            LandConfidence.Parameters(dimension="calendar_year")


class TestLandConfidenceSelection:
    params = LandConfidence.Parameters()

    def test_accepts_filters_that_do_not_name_scenario_id(self):
        LandConfidence.validate_selection(self.params, {"model_group_id": 705, "calendar_year": 2050})  # no raise

    def test_rejects_a_caller_selected_scenario_id(self):
        with pytest.raises(InvalidAnalysisParametersError, match="fixed ensemble"):
            LandConfidence.validate_selection(self.params, {"scenario_id": 34})

    def test_rejects_a_scenario_id_selecting_the_whole_ensemble_too(self):
        # Even the exact member set is rejected: the analysis supplies it itself, and
        # accepting it here would mean accepting a SUBSET next.
        with pytest.raises(InvalidAnalysisParametersError):
            LandConfidence.validate_selection(self.params, {"scenario_id": sorted(SCENARIO_WEIGHTS)})


class TestLandConfidenceWeights:
    def test_the_weights_sum_to_one(self):
        # What makes the trailing `group`/`sum` a weighted MEAN rather than an
        # unnormalized total -- a result outside 0..1 would not be a confidence.
        assert sum(SCENARIO_WEIGHTS.values()) == pytest.approx(1.0)

    def test_every_land_type_code_is_classified_exactly_once(self):
        assert set(LAND_VALUES).isdisjoint(WATER_VALUES)
        assert set(LAND_VALUES) | set(WATER_VALUES) == {1, 2, 3, 4, 5}


class TestLandConfidencePipeline:
    def _build(self, filters=None, dataset=LND_TYPE_DATASET):
        analysis = LandConfidence(
            dataset=dataset,
            filters={"model_group_id": 705, "calendar_year": 2050} if filters is None else filters,
            parameters=LandConfidence.Parameters(),
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
            ("value = 1 OR value = 3 OR value = 4 OR value = 5", "1.0"),
            ("value = 2", "0.0"),
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
        assert analysis.legend == "lnd_conf"
        assert analysis.result_kind == "value"
        assert analysis.title_suffix == " Confidence"

    def test_resolves_through_resolve_analysis(self):
        resolved = resolve_analysis(
            "lnd_conf",
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
                "lnd_conf",
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
        assert query["step.3.cases.2.when"] == "value = 2"
        assert query["step.4.cases.1.when"] == "scenario_id = 34"
        assert query["step.4.cases.1.then"] == "value * 0.33"


# ------------------------------------------------------------------------------------------
# `lnd_type_plus`
# ------------------------------------------------------------------------------------------


class TestLndTypePlusParameters:
    """It declares NO parameters. `model_group_id`/`scenario_id`/`calendar_year` were
    declared here, the first two required, back when a fixed-input analysis had nowhere to
    put a dimension filter; they are ordinary filters now, composed into the same trailing
    `filter` step every other selection goes to.

    There is no `TestLndTypePlusSelection` any more either. `validate_selection` rejected
    any `filters` at all with "takes no selection", which rejected the dimension filters
    that were every real use of it. What the blanket rejection was approximating -- that a
    `grid_cell` lowers this analysis's frame to long and cannot then be written to its
    raster sink -- is `validate_shapes`'s judgement, made per format and with a message
    naming `grid_cell`; see `test_a_grid_cell_is_a_conjunct_like_any_other` below and the
    decoder's own `test_a_raster_point_selection_cannot_be_written_as_a_surface`.
    """

    def test_it_declares_no_parameters_at_all(self):
        assert LndTypePlus.Parameters.model_fields == {}

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            LndTypePlus.Parameters(dimension="calendar_year")


class TestLndTypePlusPipeline:
    # The selection a caller writes, which used to be three `Parameters` fields.
    FILTERS: ClassVar[dict] = {"model_group_id": 705, "scenario_id": 7, "calendar_year": 2050}

    def _build(self, **overrides):
        filters = {key: value for key, value in {**self.FILTERS, **overrides}.items() if value is not None}
        analysis = LndTypePlus(dataset=None, filters=filters, parameters=LndTypePlus.Parameters())
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
        # The stage guard FIRST -- past the last hydro compartment there is no stage to
        # broadcast, and the unbounded arm below would read that absence as shallow water --
        # then the depth classes descending. The shallowest has no lower bound, so its arm
        # is the bare water conjunct with no depth test: every open-water pixel the two
        # bounded arms leave is `6`, whatever its depth says.
        assert isinstance(reclassify, CaseStep)
        assert [(arm.when, arm.then) for arm in reclassify.cases] == [
            ("stage_m IS NULL", "NULL"),
            ("land_type = 2 AND stage_m - elevation / 1000 > 1.0", "2"),
            ("land_type = 2 AND stage_m - elevation / 1000 > 0.46", "7"),
            ("land_type = 2", "6"),
        ]
        # The land type, unchanged, for everything no arm claims -- which is now exactly
        # the pixels that have a stage and are not open water, and is the whole reason
        # nothing follows this step.
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
        `DEPTH_CLASSES` names, and only a pixel that is not open water falls through to the
        land type.

        Re-implemented here as a plain first-match-wins interpreter over the pipeline's
        own `when` strings rather than against a second copy of the class table, so an
        edit to the arms that gets an interval's open/closed end wrong fails this test.
        The `else` is interpreted too, since with no reduce after the `case` it is the only
        thing standing between an unclaimed pixel and nodata.

        Every depth arm also carries the water conjunct, which is interpreted here too -- so
        a land type that stops being tested, or stops being tested for water, fails below
        rather than being quietly dropped from the interpreter. The leading STAGE GUARD is
        interpreted as well, and `None` stands for the null it writes.
        """
        _, pipeline = self._build()
        case_step = pipeline.steps[-1]
        guard, *arms = [(arm.when, arm.then) for arm in case_step.cases]
        assert case_step.else_ == "land_type"
        # The guard is an arm like any other, and being FIRST is the whole of how it works:
        # every arm below it would claim a stageless water pixel for a depth class.
        assert guard == ("stage_m IS NULL", "NULL")

        def run(depth, land_type=WATER_LAND_TYPE, stage=True):
            if not stage:
                return None  # the guard, which writes null -- i.e. nodata
            for when, then in arms:
                water, _, depth_test = when.partition(" AND ")
                assert water == "land_type = 2", f"unexpected water conjunct in {when!r}"
                if land_type != WATER_LAND_TYPE:
                    continue
                # The shallowest arm has no depth test -- `partition` leaves `depth_test`
                # empty -- and claims the pixel on the water conjunct alone.
                if not depth_test:
                    return float(then)
                if _conjunct_holds(depth, depth_test, measured="stage_m - elevation / 1000"):
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
        # The shallowest class has NO lower bound. Zero is the water level standing exactly
        # at the ground and a negative is the ground above it, but the LAND TYPE is what
        # says this pixel is water, and open water however shallow is `6` -- never `2`,
        # which would be the positive claim that it is deeper than a metre.
        assert run(0) == 6
        assert run(-3.2) == 6
        # ... and no depth is a class on a pixel that is not open water, however deeply the
        # water surface stands above its ground: the land type passes straight through.
        assert run(12.5, land_type=1) == 1
        assert run(0.47, land_type=5) == 5
        # With no stage the guard claims the pixel ahead of all of them, whatever the land
        # type and whatever a depth would have been -- nothing was evaluated, so nothing is
        # reported. Without it, the first of these would read `6` and the second `4`.
        assert run(0.20, stage=False) is None
        assert run(0.20, land_type=4, stage=False) is None

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
        resolved = resolve_analysis("lnd_type_plus", dataset=None, filters=dict(self.FILTERS), params={})
        assert isinstance(resolved, ResolvedAnalysis)
        assert resolved.pipeline.steps[-1].else_ == "land_type"
        assert resolved.collapsed_dimensions == frozenset({"aggregate_type"})

    def test_string_filter_values_off_a_query_string_are_coerced(self):
        # Every value arrives as a string on the real route (the decoder reads the raw
        # query string), so `format_dimension_literal` has to cast before it renders --
        # otherwise the conjunct would read `model_group_id = '705'` against an int column.
        resolved = resolve_analysis(
            "lnd_type_plus",
            dataset=None,
            filters={"model_group_id": "705", "scenario_id": "7", "calendar_year": "2050"},
            params={},
        )
        assert resolved.pipeline.steps[-2].where == (
            "calendar_year = 2050 AND model_group_id = 705 AND scenario_id = 7"
        )

    def test_an_entirely_unselected_request_states_no_selection_step(self):
        """Legal now: `selection_filter_where({})` is `None`, and a `filter` with no `where`
        is not a step, so the `case` chains from the join directly. Nothing replaces the
        `required` this drops -- what bounds the read is `config.ANALYSIS_BAND_CEILING`
        against the join's operand union.
        """
        analysis = LndTypePlus(dataset=None, filters={}, parameters=LndTypePlus.Parameters())
        assert [step.op for step in analysis.pipeline().steps] == [
            "filter",
            "project",
            "join",
            "case",
        ]

    def test_a_grid_cell_is_a_conjunct_like_any_other(self):
        """`validate_selection` rejected this outright. It composes into the same trailing
        conjunct now; what makes it wrong is the analysis's raster `output.format`, which
        `validate_shapes` rejects because a spatial filter lowers the frame to long -- a
        judgement about the SINK, not about the filter, and one this method never makes.
        """
        analysis = LndTypePlus(dataset=None, filters={"grid_cell": "3,5"}, parameters=LndTypePlus.Parameters())
        assert analysis.pipeline().steps[-2].where == "grid_cell IN (3, 5)"


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


# ------------------------------------------------------------------------------------------
# `Meta.subsumed_dimensions` drift: portal-facing metadata vs. the runtime truth
# ------------------------------------------------------------------------------------------


class TestSubsumedDimensionsAreASubsetOfWhatsActuallyCollapsed:
    """`Meta.subsumed_dimensions` is what `/v1/analyses.json` publishes for the portal to
    hide/not-send (`dimensions-select.tsx`, §4.4 of the plan); `collapsed_dimensions` is
    what a given resolved selection actually collapses. The declared set must never OVERSTATE
    the runtime one -- a dimension the portal hides as "already handled" that still shows up
    in a real response would silently mislead the caller. It MAY be a strict subset
    (`lnd_chng`'s three-image mode additionally subsumes `model_group_id`, which `Meta`
    does not declare, since only that mode does), so the check is subset, not equality.

    Each test is mutation-checked: asserting the same subset relationship against a
    dimension the selection does NOT actually collapse must fail, so a `Meta` that
    overstates its own declaration would be caught rather than passed vacuously.
    """

    def _assert_subset_with_teeth(self, declared, collapsed, an_uncollapsed_dimension):
        assert an_uncollapsed_dimension not in collapsed  # test setup sanity
        assert set(declared) <= collapsed
        with pytest.raises(AssertionError):
            assert set(declared) | {an_uncollapsed_dimension} <= collapsed

    def test_land_type_confidence(self):
        resolved = resolve_analysis(
            "lnd_conf",
            dataset=LND_TYPE_DATASET,
            filters={"model_group_id": 705, "calendar_year": 2050},
            params={},
        )
        self._assert_subset_with_teeth(
            LandConfidence.Meta.subsumed_dimensions,
            resolved.collapsed_dimensions,
            "calendar_year",
        )

    def test_land_change_two_image_mode(self):
        resolved = resolve_analysis(
            "lnd_chng",
            dataset=SSF_DATASET,
            filters={"calendar_year": 2040, "model_group_id": 705, "scenario_id": 7, "percentile": 50},
            params={},
        )
        self._assert_subset_with_teeth(
            LandChange.Meta.subsumed_dimensions, resolved.collapsed_dimensions, "scenario_id"
        )

    def test_land_change_three_image_mode_still_holds_even_though_it_subsumes_more(self):
        resolved = resolve_analysis(
            "lnd_chng",
            dataset=SSF_DATASET,
            filters={
                "calendar_year": 2040,
                "model_group_id": [720, 705],
                "scenario_id": 7,
                "percentile": 50,
            },
            params={},
        )
        # The extra thing three-image mode subsumes beyond what `Meta` declares -- proof
        # this mode is the strict-subset case the class docstring calls out, not equality.
        # `calendar_year` is NOT in it: every mode compares each selected year against the
        # derived baseline and reports it, so the year axis survives both of them.
        assert resolved.collapsed_dimensions == frozenset({"model_group_id"})
        assert set(LandChange.Meta.subsumed_dimensions) < resolved.collapsed_dimensions
        self._assert_subset_with_teeth(
            LandChange.Meta.subsumed_dimensions, resolved.collapsed_dimensions, "scenario_id"
        )

    def test_lnd_type_plus(self):
        resolved = resolve_analysis(
            "lnd_type_plus",
            dataset=None,
            filters={"model_group_id": 705, "scenario_id": 7, "calendar_year": 2050},
            params={},
        )
        self._assert_subset_with_teeth(
            LndTypePlus.Meta.subsumed_dimensions, resolved.collapsed_dimensions, "calendar_year"
        )
