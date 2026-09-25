"""Tests for the visualization-layer migration to named analyses.

Covers what `tests/api/test_analyses_registry.py` and `tests/test_visualizations.py` don't:

* **The `feature_colors` bug fix** (`TestFeatureColorsBugFix`) -- `Map.feature_colors` used
  to silently return `{}` for EVERY caller (a `cs.matches(r"^[0-9]")` selector that assumed
  a wide, one-column-per-grid-cell shape against data stored long). Proven here by asserting
  a non-empty result on both the resolved-analysis branch and the no-analysis-selected
  branch (dimension filters alone, no `analysis=`).
* **Pure presentation-hook coverage** (`TestResolvedAnalysisMetadataHooks`,
  `TestCompareDimensionsExclusion`) -- `_is_difference`/`_legend_name`/`difference_suffix`/
  `compare_dimensions`/`chart()`'s compare channels, on the resolved-analysis path and the
  no-analysis-selected path. The legacy fallback path this class also used to cover (the
  `lnd_type`+`multiply10` "lnd_chng" special case, and `StackedBar`'s `negate_dimension`
  sniffing) was later deleted outright (see `visualizations/map.py`'s `_is_difference`/
  `_legend_name` and `visualizations/stacked_bar.py`'s `compare_dimensions`, which now only
  ever consult `resolved_analysis`) -- there is nothing left of it to test.
  None of these need any data on disk: `resolve_analysis` only builds a `Pipeline` in
  memory, and a visualization never reads its own data (the API does, from the URL/spec it
  builds) -- see `BaseVisualization.get_data_uri`.

`SSF_DATASET`/`COMMON_ANNUAL_FILTERS`/the local fixture-writer convention mirror
`tests/api/test_analyses_registry.py` (a concurrent module this one doesn't import from, per
that module's own documented "don't couple to a module you don't own" reasoning) --
`ssf`/`ecoregion_v001` is used as the representative dataset for the same reason: the
migrated call sites care about grid TYPE and pipeline SHAPE, not real-world variable
semantics.
"""

import pytest

from cpra.mp.data import config
from cpra.mp.data.analyses import DatasetRef, resolve_analysis
from cpra.mp.data.visualizations.map import Map
from cpra.mp.data.visualizations.stacked_bar import StackedBar
from cpra.mp.data.visualizations.time_series_line import TimeSeriesLine
from tests.factories import ANNUAL_KWARGS, write_annual_dataset

SSF_DATASET = DatasetRef(
    variable=ANNUAL_KWARGS["variable"], time_unit=ANNUAL_KWARGS["time_unit"], grid=ANNUAL_KWARGS["grid"]
)
COMMON_ANNUAL_FILTERS = {
    "model_group_id": ANNUAL_KWARGS["model_group_id"],
    "scenario_id": ANNUAL_KWARGS["scenario_id"],
    "percentile": ANNUAL_KWARGS["percentile"],
}
# The same filters, but as the single-element LISTS a `Map(**kwargs)`/`FilterDimensions`
# construction needs (unlike `resolve_analysis`'s `filters=`, which accepts bare scalars).
COMMON_ANNUAL_FILTER_LISTS = {k: [v] for k, v in COMMON_ANNUAL_FILTERS.items()}


# ------------------------------------------------------------------------------------------
# feature_colors: the wide/long-format bug fix
# ------------------------------------------------------------------------------------------


class TestFeatureColorsBugFix:
    """`Map.feature_colors` used to select columns via `cs.matches(r"^[0-9]")`, a wide-
    format assumption that matches nothing against the long (one-row-per-grid_cell) shape
    both the legacy and resolved-analysis code paths actually materialize -- silently
    returning `{}` for every caller. These tests fail on `{}` (the old, broken behavior).
    """

    def test_resolved_analysis_branch_returns_real_colors(self, base_path, monkeypatch):
        write_annual_dataset(base_path, years=(2030, 2040), value_fn=lambda year, geo_id: 100.0 + geo_id)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        m = Map(
            variable=ANNUAL_KWARGS["variable"],
            grid=ANNUAL_KWARGS["grid"],
            time_unit=ANNUAL_KWARGS["time_unit"],
            analysis="difference",
            dimension="calendar_year",
            order="asc",
            calendar_year=[2030, 2040],
            **COMMON_ANNUAL_FILTER_LISTS,
        )
        colors = m.feature_colors
        assert colors != {}
        assert all(color.startswith("#") for color in colors.values())

    def test_no_analysis_branch_also_returns_real_colors(self, base_path, monkeypatch):
        """The bug affected the no-`analysis=`-selected path exactly as much as the
        resolved one -- both branches used to share the same broken `cs.matches`
        selector. Reaching that branch takes nothing but leaving `analysis=` unset, so
        the ordinary dimension filters below are the whole setup."""
        write_annual_dataset(base_path, years=(2030, 2040), value_fn=lambda year, geo_id: 100.0 + geo_id)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        m = Map(
            variable=ANNUAL_KWARGS["variable"],
            grid=ANNUAL_KWARGS["grid"],
            time_unit=ANNUAL_KWARGS["time_unit"],
            calendar_year=[2030, 2040],
            **COMMON_ANNUAL_FILTER_LISTS,
        )
        colors = m.feature_colors
        assert colors != {}
        assert all(color.startswith("#") for color in colors.values())

    def test_a_uniform_difference_result_is_a_single_color(self, base_path, monkeypatch):
        """Same setup as `TestNamedAnalysisResults.test_difference_asc`: every cell's
        difference is 0.0, so every cell should classify into the same color -- an
        absolute check on `feature_colors` itself, replacing what used to be a dict-
        equality comparison against the now-deleted legacy `negate` request that produced
        the identical 0.0-everywhere result the old-fashioned way.
        """
        write_annual_dataset(base_path, years=(2030, 2040), value_fn=lambda year, geo_id: 100.0 + geo_id)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        resolved_map = Map(
            variable=ANNUAL_KWARGS["variable"],
            grid=ANNUAL_KWARGS["grid"],
            time_unit=ANNUAL_KWARGS["time_unit"],
            analysis="difference",
            dimension="calendar_year",
            order="asc",
            calendar_year=[2030, 2040],
            **COMMON_ANNUAL_FILTER_LISTS,
        )
        colors = resolved_map.feature_colors
        assert colors != {}
        assert len(set(colors.values())) == 1

    def test_non_vector_grid_still_returns_empty(self, base_path, monkeypatch):
        """The early `if self.grid.grid_type != "vector": return {}` guard is unrelated to
        the bug and must still short-circuit before any analysis is resolved or compiled."""
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        m = Map(variable="vct", grid="veg_grid_cell_v001", time_unit="annual")
        assert m.feature_colors == {}


# ------------------------------------------------------------------------------------------
# Pure presentation-hook coverage (no data on disk needed)
# ------------------------------------------------------------------------------------------


class TestResolvedAnalysisMetadataHooks:
    """`_is_difference`/`_legend_name`/`difference_suffix`, resolved-analysis path only.
    The legacy-fallback sniffing these hooks used to fall back to
    (`self.params.get("negate"/"multiply10")`) was later deleted outright, not merely
    hidden from HTTP --
    there is no legacy branch left to test, so the equivalent legacy-fallback tests that
    used to live here (`test_legacy_negate_is_a_difference_with_no_special_legend`,
    `test_legacy_lnd_type_multiply10_uses_the_lnd_chng_legend`,
    `test_legacy_multiply10_on_another_variable_does_not_use_the_lnd_chng_legend`) were
    removed rather than converted. `resolve_analysis` only builds a `Pipeline` object in
    memory (it never reads a dataset file), so none of this needs
    `base_path`/`monkeypatch`/written data.
    """

    ANNUAL = dict(variable="ssf", grid="ecoregion_v001", time_unit="annual", **COMMON_ANNUAL_FILTER_LISTS)

    def test_difference_result_kind_is_reported_as_a_difference(self):
        m = Map(analysis="difference", dimension="calendar_year", order="asc", calendar_year=[2030, 2040], **self.ANNUAL)
        assert m._is_difference() is True
        assert m._legend_name() is None
        assert m.difference_suffix == " Difference"

    def test_land_change_result_kind_is_not_a_difference_and_uses_the_lnd_chng_legend(self):
        m = Map(analysis="land-change", calendar_year=[2040], **self.ANNUAL)
        assert m._is_difference() is False
        assert m._legend_name() == "lnd_chng"
        assert m.difference_suffix == ""

    def test_no_analysis_and_no_negate_is_not_a_difference(self):
        m = Map(**self.ANNUAL)
        assert m._is_difference() is False
        assert m._legend_name() is None
        assert m.difference_suffix == ""


class TestCompareDimensionsExclusion:
    """`StackedBar.compare_dimensions`/`TimeSeriesLine.chart()`'s own local equivalent must
    exclude whatever the resolved analysis's own `collapsed_dimensions` says it already
    collapsed -- not just some dimension happening to be absent from the selection.
    `model_group_id` here is deliberately given 2 values and NOT collapsed by either
    analysis, so it stays a real compare axis; only `calendar_year` (and, for the
    three-image case, `model_group_id` too) should ever be excluded.
    `StackedBar.negate_dimension` was later deleted outright (the no-`analysis=`-selected
    fallback exclusion it powered no longer exists), so the legacy-fallback test that used
    to sit alongside
    these (`test_legacy_negate_dimension_sniffing_is_unaffected`) was removed rather than
    converted -- there is no property left on `StackedBar` for it to call.
    """

    def test_difference_excludes_only_its_own_collapsed_dimension(self):
        sb = StackedBar(
            variable="ssf",
            grid="ecoregion_v001",
            time_unit="annual",
            grid_cell="1,2",
            analysis="difference",
            dimension="calendar_year",
            order="asc",
            calendar_year=[2030, 2040],
            model_group_id=[500, 520],
            scenario_id=[7],
            percentile=[50],
        )
        assert "calendar_year" not in sb.compare_dimensions
        assert sb.compare_dimensions == {"model_group_id": [500, 520]}

    def test_three_image_land_change_excludes_both_of_its_collapsed_dimensions(self):
        sb = StackedBar(
            variable="ssf",
            grid="ecoregion_v001",
            time_unit="annual",
            grid_cell="1,2",
            analysis="land-change",
            calendar_year=[2040],
            model_group_id=[500, 520],
            scenario_id=[7],
            percentile=[50],
        )
        assert sb.resolved_analysis.collapsed_dimensions == frozenset({"calendar_year", "model_group_id"})
        assert sb.compare_dimensions == {}

    def test_time_series_line_chart_omits_the_collapsed_dimension_as_a_compare_channel(self):
        tsl = TimeSeriesLine(
            variable="ssf",
            grid="ecoregion_v001",
            time_unit="annual",
            grid_cell="1",
            analysis="difference",
            dimension="calendar_year",
            order="asc",
            calendar_year=[2030, 2040],
            model_group_id=[500, 520],
            scenario_id=[7],
            percentile=[50],
        )
        spec = tsl.chart().to_dict()
        # A difference result (`resolved.result_kind == "difference"`) prepends a
        # zero-reference-rule layer with its own single `y` encoding ahead of the main
        # line/points/rules layers -- so the compare-channel encodings this test cares
        # about aren't necessarily on `layer[0]` anymore; scan every layer's encoding
        # rather than assuming which index the main line lands on.
        fields = [
            enc.get("field")
            for layer in spec["layer"]
            for enc in layer["encoding"].values()
            if isinstance(enc, dict)
        ]
        assert "calendar_year" not in fields
        assert "model_group_id" in fields
