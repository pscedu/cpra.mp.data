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

from typing import ClassVar

import pytest

from cpra.mp.data import config
from cpra.mp.data.analyses import DatasetRef
from cpra.mp.data.legends import LEGENDS
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
            analysis="diff",
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
            analysis="diff",
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

    ANNUAL: ClassVar[dict] = dict(variable="ssf", grid="ecoregion_v001", time_unit="annual", **COMMON_ANNUAL_FILTER_LISTS)

    def test_difference_result_kind_is_reported_as_a_difference(self):
        m = Map(analysis="diff", dimension="calendar_year", order="asc", calendar_year=[2030, 2040], **self.ANNUAL)
        assert m._is_difference() is True
        assert m._legend_name() is None
        assert m.difference_suffix == " Difference"

    def test_land_change_result_kind_is_not_a_difference_and_uses_the_lnd_chng_legend(self):
        m = Map(analysis="lnd_chng", calendar_year=[2040], **self.ANNUAL)
        assert m._is_difference() is False
        assert m._legend_name() == "lnd_chng"
        assert m.difference_suffix == ""

    def test_no_analysis_and_no_negate_is_not_a_difference(self):
        m = Map(**self.ANNUAL)
        assert m._is_difference() is False
        assert m._legend_name() is None
        assert m.difference_suffix == ""


class TestDifferenceRampVsNamedLegend:
    """`_uses_difference_ramp` -- which of the map's two colorings a difference gets.

    A difference used to have exactly one: the synthesized diverging ramp built from the
    variable's `difference_legend_limit` (`_difference_breaks`). A chain whose difference
    still carries a legend (`lnd_type_plus,diff`) makes that a choice, and the choice is made in ONE place
    because four branches read it (`feature_colors`, the raster source, `_legend_fill_expr`,
    and the legend metadata) and a map whose tiles and whose key disagree is worse than
    either answer alone.

    `_is_difference` deliberately still answers `True` for the named-legend case -- the
    result IS a difference, and the title still says so -- so the two are asserted together
    here rather than one standing in for the other.
    """

    # `lnd_conf`'s real dataset. No `scenario_id`: the analysis fixes its own ensemble and
    # `validate_selection` rejects a caller-supplied one outright.
    RASTER: ClassVar[dict] = {"variable": "lnd_type", "grid": "morph_pixel_v002", "time_unit": "annual", "model_group_id": [705]}

    def _map(self, **kwargs):
        return Map(calendar_year=[2040, 2050], dimension="calendar_year", order="asc", **self.RASTER, **kwargs)

    def test_land_confidence_difference_uses_the_ramp(self):
        m = self._map(analysis="lnd_conf,diff")
        assert m._is_difference() is True
        assert m._legend_name() is None
        assert m._uses_difference_ramp() is True

    def test_a_difference_with_no_named_legend_still_uses_the_ramp(self):
        m = self._map(analysis="diff")
        assert m._is_difference() is True
        assert m._legend_name() is None
        assert m._uses_difference_ramp() is True

    def test_an_undifferenced_analysis_uses_neither(self):
        """`lnd_conf` alone: not a difference at all, so the ramp question never arises and
        the legend is the undifferenced one."""
        m = Map(analysis="lnd_conf", **self.RASTER, calendar_year=[2040])
        assert m._is_difference() is False
        assert m._legend_name() == "lnd_conf"
        assert m._uses_difference_ramp() is False

    def test_the_legend_key_matches_the_colors_the_tiles_come_back_in(self):
        """`lnd_conf,diff` is keyed by the same continuous red/white/blue ramp its tiles
        are colored by, titled from the analysis's `Meta.label` rather than the input
        variable's."""
        m = self._map(analysis="lnd_conf,diff")
        metadata = m._get_legend_metadata(m._uses_difference_ramp())
        assert metadata["type"] == "continuous"
        assert [step["value"] for step in metadata["steps"]] == [-1, 0, 1]
        assert metadata["title"] == "Land Confidence Difference"


class TestLegendSortOrder:
    """`legends.csv`'s `sort` column reorders the legend KEY without touching the colors."""

    def _metadata(self, analysis):
        return Map(
            analysis=analysis,
            variable="lnd_type",
            grid="morph_pixel_v002",
            time_unit="annual",
            model_group_id=[705],
            calendar_year=[2040],
        )._get_legend_metadata(False)

    def test_land_confidence_change_key_reads_loss_gain_water_land(self):
        metadata = self._metadata("lnd_conf_chng")
        assert metadata["title"] == "Land Confidence Change"
        # First-seen labels, as the portal renders them.
        labels = list(dict.fromkeys(step["label"] for step in metadata["steps"]))
        assert labels == [
            "No Data",
            "Land Loss, High Confidence",
            "Land Loss, Moderate Confidence",
            "Land Loss, Low Confidence",
            "Land Gain, Low Confidence",
            "Land Gain, Moderate Confidence",
            "Land Gain, High Confidence",
            "Stays Water, High Confidence",
            "Stays Water, Moderate Confidence",
            "Stays Water, Low Confidence",
            "Stays Land, Low Confidence",
            "Stays Land, Moderate Confidence",
            "Stays Land, High Confidence",
        ]

    def test_sorting_the_key_leaves_the_legend_itself_in_break_point_order(self):
        """Tile colorizing (`colorize_pixels`) and `/v1/legends.json` read
        `break_points` directly, and need it ascending."""
        bounds = LEGENDS["lnd_conf_chng"].break_points["upper_bound"].to_list()
        assert bounds[:-1] == sorted(bounds[:-1])

    def test_a_legend_without_sort_keeps_its_order(self):
        metadata = self._metadata("lnd_chng")
        expected = [row["label"] for row in LEGENDS["lnd_chng"].break_points.iter_rows(named=True)]
        assert [step["label"] for step in metadata["steps"]] == expected

    def test_land_confidence_change_key_is_laid_out_like_the_design(self):
        metadata = self._metadata("lnd_conf_chng")
        # Still a discrete legend with its full step list, for a client that ignores `layout`.
        assert metadata["type"] == "discrete"
        assert len(metadata["steps"]) == len(LEGENDS["lnd_conf_chng"].break_points)

        rows = metadata["layout"]["rows"]
        assert [(row["kind"], row["left"], row["right"]) for row in rows] == [
            ("swatches", "Land loss", "Land gain"),
            ("arrows", "High confidence", "High confidence"),
            ("swatches", "Stays water", "Stays land"),
        ]
        assert "steps" not in rows[1]
        levels = ["High", "Moderate", "Low"]
        assert [step["label"] for step in rows[0]["steps"]] == [
            *(f"Land Loss, {level} Confidence" for level in levels),
            *(f"Land Gain, {level} Confidence" for level in reversed(levels)),
        ]
        assert [step["label"] for step in rows[2]["steps"]] == [
            *(f"Stays Water, {level} Confidence" for level in levels),
            *(f"Stays Land, {level} Confidence" for level in reversed(levels)),
        ]
        # Each swatch is one of the key's own steps -- the color the tiles use for it.
        colors = {step["label"]: step["color"] for step in metadata["steps"]}
        for row in (rows[0], rows[2]):
            assert all(step["color"] == colors[step["label"]] for step in row["steps"])

    def test_a_legend_without_a_layout_has_no_layout_key(self):
        assert "layout" not in self._metadata("lnd_chng")


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
            analysis="diff",
            dimension="calendar_year",
            order="asc",
            calendar_year=[2030, 2040],
            model_group_id=[500, 520],
            scenario_id=[7],
            percentile=[50],
        )
        assert "calendar_year" not in sb.compare_dimensions
        assert sb.compare_dimensions == {"model_group_id": [500, 520]}

    def test_three_image_land_change_excludes_its_collapsed_dimension(self):
        """`model_group_id` alone: three-image mode packs the two model groups into one
        code, so there is nothing left to compare them across. `calendar_year` is not in
        it -- `lnd_chng` reports each selected year -- and it is excluded here for the
        ORDINARY reason instead, being the chart's own time axis rather than a compare
        channel (`test_calendar_year_is_never_a_compare_dimension` above).
        """
        sb = StackedBar(
            variable="ssf",
            grid="ecoregion_v001",
            time_unit="annual",
            grid_cell="1,2",
            analysis="lnd_chng",
            calendar_year=[2040],
            model_group_id=[500, 520],
            scenario_id=[7],
            percentile=[50],
        )
        assert sb.resolved_analysis.collapsed_dimensions == frozenset({"model_group_id"})
        assert sb.compare_dimensions == {}

    def test_time_series_line_chart_omits_the_collapsed_dimension_as_a_compare_channel(self):
        tsl = TimeSeriesLine(
            variable="ssf",
            grid="ecoregion_v001",
            time_unit="annual",
            grid_cell="1",
            analysis="diff",
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


# ------------------------------------------------------------------------------------------
# A fixed-input analysis on a dataset-scoped visualization
# ------------------------------------------------------------------------------------------


class TestFixedInputAnalysisVisualization:
    """An analysis that reads its own fixed inputs (`lnd_type_plus`) selected as `analysis=`
    on a `DatasetVisualization`.

    Such an analysis names its own inputs, so it must be resolved with NO dataset. Passing
    one through anyway is what a map of `lnd_type_plus` used to do, and it was rejected
    outright (today: `AnalysisDatasetMismatchError`, raised by the analysis's own
    `pipeline()` via `reject_dataset`), so the Variable dropdown could offer the derived
    variable but never render it.

    What used to be a fork is now a single decode. The `_analysis_is_custom`/
    `_analysis_filters`/`_analysis_params` trio these tests turned on existed to reproduce, on
    the visualization's side, the filters-or-parameters split the two families of data routes
    were making on theirs; both sides run `api.unified.decode_request` now, so the assertions
    below are against `decoded.partition` -- one object, shared with the route. And the split
    itself is gone in the direction that matters: a dimension value is ALWAYS a filter, here
    exactly as everywhere else, landing in the trailing selection `filter` step that
    `lnd_type_plus.validate_selection`'s blanket rejection used to make impossible.

    NOTHING is named here -- no `variable`, no `grid`, no `time_unit`, exactly as in the URL
    the portal sends. The three used to be constructor arguments and this class passed
    `variable=None` with a grid and a time unit anyway, because the visualization needed a
    render context and there was nowhere else to get one. There is now: `grid` and
    `time_unit` are read back off `output_descriptor`, which says `morph_pixel_v002`/`annual`
    (the grid the analysis PROJECTS onto, which none of its three inputs is on) and no single
    variable, because it folds three. No data on disk is needed -- resolving only builds a
    `Pipeline` in memory.
    """

    def _map(self, **overrides):
        kwargs = {
            "analysis": "lnd_type_plus",
            "model_group_id": 700,
            "scenario_id": 34,
            "calendar_year": 2025,
        }
        kwargs.update(overrides)
        return Map(**kwargs)

    def test_the_render_context_is_the_analysis_s_output_not_an_argument(self):
        """The half of this that used to be constructor arguments. The analysis reads
        `elev`/`lnd_type` on `morph_pixel_v002` and `stg` on `hydro_compartment_v002`, and
        projects onto `morph_pixel_v002`; no input is the answer, the output is."""
        m = self._map()
        assert m.variable is None
        assert m.grid.name == "morph_pixel_v002"
        assert m.time_unit == "annual"

    def test_a_fixed_input_analysis_resolves_instead_of_raising_a_dataset_mismatch(self):
        resolved = self._map().resolved_analysis
        assert resolved is not None
        assert resolved.meta.name == "lnd_type_plus"
        # Its own three fixed inputs, and no dataset of the visualization's.
        assert {i.variable for i in resolved.pipeline.inputs.values()} == {"stg", "elev", "lnd_type"}

    def test_it_names_no_dataset_so_its_urls_carry_no_triple(self):
        m = self._map()
        assert m.decoded.partition.dataset == {}
        assert m.names_dataset is False

    def test_dimension_values_stay_filters_and_reach_the_selection_step(self):
        """The reversal of the old behaviour: no `Parameters` anywhere, and the selection is
        a trailing filter against the analysis's OUTPUT frame rather than something the
        analysis had to declare a field for."""
        m = self._map()
        assert m.decoded.partition.params == {}
        assert m.decoded.partition.filters == {
            "model_group_id": ["700"],
            "scenario_id": ["34"],
            "calendar_year": ["2025"],
        }
        selection = m.decoded.pipeline.steps[-2]
        assert selection.op == "filter"
        assert selection.where == "calendar_year = 2025 AND model_group_id = 700 AND scenario_id = 34"

    def test_a_dataset_analysis_on_the_same_visualization_still_names_its_dataset(self):
        m = Map(
            variable=ANNUAL_KWARGS["variable"],
            grid=ANNUAL_KWARGS["grid"],
            time_unit=ANNUAL_KWARGS["time_unit"],
            analysis="diff",
            dimension="calendar_year",
            order="asc",
            calendar_year=[2030, 2040],
            **COMMON_ANNUAL_FILTER_LISTS,
        )
        assert m.names_dataset is True
        assert m.decoded.partition.dataset["variable"] == ANNUAL_KWARGS["variable"]
        assert m.decoded.partition.filters["calendar_year"] == ["2030", "2040"]
        # `dimension=`/`order=` are the analysis's own parameters; the dimension VALUES aren't.
        assert set(m.decoded.partition.params) == {"dimension", "order"}

    def test_the_tile_url_is_the_one_data_tile_route_with_no_triple(self):
        url = self._map()._tile_url_template("png")
        assert url.startswith("https://api.mpdp.coastal.la.gov/v1/data/{z}/{x}/{y}.png?")
        assert "analysis=lnd_type_plus" in url
        assert "model_group_id=700" in url
        assert "variable=" not in url

    def test_the_data_url_is_the_one_data_route_with_no_triple(self):
        m = self._map()
        uri = m.get_data_uri(None, "morph_pixel_v002", "annual", **m.params)
        assert uri.startswith("https://api.mpdp.coastal.la.gov/v1/data.csv?")
        assert "analysis=lnd_type_plus" in uri
        assert "variable=" not in uri

    def test_the_legend_and_title_come_from_the_analysis_not_a_stand_in_variable(self):
        m = self._map()
        assert m._legend_name() == "lnd_type_plus"
        assert m._get_legend_metadata()["title"] == "Land Type Plus"

    def test_an_unknown_analysis_name_is_left_for_the_registry_to_reject(self):
        from cpra.mp.data.analyses import UnknownAnalysisError

        m = self._map(analysis="no_such_analysis")
        with pytest.raises(UnknownAnalysisError):
            _ = m.resolved_analysis
