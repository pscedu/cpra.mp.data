"""Tests for the static catalog endpoints, plus the /catalog.json dataset-catalog route."""

import numpy as np
import pytest
from pydantic import ValidationError

from cpra.mp.data import config
from cpra.mp.data.analyses.base import AnalysisMeta
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.legends import LEGENDS
from cpra.mp.data.raster import write_cog
from cpra.mp.data.variables import VARIABLES


class TestVariables:
    def test_returns_every_variable(self, client):
        response = client.get("/v1/variables.json")
        assert response.status_code == 200
        assert {v["name"] for v in response.json()} == set(VARIABLES)

    def test_dtype_is_serialised_as_a_string(self, client):
        by_name = {v["name"]: v for v in client.get("/v1/variables.json").json()}
        assert by_name["sal"]["dtype"] == "Float32"

    def test_legend_is_serialised_by_name(self, client):
        by_name = {v["name"]: v for v in client.get("/v1/variables.json").json()}
        assert by_name["sal"]["legend"] == "sal"

    def test_list_fields_are_lists(self, client):
        by_name = {v["name"]: v for v in client.get("/v1/variables.json").json()}
        assert by_name["sal"]["time_units"] == ["daily", "monthly", "annual"]
        assert by_name["sal"]["geographic_units"] == ["hydro_compartment"]

    def test_float_precision_is_reported(self, client):
        """Reported rather than acted on: the API has already applied it to the CSV it
        serves, so this only tells a client what it is about to receive."""
        by_name = {v["name"]: v for v in client.get("/v1/variables.json").json()}
        assert by_name["sal"]["float_precision"] == 1
        assert by_name["ssf"]["float_precision"] is None


class TestDimensions:
    def test_returns_every_dimension(self, client):
        response = client.get("/v1/dimensions.json")
        assert response.status_code == 200
        assert {d["name"] for d in response.json()} == set(DIMENSIONS)

    def test_csv_backed_values_are_used_when_present(self, client):
        by_name = {d["name"]: d for d in client.get("/v1/dimensions.json").json()}
        values = {v["value"] for v in by_name["time_unit"]["values"]}
        assert values == {"daily", "monthly", "annual", "decadal"}

    def test_database_backed_values_fill_the_gaps(self, client):
        """model_group_id has no values in the CSVs, so they come from the MPD."""
        by_name = {d["name"]: d for d in client.get("/v1/dimensions.json").json()}
        values = {v["value"] for v in by_name["model_group_id"]["values"]}
        assert values == {500, 520}

    def test_database_labels_are_returned(self, client):
        by_name = {d["name"]: d for d in client.get("/v1/dimensions.json").json()}
        labels = {v["value"]: v["label"] for v in by_name["model_group_id"]["values"]}
        assert labels[500] == "Future Without Action"

    def test_expression_backed_required_flags_become_false(self, client):
        """`required` is a Polars expression for conditional dimensions; only a literal
        True is reported as required."""
        by_name = {d["name"]: d for d in client.get("/v1/dimensions.json").json()}
        assert by_name["variable"]["required"] is True
        assert by_name["percentile"]["required"] is False

    def test_min_and_max_are_reported(self, client):
        by_name = {d["name"]: d for d in client.get("/v1/dimensions.json").json()}
        assert by_name["calendar_year"]["min_value"] == 1900
        assert by_name["calendar_year"]["max_value"] == 2100

    def test_colored_values_carry_their_color(self, client):
        """The species colors are the vegetation model's own symbology, and a client needs
        them to draw a species the same way the map does."""
        by_name = {d["name"]: d for d in client.get("/v1/dimensions.json").json()}
        colors = {v["value"]: v["color"] for v in by_name["veg_species_code"]["values"]}
        assert colors["spal"] == "#ff0000"
        assert None not in colors.values()

    def test_colored_values_are_listed_in_display_order(self, client):
        """The order is the payload here, not an accident of serialization: it is what a
        client stacks and lists a legend by, so it has to survive the round trip."""
        by_name = {d["name"]: d for d in client.get("/v1/dimensions.json").json()}
        values = [v["value"] for v in by_name["veg_species_code"]["values"]]
        assert values == DIMENSIONS["veg_species_code"].values["value"].to_list()
        assert (values[0], values[-1]) == ("notmod", "unpa")

    def test_uncolored_values_report_a_null_color(self, client):
        """Every dimension carries the field now that one dimension needs it, so the others
        answer it rather than omitting it."""
        by_name = {d["name"]: d for d in client.get("/v1/dimensions.json").json()}
        assert {v["color"] for v in by_name["time_unit"]["values"]} == {None}

    def test_database_backed_values_report_a_null_color_too(self, client):
        """These come from the MPD rather than the CSV, so they are built by a different path
        and could plausibly have skipped the field."""
        by_name = {d["name"]: d for d in client.get("/v1/dimensions.json").json()}
        assert {v["color"] for v in by_name["model_group_id"]["values"]} == {None}

    def test_float_values_are_canonical(self, client):
        """aep values must serialise as 0.1, not the Float32 repr 0.10000000149011612."""
        by_name = {d["name"]: d for d in client.get("/v1/dimensions.json").json()}
        assert 0.1 in {v["value"] for v in by_name["aep"]["values"]}


class TestGrids:
    def test_returns_every_grid(self, client):
        response = client.get("/v1/grids.json")
        assert response.status_code == 200
        assert {g["name"] for g in response.json()} == set(GRIDS)

    def test_vector_grids_report_id_bounds_and_no_geometry(self, client):
        by_name = {g["name"]: g for g in client.get("/v1/grids.json").json()}
        grid = by_name["ecoregion_v001"]
        assert grid["grid_type"] == "vector"
        assert (grid["id_min"], grid["id_max"], grid["id_count"]) == (1, 32, 32)
        assert grid["bounds"] is None

    def test_raster_grids_report_geometry_and_no_id_bounds(self, client):
        by_name = {g["name"]: g for g in client.get("/v1/grids.json").json()}
        grid = by_name["veg_grid_cell_v001"]
        assert grid["grid_type"] == "raster"
        assert grid["height"] == 365
        assert grid["width"] == 1052
        assert grid["id_count"] is None


class TestLegends:
    def test_returns_every_legend(self, client):
        response = client.get("/v1/legends.json")
        assert response.status_code == 200
        assert {legend["name"] for legend in response.json()} == set(LEGENDS)

    def test_break_points_are_serialised(self, client):
        by_name = {legend["name"]: legend for legend in client.get("/v1/legends.json").json()}
        points = by_name["accr"]["break_points"]
        assert points[0]["upper_bound"] == 2.0
        assert points[0]["color"].startswith("#")
        assert points[-1]["upper_bound"] is None


class TestVisualizations:
    def test_returns_visualization_metadata(self, client):
        response = client.get("/v1/visualizations.json")
        assert response.status_code == 200
        names = {v["name"] for v in response.json()}
        assert {
            "time-series-line",
            "stacked-bar",
            "species-stacked-bar",
            "map",
            "sediment-dynamics",
        } <= names

    def test_each_entry_declares_a_type(self, client):
        for viz in client.get("/v1/visualizations.json").json():
            assert viz["type"] in ("dataset", "custom")

    def test_only_the_map_declares_no_time_axis(self, client):
        """What a client needs to not offer a chart for an analysis that collapses the time
        column (`DatasetVisualization.validate_time_axis` is the server-side half). Only
        `map.py` states it; every other entry takes the model's `True` default, so this
        pins that the default actually reaches the response rather than being dropped as
        unset."""
        by_name = {v["name"]: v["time_axis"] for v in client.get("/v1/visualizations.json").json()}
        assert by_name["map"] is False
        assert {name for name, axis in by_name.items() if axis} == set(by_name) - {"map"}

    def test_only_sediment_dynamics_declares_an_implied_analysis(self, client):
        """The other half of what a client needs to not offer a broken combination. A chart
        declaring an analysis puts it at the head of the chain, so a caller's own
        `kind="variable"` analysis would land in a non-leading stage and
        `resolve_analysis_chain` 400s. The portal reads this to drop the chart from the
        picker instead; asserting the null for every other entry is what keeps that
        predicate from silently matching nothing.
        """
        by_name = {v["name"]: v["analysis"] for v in client.get("/v1/visualizations.json").json()}
        assert by_name["sediment-dynamics"] == "sediment_dynamics"
        assert {name for name, analysis in by_name.items() if analysis is not None} == {"sediment-dynamics"}


class _FakeCatalogAnalysis:
    # A real `AnalysisMeta`, not a stand-in class: the route serves `Meta.model_dump()`,
    # so a fake that only duck-types the attribute reads would not exercise it. Building
    # it here also keeps this fake honest about the required fields -- `kind` cannot be
    # left off a fake any more than it can be left off a real entry.
    Meta = AnalysisMeta(
        name="lnd_chng",
        label="Land Change",
        variables=["ssf"],
        legend=None,
        result_kind="value",
        kind="variable",
        subsumed_dimensions=["scenario_id"],
    )


class TestAnalyses:
    """`/v1/analyses.json`, built from `analysis.Meta.model_dump()` per registered class.
    Not quite the mirror of `TestVisualizations` above it any more: those `Meta`s are still
    bare classes served through `__dict__`, while every analysis `Meta` is a validated
    `AnalysisMeta` instance.

    This monkeypatches a small fake registry onto both places `api/__init__.py`/
    `api/utils.py` bind the name, exercising the route itself rather than the real
    analyses' content -- see `tests/api/test_analyses_endpoints.py` for the full
    route-surface coverage. The fake analysis class is module-level rather than nested
    here: a nested class body has no closure over its enclosing class body's namespace
    (unlike a nested function).
    """

    @pytest.fixture(autouse=True)
    def _fake_registry(self, monkeypatch):
        import cpra.mp.data.api as api_module
        import cpra.mp.data.api.utils as api_utils

        fake = {"lnd_chng": _FakeCatalogAnalysis}
        monkeypatch.setattr(api_utils, "ANALYSES", fake)
        monkeypatch.setattr(api_module, "ANALYSES", fake)

    def test_returns_analysis_metadata(self, client):
        response = client.get("/v1/analyses.json")
        assert response.status_code == 200
        assert {a["name"] for a in response.json()} == {"lnd_chng"}

    def test_no_entry_carries_a_type(self, client):
        """`AnalysisMeta` dropped `type` with the route split that read it, and
        `api.models.Analysis` dropped it too -- so the field is absent from the wire rather
        than serving as `null`. `kind` is the surviving partition, asserted below."""
        for analysis in client.get("/v1/analyses.json").json():
            assert "type" not in analysis
            assert analysis["kind"] in ("variable", "dimension", "internal")

    def test_kind_and_subsumed_dimensions_are_published(self, client):
        """The portal's dimension picker (`dimensions-select.tsx`, plan §4.4) hides/strips
        a dimension a selected analysis already subsumes, so it needs both fields off the
        wire -- not just present on `Meta` server-side.
        """
        by_name = {a["name"]: a for a in client.get("/v1/analyses.json").json()}
        assert by_name["lnd_chng"]["kind"] == "variable"
        assert by_name["lnd_chng"]["subsumed_dimensions"] == ["scenario_id"]

    def test_subsumed_dimensions_defaults_through_the_round_trip(self, client, monkeypatch):
        """`diff` declares no `subsumed_dimensions` -- it removes nothing from the caller's
        view that `collapsed_dimensions` doesn't already report per-request -- so
        `AnalysisMeta`'s own `[]` default has to survive the dump and this route's response
        model rather than raising or omitting the field. Re-patches over the class-level
        `_fake_registry` fixture's fake for this one test.

        `kind` gets no such test because it no longer HAS a default: see
        `test_a_meta_cannot_omit_kind` below.
        """
        import cpra.mp.data.api as api_module
        import cpra.mp.data.api.utils as api_utils

        class _BareAnalysis:
            Meta = AnalysisMeta(
                name="diff",
                label="Difference",
                kind="dimension",
                variables=None,
                legend=None,
                result_kind="difference",
            )

        fake = {"diff": _BareAnalysis}
        monkeypatch.setattr(api_utils, "ANALYSES", fake)
        monkeypatch.setattr(api_module, "ANALYSES", fake)

        by_name = {a["name"]: a for a in client.get("/v1/analyses.json").json()}
        assert by_name["diff"]["kind"] == "dimension"
        assert by_name["diff"]["subsumed_dimensions"] == []

    def test_a_meta_cannot_omit_kind(self):
        """The failure this route used to serve as `kind: null`, moved to import time.

        `sal_veg_investigation` really did ship without a `kind`, and nothing caught it
        here: the omission dumped to an absent key, `api.models.Analysis` filled in its
        `None` default, and the portal could not tell that from a deliberate declaration.
        `AnalysisMeta` has no default to fall back on, so the same omission is now a
        `ValidationError` raised by the module that declares it.
        """
        with pytest.raises(ValidationError):
            AnalysisMeta(name="x", label="X")


class TestDatasetsIndex:
    @pytest.fixture(autouse=True)
    def data(self, base_path, monkeypatch):
        from cpra.mp.data import config
        from cpra.mp.data.tabular import write_parquet
        from tests.factories import annual_frame

        write_parquet(annual_frame(), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

    def test_lists_available_datasets_as_csv(self, client):
        response = client.get("/v1/datasets.csv")
        assert response.status_code == 200
        body = response.text
        assert "variable,time_unit,grid" in body
        assert "ssf,annual,ecoregion_v001" in body

    def test_path_column_is_excluded(self, client):
        assert "path" not in client.get("/v1/datasets.csv").text

    @pytest.mark.parametrize("format", ["csv", "ndjson", "parquet"])
    def test_supported_formats(self, client, format):
        assert client.get(f"/v1/datasets.{format}").status_code == 200

    def test_rejects_a_raster_only_compression_algorithm(self, client):
        response = client.get("/v1/datasets.csv?compression_algorithm=lzw")
        assert response.status_code == 400
        assert "must be one of" in response.json()["detail"]

    def test_rejects_an_unknown_format(self, client):
        assert client.get("/v1/datasets.xlsx").status_code == 422


class TestCatalogEndpoint:
    """/v1/catalog.json -- what a request can still be steered by, and what it would
    produce.

    It takes the same parameter set as the four data routes and resolves the same pipeline,
    which is the whole point of the change: the route it replaces was keyed on a
    {variable}/{time_unit}/{grid} path triple, so an analysis that fixes its own inputs had
    no way to ask what it was allowed to select, and the portal covered for it by inventing
    a triple per derived variable and keeping it in agreement by hand.

    Two halves, tested here through the route and in tests/api/test_catalog_query.py as
    units: choices/results/exact (CatalogQuery over the pipeline's inputs, strictly
    intersected, minus what the analysis collapsed -- see that file's TestSeveralInputs for
    the intersection and its 404), and `output` (the output descriptor with the analysis's
    presentation overrides folded in).
    """

    URL = "/v1/catalog.json?variable=vct&time_unit=annual&grid=veg_grid_cell_v001"

    @pytest.fixture(autouse=True)
    def _catalog_path(self, catalog_path, monkeypatch):
        """The route builds CatalogQuery without a catalog_path argument, so it falls back
        to config.CATALOG_PATH -- point that at the isolated fixture path."""
        monkeypatch.setattr(config, "CATALOG_PATH", catalog_path)
        return catalog_path

    def _write(self, base_path, catalog_path, **overrides):
        grid = GRIDS["veg_grid_cell_v001"]
        kwargs = {
            "variable": "vct",
            "grid": "veg_grid_cell_v001",
            "time_unit": "annual",
            "model_group_id": 500,
            "scenario_id": 7,
            "model": "icm_v2023.0.0",
            "calendar_year": 2030,
            **overrides,
        }
        write_cog(
            np.zeros((grid.height, grid.width), dtype="float32"),
            base_path=base_path,
            catalog_path=catalog_path,
            **kwargs,
        )

    def test_returns_choices_results_exact_and_output(self, client, base_path, _catalog_path):
        self._write(base_path, _catalog_path)
        response = client.get(self.URL)
        assert response.status_code == 200
        assert set(response.json()) == {"choices", "results", "exact", "output"}

    def test_reports_available_scenarios(self, client, base_path, _catalog_path):
        self._write(base_path, _catalog_path, scenario_id=7)
        self._write(base_path, _catalog_path, scenario_id=8)
        body = client.get(self.URL).json()
        assert body["choices"]["scenario_id"] == [7, 8]

    def test_selection_is_reflected_in_results(self, client, base_path, _catalog_path):
        """Also the cast: `scenario_id=8` arrives as the string "8" and the catalog's
        column is Int32, so without `api.catalog.catalog_filters` the selection would miss
        and fall back to the first choice -- which is 7 here, not 8, so the assertion
        distinguishes the two."""
        self._write(base_path, _catalog_path, scenario_id=7)
        self._write(base_path, _catalog_path, scenario_id=8)
        body = client.get(f"{self.URL}&scenario_id=8").json()
        assert body["results"]["scenario_id"] == [8]

    def test_unavailable_selection_is_inexact(self, client, base_path, _catalog_path):
        self._write(base_path, _catalog_path)
        body = client.get(f"{self.URL}&scenario_id=99").json()
        assert body["exact"] is False

    def test_the_dataset_triple_is_not_echoed_back(self, client, base_path, _catalog_path):
        """It was a restatement of the caller's own path segments, it has no single answer
        once a request reads several datasets, and what a client needs from it is in
        `output` instead."""
        self._write(base_path, _catalog_path)
        body = client.get(self.URL).json()
        assert not {"variable", "time_unit", "grid"} & set(body["choices"])
        assert not {"variable", "time_unit", "grid"} & set(body["results"])

    def test_output_describes_the_result_of_a_plain_read(self, client, base_path, _catalog_path):
        self._write(base_path, _catalog_path)
        output = client.get(self.URL).json()["output"]
        assert output["grid"] == "veg_grid_cell_v001"
        assert output["grid_type"] == "raster"
        assert output["time_unit"] == "annual"
        assert output["time_column"] == "calendar_year"
        assert output["variables"] == ["vct"]
        assert output["legend"] == "vct"
        assert output["result_kind"] == "value"
        assert output["collapsed_dimensions"] == []
        # Null, which is what says `value_columns` holds columns of the result.
        assert output["folded_into"] is None

    def test_output_of_an_unpivoted_pipeline_reports_the_folded_names(self, client, base_path, _catalog_path):
        """The other reading of `value_columns`: after an `unpivot` the result carries one
        `value` column and the names are the VALUES of the label column instead. Reported
        that way, with `folded_into` naming the column, so a client can key display
        metadata per measurement without caring which reading it got -- see
        `api.models.CatalogOutput`."""
        self._write(base_path, _catalog_path)
        self._write(base_path, _catalog_path, variable="ffibs_score")
        query = (
            "input.vct.variable=vct&input.vct.time_unit=annual&input.vct.grid=veg_grid_cell_v001&"
            "input.ffibs_score.variable=ffibs_score&"
            "input.ffibs_score.time_unit=annual&"
            "input.ffibs_score.grid=veg_grid_cell_v001&"
            "step.1.op=join&step.1.type=spatial&step.1.inputs=vct,ffibs_score&step.1.as=j&"
            "step.2.op=filter&step.2.input=j&step.2.where=grid_cell%20%3D%205&step.2.as=pt&"
            "step.3.op=unpivot&step.3.input=pt&step.3.as=stacked&"
            "output.format=csv"
        )
        output = client.get(f"/v1/catalog.json?{query}").json()["output"]
        assert output["folded_into"] == "variable"
        assert output["value_columns"] == ["ffibs_score", "vct"]
        assert output["variables"] == ["ffibs_score", "vct"]

    def test_an_analysis_overrides_what_the_result_is(self, client, base_path, _catalog_path):
        """`lnd_chng` reads the caller's dataset but outputs land-change categories, not
        vegetation types, so its own legend/label/kind win over the input variable's."""
        self._write(base_path, _catalog_path)
        output = client.get(f"{self.URL}&analysis=lnd_chng&calendar_year=2030&model_group_id=500").json()["output"]
        assert output["legend"] == "lnd_chng"
        assert output["label"] == "Land Change"
        assert output["result_kind"] == "category"

    def test_a_collapsed_dimension_is_absent_rather_than_defaulted(self, client, base_path, _catalog_path):
        """`lnd_chng`'s three-image mode packs its two selected `model_group_id` values into
        one code, so the model group is not a choice the caller still has. Absent, not
        reported with a value they cannot change -- and `exact` must not be dragged down by
        defaulting it.

        `calendar_year` is the control here, and it used to be the subject: `lnd_chng` once
        fused its two years into one code and collapsed the axis. It compares each selected
        year against a derived baseline now and reports every one of them, so the year is
        a live choice and has to come back as one.
        """
        self._write(base_path, _catalog_path)
        query = f"{self.URL}&analysis=lnd_chng&calendar_year=2030&model_group_id=500&model_group_id=520"
        body = client.get(query).json()
        assert body["output"]["collapsed_dimensions"] == ["model_group_id"]
        assert "model_group_id" not in body["choices"]
        assert "model_group_id" not in body["results"]
        assert "calendar_year" in body["results"]

    def test_missing_catalog_entries_is_404(self, client, _catalog_path):
        """No data has ever been written for this variable/time_unit/grid -- CatalogQuery's
        _load_catalog finds no catalog files under catalog_path and raises ValueError,
        which the route translates into a 404."""
        response = client.get(self.URL)
        assert response.status_code == 404
        assert "No catalog entries" in response.json()["detail"]

    def test_a_compression_parameter_is_rejected(self, client, base_path, _catalog_path):
        """A JSON metadata response carries no compression, so asking for it is a 400
        rather than a silently ignored parameter."""
        self._write(base_path, _catalog_path)
        assert client.get(f"{self.URL}&compression_algorithm=gzip").status_code == 400

    def test_a_render_option_is_rejected(self, client, base_path, _catalog_path):
        """Render options belong to the visualization route. Nothing here draws anything."""
        self._write(base_path, _catalog_path)
        assert client.get(f"{self.URL}&height=400").status_code == 400
