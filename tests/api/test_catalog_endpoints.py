"""Tests for the static catalog endpoints, plus the /catalog.json dataset-catalog route."""

import numpy as np
import pytest

from cpra.mp.data import config
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


class TestDatasetCatalogEndpoint:
    """/v1/datasets/{variable}/{time_unit}/{grid}/catalog.json, backed by CatalogQuery
    reading the materialized catalog side-table rather than walking the data tree.

    See tests/api/test_catalog_query.py for CatalogQuery's own unit tests -- including why
    there is no test here for the old "no common values" 404: that ValueError is
    unreachable in the current CatalogQuery (see that file's TestChoiceIntersection).
    """

    URL = "/v1/datasets/vct/annual/veg_grid_cell_v001/catalog.json"

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

    def test_returns_choices_results_and_exact(self, client, base_path, _catalog_path):
        self._write(base_path, _catalog_path)
        response = client.get(self.URL)
        assert response.status_code == 200
        assert set(response.json()) == {"choices", "results", "exact"}

    def test_reports_available_scenarios(self, client, base_path, _catalog_path):
        self._write(base_path, _catalog_path, scenario_id=7)
        self._write(base_path, _catalog_path, scenario_id=8)
        body = client.get(self.URL).json()
        assert body["choices"]["scenario_id"] == [7, 8]

    def test_selection_is_reflected_in_results(self, client, base_path, _catalog_path):
        self._write(base_path, _catalog_path, scenario_id=7)
        self._write(base_path, _catalog_path, scenario_id=8)
        body = client.get(f"{self.URL}?scenario_id=8").json()
        assert body["results"]["scenario_id"] == [8]

    def test_unavailable_selection_is_inexact(self, client, base_path, _catalog_path):
        self._write(base_path, _catalog_path)
        body = client.get(f"{self.URL}?scenario_id=99").json()
        assert body["exact"] is False

    def test_missing_catalog_entries_is_404(self, client, _catalog_path):
        """No data has ever been written for this variable/time_unit/grid -- CatalogQuery's
        _load_catalog finds no catalog files under catalog_path and raises ValueError,
        which the route translates into a 404."""
        response = client.get(self.URL)
        assert response.status_code == 404
        assert "No catalog entries" in response.json()["detail"]
