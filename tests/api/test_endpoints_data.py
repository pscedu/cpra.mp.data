"""End-to-end tests for the data and tile endpoints through the ASGI app.

Unlike the Dataset unit tests, these go through real routing and query-string parsing, so
they cover the parameter binding that FastAPI does before a Dataset is ever built.
"""

import io
import json

import mercantile
import polars as pl
import pytest
import rasterio as rio
from rasterio.crs import CRS
from rasterio.warp import transform
from shapely.geometry import box

from cpra.mp.data import config
from cpra.mp.data.tabular import write_parquet

from tests.factories import (
    ANNUAL_KWARGS,
    RASTER_KWARGS,
    annual_frame,
    write_grid_geometry,
    write_pmtiles,
    write_raster_dataset,
)

VECTOR_URL = "/v1/datasets/ssf/ecoregion_v001/annual"
RASTER_URL = "/v1/datasets/vct/veg_grid_cell_v001/annual"


def _lng_lat(x, y):
    """Project an ecoregion_v001 (EPSG:26915) point to WGS84, as a `geography=` value."""
    lng, lat = [c[0] for c in transform(CRS.from_epsg(26915), CRS.from_epsg(4326), [x], [y])]
    return f"{lng!s},{lat!s}"


@pytest.fixture
def vector_data(base_path, monkeypatch):
    write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
    monkeypatch.setattr(config, "BASE_PATH", base_path)
    return base_path


@pytest.fixture
def ecoregion_geometry(geometry_path):
    """A single 1km square at ecoregion_v001 cell 1, for coordinate lookups."""
    return write_grid_geometry(
        geometry_path, "ecoregion_v001", [(1, box(500000, 3300000, 501000, 3301000))]
    )


@pytest.fixture
def raster_data(base_path, monkeypatch):
    write_raster_dataset(base_path, years=(2030,), fill_fn=lambda y: y - 2000)
    monkeypatch.setattr(config, "BASE_PATH", base_path)
    return base_path


class TestVectorData:
    def test_csv(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv")
        assert response.status_code == 200
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["calendar_year"].to_list() == [2030, 2040]

    def test_parquet(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.parquet")
        assert response.status_code == 200
        assert pl.read_parquet(io.BytesIO(response.content)).height == 2

    def test_ndjson(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.ndjson")
        assert response.status_code == 200
        assert pl.read_ndjson(io.BytesIO(response.content)).height == 2

    def test_dimension_filter_from_the_query_string(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv?calendar_year=2030")
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["calendar_year"].to_list() == [2030]

    def test_repeated_filter_values(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv?calendar_year=2030&calendar_year=2040")
        assert pl.read_csv(io.BytesIO(response.content)).height == 2

    def test_geography_selects_columns(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv?geography=1,2")
        df = pl.read_csv(io.BytesIO(response.content))
        assert [c for c in df.columns if c[0].isdigit()] == ["1", "2"]

    def test_geography_accepts_a_coordinate_pair(self, client, vector_data, ecoregion_geometry):
        """geography= on a vector grid also accepts lat/lng, resolved via the grid's
        FlatGeoBuf geometry and reported back under the grid cell ID it resolved to."""
        label = _lng_lat(500500, 3300500)

        response = client.get(f"{VECTOR_URL}/data.csv?geography={label}")
        assert response.status_code == 200
        df = pl.read_csv(io.BytesIO(response.content))
        assert "1" in df.columns
        assert label not in df.columns

    def test_unpivot(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv?unpivot=true&geography=1")
        df = pl.read_csv(io.BytesIO(response.content))
        assert {"geography", "value"} <= set(df.columns)

    def test_unpivot_reports_cell_ids_for_a_coordinate_selection(
        self, client, vector_data, ecoregion_geometry
    ):
        """The shape every visualization requests: unpivoted, and still keyed by ID, so the
        caller can join it against anything else the vector grid keys that way."""
        label = _lng_lat(500500, 3300500)

        response = client.get(f"{VECTOR_URL}/data.csv?unpivot=true&geography={label}")
        assert response.status_code == 200
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["geography"].unique().to_list() == [1]

    def test_coordinates_sharing_a_cell_report_it_once(
        self, client, vector_data, ecoregion_geometry
    ):
        """Two coordinates in one cell describe one series — there's a single value, so
        reporting it twice under one ID would only be ambiguous."""
        first, second = _lng_lat(500400, 3300400), _lng_lat(500600, 3300600)

        response = client.get(
            f"{VECTOR_URL}/data.csv?unpivot=true&geography={first},{second}&calendar_year=2030"
        )
        assert response.status_code == 200
        df = pl.read_csv(io.BytesIO(response.content))
        assert dict(zip(df["geography"], df["value"])) == {1: pytest.approx(2030.01)}

    def test_select(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv?select=calendar_year")
        assert pl.read_csv(io.BytesIO(response.content)).columns == ["calendar_year"]

    def test_sort_descending(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv?sort=-calendar_year")
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["calendar_year"].to_list() == [2040, 2030]

    def test_where_clause(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv?where=calendar_year%20%3E%202035")
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["calendar_year"].to_list() == [2040]

    def test_tif_is_rejected_for_a_vector_grid(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.tif")
        assert response.status_code == 400
        assert "must be one of" in response.json()["detail"]

    def test_unknown_variable_is_rejected(self, client, vector_data):
        assert client.get("/v1/datasets/nope/ecoregion_v001/annual/data.csv").status_code == 422

    def test_unknown_grid_is_rejected(self, client, vector_data):
        assert client.get("/v1/datasets/ssf/nope/annual/data.csv").status_code == 422

    def test_missing_dataset_is_404(self, client, base_path, monkeypatch):
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        response = client.get(f"{VECTOR_URL}/data.csv")
        assert response.status_code == 404

    def test_invalid_sort_is_400(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv?sort=nonsense")
        assert response.status_code == 400

    def test_group_by_without_aggregate_is_400(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv?group_by=calendar_year")
        assert response.status_code == 400


class TestRasterData:
    def test_tif(self, client, raster_data):
        response = client.get(f"{RASTER_URL}/data.tif")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff; application=geotiff"
        with rio.open(io.BytesIO(response.content)) as src:
            assert src.read(1)[0, 0] == 30

    def test_csv_is_rejected_for_a_raster_grid(self, client, raster_data):
        response = client.get(f"{RASTER_URL}/data.csv")
        assert response.status_code == 400
        assert "must be tif" in response.json()["detail"]

    def test_geography_makes_csv_valid(self, client, raster_data):
        """Sampling at coordinates converts raster output to tabular."""
        with rio.open(next(raster_data.rglob("*.tif"))) as src:
            from rasterio.warp import transform

            x, y = src.xy(src.height // 2, src.width // 2)
            lng, lat = [c[0] for c in transform(src.crs, "EPSG:4326", [x], [y])]

        response = client.get(f"{RASTER_URL}/data.csv?geography={lng},{lat}")
        assert response.status_code == 200
        df = pl.read_csv(io.BytesIO(response.content))
        assert df[f"{lng},{lat}"].to_list() == [30]

    def test_tabular_compression_is_rejected_for_tif(self, client, raster_data):
        response = client.get(f"{RASTER_URL}/data.tif?compression_algorithm=gzip")
        assert response.status_code == 400

    def test_missing_dataset_is_404(self, client, base_path, monkeypatch):
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        assert client.get(f"{RASTER_URL}/data.tif").status_code == 404


class TestRasterTiles:
    def _tile(self, raster_data, zoom=8):
        with rio.open(next(raster_data.rglob("*.tif"))) as src:
            from rasterio.warp import transform_bounds

            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return mercantile.tile((west + east) / 2, (south + north) / 2, zoom)

    def test_png(self, client, raster_data):
        tile = self._tile(raster_data)
        response = client.get(f"{RASTER_URL}/data/{tile.z}/{tile.x}/{tile.y}.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_jpg(self, client, raster_data):
        tile = self._tile(raster_data)
        response = client.get(f"{RASTER_URL}/data/{tile.z}/{tile.x}/{tile.y}.jpg")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"

    def test_tile_outside_the_dataset_is_404(self, client, raster_data):
        assert client.get(f"{RASTER_URL}/data/2/0/0.png").status_code == 404

    def test_unknown_tile_format_is_422(self, client, raster_data):
        tile = self._tile(raster_data)
        response = client.get(f"{RASTER_URL}/data/{tile.z}/{tile.x}/{tile.y}.gif")
        assert response.status_code == 422

    def test_multiband_tile_is_400(self, client, base_path, monkeypatch):
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        response = client.get(f"{RASTER_URL}/data/8/0/0.png")
        assert response.status_code == 400
        assert "more than one band" in response.json()["detail"]

    def test_color_map_is_accepted(self, client, raster_data):
        tile = self._tile(raster_data)
        response = client.get(
            f"{RASTER_URL}/data/{tile.z}/{tile.x}/{tile.y}.png?color_map=lnd_type"
        )
        assert response.status_code == 200


class TestVectorTiles:
    def test_serves_a_tile(self, client, vector_data, tiles_path):
        tile = mercantile.Tile(x=1, y=2, z=3)
        write_pmtiles(tiles_path, "ecoregion_v001", tile, [1, 2])

        response = client.get(
            f"{VECTOR_URL}/tiles/{tile.z}/{tile.x}/{tile.y}.pbf?calendar_year=2030"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"

    def test_missing_pmtiles_file_is_404(self, client, vector_data, tiles_path):
        response = client.get(f"{VECTOR_URL}/tiles/3/1/2.pbf")
        assert response.status_code == 404
        assert "no tile file for grid" in response.json()["detail"]

    def test_raster_grid_is_400(self, client, raster_data, tiles_path):
        response = client.get(f"{RASTER_URL}/tiles/3/1/2.pbf")
        assert response.status_code == 400
        assert "invalid output type for vector tile" in response.json()["detail"]


class TestCatalogEndpoint:
    def test_returns_choices_results_and_exact(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/catalog.json")
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"choices", "results", "exact"}

    def test_reports_available_years(self, client, vector_data):
        """2030 and 2040 are ten apart, so an annual series reports them separately."""
        body = client.get(f"{VECTOR_URL}/catalog.json").json()
        assert body["choices"]["calendar_year"] == [2030, 2040]

    def test_contiguous_years_are_compressed_into_a_range(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030, 2031, 2032)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        body = client.get(f"{VECTOR_URL}/catalog.json").json()
        assert body["choices"]["calendar_year"] == [[2030, 2032]]

    def test_selection_is_reflected_in_results(self, client, vector_data):
        body = client.get(f"{VECTOR_URL}/catalog.json?calendar_year=2040").json()
        assert body["results"]["calendar_year"] == [2040]

    def test_unavailable_selection_is_inexact(self, client, vector_data):
        body = client.get(f"{VECTOR_URL}/catalog.json?calendar_year=1999").json()
        assert body["exact"] is False


class TestVisualizationEndpoints:
    def test_dataset_visualization_returns_a_vega_spec(self, client, vector_data):
        response = client.get(
            f"{VECTOR_URL}/visualization/time-series-line.json?geography=1&calendar_year=2030"
        )
        assert response.status_code == 200
        assert "$schema" in response.json()

    def test_dataset_visualization_accepts_coordinates_on_a_vector_grid(
        self, client, vector_data, ecoregion_geometry
    ):
        """The spec has to describe the data.csv it points at, which reports the resolved
        cell ID however the caller selected it — a number, whatever the request looked
        like. The coordinate survives only as a label."""
        label = _lng_lat(500500, 3300500)
        response = client.get(
            f"{VECTOR_URL}/visualization/time-series-line.json?geography={label}&calendar_year=2030"
        )
        assert response.status_code == 200
        spec = response.json()
        assert label.replace(",", ", ") in json.dumps(spec)
        assert spec["data"]["format"]["parse"] == {
            "value": "number",
            "geography": "number",
            "calendar_year": "number",
        }

    def test_time_series_line_labels_a_coordinate_with_its_grid_cell(
        self, client, vector_data, ecoregion_geometry
    ):
        """The legend and tooltips report the cell the coordinate resolved to, since that's
        what the values describe, while the data stays keyed by the coordinate itself. Only
        the legend repeats the coordinate — a tooltip row has no width to spare."""
        label = _lng_lat(500500, 3300500)
        response = client.get(
            f"{VECTOR_URL}/visualization/time-series-line.json?geography={label}&calendar_year=2030"
        )
        assert response.status_code == 200
        spec = response.json()

        display = f"1 ({label.replace(',', ', ')})"
        line = spec["layer"][0]
        assert line["encoding"]["color"]["field"] == "geography_label"
        assert line["encoding"]["color"]["scale"]["domain"] == [display]
        # The lookup that puts that label on each row is keyed by the API's own value.
        lookup = line["transform"][-1]
        assert lookup["lookup"] == "geography"
        assert spec["datasets"][lookup["from"]["data"]["name"]] == [
            {"geography_idx": 0, "geography": 1, "geography_label": display}
        ]
        tooltips = spec["layer"][2]["encoding"]["tooltip"]
        assert [t["title"] for t in tooltips[1:]] == ["1: "]

    @pytest.mark.parametrize("name", ["map", "stacked-bar", "time-series-line"])
    def test_every_dataset_visualization_accepts_coordinates(
        self, client, vector_data, ecoregion_geometry, name
    ):
        label = _lng_lat(500500, 3300500)
        response = client.get(
            f"{VECTOR_URL}/visualization/{name}.json?geography={label}&calendar_year=2030"
        )
        assert response.status_code == 200

    def test_dataset_visualization_parses_grid_cell_ids_as_numbers(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/visualization/time-series-line.json?geography=1")
        assert response.json()["data"]["format"]["parse"]["geography"] == "number"

    def test_unknown_visualization_is_404(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/visualization/nope.json")
        assert response.status_code == 404

    def test_custom_visualization_rejected_on_the_dataset_route(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/visualization/sediment-dynamics.json?geography=1")
        assert response.status_code == 400

    def test_dataset_visualization_rejected_on_the_custom_route(self, client, vector_data):
        response = client.get("/v1/visualizations/time-series-line.json?geography=1")
        assert response.status_code == 400


class TestCors:
    """The portal is served from a different origin than the API, so CORS must permit it.

    Whether the allow-origin header comes back as `*` or as the echoed origin is a
    Starlette implementation detail that has changed across releases (the condition moved
    from `allow_all_origins and has_cookie` to `allow_all_origins and allow_credentials`),
    so these assert only that the requesting origin is permitted.
    """

    ORIGIN = "https://example.com"

    def test_request_origin_is_permitted(self, client, vector_data):
        response = client.get("/v1/variables.json", headers={"Origin": self.ORIGIN})
        assert response.headers["access-control-allow-origin"] in ("*", self.ORIGIN)

    def test_credentials_are_allowed(self, client, vector_data):
        response = client.get("/v1/variables.json", headers={"Origin": self.ORIGIN})
        assert response.headers["access-control-allow-credentials"] == "true"

    def test_preflight_advertises_methods_and_headers(self, client, vector_data):
        response = client.options(
            f"{VECTOR_URL}/data.csv",
            headers={
                "Origin": self.ORIGIN,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-Custom",
            },
        )
        assert response.status_code == 200
        assert "GET" in response.headers["access-control-allow-methods"]
        assert "X-Custom" in response.headers["access-control-allow-headers"]
