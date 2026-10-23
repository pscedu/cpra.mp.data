"""The OGC API - Features provider, driven through the mounted pygeoapi app."""

import datetime

import pytest
from pyproj import Transformer
from shapely.geometry import box

from cpra.mp.data import config
from cpra.mp.data.api.ogc_provider import strip_ogc_params
from cpra.mp.data.tabular import write_parquet
from tests.factories import daily_frame, write_grid_geometry

ITEMS = "/v1/ogc/collections/master_plan_data/items"
SALINITY = (
    "variable=sal&time_unit=daily&grid=hydro_compartment_v001&model_group_id=500"
    "&scenario_id=7&aggregate_type=mean"
)
DAY = "calendar_day=2030-01-01"

# Three 1 km squares in hydro_compartment_v001's own CRS (EPSG:26915), a few km apart.
CELLS = {1: box(500000, 3300000, 501000, 3301000),
         2: box(505000, 3300000, 506000, 3301000),
         3: box(510000, 3300000, 511000, 3301000)}
DAYS = [datetime.date(2030, 1, 1), datetime.date(2030, 1, 2)]


@pytest.fixture
def salinity(client, base_path, geometry_path, monkeypatch):
    """`client` over a three-cell sal grid for two days. `daily_frame` gives each cell its
    own id as its value, so cell N reads N.0."""
    write_grid_geometry(geometry_path, "hydro_compartment_v001", list(CELLS.items()))
    write_parquet(daily_frame(days=DAYS), base_path=base_path, validate=False, sort=False)
    monkeypatch.setattr(config, "BASE_PATH", base_path)
    return client


def lng_lat(x, y):
    return Transformer.from_crs("EPSG:26915", "OGC:CRS84", always_xy=True).transform(x, y)


def features(response):
    assert response.status_code == 200, response.text
    return response.json()["features"]


def ids(response):
    """Feature ids, sorted: the FlatGeoBuf's spatial index orders features by position, not id."""
    return sorted(f["id"] for f in features(response))


class TestStripOgcParams:
    def test_drops_ogc_params_case_insensitively_and_keeps_the_rest_in_order(self):
        items = [("variable", "sal"), ("f", "json"), ("LIMIT", "5"), ("scenario_id", "7"),
                 ("bbox", "1,2,3,4"), ("scenario_id", "8")]
        assert strip_ogc_params(items) == [("variable", "sal"), ("scenario_id", "7"), ("scenario_id", "8")]


class TestItems:
    def test_each_feature_carries_its_cells_value(self, salinity):
        found = features(salinity.get(f"{ITEMS}?f=json&{SALINITY}&{DAY}"))
        assert {f["id"]: f["properties"]["value"] for f in found} == {1: 1.0, 2: 2.0, 3: 3.0}
        assert all(f["properties"]["id"] == f["id"] for f in found)

    def test_geometry_is_reprojected_to_longitude_latitude(self, salinity):
        found = {f["id"]: f for f in features(salinity.get(f"{ITEMS}?f=json&{SALINITY}&{DAY}"))}
        geometry = found[1]["geometry"]
        assert geometry["type"] == "Polygon"
        ring = geometry["coordinates"][0]
        corners = {tuple(round(c, 6) for c in lng_lat(x, y)) for x, y in CELLS[1].exterior.coords}
        assert {tuple(round(c, 6) for c in point) for point in ring} == corners
        assert all(-95 < lng < -88 and 28 < lat < 31 for lng, lat in ring)

    def test_the_selection_decides_which_values_come_back(self, salinity):
        found = features(salinity.get(f"{ITEMS}?f=json&{SALINITY}&calendar_day=2030-01-02"))
        assert {f["id"]: f["properties"]["value"] for f in found} == {1: 1.0, 2: 2.0, 3: 3.0}

    def test_limit_and_offset_page_through_the_cells(self, salinity):
        first = salinity.get(f"{ITEMS}?f=json&{SALINITY}&{DAY}&limit=2")
        second = salinity.get(f"{ITEMS}?f=json&{SALINITY}&{DAY}&limit=2&offset=2")
        assert (len(features(first)), len(features(second))) == (2, 1)
        assert first.json()["numberMatched"] == 3
        assert sorted(ids(first) + ids(second)) == [1, 2, 3]

    def test_next_link_keeps_the_data_request(self, salinity):
        links = salinity.get(f"{ITEMS}?f=json&{SALINITY}&{DAY}&limit=2").json()["links"]
        next_href = next(link["href"] for link in links if link["rel"] == "next")
        assert "variable=sal" in next_href and "calendar_day=2030-01-01" in next_href

    def test_bbox_is_in_longitude_latitude_and_narrows_the_cells(self, salinity):
        west, south = lng_lat(499500, 3299500)
        east, north = lng_lat(506500, 3301500)
        response = salinity.get(f"{ITEMS}?f=json&{SALINITY}&{DAY}&bbox={west},{south},{east},{north}")
        assert ids(response) == [1, 2]
        assert response.json()["numberMatched"] == 2

    def test_hits_counts_without_returning_features(self, salinity):
        body = salinity.get(f"{ITEMS}?f=json&{SALINITY}&{DAY}&resulttype=hits").json()
        assert body["numberMatched"] == 3
        assert body["features"] == []

    def test_skip_geometry_drops_the_geometry_but_keeps_values(self, salinity):
        found = features(salinity.get(f"{ITEMS}?f=json&{SALINITY}&{DAY}&skipGeometry=true"))
        assert [f["geometry"] for f in found] == [None, None, None]
        assert {f["id"]: f["properties"]["value"] for f in found} == {1: 1.0, 2: 2.0, 3: 3.0}

    def test_a_cell_the_data_lacks_gets_a_null_value(self, salinity, geometry_path):
        write_grid_geometry(
            geometry_path, "hydro_compartment_v001", [*CELLS.items(), (4, box(515000, 3300000, 516000, 3301000))]
        )
        found = {f["id"]: f for f in features(salinity.get(f"{ITEMS}?f=json&{SALINITY}&{DAY}"))}
        assert found[4]["properties"]["value"] is None


class TestItem:
    def test_returns_one_feature_with_its_value(self, salinity):
        response = salinity.get(f"{ITEMS}/2?f=json&{SALINITY}&{DAY}")
        assert response.status_code == 200
        assert response.json()["properties"]["value"] == 2.0

    @pytest.mark.parametrize("identifier", ["99", "not-a-number"])
    def test_unknown_identifier_is_not_found(self, salinity, identifier):
        assert salinity.get(f"{ITEMS}/{identifier}?f=json&{SALINITY}&{DAY}").status_code == 404


class TestBadRequests:
    """Each is a 400 whose message reaches the client, because pygeoapi hides the text of
    a plain provider error behind 'check logs'."""

    def test_no_dataset_selected(self, salinity):
        response = salinity.get(f"{ITEMS}?f=json")
        assert response.status_code == 400
        assert "no dataset selected" in response.json()["description"]

    def test_more_than_one_value_per_cell(self, salinity):
        response = salinity.get(f"{ITEMS}?f=json&{SALINITY}")
        assert response.status_code == 400
        assert "more than one value per grid cell" in response.json()["description"]

    def test_raster_grid(self, salinity):
        response = salinity.get(f"{ITEMS}?f=json&variable=vct&time_unit=annual&grid=veg_grid_cell_v001")
        assert response.status_code == 400
        assert "band-wide (raster)" in response.json()["description"]
