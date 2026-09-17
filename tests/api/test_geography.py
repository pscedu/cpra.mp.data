"""Tests for converting between coordinates and grid cell IDs using geometry files."""

import pytest
from rasterio.crs import CRS
from rasterio.warp import transform
from shapely.geometry import box

from cpra.mp.data.analysis import AnalysisError
from cpra.mp.data.analysis.io.geography import (
    parse_coordinate_pairs,
    project_coordinates,
    representative_points,
    resolve_vector_grid_cell,
    resolve_vector_grid_cell_sync,
)
from tests.factories import write_grid_geometry


def _lng_lat(x, y):
    lng, lat = [c[0] for c in transform(CRS.from_epsg(26915), CRS.from_epsg(4326), [x], [y])]
    return lng, lat


@pytest.fixture
def ecoregion_geometry(geometry_path):
    """Two adjacent 1km squares in ecoregion_v001's CRS, at IDs 1 and 2."""
    return write_grid_geometry(
        geometry_path,
        "ecoregion_v001",
        [
            (1, box(500000, 3300000, 501000, 3301000)),
            (2, box(501000, 3300000, 502000, 3301000)),
        ],
    )


async def test_resolves_a_point_inside_a_polygon(ecoregion_geometry):
    lng, lat = _lng_lat(500500, 3300500)
    resolved = await resolve_vector_grid_cell("ecoregion_v001", [lng], [lat])
    assert resolved == [1]


async def test_resolves_multiple_points_independently(ecoregion_geometry):
    lng1, lat1 = _lng_lat(500500, 3300500)
    lng2, lat2 = _lng_lat(501500, 3300500)
    resolved = await resolve_vector_grid_cell("ecoregion_v001", [lng1, lng2], [lat1, lat2])
    assert resolved == [1, 2]


async def test_point_outside_every_polygon_resolves_to_none(ecoregion_geometry):
    lng, lat = _lng_lat(0, 0)
    resolved = await resolve_vector_grid_cell("ecoregion_v001", [lng], [lat])
    assert resolved == [None]


async def test_missing_geometry_file_raises_400(geometry_path):
    with pytest.raises(AnalysisError) as excinfo:
        await resolve_vector_grid_cell("ecoregion_v001", [-91.0], [29.5])
    assert excinfo.value.status_code == 400
    assert "No geometry file" in excinfo.value.detail


def test_representative_points_are_inside_their_polygons(ecoregion_geometry):
    points = representative_points("ecoregion_v001")
    assert set(points) == {1, 2}
    for geo_id, (lng, lat) in points.items():
        x, y = [c[0] for c in transform(CRS.from_epsg(4326), CRS.from_epsg(26915), [lng], [lat])]
        assert resolve_vector_grid_cell_sync("ecoregion_v001", [lng], [lat]) == [geo_id]
        assert 500000 <= x <= 502000 and 3300000 <= y <= 3301000


def test_representative_points_can_be_limited_to_some_geographies(ecoregion_geometry):
    assert list(representative_points("ecoregion_v001", [2])) == [2]


def test_representative_points_of_nothing_is_empty(ecoregion_geometry):
    """An empty selection has to short-circuit: `id IN ()` is not valid SQL."""
    assert representative_points("ecoregion_v001", []) == {}


class TestProjectCoordinates:
    """A coordinate can be valid WGS84 and still have no place in a grid's projection."""

    def test_projects_into_the_target_crs(self):
        xs, ys = project_coordinates([-91.0], [30.0], CRS.from_epsg(26915))
        assert (round(xs[0]), round(ys[0])) == (692915, 3320469)

    def test_unprojectable_coordinates_raise_400(self):
        """The grids' UTM zone has no image for the origin, which PROJ reports as an error
        rather than as a coordinate — a request, not a server, problem."""
        with pytest.raises(AnalysisError) as excinfo:
            project_coordinates([0.0], [0.0], CRS.from_epsg(26915))
        assert excinfo.value.status_code == 400
        assert "outside the grid's projection" in excinfo.value.detail

    def test_only_the_unprojectable_pairs_are_reported(self):
        """PROJ fails the whole batch, so the pair at fault has to be found afterwards."""
        with pytest.raises(AnalysisError) as excinfo:
            project_coordinates([-91.0, 0.0], [30.0, 0.0], CRS.from_epsg(26915))
        assert "0.0, 0.0" in excinfo.value.detail
        assert "-91.0" not in excinfo.value.detail


class TestParseCoordinatePairs:
    def test_parses_pairs_in_order(self):
        assert parse_coordinate_pairs(["-91.0", "30.0", "-90.0", "29.0"]) == (
            [-91.0, -90.0],
            [30.0, 29.0],
        )

    def test_nan_is_an_invalid_coordinate(self):
        """`float` accepts "nan" and every bounds comparison against it is False, so a
        magnitude check would pass it through to project to infinity."""
        with pytest.raises(AnalysisError) as excinfo:
            parse_coordinate_pairs(["nan", "nan"])
        assert excinfo.value.status_code == 400
        assert "invalid coordinate pair" in excinfo.value.detail

    def test_infinity_is_an_invalid_coordinate(self):
        with pytest.raises(AnalysisError) as excinfo:
            parse_coordinate_pairs(["inf", "0"])
        assert excinfo.value.status_code == 400
        assert "invalid coordinate pair" in excinfo.value.detail


async def test_unprojectable_point_raises_400(ecoregion_geometry):
    """The vector lookup projects before it reads, so it fails the same way."""
    with pytest.raises(AnalysisError) as excinfo:
        await resolve_vector_grid_cell("ecoregion_v001", [0.0], [0.0])
    assert excinfo.value.status_code == 400
    assert "outside the grid's projection" in excinfo.value.detail
