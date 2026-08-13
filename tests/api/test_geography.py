"""Tests for point-in-polygon coordinate resolution against grid geometry files."""

import pytest
from fastapi import HTTPException
from rasterio.crs import CRS
from rasterio.warp import transform
from shapely.geometry import box

from cpra.mp.data.api.geography import resolve_vector_geography

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
    resolved = await resolve_vector_geography("ecoregion_v001", [lng], [lat])
    assert resolved == [1]


async def test_resolves_multiple_points_independently(ecoregion_geometry):
    lng1, lat1 = _lng_lat(500500, 3300500)
    lng2, lat2 = _lng_lat(501500, 3300500)
    resolved = await resolve_vector_geography("ecoregion_v001", [lng1, lng2], [lat1, lat2])
    assert resolved == [1, 2]


async def test_point_outside_every_polygon_resolves_to_none(ecoregion_geometry):
    lng, lat = _lng_lat(0, 0)
    resolved = await resolve_vector_geography("ecoregion_v001", [lng], [lat])
    assert resolved == [None]


async def test_missing_geometry_file_raises_400(geometry_path):
    with pytest.raises(HTTPException) as excinfo:
        await resolve_vector_geography("ecoregion_v001", [-91.0], [29.5])
    assert excinfo.value.status_code == 400
    assert "No geometry file" in excinfo.value.detail
