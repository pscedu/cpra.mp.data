"""Tests for converting between coordinates and grid cell IDs using geometry files."""

from types import SimpleNamespace

import pytest
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.transform import xy as raster_xy
from rasterio.warp import transform
from shapely.geometry import box

from cpra.mp.data.analysis import AnalysisError
from cpra.mp.data.analysis.io.geography import (
    parse_coordinate_pairs,
    project_coordinates,
    raster_grid_cell_id,
    representative_points,
    resolve_raster_pixel,
    resolve_vector_grid_cell,
    resolve_vector_grid_cell_sync,
)
from cpra.mp.data.analysis.request.errors import NotFoundError
from cpra.mp.data.grids import GRIDS
from tests.factories import write_grid_geometry

# A real raster grid already present in `structure/grids.csv` -- `resolve_raster_pixel`
# and `raster_grid_cell_id` need nothing but a `RasterGrid`'s declared geometry
# (`bounds`/`height`/`width`/`crs`), so unlike the vector fixtures above, no `.fgb` (or any
# other file) needs to be written for these tests.
RASTER_GRID = "veg_grid_cell_v001"


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


def _raster_pixel_lng_lat(row: int, col: int) -> tuple[float, float]:
    """WGS84 (lng, lat) at the CENTER of pixel `(row, col)` on `RASTER_GRID`'s own full
    pixel grid, computed straight from `rasterio.transform` independently of
    `resolve_raster_pixel` itself -- so a test asserting `resolve_raster_pixel(..., lng,
    lat) == [(row, col)]` actually proves the round trip, rather than just restating
    whatever arithmetic `resolve_raster_pixel` happens to use.
    """
    grid = GRIDS[RASTER_GRID]
    grid_transform = from_bounds(*grid.bounds, width=grid.width, height=grid.height)
    x, y = raster_xy(grid_transform, row, col)
    lng, lat = [c[0] for c in transform(CRS.from_epsg(26915), CRS.from_epsg(4326), [x], [y])]
    return lng, lat


class TestResolveRasterPixel:
    """`resolve_raster_pixel` is the raster counterpart of `resolve_vector_grid_cell`: no
    file is opened -- `RASTER_GRID`'s declared `bounds`/`height`/`width`/`crs` (from
    `structure/grids.csv`) are the entire input.
    """

    def test_resolves_a_point_to_its_absolute_pixel_position(self):
        row, col = 10, 20
        lng, lat = _raster_pixel_lng_lat(row, col)
        assert resolve_raster_pixel(RASTER_GRID, [lng], [lat]) == [(row, col)]

    def test_resolves_a_point_far_from_the_origin_too(self):
        """A pixel near `(0, 0)` can't tell a transposed row/col apart from a correct one --
        this pins a position where `width != height` (1052 x 365) makes that mistake
        produce a different, wrong answer, the same reason a window-relative id test needs
        a point away from the raster's origin to actually distinguish itself from the
        absolute id.
        """
        row, col = 300, 900
        lng, lat = _raster_pixel_lng_lat(row, col)
        assert resolve_raster_pixel(RASTER_GRID, [lng], [lat]) == [(row, col)]

    def test_resolves_multiple_points_independently(self):
        row1, col1 = 5, 5
        row2, col2 = 100, 200
        lng1, lat1 = _raster_pixel_lng_lat(row1, col1)
        lng2, lat2 = _raster_pixel_lng_lat(row2, col2)
        resolved = resolve_raster_pixel(RASTER_GRID, [lng1, lng2], [lat1, lat2])
        assert resolved == [(row1, col1), (row2, col2)]

    def test_point_outside_the_raster_extent_resolves_to_none(self):
        """A point south of the grid's own `miny` (but still a valid, projectable UTM
        coordinate) is out of pixel-row range on the low end -- `None`, the same convention
        `resolve_vector_grid_cell` uses for a point outside every polygon, not an error.
        """
        grid = GRIDS[RASTER_GRID]
        minx, miny, maxx, maxy = grid.bounds
        center_x = (minx + maxx) / 2
        lng, lat = [c[0] for c in transform(CRS.from_epsg(26915), CRS.from_epsg(4326), [center_x], [miny - 10_000])]
        assert resolve_raster_pixel(RASTER_GRID, [lng], [lat]) == [None]

    def test_mixed_in_and_out_of_extent_points_resolve_independently(self):
        """Each point is resolved on its own -- one point landing outside the grid must not
        affect another point in the same call that lands inside it.
        """
        grid = GRIDS[RASTER_GRID]
        minx, miny, maxx, maxy = grid.bounds
        lng_in, lat_in = _raster_pixel_lng_lat(0, 0)
        center_x = (minx + maxx) / 2
        lng_out, lat_out = [
            c[0] for c in transform(CRS.from_epsg(26915), CRS.from_epsg(4326), [center_x], [miny - 10_000])
        ]
        resolved = resolve_raster_pixel(RASTER_GRID, [lng_in, lng_out], [lat_in, lat_out])
        assert resolved == [(0, 0), None]

    def test_unprojectable_point_raises_the_same_400_the_vector_path_raises(self):
        """`resolve_raster_pixel` reuses `project_coordinates` directly, so an unprojectable
        point (the UTM zone's own origin) fails identically for either grid kind.
        """
        with pytest.raises(AnalysisError) as excinfo:
            resolve_raster_pixel(RASTER_GRID, [0.0], [0.0])
        assert excinfo.value.status_code == 400
        assert "outside the grid's projection" in excinfo.value.detail

    def test_a_vector_grid_name_raises_not_found(self):
        """`resolve_raster_pixel` is raster-only -- a name that resolves to a real grid of
        the WRONG kind is exactly as unusable to it as a name that doesn't exist at all.
        """
        with pytest.raises(NotFoundError) as excinfo:
            resolve_raster_pixel("ecoregion_v001", [-91.0], [29.5])
        assert "is not a raster grid" in str(excinfo.value)

    def test_an_unknown_grid_name_raises_not_found(self):
        with pytest.raises(NotFoundError) as excinfo:
            resolve_raster_pixel("not_a_real_grid_at_all", [-91.0], [29.5])
        assert "is not a raster grid" in str(excinfo.value)

    def test_not_found_is_the_new_engine_error_family_not_analysis_error(self):
        """The one failure mode `resolve_raster_pixel` invented itself (as opposed to
        reusing `project_coordinates`'s existing `AnalysisError`) must raise from the NEW
        engine's error family, per this module's own docstring -- not the old engine's
        `AnalysisError`, which carries an HTTP `status_code` this exception does not.
        """
        with pytest.raises(NotFoundError) as excinfo:
            resolve_raster_pixel("not_a_real_grid_at_all", [-91.0], [29.5])
        assert not isinstance(excinfo.value, AnalysisError)


class TestRasterGridCellId:
    """`raster_grid_cell_id` computes FORWARD, from a known absolute pixel, the same id
    `execution.frame.row_major_grid_cell` would DECODE from a real read of that window --
    see that function's own docstring for why the two are the same arithmetic.
    """

    def test_matches_row_major_numbering_for_a_full_grid_window(self):
        """A window that starts at the raster's own origin and spans its full width is the
        convention every non-point-sample read uses today -- the id should reduce to plain
        `row * width + col` numbering.
        """
        grid = GRIDS[RASTER_GRID]
        window = SimpleNamespace(row_off=0, col_off=0, width=grid.width)
        row, col = 10, 20
        assert raster_grid_cell_id((row, col), window) == row * grid.width + col

    def test_offsets_the_id_for_a_narrower_window(self):
        """A window that does NOT start at the raster's own origin must shift both the row
        and column components of the id by the window's own offset before applying the
        window's (possibly narrower) width -- this is the exact arithmetic a point-sparse
        read's window needs, once one exists.
        """
        window = SimpleNamespace(row_off=5, col_off=3, width=10)
        assert raster_grid_cell_id((7, 8), window) == 25

    def test_a_pixel_at_the_windows_own_origin_is_cell_zero(self):
        window = SimpleNamespace(row_off=5, col_off=3, width=10)
        assert raster_grid_cell_id((5, 3), window) == 0

    def test_window_width_not_raster_width_scales_the_row_component(self):
        """The multiplier in the row component must be the WINDOW's width, not the full
        raster's -- using the wrong one is exactly the bug a window narrower than the whole
        raster would otherwise hide, since the two widths only diverge once a window is
        actually narrower than the raster it was read from.
        """
        grid = GRIDS[RASTER_GRID]
        assert grid.width != 10  # guards the premise: the window below must actually differ
        window = SimpleNamespace(row_off=0, col_off=0, width=10)
        assert raster_grid_cell_id((2, 4), window) == 2 * 10 + 4
