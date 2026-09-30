"""Tests for `cpra.mp.data.api.data_query.resolve_grid_cell` -- turning the one
`grid_cell=` spelling that needs I/O (a coordinate pair) into a `grid_cell` conjunct and,
for a raster grid, the relabel map that names the pixel back.

Asserted against the returned `(where, relabel)` directly rather than through the ASGI
app: what a route then does with those is `test_endpoints_data.py`'s subject, and keeping
the two apart means a `grid_cell` resolution bug is never diagnosed as a response-plumbing
bug. The composition this used to sit under, `build_data_query`, is gone -- see the note
at the bottom of this file.

The vector coordinate path is the only one that opens a file, so only its tests take the
`geometry_path`-backed fixture; the raster path needs nothing but `structure/grids.csv`'s
declared `bounds`/`width`/`height`/`crs`.
"""

import pytest
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.transform import xy as raster_xy
from rasterio.warp import transform
from shapely.geometry import box

from cpra.mp.data.analysis.request.errors import NotFoundError, RequestError
from cpra.mp.data.api.boundary import resolve_status
from cpra.mp.data.api.data_query import resolve_grid_cell
from cpra.mp.data.grids import GRIDS
from tests.factories import write_grid_geometry

VECTOR_GRID = "ecoregion_v001"
RASTER_GRID = "veg_grid_cell_v001"

assert GRIDS[VECTOR_GRID].grid_type == "vector"
assert GRIDS[RASTER_GRID].grid_type == "raster"


def _lng_lat(x, y):
    lng, lat = [c[0] for c in transform(CRS.from_epsg(26915), CRS.from_epsg(4326), [x], [y])]
    return lng, lat


def _raster_pixel_lng_lat(row: int, col: int) -> tuple[float, float]:
    """WGS84 (lng, lat) at the center of `RASTER_GRID`'s pixel `(row, col)`, computed from
    `rasterio.transform` directly so a test naming an expected cell ID proves the round
    trip rather than restating `resolve_raster_pixel`'s own arithmetic.
    """
    grid = GRIDS[RASTER_GRID]
    grid_transform = from_bounds(*grid.bounds, width=grid.width, height=grid.height)
    x, y = raster_xy(grid_transform, row, col)
    return _lng_lat(x, y)


def _far_south_lng_lat() -> tuple[float, float]:
    """A valid, projectable UTM coordinate 10km south of `RASTER_GRID`'s own `miny` -- out
    of the raster's extent, but nowhere near the edge of the CRS's projection domain, so
    it exercises the "resolved to nothing" path rather than the unprojectable-input 400.
    """
    minx, miny, maxx, _maxy = GRIDS[RASTER_GRID].bounds
    return _lng_lat((minx + maxx) / 2, miny - 10_000)


@pytest.fixture
def ecoregion_geometry(geometry_path):
    """Two adjacent 1km squares in ecoregion_v001's CRS, at IDs 1 and 2."""
    return write_grid_geometry(
        geometry_path,
        VECTOR_GRID,
        [
            (1, box(500000, 3300000, 501000, 3301000)),
            (2, box(501000, 3300000, 502000, 3301000)),
        ],
    )


class TestResolveGridCellNoop:
    """The two cases that resolve nothing and say so with `(None, None)`, leaving
    `options.grid_cell` for `build_data_query_pipeline` to handle (or ignore) itself.
    """

    async def test_no_grid_cell_resolves_to_nothing(self):
        assert await resolve_grid_cell(VECTOR_GRID, None) == (None, None)

    async def test_an_empty_grid_cell_resolves_to_nothing(self):
        assert await resolve_grid_cell(VECTOR_GRID, "") == (None, None)

    async def test_a_vector_grids_digit_ids_are_left_to_the_builder(self):
        """Digit-only tokens on a vector grid are literal polygon IDs, and the ID ->
        `where` rendering already exists in `request.data_query.grid_cell_conjunct`. This
        layer deliberately does not duplicate it -- it returns `None` so exactly one
        implementation of "these are literal IDs" runs.
        """
        assert await resolve_grid_cell(VECTOR_GRID, "1,2,3") == (None, None)

    async def test_leaving_digit_ids_to_the_builder_opens_no_geometry_file(self):
        """Without a `geometry_path` fixture, `GRIDS_PATH` holds no `.fgb` for this grid,
        so a digit-ID query that reached the point-in-polygon lookup would raise. Passing
        proves the digit branch short-circuits before any I/O.
        """
        assert await resolve_grid_cell(VECTOR_GRID, "7") == (None, None)


class TestResolveGridCellVector:
    """A vector grid's coordinate form: point-in-polygon, one ID per point, no relabeling
    (a vector `grid_cell` column already reports the polygon ID the caller can act on).
    """

    async def test_a_coordinate_pair_resolves_to_its_polygons_id(self, ecoregion_geometry):
        lng, lat = _lng_lat(500500, 3300500)
        where, relabel = await resolve_grid_cell(VECTOR_GRID, f"{lng},{lat}")
        assert where == "grid_cell = 1"
        assert relabel is None

    async def test_two_coordinate_pairs_resolve_to_an_in_list(self, ecoregion_geometry):
        lng1, lat1 = _lng_lat(500500, 3300500)
        lng2, lat2 = _lng_lat(501500, 3300500)
        where, _relabel = await resolve_grid_cell(VECTOR_GRID, f"{lng1},{lat1},{lng2},{lat2}")
        assert where == "grid_cell IN (1, 2)"

    async def test_a_point_outside_every_polygon_is_dropped_not_rejected(self, ecoregion_geometry):
        """The long-standing convention: a coordinate matching no polygon contributes no
        ID, and a query where every point misses becomes the "matched nothing" conjunct
        (an empty result) rather than an error.
        """
        lng, lat = _lng_lat(400000, 3200000)
        where, _relabel = await resolve_grid_cell(VECTOR_GRID, f"{lng},{lat}")
        assert where == "grid_cell != grid_cell"

    async def test_an_unmatched_point_does_not_suppress_a_matched_one(self, ecoregion_geometry):
        lng_in, lat_in = _lng_lat(500500, 3300500)
        lng_out, lat_out = _lng_lat(400000, 3200000)
        where, _relabel = await resolve_grid_cell(
            VECTOR_GRID, f"{lng_in},{lat_in},{lng_out},{lat_out}"
        )
        assert where == "grid_cell = 1"

    async def test_whitespace_around_tokens_is_tolerated(self, ecoregion_geometry):
        lng, lat = _lng_lat(500500, 3300500)
        where, _relabel = await resolve_grid_cell(VECTOR_GRID, f" {lng} , {lat} ")
        assert where == "grid_cell = 1"


class TestResolveGridCellRaster:
    """A raster grid's `grid_cell` is ALWAYS coordinates -- no digit check, matching what
    the point-sampling path has always done -- and always comes back with a `relabel` map
    so the result reports the caller's own coordinate strings.
    """

    async def test_a_coordinate_pair_resolves_to_its_absolute_pixel_id(self):
        row, col = 10, 20
        lng, lat = _raster_pixel_lng_lat(row, col)
        expected_id = row * GRIDS[RASTER_GRID].width + col
        where, relabel = await resolve_grid_cell(RASTER_GRID, f"{lng},{lat}")
        assert where == f"grid_cell = {expected_id}"
        assert relabel == {expected_id: [f"{lng},{lat}"]}

    async def test_the_label_is_the_callers_own_string_not_a_reformatted_float(self):
        """The label must be the token as typed: a round trip through `float()` would
        rewrite `-93.74873032000` (and every other non-canonical spelling) into something
        the caller never sent and cannot match against their own request.
        """
        lng, lat = _raster_pixel_lng_lat(10, 20)
        padded = f"{lng:.11f},{lat:.11f}"
        _where, relabel = await resolve_grid_cell(RASTER_GRID, padded)
        assert list(relabel.values()) == [[padded]]

    async def test_two_points_on_one_pixel_keep_both_labels(self):
        """Two distinct coordinate strings landing on the same pixel share one ID, and both
        labels ride along -- that is what makes `relabel_raster_grid_cell` fan the pixel's
        single data row back out into one row per point the caller asked about.
        """
        grid = GRIDS[RASTER_GRID]
        pixel_width = (grid.bounds[2] - grid.bounds[0]) / grid.width
        x, y = raster_xy(from_bounds(*grid.bounds, width=grid.width, height=grid.height), 10, 20)
        first = _lng_lat(x - pixel_width / 4, y)
        second = _lng_lat(x + pixel_width / 4, y)
        expected_id = 10 * grid.width + 20
        where, relabel = await resolve_grid_cell(
            RASTER_GRID, f"{first[0]},{first[1]},{second[0]},{second[1]}"
        )
        assert where == f"grid_cell = {expected_id}"
        assert relabel == {
            expected_id: [f"{first[0]},{first[1]}", f"{second[0]},{second[1]}"]
        }

    async def test_two_points_on_different_pixels_resolve_to_an_in_list(self):
        grid = GRIDS[RASTER_GRID]
        lng1, lat1 = _raster_pixel_lng_lat(10, 20)
        lng2, lat2 = _raster_pixel_lng_lat(100, 200)
        ids = sorted([10 * grid.width + 20, 100 * grid.width + 200])
        where, relabel = await resolve_grid_cell(RASTER_GRID, f"{lng1},{lat1},{lng2},{lat2}")
        assert where == f"grid_cell IN ({ids[0]}, {ids[1]})"
        assert sorted(relabel) == ids

    async def test_a_point_outside_the_extent_is_dropped_not_rejected(self):
        lng, lat = _far_south_lng_lat()
        where, relabel = await resolve_grid_cell(RASTER_GRID, f"{lng},{lat}")
        assert where == "grid_cell != grid_cell"
        assert relabel == {}

    async def test_digits_are_coordinates_on_a_raster_not_pixel_ids(self):
        """The asymmetry this module exists to preserve: `grid_cell=1,2` on a raster has
        always meant the point at lng=1, lat=2, so reading it as pixel IDs 1 and 2 (as a
        vector grid would) would reinterpret existing URLs rather than extend them. That
        point is nowhere near this grid's UTM zone, and the error names it as the
        coordinate pair it was parsed as.
        """
        with pytest.raises(RequestError) as excinfo:
            await resolve_grid_cell(RASTER_GRID, "1,2")
        assert "1.0, 2.0" in str(excinfo.value)

    async def test_the_raster_branch_opens_no_geometry_file(self):
        """No `geometry_path` fixture here either: a raster `grid_cell` is resolved from the
        grid's declared geometry alone, so nothing under `GRIDS_PATH` is ever consulted.
        """
        lng, lat = _raster_pixel_lng_lat(0, 0)
        where, _relabel = await resolve_grid_cell(RASTER_GRID, f"{lng},{lat}")
        assert where == "grid_cell = 0"


class TestResolveGridCellErrors:
    async def test_an_unknown_grid_is_not_found(self):
        with pytest.raises(NotFoundError):
            await resolve_grid_cell("no_such_grid_v001", "1,2")

    async def test_an_odd_number_of_coordinates_is_rejected(self, ecoregion_geometry):
        with pytest.raises(RequestError) as excinfo:
            await resolve_grid_cell(VECTOR_GRID, "-91.0,29.5,-91.5")
        assert resolve_status(excinfo.value) == 400

    async def test_a_non_numeric_coordinate_is_rejected(self, ecoregion_geometry):
        with pytest.raises(RequestError) as excinfo:
            await resolve_grid_cell(VECTOR_GRID, "-91.0,north")
        assert resolve_status(excinfo.value) == 400


# There is no `TestBuildDataQuery` here any more. `build_data_query` -- `resolve_grid_cell`
# composed with `request.data_query.build_data_query_pipeline`, plus the `relabel` and
# `float_precision` a route had to plumb separately -- is gone with the flat routes it was
# the entry point for. `api.unified.decode_request` is that composition now, for every
# route rather than three of them, and `test_unified_decoder.py` is where the cases that
# class pinned live: the trailing filter conjunct, the stepless no-filter pipeline, the
# coordinate `grid_cell` and its relabel map, and `float_precision` off the descriptor.
#
# What stays here is `resolve_grid_cell` itself, still a distinct unit with its own two
# grid-type paths and its own error vocabulary, and still worth testing apart from whatever
# assembles a pipeline around it.
