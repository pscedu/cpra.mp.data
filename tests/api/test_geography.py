"""Tests for converting between coordinates and grid cell IDs using geometry files."""

import polars as pl
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.transform import xy as raster_xy
from rasterio.warp import transform
from shapely.geometry import box

from cpra.mp.data.analysis.io.geography import (
    grid_cell_id_where,
    parse_coordinate_pairs,
    project_coordinates,
    raster_grid_cell_id,
    relabel_raster_grid_cell,
    representative_points,
    resolve_raster_pixel,
    resolve_raster_point_labels,
    resolve_vector_grid_cell,
    resolve_vector_grid_cell_sync,
)
from cpra.mp.data.analysis.request.errors import NotFoundError, RequestError
from cpra.mp.data.api.boundary import resolve_status
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
    """A plain `RequestError` -- 400 via `api.boundary`, not `NotFoundError`'s 404: the
    grid itself is real and the request names it correctly; what is absent is a server-side
    artifact the caller can neither see nor fix by asking for something else.
    """
    with pytest.raises(RequestError) as excinfo:
        await resolve_vector_grid_cell("ecoregion_v001", [-91.0], [29.5])
    assert resolve_status(excinfo.value) == 400
    assert "No geometry file" in str(excinfo.value)


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
        with pytest.raises(RequestError) as excinfo:
            project_coordinates([0.0], [0.0], CRS.from_epsg(26915))
        assert resolve_status(excinfo.value) == 400
        assert "outside the grid's projection" in str(excinfo.value)

    def test_only_the_unprojectable_pairs_are_reported(self):
        """PROJ fails the whole batch, so the pair at fault has to be found afterwards."""
        with pytest.raises(RequestError) as excinfo:
            project_coordinates([-91.0, 0.0], [30.0, 0.0], CRS.from_epsg(26915))
        assert "0.0, 0.0" in str(excinfo.value)
        assert "-91.0" not in str(excinfo.value)


class TestParseCoordinatePairs:
    def test_parses_pairs_in_order(self):
        assert parse_coordinate_pairs(["-91.0", "30.0", "-90.0", "29.0"]) == (
            [-91.0, -90.0],
            [30.0, 29.0],
        )

    def test_nan_is_an_invalid_coordinate(self):
        """`float` accepts "nan" and every bounds comparison against it is False, so a
        magnitude check would pass it through to project to infinity."""
        with pytest.raises(RequestError) as excinfo:
            parse_coordinate_pairs(["nan", "nan"])
        assert resolve_status(excinfo.value) == 400
        assert "invalid coordinate pair" in str(excinfo.value)

    def test_infinity_is_an_invalid_coordinate(self):
        with pytest.raises(RequestError) as excinfo:
            parse_coordinate_pairs(["inf", "0"])
        assert resolve_status(excinfo.value) == 400
        assert "invalid coordinate pair" in str(excinfo.value)


async def test_unprojectable_point_raises_400(ecoregion_geometry):
    """The vector lookup projects before it reads, so it fails the same way."""
    with pytest.raises(RequestError) as excinfo:
        await resolve_vector_grid_cell("ecoregion_v001", [0.0], [0.0])
    assert resolve_status(excinfo.value) == 400
    assert "outside the grid's projection" in str(excinfo.value)


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
        minx, miny, maxx, _maxy = grid.bounds
        center_x = (minx + maxx) / 2
        lng, lat = [c[0] for c in transform(CRS.from_epsg(26915), CRS.from_epsg(4326), [center_x], [miny - 10_000])]
        assert resolve_raster_pixel(RASTER_GRID, [lng], [lat]) == [None]

    def test_mixed_in_and_out_of_extent_points_resolve_independently(self):
        """Each point is resolved on its own -- one point landing outside the grid must not
        affect another point in the same call that lands inside it.
        """
        grid = GRIDS[RASTER_GRID]
        minx, miny, maxx, _maxy = grid.bounds
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
        with pytest.raises(RequestError) as excinfo:
            resolve_raster_pixel(RASTER_GRID, [0.0], [0.0])
        assert resolve_status(excinfo.value) == 400
        assert "outside the grid's projection" in str(excinfo.value)

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

    def test_a_missing_raster_grid_is_a_404_not_a_400(self):
        """The whole point of this module raising `NotFoundError` here rather than the
        plain `RequestError` every coordinate rejection uses: naming a grid that does not
        exist is a different answer from sending a malformed coordinate, and
        `api.boundary` is where that difference becomes 404 vs 400.
        """
        with pytest.raises(NotFoundError) as excinfo:
            resolve_raster_pixel("not_a_real_grid_at_all", [-91.0], [29.5])
        assert resolve_status(excinfo.value) == 404


class TestRasterGridCellId:
    """`raster_grid_cell_id` computes FORWARD, from a known absolute pixel, the same
    ABSOLUTE id `execution.band_ops.sample_grid_cells_band_wide` would emit for that pixel
    regardless of what window an actual read covered -- see that function's own docstring
    for why the two are the same arithmetic, and why no window is involved at all any more.
    """

    def test_matches_row_major_numbering_over_the_grids_own_width(self):
        row, col = 10, 20
        grid = GRIDS[RASTER_GRID]
        assert raster_grid_cell_id(RASTER_GRID, (row, col)) == row * grid.width + col

    def test_the_origin_pixel_is_cell_zero(self):
        assert raster_grid_cell_id(RASTER_GRID, (0, 0)) == 0

    def test_the_id_is_independent_of_any_window(self):
        """The whole point of the fix this test guards: an id must mean the same pixel
        whether or not a read of it was ever narrowed to some window smaller than the full
        grid -- so there is no `window` argument left to pass here at all, unlike the
        window-relative id this function used to compute.
        """
        grid = GRIDS[RASTER_GRID]
        row, col = 200, 800
        assert raster_grid_cell_id(RASTER_GRID, (row, col)) == row * grid.width + col


class TestGridCellIdWhere:
    """Shared by both grid kinds' non-digit `grid_cell` handling once a coordinate has
    already been resolved down to a plain list of ints.
    """

    def test_a_single_id_is_an_equality_test(self):
        assert grid_cell_id_where([7]) == "grid_cell = 7"

    def test_several_ids_are_an_in_list(self):
        assert grid_cell_id_where([3, 1, 2]) == "grid_cell IN (1, 2, 3)"

    def test_duplicate_ids_collapse_to_a_single_equality_test(self):
        assert grid_cell_id_where([5, 5, 5]) == "grid_cell = 5"

    def test_no_ids_is_the_pipelines_own_matched_nothing_convention(self):
        assert grid_cell_id_where([]) == "grid_cell != grid_cell"


def _raster_point_label(row: int, col: int) -> tuple[str, str]:
    """A raw `(lng_token, lat_token)` pair for the CENTER of pixel `(row, col)` on
    `RASTER_GRID`, formatted with trailing zeros a plain `float(...)!s` round trip would
    strip -- so a test asserting the resolved label equals this exact string actually
    proves nothing was reformatted, rather than passing merely because `str(float(x)) ==
    x` already held for a plainly-formatted value.
    """
    lng, lat = _raster_pixel_lng_lat(row, col)
    return f"{lng:.10f}0", f"{lat:.10f}0"


class TestResolveRasterPointLabels:
    """`resolve_raster_point_labels` is the raster point-query-specific half of Milestone
    3: resolving each raw token pair to an ABSOLUTE id, while keeping the caller's own
    strings verbatim for `relabel_raster_grid_cell` to hand back later.
    """

    def test_the_returned_label_is_the_callers_own_string_verbatim(self):
        """The deliberate departure from the OLD engine's `f"{x!s},{y!s}"` convention,
        which reformats through `float()` and back (e.g. `"-90.10"` -> `"-90.1"`). Here the
        trailing zero must survive untouched.
        """
        row, col = 50, 60
        lng_token, lat_token = _raster_point_label(row, col)
        tokens = [lng_token, lat_token]
        cell_id = raster_grid_cell_id(RASTER_GRID, (row, col))

        resolved = resolve_raster_point_labels(RASTER_GRID, tokens)

        assert resolved == {cell_id: [f"{lng_token},{lat_token}"]}

    def test_two_points_on_the_same_pixel_both_keep_their_own_distinct_label(self):
        """Two different requested points that happen to resolve to the same pixel must
        not collapse into one label -- each of the caller's original strings for that pixel
        is preserved, in request order, ready for `relabel_raster_grid_cell` to fan back
        out into two rows.
        """
        row, col = 50, 60
        lng_token, lat_token = _raster_point_label(row, col)
        first_label = f"{lng_token},{lat_token}"
        second_label = f"{lng_token}000,{lat_token}000"
        tokens = [lng_token, lat_token, f"{lng_token}000", f"{lat_token}000"]
        cell_id = raster_grid_cell_id(RASTER_GRID, (row, col))

        resolved = resolve_raster_point_labels(RASTER_GRID, tokens)

        assert resolved == {cell_id: [first_label, second_label]}

    def test_a_point_outside_the_extent_is_absent_not_an_error(self):
        """Mirrors `resolve_raster_pixel`'s own `None`-for-out-of-extent convention: the
        point simply does not appear in the returned map at all.
        """
        grid = GRIDS[RASTER_GRID]
        minx, miny, maxx, _maxy = grid.bounds
        center_x = (minx + maxx) / 2
        lng, lat = [c[0] for c in transform(CRS.from_epsg(26915), CRS.from_epsg(4326), [center_x], [miny - 10_000])]

        resolved = resolve_raster_point_labels(RASTER_GRID, [str(lng), str(lat)])

        assert resolved == {}

    def test_an_in_extent_point_is_kept_when_paired_with_an_out_of_extent_one(self):
        """Each point is resolved independently -- one point landing outside the grid must
        not drop another point's label from the result.
        """
        grid = GRIDS[RASTER_GRID]
        minx, miny, maxx, _maxy = grid.bounds
        in_lng_token, in_lat_token = _raster_point_label(0, 0)
        center_x = (minx + maxx) / 2
        out_lng, out_lat = [
            c[0] for c in transform(CRS.from_epsg(26915), CRS.from_epsg(4326), [center_x], [miny - 10_000])
        ]
        tokens = [in_lng_token, in_lat_token, str(out_lng), str(out_lat)]
        cell_id = raster_grid_cell_id(RASTER_GRID, (0, 0))

        resolved = resolve_raster_point_labels(RASTER_GRID, tokens)

        assert resolved == {cell_id: [f"{in_lng_token},{in_lat_token}"]}


class TestRelabelRasterGridCell:
    """`relabel_raster_grid_cell` replaces an executed frame's ABSOLUTE integer
    `grid_cell` with the caller's own coordinate string(s) -- an inner join, so one id
    mapped to several labels fans a single data row out into several.
    """

    def test_replaces_the_id_with_the_single_matching_label(self):
        lf = pl.LazyFrame({"grid_cell": [42], "value": [1.5]})

        result = relabel_raster_grid_cell(lf, {42: ["-90.10,29.90"]}).collect()

        assert result.to_dicts() == [{"grid_cell": "-90.10,29.90", "value": 1.5}]

    def test_two_labels_on_one_id_fan_out_into_two_rows(self):
        """The property Milestone 3 explicitly calls out: two requested points landing on
        the same pixel must produce two rows, each with its own distinct original string --
        not one row, and not a row with both strings run together.
        """
        lf = pl.LazyFrame({"grid_cell": [42], "value": [1.5]})

        result = relabel_raster_grid_cell(lf, {42: ["-90.10,29.90", "-90.1,29.9"]}).collect()

        assert result.height == 2
        assert set(result["grid_cell"]) == {"-90.10,29.90", "-90.1,29.9"}
        assert list(result["value"]) == [1.5, 1.5]

    def test_a_row_whose_id_has_no_label_is_dropped(self):
        """An inner join: a `grid_cell` value with nothing in `id_to_labels` simply does
        not appear in the output, the same as any other row a `where` would exclude.
        """
        lf = pl.LazyFrame({"grid_cell": [42, 99], "value": [1.5, 2.5]})

        result = relabel_raster_grid_cell(lf, {42: ["-90.10,29.90"]}).collect()

        assert result.to_dicts() == [{"grid_cell": "-90.10,29.90", "value": 1.5}]
