"""Tests for converting grid_cells between grids.

The four conversions have nothing in common but their output table: two read a crosswalk
file, one reads the source grid's polygons, and one is a no-op. What binds them is that
each has to name a grid_cell the destination grid actually accepts — a cell ID in `dst_id`
for a vector grid, a coordinate in `dst_lng`/`dst_lat` for a raster one — so that it can be
handed straight back to the data API as a `grid_cell` parameter.
"""

import io
import threading

import numpy as np
import polars as pl
import pytest
from fastapi import HTTPException
from rasterio.crs import CRS
from rasterio.warp import transform
from shapely.geometry import Point, Polygon, box

from cpra.mp.data.analysis import AnalysisError
from cpra.mp.data.api import crosswalk as crosswalk_module
from cpra.mp.data.api.crosswalk import Crosswalk
from cpra.mp.data.grids import GRIDS, WGS84
from tests.factories import (
    write_grid_geometry,
    write_raster_crosswalk,
    write_vector_crosswalk,
)

VECTOR_GRID = "hydro_compartment_v001"
RASTER_GRID = "veg_grid_cell_v001"
GRID = GRIDS[RASTER_GRID]

URL = "/v1/grids/{grid}/crosswalk.{format}"

COLUMNS = [
    "src_id",
    "src_lng",
    "src_lat",
    "dst_id",
    "dst_lng",
    "dst_lat",
    "scale_factor",
    "dst_grid",
]

# A point in the northern half of the raster grid, which the zone crosswalk below maps to
# ecoregion 1, and one in the southern half, which it maps to nothing.
NORTH = (657190, 3350000)
SOUTH = (657190, 3220000)


def _lng_lat(x, y, grid_name=RASTER_GRID):
    """Convert a point in a grid's own CRS to the WGS84 pair the API takes."""
    crs = CRS.from_user_input(GRIDS[grid_name].crs)
    lng, lat = [c[0] for c in transform(crs, CRS.from_epsg(4326), [x], [y])]
    return lng, lat


def _projected(lng, lat, epsg=26915):
    """Convert a WGS84 pair from the output table back into a grid's own CRS."""
    x, y = [c[0] for c in transform(CRS.from_epsg(4326), CRS.from_epsg(epsg), [lng], [lat])]
    return x, y


def _geography(*points, grid_name=RASTER_GRID):
    """Format projected points as a `grid_cell` query value."""
    return ",".join(str(c) for point in points for c in _lng_lat(*point, grid_name=grid_name))


def _u_shape(x, y, size=1000):
    """A U with its opening at the top, whose centroid falls in the gap between the arms."""
    corners = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]
    return Polygon([(x + cx * size, y + cy * size) for cx, cy in corners])


@pytest.fixture
def vector_crosswalk(crosswalk_path):
    """Compartment 1 split evenly between two ecoregions, compartment 2 wholly in one."""
    write_vector_crosswalk(
        crosswalk_path,
        VECTOR_GRID,
        "ecoregion_v001",
        [(1, 10, 0.5), (1, 11, 0.5), (2, 11, 1.0)],
    )
    return crosswalk_path


def _northern_zones():
    """Zone IDs covering the raster grid's northern half with zone 1, the rest with none."""
    zones = np.zeros((GRID.height, GRID.width), dtype="int32")
    zones[: GRID.height // 2, :] = 1
    return zones


@pytest.fixture
def zone_crosswalk(crosswalk_path):
    """A zone crosswalk covering the raster grid's northern half with ecoregion 1."""
    write_raster_crosswalk(crosswalk_path, RASTER_GRID, "ecoregion_v001", _northern_zones())
    return crosswalk_path


@pytest.fixture
def vector_geometry(geometry_path):
    """Two compartments: a square, and a U whose centroid lies outside itself."""
    return write_grid_geometry(
        geometry_path,
        VECTOR_GRID,
        [(1, box(500000, 3300000, 501000, 3301000)), (2, _u_shape(600000, 3300000))],
    )


class TestDestinationGrids:
    """Which grids a source converts to when the request doesn't say."""

    def test_defaults_to_grids_with_a_crosswalk_file(self, vector_crosswalk):
        assert Crosswalk(VECTOR_GRID).destination_grids == ["ecoregion_v001"]

    def test_a_raster_source_also_reaches_every_other_raster_grid(self, zone_crosswalk):
        """Raster-to-raster needs no file, so those destinations can't be discovered by
        looking for one."""
        assert Crosswalk(RASTER_GRID).destination_grids == [
            "ecoregion_v001",
            "morph_pixel_v001",
            "morph_pixel_v002",
            "veg_grid_cell_v002",
        ]

    def test_wgs84_reaches_the_grids_that_can_place_a_coordinate(self, vector_geometry):
        """A vector grid whose polygons are on hand, and every raster grid, whose values
        are at the coordinate itself. Neither is discovered from a crosswalk file."""
        assert Crosswalk(WGS84.name).destination_grids == [
            VECTOR_GRID,
            "morph_pixel_v001",
            "morph_pixel_v002",
            "veg_grid_cell_v001",
            "veg_grid_cell_v002",
        ]

    def test_the_source_grid_is_not_a_destination(self, zone_crosswalk):
        assert RASTER_GRID not in Crosswalk(RASTER_GRID).destination_grids

    def test_no_reachable_grid_is_a_400(self, crosswalk_path):
        with pytest.raises(HTTPException) as excinfo:
            Crosswalk(VECTOR_GRID)
        assert excinfo.value.status_code == 400
        assert "No crosswalks from grid: hydro_compartment_v001" in excinfo.value.detail

    def test_requested_grids_are_used_in_the_order_given(self, vector_crosswalk):
        crosswalk = Crosswalk(VECTOR_GRID, to_grid=["parish_v2020", "ecoregion_v001"])
        assert crosswalk.destination_grids == ["parish_v2020", "ecoregion_v001"]

    def test_a_repeated_destination_is_converted_once(self, vector_crosswalk):
        crosswalk = Crosswalk(VECTOR_GRID, to_grid=["ecoregion_v001", "ecoregion_v001"])
        assert crosswalk.destination_grids == ["ecoregion_v001"]

    async def test_a_requested_grid_with_no_crosswalk_file_is_a_400(self, vector_crosswalk):
        """Unlike an unmapped grid_cell, which is simply left out: asking for a conversion
        that can't be made at all is a mistake in the request."""
        crosswalk = Crosswalk(VECTOR_GRID, grid_cell="1", to_grid=["parish_v2020"])
        with pytest.raises(HTTPException) as excinfo:
            await crosswalk.execute()
        assert excinfo.value.status_code == 400
        assert "No crosswalk from hydro_compartment_v001 to parish_v2020" in excinfo.value.detail


class TestVectorToVector:
    """The crosswalk table's own rows, filtered and reordered but not otherwise touched."""

    async def test_rows_are_returned_as_stored(self, vector_crosswalk):
        df = await Crosswalk(VECTOR_GRID, grid_cell="2").execute()
        assert df.select("src_id", "dst_id", "scale_factor").rows() == [(2, 11, 1.0)]

    async def test_a_split_geography_keeps_a_row_per_destination(self, vector_crosswalk):
        """Compartment 1 straddles two ecoregions, and the scale factors say how much of
        it went to each."""
        df = await Crosswalk(VECTOR_GRID, grid_cell="1").execute()
        assert df.select("dst_id", "scale_factor").rows() == [(10, 0.5), (11, 0.5)]

    async def test_geographies_are_reported_in_the_order_requested(self, vector_crosswalk):
        """The stored table is ordered by destination, which a caller matching rows back to
        its own request would have to undo."""
        df = await Crosswalk(VECTOR_GRID, grid_cell="2,1").execute()
        assert df["src_id"].to_list() == [2, 1, 1]

    async def test_an_unmapped_geography_is_left_out(self, vector_crosswalk):
        df = await Crosswalk(VECTOR_GRID, grid_cell="1,99").execute()
        assert df["src_id"].unique().to_list() == [1]

    async def test_no_geography_returns_the_whole_table(self, vector_crosswalk):
        df = await Crosswalk(VECTOR_GRID).execute()
        assert df.select("src_id", "dst_id").rows() == [(1, 10), (1, 11), (2, 11)]

    async def test_a_coordinate_selects_the_geography_containing_it(
        self, vector_crosswalk, vector_geometry
    ):
        """A vector source accepts coordinates on the same terms as the data API's
        grid_cell filter, and reports the cell they resolved to."""
        grid_cell = _geography((500500, 3300500), grid_name=VECTOR_GRID)
        df = await Crosswalk(VECTOR_GRID, grid_cell=grid_cell).execute()
        assert df["src_id"].unique().to_list() == [1]


class TestVectorToRaster:
    """A vector grid_cell converts to a coordinate, since a raster grid has no IDs."""

    async def test_the_point_is_inside_the_geography(self, vector_crosswalk, vector_geometry):
        df = await Crosswalk(VECTOR_GRID, grid_cell="1", to_grid=[RASTER_GRID]).execute()
        x, y = _projected(df["dst_lng"][0], df["dst_lat"][0])
        assert box(500000, 3300000, 501000, 3301000).contains(Point(x, y))

    async def test_a_concave_geography_still_gets_a_point_inside_itself(
        self, vector_crosswalk, vector_geometry
    ):
        """The reason for a point on surface rather than a centroid: compartment 2's
        centroid lands in the gap between its arms, and would name a location the
        grid_cell doesn't cover."""
        shape = _u_shape(600000, 3300000)
        assert not shape.contains(shape.centroid)

        df = await Crosswalk(VECTOR_GRID, grid_cell="2", to_grid=[RASTER_GRID]).execute()
        assert shape.contains(Point(*_projected(df["dst_lng"][0], df["dst_lat"][0])))

    async def test_the_destination_raster_grid_does_not_change_the_point(
        self, vector_crosswalk, vector_geometry
    ):
        """The point comes from the source polygon; every raster grid covers it."""
        crosswalk = Crosswalk(VECTOR_GRID, grid_cell="1", to_grid=[RASTER_GRID, "morph_pixel_v001"])
        assert crosswalk.destination_grids == [RASTER_GRID, "morph_pixel_v001"]
        df = await crosswalk.execute()
        assert df.select("dst_lng", "dst_lat").row(0) == df.select("dst_lng", "dst_lat").row(1)

    async def test_a_geography_with_no_polygon_is_left_out(self, vector_crosswalk, vector_geometry):
        df = await Crosswalk(VECTOR_GRID, grid_cell="1,99", to_grid=[RASTER_GRID]).execute()
        assert df["src_id"].to_list() == [1]

    async def test_no_geography_matches_a_polygon(self, vector_crosswalk, vector_geometry):
        df = await Crosswalk(VECTOR_GRID, grid_cell="99", to_grid=[RASTER_GRID]).execute()
        assert df.height == 0

    async def test_coordinates_that_match_nothing_convert_to_nothing(
        self, vector_crosswalk, vector_geometry
    ):
        """The coordinates resolve to no source grid_cell at all, so there is nothing left
        to look up a point for."""
        grid_cell = _geography((400000, 3200000), grid_name=VECTOR_GRID)
        df = await Crosswalk(VECTOR_GRID, grid_cell=grid_cell, to_grid=[RASTER_GRID]).execute()
        assert df.height == 0

    async def test_every_geography_is_converted_when_none_is_given(
        self, vector_crosswalk, vector_geometry
    ):
        df = await Crosswalk(VECTOR_GRID, to_grid=[RASTER_GRID]).execute()
        assert df["src_id"].to_list() == [1, 2]

    async def test_a_missing_geometry_file_is_a_400(self, vector_crosswalk, geometry_path):
        crosswalk = Crosswalk(VECTOR_GRID, grid_cell="1", to_grid=[RASTER_GRID])
        with pytest.raises(AnalysisError) as excinfo:
            await crosswalk.execute()
        assert excinfo.value.status_code == 400
        assert "No geometry file for grid" in excinfo.value.detail


class TestRasterToVector:
    """Coordinates convert through the zone-ID crosswalk raster."""

    async def test_a_coordinate_becomes_the_zone_containing_it(self, zone_crosswalk):
        crosswalk = Crosswalk(RASTER_GRID, grid_cell=_geography(NORTH), to_grid=["ecoregion_v001"])
        df = await crosswalk.execute()
        assert df["dst_id"].to_list() == [1]

    async def test_the_coordinate_is_reported_as_the_source(self, zone_crosswalk):
        lng, lat = _lng_lat(*NORTH)
        crosswalk = Crosswalk(RASTER_GRID, grid_cell=_geography(NORTH), to_grid=["ecoregion_v001"])
        df = await crosswalk.execute()
        assert df.select("src_id", "src_lng", "src_lat").rows() == [(None, lng, lat)]

    async def test_a_coordinate_outside_every_zone_is_left_out(self, zone_crosswalk):
        grid_cell = _geography(NORTH, SOUTH)
        df = await Crosswalk(RASTER_GRID, grid_cell=grid_cell, to_grid=["ecoregion_v001"]).execute()
        assert df.height == 1

    async def test_a_repeated_coordinate_is_converted_once(self, zone_crosswalk):
        grid_cell = _geography(NORTH, NORTH)
        df = await Crosswalk(RASTER_GRID, grid_cell=grid_cell, to_grid=["ecoregion_v001"]).execute()
        assert df.height == 1

    async def test_geography_is_required(self, zone_crosswalk):
        """There is no enumerable set of pixels to convert without one."""
        crosswalk = Crosswalk(RASTER_GRID, to_grid=["ecoregion_v001"])
        with pytest.raises(HTTPException) as excinfo:
            await crosswalk.execute()
        assert excinfo.value.status_code == 400
        assert "grid_cell is required for raster grid" in excinfo.value.detail


class TestRasterToRaster:
    """Nothing to convert: a raster grid_cell is the location itself."""

    async def test_the_coordinate_is_returned_unchanged(self, zone_crosswalk):
        lng, lat = _lng_lat(*NORTH)
        crosswalk = Crosswalk(RASTER_GRID, grid_cell=f"{lng},{lat}", to_grid=["morph_pixel_v001"])
        df = await crosswalk.execute()
        assert df.select("src_lng", "src_lat", "dst_lng", "dst_lat").rows() == [(lng, lat, lng, lat)]

    async def test_no_crosswalk_file_is_needed(self, crosswalk_path):
        crosswalk = Crosswalk(RASTER_GRID, grid_cell="-91.5,29.5", to_grid=["morph_pixel_v001"])
        df = await crosswalk.execute()
        assert df.height == 1

    async def test_a_coordinate_outside_the_grid_is_still_returned(self, crosswalk_path):
        """A location is a location; whether either grid has data there is the data
        endpoint's business."""
        crosswalk = Crosswalk(RASTER_GRID, grid_cell="-80.0,40.0", to_grid=["morph_pixel_v001"])
        df = await crosswalk.execute()
        assert df.select("dst_lng", "dst_lat").rows() == [(-80.0, 40.0)]


class TestWGS84ToVector:
    """Coordinates as source grid_cells, resolved from the destination grid's polygons.

    The lookup is the data API's own, so a coordinate converts to the grid_cell whose data
    it selects — which is what a caller matching a chart to a map depends on, and what
    sampling a zone raster only approximates.
    """

    def _crosswalk(self, *points, to_grid=None):
        grid_cell = _geography(*points, grid_name=VECTOR_GRID)
        return Crosswalk(WGS84.name, grid_cell=grid_cell, to_grid=to_grid or [VECTOR_GRID])

    async def test_a_coordinate_becomes_the_geography_containing_it(self, vector_geometry):
        df = await self._crosswalk((500500, 3300500)).execute()
        assert df["dst_id"].to_list() == [1]

    async def test_the_coordinate_is_reported_as_the_source(self, vector_geometry):
        """There is no source ID to report: the coordinate is the source grid_cell."""
        point = (500500, 3300500)
        df = await self._crosswalk(point).execute()
        lng, lat = _lng_lat(*point, grid_name=VECTOR_GRID)
        assert df.select("src_id", "src_lng", "src_lat").rows() == [(None, lng, lat)]

    async def test_every_coordinate_gets_its_own_row(self, vector_geometry):
        """Two coordinates in one grid_cell are two rows, so that every coordinate asked
        about is answered for by name rather than by its place in the results."""
        points = [(500400, 3300400), (500600, 3300600)]
        df = await self._crosswalk(*points).execute()
        assert df.select("src_lng", "src_lat", "dst_id").rows() == [
            (*_lng_lat(*point, grid_name=VECTOR_GRID), 1) for point in points
        ]

    async def test_a_coordinate_outside_every_geography_is_left_out(self, vector_geometry):
        """A point in the gap between compartment 2's arms: inside the bounding box the
        geometry file is indexed by, outside the polygon itself."""
        gap = (601500, 3302000)
        assert not _u_shape(600000, 3300000).contains(Point(*gap))
        assert (await self._crosswalk(gap).execute()).height == 0

    async def test_a_repeated_coordinate_is_converted_once(self, vector_geometry):
        point = (500500, 3300500)
        assert (await self._crosswalk(point, point).execute()).height == 1

    async def test_an_integer_is_a_coordinate_rather_than_an_id(self, vector_geometry):
        """A vector source reads a whole number as a cell ID, but `wgs84` has no cells to
        name — every value it takes is part of a location."""
        crosswalk = Crosswalk(WGS84.name, grid_cell="-91,30", to_grid=[VECTOR_GRID])
        await crosswalk.execute()
        assert crosswalk.source_ids is None
        assert crosswalk.source_coordinates == [(-91.0, 30.0)]

    async def test_geography_is_required(self, vector_geometry):
        """There is nothing to convert without one: the coordinates are the grid_cells."""
        crosswalk = Crosswalk(WGS84.name, to_grid=[VECTOR_GRID])
        with pytest.raises(HTTPException) as excinfo:
            await crosswalk.execute()
        assert excinfo.value.status_code == 400
        assert "grid_cell is required for coordinate grid: wgs84" in excinfo.value.detail

    async def test_a_missing_geometry_file_is_a_400(self, geometry_path):
        crosswalk = self._crosswalk((500500, 3300500), to_grid=["parish_v2020"])
        with pytest.raises(AnalysisError) as excinfo:
            await crosswalk.execute()
        assert excinfo.value.status_code == 400
        assert "No geometry file for grid: parish_v2020" in excinfo.value.detail

    async def test_a_raster_destination_returns_the_coordinate_unchanged(self, geometry_path):
        """The same conversion a raster source makes to a raster grid: the coordinate is
        already the grid_cell, and no geometry is read to say so."""
        crosswalk = Crosswalk(WGS84.name, grid_cell="-91.5,29.5", to_grid=[RASTER_GRID])
        df = await crosswalk.execute()
        assert df.select("src_lng", "src_lat", "dst_lng", "dst_lat").rows() == [
            (-91.5, 29.5, -91.5, 29.5)
        ]


class TestOutputTable:
    """One table for several destinations, so every column has to hold every conversion."""

    async def test_the_destination_grid_is_named_on_every_row(self, vector_crosswalk):
        df = await Crosswalk(VECTOR_GRID, grid_cell="1").execute()
        assert df["dst_grid"].unique().to_list() == ["ecoregion_v001"]

    async def test_destinations_are_stacked_in_order(self, vector_crosswalk, vector_geometry):
        crosswalk = Crosswalk(VECTOR_GRID, grid_cell="2", to_grid=[RASTER_GRID, "ecoregion_v001"])
        df = await crosswalk.execute()
        assert df["dst_grid"].to_list() == [RASTER_GRID, "ecoregion_v001"]

    async def test_the_column_types_do_not_depend_on_the_grids(
        self, vector_crosswalk, vector_geometry
    ):
        """The whole point of separate ID and coordinate columns: a raster destination adds
        coordinates without changing what the ID column holds for the vector one."""
        vector_only = await Crosswalk(VECTOR_GRID, grid_cell="2").execute()
        crosswalk = Crosswalk(VECTOR_GRID, grid_cell="2", to_grid=["ecoregion_v001", RASTER_GRID])
        both = await crosswalk.execute()

        assert both.schema == vector_only.schema
        assert both.schema["src_id"] == pl.Int64
        assert both.schema["dst_id"] == pl.Int64
        assert both.schema["dst_lng"] == pl.Float64
        assert both.schema["scale_factor"] == pl.Float64

    async def test_only_the_columns_a_conversion_uses_are_populated(
        self, vector_crosswalk, vector_geometry
    ):
        """Compartment 2 to an ecoregion and to a raster grid: the first names a cell, the
        second a location, and neither fills in the other's columns."""
        crosswalk = Crosswalk(VECTOR_GRID, grid_cell="2", to_grid=["ecoregion_v001", RASTER_GRID])
        df = await crosswalk.execute()

        assert df.select("src_id", "src_lng", "src_lat").rows() == [(2, None, None), (2, None, None)]
        assert df.select("dst_id", "dst_lng", "dst_lat").row(0) == (11, None, None)
        assert df.select("dst_id").row(1) == (None,)

    async def test_an_empty_result_still_has_the_full_schema(self, vector_crosswalk):
        df = await Crosswalk(VECTOR_GRID, grid_cell="99").execute()
        assert df.height == 0
        assert df.columns == COLUMNS

    async def test_every_conversion_produces_the_same_columns(
        self, vector_crosswalk, zone_crosswalk, vector_geometry
    ):
        vector_source = Crosswalk(VECTOR_GRID, grid_cell="1", to_grid=["ecoregion_v001", RASTER_GRID])
        raster_source = Crosswalk(
            RASTER_GRID, grid_cell=_geography(NORTH), to_grid=["ecoregion_v001", "morph_pixel_v001"]
        )
        wgs84_source = Crosswalk(
            WGS84.name,
            grid_cell=_geography((500500, 3300500), grid_name=VECTOR_GRID),
            to_grid=[VECTOR_GRID, "morph_pixel_v001"],
        )
        schema = (await vector_source.execute()).schema
        assert (await raster_source.execute()).schema == schema
        assert (await wgs84_source.execute()).schema == schema


class TestParallelConversion:
    """Destinations are independent of each other, so they don't wait on each other."""

    async def test_destinations_are_converted_at_the_same_time(
        self, monkeypatch, zone_crosswalk, crosswalk_path
    ):
        """A barrier only two conversions together can clear: were they run one after the
        other, the first would block waiting for a partner that hadn't started yet, and
        time out rather than pass."""
        write_raster_crosswalk(crosswalk_path, RASTER_GRID, "parish_v2020", _northern_zones())
        barrier = threading.Barrier(2, timeout=10)
        convert = Crosswalk._raster_to_vector

        def _wait_for_the_other(self, dst_grid_name):
            barrier.wait()
            return convert(self, dst_grid_name)

        monkeypatch.setattr(Crosswalk, "_raster_to_vector", _wait_for_the_other)
        crosswalk = Crosswalk(
            RASTER_GRID, grid_cell=_geography(NORTH), to_grid=["ecoregion_v001", "parish_v2020"]
        )
        df = await crosswalk.execute()
        assert df["dst_grid"].to_list() == ["ecoregion_v001", "parish_v2020"]

    async def test_the_geometry_file_is_read_once_for_all_raster_destinations(
        self, monkeypatch, vector_crosswalk, vector_geometry
    ):
        """Every raster destination gets the same points, so reading them per destination
        would be both wasted work and a race between the threads doing it."""
        reads = []
        read_points = crosswalk_module.representative_points

        def _counted(grid_name, geography_ids=None):
            reads.append(grid_name)
            return read_points(grid_name, geography_ids)

        monkeypatch.setattr(crosswalk_module, "representative_points", _counted)
        crosswalk = Crosswalk(VECTOR_GRID, grid_cell="1", to_grid=[RASTER_GRID, "morph_pixel_v001"])
        df = await crosswalk.execute()
        assert reads == [VECTOR_GRID]
        assert df.height == 2

    async def test_the_geometry_file_is_left_alone_without_a_raster_destination(
        self, monkeypatch, vector_crosswalk
    ):
        def _fail(*args, **kwargs):
            raise AssertionError("geometry should not be read for a vector destination")

        monkeypatch.setattr(crosswalk_module, "representative_points", _fail)
        assert (await Crosswalk(VECTOR_GRID, grid_cell="1").execute()).height == 2

    async def test_one_failed_conversion_fails_the_request(self, zone_crosswalk):
        """The bad destination is converted alongside a good one, and still reports its
        own 400 rather than being lost among the results."""
        crosswalk = Crosswalk(
            RASTER_GRID,
            grid_cell=_geography(NORTH),
            to_grid=["ecoregion_v001", "hydro_compartment_v001"],
        )
        with pytest.raises(HTTPException) as excinfo:
            await crosswalk.execute()
        assert excinfo.value.status_code == 400
        assert "to hydro_compartment_v001" in excinfo.value.detail


class TestCrosswalkEndpoint:
    def _get(self, client, grid, format="csv", **params):
        query = "&".join(
            f"{key}={value}"
            for key, values in params.items()
            for value in (values if isinstance(values, list) else [values])
        )
        return client.get(f"{URL.format(grid=grid, format=format)}?{query}")

    def test_csv_carries_the_crosswalk_columns(self, client, vector_crosswalk):
        response = self._get(client, VECTOR_GRID, grid_cell="1")
        df = pl.read_csv(io.BytesIO(response.content))
        assert df.columns == COLUMNS

    def test_parquet_keeps_the_geography_types(self, client, vector_crosswalk):
        response = self._get(client, VECTOR_GRID, format="parquet", grid_cell="1")
        df = pl.read_parquet(io.BytesIO(response.content))
        assert df.schema["src_id"] == pl.Int64
        assert df.schema["dst_id"] == pl.Int64
        assert df.schema["dst_lng"] == pl.Float64

    def test_to_grid_can_be_repeated(self, client, vector_crosswalk, vector_geometry):
        response = self._get(
            client, VECTOR_GRID, grid_cell="2", to_grid=["ecoregion_v001", RASTER_GRID]
        )
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["dst_grid"].to_list() == ["ecoregion_v001", RASTER_GRID]

    def test_a_raster_source_converts_coordinates(self, client, zone_crosswalk):
        response = self._get(
            client, RASTER_GRID, grid_cell=_geography(NORTH), to_grid="ecoregion_v001"
        )
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["dst_id"].to_list() == [1]

    def test_wgs84_is_a_source_grid(self, client, vector_geometry):
        """The one source that isn't a grid of the model's, so the route has to accept a
        name that the grid catalog doesn't list."""
        response = self._get(
            client,
            WGS84.name,
            format="ndjson",
            grid_cell=_geography((500500, 3300500), grid_name=VECTOR_GRID),
            to_grid=VECTOR_GRID,
        )
        assert response.status_code == 200
        df = pl.read_ndjson(io.BytesIO(response.content))
        assert df.columns == COLUMNS
        assert df["dst_id"].to_list() == [1]

    def test_wgs84_is_not_a_destination_grid(self, client, vector_crosswalk):
        """Nothing to convert to: a destination coordinate is what a raster grid already
        reports."""
        response = self._get(client, VECTOR_GRID, grid_cell="1", to_grid=WGS84.name)
        assert response.status_code == 422

    def test_an_unknown_grid_is_rejected(self, client, vector_crosswalk):
        assert self._get(client, "not_a_grid", grid_cell="1").status_code == 422

    def test_an_unknown_destination_grid_is_rejected(self, client, vector_crosswalk):
        response = self._get(client, VECTOR_GRID, grid_cell="1", to_grid="not_a_grid")
        assert response.status_code == 422

    def test_a_raster_format_is_rejected(self, client, vector_crosswalk):
        response = self._get(client, VECTOR_GRID, format="tif", grid_cell="1")
        assert response.status_code == 400
        assert "must be one of" in response.json()["detail"]

    def test_a_raster_compression_algorithm_is_rejected(self, client, vector_crosswalk):
        response = self._get(client, VECTOR_GRID, grid_cell="1", compression_algorithm="lzw")
        assert response.status_code == 400

    def test_a_missing_crosswalk_is_reported(self, client, vector_crosswalk):
        response = self._get(client, VECTOR_GRID, grid_cell="1", to_grid="parish_v2020")
        assert response.status_code == 400
        assert response.json()["detail"].startswith("No crosswalk from")

    def test_non_numeric_coordinates_are_rejected(self, client, zone_crosswalk):
        response = self._get(client, RASTER_GRID, grid_cell="nope,29.5", to_grid="ecoregion_v001")
        assert response.status_code == 400
        assert "non-numeric" in response.json()["detail"]

    def test_unprojectable_coordinates_are_rejected(self, client, zone_crosswalk):
        """Sampling the crosswalk raster projects first, so an unprojectable pair is a bad
        request rather than the error PROJ raises out of the transform."""
        response = self._get(client, RASTER_GRID, grid_cell="0,0", to_grid="ecoregion_v001")
        assert response.status_code == 400
        assert "outside the grid's projection" in response.json()["detail"]

    def test_unprojectable_coordinates_are_rejected_for_a_vector_source(
        self, client, vector_crosswalk, vector_geometry
    ):
        """Written with decimal points, since a vector grid reads bare digits as cell IDs."""
        response = self._get(client, VECTOR_GRID, grid_cell="0.0,0.0", to_grid="ecoregion_v001")
        assert response.status_code == 400
        assert "outside the grid's projection" in response.json()["detail"]
