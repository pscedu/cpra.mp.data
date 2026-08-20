"""Tests for the Dataset query pipeline on raster (COG) grids.

Raster datasets carry a band-metadata table (dimension columns plus `path`) rather than
values, so each query stage has a separate raster implementation. `summarize_by` converts
the result to the same tabular shape the vector path produces.
"""

import numpy as np
import polars as pl
import pytest
import rasterio as rio
from fastapi import HTTPException

from cpra.mp.data.api.dataset import _get_zone_segments
from cpra.mp.data.api.types import OutputType
from cpra.mp.data.api.utils import group_rows_by_path
from cpra.mp.data.grids import GRIDS

from tests.factories import RASTER_KWARGS, write_raster_crosswalk, write_raster_dataset

DIMENSIONS = {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}
GRID = GRIDS[RASTER_KWARGS["grid"]]


@pytest.fixture
def raster_data(base_path):
    """Two COGs: 2030 filled with 30, 2040 filled with 40."""
    write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
    return base_path


@pytest.fixture
def one_raster(base_path):
    write_raster_dataset(base_path, years=(2030,), fill_fn=lambda y: y - 2000)
    return base_path


@pytest.fixture
def zone_crosswalk(crosswalk_path):
    """Split the grid into two zones by row: zone 1 on top, zone 2 below."""
    zones = np.zeros((GRID.height, GRID.width), dtype="int32")
    zones[: GRID.height // 2, :] = 1
    zones[GRID.height // 2 :, :] = 2
    return write_raster_crosswalk(
        crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones
    )


class TestOutputType:
    def test_raster_grid_defaults_to_raster_output(self, dataset_factory, raster_data):
        assert dataset_factory(DIMENSIONS).output_type == OutputType.raster

    def test_grid_cell_switches_to_vector_output(self, dataset_factory, raster_data):
        dataset = dataset_factory(DIMENSIONS, grid_cell="-91.0,29.5")
        assert dataset.output_type == OutputType.vector

    def test_summarize_by_switches_to_vector_output(self, dataset_factory, raster_data):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="mean"
        )
        assert dataset.output_type == OutputType.vector

    def test_metadata_table_is_detected(self, dataset_factory, raster_data):
        assert dataset_factory(DIMENSIONS)._is_raster_metadata_table() is True

    async def test_metadata_table_is_false_after_summarizing(
        self, dataset_factory, raster_data, zone_crosswalk
    ):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="mean"
        )
        await dataset.apply_params()
        assert dataset._is_raster_metadata_table() is False


class TestGridCellSampling:
    async def test_samples_at_a_coordinate(self, dataset_factory, raster_data):
        """The grid covers coastal Louisiana; this point falls inside it."""
        left, bottom, right, top = GRID.bounds
        with rio.open(next(raster_data.rglob("*.tif"))) as src:
            x, y = src.xy(src.height // 2, src.width // 2)
        from rasterio.warp import transform

        lng, lat = [c[0] for c in transform(src.crs, "EPSG:4326", [x], [y])]

        dataset = dataset_factory(DIMENSIONS, grid_cell=f"{lng},{lat}")
        await dataset.apply_params()
        df = dataset.materialize().sort("calendar_year")
        # A sampled coordinate is reported as a "lng,lat" grid_cell label rather than an
        # integer ID, in the long-format grid_cell/value shape used everywhere else.
        label = f"{lng},{lat}"
        assert df["grid_cell"].to_list() == [label, label]
        assert df["value"].to_list() == [30, 40]

    async def test_path_column_is_dropped_after_sampling(self, dataset_factory, raster_data):
        dataset = dataset_factory(DIMENSIONS, grid_cell="-91.0,29.5")
        await dataset.apply_params()
        assert "path" not in dataset.materialize().columns

    async def test_non_numeric_coordinates_raise_400(self, dataset_factory, raster_data):
        dataset = dataset_factory(DIMENSIONS, grid_cell="west,north")
        with pytest.raises(HTTPException) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "non-numeric coordinates" in excinfo.value.detail

    async def test_odd_coordinate_count_raises_400(self, dataset_factory, raster_data):
        dataset = dataset_factory(DIMENSIONS, grid_cell="-91.0,29.5,-90.0")
        with pytest.raises(HTTPException) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "even number of coordinates" in excinfo.value.detail

    @pytest.mark.parametrize(
        "grid_cell", ["181.0,29.5", "-91.0,91.0", "-200.0,-100.0", "nan,nan"]
    )
    async def test_out_of_range_coordinates_raise_400(
        self, dataset_factory, raster_data, grid_cell
    ):
        dataset = dataset_factory(DIMENSIONS, grid_cell=grid_cell)
        with pytest.raises(HTTPException) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "invalid coordinate pair" in excinfo.value.detail

    async def test_unprojectable_coordinates_raise_400(self, dataset_factory, raster_data):
        """A pair can be valid WGS84 and still have no place in the grid's UTM zone."""
        dataset = dataset_factory(DIMENSIONS, grid_cell="0,0")
        with pytest.raises(HTTPException) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "outside the grid's projection" in excinfo.value.detail


class TestCalculateStage:
    async def test_multiplier_column_is_added(self, dataset_factory, raster_data):
        """Raster calculate params become a per-band multiplier applied at read time."""
        dataset = dataset_factory(DIMENSIONS)
        dataset._apply_calculate_params()
        assert dataset.materialize()["multiplier"].to_list() == [1, 1]

    async def test_negate_sets_a_negative_multiplier(self, dataset_factory, raster_data):
        dataset = dataset_factory(DIMENSIONS, negate="calendar_year = 2030")
        dataset._apply_calculate_params()
        assert dataset.materialize().sort("calendar_year")["multiplier"].to_list() == [-1, 1]

    async def test_multiply10(self, dataset_factory, raster_data):
        dataset = dataset_factory(DIMENSIONS, multiply10="calendar_year = 2040")
        dataset._apply_calculate_params()
        assert dataset.materialize().sort("calendar_year")["multiplier"].to_list() == [1, 10]

    async def test_multiplier_reaches_the_pixels(self, dataset_factory, raster_data):
        dataset = dataset_factory(DIMENSIONS, negate="calendar_year = 2030")
        await dataset.apply_params()
        df = dataset.materialize().sort("calendar_year")
        _paths, row_indices, bands = group_rows_by_path(df)
        with rio.open(df["path"][0]) as first, rio.open(df["path"][1]) as second:
            data = await dataset.get_raster_array(df, [first, second], row_indices, bands, None)
        assert data[0][0, 0] == -30
        assert data[1][0, 0] == 40


class TestGroupStage:
    async def test_grouping_records_row_indices(self, dataset_factory, raster_data):
        dataset = dataset_factory(DIMENSIONS, group_aggregate="max")
        await dataset.apply_params()
        assert dataset.raster_groups is not None
        assert dataset.raster_groups.collect().height == 1

    async def test_group_by_a_dimension_keeps_one_group_per_value(
        self, dataset_factory, raster_data
    ):
        dataset = dataset_factory(DIMENSIONS, group_by="calendar_year", group_aggregate="max")
        await dataset.apply_params()
        assert dataset.raster_groups.collect().height == 2

    async def test_aggregate_collapses_bands(self, dataset_factory, raster_data):
        dataset = dataset_factory(DIMENSIONS, group_aggregate="max")
        await dataset.apply_params()
        df = dataset.materialize()
        _paths, row_indices, bands = group_rows_by_path(df)
        with rio.open(df["path"][0]) as first, rio.open(df["path"][1]) as second:
            data = await dataset.get_raster_array(df, [first, second], row_indices, bands, None)
        assert data.shape[0] == 1
        assert data[0][0, 0] == 40

    @pytest.mark.parametrize(
        ("aggregate", "expected"), [("min", 30), ("max", 40), ("sum", 70), ("mean", 35)]
    )
    async def test_each_aggregate_function(
        self, dataset_factory, raster_data, aggregate, expected
    ):
        dataset = dataset_factory(DIMENSIONS, group_aggregate=aggregate)
        await dataset.apply_params()
        df = dataset.materialize()
        _paths, row_indices, bands = group_rows_by_path(df)
        with rio.open(df["path"][0]) as first, rio.open(df["path"][1]) as second:
            data = await dataset.get_raster_array(df, [first, second], row_indices, bands, None)
        assert data[0][0, 0] == expected


class TestZoneSegments:
    def test_groups_pixels_by_zone(self):
        zones = np.array([[1, 2, 1, 2]], dtype="int64")
        zone_ids = np.array([1, 2], dtype="int64")
        positions, pixels, starts, counts = _get_zone_segments(zones, zone_ids)
        assert positions.tolist() == [0, 1]
        assert counts.tolist() == [2, 2]
        assert sorted(pixels[starts[0] : starts[0] + counts[0]].tolist()) == [0, 2]

    def test_drops_zones_that_are_not_summarized(self):
        zones = np.array([[1, 99, 1]], dtype="int64")
        zone_ids = np.array([1], dtype="int64")
        positions, pixels, _starts, counts = _get_zone_segments(zones, zone_ids)
        assert positions.tolist() == [0]
        assert counts.tolist() == [2]
        assert sorted(pixels.tolist()) == [0, 2]

    def test_drops_nodata_zeros(self):
        zones = np.array([[0, 1, 0]], dtype="int64")
        zone_ids = np.array([1, 2], dtype="int64")
        positions, pixels, _starts, counts = _get_zone_segments(zones, zone_ids)
        assert positions.tolist() == [0]
        assert counts.tolist() == [1]

    def test_no_matching_zones_returns_empty(self):
        zones = np.array([[7, 8]], dtype="int64")
        zone_ids = np.array([1, 2], dtype="int64")
        positions, _pixels, _starts, _counts = _get_zone_segments(zones, zone_ids)
        assert positions.size == 0

    def test_counts_sum_to_the_matched_pixel_count(self):
        rng = np.random.default_rng(0)
        zones = rng.integers(0, 4, size=(8, 8)).astype("int64")
        zone_ids = np.array([1, 2, 3], dtype="int64")
        _positions, pixels, _starts, counts = _get_zone_segments(zones, zone_ids)
        assert counts.sum() == pixels.size


class TestSummarizeStage:
    async def test_missing_crosswalk_raises_400(self, dataset_factory, raster_data, crosswalk_path):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="parish_v2020", summarize_aggregate="mean"
        )
        with pytest.raises(HTTPException) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "No crosswalk from veg_grid_cell_v001 to parish_v2020" in excinfo.value.detail

    async def test_produces_one_row_per_zone_per_band(
        self, dataset_factory, raster_data, zone_crosswalk
    ):
        """Summarizing no longer produces one column per zone; each zone is a row,
        identified by an Int32 grid_cell value, in the long-format result."""
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="mean"
        )
        await dataset.apply_params()
        df = dataset.materialize()
        assert df["grid_cell"].dtype == pl.Int32
        assert sorted(df["grid_cell"].unique().to_list()) == [1, 2]
        assert df.height == 4  # 2 bands (years) x 2 zones

    @pytest.mark.parametrize("aggregate", ["min", "max", "mean", "sum"])
    async def test_uniform_rasters_summarize_to_the_fill_value(
        self, dataset_factory, raster_data, zone_crosswalk, aggregate
    ):
        """Each COG is uniform, so min/max/mean all equal the fill and sum scales by area."""
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate=aggregate
        )
        await dataset.apply_params()
        df = dataset.materialize()

        zone_pixels = (GRID.height // 2) * GRID.width
        expected = {"min": 30, "max": 30, "mean": 30, "sum": 30 * zone_pixels}[aggregate]
        zone_1 = df.filter(pl.col("grid_cell") == 1).sort("calendar_year")
        assert zone_1["value"][0] == pytest.approx(expected)

    async def test_geographic_unit_column_is_added(
        self, dataset_factory, raster_data, zone_crosswalk
    ):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="mean"
        )
        await dataset.apply_params()
        assert dataset.materialize()["geographic_unit"].unique().to_list() == [
            "hydro_compartment_v001"
        ]

    async def test_summarize_grid_cell_restricts_zones(
        self, dataset_factory, raster_data, zone_crosswalk
    ):
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
            summarize_grid_cell="2",
        )
        await dataset.apply_params()
        df = dataset.materialize()
        assert sorted(df["grid_cell"].unique().to_list()) == [2]

    async def test_zone_with_no_pixels_is_nan(self, dataset_factory, raster_data, zone_crosswalk):
        """Zone 5 is not present in the crosswalk raster, so it has no contributing pixels."""
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
            summarize_grid_cell="1,5",
        )
        await dataset.apply_params()
        df = dataset.materialize()
        zone_5 = df.filter(pl.col("grid_cell") == 5)
        zone_1 = df.filter(pl.col("grid_cell") == 1).sort("calendar_year")
        assert all(np.isnan(v) for v in zone_5["value"].to_list())
        assert zone_1["value"][0] == pytest.approx(30)

    async def test_empty_dataset_raises_404(self, dataset_factory, base_path, zone_crosswalk):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="mean"
        )
        with pytest.raises(HTTPException) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 404

    async def test_grouping_collapses_summary_rows(
        self, dataset_factory, raster_data, zone_crosswalk
    ):
        dataset = dataset_factory(
            DIMENSIONS,
            group_aggregate="max",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
        )
        await dataset.apply_params()
        df = dataset.materialize()
        assert df.height == 2  # one group (band) x 2 zones
        zone_1 = df.filter(pl.col("grid_cell") == 1)
        assert zone_1["value"][0] == pytest.approx(40)

    async def test_grouped_dimensions_become_null(
        self, dataset_factory, raster_data, zone_crosswalk
    ):
        dataset = dataset_factory(
            DIMENSIONS,
            group_aggregate="max",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
        )
        await dataset.apply_params()
        df = dataset.materialize()
        assert df["calendar_year"].null_count() == df.height

    async def test_unpivot_is_a_no_op_after_summarizing(
        self, dataset_factory, raster_data, zone_crosswalk
    ):
        """unpivot is retained only for API compatibility: the summarize result is
        already long-format (grid_cell/value), so the parameter changes nothing."""
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
            unpivot=True,
        )
        await dataset.apply_params()
        df = dataset.materialize()
        assert {"grid_cell", "value"} <= set(df.columns)
        assert df.height == 4


class TestValidateTile:
    async def test_single_band_passes(self, dataset_factory, one_raster):
        dataset = dataset_factory(DIMENSIONS)
        await dataset.apply_params()
        dataset.validate_tile()

    async def test_multiple_bands_raise_400(self, dataset_factory, raster_data):
        dataset = dataset_factory(DIMENSIONS)
        await dataset.apply_params()
        with pytest.raises(HTTPException) as excinfo:
            dataset.validate_tile()
        assert excinfo.value.status_code == 400
        assert "2 bands found" in excinfo.value.detail

    async def test_grouping_makes_a_multiband_dataset_tileable(
        self, dataset_factory, raster_data
    ):
        dataset = dataset_factory(DIMENSIONS, group_aggregate="max")
        await dataset.apply_params()
        dataset.validate_tile()

    async def test_vector_output_raises_400(self, dataset_factory, raster_data):
        dataset = dataset_factory(DIMENSIONS, grid_cell="-91.0,29.5")
        with pytest.raises(HTTPException) as excinfo:
            dataset.validate_tile()
        assert "invalid output type for tile: vector" in excinfo.value.detail


class TestValidateVectorTile:
    async def test_raster_output_raises_400(self, dataset_factory, raster_data, tiles_path):
        dataset = dataset_factory(DIMENSIONS)
        with pytest.raises(HTTPException) as excinfo:
            dataset.validate_vector_tile()
        assert excinfo.value.status_code == 400
        assert "invalid output type for vector tile: raster" in excinfo.value.detail

    async def test_missing_pmtiles_raises_404(self, dataset_factory, raster_data, tiles_path):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="mean"
        )
        with pytest.raises(HTTPException) as excinfo:
            dataset.validate_vector_tile()
        assert excinfo.value.status_code == 404
        assert "no tile file for grid: hydro_compartment_v001" in excinfo.value.detail

    async def test_present_pmtiles_passes(self, dataset_factory, raster_data, tiles_path):
        (tiles_path / "hydro_compartment_v001.pmtiles").touch()
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="mean"
        )
        dataset.validate_vector_tile()


class TestWriteRaster:
    async def test_writes_one_band_per_matched_cog(self, dataset_factory, raster_data, tmp_path):
        dataset = dataset_factory(DIMENSIONS)
        await dataset.apply_params()
        out = tmp_path / "out.tif"

        assert await dataset.write_raster(out) is True
        with rio.open(out) as src:
            assert src.count == 2
            assert src.read(1)[0, 0] == 30
            assert src.read(2)[0, 0] == 40

    async def test_band_count_follows_grouping(self, dataset_factory, raster_data, tmp_path):
        """group_by can collapse matched bands, so the count can't come from the sources."""
        dataset = dataset_factory(DIMENSIONS, group_aggregate="sum")
        await dataset.apply_params()
        out = tmp_path / "out.tif"

        await dataset.write_raster(out)
        with rio.open(out) as src:
            assert src.count == 1
            assert src.read(1)[0, 0] == 70

    async def test_empty_dataset_returns_false(self, dataset_factory, base_path, tmp_path):
        dataset = dataset_factory(DIMENSIONS)
        await dataset.apply_params()
        assert await dataset.write_raster(tmp_path / "out.tif") is False

    async def test_profile_options_are_applied(self, dataset_factory, one_raster, tmp_path):
        dataset = dataset_factory(DIMENSIONS)
        await dataset.apply_params()
        out = tmp_path / "out.tif"

        await dataset.write_raster(out, compress="DEFLATE")
        with rio.open(out) as src:
            assert src.profile["compress"].lower() == "deflate"


class TestGetTile:
    async def test_returns_png_bytes(self, dataset_factory, one_raster):
        import mercantile

        with rio.open(next(one_raster.rglob("*.tif"))) as src:
            from rasterio.warp import transform_bounds

            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        tile = mercantile.tile((west + east) / 2, (south + north) / 2, 8)

        dataset = dataset_factory(DIMENSIONS)
        await dataset.apply_params()
        body = await dataset.get_tile(tile.x, tile.y, tile.z, "png")
        assert body is not None
        assert body[:8] == b"\x89PNG\r\n\x1a\n"

    async def test_returns_jpeg_bytes(self, dataset_factory, one_raster):
        import mercantile

        with rio.open(next(one_raster.rglob("*.tif"))) as src:
            from rasterio.warp import transform_bounds

            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        tile = mercantile.tile((west + east) / 2, (south + north) / 2, 8)

        dataset = dataset_factory(DIMENSIONS)
        await dataset.apply_params()
        body = await dataset.get_tile(tile.x, tile.y, tile.z, "jpg")
        assert body[:3] == b"\xff\xd8\xff"

    async def test_tile_outside_the_dataset_returns_none(self, dataset_factory, one_raster):
        """A tile over Antarctica cannot overlap a Louisiana grid."""
        dataset = dataset_factory(DIMENSIONS)
        await dataset.apply_params()
        assert await dataset.get_tile(0, 0, 2, "png") is None
