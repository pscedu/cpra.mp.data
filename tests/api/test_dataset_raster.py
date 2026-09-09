"""Tests for the Dataset query pipeline on raster (COG) grids.

Raster datasets carry a band-metadata table (dimension columns plus `path`) rather than
values, so each query stage has a separate raster implementation. `summarize_by` converts
the result to the same tabular shape the vector path produces.
"""

from io import BytesIO

import numpy as np
import polars as pl
import pytest
import rasterio as rio
from fastapi import HTTPException
from PIL import Image
from rasterio.windows import Window

from cpra.mp.data.api import dataset as dataset_module
from cpra.mp.data.api.dataset import _batched_block_windows, _get_zone_segments
from cpra.mp.data.api.types import OutputType
from cpra.mp.data.api.utils import group_rows_by_path
from cpra.mp.data.grids import GRIDS

from tests.factories import (
    RASTER_KWARGS,
    raster_array,
    write_raster_crosswalk,
    write_raster_dataset,
)

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
def nodata_raster(base_path):
    """One COG that is entirely nodata, for rendering a tile with no real data in it."""
    write_raster_dataset(base_path, years=(2030,), data_fn=lambda _year: raster_array(fill=-9999))
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


class TestBatchedBlockWindows:
    """Batches are capped at `blocks_per_batch` blocks, independent of how
    many blocks make up a full block row."""

    @staticmethod
    def _row_of_blocks(row, count, block_width=10, block_height=10):
        return [
            ((row, col), Window(col * block_width, row * block_height, block_width, block_height))
            for col in range(count)
        ]

    def test_caps_batch_size_below_a_full_row(self):
        block_windows = self._row_of_blocks(0, 10)
        batches = list(_batched_block_windows(block_windows, blocks_per_batch=3))
        assert len(batches) == 4  # ceil(10 / 3)
        assert [b.width for b in batches] == [30, 30, 30, 10]

    def test_never_spans_more_than_one_block_row(self):
        block_windows = self._row_of_blocks(0, 5) + self._row_of_blocks(1, 5)
        batches = list(_batched_block_windows(block_windows, blocks_per_batch=1000))
        assert len(batches) == 2  # one batch per row, despite the huge cap
        assert [b.row_off for b in batches] == [0, 10]

    def test_batch_count_is_independent_of_row_width(self):
        """The same blocks_per_batch yields the same number of batches per row whether
        the row is 10 blocks wide or 100 -- the original defect was exactly the
        batch size scaling with row width instead of staying capped."""
        narrow = list(_batched_block_windows(self._row_of_blocks(0, 10), blocks_per_batch=3))
        wide = list(_batched_block_windows(self._row_of_blocks(0, 100), blocks_per_batch=3))
        assert len(narrow) == 4
        assert len(wide) == 34  # ceil(100 / 3), not a single 100-block union

    def test_covers_every_block_exactly_once(self):
        block_windows = self._row_of_blocks(0, 7)
        batches = list(_batched_block_windows(block_windows, blocks_per_batch=3))
        total_width = sum(b.width for b in batches)
        assert total_width == 7 * 10


class TestSummarizeStage:
    async def test_missing_crosswalk_raises_400(self, dataset_factory, raster_data, crosswalk_path):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="parish_v2020", summarize_aggregate="mean"
        )
        with pytest.raises(HTTPException) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "No crosswalk from veg_grid_cell_v001 to parish_v2020" in excinfo.value.detail

    async def test_batch_size_does_not_change_the_result(
        self, dataset_factory, raster_data, zone_crosswalk, monkeypatch
    ):
        """A block row is chunked into batches of at most `RASTER_BLOCKS_PER_BATCH`
        blocks purely to bound memory -- the summarized result must be identical
        regardless of the chunk size. The test grid's single block row is split into 3
        column blocks (512, 512, 28 px wide), so 1 vs. 1000 genuinely exercises chunked
        vs. whole-row batching, not just a no-op override.
        """
        from cpra.mp.data.api import utils as api_utils

        monkeypatch.setattr(api_utils, "RASTER_BLOCKS_PER_BATCH", 1)
        chunked_dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="mean"
        )
        await chunked_dataset.apply_params()
        chunked = chunked_dataset.materialize().sort(["calendar_year", "grid_cell"])

        monkeypatch.setattr(api_utils, "RASTER_BLOCKS_PER_BATCH", 1000)
        whole_row_dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="mean"
        )
        await whole_row_dataset.apply_params()
        whole_row = whole_row_dataset.materialize().sort(["calendar_year", "grid_cell"])

        assert chunked["grid_cell"].to_list() == whole_row["grid_cell"].to_list()
        assert np.array_equal(
            chunked["value"].to_numpy(), whole_row["value"].to_numpy(), equal_nan=True
        )

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


@pytest.fixture
def raster_data_with_nodata(base_path):
    """Two COGs whose nodata pixels (-9999, `write_cog`'s fixed sentinel) probe the
    nodata masking against `zone_crosswalk`'s top/bottom row split:

    * 2030: zone 1 (top half) is real (10) except its first row, which is nodata; zone 2
      (bottom half) is entirely nodata.
    * 2040: the opposite -- zone 1 entirely nodata, zone 2 real (20) throughout.

    The two bands disagree on which zone is all-nodata so a per-zone (rather than
    per-band-per-zone) contributing-pixel count would wrongly mark the real band NaN too.
    """
    half = GRID.height // 2

    def data_fn(year):
        data = np.full((GRID.height, GRID.width), 10 if year == 2030 else 20, dtype="int32")
        if year == 2030:
            data[0, :] = -9999  # one nodata row inside real zone 1
            data[half:, :] = -9999  # all of zone 2
        else:
            data[:half, :] = -9999  # all of zone 1
        return data

    return write_raster_dataset(base_path, years=(2030, 2040), data_fn=data_fn)


class TestNodataMasking:
    """Nodata pixels are excluded from every aggregate, and `mean` divides by
    the count of contributing (non-nodata) pixels, not the zone's raw pixel count."""

    async def test_nodata_excluded_from_min_max_sum(
        self, dataset_factory, raster_data_with_nodata, zone_crosswalk
    ):
        zone_pixels = (GRID.height // 2) * GRID.width
        contributing = zone_pixels - GRID.width  # one nodata row removed from zone 1

        for aggregate, expected in (("min", 10), ("max", 10), ("sum", 10 * contributing)):
            dataset = dataset_factory(
                DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate=aggregate
            )
            await dataset.apply_params()
            df = dataset.materialize()
            zone_1_2030 = df.filter(
                (pl.col("grid_cell") == 1) & (pl.col("calendar_year") == 2030)
            )
            assert zone_1_2030["value"][0] == pytest.approx(expected), aggregate

    async def test_mean_denominator_is_contributing_pixels_not_zone_pixels(
        self, dataset_factory, raster_data_with_nodata, zone_crosswalk
    ):
        """Every contributing pixel is 10; a raw-pixel-count denominator (the pre-fix
        behaviour) would pull this below 10 because it would also divide by the nodata
        row, and treating nodata as 0 as well as counting it would pull it lower still."""
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="mean"
        )
        await dataset.apply_params()
        df = dataset.materialize()
        zone_1_2030 = df.filter((pl.col("grid_cell") == 1) & (pl.col("calendar_year") == 2030))
        assert zone_1_2030["value"][0] == pytest.approx(10)

    async def test_all_nodata_band_zone_is_nan_per_band(
        self, dataset_factory, raster_data_with_nodata, zone_crosswalk
    ):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="mean"
        )
        await dataset.apply_params()
        df = dataset.materialize()

        def value(grid_cell, year):
            return df.filter(
                (pl.col("grid_cell") == grid_cell) & (pl.col("calendar_year") == year)
            )["value"][0]

        # The all-nodata (band, zone) cells are NaN...
        assert np.isnan(value(2, 2030))
        assert np.isnan(value(1, 2040))
        # ...but the real data in the *other* band for the very same zones is not --
        # this is the per-(band, zone) granularity the fix requires: a per-zone-only
        # contributing count would incorrectly mark these NaN too, since each zone has
        # an all-nodata band somewhere.
        assert value(1, 2030) == pytest.approx(10)
        assert value(2, 2040) == pytest.approx(20)


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

    async def test_all_nodata_tile_renders_as_the_sentinel(self, dataset_factory, nodata_raster):
        """A tile with nothing but nodata colors exactly as the sentinel value itself.

        The end-to-end check that no null escapes `_collapse_tile_bands` into the legend:
        a null would reach `colorize_pixels` as NaN, which `np.digitize` puts in the LAST
        bin instead of the below-range first one, so an empty tile would come back painted
        the top color.
        """
        import mercantile

        with rio.open(next(nodata_raster.rglob("*.tif"))) as src:
            from rasterio.warp import transform_bounds

            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        tile = mercantile.tile((west + east) / 2, (south + north) / 2, 8)

        dataset = dataset_factory(DIMENSIONS)
        await dataset.apply_params()
        body = await dataset.get_tile(tile.x, tile.y, tile.z, "png")

        expected = dataset.legend.colorize_pixels(np.full((1, 1, 1), -9999, dtype="int32"))
        with Image.open(BytesIO(body)) as image:
            pixels = np.array(image)
        assert (pixels == expected[0, 0]).all()


class TestTileBandCollapse:
    """`Dataset._collapse_tile_bands` -- the tile path's multiplier/group_aggregate step.

    Exercised directly on a pixel array rather than through `get_tile`, so the assertions
    are about values rather than colors: a legend maps whole ranges onto one color, which
    would hide most of what these pin. `get_tile`'s own tests cover the wiring.

    Nodata is handled here the way the write and summarize paths handle it (one shared
    builder, `raster_source.band_collapse_exprs`). The NumPy reduce this replaced did not:
    it scaled the sentinel by any calculate multiplier and averaged it in as a real value.
    """

    async def test_nodata_is_excluded_from_the_aggregate(self, dataset_factory, raster_data):
        dataset = dataset_factory(DIMENSIONS, group_aggregate="mean")
        await dataset.apply_params()
        df = dataset.materialize()
        # Three pixels: nodata in one band, nodata in both, real data in both.
        data = np.array([[[-9999, -9999, 30]], [[40, -9999, 40]]], dtype="int32")

        out = dataset._collapse_tile_bands(df, data, -9999)

        assert out.shape == (1, 1, 3)
        assert out[0, 0, 0] == 40, "the nodata band drops out instead of averaging in"
        assert out[0, 0, 1] == -9999, "no band contributed, so the sentinel comes back"
        assert out[0, 0, 2] == 35

    async def test_all_nodata_group_is_the_sentinel_not_zero(self, dataset_factory, raster_data):
        """`sum`'s additive identity is 0, which would render as a real value of zero."""
        dataset = dataset_factory(DIMENSIONS, group_aggregate="sum")
        await dataset.apply_params()
        df = dataset.materialize()
        data = np.array([[[-9999, 30]], [[-9999, 40]]], dtype="int32")

        out = dataset._collapse_tile_bands(df, data, -9999)

        assert out[0, 0, 0] == -9999
        assert out[0, 0, 1] == 70

    async def test_multiplier_does_not_scale_the_sentinel(self, dataset_factory, raster_data):
        """Masking happens before the multiply, so `negate` can't turn -9999 into 9999.

        Unaggregated, so this is also the case where the tile path deliberately diverges
        from the write path: a written file keeps the sentinel for its reader to interpret
        (test_non_aggregated_output_preserves_nodata), but a rendered pixel has to commit to
        a color, and 9999 is a legitimate value in some legends.
        """
        dataset = dataset_factory(DIMENSIONS, negate="calendar_year = 2030")
        await dataset.apply_params()
        df = dataset.materialize().sort("calendar_year")
        assert df["multiplier"].to_list() == [-1, 1]
        data = np.array([[[-9999, 30]], [[40, 40]]], dtype="int32")

        out = dataset._collapse_tile_bands(df, data, -9999)

        assert out.shape == (2, 1, 2), "no aggregate, so every selected band is an output band"
        assert out[0].tolist() == [[-9999, -30]]
        assert out[1].tolist() == [[40, 40]]

    async def test_a_source_without_nodata_is_left_alone(self, dataset_factory, raster_data):
        """`nodata=None` means every pixel is real, including one that looks like -9999."""
        dataset = dataset_factory(DIMENSIONS, group_aggregate="mean")
        await dataset.apply_params()
        df = dataset.materialize()
        data = np.array([[[-9999]], [[40]]], dtype="int32")

        out = dataset._collapse_tile_bands(df, data, None)

        assert out[0, 0, 0] == pytest.approx((-9999 + 40) / 2)


class TestPluginDifferential:
    """The Polars IO-plugin summarize path (config.RASTER_USE_IO_PLUGIN) must agree
    exactly with the numpy path it can replace -- see
    Dataset._summarize_zone_stats_plugin vs _summarize_zone_stats_numpy. Eligibility no
    longer depends on which query parameters are in play (see TestPluginEligibility
    below), so a group_aggregate collapse and a calculate (multiplier) parameter both need
    differential coverage here too, not just the "plain" case.
    """

    async def _run(self, dataset_factory, monkeypatch, use_plugin, **options):
        monkeypatch.setattr(dataset_module, "RASTER_USE_IO_PLUGIN", use_plugin)
        dataset = dataset_factory(DIMENSIONS, **options)
        await dataset.apply_params()
        return dataset.materialize().sort(["calendar_year", "grid_cell"])

    @pytest.mark.parametrize("aggregate", ["min", "max", "sum", "mean"])
    @pytest.mark.parametrize("data_fixture", ["one_raster", "raster_data"])
    async def test_plugin_matches_numpy(
        self, request, dataset_factory, zone_crosswalk, monkeypatch, data_fixture, aggregate
    ):
        """Single-file (`one_raster`) and multi-file (`raster_data`) cases, all four
        aggregates -- the exit criterion this whole path is measured against."""
        request.getfixturevalue(data_fixture)
        options = dict(summarize_by="hydro_compartment_v001", summarize_aggregate=aggregate)

        numpy_df = await self._run(dataset_factory, monkeypatch, False, **options)
        plugin_df = await self._run(dataset_factory, monkeypatch, True, **options)

        assert plugin_df["grid_cell"].to_list() == numpy_df["grid_cell"].to_list()
        assert np.array_equal(
            plugin_df["value"].to_numpy(), numpy_df["value"].to_numpy(), equal_nan=True
        ), aggregate

    @pytest.mark.parametrize("aggregate", ["min", "max", "sum", "mean"])
    async def test_plugin_matches_numpy_with_nodata(
        self, dataset_factory, raster_data_with_nodata, zone_crosswalk, monkeypatch, aggregate
    ):
        """Nodata handling differs in mechanism between the two paths (explicit masking
        plus a hand-corrected mean denominator in numpy, vs null-propagation plus Polars'
        own null-skipping count/mean in the plugin -- see scan_rasters) but must agree in
        result, including the all-nodata (band, zone) NaN case."""
        options = dict(summarize_by="hydro_compartment_v001", summarize_aggregate=aggregate)

        numpy_df = await self._run(dataset_factory, monkeypatch, False, **options)
        plugin_df = await self._run(dataset_factory, monkeypatch, True, **options)

        assert plugin_df["grid_cell"].to_list() == numpy_df["grid_cell"].to_list()
        assert np.array_equal(
            plugin_df["value"].to_numpy(), numpy_df["value"].to_numpy(), equal_nan=True
        ), aggregate

    async def test_plugin_zone_with_no_pixels_is_nan(
        self, dataset_factory, raster_data, zone_crosswalk, monkeypatch
    ):
        monkeypatch.setattr(dataset_module, "RASTER_USE_IO_PLUGIN", True)
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

    @pytest.mark.parametrize("aggregate", ["min", "max", "sum", "mean"])
    async def test_plugin_matches_numpy_with_group_aggregate(
        self, dataset_factory, raster_data, zone_crosswalk, monkeypatch, aggregate
    ):
        """Closing the group_aggregate/multiplier bypass (see TestPluginEligibility) means
        the plugin path must now agree with numpy for these cases too, not just the plain
        one -- see Dataset._summarize_zone_stats_plugin.

        Uses `raster_data` (no nodata) rather than `raster_data_with_nodata` deliberately:
        the numpy path applies group_aggregate *before* nodata masking
        (`get_raster_array` collapses first; `_accumulate_zone_stats` then masks the
        already-collapsed value against the raw nodata sentinel, which will essentially
        never match), so numpy's own group_aggregate+nodata combination has a pre-existing
        asymmetry unrelated to closing this bypass. Exercising it here would pin that
        numpy quirk into a test named for something else instead of documenting it as its
        own, separate known behavior.
        """
        options = dict(
            group_aggregate=aggregate,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
            summarize_grid_cell="1,5",  # zone 5 has no pixels -- exercises NaN placement too
        )
        numpy_df = await self._run(dataset_factory, monkeypatch, False, **options)
        plugin_df = await self._run(dataset_factory, monkeypatch, True, **options)

        assert plugin_df["grid_cell"].to_list() == numpy_df["grid_cell"].to_list()
        assert np.array_equal(
            plugin_df["value"].to_numpy(), numpy_df["value"].to_numpy(), equal_nan=True
        ), aggregate

    async def test_plugin_matches_numpy_with_multiplier(
        self, dataset_factory, raster_data, zone_crosswalk, monkeypatch
    ):
        """Same closed bypass, for the calculate (multiplier) parameter."""
        options = dict(
            negate="calendar_year = 2030",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
        )
        numpy_df = await self._run(dataset_factory, monkeypatch, False, **options)
        plugin_df = await self._run(dataset_factory, monkeypatch, True, **options)

        assert plugin_df["grid_cell"].to_list() == numpy_df["grid_cell"].to_list()
        assert np.array_equal(
            plugin_df["value"].to_numpy(), numpy_df["value"].to_numpy(), equal_nan=True
        )

    async def test_plugin_matches_numpy_with_group_aggregate_and_multiplier(
        self, dataset_factory, raster_data, zone_crosswalk, monkeypatch
    ):
        """Both parameters together -- the combination the bypass used to rule out
        entirely, regardless of RASTER_USE_IO_PLUGIN."""
        options = dict(
            group_aggregate="sum",
            negate="calendar_year = 2030",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
        )
        numpy_df = await self._run(dataset_factory, monkeypatch, False, **options)
        plugin_df = await self._run(dataset_factory, monkeypatch, True, **options)

        assert plugin_df["grid_cell"].to_list() == numpy_df["grid_cell"].to_list()
        assert np.array_equal(
            plugin_df["value"].to_numpy(), numpy_df["value"].to_numpy(), equal_nan=True
        )


class TestPluginEligibility:
    """The plugin path applies the same multiplier/group_aggregate pre-collapse as the
    numpy path (see Dataset._summarize_zone_stats_plugin and
    raster_source.horizontal_aggregate_expr), so whether it is used depends only on
    RASTER_USE_IO_PLUGIN -- not on which query parameters are in play."""

    async def _plugin_was_called(self, dataset, monkeypatch):
        called = []
        original = dataset._summarize_zone_stats_plugin

        def spy(*args, **kwargs):
            called.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(dataset, "_summarize_zone_stats_plugin", spy)
        await dataset.apply_params()
        return bool(called)

    async def test_group_aggregate_uses_the_plugin_when_flagged_on(
        self, dataset_factory, raster_data, zone_crosswalk, monkeypatch
    ):
        monkeypatch.setattr(dataset_module, "RASTER_USE_IO_PLUGIN", True)
        dataset = dataset_factory(
            DIMENSIONS,
            group_aggregate="max",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
        )
        assert await self._plugin_was_called(dataset, monkeypatch) is True

    async def test_multiplier_uses_the_plugin_when_flagged_on(
        self, dataset_factory, raster_data, zone_crosswalk, monkeypatch
    ):
        monkeypatch.setattr(dataset_module, "RASTER_USE_IO_PLUGIN", True)
        dataset = dataset_factory(
            DIMENSIONS,
            negate="calendar_year = 2030",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
        )
        assert await self._plugin_was_called(dataset, monkeypatch) is True

    async def test_group_aggregate_skips_the_plugin_when_flagged_off(
        self, dataset_factory, raster_data, zone_crosswalk, monkeypatch
    ):
        monkeypatch.setattr(dataset_module, "RASTER_USE_IO_PLUGIN", False)
        dataset = dataset_factory(
            DIMENSIONS,
            group_aggregate="max",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
        )
        assert await self._plugin_was_called(dataset, monkeypatch) is False

    async def test_plain_request_uses_the_plugin_when_flagged_on(
        self, dataset_factory, raster_data, zone_crosswalk, monkeypatch
    ):
        monkeypatch.setattr(dataset_module, "RASTER_USE_IO_PLUGIN", True)
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="mean"
        )
        assert await self._plugin_was_called(dataset, monkeypatch) is True

    async def test_plain_request_skips_the_plugin_when_flagged_off(
        self, dataset_factory, raster_data, zone_crosswalk, monkeypatch
    ):
        monkeypatch.setattr(dataset_module, "RASTER_USE_IO_PLUGIN", False)
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="mean"
        )
        assert await self._plugin_was_called(dataset, monkeypatch) is False
