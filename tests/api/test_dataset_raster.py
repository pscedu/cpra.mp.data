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
from PIL import Image
from rasterio.windows import Window

from cpra.mp.data.analysis import AnalysisError
from cpra.mp.data.analysis import raster as dataset_module
from cpra.mp.data.analysis.raster import _batched_block_windows
from cpra.mp.data.analysis.types import OutputType
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
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "non-numeric coordinates" in excinfo.value.detail

    async def test_odd_coordinate_count_raises_400(self, dataset_factory, raster_data):
        dataset = dataset_factory(DIMENSIONS, grid_cell="-91.0,29.5,-90.0")
        with pytest.raises(AnalysisError) as excinfo:
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
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "invalid coordinate pair" in excinfo.value.detail

    async def test_unprojectable_coordinates_raise_400(self, dataset_factory, raster_data):
        """A pair can be valid WGS84 and still have no place in the grid's UTM zone."""
        dataset = dataset_factory(DIMENSIONS, grid_cell="0,0")
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "outside the grid's projection" in excinfo.value.detail


class TestCalculateStage:
    async def test_no_calculate_params_records_no_plans(self, dataset_factory, raster_data):
        """A raster's calculate stage never touches the band-metadata table itself -- it
        records validated plans for the pixel-read paths to apply per band."""
        dataset = dataset_factory(DIMENSIONS)
        dataset._apply_calculate_params()
        assert dataset._calculate_plans == []
        assert "multiplier" not in dataset.materialize().columns

    async def test_a_generic_negate_entry_records_its_plan(self, dataset_factory, raster_data):
        """`negate` was flat-parameter sugar for this exact `calculate` entry -- retired
        from `Options` along with its HTTP spelling; the generic form it
        expanded into is what's left to record a plan."""
        dataset = dataset_factory(
            DIMENSIONS, calculate=[{"expression": "value * -1", "where": "calendar_year = 2030"}]
        )
        dataset._apply_calculate_params()
        (plan,) = dataset._calculate_plans
        assert plan.label == "calculate.1"
        assert plan.expression == "value * -1"
        assert plan.dimension_where == "(calendar_year = 2030)"

    async def test_dimension_where_selects_bands(self, dataset_factory, raster_data):
        """The dimension half of a calculate where is evaluated against the metadata
        rows, in df row order -- the band-selection half of the raster contract."""
        dataset = dataset_factory(
            DIMENSIONS, calculate=[{"expression": "value * 10", "where": "calendar_year = 2040"}]
        )
        dataset._apply_calculate_params()
        df = dataset.materialize().sort("calendar_year")
        (plan,) = dataset._calculate_plans
        assert dataset._band_applicability(df, plan) == [False, True]

    async def test_calculate_reaches_the_pixels(self, dataset_factory, raster_data):
        """A calculate entry recorded at the calculate stage reaches the pixels through
        `_collapse_tile_bands`, the shared Polars-expression calculate/group_aggregate
        step every read path (tile, write, summarize) uses."""
        dataset = dataset_factory(
            DIMENSIONS, calculate=[{"expression": "value * -1", "where": "calendar_year = 2030"}]
        )
        await dataset.apply_params()
        df = dataset.materialize().sort("calendar_year")
        data = np.array([[[30]], [[40]]], dtype="int32")

        out = dataset._collapse_tile_bands(df, data, -9999)

        assert out[0, 0, 0] == -30
        assert out[1, 0, 0] == 40

    async def test_entries_compose_sequentially_per_band(self, dataset_factory, raster_data):
        """Entry 2 sees entry 1's output on the band both select; the other band passes
        through both entries unchanged."""
        dataset = dataset_factory(
            DIMENSIONS,
            calculate=[
                {"expression": "value * 10", "where": "calendar_year = 2030"},
                {"expression": "value * -1", "where": "calendar_year = 2030"},
            ],
        )
        await dataset.apply_params()
        df = dataset.materialize().sort("calendar_year")
        data = np.array([[[30]], [[40]]], dtype="int32")

        out = dataset._collapse_tile_bands(df, data, -9999)

        assert out[0, 0, 0] == -300
        assert out[1, 0, 0] == 40

    async def test_value_condition_masks_per_pixel(self, dataset_factory, raster_data):
        """A `value` condition in a calculate where is a per-pixel elementwise mask on
        the render path too: only pixels passing it are rewritten, the rest pass through."""
        dataset = dataset_factory(
            DIMENSIONS, calculate=[{"expression": "value * -1", "where": "value > 35"}]
        )
        await dataset.apply_params()
        df = dataset.materialize().sort("calendar_year")
        data = np.array([[[30, 40]], [[40, 30]]], dtype="int32")

        out = dataset._collapse_tile_bands(df, data, -9999)

        assert out[0].tolist() == [[30, -40]]
        assert out[1].tolist() == [[-40, 30]]

    async def test_where_naming_a_dimension_this_dataset_lacks_is_400(
        self, dataset_factory, raster_data
    ):
        """`aep` is a real dimension globally, but not one vct's band-metadata table
        carries -- the calculate plan records fine (band applicability is a
        pixel-read-time question) and 400s, naming the entry, once a read path tries to
        resolve which bands it selects."""
        dataset = dataset_factory(
            DIMENSIONS, calculate=[{"expression": "value * 2", "where": "aep = 0.2"}]
        )
        await dataset.apply_params()
        df = dataset.materialize()
        data = np.array([[[30]], [[40]]], dtype="int32")

        with pytest.raises(AnalysisError) as excinfo:
            dataset._collapse_tile_bands(df, data, -9999)
        assert excinfo.value.status_code == 400
        assert "calculate.1" in excinfo.value.detail

    async def test_calculate_does_not_rewrite_the_sentinel(self, dataset_factory, raster_data):
        """NoData in, NoData out on the tile path: masking runs before every entry, so a
        no-where entry cannot turn -9999 into anything else."""
        dataset = dataset_factory(DIMENSIONS, calculate=[{"expression": "value + 1"}])
        await dataset.apply_params()
        df = dataset.materialize().sort("calendar_year")
        data = np.array([[[-9999, 30]], [[40, -9999]]], dtype="int32")

        out = dataset._collapse_tile_bands(df, data, -9999)

        assert out[0].tolist() == [[-9999, 31]]
        assert out[1].tolist() == [[41, -9999]]


class TestGroupStage:
    async def test_grouping_records_row_indices(self, dataset_factory, raster_data):
        dataset = dataset_factory(DIMENSIONS, group_aggregate="max")
        await dataset.apply_params()
        assert dataset._band_groups is not None
        assert len(dataset._band_groups) == 1

    async def test_group_by_a_dimension_keeps_one_group_per_value(
        self, dataset_factory, raster_data
    ):
        dataset = dataset_factory(DIMENSIONS, group_by="calendar_year", group_aggregate="max")
        await dataset.apply_params()
        assert len(dataset._band_groups) == 2

    async def test_aggregate_collapses_bands(self, dataset_factory, raster_data):
        dataset = dataset_factory(DIMENSIONS, group_aggregate="max")
        await dataset.apply_params()
        df = dataset._band_frame()
        data = np.array([[[30]], [[40]]], dtype="int32")

        out = dataset._collapse_tile_bands(df, data, -9999)

        assert out.shape == (1, 1, 1)
        assert out[0, 0, 0] == 40

    @pytest.mark.parametrize(
        ("aggregate", "expected"), [("min", 30), ("max", 40), ("sum", 70), ("mean", 35)]
    )
    async def test_each_aggregate_function(
        self, dataset_factory, raster_data, aggregate, expected
    ):
        dataset = dataset_factory(DIMENSIONS, group_aggregate=aggregate)
        await dataset.apply_params()
        df = dataset._band_frame()
        data = np.array([[[30]], [[40]]], dtype="int32")

        out = dataset._collapse_tile_bands(df, data, -9999)

        assert out[0, 0, 0] == expected


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
        with pytest.raises(AnalysisError) as excinfo:
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
        from cpra.mp.data.analysis import utils as api_utils

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
    @pytest.mark.parametrize("data_fixture", ["one_raster", "raster_data"])
    async def test_uniform_rasters_summarize_to_the_fill_value(
        self, request, dataset_factory, zone_crosswalk, data_fixture, aggregate
    ):
        """Each COG is uniform, so min/max/mean all equal the fill and sum scales by area.

        Parametrized over both a single-file (`one_raster`) and multi-file (`raster_data`)
        dataset: rows sharing a multiband COG are grouped onto one raster/band-list by
        `group_rows_by_path`, so this exercises both the single- and multi-source shapes
        of that grouping through the same summarize call.
        """
        request.getfixturevalue(data_fixture)
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
        with pytest.raises(AnalysisError) as excinfo:
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

    async def test_grouped_dimensions_are_dropped(
        self, dataset_factory, raster_data, zone_crosswalk
    ):
        """Mirrors vector's `test_ungrouped_dimensions_are_dropped`: calendar_year is
        collapsed away here too (group_aggregate with no group_by), so it is absent from
        the result entirely rather than an all-null placeholder column -- see
        `RasterGridAnalysis._get_raster_dimension_frame`'s docstring."""
        dataset = dataset_factory(
            DIMENSIONS,
            group_aggregate="max",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
        )
        await dataset.apply_params()
        df = dataset.materialize()
        assert "calendar_year" not in df.columns

    async def test_summarize_result_is_already_long_format(
        self, dataset_factory, raster_data, zone_crosswalk
    ):
        """A zonal summarize result is already long-format (grid_cell/value) unconditionally
        -- there used to be an `unpivot` parameter that was a no-op here for exactly that
        reason, but it was deleted from `Options`/`DataOptions` outright rather than
        keeping a flag that never did anything after a summarize."""
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
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
        with pytest.raises(AnalysisError) as excinfo:
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
        with pytest.raises(AnalysisError) as excinfo:
            dataset.validate_tile()
        assert "invalid output type for tile: vector" in excinfo.value.detail


class TestValidateVectorTile:
    async def test_raster_output_raises_400(self, dataset_factory, raster_data, tiles_path):
        dataset = dataset_factory(DIMENSIONS)
        with pytest.raises(AnalysisError) as excinfo:
            dataset.validate_vector_tile()
        assert excinfo.value.status_code == 400
        assert "invalid output type for vector tile: raster" in excinfo.value.detail

    async def test_missing_pmtiles_raises_404(self, dataset_factory, raster_data, tiles_path):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="mean"
        )
        with pytest.raises(AnalysisError) as excinfo:
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
        df = dataset._band_frame()
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
        df = dataset._band_frame()
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
        dataset = dataset_factory(
            DIMENSIONS, calculate=[{"expression": "value * -1", "where": "calendar_year = 2030"}]
        )
        await dataset.apply_params()
        df = dataset.materialize().sort("calendar_year")
        data = np.array([[[-9999, 30]], [[40, 40]]], dtype="int32")

        out = dataset._collapse_tile_bands(df, data, -9999)

        assert out.shape == (2, 1, 2), "no aggregate, so every selected band is an output band"
        assert out[0].tolist() == [[-9999, -30]]
        assert out[1].tolist() == [[40, 40]]

    async def test_a_source_without_nodata_is_left_alone(self, dataset_factory, raster_data):
        """`nodata=None` means every pixel is real, including one that looks like -9999."""
        dataset = dataset_factory(DIMENSIONS, group_aggregate="mean")
        await dataset.apply_params()
        df = dataset._band_frame()
        data = np.array([[[-9999]], [[40]]], dtype="int32")

        out = dataset._collapse_tile_bands(df, data, None)

        assert out[0, 0, 0] == pytest.approx((-9999 + 40) / 2)


class TestSummarizeCalculateAndGroupAggregate:
    """The zonal summarize (`_summarize_zone_stats_plugin`) applies a calculate
    (multiplier) parameter and a group_aggregate band collapse before the zonal reduce
    itself, via the shared `horizontal_aggregate_expr`/`HORIZONTAL_RASTER_AGGREGATES`
    vocabulary (see `_apply_transform_params_raster`'s docstring) -- these pin that
    composition with hand-computed values, since `TestSummarizeStage`'s plain-selection
    coverage doesn't exercise either parameter.
    """

    @pytest.mark.parametrize(
        ("aggregate", "expected"), [("min", 30), ("max", 40), ("sum", 70), ("mean", 35)]
    )
    async def test_group_aggregate_precedes_the_zonal_summarize(
        self, dataset_factory, raster_data, zone_crosswalk, aggregate, expected
    ):
        """group_aggregate (band collapse) and summarize_aggregate (zonal reduce) are two
        different axes of the same pipeline. Both source bands are uniform (30, 40), so
        the collapsed per-pixel value is the same everywhere in the zone, and the zonal
        `summarize_aggregate=mean` just confirms that collapsed value reaches every pixel
        in the zone unchanged. Zone 5 has no pixels in the crosswalk, exercising NaN
        placement here too.
        """
        dataset = dataset_factory(
            DIMENSIONS,
            group_aggregate=aggregate,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
            summarize_grid_cell="1,5",
        )
        await dataset.apply_params()
        df = dataset.materialize()

        zone_1 = df.filter(pl.col("grid_cell") == 1)
        zone_5 = df.filter(pl.col("grid_cell") == 5)
        assert zone_1["value"][0] == pytest.approx(expected)
        assert all(np.isnan(v) for v in zone_5["value"].to_list())

    async def test_multiplier_precedes_the_zonal_summarize(
        self, dataset_factory, raster_data, zone_crosswalk
    ):
        """The calculate (multiplier) parameter applies before the zonal summarize
        aggregate, band by band -- negating one band doesn't touch the other."""
        dataset = dataset_factory(
            DIMENSIONS,
            calculate=[{"expression": "value * -1", "where": "calendar_year = 2030"}],
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
        )
        await dataset.apply_params()
        df = dataset.materialize()

        zone_1 = df.filter(pl.col("grid_cell") == 1).sort("calendar_year")
        assert zone_1["value"].to_list() == pytest.approx([-30, 40])

    async def test_multiplier_and_group_aggregate_together(
        self, dataset_factory, raster_data, zone_crosswalk
    ):
        """Both parameters compose: multiply then collapse (see
        `_summarize_zone_stats_plugin`'s docstring) -- sum(-30, 40) = 10, uniform across
        the grid, so the zonal mean is also 10."""
        dataset = dataset_factory(
            DIMENSIONS,
            group_aggregate="sum",
            calculate=[{"expression": "value * -1", "where": "calendar_year = 2030"}],
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
        )
        await dataset.apply_params()
        df = dataset.materialize()

        zone_1 = df.filter(pl.col("grid_cell") == 1)
        zone_2 = df.filter(pl.col("grid_cell") == 2)
        assert zone_1["value"][0] == pytest.approx(10)
        assert zone_2["value"][0] == pytest.approx(10)

    async def test_generic_entries_weight_bands_before_the_zonal_summarize(
        self, dataset_factory, raster_data, zone_crosswalk
    ):
        """The confidence-weighting idiom, on the summarize path: each band scaled by its
        own coefficient (a dimension `where` selecting it), collapsed per pixel with
        `group_aggregate=sum` -- 30*0.6 + 40*0.3 = 30, uniform, so the zonal mean is 30."""
        dataset = dataset_factory(
            DIMENSIONS,
            calculate=[
                {"expression": "value * 0.6", "where": "calendar_year = 2030"},
                {"expression": "value * 0.3", "where": "calendar_year = 2040"},
            ],
            group_aggregate="sum",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
        )
        await dataset.apply_params()
        df = dataset.materialize()
        zone_1 = df.filter(pl.col("grid_cell") == 1)
        assert zone_1["value"][0] == pytest.approx(30 * 0.6 + 40 * 0.3)


@pytest.fixture
def group_aggregate_nodata_raster(base_path):
    """Two COGs (2030, 2040) built so a `group_aggregate` collapse across them, combined
    with nodata, has a numerically distinct answer depending on whether nodata is excluded
    from the collapse -- see `TestGroupAggregateNodataExclusion` below.

    Zone 1 (top half, via `zone_crosswalk`) is real in both bands: 10 in 2030, 20 in 2040.
    Zone 2 (bottom half) is nodata (-9999, `write_cog`'s fixed sentinel) in 2030 and real
    (30) in 2040 -- nodata in one of the two bands being collapsed, not all of them, so the
    correct answer excludes it rather than landing on an all-nodata marker.
    """
    half = GRID.height // 2

    def data_fn(year):
        data = np.full((GRID.height, GRID.width), 10 if year == 2030 else 20, dtype="int32")
        data[half:, :] = -9999 if year == 2030 else 30
        return data

    return write_raster_dataset(base_path, years=(2030, 2040), data_fn=data_fn)


class TestGroupAggregateNodataExclusion:
    """A nodata pixel in one of the raw bands being collapsed by `group_aggregate` is
    excluded from that collapse rather than contaminating it.

    `scan_rasters(mask_nodata=True)` nulls out a nodata pixel against its own raw value
    before `horizontal_aggregate_expr` (`raster_source.py`) ever sees it, so the collapse
    runs over only the surviving real values -- `mean_horizontal` skips nulls the same way
    Polars' own null-skipping `mean()` does. This is the single summarize implementation's
    only behavior; there is no other order of operations to compare it against.
    """

    async def test_group_aggregate_excludes_nodata_from_the_collapse(
        self, dataset_factory, group_aggregate_nodata_raster, zone_crosswalk
    ):
        """Zone 1 has no nodata, so it collapses normally: mean(10, 2030; 20, 2040) = 15,
        uniform across the zone, so the zonal mean is also 15 -- a sanity check that both
        zones ran through the same collapse.

        Zone 2's 2030 band is nodata (-9999) and its 2040 band is real (30). The
        nodata-excluding collapse mean is over the single surviving value, so it is
        exactly 30 -- not `mean(-9999, 30)`, which the masking exists to avoid. That is
        not the all-nodata case (`horizontal_aggregate_expr`'s `no_contributor` branch,
        which would apply only if *both* zone-2 bands were nodata) -- one band still
        contributes, so the collapsed value is exactly that band's value, not NaN/null.
        """
        dataset = dataset_factory(
            DIMENSIONS,
            group_aggregate="mean",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
        )
        await dataset.apply_params()
        df = dataset.materialize()

        def zone_value(grid_cell):
            return df.filter(pl.col("grid_cell") == grid_cell)["value"][0]

        assert zone_value(1) == pytest.approx(15.0)
        assert zone_value(2) == pytest.approx(30.0)

    async def test_calculate_never_rewrites_nodata_on_the_summarize_path(
        self, dataset_factory, group_aggregate_nodata_raster, zone_crosswalk
    ):
        """A no-where calculate entry rewrites every REAL pixel, and a nodata pixel still
        drops out of the collapse exactly as without calculate -- zone 2's 2030 band is
        nodata, so its collapse mean is just its rewritten 2040 value (30*10), not a
        blend with a rewritten sentinel."""
        dataset = dataset_factory(
            DIMENSIONS,
            calculate=[{"expression": "value * 10"}],
            group_aggregate="mean",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
        )
        await dataset.apply_params()
        df = dataset.materialize()

        def zone_value(grid_cell):
            return df.filter(pl.col("grid_cell") == grid_cell)["value"][0]

        assert zone_value(1) == pytest.approx(150.0)
        assert zone_value(2) == pytest.approx(300.0)


class TestSummarizeRoutesThroughThePlugin:
    """Every summarize request, regardless of which query parameters are in play, reaches
    `_summarize_zone_stats_plugin` (see `_apply_transform_params_raster`'s docstring) --
    there is only the one implementation to route to."""

    async def _plugin_was_called(self, dataset, monkeypatch):
        called = []
        original = dataset._summarize_zone_stats_plugin

        def spy(*args, **kwargs):
            called.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(dataset, "_summarize_zone_stats_plugin", spy)
        await dataset.apply_params()
        return bool(called)

    async def test_group_aggregate_uses_the_plugin(
        self, dataset_factory, raster_data, zone_crosswalk, monkeypatch
    ):
        dataset = dataset_factory(
            DIMENSIONS,
            group_aggregate="max",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
        )
        assert await self._plugin_was_called(dataset, monkeypatch) is True

    async def test_multiplier_uses_the_plugin(
        self, dataset_factory, raster_data, zone_crosswalk, monkeypatch
    ):
        dataset = dataset_factory(
            DIMENSIONS,
            calculate=[{"expression": "value * -1", "where": "calendar_year = 2030"}],
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
        )
        assert await self._plugin_was_called(dataset, monkeypatch) is True

    async def test_plain_request_uses_the_plugin(
        self, dataset_factory, raster_data, zone_crosswalk, monkeypatch
    ):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="mean"
        )
        assert await self._plugin_was_called(dataset, monkeypatch) is True


class TestValueWherePixelPredicate:
    """A `where` VALUE conjunct -- a predicate over pixel VALUES, not dimensions,
    written against the placeholder column `value` (see `expressions.classify_where`)
    -- is only meaningful once pixels are actually being summarized (see
    `_apply_transform_params_raster`'s docstring): it is masked in per band, before the
    multiplier/group_aggregate collapse (`_summarize_zone_stats_plugin`), so
    `summarize_aggregate=count` together with it is exactly "count pixels per zone where
    value > X" -- and any other summarize_aggregate becomes its conditional counterpart
    ("sum of pixels where...", etc.) for free via the same null-skipping mechanism nodata
    already uses.
    """

    @pytest.fixture
    def gradient_raster_data(self, base_path):
        """One COG whose pixels are a distinct, strictly increasing gradient (row-major
        0..N-1 -- see `raster_array`'s no-fill case) rather than a uniform fill, so a
        value threshold selects a known, non-trivial subset of pixels per zone."""
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _year: raster_array())
        return base_path

    def _zone_values(self):
        """The same zone split `zone_crosswalk` writes (zone 1 on top, zone 2 below),
        applied directly to the gradient array, so expected counts/sums are derived from
        the fixture's own data rather than hand-computed and liable to drift from it."""
        zones = np.zeros((GRID.height, GRID.width), dtype="int32")
        zones[: GRID.height // 2, :] = 1
        zones[GRID.height // 2 :, :] = 2
        gradient = raster_array()
        return {zone_id: gradient[zones == zone_id] for zone_id in (1, 2)}

    async def test_conditional_count_per_zone(
        self, dataset_factory, gradient_raster_data, zone_crosswalk
    ):
        threshold = 100_000
        zone_values = self._zone_values()
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="count",
            where=f"value > {threshold}",
        )
        await dataset.apply_params()
        df = dataset.materialize()
        for zone_id, values in zone_values.items():
            actual = df.filter(pl.col("grid_cell") == zone_id)["value"][0]
            assert actual == (values > threshold).sum()

    async def test_conditional_sum_masks_non_matching_pixels(
        self, dataset_factory, gradient_raster_data, zone_crosswalk
    ):
        """A non-count aggregate with a `where` value conjunct is a conditional
        aggregate for free, not a capability specific to count."""
        threshold = 383_000
        zone_values = self._zone_values()
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="sum",
            where=f"value > {threshold}",
        )
        await dataset.apply_params()
        df = dataset.materialize()
        for zone_id, values in zone_values.items():
            matching = values[values > threshold].astype("float64")
            actual = df.filter(pl.col("grid_cell") == zone_id)["value"][0]
            if matching.size == 0:
                # Every pixel in this zone was masked out -- the all-nodata-equivalent
                # NaN, exactly as guarded_aggregate already gives an empty group.
                assert np.isnan(actual)
            else:
                assert actual == pytest.approx(matching.sum())

    async def test_where_value_conjunct_without_summarize_is_rejected(self, dataset_factory, raster_data):
        """A raster grid has no per-pixel frame at all for a plain (non-summarized)
        response -- see `_apply_transform_params_raster`'s docstring -- so a `where`
        value conjunct given without summarize_by/summarize_aggregate is rejected
        outright, with an error that says so."""
        dataset = dataset_factory(DIMENSIONS, where="value > 3")
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "summarize" in excinfo.value.detail

    async def test_where_value_conjunct_uses_the_plugin(
        self, dataset_factory, gradient_raster_data, zone_crosswalk, monkeypatch
    ):
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="sum",
            where="value > 100000",
        )
        called = []
        original = dataset._summarize_zone_stats_plugin

        def spy(*args, **kwargs):
            called.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(dataset, "_summarize_zone_stats_plugin", spy)
        await dataset.apply_params()
        assert called

    async def test_summarize_aggregate_count_uses_the_plugin(
        self, dataset_factory, raster_data, zone_crosswalk, monkeypatch
    ):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="count"
        )
        called = []
        original = dataset._summarize_zone_stats_plugin

        def spy(*args, **kwargs):
            called.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(dataset, "_summarize_zone_stats_plugin", spy)
        await dataset.apply_params()
        assert called

    async def test_group_aggregate_count_is_rejected(self, dataset_factory, raster_data):
        """group_aggregate is the band-COLLAPSE step, not the zonal summarize aggregate
        `summarize_aggregate` shares its enum values with -- "how many bands contributed
        to this pixel" has no natural per-pixel meaning (see the AggregateFunction.count
        comment in types.py)."""
        dataset = dataset_factory(DIMENSIONS, group_aggregate="count")
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "count" in excinfo.value.detail

    async def test_mixed_conjunct_is_rejected_before_any_pixel_data_is_read(
        self, dataset_factory, gradient_raster_data, zone_crosswalk, monkeypatch
    ):
        """A single AND-term referencing both `value` and some other column has no
        ambiguity-free classification (`expressions.classify_where`) and is rejected at
        `_apply_filter_params` time -- before `_apply_transform_params_raster` even
        starts, so before `scan_rasters`, any COG, or the crosswalk file is touched.
        Substituted here for the retired value_where's own "unknown column" rejection,
        since a bare dimension-only conjunct (no `value` reference at all) is no longer
        eagerly namespace-checked -- it goes to the same unrestricted `lf.sql()` path any
        other `where` dimension conjunct does; this still pins the same observable
        contract the old test did: a 400 naming the offending token, before any data is
        read.
        """
        called = []

        def fail_if_called(*args, **kwargs):
            called.append(True)
            raise AssertionError("scan_rasters should not run: the expression should have been rejected first")

        monkeypatch.setattr(dataset_module, "scan_rasters", fail_if_called)

        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="count",
            where="value > not_a_column",
        )
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "not_a_column" in excinfo.value.detail
        assert not called

    async def test_aggregate_inside_the_predicate_is_rejected_before_any_pixel_data_is_read(
        self, dataset_factory, gradient_raster_data, zone_crosswalk, monkeypatch
    ):
        """The plan-time rule the 94x memory cliff (see api/expressions.py's module
        docstring) requires: an aggregate inside a value predicate is refused before
        `scan_rasters` -- and so before any COG or crosswalk file -- is ever touched."""
        called = []

        def fail_if_called(*args, **kwargs):
            called.append(True)
            raise AssertionError("scan_rasters should not run: the expression should have been rejected first")

        monkeypatch.setattr(dataset_module, "scan_rasters", fail_if_called)

        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="count",
            where="value = min(value)",
        )
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "min" in excinfo.value.detail
        assert not called

    async def test_dimension_subquery_is_rejected_before_any_pixel_data_is_read(
        self, dataset_factory, gradient_raster_data, zone_crosswalk, monkeypatch
    ):
        """The dimension-conjunct half of the closed 94x/26.9GB pushdown-defeating hole
        (see api/expressions.py's module docstring and `classify_where`): a subquery
        inside a DIMENSION conjunct is refused just as eagerly as one inside a value
        conjunct is, before `scan_rasters` or any file is touched.
        """
        called = []

        def fail_if_called(*args, **kwargs):
            called.append(True)
            raise AssertionError("scan_rasters should not run: the expression should have been rejected first")

        monkeypatch.setattr(dataset_module, "scan_rasters", fail_if_called)

        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="count",
            where="calendar_year in (select max(calendar_year) from self)",
        )
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert not called

    async def test_value_subquery_is_rejected_before_any_pixel_data_is_read(
        self, dataset_factory, gradient_raster_data, zone_crosswalk, monkeypatch
    ):
        """The value-conjunct half of the same closed hole: `value in (select ... from
        self)` fails to parse at `classify_where` time (no `self` relation is
        registered there), the same as the dimension form above, before any file is
        touched.
        """
        called = []

        def fail_if_called(*args, **kwargs):
            called.append(True)
            raise AssertionError("scan_rasters should not run: the expression should have been rejected first")

        monkeypatch.setattr(dataset_module, "scan_rasters", fail_if_called)

        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="count",
            where="value in (select max(value) from self)",
        )
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert not called


class TestCountByValue:
    """count_by_value: one row per (zone, distinct pixel value), with the count of
    contributing pixels sharing that value -- a categorical count, a genuinely different
    output SHAPE from every other summarize_aggregate (which collapse a zone to one row).
    See `raster_source.count_by_value` and `expressions.count_by_value_frame` for the
    single-raw-band scoping and the cardinality-ceiling design this class exercises.
    """

    N_CATEGORIES = 5

    @pytest.fixture
    def categorical_raster_data(self, base_path):
        """One COG whose pixels cycle through a handful of small integer categories
        (0..N_CATEGORIES-1) -- a bounded categorical column, e.g. land-type codes, not a
        uniform fill or an unbounded gradient."""
        write_raster_dataset(
            base_path, years=(2030,), data_fn=lambda _year: raster_array() % self.N_CATEGORIES
        )
        return base_path

    @pytest.fixture
    def two_band_categorical_raster_data(self, base_path):
        write_raster_dataset(
            base_path, years=(2030, 2040), data_fn=lambda _year: raster_array() % self.N_CATEGORIES
        )
        return base_path

    def _zone_category_counts(self):
        """Expected {zone: {value: count}}, derived from the fixture's own data (the same
        zone split `zone_crosswalk` writes: zone 1 on top, zone 2 below) rather than
        hand-computed, so it can't drift from what the fixture actually contains."""
        zones = np.zeros((GRID.height, GRID.width), dtype="int32")
        zones[: GRID.height // 2, :] = 1
        zones[GRID.height // 2 :, :] = 2
        categories = raster_array() % self.N_CATEGORIES
        counts = {}
        for zone_id in (1, 2):
            values, value_counts = np.unique(categories[zones == zone_id], return_counts=True)
            counts[zone_id] = dict(zip(values.tolist(), value_counts.tolist()))
        return counts

    async def test_counts_pixels_per_zone_per_value(
        self, dataset_factory, categorical_raster_data, zone_crosswalk
    ):
        expected = self._zone_category_counts()
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="count_by_value"
        )
        await dataset.apply_params()
        df = dataset.materialize()

        assert set(df.columns) >= {"grid_cell", "value", "count", "geographic_unit"}
        for zone_id, value_counts in expected.items():
            zone_df = df.filter(pl.col("grid_cell") == zone_id)
            actual = dict(zip(zone_df["value"].to_list(), zone_df["count"].to_list()))
            assert actual == value_counts

    async def test_composes_with_where_value_conjunct_mask_then_count(
        self, dataset_factory, categorical_raster_data, zone_crosswalk
    ):
        """A `where` value conjunct masks non-matching pixels to null BEFORE counting --
        see `raster_source.count_by_value`'s docstring -- so a masked-out pixel
        contributes to no category at all, rather than being counted under some sentinel
        value."""
        expected = self._zone_category_counts()
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="count_by_value",
            where="value != 2",
        )
        await dataset.apply_params()
        df = dataset.materialize()

        for zone_id, value_counts in expected.items():
            zone_df = df.filter(pl.col("grid_cell") == zone_id)
            actual = dict(zip(zone_df["value"].to_list(), zone_df["count"].to_list()))
            expected_masked = {v: c for v, c in value_counts.items() if v != 2}
            assert actual == expected_masked

    async def test_nodata_pixels_are_excluded_not_a_category(
        self, dataset_factory, zone_crosswalk, base_path
    ):
        def data_fn(_year):
            arr = raster_array() % self.N_CATEGORIES
            arr.reshape(-1)[:10] = -9999
            return arr

        write_raster_dataset(base_path, years=(2030,), data_fn=data_fn)
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="count_by_value"
        )
        await dataset.apply_params()
        df = dataset.materialize()

        assert -9999 not in df["value"].to_list()
        assert df["count"].sum() == GRID.height * GRID.width - 10

    async def test_multi_band_request_is_rejected_before_any_pixel_data_is_read(
        self, dataset_factory, two_band_categorical_raster_data, zone_crosswalk, monkeypatch
    ):
        called = []

        def fail_if_called(*args, **kwargs):
            called.append(True)
            raise AssertionError(
                "raster_count_by_value should not run: the multi-band request should have "
                "been rejected first"
            )

        monkeypatch.setattr(dataset_module, "raster_count_by_value", fail_if_called)

        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="count_by_value"
        )
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "single" in excinfo.value.detail
        assert not called

    async def test_cardinality_ceiling_is_enforced(
        self, dataset_factory, base_path, zone_crosswalk, monkeypatch
    ):
        """An unbounded (per-pixel-distinct) column is exactly what the ceiling exists to
        catch -- see config.COUNT_BY_VALUE_MAX_DISTINCT and
        expressions.count_by_value_frame's docstring for the bound math. Unlike the
        multi-band rejection above, this check necessarily reads pixel data (the
        `approx_n_unique` probe) before it can fire -- an accepted cost, not a plan-time
        rejection."""
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _year: raster_array())
        monkeypatch.setattr(dataset_module, "COUNT_BY_VALUE_MAX_DISTINCT", 10)

        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="count_by_value"
        )
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "distinct" in excinfo.value.detail

    async def test_group_aggregate_count_by_value_is_rejected(self, dataset_factory, raster_data):
        dataset = dataset_factory(DIMENSIONS, group_aggregate="count_by_value")
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "count_by_value" in excinfo.value.detail

    async def test_calculate_params_are_rejected_with_count_by_value(
        self, dataset_factory, categorical_raster_data, zone_crosswalk
    ):
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="count_by_value",
            calculate=[{"expression": "value * 10"}],
        )
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "count_by_value" in excinfo.value.detail
