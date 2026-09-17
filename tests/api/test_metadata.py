"""Narrow characterization tests for `cpra.mp.data.analysis.metadata`.

Every dataset/filter object here is built directly from `DatasetDimensions`/
`FilterDimensions` and a real `GRIDS[...]` entry -- no `dataset_factory`, no catalog
files on disk, and no `BaseAnalysis` subclass. `estimate_data_size` and
`is_raster_metadata_table` need none of that: they take exactly what they read (a
dimensions pair, a grid, a lazyframe) and answer from declared structure alone, so a
dedicated file can exercise them as plain functions.
"""

import types

import polars as pl
import pytest

from cpra.mp.data.analysis import metadata as metadata_module
from cpra.mp.data.analysis.metadata import estimate_data_size, is_raster_metadata_table, variable_for
from cpra.mp.data.analysis.parameters import DatasetDimensions, FilterDimensions
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.variables import VARIABLES

# Plain-kwarg dataset shapes, mirroring tests/factories.py's ANNUAL_KWARGS / DAILY_KWARGS /
# RASTER_KWARGS but kept local so this file has no dependency on dataset_factory or the
# fixtures built around it.
ANNUAL_KWARGS = dict(
    variable="ssf",
    grid="ecoregion_v001",
    time_unit="annual",
    model_group_id=500,
    scenario_id=7,
    percentile=50,
    model="icm_v2023.0.0",
)
DAILY_MONTHLY_KWARGS = dict(
    variable="sal",
    grid="hydro_compartment_v001",
    time_unit="monthly",
    model_group_id=500,
    scenario_id=7,
    aggregate_type="mean",
    model="icm_v2023.0.0",
)
RASTER_KWARGS = dict(
    variable="vct",
    grid="veg_grid_cell_v001",
    time_unit="annual",
    model_group_id=500,
    scenario_id=7,
    model="icm_v2023.0.0",
)


class TestVariableFor:
    def test_looks_up_the_real_catalog_entry(self):
        dataset_dimensions = DatasetDimensions(**ANNUAL_KWARGS)
        resolved = variable_for(dataset_dimensions)
        assert resolved is VARIABLES["ssf"]
        assert resolved.name == "ssf"


class TestIsRasterMetadataTable:
    def test_raster_grid_with_a_path_column_is_metadata(self):
        band_metadata = pl.LazyFrame({"path": ["a.tif"], "band": [1], "calendar_year": [2030]})
        assert is_raster_metadata_table(band_metadata, "raster") is True

    def test_raster_grid_without_a_path_column_is_not_metadata(self):
        sampled = pl.LazyFrame({"grid_cell": [1], "value": [1.0]})
        assert is_raster_metadata_table(sampled, "raster") is False

    def test_a_vector_grid_is_never_metadata_even_with_a_path_column(self):
        band_metadata = pl.LazyFrame({"path": ["a.tif"], "band": [1]})
        assert is_raster_metadata_table(band_metadata, "vector") is False


class TestEstimateDataSize:
    def test_every_dimension_explicitly_filtered_gives_an_exact_estimate(self):
        """ssf is Float32 on the 32-cell ecoregion grid; every required column has an
        explicit single-value filter, so rows and columns are both exactly known."""
        dataset_dimensions = DatasetDimensions(**ANNUAL_KWARGS)
        filter_dimensions = FilterDimensions(
            model_group_id=[500], scenario_id=[7], percentile=[50], calendar_year=[2030]
        )
        grid = GRIDS["ecoregion_v001"]

        result = estimate_data_size(
            dataset_dimensions, filter_dimensions, grid, [], has_group_by=False, grid_cell_ids=None
        )

        assert result == pytest.approx(32 * 32 / 8)

    def test_grid_cell_ids_override_the_grids_column_count(self):
        dataset_dimensions = DatasetDimensions(**ANNUAL_KWARGS)
        filter_dimensions = FilterDimensions(
            model_group_id=[500], scenario_id=[7], percentile=[50], calendar_year=[2030]
        )
        grid = GRIDS["ecoregion_v001"]

        result = estimate_data_size(
            dataset_dimensions,
            filter_dimensions,
            grid,
            [],
            has_group_by=False,
            grid_cell_ids=["1", "2"],
        )

        assert result == pytest.approx(2 * 4)

    def test_a_required_dimension_with_no_filter_and_no_declared_values_is_unknowable(self):
        """model_group_id has no CSV-declared values and no filter here, so rows cannot be
        estimated -- which must come back as None, not as a zero-byte estimate."""
        dataset_dimensions = DatasetDimensions(**ANNUAL_KWARGS)
        grid = GRIDS["ecoregion_v001"]

        result = estimate_data_size(
            dataset_dimensions, FilterDimensions(), grid, [], has_group_by=False, grid_cell_ids=None
        )

        assert result is None

    @pytest.mark.parametrize(
        ("time_unit", "expected_year_count"),
        [("decadal", 6), ("annual", 52)],
    )
    def test_calendar_year_falls_back_to_a_default_count_when_unfiltered(
        self, time_unit, expected_year_count
    ):
        """calendar_year has no CSV-declared values at all (see dimensions.py), so whenever
        it is required and not explicitly filtered, estimate_data_size falls back to a
        hardcoded count that depends on time_unit."""
        dataset_dimensions = DatasetDimensions(**{**ANNUAL_KWARGS, "time_unit": time_unit})
        filter_dimensions = FilterDimensions(model_group_id=[500], scenario_id=[7], percentile=[50])
        grid = GRIDS["ecoregion_v001"]

        result = estimate_data_size(
            dataset_dimensions, filter_dimensions, grid, [], has_group_by=False, grid_cell_ids=None
        )

        assert result == pytest.approx(expected_year_count * 32 * 4)

    def test_calendar_day_falls_back_to_the_monthly_default_count_when_unfiltered(self):
        """Same fallback as calendar_year, for the monthly branch specifically (the daily
        branch is exercised elsewhere via a plain, unfiltered daily dataset)."""
        dataset_dimensions = DatasetDimensions(**DAILY_MONTHLY_KWARGS)
        filter_dimensions = FilterDimensions(model_group_id=[500], scenario_id=[7], aggregate_type=["mean"])
        grid = GRIDS["hydro_compartment_v001"]

        result = estimate_data_size(
            dataset_dimensions, filter_dimensions, grid, [], has_group_by=False, grid_cell_ids=None
        )

        assert result == pytest.approx(52 * 12 * grid.id_count * 4)

    def test_a_raster_grid_falls_back_to_height_times_width_when_it_has_no_id_count(self):
        dataset_dimensions = DatasetDimensions(**RASTER_KWARGS)
        filter_dimensions = FilterDimensions(model_group_id=[500], scenario_id=[7], calendar_year=[2030])
        grid = GRIDS["veg_grid_cell_v001"]

        result = estimate_data_size(
            dataset_dimensions, filter_dimensions, grid, [], has_group_by=False, grid_cell_ids=None
        )

        assert result == pytest.approx(grid.height * grid.width * 4)

    def test_a_grid_with_neither_an_id_count_nor_a_height_and_width_is_unknowable(self):
        """A grid-like object exposing none of the attributes estimate_data_size looks for
        (`id_count`, `height`/`width`) can't yield a column count, so the estimate must
        come back as None rather than guessing."""
        dataset_dimensions = DatasetDimensions(**RASTER_KWARGS)
        filter_dimensions = FilterDimensions(model_group_id=[500], scenario_id=[7], calendar_year=[2030])
        columnless_grid = types.SimpleNamespace()

        result = estimate_data_size(
            dataset_dimensions,
            filter_dimensions,
            columnless_grid,
            [],
            has_group_by=False,
            grid_cell_ids=None,
        )

        assert result is None

    def test_a_dtype_name_with_no_single_bit_width_is_unknowable(self, monkeypatch):
        """`variable.dtype`'s string form is parsed for exactly one run of digits (e.g.
        `Float32` -> `32`). A dtype name with zero or multiple digit runs can't be turned
        into a bytes-per-value figure, so the estimate must come back as None."""
        dataset_dimensions = DatasetDimensions(**ANNUAL_KWARGS)
        filter_dimensions = FilterDimensions(
            model_group_id=[500], scenario_id=[7], percentile=[50], calendar_year=[2030]
        )
        grid = GRIDS["ecoregion_v001"]

        monkeypatch.setattr(
            metadata_module, "variable_for", lambda _: types.SimpleNamespace(dtype="Boolean")
        )

        result = estimate_data_size(
            dataset_dimensions, filter_dimensions, grid, [], has_group_by=False, grid_cell_ids=None
        )

        assert result is None
