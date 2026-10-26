"""Tests for `cpra.mp.data.analysis.io.catalog`'s vector half -- `band_dimension_columns`
and `resolve_vector_bands`, the plan-time answer to "how many bands will a
`("project", ("vector", "raster"))` step off this input produce".

Real structure -- `ssf` on `ecoregion_v001` -- via `tests/factories.py`'s
`write_annual_dataset`, matching `test_vector_scan.py`'s convention (this reads the same
datasets through the same scan). Every fixture writes through `base_path`
(`tmp_path / "data"`, never a real data directory -- see `tests/conftest.py`).

Which bands the enumeration then becomes at execution is `test_project_raster.py`'s
`TestVectorToRasterPlannerSuppliedBands`; which pushdown it is made under is
`test_execution_plan.py`'s `TestVectorBandEnumeration`.
"""

import polars as pl
import pytest

from cpra.mp.data.analysis.io.catalog import band_dimension_columns, resolve_vector_bands
from cpra.mp.data.analysis.request.errors import NotFoundError
from cpra.mp.data.analysis.request.pipeline import RESERVED_BAND_INDEX_COLUMN, InputSelection
from tests.factories import write_annual_dataset

SELECTION = InputSelection(variable="ssf", time_unit="annual", grid="ecoregion_v001")


@pytest.fixture
def annual_dataset(base_path):
    write_annual_dataset(base_path, years=(2030, 2040))
    return base_path


class TestBandDimensionColumns:
    def test_drops_the_two_long_shape_columns_and_the_three_dataset_ones(self):
        names = ["variable", "time_unit", "grid", "scenario_id", "calendar_year", "grid_cell", "value"]
        assert band_dimension_columns(names) == ["scenario_id", "calendar_year"]

    def test_preserves_the_order_it_is_given(self):
        # The pivot key is positional on both sides of the seam (the plan's enumeration and
        # `project_vector_to_raster`'s own schema read), so this rule must never re-sort.
        assert band_dimension_columns(["calendar_year", "scenario_id"]) == ["calendar_year", "scenario_id"]

    def test_a_frame_of_nothing_but_the_excluded_columns_has_no_band_dimension(self):
        assert band_dimension_columns(["grid_cell", "value", "variable"]) == []


class TestResolveVectorBands:
    def test_one_row_per_distinct_combination_sorted(self, annual_dataset):
        # `write_annual_dataset` varies only `calendar_year`; every other dimension it
        # writes (`model_group_id`, `scenario_id`, `percentile`, `model`) is constant, so
        # two years across 32 cells is two BANDS, not 64.
        bands = resolve_vector_bands(SELECTION, base_path=annual_dataset)
        assert bands.height == 2
        assert bands["calendar_year"].to_list() == [2030, 2040]

    def test_every_varying_dimension_is_part_of_the_combination(self, base_path):
        write_annual_dataset(base_path, years=(2030, 2040), scenario_id=7)
        write_annual_dataset(base_path, years=(2030, 2040), scenario_id=8)

        bands = resolve_vector_bands(SELECTION, base_path=base_path)

        assert bands.height == 4
        assert list(zip(bands["scenario_id"].to_list(), bands["calendar_year"].to_list())) == [
            (7, 2030),
            (7, 2040),
            (8, 2030),
            (8, 2040),
        ]

    def test_the_predicate_narrows_the_enumeration(self, base_path):
        write_annual_dataset(base_path, years=(2030, 2040), scenario_id=7)
        write_annual_dataset(base_path, years=(2030, 2040), scenario_id=8)

        bands = resolve_vector_bands(SELECTION, base_path=base_path, predicate=pl.col("scenario_id") == 8)

        assert bands["scenario_id"].to_list() == [8, 8]
        assert bands["calendar_year"].to_list() == [2030, 2040]

    def test_dimensions_narrows_to_the_names_still_standing(self, base_path):
        write_annual_dataset(base_path, years=(2030, 2040), scenario_id=7)
        write_annual_dataset(base_path, years=(2030, 2040), scenario_id=8)

        # What an upstream `group by [grid_cell, calendar_year]` leaves: `scenario_id` no
        # longer has a value per row, so it cannot split bands either, and four
        # combinations collapse to two.
        bands = resolve_vector_bands(
            SELECTION, base_path=base_path, dimensions=frozenset({"grid_cell", "calendar_year"})
        )

        assert bands.columns == ["calendar_year"]
        assert bands["calendar_year"].to_list() == [2030, 2040]

    def test_no_dimension_at_all_is_one_band_carrying_the_reserved_index(self, annual_dataset):
        bands = resolve_vector_bands(SELECTION, base_path=annual_dataset, dimensions=frozenset({"grid_cell"}))

        assert bands.columns == [RESERVED_BAND_INDEX_COLUMN]
        assert bands[RESERVED_BAND_INDEX_COLUMN].to_list() == [0]

    def test_columns_projection_is_applied_like_the_scan_applies_it(self, base_path):
        write_annual_dataset(base_path, years=(2030, 2040), scenario_id=7)
        write_annual_dataset(base_path, years=(2030, 2040), scenario_id=8)

        # A `group` is what makes the plan project the scan's columns; a dimension left out
        # of that projection is not read, so it cannot split bands here either.
        bands = resolve_vector_bands(SELECTION, base_path=base_path, columns=["calendar_year", "grid_cell", "value"])

        assert bands.columns == ["calendar_year"]
        assert bands["calendar_year"].to_list() == [2030, 2040]

    def test_a_dataset_with_nothing_on_disk_is_not_enumerated(self, base_path):
        # Nothing written: compiling a plan must stay possible, so this is `None` ("not a
        # plan-time fact") rather than an error or a fabricated zero.
        assert resolve_vector_bands(SELECTION, base_path=base_path) is None

    def test_a_combination_that_is_not_a_real_dataset_raises_not_found(self, annual_dataset):
        # `ssf` is an ecoregion variable; `hydro_compartment_v001` is not its grid, which
        # the underlying read rejects structurally rather than by looking for files.
        selection = InputSelection(variable="ssf", time_unit="annual", grid="hydro_compartment_v001")
        with pytest.raises(NotFoundError):
            resolve_vector_bands(selection, base_path=annual_dataset)
