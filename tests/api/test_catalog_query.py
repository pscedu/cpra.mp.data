"""Tests for CatalogQuery, which walks the data tree to report available dimension values.

The query returns three things: `choices` (everything available), `results` (what was
selected, defaulting when the request was unsatisfiable), and `exact` (whether the request
was honoured in full).
"""

import datetime

import pytest
from fastapi import HTTPException

from cpra.mp.data.api.models import DatasetDimensions, FilterDimensions
from cpra.mp.data.catalog import CatalogQuery
from cpra.mp.data.raster import write_cog
from cpra.mp.data.tabular import write_parquet

from tests.factories import ANNUAL_KWARGS, RASTER_KWARGS, annual_frame, raster_array


def query(base_path, dimensions, **filters):
    return CatalogQuery(
        DatasetDimensions(**dimensions), FilterDimensions(**filters), base_path=base_path
    ).execute()


DAMAGE_DIMENSIONS = {
    "variable": "damage",
    "grid": "clara_grid_cell_v001",
    "time_unit": "decadal",
}

# damage is the decadal CLARA variable, and the one with a Float32 `aep` dimension.
DAMAGE_KWARGS = dict(
    variable="damage",
    grid="clara_grid_cell_v001",
    time_unit="decadal",
    model_group_id=500,
    scenario_id=7,
    model="clara_v2023.0.0",
    pumping_id=0.5,
    fragility_scenario=0,
    population_variant=0,
    clara_variant=0,
    asset_type=3,
    percentile=50,
)


def write_damage(base_path, years=(2030,), aeps=(2.0,)):
    """Write a minimal decadal CLARA dataset, one file per (year, aep) pair."""
    import polars as pl

    for year in years:
        for aep in aeps:
            df = pl.DataFrame({**{k: [v] for k, v in DAMAGE_KWARGS.items()}}).with_columns(
                pl.col("model_group_id").cast(pl.Int32),
                pl.col("scenario_id").cast(pl.Int32),
                pl.col("pumping_id").cast(pl.Float32),
                pl.col("fragility_scenario").cast(pl.Int8),
                pl.col("population_variant").cast(pl.Int8),
                pl.col("clara_variant").cast(pl.Int8),
                pl.col("asset_type").cast(pl.Int8),
                pl.col("percentile").cast(pl.Int8),
                aep=pl.lit(aep, dtype=pl.Float32),
                calendar_year=pl.lit(year, dtype=pl.Int32),
            )
            df = df.with_columns(pl.lit(1.0, dtype=pl.Float64).alias("1"))
            write_parquet(df, base_path=base_path, validate=False, sort=False)


@pytest.fixture
def raster_tree(base_path):
    """COGs for two scenarios across three consecutive years."""
    for scenario in (7, 8):
        for year in (2030, 2031, 2032):
            write_cog(
                raster_array(fill=1),
                base_path=base_path,
                calendar_year=year,
                **{**RASTER_KWARGS, "scenario_id": scenario},
            )
    return base_path


@pytest.fixture
def raster_dimensions():
    return {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}


@pytest.fixture
def parquet_tree(base_path):
    write_parquet(annual_frame(years=(2030, 2031, 2032)), base_path=base_path)
    return base_path


@pytest.fixture
def annual_dimensions():
    return {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}


class TestDirectoryWalk:
    def test_reports_available_choices(self, raster_tree, raster_dimensions):
        result = query(raster_tree, raster_dimensions)
        assert result["choices"]["scenario_id"] == [7, 8]
        assert result["choices"]["model_group_id"] == [500]

    def test_choices_are_typed_by_dimension_dtype(self, raster_tree, raster_dimensions):
        result = query(raster_tree, raster_dimensions)
        assert all(isinstance(v, int) for v in result["choices"]["scenario_id"])

    def test_selecting_an_available_value_is_exact(self, raster_tree, raster_dimensions):
        """`exact` is only true when every walked dimension was pinned by the request."""
        result = query(
            raster_tree,
            raster_dimensions,
            model_group_id=[500],
            scenario_id=[8],
            calendar_year=[2031],
        )
        assert result["results"]["scenario_id"] == [8]
        assert result["results"]["calendar_year"] == [2031]
        assert result["exact"] is True

    def test_leaving_a_dimension_unspecified_is_inexact(self, raster_tree, raster_dimensions):
        """model_group_id is unspecified, so it falls back to a default."""
        result = query(raster_tree, raster_dimensions, scenario_id=[8], calendar_year=[2031])
        assert result["results"]["scenario_id"] == [8]
        assert result["exact"] is False

    def test_defaults_to_the_first_choice_when_nothing_is_requested(
        self, raster_tree, raster_dimensions
    ):
        result = query(raster_tree, raster_dimensions)
        assert result["results"]["scenario_id"] == [7]
        assert result["exact"] is False

    def test_unavailable_value_falls_back_and_is_inexact(self, raster_tree, raster_dimensions):
        result = query(raster_tree, raster_dimensions, scenario_id=[99])
        assert result["results"]["scenario_id"] == [7]
        assert result["exact"] is False

    def test_partially_available_selection_is_inexact(self, raster_tree, raster_dimensions):
        result = query(raster_tree, raster_dimensions, scenario_id=[7, 99])
        assert result["results"]["scenario_id"] == [7]
        assert result["exact"] is False

    def test_multiple_values_are_all_selected(self, raster_tree, raster_dimensions):
        result = query(raster_tree, raster_dimensions, scenario_id=[7, 8])
        assert result["results"]["scenario_id"] == [7, 8]

    def test_dataset_dimensions_are_read_from_the_dataset_model(
        self, raster_tree, raster_dimensions
    ):
        result = query(raster_tree, raster_dimensions)
        assert result["results"]["variable"] == ["vct"]
        assert result["results"]["grid"] == ["veg_grid_cell_v001"]

    def test_entries_without_an_equals_sign_are_ignored(self, raster_tree, raster_dimensions):
        (raster_tree / "variable=vct" / "stray_directory").mkdir()
        result = query(raster_tree, raster_dimensions)
        assert result["choices"]["grid"] == ["veg_grid_cell_v001"]


class TestChoiceIntersection:
    """Choices are intersected across every branch the walk actually descends into.

    Selecting a value also prunes the walk (`items[:]` is rewritten in `_update`), so a
    dimension is only intersected across model groups when more than one is selected --
    otherwise the walk never visits the second branch.
    """

    def _write(self, base_path, pairs):
        for model_group, scenario in pairs:
            write_cog(
                raster_array(fill=1),
                base_path=base_path,
                calendar_year=2030,
                **{**RASTER_KWARGS, "model_group_id": model_group, "scenario_id": scenario},
            )

    def test_common_values_across_branches_are_kept(self, base_path, raster_dimensions):
        """Two model groups that share only scenario 7 leave scenario 7 as the choice."""
        self._write(base_path, [(500, 7), (500, 8), (520, 7)])
        result = query(base_path, raster_dimensions, model_group_id=[500, 520])
        assert result["choices"]["scenario_id"] == [7]

    def test_unselected_branches_are_pruned_from_the_walk(self, base_path, raster_dimensions):
        """With only the default model group walked, its own scenarios are the choices."""
        self._write(base_path, [(500, 7), (500, 8), (520, 7)])
        result = query(base_path, raster_dimensions)
        assert result["choices"]["model_group_id"] == [500, 520]
        assert result["choices"]["scenario_id"] == [7, 8]

    def test_no_common_values_raises_404(self, base_path, raster_dimensions):
        self._write(base_path, [(500, 7), (520, 8)])
        with pytest.raises(HTTPException) as excinfo:
            query(base_path, raster_dimensions, model_group_id=[500, 520])
        assert excinfo.value.status_code == 404
        assert "No common values" in excinfo.value.detail


class TestCompressYears:
    def test_contiguous_years_collapse_to_a_range(self, raster_tree, raster_dimensions):
        result = query(raster_tree, raster_dimensions)
        assert result["choices"]["calendar_year"] == [[2030, 2032]]

    def test_gaps_produce_separate_ranges(self, base_path, raster_dimensions):
        for year in (2030, 2031, 2040, 2041):
            write_cog(
                raster_array(fill=1), base_path=base_path, calendar_year=year, **RASTER_KWARGS
            )
        result = query(base_path, raster_dimensions)
        assert result["choices"]["calendar_year"] == [[2030, 2031], [2040, 2041]]

    def test_a_lone_year_is_reported_as_a_scalar(self, base_path, raster_dimensions):
        write_cog(raster_array(fill=1), base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        result = query(base_path, raster_dimensions)
        assert result["choices"]["calendar_year"] == [2030]

    def test_decadal_ranges_align_to_the_decade(self, base_path):
        """A decadal series starting mid-decade is still stepped by 10 from floor(min/10)*10.

        Uses damage/clara_grid_cell_v001 because every decadal variable is a CLARA
        (vector) variable -- no raster grid carries decadal data.
        """
        write_damage(base_path, years=(2035, 2045, 2055))
        result = query(base_path, DAMAGE_DIMENSIONS)
        assert result["choices"]["calendar_year"] == [[2035, 2055]]

    def test_decadal_gaps_still_split(self, base_path):
        write_damage(base_path, years=(2035, 2045, 2075))
        result = query(base_path, DAMAGE_DIMENSIONS)
        assert result["choices"]["calendar_year"] == [[2035, 2045], 2075]


class TestParquetScan:
    def test_reads_choices_from_parquet_columns(self, parquet_tree, annual_dimensions):
        """Non-partition dimensions live inside the file, not in directory names."""
        result = query(parquet_tree, annual_dimensions)
        assert result["choices"]["calendar_year"] == [[2030, 2032]]

    def test_partition_columns_come_from_the_file_too(self, parquet_tree, annual_dimensions):
        result = query(parquet_tree, annual_dimensions)
        assert result["choices"]["model_group_id"] == [500]
        assert result["choices"]["percentile"] == [50]

    def test_selection_narrows_results(self, parquet_tree, annual_dimensions):
        """Only calendar_day is stringified by _compress; calendar_year stays an int."""
        result = query(parquet_tree, annual_dimensions, calendar_year=[2031])
        assert result["results"]["calendar_year"] == [2031]

    def test_multiple_files_are_scanned_together(self, base_path, annual_dimensions):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        write_parquet(annual_frame(years=(2040,), scenario_id=8), base_path=base_path)
        result = query(base_path, annual_dimensions)
        assert result["choices"]["scenario_id"] == [7, 8]


class TestDailyCompression:
    def test_contiguous_days_collapse_to_a_string_range(self, base_path):
        from tests.factories import daily_frame

        days = [datetime.date(2030, 1, 1) + datetime.timedelta(days=i) for i in range(5)]
        write_parquet(daily_frame(days=days), base_path=base_path, validate=False, sort=False)

        result = query(
            base_path, {"variable": "sal", "grid": "hydro_compartment_v001", "time_unit": "daily"}
        )
        assert result["choices"]["calendar_day"] == [["2030-01-01", "2030-01-05"]]

    def test_selected_days_are_stringified(self, base_path):
        from tests.factories import daily_frame

        days = [datetime.date(2030, 1, 1), datetime.date(2030, 1, 2)]
        write_parquet(daily_frame(days=days), base_path=base_path, validate=False, sort=False)

        result = query(
            base_path, {"variable": "sal", "grid": "hydro_compartment_v001", "time_unit": "daily"}
        )
        assert all(isinstance(v, str) for v in result["results"]["calendar_day"])


class TestFloatNormalisation:
    """Float32 storage means a canonical 0.1 reads back as 0.10000000149011612.

    CatalogQuery maps values back onto the dimension's canonical list so that choices,
    results, and /v1/dimensions.json all agree, and so that a client can select using
    either form.
    """

    @pytest.fixture
    def aep_tree(self, base_path):
        write_damage(base_path, aeps=(0.1, 2.0))
        return base_path

    def test_choices_use_canonical_values(self, aep_tree):
        result = query(aep_tree, DAMAGE_DIMENSIONS)
        assert 0.1 in result["choices"]["aep"]

    def test_canonical_input_selects_the_stored_value(self, aep_tree):
        result = query(aep_tree, DAMAGE_DIMENSIONS, aep=[0.1])
        assert result["results"]["aep"] == [0.1]

    def test_float32_repr_input_resolves_to_the_same_value(self, aep_tree):
        """A client echoing back the Float32 repr must not be treated as a miss."""
        result = query(aep_tree, DAMAGE_DIMENSIONS, aep=[0.10000000149011612])
        assert result["results"]["aep"] == [0.1]

    def test_unknown_float_still_falls_back(self, aep_tree):
        result = query(aep_tree, DAMAGE_DIMENSIONS, aep=[99.0])
        assert result["results"]["aep"] == [0.1]
        assert result["exact"] is False
