"""Tests for cpra.mp.data.utils, the shared helpers behind reading and writing.

These functions are pure and carry most of the library's implicit contract: which columns
are required, which become directories, how keyword arguments turn into glob patterns and
filter expressions, and what counts as valid data.
"""

import datetime

import numpy as np
import polars as pl
import pytest
from scipy.io import FortranFile

from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.utils import (
    build_glob_pattern,
    cast_columns,
    dataset_path,
    extract_dimension_value,
    get_filter_expressions,
    get_input_data_frame,
    glob_lazy_frame,
    partition_columns,
    read_fortran_array,
    required_columns,
    validate_data_frame,
    validate_grid,
)
from tests.factories import annual_frame, daily_frame


class TestPartitionColumns:
    def test_annual_partitions_on_dataset_dimensions_only(self):
        """For non-daily data, only variable/time_unit/grid become directories."""
        assert partition_columns(annual_frame()) == ["variable", "time_unit", "grid"]

    def test_daily_adds_partition_dimensions(self):
        """Daily data additionally partitions on the required partition dimensions."""
        assert partition_columns(daily_frame()) == [
            "variable",
            "time_unit",
            "grid",
            "model_group_id",
            "scenario_id",
            "aggregate_type",
        ]

    def test_exclude_dataset_dimensions(self):
        assert partition_columns(annual_frame(), include_dataset=False) == []

    def test_daily_excluding_dataset_dimensions(self):
        assert partition_columns(daily_frame(), include_dataset=False) == [
            "model_group_id",
            "scenario_id",
            "aggregate_type",
        ]


class TestRequiredColumns:
    def test_annual_required_columns(self):
        assert required_columns(annual_frame()) == [
            "variable",
            "time_unit",
            "grid",
            "model_group_id",
            "scenario_id",
            "percentile",
            "calendar_year",
            "model",
        ]

    def test_daily_requires_calendar_day_not_year(self):
        cols = required_columns(daily_frame())
        assert "calendar_day" in cols
        assert "calendar_year" not in cols

    def test_annual_requires_calendar_year_not_day(self):
        cols = required_columns(annual_frame())
        assert "calendar_year" in cols
        assert "calendar_day" not in cols

    def test_exclude_dataset(self):
        cols = required_columns(annual_frame(), include_dataset=False)
        assert not {"variable", "grid", "time_unit"} & set(cols)

    def test_exclude_metadata(self):
        assert "model" not in required_columns(annual_frame(), include_metadata=False)

    def test_include_geography_and_value_append_in_order(self):
        cols = required_columns(annual_frame(), include_grid_cell=True, include_value=True)
        assert cols[-2:] == ["grid_cell", "value"]

    def test_variable_specific_requirement(self):
        """percentile is required for ssf but not for sal."""
        assert "percentile" in required_columns(annual_frame())
        assert "aggregate_type" in required_columns(daily_frame())
        assert "percentile" not in required_columns(daily_frame())


class TestValidateDataFrame:
    def test_valid_frame_passes(self):
        validate_data_frame(annual_frame())

    def test_missing_required_column_raises(self):
        df = annual_frame().drop("percentile")
        with pytest.raises(ValidationError, match="Required columns are missing: percentile"):
            validate_data_frame(df)

    def test_wrong_dtype_raises(self):
        df = annual_frame().with_columns(pl.col("model_group_id").cast(pl.Int64))
        with pytest.raises(ValidationError, match="expected Int32"):
            validate_data_frame(df)

    def test_null_value_raises(self):
        df = annual_frame().with_columns(model_group_id=pl.lit(None, dtype=pl.Int32))
        with pytest.raises(ValidationError, match="model_group_id_is_not_null"):
            validate_data_frame(df)

    def test_value_below_minimum_raises(self):
        df = annual_frame().with_columns(calendar_year=pl.lit(1800, dtype=pl.Int32))
        with pytest.raises(ValidationError, match="calendar_year_minimum_value"):
            validate_data_frame(df)

    def test_value_above_maximum_raises(self):
        df = annual_frame().with_columns(calendar_year=pl.lit(2500, dtype=pl.Int32))
        with pytest.raises(ValidationError, match="calendar_year_maximum_value"):
            validate_data_frame(df)

    def test_unknown_dimension_value_raises(self):
        df = annual_frame().with_columns(variable=pl.lit("not_a_variable"))
        with pytest.raises(ValidationError, match="variable_values"):
            validate_data_frame(df)

    def test_variable_grid_mismatch_raises(self):
        """ssf is an ecoregion variable, so pairing it with a hydro grid must fail."""
        df = annual_frame().with_columns(grid=pl.lit("hydro_compartment_v001"))
        with pytest.raises(ValidationError, match="variable_matches_grid"):
            validate_data_frame(df)

    def test_variable_time_unit_mismatch_raises(self):
        """ssf is annual only.

        Uses decadal rather than daily: both annual and decadal require calendar_year, so
        the frame stays structurally valid and the mismatch check is what fails. Switching
        to daily would trip the missing-calendar_day check first.
        """
        df = annual_frame().with_columns(time_unit=pl.lit("decadal"))
        with pytest.raises(ValidationError, match="variable_matches_time_unit"):
            validate_data_frame(df)

    def test_time_unit_change_can_alter_which_columns_are_required(self):
        """Marking annual data as daily makes calendar_day required and year irrelevant."""
        df = annual_frame().with_columns(time_unit=pl.lit("daily"))
        with pytest.raises(ValidationError, match="Required columns are missing: calendar_day"):
            validate_data_frame(df)

    def test_error_reports_failed_row_count(self):
        df = annual_frame(years=(2030, 2040)).with_columns(calendar_year=pl.Series([1800, 1801], dtype=pl.Int32))
        with pytest.raises(ValidationError, match="^2 rows failed validation"):
            validate_data_frame(df)

    def test_explicit_required_cols_narrows_the_check(self):
        """read_parquet validates only the dataset dimensions of its kwargs."""
        df = pl.DataFrame({"variable": ["ssf"], "grid": ["ecoregion_v001"], "time_unit": ["annual"]})
        validate_data_frame(df, required_cols=["variable", "grid", "time_unit"])

    def test_grid_version_suffix_is_stripped_before_matching(self):
        """The variable/grid check compares against the unversioned geographic unit."""
        validate_data_frame(annual_frame(grid="ecoregion_v001"))


class TestExtractDimensionValue:
    def test_kwargs_take_priority_over_data(self):
        assert extract_dimension_value("grid", {"grid": "from_data"}, grid="from_kwargs") == "from_kwargs"

    def test_reads_from_a_dict(self):
        assert extract_dimension_value("grid", {"grid": "ecoregion_v001"}) == "ecoregion_v001"

    def test_reads_first_row_of_a_data_frame(self):
        df = pl.DataFrame({"grid": ["ecoregion_v001", "other"]})
        assert extract_dimension_value("grid", df) == "ecoregion_v001"

    @pytest.mark.parametrize(
        "data",
        [None, {}, pl.DataFrame({"other": [1]}), pl.DataFrame({"grid": []}), 42, "a string"],
        ids=["none", "empty-dict", "missing-column", "empty-column", "int", "str"],
    )
    def test_returns_none_when_unavailable(self, data):
        assert extract_dimension_value("grid", data) is None


class TestDatasetPath:
    def test_builds_dataset_dimension_prefix(self, tmp_path):
        path = dataset_path(tmp_path, variable="ssf", grid="ecoregion_v001", time_unit="annual")
        assert path == tmp_path / "variable=ssf" / "time_unit=annual" / "grid=ecoregion_v001"

    def test_ignores_extra_kwargs(self, tmp_path):
        path = dataset_path(tmp_path, variable="ssf", grid="ecoregion_v001", time_unit="annual", scenario_id=7)
        assert "scenario_id" not in str(path)

    def test_missing_dataset_dimension_raises(self, tmp_path):
        with pytest.raises(KeyError):
            dataset_path(tmp_path, variable="ssf")


class TestValidateGrid:
    def test_returns_the_grid_object(self):
        assert validate_grid(grid="ecoregion_v001").name == "ecoregion_v001"

    def test_reads_the_grid_from_a_data_frame(self):
        assert validate_grid(annual_frame()).name == "ecoregion_v001"

    def test_missing_grid_raises(self):
        with pytest.raises(ValidationError, match="Grid is required"):
            validate_grid(variable="ssf")

    def test_unknown_grid_raises(self):
        with pytest.raises(ValidationError, match="not a valid grid"):
            validate_grid(grid="nope_v001")


class TestCastColumns:
    def test_casts_known_dimension_columns(self):
        df = pl.DataFrame({"model_group_id": [500], "scenario_id": [7], "calendar_year": [2030]})
        result = cast_columns(df)
        assert result["model_group_id"].dtype == pl.Int32
        assert result["scenario_id"].dtype == pl.Int32
        assert result["calendar_year"].dtype == pl.Int32

    def test_leaves_unknown_columns_alone(self):
        df = pl.DataFrame({"model_group_id": [500], "1": [1.5]})
        result = cast_columns(df)
        assert result["1"].dtype == pl.Float64

    def test_casts_dates_from_strings(self):
        df = pl.DataFrame({"calendar_day": ["2030-01-01"]})
        assert cast_columns(df)["calendar_day"].item() == datetime.date(2030, 1, 1)


class TestBuildGlobPattern:
    def test_scalar_values_are_substituted(self):
        pattern = build_glob_pattern(["variable", "grid"], variable="ssf", grid="ecoregion_v001")
        assert pattern == "variable=ssf/grid=ecoregion_v001"

    def test_missing_values_become_wildcards(self):
        pattern = build_glob_pattern(["variable", "scenario_id"], variable="ssf")
        assert pattern == "variable=ssf/scenario_id=*"

    def test_single_element_list_is_treated_as_a_scalar(self):
        pattern = build_glob_pattern(["scenario_id"], scenario_id=[7])
        assert pattern == "scenario_id=7"

    def test_multi_element_list_becomes_a_wildcard(self):
        """A glob can't express alternatives, so multi-valued filters widen the match."""
        pattern = build_glob_pattern(["scenario_id"], scenario_id=[7, 8])
        assert pattern == "scenario_id=*"

    def test_extension_is_appended(self):
        pattern = build_glob_pattern(["variable"], extension="tif", variable="ssf")
        assert pattern == "variable=ssf.tif"

    def test_no_extension_leaves_the_pattern_bare(self):
        assert build_glob_pattern(["variable"], variable="ssf") == "variable=ssf"


class TestGlobLazyFrame:
    @pytest.fixture
    def tree(self, tmp_path):
        for scenario in (7, 8):
            for year in (2030, 2040):
                path = tmp_path / f"variable=ssf/grid=ecoregion_v001/scenario_id={scenario}"
                path.mkdir(parents=True, exist_ok=True)
                (path / f"calendar_year={year}.tif").touch()
        return tmp_path

    def test_extracts_dimension_values_from_paths(self, tree):
        cols = ["variable", "grid", "scenario_id", "calendar_year"]
        df = glob_lazy_frame(cols, base_path=tree, extension="tif").collect()
        assert df["scenario_id"].to_list() == [7, 7, 8, 8]
        assert df["calendar_year"].to_list() == [2030, 2040, 2030, 2040]

    def test_casts_extracted_values_to_dimension_dtypes(self, tree):
        df = glob_lazy_frame(["variable", "scenario_id"], base_path=tree, extension="tif").collect()
        assert df["scenario_id"].dtype == pl.Int32

    def test_sorted_by_required_columns(self, tree):
        cols = ["variable", "grid", "scenario_id", "calendar_year"]
        df = glob_lazy_frame(cols, base_path=tree, extension="tif").collect()
        assert df.sort(cols).equals(df)

    def test_path_column_is_included_by_default(self, tree):
        df = glob_lazy_frame(["variable"], base_path=tree, extension="tif").collect()
        assert "path" in df.columns

    def test_path_column_can_be_suppressed(self, tree):
        df = glob_lazy_frame(["variable"], base_path=tree, extension="tif", include_file_paths=None).collect()
        assert "path" not in df.columns

    def test_path_column_can_be_renamed(self, tree):
        df = glob_lazy_frame(["variable"], base_path=tree, extension="tif", include_file_paths="src").collect()
        assert "src" in df.columns

    def test_filters_narrow_the_glob(self, tree):
        cols = ["variable", "grid", "scenario_id", "calendar_year"]
        df = glob_lazy_frame(cols, base_path=tree, extension="tif", scenario_id=7).collect()
        assert df["scenario_id"].unique().to_list() == [7]

    def test_no_matches_yields_an_empty_frame(self, tmp_path):
        df = glob_lazy_frame(["variable"], base_path=tmp_path, extension="tif").collect()
        assert df.height == 0

    def test_base_path_defaults_to_the_configured_path(self, tmp_path, monkeypatch):
        """base_path is resolved at call time, so overriding the config takes effect."""
        from cpra.mp.data import config

        # With only `variable` required, the glob is "variable=*.tif", so the dimension
        # value lives in the filename rather than in a directory name.
        (tmp_path / "variable=ssf.tif").touch()
        monkeypatch.setattr(config, "BASE_PATH", tmp_path)
        df = glob_lazy_frame(["variable"], extension="tif").collect()
        assert df["variable"].to_list() == ["ssf"]


class TestGetInputDataFrame:
    def test_scalar_kwargs_produce_one_row(self):
        df = get_input_data_frame(variable="ssf", grid="ecoregion_v001")
        assert df.shape == (1, 2)

    def test_lists_are_cross_joined(self):
        df = get_input_data_frame(scenario_id=[7, 8], calendar_year=[2030, 2040, 2050])
        assert df.height == 6

    def test_column_order_follows_kwargs(self):
        df = get_input_data_frame(grid="ecoregion_v001", variable="ssf")
        assert df.columns == ["grid", "variable"]

    def test_no_kwargs_returns_none(self):
        assert get_input_data_frame() is None


class TestGetFilterExpressions:
    def _apply(self, df, **kwargs):
        exprs = get_filter_expressions(**kwargs)
        return df.filter(*exprs) if exprs else df

    @pytest.fixture
    def df(self):
        return pl.DataFrame(
            {
                "scenario_id": pl.Series([7, 8, 9], dtype=pl.Int32),
                "calendar_year": pl.Series([2030, 2040, 2050], dtype=pl.Int32),
            }
        )

    def test_scalar_becomes_equality(self, df):
        assert self._apply(df, scenario_id=7)["scenario_id"].to_list() == [7]

    def test_list_becomes_membership(self, df):
        assert self._apply(df, scenario_id=[7, 9])["scenario_id"].to_list() == [7, 9]

    def test_tuple_becomes_membership(self, df):
        assert self._apply(df, scenario_id=(8, 9))["scenario_id"].to_list() == [8, 9]

    def test_dataset_dimensions_are_excluded(self):
        """Dataset dimensions are directories, not columns, so they never become filters."""
        assert get_filter_expressions(variable="ssf", grid="ecoregion_v001", time_unit="annual") == []

    def test_values_are_cast_to_the_dimension_dtype(self, df):
        """A Python int must still match an Int32 column."""
        assert self._apply(df, scenario_id=7).height == 1

    def test_multiple_filters_are_combined(self, df):
        result = self._apply(df, scenario_id=[7, 8], calendar_year=2040)
        assert result["scenario_id"].to_list() == [8]


class TestReadFortranArray:
    def _write(self, path, array):
        with FortranFile(path, "w") as f:
            f.write_record(array)

    def test_round_trip_with_explicit_dtype(self, tmp_path):
        path = tmp_path / "a.xyz.b"
        array = np.array([1.5, 2.5, 3.5], dtype="float32")
        self._write(path, array)
        assert np.array_equal(read_fortran_array(path, dtype="float32"), array)

    def test_dtype_derived_from_variable(self, tmp_path):
        """lnd_type is Int32, so the record is read as int32 without an explicit dtype."""
        path = tmp_path / "a.xyz.b"
        array = np.array([1, 2, 3], dtype="int32")
        self._write(path, array)
        result = read_fortran_array(path, variable="lnd_type")
        assert result.dtype == np.int32
        assert np.array_equal(result, array)

    def test_float_variable_dtype(self, tmp_path):
        """inun is Float32, so the record is read as float32 without an explicit dtype."""
        path = tmp_path / "a.xyz.b"
        array = np.array([1.5, 2.5], dtype="float32")
        self._write(path, array)
        assert read_fortran_array(path, variable="inun").dtype == np.float32

    def test_missing_dtype_and_variable_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Either variable or dtype must be provided"):
            read_fortran_array(tmp_path / "a.xyz.b")

    def test_unknown_variable_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid variable"):
            read_fortran_array(tmp_path / "a.xyz.b", variable="not_a_variable")
