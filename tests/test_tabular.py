"""Tests for reading, writing, and compacting wide-format Parquet datasets."""

import datetime
from pathlib import Path

import polars as pl
import pytest

from cpra.mp.data import config
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.tabular import compact_parquet, read_parquet, write_parquet

from tests.factories import (
    ANNUAL_KWARGS,
    DAILY_KWARGS,
    ECOREGION_IDS,
    annual_frame,
    daily_frame,
)

DATASET_KWARGS = {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}


class TestWriteParquet:
    def test_writes_a_partitioned_dataset(self, base_path):
        write_parquet(annual_frame(), base_path=base_path)
        files = list(base_path.rglob("*.parquet"))
        assert len(files) == 1
        assert "variable=ssf/grid=ecoregion_v001/time_unit=annual" in str(files[0])

    def test_annual_data_keeps_partition_dimensions_as_columns(self, base_path):
        """Only daily data promotes model_group_id and friends to directories."""
        write_parquet(annual_frame(), base_path=base_path)
        path = next(base_path.rglob("*.parquet"))
        assert "model_group_id=" not in str(path)
        assert "model_group_id" in pl.read_parquet(path).columns

    def test_daily_data_promotes_partition_dimensions_to_directories(self, base_path):
        write_parquet(daily_frame(), base_path=base_path, validate=False, sort=False)
        path = str(next(base_path.rglob("*.parquet")))
        for segment in ("model_group_id=500", "scenario_id=7", "aggregate_type=mean"):
            assert segment in path

    def test_kwargs_are_added_as_columns(self, base_path):
        """Dimension values can be supplied as keyword arguments instead of columns."""
        df = annual_frame().drop("model_group_id")
        write_parquet(df, base_path=base_path, model_group_id=501)
        result = pl.read_parquet(next(base_path.rglob("*.parquet")))
        assert result["model_group_id"].unique().to_list() == [501]

    def test_long_format_input_is_pivoted(self, base_path):
        """A frame with geography/value columns is pivoted to wide before writing."""
        long_df = pl.DataFrame(
            {
                **{k: [v] * 3 for k, v in ANNUAL_KWARGS.items() if k != "grid"},
                "grid": ["ecoregion_v001"] * 3,
                "calendar_year": [2030] * 3,
                "geography": ["1", "2", "3"],
                "value": [1.0, 2.0, 3.0],
            }
        ).with_columns(
            pl.col("model_group_id").cast(pl.Int32),
            pl.col("scenario_id").cast(pl.Int32),
            pl.col("percentile").cast(pl.Int8),
            pl.col("calendar_year").cast(pl.Int32),
        )
        write_parquet(long_df, base_path=base_path, validate=False, sort=False)
        result = pl.read_parquet(next(base_path.rglob("*.parquet")))
        assert {"1", "2", "3"} <= set(result.columns)
        assert "geography" not in result.columns

    def test_accepts_a_dict_of_series(self, base_path):
        """A plain dict is accepted, but only Series preserve the required dtypes."""
        write_parquet(annual_frame().to_dict(as_series=True), base_path=base_path)
        assert list(base_path.rglob("*.parquet"))

    def test_dict_of_python_lists_loses_dtypes(self, base_path):
        """Round-tripping through plain lists widens Int32 to Int64 and fails validation."""
        with pytest.raises(ValidationError, match="expected Int32"):
            write_parquet(annual_frame().to_dict(as_series=False), base_path=base_path)

    def test_sorts_geography_columns_numerically(self, base_path):
        """Column order must be 1, 2, ... 10, not the lexical 1, 10, 2."""
        write_parquet(annual_frame(), base_path=base_path)
        result = pl.read_parquet(next(base_path.rglob("*.parquet")))
        numeric = [c for c in result.columns if c[0].isdigit()]
        assert numeric == [str(i) for i in ECOREGION_IDS]

    def test_sorts_rows_by_required_columns(self, base_path):
        write_parquet(annual_frame(years=(2050, 2030, 2040)), base_path=base_path)
        result = pl.read_parquet(next(base_path.rglob("*.parquet")))
        assert result["calendar_year"].to_list() == [2030, 2040, 2050]

    def test_sort_false_preserves_input_order(self, base_path):
        write_parquet(annual_frame(years=(2050, 2030)), base_path=base_path, sort=False)
        result = pl.read_parquet(next(base_path.rglob("*.parquet")))
        assert result["calendar_year"].to_list() == [2050, 2030]


class TestWriteParquetValidation:
    def test_unexpected_alpha_column_raises(self, base_path):
        df = annual_frame().with_columns(surprise=pl.lit("x"))
        with pytest.raises(ValidationError, match="Invalid columns: surprise"):
            write_parquet(df, base_path=base_path)

    def test_missing_required_column_raises(self, base_path):
        df = annual_frame().drop("percentile")
        with pytest.raises(ValidationError, match="percentile"):
            write_parquet(df, base_path=base_path)

    def test_sort_reports_a_column_required_by_the_data_itself(self, base_path):
        """With validation off, the sort branch still catches a column that the frame's
        own dimension values imply: annual data must carry calendar_year."""
        df = annual_frame().drop("calendar_year")
        with pytest.raises(ValidationError, match="Missing columns: calendar_year"):
            write_parquet(df, base_path=base_path, validate=False, sort=True)

    def test_wrong_geography_column_count_raises(self, base_path):
        """ecoregion_v001 declares id_count=32."""
        df = annual_frame(geo_ids=(1, 2, 3))
        with pytest.raises(ValidationError, match="expected 32, found 3"):
            write_parquet(df, base_path=base_path)

    def test_geography_id_above_max_raises(self, base_path):
        df = annual_frame(geo_ids=tuple(range(2, 34)))
        with pytest.raises(ValidationError, match="Invalid geography ID"):
            write_parquet(df, base_path=base_path)

    def test_geography_id_below_min_raises(self, base_path):
        df = annual_frame(geo_ids=tuple(range(0, 32)))
        with pytest.raises(ValidationError, match="Invalid geography ID"):
            write_parquet(df, base_path=base_path)

    def test_wrong_value_dtype_raises(self, base_path):
        """ssf values are Float32; Float64 columns must be rejected."""
        df = annual_frame().with_columns(pl.col("1").cast(pl.Float64))
        with pytest.raises(ValidationError, match="expected Float32"):
            write_parquet(df, base_path=base_path)

    def test_validate_and_sort_false_skips_all_checks(self, base_path):
        df = annual_frame(geo_ids=(1, 2, 3))
        write_parquet(df, base_path=base_path, validate=False, sort=False)
        assert list(base_path.rglob("*.parquet"))

    def test_sort_enforces_column_checks_even_when_validate_is_false(self, base_path):
        """The column-name and geography-ID checks live in the `validate or sort` block,
        so turning validation off does not disable them unless sorting is off too."""
        df = annual_frame(geo_ids=(1, 2, 3))
        with pytest.raises(ValidationError, match="expected 32, found 3"):
            write_parquet(df, base_path=base_path, validate=False, sort=True)

    def test_validate_false_skips_dimension_validation(self, base_path):
        """An out-of-range dimension value passes when validate=False."""
        df = annual_frame().with_columns(calendar_year=pl.lit(1800, dtype=pl.Int32))
        write_parquet(df, base_path=base_path, validate=False)
        assert list(base_path.rglob("*.parquet"))


class TestReadParquet:
    def test_round_trip(self, base_path):
        original = annual_frame()
        write_parquet(original, base_path=base_path)
        result = read_parquet(base_path=base_path, **DATASET_KWARGS).collect()
        assert result["calendar_year"].to_list() == original["calendar_year"].to_list()
        assert result["1"].to_list() == pytest.approx(original["1"].to_list())

    def test_dataset_dimensions_are_restored_as_columns(self, base_path):
        """variable/grid/time_unit live in the path, not the file, and are re-added."""
        write_parquet(annual_frame(), base_path=base_path)
        result = read_parquet(base_path=base_path, **DATASET_KWARGS).collect()
        assert result["variable"].unique().to_list() == ["ssf"]
        assert result["grid"].unique().to_list() == ["ecoregion_v001"]
        assert result["time_unit"].unique().to_list() == ["annual"]

    def test_column_order(self, base_path):
        write_parquet(annual_frame(), base_path=base_path)
        result = read_parquet(base_path=base_path, **DATASET_KWARGS).collect()
        assert result.columns[:3] == ["variable", "grid", "time_unit"]
        assert result.columns[-1] == "32"

    def test_scalar_filter(self, base_path):
        write_parquet(annual_frame(years=(2030, 2040, 2050)), base_path=base_path)
        result = read_parquet(base_path=base_path, calendar_year=2040, **DATASET_KWARGS).collect()
        assert result["calendar_year"].to_list() == [2040]

    def test_list_filter(self, base_path):
        write_parquet(annual_frame(years=(2030, 2040, 2050)), base_path=base_path)
        result = read_parquet(
            base_path=base_path, calendar_year=[2030, 2050], **DATASET_KWARGS
        ).collect()
        assert result["calendar_year"].to_list() == [2030, 2050]

    def test_include_file_paths(self, base_path):
        write_parquet(annual_frame(), base_path=base_path)
        result = read_parquet(
            base_path=base_path, include_file_paths="src", **DATASET_KWARGS
        ).collect()
        assert result["src"].str.ends_with(".parquet").all()

    def test_missing_dataset_raises_on_collect(self, base_path):
        """Reading a dataset that was never written fails at materialization time."""
        lf = read_parquet(base_path=base_path, **DATASET_KWARGS)
        with pytest.raises(FileNotFoundError):
            lf.collect()

    def test_daily_hive_partitions_are_typed(self, base_path):
        """Partition directories are parsed back using the declared dimension dtypes."""
        write_parquet(daily_frame(), base_path=base_path, validate=False, sort=False)
        result = read_parquet(
            base_path=base_path,
            **{k: DAILY_KWARGS[k] for k in ("variable", "grid", "time_unit")},
        ).collect()
        assert result["model_group_id"].dtype == pl.Int32
        assert result["model_group_id"].unique().to_list() == [500]

    def test_missing_dataset_dimension_raises(self, base_path):
        with pytest.raises(ValidationError, match="Required columns are missing"):
            read_parquet(base_path=base_path, variable="ssf")


DAILY_DATASET_KWARGS = {k: DAILY_KWARGS[k] for k in ("variable", "grid", "time_unit")}


class TestCompactParquetGlob:
    """The glob compact_parquet searches with must match the layout write_parquet produces.

    Only partition columns become directories, and the files below them are UUID-named,
    so the pattern is the partition directories followed by any Parquet file. Getting this
    wrong is silent: a non-matching glob just finds nothing to compact.
    """

    def _glob(self, **kwargs):
        """Rebuild the pattern the same way compact_parquet does."""
        from cpra.mp.data.utils import partition_columns

        input_df = pl.DataFrame(kwargs)
        cols = partition_columns(input_df, include_dataset=False)
        return "/".join([f"{col}={kwargs.get(col, '*')}" for col in cols] + ["*.parquet"])

    def test_annual_pattern_is_a_bare_file_glob(self):
        """For annual data nothing below the dataset directory is a directory at all."""
        assert self._glob(**DATASET_KWARGS) == "*.parquet"

    def test_daily_pattern_covers_the_partition_dimensions(self):
        """calendar_day is a nonpartition dimension, so it is never a directory."""
        assert self._glob(**DAILY_DATASET_KWARGS) == (
            "model_group_id=*/scenario_id=*/aggregate_type=*/*.parquet"
        )

    def test_written_annual_layout_is_flat(self, base_path):
        write_parquet(annual_frame(), base_path=base_path)
        path = next(base_path.rglob("*.parquet"))
        assert path.parent == base_path / "variable=ssf/grid=ecoregion_v001/time_unit=annual"

    def test_written_daily_layout_stops_at_the_partition_dimensions(self, base_path):
        write_parquet(daily_frame(), base_path=base_path, validate=False, sort=False)
        path = next(base_path.rglob("*.parquet"))
        assert path.parent.name == "aggregate_type=mean"

    @pytest.mark.parametrize("kwargs", [DATASET_KWARGS, DAILY_DATASET_KWARGS], ids=["annual", "daily"])
    def test_pattern_matches_what_was_written(self, base_path, kwargs):
        """The pattern and the layout are derived independently; check they agree."""
        from cpra.mp.data.utils import dataset_path

        if kwargs is DATASET_KWARGS:
            write_parquet(annual_frame(), base_path=base_path)
        else:
            write_parquet(daily_frame(), base_path=base_path, validate=False, sort=False)

        matched = list(dataset_path(base_path, **kwargs).glob(self._glob(**kwargs)))
        assert matched == list(base_path.rglob("*.parquet"))


class TestCompactParquet:
    def _write_two_annual_files(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        write_parquet(annual_frame(years=(2040,)), base_path=base_path)

    def _write_two_daily_files(self, base_path):
        for day in (datetime.date(2030, 1, 1), datetime.date(2030, 1, 2)):
            write_parquet(daily_frame(days=[day]), base_path=base_path, validate=False, sort=False)

    def test_empty_tree_is_a_clean_no_op(self, base_path):
        """No paths means the directory/file frame would be Null-typed, so bail out early."""
        result = compact_parquet(base_path=base_path, **DATASET_KWARGS)
        assert result == {"outputs": [], "sources_merged": 0}
        assert list(base_path.rglob("*.parquet")) == []

    def test_does_not_touch_the_configured_base_path(self, base_path, tmp_path, monkeypatch):
        """Regression: compact_parquet passed no base_path to its inner read_parquet and
        write_parquet calls, so both fell back to the module-level BASE_PATH. Compacting a
        copy would read from, and rewrite into, the production data directory."""
        production = tmp_path / "production"
        production.mkdir()
        monkeypatch.setattr(config, "BASE_PATH", production)

        self._write_two_annual_files(base_path)
        compact_parquet(base_path=base_path, **DATASET_KWARGS)

        assert list(production.rglob("*")) == []
        files = list(base_path.rglob("*.parquet"))
        assert len(files) == 1
        assert pl.read_parquet(files[0])["calendar_year"].to_list() == [2030, 2040]

    def test_merges_annual_files_within_a_dataset(self, base_path):
        self._write_two_annual_files(base_path)
        assert len(list(base_path.rglob("*.parquet"))) == 2

        compact_parquet(base_path=base_path, **DATASET_KWARGS)

        files = list(base_path.rglob("*.parquet"))
        assert len(files) == 1
        assert pl.read_parquet(files[0])["calendar_year"].to_list() == [2030, 2040]

    def test_merges_daily_files_within_a_partition(self, base_path):
        self._write_two_daily_files(base_path)
        assert len(list(base_path.rglob("*.parquet"))) == 2

        compact_parquet(base_path=base_path, sort=False, **DAILY_DATASET_KWARGS)

        assert len(list(base_path.rglob("*.parquet"))) == 1

    def test_values_survive_compaction(self, base_path):
        source = annual_frame(years=(2030, 2040))
        self._write_two_annual_files(base_path)

        compact_parquet(base_path=base_path, **DATASET_KWARGS)

        result = read_parquet(base_path=base_path, **DATASET_KWARGS).collect()
        for col in ("1", "16", "32"):
            assert result[col].to_list() == pytest.approx(source[col].to_list())

    def test_single_file_partitions_are_left_alone(self, base_path):
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        before = {p.name for p in base_path.rglob("*.parquet")}

        result = compact_parquet(base_path=base_path, **DATASET_KWARGS)

        assert result == {"outputs": [], "sources_merged": 0}
        assert {p.name for p in base_path.rglob("*.parquet")} == before

    def test_partitions_are_compacted_independently(self, base_path):
        """Two files under scenario 7 merge; the lone scenario 8 file is untouched."""
        self._write_two_daily_files(base_path)
        write_parquet(
            daily_frame(days=[datetime.date(2030, 1, 3)], scenario_id=8),
            base_path=base_path,
            validate=False,
            sort=False,
        )

        result = compact_parquet(base_path=base_path, sort=False, **DAILY_DATASET_KWARGS)

        assert result["sources_merged"] == 2
        by_dir = {}
        for path in base_path.rglob("*.parquet"):
            by_dir.setdefault(path.parent, []).append(path)
        assert sorted(len(v) for v in by_dir.values()) == [1, 1]

    def test_only_the_requested_dataset_is_compacted(self, base_path):
        """A second variable in the same tree must be left alone."""
        self._write_two_annual_files(base_path)
        self._write_two_daily_files(base_path)

        compact_parquet(base_path=base_path, **DATASET_KWARGS)

        daily_files = list((base_path / "variable=sal").rglob("*.parquet"))
        assert len(daily_files) == 2


class TestCompactParquetSummary:
    """compact_parquet reports what it did, mirroring raster.compact_cog's contract."""

    def test_returns_outputs_and_sources_merged(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        write_parquet(annual_frame(years=(2040,)), base_path=base_path)

        result = compact_parquet(base_path=base_path, **DATASET_KWARGS)

        assert set(result) == {"outputs", "sources_merged"}
        assert result["sources_merged"] == 2
        assert result["outputs"] == list(base_path.rglob("*.parquet"))

    def test_outputs_are_paths_that_exist(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        write_parquet(annual_frame(years=(2040,)), base_path=base_path)

        result = compact_parquet(base_path=base_path, **DATASET_KWARGS)

        assert all(isinstance(p, Path) and p.exists() for p in result["outputs"])

    def test_outputs_exclude_the_consumed_sources(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        sources = {p for p in base_path.rglob("*.parquet")}
        write_parquet(annual_frame(years=(2040,)), base_path=base_path)

        result = compact_parquet(base_path=base_path, **DATASET_KWARGS)

        assert not sources & set(result["outputs"])

    def test_one_output_per_compacted_partition(self, base_path):
        self._write_daily(base_path, datetime.date(2030, 1, 1), 7)
        self._write_daily(base_path, datetime.date(2030, 1, 2), 7)
        self._write_daily(base_path, datetime.date(2030, 1, 1), 8)
        self._write_daily(base_path, datetime.date(2030, 1, 2), 8)

        result = compact_parquet(base_path=base_path, sort=False, **DAILY_DATASET_KWARGS)

        assert len(result["outputs"]) == 2
        assert result["sources_merged"] == 4

    @staticmethod
    def _write_daily(base_path, day, scenario_id):
        write_parquet(
            daily_frame(days=[day], scenario_id=scenario_id),
            base_path=base_path,
            validate=False,
            sort=False,
        )

