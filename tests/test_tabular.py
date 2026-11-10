"""Tests for reading, writing, and compacting long-format Parquet datasets.

``write_parquet`` accepts wide input (one column per grid cell, as produced by the model
and built by ``tests/factories.py``) and unpivots it to long (a ``grid_cell`` column plus
a ``value`` column, one row per cell x time step) before writing. ``read_parquet`` always
returns long-format data."""

import datetime
import tempfile
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from cpra.mp.data import config, merge, staging, tabular
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
        assert "variable=ssf/time_unit=annual/grid=ecoregion_v001" in str(files[0])

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

    def test_long_format_input_is_written_directly(self, base_path):
        """A frame already in long format (grid_cell/value columns, not one column per
        grid cell) has no numeric-named columns, so the unpivot step is a no-op and the
        frame is written straight through."""
        long_df = pl.DataFrame(
            {
                **{k: [v] * 3 for k, v in ANNUAL_KWARGS.items() if k != "grid"},
                "grid": ["ecoregion_v001"] * 3,
                "calendar_year": [2030] * 3,
                "grid_cell": [1, 2, 3],
                "value": [1.0, 2.0, 3.0],
            }
        ).with_columns(
            pl.col("model_group_id").cast(pl.Int32),
            pl.col("scenario_id").cast(pl.Int32),
            pl.col("percentile").cast(pl.Int8),
            pl.col("calendar_year").cast(pl.Int32),
            pl.col("grid_cell").cast(pl.Int32),
            pl.col("value").cast(pl.Float32),
        )
        write_parquet(long_df, base_path=base_path, validate=False, sort=False)
        result = pl.read_parquet(next(base_path.rglob("*.parquet")))
        assert result["grid_cell"].to_list() == [1, 2, 3]
        assert result["value"].to_list() == pytest.approx([1.0, 2.0, 3.0])

    def test_accepts_a_dict_of_series(self, base_path):
        """A plain dict is accepted, but only Series preserve the required dtypes."""
        write_parquet(annual_frame().to_dict(as_series=True), base_path=base_path)
        assert list(base_path.rglob("*.parquet"))

    def test_dict_of_python_lists_loses_dtypes(self, base_path):
        """Round-tripping through plain lists widens Int32 to Int64 and fails validation."""
        with pytest.raises(ValidationError, match="expected Int32"):
            write_parquet(annual_frame().to_dict(as_series=False), base_path=base_path)

    def test_sorts_grid_cells_numerically(self, base_path):
        """Row order must be grid_cell 1, 2, ... 10, not the lexical 1, 10, 2 that the
        wide column names (unpivoted into grid_cell) would sort to as strings."""
        write_parquet(annual_frame(), base_path=base_path)
        result = pl.read_parquet(next(base_path.rglob("*.parquet")))
        grid_cells = result.filter(pl.col("calendar_year") == 2030)["grid_cell"].to_list()
        assert grid_cells == list(ECOREGION_IDS)

    def test_sorts_rows_by_required_columns(self, base_path):
        write_parquet(annual_frame(years=(2050, 2030, 2040)), base_path=base_path)
        result = pl.read_parquet(next(base_path.rglob("*.parquet")))
        years = result.filter(pl.col("grid_cell") == 1)["calendar_year"].to_list()
        assert years == [2030, 2040, 2050]

    def test_sort_false_preserves_input_order(self, base_path):
        write_parquet(annual_frame(years=(2050, 2030)), base_path=base_path, sort=False)
        result = pl.read_parquet(next(base_path.rglob("*.parquet")))
        years = result.filter(pl.col("grid_cell") == 1)["calendar_year"].to_list()
        assert years == [2050, 2030]


class TestWriteParquetValidation:
    def test_missing_required_column_raises(self, base_path):
        df = annual_frame().drop("percentile")
        with pytest.raises(ValidationError, match="percentile"):
            write_parquet(df, base_path=base_path)

    def test_sort_reports_a_column_required_by_the_data_itself(self, base_path):
        """With validation off, the sort branch still needs a column that the frame's
        own dimension values imply: annual data must carry calendar_year. It is checked
        eagerly, so the caller gets this package's own error rather than a Polars
        ColumnNotFoundError raised from inside a sink producer thread."""
        df = annual_frame().drop("calendar_year")
        with pytest.raises(ValidationError, match="calendar_year"):
            write_parquet(df, base_path=base_path, validate=False, sort=True)

    def test_unexpected_column_raises(self, base_path):
        """An unrecognized column is a typo, not data. Rejecting it up front matters
        because the alternative is silent and depends on an unrelated argument: the
        sort's `.select()` drops the column when sort=True, and it is written into the
        output file when sort=False.
        """
        df = annual_frame().with_columns(surprise=pl.lit("boom"))
        with pytest.raises(ValidationError, match="Invalid columns: surprise"):
            write_parquet(df, base_path=base_path)

    @pytest.mark.parametrize("sort", [True, False])
    def test_unexpected_column_never_reaches_the_output(self, base_path, sort):
        """The guard runs before either sort branch, so neither can leak the column."""
        df = annual_frame().with_columns(surprise=pl.lit("boom"))
        with pytest.raises(ValidationError, match="surprise"):
            write_parquet(df, base_path=base_path, sort=sort)
        assert list(base_path.rglob("*.parquet")) == []

    def test_grid_cell_id_above_max_raises(self, base_path):
        df = annual_frame(geo_ids=tuple(range(2, 34)))
        with pytest.raises(ValidationError, match="Grid cell ID above maximum"):
            write_parquet(df, base_path=base_path)

    def test_grid_cell_id_below_min_raises(self, base_path):
        df = annual_frame(geo_ids=tuple(range(32)))
        with pytest.raises(ValidationError, match="Grid cell ID below minimum"):
            write_parquet(df, base_path=base_path)

    def test_grid_cell_id_count_incomplete_raises(self, base_path):
        """Partial writes are not allowed: ecoregion_v001 has exactly 32 grid cells, so a
        frame missing one -- even though every ID present is in range -- must be rejected
        rather than silently written as an incomplete dataset."""
        df = annual_frame(geo_ids=tuple(range(1, 32)))
        with pytest.raises(ValidationError, match="Grid cell IDs incomplete"):
            write_parquet(df, base_path=base_path)

    def test_grid_cell_id_inconsistent_row_counts_raises(self, base_path):
        """Every grid cell must appear the same number of times (once per implied time
        step); a cell with extra or missing time steps relative to the others is also a
        partial write, even though every ID from 1-32 is present."""
        rows = [(gc, 2030) for gc in ECOREGION_IDS] + [(1, 2040)]
        df = pl.DataFrame(
            {
                **{k: [v] * len(rows) for k, v in ANNUAL_KWARGS.items()},
                "grid_cell": [r[0] for r in rows],
                "calendar_year": [r[1] for r in rows],
                "value": [1.0] * len(rows),
            }
        ).with_columns(
            pl.col("model_group_id").cast(pl.Int32),
            pl.col("scenario_id").cast(pl.Int32),
            pl.col("percentile").cast(pl.Int8),
            pl.col("calendar_year").cast(pl.Int32),
            pl.col("grid_cell").cast(pl.Int32),
            pl.col("value").cast(pl.Float32),
        )
        with pytest.raises(ValidationError, match="inconsistent row counts"):
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
        cell = result.filter(pl.col("grid_cell") == 1).sort("calendar_year")
        assert cell["calendar_year"].to_list() == original["calendar_year"].to_list()
        assert cell["value"].to_list() == pytest.approx(original["1"].to_list())

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
        assert result.columns[:3] == ["variable", "time_unit", "grid"]
        assert result.columns[-2:] == ["grid_cell", "value"]

    def test_scalar_filter(self, base_path):
        write_parquet(annual_frame(years=(2030, 2040, 2050)), base_path=base_path)
        result = read_parquet(base_path=base_path, calendar_year=2040, **DATASET_KWARGS).collect()
        assert result["calendar_year"].unique().to_list() == [2040]
        assert result.height == len(ECOREGION_IDS)

    def test_list_filter(self, base_path):
        write_parquet(annual_frame(years=(2030, 2040, 2050)), base_path=base_path)
        result = read_parquet(base_path=base_path, calendar_year=[2030, 2050], **DATASET_KWARGS).collect()
        assert sorted(result["calendar_year"].unique().to_list()) == [2030, 2050]
        assert result.height == len(ECOREGION_IDS) * 2

    def test_include_file_paths(self, base_path):
        write_parquet(annual_frame(), base_path=base_path)
        result = read_parquet(base_path=base_path, include_file_paths="src", **DATASET_KWARGS).collect()
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
        assert self._glob(**DAILY_DATASET_KWARGS) == ("model_group_id=*/scenario_id=*/aggregate_type=*/*.parquet")

    def test_written_annual_layout_is_flat(self, base_path):
        write_parquet(annual_frame(), base_path=base_path)
        path = next(base_path.rglob("*.parquet"))
        assert path.parent == base_path / "variable=ssf/time_unit=annual/grid=ecoregion_v001"

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
        result = pl.read_parquet(files[0])
        years = result.filter(pl.col("grid_cell") == 1)["calendar_year"].to_list()
        assert years == [2030, 2040]

    def test_merges_annual_files_within_a_dataset(self, base_path):
        self._write_two_annual_files(base_path)
        assert len(list(base_path.rglob("*.parquet"))) == 2

        compact_parquet(base_path=base_path, **DATASET_KWARGS)

        files = list(base_path.rglob("*.parquet"))
        assert len(files) == 1
        result = pl.read_parquet(files[0])
        years = result.filter(pl.col("grid_cell") == 1)["calendar_year"].to_list()
        assert years == [2030, 2040]

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
        for geo_id in (1, 16, 32):
            cell = result.filter(pl.col("grid_cell") == geo_id).sort("calendar_year")
            assert cell["value"].to_list() == pytest.approx(source[str(geo_id)].to_list())

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

    @pytest.mark.parametrize("extra", [{"model_group_id": 500}, {"scenario_id": 7}, {"calendar_year": 2030}])
    def test_column_filter_on_annual_data_is_refused(self, base_path, extra):
        """Regression: annual files hold every run, so compacting with model_group_id=500
        rewrote only 500's rows and then deleted the files holding everyone else's."""
        self._write_two_annual_files(base_path)
        write_parquet(annual_frame(years=(2030,), model_group_id=501), base_path=base_path)
        before = read_parquet(base_path=base_path, **DATASET_KWARGS).collect()

        with pytest.raises(ValidationError, match="can only narrow by partition directories"):
            compact_parquet(base_path=base_path, **DATASET_KWARGS, **extra)

        assert len(list(base_path.rglob("*.parquet"))) == 3
        after = read_parquet(base_path=base_path, **DATASET_KWARGS).collect()
        assert after.sort(after.columns).equals(before.sort(before.columns))

    def test_partition_filter_on_daily_data_is_allowed(self, base_path):
        self._write_two_daily_files(base_path)
        write_parquet(
            daily_frame(days=[datetime.date(2030, 1, 3)], scenario_id=8),
            base_path=base_path,
            validate=False,
            sort=False,
        )
        write_parquet(
            daily_frame(days=[datetime.date(2030, 1, 4)], scenario_id=8),
            base_path=base_path,
            validate=False,
            sort=False,
        )

        result = compact_parquet(base_path=base_path, sort=False, scenario_id=7, **DAILY_DATASET_KWARGS)

        assert result["sources_merged"] == 2
        assert len(list(base_path.rglob("scenario_id=8/**/*.parquet"))) == 2

    def test_non_partition_filter_on_daily_data_is_refused(self, base_path):
        self._write_two_daily_files(base_path)

        with pytest.raises(ValidationError, match="calendar_day"):
            compact_parquet(
                base_path=base_path, sort=False, calendar_day=datetime.date(2030, 1, 1), **DAILY_DATASET_KWARGS
            )

        assert len(list(base_path.rglob("*.parquet"))) == 2


def _forbid_windowed_merge(monkeypatch):
    """Runs never interleave with each other, so these cases must not need the slow path."""

    def forbidden(*args):
        raise AssertionError("the windowed merge was used")

    monkeypatch.setattr(merge, "_window_merge", forbidden)


class TestCompactParquetStreaming:
    """compact_parquet merges the already-sorted files rather than re-sorting them, and must
    end up with exactly the file a single write_parquet of all the data would have made."""

    @staticmethod
    def _single_write(tmp_path, frames):
        reference = tmp_path / "reference"
        reference.mkdir()
        write_parquet(pl.concat(frames), base_path=reference)
        (file,) = reference.rglob("*.parquet")
        return pl.read_parquet(file)

    @staticmethod
    def _compacted(base_path):
        (file,) = base_path.rglob("*.parquet")
        return pl.read_parquet(file)

    def test_one_year_at_a_time_matches_a_single_write(self, base_path, tmp_path):
        frames = [annual_frame(years=(year,)) for year in (2033, 2030, 2032, 2031)]
        for frame in frames:
            write_parquet(frame, base_path=base_path)

        compact_parquet(base_path=base_path, **DATASET_KWARGS)

        assert self._compacted(base_path).equals(self._single_write(tmp_path, frames))

    def test_a_new_run_inside_a_compacted_file(self, base_path, tmp_path, monkeypatch):
        _forbid_windowed_merge(monkeypatch)
        frames = [annual_frame(years=(year,), model_group_id=mg) for mg in (400, 600) for year in (2030, 2031)]
        for frame in frames:
            write_parquet(frame, base_path=base_path)
        compact_parquet(base_path=base_path, **DATASET_KWARGS)
        new = [annual_frame(years=(year,), model_group_id=500) for year in (2030, 2031)]
        for frame in new:
            write_parquet(frame, base_path=base_path)

        compact_parquet(base_path=base_path, **DATASET_KWARGS)

        assert self._compacted(base_path).equals(self._single_write(tmp_path, frames + new))

    def test_years_added_to_a_compacted_run(self, base_path, tmp_path, monkeypatch):
        _forbid_windowed_merge(monkeypatch)
        frames = [annual_frame(years=(2030,), model_group_id=mg) for mg in (500, 600)]
        for frame in frames:
            write_parquet(frame, base_path=base_path)
        compact_parquet(base_path=base_path, **DATASET_KWARGS)
        added = [annual_frame(years=(year,), model_group_id=500) for year in (2031, 2032)]
        for frame in added:
            write_parquet(frame, base_path=base_path)

        compact_parquet(base_path=base_path, **DATASET_KWARGS)

        assert self._compacted(base_path).equals(self._single_write(tmp_path, frames + added))

    def test_file_cap_leaves_no_temporary_files(self, base_path, tmp_path):
        frames = [annual_frame(years=(year,)) for year in range(2030, 2037)]
        for frame in frames:
            write_parquet(frame, base_path=base_path)

        compact_parquet(base_path=base_path, max_open_files=2, **DATASET_KWARGS)

        assert self._compacted(base_path).equals(self._single_write(tmp_path, frames))
        assert [p for p in base_path.rglob("*") if p.name.startswith(".")] == []

    def test_leftovers_of_an_interrupted_compaction_are_removed(self, base_path, tmp_path):
        frames = [annual_frame(years=(year,)) for year in (2030, 2031)]
        for frame in frames:
            write_parquet(frame, base_path=base_path)
        (base_path / ".compact-dead").mkdir()
        (base_path / ".compact-dead" / "merged.parquet").write_bytes(b"truncated")
        keep = base_path / ".compact-not-a-dir"
        keep.write_bytes(b"")

        compact_parquet(base_path=base_path, **DATASET_KWARGS)

        assert self._compacted(base_path).equals(self._single_write(tmp_path, frames))
        assert not (base_path / ".compact-dead").exists()
        assert keep.exists()

    def test_leftovers_are_removed_from_temp_dir(self, base_path, tmp_path):
        temp_dir = tmp_path / "scratch"
        (temp_dir / ".compact-dead").mkdir(parents=True)
        (temp_dir / "other").mkdir()
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)

        compact_parquet(base_path=base_path, temp_dir=temp_dir, **DATASET_KWARGS)

        assert [p.name for p in temp_dir.iterdir()] == ["other"]

    def test_dataset_stays_readable_while_compacting(self, base_path, monkeypatch):
        """Polars takes every file under a scanned directory, so nothing unfinished may sit
        in the dataset directory -- not even under a hidden name."""
        for year in (2030, 2031):
            write_parquet(annual_frame(years=(year,)), base_path=base_path)
        expected = read_parquet(base_path=base_path, **DATASET_KWARGS).collect()
        original = tabular.merge_sorted_files
        during = []

        def checking(paths, out_path, *args, **kwargs):
            stats = original(paths, out_path, *args, **kwargs)
            during.append(read_parquet(base_path=base_path, **DATASET_KWARGS).collect())
            return stats

        monkeypatch.setattr(tabular, "merge_sorted_files", checking)

        compact_parquet(base_path=base_path, **DATASET_KWARGS)

        (frame,) = during
        assert frame.equals(expected)

    def test_a_killed_compaction_leaves_the_sources_alone(self, base_path, monkeypatch):
        for year in (2030, 2031):
            write_parquet(annual_frame(years=(year,)), base_path=base_path)
        before = sorted(base_path.rglob("*"))

        def killed(*args, **kwargs):
            raise MemoryError

        monkeypatch.setattr(staging, "publish", killed)
        with pytest.raises(MemoryError):
            compact_parquet(base_path=base_path, **DATASET_KWARGS)

        assert sorted(base_path.rglob("*")) == before

    def test_temp_dir_is_used_and_cleaned(self, base_path, tmp_path, monkeypatch):
        temp_dir = tmp_path / "scratch"
        temp_dir.mkdir()
        seen = []
        original = tempfile.TemporaryDirectory

        def recording(*args, **kwargs):
            seen.append(kwargs.get("dir"))
            return original(*args, **kwargs)

        monkeypatch.setattr(tempfile, "TemporaryDirectory", recording)
        for year in range(2030, 2035):
            write_parquet(annual_frame(years=(year,)), base_path=base_path)

        compact_parquet(base_path=base_path, max_open_files=2, temp_dir=temp_dir, **DATASET_KWARGS)

        assert temp_dir in seen
        assert list(temp_dir.iterdir()) == []
        assert [p for p in base_path.iterdir() if p.name.startswith(".")] == []

    def test_compacted_file_is_encoded_like_a_written_one(self, base_path):
        for year in (2030, 2031):
            write_parquet(annual_frame(years=(year,)), base_path=base_path)

        compact_parquet(base_path=base_path, **DATASET_KWARGS)

        (file,) = base_path.rglob("*.parquet")
        metadata = pq.read_metadata(file)
        columns = {metadata.schema.column(i).name: metadata.row_group(0).column(i) for i in range(metadata.num_columns)}
        assert "DELTA_BINARY_PACKED" in columns["grid_cell"].encodings
        assert "DELTA_BINARY_PACKED" in columns["calendar_year"].encodings
        assert columns["value"].compression == "ZSTD"

    def test_failure_leaves_the_sources_and_no_partial_file(self, base_path, monkeypatch):
        for year in (2030, 2031):
            write_parquet(annual_frame(years=(year,)), base_path=base_path)
        before = {p: p.read_bytes() for p in base_path.rglob("*") if p.is_file()}

        def failing_write(self, table):
            raise OSError("disk full")

        monkeypatch.setattr(merge._Writer, "write", failing_write)

        with pytest.raises(OSError, match="disk full"):
            compact_parquet(base_path=base_path, **DATASET_KWARGS)

        after = {p: p.read_bytes() for p in base_path.rglob("*") if p.is_file()}
        assert after == before
        assert [p for p in base_path.rglob("*") if p.name.startswith(".")] == []

    def test_catalog_rows_cover_the_merged_file(self, base_path, catalog_path):
        for year in (2030, 2031, 2032):
            write_parquet(annual_frame(years=(year,)), base_path=base_path, catalog_path=catalog_path)

        compact_parquet(base_path=base_path, catalog_path=catalog_path, **DATASET_KWARGS)

        rows = pl.read_parquet(catalog_path / "**" / "*.parquet", hive_partitioning=True)
        assert rows.height == 1
        assert rows.row(0, named=True)["calendar_year_min"] == 2030
        assert rows.row(0, named=True)["calendar_year_max"] == 2032


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
