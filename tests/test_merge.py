"""Tests for merging already-sorted Parquet files (``cpra.mp.data.merge``).

The files here are written directly with Polars, with tiny row groups and batches, so a
handful of rows exercises the chunking that real 500k-row row groups would."""

import datetime

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from cpra.mp.data import merge
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.merge import merge_sorted_files

KEY = ["model_group_id", "scenario_id", "grid_cell", "calendar_year"]
OPTIONS = {"batch_size": 7, "row_group_size": 10, "write_options": {}}


def run_frame(model_group_id, years, cells=range(1, 21), scenario_id=1):
    """One run's rows for ``years``, sorted by KEY, with a value unique to each row."""
    rows = [(model_group_id, scenario_id, cell, year) for cell in cells for year in years]
    frame = pl.DataFrame(rows, schema=dict(zip(KEY, [pl.Int32, pl.Int32, pl.Int32, pl.Int32])), orient="row")
    return frame.with_columns(
        value=(pl.col("model_group_id") * 1e6 + pl.col("grid_cell") * 1e4 + pl.col("calendar_year")).cast(pl.Float32)
    )


def write(path, frame, row_group_size=5):
    frame.write_parquet(path, row_group_size=row_group_size)
    return path


def full_sort(paths):
    return pl.concat([pl.read_parquet(path) for path in paths]).sort(KEY, maintain_order=True)


def merged(tmp_path, paths, **options):
    out = tmp_path / "out.parquet"
    stats = merge_sorted_files(paths, out, KEY, temp_dir=tmp_path, **{**OPTIONS, **options})
    return pl.read_parquet(out), stats


def one_year_files(tmp_path, years, **run):
    return [write(tmp_path / f"{year}.parquet", run_frame(500, [year], **run)) for year in years]


class TestInterleave:
    def test_one_year_files_match_a_full_sort(self, tmp_path):
        # Written out of calendar order, as the directory listing might return them.
        paths = one_year_files(tmp_path, [2033, 2030, 2032, 2031])

        result, stats = merged(tmp_path, paths)

        assert result.equals(full_sort(paths))
        assert stats == {"rows": 80, "copied": 0, "interleaved": 1, "merged": 0, "rounds": 0}

    def test_files_of_several_years_each(self, tmp_path):
        paths = [
            write(tmp_path / "a.parquet", run_frame(500, [2030, 2031, 2032])),
            write(tmp_path / "b.parquet", run_frame(500, [2033])),
            write(tmp_path / "c.parquet", run_frame(500, [2034, 2035])),
        ]

        result, stats = merged(tmp_path, paths)

        assert result.equals(full_sort(paths))
        assert stats["interleaved"] == 1

    def test_daily_files(self, tmp_path):
        key = ["grid_cell", "calendar_day"]
        paths = []
        for month in (3, 1, 2):
            days = [datetime.date(2030, month, d) for d in (1, 2)]
            frame = pl.DataFrame(
                {"grid_cell": np.repeat(np.arange(1, 11, dtype=np.int32), 2), "calendar_day": days * 10}
            ).with_columns(value=pl.int_range(pl.len()).cast(pl.Float32))
            paths.append(write(tmp_path / f"{month}.parquet", frame))
        out = tmp_path / "out.parquet"

        stats = merge_sorted_files(paths, out, key, temp_dir=tmp_path, **OPTIONS)

        expected = pl.concat([pl.read_parquet(p) for p in paths]).sort(key, maintain_order=True)
        assert pl.read_parquet(out).equals(expected)
        assert stats["interleaved"] == 1

    def test_output_row_groups_have_the_requested_size(self, tmp_path):
        paths = one_year_files(tmp_path, [2030, 2031, 2032])
        out = tmp_path / "out.parquet"

        merge_sorted_files(paths, out, KEY, temp_dir=tmp_path, **OPTIONS)

        metadata = pq.read_metadata(out)
        sizes = [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]
        assert sizes == [10] * 6


RUN_COLS = ["model_group_id", "scenario_id"]


class TestRuns:
    """With run_cols, each file is split into one piece per run, and only a run's own
    pieces are ever merged with each other."""

    @pytest.mark.parametrize("statistics", [True, False])
    @pytest.mark.parametrize("row_group_size", [5, 7, 1000])
    def test_a_run_sorting_inside_a_compacted_file_is_copied(self, tmp_path, statistics, row_group_size):
        """Row groups of 7 rows straddle runs, and of 1000 hold the whole file, so the run
        boundaries are found by decoding; without statistics, always so."""
        compacted = pl.concat([run_frame(1, [2030, 2031]), run_frame(3, [2030, 2031])])
        compacted_path = tmp_path / "compacted.parquet"
        compacted.write_parquet(compacted_path, row_group_size=row_group_size, statistics=statistics)
        paths = [compacted_path, write(tmp_path / "new.parquet", run_frame(2, [2030, 2031]))]

        result, stats = merged(tmp_path, paths, run_cols=RUN_COLS)

        assert result.equals(full_sort(paths))
        assert stats == {"rows": 120, "copied": 3, "interleaved": 0, "merged": 0, "rounds": 0}

    def test_without_run_cols_the_same_files_need_the_windowed_merge(self, tmp_path):
        compacted = pl.concat([run_frame(1, [2030, 2031]), run_frame(3, [2030, 2031])])
        paths = [
            write(tmp_path / "compacted.parquet", compacted),
            write(tmp_path / "new.parquet", run_frame(2, [2030, 2031])),
        ]

        result, stats = merged(tmp_path, paths)

        assert result.equals(full_sort(paths))
        assert stats["merged"] == 1

    def test_years_added_to_a_compacted_run_are_interleaved(self, tmp_path):
        compacted = pl.concat([run_frame(1, [2030, 2031]), run_frame(3, [2030, 2031])])
        paths = [
            write(tmp_path / "compacted.parquet", compacted, row_group_size=7),
            write(tmp_path / "2032.parquet", run_frame(1, [2032])),
            write(tmp_path / "2033.parquet", run_frame(1, [2033])),
        ]

        result, stats = merged(tmp_path, paths, run_cols=RUN_COLS)

        assert result.equals(full_sort(paths))
        assert stats["interleaved"] == 1
        assert stats["copied"] == 1
        assert stats["merged"] == 0

    def test_new_runs_written_a_year_at_a_time(self, tmp_path):
        compacted = pl.concat([run_frame(1, [2030, 2031]), run_frame(3, [2030, 2031])])
        paths = [
            write(tmp_path / "compacted.parquet", compacted),
            *[write(tmp_path / f"2-{y}.parquet", run_frame(2, [y])) for y in (2031, 2030)],
            *[write(tmp_path / f"4-{y}.parquet", run_frame(4, [y])) for y in (2030, 2031)],
        ]

        result, stats = merged(tmp_path, paths, run_cols=RUN_COLS, max_open_files=2)

        assert result.equals(full_sort(paths))
        assert stats == {"rows": 160, "copied": 2, "interleaved": 2, "merged": 0, "rounds": 0}

    def test_runs_out_of_order_raise(self, tmp_path):
        paths = [
            write(tmp_path / "a.parquet", pl.concat([run_frame(3, [2030]), run_frame(1, [2030])])),
            write(tmp_path / "b.parquet", run_frame(2, [2030])),
        ]

        with pytest.raises(ValidationError, match="isn't sorted by model_group_id"):
            merged(tmp_path, paths, run_cols=RUN_COLS)

        assert not (tmp_path / "out.parquet").exists()

    def test_run_cols_must_lead_the_key(self, tmp_path):
        paths = one_year_files(tmp_path, [2030, 2031])

        with pytest.raises(ValueError, match="must lead"):
            merged(tmp_path, paths, run_cols=["scenario_id"])


class TestFallback:
    def test_layout_break_part_way_through(self, tmp_path):
        """One file swaps a late grid cell for another -- same row count, so the
        interleave writes the chunks before it, and the windowed merge finishes the rest."""
        paths = [
            write(tmp_path / "a.parquet", run_frame(500, [2030])),
            write(tmp_path / "b.parquet", run_frame(500, [2031], cells=[*range(1, 18), 19, 20, 21])),
        ]

        # The interleave really does get part-way before it gives up.
        sources = [merge._Source(merge._Piece.whole(path), pq.read_schema(path), 7) for path in paths]
        written = 0
        with pytest.raises(merge._LayoutError):
            for chunk in merge._interleave(sources, KEY[:-1], KEY[-1], 10):
                written += chunk.num_rows
        assert written > 0
        for source in sources:
            source.close()

        result, stats = merged(tmp_path, paths)

        assert result.equals(full_sort(paths))
        assert stats["merged"] == 1

    def test_duplicate_years_keep_every_row(self, tmp_path):
        paths = [
            write(tmp_path / "a.parquet", run_frame(500, [2030, 2031])),
            write(tmp_path / "b.parquet", run_frame(500, [2031])),
        ]

        result, stats = merged(tmp_path, paths)

        assert result.height == 60
        assert result.equals(full_sort(paths))
        assert stats["merged"] == 1

    def test_differing_block_lengths_across_groups(self, tmp_path):
        """A file whose cells don't all hold the same years can't be interleaved."""
        ragged = pl.concat([run_frame(500, [2030], cells=[1]), run_frame(500, [2030, 2031], cells=range(2, 21))])
        paths = [
            write(tmp_path / "a.parquet", ragged),
            write(tmp_path / "b.parquet", run_frame(500, [2032])),
        ]

        result, stats = merged(tmp_path, paths)

        assert result.equals(full_sort(paths))
        assert stats["merged"] == 1


class TestClusters:
    def test_non_overlapping_files_are_copied(self, tmp_path):
        paths = [
            write(tmp_path / "b.parquet", run_frame(2, [2030, 2031])),
            write(tmp_path / "a.parquet", run_frame(1, [2030, 2031])),
        ]

        result, stats = merged(tmp_path, paths)

        assert result.equals(full_sort(paths))
        assert stats == {"rows": 80, "copied": 2, "interleaved": 0, "merged": 0, "rounds": 0}

    def test_overlapping_and_separate_files_together(self, tmp_path):
        paths = [
            *one_year_files(tmp_path, [2030, 2031]),
            write(tmp_path / "other.parquet", run_frame(600, [2030])),
        ]

        result, stats = merged(tmp_path, paths)

        assert result.equals(full_sort(paths))
        assert stats["copied"] == 1
        assert stats["interleaved"] == 1

    def test_empty_files_are_skipped(self, tmp_path):
        paths = [
            *one_year_files(tmp_path, [2030, 2031]),
            write(tmp_path / "empty.parquet", run_frame(500, [2030]).clear()),
        ]

        result, _ = merged(tmp_path, paths)

        assert result.equals(full_sort(paths))


class TestFileCap:
    def test_many_files_are_merged_in_rounds(self, tmp_path):
        paths = one_year_files(tmp_path, range(2030, 2040))

        result, stats = merged(tmp_path, paths, max_open_files=3)

        assert result.equals(full_sort(paths))
        assert stats["rounds"] == 2
        assert stats["merged"] == 0

    def test_temporary_files_are_removed(self, tmp_path):
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        paths = one_year_files(tmp_path, range(2030, 2040))

        merge_sorted_files(paths, tmp_path / "out.parquet", KEY, temp_dir=temp_dir, max_open_files=2, **OPTIONS)

        assert list(temp_dir.iterdir()) == []

    def test_temporary_files_are_removed_after_a_failure(self, tmp_path, monkeypatch):
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        paths = one_year_files(tmp_path, range(2030, 2040))
        calls = []

        def failing_merge(*args):
            if calls:
                raise OSError("disk full")
            calls.append(args)
            yield from original(*args)

        original = merge._merge
        monkeypatch.setattr(merge, "_merge", failing_merge)

        with pytest.raises(OSError, match="disk full"):
            merge_sorted_files(paths, tmp_path / "out.parquet", KEY, temp_dir=temp_dir, max_open_files=2, **OPTIONS)

        assert list(temp_dir.iterdir()) == []
        assert not (tmp_path / "out.parquet").exists()

    def test_cap_below_two_is_rejected(self, tmp_path):
        paths = one_year_files(tmp_path, [2030, 2031])

        with pytest.raises(ValueError, match="at least 2"):
            merged(tmp_path, paths, max_open_files=1)


class TestSortFalse:
    def test_files_are_concatenated_in_the_given_order(self, tmp_path):
        paths = one_year_files(tmp_path, [2031, 2030])

        result, stats = merged(tmp_path, paths, sort=False)

        assert result.equals(pl.concat([pl.read_parquet(path) for path in paths]))
        assert stats["copied"] == 2


class TestErrors:
    def test_differing_columns_raise(self, tmp_path):
        paths = [
            write(tmp_path / "a.parquet", run_frame(500, [2030])),
            write(tmp_path / "b.parquet", run_frame(500, [2031]).with_columns(extra=pl.lit(1))),
        ]

        with pytest.raises(ValidationError, match="columns"):
            merged(tmp_path, paths)

        assert not (tmp_path / "out.parquet").exists()

    def test_missing_key_column_raises(self, tmp_path):
        paths = [write(tmp_path / f"{y}.parquet", run_frame(500, [y]).drop("scenario_id")) for y in (2030, 2031)]

        with pytest.raises(ValidationError, match="scenario_id"):
            merged(tmp_path, paths)

    def test_differing_column_types_are_cast_to_the_first_files(self, tmp_path):
        paths = [
            write(tmp_path / "a.parquet", run_frame(500, [2030])),
            write(tmp_path / "b.parquet", run_frame(500, [2031]).with_columns(pl.col("value").cast(pl.Float64))),
        ]

        result, _ = merged(tmp_path, paths)

        assert result.schema["value"] == pl.Float32
        assert result.height == 40

    def test_failure_mid_write_leaves_no_output(self, tmp_path, monkeypatch):
        paths = one_year_files(tmp_path, [2030, 2031])
        original = merge._Writer.write

        def failing_write(self, table):
            if self.rows:
                raise OSError("disk full")
            original(self, table)

        monkeypatch.setattr(merge._Writer, "write", failing_write)

        with pytest.raises(OSError, match="disk full"):
            merged(tmp_path, paths)

        assert not (tmp_path / "out.parquet").exists()
        assert all(path.exists() for path in paths)

    def test_row_count_mismatch_raises(self, tmp_path, monkeypatch):
        paths = one_year_files(tmp_path, [2030, 2031])
        original = merge._Writer.write
        monkeypatch.setattr(merge._Writer, "write", lambda self, table: original(self, table.slice(1)))

        with pytest.raises(ValidationError, match="expected 40"):
            merged(tmp_path, paths)

        assert not (tmp_path / "out.parquet").exists()
