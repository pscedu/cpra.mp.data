"""Tests for `python -m cpra.mp.data.cli catalog` -- the `build`/`compact`/`query`
commands in cpra/mp/data/cli/catalog.py.

Importing this module attaches the `catalog` command group to the shared `cli` Click
group as a side effect (see cli/catalog.py's `@cli.group()`), the same as it would in
production when `cli/__main__.py` imports it. See tests/test_cli.py for why that makes
`cli.commands == {}` an order-dependent assertion elsewhere in the suite.

All data lives under tmp_path -- never the real data/ directory (see repo CLAUDE.md).
`build`'s `--base-path`/`--out` default to relative paths ("data-long"/"catalog"), so
every invocation here passes explicit tmp_path-based values except the one test that
specifically exercises the defaults, which uses `CliRunner.isolated_filesystem()` to
sandbox the current working directory instead.
"""

from pathlib import Path

import numpy as np
import polars as pl
from click.testing import CliRunner

import cpra.mp.data.cli.catalog  # noqa: F401  (registers `catalog` on `cli`)
from cpra.mp.data.cli import cli
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.raster import write_cog
from cpra.mp.data.tabular import write_parquet

TABULAR_KWARGS = dict(
    variable="sal",
    grid="hydro_compartment_v001",
    time_unit="annual",
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


def _tabular_frame(years, grid_cells=(1, 2)):
    rows = [(gc, yr) for gc in grid_cells for yr in years]
    return pl.DataFrame(
        {
            "grid_cell": [r[0] for r in rows],
            "calendar_year": [r[1] for r in rows],
            "value": [1.0] * len(rows),
        },
        schema={"grid_cell": pl.Int32, "calendar_year": pl.Int32, "value": pl.Float32},
    )


def _raster_array():
    grid = GRIDS[RASTER_KWARGS["grid"]]
    return np.zeros((grid.height, grid.width), dtype="float32")


def _read_catalog(catalog_path):
    return pl.read_parquet(catalog_path, hive_partitioning=True)


def invoke(*args):
    return CliRunner().invoke(cli, ["catalog", *args])


class TestBuild:
    def test_scans_tabular_dataset_and_writes_catalog_rows(self, tmp_path):
        base = tmp_path / "data-long"
        base.mkdir()
        write_parquet(
            _tabular_frame([2020, 2021, 2022]), base_path=base, catalog_path=None, validate=False, **TABULAR_KWARGS
        )
        out = tmp_path / "catalog"

        result = invoke("build", "--base-path", str(base), "--out", str(out))

        assert result.exit_code == 0
        assert "tabular:" in result.output
        assert "DONE: 1 rows" in result.output

        catalog = _read_catalog(out)
        assert catalog.height == 1
        row = catalog.row(0, named=True)
        assert row["data_type"] == "tabular"
        assert (row["calendar_year_min"], row["calendar_year_max"], row["calendar_year_count"]) == (2020, 2022, 3)

    def test_splits_a_gap_into_separate_rows(self, tmp_path):
        """The docstring's core row model: one row per contiguous calendar run, so a
        gap in the data produces two rows rather than one row with a wrong extent."""
        base = tmp_path / "data-long"
        base.mkdir()
        write_parquet(_tabular_frame([2020, 2021]), base_path=base, catalog_path=None, validate=False, **TABULAR_KWARGS)
        write_parquet(
            _tabular_frame([2025, 2026]),
            base_path=base,
            catalog_path=None,
            validate=False,
            **{**TABULAR_KWARGS, "scenario_id": 7},
        )
        out = tmp_path / "catalog"

        result = invoke("build", "--base-path", str(base), "--out", str(out))

        assert result.exit_code == 0
        catalog = _read_catalog(out)
        assert catalog.height == 2
        ranges = sorted(zip(catalog["calendar_year_min"].to_list(), catalog["calendar_year_max"].to_list()))
        assert ranges == [(2020, 2021), (2025, 2026)]

    def test_scans_raster_dataset_and_writes_catalog_row(self, tmp_path):
        base = tmp_path / "data-long"
        base.mkdir()
        write_cog(_raster_array(), base_path=base, catalog_path=None, calendar_year=2030, **RASTER_KWARGS)
        out = tmp_path / "catalog"

        result = invoke("build", "--base-path", str(base), "--out", str(out))

        assert result.exit_code == 0
        assert "raster: 1 .tif files" in result.output
        catalog = _read_catalog(out)
        assert catalog.height == 1
        row = catalog.row(0, named=True)
        assert row["data_type"] == "raster"
        assert row["model"] == "icm_v2023.0.0"

    def test_scans_mixed_tabular_and_raster_datasets_together(self, tmp_path):
        base = tmp_path / "data-long"
        base.mkdir()
        write_parquet(_tabular_frame([2030]), base_path=base, catalog_path=None, validate=False, **TABULAR_KWARGS)
        write_cog(_raster_array(), base_path=base, catalog_path=None, calendar_year=2030, **RASTER_KWARGS)
        out = tmp_path / "catalog"

        result = invoke("build", "--base-path", str(base), "--out", str(out))

        assert result.exit_code == 0
        catalog = _read_catalog(out)
        assert sorted(catalog["data_type"].to_list()) == ["raster", "tabular"]

    def test_skips_a_dataset_that_fails_to_parse_and_continues(self, tmp_path):
        """A directory whose path is missing an expected dimension segment (here
        `time_unit=...`) makes `tabular_catalog_from_files` raise KeyError; `build`
        reports it as SKIPPED rather than aborting the whole scan."""
        base = tmp_path / "data-long"
        base.mkdir()
        write_parquet(_tabular_frame([2030]), base_path=base, catalog_path=None, validate=False, **TABULAR_KWARGS)

        broken = base / "variable=broken" / "grid=ecoregion_v001"
        broken.mkdir(parents=True)
        pl.DataFrame({"grid_cell": [1], "calendar_year": [2020], "value": [1.0]}).write_parquet(
            broken / "x.parquet"
        )
        out = tmp_path / "catalog"

        result = invoke("build", "--base-path", str(base), "--out", str(out))

        assert result.exit_code == 0
        assert "SKIPPED (KeyError:" in result.output
        catalog = _read_catalog(out)
        assert catalog.height == 1  # only the good dataset made it in

    def test_reports_done_with_no_crash_when_no_valid_datasets_found(self, tmp_path):
        """An empty base directory -- or one where every dataset was skipped -- used to
        crash with `ValueError: cannot concat empty list` from `pl.concat(frames, ...)`,
        since `frames` stays empty when nothing parses. `build` should report zero rows
        instead of raising."""
        base = tmp_path / "data-long"
        base.mkdir()
        out = tmp_path / "catalog"

        result = invoke("build", "--base-path", str(base), "--out", str(out))

        assert result.exit_code == 0
        assert "DONE: no valid datasets found" in result.output
        assert not out.exists()

    def test_reports_done_with_no_crash_when_every_dataset_is_skipped(self, tmp_path):
        base = tmp_path / "data-long"
        broken = base / "variable=broken" / "grid=ecoregion_v001"
        broken.mkdir(parents=True)
        pl.DataFrame({"grid_cell": [1], "calendar_year": [2020], "value": [1.0]}).write_parquet(
            broken / "x.parquet"
        )
        out = tmp_path / "catalog"

        result = invoke("build", "--base-path", str(base), "--out", str(out))

        assert result.exit_code == 0
        assert "SKIPPED (KeyError:" in result.output
        assert "DONE: no valid datasets found" in result.output
        assert not out.exists()

    def test_uses_relative_defaults_from_the_current_directory(self, tmp_path):
        """`--base-path`/`--out` default to relative paths ("data-long"/"catalog"), so
        an invocation with no options resolves them against the cwd -- sandboxed here
        with `isolated_filesystem()` rather than the real repo cwd."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
            base = Path(cwd) / "data-long"
            base.mkdir()
            write_parquet(_tabular_frame([2030]), base_path=base, catalog_path=None, validate=False, **TABULAR_KWARGS)

            result = runner.invoke(cli, ["catalog", "build"])

            assert result.exit_code == 0
            assert (Path(cwd) / "catalog").exists()


class TestCompact:
    def test_merges_adjacent_runs_into_one_file(self, base_path, catalog_path):
        write_parquet(_tabular_frame([2020, 2021, 2022]), base_path=base_path, catalog_path=catalog_path, validate=False, **TABULAR_KWARGS)
        write_parquet(_tabular_frame([2023, 2024]), base_path=base_path, catalog_path=catalog_path, validate=False, **TABULAR_KWARGS)
        assert _read_catalog(catalog_path).height == 2

        result = invoke("compact", "--catalog", str(catalog_path))

        assert result.exit_code == 0
        assert "compacted" in result.output
        assert "DONE: 1 partitions, 2 -> 1 rows" in result.output
        catalog = _read_catalog(catalog_path)
        assert catalog.height == 1
        row = catalog.row(0, named=True)
        assert (row["calendar_year_min"], row["calendar_year_max"], row["calendar_year_count"]) == (2020, 2024, 5)

    def test_preserves_a_real_gap_as_two_rows(self, base_path, catalog_path):
        write_parquet(_tabular_frame([2020, 2021]), base_path=base_path, catalog_path=catalog_path, validate=False, **TABULAR_KWARGS)
        write_parquet(_tabular_frame([2025, 2026]), base_path=base_path, catalog_path=catalog_path, validate=False, **TABULAR_KWARGS)

        result = invoke("compact", "--catalog", str(catalog_path))

        assert result.exit_code == 0
        catalog = _read_catalog(catalog_path)
        assert catalog.height == 2
        ranges = sorted(zip(catalog["calendar_year_min"].to_list(), catalog["calendar_year_max"].to_list()))
        assert ranges == [(2020, 2021), (2025, 2026)]

    def test_leaves_a_single_file_partition_untouched(self, base_path, catalog_path):
        """`build` (and each write_parquet call) already emits minimal per-run files, so
        a partition with only one file needs no merge -- `compact` should report it as
        untouched (0 partitions) rather than rewriting it."""
        write_parquet(_tabular_frame([2020, 2021]), base_path=base_path, catalog_path=catalog_path, validate=False, **TABULAR_KWARGS)
        before = list(catalog_path.rglob("*.parquet"))

        result = invoke("compact", "--catalog", str(catalog_path))

        assert result.exit_code == 0
        assert "DONE: 0 partitions, 0 -> 0 rows" in result.output
        assert list(catalog_path.rglob("*.parquet")) == before


class TestQuery:
    def test_reports_choices_and_results_for_a_tabular_dataset(self, base_path, catalog_path):
        write_parquet(_tabular_frame([2020, 2021, 2022]), base_path=base_path, catalog_path=catalog_path, validate=False, **TABULAR_KWARGS)

        result = invoke(
            "query",
            "--catalog",
            str(catalog_path),
            "--variable",
            "sal",
            "--time-unit",
            "annual",
            "--grid",
            "hydro_compartment_v001",
        )

        assert result.exit_code == 0
        assert "[table]" in result.output
        assert "scenario_id: choices=[7]  result=[7]" in result.output
        assert "calendar_year: choices=[[2020, 2022]]  result=[2020]" in result.output

    def test_no_catalog_entries_reports_an_error_rather_than_a_crash(self, catalog_path):
        catalog_path.mkdir()

        result = invoke(
            "query",
            "--catalog",
            str(catalog_path),
            "--variable",
            "sal",
            "--time-unit",
            "annual",
            "--grid",
            "hydro_compartment_v001",
        )

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
