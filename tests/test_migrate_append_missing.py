"""Tests for ``scripts/migrate_to_long_format.py --append-missing``.

The migration is a long pass over a tree that does not hold still: model runs keep writing
new wide-format Parquet and new single-band COGs into the input while it runs, so a
partition converted early can be short by the time the pass finishes -- a model group that
did not exist yet, or extra calendar values inside one that did. ``--append-missing``
re-checks every partition against its source and writes only what is absent, as additional
files in the same partition.

Two properties are what the option is worth anything for, and most of the tests below pin
one or the other:

* it appends *only* what is missing -- no duplicate rows, no rewritten file; and
* it never deletes, so an output that has more than its source is reported, not repaired.

``scripts/`` is not an importable package, so the module is loaded by path.
"""

import importlib.util
import os
import subprocess
import sys
import uuid
from pathlib import Path

import polars as pl
import rasterio as rio
from click.testing import CliRunner

from cpra.mp.data.tabular import read_parquet
from tests.factories import ANNUAL_KWARGS, DAILY_KWARGS, annual_frame, daily_frame, write_raster_dataset

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "migrate_to_long_format.py"


def _load_migrate_module():
    spec = importlib.util.spec_from_file_location("migrate_to_long_format", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migrate = _load_migrate_module()

ANNUAL_DATASET = {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}
DAILY_DATASET = {k: DAILY_KWARGS[k] for k in ("variable", "grid", "time_unit")}
N_CELLS = 32  # ecoregion_v001's full cell count, which annual_frame supplies


def _write_wide(base_path, frame, path_dims):
    """Write one wide-format Parquet file where the migration expects to find it.

    Dimensions encoded in the directory path are dropped from the frame, matching the
    real input tree (checked against it: an annual file carries model_group_id,
    scenario_id, aggregate_type, calendar_year and model as columns, and nothing that is
    already a directory). The directory order is the pre-migration one --
    variable/grid/time_unit -- which is what ``find_partitions`` globs for.
    """
    directory = base_path.joinpath(*(f"{name}={value}" for name, value in path_dims.items()))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}-0.parquet"
    frame.drop(*(name for name in path_dims if name in frame.columns)).write_parquet(path)
    return path


def _write_annual(base_path, years=(2030, 2040), **overrides):
    path_dims = {name: ANNUAL_KWARGS[name] for name in ("variable", "grid", "time_unit")}
    return _write_wide(base_path, annual_frame(years=years, **overrides), path_dims)


def _write_daily(base_path, days, **overrides):
    kwargs = {**DAILY_KWARGS, **overrides}
    path_dims = {
        name: kwargs[name]
        for name in ("variable", "grid", "time_unit", "model_group_id", "scenario_id", "aggregate_type")
    }
    return _write_wide(base_path, daily_frame(days=days, **overrides), path_dims)


def _run(base_path, output_path, catalog_path=None, **kwargs):
    """Migrate the one tabular partition under ``base_path``."""
    partitions = migrate.find_partitions(base_path)
    assert len(partitions) == 1, f"expected one partition, got {list(partitions)}"
    paths = next(iter(partitions.values()))
    return migrate.migrate_partition(paths, base_path, output_path, catalog_path, **kwargs)


def _output_files(output_path):
    return sorted(output_path.rglob("*.parquet"))


def _rows(output_path, dataset):
    return read_parquet(base_path=output_path, **dataset).collect()


class TestAppendMissingTabular:
    def test_appends_a_model_group_that_appeared_after_conversion(self, base_path, tmp_path):
        """The headline case: partition converted, model run adds another model group to the
        same partition, append picks up that group and nothing else."""
        output_path = tmp_path / "out"
        _write_annual(base_path, model_group_id=500)
        _run(base_path, output_path, catalog_path=None)
        converted = _output_files(output_path)
        assert len(converted) == 1

        _write_annual(base_path, model_group_id=501)
        result, _messages = _run(base_path, output_path, catalog_path=None, append_missing=True)

        assert result["appended"] is True
        assert result["long_rows"] == 2 * N_CELLS
        # The appended rows go to a new file; the converted one is left exactly as it was.
        files = _output_files(output_path)
        assert len(files) == 2
        assert converted[0] in files

        rows = _rows(output_path, ANNUAL_DATASET)
        assert rows.height == 2 * 2 * N_CELLS
        assert sorted(rows["model_group_id"].unique()) == [500, 501]

    def test_appends_calendar_values_missing_from_a_group_already_present(self, base_path, tmp_path):
        """The harder half: the group is in the output, just short. Row counts alone say how
        many are missing; the anti-join says which, so nothing already there is duplicated."""
        output_path = tmp_path / "out"
        _write_annual(base_path, years=(2030, 2040))
        _run(base_path, output_path, catalog_path=None)

        _write_annual(base_path, years=(2050,))
        result, _messages = _run(base_path, output_path, catalog_path=None, append_missing=True)

        assert result["long_rows"] == N_CELLS
        rows = _rows(output_path, ANNUAL_DATASET)
        assert rows.height == 3 * N_CELLS
        assert sorted(rows["calendar_year"].unique()) == [2030, 2040, 2050]

    def test_appends_missing_days_to_a_daily_partition(self, base_path, tmp_path):
        """Daily is the shape where the partition dimensions are directories rather than
        columns, so the whole partition is effectively one group."""
        import datetime

        output_path = tmp_path / "out"
        days = [datetime.date(2030, 1, 1), datetime.date(2030, 1, 2)]
        _write_daily(base_path, days=days)
        _run(base_path, output_path, catalog_path=None)

        _write_daily(base_path, days=[datetime.date(2030, 1, 3)])
        result, _messages = _run(base_path, output_path, catalog_path=None, append_missing=True)

        assert result["long_rows"] == 3  # daily_frame carries three cells
        rows = _rows(output_path, DAILY_DATASET)
        assert rows.height == 9
        assert sorted(rows["calendar_day"].unique()) == [*days, datetime.date(2030, 1, 3)]

    def test_leaves_a_complete_partition_alone(self, base_path, tmp_path):
        """Every partition is checked, including ones already converted -- so the no-op case
        has to stay a true no-op, not an empty file per pass."""
        output_path = tmp_path / "out"
        _write_annual(base_path)
        _run(base_path, output_path, catalog_path=None)
        before = _output_files(output_path)

        result, messages = _run(base_path, output_path, catalog_path=None, append_missing=True)

        assert result["skipped"] is True
        assert _output_files(output_path) == before
        assert any("Complete" in message for message in messages)

    def test_reports_output_rows_the_source_no_longer_has(self, base_path, tmp_path):
        """A group in the output but not (or no longer) in the source is a judgement call --
        a stale purge, a source that shrank, duplicates. Appending says so and moves on."""
        output_path = tmp_path / "out"
        _write_annual(base_path, model_group_id=500)
        stale = _write_annual(base_path, model_group_id=501)
        _run(base_path, output_path, catalog_path=None)
        before = _output_files(output_path)
        stale.unlink()

        result, messages = _run(base_path, output_path, catalog_path=None, append_missing=True)

        assert result["skipped"] is True
        assert _output_files(output_path) == before
        assert _rows(output_path, ANNUAL_DATASET).height == 2 * 2 * N_CELLS
        assert any("WARNING" in message and "more than the source" in message for message in messages)

    def test_dry_run_writes_nothing(self, base_path, tmp_path):
        output_path = tmp_path / "out"
        _write_annual(base_path, model_group_id=500)
        _run(base_path, output_path, catalog_path=None)
        before = _output_files(output_path)
        _write_annual(base_path, model_group_id=501)

        result, messages = _run(
            base_path, output_path, catalog_path=None, append_missing=True, dry_run=True
        )

        assert result["long_rows"] == 2 * N_CELLS
        assert _output_files(output_path) == before
        assert any("[DRY RUN]" in message for message in messages)

    def test_catalogs_the_appended_rows(self, base_path, tmp_path):
        """The appended data has to be findable, or the next run would 'discover' it missing
        all over again -- and the API reads the catalog, not the tree."""
        output_path = tmp_path / "out"
        out_catalog = tmp_path / "out-catalog"
        _write_annual(base_path, model_group_id=500)
        _run(base_path, output_path, out_catalog)

        _write_annual(base_path, model_group_id=501)
        _run(base_path, output_path, out_catalog, append_missing=True)

        rows = pl.concat(
            [pl.read_parquet(p) for p in out_catalog.rglob("*.parquet")], how="diagonal_relaxed"
        )
        assert sorted(rows["model_group_id"].unique()) == [500, 501]


class TestAppendMissingRaster:
    def _band_values(self, output_path):
        values = []
        for path in sorted(output_path.rglob("*.tif")):
            with rio.open(path) as src:
                values += [int(desc.split("=")[1]) for desc in src.descriptions]
        return sorted(values)

    def _dims(self, base_path):
        datasets = migrate.find_raster_datasets(base_path)
        assert len(datasets) == 1, f"expected one raster partition, got {datasets}"
        return datasets[0]

    def test_appends_a_band_that_appeared_after_conversion(self, base_path, tmp_path):
        """2032 arrives on its own after 2030-2031 were compacted. On its own it is already
        an optimally-stored single band, which is exactly the shape an out-of-place
        compaction used to skip -- so this pins that it reaches the output."""
        output_path = tmp_path / "out"
        write_raster_dataset(base_path, years=(2030, 2031))
        dims = self._dims(base_path)
        migrate.migrate_raster_dataset(dims, base_path, output_path, catalog_path=None)
        assert self._band_values(output_path) == [2030, 2031]

        write_raster_dataset(base_path, years=(2032,))
        result, _messages = migrate.migrate_raster_dataset(
            dims, base_path, output_path, catalog_path=None, append_missing=True
        )

        assert result["appended"] is True
        assert self._band_values(output_path) == [2030, 2031, 2032]
        # The new band lands beside the compacted run rather than in it; consolidating the
        # two is what a later in-place compact_cog over the output tree is for.
        assert len(list(output_path.rglob("*.tif"))) == 2

    def test_leaves_a_complete_raster_partition_alone(self, base_path, tmp_path):
        output_path = tmp_path / "out"
        write_raster_dataset(base_path, years=(2030, 2031))
        dims = self._dims(base_path)
        migrate.migrate_raster_dataset(dims, base_path, output_path, catalog_path=None)
        before = sorted(output_path.rglob("*.tif"))

        result, messages = migrate.migrate_raster_dataset(
            dims, base_path, output_path, catalog_path=None, append_missing=True
        )

        assert result["skipped"] is True
        assert sorted(output_path.rglob("*.tif")) == before
        assert any("Complete" in message for message in messages)

    def test_reports_output_bands_the_source_no_longer_has(self, base_path, tmp_path):
        output_path = tmp_path / "out"
        write_raster_dataset(base_path, years=(2030, 2031))
        dims = self._dims(base_path)
        migrate.migrate_raster_dataset(dims, base_path, output_path, catalog_path=None)
        before = sorted(output_path.rglob("*.tif"))
        for path in migrate._raster_source_tifs(base_path, dims):
            with rio.open(path) as src:
                if src.descriptions[0] == "calendar_year=2031":
                    stale = path
        stale.unlink()

        result, messages = migrate.migrate_raster_dataset(
            dims, base_path, output_path, catalog_path=None, append_missing=True
        )

        assert result["skipped"] is True
        assert sorted(output_path.rglob("*.tif")) == before
        assert any("WARNING" in message and "the source does not" in message for message in messages)


class TestAppendMissingCli:
    def test_rejects_unconverted_only(self, tmp_path):
        """--unconverted-only trims the work list to what the catalog hasn't recorded;
        --append-missing deliberately re-examines what it has. Asking for both is a typo."""
        result = CliRunner().invoke(
            migrate.main,
            ["--output-path", str(tmp_path / "out"), "--catalog-path", str(tmp_path / "cat"),
             "--estimate", "--unconverted-only", "--append-missing"],
        )

        assert result.exit_code != 0
        assert "cannot be combined" in result.output

    def test_end_to_end_ignores_the_catalogs_already_converted_list(self, base_path, tmp_path):
        """The whole point of the flag: a partition the catalog calls done is still checked.

        Run as a subprocess rather than through CliRunner because the migration fans
        partitions out to spawned worker processes, which need the script to be __main__.
        """
        output_path = tmp_path / "out"
        catalog_path = tmp_path / "cat"
        _write_annual(base_path, model_group_id=500)

        def run(*extra):
            env = {
                **os.environ,
                "CPRA_MP_DATA_BASE_PATH": str(base_path),
                "CPRA_MP_DATA_CATALOG_PATH": str(catalog_path),
            }
            completed = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), "--data-path", str(base_path),
                 "--output-path", str(output_path), "--catalog-path", str(catalog_path),
                 "--no-progress", "--log-path", str(tmp_path / "log.csv"), *extra],
                capture_output=True, text=True, cwd=SCRIPT_PATH.parent.parent, env=env, check=False,
            )
            assert completed.returncode == 0, completed.stderr
            return completed.stdout

        run()
        _write_annual(base_path, model_group_id=501)

        # Without the flag the catalog says this partition is done, so nothing happens.
        assert "already converted per the catalog" in run()
        assert _rows(output_path, ANNUAL_DATASET).height == 2 * N_CELLS

        assert "Appended to 1 unit" in run("--append-missing")
        rows = _rows(output_path, ANNUAL_DATASET)
        assert rows.height == 2 * 2 * N_CELLS
        assert sorted(rows["model_group_id"].unique()) == [500, 501]
