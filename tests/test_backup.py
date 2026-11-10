"""Tests for run backup/restore/compaction -- cpra/mp/data/backup.py, the `backup`/`restore`
commands in cpra/mp/data/cli/backup.py and the `data compact` command
in cpra/mp/data/cli/data.py.

Every run here is written to three datasets, one of each shape the archive handles
differently: a raster (COGs, archived as-is), daily Parquet (partitioned by run, archived
as-is) and annual Parquet (one file for every run, so the run's rows are extracted).

All data lives under tmp_path -- never the real data/ directory (see repo CLAUDE.md).
"""

import io
import json
import shutil
import tarfile

import polars as pl
import pytest
from click.testing import CliRunner
from polars.testing import assert_frame_equal

import cpra.mp.data.cli.backup  # (registers `backup`/`restore` on `cli`)
import cpra.mp.data.cli.data  # noqa: F401  (registers `data compact` on `cli`)
from cpra.mp.data.backup import MANIFEST_NAME, ArchiveError, inventory_run, restore_archive
from cpra.mp.data.cli import cli
from cpra.mp.data.tabular import write_parquet
from tests.factories import daily_frame, write_annual_dataset, write_raster_dataset

RUN = {"model_group_id": 500, "scenario_id": 7}
OTHER_RUNS = [{"model_group_id": 500, "scenario_id": 8}, {"model_group_id": 501, "scenario_id": 7}]

ANNUAL_DIR = "variable=ssf/time_unit=annual/grid=ecoregion_v001"
DAILY_DIR = "variable=sal/time_unit=daily/grid=hydro_compartment_v001"
RASTER_DIR = "variable=vct/time_unit=annual/grid=veg_grid_cell_v001"


def _write_run(base, catalog, run, years=(2030, 2031)):
    # Each call to a writer adds a file, so writing year by year leaves uncompacted data.
    for year in years:
        write_annual_dataset(base, years=(year,), **run)
        write_parquet(daily_frame(**run), base_path=base, catalog_path=catalog, validate=False)
        write_raster_dataset(base, years=(year,), **run)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A data tree and catalog holding RUN and OTHER_RUNS, written without compaction."""
    from cpra.mp.data import config

    base, catalog = tmp_path / "data", tmp_path / "catalog"
    base.mkdir()
    monkeypatch.setattr(config, "CATALOG_PATH", catalog)
    for run in [RUN, *OTHER_RUNS]:
        _write_run(base, catalog, run)
    return base, catalog


def _invoke(*args):
    return CliRunner().invoke(cli, [str(arg) for arg in args], catch_exceptions=False)


def _backup(base, catalog, out, *extra):
    result = _invoke(
        "backup", "--model-group-id", RUN["model_group_id"], "--scenario-id", RUN["scenario_id"],
        "--output-dir", out, "--base-path", base, "--catalog", catalog, *extra,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    archives = sorted(out.glob("*.tar"))
    assert len(archives) == 1
    return archives[0]


def _read(root, dataset_dir, run=None):
    """Every row of a Parquet dataset (or catalog partition), in a stable order."""
    frame = pl.read_parquet(root / dataset_dir, hive_partitioning=True)
    if run is not None:
        frame = frame.filter(**run)
    return frame.sort(frame.columns)


def _files(root, dataset_dir, run):
    """``{relative path: bytes}`` for a run's partition directory."""
    run_dir = root / dataset_dir / f"model_group_id={run['model_group_id']}" / f"scenario_id={run['scenario_id']}"
    return {p.relative_to(root): p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}


def _purge(base, catalog, run):
    """Remove a run by hand: its partition directories, and its rows from mixed files."""
    for dataset_dir in (DAILY_DIR, RASTER_DIR):
        shutil.rmtree(
            base / dataset_dir / f"model_group_id={run['model_group_id']}" / f"scenario_id={run['scenario_id']}"
        )
    keep = ~((pl.col("model_group_id") == run["model_group_id"]) & (pl.col("scenario_id") == run["scenario_id"]))
    for path in [*(base / ANNUAL_DIR).glob("*.parquet"), *catalog.rglob("*.parquet")]:
        pl.read_parquet(path, hive_partitioning=False).filter(keep).write_parquet(path)


class TestInventory:
    def test_finds_each_kind_of_member(self, tree):
        base, catalog = tree
        inventory = inventory_run(base, catalog, **RUN)

        assert inventory.datasets == sorted([ANNUAL_DIR, DAILY_DIR, RASTER_DIR])
        assert inventory.warnings == []
        by_kind = {}
        for member in inventory.members:
            by_kind.setdefault((member.kind, member.method), []).append(member)
        assert len(by_kind[("cog", "copy")]) == 2
        assert len(by_kind[("parquet", "copy")]) == 2
        # Both annual files hold this run's rows, so both are sources of one extract.
        [annual] = by_kind[("parquet", "extract")]
        assert len(annual.sources) == 2
        assert annual.rows == _read(base, ANNUAL_DIR, RUN).height
        assert len(by_kind[("catalog", "extract")]) == 3

    def test_reports_data_without_catalog_rows(self, tree):
        base, catalog = tree
        shutil.rmtree(catalog / RASTER_DIR)
        inventory = inventory_run(base, catalog, **RUN)
        assert inventory.warnings == [f"{RASTER_DIR}: data files but no catalog rows"]


class TestRoundTrip:
    def test_restore_into_empty_tree_reproduces_the_run(self, tree, tmp_path):
        base, catalog = tree
        archive = _backup(base, catalog, tmp_path / "out")

        new_base, new_catalog = tmp_path / "new-data", tmp_path / "new-catalog"
        result = _invoke("restore", archive, "--base-path", new_base, "--catalog", new_catalog)
        assert result.exit_code == 0, result.output

        assert _read(new_base, ANNUAL_DIR).equals(_read(base, ANNUAL_DIR, RUN))
        for dataset_dir in (DAILY_DIR, RASTER_DIR):
            assert _files(new_base, dataset_dir, RUN) == _files(base, dataset_dir, RUN)
            assert _read(new_catalog, dataset_dir).equals(_read(catalog, dataset_dir, RUN))
        # No other run came along, and no staging directory was left behind.
        assert _read(new_base, ANNUAL_DIR).select("model_group_id", "scenario_id").unique().height == 1
        assert not list(new_base.glob(".restore-*")) and not list(new_catalog.glob(".restore-*"))

    def test_restore_after_purge_leaves_other_runs_untouched(self, tree, tmp_path):
        base, catalog = tree
        before = {d: _read(base, d) for d in (ANNUAL_DIR, DAILY_DIR)}
        catalog_before = {d: _read(catalog, d) for d in (ANNUAL_DIR, DAILY_DIR, RASTER_DIR)}
        raster_before = _files(base, RASTER_DIR, RUN)

        archive = _backup(base, catalog, tmp_path / "out")
        _purge(base, catalog, RUN)
        assert _read(base, ANNUAL_DIR, RUN).height == 0

        restore_archive(archive, base, catalog, echo=lambda *_: None)

        for dataset_dir, frame in before.items():
            assert _read(base, dataset_dir).equals(frame)
        for dataset_dir, frame in catalog_before.items():
            assert _read(catalog, dataset_dir).equals(frame)
        assert _files(base, RASTER_DIR, RUN) == raster_before

    def test_manifest_describes_the_archive(self, tree, tmp_path):
        base, catalog = tree
        archive = _backup(base, catalog, tmp_path / "out")

        assert archive.name.startswith("g500-s7-")
        with tarfile.open(archive) as tar:
            names = tar.getnames()
            manifest = json.load(tar.extractfile(MANIFEST_NAME))
        assert names[-1] == MANIFEST_NAME
        assert manifest["model_group_id"] == 500 and manifest["scenario_id"] == 7
        assert sorted(f["path"] for f in manifest["files"]) == sorted(names[:-1])
        assert all(len(f["sha256"]) == 64 for f in manifest["files"])


class TestRestoreRefuses:
    def test_when_the_run_is_still_present(self, tree, tmp_path):
        base, catalog = tree
        archive = _backup(base, catalog, tmp_path / "out")
        before = _read(base, ANNUAL_DIR)

        with pytest.raises(ArchiveError, match="already present"):
            restore_archive(archive, base, catalog)
        assert _read(base, ANNUAL_DIR).equals(before)

        result = CliRunner().invoke(cli, ["restore", str(archive), "--base-path", str(base), "--catalog", str(catalog)])
        assert result.exit_code == 1
        assert "already present" in result.output

    def test_when_a_member_is_corrupt_and_writes_nothing(self, tree, tmp_path):
        base, catalog = tree
        archive = _backup(base, catalog, tmp_path / "out")

        # Rebuild the archive with one member's bytes changed but its manifest entry intact.
        corrupt = tmp_path / "corrupt.tar"
        with tarfile.open(archive) as src, tarfile.open(corrupt, "w") as dst:
            for i, member in enumerate(src.getmembers()):
                data = src.extractfile(member).read()
                if i == 0:
                    data = bytes([data[0] ^ 0xFF]) + data[1:]
                dst.addfile(member, fileobj=io.BytesIO(data))

        new_base, new_catalog = tmp_path / "new-data", tmp_path / "new-catalog"
        with pytest.raises(ArchiveError, match="checksum mismatch"):
            restore_archive(corrupt, new_base, new_catalog)
        assert not [p for p in new_base.rglob("*") if p.is_file()]
        assert not [p for p in new_catalog.rglob("*") if p.is_file()]

    def test_dry_run_verifies_and_writes_nothing(self, tree, tmp_path):
        base, catalog = tree
        archive = _backup(base, catalog, tmp_path / "out")

        new_base = tmp_path / "new-data"
        result = _invoke(
            "restore", archive, "--base-path", new_base, "--catalog", tmp_path / "new-catalog", "--dry-run"
        )
        assert result.exit_code == 0, result.output
        assert "verified" in result.output
        assert not new_base.exists()


class TestBackup:
    def test_compact_merges_files_and_keeps_every_run(self, tree, tmp_path):
        base, catalog = tree
        annual_before = _read(base, ANNUAL_DIR)
        assert len(list((base / ANNUAL_DIR).glob("*.parquet"))) == 6

        _backup(base, catalog, tmp_path / "out", "--compact")

        # The mixed annual directory is compacted whole, and no run's rows are lost.
        assert len(list((base / ANNUAL_DIR).glob("*.parquet"))) == 1
        assert _read(base, ANNUAL_DIR).equals(annual_before)
        # Only this run's partitions are compacted.
        raster_run = base / RASTER_DIR / "model_group_id=500"
        assert len(list((raster_run / "scenario_id=7").glob("*.tif"))) == 1
        assert len(list((raster_run / "scenario_id=8").glob("*.tif"))) == 2
        for dataset_dir in (ANNUAL_DIR, DAILY_DIR, RASTER_DIR):
            assert len(list((catalog / dataset_dir).glob("*.parquet"))) == 1

    def test_compact_twice_changes_nothing(self, tree, tmp_path):
        base, catalog = tree
        _backup(base, catalog, tmp_path / "first", "--compact")
        files = sorted(p for p in base.rglob("*") if p.is_file())
        _backup(base, catalog, tmp_path / "second", "--compact")
        assert sorted(p for p in base.rglob("*") if p.is_file()) == files

    def test_dry_run_writes_and_compacts_nothing(self, tree, tmp_path):
        base, catalog = tree
        files = sorted(p for p in base.rglob("*") if p.is_file())
        out = tmp_path / "out"
        result = _invoke(
            "backup", "--model-group-id", 500, "--scenario-id", 7, "-o", out,
            "--base-path", base, "--catalog", catalog, "--compact", "--dry-run",
        )  # fmt: skip
        assert result.exit_code == 0, result.output
        assert RASTER_DIR in result.output
        assert not out.exists()
        assert sorted(p for p in base.rglob("*") if p.is_file()) == files

    def test_run_can_come_from_the_environment(self, tree, tmp_path):
        base, catalog = tree
        out = tmp_path / "out"
        result = CliRunner().invoke(
            cli,
            ["backup", "-o", str(out), "--base-path", str(base), "--catalog", str(catalog)],
            env={"MODEL_GROUP_ID": str(RUN["model_group_id"]), "SCENARIO_ID": str(RUN["scenario_id"])},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert [p.name[: p.name.rindex("-")] for p in out.glob("*.tar")] == [
            f"g{RUN['model_group_id']}-s{RUN['scenario_id']}"
        ]

    def test_unknown_run_is_an_error(self, tree, tmp_path):
        base, catalog = tree
        result = CliRunner().invoke(
            cli,
            ["backup", "--model-group-id", "999", "--scenario-id", "7", "-o", str(tmp_path / "out"),
             "--base-path", str(base), "--catalog", str(catalog)],
        )  # fmt: skip
        assert result.exit_code == 1
        assert "no data or catalog rows found for g999-s7" in result.output
        assert not (tmp_path / "out").exists()


def _crowded(root, suffix):
    """Directories under ``root`` still holding more than one ``suffix`` file."""
    return [d for d in [root, *root.rglob("*")] if d.is_dir() and len(list(d.glob(f"*{suffix}"))) > 1]


def _compact(base, catalog, *extra, env=None):
    result = CliRunner().invoke(
        cli,
        ["data", "compact", "--base-path", str(base), "--catalog", str(catalog), *map(str, extra)],
        env=env,
        catch_exceptions=False,
    )
    return result


class TestCompact:
    def test_compacts_like_backup_compact(self, tree, tmp_path):
        base, catalog = tree
        twin = tmp_path / "twin"
        shutil.copytree(base.parent, twin)
        annual_before = _read(base, ANNUAL_DIR)

        result = _compact(base, catalog, "--model-group-id", 500, "--scenario-id", 7)
        assert result.exit_code == 0, result.output
        _backup(twin / "data", twin / "catalog", tmp_path / "out", "--compact")

        def layout(root):
            return sorted((p.parent.relative_to(root), p.suffix) for p in root.rglob("*") if p.is_file())

        assert layout(base) == layout(twin / "data")
        assert layout(catalog) == layout(twin / "catalog")
        assert _read(base, ANNUAL_DIR).equals(annual_before)

    def test_dry_run_changes_nothing(self, tree):
        base, catalog = tree
        files = sorted(p for p in base.parent.rglob("*") if p.is_file())

        result = _compact(base, catalog, "--model-group-id", 500, "--scenario-id", 7, "--dry-run")

        assert result.exit_code == 0, result.output
        assert f"{ANNUAL_DIR}: would merge 6 files (all runs)" in result.output
        assert RASTER_DIR in result.output
        assert sorted(p for p in base.parent.rglob("*") if p.is_file()) == files

    def test_run_can_come_from_the_environment(self, tree):
        base, catalog = tree
        result = _compact(base, catalog, env={"MODEL_GROUP_ID": "500", "SCENARIO_ID": "7"})
        assert result.exit_code == 0, result.output
        assert len(list((base / ANNUAL_DIR).glob("*.parquet"))) == 1

    def test_compacts_the_whole_tree_by_default(self, tree):
        base, catalog = tree
        before = {d: _read(base, d) for d in (ANNUAL_DIR, DAILY_DIR)}
        # The fixture writes each run's daily frame twice; the catalog keeps one row each.
        catalog_before = _read(catalog, DAILY_DIR).unique(maintain_order=True)
        n_tifs = len(list((base / RASTER_DIR).rglob("*.tif")))

        result = _compact(base, catalog)

        assert result.exit_code == 0, result.output
        assert "compacting everything (3 dataset(s))" in result.output
        assert _crowded(base, ".parquet") == []
        assert _crowded(catalog, ".parquet") == []
        assert len(list((base / RASTER_DIR).rglob("*.tif"))) < n_tifs
        for dataset_dir, frame in before.items():
            assert _read(base, dataset_dir).equals(frame)
        assert_frame_equal(_read(catalog, DAILY_DIR), catalog_before, check_column_order=False, check_dtypes=False)

    def test_whole_tree_dry_run_changes_nothing(self, tree):
        base, catalog = tree
        files = sorted(p for p in base.parent.rglob("*") if p.is_file())

        result = _compact(base, catalog, "--dry-run")

        assert result.exit_code == 0, result.output
        assert f"{ANNUAL_DIR}: would merge 6 files in 1 directory(ies)" in result.output
        assert f"{DAILY_DIR}: would merge" in result.output
        assert f"{RASTER_DIR}: would check" in result.output
        assert sorted(p for p in base.parent.rglob("*") if p.is_file()) == files

    def test_dataset_filters_leave_other_datasets_alone(self, tree):
        base, catalog = tree
        untouched = {d: sorted((base / d).rglob("*")) for d in (ANNUAL_DIR, RASTER_DIR)}
        untouched_catalog = sorted((catalog / ANNUAL_DIR).rglob("*"))

        result = _compact(base, catalog, "--variable", "sal", "--time-unit", "daily")

        assert result.exit_code == 0, result.output
        assert _crowded(base / DAILY_DIR, ".parquet") == []
        assert _crowded(catalog / DAILY_DIR, ".parquet") == []
        for dataset_dir, paths in untouched.items():
            assert sorted((base / dataset_dir).rglob("*")) == paths
        assert sorted((catalog / ANNUAL_DIR).rglob("*")) == untouched_catalog

    def test_a_run_and_a_dataset_filter_combine(self, tree):
        base, catalog = tree
        other = OTHER_RUNS[0]
        other_files = _files(base, DAILY_DIR, other)

        result = _compact(base, catalog, "--model-group-id", 500, "--scenario-id", 7, "--variable", "sal")

        assert result.exit_code == 0, result.output
        assert "compacting g500-s7, variable=sal (1 dataset(s))" in result.output
        assert _crowded(base / DAILY_DIR / "model_group_id=500" / "scenario_id=7", ".parquet") == []
        assert _files(base, DAILY_DIR, other) == other_files
        assert len(list((base / ANNUAL_DIR).glob("*.parquet"))) == 6

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["--model-group-id", "500"], "--model-group-id and --scenario-id go together"),
            (["--variable", "nope"], "no datasets match variable=nope"),
            (
                ["--model-group-id", "500", "--scenario-id", "7", "--grid", "nope"],
                "no datasets match g500-s7, grid=nope",
            ),
        ],
    )
    def test_bad_selections_are_errors(self, tree, args, message):
        base, catalog = tree
        files = sorted(p for p in base.parent.rglob("*") if p.is_file())

        result = CliRunner().invoke(
            cli, ["data", "compact", "--base-path", str(base), "--catalog", str(catalog), *args]
        )

        assert result.exit_code != 0
        assert message in result.output
        assert sorted(p for p in base.parent.rglob("*") if p.is_file()) == files

    def test_unknown_run_is_an_error(self, tree):
        base, catalog = tree
        files = sorted(p for p in base.parent.rglob("*") if p.is_file())
        result = CliRunner().invoke(
            cli,
            ["data", "compact", "--model-group-id", "999", "--scenario-id", "7",
             "--base-path", str(base), "--catalog", str(catalog)],
        )  # fmt: skip
        assert result.exit_code == 1
        assert "no data or catalog rows found for g999-s7" in result.output
        assert sorted(p for p in base.parent.rglob("*") if p.is_file()) == files
