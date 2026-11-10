"""Back up and restore one model run -- a ``model_group_id``/``scenario_id`` pair -- as a
single uncompressed tar.

Where a run's data lives depends on the dataset, and decides how it gets into the archive:

* **COGs** encode every partition dimension in their directory path (see
  ``raster.cog_path_expression``), and **daily Parquet** is Hive-partitioned by the same
  dimensions (``Dimension.partition``), so everything for one run sits under
  ``<dataset>/model_group_id=X/scenario_id=Y/``. Those files are archived as-is.
* **Every other Parquet** (annual/decadal/monthly) keeps ``model_group_id`` and
  ``scenario_id`` as columns, and one file holds every run in the dataset. The run's rows
  are extracted into a new file of their own.
* **The catalog** is one row per dimension combination, so its files mix runs too, and the
  run's rows are extracted the same way.

The archive mirrors the live layout under ``data/`` and ``catalog/``, plus a
``MANIFEST.json`` (written last) recording each member's checksum, row count and source.
Restoring is a matter of putting each member back at its relative path: readers pick up
every ``*.parquet`` in a dataset directory, so an extracted file can sit beside the
combined file it came from until the next compaction merges them. That only holds while
the run is absent -- restoring over a run that is still present would duplicate its rows
-- so ``restore_archive`` refuses to.

Plain tar, not gzip: COGs and Parquet are already compressed, so gzip would cost CPU for
almost nothing.
"""

import hashlib
import json
import shutil
import tarfile
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath

import polars as pl

from cpra.mp.data.catalog import Catalog, parse_path_dims
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.raster import compact_cog
from cpra.mp.data.tabular import _DEFAULT_ROW_GROUP_SIZE, compact_parquet

FORMAT_VERSION = 1
MANIFEST_NAME = "MANIFEST.json"
DATA_PREFIX = "data"
CATALOG_PREFIX = "catalog"
DATASET_DIMS = ("variable", "time_unit", "grid")
_CHUNK_SIZE = 1 << 20


@dataclass
class Member:
    """One file the archive will hold.

    ``method`` is ``copy`` for a live file archived as-is (``sources`` is that one file),
    or ``extract`` for rows filtered out of the mixed files in ``sources``.
    """

    arcname: str
    kind: str  # "cog", "parquet", "catalog" or "other"
    method: str  # "copy" or "extract"
    dataset: str
    sources: list[Path]
    size: int | None = None  # copy members only; an extract's size is known once written
    rows: int | None = None  # extract members only


@dataclass
class RunInventory:
    """Everything a run has on disk, as ``Member``s, plus anything that looked wrong."""

    model_group_id: int
    scenario_id: int
    members: list[Member] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def datasets(self) -> list[str]:
        return sorted({member.dataset for member in self.members})


def run_name(model_group_id, scenario_id) -> str:
    return f"g{model_group_id}-s{scenario_id}"


def _cast_run(model_group_id, scenario_id) -> tuple[int, int]:
    return DIMENSIONS["model_group_id"].cast(model_group_id), DIMENSIONS["scenario_id"].cast(scenario_id)


def _run_filter(model_group_id, scenario_id) -> pl.Expr:
    return (pl.col("model_group_id") == model_group_id) & (pl.col("scenario_id") == scenario_id)


def _dataset_key(dims: dict) -> str:
    return "/".join(f"{name}={dims[name]}" for name in DATASET_DIMS)


def _datasets_under(root: Path) -> dict[str, dict]:
    """The ``variable``/``time_unit``/``grid`` dataset directories under ``root``."""
    datasets = {}
    for directory in root.glob("variable=*/time_unit=*/grid=*"):
        if directory.is_dir():
            dims = parse_path_dims(directory.relative_to(root))
            datasets[_dataset_key(dims)] = dims
    return datasets


def _run_dir(base_path: Path, dataset: str, model_group_id, scenario_id) -> Path:
    return base_path / dataset / f"model_group_id={model_group_id}" / f"scenario_id={scenario_id}"


def _kind(path: Path) -> str:
    return {".tif": "cog", ".parquet": "parquet"}.get(path.suffix, "other")


def _mixed_parquet_files(directory: Path) -> list[Path]:
    """The Parquet files sitting directly in a dataset directory -- the unpartitioned kind
    that holds every run in one file."""
    return sorted(path for path in directory.glob("*.parquet") if path.is_file())


def _count_run_rows(paths: list[Path], model_group_id, scenario_id) -> int:
    # hive_partitioning=False so the count reflects the files' own columns rather than
    # anything inferred from the directories above them.
    lf = pl.scan_parquet(paths, hive_partitioning=False)
    if not {"model_group_id", "scenario_id"} <= set(lf.collect_schema().names()):
        return 0
    return lf.filter(_run_filter(model_group_id, scenario_id)).select(pl.len()).collect().item()


def _catalog_rows(directory: Path, model_group_id, scenario_id) -> tuple[pl.DataFrame | None, list[Path]]:
    """The run's catalog rows in one catalog partition, and the files they came from.

    Files are read one at a time and concatenated with ``diagonal_relaxed`` for the same
    reason ``Catalog.compact_partition`` does: older files can lack columns newer ones have.
    """
    frames, sources = [], []
    for path in sorted(directory.glob("*.parquet")):
        frame = pl.read_parquet(path, hive_partitioning=False)
        if not {"model_group_id", "scenario_id"} <= set(frame.columns):
            continue
        frame = frame.filter(_run_filter(model_group_id, scenario_id))
        if frame.height:
            frames.append(frame)
            sources.append(path)
    if not frames:
        return None, []
    return pl.concat(frames, how="diagonal_relaxed"), sources


def inventory_run(base_path, catalog_path, model_group_id, scenario_id) -> RunInventory:
    """Find every file and row belonging to one run. Reads only; changes nothing.

    Datasets are taken from both the data tree and the catalog, so a run whose files are
    present but whose catalog rows are not (or the reverse) is still found, and the
    mismatch is reported in ``warnings``.
    """
    base_path, catalog_path = Path(base_path), Path(catalog_path)
    model_group_id, scenario_id = _cast_run(model_group_id, scenario_id)
    inventory = RunInventory(model_group_id, scenario_id)

    data_datasets = _datasets_under(base_path)
    catalog_datasets = _datasets_under(catalog_path) if catalog_path.is_dir() else {}

    # LEGACY COG SHIM (see raster._LEGACY_DATASET_DIMENSION_ORDER): pre-migration COGs sit
    # under variable/grid/time_unit. Nothing here reads that layout, so say so rather than
    # silently leaving those files out of the archive.
    if any(path.is_dir() for path in base_path.glob("variable=*/grid=*")):
        inventory.warnings.append("found legacy variable=/grid=/time_unit= directories; files there are not backed up")

    for dataset in sorted(data_datasets.keys() | catalog_datasets.keys()):
        has_data = False

        run_dir = _run_dir(base_path, dataset, model_group_id, scenario_id)
        if run_dir.is_dir():
            for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
                inventory.members.append(
                    Member(
                        arcname=f"{DATA_PREFIX}/{path.relative_to(base_path).as_posix()}",
                        kind=_kind(path),
                        method="copy",
                        dataset=dataset,
                        sources=[path],
                        size=path.stat().st_size,
                    )
                )
                has_data = True

        counts = {
            path: _count_run_rows([path], model_group_id, scenario_id)
            for path in _mixed_parquet_files(base_path / dataset)
        }
        if sources := [path for path, rows in counts.items() if rows]:
            inventory.members.append(
                Member(
                    arcname=f"{DATA_PREFIX}/{dataset}/{uuid.uuid4()}-0.parquet",
                    kind="parquet",
                    method="extract",
                    dataset=dataset,
                    sources=sources,
                    rows=sum(counts.values()),
                )
            )
            has_data = True

        catalog_frame, catalog_sources = _catalog_rows(catalog_path / dataset, model_group_id, scenario_id)
        if catalog_frame is not None:
            inventory.members.append(
                Member(
                    arcname=f"{CATALOG_PREFIX}/{dataset}/{uuid.uuid4()}-0.parquet",
                    kind="catalog",
                    method="extract",
                    dataset=dataset,
                    sources=catalog_sources,
                    rows=catalog_frame.height,
                )
            )

        if has_data and catalog_frame is None:
            inventory.warnings.append(f"{dataset}: data files but no catalog rows")
        elif catalog_frame is not None and not has_data:
            inventory.warnings.append(f"{dataset}: catalog rows but no data files")

    return inventory


def compact_run(base_path, catalog_path, model_group_id, scenario_id, datasets, echo=print) -> None:
    """Compact the run's data and catalog files in ``datasets``, skipping what is already
    compact.

    Every compactor called here already leaves compact data alone: ``compact_cog`` skips a
    calendar run living in one optimal file, ``compact_parquet`` skips a directory with one
    file, and ``Catalog.compact_partition`` skips a partition with one file.

    COGs and daily Parquet are compacted for this run only. Annual/decadal Parquet holds
    every run in one directory, so that directory is compacted as a whole -- which rewrites
    the other runs' rows too, and reads the whole dataset into memory to do it.

    The data compactors are called with ``catalog_path=None``: merging files changes no
    dimension combination, so the rows already in the catalog stay correct, and the
    catalog partition is compacted separately afterwards.
    """
    base_path, catalog_path = Path(base_path), Path(catalog_path)
    model_group_id, scenario_id = _cast_run(model_group_id, scenario_id)

    for dataset in datasets:
        dims = parse_path_dims(dataset)
        grid = GRIDS.get(dims["grid"])
        if grid is None:
            echo(f"  {dataset}: unknown grid, not compacted")
            continue

        run_kwargs = {**dims, "model_group_id": model_group_id, "scenario_id": scenario_id}
        if _run_dir(base_path, dataset, model_group_id, scenario_id).is_dir():
            compactor = compact_cog if grid.grid_type == "raster" else compact_parquet
            result = compactor(base_path=base_path, catalog_path=None, **run_kwargs)
            if result["sources_merged"]:
                echo(f"  {dataset}: merged {result['sources_merged']} files into {len(result['outputs'])}")

        # Never pass the run's dimensions here: this directory has no partition
        # directories, so compact_parquet would read back only the run's rows and then
        # delete every source file, taking every other run's rows with them.
        if len(_mixed_parquet_files(base_path / dataset)) > 1:
            result = compact_parquet(base_path=base_path, catalog_path=None, **dims)
            echo(f"  {dataset}: merged {result['sources_merged']} files into {len(result['outputs'])} (all runs)")

        catalog_dir = catalog_path / dataset
        n_catalog = len(list(catalog_dir.glob("*.parquet"))) if catalog_dir.is_dir() else 0
        if n_catalog > 1:
            Catalog(catalog_path).compact_partition(dims["variable"], dims["time_unit"], dims["grid"])
            echo(f"  {dataset}: compacted {n_catalog} catalog files")


class _HashingReader:
    """A read-only file wrapper that hashes everything read through it."""

    def __init__(self, handle):
        self.handle = handle
        self.sha256 = hashlib.sha256()

    def read(self, size=-1):
        chunk = self.handle.read(size)
        self.sha256.update(chunk)
        return chunk


def _add_file(tar: tarfile.TarFile, path: Path, arcname: str) -> str:
    """Add ``path`` to ``tar`` and return its SHA-256, reading the file only once."""
    info = tar.gettarinfo(str(path), arcname=arcname)
    with path.open("rb") as handle:
        reader = _HashingReader(handle)
        tar.addfile(info, fileobj=reader)
    return reader.sha256.hexdigest()


def _package_version() -> str:
    try:
        return version("cpra.mp.data")
    except PackageNotFoundError:
        return "unknown"


def write_archive(inventory: RunInventory, output_dir, base_path, catalog_path, *, compacted=False, echo=print) -> Path:
    """Write ``inventory`` to ``<output_dir>/g<id>-s<id>-<UTC timestamp>.tar``.

    Written under a ``.partial`` name and renamed into place only once complete, so an
    interrupted backup never leaves something that looks like a finished archive.
    Extracted Parquet is staged in a temporary directory inside ``output_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC)
    final = output_dir / f"{run_name(inventory.model_group_id, inventory.scenario_id)}-{timestamp:%Y%m%dT%H%M%SZ}.tar"
    partial = final.with_name(final.name + ".partial")
    run = _run_filter(inventory.model_group_id, inventory.scenario_id)

    files = []
    try:
        with (
            tempfile.TemporaryDirectory(dir=output_dir, prefix=".backup-") as staging,
            tarfile.open(partial, "w") as tar,
        ):
            for member in inventory.members:
                if member.method == "copy":
                    path = member.sources[0]
                elif member.kind == "catalog":
                    path = Path(staging) / f"{uuid.uuid4()}.parquet"
                    frame, _sources = _catalog_rows(
                        member.sources[0].parent, inventory.model_group_id, inventory.scenario_id
                    )
                    frame.write_parquet(path)
                else:
                    path = Path(staging) / f"{uuid.uuid4()}.parquet"
                    pl.scan_parquet(member.sources, hive_partitioning=False).filter(run).sink_parquet(
                        path, compression="zstd", compression_level=6, row_group_size=_DEFAULT_ROW_GROUP_SIZE
                    )

                sha256 = _add_file(tar, path, member.arcname)
                files.append(
                    {
                        "path": member.arcname,
                        "kind": member.kind,
                        "method": member.method,
                        "size": path.stat().st_size,
                        "sha256": sha256,
                        "rows": member.rows,
                        "sources": [
                            str(source.relative_to(catalog_path if member.kind == "catalog" else base_path))
                            for source in member.sources
                        ],
                    }
                )
                if member.method == "extract":
                    path.unlink()
                echo(f"  {member.arcname}")

            manifest = {
                "format_version": FORMAT_VERSION,
                "model_group_id": inventory.model_group_id,
                "scenario_id": inventory.scenario_id,
                "created": timestamp.isoformat(),
                "cpra_mp_data_version": _package_version(),
                "base_path": str(base_path),
                "catalog_path": str(catalog_path),
                "compacted": compacted,
                "datasets": inventory.datasets,
                "warnings": inventory.warnings,
                "files": files,
            }
            payload = json.dumps(manifest, indent=2).encode()
            info = tarfile.TarInfo(MANIFEST_NAME)
            info.size = len(payload)
            info.mtime = int(timestamp.timestamp())
            info.mode = 0o644
            tar.addfile(info, fileobj=_BytesReader(payload))
        partial.rename(final)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return final


class _BytesReader:
    def __init__(self, payload: bytes):
        self.payload = memoryview(payload)
        self.offset = 0

    def read(self, size=-1):
        end = len(self.payload) if size < 0 else self.offset + size
        chunk = bytes(self.payload[self.offset : end])
        self.offset += len(chunk)
        return chunk


class ArchiveError(Exception):
    """An archive is malformed, corrupt, or can't be restored where asked."""


def _safe_parts(name: str) -> tuple[str, ...]:
    """The member's path parts, refusing anything that could land outside the target."""
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) < 2
        or path.parts[0] not in (DATA_PREFIX, CATALOG_PREFIX)
    ):
        raise ArchiveError(f"unexpected member path in archive: {name}")
    return path.parts


def read_manifest(archive) -> dict:
    with tarfile.open(archive, "r") as tar:
        return _read_manifest(tar)


def _read_manifest(tar: tarfile.TarFile) -> dict:
    try:
        handle = tar.extractfile(MANIFEST_NAME)
    except KeyError:
        raise ArchiveError(f"{MANIFEST_NAME} missing; not a run backup, or the backup is incomplete") from None
    manifest = json.load(handle)
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ArchiveError(f"unsupported archive format version: {manifest.get('format_version')!r}")
    return manifest


def restore_archive(archive, base_path, catalog_path, *, dry_run=False, echo=print) -> dict:
    """Restore a run archive into ``base_path``/``catalog_path``.

    Refuses if any of the run is still present, since its rows would then be duplicated.
    Every member is checksummed against the manifest before anything is put in place:
    members are extracted into a staging directory inside each target tree first (so the
    final moves are same-filesystem renames), and only once all of them verify are they
    moved into place. A failed or corrupt restore therefore leaves the live trees as they
    were.

    With ``dry_run`` the archive is read and verified the same way, but nothing is written.

    Returns the manifest.
    """
    base_path, catalog_path = Path(base_path), Path(catalog_path)

    with tarfile.open(archive, "r") as tar:
        manifest = _read_manifest(tar)
        model_group_id, scenario_id = manifest["model_group_id"], manifest["scenario_id"]

        expected = {entry["path"]: entry for entry in manifest["files"]}
        members = [member for member in tar.getmembers() if member.name != MANIFEST_NAME]
        for member in members:
            _safe_parts(member.name)
            if not member.isfile():
                raise ArchiveError(f"not a regular file: {member.name}")
            if member.name not in expected:
                raise ArchiveError(f"member not listed in the manifest: {member.name}")
        missing = expected.keys() - {member.name for member in members}
        if missing:
            raise ArchiveError(f"manifest lists {len(missing)} file(s) the archive doesn't hold, e.g. {min(missing)}")

        present = inventory_run(base_path, catalog_path, model_group_id, scenario_id)
        if present.members:
            datasets = ", ".join(present.datasets)
            raise ArchiveError(
                f"{run_name(model_group_id, scenario_id)} is already present in {len(present.members)} file(s) "
                f"({datasets}); remove it before restoring, or its rows will be duplicated"
            )

        roots = {DATA_PREFIX: base_path, CATALOG_PREFIX: catalog_path}
        staging = {}
        try:
            if not dry_run:
                for prefix, root in roots.items():
                    root.mkdir(parents=True, exist_ok=True)
                    staging[prefix] = Path(tempfile.mkdtemp(dir=root, prefix=".restore-"))

            staged = []
            for member in members:
                parts = _safe_parts(member.name)
                sha256 = hashlib.sha256()
                source = tar.extractfile(member)
                if dry_run:
                    while chunk := source.read(_CHUNK_SIZE):
                        sha256.update(chunk)
                else:
                    target = staging[parts[0]].joinpath(*parts[1:])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("wb") as handle:
                        while chunk := source.read(_CHUNK_SIZE):
                            sha256.update(chunk)
                            handle.write(chunk)
                    staged.append((target, roots[parts[0]].joinpath(*parts[1:])))
                if sha256.hexdigest() != expected[member.name]["sha256"]:
                    raise ArchiveError(f"checksum mismatch: {member.name}")

            if dry_run:
                echo(f"verified {len(members)} file(s); nothing written (--dry-run)")
                return manifest

            for target, destination in staged:
                if destination.exists():
                    raise ArchiveError(f"refusing to overwrite {destination}")
            for target, destination in staged:
                destination.parent.mkdir(parents=True, exist_ok=True)
                target.rename(destination)
                echo(f"  {destination}")
        finally:
            for directory in staging.values():
                shutil.rmtree(directory, ignore_errors=True)

    return manifest
