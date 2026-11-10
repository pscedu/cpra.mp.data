"""Compact the data tree and the catalog in place: the logic behind ``data compact`` and
``backup --compact``.

Each dataset (a ``variable``/``time_unit``/``grid`` directory) is handed to the compactor
for its kind -- ``raster.compact_cog`` for raster grids, ``tabular.compact_parquet`` for
vector ones -- and then its catalog partition to ``Catalog.compact_partition``. All three
leave what is already compact alone and stage new files out of readers' sight (see
``cpra.mp.data.staging``), so compacting a tree twice is cheap and safe.

Narrowed to one run (``model_group_id`` + ``scenario_id``), COGs and daily Parquet are
compacted for that run only, since they keep each run in its own directory. Annual/decadal
Parquet keeps every run in one directory, so that directory is compacted whole, other runs
included. It streams the files (see ``cpra.mp.data.merge``), so memory stays bounded
however large the dataset is, but the time grows with it.

The data compactors are called with ``catalog_path=None``: merging files changes no
dimension combination, so the rows already in the catalog stay correct, and the catalog
partition is compacted separately afterwards.
"""

from pathlib import Path

from cpra.mp.data.catalog import Catalog, parse_path_dims
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.raster import compact_cog
from cpra.mp.data.tabular import compact_parquet

DATASET_DIMS = ("variable", "time_unit", "grid")


def dataset_key(dims: dict) -> str:
    return "/".join(f"{name}={dims[name]}" for name in DATASET_DIMS)


def datasets_under(root: Path) -> dict[str, dict]:
    """The ``variable``/``time_unit``/``grid`` dataset directories under ``root``."""
    datasets = {}
    for directory in Path(root).glob("variable=*/time_unit=*/grid=*"):
        if directory.is_dir():
            dims = parse_path_dims(directory.relative_to(root))
            datasets[dataset_key(dims)] = dims
    return datasets


def find_datasets(base_path, catalog_path, **filters) -> list[str]:
    """Every dataset in the data tree or the catalog whose ``variable``/``time_unit``/``grid``
    match ``filters`` (a missing or ``None`` filter matches anything)."""
    found = datasets_under(Path(base_path)) | datasets_under(Path(catalog_path))
    return sorted(
        dataset
        for dataset, dims in found.items()
        if all(filters.get(name) in (None, dims[name]) for name in DATASET_DIMS)
    )


def run_dir(base_path: Path, dataset: str, model_group_id, scenario_id) -> Path:
    return base_path / dataset / f"model_group_id={model_group_id}" / f"scenario_id={scenario_id}"


def mixed_parquet_files(directory: Path) -> list[Path]:
    """The Parquet files sitting directly in a dataset directory -- the unpartitioned kind
    that holds every run in one file."""
    return sorted(path for path in directory.glob("*.parquet") if path.is_file())


def _files_to_merge(directory: Path, suffix: str) -> tuple[int, int]:
    """``(files, directories)`` for the directories under ``directory`` holding more than one
    ``suffix`` file -- what ``compact_parquet`` would merge. Hidden directories (staging)
    are skipped."""
    n_files = n_dirs = 0
    for path in [directory, *directory.rglob("*")]:
        if not path.is_dir() or any(part.startswith(".") for part in path.relative_to(directory).parts):
            continue
        n = sum(1 for f in path.glob(f"*{suffix}") if f.is_file())
        if n > 1:
            n_files, n_dirs = n_files + n, n_dirs + 1
    return n_files, n_dirs


def _compact_data(base_path, dataset, dims, raster, run, dry_run, echo):
    """Compact one dataset's data files, or the run's share of them."""
    if run is None:
        directory, kwargs, scope = base_path / dataset, dims, ""
    else:
        model_group_id, scenario_id = run
        directory = run_dir(base_path, dataset, model_group_id, scenario_id)
        kwargs = {**dims, "model_group_id": model_group_id, "scenario_id": scenario_id}
        scope = " of this run"
    if not directory.is_dir():
        return

    if dry_run:
        if raster:
            # compact_cog decides what needs merging: a calendar run already in one
            # optimal file is left alone.
            n_files = sum(1 for path in directory.rglob("*.tif") if path.is_file())
            echo(f"  {dataset}: would check {n_files} file(s){scope}")
        else:
            n_files, n_dirs = _files_to_merge(directory, ".parquet")
            if n_files:
                echo(f"  {dataset}: would merge {n_files} files in {n_dirs} directory(ies){scope}")
        return

    compactor = compact_cog if raster else compact_parquet
    result = compactor(base_path=base_path, catalog_path=None, **kwargs)
    if result["sources_merged"]:
        echo(f"  {dataset}: merged {result['sources_merged']} files into {len(result['outputs'])}")


def compact(
    base_path,
    catalog_path,
    datasets,
    *,
    model_group_id=None,
    scenario_id=None,
    dry_run=False,
    echo=print,
) -> None:
    """Compact the data and catalog files of ``datasets``, skipping what is already compact.
    With ``model_group_id`` and ``scenario_id``, only that run's data (see the module
    docstring). With ``dry_run``, only report what would be handed to each compactor."""
    if (model_group_id is None) != (scenario_id is None):
        msg = "model_group_id and scenario_id must be given together"
        raise ValueError(msg)
    run = None if model_group_id is None else (int(model_group_id), int(scenario_id))
    base_path, catalog_path = Path(base_path), Path(catalog_path)
    catalog = Catalog(catalog_path)

    for dataset in datasets:
        dims = parse_path_dims(dataset)
        grid = GRIDS.get(dims["grid"])
        if grid is None:
            echo(f"  {dataset}: unknown grid, not compacted")
            continue
        raster = grid.grid_type == "raster"

        _compact_data(base_path, dataset, dims, raster, run, dry_run, echo)

        # A run's annual/decadal rows share files with every other run. Never pass the
        # run's dimensions here: this directory has no partition directories, so
        # compact_parquet would read back only the run's rows and then delete every source
        # file, taking every other run's rows with them.
        n_mixed = len(mixed_parquet_files(base_path / dataset))
        if run is not None and not raster and n_mixed > 1:
            if dry_run:
                echo(f"  {dataset}: would merge {n_mixed} files (all runs)")
            else:
                result = compact_parquet(base_path=base_path, catalog_path=None, **dims)
                echo(f"  {dataset}: merged {result['sources_merged']} files into {len(result['outputs'])} (all runs)")

        catalog_dir = catalog_path / dataset
        n_catalog = len(list(catalog_dir.glob("*.parquet"))) if catalog_dir.is_dir() else 0
        if n_catalog > 1:
            if dry_run:
                echo(f"  {dataset}: would compact {n_catalog} catalog files")
            else:
                catalog.compact_partition(dims["variable"], dims["time_unit"], dims["grid"])
                echo(f"  {dataset}: compacted {n_catalog} catalog files")
