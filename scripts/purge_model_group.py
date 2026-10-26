"""Find (and, only when told to, delete) every migrated file carrying a model_group_id.

Written for one situation: ``migrate_to_long_format.py`` was run while a model group was
still being written, so everything the migration produced for that group is incomplete.
This script locates that output so it can be removed and the migration re-run against the
finished source data.

It iterates the datasets recorded in the catalog side-table (``--catalog-path``), scans
each one's directory in the migrated tree (``--data-path``), and classifies every Parquet
and COG it finds by whether that model group is in it.

\b
**Nothing is deleted without ``--execute``.**  The default run only reports and, with
``--manifest``, writes a CSV of what it would delete.

How the model group is carried differs by dataset, and that decides how much collateral
a deletion has:

\b
* **COGs** always encode every partition dimension in their directory path (see
  ``raster.cog_path_expression``), so a matching COG holds exactly one model group.
  Deleting it costs nothing else.  ``model_group_id`` is also a dataset tag
  (``write_cog``), used here as a fallback for any COG whose path lacks it.
* **Daily Parquet** is Hive-partitioned by ``model_group_id`` (``Dimension.partition``
  only partitions when ``time_unit == "daily"``), so the same holds: one group per file.
* **Every other Parquet** (annual/decadal/monthly) keeps ``model_group_id`` as a *column*,
  and ``migrate_partition`` writes one output file per source partition directory -- which
  for these time units is the whole dataset directory.  One file therefore holds every
  model group in the dataset (97 of them, for ``sal``/``annual``/``hydro_compartment_v001``).
  These are reported separately as MIXED: deleting one is still correct -- the incomplete
  group cannot be excised from it any other way -- but re-running the migration then
  regenerates every group that file held, not just the bad one.  ``--skip-mixed`` leaves
  them alone if that trade needs thinking about first.

\b
**The catalog must be pruned too, or the re-run does nothing.**  ``migrate_to_long_format``
decides what to skip from the catalog (``_catalog_completed``), not from the data tree, so
a partition whose files are gone but whose catalog rows remain is still "already converted"
and is skipped.  ``--prune-catalog`` removes the matching rows.  Catalog files mix model
groups (one row per dimension combination), so pruning rewrites them in place rather than
deleting them, and only drops a file when no rows survive.

Note that dimension filtering is deliberately limited to ``model_group_id``.  Narrowing
further (a single scenario, say) would leave a partially-purged partition behind, which is
the state this script exists to get out of.

Usage:

\b
    # Report what carries model group 690 -- writes nothing
    python scripts/purge_model_group.py --model-group-id 690 \\
        --data-path /path/to/long-format --catalog-path /path/to/catalog

\b
    # Same, but record the file list for review before acting on it
    python scripts/purge_model_group.py --model-group-id 690 \\
        --data-path /path/to/long-format --catalog-path /path/to/catalog \\
        --manifest purge-690.csv

\b
    # Delete for real, and prune the catalog rows so the re-run picks the work back up
    python scripts/purge_model_group.py --model-group-id 690 \\
        --data-path /path/to/long-format --catalog-path /path/to/catalog \\
        --manifest purge-690.csv --execute --prune-catalog

Then re-run the migration for the affected datasets as usual.
"""

import csv
import sys
import uuid
from pathlib import Path

import click
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

# Add the project root to sys.path so the package is importable when the
# script is run from any working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpra.mp.data.catalog import parse_path_dims
from cpra.mp.data.config import BASE_PATH, CATALOG_PATH
from cpra.mp.data.dimensions import DIMENSIONS

DIMENSION = "model_group_id"
DTYPE = DIMENSIONS[DIMENSION].dtype

# Classification of one matched file -- what deleting it would actually cost.
#   "exclusive"  the file holds only the requested model group(s); nothing else goes with it
#   "mixed"      the file also holds other model groups, which the re-run must regenerate
EXCLUSIVE = "exclusive"
MIXED = "mixed"


def catalog_datasets(catalog_path: Path, variable=None, time_unit=None, grid=None) -> list[dict]:
    """The ``variable``/``time_unit``/``grid`` datasets the catalog knows about.

    The catalog mirrors the data tree's Hive partitioning (``catalog.PART_COLS``), so its
    directory layout alone enumerates the datasets -- no need to read any of its files.
    This is what makes the catalog, rather than a walk of the data tree, the list of
    datasets to visit: it is the same source ``migrate_to_long_format`` consults to decide
    what has already been converted.
    """
    parts = [
        f"variable={variable or '*'}",
        f"time_unit={time_unit or '*'}",
        f"grid={grid or '*'}",
    ]
    datasets = {}
    for directory in catalog_path.glob("/".join(parts)):
        if not directory.is_dir():
            continue
        dims = parse_path_dims(directory.relative_to(catalog_path))
        if {"variable", "time_unit", "grid"} <= dims.keys():
            datasets[(dims["variable"], dims["time_unit"], dims["grid"])] = dims
    return [datasets[key] for key in sorted(datasets)]


def dataset_dirs(base_path: Path, dims: dict) -> list[Path]:
    """The on-disk directories one dataset's files can live under.

    Both the canonical (variable/time_unit/grid) and the legacy (variable/grid/time_unit)
    dataset-directory orders are returned -- ``raster.py``'s LEGACY COG SHIM means COGs
    written before the migration sit under the latter, and ``compact_cog`` pools both.
    Migrated output only ever uses the canonical order, but scanning both costs nothing
    and keeps this correct if it is ever pointed at a pre-migration tree.
    """
    variable, time_unit, grid = dims["variable"], dims["time_unit"], dims["grid"]
    candidates = [
        base_path / f"variable={variable}" / f"time_unit={time_unit}" / f"grid={grid}",
        base_path / f"variable={variable}" / f"grid={grid}" / f"time_unit={time_unit}",
    ]
    return [directory for directory in candidates if directory.is_dir()]


def _path_model_group(path: Path, base_path: Path):
    """The model group encoded in ``path``'s Hive directories, or None if it isn't there."""
    value = parse_path_dims(path.relative_to(base_path)).get(DIMENSION)
    return None if value is None else pl.Series([value], dtype=pl.String).cast(DTYPE, strict=False)[0]


def _parquet_may_contain(path: Path, targets: set) -> bool:
    """Whether ``path``'s row-group statistics leave room for any of ``targets``.

    A cheap superset test read from the Parquet footer, so files that cannot hold the
    model group are rejected without touching a single data page.  It is worth doing:
    ``write_parquet`` sorts model_group_id-first, so each row group covers one group (or
    very few) and the min/max bounds are about as tight as they get.

    Returns True whenever the answer isn't certain -- absent statistics, an unreadable
    footer, a file with no row groups -- leaving the decision to a real read.
    """
    try:
        metadata = pq.read_metadata(path)
    except Exception:
        return True

    names = metadata.schema.names
    if DIMENSION not in names:
        return False
    index = names.index(DIMENSION)

    for group in range(metadata.num_row_groups):
        statistics = metadata.row_group(group).column(index).statistics
        if statistics is None or not statistics.has_min_max:
            return True
        if any(statistics.min <= target <= statistics.max for target in targets):
            return True
    return metadata.num_row_groups == 0


def _parquet_model_groups(path: Path) -> set | None:
    """Every model group present in ``path``, or None if it has no such column.

    ``hive_partitioning=False`` so the answer describes the file's own contents rather
    than being back-filled from the directories above it -- the path is handled separately
    and first, by ``_path_model_group``.
    """
    lf = pl.scan_parquet(path, hive_partitioning=False)
    if DIMENSION not in lf.collect_schema().names():
        return None
    return set(lf.select(pl.col(DIMENSION).unique()).collect()[DIMENSION].drop_nulls().to_list())


def _cog_model_group(path: Path):
    """The model group from a COG's dataset tags, or None if it carries no such tag.

    ``write_cog`` tags every non-calendar dimension it is given, so this is a real second
    source for a COG whose path somehow lacks the directory.  Only consulted in that case:
    opening every COG in a tree to read a value already in its path would be pure cost.
    """
    import rasterio as rio

    try:
        with rio.open(path) as dataset:
            value = dataset.tags().get(DIMENSION)
    except Exception:
        return None
    return None if value is None else pl.Series([value], dtype=pl.String).cast(DTYPE, strict=False)[0]


def classify(path: Path, base_path: Path, targets: set) -> dict | None:
    """Classify one data file, or return None if it holds none of ``targets``.

    The returned record carries ``kind`` (parquet/cog), ``classification``
    (``EXCLUSIVE``/``MIXED``), ``source`` (where the model group was read from), and
    ``present``/``others`` -- the model groups in the file, split by whether they were
    asked for.  ``others`` is what a deletion takes with it.
    """
    from_path = _path_model_group(path, base_path)
    kind = "cog" if path.suffix == ".tif" else "parquet"

    if from_path is not None:
        if from_path not in targets:
            return None
        # A partition dimension in the path is the whole file's value, by construction.
        return {
            "path": path,
            "kind": kind,
            "classification": EXCLUSIVE,
            "source": "path",
            "present": {from_path},
            "others": set(),
        }

    if kind == "cog":
        from_tags = _cog_model_group(path)
        if from_tags is None or from_tags not in targets:
            return None
        return {
            "path": path,
            "kind": kind,
            "classification": EXCLUSIVE,
            "source": "tags",
            "present": {from_tags},
            "others": set(),
        }

    if not _parquet_may_contain(path, targets):
        return None
    contents = _parquet_model_groups(path)
    if not contents:
        return None
    present = contents & targets
    if not present:
        return None
    others = contents - targets
    return {
        "path": path,
        "kind": kind,
        "classification": MIXED if others else EXCLUSIVE,
        "source": "column",
        "present": present,
        "others": others,
    }


def find_matches(base_path: Path, datasets: list[dict], targets: set, progress: bool = True) -> list[dict]:
    """Every file under ``datasets`` that carries one of ``targets``.

    Each record gains ``dataset`` (the canonical ``variable=.../time_unit=.../grid=...``
    key) and ``size_bytes``.  A dataset reachable under both the canonical and legacy
    directory orders is deduplicated by resolved path, but only then -- ``resolve()`` is a
    syscall per file, and a migrated tree only ever uses the canonical order, so paying it
    for every file in a multi-million-file tree would be for nothing.
    """
    matches: list[dict] = []

    for dims in tqdm(datasets, desc="Scanning datasets", disable=not progress, unit="dataset"):
        dataset = "/".join(f"{name}={dims[name]}" for name in ("variable", "time_unit", "grid"))
        directories = dataset_dirs(base_path, dims)
        seen: set[Path] | None = set() if len(directories) > 1 else None
        for directory in directories:
            for pattern in ("*.parquet", "*.tif"):
                for path in sorted(directory.rglob(pattern)):
                    if not path.is_file():
                        continue
                    if seen is not None:
                        resolved = path.resolve()
                        if resolved in seen:
                            continue
                        seen.add(resolved)
                    match = classify(path, base_path, targets)
                    if match is not None:
                        match["dataset"] = dataset
                        match["size_bytes"] = path.stat().st_size
                        matches.append(match)
    return matches


def catalog_matches(catalog_path: Path, datasets: list[dict], targets: set) -> list[dict]:
    """Catalog files holding rows for ``targets``, with the row counts a prune would change.

    A catalog file is one row per dimension combination per contiguous calendar run, so it
    mixes model groups freely -- ``kept`` is what would survive, and is what decides
    between rewriting the file and removing it.
    """
    results = []
    for dims in datasets:
        directory = (
            catalog_path / f"variable={dims['variable']}" / f"time_unit={dims['time_unit']}" / f"grid={dims['grid']}"
        )
        dataset = "/".join(f"{name}={dims[name]}" for name in ("variable", "time_unit", "grid"))
        for path in sorted(directory.glob("*.parquet")):
            frame = pl.read_parquet(path)
            if DIMENSION not in frame.columns:
                continue
            # A plain list, not a Series: `is_in` against a Series of the same dtype is
            # deprecated in polars as ambiguous with the implode/list-column meaning.
            selected = pl.col(DIMENSION).is_in(sorted(targets))
            matched = frame.filter(selected).height
            if matched:
                results.append(
                    {
                        "path": path,
                        "dataset": dataset,
                        "matched": matched,
                        "kept": frame.height - matched,
                    }
                )
    return results


def delete_data(paths, *, execute: bool = False) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Delete the given data files, returning ``(deleted, failures)``.

    A local stand-in: ``cpra.mp.data`` has no ``delete_data`` -- its write-side surface is
    ``write_data``/``compact_data`` -- so there is nothing in the package to call.  The
    signature is kept deliberately narrow (a list of already-resolved paths) so that if a
    real ``delete_data`` lands later, swapping it in is a one-line change here.

    Without ``execute`` this does nothing at all and reports nothing deleted; the caller
    decides when a run is for real.
    """
    deleted: list[Path] = []
    failures: list[tuple[Path, str]] = []
    if not execute:
        return deleted, failures
    for path in paths:
        try:
            Path(path).unlink()
            deleted.append(Path(path))
        except OSError as exc:
            failures.append((Path(path), str(exc)))
    return deleted, failures


def prune_empty_dirs(paths, base_path: Path) -> list[Path]:
    """Remove directories emptied by a deletion, walking up but never past ``base_path``.

    A leftover empty ``model_group_id=.../scenario_id=...`` directory is harmless to the
    migration (its discovery globs for files, not directories), so this is tidiness rather
    than correctness -- but it keeps a purged tree from looking like it still holds the
    group.
    """
    removed = []
    for directory in sorted({Path(p).parent for p in paths}, key=lambda d: len(d.parts), reverse=True):
        while directory != base_path and base_path in directory.parents:
            try:
                directory.rmdir()
            except OSError:  # not empty, or gone already
                break
            removed.append(directory)
            directory = directory.parent
    return removed


def prune_catalog(entries: list[dict], targets: set, *, execute: bool = False) -> tuple[int, int, list]:
    """Drop ``targets``' rows from the catalog files in ``entries``.

    Returns ``(rewritten, removed, failures)``.  A file with rows left over is rewritten
    (written beside itself and renamed over, so an interrupted prune leaves the original
    intact); a file with none left is removed outright, matching what ``Catalog.append``
    would have produced had the group never been written.

    Without ``execute`` this changes nothing and reports zeroes.
    """
    rewritten = removed = 0
    failures = []
    if not execute:
        return rewritten, removed, failures

    for entry in entries:
        path = entry["path"]
        try:
            if entry["kept"] == 0:
                path.unlink()
                removed += 1
                continue
            frame = pl.read_parquet(path)
            keep = pl.col(DIMENSION).is_in(sorted(targets)).not_()
            temporary = path.parent / f".{uuid.uuid4()}.parquet.tmp"
            frame.filter(keep).write_parquet(temporary)
            temporary.replace(path)
            rewritten += 1
        except Exception as exc:
            failures.append((path, str(exc)))
    return rewritten, removed, failures


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _fmt_groups(values) -> str:
    return " ".join(str(v) for v in sorted(values))


def write_manifest(path: Path, matches: list[dict], selected: set) -> None:
    """Write every match to a CSV, including the ones this run would not delete.

    ``action`` is ``delete`` or ``skip``, so the manifest doubles as the record of what a
    ``--execute`` run with these same flags would do.
    """
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "action",
                "kind",
                "classification",
                "source",
                "dataset",
                "path",
                "model_groups_matched",
                "model_groups_collateral",
                "size_bytes",
            ]
        )
        for match in matches:
            writer.writerow(
                [
                    "delete" if match["path"] in selected else "skip",
                    match["kind"],
                    match["classification"],
                    match["source"],
                    match["dataset"],
                    str(match["path"]),
                    _fmt_groups(match["present"]),
                    _fmt_groups(match["others"]),
                    match["size_bytes"],
                ]
            )


def report(matches: list[dict], selected: set) -> None:
    """Print the findings: the clean deletions, then the mixed files and their collateral."""
    exclusive = [m for m in matches if m["classification"] == EXCLUSIVE]
    mixed = [m for m in matches if m["classification"] == MIXED]

    print(
        f"\nFound {len(matches)} file(s) carrying the requested model group(s), "
        f"{_fmt_bytes(sum(m['size_bytes'] for m in matches))} total."
    )

    if exclusive:
        print(
            f"\n  {len(exclusive)} file(s) hold only the requested group(s) "
            f"({_fmt_bytes(sum(m['size_bytes'] for m in exclusive))}):"
        )
        by_dataset: dict[str, list[dict]] = {}
        for match in exclusive:
            by_dataset.setdefault(match["dataset"], []).append(match)
        for dataset in sorted(by_dataset):
            group = by_dataset[dataset]
            kinds = sorted({m["kind"] for m in group})
            print(
                f"    {dataset}: {len(group)} {'/'.join(kinds)} file(s), "
                f"{_fmt_bytes(sum(m['size_bytes'] for m in group))}"
            )

    if mixed:
        print(
            f"\n  {len(mixed)} file(s) ALSO hold other model groups "
            f"({_fmt_bytes(sum(m['size_bytes'] for m in mixed))})."
        )
        print("    The requested group cannot be removed from these without removing the")
        print("    file, so re-running the migration regenerates every group listed below.")
        for match in sorted(mixed, key=lambda m: m["dataset"]):
            action = "DELETE" if match["path"] in selected else "skip"
            print(f"    [{action}] {match['dataset']}  ({_fmt_bytes(match['size_bytes'])})")
            print(f"        {match['path'].name}")
            print(f"        also regenerated: {len(match['others'])} group(s): {_fmt_groups(match['others'])}")


@click.command(
    help=__doc__,
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)
@click.option(
    "--model-group-id",
    "model_group_ids",
    type=int,
    multiple=True,
    required=True,
    help="Model group to purge. Repeat the flag to purge more than one.",
)
@click.option(
    "--data-path", default=str(BASE_PATH), show_default=True, help="Migrated (long-format) data tree to scan."
)
@click.option(
    "--catalog-path",
    metavar="DIR",
    default=None,
    help="Root of the unified catalog side-table. Its partitions are the list of datasets to "
    "visit, and with --prune-catalog its rows for the model group are removed. Defaults to "
    "CATALOG_PATH (the CPRA_MP_DATA_CATALOG_PATH environment variable).",
)
@click.option("--variable", help="Restrict to this variable.")
@click.option("--grid", help="Restrict to this grid.")
@click.option("--time-unit", help="Restrict to this time unit.")
@click.option(
    "--skip-mixed",
    is_flag=True,
    help="Leave Parquet files that also hold other model groups alone. They cannot be purged "
    "without being deleted whole, so this leaves the model group's data in place -- use it to "
    "take the unambiguous deletions first and decide about the rest separately.",
)
@click.option(
    "--manifest",
    metavar="PATH",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write every match to this CSV, each marked delete or skip. Written on dry runs too.",
)
@click.option(
    "--execute",
    is_flag=True,
    help="Actually delete. Without this flag the script only reports and writes --manifest.",
)
@click.option(
    # Named `prune_catalog_rows` rather than the default `prune_catalog` so the flag does
    # not shadow the `prune_catalog` function inside main.
    "--prune-catalog",
    "prune_catalog_rows",
    is_flag=True,
    help="Also remove the model group's rows from the catalog. REQUIRED for a re-run to redo "
    "this work: migrate_to_long_format skips partitions the catalog shows as converted, so "
    "deleted files whose catalog rows survive are simply never rebuilt. Needs --execute.",
)
@click.option(
    # Named `prune_empty` for the same reason as `prune_catalog_rows` above -- the default
    # name would shadow the `prune_empty_dirs` function inside main.
    "--prune-empty-dirs/--no-prune-empty-dirs",
    "prune_empty",
    default=True,
    help="Remove directories left empty by the deletion.",
)
@click.option("--progress/--no-progress", default=True, help="Show a progress bar (use --no-progress for batch jobs).")
def main(
    model_group_ids,
    data_path,
    catalog_path,
    variable,
    grid,
    time_unit,
    skip_mixed,
    manifest,
    execute,
    prune_catalog_rows,
    prune_empty,
    progress,
):
    catalog_path = Path(catalog_path) if catalog_path else CATALOG_PATH
    if not catalog_path:
        raise click.UsageError("--catalog-path is required (CATALOG_PATH is not set)")
    catalog_path = Path(catalog_path).resolve()
    if not catalog_path.is_dir():
        raise click.UsageError(f"--catalog-path does not exist: {catalog_path}")

    base_path = Path(data_path).resolve()
    if not base_path.is_dir():
        raise click.UsageError(f"--data-path does not exist: {base_path}")

    if prune_catalog_rows and not execute:
        raise click.UsageError("--prune-catalog needs --execute (it rewrites catalog files)")

    targets = {DIMENSIONS[DIMENSION].cast(value) for value in model_group_ids}

    datasets = catalog_datasets(catalog_path, variable, time_unit, grid)
    if not datasets:
        print("No datasets in the catalog match the given filters.")
        return

    print(
        f"{'DELETING' if execute else 'DRY RUN -- nothing will be deleted'}: {DIMENSION} in {{{_fmt_groups(targets)}}}"
    )
    print(f"  data:    {base_path}")
    print(f"  catalog: {catalog_path}")
    print(f"  {len(datasets)} dataset(s) to scan.")

    matches = find_matches(base_path, datasets, targets, progress=progress)
    selected = {m["path"] for m in matches if not (skip_mixed and m["classification"] == MIXED)}

    # Checked even when no data file matched: a tree whose files were already removed by
    # hand still has catalog rows saying otherwise, and those rows alone are enough to make
    # migrate_to_long_format skip the partition. Reporting "nothing to do" there would be
    # wrong in exactly the case that is hardest to notice.
    catalog_entries = catalog_matches(catalog_path, datasets, targets)

    if not matches:
        print("\nNo files carry the requested model group(s).")
    else:
        report(matches, selected)
        if manifest:
            write_manifest(manifest, matches, selected)
            print(f"\nManifest written: {manifest}")

    if catalog_entries:
        rows = sum(entry["matched"] for entry in catalog_entries)
        print(f"\nCatalog: {rows} row(s) for the requested model group(s) across {len(catalog_entries)} file(s).")
    elif not matches:
        print("Catalog has no rows for them either. Nothing to do.")
        return

    if not execute:
        print(
            f"\nWould delete {len(selected)} file(s), "
            f"{_fmt_bytes(sum(m['size_bytes'] for m in matches if m['path'] in selected))}."
        )
        if skip_mixed:
            held = [m for m in matches if m["path"] not in selected]
            if held:
                print(
                    f"Would leave {len(held)} mixed file(s) in place (--skip-mixed), so the "
                    "model group's data would still be present in them."
                )
        if catalog_entries and not prune_catalog_rows:
            print("\nNOTE: without --prune-catalog the catalog still lists this model group as")
            print("      converted, and migrate_to_long_format will skip those partitions --")
            print("      the deleted files would never be rebuilt.")
        print("\nRe-run with --execute (and --prune-catalog) to apply.")
        return

    deleted, failures = delete_data(sorted(selected), execute=True)
    print(f"\nDeleted {len(deleted)} file(s).")
    for path, error in failures:
        print(f"  FAILED: {path}: {error}")

    if prune_empty and deleted:
        removed = prune_empty_dirs(deleted, base_path)
        if removed:
            print(f"Removed {len(removed)} empty directory/directories.")

    if prune_catalog_rows and catalog_entries:
        rewritten, removed_files, catalog_failures = prune_catalog(catalog_entries, targets, execute=True)
        print(f"Catalog: rewrote {rewritten} file(s), removed {removed_files} emptied file(s).")
        for path, error in catalog_failures:
            print(f"  FAILED: {path}: {error}")
    elif catalog_entries:
        print("\nNOTE: catalog rows for this model group were left in place (--prune-catalog not")
        print("      given). migrate_to_long_format will skip these partitions as already")
        print("      converted, so the deleted files will not be rebuilt.")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
