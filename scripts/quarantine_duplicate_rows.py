"""Find wide-format source rows duplicated across files with differing values, and move the stale copies aside.

The situation this exists for: a model run is restarted from a checkpoint, and the restart
re-writes years the first attempt had already written -- as *new* files, beside the old
ones.  If the re-run reproduces those years exactly, ``--unique-parquet-rows`` hides the
duplicates.  If it does not (a run that was not bit-for-bit reproducible), the migration
keeps both values of every differing cell, and ``repair_excess_rows.py`` reports the
partition as over no matter how often it is rebuilt: the check deduplicates wide rows by
key, the conversion deduplicates long rows by value.

For each source partition this script finds keys (the partition's non-calendar dimension
columns plus its calendar column) that occur more than once, and measures the *excess*:
long rows the conversion would write beyond one per key and grid cell -- the same number
``repair_excess_rows.py`` reports as ``over_rows``.  In every partition with excess, the
newest file (by mtime) holding a duplicated key wins, and every older file all of whose rows
are superseded copies is moved to ``--quarantine-path`` (mirroring its path under
``--data-path``).  Superseded copies that are identical are moved too: they are redundant,
and the move keeps one rule for the whole partition.

\b
Safety:
* **Nothing is moved without ``--execute``.**  The default run only reports.
* Files are moved, never deleted; ``--execute`` writes ``quarantine-manifest.csv`` into
  ``--quarantine-path`` listing every move, so it can be undone.
* A file that also holds rows that are not superseded is never moved.  If such a file
  holds a *differing* superseded row, the partition is reported as unresolved.
* So is a duplicated key whose copies sit in the same file, or whose newest copies share an
  mtime -- "newest wins" can't decide those.
* Each partition's excess is recomputed over the files that would remain, and is printed
  beside the current one; a partition is only touched if that comes out to zero.
* Partitions whose duplicates are all identical have no excess and are left alone.

Afterwards, re-run ``repair_excess_rows.py --execute`` to rebuild the affected output.

Usage:

\b
    # Report only
    python scripts/quarantine_duplicate_rows.py \\
        --data-path /path/to/wide --quarantine-path /path/to/wide-quarantine

\b
    # Move the stale copies
    python scripts/quarantine_duplicate_rows.py \\
        --data-path /path/to/wide --quarantine-path /path/to/wide-quarantine --execute
"""

import csv
import datetime
import shutil
import sys
from pathlib import Path

import click
import polars as pl
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import migrate_to_long_format as mig  # noqa: E402
from cpra.mp.data.catalog import cal_for  # noqa: E402
from cpra.mp.data.config import BASE_PATH  # noqa: E402
from cpra.mp.data.dimensions import DIMENSIONS  # noqa: E402

FILE = "__file__"


def key_columns(schema: dict, path: Path) -> list[str]:
    """The columns that identify one wide row: non-calendar dimensions, then the calendar."""
    alpha_cols = set(col for col in schema if not col[:1].isdigit())
    other_dim_cols = [col for col in DIMENSIONS if col in alpha_cols and not col.startswith("calendar_")]
    time_unit = str(mig._dimension_cols_from_hive(path).get("time_unit", ""))
    return [*other_dim_cols, cal_for(time_unit)]


def excess_rows(paths: list[Path], schema: dict, key_cols: list[str], dup_keys: pl.DataFrame) -> int:
    """Long rows the conversion (with ``--unique-parquet-rows``) writes beyond one per key and grid cell.

    Only rows with a duplicated key can contribute, so only those are read and unpivoted.
    """
    if not paths or dup_keys.is_empty():
        return 0
    alpha_cols = [col for col in schema if not col[:1].isdigit()]
    long = (
        pl.scan_parquet(paths)
        .join(dup_keys.lazy(), on=key_cols, how="semi")
        .unpivot(index=alpha_cols, variable_name="grid_cell", value_name="value")
    )
    rows = long.unique().select(pl.len()).collect().item()
    keys = long.select(*key_cols, "grid_cell").unique().select(pl.len()).collect().item()
    return rows - keys


def duplicated_keys(keys: pl.DataFrame, key_cols: list[str]) -> pl.DataFrame:
    return keys.filter(pl.len().over(key_cols) > 1)


def plan_partition(paths: list[Path]) -> dict | None:
    """Work out which files to move for one partition; None if it has no duplicated keys."""
    schema = mig._wide_schema(paths[0])
    if schema is None:
        return None
    key_cols = key_columns(schema, paths[0])
    keys = pl.scan_parquet(paths, include_file_paths=FILE).select(*key_cols, FILE).collect()
    dups = duplicated_keys(keys, key_cols)
    if dups.is_empty():
        return None
    dup_keys = dups.select(key_cols).unique()
    plan = {
        "excess": excess_rows(paths, schema, key_cols, dup_keys),
        "move": [], "unresolved": [], "after": None, "dup_keys": dup_keys.height,
    }
    if not plan["excess"]:
        return plan

    # Per duplicated key, which copies differ, so a stuck file can be judged harmless or not.
    alpha_cols = [col for col in schema if not col[:1].isdigit()]
    differing = (
        pl.scan_parquet(paths)
        .join(dup_keys.lazy(), on=key_cols, how="semi")
        .unpivot(index=alpha_cols, variable_name="grid_cell", value_name="value")
        .unique()
        .group_by(*key_cols, "grid_cell").len()
        .filter(pl.col("len") > 1)
        .select(key_cols).unique()
        .collect()
    )

    mtimes = pl.DataFrame(
        {FILE: list(set(dups[FILE])), "mtime": [Path(f).stat().st_mtime for f in set(dups[FILE])]},
        schema={FILE: pl.String, "mtime": pl.Float64},
    )
    dups = dups.join(mtimes, on=FILE)

    same_file = dups.filter(pl.len().over(*key_cols, FILE) > 1)
    if not same_file.is_empty():
        plan["unresolved"].append(f"{same_file.select(key_cols).unique().height} key(s) duplicated within one file")
    tied = dups.filter(pl.col("mtime") == pl.col("mtime").max().over(key_cols)).filter(
        pl.col(FILE).n_unique().over(key_cols) > 1
    )
    if not tied.is_empty():
        plan["unresolved"].append(f"{tied.select(key_cols).unique().height} key(s) whose newest copies share an mtime")

    losers = dups.filter(pl.col("mtime") < pl.col("mtime").max().over(key_cols))
    file_rows = keys.group_by(FILE).len()
    loser_rows = losers.group_by(FILE).len()
    for file, n_rows, n_lost in (
        loser_rows.join(file_rows, on=FILE, suffix="_file").select(FILE, "len_file", "len").iter_rows()
    ):
        if n_rows == n_lost:
            plan["move"].append(Path(file))
            continue
        stuck = losers.filter(pl.col(FILE) == file).join(differing, on=key_cols, how="semi")
        if not stuck.is_empty():
            plan["unresolved"].append(
                f"{Path(file).name} holds {stuck.height} differing superseded row(s) beside "
                f"{n_rows - n_lost} that are not superseded"
            )
    plan["move"].sort(key=lambda path: path.stat().st_mtime)

    remaining = [path for path in paths if path not in set(plan["move"])]
    remaining_keys = keys.filter(pl.col(FILE).is_in([str(path) for path in remaining]))
    plan["after"] = excess_rows(
        remaining, schema, key_cols, duplicated_keys(remaining_keys, key_cols).select(key_cols).unique()
    )
    if plan["after"]:
        plan["unresolved"].append(f"{plan['after']:,} excess row(s) would remain")
    return plan


@click.command(
    help=__doc__,
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)
@click.option("--data-path", default=str(BASE_PATH), show_default=True, help="Wide-format source tree.")
@click.option(
    "--quarantine-path",
    required=True,
    help="Where superseded files are moved, mirroring their path under --data-path. Must be outside it.",
)
@click.option("--variable", help="Restrict to this variable.")
@click.option("--grid", help="Restrict to this grid.")
@click.option("--time-unit", help="Restrict to this time unit.")
@click.option("--execute", is_flag=True, help="Actually move files. Without it, only report.")
@click.option("--progress/--no-progress", default=True, help="Show a progress bar (use --no-progress for batch jobs).")
def main(data_path, quarantine_path, variable, grid, time_unit, execute, progress):
    base_path = Path(data_path).resolve()
    quarantine_path = Path(quarantine_path).resolve()
    if not base_path.is_dir():
        raise click.UsageError(f"--data-path does not exist: {base_path}")
    if quarantine_path.is_relative_to(base_path):
        raise click.UsageError("--quarantine-path must be outside --data-path, or the migration would read it")

    partitions = mig.find_partitions(base_path, variable, grid, time_unit)
    if not partitions:
        print("No tabular partitions found matching the given filters.")
        return

    plans: dict[str, dict] = {}
    identical = 0
    for paths in tqdm(partitions.values(), desc="Checking", unit="partition", disable=not progress):
        rel = str(paths[0].parent.relative_to(base_path))
        try:
            plan = plan_partition(paths)
        except Exception as exc:
            tqdm.write(f"  ERROR checking {rel}: {exc}", file=sys.stderr)
            continue
        if plan is None:
            continue
        if plan["excess"]:
            plans[rel] = plan
        else:
            identical += 1

    print(f"\nChecked {len(partitions)} partition(s): {len(plans)} with differing duplicates, "
          f"{identical} with identical duplicates only (left alone).")
    unresolved = [rel for rel, plan in plans.items() if plan["unresolved"]]
    for rel, plan in plans.items():
        print(f"\n  {rel}")
        print(f"    {plan['dup_keys']:,} duplicated key(s), excess {plan['excess']:,} -> {plan['after']:,}")
        for path in plan["move"]:
            mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ", timespec="seconds")
            print(f"    move {path.name}  ({mtime})")
        for problem in plan["unresolved"]:
            print(f"    UNRESOLVED: {problem} -- nothing moved")

    movable = {rel: plan for rel, plan in plans.items() if rel not in unresolved and plan["move"]}
    n_files = sum(len(plan["move"]) for plan in movable.values())
    if not execute:
        if n_files:
            print(f"\nDRY RUN -- nothing moved. {n_files} file(s) in {len(movable)} partition(s) would be "
                  f"moved to {quarantine_path}. Re-run with --execute to move them.")
        if unresolved:
            sys.exit(1)
        return

    manifest = quarantine_path / "quarantine-manifest.csv"
    quarantine_path.mkdir(parents=True, exist_ok=True)
    new_manifest = not manifest.exists()
    moved = 0
    with manifest.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new_manifest:
            writer.writerow(["moved_at", "partition", "source", "destination"])
        for rel, plan in movable.items():
            for path in plan["move"]:
                destination = quarantine_path / path.relative_to(base_path)
                if destination.exists():
                    print(f"  ERROR: {destination} already exists -- {path} not moved", file=sys.stderr)
                    unresolved.append(rel)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(path, destination)
                writer.writerow([datetime.datetime.now().isoformat(timespec="seconds"), rel, path, destination])
                handle.flush()
                moved += 1

    print(f"\nMoved {moved} file(s) to {quarantine_path} (manifest: {manifest}).")
    if moved:
        print("Re-run repair_excess_rows.py --execute to rebuild the affected output.")
    if unresolved:
        sys.exit(1)


if __name__ == "__main__":
    main()
