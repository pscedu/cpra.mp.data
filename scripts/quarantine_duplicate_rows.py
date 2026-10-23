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

A full scan reads the keys of every partition in the tree.  ``--from-report`` limits the
run to the partitions a ``repair_excess_rows.py --report`` CSV shows as still over (its
``after_status``, or ``status`` for a check-only report), and prints each one's
``over_rows`` beside the excess measured here -- they should match.

Afterwards, re-run ``repair_excess_rows.py --execute`` to rebuild the affected output.

Usage:

\b
    # Report only
    python scripts/quarantine_duplicate_rows.py \\
        --data-path /path/to/wide --quarantine-path /path/to/wide-quarantine

\b
    # Only the partitions a repair run left over
    python scripts/quarantine_duplicate_rows.py \\
        --data-path /path/to/wide --quarantine-path /path/to/wide-quarantine \\
        --from-report repair_report.csv

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

# Upper bound on long rows (key x grid cell x copy) unpivoted at once when measuring excess.
CHUNK_LONG_ROWS = 5_000_000


def key_columns(schema: dict, path: Path) -> list[str]:
    """The columns that identify one wide row: non-calendar dimensions, then the calendar."""
    alpha_cols = set(col for col in schema if not col[:1].isdigit())
    other_dim_cols = [col for col in DIMENSIONS if col in alpha_cols and not col.startswith("calendar_")]
    time_unit = str(mig._dimension_cols_from_hive(path).get("time_unit", ""))
    return [*other_dim_cols, cal_for(time_unit)]


def read_keys(paths: list[Path], key_cols: list[str]) -> pl.DataFrame:
    """Every row's key, tagged with its file -- read one file at a time.

    A wide file's footer describes every grid-cell column (~100k of them on the large
    grids); scanning a whole partition at once holds all of those footers together.
    """
    return pl.concat(
        [
            pl.read_parquet(path, columns=key_cols, hive_partitioning=False).with_columns(
                pl.lit(str(path)).alias(FILE)
            )
            for path in paths
        ],
        how="vertical_relaxed",
    )


def row_hashes(dups: pl.DataFrame, key_cols: list[str]) -> pl.DataFrame:
    """One hash of every non-key column per duplicated row, read file by file.

    Copies of a key with equal hashes are identical and can't add excess, so the expensive
    per-cell comparison only ever sees the keys whose copies actually differ.
    """
    frames = []
    for (file,), file_keys in dups.group_by(FILE):
        rows = (
            pl.scan_parquet(file, hive_partitioning=False)
            .join(file_keys.select(key_cols).unique().lazy(), on=key_cols, how="semi")
            .collect(engine="streaming")
        )
        frames.append(
            rows.select(key_cols).with_columns(
                rows.drop(key_cols).hash_rows().alias("hash"), pl.lit(file).alias(FILE)
            )
        )
        del rows
    return pl.concat(frames, how="vertical_relaxed")


def excess_rows(
    schema: dict, key_cols: list[str], dups: pl.DataFrame, moved: set[str]
) -> tuple[int, int]:
    """Long rows the conversion (with ``--unique-parquet-rows``) writes beyond one per key and grid cell.

    Returned for the partition as it is and as it would be without the ``moved`` files.
    ``dups`` is the key/file rows of the keys whose copies differ; they are unpivoted a
    slice of keys at a time, reading only the files that hold that slice.
    """
    alpha_cols = [col for col in schema if not col[:1].isdigit()]
    n_geo = len(schema) - len(alpha_cols)
    keys = dups.select(key_cols).unique(maintain_order=True)
    copies = dups.group_by(key_cols).len()["len"].max()
    step = max(1, CHUNK_LONG_ROWS // (n_geo * copies))
    def excess(long: pl.DataFrame) -> int:
        return long.drop(FILE).unique().height - long.select(*key_cols, "grid_cell").unique().height

    before = after = 0
    for offset in range(0, keys.height, step):
        chunk = keys.slice(offset, step)
        files = dups.join(chunk, on=key_cols, how="semi")[FILE].unique().to_list()
        long = (
            pl.scan_parquet(files, include_file_paths=FILE, hive_partitioning=False)
            .join(chunk.lazy(), on=key_cols, how="semi")
            .unpivot(index=[*alpha_cols, FILE], variable_name="grid_cell", value_name="value")
            .collect(engine="streaming")
        )
        before += excess(long)
        after += excess(long.filter(pl.col(FILE).is_in(moved).not_()))
        del long
    return before, after


def plan_partition(paths: list[Path]) -> dict | None:
    """Work out which files to move for one partition; None if it has no duplicated keys."""
    schema = mig._wide_schema(paths[0])
    if schema is None:
        return None
    key_cols = key_columns(schema, paths[0])
    keys = read_keys(paths, key_cols)
    dups = keys.filter(pl.len().over(key_cols) > 1)
    if dups.is_empty():
        return None

    hashes = row_hashes(dups, key_cols)
    differing = hashes.filter(pl.col("hash").n_unique().over(key_cols) > 1).select(key_cols).unique()
    plan = {
        "excess": 0, "move": [], "unresolved": [], "after": None,
        "dup_keys": dups.select(key_cols).unique().height,
    }
    if differing.is_empty():
        return plan

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

    # Identical copies add no excess before or after, so only the differing keys are measured.
    plan["excess"], plan["after"] = excess_rows(
        schema, key_cols, dups.join(differing, on=key_cols, how="semi"), {str(path) for path in plan["move"]}
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
@click.option(
    "--from-report",
    metavar="CSV",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Check only the partitions this repair_excess_rows.py --report CSV shows as still over.",
)
@click.option("--variable", help="Restrict to this variable.")
@click.option("--grid", help="Restrict to this grid.")
@click.option("--time-unit", help="Restrict to this time unit.")
@click.option("--execute", is_flag=True, help="Actually move files. Without it, only report.")
@click.option("--progress/--no-progress", default=True, help="Show a progress bar (use --no-progress for batch jobs).")
def main(data_path, quarantine_path, from_report, variable, grid, time_unit, execute, progress):
    base_path = Path(data_path).resolve()
    quarantine_path = Path(quarantine_path).resolve()
    if not base_path.is_dir():
        raise click.UsageError(f"--data-path does not exist: {base_path}")
    if quarantine_path.is_relative_to(base_path):
        raise click.UsageError("--quarantine-path must be outside --data-path, or the migration would read it")

    reported: dict[str, int] = {}
    if from_report:
        if variable or grid or time_unit:
            raise click.UsageError("--from-report can't be combined with --variable/--grid/--time-unit")
        with from_report.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if (row.get("after_status") or row["status"]) == "over":
                    reported[row["partition"]] = int(row["over_rows"])
        if not reported:
            print(f"No partitions in {from_report} are still over.")
            return
        partitions = {}
        for rel in reported:
            paths = sorted((base_path / rel).glob("*.parquet"))
            if paths:
                partitions[base_path / rel] = paths
            else:
                print(f"  ERROR: no Parquet files in {base_path / rel}", file=sys.stderr)
    else:
        partitions = mig.find_partitions(base_path, variable, grid, time_unit)
    if not partitions:
        print("No tabular partitions found matching the given filters.")
        return

    plans: dict[str, dict] = {}
    identical = 0
    unexplained = []  # reported partitions the duplicates here don't account for
    bar = tqdm(partitions.values(), desc="Checking", unit="partition", disable=not progress)
    for paths in bar:
        rel = str(paths[0].parent.relative_to(base_path))
        bar.set_postfix_str(rel, refresh=False)
        try:
            plan = plan_partition(paths)
        except Exception as exc:
            tqdm.write(f"  ERROR checking {rel}: {exc}", file=sys.stderr)
            continue
        if plan is None or not plan["excess"]:
            if rel in reported:
                unexplained.append(rel)
                tqdm.write(f"  WARNING: {rel} has no differing duplicates -- its excess has another cause")
            if plan is not None:
                identical += 1
            continue
        if rel in reported and plan["excess"] != reported[rel]:
            unexplained.append(rel)
            plan["unresolved"].append(
                f"excess {plan['excess']:,} doesn't match the report's {reported[rel]:,} over -- something else is going on"
            )
        plans[rel] = plan

    print(f"\nChecked {len(partitions)} partition(s): {len(plans)} with differing duplicates, "
          f"{identical} with identical duplicates only (left alone).")
    unresolved = [rel for rel, plan in plans.items() if plan["unresolved"]]
    for rel, plan in plans.items():
        print(f"\n  {rel}")
        line = f"    {plan['dup_keys']:,} duplicated key(s), excess {plan['excess']:,} -> {plan['after']:,}"
        if rel in reported:
            line += f"  (report: {reported[rel]:,} over)"
        print(line)
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
        if unresolved or unexplained:
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
    if unresolved or unexplained:
        sys.exit(1)


if __name__ == "__main__":
    main()
