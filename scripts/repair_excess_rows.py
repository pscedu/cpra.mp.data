"""Find long-format tabular partitions holding more rows than their wide source, and rebuild them.

``migrate_to_long_format.py --append-missing`` reports a partition as having "N group(s)
with M row(s) more than the source" and deliberately leaves it alone -- appending never
deletes.  This script is the follow-up: it checks every tabular partition the same way,
and for each one with excess rows it

\b
1. deletes the partition's long-format output files,
2. removes the partition's ``tabular`` rows from the catalog (without this the migration
   treats the partition as already converted and never rebuilds it -- see
   ``purge_model_group.py``), and
3. re-runs ``migrate_partition`` for it from scratch, then re-checks the result.

The comparison is the one ``_append_missing`` makes: per group of the partition's
non-calendar dimension columns, expected long rows (wide rows x grid_cell columns,
deduplicated with ``--unique-parquet-rows``) against the rows actually in the output.  A
partition is flagged when any group is *over*.  ``--include-short`` also flags partitions
with groups that are short (normally ``--append-missing``'s job, but a full rebuild is the
blunter fix if you'd rather).

Pass the same ``--unique-parquet-rows`` the migration was run with: it changes both the
expected counts and what the rebuild writes, so a mismatch either flags complete
partitions or rebuilds them to a count that is flagged again.

\b
Safety:
* **Nothing is changed without ``--execute``.**  The default run only checks and reports
  (and writes ``--report``).
* An output file that also holds rows from outside the partition is never deleted; the
  partition is reported as ``shared`` and left alone.  (Migrated output maps one file per
  source partition, so this should not happen -- it is a guard, not a feature.)
* Every flagged partition is purged before any is rebuilt.  If the run is interrupted
  partway through the rebuild, the remaining partitions have neither output nor catalog
  rows, so an ordinary ``migrate_to_long_format.py`` run picks them back up.
* Don't run this against an output tree / catalog that a migration is still writing to.

Usage:

\b
    # Check only -- report partitions with excess rows, change nothing
    python scripts/repair_excess_rows.py \\
        --data-path /path/to/wide --output-path /path/to/long --catalog-path /path/to/catalog \\
        --report excess.csv

\b
    # Purge and rebuild them (same --unique-parquet-rows / --memory-limit as the migration)
    python scripts/repair_excess_rows.py \\
        --data-path /path/to/wide --output-path /path/to/long --catalog-path /path/to/catalog \\
        --unique-parquet-rows --memory-limit 32GB --execute
"""

import csv
import multiprocessing
import queue
import sys
import uuid
from pathlib import Path

import click
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

# The project root makes the package importable; the scripts directory makes the migration
# script importable, so the check and rebuild use exactly its discovery and conversion code.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import migrate_to_long_format as mig

from cpra.mp.data.catalog import Catalog, cal_for, parse_path_dims
from cpra.mp.data.config import BASE_PATH, CATALOG_PATH
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.tabular import read_parquet

DATASET_DIMS = ("variable", "time_unit", "grid")

# Partition statuses from check_partition.
OK = "ok"
OVER = "over"  # at least one group has more rows than the source
SHORT = "short"  # no group over, at least one short
MISSING = "missing"  # no output at all -- never converted; not this script's business
SHARED = "shared"  # an output file holds rows from outside the partition; never purged
NOT_WIDE = "not_wide"  # the source isn't wide-format


_output_index: dict[Path, list[tuple[dict, Path]]] = {}


def output_files(output_path: Path, hive_dims: dict) -> list[Path]:
    """The output Parquet files in the partition ``hive_dims`` describes, found by path.

    Deliberately a directory listing rather than ``read_parquet(include_file_paths=...)``:
    that materializes a path string for every *row*, which for an annual dataset (every
    model group in one partition) is enough on its own to get the process OOM-killed.
    Each output dataset directory is walked once and cached, since the daily datasets
    hold thousands of partitions apiece.
    """
    dataset_dir = output_path.joinpath(*(f"{name}={hive_dims[name]}" for name in DATASET_DIMS))
    if dataset_dir not in _output_index:
        _output_index[dataset_dir] = [
            (parse_path_dims(path.parent.relative_to(output_path)), path)
            for path in sorted(dataset_dir.rglob("*.parquet")) if path.is_file()
        ] if dataset_dir.is_dir() else []
    wanted = {name: str(value) for name, value in hive_dims.items()}
    return [path for dims, path in _output_index[dataset_dir] if all(dims.get(k) == v for k, v in wanted.items())]


def check_partition(paths: list[Path], output_path: Path, unique_parquet_rows: bool) -> dict:
    """Compare one source partition against its long-format output, group by group.

    Mirrors the counting in ``migrate_to_long_format._append_missing`` so a partition is
    flagged here exactly when the migration warns about it.  Returns a record with
    ``status``, ``expected``/``existing`` totals, ``over_rows``/``short_rows`` (summed over
    the groups that are over/short), ``over_groups``/``short_groups`` and the output
    ``files``.
    """
    record = {
        "status": OK, "expected": 0, "existing": 0, "over_rows": 0, "short_rows": 0,
        "over_groups": 0, "short_groups": 0, "files": [],
    }
    schema = mig._wide_schema(paths[0])
    if schema is None:
        return {**record, "status": NOT_WIDE}

    geo_cols = [col for col in schema if col[:1].isdigit()]
    alpha_cols = {col for col in schema if not col[:1].isdigit()}
    other_dim_cols = [col for col in DIMENSIONS if col in alpha_cols and not col.startswith("calendar_")]
    hive_dims = mig._dimension_cols_from_hive(paths[0])
    cal_col = cal_for(str(hive_dims.get("time_unit", "")))
    n_geo = len(geo_cols)

    try:
        existing_lf = read_parquet(base_path=output_path, **hive_dims)
        partition_rows = existing_lf.select(pl.len()).collect(engine="streaming").item()
    except Exception:
        partition_rows = 0
    if not partition_rows:
        return {**record, "status": MISSING}

    files = output_files(output_path, hive_dims)
    record["files"] = [str(path) for path in files]

    # Every row in the partition's files must belong to the partition, or deleting the
    # files would take other partitions' data with them.
    file_rows = sum(pq.read_metadata(path).num_rows for path in files)
    if file_rows != partition_rows:
        return {**record, "status": SHARED, "existing": partition_rows}

    wide_keys = pl.scan_parquet(paths).select([*other_dim_cols, cal_col])
    if unique_parquet_rows:
        wide_keys = wide_keys.unique()

    if other_dim_cols:
        group_dtypes = {col: DIMENSIONS[col].dtype for col in other_dim_cols}
        expected = (
            wide_keys.group_by(other_dim_cols)
            .agg((pl.len() * n_geo).cast(pl.Int64).alias("expected"))
            .collect(engine="streaming")
            .cast(group_dtypes)
        )
        existing = (
            existing_lf.group_by(other_dim_cols)
            .agg(pl.len().cast(pl.Int64).alias("existing"))
            .collect(engine="streaming")
            .cast(group_dtypes)
        )
        counts = expected.join(existing, on=other_dim_cols, how="full", coalesce=True).with_columns(
            pl.col("expected").fill_null(0), pl.col("existing").fill_null(0)
        )
    else:
        if unique_parquet_rows:
            expected_rows = wide_keys.select(pl.len()).collect(engine="streaming").item() * n_geo
        else:
            # Footer metadata only -- this is the common daily case, so it matters.
            expected_rows = sum(pq.read_metadata(path).num_rows for path in paths) * n_geo
        counts = pl.DataFrame(
            {"expected": [expected_rows], "existing": [partition_rows]},
            schema={"expected": pl.Int64, "existing": pl.Int64},
        )

    over = counts.filter(pl.col("existing") > pl.col("expected"))
    short = counts.filter(pl.col("existing") < pl.col("expected"))
    record.update(
        expected=int(counts["expected"].sum()),
        existing=int(counts["existing"].sum()),
        over_rows=int((over["existing"] - over["expected"]).sum()),
        short_rows=int((short["expected"] - short["existing"]).sum()),
        over_groups=over.height,
        short_groups=short.height,
    )
    if over.height:
        record["status"] = OVER
    elif short.height:
        record["status"] = SHORT
    return record


def catalog_prune_partition(catalog_path: Path, rel: str) -> tuple[int, int]:
    """Remove the catalog's ``tabular`` rows for one partition, returning (rows, files) changed.

    A row belongs to the partition when every dimension in the partition's path (beyond
    variable/time_unit/grid, which select the catalog directory) matches it -- the same
    test ``_catalog_completed`` uses to decide a partition is already converted.  For a
    partition that is a whole dataset directory (annual/decadal/monthly) that is every
    tabular row of the dataset.  Files are rewritten beside themselves and renamed over;
    a file left with no rows is removed.
    """
    dims = parse_path_dims(rel)
    directory = catalog_path / "/".join(f"{name}={dims[name]}" for name in DATASET_DIMS)
    removed_rows = changed_files = 0
    for path in sorted(directory.glob("*.parquet")) if directory.is_dir() else []:
        frame = pl.read_parquet(path)
        if "data_type" not in frame.columns:
            continue
        match = pl.col("data_type").cast(pl.String) == "tabular"
        for name, value in dims.items():
            if name not in DATASET_DIMS and name in frame.columns:
                match &= pl.col(name).cast(pl.String) == str(value)
        kept = frame.filter(match.not_())
        if kept.height == frame.height:
            continue
        removed_rows += frame.height - kept.height
        changed_files += 1
        if kept.height == 0:
            path.unlink()
        else:
            temporary = path.parent / f".{uuid.uuid4()}.parquet.tmp"
            kept.write_parquet(temporary)
            temporary.replace(path)
    return removed_rows, changed_files


def rebuild(
    tasks, base_path, output_path, catalog_path, memory_limit, temp_path, unique_parquet_rows, workers, progress=True,
):
    """Re-migrate ``tasks`` (``("tabular", rel, paths)``), each in its own spawned process.

    The same isolation ``migrate_to_long_format`` uses, so allocator memory is released per
    partition and an OOM-killed worker is detected by its exit code.  Returns
    ``{rel: error or None}``.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    pending = iter(tasks)
    active: dict[int, tuple] = {}  # pid -> (process, rel)
    outcome: dict[str, str | None] = {}

    def spawn() -> None:
        task = next(pending, None)
        if task is None:
            return
        args = (
            task, base_path, output_path, catalog_path, False, memory_limit, False, temp_path,
            unique_parquet_rows, False,
        )
        process = ctx.Process(target=mig._run_partition, args=(args, result_queue), daemon=True)
        process.start()
        active[process.pid] = (process, task[1])

    for _ in range(workers):
        spawn()

    with tqdm(total=len(tasks), desc="Rebuilding", unit="partition", disable=not progress) as bar:
        while active:
            try:
                message = result_queue.get(timeout=0.5)
            except queue.Empty:
                for pid, (process, rel) in list(active.items()):
                    if not process.is_alive():
                        process.join()
                        if process.exitcode != 0:
                            del active[pid]
                            outcome[rel] = f"worker killed (OOM?), exit code {process.exitcode}"
                            tqdm.write(f"  ERROR: {rel}: {outcome[rel]}", file=sys.stderr)
                            bar.update(1)
                            spawn()
                continue
            if message[0] != "result":
                continue
            _, pid, rel, _result, messages, error = message
            if pid in active:
                active.pop(pid)[0].join()
            for line in messages:
                tqdm.write(line)
            if error:
                tqdm.write(f"  ERROR: {rel}: {error}", file=sys.stderr)
            outcome[rel] = error
            bar.update(1)
            spawn()
    return outcome


REPORT_FIELDS = [
    "partition", "status", "expected", "existing", "over_rows", "over_groups",
    "short_rows", "short_groups", "files", "action", "after_status", "after_existing", "error",
]


@click.command(
    help=__doc__,
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)
@click.option("--data-path", default=str(BASE_PATH), show_default=True, help="Wide-format source tree.")
@click.option("--output-path", required=True, help="Long-format tree the migration wrote.")
@click.option(
    "--catalog-path",
    metavar="DIR",
    default=None,
    help="Catalog side-table the migration wrote. Defaults to CATALOG_PATH "
    "(the CPRA_MP_DATA_CATALOG_PATH environment variable).",
)
@click.option("--variable", help="Restrict to this variable.")
@click.option("--grid", help="Restrict to this grid.")
@click.option("--time-unit", help="Restrict to this time unit.")
@click.option(
    "--include-short",
    is_flag=True,
    help="Also purge and rebuild partitions that are short of their source (no group over).",
)
@click.option(
    "--unique-parquet-rows",
    is_flag=True,
    help="Count and rebuild with duplicate rows dropped. Use whatever the migration used.",
)
@click.option("--memory-limit", metavar="SIZE", help="Per-worker memory budget for the rebuild (e.g. '32GB').")
@click.option(
    "--temp-path",
    metavar="DIR",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory for the rebuild's intermediate batch files.",
)
@click.option("--workers", type=int, default=1, show_default=True, help="Partitions rebuilt in parallel.")
@click.option(
    "--report",
    metavar="PATH",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write every non-ok partition (and, after --execute, its rebuilt state) to this CSV.",
)
@click.option("--execute", is_flag=True, help="Actually purge and rebuild. Without it, only check and report.")
@click.option("--progress/--no-progress", default=True, help="Show progress bars (use --no-progress for batch jobs).")
def main(
    data_path, output_path, catalog_path, variable, grid, time_unit, include_short, unique_parquet_rows,
    memory_limit, temp_path, workers, report, execute, progress,
):
    catalog_path = Path(catalog_path) if catalog_path else CATALOG_PATH
    if not catalog_path:
        raise click.UsageError("--catalog-path is required (CATALOG_PATH is not set)")
    catalog_path = Path(catalog_path).resolve()
    base_path = Path(data_path).resolve()
    output_path = Path(output_path).resolve()
    for name, path in (("--data-path", base_path), ("--output-path", output_path), ("--catalog-path", catalog_path)):
        if not path.is_dir():
            raise click.UsageError(f"{name} does not exist: {path}")
    if temp_path is not None:
        temp_path = temp_path.resolve()
    try:
        memory_limit = mig._parse_memory(memory_limit) if memory_limit else None
    except ValueError:
        raise click.BadParameter(f"invalid value: {memory_limit!r}", param_hint="'--memory-limit'")

    partitions = mig.find_partitions(base_path, variable, grid, time_unit)
    if not partitions:
        print("No tabular partitions found matching the given filters.")
        return

    # --- check ---------------------------------------------------------------------------
    records: dict[str, dict] = {}
    for paths in tqdm(partitions.values(), desc="Checking", unit="partition", disable=not progress):
        rel = str(paths[0].parent.relative_to(base_path))
        try:
            records[rel] = {"paths": paths, **check_partition(paths, output_path, unique_parquet_rows)}
        except Exception as exc:
            records[rel] = {"paths": paths, "status": "error", "files": [], "error": str(exc)}
            tqdm.write(f"  ERROR checking {rel}: {exc}", file=sys.stderr)

    flag_statuses = {OVER, SHORT} if include_short else {OVER}
    flagged = [rel for rel, rec in records.items() if rec["status"] in flag_statuses]

    # The migration skips a partition whose estimated peak exceeds --memory-limit even at
    # the minimum batch size, rather than lose a worker to the OOM killer.  Here that
    # check has to come before the purge: a partition purged and then OOM-killed in the
    # rebuild is left with no output at all.
    too_big = []
    if memory_limit is not None:
        for rel in flagged:
            est = mig._partition_estimates(records[rel]["paths"], memory_limit)
            if est is not None and est["est_peak"] > memory_limit:
                records[rel]["action"] = "skipped_memory_limit"
                records[rel]["error"] = f"est. peak {mig._fmt_bytes(est['est_peak'])}"
                too_big.append(rel)
        flagged = [rel for rel in flagged if rel not in too_big]

    tally: dict[str, int] = {}
    for rec in records.values():
        tally[rec["status"]] = tally.get(rec["status"], 0) + 1
    print(f"\nChecked {len(records)} partition(s): " + ", ".join(f"{n} {s}" for s, n in sorted(tally.items())))
    for status, note in (
        (SHARED, "output file(s) also hold other partitions' rows -- left alone"),
        (SHORT, "short of the source" + ("" if include_short else " -- not flagged (see --append-missing / --include-short)")),
    ):
        rels = [rel for rel, rec in records.items() if rec["status"] == status]
        if rels:
            print(f"\n  {len(rels)} {status}: {note}")
            for rel in rels[:20]:
                print(f"    {rel}")
            if len(rels) > 20:
                print(f"    ... and {len(rels) - 20} more")

    if too_big:
        print(f"\n  {len(too_big)} flagged partition(s) need more than --memory-limit to rebuild -- left alone:")
        for rel in too_big:
            print(f"    {rel} ({records[rel]['error']})")

    if flagged:
        print(f"\n{len(flagged)} partition(s) to purge and rebuild:")
        for rel in flagged:
            rec = records[rel]
            detail = f"+{rec['over_rows']:,} over in {rec['over_groups']} group(s)" if rec["over_rows"] else ""
            if rec["short_rows"]:
                detail += f"{', ' if detail else ''}-{rec['short_rows']:,} short in {rec['short_groups']} group(s)"
            print(f"  {rel}: {rec['existing']:,} rows vs {rec['expected']:,} expected ({detail})")
    else:
        print("\nNothing to rebuild.")

    def write_report() -> None:
        if not report:
            return
        with report.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS, restval="", extrasaction="ignore")
            writer.writeheader()
            for rel, rec in records.items():
                if rec["status"] in (OK, NOT_WIDE):
                    continue
                writer.writerow({**rec, "partition": rel, "files": " ".join(rec["files"])})
        print(f"\nReport written: {report}")

    if not flagged or not execute:
        for rel in flagged:
            records[rel]["action"] = "would_rebuild"
        write_report()
        if flagged:
            print("\nDRY RUN -- nothing changed. Re-run with --execute to purge and rebuild.")
        elif too_big and execute:
            sys.exit(1)
        return

    # --- purge: every flagged partition, before any is rebuilt ---------------------------
    purged = []
    for rel in flagged:
        rec = records[rel]
        try:
            for path in rec["files"]:
                Path(path).unlink(missing_ok=True)
            rows, files = catalog_prune_partition(catalog_path, rel)
            rec["action"] = "purged"
            purged.append(rel)
            print(f"  Purged {rel}: {len(rec['files'])} data file(s), {rows} catalog row(s) in {files} file(s)")
        except Exception as exc:
            rec["action"], rec["error"] = "purge_failed", str(exc)
            print(f"  ERROR purging {rel}: {exc}", file=sys.stderr)

    # --- rebuild -------------------------------------------------------------------------
    tasks = [("tabular", rel, records[rel]["paths"]) for rel in purged]
    outcome = rebuild(
        tasks, base_path, output_path, catalog_path, memory_limit, temp_path, unique_parquet_rows, max(1, workers),
        progress=progress,
    )

    datasets = {tuple(parse_path_dims(rel)[name] for name in DATASET_DIMS) for rel in purged}
    catalog = Catalog(catalog_path)
    for variable_, time_unit_, grid_ in sorted(datasets):
        catalog.compact_partition(variable_, time_unit_, grid_)

    # --- verify --------------------------------------------------------------------------
    _output_index.clear()  # the purge and rebuild replaced the files it lists
    failed = []
    for rel in purged:
        rec = records[rel]
        if outcome.get(rel):
            rec["action"], rec["error"] = "rebuild_failed", outcome[rel]
            failed.append(rel)
            continue
        after = check_partition(rec["paths"], output_path, unique_parquet_rows)
        rec["action"] = "rebuilt"
        rec["after_status"], rec["after_existing"] = after["status"], after["existing"]
        if after["status"] != OK:
            failed.append(rel)
    write_report()

    print(f"\nRebuilt {len(purged) - len(failed)} of {len(flagged) + len(too_big)} flagged partition(s) cleanly.")
    if failed or len(purged) < len(flagged) or too_big:
        print("Not resolved:", file=sys.stderr)
        for rel in too_big:
            print(f"  {rel}: not purged, needs more than --memory-limit ({records[rel]['error']})", file=sys.stderr)
        for rel in flagged:
            rec = records[rel]
            if rel in failed or rec["action"] == "purge_failed":
                state = rec.get("error") or f"{rec['after_status']}, {rec['after_existing']:,} rows"
                print(f"  {rel}: {state}", file=sys.stderr)
        if any(records[rel].get("after_status") == OVER for rel in failed):
            print(
                "Partitions still over after a clean rebuild usually mean --unique-parquet-rows "
                "differs from the migration's (or the source itself holds duplicate rows).",
                file=sys.stderr,
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
