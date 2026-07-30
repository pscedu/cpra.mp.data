"""Migrate model datasets into the long-format tree, out-of-place.

Reads from ``--data-path`` and writes a complete long-format tree under
``--output-path``, leaving the input untouched.  Two kinds of data are handled:

* **Tabular (Parquet).**  All wide-format files in the same partition directory
  are converted together and written as a single long-format Parquet file, so
  each output partition contains exactly one file.
* **Raster (COG).**  Single-band COGs (including legacy single-band files) are
  compacted into multi-band COGs wherever a contiguous calendar run spans more
  than one file, via ``raster.compact_cog`` writing into the output tree.

Everything processed is recorded in the unified catalog side-table under
``--catalog-path`` (per-write rows appended by the write/compact functions, then
coalesced into minimal contiguous runs by a final compaction pass).

Usage:

\b
    # Dry-run: show what would be migrated without writing anything
    python scripts/migrate_to_long_format.py --output-path /path/to/output --dry-run

\b
    # Migrate a single dataset (works for tabular or raster variables)
    python scripts/migrate_to_long_format.py \\
        --output-path /path/to/output --catalog-path /path/to/catalog \\
        --variable sal --grid hydro_compartment_v001 --time-unit daily

\b
    # Migrate all datasets
    python scripts/migrate_to_long_format.py \\
        --output-path /path/to/output --catalog-path /path/to/catalog

After verifying correctness on one dataset, run without --dry-run to apply.
"""

import multiprocessing
import queue
import os
import resource
import signal
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import click
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

# Add the project root to sys.path so the package is importable when the
# script is run from any working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpra.mp.data.catalog import Catalog, parse_path_dims
from cpra.mp.data.config import BASE_PATH, CATALOG_PATH
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.raster import compact_cog
from cpra.mp.data.tabular import read_parquet, write_parquet
from cpra.mp.data.utils import long_sort_expressions


def _db_init(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS migrations (
            partition TEXT PRIMARY KEY,
            data_type TEXT NOT NULL DEFAULT 'tabular',
            files INTEGER NOT NULL,
            geo_cols INTEGER NOT NULL,
            long_rows INTEGER NOT NULL,
            elapsed_s REAL NOT NULL,
            peak_bytes INTEGER NOT NULL,
            completed_at TEXT NOT NULL DEFAULT (datetime('now'))
        )"""
    )
    # Add data_type to progress DBs created before the raster/unified migration.
    if "data_type" not in {row[1] for row in conn.execute("PRAGMA table_info(migrations)")}:
        conn.execute("ALTER TABLE migrations ADD COLUMN data_type TEXT NOT NULL DEFAULT 'tabular'")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            partition TEXT NOT NULL,
            error TEXT,
            cpu_cores INTEGER,
            mem_available_bytes INTEGER,
            attempted_at TEXT NOT NULL DEFAULT (datetime('now'))
        )"""
    )
    conn.commit()
    return conn


def _db_completed(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT partition FROM migrations")}


def _available_memory_bytes() -> int | None:
    """Return available system memory in bytes, or None if it cannot be determined."""
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        return None


def _db_record_failure(
    conn: sqlite3.Connection,
    partition: str,
    error: str | None,
) -> None:
    conn.execute(
        "INSERT INTO failures (partition, error, cpu_cores, mem_available_bytes, attempted_at)"
        " VALUES (?, ?, ?, ?, datetime('now'))",
        (partition, error, os.cpu_count(), _available_memory_bytes()),
    )
    conn.commit()


def _db_record(
    conn: sqlite3.Connection,
    partition: str,
    files: int,
    geo_cols: int,
    long_rows: int,
    elapsed_s: float,
    peak_bytes: int,
    data_type: str = "tabular",
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO migrations"
        " (partition, data_type, files, geo_cols, long_rows, elapsed_s, peak_bytes, completed_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        (partition, data_type, files, geo_cols, long_rows, elapsed_s, peak_bytes),
    )
    conn.commit()


def _wide_schema(path: Path) -> dict | None:
    """Return the schema if the file is wide-format, else None.

    Reads the file footer once so callers don't need a second schema fetch.
    """
    schema = pl.read_parquet_schema(path)
    return schema if any(name[:1].isdigit() for name in schema) else None


def _dimension_cols_from_hive(path: Path) -> dict:
    """Extract dimension values encoded in the Hive partition path."""
    parts = {}
    for part in path.parts:
        if "=" in part:
            key, value = part.split("=", 1)
            if key in DIMENSIONS:
                parts[key] = pl.Series([value]).cast(DIMENSIONS[key].dtype)[0]
    return parts


def _convert_batched(
    parquet_paths, output_path, schema, geo_cols, hive_dims, time_unit, est, memory_limit,
    progress_cb=None, temp_path=None, catalog_path=None, round_to_decimal_places=None,
    unique_parquet_rows=False,
) -> list[str]:
    """Batched conversion via temp files, bounded to roughly one batch in memory.

    Works for every time unit.  The wide frame has one column per grid cell, so
    grid_cell is produced by unpivot rather than being a row attribute.  The
    target sort places grid_cell ahead of the calendar tail for every time unit:

      other_dims → grid_cell → calendar_year (annual/decadal)
      other_dims → grid_cell → calendar_day  (daily/monthly)

    A "group" is the set of wide rows that share every sort key *outside*
    grid_cell — i.e. other_dims, for every time unit (the calendar column sorts
    *inside* grid_cell, which is produced by unpivot, so all wide rows for one
    other_dims combo must stay together).  Groups must never be split across
    batches, or a grid_cell would appear in two batches and break the global
    order on concatenation.

    The wide frame is sorted so groups are contiguous; consecutive groups are
    packed into batches under a long-row budget; each batch is unpivoted, sorted
    to the global order, and streamed to a temp file.  Batches are written in
    order, so concatenating the temp files is globally sorted.  partition_by()
    is deliberately avoided — it would materialize every group at once, each
    retaining all (~100k) grid-cell columns.

    round_to_decimal_places, when given, rounds the unpivoted "value" column to
    that many decimal places before sorting/writing (only when the value dtype
    is floating-point; ignored otherwise).

    unique_parquet_rows, when True, drops duplicate rows (all columns equal,
    including grid_cell and value) from each batch's unpivoted long frame before
    sorting/writing. Batches never split a group, and duplicate wide rows always
    share the same other_dims group, so duplicates are always caught within a
    single batch.
    """
    messages: list[str] = []
    alpha_cols = [col for col in schema if not col[:1].isdigit()]
    other_dim_cols = [
        col for col in DIMENSIONS if col in set(alpha_cols) and not col.startswith("calendar_")
    ]
    value_dtype = schema[geo_cols[0]]
    round_value = round_to_decimal_places is not None and value_dtype in (pl.Float32, pl.Float64)

    wide_df = pl.scan_parquet(parquet_paths).collect()

    if other_dim_cols:
        wide_df = wide_df.sort(other_dim_cols)
        group_lengths = (
            wide_df.lazy()
            .select(other_dim_cols)
            .group_by(other_dim_cols, maintain_order=True)
            .len()
            .select("len")
            .collect()["len"]
            .to_list()
        )
    else:
        # No outer keys: the whole partition is one group (grid_cell → calendar).
        group_lengths = [wide_df.height]

    # The global long-format order; shared with write_parquet so it cannot drift.
    batch_sort = long_sort_expressions(other_dim_cols, time_unit)

    n_geo = len(geo_cols)
    target_long_rows = _batch_target_rows(est, memory_limit)
    if memory_limit is not None and est["wide_in_memory"] > memory_limit:
        messages.append(
            f"    WARNING: wide frame (~{_fmt_bytes(est['wide_in_memory'])}) exceeds the "
            f"{_fmt_bytes(memory_limit)} limit; it must be loaded in full, so the limit "
            f"cannot be honored for this partition."
        )

    # Pack consecutive groups into batches under the long-row budget (never
    # splitting a group).
    batches: list[tuple[int, int]] = []
    start = 0
    acc = 0
    for length in group_lengths:
        if acc > 0 and (acc + length) * n_geo > target_long_rows:
            batches.append((start, acc))
            start += acc
            acc = 0
        acc += length
    if acc > 0:
        batches.append((start, acc))

    messages.append(
        f"    batched conversion: {len(batches)} batch(es) via temp files "
        f"(≈{est['long_bytes_per_row'] * target_long_rows / 1024**3:.1f} GB/batch)"
    )
    if progress_cb:
        progress_cb(0, len(batches))
    with tempfile.TemporaryDirectory(dir=temp_path) as _tmpdir:
        tmp_paths = []
        for i, (b_offset, b_rows) in enumerate(batches):
            tmp_path = Path(_tmpdir) / f"chunk_{i:06d}.parquet"
            batch_lazy = (
                wide_df.slice(b_offset, b_rows)
                .lazy()
                .unpivot(index=alpha_cols, variable_name="grid_cell", value_name="value")
                .with_columns(pl.col("grid_cell").cast(pl.Int32))
            )
            if round_value:
                batch_lazy = batch_lazy.with_columns(pl.col("value").round(round_to_decimal_places))
            if unique_parquet_rows:
                batch_lazy = batch_lazy.unique()
            batch_lazy.sort(*batch_sort).sink_parquet(tmp_path)
            tmp_paths.append(tmp_path)
            if progress_cb:
                progress_cb(i + 1, len(batches))
        del wide_df  # free the wide frame before the final streaming write
        write_parquet(
            pl.scan_parquet(tmp_paths),
            base_path=output_path,
            catalog_path=catalog_path,
            validate=False,
            sort=False,
            temp_dir=temp_path,
            **hive_dims,
        )
    return messages


def migrate_partition(
    parquet_paths: list[Path],
    base_path: Path,
    output_path: Path,
    catalog_path: Path | None = None,
    dry_run: bool = False,
    memory_limit: int | None = None,
    progress_cb=None,
    temp_path=None,
    round_to_decimal_places: int | None = None,
    unique_parquet_rows: bool = False,
) -> tuple[dict | None, list[str]]:
    """Convert all wide-format Parquet files in one partition to a single long-format file.

    All paths must share the same parent directory (partition).  Returns (result, messages)
    where result is a stats dict if the partition was (or would be) migrated, or None if it
    was already in long format.  messages contains lines to print via tqdm.write in the caller.
    Stats keys: files, geo_cols, long_rows; plus elapsed_s and peak_bytes when not a dry run.

    Conversion always uses the batched temp-file path.  memory_limit (bytes),
    when given, sizes each batch so the per-worker peak stays within it;
    otherwise a default batch size is used.

    round_to_decimal_places, when given, is forwarded to ``_convert_batched`` to
    round floating-point values before writing.

    unique_parquet_rows, when True, is forwarded to ``_convert_batched`` to drop
    duplicate rows from the long-format output.

    temp_path, when given, is the directory under which the intermediate batch
    files are written (e.g. a fast dedicated scratch filesystem); otherwise the
    system default temp location is used.

    catalog_path, when given, is passed to ``write_parquet`` so its catalog hook
    appends rows to the migration's catalog (None -- e.g. dry runs -- disables it).

    progress_cb, when given, is called as progress_cb(done, total) to report
    intra-partition progress (number of batches written).
    """
    messages: list[str] = []
    schema = _wide_schema(parquet_paths[0])
    if schema is None:
        return None, messages

    partition_dir = parquet_paths[0].parent
    rel_dir = partition_dir.relative_to(base_path)
    n = len(parquet_paths)
    geo_cols = [col for col in schema if col[:1].isdigit()]

    # Row counts come from Parquet footer metadata — no data scan needed.
    wide_rows = sum(pq.read_metadata(p).num_rows for p in parquet_paths)
    row_count = wide_rows * len(geo_cols)

    hive_dims = _dimension_cols_from_hive(parquet_paths[0])

    # Discover existing output files via read_parquet so the directory ordering
    # of the output (which may differ from the input) is handled correctly.
    try:
        existing_files = (
            read_parquet(base_path=output_path, include_file_paths="__path__", **hive_dims)
            .select("__path__")
            .collect()["__path__"]
            .unique()
            .to_list()
        )
    except Exception:
        existing_files = []

    if existing_files:
        existing_rows = sum(pq.read_metadata(p).num_rows for p in existing_files)
        if existing_rows == row_count:
            messages.append(f"  Skipping {rel_dir} ({row_count:,} rows already written)")
            return {"skipped": True, "files": n, "geo_cols": len(geo_cols), "long_rows": row_count}, messages
        messages.append(f"  Output incomplete ({existing_rows:,}/{row_count:,} rows), clearing: {rel_dir}")
        if not dry_run:
            for p in existing_files:
                Path(p).unlink()

    messages.append(
        f"  {'[DRY RUN] ' if dry_run else ''}Migrating: {rel_dir} "
        f"({n} file{'s' if n > 1 else ''}, {len(geo_cols)} grid_cell cols, {row_count:,} long rows)"
    )

    if dry_run:
        return {"files": n, "geo_cols": len(geo_cols), "long_rows": row_count}, messages

    t0 = time.perf_counter()

    est = _peak_estimates(schema, wide_rows)
    time_unit = str(hive_dims.get("time_unit", ""))

    # Always use the batched temp-file path.  Benchmarks across all time units
    # showed it as fast or faster than a single in-memory sort while using far
    # less memory on large partitions, with byte-identical output.  memory_limit
    # tunes the per-batch budget.
    messages += _convert_batched(
        parquet_paths, output_path, schema, geo_cols, hive_dims, time_unit, est, memory_limit,
        progress_cb, temp_path, catalog_path, round_to_decimal_places, unique_parquet_rows,
    )

    elapsed = time.perf_counter() - t0
    # ru_maxrss is KiB on Linux, bytes on macOS.
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak_rss * (1 if sys.platform == "darwin" else 1024)
    messages.append(f"    done in {elapsed:.1f}s, peak RSS {_fmt_bytes(peak_bytes)}")
    return {"files": n, "geo_cols": len(geo_cols), "long_rows": row_count, "elapsed_s": elapsed, "peak_bytes": peak_bytes}, messages


def _raster_source_tifs(base_path: Path, variable: str, time_unit: str, grid: str) -> list[Path]:
    """All source COGs for one dataset, looked up under both directory orders.

    ``compact_cog`` reads from the canonical (variable/time_unit/grid) order and
    the legacy (variable/grid/time_unit) order; mirror both here so the reported
    source count matches what it will actually consolidate.
    """
    dirs = [
        base_path / f"variable={variable}" / f"time_unit={time_unit}" / f"grid={grid}",
        base_path / f"variable={variable}" / f"grid={grid}" / f"time_unit={time_unit}",
    ]
    return [p for d in dirs if d.exists() for p in d.rglob("*.tif")]


def migrate_raster_dataset(
    dataset_kwargs: dict,
    base_path: Path,
    output_path: Path,
    catalog_path: Path | None = None,
    dry_run: bool = False,
    round_to_decimal_places: int | None = None,
) -> tuple[dict, list[str]]:
    """Compact one raster dataset's single-band COGs into multi-band COGs, written
    into the output tree with the input left untouched.

    ``dataset_kwargs`` holds variable/time_unit/grid.  Every matching COG under
    ``base_path`` (canonical or legacy directory order) is pooled per partition and
    consolidated by ``raster.compact_cog`` writing into ``output_path``; catalog
    rows are appended and the partition coalesced by compact_cog itself when
    ``catalog_path`` is given.  Returns (result, messages); result carries
    ``data_type="raster"``, ``files`` (source COG count) and ``geo_cols`` (output
    COG count), plus ``elapsed_s``/``peak_bytes`` when not a dry run.

    round_to_decimal_places, when given, is forwarded to ``raster.compact_cog`` to
    round floating-point pixel values of every band that gets consolidated into a
    multi-band COG (bands whose run is already optimally stored are left untouched,
    same as compact_cog's usual skip behavior).
    """
    messages: list[str] = []
    variable, time_unit, grid = dataset_kwargs["variable"], dataset_kwargs["time_unit"], dataset_kwargs["grid"]
    rel = f"variable={variable}/time_unit={time_unit}/grid={grid}"

    n_src = len(_raster_source_tifs(base_path, variable, time_unit, grid))
    messages.append(
        f"  {'[DRY RUN] ' if dry_run else ''}Compacting rasters: {rel} ({n_src} source COG{'s' if n_src != 1 else ''})"
    )
    if dry_run:
        return {"data_type": "raster", "files": n_src, "geo_cols": 0, "long_rows": 0}, messages

    t0 = time.perf_counter()
    result = compact_cog(
        base_path=base_path,
        output_base_path=output_path,
        catalog_path=catalog_path,
        round_to_decimal_places=round_to_decimal_places,
        **dataset_kwargs,
    )
    elapsed = time.perf_counter() - t0

    n_out = len(result["outputs"])
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak_rss * (1 if sys.platform == "darwin" else 1024)
    messages.append(
        f"    done in {elapsed:.1f}s, {n_src} -> {n_out} COG(s), peak RSS {_fmt_bytes(peak_bytes)}"
    )
    return {
        "data_type": "raster",
        "files": n_src,
        "geo_cols": n_out,
        "long_rows": 0,
        "elapsed_s": elapsed,
        "peak_bytes": peak_bytes,
    }, messages


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


_DTYPE_SIZES = {
    pl.Float32: 4,
    pl.Float64: 8,
    pl.Int8: 1,
    pl.Int16: 2,
    pl.Int32: 4,
    pl.Int64: 8,
    pl.UInt8: 1,
    pl.UInt16: 2,
    pl.UInt32: 4,
    pl.UInt64: 8,
    pl.Date: 4,
    pl.Boolean: 1,
}


def _dtype_itemsize(dtype) -> int:
    if dtype in _DTYPE_SIZES:
        return _DTYPE_SIZES[dtype]
    try:
        if dtype.base_type() in (pl.Datetime, pl.Duration, pl.Time):
            return 8
    except AttributeError:
        pass
    return 50  # String, Categorical, List, Struct, etc.


def _parse_memory(value: str) -> int:
    """Parse a human-readable memory size into bytes.

    Accepts forms like '32GB', '32G', '500MB', '16gib', or a raw byte count.
    Binary units (1 K = 1024) are used throughout.
    """
    s = value.strip().upper().replace("IB", "").rstrip("B")
    mult = 1
    for suffix, m in (("K", 1024), ("M", 1024**2), ("G", 1024**3), ("T", 1024**4)):
        if s.endswith(suffix):
            mult = m
            s = s[:-1]
            break
    return int(float(s) * mult)


# Tuning constants for the batched memory model.  BATCH_SORT_FACTOR is empirical:
# a Polars unpivot→sort→sink pipeline transiently holds several copies of the
# data (the unpivoted frame, the sort's gather/index buffers, and the parquet
# encode buffer).  Measured batch-sort peaks were ~2.2× the batch's long form;
# 2.5 adds a safety margin so the memory limit is not exceeded.
DEFAULT_BATCH_LONG_ROWS = 50_000_000
MIN_BATCH_LONG_ROWS = 5_000_000
BATCH_SORT_FACTOR = 2.5


def _peak_estimates(schema: dict, wide_rows: int) -> dict:
    """Compute base memory figures for one wide-format partition.

    Returns a dict with:
      wide_in_memory      — decoded Arrow size of the wide DataFrame.
      long_rows           — number of rows in the long form.
      long_bytes_per_row  — bytes per row of the long form.
      long_in_memory      — decoded Arrow size of the full long form.
    """
    geo_cols = [c for c in schema if c[:1].isdigit()]
    alpha_cols = [c for c in schema if not c[:1].isdigit()]
    value_dtype = schema[geo_cols[0]]
    long_rows = wide_rows * len(geo_cols)
    wide_in_memory = wide_rows * sum(_dtype_itemsize(schema[c]) for c in schema)
    long_bytes_per_row = (
        sum(_dtype_itemsize(schema[c]) for c in alpha_cols)
        + _dtype_itemsize(pl.Int32)  # grid_cell
        + _dtype_itemsize(value_dtype)
    )
    long_in_memory = long_rows * long_bytes_per_row
    return {
        "wide_in_memory": wide_in_memory,
        "long_rows": long_rows,
        "long_bytes_per_row": long_bytes_per_row,
        "long_in_memory": long_in_memory,
    }


def _batch_target_rows(est: dict, memory_limit: int | None) -> int:
    """Long-row budget per batch for the batched temp-file path.

    Without a limit, a fixed default is used.  With a limit, the budget is sized
    so the resident wide frame plus one batch's sort stays within it:
    wide + BATCH_SORT_FACTOR × batch_long_bytes ≤ limit.
    """
    if memory_limit is None:
        return DEFAULT_BATCH_LONG_ROWS
    budget = memory_limit - est["wide_in_memory"]
    if budget <= 0:
        return MIN_BATCH_LONG_ROWS
    return max(MIN_BATCH_LONG_ROWS, int(budget / BATCH_SORT_FACTOR / est["long_bytes_per_row"]))


def _batched_peak(est: dict, memory_limit: int | None) -> int:
    """Conservative peak for the batched path, given its batch budget."""
    # A batch never holds more than the whole partition's long form.
    target = min(_batch_target_rows(est, memory_limit), est["long_rows"])
    loop_peak = est["wide_in_memory"] + BATCH_SORT_FACTOR * target * est["long_bytes_per_row"]
    # The one-time sort of the wide frame (≈ 2× wide) can also be the high-water mark.
    return int(max(2 * est["wide_in_memory"], loop_peak))


def estimate_partitions(
    partitions: dict[Path, list[Path]], base_path: Path, progress: bool = True, memory_limit: int | None = None
) -> None:
    """Print a memory-estimate table for all wide-format partitions, sorted by estimated peak memory.

    Sizes are derived from schema dtypes (num_rows × Σ dtype.itemsize), which reflects the
    actual decoded Arrow representation rather than the Parquet-encoded size.

    Every partition is converted via the batched temp-file path.  The estimated
    peak is the larger of the one-time wide-frame sort (≈ 2× wide) and a single
    batch's sort (≈ wide + BATCH_SORT_FACTOR × batch long form), where the batch
    size follows from memory_limit (or the default when none is given).
    """
    rows = []
    with tqdm(total=len(partitions), desc="Estimating", unit="partition", disable=not progress) as bar:
        for paths in partitions.values():
            schema = _wide_schema(paths[0])
            bar.update(1)
            if schema is None:
                continue

            wide_rows = sum(pq.read_metadata(p).num_rows for p in paths)
            est = _peak_estimates(schema, wide_rows)
            rows.append(
                {
                    "partition": str(paths[0].parent.relative_to(base_path)),
                    "files": len(paths),
                    "long_rows": est["long_rows"],
                    "wide_in_memory": est["wide_in_memory"],
                    "est_peak": _batched_peak(est, memory_limit),
                }
            )

    if not rows:
        print("No wide-format partitions found.")
        return

    rows.sort(key=lambda r: r["est_peak"], reverse=True)

    part_w = max(len(r["partition"]) for r in rows)
    part_w = max(part_w, len("Partition"))

    def _row(partition, files, long_rows, wide_data, est_peak):
        return f"{partition:<{part_w}}  {files:>5}  {long_rows:>14}  {wide_data:>10}  {est_peak:>10}"

    header = _row("Partition", "Files", "Long rows", "Wide data", "Est. peak")
    sep = "-" * len(header)
    print(f"\n{header}\n{sep}")
    for r in rows:
        print(
            _row(
                r["partition"],
                r["files"],
                f"{r['long_rows']:,}",
                _fmt_bytes(r["wide_in_memory"]),
                _fmt_bytes(r["est_peak"]),
            )
        )
    print(sep)
    print(
        _row(
            "TOTAL",
            sum(r["files"] for r in rows),
            f"{sum(r['long_rows'] for r in rows):,}",
            _fmt_bytes(sum(r["wide_in_memory"] for r in rows)),
            _fmt_bytes(sum(r["est_peak"] for r in rows)),
        )
    )


def _run_partition(args, result_queue):
    """Target for worker processes spawned directly (not via Pool).

    Each task runs in a fresh subprocess so allocator memory from Polars/Arrow (or
    GDAL) is fully released when the process exits.  A task is either a tabular
    partition (``("tabular", rel, [parquet paths])``) or a raster dataset
    (``("raster", rel, {dim kwargs})``); the kind selects the migration function.
    Messages are tagged tuples on result_queue:
      ("progress", pid, rel, done, total)            — intra-partition progress
      ("result",   pid, rel, result, messages, error) — terminal result
    If the process is killed by a signal the queue is never written; the main
    process detects this via p.exitcode.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    (
        task, base_path, output_path, catalog_path, dry_run, memory_limit, progress, temp_path,
        round_to_decimal_places, unique_parquet_rows,
    ) = args
    kind, rel, payload = task
    pid = os.getpid()

    def progress_cb(done, total):
        result_queue.put(("progress", pid, rel, done, total))

    try:
        if kind == "raster":
            result, messages = migrate_raster_dataset(
                payload, base_path, output_path, catalog_path, dry_run,
                round_to_decimal_places=round_to_decimal_places,
            )
        else:
            result, messages = migrate_partition(
                payload, base_path, output_path, catalog_path, dry_run,
                memory_limit=memory_limit,
                progress_cb=progress_cb if progress else None,
                temp_path=temp_path,
                round_to_decimal_places=round_to_decimal_places,
                unique_parquet_rows=unique_parquet_rows,
            )
        result_queue.put(("result", pid, rel, result, messages, None))
    except Exception as exc:
        result_queue.put(("result", pid, rel, None, [], str(exc)))


def find_partitions(base_path: Path, variable=None, grid=None, time_unit=None) -> dict[Path, list[Path]]:
    """Find all Parquet files under base_path grouped by partition directory."""
    pattern_parts = []
    for dim_name, value in [("variable", variable), ("grid", grid), ("time_unit", time_unit)]:
        pattern_parts.append(f"{dim_name}={value}" if value else f"{dim_name}=*")

    partitions: dict[Path, list[Path]] = {}
    for dataset_dir in base_path.glob("/".join(pattern_parts)):
        for path in sorted(dataset_dir.rglob("*.parquet")):
            partitions.setdefault(path.parent, []).append(path)
    return dict(sorted(partitions.items()))


def find_raster_datasets(base_path: Path, variable=None, grid=None, time_unit=None) -> list[dict]:
    """Find distinct raster (COG) datasets under base_path, keyed by variable/time_unit/grid.

    COGs live under either the canonical (variable/time_unit/grid) or the legacy
    (variable/grid/time_unit) directory order; both are globbed and collapsed to
    one entry per (variable, time_unit, grid).  A dataset is included only if it
    actually contains .tif files (so Parquet-only trees are skipped).  Returns a
    sorted list of dimension-kwargs dicts suitable for ``compact_cog``.
    """
    v, g, t = variable or "*", grid or "*", time_unit or "*"
    patterns = [
        f"variable={v}/time_unit={t}/grid={g}",  # canonical
        f"variable={v}/grid={g}/time_unit={t}",  # legacy
    ]
    datasets: dict[tuple, dict] = {}
    for pattern in patterns:
        for dataset_dir in base_path.glob(pattern):
            if next(dataset_dir.rglob("*.tif"), None) is None:
                continue
            dims = parse_path_dims(dataset_dir.relative_to(base_path))
            key = (dims["variable"], dims["time_unit"], dims["grid"])
            datasets[key] = {"variable": dims["variable"], "time_unit": dims["time_unit"], "grid": dims["grid"]}
    return [datasets[key] for key in sorted(datasets)]


@click.command(
    help=__doc__,
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)
@click.option("--output-path", help="Root directory for long-format output (required unless --estimate).")
@click.option(
    "--catalog-path",
    metavar="DIR",
    default=str(CATALOG_PATH),
    show_default=True,
    help="Root directory for the unified catalog side-table. Rows are appended as data is "
    "written and coalesced into minimal contiguous runs at the end. Ignored for --dry-run.",
)
@click.option("--estimate", is_flag=True, help="Print a memory-estimate table sorted by peak memory and exit.")
@click.option("--variable", help="Migrate only this variable.")
@click.option("--grid", help="Migrate only this grid.")
@click.option("--time-unit", help="Migrate only this time unit.")
@click.option("--dry-run", is_flag=True, help="Show what would be migrated without writing.")
@click.option(
    "--progress/--no-progress", default=True, help="Show progress bar (use --no-progress for batch jobs)."
)
@click.option("--data-path", default=str(BASE_PATH), show_default=True, help="Base data path.")
@click.option(
    "--workers",
    type=int,
    default=1,
    show_default=True,
    help="Number of partitions to process in parallel. Keep low (2–4) — Polars already "
    "saturates CPUs within each partition.",
)
@click.option(
    "--progress-db",
    metavar="PATH",
    help="SQLite database for tracking migration progress. Completed partitions are "
    "recorded and skipped on subsequent runs.",
)
@click.option(
    "--memory-limit",
    metavar="SIZE",
    help="Per-worker memory budget (e.g. '32GB'). Sizes each conversion batch so the "
    "worker's peak memory stays within the budget. Without this flag a default batch size is used.",
)
@click.option(
    "--temp-path",
    metavar="DIR",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory for intermediate batch files (e.g. a fast dedicated scratch "
    "filesystem). Defaults to the system temp location.",
)
@click.option(
    "--round-to-decimal-places",
    type=int,
    metavar="N",
    help="Round floating-point values to N decimal places before writing the long-format "
    "output. Applies to tabular (Parquet) value columns and to raster (COG) bands that get "
    "consolidated into a multi-band COG; ignored where the value/band dtype isn't floating-point.",
)
@click.option(
    "--unique-parquet-rows",
    is_flag=True,
    help="Drop duplicate rows (all columns equal) from the long-format Parquet output. "
    "Tabular data only; no equivalent is applied to raster (COG) output.",
)
def main(
    output_path, catalog_path, estimate, variable, grid, time_unit, dry_run, progress,
    data_path, workers, progress_db, memory_limit, temp_path, round_to_decimal_places,
    unique_parquet_rows,
):
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(1))

    if not estimate and not output_path:
        raise click.UsageError("--output-path is required unless --estimate is set")

    if temp_path is not None:
        temp_path = temp_path.resolve()

    try:
        memory_limit = _parse_memory(memory_limit) if memory_limit else None
    except ValueError:
        raise click.BadParameter(f"invalid value: {memory_limit!r}", param_hint="'--memory-limit'")

    base_path = Path(data_path).resolve()
    partitions = find_partitions(base_path, variable, grid, time_unit)
    raster_datasets = find_raster_datasets(base_path, variable, grid, time_unit)

    if not partitions and not raster_datasets:
        print("No datasets found matching the given filters.")
        return

    if estimate:
        estimate_partitions(partitions, base_path, progress=progress, memory_limit=memory_limit)
        if raster_datasets:
            n_tifs = sum(
                len(_raster_source_tifs(base_path, d["variable"], d["time_unit"], d["grid"]))
                for d in raster_datasets
            )
            print(
                f"\nRaster: {len(raster_datasets)} dataset(s), {n_tifs} source COG(s) "
                "(compacted via VRT — memory-light, not modeled above)."
            )
        return

    output_path = Path(output_path).resolve()
    # --dry-run writes nothing (no data, no catalog), so disable the catalog hooks
    # by passing None through to the write/compact functions.
    catalog_path = None if dry_run else Path(catalog_path).resolve()

    # One unified task list: tabular partitions and raster datasets, each tagged
    # with its kind and a stable progress key (rel).
    tasks: list[tuple[str, str, object]] = []
    for paths in partitions.values():
        tasks.append(("tabular", str(paths[0].parent.relative_to(base_path)), paths))
    for d in raster_datasets:
        rel = f"variable={d['variable']}/time_unit={d['time_unit']}/grid={d['grid']}"
        tasks.append(("raster", rel, d))

    db_conn: sqlite3.Connection | None = None
    db_skipped = 0
    if progress_db:
        db_conn = _db_init(Path(progress_db))
        completed = _db_completed(db_conn)
        if completed:
            kept = [t for t in tasks if t[1] not in completed]
            db_skipped = len(tasks) - len(kept)
            tasks = kept

    n_tabular = sum(1 for t in tasks if t[0] == "tabular")
    n_raster = sum(1 for t in tasks if t[0] == "raster")
    total_partitions = len(tasks)
    summary = f"Found {n_tabular} tabular partition(s) and {n_raster} raster dataset(s)."
    if db_skipped:
        summary += f" ({db_skipped} already in progress database, skipped.)"
    print(summary)
    migrated = 0
    skipped = 0
    errors: list[str] = []
    touched_datasets: set[tuple] = set()  # (variable, time_unit, grid) to compact in the catalog

    desc = "[DRY RUN]" if dry_run else "Migrating"
    packed = [
        (
            task, base_path, output_path, catalog_path, dry_run, memory_limit, progress, temp_path,
            round_to_decimal_places, unique_parquet_rows,
        )
        for task in tasks
    ]

    # spawn avoids fork-after-threads problems with Polars' internal thread pool.
    # Each partition runs in a dedicated Process so the allocator (jemalloc/mimalloc)
    # releases its pages back to the OS when the process exits, and a SIGKILL'd
    # worker is detectable via p.exitcode without relying on Pool's dead-worker logic.
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    active: dict[int, tuple[multiprocessing.Process, str]] = {}  # pid -> (proc, rel)
    task_iter = iter(packed)

    def _spawn_next() -> bool:
        try:
            item = next(task_iter)
        except StopIteration:
            return False
        task = item[0]
        p = ctx.Process(target=_run_partition, args=(item, result_queue), daemon=True)
        p.start()
        active[p.pid] = (p, task[1])
        return True

    # Nested per-worker progress bars (one slot per active worker, below the
    # partition bar at position 0).  Only created for multi-batch conversions.
    inner_bars: dict[int, tqdm] = {}
    used_positions: set[int] = set()

    def _short(rel: str, width: int = 48) -> str:
        return rel if len(rel) <= width else "…" + rel[-(width - 1):]

    def _update_inner(pid: int, rel: str, done: int, total: int) -> None:
        if not progress or total <= 1:
            return
        bar_inner = inner_bars.get(pid)
        if bar_inner is None:
            pos = 1
            while pos in used_positions:
                pos += 1
            used_positions.add(pos)
            bar_inner = tqdm(total=total, position=pos, leave=False, desc=_short(rel), unit="batch")
            bar_inner._slot = pos
            inner_bars[pid] = bar_inner
        bar_inner.total = total
        bar_inner.n = done
        bar_inner.refresh()

    def _close_inner(pid: int) -> None:
        bar_inner = inner_bars.pop(pid, None)
        if bar_inner is not None:
            used_positions.discard(getattr(bar_inner, "_slot", None))
            bar_inner.close()

    try:
        for _ in range(workers):
            if not _spawn_next():
                break

        with tqdm(total=total_partitions, desc=desc, unit="partition", position=0, disable=not progress) as bar:
            remaining = total_partitions
            while remaining > 0:
                try:
                    message = result_queue.get(timeout=0.2)
                except queue.Empty:
                    # No result yet — check whether any worker was killed without
                    # writing to the queue (OOM kill, SIGKILL, etc.).
                    for pid, (p, rel) in list(active.items()):
                        if not p.is_alive():
                            p.join()
                            if p.exitcode != 0:
                                del active[pid]
                                remaining -= 1
                                _close_inner(pid)
                                msg = f"{rel}: worker killed (OOM?), exit code {p.exitcode}"
                                errors.append(msg)
                                tqdm.write(f"  ERROR: {msg}", file=sys.stderr)
                                if db_conn is not None:
                                    _db_record_failure(db_conn, rel, f"exit code {p.exitcode}")
                                _spawn_next()
                                bar.update(1)
                                bar.set_postfix(migrated=migrated, skipped=skipped)
                    continue
                except KeyboardInterrupt:
                    for p, _ in active.values():
                        p.terminate()
                    for p, _ in active.values():
                        p.join()
                    raise

                if message[0] == "progress":
                    _, pid, rel, done, total = message
                    _update_inner(pid, rel, done, total)
                    continue

                _, pid, rel, result, messages, error = message
                _close_inner(pid)
                if pid in active:
                    active.pop(pid)[0].join()
                _spawn_next()
                remaining -= 1

                for msg in messages:
                    tqdm.write(msg)
                if error:
                    errors.append(f"{rel}: {error}")
                    tqdm.write(f"  ERROR migrating {rel}: {error}", file=sys.stderr)
                    if db_conn is not None:
                        _db_record_failure(db_conn, rel, error)
                elif result is None:
                    skipped += 1  # not wide format
                else:
                    data_type = result.get("data_type", "tabular")
                    dims = parse_path_dims(rel)
                    if {"variable", "time_unit", "grid"} <= dims.keys():
                        touched_datasets.add((dims["variable"], dims["time_unit"], dims["grid"]))
                    if result.get("skipped"):
                        skipped += 1  # already in long format on disk
                        if db_conn is not None:
                            _db_record(db_conn, rel, result["files"], result["geo_cols"], result["long_rows"], 0.0, 0, data_type)
                    elif "elapsed_s" in result:
                        migrated += 1
                        if db_conn is not None:
                            _db_record(db_conn, rel, result["files"], result["geo_cols"], result["long_rows"], result["elapsed_s"], result["peak_bytes"], data_type)
                    else:
                        migrated += 1  # dry run
                bar.update(1)
                bar.set_postfix(migrated=migrated, skipped=skipped)
    except KeyboardInterrupt:
        for p, _ in active.values():
            p.terminate()
        for p, _ in active.values():
            p.join()
        tqdm.write("\nInterrupted by user.", file=sys.stderr)
        sys.exit(1)
    finally:
        for pid in list(inner_bars):
            _close_inner(pid)
        if db_conn is not None:
            db_conn.close()

    # Coalesce the catalog for every dataset touched: write_parquet appended
    # per-write rows and compact_cog appended raster rows, so a final
    # compact_partition per variable/time_unit/grid collapses them into minimal
    # contiguous runs.  Idempotent (a single-file catalog partition is a no-op).
    if catalog_path is not None and touched_datasets:
        catalog = Catalog(catalog_path)
        for variable_, time_unit_, grid_ in sorted(touched_datasets):
            catalog.compact_partition(variable_, time_unit_, grid_)

    label = "Would migrate" if dry_run else "Migrated"
    print(f"\n{label} {migrated} unit(s), skipped {skipped}.")
    if errors:
        print(f"\n{len(errors)} unit(s) failed:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
