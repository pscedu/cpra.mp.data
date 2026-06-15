"""Migrate tabular Parquet datasets from wide format to long format.

All wide-format files in the same partition directory are processed together
and written as a single long-format Parquet file, so each output partition
contains exactly one file.  The original wide-format files are left untouched.

Usage
-----
    # Dry-run: show what would be migrated without writing anything
    python scripts/migrate_to_long_format.py --output-path /path/to/output --dry-run

    # Migrate a single dataset
    python scripts/migrate_to_long_format.py \\
        --output-path /path/to/output \\
        --variable sal --grid hydro_compartment_v001 --time-unit daily

    # Migrate all datasets
    python scripts/migrate_to_long_format.py --output-path /path/to/output

After verifying correctness on one dataset, run without --dry-run to apply.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

# Add the project root to sys.path so the package is importable when the
# script is run from any working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpra.mp.data.config import BASE_PATH
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.tabular import read_parquet, write_parquet


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


def migrate_partition(parquet_paths: list[Path], base_path: Path, output_path: Path, dry_run: bool = False) -> bool:
    """Convert all wide-format Parquet files in one partition to a single long-format file.

    All paths must share the same parent directory (partition).  Returns True if
    the partition was (or would be) migrated, False if it was already in long format.
    """
    schema = _wide_schema(parquet_paths[0])
    if schema is None:
        return False

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
            tqdm.write(f"  Skipping {rel_dir} ({row_count:,} rows already written)")
            return False
        tqdm.write(f"  Output incomplete ({existing_rows:,}/{row_count:,} rows), clearing: {rel_dir}")
        if not dry_run:
            for p in existing_files:
                Path(p).unlink()

    tqdm.write(
        f"  {'[DRY RUN] ' if dry_run else ''}Migrating: {rel_dir} "
        f"({n} file{'s' if n > 1 else ''}, {len(geo_cols)} grid_cell cols, {row_count:,} long rows)"
    )

    if dry_run:
        return True

    t0 = time.perf_counter()

    lf = pl.scan_parquet(parquet_paths)

    # write_parquet detects wide-format columns (numeric names), unpivots to
    # long internally, sorts, and streams via sink_parquet so the full frame
    # is never materialized.
    write_parquet(lf, base_path=output_path, validate=False, sort=True, **hive_dims)

    elapsed = time.perf_counter() - t0
    tqdm.write(f"    done in {elapsed:.1f}s")
    return True


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


def estimate_partitions(partitions: dict[Path, list[Path]], base_path: Path, progress: bool = True) -> None:
    """Print a memory-estimate table for all wide-format partitions, sorted by estimated peak memory.

    Sizes are derived from schema dtypes (num_rows × Σ dtype.itemsize), which reflects the
    actual decoded Arrow representation rather than the Parquet-encoded size.

    Est. peak = wide_in_memory + long_in_memory × 2, where the ×2 accounts for sort
    indices and a permutation buffer during the sort step in write_parquet.
    """
    rows = []
    with tqdm(total=len(partitions), desc="Estimating", unit="partition", disable=not progress) as bar:
        for paths in partitions.values():
            schema = _wide_schema(paths[0])
            bar.update(1)
            if schema is None:
                continue

            geo_cols = [col for col in schema if col[:1].isdigit()]
            alpha_cols = [col for col in schema if not col[:1].isdigit()]
            value_dtype = schema[geo_cols[0]]

            wide_rows = sum(pq.read_metadata(p).num_rows for p in paths)
            long_rows = wide_rows * len(geo_cols)

            # Decoded in-memory size of the wide DataFrame.
            wide_in_memory = wide_rows * sum(_dtype_itemsize(schema[c]) for c in schema)

            # Decoded in-memory size of the long DataFrame after unpivot + select:
            # all alpha dimension columns (repeated long_rows times) + grid_cell (Int32) + value.
            long_bytes_per_row = (
                sum(_dtype_itemsize(schema[c]) for c in alpha_cols)
                + _dtype_itemsize(pl.Int32)  # grid_cell cast from column name
                + _dtype_itemsize(value_dtype)  # value column
            )
            long_in_memory = long_rows * long_bytes_per_row

            # Sort overhead: ~1× long data for indices and permutation buffer.
            est_peak = wide_in_memory + long_in_memory * 2

            rows.append(
                {
                    "partition": str(paths[0].parent.relative_to(base_path)),
                    "files": len(paths),
                    "long_rows": long_rows,
                    "wide_in_memory": wide_in_memory,
                    "est_peak": est_peak,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-path", help="Root directory for long-format output (required unless --estimate)")
    parser.add_argument(
        "--estimate", action="store_true", help="Print a memory-estimate table sorted by peak memory and exit"
    )
    parser.add_argument("--variable", help="Migrate only this variable")
    parser.add_argument("--grid", help="Migrate only this grid")
    parser.add_argument("--time-unit", dest="time_unit", help="Migrate only this time unit")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without writing")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bar (use --no-progress for batch jobs)",
    )
    parser.add_argument("--data-path", default=str(BASE_PATH), help=f"Base data path (default: {BASE_PATH})")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of partitions to process in parallel (default: 1). "
        "Keep low (2–4) — Polars already saturates CPUs within each partition.",
    )
    args = parser.parse_args()

    if not args.estimate and not args.output_path:
        parser.error("--output-path is required unless --estimate is set")

    base_path = Path(args.data_path).resolve()
    partitions = find_partitions(base_path, args.variable, args.grid, args.time_unit)

    if not partitions:
        print("No Parquet files found matching the given filters.")
        return

    if args.estimate:
        estimate_partitions(partitions, base_path, progress=args.progress)
        return

    output_path = Path(args.output_path).resolve()
    total_files = sum(len(v) for v in partitions.values())
    total_partitions = len(partitions)
    print(f"Found {total_files} Parquet file(s) across {total_partitions} partition(s).")
    migrated = 0
    skipped = 0

    def _run(paths):
        return migrate_partition(paths, base_path=base_path, output_path=output_path, dry_run=args.dry_run)

    desc = "[DRY RUN]" if args.dry_run else "Migrating"
    with tqdm(total=total_partitions, desc=desc, unit="partition", disable=not args.progress) as bar:
        if args.workers == 1:
            for paths in partitions.values():
                try:
                    if _run(paths):
                        migrated += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    tqdm.write(f"  ERROR migrating {paths[0].parent}: {exc}", file=sys.stderr)
                bar.update(1)
                bar.set_postfix(migrated=migrated, skipped=skipped)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_run, paths): paths for paths in partitions.values()}
                for future in as_completed(futures):
                    paths = futures[future]
                    try:
                        if future.result():
                            migrated += 1
                        else:
                            skipped += 1
                    except Exception as exc:
                        tqdm.write(f"  ERROR migrating {paths[0].parent}: {exc}", file=sys.stderr)
                    bar.update(1)
                    bar.set_postfix(migrated=migrated, skipped=skipped)

    label = "Would migrate" if args.dry_run else "Migrated"
    print(f"\n{label} {migrated} partition(s), skipped {skipped} already-long partition(s).")


if __name__ == "__main__":
    main()
