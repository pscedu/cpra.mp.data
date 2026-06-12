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
from pathlib import Path

import polars as pl

# Add the project root to sys.path so the package is importable when the
# script is run from any working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpra.mp.data.config import BASE_PATH
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.tabular import write_parquet


def _is_wide(path: Path) -> bool:
    """Return True if the Parquet file at path is in wide format."""
    schema = pl.read_parquet_schema(path)
    return any(name[:1].isdigit() for name in schema)


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
    if not _is_wide(parquet_paths[0]):
        return False

    partition_dir = parquet_paths[0].parent
    rel_dir = partition_dir.relative_to(base_path)
    n = len(parquet_paths)
    print(f"  {'[DRY RUN] ' if dry_run else ''}Migrating: {rel_dir} ({n} file{'s' if n > 1 else ''})")

    schema = pl.read_parquet_schema(parquet_paths[0])
    geo_cols = [n for n in schema if n[:1].isdigit()]
    print(f"    {len(geo_cols)} grid_cell columns → long format")

    if dry_run:
        return True

    t0 = time.perf_counter()

    lf = pl.scan_parquet(parquet_paths)

    # Inject dimension values encoded in the Hive partition path as literal
    # columns — they are not stored inside the leaf file itself.
    hive_dims = _dimension_cols_from_hive(parquet_paths[0])
    if hive_dims:
        lf = lf.with_columns(**{k: pl.lit(v) for k, v in hive_dims.items()})

    # write_parquet detects wide-format columns (numeric names), unpivots to
    # long internally, sorts, and streams via sink_parquet so the full frame
    # is never materialized.
    write_parquet(lf, base_path=output_path, validate=False, sort=True)

    elapsed = time.perf_counter() - t0
    print(f"    done in {elapsed:.1f}s")
    return True


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
    parser.add_argument("--output-path", required=True, help="Root directory for long-format output")
    parser.add_argument("--variable", help="Migrate only this variable")
    parser.add_argument("--grid", help="Migrate only this grid")
    parser.add_argument("--time-unit", dest="time_unit", help="Migrate only this time unit")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without writing")
    parser.add_argument("--data-path", default=str(BASE_PATH), help=f"Base data path (default: {BASE_PATH})")
    args = parser.parse_args()

    base_path = Path(args.data_path)
    output_path = Path(args.output_path)
    partitions = find_partitions(base_path, args.variable, args.grid, args.time_unit)

    if not partitions:
        print("No Parquet files found matching the given filters.")
        return

    total_files = sum(len(v) for v in partitions.values())
    print(f"Found {total_files} Parquet file(s) across {len(partitions)} partition(s).")
    migrated = 0
    skipped = 0

    for paths in partitions.values():
        try:
            if migrate_partition(paths, base_path=base_path, output_path=output_path, dry_run=args.dry_run):
                migrated += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"  ERROR migrating {paths[0].parent}: {exc}", file=sys.stderr)

    label = "Would migrate" if args.dry_run else "Migrated"
    print(f"\n{label} {migrated} partition(s), skipped {skipped} already-long partition(s).")


if __name__ == "__main__":
    main()
