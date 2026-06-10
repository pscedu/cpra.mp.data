"""Migrate tabular Parquet datasets from wide format to long format.

Each existing wide-format file (one row per time step, grid_cell IDs as column
names) is converted to long format (one row per time-step × grid_cell pair, with
explicit `grid_cell` and `value` columns), then rewritten in place with the
correct sort order and row group size for the `long_year_geo` layout.

Usage
-----
    # Dry-run: show what would be migrated without writing anything
    python scripts/migrate_to_long_format.py --dry-run

    # Migrate a single dataset
    python scripts/migrate_to_long_format.py \\
        --variable sal --grid hydro_compartment_v001 --time-unit daily

    # Migrate all datasets
    python scripts/migrate_to_long_format.py

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


def migrate_file(parquet_path: Path, base_path: Path, dry_run: bool = False) -> bool:
    """Convert one wide-format Parquet file to long format in place.

    Returns True if the file was (or would be) migrated, False if it was
    already in long format or could not be processed.
    """
    if not _is_wide(parquet_path):
        return False

    print(f"  {'[DRY RUN] ' if dry_run else ''}Migrating: {parquet_path.relative_to(base_path)}")

    if dry_run:
        schema = pl.read_parquet_schema(parquet_path)
        geo_cols = [n for n in schema if n[:1].isdigit()]
        print(f"    {len(geo_cols)} grid_cell columns → long format")
        return True

    t0 = time.perf_counter()

    schema = pl.read_parquet_schema(parquet_path)
    geo_cols = [n for n in schema if n[:1].isdigit()]

    # Rename the source to a temp name before scanning it.  The streaming
    # write targets the same Hive partition directory, so leaving the source
    # at its original path risks a filename collision with the output file.
    # Renaming eliminates the conflict; on failure the original is restored.
    tmp_path = parquet_path.with_suffix(".migrating")
    parquet_path.rename(tmp_path)

    try:
        lf = pl.scan_parquet(tmp_path)

        # Inject dimension values encoded in the Hive partition path as
        # literal columns — they are not stored inside the leaf file.
        hive_dims = _dimension_cols_from_hive(parquet_path)
        if hive_dims:
            lf = lf.with_columns(**{k: pl.lit(v) for k, v in hive_dims.items()})

        # write_parquet detects wide-format columns (numeric names), unpivots
        # to long internally, sorts, and streams via sink_parquet so the full
        # frame is never materialized.
        write_parquet(lf, base_path=base_path, validate=False, sort=True)
    except Exception:
        tmp_path.rename(parquet_path)
        raise

    tmp_path.unlink()

    elapsed = time.perf_counter() - t0
    print(f"    {len(geo_cols)} grid_cell columns → long format in {elapsed:.1f}s")
    return True


def find_parquet_files(base_path: Path, variable=None, grid=None, time_unit=None) -> list[Path]:
    """Find all wide-format Parquet files under base_path, optionally filtered."""
    pattern_parts = []
    for dim_name, value in [("variable", variable), ("grid", grid), ("time_unit", time_unit)]:
        pattern_parts.append(f"{dim_name}={value}" if value else f"{dim_name}=*")

    # Walk and collect all .parquet files under the matching dataset directories.
    paths = []
    for dataset_dir in base_path.glob("/".join(pattern_parts)):
        paths.extend(dataset_dir.rglob("*.parquet"))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variable", help="Migrate only this variable")
    parser.add_argument("--grid", help="Migrate only this grid")
    parser.add_argument("--time-unit", dest="time_unit", help="Migrate only this time unit")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without writing")
    parser.add_argument("--data-path", default=str(BASE_PATH), help=f"Base data path (default: {BASE_PATH})")
    args = parser.parse_args()

    base_path = Path(args.data_path)
    paths = find_parquet_files(base_path, args.variable, args.grid, args.time_unit)

    if not paths:
        print("No Parquet files found matching the given filters.")
        return

    print(f"Found {len(paths)} Parquet file(s) to inspect.")
    migrated = 0
    skipped = 0

    for path in paths:
        try:
            if migrate_file(path, base_path=base_path, dry_run=args.dry_run):
                migrated += 1
            else:
                skipped += 1
        except Exception as exc:
            print(f"  ERROR migrating {path}: {exc}", file=sys.stderr)

    label = "Would migrate" if args.dry_run else "Migrated"
    print(f"\n{label} {migrated} file(s), skipped {skipped} already-long file(s).")


if __name__ == "__main__":
    main()
