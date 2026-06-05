#!/usr/bin/env python3
"""Convert Parquet datasets from wide format to long format.

Uses LazyFrames and sink_parquet with pl.PartitionBy so the pipeline never
materialises the full unpivoted dataset in memory — large decadal files with
O(100 K) geography columns are handled without OOM.

Old layout:  variable=X/grid=G/time_unit=T/[hive dims.../]  *.parquet
             rows = one time step, cols = geography IDs "1".."N"

New layout:  variable=X/time_unit=T/[hive dims.../]calendar_year=Y/  *.parquet
             rows = (grid, geography_id, time step), cols = geography_id + value
"""

import re
import sys
from pathlib import Path

import click
import polars as pl

from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.utils import partition_columns

NUMERIC_COL = re.compile(r"^\d+$")


def parse_path_dims(path: Path, src: Path) -> dict[str, str]:
    """Return {name: raw_string} for every key=value segment in path relative to src."""
    return dict(part.split("=", 1) for part in path.relative_to(src).parts if "=" in part)


def get_partition_col_names(raw_path_dims: dict[str, str]) -> list[str]:
    """Determine hive partition column names for the new format.

    Builds a 1-row DataFrame from the known path dimension values and passes it
    to partition_columns() so the time_unit-conditional logic evaluates correctly.
    """
    dummy: dict[str, pl.Series] = {}
    for name, raw in raw_path_dims.items():
        dim = DIMENSIONS.get(name)
        if dim:
            dummy[name] = pl.Series([raw]).cast(dim.dtype)
    # calendar_year is always a partition column; supply a placeholder if the
    # source path does not include it (e.g. daily data — it will be derived).
    if "calendar_year" not in dummy:
        dummy["calendar_year"] = pl.Series([2030], dtype=pl.Int32)
    return partition_columns(pl.DataFrame(dummy))


def convert_directory(leaf_dir: Path, src: Path, dst: Path) -> None:
    parquet_files = sorted(leaf_dir.glob("*.parquet"))
    if not parquet_files:
        return

    # Scan all files in this partition directory as a single LazyFrame.
    lf = pl.scan_parquet([str(p) for p in parquet_files])
    initial_cols = set(lf.collect_schema().names())

    # Identify geography (numeric-named) columns.
    geo_cols = [c for c in initial_cols if NUMERIC_COL.match(c)]
    if not geo_cols:
        click.echo(
            f"  WARNING: no geography columns in {leaf_dir.relative_to(src)}, skipping",
            err=True,
        )
        return

    # Inject dimension values encoded in the directory path that are absent from
    # the file itself (daily data encodes model_group_id/scenario_id/aggregate_type
    # only in the directory name, not in the parquet columns).
    raw_path_dims = parse_path_dims(leaf_dir, src)
    add_exprs: list[pl.Expr] = []
    added_names: set[str] = set()
    for name, raw in raw_path_dims.items():
        if name not in initial_cols:
            dim = DIMENSIONS.get(name)
            expr = pl.lit(raw).cast(dim.dtype) if dim else pl.lit(raw)
            add_exprs.append(expr.alias(name))
            added_names.add(name)
    if add_exprs:
        lf = lf.with_columns(*add_exprs)

    # Derive calendar_year from calendar_day when absent (daily / monthly).
    available_cols = initial_cols | added_names
    if "calendar_year" not in available_cols and "calendar_day" in available_cols:
        lf = lf.with_columns(pl.col("calendar_day").dt.year().cast(pl.Int32).alias("calendar_year"))
        available_cols = available_cols | {"calendar_year"}

    # Unpivot: geography ID columns → geography_id (Int32) + value.
    index_cols = [c for c in lf.collect_schema().names() if c not in geo_cols]
    lf = lf.unpivot(
        on=geo_cols,
        index=index_cols,
        variable_name="geography_id",
        value_name="value",
    ).with_columns(pl.col("geography_id").cast(pl.Int32))

    # Sort: partition columns first (ensures each partition's rows are contiguous
    # for the sink), then grid, geography_id, date within each partition file.
    part_col_names = get_partition_col_names(raw_path_dims)
    date_col = "calendar_day" if "calendar_day" in available_cols else "calendar_year"
    trailing = [k for k in ["grid", "geography_id", date_col] if k not in part_col_names]
    lf = lf.sort(*part_col_names, *trailing)

    # Write to disk with hive partitioning. include_key=False keeps partition
    # column values only in the directory path, not duplicated inside the file.
    lf.sink_parquet(
        pl.PartitionBy(str(dst), key=part_col_names, include_key=False),
        mkdir=True,
        compression="zstd",
        compression_level=6,
        row_group_size=500_000,
    )


@click.command()
@click.argument("src", default="data", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("dst", type=click.Path(file_okay=False, path_type=Path))
def main(src: Path, dst: Path) -> None:
    """Convert wide-format Parquet data in SRC to long format in DST.

    SRC   root of the old wide-format data tree  [default: data]
    DST   root of the new long-format data tree
    """
    dst.mkdir(parents=True, exist_ok=True)

    leaf_dirs = sorted({p.parent for p in src.rglob("*.parquet") if "clara" not in str(p)})
    n = len(leaf_dirs)
    click.echo(f"Found {n} partition director{'y' if n == 1 else 'ies'} to convert.")

    errors: list[tuple[str, Exception]] = []
    for i, leaf_dir in enumerate(leaf_dirs, 1):
        label = str(leaf_dir.relative_to(src))
        click.echo(f"[{i}/{n}] {label}")
        try:
            convert_directory(leaf_dir, src, dst)
        except Exception as exc:
            click.echo(f"  ERROR: {exc}", err=True)
            errors.append((label, exc))

    if errors:
        click.echo(
            f"\n{len(errors)} director{'y' if len(errors) == 1 else 'ies'} failed:",
            err=True,
        )
        for label, exc in errors:
            click.echo(f"  {label}: {exc}", err=True)
        sys.exit(1)

    click.echo("Done.")


if __name__ == "__main__":
    main()
