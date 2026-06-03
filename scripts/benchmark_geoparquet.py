#!/usr/bin/env python3
"""
Benchmark comparing wide-format Parquet vs. long-format storage configurations
to evaluate the feasibility of migrating from wide to long format.

Tests three primary query patterns:
  1. Map query:   1 date × all geographies        (portal map)
  2. Chart query: all dates × 1 geography         (portal chart)
  3. Model query: all dates × all geographies     (full model read)

Storage configurations tested:
  wide     - current wide-format production files (read-only, no write)
  long_geo - long format, single file, sorted by (geography_id, date)
  year_geo - long format, year-partitioned, geography-sorted within each year

Key finding: year_geo compresses better than long_geo because splitting by year
limits calendar_day to ~730 unique values per row group vs 37,986 in long_geo.
The repeating annual date sequence is far more compressible than the full plan
period sequence.

Usage:
    python scripts/benchmark_geoparquet.py \\
        --variable sal \\
        --grid hydro_compartment_v001 \\
        --time-unit daily \\
        --scenario-id 7 \\
        --model-group-id 500 \\
        --output-dir /tmp/geoparquet_benchmark \\
        --runs 3 \\
        --threads 2
"""

import argparse
import gc
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

# POLARS_MAX_THREADS must be set before polars initialises its thread pool.
# Pre-parse --threads here so we can export it before the polars import.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=None)
_pre_threads = _pre.parse_known_args()[0].threads
if _pre_threads:
    os.environ["POLARS_MAX_THREADS"] = str(_pre_threads)

import polars as pl
import polars.selectors as cs
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))

from cpra.mp.data import read_parquet
from cpra.mp.data.dimensions import DIMENSIONS, DATASET_DIMENSIONS

ROW_GROUP_SIZE = 500_000
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 6

# ── helpers ───────────────────────────────────────────────────────────────────


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_048_576


def run_timed(fn, n_runs: int) -> tuple[list[float], float]:
    times = []
    for _ in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return times, statistics.median(times)


def fmt_times(times: list[float]) -> str:
    med = statistics.median(times)
    return f"{med:>7.0f} ms  [{min(times):.0f}–{max(times):.0f}]"


def pick_target_date(df: pl.DataFrame):
    dates = df["calendar_day"].sort()
    return dates[len(dates) // 2]


def pick_target_geo(geo_cols: list[str]) -> str:
    return geo_cols[len(geo_cols) // 2]


def dim_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.isdigit() and c != "year"]


def geo_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.isdigit()]


# ── data preparation ──────────────────────────────────────────────────────────


def load_wide(kwargs: dict) -> pl.DataFrame:
    print("  Reading wide-format data …", end=" ", flush=True)
    df = read_parquet(**kwargs).collect()
    print(f"{len(df):,} rows × {len(geo_cols(df)):,} geography columns")
    return df


def wide_to_long(wide_df: pl.DataFrame) -> pl.DataFrame:
    dims = dim_cols(wide_df)
    geos = geo_cols(wide_df)
    return (
        wide_df
        .unpivot(index=dims, on=geos, variable_name="geography_id", value_name="value")
        .with_columns(pl.col("geography_id").cast(pl.Int32))
    )


def add_year(df: pl.DataFrame) -> pl.DataFrame:
    if "calendar_day" in df.columns:
        return df.with_columns(pl.col("calendar_day").dt.year().cast(pl.Int32).alias("year"))
    if "calendar_year" in df.columns:
        return df.with_columns(pl.col("calendar_year").cast(pl.Int32).alias("year"))
    raise ValueError("No date column found (calendar_day or calendar_year)")


# ── writer ────────────────────────────────────────────────────────────────────


def write_long(
    df: pl.DataFrame,
    path: Path,
    sort_by: list[str],
    partition_by_year: bool = False,
):
    """Write a sorted long-format DataFrame to Parquet."""
    path.mkdir(parents=True, exist_ok=True)
    df = df.sort(sort_by)

    if partition_by_year:
        for (year,), group in df.group_by(["year"], maintain_order=True):
            year_dir = path / f"year={year}"
            year_dir.mkdir(exist_ok=True)
            pq.write_table(
                group.to_arrow(),
                str(year_dir / "data.parquet"),
                compression=COMPRESSION,
                compression_level=COMPRESSION_LEVEL,
                row_group_size=ROW_GROUP_SIZE,
                write_statistics=True,
            )
    else:
        pq.write_table(
            df.to_arrow(),
            str(path / "data.parquet"),
            compression=COMPRESSION,
            compression_level=COMPRESSION_LEVEL,
            row_group_size=ROW_GROUP_SIZE,
            write_statistics=True,
        )


# ── benchmarks ────────────────────────────────────────────────────────────────


def bench_wide(read_kwargs: dict, target_date, target_geo: str, n_runs: int):
    def map_query():
        read_parquet(**{**read_kwargs, "calendar_day": target_date}).collect()

    def chart_query():
        read_parquet(**read_kwargs).select(cs.matches(r"^[a-z]"), pl.col(target_geo)).collect()

    def model_query():
        read_parquet(**read_kwargs).collect()

    return {
        "map": run_timed(map_query, n_runs),
        "chart": run_timed(chart_query, n_runs),
        "model": run_timed(model_query, n_runs),
    }


def bench_long(path: Path, target_date, target_geo: int, n_runs: int, hive: bool = False):
    scan_kw = dict(hive_partitioning=hive)

    def map_query():
        pl.scan_parquet(path, **scan_kw).filter(pl.col("calendar_day") == target_date).collect()

    def chart_query():
        pl.scan_parquet(path, **scan_kw).filter(pl.col("geography_id") == target_geo).collect()

    def model_query():
        pl.scan_parquet(path, **scan_kw).collect()

    return {
        "map": run_timed(map_query, n_runs),
        "chart": run_timed(chart_query, n_runs),
        "model": run_timed(model_query, n_runs),
    }


# ── reporting ─────────────────────────────────────────────────────────────────


def print_results(results: dict[str, dict]):
    col_w = 18
    hdr = (
        f"{'Configuration':<{col_w}}  {'Map query':^25}  "
        f"{'Chart query':^25}  {'Model query':^25}  {'Size (MB)':>10}"
    )
    sep = "-" * len(hdr)
    print()
    print(sep)
    print(hdr)
    print(sep)
    for name, data in results.items():
        map_t, _ = data["map"]
        chart_t, _ = data["chart"]
        model_t, _ = data["model"]
        size_mb = data.get("size_mb", float("nan"))
        size_str = f"{size_mb:.1f}" if size_mb == size_mb else "n/a"
        print(
            f"{name:<{col_w}}  {fmt_times(map_t):^25}  "
            f"{fmt_times(chart_t):^25}  {fmt_times(model_t):^25}  {size_str:>10}"
        )
    print(sep)
    print("  Timings: median ms  [min–max across runs]")
    print()


# ── main ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark wide vs. long-format Parquet storage for CPRA data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variable", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--time-unit", required=True, dest="time_unit")

    for dim in DIMENSIONS.values():
        if dim.name not in DATASET_DIMENSIONS and dim.type == "partition":
            p.add_argument(
                f"--{dim.name.replace('_', '-')}",
                type=int,
                dest=dim.name,
                help=f"Filter: {dim.label}",
            )

    p.add_argument("--output-dir", default="/tmp/geoparquet_benchmark")
    p.add_argument("--runs", type=int, default=3, help="Benchmark repetitions per query")
    p.add_argument(
        "--threads", type=int, default=None,
        help="Polars thread pool size (set via POLARS_MAX_THREADS before import; "
             "default: all available cores)",
    )
    p.add_argument("--keep-files", action="store_true", help="Retain benchmark files after run")
    return p.parse_args()


def build_read_kwargs(args) -> dict:
    kwargs = dict(variable=args.variable, grid=args.grid, time_unit=args.time_unit)
    for dim in DIMENSIONS.values():
        if dim.name not in DATASET_DIMENSIONS and dim.type == "partition":
            val = getattr(args, dim.name, None)
            if val is not None:
                kwargs[dim.name] = val
    return kwargs


def main():
    args = parse_args()
    read_kwargs = build_read_kwargs(args)
    output_dir = Path(args.output_dir)

    print(f"\n{'='*70}")
    print("Long-Format Parquet Benchmark")
    print(f"{'='*70}")
    print(f"  Variable : {args.variable}")
    print(f"  Grid     : {args.grid}")
    print(f"  Time unit: {args.time_unit}")
    filters = {k: v for k, v in read_kwargs.items() if k not in DATASET_DIMENSIONS}
    print(f"  Filters  : {', '.join(f'{k}={v}' for k, v in filters.items()) or 'none'}")
    print(f"  Runs     : {args.runs}")
    print(f"  Threads  : {pl.thread_pool_size()} (use --threads N to limit)")
    print(f"  Output   : {output_dir}")
    print()

    # ── Step 1: load wide data ────────────────────────────────────────────────
    print("Step 1/4: Loading wide-format data")
    wide_df = load_wide(read_kwargs)
    g_cols = geo_cols(wide_df)

    target_date = pick_target_date(wide_df)
    target_geo_str = pick_target_geo(g_cols)
    target_geo_int = int(target_geo_str)
    n_geos = len(g_cols)

    print(f"  Target date      : {target_date}")
    print(f"  Target geography : {target_geo_str}")
    print(f"  Wide shape       : {len(wide_df):,} rows × {n_geos:,} geography columns")

    # ── Step 2: convert to long format ───────────────────────────────────────
    print("\nStep 2/4: Converting to long format …", end=" ", flush=True)
    long_df = wide_to_long(wide_df)
    long_year_df = add_year(long_df)
    days_per_year = int(
        long_year_df.group_by("year").agg(pl.len()).select(pl.col("len").max()).item()
    ) // n_geos
    print(f"{len(long_df):,} rows × {len(long_df.columns)} columns")

    # ── Step 3: write benchmark files ────────────────────────────────────────
    print("\nStep 3/4: Writing benchmark files")
    output_dir.mkdir(parents=True, exist_ok=True)

    configs: dict[str, tuple[Path, bool]] = {}

    p = output_dir / "long_geo"
    print(f"  Writing long_geo  (single file, geo-sorted) …", end=" ", flush=True)
    write_long(long_df, p, sort_by=["geography_id", "calendar_day"])
    print(f"{dir_size_mb(p):.1f} MB")
    configs["long_geo"] = (p, False)

    p = output_dir / "year_geo"
    print(f"  Writing year_geo  (year-partitioned, geo-sorted) …", end=" ", flush=True)
    write_long(long_year_df, p, sort_by=["year", "geography_id", "calendar_day"], partition_by_year=True)
    print(f"{dir_size_mb(p):.1f} MB")
    configs["year_geo"] = (p, True)

    # ── Step 4: run benchmarks ────────────────────────────────────────────────
    print("\nStep 4/4: Running benchmarks …")
    results: dict[str, dict] = {}

    print("  Benchmarking: wide (baseline)")
    r = bench_wide(read_kwargs, target_date, target_geo_str, args.runs)
    r["size_mb"] = float("nan")
    results["wide (baseline)"] = r

    for name, (path, is_hive) in configs.items():
        print(f"  Benchmarking: {name}")
        r = bench_long(path, target_date, target_geo_int, args.runs, hive=is_hive)
        r["size_mb"] = dir_size_mb(path)
        results[name] = r

    # ── Report ────────────────────────────────────────────────────────────────
    print_results(results)

    print("Notes:")
    print(f"  Target date: {target_date}  |  Target geography: {target_geo_int}")
    print(f"  Row group size: {ROW_GROUP_SIZE:,}  |  Compression: {COMPRESSION} level {COMPRESSION_LEVEL}")
    print(f"  Long-format row count: {len(long_df):,}  |  Geographies: {n_geos:,}")
    print(f"  Threads: {pl.thread_pool_size()}  |  Use --threads N to isolate format differences from parallelism")
    print(f"  year_geo < long_geo: year files limit calendar_day to ~{days_per_year} unique values/row-group")
    print(f"    vs {len(wide_df):,} in long_geo — the repeating {days_per_year}-day sequence compresses far better")
    print(f"  Model query: long format materialises ~3-4× more data (calendar_day + geography_id explicit)")
    if args.runs < 3:
        print("  Tip: use --runs 3 or more for stable medians")
    print()

    if not args.keep_files:
        print(f"Cleaning up {output_dir} …")
        shutil.rmtree(output_dir, ignore_errors=True)
    else:
        print(f"Benchmark files retained at {output_dir}")


if __name__ == "__main__":
    main()
