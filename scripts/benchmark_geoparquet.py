#!/usr/bin/env python3
"""
Benchmark comparing wide-format Parquet vs. long-format storage configurations
to evaluate the feasibility of migrating to GeoParquet.

Tests three primary query patterns:
  1. Map query:   1 date × all geographies        (portal map)
  2. Chart query: all dates × 1 geography         (portal chart)
  3. Model query: all dates × all geographies     (full model read)

Storage configurations tested:
  wide           - current wide-format production files (read-only, no write)
  long_date      - long format, single file, sorted by (date, geography_id)
  long_geo       - long format, single file, sorted by (geography_id, date)
  year_date      - long format, year-partitioned, date-sorted within each year
  year_geo       - long format, year-partitioned, geography-sorted within each year
  geoparquet     - year_date + embedded geometry column (requires geopandas)
  geoparquet_geo - year_geo + embedded geometry column (requires geopandas)

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

    # Skip slow single-file configs and geometry join:
    python scripts/benchmark_geoparquet.py ... --skip-unpartitioned --skip-geoparquet
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

try:
    import geopandas as gpd

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# Bloom filter support was added to pq.write_table after pyarrow 21.0.0.
# Detect at import time so we can skip the year_bloom config gracefully.
try:
    import inspect as _inspect

    HAS_BLOOM = "bloom_filter_options" in _inspect.signature(pq.write_table).parameters
except Exception:
    HAS_BLOOM = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from cpra.mp.data import read_parquet
from cpra.mp.data.dimensions import DIMENSIONS, DATASET_DIMENSIONS

GEOMETRY_PATH = Path(__file__).parent.parent / "cpra/mp/data/geometry"
ROW_GROUP_SIZE = 500_000
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 6

# ── helpers ──────────────────────────────────────────────────────────────────


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


# ── data preparation ─────────────────────────────────────────────────────────


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


# ── writers ───────────────────────────────────────────────────────────────────


def _pq_write(table, file_path: Path, bloom_filter_options=None, write_statistics=True, **extra):
    """Write a single PyArrow table to one Parquet file."""
    kwargs = dict(
        compression=COMPRESSION,
        compression_level=COMPRESSION_LEVEL,
        row_group_size=ROW_GROUP_SIZE,
        write_statistics=write_statistics,
        **extra,
    )
    if bloom_filter_options:
        kwargs["bloom_filter_options"] = bloom_filter_options
    pq.write_table(table, str(file_path), **kwargs)


def write_long(
    df: pl.DataFrame,
    path: Path,
    sort_by: list[str],
    partition_by_year: bool = False,
    bloom_filter_options: dict | None = None,
):
    """Write a long-format DataFrame to Parquet using pq.write_table."""
    path.mkdir(parents=True, exist_ok=True)
    df = df.sort(sort_by)

    if partition_by_year:
        for (year,), group in df.group_by(["year"], maintain_order=True):
            year_dir = path / f"year={year}"
            year_dir.mkdir(exist_ok=True)
            _pq_write(group.to_arrow(), year_dir / "data.parquet", bloom_filter_options)
    else:
        _pq_write(df.to_arrow(), path / "data.parquet", bloom_filter_options)


def write_geoparquet(
    long_year_df: pl.DataFrame,
    path: Path,
    grid: str,
    sort_by: list[str],
    bloom_filter_options: dict | None,
) -> bool:
    import json

    if not HAS_GEOPANDAS:
        return False

    fgb_path = GEOMETRY_PATH / f"{grid}.fgb"
    if not fgb_path.exists():
        print(f"    Warning: geometry file not found: {fgb_path}")
        return False

    # Read geometry once and convert to WKB bytes in a small polars table
    # (n_geoms rows).  This avoids a huge pandas merge across all long-format rows.
    print("    Reading geometry and extracting WKB …", end=" ", flush=True)
    geom_gdf = gpd.read_file(str(fgb_path), engine="pyogrio")[["id", "geometry"]]
    wkb_pl = pl.DataFrame({
        "geography_id": pl.Series(geom_gdf["id"].astype("int32").tolist(), dtype=pl.Int32),
        "geometry": pl.Series(list(geom_gdf.geometry.to_wkb()), dtype=pl.Binary),
    })
    # Compute total WKB bytes so we can set dictionary_pagesize_limit large enough
    # to hold all unique geometries.  Parquet's default (1 MB) is smaller than the
    # total WKB for most grids, causing a silent fallback to plain encoding — which
    # stores the full geometry blob on every row and inflates the file enormously.
    total_wkb_bytes = int(wkb_pl["geometry"].map_elements(len, return_dtype=pl.Int64).sum())
    # 25% headroom; Parquet overhead is small but non-zero.
    dict_page_limit = int(total_wkb_bytes * 1.25)
    print(f"{len(wkb_pl):,} geometries  ({total_wkb_bytes / 1_048_576:.1f} MB WKB, "
          f"dict_pagesize_limit={dict_page_limit // 1_048_576} MB)")

    # Non-geometry columns to write statistics for (geometry statistics are
    # expensive — min/max WKB per row group — and never used for filtering).
    non_geo_cols = [c for c in long_year_df.columns if c != "geometry"]

    # Build the GeoParquet `geo` metadata block (identical for every year file).
    geo_meta = json.dumps({
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": [],
                "crs": geom_gdf.crs.to_json_dict(),
            }
        }
    }).encode()

    # Write one year at a time: polars filter → join WKB (1,657-row table) →
    # Arrow → attach geo metadata → pq.write_table.  Peak memory stays
    # proportional to one year's data rather than the full long-format table.
    path.mkdir(parents=True, exist_ok=True)
    years = sorted(long_year_df["year"].unique().to_list())
    n_years = len(years)

    for i, year in enumerate(years, 1):
        print(f"\r    Year {year}  ({i}/{n_years}) …", end="", flush=True)
        year_dir = path / f"year={year}"
        year_dir.mkdir(exist_ok=True)

        year_with_geom = (
            long_year_df
            .filter(pl.col("year") == year)
            .join(wkb_pl, on="geography_id", how="left")
            .sort(sort_by)
        )
        table = year_with_geom.to_arrow()
        existing_meta = table.schema.metadata or {}
        table = table.replace_schema_metadata({**existing_meta, b"geo": geo_meta})
        _pq_write(
            table,
            year_dir / "data.parquet",
            bloom_filter_options,
            dictionary_pagesize_limit=dict_page_limit,
            write_statistics=non_geo_cols,
        )

    print(f"\r    {n_years} year partitions written.                              ")
    return True


# ── benchmarks ───────────────────────────────────────────────────────────────


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


def bench_long(
    path: Path,
    target_date,
    target_geo: int,
    n_runs: int,
    hive: bool = False,
    exclude_cols: list[str] | None = None,
):
    scan_kw = dict(hive_partitioning=hive)

    def _scan():
        lf = pl.scan_parquet(path, **scan_kw)
        if exclude_cols:
            lf = lf.select(pl.exclude(exclude_cols))
        return lf

    def map_query():
        _scan().filter(pl.col("calendar_day") == target_date).collect()

    def chart_query():
        _scan().filter(pl.col("geography_id") == target_geo).collect()

    def model_query():
        _scan().collect()

    return {
        "map": run_timed(map_query, n_runs),
        "chart": run_timed(chart_query, n_runs),
        "model": run_timed(model_query, n_runs),
    }


# ── reporting ─────────────────────────────────────────────────────────────────


def print_results(results: dict[str, dict]):
    col_w = 22
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
    p.add_argument(
        "--skip-unpartitioned",
        action="store_true",
        help="Skip single-file long_date/long_geo (slow write for large datasets)",
    )
    p.add_argument(
        "--skip-geoparquet",
        action="store_true",
        help="Skip GeoParquet config (requires geopandas; geometry join can be slow)",
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
    print("GeoParquet Feasibility Benchmark")
    print(f"{'='*70}")
    print(f"  Variable : {args.variable}")
    print(f"  Grid     : {args.grid}")
    print(f"  Time unit: {args.time_unit}")
    filters = {k: v for k, v in read_kwargs.items() if k not in DATASET_DIMENSIONS}
    print(f"  Filters  : {', '.join(f'{k}={v}' for k,v in filters.items()) or 'none'}")
    print(f"  Runs     : {args.runs}")
    print(f"  Threads  : {pl.thread_pool_size()} (use --threads N to limit)")
    print(f"  Output   : {output_dir}")
    print()

    # ── Step 1: load wide data ─────────────────────────────────────────────
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

    # ── Step 2: convert to long format ────────────────────────────────────
    print("\nStep 2/4: Converting to long format …", end=" ", flush=True)
    long_df = wide_to_long(wide_df)
    long_year_df = add_year(long_df)
    print(f"{len(long_df):,} rows × {len(long_df.columns)} columns")

    # ── Step 3: write benchmark files ─────────────────────────────────────
    print("\nStep 3/4: Writing benchmark files")
    output_dir.mkdir(parents=True, exist_ok=True)

    configs: dict[str, tuple[Path, bool, list[str] | None]] = {}  # name -> (path, is_hive, exclude_cols)

    if not args.skip_unpartitioned:
        p = output_dir / "long_date"
        print(f"  Writing long_date (single file, date-sorted) …", end=" ", flush=True)
        write_long(long_df, p, sort_by=["calendar_day", "geography_id"])
        print(f"{dir_size_mb(p):.1f} MB")
        configs["long_date"] = (p, False, None)

        p = output_dir / "long_geo"
        print(f"  Writing long_geo  (single file, geo-sorted) …", end=" ", flush=True)
        write_long(long_df, p, sort_by=["geography_id", "calendar_day"])
        print(f"{dir_size_mb(p):.1f} MB")
        configs["long_geo"] = (p, False, None)

    p = output_dir / "year_date"
    print(f"  Writing year_date  (year-partitioned, date-sorted) …", end=" ", flush=True)
    write_long(long_year_df, p, sort_by=["year", "calendar_day", "geography_id"], partition_by_year=True)
    print(f"{dir_size_mb(p):.1f} MB")
    configs["year_date"] = (p, True, None)

    p = output_dir / "year_geo"
    print(f"  Writing year_geo   (year-partitioned, geo-sorted) …", end=" ", flush=True)
    write_long(long_year_df, p, sort_by=["year", "geography_id", "calendar_day"], partition_by_year=True)
    print(f"{dir_size_mb(p):.1f} MB")
    configs["year_geo"] = (p, True, None)

    if not args.skip_geoparquet:
        if not HAS_GEOPANDAS:
            print("  Skipping geoparquet configs (geopandas not installed; install the bridges2 extras)")
        else:
            p = output_dir / "geoparquet"
            print(f"  Writing geoparquet     (year-partitioned, date-sorted, geometry) …")
            ok = write_geoparquet(
                long_year_df, p,
                grid=args.grid,
                sort_by=["year", "calendar_day", "geography_id"],
                bloom_filter_options=None,
            )
            if ok:
                print(f"    {dir_size_mb(p):.1f} MB")
                configs["geoparquet"] = (p, True, ["geometry"])

            p = output_dir / "geoparquet_geo"
            print(f"  Writing geoparquet_geo (year-partitioned, geo-sorted, geometry) …")
            ok = write_geoparquet(
                long_year_df, p,
                grid=args.grid,
                sort_by=["year", "geography_id", "calendar_day"],
                bloom_filter_options=None,
            )
            if ok:
                print(f"    {dir_size_mb(p):.1f} MB")
                configs["geoparquet_geo"] = (p, True, ["geometry"])

    # ── Step 4: run benchmarks ─────────────────────────────────────────────
    print("\nStep 4/4: Running benchmarks …")
    results: dict[str, dict] = {}

    print("  Benchmarking: wide (baseline)")
    r = bench_wide(read_kwargs, target_date, target_geo_str, args.runs)
    r["size_mb"] = float("nan")
    results["wide (baseline)"] = r

    for name, (path, is_hive, exclude_cols) in configs.items():
        print(f"  Benchmarking: {name}")
        r = bench_long(path, target_date, target_geo_int, args.runs, hive=is_hive, exclude_cols=exclude_cols)
        r["size_mb"] = dir_size_mb(path)
        results[name] = r

    # ── Report ─────────────────────────────────────────────────────────────
    print_results(results)

    print("Notes:")
    print(f"  Target date: {target_date}  |  Target geography: {target_geo_int}")
    print(f"  Row group size: {ROW_GROUP_SIZE:,}  |  Compression: {COMPRESSION} level {COMPRESSION_LEVEL}")
    print(f"  Long-format row count: {len(long_df):,}  |  Geographies: {n_geos:,}")
    print(f"  Threads: {pl.thread_pool_size()}  |  Use --threads N to isolate format differences from parallelism")
    print(f"  geoparquet/geoparquet_geo read benchmarks exclude the geometry column for fair comparison")
    print(f"  year_geo is smaller than long_geo (same sort order) because splitting by year makes the")
    print(f"    calendar_day column compressible: within each year file the same {len(long_df) // len(wide_df)}-day")
    print(f"    sequence repeats once per geography, while long_geo has one {len(wide_df):,}-day sequence per geography")
    print(f"  geoparquet_geo expected to be ~half the size of geoparquet: geo-sorting ensures each unique")
    print(f"    geometry WKB appears in exactly one row-group dictionary per year file rather than every one")
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
