"""Generate, compact, and query a unified catalog side-table for tabular
(Parquet) and raster (COG) datasets.

The catalog is one row per (distinct combination of every dimension except
grid_cell/value/calendar)  x  (contiguous calendar run).  The calendar axis is
stored as extent columns -- ``calendar_year_min/_max/_count`` and the same for
``calendar_day`` -- and a combination whose calendar values are *not* gapless is
split into one row per contiguous run (at the time-unit step), so the [min, max]
extent of every row is exact.

The catalog is itself a Hive dataset partitioned by the dataset dimensions
(``variable``/``time_unit``/``grid``), mirroring the data layout, so a catalog
query can prune to the relevant partitions.

Commands:

\b
  build    scan data, write catalog rows (one file per partition per run;
           re-running appends files, like tabular.write_parquet). Intended as a
           one-time/backfill tool for data written before write_parquet/write_cog
           started keeping the catalog in sync incrementally on every write.
  compact  per partition, combine files AND coalesce adjacent/overlapping
           calendar runs into minimal contiguous rows (like compact_parquet)
  query    run the table-backed catalog query (``cpra.mp.data.catalog.CatalogQuery``,
           the same class the ``/catalog.json`` API route uses)

The shared row-building/compaction logic lives in ``cpra.mp.data.catalog`` --
it's also used by the incremental catalog-sync hooks in
``tabular.write_parquet``/``compact_parquet`` and ``raster.write_cog``/
``compact_cog``.

Usage:

\b
    python -m cpra.mp.data.cli catalog build   --base-path data-long --out catalog
    python -m cpra.mp.data.cli catalog compact --catalog catalog
    python -m cpra.mp.data.cli catalog query   --catalog catalog --variable damage
"""

import time
import uuid
from pathlib import Path

import click
import polars as pl

from cpra.mp.data.catalog import (
    CAL_DIMS,
    CATALOG_DIMS,
    EXTENT_COLS,
    Catalog,
    CatalogQuery,
    merge_partition,
    parse_path_dims,
    raster_catalog_from_paths,
    tabular_catalog_from_files,
    unify_dtypes,
)
from cpra.mp.data.cli import cli


@cli.group()
def catalog():
    """Generate, compact, and query the unified catalog side-table."""


@catalog.command()
@click.option("--base-path", default="data-long")
@click.option("--out", default="catalog")
def build(base_path, out):
    base = Path(base_path)
    t0 = time.time()
    parquet_dirs, tif_paths = {}, []
    for root, _dirs, files in base.walk():
        if pq := [f for f in files if f.endswith(".parquet")]:
            parquet_dirs[root] = pq
        tif_paths.extend(root / f for f in files if f.endswith(".tif"))

    frames = []
    if tif_paths:
        click.echo(f"raster: {len(tif_paths)} .tif files")
        frames.append(unify_dtypes(raster_catalog_from_paths(base, tif_paths)))
    for d, files in parquet_dirs.items():
        rel = d.relative_to(base)
        try:
            cat = unify_dtypes(tabular_catalog_from_files(base, d, files))
        except Exception as exc:
            click.echo(f"tabular: {rel}  ->  SKIPPED ({type(exc).__name__}: {exc})", err=True)
            continue
        click.echo(f"tabular: {rel}  ->  {cat.height:,} rows")
        frames.append(cat)

    if not frames:
        click.echo(f"\nDONE: no valid datasets found under {base} in {time.time()-t0:.1f}s")
        return

    catalog_df = pl.concat(frames, how="diagonal_relaxed")
    ordered = [c for c in CATALOG_DIMS if c in catalog_df.columns and c not in CAL_DIMS] + EXTENT_COLS + ["data_type"]
    catalog_df = catalog_df.select([c for c in ordered if c in catalog_df.columns])
    Catalog(out).append(catalog_df)
    click.echo(f"\nDONE: {catalog_df.height:,} rows -> {out}/ in {time.time()-t0:.1f}s")


@catalog.command()
@click.option("--catalog", "catalog_dir", default="catalog")
def compact(catalog_dir):
    base = Path(catalog_dir)
    n_parts = n_before = n_after = 0
    for root, _dirs, files in base.walk():
        pqs = [f for f in files if f.endswith(".parquet")]
        if len(pqs) <= 1:
            continue  # build emits minimal runs; a single file needs no merge
        time_unit = parse_path_dims(root.relative_to(base))["time_unit"]
        # Concat files individually with diagonal_relaxed rather than pl.read_parquet(pqs) --
        # older files may predate a fix to tabular_catalog_rows and lack the unused calendar
        # dim's extent columns entirely, which pl.read_parquet rejects as a schema mismatch.
        df = pl.concat([pl.read_parquet(root / f) for f in pqs], how="diagonal_relaxed")
        merged = merge_partition(df, time_unit)
        merged.write_parquet(root / f"{uuid.uuid4()}-0.parquet")
        for f in pqs:
            (root / f).unlink()
        n_parts += 1
        n_before += df.height
        n_after += merged.height
        click.echo(
            f"compacted {root.relative_to(base)}: {len(pqs)} files / {df.height:,} rows -> 1 file / {merged.height:,} rows"
        )
    click.echo(f"\nDONE: {n_parts} partitions, {n_before:,} -> {n_after:,} rows")


@catalog.command()
@click.option("--catalog", "catalog_dir", default="catalog")
@click.option("--variable", default=None)
@click.option("--time-unit", default=None)
@click.option("--grid", default=None)
def query(catalog_dir, variable, time_unit, grid):
    from cpra.mp.data.api.models import DatasetDimensions, FilterDimensions

    dd_kwargs = {k: v for k, v in dict(variable=variable, time_unit=time_unit, grid=grid).items() if v is not None}
    dd = DatasetDimensions(**dd_kwargs)
    fd = FilterDimensions()

    t = time.time()
    out = CatalogQuery(dd, fd, catalog_path=catalog_dir).execute()
    click.echo(f"[table] {(time.time()-t)*1000:.1f} ms")
    for k, v in out["choices"].items():
        click.echo(f"  {k}: choices={v}  result={out['results'][k]}")
