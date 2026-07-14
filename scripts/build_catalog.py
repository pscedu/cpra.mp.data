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
  query    run the simplified, table-backed catalog query (``--compare`` checks
           it against the real filesystem CatalogQuery)

``TableCatalogQuery`` subclasses the real ``CatalogQuery`` (``cpra.mp.data.catalog``)
and reuses its ``_update``/``_compress`` verbatim, overriding only ``execute`` -- the
simplification this prototype is for.

The shared row-building/compaction logic lives in ``cpra.mp.data.catalog`` --
it's also used by the incremental catalog-sync hooks in
``tabular.write_parquet``/``compact_parquet`` and ``raster.write_cog``/
``compact_cog``.

Usage:

\b
    python scripts/build_catalog.py build   --base-path data-long --out catalog
    python scripts/build_catalog.py compact --catalog catalog
    python scripts/build_catalog.py query   --catalog catalog --variable damage --compare
"""

import sys
import time
import uuid
from pathlib import Path

import click
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from cpra.mp.data.catalog import (
    CAL_DIMS,
    CATALOG_DIMS,
    EXTENT_COLS,
    PART_COLS,
    Catalog,
    CatalogQuery,
    expand_calendar_range,
    merge_partition,
    parse_path_dims,
    raster_catalog_from_paths,
    tabular_catalog_from_files,
    unify_dtypes,
)
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.utils import get_input_data_frame, required_columns


@click.group()
def cli():
    pass


@cli.command()
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

    catalog = pl.concat(frames, how="diagonal_relaxed")
    ordered = [c for c in CATALOG_DIMS if c in catalog.columns and c not in CAL_DIMS] + EXTENT_COLS + ["data_type"]
    catalog = catalog.select([c for c in ordered if c in catalog.columns])
    Catalog(out).append(catalog)
    click.echo(f"\nDONE: {catalog.height:,} rows -> {out}/ in {time.time()-t0:.1f}s")


@cli.command()
@click.option("--catalog", "catalog_dir", default="catalog")
def compact(catalog_dir):
    base = Path(catalog_dir)
    n_parts = n_before = n_after = 0
    for root, _dirs, files in base.walk():
        pqs = [f for f in files if f.endswith(".parquet")]
        if len(pqs) <= 1:
            continue  # build emits minimal runs; a single file needs no merge
        time_unit = parse_path_dims(root.relative_to(base))["time_unit"]
        df = pl.read_parquet([str(root / f) for f in pqs])
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


def _load_catalog(catalog_dir):
    return pl.read_parquet(Path(catalog_dir), hive_partitioning=True).with_columns(
        pl.col(c).cast(pl.String) for c in PART_COLS
    )


class TableCatalogQuery(CatalogQuery):
    """Catalog query backed by the unified table; reuses _update/_compress."""

    def __init__(self, catalog, dataset_dimensions, filter_dimensions):
        super().__init__(dataset_dimensions, filter_dimensions)
        self.catalog = catalog

    def execute(self):
        input_df = get_input_data_frame(**dict(self.dataset_dimensions), **dict(self.filter_dimensions))
        required_cols = required_columns(input_df)
        non_cal = [c for c in CATALOG_DIMS if c in self.catalog.columns and c in required_cols and c not in CAL_DIMS]

        lf = self.catalog.lazy()
        for col in non_cal:
            dim = DIMENSIONS[col]
            raw = lf.select(col).drop_nulls().unique().sort(col).collect()[col].to_list()
            if not raw:
                continue
            if dim.values is not None and str(dim.dtype).startswith("Float"):
                cmap = dict(zip(dim.values["value"].cast(dim.dtype).to_list(), dim.values["value"].to_list()))
                values = [cmap.get(v, v) for v in raw]
            else:
                values = raw
            self._update(dimension_name=col, dimension_values=values)
            sel = pl.Series(self._results[col], dtype=dim.dtype)
            lf = lf.filter(pl.col(col).is_in(sel) | pl.col(col).is_null())

        time_unit = (self._results.get("time_unit") or [None])[0]
        for cal in CAL_DIMS:
            if cal not in required_cols or f"{cal}_min" not in self.catalog.columns:
                continue
            ext = lf.select(f"{cal}_min", f"{cal}_max").drop_nulls().unique().collect()
            values = set()
            for mn, mx in ext.iter_rows():
                values.update(expand_calendar_range(cal, time_unit, mn, mx).to_list())
            if values:
                self._update(dimension_name=cal, dimension_values=sorted(values))

        self._compress()
        return {"choices": self._choices, "results": self._results, "exact": self._exact}


@cli.command()
@click.option("--catalog", "catalog_dir", default="catalog")
@click.option("--base-path", default="data-long")
@click.option("--variable", default=None)
@click.option("--time-unit", default=None)
@click.option("--grid", default=None)
@click.option("--compare", is_flag=True)
def query(catalog_dir, base_path, variable, time_unit, grid, compare):
    from cpra.mp.data.api.models import DatasetDimensions, FilterDimensions

    catalog = _load_catalog(catalog_dir)
    dd_kwargs = {k: v for k, v in dict(variable=variable, time_unit=time_unit, grid=grid).items() if v is not None}
    dd = DatasetDimensions(**dd_kwargs)
    fd = FilterDimensions()

    t = time.time()
    out = TableCatalogQuery(catalog, dd, fd).execute()
    click.echo(f"[table] {(time.time()-t)*1000:.1f} ms")
    for k, v in out["choices"].items():
        click.echo(f"  {k}: choices={v}  result={out['results'][k]}")

    if compare:
        import cpra.mp.data.catalog as catalog_module

        catalog_module.BASE_PATH = Path(base_path)
        t = time.time()
        ref = CatalogQuery(dd, fd).execute()
        same = ref["choices"] == out["choices"] and ref["results"] == out["results"]
        click.echo(f"\n[filesystem] {time.time()-t:.2f} s   choices/results identical: {same}")
        if not same:
            for k in sorted(set(ref["choices"]) | set(out["choices"])):
                if ref["choices"].get(k) != out["choices"].get(k):
                    click.echo(f"  DIFF choices[{k}]: fs={ref['choices'].get(k)} table={out['choices'].get(k)}")
                if ref["results"].get(k) != out["results"].get(k):
                    click.echo(f"  DIFF results[{k}]: fs={ref['results'].get(k)} table={out['results'].get(k)}")


if __name__ == "__main__":
    cli()
