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
           re-running appends files, like tabular.write_parquet)
  compact  per partition, combine files AND coalesce adjacent/overlapping
           calendar runs into minimal contiguous rows (like compact_parquet)
  query    run the simplified, table-backed catalog query (``--compare`` checks
           it against the real filesystem CatalogQuery)

``TableCatalogQuery`` subclasses the real ``CatalogQuery`` and reuses its
``_update``/``_compress`` verbatim, overriding only ``execute`` -- the
``api/catalog.py`` simplification this prototype is for.

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
import pyarrow as pa
import pyarrow.dataset as pads
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))

from cpra.mp.data.api.catalog import RANGE_FUNCTIONS, RANGE_KWARGS, CatalogQuery
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.utils import get_input_data_frame, required_columns

CATALOG_DIMS = list(DIMENSIONS)
CAL_DIMS = ("calendar_year", "calendar_day")
PART_COLS = ["variable", "time_unit", "grid"]
EXCLUDE = {"grid_cell", "value"}
EXTENT_COLS = [f"{c}_{s}" for c in CAL_DIMS for s in ("min", "max", "count")]


def _cal_for(time_unit):
    return "calendar_year" if time_unit in ("annual", "decadal") else "calendar_day"


def _period_of(expr, cal, time_unit):
    """Map a calendar value to a monotonic integer where consecutive periods
    differ by exactly 1 (so contiguity is just a diff of 1)."""
    if cal == "calendar_year":
        return expr // 10 if time_unit == "decadal" else expr
    if time_unit == "monthly":
        return expr.dt.year() * 12 + (expr.dt.month() - 1)
    return expr.cast(pl.Int32)  # daily: days since epoch


def _expand(cal, time_unit, mn, mx):
    """Expand a calendar [min, max] run back to its full value list."""
    fn, kw = RANGE_FUNCTIONS[cal], RANGE_KWARGS[time_unit]
    return fn(mn, mx, eager=True, **kw) if cal == "calendar_day" else fn(mn, mx + 1, eager=True, **kw)


def _cast_str(name, value):
    dtype = DIMENSIONS[name].dtype
    s = pl.Series([value], dtype=pl.String)
    return s.str.to_date(strict=False)[0] if dtype == pl.Date else s.cast(dtype, strict=False)[0]


def _parse_path_dims(rel_path):
    dims = {}
    for seg in Path(rel_path).parts:
        seg = seg[:-4] if seg.endswith(".tif") else seg
        name, sep, val = seg.partition("=")
        if sep and name in DIMENSIONS:
            dims[name] = val
    return dims


def _contiguous_runs(frame, group_cols, cal, time_unit):
    """One row per (group_cols combination, contiguous calendar run)."""
    lf = frame.lazy().select([*group_cols, cal]).unique()
    lf = lf.with_columns(_period_of(pl.col(cal), cal, time_unit).alias("__p")).sort([*group_cols, "__p"])
    brk = (pl.col("__p") - pl.col("__p").shift(1).over(group_cols) != 1).fill_null(True)
    lf = lf.with_columns(brk.alias("__brk")).with_columns(pl.col("__brk").cum_sum().over(group_cols).alias("__run"))
    return (
        lf.group_by([*group_cols, "__run"])
        .agg(
            pl.col(cal).min().alias(f"{cal}_min"),
            pl.col(cal).max().alias(f"{cal}_max"),
            pl.len().alias(f"{cal}_count"),
        )
        .drop("__run")
        .collect(engine="streaming")
    )


def _unify(df):
    exprs = []
    for c in df.columns:
        if c in DIMENSIONS:
            exprs.append(pl.col(c).cast(DIMENSIONS[c].dtype))
        elif c.endswith(("_min", "_max")):
            exprs.append(pl.col(c).cast(DIMENSIONS[c.rsplit("_", 1)[0]].dtype))
        elif c.endswith("_count"):
            exprs.append(pl.col(c).cast(pl.Int64))
    return df.with_columns(exprs)


def _raster_rows(base, tif_paths):
    """One dict per (COG, band): path dimensions plus the band's calendar value.

    Every COG -- single-band or a multi-band contiguous-run merge from ``compact_cog`` --
    stores each band's calendar value in its band description (e.g. ``"calendar_year=2032"``),
    which is the sole source used here.
    """
    for path in tif_paths:
        dims = _parse_path_dims(path.relative_to(base))
        with rasterio.open(path) as ds:
            for desc in ds.descriptions:
                name, _, value = desc.partition("=")
                yield {**dims, name: value}


def _raster_catalog(base, tif_paths):
    rows = list(_raster_rows(base, tif_paths))
    cols = sorted({k for r in rows for k in r}, key=CATALOG_DIMS.index)
    df = pl.DataFrame({c: [_cast_str(c, r[c]) if c in r else None for r in rows] for c in cols})
    frames = []
    # Each raster dataset uses one calendar dim; split per time_unit value.
    for tu in df["time_unit"].unique().to_list():
        sub = df.filter(pl.col("time_unit") == tu)
        cal = _cal_for(tu)
        group = [c for c in sub.columns if c not in CAL_DIMS]
        frames.append(_contiguous_runs(sub, group, cal, tu))
    return pl.concat(frames, how="diagonal_relaxed").with_columns(data_type=pl.lit("raster"))


def _tabular_catalog(base, parquet_dir, files):
    path_dims = _parse_path_dims(Path(parquet_dir).relative_to(base))
    time_unit = path_dims["time_unit"]
    cal = _cal_for(time_unit)
    lf = pl.scan_parquet([str(Path(parquet_dir) / f) for f in files])
    names = lf.collect_schema().names()
    in_dims = [c for c in names if c in DIMENSIONS and c not in EXCLUDE and c not in CAL_DIMS]
    runs = _contiguous_runs(lf, in_dims, cal, time_unit)
    runs = runs.with_columns(pl.lit(_cast_str(n, v)).alias(n) for n, v in path_dims.items())
    return runs.with_columns(data_type=pl.lit("tabular"))


def _write_partitioned(catalog, out):
    tbl = catalog.to_arrow()
    part = pads.partitioning(pa.schema([tbl.schema.field(c) for c in PART_COLS]), flavor="hive")
    pads.write_dataset(
        tbl,
        base_dir=str(out),
        format="parquet",
        partitioning=part,
        existing_data_behavior="overwrite_or_ignore",
        basename_template=f"{uuid.uuid4()}-{{i}}.parquet",
    )


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
        frames.append(_unify(_raster_catalog(base, tif_paths)))
    for d, files in parquet_dirs.items():
        rel = d.relative_to(base)
        try:
            cat = _unify(_tabular_catalog(base, d, files))
        except Exception as exc:
            click.echo(f"tabular: {rel}  ->  SKIPPED ({type(exc).__name__}: {exc})", err=True)
            continue
        click.echo(f"tabular: {rel}  ->  {cat.height:,} rows")
        frames.append(cat)

    catalog = pl.concat(frames, how="diagonal_relaxed")
    ordered = [c for c in CATALOG_DIMS if c in catalog.columns and c not in CAL_DIMS] + EXTENT_COLS + ["data_type"]
    catalog = catalog.select([c for c in ordered if c in catalog.columns])
    _write_partitioned(catalog, out)
    click.echo(f"\nDONE: {catalog.height:,} rows -> {out}/ in {time.time()-t0:.1f}s")


def _merge_partition(df, time_unit):
    """Coalesce adjacent/overlapping calendar runs within each combination."""
    cal = _cal_for(time_unit)
    group_cols = [c for c in df.columns if c not in EXTENT_COLS]
    other = "calendar_day" if cal == "calendar_year" else "calendar_year"

    work = (
        df.filter(pl.col(f"{cal}_min").is_not_null())
        .with_columns(
            _period_of(pl.col(f"{cal}_min"), cal, time_unit).alias("__pmin"),
            _period_of(pl.col(f"{cal}_max"), cal, time_unit).alias("__pmax"),
        )
        .sort([*group_cols, "__pmin"])
    )
    new_run = (pl.col("__pmin") > pl.col("__pmax").cum_max().shift(1).over(group_cols) + 1).fill_null(True)
    work = work.with_columns(new_run.alias("__new")).with_columns(
        pl.col("__new").cum_sum().over(group_cols).alias("__mid")
    )
    merged = (
        work.group_by([*group_cols, "__mid"])
        .agg(
            pl.col(f"{cal}_min").min().alias(f"{cal}_min"),
            pl.col(f"{cal}_max").max().alias(f"{cal}_max"),
            (pl.col("__pmax").max() - pl.col("__pmin").min() + 1).cast(pl.Int64).alias(f"{cal}_count"),
        )
        .drop("__mid")
        .with_columns(
            pl.lit(None, dtype=DIMENSIONS[other].dtype).alias(f"{other}_min"),
            pl.lit(None, dtype=DIMENSIONS[other].dtype).alias(f"{other}_max"),
            pl.lit(None, dtype=pl.Int64).alias(f"{other}_count"),
        )
    )
    return merged.select([*group_cols, *EXTENT_COLS])


@cli.command()
@click.option("--catalog", "catalog_dir", default="catalog")
def compact(catalog_dir):
    base = Path(catalog_dir)
    n_parts = n_before = n_after = 0
    for root, _dirs, files in base.walk():
        pqs = [f for f in files if f.endswith(".parquet")]
        if len(pqs) <= 1:
            continue  # build emits minimal runs; a single file needs no merge
        time_unit = _parse_path_dims(root.relative_to(base))["time_unit"]
        df = pl.read_parquet([str(root / f) for f in pqs])
        merged = _merge_partition(df, time_unit)
        tmp = root / f"{uuid.uuid4()}-0.parquet"
        merged.write_parquet(tmp)
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
                values.update(_expand(cal, time_unit, mn, mx).to_list())
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
        import cpra.mp.data.api.catalog as cat

        cat.BASE_PATH = Path(base_path)
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
