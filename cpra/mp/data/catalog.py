"""Shared logic for building and maintaining the unified catalog side-table.

The catalog is one row per (distinct combination of every dimension except
grid_cell/value/calendar) x (contiguous calendar run). The calendar axis is
stored as extent columns -- ``calendar_year_min/_max/_count`` and the same for
``calendar_day`` -- and a combination whose calendar values are *not* gapless is
split into one row per contiguous run (at the time-unit step), so the [min, max]
extent of every row is exact.

The catalog is itself a Hive dataset partitioned by the dataset dimensions
(``variable``/``time_unit``/``grid``), mirroring the data layout.

This module holds the logic shared by:

* ``cpra.mp.data.cli.catalog`` (``python -m cpra.mp.data.cli catalog ...``), which
  does a one-time/backfill full rescan of a data tree
  (``raster_catalog_from_paths``/``tabular_catalog_from_files``), and
* the incremental hooks in ``tabular.write_parquet``/``compact_parquet`` and
  ``raster.write_cog``/``compact_cog``, which keep the catalog in sync on every
  write/compact without a full rescan (``tabular_catalog_rows``/
  ``raster_catalog_rows`` build the rows; the ``Catalog`` class -- initialized with
  a catalog path -- appends/coalesces them).

Also holds ``CatalogQuery``, the production catalog-query implementation, which
resolves a request's dimension choices/results by scanning the materialized
side-table above (loaded scoped to the requested variable/time_unit/grid) rather
than walking the data tree -- used by the ``/catalog.json`` API route. It raises
``ValueError`` rather than an HTTP-framework exception so this module has no
web-framework dependency; callers (e.g. the FastAPI route in ``api/__init__.py``)
catch it and translate it into an ``HTTPException``.

Raster catalog rows include metadata-typed dimensions (e.g. ``model``) the same
as tabular rows do. Metadata dimensions are never COG path components (see
``raster.cog_path_expression``), so they can only be recovered from each COG's
dataset tags, not the path -- ``raster_rows``/``raster_catalog_rows`` source them
from tags accordingly.
"""

import uuid
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.dataset as pads

from cpra.mp.data.config import get_catalog_path
from cpra.mp.data.dimensions import DIMENSIONS, METADATA_DIMENSIONS
from cpra.mp.data.utils import (
    RANGE_FUNCTIONS,
    RANGE_KWARGS,
    calendar_run_offsets,
    get_input_data_frame,
    required_columns,
)

CATALOG_DIMS = list(DIMENSIONS)
CAL_DIMS = ("calendar_year", "calendar_day")
PART_COLS = ["variable", "time_unit", "grid"]
EXCLUDE = {"grid_cell", "value"}
EXTENT_COLS = [f"{c}_{s}" for c in CAL_DIMS for s in ("min", "max", "count")]


def cal_for(time_unit):
    return "calendar_year" if time_unit in ("annual", "decadal") else "calendar_day"


def period_of(expr, cal, time_unit):
    """Map a calendar value to a monotonic integer where consecutive periods
    differ by exactly 1 (so contiguity is just a diff of 1)."""
    if cal == "calendar_year":
        return expr // 10 if time_unit == "decadal" else expr
    if time_unit == "monthly":
        return expr.dt.year() * 12 + (expr.dt.month() - 1)
    return expr.cast(pl.Int32)  # daily: days since epoch


def expand_calendar_range(cal, time_unit, mn, mx):
    """Expand a calendar [min, max] run back to its full value list."""
    fn, kw = RANGE_FUNCTIONS[cal], RANGE_KWARGS[time_unit]
    return fn(mn, mx, eager=True, **kw) if cal == "calendar_day" else fn(mn, mx + 1, eager=True, **kw)


def cast_str(name, value):
    dtype = DIMENSIONS[name].dtype
    s = pl.Series([value], dtype=pl.String)
    return s.str.to_date(strict=False)[0] if dtype == pl.Date else s.cast(dtype, strict=False)[0]


def parse_path_dims(rel_path):
    dims = {}
    for seg in Path(rel_path).parts:
        seg = seg[:-4] if seg.endswith(".tif") else seg
        name, sep, val = seg.partition("=")
        if sep and name in DIMENSIONS:
            dims[name] = val
    return dims


def contiguous_runs(frame, group_cols, cal, time_unit):
    """One row per (group_cols combination, contiguous calendar run)."""
    lf = frame.lazy().select([*group_cols, cal]).unique()
    lf = lf.with_columns(period_of(pl.col(cal), cal, time_unit).alias("__p")).sort([*group_cols, "__p"])
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


def unify_dtypes(df):
    exprs = []
    for c in df.columns:
        if c in DIMENSIONS:
            exprs.append(pl.col(c).cast(DIMENSIONS[c].dtype))
        elif c.endswith(("_min", "_max")):
            exprs.append(pl.col(c).cast(DIMENSIONS[c.rsplit("_", 1)[0]].dtype))
        elif c.endswith("_count"):
            exprs.append(pl.col(c).cast(pl.Int64))
    return df.with_columns(exprs)


def merge_partition(df, time_unit):
    """Coalesce adjacent/overlapping calendar runs within each combination."""
    cal = cal_for(time_unit)
    group_cols = [c for c in df.columns if c not in EXTENT_COLS]
    other = "calendar_day" if cal == "calendar_year" else "calendar_year"

    work = (
        df.filter(pl.col(f"{cal}_min").is_not_null())
        .with_columns(
            period_of(pl.col(f"{cal}_min"), cal, time_unit).alias("__pmin"),
            period_of(pl.col(f"{cal}_max"), cal, time_unit).alias("__pmax"),
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


def raster_rows(base, tif_paths):
    """One dict per (COG, band): path dimensions plus the band's calendar value.

    Every COG -- single-band or a multi-band contiguous-run merge from ``compact_cog`` --
    stores each band's calendar value in its band description (e.g. ``"calendar_year=2032"``).
    Metadata-typed dimensions (e.g. ``model``) are never path components (see
    ``raster.cog_path_expression``), so they're read from the COG's dataset tags instead
    -- the sole other source ``write_cog`` persists them to.

    Delegates the parallel open + band-description parse to ``raster._read_cog_band_rows``
    (imported locally to avoid a raster<->catalog import cycle), which also synthesizes the
    calendar description for pre-migration single-band COGs that stored it as a dataset tag.
    """
    from cpra.mp.data.raster import _read_cog_band_rows

    for path, _band, name, value, _dtype, _is_legacy, tags in _read_cog_band_rows(tif_paths):
        dims = parse_path_dims(path.relative_to(base))
        dims.update({k: v for k, v in tags.items() if k in METADATA_DIMENSIONS})
        yield {**dims, name: value}


def raster_catalog_from_paths(base, tif_paths):
    rows = list(raster_rows(base, tif_paths))
    cols = sorted({k for r in rows for k in r}, key=CATALOG_DIMS.index)
    df = pl.DataFrame({c: [cast_str(c, r[c]) if c in r else None for r in rows] for c in cols})
    frames = []
    # Each raster dataset uses one calendar dim; split per time_unit value.
    for tu in df["time_unit"].unique().to_list():
        sub = df.filter(pl.col("time_unit") == tu)
        cal = cal_for(tu)
        group = [c for c in sub.columns if c not in CAL_DIMS]
        frames.append(contiguous_runs(sub, group, cal, tu))
    return pl.concat(frames, how="diagonal_relaxed").with_columns(data_type=pl.lit("raster"))


def tabular_catalog_from_files(base, parquet_dir, files):
    path_dims = parse_path_dims(Path(parquet_dir).relative_to(base))
    time_unit = path_dims["time_unit"]
    cal = cal_for(time_unit)
    lf = pl.scan_parquet([str(Path(parquet_dir) / f) for f in files])
    names = lf.collect_schema().names()
    in_dims = [c for c in names if c in DIMENSIONS and c not in EXCLUDE and c not in CAL_DIMS]
    runs = contiguous_runs(lf, in_dims, cal, time_unit)
    runs = runs.with_columns(pl.lit(cast_str(n, v)).alias(n) for n, v in path_dims.items())
    return runs.with_columns(data_type=pl.lit("tabular"))


def tabular_catalog_rows(lf, group_cols, time_unit):
    """Catalog rows for the data ``lf`` is about to write, computed in-memory
    (no re-scan from disk) -- used by ``tabular.write_parquet``'s catalog hook."""
    cal = cal_for(time_unit)
    other = CAL_DIMS[1] if cal == CAL_DIMS[0] else CAL_DIMS[0]
    runs = contiguous_runs(lf, group_cols, cal, time_unit)
    runs = runs.with_columns(
        pl.lit(None, dtype=DIMENSIONS[other].dtype).alias(f"{other}_min"),
        pl.lit(None, dtype=DIMENSIONS[other].dtype).alias(f"{other}_max"),
        pl.lit(None, dtype=pl.Int64).alias(f"{other}_count"),
        data_type=pl.lit("tabular"),
    )
    return unify_dtypes(runs)


def raster_catalog_rows(dims, calendar_dim, calendar_values):
    """One catalog row spanning ``calendar_values`` for a fixed dimension
    combination -- used by ``raster.write_cog``'s and ``compact_cog``'s catalog hooks."""
    other = CAL_DIMS[1] if calendar_dim == CAL_DIMS[0] else CAL_DIMS[0]
    row = {
        **dims,
        f"{calendar_dim}_min": min(calendar_values),
        f"{calendar_dim}_max": max(calendar_values),
        f"{calendar_dim}_count": len(calendar_values),
        f"{other}_min": None,
        f"{other}_max": None,
        f"{other}_count": None,
        "data_type": "raster",
    }
    return unify_dtypes(pl.DataFrame({k: [v] for k, v in row.items()}))


class Catalog:
    """The catalog side-table rooted at ``catalog_path``.

    ``write_parquet``/``write_cog``'s catalog hooks call ``append`` with rows built by
    ``tabular_catalog_rows``/``raster_catalog_rows``; ``compact_parquet``/``compact_cog``
    call ``compact_partition`` afterward to coalesce whatever rows now exist for that
    variable/time_unit/grid partition. ``cpra.mp.data.cli.catalog``'s ``build`` command
    uses ``append`` for its one-time/backfill full rescan of a data tree
    (``raster_catalog_from_paths``/``tabular_catalog_from_files``); its ``compact``
    command walks the whole catalog tree directly rather than through this class, since
    it doesn't know each partition's dimension values in advance.
    """

    def __init__(self, catalog_path):
        self.catalog_path = Path(catalog_path)

    def append(self, catalog_df):
        """Append rows as a new file (additive, uuid-named -- concurrency-safe)."""
        tbl = catalog_df.to_arrow()
        part = pads.partitioning(pa.schema([tbl.schema.field(c) for c in PART_COLS]), flavor="hive")
        pads.write_dataset(
            tbl,
            base_dir=str(self.catalog_path),
            format="parquet",
            partitioning=part,
            existing_data_behavior="overwrite_or_ignore",
            basename_template=f"{uuid.uuid4()}-{{i}}.parquet",
        )

    def compact_partition(self, variable, time_unit, grid):
        """Coalesce all rows for one variable/time_unit/grid partition into minimal
        contiguous-run rows."""
        directory = self.catalog_path / f"variable={variable}" / f"time_unit={time_unit}" / f"grid={grid}"
        files = list(directory.glob("*.parquet")) if directory.exists() else []
        if len(files) <= 1:
            return
        # Read files individually and concat with diagonal_relaxed rather than
        # pl.read_parquet(files) -- older files may predate a fix to tabular_catalog_rows
        # and lack the unused calendar dim's extent columns entirely, which pl.read_parquet
        # rejects as a schema mismatch across a file list.
        merged = merge_partition(
            pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed"), time_unit
        )
        tmp = directory / f"{uuid.uuid4()}-0.parquet"
        merged.write_parquet(tmp)
        for f in files:
            f.unlink()


class CatalogQuery:
    """Data catalog query backed by the materialized ``Catalog`` side-table above.

    Given a request's dataset dimensions (variable/time_unit/grid) and filter
    dimensions, it resolves the available ``choices`` and selected ``results`` for
    every dimension by scanning the side-table (one row per dimension combination x
    contiguous calendar run) rather than walking the data tree -- so it works for
    tabular (Parquet) and raster (multi-band COG) datasets alike.  Used by the
    ``/catalog.json`` API route.

    The catalog is loaded from ``catalog_path`` scoped to the requested
    variable/time_unit/grid partition(s) unless a preloaded ``catalog`` frame is
    supplied (used by the CLI/tests).

    Raises ``ValueError`` (not an HTTP-framework exception, to keep this module free of
    a web-framework dependency) when a query has no catalog entries or no valid values
    for some dimension; callers translate that into the appropriate HTTP error.
    """

    def __init__(self, dataset_dimensions, filter_dimensions, catalog=None, catalog_path=None):
        self.dataset_dimensions = dataset_dimensions
        self.filter_dimensions = filter_dimensions
        self._catalog = catalog
        self._catalog_path = get_catalog_path(catalog_path)
        self._choices = {}
        self._results = {}
        self._exact = True

    def _load_catalog(self):
        """Load the side-table scoped to the requested dataset partition(s).

        A preloaded frame (the ``catalog`` constructor arg) is returned as-is.  Otherwise
        the Hive-partitioned catalog under ``catalog_path`` is read for the
        variable/time_unit/grid selected by the dataset dimensions (a wildcard for any
        left unset), so a single request never unions the heterogeneous per-variable
        schemas of the whole tree.  ``PART_COLS`` are taken from each file's own path
        (the partitioned write strips them from the file itself) as Strings, to match
        the values the dataset dimensions carry.

        Files are read one at a time and concatenated with ``diagonal_relaxed`` rather
        than handed to ``pl.read_parquet`` as a list, for the same reason
        ``compact_partition`` does it: a partition holds one file per append, and appends
        made at different times need not agree on their columns.  The ``cli.catalog
        build`` rescan gives every file it writes the union of every dimension found
        anywhere in the tree, while ``write_parquet``/``write_cog``'s incremental hooks
        write only the dimensions of the one dataset they wrote -- so the two disagree
        the moment a rescan sees a variable the older files never had.
        ``pl.read_parquet`` over a file list rejects that as a schema mismatch; here the
        absent columns come back null, which is already what a dimension a row doesn't
        have means.
        """
        if self._catalog is not None:
            return self._catalog
        parts = []
        for name in PART_COLS:
            value = getattr(self.dataset_dimensions, name, None)
            parts.append(f"{name}={value}" if value else f"{name}=*")
        frames = [
            pl.read_parquet(path).with_columns(
                pl.lit(v, dtype=pl.String).alias(k)
                for k, v in parse_path_dims(path.parent.relative_to(self._catalog_path)).items()
            )
            for path in self._catalog_path.glob("/".join(parts) + "/*.parquet")
        ]
        if not frames:
            msg = "No catalog entries for the requested dataset"
            raise ValueError(msg)
        return pl.concat(frames, how="diagonal_relaxed")

    def _update(self, dimension_name, dimension_values):
        dimension = DIMENSIONS.get(dimension_name)

        # Update choices with the dimension values.
        if dimension_name in self._choices:
            # Calculate choices that are common to this branch and previous branches.
            dimension_values_set = set(dimension_values)
            common_choices = [val for val in self._choices[dimension_name] if val in dimension_values_set]
            # If there are common choices, assign them as the choices for
            # this dimension. Otherwise, return an error.
            if common_choices:
                self._choices[dimension_name] = common_choices
            else:
                msg = f"No common values for dimension {dimension_name}"
                raise ValueError(msg)
        else:
            self._choices[dimension_name] = dimension_values

        selected_values = []
        prop_value = getattr(
            self.dataset_dimensions if dimension.type == "dataset" else self.filter_dimensions,
            dimension_name,
            # Metadata dimensions (e.g. `model`) are stored as file tags but are not
            # filterable via FilterDimensions, so there's no attribute to read here --
            # fall through to the default-value selection below.
            None,
        )
        if prop_value:
            if not isinstance(prop_value, (list, tuple)):
                prop_value = (prop_value,)
            # For float dimensions with known canonical values, normalise
            # prop_value through dtype → canonical map so that both a clean
            # input (0.1) and its float32 repr (0.10000000149011612) resolve
            # to the same canonical value as the one stored in choices.
            if dimension.values is not None and str(dimension.dtype).startswith("Float"):
                canonical_map = dict(zip(
                    dimension.values["value"].cast(dimension.dtype).to_list(),
                    dimension.values["value"].to_list(),
                ))
                prop_value = [
                    canonical_map.get(pl.Series([v], dtype=dimension.dtype).to_list()[0], v)
                    for v in prop_value
                ]
            choices_set = set(self._choices[dimension_name])
            selected_values = sorted([val for val in prop_value if val in choices_set])
            if len(selected_values) < len(prop_value):
                self._exact = False

        if not selected_values:
            # TODO: Add more sophisticated logic for selecting a default value
            # from the available choices.
            selected_values = [self._choices[dimension_name][0]]
            # Metadata dimensions have no FilterDimensions field a caller could ever pin,
            # so defaulting one doesn't make the result any less exact than the caller
            # asked for.
            if dimension.type != "metadata":
                self._exact = False

        self._results[dimension_name] = selected_values

    def _compress(self):
        """Compress date/time values into continuous [start, end] intervals."""
        for dimension_name, values in self._choices.items():
            if dimension_name in ("calendar_day", "calendar_year"):
                if not isinstance(values, pl.Series):
                    values = pl.Series(values)
                offsets = calendar_run_offsets(values, dimension_name, self.dataset_dimensions.time_unit)
                groups = (
                    pl.DataFrame({"value": values, "offset": offsets})
                    .group_by("offset")
                    .agg(start=pl.col("value").min(), end=pl.col("value").max())
                    .sort(pl.col("start"))
                )
                if dimension_name == "calendar_day":
                    groups = groups.with_columns(
                        pl.col("start").cast(pl.String),
                        pl.col("end").cast(pl.String),
                    )
                    self._results[dimension_name] = [str(v) for v in self._results[dimension_name]]

                self._choices[dimension_name] = [
                    row["start"] if row["start"] == row["end"] else [row["start"], row["end"]]
                    for row in groups.iter_rows(named=True)
                ]

    def execute(self):
        """Resolve dimension choices/results from the catalog side-table."""
        catalog = self._load_catalog()
        input_df = get_input_data_frame(**dict(self.dataset_dimensions), **dict(self.filter_dimensions))
        required_cols = required_columns(input_df)
        non_cal = [c for c in CATALOG_DIMS if c in catalog.columns and c in required_cols and c not in CAL_DIMS]

        lf = catalog.lazy()
        for col in non_cal:
            dim = DIMENSIONS[col]
            raw = lf.select(col).drop_nulls().unique().sort(col).collect()[col].to_list()
            if not raw:
                continue
            # Map float32-repr values (e.g. 0.10000000149011612) back to their canonical
            # display values (e.g. 0.1) using the dimension's known values list.
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
            if cal not in required_cols or f"{cal}_min" not in catalog.columns:
                continue
            ext = lf.select(f"{cal}_min", f"{cal}_max").drop_nulls().unique().collect()
            values = set()
            for mn, mx in ext.iter_rows():
                values.update(expand_calendar_range(cal, time_unit, mn, mx).to_list())
            if values:
                self._update(dimension_name=cal, dimension_values=sorted(values))

        self._compress()
        return {"choices": self._choices, "results": self._results, "exact": self._exact}
