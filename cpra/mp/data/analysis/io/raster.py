"""The raster scan: `scan_raster` opens a band-wide dataset as the band-wide triple
(`execution/frame.py`'s `BandWideFrame`), assembling a per-request VRT rather than reading
one file at a time.

**Why a VRT.** A band-wide frame's columns are files -- `metadata_df` row *i* becomes
column `b<i>`, and `execution/frame.py`'s band-naming rule is that this correspondence is
exact and positional. GDAL's own VRT format already expresses "N bands, each sourced from a
different file (or a different band of the same file), in a declared order" as a single
virtual dataset, so building one turns "read N band-wide columns in step" into an ordinary
single-dataset windowed read, with no separate bookkeeping to keep N independent file
handles' windows in sync batch to batch. `cpra.mp.data.raster.vrt` already builds this XML
from a `(path, band)` list and needs no change for this -- see that function's own
docstring for why a bare path means band 1 and a `(path, band)` pair selects a specific
one.

The VRT is assembled fresh on every call (no caching -- deferred, along with
whatever invalidation a cache would need, to a later stage) and opened straight out of
memory via `rasterio.io.MemoryFile`, never written to disk: a VRT's own bytes are pure XML
referencing the real files by path, so nothing about opening or re-opening one touches a
temp directory this module would otherwise have to clean up. Confirmed empirically (not
merely assumed) that a `MemoryFile`-opened VRT reads identically, band for band and window
for window including per-band NoData, to opening the same sources directly.

**A crosswalk is read as a band but is NOT a band column.** A raster-to-vector crosswalk is
co-registered with the raster grid, so reading it is an ordinary windowed read with no
resampling and it can ride along as one more band of the same VRT -- that is a fact about
the READ, and this module uses it. What comes back is a different matter: a crosswalk holds
the ID of the vector cell containing each pixel, which is not a measurement, and a band-wide
frame's non-band columns are exactly the place for it (`execution/frame.py`'s
`validate_extra_columns` exists for these, and every band-wide lowering names what to carry
through).

So `scan_raster`'s `crosswalks` argument takes `(name, path)` pairs, and each one's pixel
values come back in a column called `name` -- ONE column, not one per band, since "which
zone is this pixel in" is a property of the pixel alone and is shared by every band in the
row. Crucially, a crosswalk gets NO row in `metadata_df`: `band_column_names(metadata_df)`
is the single source of truth for what a band is, so a crosswalk row there would make the
zone IDs indistinguishable from a measurement. A horizontal reduce across a row's bands
would then fold zone IDs into the statistic and a band-wide `calculate` would compute over
them -- a wrong number with nothing about the frame's shape to reveal it.

**NoData is two different questions, not one.** A data band's NoData sentinel
means "no measurement here" and is masked to null only when `mask_nodata` says to (default
on). A crosswalk's sentinel means "this pixel is outside that vector grid's domain
entirely" -- a fact a caller can never opt out of knowing, because there is no request under
which "the zone whose ID is the NoData sentinel" is a real zone. So a crosswalk column is
masked UNCONDITIONALLY, regardless of `mask_nodata`. A band with no declared NoData
(`nodataval is None`) is left alone either way -- such a raster is treated as never having
NoData at all, the same convention the old per-file scan used.

**Batching.** Reads are chunked through `io/raster_io.py`'s `batched_windows`/
`raster_blocks_per_batch`, the same block-row-chunked budget the old scan used, so a
request over a large raster does not pull the whole window into memory as one array. Each
batch reopens the VRT from its own in-memory bytes rather than holding a dataset handle
open across a `yield`: Polars' streaming engine may resume this generator's `source`
callback from a different thread than the one that entered it, and both `rio.Env` and
rasterio dataset handles are thread-local, so nothing opened before a `yield` may be
assumed valid after it (see `sources.py`'s identical discipline and its docstring for the
measurement behind it).

`window`, when given, restricts the scan to that single `rasterio.windows.Window` read
whole, in one batch, rather than tiling the full raster -- a caller specific enough to name
its own window has already decided how much of it to hold in memory at once. This is
plumbing only: nothing in this module computes a window on its own (that is a pushdown
decision for a later stage), and no resampling parameter exists yet -- every read is
`Resampling.nearest`, GDAL's own default for an unspecified resampling method, and the only
one a decimation-free window read ever needs.

`io` may import `request` but never `execution` -- see
`tests/api/test_analysis_layering.py`. What `scan_raster` takes from `request` is only
the two naming constants a crosswalk column name must not collide with; the dataset
selection itself never reaches here, `io/catalog.py` having already resolved one into
`metadata_df`. In particular, `execution/frame.py`'s `band_column_names` is NOT imported
here even though it computes the identical `f"b{i}"` convention, and neither is its
`validate_extra_columns` even though `_validate_crosswalk_names` below applies the same
rule: `io` may not import `execution` (`ReadWindow` is defined here precisely so the scan
can return a complete triple without that import -- see `io/window.py`'s docstring), so
both are reproduced locally against `request/pipeline.py`'s constants, as the one
deliberate duplication that layering requires.
"""

from collections.abc import Sequence

import numpy as np
import polars as pl
from polars.io.plugins import register_io_source
from rasterio.io import MemoryFile
from rasterio.windows import Window, transform as window_transform

from cpra.mp.data.analysis.io.raster_io import batched_windows, raster_blocks_per_batch, raster_gdal_env
from cpra.mp.data.analysis.io.window import ReadWindow
from cpra.mp.data.analysis.request.pipeline import BAND_NAME_RE, RESERVED_BAND_INDEX_COLUMN
from cpra.mp.data.raster import vrt

__all__ = ["scan_raster"]

# Mirrors `sources.py`'s identical dict -- duplicated rather than imported because
# `sources.py` is delete-tier for this scan (it belongs to the old engine, kept in place
# only until that engine is retired) and this module may not depend on it.
NUMPY_TO_POLARS = {
    "int8": pl.Int8,
    "uint8": pl.UInt8,
    "int16": pl.Int16,
    "uint16": pl.UInt16,
    "int32": pl.Int32,
    "uint32": pl.UInt32,
    "int64": pl.Int64,
    "float32": pl.Float32,
    "float64": pl.Float64,
}


def _validate_crosswalk_names(names: list[str]) -> None:
    """Reject a crosswalk column name that could be mistaken for a positional band name.

    The same rule `execution/frame.py` applies to any non-band column a lowering carries
    through, and the same one `request/pipeline.py` applies to a user's own input and step
    names -- checked here against those modules' own constants rather than a third copy of
    the pattern, since a crosswalk column is the first non-band column any frame gets and
    this is the earliest point it can be caught. A name like `b2` would be shadowed the
    moment the band columns are built, and the failure would present as a crosswalk that
    silently holds measurements.
    """
    seen = set()
    for name in names:
        if name == RESERVED_BAND_INDEX_COLUMN or BAND_NAME_RE.match(name):
            raise ValueError(
                f"crosswalk column name {name!r} matches the positional band-name pattern "
                f"(b0, b1, ...) or the reserved {RESERVED_BAND_INDEX_COLUMN!r} index column, "
                "and would be shadowed by it"
            )
        if name in seen:
            raise ValueError(f"duplicate crosswalk column name {name!r}")
        seen.add(name)


def scan_raster(
    metadata_df: pl.DataFrame,
    *,
    crosswalks: Sequence[tuple[str, object]] = (),
    window: Window | None = None,
    mask_nodata: bool = True,
) -> tuple[pl.LazyFrame, pl.DataFrame, ReadWindow]:
    """Open `metadata_df`'s bands as a band-wide triple, via a per-request VRT.

    `metadata_df` must carry `path` and `band` columns -- exactly what
    `io/catalog.py::resolve_raster_bands` returns -- and at least one row: a
    `ReadWindow`'s transform, CRS, and
    extent are properties of an actually-opened raster, so there is no legitimate way for
    this function itself to produce one for zero bands. (The zero-band degenerate frame,
    `execution/frame.py`'s `empty_band_wide`, can only arise downstream, from a dimension
    filter narrowing an EXISTING scan's metadata to nothing -- it already holds a `window`
    from that earlier scan and has no need to ask this function for one.) A zero-row
    `metadata_df` therefore raises `ValueError` rather than guessing at a window.

    Band order is `metadata_df`'s own row order, exactly -- row *i* becomes source *i* of
    the VRT, becomes column `b<i>` (see `execution/frame.py`'s band-naming invariant). This
    module computes `b<i>` names itself rather than importing `band_column_names` (see
    module docstring); the two must never diverge, since the whole contract rests on both
    halves of the walk agreeing on the same convention.

    `crosswalks` is a sequence of `(name, path)` pairs, each a crosswalk raster
    co-registered with the rasters `metadata_df` describes (same grid, so it shares their
    window and needs no resampling). Each is appended to the VRT AFTER every data band --
    so band positions, and therefore the `b<i>` names, are unaffected by how many
    crosswalks ride along -- and its pixel values come back in a column called `name`
    rather than `b<something>`. `metadata_df` is returned unchanged: a crosswalk is not a
    band, and giving it a metadata row is what would make it look like one. Resolving which
    file backs a given pair of grids is a caller concern, mirroring
    `io/crosswalk_read.py`'s own path resolution for the vector-to-vector artifact; this
    function takes an already-resolved path.

    `window`, when omitted, reads the FULL raster, tiled into memory-bounded batches (see
    module docstring); when given, restricts the scan to exactly that window, read whole in
    one batch. Either way, `window` is echoed into the returned `ReadWindow` transform/CRS/
    extent -- see `execution/frame.py`'s `assert_window_unchanged`, the invariant every
    downstream lowering checks against what this function handed it first.

    `mask_nodata` (default on) governs DATA bands only. A `crosswalks` column is masked
    unconditionally regardless of this flag -- see module docstring.
    """
    if metadata_df.height == 0:
        raise ValueError(
            "scan_raster requires at least one band -- a window's transform/CRS/extent "
            "can only come from an opened raster, so there is no zero-band answer this "
            "function can give (see execution/frame.py's empty_band_wide for the "
            "downstream-only degenerate case this is not)"
        )
    if "path" not in metadata_df.columns or "band" not in metadata_df.columns:
        raise ValueError("scan_raster requires metadata_df to carry 'path' and 'band' columns")

    crosswalk_names = [str(name) for name, _ in crosswalks]
    _validate_crosswalk_names(crosswalk_names)

    band_sources = list(zip(metadata_df["path"].to_list(), metadata_df["band"].to_list()))
    band_names = [f"b{i}" for i in range(len(band_sources))]
    # A crosswalk raster is single-band by construction, so band 1. Appended after every
    # data band so that adding one never shifts a `b<i>` name.
    sources = band_sources + [(str(path), 1) for _, path in crosswalks]
    column_names = band_names + crosswalk_names
    # Data bands honor the flag; crosswalk columns never do -- the two-tier rule.
    mask_this_band = [mask_nodata] * len(band_sources) + [True] * len(crosswalk_names)

    vrt_bytes = vrt(sources).encode("utf-8")

    # Eager, metadata-only open: a VRT's own bytes carry no pixel data, so this costs one
    # XML parse and one open-per-underlying-file (already paid for by `vrt()` itself just
    # above), not a read. Everything this function returns immediately (the `ReadWindow`,
    # the per-band schema) comes from this single open; every batch below reopens fresh
    # from `vrt_bytes` rather than reusing this handle across a `yield` (see module
    # docstring).
    with raster_gdal_env(), MemoryFile(vrt_bytes, ext=".vrt") as memfile, memfile.open() as ref:
        if ref.count != len(sources):
            raise ValueError(
                f"scan_raster: VRT assembled {ref.count} bands from {len(sources)} sources "
                "-- a source path/band must have resolved to more or fewer bands than "
                "expected"
            )
        raster_transform = ref.transform
        raster_crs = ref.crs
        dtypes = ref.dtypes
        nodatavals = ref.nodatavals
        read_window = window if window is not None else Window(0, 0, ref.width, ref.height)
        if window is not None:
            window_list = [window]
        else:
            block_height, block_width = ref.block_shapes[0]
            itemsize = max(np.dtype(dt).itemsize for dt in dtypes)
            blocks_per_batch = raster_blocks_per_batch(len(sources), block_height, block_width, itemsize)
            window_list = list(batched_windows(ref, blocks_per_batch))

    result_window = ReadWindow(
        source=tuple(sources),
        col_off=int(read_window.col_off),
        row_off=int(read_window.row_off),
        width=int(read_window.width),
        height=int(read_window.height),
        transform=window_transform(read_window, raster_transform),
        crs=raster_crs,
        decimation=1,
    )

    schema = pl.Schema({name: NUMPY_TO_POLARS[dt] for name, dt in zip(column_names, dtypes)})

    def source(with_columns, predicate, n_rows, batch_size):
        # No pushdown yet (projection/predicate pushdown is later-stage work):
        # `with_columns`/`predicate` are accepted (the plugin protocol requires the
        # signature) but not applied here -- Polars applies any `.select`/`.filter` a
        # caller chains onto this scan's own LazyFrame afterward, which is correct, just
        # not optimized. `n_rows` IS honored below: unlike the other two it is not an
        # optimization a caller could apply just as well downstream (a `.head(n)` needs
        # the source to actually stop producing rows, not merely trim them after the
        # fact once memory is already spent).
        remaining = n_rows
        for batch_window in window_list:
            if remaining is not None and remaining <= 0:
                break

            # Nothing GDAL-related crosses the `yield` below -- see module docstring.
            with raster_gdal_env(), MemoryFile(vrt_bytes, ext=".vrt") as memfile, memfile.open() as src:
                data = src.read(window=batch_window)

            cols = {name: data[i].ravel() for i, name in enumerate(column_names)}
            df = pl.DataFrame(cols, schema=schema)
            del data, cols

            mask_exprs = [
                pl.when(pl.col(name) == nd).then(None).otherwise(pl.col(name)).alias(name)
                for name, nd, masked in zip(column_names, nodatavals, mask_this_band)
                if nd is not None and masked
            ]
            if mask_exprs:
                df = df.with_columns(mask_exprs)

            if remaining is not None and df.height > remaining:
                df = df.head(remaining)
            if remaining is not None:
                remaining -= df.height

            yield df

    return register_io_source(source, schema=schema), metadata_df, result_window
