"""The raster scan: `scan_raster` opens a band-wide dataset as the band-wide triple
(`execution/frame.py`'s `BandWideFrame`), reading each source file directly.

**Bands in a declared order, read one file at a time.** A band-wide frame's columns are
bands of files -- `metadata_df` row *i* becomes column `b<i>`, and `execution/frame.py`'s
band-naming rule is that this correspondence is exact and positional. Each batch opens
every distinct file once and reads all the bands the request wants from it in one
`read(indexes=[...])`, then places each band at its row's position. A bare path in the
sources means band 1 (a crosswalk is single-band by construction).

**One read per file.** This scan used to assemble a per-request VRT over the `(path,
band)` sources and read through it, which kept N files' windows in step for free. It also
made every read band by band: the COGs this scans are PIXEL-interleaved
(`cpra.mp.data.raster`'s `compact_cog`), so each band read decompressed every block's bands
into GDAL's block cache, once per band wanted. A single multi-band read of a file decodes
each block once, straight into the output, across `GDAL_NUM_THREADS`. Measured over the
full 16640x5824 `morph_pixel_v002` extent at prod's settings: two bands of one file 7.7 s /
811 MiB peak through the VRT against 2.0 s / 486 MiB direct; one band from each of two
files 11.0 s against 6.8 s; 25 bands across two files 15.3 s against 10.2 s. A lone band,
and all of one file's bands, read about the same either way. Decimated tile reads gain
little, because GDAL already serves them off overviews, and they come back pixel-identical
to the VRT's read (checked across tile-sized, unaligned and full-extent windows at
decimations 1 to 16; `tests/api/test_raster_scan.py` keeps a VRT read as its reference).
The VRT's metadata open went with it: opening the files directly is 3-4x cheaper.

Reading every file through one window and one `out_shape` requires every source to share
one grid -- size, transform and CRS. The VRT assumed that silently (it placed each source at
the full extent); `_require_same_grid` checks it against the first file.

**A crosswalk is read as a band but is NOT a band column.** A raster-to-vector crosswalk is
co-registered with the raster grid, so reading it is an ordinary windowed read with no
resampling and it can ride along as one more source of the same scan -- that is a fact
about the READ, and this module uses it. What comes back is a different matter: a crosswalk holds
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
`raster_blocks_per_batch`, a block-row-chunked budget, so a request over a large raster
does not pull the whole window into memory as one array. Each
batch reopens its files rather than holding a dataset handle open across a `yield`: Polars' streaming engine may resume this generator's `source`
callback from a different thread than the one that entered it, and both `rio.Env` and
rasterio dataset handles are thread-local, so nothing opened before a `yield` may be
assumed valid after it.

`window`, when given, restricts the scan to that single `rasterio.windows.Window` read
whole, in one batch, rather than tiling the full raster -- a caller specific enough to name
its own window has already decided how much of it to hold in memory at once. This is
plumbing only: nothing in this module computes a window on its own, nor how coarsely to
read it (both are the caller's decision, not this scan's -- `execution.plan.compile_plan`
makes them together, since which pixels a tile needs and whether that tile may be served
off overviews are the same request's questions).

**Decimation.** `decimation` is a per-axis `(row_factor, col_factor)`, and reading a window
into a correspondingly smaller buffer is what makes GDAL serve it out of the source's own
overview pyramid (rasterio passes `out_shape` straight through to GDAL's overview
selection) instead of decompressing every full-resolution pixel. It is only accepted
alongside an explicit `window`: a decimated read is a single-batch read by construction,
because the batch layout a full read is chunked into is expressed in source pixels and
floor-dividing each batch separately would not tile the decimated array (`execution/
frame.py`'s `row_major_grid_cell` decodes a row index through that layout, and would
decode it to the wrong pixel). Every read here stays `Resampling.nearest` -- GDAL's own
default for an unspecified resampling method -- decimated or not, which is what keeps every
value a real source value: the COGs this reads carry overviews built with NEAREST (see
`cpra.mp.data.raster`'s `_cog_profile`), so a decimated pixel is a pixel that was really
there, and a band's NoData sentinel survives decimation intact for the masking below to
find.

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
import rasterio
from polars.io.plugins import register_io_source
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from cpra.mp.data.analysis.io.raster_io import batched_windows, raster_blocks_per_batch, raster_gdal_env
from cpra.mp.data.analysis.io.window import ReadWindow
from cpra.mp.data.analysis.request.pipeline import BAND_NAME_RE, RESERVED_BAND_INDEX_COLUMN

__all__ = ["scan_raster"]

# numpy dtype name -> the Polars dtype a band of that raster reads back as.
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


def _require_same_grid(src, ref) -> None:
    """Reject a source file that is not on `ref`'s grid.

    Every read in a scan uses one window and one `out_shape` across all files, so a file
    with a different size, transform or CRS would come back as pixels from different ground
    under the same row, with nothing in the frame's shape to show it. Exact comparison:
    a grid's data files and its crosswalks are all written from the same grid definition,
    so they agree bit for bit.
    """
    if (src.width, src.height) != (ref.width, ref.height) or src.transform != ref.transform or src.crs != ref.crs:
        raise ValueError(
            f"scan_raster: {src.name} is not on the same grid as {ref.name} -- every source "
            f"must share one size, transform and CRS ({src.width}x{src.height} {src.transform!r} "
            f"{src.crs} against {ref.width}x{ref.height} {ref.transform!r} {ref.crs})"
        )


def scan_raster(
    metadata_df: pl.DataFrame,
    *,
    crosswalks: Sequence[tuple[str, object]] = (),
    window: Window | None = None,
    decimation: tuple[int, int] = (1, 1),
    mask_nodata: bool = True,
) -> tuple[pl.LazyFrame, pl.DataFrame, ReadWindow]:
    """Open `metadata_df`'s bands as a band-wide triple, reading each file directly.

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
    the scan, becomes column `b<i>` (see `execution/frame.py`'s band-naming invariant). This
    module computes `b<i>` names itself rather than importing `band_column_names` (see
    module docstring); the two must never diverge, since the whole contract rests on both
    halves of the walk agreeing on the same convention.

    `crosswalks` is a sequence of `(name, path)` pairs, each a crosswalk raster
    co-registered with the rasters `metadata_df` describes (same grid, so it shares their
    window and needs no resampling). Each is appended to the sources AFTER every data band --
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

    `decimation` is the `(row_factor, col_factor)` to read `window` at (see module
    docstring), `(1, 1)` -- full resolution -- by default, and REQUIRES an explicit
    `window`: a factor without one raises `ValueError` rather than silently reading whole.
    A crosswalk rides the same decimated read as the data bands, so its zone ids stay
    positionally aligned with them for free, and stay real zone ids rather than blends
    (`Resampling.nearest`). The returned `ReadWindow` keeps `window`'s own SOURCE extent in
    `col_off`/`row_off`/`width`/`height` -- it still describes the same ground -- while its
    `transform` is scaled to the pixels that actually came back, so a consumer holding only
    the window (`execution/sink.py`) can georeference the array without knowing it was
    decimated at all. `ReadWindow.read_shape` is that array's shape.

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

    row_factor, col_factor = (int(factor) for factor in decimation)
    if row_factor < 1 or col_factor < 1:
        raise ValueError(f"scan_raster: decimation factors must be >= 1, got {decimation!r}")
    decimated = row_factor != 1 or col_factor != 1
    if decimated and window is None:
        raise ValueError(
            f"scan_raster: decimation {decimation!r} requires an explicit window -- a full "
            "read is chunked into batches whose layout is expressed in source pixels, which "
            "a decimated array does not tile (see the module docstring)"
        )

    crosswalk_names = [str(name) for name, _ in crosswalks]
    _validate_crosswalk_names(crosswalk_names)

    band_sources = [(str(path), int(band)) for path, band in zip(metadata_df["path"], metadata_df["band"])]
    band_names = [f"b{i}" for i in range(len(band_sources))]
    # A crosswalk raster is single-band by construction, so band 1. Appended after every
    # data band so that adding one never shifts a `b<i>` name.
    sources = band_sources + [(str(path), 1) for _, path in crosswalks]
    column_names = band_names + crosswalk_names
    # Data bands honor the flag; crosswalk columns never do -- the two-tier rule.
    mask_this_band = [mask_nodata] * len(band_sources) + [True] * len(crosswalk_names)

    # Eager, metadata-only opens, one per distinct file (in first-appearance order), not a
    # read. Everything this function returns immediately (the `ReadWindow`, the per-band
    # schema) comes from here; the batches below reopen each file fresh rather than
    # holding a handle across a `yield` (see module docstring). The first file is the
    # reference: its grid is the scan's, and every other file must match it.
    paths = list(dict.fromkeys(path for path, _band in sources))
    with raster_gdal_env(), rasterio.open(paths[0]) as ref:
        file_bands = {paths[0]: (ref.count, ref.dtypes, ref.nodatavals)}
        for path in paths[1:]:
            with rasterio.open(path) as src:
                _require_same_grid(src, ref)
                file_bands[path] = (src.count, src.dtypes, src.nodatavals)
        dtypes, nodatavals = [], []
        for path, band in sources:
            count, file_dtypes, file_nodatavals = file_bands[path]
            if not 1 <= band <= count:
                raise ValueError(f"scan_raster: {path} has {count} band(s), so it has no band {band}")
            dtypes.append(file_dtypes[band - 1])
            nodatavals.append(file_nodatavals[band - 1])
        raster_transform = ref.transform
        raster_crs = ref.crs
        read_window = window if window is not None else Window(0, 0, ref.width, ref.height)
        if window is not None:
            window_list = [window]
        else:
            block_height, block_width = ref.block_shapes[0]
            itemsize = max(np.dtype(dt).itemsize for dt in dtypes)
            blocks_per_batch = raster_blocks_per_batch(len(sources), block_height, block_width, itemsize)
            window_list = list(batched_windows(ref, blocks_per_batch))

    source_height, source_width = int(read_window.height), int(read_window.width)
    out_shape = (max(1, source_height // row_factor), max(1, source_width // col_factor)) if decimated else None

    window_affine = window_transform(read_window, raster_transform)
    if out_shape is not None:
        # The array covers the window's whole ground extent at a coarser resolution, so its
        # origin is the window's and only its pixel size changes. Scaled by the EFFECTIVE
        # ratio (source extent over the shape actually returned) rather than by the
        # requested factors, since `out_shape` is floor-divided and the two differ whenever
        # the extent is not a multiple of the factor.
        window_affine *= Affine.scale(source_width / out_shape[1], source_height / out_shape[0])

    result_window = ReadWindow(
        source=tuple(sources),
        col_off=int(read_window.col_off),
        row_off=int(read_window.row_off),
        width=source_width,
        height=source_height,
        transform=window_affine,
        crs=raster_crs,
        decimation=(row_factor, col_factor),
        # Empty for a decimated read -- "the whole window, one batch", which is what a
        # decimated read always is. Recording the source-pixel extents a full-resolution
        # single-batch read records would be a unit mismatch against the smaller array the
        # rows actually came from, and `frame.row_major_grid_cell` reads this layout in
        # array units.
        batch_layout=()
        if decimated
        else tuple((int(w.col_off), int(w.row_off), int(w.width), int(w.height)) for w in window_list),
    )

    schema = pl.Schema({name: NUMPY_TO_POLARS[dt] for name, dt in zip(column_names, dtypes)})

    # One `rasterio` call per distinct file per batch, asking for every band this request
    # wants from it at once -- see the module docstring's "One read per file". Files are in
    # first-appearance order and each keeps the output positions its bands fill, so `b<i>`
    # order is unaffected. A GeoTIFF's bands all share one dtype, so a per-file read never
    # has to reconcile two.
    file_reads: dict[str, tuple[list[int], list[int]]] = {}
    for position, (path, band) in enumerate(sources):
        indexes, positions = file_reads.setdefault(path, ([], []))
        indexes.append(band)
        positions.append(position)

    def source(with_columns, predicate, n_rows, batch_size):
        # Every one of the four arguments MUST be honored here. An IO plugin does not
        # "accept but ignore" a pushed-down projection or predicate: Polars removes the
        # `.select`/`.filter` from the plan when it hands it over and does NOT re-apply it
        # afterward, so a plugin that takes one and drops it silently returns unfiltered
        # rows (and unprojected columns) that the plan says were already dealt with. Both
        # were confirmed directly: a `.filter()` chained onto this scan returned every row,
        # and a `.select("b1")` returned `b0` alongside it.
        #
        # Honoring them here is not a read-side optimization -- the pixels are read either
        # way, since a projection arrives as column names after every band has been read. It is what makes the scan CORRECT. `n_rows` is the
        # one that does cut the read short, and it is applied last, after the predicate,
        # because it means "the first n rows of what this source produces".
        remaining = n_rows
        for batch_window in window_list:
            if remaining is not None and remaining <= 0:
                break

            # Nothing GDAL-related crosses the `yield` below -- see module docstring.
            by_position = {}
            with raster_gdal_env():
                for path, (indexes, positions) in file_reads.items():
                    with rasterio.open(path) as src:
                        # `out_shape` is a 2-tuple even for a multi-band read: rasterio
                        # broadcasts it across `indexes` and returns
                        # `(len(indexes), *out_shape)`. `None` (the undecimated case) reads
                        # the window at full resolution.
                        file_data = src.read(indexes=indexes, window=batch_window, out_shape=out_shape)
                    for i, position in enumerate(positions):
                        by_position[position] = file_data[i].ravel()
                    del file_data

            cols = {name: by_position[i] for i, name in enumerate(column_names)}
            df = pl.DataFrame(cols, schema=schema)
            del by_position, cols

            mask_exprs = [
                pl.when(pl.col(name) == nd).then(None).otherwise(pl.col(name)).alias(name)
                for name, nd, masked in zip(column_names, nodatavals, mask_this_band)
                if nd is not None and masked
            ]
            if mask_exprs:
                df = df.with_columns(mask_exprs)

            # After masking, never before: a mask expression reads every column of the
            # batch, including ones a projection is about to drop.
            if with_columns is not None:
                df = df.select(with_columns)
            if predicate is not None:
                df = df.filter(predicate)

            if remaining is not None and df.height > remaining:
                df = df.head(remaining)
            if remaining is not None:
                remaining -= df.height

            yield df

    return register_io_source(source, schema=schema), metadata_df, result_window
