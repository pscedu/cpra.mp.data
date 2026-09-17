"""Analysis-layer helpers (the former api/utils.py, HTTP-free half). Nothing here may import FastAPI."""
import asyncio
import math
from enum import Enum
from functools import wraps
from time import time

import rasterio as rio
from cachetools import TTLCache, cached
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import Affine, from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from cpra.mp.data import read_data
from cpra.mp.data.analysis.types import CompressionAlgorithm
from cpra.mp.data.config import (
    DEBUG,
    RASTER_BATCH_BUFFER_MB,
    RASTER_BLOCKS_PER_BATCH,
    RASTER_CACHE_MAX_MB,
    RASTER_NUM_THREADS,
    RASTER_SWATH_SIZE_MB,
)
from cpra.mp.data.exceptions import ValidationError

RASTER_COMPRESSION_ALGORITHMS = [
    CompressionAlgorithm.deflate,
    CompressionAlgorithm.lzw,
    CompressionAlgorithm.lzma,
    CompressionAlgorithm.none,
    CompressionAlgorithm.zstd,
]

TABULAR_COMPRESSION_ALGORITHMS = [
    CompressionAlgorithm.gzip,
    CompressionAlgorithm.lz4,
    CompressionAlgorithm.none,
    CompressionAlgorithm.zstd,
]


def unpack_values(values_str, param_name, choices=None, separator=","):
    """Unpack values from a comma-separated list."""
    values = [val.strip() for val in values_str.split(separator)]

    # Validate choices.
    if choices:
        invalid_values = [val for val in values if val not in choices]
        if invalid_values:
            # Local import: avoids a cpra.mp.data.analysis.base <-> .io.raster_io import
            # cycle (base.py imports get_lazy_frame/unpack_values/timing from this module
            # at module scope; AnalysisError itself is defined in base.py per that
            # module's docstring).
            from cpra.mp.data.analysis.base import AnalysisError

            msg = f"Invalid values for {param_name}: {', '.join(invalid_values)}"
            raise AnalysisError(msg, status_code=400)

    return values


@cached(cache=TTLCache(maxsize=128, ttl=600))
def _read_data_cached(*args, **kwargs):
    return read_data(*args, **kwargs)


def get_lazy_frame(dataset_dimensions, filter_dimensions, base_path=None):
    """Get a LazyFrame for the given dimensions with error handling."""
    try:
        return _read_data_cached(
            **({"base_path": base_path} if base_path is not None else {}),
            **dict(dataset_dimensions),
            **{k: tuple(v) for (k, v) in dict(filter_dimensions).items() if isinstance(v, list)},
        )
    except ValidationError as err:
        # Local import: see the note in unpack_values above.
        from cpra.mp.data.analysis.base import AnalysisError

        # TODO: Also handle directory does not exist errors.
        raise AnalysisError(str(err), status_code=422)


def enum_value(inst):
    """Get the value from an enum."""
    return inst.value if isinstance(inst, Enum) else inst


def asynctiming(func):
    """Time an asynchronous function call."""

    @wraps(func)
    async def wrapped(*args, **kwargs):
        start = time()
        result = await func(*args, **kwargs)
        if DEBUG:
            print("func:%r args:[%r, %r] took: %2.4f sec" % (func.__name__, args, kwargs, time() - start))
        return result

    return wrapped


def timing(func):
    """Time a synchronous function call."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        if DEBUG:
            print("func:%r args:[%r, %r] took: %2.4f sec" % (func.__name__, args, kwargs, time() - start))
        return result

    return wrapped


def raster_gdal_env():
    """GDAL configuration for reading and writing rasters.

    ``GDAL_NUM_THREADS`` parallelizes tile decompression, which dominates every read of
    a multiband COG. Setting it measured 2.2-3.7x faster across band counts with
    byte-identical output -- but only for a dataset *opened* with it in effect (measured on
    real data: opening under this env then reading plainly took ~5s for a 10-band window
    vs ~15s with no env anywhere; entering the env only around the later ``.read()`` call,
    on an already-opened dataset, recovered almost none of that, ~12-14s). GDAL_NUM_THREADS
    is effectively latched in at open time for the GeoTIFF/COG driver, not re-read per call
    -- see the callers below, which enter this env around ``rio.open()`` for exactly that
    reason, narrowly (not held open for the whole read loop after).

    CORRECTED: an earlier version of this docstring claimed that selecting a
    subset of bands "still decompresses each tile's full band payload, so a 1-band read
    costs nearly as much as a 52-band one." Re-measured against real long-format data and
    found false: I/O-only reads scale with the number of bands actually read, not flat (on
    this machine: 1.22s for 2 bands vs 2.42s for 11 from one file; 1.11s for 2 bands vs
    4.81s for 21 of 52 from two files -- a large fixed per-read overhead means it isn't
    perfectly linear, but a narrow band selection is genuinely cheaper, not merely
    memory-light). ``RasterGridAnalysis._windowed_read_is_lighter``'s routing already keys
    on the windowed path's buffer *size* (bands x block height x width) rather than a time
    assumption, so it needed no change once this claim was checked -- confirmed by reading
    it, not merely assumed.

    Capping ``GDAL_CACHEMAX`` is what keeps the windowed path's peak memory down, and unlike
    ``GDAL_NUM_THREADS`` it must be in effect at *read* time to matter, not open time
    (measured on real data, holding ``GDAL_NUM_THREADS`` fixed: capping ``GDAL_CACHEMAX``
    only at open produced no reduction in peak RSS at all -- 3706 MB either way -- while
    capping it at read time took the same read down to 491 MB). Reading one band of a
    52-band float32 COG this way went from 4226 MB to 1027 MB peak RSS, at no measurable
    cost in time. Above a handful of bands the path's own NumPy buffers dominate instead and
    the cap can't help on its own -- see ``raster_blocks_per_batch``, which bounds that
    buffer directly rather than relying on the GDAL cache to do it. Being process-global,
    the cap also bounds cache growth across concurrent requests rather than leaving it at
    GDAL's default 5% of RAM.

    ``rio.Env`` is thread-local storage (``rasterio.env.ThreadEnv`` subclasses
    ``threading.local``), but the GDAL config option it sets is not always thread-scoped:
    rasterio's ``_env.pyx`` uses ``CPLSetConfigOption`` (process-global) when the *calling*
    thread is the main thread, and ``CPLSetThreadLocalConfigOption`` otherwise, and every
    thread falls back to the main thread's global value unless it has set its own override.
    Confirmed empirically (a main-thread-entered env's config option was visible from a
    ``loop.run_in_executor`` worker thread without that thread entering anything itself).
    That means entering this env once in the calling coroutine *would* reach worker-thread
    reads dispatched via ``run_in_executor``/``to_thread`` -- but it would just as easily
    reach any *other* concurrent request's reads sharing the same default thread pool for as
    long as the env stayed open, since the option is genuinely process-global with no
    per-request scoping. Confirmed empirically too: an unrelated concurrent "tile" read that
    never touched this env observed ``GDAL_NUM_THREADS`` leaked in from a second, overlapping
    "summarize" coroutine's still-open env. Entering the env inside the worker-thread
    function that performs the actual read (see ``_read_window``'s ``use_gdal_env``
    argument) avoids that leak (that thread is not the main thread, so it goes through the
    thread-local path and reverts on exit) while entering it narrowly around a ``rio.open()``
    call in the calling coroutine (see callers) is an acceptably small, bounded exposure
    given how fast an open is relative to the read/write loop that follows it.

    Deliberately not applied to tile requests. Those read a window decimated to roughly the
    tile's own resolution (``tile_read_shape``), which GDAL serves out of the source's
    overviews -- a small enough read that threading measured no benefit on it.
    """

    return rio.Env(GDAL_NUM_THREADS=RASTER_NUM_THREADS, GDAL_CACHEMAX=RASTER_CACHE_MAX_MB)


def raster_copy_gdal_env():
    """GDAL configuration for the VRT + ``CreateCopy`` download path.

    ``CreateCopy`` streams the output in ``GDAL_SWATH_SIZE`` chunks, only 10MB by default.
    Each chunk re-enters GDAL's per-block path, which for a PIXEL-interleaved multiband
    COG decompresses every band of every tile it touches -- the same cost that makes a
    one-block-row-at-a-time loop slow. Raising the swath amortizes it: measured 91.8s ->
    22.8s for one band of a 52-band float32 COG. GDAL clamps the swath to half of
    ``GDAL_CACHEMAX``, so the cache has to be raised alongside it; capping the cache is
    also what keeps peak memory down (3-4x lower than the windowed path).

    ``GDAL_CACHEMAX`` must be an int -- a str raises ``TypeError`` from ``rio.Env``.

    ``GDAL_PAM_ENABLED=NO`` stops GDAL writing a ``.aux.xml`` sidecar for the per-band
    descriptions the copy carries over (the GTiff output embeds them natively). The
    response streams only the TIFF, so a sidecar would just be orphaned in the temp dir.
    """

    return rio.Env(
        GDAL_NUM_THREADS=RASTER_NUM_THREADS,
        GDAL_SWATH_SIZE=str(RASTER_SWATH_SIZE_MB * 1024 * 1024),
        GDAL_CACHEMAX=RASTER_CACHE_MAX_MB,
        GDAL_PAM_ENABLED="NO",
    )


def raster_blocks_per_batch(n_bands, block_height, block_width, itemsize):
    """Number of adjacent raster blocks to union into one summarize-loop batch.

    ``RasterGridAnalysis._apply_transform_params_impl`` used to union a whole block row per
    iteration regardless of how many blocks that row contained, so one iteration's read
    buffer (``n_bands`` bands x ``block_height`` x the row's full width) grew with the
    raster's width and the request's band count alike -- 20674 px wide block rows meant
    ~10M pixels x N bands per iteration no matter how small the actual query. Chunking the
    row into groups of ``K`` blocks bounds that buffer directly (measured 4 block rows, 52
    bands: whole row 12.0s/4143MB -> 8 blocks 10.6s/2243MB -> 4 blocks 9.8s/1753MB -> 2
    blocks 9.1s/1363MB -- chunking is both faster and lighter, so there's no tradeoff to
    tune against).

    ``RASTER_BLOCKS_PER_BATCH``, if set, overrides the computed value outright. Otherwise
    ``K`` is derived from ``RASTER_BATCH_BUFFER_MB`` (itself defaulted from
    ``RASTER_CACHE_MAX_MB``) and the actual band count and block shape of the request, so a
    wide band selection or a grid with a different block/dtype size doesn't inherit a
    constant tuned for 52 int32 bands on 512x512 blocks.
    """
    if RASTER_BLOCKS_PER_BATCH is not None:
        return RASTER_BLOCKS_PER_BATCH
    per_block_bytes = max(n_bands, 1) * block_height * block_width * itemsize
    if per_block_bytes <= 0:
        return 1
    budget_bytes = RASTER_BATCH_BUFFER_MB * 1024 * 1024
    return max(1, budget_bytes // per_block_bytes)


def group_rows_by_path(df):
    """Group a materialized index DataFrame's rows by unique ``path``.

    Multiband COGs mean several df rows (different bands, e.g. different
    ``calendar_year`` values) can share the same file. Callers use this to open each
    file once and request all of its needed bands in a single read/sample call, rather
    than reopening the same file per row.

    Returns ``(paths, row_indices, bands)`` where ``paths`` are the unique paths in
    first-seen order, ``row_indices[i]`` are the df row positions backed by
    ``paths[i]``, and ``bands[i]`` are the corresponding band numbers (1-based, in the
    same order as ``row_indices[i]``), or None if df has no ``band`` column.
    """
    paths = df["path"].to_list()
    has_bands = "band" in df.columns
    row_bands = df["band"].to_list() if has_bands else [None] * len(paths)

    order = []
    row_indices_by_path = {}
    bands_by_path = {}
    for row_idx, (path, band) in enumerate(zip(paths, row_bands)):
        if path not in row_indices_by_path:
            order.append(path)
            row_indices_by_path[path] = []
            bands_by_path[path] = []
        row_indices_by_path[path].append(row_idx)
        bands_by_path[path].append(band)

    row_indices = [row_indices_by_path[path] for path in order]
    bands = [bands_by_path[path] if has_bands else None for path in order]
    return order, row_indices, bands


#: The CRS every map tile is rendered in.
TILE_CRS = "EPSG:3857"

#: Whole source pixels read around a tile's own footprint, so the warp onto the tile grid
#: has neighbouring pixels to sample at the tile's edge rather than running off the end of
#: the buffer it was handed. Four is what `tile_source_window`'s callers were measured
#: against; with `Resampling.nearest` (the only resampling either tile path uses) a single
#: pixel would do, and a larger margin costs only the extra pixels read.
TILE_READ_BUFFER_PX = 4


def tile_source_window(src, tile_bounds, buffer_px=TILE_READ_BUFFER_PX):
    """The window of `src` (in its OWN CRS) covering the Web Mercator `tile_bounds`, as
    whole source pixels clipped to the source's extent -- or `None` when the tile's
    footprint lands entirely outside the source.

    Two properties of the returned window are what let a caller read it, run per-pixel work
    over the result in the SOURCE CRS, and only then warp onto the tile grid (see
    `warp_window_to_tile`):

    * It is a whole-pixel window: every offset and extent is an integer, so
      `rasterio.windows.transform(window, src.transform)` describes the returned buffer
      exactly -- same resolution and same grid phase as `src` itself, offset by an integer
      number of pixels.
    * It is INTERSECTED with the source extent rather than read `boundless=True` with
      nodata padding. GDAL treats "outside the source extent" and "inside it but nodata"
      differently when it snaps the warp at the data boundary, and padding measured one
      pixel of erosion along that boundary on tiles that straddle it. Clipping instead
      measured none. `tests/api/test_tile_warp.py` pins both halves of that.

    `transform_bounds` densifies the tile's edges before reprojecting (a straight
    Web-Mercator edge is curved in a projected source CRS, so its corners alone understate
    the footprint). A tile large enough that its reprojection is not finite at all -- a
    whole-world tile at low zoom, say -- falls back to the source's full extent, which is
    the smallest window that certainly contains the footprint.
    """
    src_bounds = transform_bounds(TILE_CRS, src.crs, *tile_bounds, densify_pts=21)
    if not all(math.isfinite(value) for value in src_bounds):
        return windows.Window(0, 0, src.width, src.height)

    window = windows.from_bounds(*src_bounds, transform=src.transform)
    col_off = max(0, math.floor(window.col_off) - buffer_px)
    row_off = max(0, math.floor(window.row_off) - buffer_px)
    col_end = min(src.width, math.ceil(window.col_off + window.width) + buffer_px)
    row_end = min(src.height, math.ceil(window.row_off + window.height) + buffer_px)
    if col_end <= col_off or row_end <= row_off:
        return None
    return windows.Window(col_off, row_off, col_end - col_off, row_end - row_off)


#: How much finer than the tile's own pixel grid a decimated tile read is kept.
#:
#: A tile pixel takes ONE source pixel under `Resampling.nearest`, so if the decimated grid
#: were at the tile's own resolution the pixel it picks could be up to ~1.5 decimated pixels
#: -- i.e. more than a whole tile pixel -- away from the one a full-resolution read would
#: have picked. Measured over 38 tiles (z=7..14, 1,741,449 rendered pixels compared) of the
#: 2176x2176 unique-per-pixel fixture `tests/api/test_tile_warp.py` builds: reading at the
#: tile's own resolution displaces a rendered pixel by up to 0.75 tile pixels, reading twice
#: as fine by up to 0.15. The cost is 4x the pixels of a 1:1 rule, which at low zoom is still
#: two orders of magnitude below a full-resolution read (434 MB -> 3.4 MB per band at z=6 on
#: `morph_pixel_v001`, versus 0.8 MB at 1:1).
#:
#: It is a margin, not a guarantee: a decimated read still takes a different source pixel
#: than a full-resolution one, which is what `TestDecimatedTileReads` bounds rather than
#: asserts away.
TILE_READ_OVERSAMPLE = 2


def _axis_decimation_factor(extent, target):
    """The largest power of two by which `extent` source pixels can be decimated and still
    leave at least `target` of them -- 1 (read this axis whole) when it cannot be halved."""
    ratio = int(extent // target)
    if ratio < 2:
        return 1
    return 1 << (ratio.bit_length() - 1)


def tile_read_shape(window, tile_size, buffer_px=TILE_READ_BUFFER_PX, oversample=TILE_READ_OVERSAMPLE):
    """The ``(height, width)`` to decimate `window` to while reading it for a tile, or None
    when `window` is already small enough to read whole.

    A tile's source window is only as small as the tile's ground footprint, which at low
    zoom is most of the raster: `tile_source_window` on a 20674x8285 source returns a
    13107x8285 window at z=6, 434 MB per band for the 65,536 pixels a 256x256 tile actually
    renders. Reading that window into a smaller buffer makes GDAL serve it from the source's
    overviews (rasterio passes ``out_shape`` straight through to GDAL's own overview
    selection), so the decompressed volume drops with the square of the factor.

    The factor is chosen so that:

    * it never decimates below `oversample` times the tile's own resolution -- the target is
      `tile_size` plus `tile_source_window`'s buffer on both sides, so the returned shape
      still covers the same ground as the window at a resolution finer than the tile grid
      it is about to be warped onto (see `TILE_READ_OVERSAMPLE` for what the margin buys);
    * it never UPSAMPLES -- an axis already at or below that size keeps every one of its
      pixels, and a window small enough on BOTH axes returns None, which the callers read
      exactly as they did before this existed;
    * it is a power of two, so it lands on a real overview level (the COGs this reads carry
      ``[2, 4, 8, 16, 32, 64]``, built with NEAREST -- see `raster._cog_profile` -- which is
      what keeps every decimated pixel a real source value rather than a blend) rather than
      making GDAL resample down from a finer one;
    * it is chosen PER AXIS. A single factor taken from the shorter axis would leave the
      read unbounded on the longer one: a tile straddling the source's edge gets a window
      clipped to a sliver -- 2176x817 on the test fixture, 20674 x a few hundred on
      morph_pixel -- whose short axis forbids decimating at all while its long axis is still
      the whole raster. Per axis, the returned shape is under
      ``2 * oversample * (tile_size + 2 * buffer_px)`` on BOTH axes for every window there
      is, which is the bound that makes a tile's cost a function of the tile rather than of
      the zoom (`tests/api/test_tile_warp.py::TestDecimatedTileReads` pins it). When the two
      factors differ GDAL can serve only one axis straight off the matching overview level
      and decimates the other itself; the array handed back is bounded either way, and under
      `Resampling.nearest` both axes stay whole source values.

    The returned shape is `window` floor-divided by those factors, not ceil-divided: the
    effective ratio has to be at least the factor for GDAL to pick that overview level
    rather than the next finer one. The shape still describes the WHOLE window -- it is the
    same ground extent at a coarser resolution -- which is what `warp_window_to_tile` needs
    to scale the georeferencing by.
    """
    target = oversample * (tile_size + 2 * buffer_px)
    height_factor = _axis_decimation_factor(window.height, target)
    width_factor = _axis_decimation_factor(window.width, target)
    if height_factor == 1 and width_factor == 1:
        return None
    return (max(1, int(window.height) // height_factor), max(1, int(window.width) // width_factor))


def _warp_array_to_tile(data, array_transform, src_crs, tile_bounds, tile_size, nodata):
    """Warp `data` (a ``(bands, height, width)`` array georeferenced by `array_transform`
    in `src_crs`) onto a tile's own Web Mercator pixel grid, through an in-memory dataset.

    `array_transform` must describe the array PASSED IN. `warp_window_to_tile` is the entry
    point that guarantees that; this function takes the transform as a bare argument so a
    test can hand it a deliberately wrong one and measure what that costs
    (`tests/api/test_tile_warp.py`'s control).

    The warp targets the tile grid directly -- explicit `transform`/`width`/`height`, no
    intermediate output grid -- and `Resampling.nearest` keeps every rendered pixel a real
    source pixel value rather than a blend of several.
    """
    tile_transform = from_bounds(*tile_bounds, tile_size, tile_size)
    count, height, width = data.shape
    dtype = data.dtype.name
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=dtype,
            crs=src_crs,
            transform=array_transform,
            nodata=nodata,
        ) as mem:
            mem.write(data)
        # Reopened read-only: rasterio deprecates warping a dataset that is still open for
        # writing, and the written bytes are already complete once the block above exits.
        with memfile.open() as mem:
            with WarpedVRT(
                mem,
                crs=TILE_CRS,
                transform=tile_transform,
                width=tile_size,
                height=tile_size,
                nodata=nodata,
                src_nodata=nodata,
                add_alpha=False,
                dtype=dtype,
                resampling=Resampling.nearest,
            ) as vrt:
                return vrt.read()


def warp_window_to_tile(data, *, window, src_transform, src_crs, tile_bounds, tile_size, nodata, out_shape=None):
    """Warp one window's worth of per-pixel results onto a tile's pixel grid.

    `data` is whatever a caller computed over the pixels of `window`, still in the source
    CRS and still on the source's own grid, and `window`/`src_transform` are the window that
    was READ and the transform of the dataset it was read from. `out_shape` is the
    ``(height, width)`` that window was DECIMATED to on the way in (`tile_read_shape`), or
    None when it was read whole -- `data` is ``(bands, *out_shape)`` in the first case and
    ``(bands, window.height, window.width)`` in the second.

    The georeferencing handed to the warp is derived here, from those three, rather than
    accepted from the caller: a transform built from anything other than the array actually
    handed to the warp is a sub-pixel origin error that displaces every rendered pixel, with
    no symptom short of a differential against another implementation (see
    `tests/api/test_tile_warp.py`, which measures exactly that). A decimated read is where
    that is easiest to get wrong -- the array is smaller than the window while covering the
    same ground, so the window's own transform describes it at the wrong resolution -- which
    is why the scaling lives here too and not at either call site.

    The shape check below is the mechanical half of the same guarantee -- an array that
    matches neither the window nor the shape it was decimated to cannot reach the warp.
    """
    expected = tuple(out_shape) if out_shape is not None else (window.height, window.width)
    if tuple(data.shape[-2:]) != expected:
        raise ValueError(
            f"warp_window_to_tile: data is {data.shape[-2:]} but window is "
            f"{expected} -- the array must be exactly the window "
            "that was read, or its georeferencing is wrong"
        )
    array_transform = windows.transform(window, src_transform)
    if out_shape is not None:
        array_transform *= Affine.scale(window.width / expected[1], window.height / expected[0])
    return _warp_array_to_tile(data, array_transform, src_crs, tile_bounds, tile_size, nodata)


def _read_window(dataset, window, out_shape=None, resampling=Resampling.nearest, indexes=None, use_gdal_env=False):
    """Read a window of data from a single raster (optionally a subset of bands).

    ``use_gdal_env`` enters ``raster_gdal_env()`` around the read itself, which is what
    keeps ``GDAL_CACHEMAX`` bounding memory for *this* read specifically (it must be live
    at read time -- see ``raster_gdal_env``'s docstring) without leaving a process-global
    GDAL config change sitting open for the whole duration of an unrelated concurrent
    request that happens to dispatch onto the same worker thread pool. Kept as an opt-in
    argument rather than always-on because tile requests deliberately skip it (see
    ``raster_gdal_env``).
    """
    if use_gdal_env:
        with raster_gdal_env():
            return dataset.read(indexes=indexes, window=window, out_shape=out_shape, resampling=resampling)
    return dataset.read(indexes=indexes, window=window, out_shape=out_shape, resampling=resampling)


async def parallel_read_rasters(
    datasets, window=None, out_shape=None, resampling=Resampling.nearest, bands=None, use_gdal_env=False
):
    """Read several rasters in multiple threads.

    ``bands`` optionally selects, per dataset, the list of band numbers (1-based) to
    read; each dataset's selected bands are read together as one ``(n, height, width)``
    array. When ``bands`` is None every band of each dataset is read.

    ``use_gdal_env`` is forwarded to ``_read_window``, entering ``raster_gdal_env()`` on
    each worker thread that performs a read, scoped to just that read -- see that
    function's and ``raster_gdal_env``'s docstrings for why that scoping (rather than
    entering the env once in the calling coroutine) matters here.
    """
    loop = asyncio.get_running_loop()
    if bands is None:
        bands = [None] * len(datasets)
    return await asyncio.gather(
        *[
            loop.run_in_executor(None, _read_window, dataset, window, out_shape, resampling, band, use_gdal_env)
            for dataset, band in zip(datasets, bands)
        ]
    )


def _sample_raster(dataset, coordinates, bands=None):
    """Sample a single raster at the given coordinates (optionally a subset of bands)."""
    return list(dataset.sample(coordinates, indexes=bands))


async def parallel_sample_rasters(datasets, coordinates, bands=None):
    """Sample several rasters at the given coordinates in multiple threads.

    ``bands`` optionally selects, per dataset, the list of band numbers (1-based) to
    sample; each dataset's selected bands are sampled together in one call. When
    ``bands`` is None every band of each dataset is sampled.
    """
    loop = asyncio.get_running_loop()
    if bands is None:
        bands = [None] * len(datasets)
    return await asyncio.gather(
        *[
            loop.run_in_executor(None, _sample_raster, dataset, coordinates, band)
            for dataset, band in zip(datasets, bands)
        ]
    )
