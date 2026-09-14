"""Analysis-layer helpers (the former api/utils.py, HTTP-free half). Nothing here may import FastAPI."""
import asyncio
from enum import Enum
from functools import wraps
from time import time

import rasterio as rio
from cachetools import TTLCache, cached
from rasterio.enums import Resampling

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
            # Local import: avoids a cpra.mp.data.analysis.base <-> .utils import cycle
            # (base.py imports get_lazy_frame/unpack_values/timing from this module at
            # module scope; AnalysisError itself is defined in base.py per that module's
            # docstring).
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
    memory-light). ``Dataset._windowed_read_is_lighter``'s routing already keys on the
    windowed path's buffer *size* (bands x block height x width) rather than a time
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

    Deliberately not applied to tile requests -- those read a decimated 256x256 window
    through existing overviews and measured no benefit from threading.
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

    ``Dataset._apply_transform_params_raster`` used to union a whole block row per
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


