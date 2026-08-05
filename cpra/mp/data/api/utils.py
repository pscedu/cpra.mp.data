import asyncio
from enum import Enum
from functools import wraps
from time import time

import rasterio as rio
from cachetools import TTLCache, cached
from fastapi import HTTPException
from rasterio.enums import Resampling

from cpra.mp.data import read_data
from cpra.mp.data.api.types import CompressionAlgorithm, DataFormat, OutputType
from cpra.mp.data.config import (
    DEBUG,
    RASTER_CACHE_MAX_MB,
    RASTER_NUM_THREADS,
    RASTER_SWATH_SIZE_MB,
)
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.visualizations import VISUALIZATIONS

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
    a multiband COG: selecting a subset of bands still decompresses each tile's full
    band payload, so a 1-band read costs nearly as much as a 52-band one. Setting it
    measured 2.2-3.7x faster across band counts with byte-identical output.

    Capping ``GDAL_CACHEMAX`` trades ~1% time for ~11% lower peak memory on the windowed
    path (measured on a 52-band aggregate), much less dramatic than on the copy path but
    worth having -- the windowed path's peak is dominated by its own NumPy buffers, not by
    GDAL's cache. Being process-global, it also bounds cache growth across concurrent
    requests rather than leaving it at GDAL's default 5% of RAM.

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


def _read_window(dataset, window, out_shape=None, resampling=Resampling.nearest, indexes=None):
    """Read a window of data from a single raster (optionally a subset of bands)."""
    return dataset.read(indexes=indexes, window=window, out_shape=out_shape, resampling=resampling)


async def parallel_read_rasters(datasets, window=None, out_shape=None, resampling=Resampling.nearest, bands=None):
    """Read several rasters in multiple threads.

    ``bands`` optionally selects, per dataset, the list of band numbers (1-based) to
    read; each dataset's selected bands are read together as one ``(n, height, width)``
    array. When ``bands`` is None every band of each dataset is read.
    """
    loop = asyncio.get_running_loop()
    if bands is None:
        bands = [None] * len(datasets)
    return await asyncio.gather(
        *[
            loop.run_in_executor(None, _read_window, dataset, window, out_shape, resampling, band)
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


def unpack_values(values_str, param_name, choices=None, separator=","):
    """Unpack values from a comma-separated list."""
    values = [val.strip() for val in values_str.split(separator)]

    # Validate choices.
    if choices:
        invalid_values = [val for val in values if val not in choices]
        if invalid_values:
            msg = f"Invalid values for {param_name}: {', '.join(invalid_values)}"
            raise HTTPException(status_code=400, detail=msg)

    return values


@cached(cache=TTLCache(maxsize=128, ttl=600))
def _read_data_cached(*args, **kwargs):
    return read_data(*args, **kwargs)


def get_lazy_frame(dataset_dimensions, filter_dimensions):
    """Get a LazyFrame for the given dimensions with error handling."""
    try:
        return _read_data_cached(
            **dict(dataset_dimensions),
            **{k: tuple(v) for (k, v) in dict(filter_dimensions).items() if isinstance(v, list)},
        )
    except ValidationError as err:
        # TODO: Also handle directory does not exist errors.
        raise HTTPException(status_code=422, detail=str(err))


def validate_compression(compression, format):
    """Validate that the compression algorithm matches the requested data format."""
    # TODO: Also validate compression level.
    if format == DataFormat.tif:
        if compression.compression_algorithm in RASTER_COMPRESSION_ALGORITHMS:
            return True
        allowed_algorithms = RASTER_COMPRESSION_ALGORITHMS
    else:
        if compression.compression_algorithm in TABULAR_COMPRESSION_ALGORITHMS:
            return True
        allowed_algorithms = TABULAR_COMPRESSION_ALGORITHMS

    algs_str = ", ".join([alg.value for alg in allowed_algorithms])
    msg = f"Compression algorithm for format {format.value} must be one of: {algs_str}"
    raise HTTPException(status_code=400, detail=msg)


def validate_format(format, output_type):
    """Validate that the specified data format matches the expected output type."""
    if output_type == OutputType.raster and format != DataFormat.tif:
        msg = "Format for raster output must be tif"
        raise HTTPException(status_code=400, detail=msg)
    elif output_type == OutputType.vector and format == DataFormat.tif:
        formats_str = ", ".join([f.value for f in DataFormat if f != DataFormat.tif])
        msg = f"Format for vector output must be one of: {formats_str}"
        raise HTTPException(status_code=400, detail=msg)


def get_visualization_class(name: str, type: str | None = None):
    """Get a visualization class by name, and optionally, by type."""

    VisualizationClass = VISUALIZATIONS.get(name, None)

    if VisualizationClass is None:
        msg = f"Unknown visualization name: {name}"
        raise HTTPException(status_code=404, detail=msg)

    if type is not None and VisualizationClass.Meta.type != type:
        msg = f"Invalid visualization type: expected {type}, got {VisualizationClass.Meta.type}"
        raise HTTPException(status_code=400, detail=msg)

    return VisualizationClass
