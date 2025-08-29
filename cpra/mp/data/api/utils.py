import asyncio
from enum import Enum
from functools import wraps
from time import time

from cachetools import TTLCache, cached
from fastapi import HTTPException
from rasterio.enums import Resampling

from cpra.mp.data import read_data
from cpra.mp.data.api.types import CompressionAlgorithm, DataFormat, OutputType
from cpra.mp.data.config import DEBUG
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


def _read_window(dataset, window, out_shape=None, resampling=Resampling.nearest):
    """Read a window of data from a single raster."""
    return dataset.read(window=window, out_shape=out_shape, resampling=resampling)


async def parallel_read_rasters(datasets, window=None, out_shape=None, resampling=Resampling.nearest):
    """Read several rasters in multiple threads."""
    loop = asyncio.get_running_loop()
    return await asyncio.gather(
        *[loop.run_in_executor(None, _read_window, dataset, window, out_shape, resampling) for dataset in datasets]
    )


def _sample_raster(dataset, coordinates):
    """Sample a single raster at the given coordinates."""
    return list(dataset.sample(coordinates))


async def parallel_sample_rasters(datasets, coordinates):
    """Sample several rasters at the given coordinates in multiple threads."""
    loop = asyncio.get_running_loop()
    return await asyncio.gather(
        *[loop.run_in_executor(None, _sample_raster, dataset, coordinates) for dataset in datasets]
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
