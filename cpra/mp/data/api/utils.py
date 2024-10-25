import asyncio
from cachetools import cached, TTLCache
from cpra.mp.data import read_data
from cpra.mp.data.config import DEBUG
from cpra.mp.data.exceptions import ValidationError
from enum import Enum
from fastapi import HTTPException
from functools import wraps
from rasterio.enums import Resampling
from time import time


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
            print(
                "func:%r args:[%r, %r] took: %2.4f sec"
                % (func.__name__, args, kwargs, time() - start)
            )
        return result

    return wrapped


def timing(func):
    """Time a synchronous function call."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        if DEBUG:
            print(
                "func:%r args:[%r, %r] took: %2.4f sec"
                % (func.__name__, args, kwargs, time() - start)
            )
        return result

    return wrapped


def _read_window(dataset, window, out_shape=None, resampling=Resampling.nearest):
    """Read a window of data from a single raster."""
    return dataset.read(window=window, out_shape=out_shape, resampling=resampling)


async def parallel_read_rasters(
    datasets, window=None, out_shape=None, resampling=Resampling.nearest
):
    """Read several rasters in multiple threads."""
    loop = asyncio.get_running_loop()
    return await asyncio.gather(
        *[
            loop.run_in_executor(
                None, _read_window, dataset, window, out_shape, resampling
            )
            for dataset in datasets
        ]
    )


def _sample_raster(dataset, coordinates):
    """Sample a single raster at the given coordinates."""
    return list(dataset.sample(coordinates))


async def parallel_sample_rasters(datasets, coordinates):
    """Sample several rasters at the given coordinates in multiple threads."""
    loop = asyncio.get_running_loop()
    return await asyncio.gather(
        *[
            loop.run_in_executor(None, _sample_raster, dataset, coordinates)
            for dataset in datasets
        ]
    )


def unpack_values(values_str, param_name, choices=None, separator=","):
    """Unpack values from a comma-separated list."""
    values = [val.strip() for val in values_str.split(separator)]

    # Validate choices.
    if choices:
        invalid_values = [val for val in values if val not in choices]
        if invalid_values:
            msg = f"Invalid values for {param_name}: {", ".join(invalid_values)}"
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
            **{
                k: tuple(v)
                for (k, v) in dict(filter_dimensions).items()
                if isinstance(v, list)
            },
        )
    except ValidationError as err:
        # TODO: Also handle directory does not exist errors.
        raise HTTPException(status_code=422, detail=str(err))
