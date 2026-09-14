"""Tests for RasterioStreamingResponse's compression selection.

``write_raster`` is stubbed to return False, so each call raises the expected 404
before reaching Starlette's file-send machinery (which needs a real ASGI scope) --
these tests only need to see which creation options __call__ chose.
"""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from cpra.mp.data.analysis.types import CompressionAlgorithm
from cpra.mp.data.api.models import Compression
from cpra.mp.data.api.raster import RasterioStreamingResponse


def _response(max_z_error, compression=None):
    calls = []

    async def write_raster(path, **kwargs):
        calls.append(kwargs)
        return False

    dataset = SimpleNamespace(
        variable=SimpleNamespace(max_z_error=max_z_error),
        write_raster=write_raster,
    )
    response = RasterioStreamingResponse(
        dataset=dataset,
        format="tif",
        compression=compression or Compression(),
    )
    with pytest.raises(HTTPException):
        asyncio.run(response(None, None, None))
    return calls[0]


def test_lerc_variable_overrides_client_compression():
    """A variable with max_z_error always gets a LERC_ZSTD response, regardless of request."""

    kwargs = _response(
        max_z_error=0.0005,
        compression=Compression(compression_algorithm=CompressionAlgorithm.lzma, compression_level=9),
    )
    assert kwargs == {"compress": "LERC_ZSTD", "max_z_error": 0.0005, "zstd_level": 9}


def test_non_lerc_variable_uses_requested_compression():
    """A variable without max_z_error keeps the ordinary client-driven compression choice."""

    kwargs = _response(
        max_z_error=None,
        compression=Compression(compression_algorithm=CompressionAlgorithm.deflate, compression_level=9),
    )
    assert kwargs == {"compress": "DEFLATE", "zlevel": 9}


def test_default_compression_is_zstd_without_max_z_error():
    """The Compression model's own default (zstd, level 6) is untouched when not overridden."""

    kwargs = _response(max_z_error=None)
    assert kwargs == {"compress": "ZSTD", "zstd_level": 6}
