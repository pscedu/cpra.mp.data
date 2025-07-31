import tempfile

from fastapi import HTTPException
from starlette.responses import FileResponse, Response
from starlette.types import Receive, Scope, Send

from cpra.mp.data.api.dataset import Dataset
from cpra.mp.data.api.models import Compression
from cpra.mp.data.api.types import CompressionAlgorithm

COMPRESSION_LEVEL_ALGORITHMS = [
    CompressionAlgorithm.deflate,
    CompressionAlgorithm.lzma,
    CompressionAlgorithm.zstd,
]


class RasterioStreamingResponse(FileResponse):
    background = None
    chunk_size = 64 * 1024
    mime_types = {"tif": "image/tiff; application=geotiff"}

    def __init__(
        self, dataset: Dataset, format: str, compression: Compression, **kwargs
    ) -> None:
        self.dataset = dataset
        self.format = format
        self.compression = compression
        self.media_type = self.mime_types[format]
        self.init_headers()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        with tempfile.NamedTemporaryFile() as fp:
            profile_options = {
                "compress": self.compression.compression_algorithm.value.upper()
            }
            if self.compression.compression_algorithm in COMPRESSION_LEVEL_ALGORITHMS:
                profile_options["level"] = self.compression.compression_level

            if await self.dataset.write_raster(fp.name, **profile_options):
                self.path = fp.name
                self.stat_result = None
                self.status_code = 200
                await super().__call__(scope, receive, send)

            else:
                raise HTTPException(status_code=404, detail="Source data do not exist")


class RasterioTileResponse(Response):
    background = None
    mime_types = {"jpg": "image/jpeg", "png": "image/png"}
    status_code = 200

    def __init__(
        self,
        dataset: Dataset,
        x: int,
        y: int,
        z: int,
        format: str = "png",
        **kwargs,
    ) -> None:
        self.dataset = dataset
        self.format = format
        self.tile = (x, y, z) if x is not None else None
        self.media_type = self.mime_types[format]
        self.init_headers()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.body = await self.dataset.get_tile(*self.tile, self.format)

        # Handle tile requests that are outside the dataset bounds.
        if self.body is None:
            raise HTTPException(status_code=404, detail="Tile does not exist")

        await super().__call__(scope, receive, send)
