import tempfile

from fastapi import HTTPException
from starlette.responses import FileResponse, Response
from starlette.types import Receive, Scope, Send

from cpra.mp.data.analysis.paired import PairedAnalysis
from cpra.mp.data.analysis.raster import RasterGridAnalysis
from cpra.mp.data.analysis.types import CompressionAlgorithm
from cpra.mp.data.api.models import Compression

# Either a raster-grid analysis or a paired (notebook-shape) plan -- both expose the same
# duck-typed `write_raster`/`get_tile`/`variable` surface these responses need (see
# `PairedAnalysis`'s class docstring for why it's a sibling of `BaseAnalysis`, not a
# subclass, despite implementing the same methods).
RasterLike = RasterGridAnalysis | PairedAnalysis

COMPRESSION_LEVEL_OPTIONS = {
    CompressionAlgorithm.deflate: "zlevel",
    CompressionAlgorithm.lzma: "lzma_preset",
    CompressionAlgorithm.zstd: "zstd_level",
}


class RasterioStreamingResponse(FileResponse):
    background = None
    chunk_size = 64 * 1024
    mime_types = {"tif": "image/tiff; application=geotiff"}

    def __init__(
        self, dataset: RasterLike, format: str, compression: Compression, **kwargs
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
            level_option = COMPRESSION_LEVEL_OPTIONS.get(self.compression.compression_algorithm)
            if level_option is not None:
                profile_options[level_option] = self.compression.compression_level

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
        dataset: RasterLike,
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
