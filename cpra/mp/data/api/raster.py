import tempfile
from cpra.mp.data.api.dataset import Dataset
from fastapi import HTTPException
from starlette.responses import FileResponse, Response
from starlette.types import Receive, Scope, Send


class RasterioStreamingResponse(FileResponse):
    background = None
    chunk_size = 64 * 1024
    mime_types = {"tif": "image/tiff; application=geotiff"}

    def __init__(self, dataset: Dataset, format: str = "tif", **kwargs) -> None:
        self.dataset = dataset
        self.format = format
        self.media_type = self.mime_types[format]
        self.init_headers()
        self.profile_options = {"compression_level": 6}
        self.profile_options.update(kwargs)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        with tempfile.NamedTemporaryFile() as fp:
            await self.dataset.write_raster(fp.name, **self.profile_options)
            self.path = fp.name
            self.stat_result = None
            self.status_code = 200
            await super().__call__(scope, receive, send)


class RasterioTileResponse(Response):
    background = None
    mime_types = {"png": "image/png"}
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
        self.body = await self.dataset.get_tile(*self.tile)

        # Handle tile requests that are outside the dataset bounds.
        if self.body is None:
            raise HTTPException(status_code=404, detail="Tile does not exist")

        await super().__call__(scope, receive, send)
