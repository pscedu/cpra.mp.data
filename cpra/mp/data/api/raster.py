import polars as pl
import rasterio as rio
import tempfile
from contextlib import ExitStack
from starlette.responses import FileResponse
from starlette.types import Receive, Scope, Send
from time import time


class RasterioStreamingResponse(FileResponse):
    background = None
    chunk_size = 64 * 1024
    mime_types = {"tif": "image/tiff; application=geotiff"}

    def __init__(self, lf: pl.LazyFrame, format: str = "tif", **kwargs) -> None:
        self.lf = lf
        self.format = format
        self.media_type = self.mime_types[format]
        self.init_headers()
        self.profile_options = {"compression_level": 6}
        self.profile_options.update(kwargs)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        start = time()
        with tempfile.NamedTemporaryFile() as fp:
            write_tiff(self.lf, fp.name, **self.profile_options)
            self.path = fp.name
            self.stat_result = None
            self.status_code = 200
            await super().__call__(scope, receive, send)
        print(f"Download took {time() - start} seconds")


def write_tiff(lf, out_path, **kwargs):
    """Write a GeoTIFF to a temporary file using windowed reading
    and writing to conserve memory."""

    start = time()
    df = lf.collect()
    with ExitStack() as stack:
        inputs = [stack.enter_context(rio.open(path)) for path in df["path"]]
        profile = inputs[0].profile.copy()
        profile.update(kwargs)
        output = stack.enter_context(rio.open(out_path, "w", **profile))
        for ji, window in inputs[0].block_windows(1):
            data = [input.read(window=window) for input in inputs]
            diff = data[1] - data[0]
            output.write(diff, window=window)
    print(f"Save took {time() - start} seconds")
