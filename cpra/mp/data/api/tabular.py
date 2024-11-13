import anyio
import tempfile
from cpra.mp.data.api.models import Compression
from cpra.mp.data.api.sources import LazyFrameSource
from cpra.mp.data.api.types import CompressionAlgorithm, DataFormat
from starlette.responses import Response
from starlette.types import Receive, Scope, Send
from threading import Thread


def stream_to_file(source, out_path, format, compression):
    stream_kwargs = {}
    if format == DataFormat.parquet:
        uncompressed = compression.compression_algorithm == CompressionAlgorithm.none
        stream_kwargs = {
            "compression": "uncompressed"
            if uncompressed
            else compression.compression_algorithm.value,
            "compression_level": None
            if uncompressed
            else compression.compression_level,
        }
    source.stream_tabular(out_path, format, **stream_kwargs)


class StreamingThread(Thread):
    def __init__(self, source, out_path, format, compression, **kwargs):
        kwargs.update(
            {
                "target": stream_to_file,
                "name": "stream_to_file",
                "args": [source, out_path, format, compression],
            }
        )
        self.exception = None
        super().__init__(**kwargs)

    def run(self):
        try:
            super().run()
        except Exception as err:
            self.exception = err


class PolarsStreamingResponse(Response):
    background = None
    chunk_size = 64 * 1024
    mime_types = {
        "arrow": "application/vnd.apache.arrow.file",
        "csv": "text/csv",
        "ndjson": "application/x-ndjson",
        "parquet": "application/vnd.apache.parquet",
    }

    def __init__(
        self, source: LazyFrameSource, format: str, compression: Compression
    ) -> None:
        self.source = source
        self.format = format
        self.compression = compression
        self.media_type = self.mime_types[format]
        self.init_headers()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["method"].upper() == "HEAD":
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        else:
            with tempfile.NamedTemporaryFile() as fp:
                write_thread = StreamingThread(
                    self.source, fp.name, self.format, self.compression
                )
                write_thread.start()
                sent_headers = False
                async with await anyio.open_file(fp.name, mode="rb") as file:
                    more_body = True
                    while more_body:
                        # Check whether the thread is alive *before* attempting to read data
                        # to avoid a race condition where the thread terminates immediately
                        # after the failed read attempt but before calling .is_alive().
                        is_alive = write_thread.is_alive()
                        chunk = await file.read(self.chunk_size)
                        chunk_has_data = len(chunk) > 0
                        more_body = chunk_has_data or is_alive
                        if chunk_has_data or not more_body:
                            if not sent_headers:
                                sent_headers = True
                                err = write_thread.exception
                                await send(
                                    {
                                        "type": "http.response.start",
                                        "status": getattr(err, "status_code", 500)
                                        if err
                                        else 200,
                                        "headers": self.raw_headers,
                                    }
                                )
                                if err:
                                    chunk = getattr(err, "detail", str(err)).encode(
                                        "utf-8"
                                    )
                                    more_body = False
                            await send(
                                {
                                    "type": "http.response.body",
                                    "body": chunk,
                                    "more_body": more_body,
                                }
                            )
