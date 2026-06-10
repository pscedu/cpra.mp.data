import tempfile
from threading import Thread

import anyio
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from cpra.mp.data.api.models import Compression
from cpra.mp.data.api.sources import DuckDBSource, LazyFrameSource
from cpra.mp.data.api.types import CompressionAlgorithm, DataFormat


def _stream_to_file(source, out_path, format, compression):
    """Write source data to out_path in the requested format."""
    kwargs = {}
    if format == DataFormat.parquet:
        uncompressed = compression.compression_algorithm == CompressionAlgorithm.none
        kwargs = {
            "compression": "uncompressed" if uncompressed else compression.compression_algorithm.value,
            "compression_level": None if uncompressed else compression.compression_level,
        }

    if isinstance(source, DuckDBSource):
        source.stream_tabular(out_path, format, **kwargs)
    else:
        # Polars LazyFrameSource path (used for smaller / cached datasets)
        source.stream_tabular(out_path, format, **kwargs)


class StreamingThread(Thread):
    def __init__(self, source, out_path, format, compression, **kwargs):
        kwargs.update({
            "target": _stream_to_file,
            "name": "stream_to_file",
            "args": [source, out_path, format, compression],
        })
        self.exception = None
        super().__init__(**kwargs)

    def run(self):
        try:
            super().run()
        except Exception as err:
            self.exception = err


MIME_TYPES = {
    "arrow": "application/vnd.apache.arrow.file",
    "csv": "text/csv",
    "ndjson": "application/x-ndjson",
    "parquet": "application/vnd.apache.parquet",
}

CHUNK_SIZE = 64 * 1024


class PolarsStreamingResponse(Response):
    """Streaming response for both Polars and DuckDB sources."""

    background = None
    chunk_size = CHUNK_SIZE

    def __init__(
        self,
        source: LazyFrameSource | DuckDBSource,
        format: str,
        compression: Compression,
    ) -> None:
        self.source = source
        self.format = format
        self.compression = compression
        self.media_type = MIME_TYPES[format]
        self.init_headers()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["method"].upper() == "HEAD":
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return

        with tempfile.NamedTemporaryFile() as fp:
            write_thread = StreamingThread(self.source, fp.name, self.format, self.compression)
            write_thread.start()
            sent_headers = False
            async with await anyio.open_file(fp.name, mode="rb") as file:
                more_body = True
                while more_body:
                    is_alive = write_thread.is_alive()
                    chunk = await file.read(self.chunk_size)
                    chunk_has_data = len(chunk) > 0
                    more_body = chunk_has_data or is_alive
                    if chunk_has_data or not more_body:
                        if not sent_headers:
                            sent_headers = True
                            err = write_thread.exception
                            await send({
                                "type": "http.response.start",
                                "status": getattr(err, "status_code", 500) if err else 200,
                                "headers": self.raw_headers,
                            })
                            if err:
                                chunk = getattr(err, "detail", str(err)).encode("utf-8")
                                more_body = False
                        await send({
                            "type": "http.response.body",
                            "body": chunk,
                            "more_body": more_body,
                        })
