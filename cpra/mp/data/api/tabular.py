import tempfile
from threading import Thread

import anyio
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from cpra.mp.data.analysis.types import CompressionAlgorithm, DataFormat
from cpra.mp.data.api.models import Compression
from cpra.mp.data.api.sources import LazyFrameSource


def stream_or_write_to_file(source, out_path, format, compression, collect_in_memory):
    stream_kwargs = {}
    if format == DataFormat.parquet:
        uncompressed = compression.compression_algorithm == CompressionAlgorithm.none
        stream_kwargs = {
            "compression": "uncompressed" if uncompressed else compression.compression_algorithm.value,
            "compression_level": None if uncompressed else compression.compression_level,
        }

    # Whatever the source itself needs for this format, on top of the request's own
    # arguments -- a variable's `float_precision`, for one. See LazyFrameSource.
    stream_kwargs |= source.tabular_kwargs(format)

    if collect_in_memory:
        source.write_tabular(out_path, format, **stream_kwargs)
    else:
        source.stream_tabular(out_path, format, **stream_kwargs)


class StreamingThread(Thread):
    def __init__(self, source, out_path, format, compression, collect_in_memory=False, **kwargs):
        kwargs.update(
            {
                "target": stream_or_write_to_file,
                "name": "stream_or_write_to_file",
                "args": [source, out_path, format, compression, collect_in_memory],
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

    def __init__(self, source: LazyFrameSource, format: str, compression: Compression, collect_in_memory=False) -> None:
        self.source = source
        self.format = format
        self.compression = compression
        self.media_type = self.mime_types[format]
        self.collect_in_memory = collect_in_memory
        self.init_headers()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["method"].upper() == "HEAD":
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        else:
            with tempfile.NamedTemporaryFile() as fp:
                write_thread = StreamingThread(
                    self.source, fp.name, self.format, self.compression, self.collect_in_memory
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
                                        "status": getattr(err, "status_code", 500) if err else 200,
                                        "headers": self.raw_headers,
                                    }
                                )
                                if err:
                                    chunk = getattr(err, "detail", str(err)).encode("utf-8")
                                    more_body = False
                            await send(
                                {
                                    "type": "http.response.body",
                                    "body": chunk,
                                    "more_body": more_body,
                                }
                            )
