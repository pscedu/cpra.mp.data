import tempfile
from threading import Thread

import anyio
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from cpra.mp.data.analysis.request.types import CompressionAlgorithm, DataFormat
from cpra.mp.data.api.models import Compression
from cpra.mp.data.api.sources import LazyFrameSource


# Poll pacing for the temp-file read loop in `PolarsStreamingResponse.__call__` below, and
# for `api.pipeline.PipelineTabularResponse`'s copy of that loop. Only `sleep_between_polls`
# reads these, and both loops reach them only through it -- see that function for what the
# two bounds buy and why neither loop names them directly.
POLL_INTERVAL_MIN = 0.0005
POLL_INTERVAL_MAX = 0.01


async def sleep_between_polls(interval: float | None) -> float:
    """Wait out one empty read of the temp file, and return the interval the next one should
    use -- double this one, capped at `POLL_INTERVAL_MAX`. `None` means "first poll of this
    stall" and waits `POLL_INTERVAL_MIN`, which is how a caller spells the reset without
    naming either bound itself.

    The loops that call this read a temp file a background thread is still writing, and an
    empty read while that thread is alive is the "nothing yet" signal. Unpaced, that is a
    spin, and each iteration of it is a real thread-pool dispatch: a request whose writer
    took 3.5 seconds measured ~70,000 reads of an empty file, at ~20,000/s, to deliver 5 KB.
    The cost is partly latency (the spin contends with the writer for the GIL and for the
    pool) but mostly concurrency -- the iteration count scales with request duration, so one
    pool slot stays occupied reading nothing for the whole of it, which is worst for exactly
    the slow requests where a wasted slot matters most.

    Backoff rather than a fixed interval, because the two ends of the range want opposite
    things. A response whose writer finishes in ~2 ms does 3 reads in total and never waits
    more than `POLL_INTERVAL_MIN`, so it is unaffected -- where a fixed interval big enough
    to matter to a slow request is paid in full by a fast one (a flat 5 ms measured 8-9 ms
    on a 4 ms response, roughly doubling it, for no benefit the backoff does not also get).

    The cap is the tail-latency knob, and 10 ms is where its two costs cross. It bounds how
    long a finished write can go unnoticed, since the loop learns the writer is done only by
    polling: on the same 3.5-second request, 10 ms measured 340 reads and ~10 ms of added
    wall clock, against 50 ms measured 77 reads and ~50 ms added. Both are a 200x-900x cut
    from 70,000 and both leave the slot essentially idle, so the extra reduction buys
    nothing that the added tail does not cost back on shorter requests.

    Callers reset to `None` whenever a read does return data, so a stall partway through a
    long response backs off from scratch rather than inheriting whatever interval an
    earlier stall ended at.
    """
    interval = POLL_INTERVAL_MIN if interval is None else interval
    await anyio.sleep(interval)
    return min(interval * 2, POLL_INTERVAL_MAX)


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
                    poll_interval = None
                    while more_body:
                        # Check whether the thread is alive *before* attempting to read data
                        # to avoid a race condition where the thread terminates immediately
                        # after the failed read attempt but before calling .is_alive().
                        is_alive = write_thread.is_alive()
                        chunk = await file.read(self.chunk_size)
                        chunk_has_data = len(chunk) > 0
                        more_body = chunk_has_data or is_alive
                        # Nothing written yet and the writer still going: the one case with
                        # nothing to send, and so the one the loop would otherwise spin in.
                        # This `continue` is the exact complement of the send condition it
                        # replaces (`chunk_has_data or not more_body`), so it changes when
                        # the loop waits and nothing about what it sends.
                        if not chunk_has_data and more_body:
                            poll_interval = await sleep_between_polls(poll_interval)
                            continue
                        poll_interval = None
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
