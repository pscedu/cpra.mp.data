"""Tests for the streaming response classes.

These wrap a background writer thread around a temp file and stream it out, which means
the HTTP status has to be decided from whatever the thread has done by the time the first
chunk is ready.
"""

import time

import anyio
import polars as pl
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from cpra.mp.data.analysis.request.types import CompressionAlgorithm, DataFormat
from cpra.mp.data.api import tabular
from cpra.mp.data.api.models import Compression
from cpra.mp.data.api.pipeline import (
    _TABULAR_MIME_TYPES,
    COMPRESSION_LEVEL_OPTIONS,
    PipelineTabularResponse,
)
from cpra.mp.data.api.sources import LazyFrameSource
from cpra.mp.data.api.tabular import PolarsStreamingResponse, StreamingThread


def serve(response_factory):
    """Mount a single route returning the given response and hand back a client."""
    app = FastAPI()

    @app.get("/test")
    async def route():
        return response_factory()

    return TestClient(app, raise_server_exceptions=False)


class TestPolarsStreamingResponse:
    def _client(self, source, format="csv", compression=None, collect_in_memory=False):
        compression = compression or Compression()
        return serve(
            lambda: PolarsStreamingResponse(source, format, compression, collect_in_memory)
        )

    def test_streams_csv(self):
        source = LazyFrameSource(pl.LazyFrame({"a": [1, 2]}))
        response = self._client(source).get("/test")
        assert response.status_code == 200
        assert "a\n1\n2" in response.text

    def test_collect_in_memory_path(self):
        source = LazyFrameSource(pl.LazyFrame({"a": [1, 2]}))
        response = self._client(source, collect_in_memory=True).get("/test")
        assert response.status_code == 200
        assert "a\n1\n2" in response.text

    @pytest.mark.parametrize(
        ("format", "media_type"),
        [
            ("csv", "text/csv"),
            ("ndjson", "application/x-ndjson"),
            ("parquet", "application/vnd.apache.parquet"),
        ],
    )
    def test_media_types(self, format, media_type):
        source = LazyFrameSource(pl.LazyFrame({"a": [1]}))
        response = self._client(source, format=format).get("/test")
        assert response.headers["content-type"].startswith(media_type)

    def test_both_mime_maps_carry_exactly_the_tabular_data_formats(self):
        """The two maps are deliberately separate copies (see `_TABULAR_MIME_TYPES`'s own
        comment for why this module's response class does not import that one's), so they
        can drift from each other and from the format vocabulary alike -- and did: both
        carried an `arrow` entry that no route could reach, since every `{format}` path
        segment binds straight to `DataFormat` and that enum has no `arrow` member.

        `tif` is absent from both because a GeoTIFF is not streamed by either class --
        `api.pipeline.PipelineRasterResponse` serves it from a temp file instead.
        """
        tabular_formats = {format.value for format in DataFormat if format is not DataFormat.tif}
        assert set(PolarsStreamingResponse.mime_types) == tabular_formats
        assert set(_TABULAR_MIME_TYPES) == tabular_formats
        assert PolarsStreamingResponse.mime_types == _TABULAR_MIME_TYPES

    def test_parquet_body_is_readable(self, tmp_path):
        import io

        source = LazyFrameSource(pl.LazyFrame({"a": [1, 2]}))
        response = self._client(source, format="parquet").get("/test")
        assert pl.read_parquet(io.BytesIO(response.content))["a"].to_list() == [1, 2]

    def test_uncompressed_parquet(self):
        import io

        source = LazyFrameSource(pl.LazyFrame({"a": [1]}))
        compression = Compression(compression_algorithm=CompressionAlgorithm.none)
        response = self._client(source, format="parquet", compression=compression).get("/test")
        assert pl.read_parquet(io.BytesIO(response.content))["a"].to_list() == [1]

    def test_head_request_returns_no_body(self):
        source = LazyFrameSource(pl.LazyFrame({"a": [1, 2]}))
        response = self._client(source).head("/test")
        assert response.content == b""

    def test_writer_exception_sets_the_http_status(self):
        """A failure inside the writer thread must surface as its status, not a 200."""
        source = LazyFrameSource(pl.scan_parquet("/nonexistent/path.parquet"))
        response = self._client(source).get("/test")
        assert response.status_code == 404
        assert response.text == "Dataset does not exist"

    def test_writer_exception_without_a_status_becomes_500(self):
        class Boom(LazyFrameSource):
            def stream_tabular(self, *args, **kwargs):
                raise RuntimeError("boom")

        response = self._client(Boom(pl.LazyFrame({"a": [1]}))).get("/test")
        assert response.status_code == 500
        assert response.text == "boom"


class SlowSource(LazyFrameSource):
    """A writer that produces nothing for `delay` seconds and then writes normally -- the
    shape of a slow read, and the only shape in which the response's read loop has anything
    to wait on. A fast source writes before the loop's first read and never waits at all.
    """

    def __init__(self, lf, delay):
        super().__init__(lf)
        self.delay = delay

    def stream_tabular(self, out_path, format, *args, **kwargs):
        time.sleep(self.delay)
        super().stream_tabular(out_path, format, *args, **kwargs)


def count_reads(monkeypatch):
    """Count every `file.read` the response's poll loop performs. Returns a one-key dict so
    the caller can read the running total.
    """
    counted = {"n": 0}
    original = anyio.AsyncFile.read

    async def counting_read(self, *args, **kwargs):
        counted["n"] += 1
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(anyio.AsyncFile, "read", counting_read)
    return counted


class TestPollPacing:
    """The read loop polls a temp file a background thread is still writing, and an empty
    read while that thread is alive means "nothing yet". Unpaced, that is a spin -- ~70,000
    thread-pool-dispatched reads of an empty file measured on one 3.5-second request -- and
    `api.tabular.sleep_between_polls` is what bounds it.
    """

    DELAY = 0.25

    def test_the_interval_doubles_and_then_holds_at_the_cap(self):
        async def intervals():
            # `None` is how a caller spells "first poll of this stall" -- neither response
            # class names a bound itself, so this is the only entry point to the series.
            out, interval = [], None
            for _ in range(8):
                interval = await tabular.sleep_between_polls(interval)
                out.append(interval)
            return out

        got = anyio.run(intervals)
        assert got[0] == tabular.POLL_INTERVAL_MIN * 2
        assert got[1] == tabular.POLL_INTERVAL_MIN * 4
        # It reaches the cap and stays there rather than growing without bound, which is
        # what keeps a finished write from going unnoticed for an arbitrarily long time.
        assert got[-1] == tabular.POLL_INTERVAL_MAX
        assert max(got) == tabular.POLL_INTERVAL_MAX

    def test_the_cap_is_short_enough_to_not_dominate_a_quick_response(self):
        # The pacing's whole cost is that a finished write is noticed up to one interval
        # late, so the cap IS the worst-case added latency. Pinned because raising it is
        # the tempting way to cut the poll count further, and it is paid by every request.
        assert tabular.POLL_INTERVAL_MAX <= 0.01
        assert tabular.POLL_INTERVAL_MIN < tabular.POLL_INTERVAL_MAX

    def _reads_for(self, response_factory, *, paced):
        # Its own patch context per call, not the `monkeypatch` fixture: the two calls run
        # inside one test, and a fixture would not undo the control's zeroed bounds until
        # the test ended -- which would silently make the paced run unpaced too.
        with pytest.MonkeyPatch.context() as patch:
            if not paced:
                # The same loop with the pacing neutralized -- what it did before this fix.
                patch.setattr(tabular, "POLL_INTERVAL_MIN", 0)
                patch.setattr(tabular, "POLL_INTERVAL_MAX", 0)
            counted = count_reads(patch)
            response = serve(response_factory).get("/test")
        assert response.status_code == 200
        assert "a\n1\n2" in response.text
        return counted["n"]

    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(
                lambda source: PolarsStreamingResponse(source, "csv", Compression()),
                id="PolarsStreamingResponse",
            ),
            pytest.param(
                lambda source: PipelineTabularResponse(source, "csv", Compression()),
                id="PipelineTabularResponse",
            ),
        ],
    )
    def test_a_slow_writer_does_not_spin_the_read_loop(self, factory):
        """Both classes carry a copy of the loop, so both have to be paced -- and the second
        one's docstring says it shares the pacing rather than copying it, which only means
        anything if it is checked here too.

        Measured against the unpaced loop rather than an absolute count: how many reads a
        spin fits into a quarter second is a property of the machine, but the ratio between
        spinning and not is the behavior being fixed. `PipelineTabularResponse` is built on
        a duck-typed source here, the same way `stream_or_write_to_file` consumes one.
        """
        source = SlowSource(pl.LazyFrame({"a": [1, 2]}), self.DELAY)
        spun = self._reads_for(lambda: factory(source), paced=False)
        paced = self._reads_for(lambda: factory(source), paced=True)

        # A quarter second at a 10 ms cap is tens of polls (measured: 30 for both classes);
        # unpaced it is thousands (measured: ~4,000, a ratio of ~135x).
        assert paced < 100
        assert spun > 20 * paced

    def test_a_fast_response_is_not_slowed_by_the_pacing(self):
        # The backoff starts at `POLL_INTERVAL_MIN` precisely so this case pays nothing
        # noticeable: a writer that finishes in milliseconds does a handful of reads, none
        # of which waits longer than half a millisecond.
        source = LazyFrameSource(pl.LazyFrame({"a": [1, 2]}))
        started = time.perf_counter()
        response = serve(lambda: PolarsStreamingResponse(source, "csv", Compression())).get("/test")
        elapsed = time.perf_counter() - started

        assert response.status_code == 200
        assert "a\n1\n2" in response.text
        assert elapsed < 20 * tabular.POLL_INTERVAL_MAX

    def test_a_writer_that_fails_slowly_still_reports_its_status(self):
        # The paced branch is the one with nothing to send, so it must not swallow the
        # not-alive-and-empty exit that carries a failed writer's status out.
        class SlowBoom(SlowSource):
            def stream_tabular(self, *args, **kwargs):
                time.sleep(self.delay)
                raise RuntimeError("boom")

        source = SlowBoom(pl.LazyFrame({"a": [1]}), self.DELAY)
        response = serve(lambda: PolarsStreamingResponse(source, "csv", Compression())).get("/test")
        assert response.status_code == 500
        assert response.text == "boom"


class TestStreamingThread:
    def test_captures_an_exception_instead_of_raising(self):
        source = LazyFrameSource(pl.scan_parquet("/nonexistent/path.parquet"))
        thread = StreamingThread(source, "/tmp/unused", "csv", Compression())
        thread.start()
        thread.join()
        assert isinstance(thread.exception, HTTPException)

    def test_no_exception_on_success(self, tmp_path):
        source = LazyFrameSource(pl.LazyFrame({"a": [1]}))
        out = tmp_path / "out.csv"
        thread = StreamingThread(source, str(out), "csv", Compression())
        thread.start()
        thread.join()
        assert thread.exception is None
        assert out.read_text().startswith("a")


class TestCompressionLevelOptions:
    """`api.pipeline.COMPRESSION_LEVEL_OPTIONS` -- the rasterio profile key each
    compression algorithm spells its level with. Only the three that HAVE a level appear;
    `.get()` returning nothing is what "this algorithm takes no level" means.
    """

    def test_compression_level_option_mapping(self):
        assert COMPRESSION_LEVEL_OPTIONS[CompressionAlgorithm.deflate] == "zlevel"
        assert COMPRESSION_LEVEL_OPTIONS[CompressionAlgorithm.lzma] == "lzma_preset"
        assert COMPRESSION_LEVEL_OPTIONS[CompressionAlgorithm.zstd] == "zstd_level"

    def test_algorithms_without_a_level_option(self):
        assert CompressionAlgorithm.lzw not in COMPRESSION_LEVEL_OPTIONS
        assert CompressionAlgorithm.none not in COMPRESSION_LEVEL_OPTIONS
