"""Tests for the streaming response classes.

These wrap a background writer thread around a temp file and stream it out, which means
the HTTP status has to be decided from whatever the thread has done by the time the first
chunk is ready.
"""

import polars as pl
import pytest
import rasterio as rio
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from cpra.mp.data.analysis.types import CompressionAlgorithm
from cpra.mp.data.api.models import Compression
from cpra.mp.data.api.raster import COMPRESSION_LEVEL_OPTIONS, RasterioStreamingResponse
from cpra.mp.data.api.sources import LazyFrameSource
from cpra.mp.data.api.tabular import PolarsStreamingResponse, StreamingThread
from cpra.mp.data.api.vector import VectorTileResponse
from tests.factories import RASTER_KWARGS, write_raster_dataset

DIMENSIONS = {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}


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


class TestRasterioStreamingResponse:
    def test_compression_level_option_mapping(self):
        assert COMPRESSION_LEVEL_OPTIONS[CompressionAlgorithm.deflate] == "zlevel"
        assert COMPRESSION_LEVEL_OPTIONS[CompressionAlgorithm.lzma] == "lzma_preset"
        assert COMPRESSION_LEVEL_OPTIONS[CompressionAlgorithm.zstd] == "zstd_level"

    def test_algorithms_without_a_level_option(self):
        assert CompressionAlgorithm.lzw not in COMPRESSION_LEVEL_OPTIONS
        assert CompressionAlgorithm.none not in COMPRESSION_LEVEL_OPTIONS

    async def test_streams_a_geotiff(self, dataset_factory, base_path, tmp_path):
        import io

        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda y: y - 2000)
        dataset = dataset_factory(DIMENSIONS)
        await dataset.apply_params()

        client = serve(lambda: RasterioStreamingResponse(dataset, "tif", Compression()))
        response = client.get("/test")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff; application=geotiff"
        with rio.open(io.BytesIO(response.content)) as src:
            assert src.read(1)[0, 0] == 30

    async def test_compression_reaches_the_output_profile(
        self, dataset_factory, base_path, tmp_path
    ):
        import io

        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda y: y - 2000)
        dataset = dataset_factory(DIMENSIONS)
        await dataset.apply_params()

        compression = Compression(
            compression_algorithm=CompressionAlgorithm.deflate, compression_level=9
        )
        client = serve(lambda: RasterioStreamingResponse(dataset, "tif", compression))
        response = client.get("/test")

        with rio.open(io.BytesIO(response.content)) as src:
            assert src.profile["compress"].lower() == "deflate"

    async def test_missing_source_data_raises_404(self, dataset_factory, base_path):
        """An HTTPException raised inside __call__ is still handled by FastAPI."""
        dataset = dataset_factory(DIMENSIONS)
        await dataset.apply_params()

        client = serve(lambda: RasterioStreamingResponse(dataset, "tif", Compression()))
        response = client.get("/test")
        assert response.status_code == 404
        assert response.json()["detail"] == "Source data do not exist"


class TestVectorTileResponse:
    async def test_missing_tile_raises(self, dataset_factory, base_path, tiles_path):
        from cpra.mp.data.tabular import write_parquet
        from tests.factories import ANNUAL_KWARGS, annual_frame

        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        dimensions = {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}
        dataset = dataset_factory(dimensions)
        await dataset.apply_params()

        client = serve(lambda: VectorTileResponse(dataset, 1, 2, 3))
        response = client.get("/test")
        assert response.status_code == 404
        assert response.json()["detail"] == "Tile does not exist"

    async def test_serves_a_tile(self, dataset_factory, base_path, tiles_path):
        import mercantile
        from mapbox_vector_tile.Mapbox.vector_tile_pb2 import tile as VectorTile

        from cpra.mp.data.tabular import write_parquet
        from tests.factories import ANNUAL_KWARGS, annual_frame, write_pmtiles

        tile = mercantile.Tile(x=1, y=2, z=3)
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        write_pmtiles(tiles_path, "ecoregion_v001", tile, [1, 2])

        dimensions = {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}
        dataset = dataset_factory(dimensions)
        await dataset.apply_params()

        client = serve(lambda: VectorTileResponse(dataset, tile.x, tile.y, tile.z))
        response = client.get("/test")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"
        parsed = VectorTile()
        parsed.ParseFromString(response.content)
        assert len(parsed.layers[0].values) == 2
