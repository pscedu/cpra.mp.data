"""Tests for cpra.mp.data.api.utils / cpra.mp.data.analysis.io.raster_io and the
LazyFrameSource base class. `unpack_values` lives under `analysis/`, which may not import
FastAPI, so it raises a `request.errors.RequestError` that `api/boundary.py` maps to 400
rather than an `HTTPException` of its own.
`validate_compression`/`validate_format`/`get_visualization_class`/`LazyFrameSource` stayed
in `api/` and still raise `HTTPException`.
"""

import numpy as np
import polars as pl
import pytest
import rasterio as rio
from fastapi import HTTPException
from rasterio.crs import CRS
from rasterio.env import get_gdal_config
from rasterio.transform import from_bounds

import cpra.mp.data.analysis.io.raster_io as raster_io
from cpra.mp.data.analysis.request.types import CompressionAlgorithm, DataFormat, OutputType
from cpra.mp.data.analysis.io.raster_io import (
    enum_value,
    parallel_read_rasters,
    parallel_sample_rasters,
    raster_blocks_per_batch,
    raster_gdal_env,
    unpack_values,
)
from cpra.mp.data.analysis.request.errors import RequestError
from cpra.mp.data.api.boundary import resolve_status
from cpra.mp.data.api.models import Compression, DatasetDimensions, FilterDimensions
from cpra.mp.data.api.sources import LazyFrameSource
from cpra.mp.data.api.utils import (
    get_analysis_class,
    get_visualization_class,
    validate_compression,
    validate_format,
)
from tests.factories import RASTER_KWARGS, write_raster_dataset


class TestEnumValue:
    def test_unwraps_an_enum(self):
        assert enum_value(DataFormat.csv) == "csv"

    def test_passes_through_a_plain_value(self):
        assert enum_value("csv") == "csv"


class TestUnpackValues:
    def test_splits_on_commas(self):
        assert unpack_values("a,b,c", "param") == ["a", "b", "c"]

    def test_strips_whitespace(self):
        assert unpack_values("a , b ", "param") == ["a", "b"]

    def test_single_value(self):
        assert unpack_values("a", "param") == ["a"]

    def test_custom_separator(self):
        assert unpack_values("a|b", "param", separator="|") == ["a", "b"]

    def test_valid_choices_pass(self):
        assert unpack_values("a,b", "param", choices=["a", "b", "c"]) == ["a", "b"]

    def test_invalid_choice_raises_400(self):
        with pytest.raises(RequestError) as excinfo:
            unpack_values("a,zzz", "sort", choices=["a", "b"])
        assert resolve_status(excinfo.value) == 400
        assert "Invalid values for sort: zzz" in str(excinfo.value)

    def test_all_invalid_choices_are_listed(self):
        with pytest.raises(RequestError) as excinfo:
            unpack_values("x,y", "sort", choices=["a"])
        assert "x, y" in str(excinfo.value)


class TestValidateCompression:
    @pytest.mark.parametrize("algorithm", ["deflate", "lzw", "lzma", "none", "zstd"])
    def test_raster_algorithms_allowed_for_tif(self, algorithm):
        compression = Compression(compression_algorithm=algorithm)
        assert validate_compression(compression, DataFormat.tif) is True

    @pytest.mark.parametrize("algorithm", ["gzip", "lz4", "none", "zstd"])
    def test_tabular_algorithms_allowed_for_csv(self, algorithm):
        compression = Compression(compression_algorithm=algorithm)
        assert validate_compression(compression, DataFormat.csv) is True

    def test_tabular_algorithm_rejected_for_tif(self):
        compression = Compression(compression_algorithm=CompressionAlgorithm.gzip)
        with pytest.raises(HTTPException) as excinfo:
            validate_compression(compression, DataFormat.tif)
        assert excinfo.value.status_code == 400
        assert "must be one of" in excinfo.value.detail

    def test_raster_algorithm_rejected_for_csv(self):
        compression = Compression(compression_algorithm=CompressionAlgorithm.lzw)
        with pytest.raises(HTTPException) as excinfo:
            validate_compression(compression, DataFormat.csv)
        assert excinfo.value.status_code == 400

    def test_error_lists_the_allowed_algorithms(self):
        compression = Compression(compression_algorithm=CompressionAlgorithm.lzw)
        with pytest.raises(HTTPException) as excinfo:
            validate_compression(compression, DataFormat.parquet)
        assert "gzip" in excinfo.value.detail


class TestValidateFormat:
    def test_raster_output_requires_tif(self):
        assert validate_format(DataFormat.tif, OutputType.raster) is None

    def test_raster_output_rejects_csv(self):
        with pytest.raises(HTTPException) as excinfo:
            validate_format(DataFormat.csv, OutputType.raster)
        assert excinfo.value.status_code == 400
        assert "must be tif" in excinfo.value.detail

    @pytest.mark.parametrize("format", [DataFormat.csv, DataFormat.ndjson, DataFormat.parquet])
    def test_vector_output_accepts_tabular_formats(self, format):
        assert validate_format(format, OutputType.vector) is None

    def test_vector_output_rejects_tif(self):
        with pytest.raises(HTTPException) as excinfo:
            validate_format(DataFormat.tif, OutputType.vector)
        assert excinfo.value.status_code == 400
        assert "tif" not in excinfo.value.detail.split(":")[1]


class TestGetVisualizationClass:
    def test_returns_a_known_visualization(self):
        assert get_visualization_class("time-series-line").Meta.name == "time-series-line"

    def test_unknown_name_raises_404(self):
        with pytest.raises(HTTPException) as excinfo:
            get_visualization_class("nope")
        assert excinfo.value.status_code == 404
        assert "Unknown visualization name" in excinfo.value.detail

    def test_matching_type_is_accepted(self):
        assert get_visualization_class("time-series-line", "dataset") is not None

    def test_mismatched_type_raises_400(self):
        with pytest.raises(HTTPException) as excinfo:
            get_visualization_class("time-series-line", "custom")
        assert excinfo.value.status_code == 400
        assert "Invalid visualization type" in excinfo.value.detail

    def test_type_is_optional(self):
        assert get_visualization_class("sediment-dynamics") is not None


class _FakeAnalysisDatasetMeta:
    type = "dataset"


class _FakeAnalysisCustomMeta:
    type = "custom"


class _FakeDatasetAnalysisClass:
    Meta = _FakeAnalysisDatasetMeta


class _FakeCustomAnalysisClass:
    Meta = _FakeAnalysisCustomMeta


class TestGetAnalysisClass:
    """`get_analysis_class` -- the named-analysis counterpart to
    `get_visualization_class` above, generalizing its exact 404 (unknown name) / 400
    (wrong route kind) split rather than reinventing it. The real registry
    (`cpra.mp.data.analyses.ANALYSES`) is a concurrent change that may not exist
    yet (see `api/utils.py`'s guarded import) -- these tests monkeypatch a small fake
    registry directly onto `cpra.mp.data.api.utils.ANALYSES` (the module-global name
    `get_analysis_class` itself reads), so they pass identically whether or not the real
    package has landed.

    The fake `Meta`/analysis classes above are module-level, not nested inside this test
    class: a nested class body has no closure over its enclosing class body's namespace
    (unlike a nested function), so `class _FakeDatasetAnalysis: Meta = _DatasetMeta` would
    raise `NameError` at collection time if `_DatasetMeta` were itself just another
    sibling nested class here.
    """

    @pytest.fixture(autouse=True)
    def _fake_registry(self, monkeypatch):
        import cpra.mp.data.api.utils as api_utils

        monkeypatch.setattr(
            api_utils,
            "ANALYSES",
            {"lnd_chng": _FakeDatasetAnalysisClass, "sal_veg_investigation": _FakeCustomAnalysisClass},
        )

    def test_returns_a_known_analysis(self):
        assert get_analysis_class("lnd_chng").Meta.type == "dataset"

    def test_unknown_name_raises_404(self):
        with pytest.raises(HTTPException) as excinfo:
            get_analysis_class("nope")
        assert excinfo.value.status_code == 404
        assert "Unknown analysis name" in excinfo.value.detail

    def test_matching_type_is_accepted(self):
        assert get_analysis_class("lnd_chng", "dataset") is not None

    def test_mismatched_type_raises_400(self):
        with pytest.raises(HTTPException) as excinfo:
            get_analysis_class("lnd_chng", "custom")
        assert excinfo.value.status_code == 400
        assert "Invalid analysis type" in excinfo.value.detail

    def test_custom_type_mismatch_the_other_direction(self):
        with pytest.raises(HTTPException) as excinfo:
            get_analysis_class("sal_veg_investigation", "dataset")
        assert excinfo.value.status_code == 400

    def test_type_is_optional(self):
        assert get_analysis_class("sal_veg_investigation") is not None


class TestLazyFrameSource:
    def test_materialize_collects(self):
        source = LazyFrameSource(pl.LazyFrame({"a": [1, 2]}))
        assert source.materialize().height == 2

    def test_missing_column_becomes_400(self):
        source = LazyFrameSource(pl.LazyFrame({"a": [1]}).select("missing"))
        with pytest.raises(HTTPException) as excinfo:
            source.materialize()
        assert excinfo.value.status_code == 400

    def test_missing_file_becomes_404(self, tmp_path):
        source = LazyFrameSource(pl.scan_parquet(str(tmp_path / "absent.parquet")))
        with pytest.raises(HTTPException) as excinfo:
            source.materialize()
        assert excinfo.value.status_code == 404
        assert excinfo.value.detail == "Dataset does not exist"

    @pytest.mark.parametrize(
        ("format", "reader"),
        [("csv", pl.read_csv), ("ndjson", pl.read_ndjson), ("parquet", pl.read_parquet)],
    )
    def test_write_tabular_formats(self, tmp_path, format, reader):
        source = LazyFrameSource(pl.LazyFrame({"a": [1, 2]}))
        out = tmp_path / f"out.{format}"
        source.write_tabular(out, format)
        assert reader(out)["a"].to_list() == [1, 2]

    @pytest.mark.parametrize(
        ("format", "reader"),
        [("csv", pl.read_csv), ("ndjson", pl.read_ndjson), ("parquet", pl.read_parquet)],
    )
    def test_stream_tabular_formats(self, tmp_path, format, reader):
        source = LazyFrameSource(pl.LazyFrame({"a": [1, 2]}))
        out = tmp_path / f"out.{format}"
        source.stream_tabular(out, format)
        assert reader(out)["a"].to_list() == [1, 2]

    def test_unknown_format_raises(self, tmp_path):
        source = LazyFrameSource(pl.LazyFrame({"a": [1]}))
        with pytest.raises(KeyError):
            source.write_tabular(tmp_path / "out.xyz", "xyz")

    def test_stream_sinks_lazily_and_write_collects_first(self, tmp_path):
        """The two methods now differ only in which `io.output` writer they hand
        `materialize` to, and the format tests above cannot tell them apart -- both leave a
        correct file behind either way. Pin the difference where it is observable: what
        reaches `materialize`. Streaming passes a `sink_*` method name and the destination;
        collecting calls it with nothing, taking its `collect` default.
        """
        calls = []

        class Recording(LazyFrameSource):
            def materialize(self, method="collect", *args, **kwargs):
                calls.append(method)
                return super().materialize(method, *args, **kwargs)

        source = Recording(pl.LazyFrame({"a": [1]}))
        source.stream_tabular(tmp_path / "streamed.parquet", "parquet")
        source.write_tabular(tmp_path / "written.parquet", "parquet")

        assert calls == ["sink_parquet", "collect"]

    @pytest.mark.parametrize("format", list(DataFormat))
    def test_tabular_kwargs_are_empty_for_every_format(self, format):
        """A source with no variable behind it, like a crosswalk or the dataset list, adds
        nothing to the writer arguments the request supplies."""
        assert LazyFrameSource(pl.LazyFrame({"a": [1]})).tabular_kwargs(format) == {}


class TestParallelRasterHelpers:
    @pytest.fixture
    def rasters(self, base_path):
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)

        # COGs are named with a UUID, so filename order says nothing about which
        # year a file holds -- order by the calendar value in the band description
        # instead, which is where write_cog records it.
        def calendar_year(path):
            with rio.open(path) as dataset:
                return int(dataset.descriptions[0].split("=")[1])

        return sorted(base_path.rglob("*.tif"), key=calendar_year)

    async def test_parallel_read_returns_one_array_per_dataset(self, rasters):
        datasets = [rio.open(p) for p in rasters]
        try:
            results = await parallel_read_rasters(datasets, None)
        finally:
            for dataset in datasets:
                dataset.close()
        assert len(results) == 2
        assert results[0][0][0, 0] == 30
        assert results[1][0][0, 0] == 40

    async def test_parallel_read_honours_a_window(self, rasters):
        from rasterio.windows import Window

        datasets = [rio.open(p) for p in rasters]
        try:
            results = await parallel_read_rasters(datasets, Window(0, 0, 4, 3))
        finally:
            for dataset in datasets:
                dataset.close()
        assert results[0].shape == (1, 3, 4)

    async def test_parallel_sample_returns_values_per_coordinate(self, rasters):
        datasets = [rio.open(p) for p in rasters]
        try:
            centre = datasets[0].xy(datasets[0].height // 2, datasets[0].width // 2)
            results = await parallel_sample_rasters(datasets, [centre])
        finally:
            for dataset in datasets:
                dataset.close()
        assert len(results) == 2
        assert np.asarray(results[0]).ravel()[0] == 30

    async def test_use_gdal_env_true_applies_config_on_the_worker_thread(self, rasters):
        """Proves the env genuinely reaches the thread that performs the
        read (not just that the call completes without error) by having a stand-in
        dataset record the GDAL config actually in effect, and which thread it ran on,
        at the moment `.read()` is called -- the same call `_read_window` makes."""
        import threading

        from rasterio._env import get_gdal_config

        from cpra.mp.data.config import RASTER_NUM_THREADS

        class RecordingDataset:
            def __init__(self, dataset):
                self._dataset = dataset
                self.thread = None
                self.num_threads = None

            def read(self, indexes=None, window=None, out_shape=None, resampling=None):
                self.thread = threading.current_thread()
                self.num_threads = get_gdal_config("GDAL_NUM_THREADS")
                return self._dataset.read(indexes=indexes, window=window, out_shape=out_shape, resampling=resampling)

        dataset = rio.open(rasters[0])
        try:
            recorder = RecordingDataset(dataset)
            results = await parallel_read_rasters([recorder], None, use_gdal_env=True)
        finally:
            dataset.close()

        assert len(results) == 1
        # run_in_executor's default pool never reuses the calling (main) thread.
        assert recorder.thread is not threading.main_thread()
        assert recorder.num_threads == RASTER_NUM_THREADS

    async def test_use_gdal_env_false_does_not_set_the_config(self, rasters):
        from rasterio._env import get_gdal_config

        class RecordingDataset:
            def __init__(self, dataset):
                self._dataset = dataset
                self.num_threads = "not observed"

            def read(self, indexes=None, window=None, out_shape=None, resampling=None):
                self.num_threads = get_gdal_config("GDAL_NUM_THREADS")
                return self._dataset.read(indexes=indexes, window=window, out_shape=out_shape, resampling=resampling)

        dataset = rio.open(rasters[0])
        try:
            recorder = RecordingDataset(dataset)
            await parallel_read_rasters([recorder], None, use_gdal_env=False)
        finally:
            dataset.close()

        assert recorder.num_threads is None


class TestBlockGroupedSampling:
    """`_sample_raster` groups coordinates by COG block and issues one windowed read per
    block instead of one 1x1 read per coordinate. The contract it has to keep while doing
    that is `rasterio`'s own `sample`, value for value, so most of these compare against it
    directly rather than against hand-written expectations.
    """

    BLOCK = 16  # GTiff tiles must be a multiple of 16; this is the smallest legal one.

    @pytest.fixture
    def tiled(self, base_path):
        """A 64x64 two-band raster in 16x16 blocks -- 16 blocks, so coordinates can be
        placed to share a block or to land on different ones. Every pixel holds a distinct
        value, so a sample attributed to the wrong pixel cannot coincidentally be right.
        """
        path = base_path / "tiled.tif"
        rows = np.arange(64).reshape(64, 1) * 100
        cols = np.arange(64).reshape(1, 64)
        data = np.stack([rows + cols, rows + cols + 500_000]).astype("int32")
        with rio.open(
            path,
            "w",
            driver="GTiff",
            width=64,
            height=64,
            count=2,
            dtype="int32",
            nodata=-7,
            tiled=True,
            blockxsize=self.BLOCK,
            blockysize=self.BLOCK,
            crs=CRS.from_epsg(4326),
            transform=from_bounds(0, 0, 64, 64, 64, 64),
        ) as dst:
            dst.write(data)
        return path

    @staticmethod
    def _xy(dataset, positions):
        return [dataset.xy(row, col) for row, col in positions]

    def test_it_matches_rasterios_own_sample_value_for_value(self, tiled):
        # Coordinates chosen to hit every case the grouping distinguishes at once: several
        # in one block, several alone in their own, a repeat, and an order that is not the
        # order the blocks are visited in.
        positions = [(0, 0), (40, 40), (1, 2), (63, 63), (2, 1), (40, 41), (0, 0), (17, 33)]
        with rio.open(tiled) as dataset:
            coordinates = self._xy(dataset, positions)
            got = raster_io._sample_raster(dataset, coordinates, [1, 2])
            want = list(dataset.sample(coordinates, indexes=[1, 2]))

        assert [tuple(v) for v in got] == [tuple(v) for v in want]
        # Not a comparison of two lists of nodata: the values are the planted ones.
        assert tuple(got[1]) == (4040, 504040)
        assert tuple(got[3]) == (6363, 506363)

    def test_a_coordinate_off_the_raster_yields_the_nodata_fill(self, tiled):
        # `sample`'s out-of-extent behaviour is to yield the dataset nodata rather than
        # raise, and callers code to that; grouping must not quietly turn it into an error
        # or into a read of some clamped in-range pixel.
        with rio.open(tiled) as dataset:
            inside = dataset.xy(5, 5)
            outside = [(-100.0, -100.0), (1000.0, 1000.0)]
            coordinates = [inside, *outside]
            got = raster_io._sample_raster(dataset, coordinates, [1, 2])
            want = list(dataset.sample(coordinates, indexes=[1, 2]))

        assert [tuple(v) for v in got] == [tuple(v) for v in want]
        assert tuple(got[0]) == (505, 500505)
        assert tuple(got[1]) == (-7, -7) and tuple(got[2]) == (-7, -7)

    def test_it_reads_once_per_block_not_once_per_coordinate(self, tiled):
        # The behaviour change itself. 12 coordinates inside one 16x16 block must cost one
        # read; spread one per block they cost one read each. Checked by counting, because
        # the values alone cannot tell the two apart.
        class CountingDataset:
            """Forwards everything `_sample_raster` touches, recording `read` windows."""

            def __init__(self, dataset):
                self._dataset = dataset
                self.windows = []

            def __getattr__(self, name):
                return getattr(self._dataset, name)

            def read(self, **kwargs):
                self.windows.append(kwargs["window"])
                return self._dataset.read(**kwargs)

        shared = [(r, c) for r in range(3) for c in range(4)]
        spread = [(r * self.BLOCK, c * self.BLOCK) for r in range(4) for c in range(3)]
        with rio.open(tiled) as dataset:
            counter = CountingDataset(dataset)
            raster_io._sample_raster(counter, self._xy(dataset, shared), [1])
            shared_reads = list(counter.windows)

            counter.windows.clear()
            raster_io._sample_raster(counter, self._xy(dataset, spread), [1])
            spread_reads = list(counter.windows)

        assert len(shared) == len(spread) == 12
        assert len(shared_reads) == 1
        assert len(spread_reads) == 12
        # And the one read covers only the coordinates' own bounding box, not the whole
        # block -- 3 rows by 4 columns, which is what was asked for.
        assert (shared_reads[0].height, shared_reads[0].width) == (3, 4)

    def test_a_sample_does_not_retain_the_block_it_came_from(self, tiled):
        # A numpy slice of the block array is a VIEW, and a view keeps its whole parent
        # alive -- one retained view per coordinate turns a bounded per-block read into
        # every block ever read, held until the caller drops the last sample. That cost
        # 5.8 GiB of peak RSS for 1,000 scattered coordinates when this was first written,
        # and nothing about the returned VALUES would have shown it.
        positions = [(r * 4, c * 4) for r in range(16) for c in range(16)]
        with rio.open(tiled) as dataset:
            got = raster_io._sample_raster(dataset, self._xy(dataset, positions), [1, 2])

        assert len(got) == 256
        assert all(value.base is None for value in got)

    def test_the_band_argument_is_accepted_in_every_form_sample_accepts(self, tiled):
        # `sample` normalizes `indexes` itself -- None for every band, a bare int for one,
        # a list for a subset -- and this reimplements that, since it no longer calls
        # `sample` to do it. All three have to keep meaning what they meant.
        with rio.open(tiled) as dataset:
            coordinates = self._xy(dataset, [(9, 9), (40, 40)])
            every = raster_io._sample_raster(dataset, coordinates)
            one = raster_io._sample_raster(dataset, coordinates, 2)
            listed = raster_io._sample_raster(dataset, coordinates, [2])

        assert [tuple(v) for v in every] == [(909, 500909), (4040, 504040)]
        assert [tuple(v) for v in one] == [(500909,), (504040,)]
        assert [tuple(v) for v in one] == [tuple(v) for v in listed]

    def test_no_coordinates_is_no_samples_rather_than_a_read(self, tiled):
        # Not reachable through the point reader, which rejects an empty id list before it
        # gets here, but `sample` answers `[]` and this claims to answer what `sample`
        # answers. Zipping an empty coordinate list is also the one input that would raise
        # on the unpacking below rather than return anything at all.
        with rio.open(tiled) as dataset:
            assert raster_io._sample_raster(dataset, [], [1]) == []
            assert list(dataset.sample([], indexes=[1])) == []


class TestRasterBlocksPerBatch:
    """The batch size is derived from the configured memory budget and the
    request's own band count/block shape, not hardcoded."""

    def test_more_bands_means_fewer_blocks_per_batch(self):
        few_bands = raster_blocks_per_batch(1, 512, 512, 4)
        many_bands = raster_blocks_per_batch(52, 512, 512, 4)
        assert many_bands < few_bands

    def test_never_returns_less_than_one(self):
        assert raster_blocks_per_batch(10_000, 512, 512, 8) >= 1

    def test_explicit_override_takes_precedence(self, monkeypatch):
        import cpra.mp.data.analysis.io.raster_io as api_utils

        monkeypatch.setattr(api_utils, "RASTER_BLOCKS_PER_BATCH", 3)
        assert raster_blocks_per_batch(52, 512, 512, 4) == 3

    def test_derives_from_the_configured_buffer_budget(self, monkeypatch):
        import cpra.mp.data.analysis.io.raster_io as api_utils

        monkeypatch.setattr(api_utils, "RASTER_BLOCKS_PER_BATCH", None)
        monkeypatch.setattr(api_utils, "RASTER_BATCH_BUFFER_MB", 8)
        # 8 MB budget / (2 bands x 512 x 512 x 4 bytes = 2 MiB per block) -> 4 blocks.
        assert raster_blocks_per_batch(2, 512, 512, 4) == 4


class TestRasterGdalEnv:
    """What the read env actually puts into GDAL, read back out of GDAL rather than off the
    `rio.Env` object -- the two disagreed, and the whole point of these tests is the
    disagreement.
    """

    def test_the_cache_cap_reaches_gdal_as_bytes_not_megabytes(self, monkeypatch):
        """`GDAL_CACHEMAX` is the one config option rasterio does not hand to GDAL's own
        parser: it special-cases the name and calls `GDALSetCacheMax64`, which takes a byte
        count, so the megabyte figure the constant is named for has to be converted first.
        Passed through as it stood, the 2048 in `RASTER_CACHE_MAX_MB` asked GDAL for a 2 KB
        block cache -- which cost nothing on a full-resolution windowed read (each source
        block is touched once) and roughly 100x on an overview-served one (blocks are
        revisited, and every revisit re-decompressed).

        The expected value is written out as a literal rather than as the same arithmetic
        the code under test performs, so this fails if that arithmetic is dropped.
        """
        monkeypatch.setattr(raster_io, "RASTER_CACHE_MAX_MB", 64)
        with raster_gdal_env():
            assert get_gdal_config("GDAL_CACHEMAX") == 67_108_864

    def test_the_cap_is_not_the_bare_megabyte_figure(self, monkeypatch):
        """The specific regression, stated as itself: a cap that arrives as the raw constant
        is a cache three orders of magnitude smaller than the one asked for, and nothing
        about the read that follows reports it -- it only runs slowly.
        """
        monkeypatch.setattr(raster_io, "RASTER_CACHE_MAX_MB", 2048)
        with raster_gdal_env():
            cache_max = get_gdal_config("GDAL_CACHEMAX")
        assert cache_max != 2048
        assert cache_max >= 64 * 1024 * 1024

    def test_the_thread_count_is_passed_through_unchanged(self, monkeypatch):
        """Unlike the cap, `GDAL_NUM_THREADS` is an ordinary config option and takes GDAL's
        own spellings, so it is forwarded verbatim -- no conversion, and in particular none
        borrowed from the cap beside it.

        Checked with `ALL_CPUS` (the configured default) rather than a count because
        `get_gdal_config` parses a numeric string back as an `int`, which would leave the
        assertion testing that round trip instead of the forwarding.
        """
        monkeypatch.setattr(raster_io, "RASTER_NUM_THREADS", "ALL_CPUS")
        with raster_gdal_env():
            assert get_gdal_config("GDAL_NUM_THREADS") == "ALL_CPUS"

    def test_the_settings_do_not_outlive_the_env(self, monkeypatch):
        """Scoping the cap matters as much as its size: it is process-global while the env is
        open (see `raster_gdal_env`), so a value that leaked past `__exit__` would follow
        every later read in the process, including ones that never entered this env.
        """
        monkeypatch.setattr(raster_io, "RASTER_CACHE_MAX_MB", 64)
        with raster_gdal_env():
            inside = get_gdal_config("GDAL_CACHEMAX")
        assert inside == 67_108_864
        assert get_gdal_config("GDAL_CACHEMAX") != inside
