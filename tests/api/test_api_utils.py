"""Tests for cpra.mp.data.api.utils / cpra.mp.data.analysis.utils and the LazyFrameSource
base class. `get_lazy_frame`/`unpack_values` moved to `analysis.utils` (and raise
`AnalysisError`, not `HTTPException`) since the `BaseAnalysis` hierarchy needs them and
`analysis/` may not import FastAPI; `validate_compression`/`validate_format`/
`get_visualization_class`/`LazyFrameSource` stayed in `api/` and still raise `HTTPException`.
"""

import numpy as np
import polars as pl
import pytest
import rasterio as rio
from fastapi import HTTPException

from cpra.mp.data.analysis import AnalysisError
from cpra.mp.data.analysis.types import CompressionAlgorithm, DataFormat, OutputType
from cpra.mp.data.analysis.utils import (
    enum_value,
    get_lazy_frame,
    parallel_read_rasters,
    parallel_sample_rasters,
    raster_blocks_per_batch,
    unpack_values,
)
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
        with pytest.raises(AnalysisError) as excinfo:
            unpack_values("a,zzz", "sort", choices=["a", "b"])
        assert excinfo.value.status_code == 400
        assert "Invalid values for sort: zzz" in excinfo.value.detail

    def test_all_invalid_choices_are_listed(self):
        with pytest.raises(AnalysisError) as excinfo:
            unpack_values("x,y", "sort", choices=["a"])
        assert "x, y" in excinfo.value.detail


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
    """`get_analysis_class` (plan §3.2a) -- the named-analysis counterpart to
    `get_visualization_class` above, generalizing its exact 404 (unknown name) / 400
    (wrong route kind) split rather than reinventing it. The real registry
    (`cpra.mp.data.analyses.ANALYSES`) is a concurrent Phase 2 change that may not exist
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
            {"concatenate": _FakeDatasetAnalysisClass, "exit-criterion": _FakeCustomAnalysisClass},
        )

    def test_returns_a_known_analysis(self):
        assert get_analysis_class("concatenate").Meta.type == "dataset"

    def test_unknown_name_raises_404(self):
        with pytest.raises(HTTPException) as excinfo:
            get_analysis_class("nope")
        assert excinfo.value.status_code == 404
        assert "Unknown analysis name" in excinfo.value.detail

    def test_matching_type_is_accepted(self):
        assert get_analysis_class("concatenate", "dataset") is not None

    def test_mismatched_type_raises_400(self):
        with pytest.raises(HTTPException) as excinfo:
            get_analysis_class("concatenate", "custom")
        assert excinfo.value.status_code == 400
        assert "Invalid analysis type" in excinfo.value.detail

    def test_custom_type_mismatch_the_other_direction(self):
        with pytest.raises(HTTPException) as excinfo:
            get_analysis_class("exit-criterion", "dataset")
        assert excinfo.value.status_code == 400

    def test_type_is_optional(self):
        assert get_analysis_class("exit-criterion") is not None


class TestGetLazyFrame:
    def test_returns_a_lazy_frame(self, base_path):
        write_raster_dataset(base_path, years=(2030,))
        dims = DatasetDimensions(**{k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")})
        lf = get_lazy_frame(dims, FilterDimensions(), base_path=base_path)
        assert lf.collect().height == 1

    def test_list_filters_are_applied(self, base_path):
        write_raster_dataset(base_path, years=(2030, 2040))
        dims = DatasetDimensions(**{k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")})
        lf = get_lazy_frame(dims, FilterDimensions(calendar_year=[2040]), base_path=base_path)
        assert lf.collect()["calendar_year"].to_list() == [2040]

    def test_validation_errors_become_422(self, base_path):
        """An omitted grid leaves the field None, which read_data rejects."""
        dims = DatasetDimensions(variable="sal", time_unit="daily")
        with pytest.raises(AnalysisError) as excinfo:
            get_lazy_frame(dims, FilterDimensions(), base_path=base_path)
        assert excinfo.value.status_code == 422
        assert "Grid is required" in excinfo.value.detail

    def test_results_are_cached(self, base_path):
        """Repeated identical queries reuse the cached LazyFrame."""
        write_raster_dataset(base_path, years=(2030,))
        dims = DatasetDimensions(**{k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")})
        first = get_lazy_frame(dims, FilterDimensions(), base_path=base_path)
        second = get_lazy_frame(dims, FilterDimensions(), base_path=base_path)
        assert first is second


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
        import cpra.mp.data.analysis.utils as api_utils

        monkeypatch.setattr(api_utils, "RASTER_BLOCKS_PER_BATCH", 3)
        assert raster_blocks_per_batch(52, 512, 512, 4) == 3

    def test_derives_from_the_configured_buffer_budget(self, monkeypatch):
        import cpra.mp.data.analysis.utils as api_utils

        monkeypatch.setattr(api_utils, "RASTER_BLOCKS_PER_BATCH", None)
        monkeypatch.setattr(api_utils, "RASTER_BATCH_BUFFER_MB", 8)
        # 8 MB budget / (2 bands x 512 x 512 x 4 bytes = 2 MiB per block) -> 4 blocks.
        assert raster_blocks_per_batch(2, 512, 512, 4) == 4
