"""Tests for cpra.mp.data.api.utils and the LazyFrameSource base class."""

import numpy as np
import polars as pl
import pytest
import rasterio as rio
from fastapi import HTTPException

from cpra.mp.data.api.models import Compression, DatasetDimensions, FilterDimensions
from cpra.mp.data.api.sources import LazyFrameSource
from cpra.mp.data.api.types import CompressionAlgorithm, DataFormat, OutputType
from cpra.mp.data.api.utils import (
    enum_value,
    get_lazy_frame,
    get_visualization_class,
    parallel_read_rasters,
    parallel_sample_rasters,
    unpack_values,
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
        with pytest.raises(HTTPException) as excinfo:
            unpack_values("a,zzz", "sort", choices=["a", "b"])
        assert excinfo.value.status_code == 400
        assert "Invalid values for sort: zzz" in excinfo.value.detail

    def test_all_invalid_choices_are_listed(self):
        with pytest.raises(HTTPException) as excinfo:
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
        with pytest.raises(HTTPException) as excinfo:
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


class TestParallelRasterHelpers:
    @pytest.fixture
    def rasters(self, base_path):
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        return sorted(base_path.rglob("*.tif"))

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
