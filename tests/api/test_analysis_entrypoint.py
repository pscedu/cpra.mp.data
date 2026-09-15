"""Tests for `create_analysis`'s plain-value calling convention -- the public library
entry point `cpra.mp.data.analysis/__init__.py`'s module docstring documents for a
notebook, a script, or a future `read_data` extension. These call `create_analysis`
directly (no ASGI, no `dataset_factory`/query-model fixtures) to exercise exactly what a
library caller would use.

`TestAnalysisCompression` below is the one exception: G6 (compression on analysis
responses) is a query-binding/routing concern, not a `create_analysis` calling-convention
one, so it goes through the real ASGI app (the `client` fixture from
`tests/api/conftest.py`) instead.
"""

import io

import numpy as np
import polars as pl
import pytest
import rasterio as rio
from polars.testing import assert_frame_equal

from cpra.mp.data import config
from cpra.mp.data.analysis import AnalysisError, create_analysis
from cpra.mp.data.analysis.parameters import (
    DatasetDimensions,
    FilterDimensions,
    Options,
)
from cpra.mp.data.analysis.pipeline import parse_pipeline, pipeline_to_query
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.tabular import write_parquet
from tests.factories import (
    ANNUAL_KWARGS,
    RASTER_KWARGS,
    annual_frame,
    write_raster_crosswalk,
    write_raster_dataset,
)

RASTER_GRID = GRIDS[RASTER_KWARGS["grid"]]


@pytest.fixture
def two_scenario_annual_data(base_path):
    """Two years x two scenarios of ssf/ecoregion_v001, so a `scenario_id` filter has
    something to actually exclude rather than trivially matching everything written."""
    write_parquet(annual_frame(years=(2030, 2040), scenario_id=7), base_path=base_path)
    write_parquet(annual_frame(years=(2030, 2040), scenario_id=8), base_path=base_path)
    return base_path


@pytest.fixture
def two_scenario_raster_data(base_path):
    """Two uniform-fill rasters (scenario 7 filled 30, scenario 8 filled 999), so a
    `scenario_id` filter has something to actually exclude."""
    write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _year: 30, scenario_id=7)
    write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _year: 999, scenario_id=8)
    return base_path


@pytest.fixture
def zone_crosswalk(crosswalk_path):
    """Split the raster grid into two zones by row: zone 1 on top, zone 2 below."""
    zones = np.zeros((RASTER_GRID.height, RASTER_GRID.width), dtype="int32")
    zones[: RASTER_GRID.height // 2, :] = 1
    zones[RASTER_GRID.height // 2 :, :] = 2
    return write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)


class TestVectorPlainValues:
    async def test_dimension_filter_where_and_group_aggregate(
        self, two_scenario_annual_data
    ):
        """scenario_id=7 (scalar) excludes the scenario-8 rows entirely; `where` then
        drops calendar_year=2030, leaving one (2040) row per cell for `group_aggregate`
        to collapse -- so the result is just that cell's 2040 value."""
        analysis = create_analysis(
            variable=ANNUAL_KWARGS["variable"],
            time_unit=ANNUAL_KWARGS["time_unit"],
            grid=ANNUAL_KWARGS["grid"],
            scenario_id=7,
            where="calendar_year > 2035",
            group_aggregate="mean",
            base_path=two_scenario_annual_data,
        )
        await analysis.apply_params()
        df = analysis.materialize()

        assert df.height == 32
        cell_1 = df.filter(pl.col("grid_cell") == 1)
        assert cell_1["value"][0] == pytest.approx(2040 + 1 / 100)

    async def test_list_valued_dimension_filter_keeps_both_scenarios(
        self, two_scenario_annual_data
    ):
        analysis = create_analysis(
            variable=ANNUAL_KWARGS["variable"],
            time_unit=ANNUAL_KWARGS["time_unit"],
            grid=ANNUAL_KWARGS["grid"],
            scenario_id=[7, 8],
            group_by="scenario_id",
            group_aggregate="max",
            base_path=two_scenario_annual_data,
        )
        await analysis.apply_params()
        df = analysis.materialize()

        assert df.height == 2 * 32
        assert sorted(df["scenario_id"].unique().to_list()) == [7, 8]

    async def test_unknown_keyword_is_rejected(self):
        with pytest.raises(AnalysisError) as excinfo:
            create_analysis(
                variable=ANNUAL_KWARGS["variable"],
                time_unit=ANNUAL_KWARGS["time_unit"],
                grid=ANNUAL_KWARGS["grid"],
                scenario_id=7,
                scenario_idd=7,  # typo
            )
        assert "scenario_idd" in excinfo.value.detail
        # The valid-name list is there to make the typo obvious, not just the fact of one.
        assert "scenario_id" in excinfo.value.detail

    async def test_plain_value_and_query_model_calls_agree(self, two_scenario_annual_data):
        """The same request, built the two supported ways, must materialize identically --
        the plain-value surface is a convenience layer over the same objects, not a
        second code path with its own behaviour."""
        plain = create_analysis(
            variable=ANNUAL_KWARGS["variable"],
            time_unit=ANNUAL_KWARGS["time_unit"],
            grid=ANNUAL_KWARGS["grid"],
            scenario_id=7,
            where="calendar_year > 2035",
            group_aggregate="mean",
            base_path=two_scenario_annual_data,
        )
        await plain.apply_params()
        plain_df = plain.materialize().sort("grid_cell")

        objects = create_analysis(
            DatasetDimensions(
                variable=ANNUAL_KWARGS["variable"],
                time_unit=ANNUAL_KWARGS["time_unit"],
                grid=ANNUAL_KWARGS["grid"],
            ),
            FilterDimensions(scenario_id=[7]),
            Options(where="calendar_year > 2035", group_aggregate="mean"),
            base_path=two_scenario_annual_data,
        )
        await objects.apply_params()
        objects_df = objects.materialize().sort("grid_cell")

        assert_frame_equal(plain_df, objects_df)


class TestRasterPlainValues:
    async def test_dimension_filter_and_summarize(
        self, two_scenario_raster_data, zone_crosswalk
    ):
        """scenario_id=7 excludes the scenario-8 (fill=999) raster; summarizing the
        uniform scenario-7 (fill=30) raster over zone 1 must average back to 30."""
        analysis = create_analysis(
            variable=RASTER_KWARGS["variable"],
            time_unit=RASTER_KWARGS["time_unit"],
            grid=RASTER_KWARGS["grid"],
            scenario_id=7,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
            base_path=two_scenario_raster_data,
        )
        await analysis.apply_params()
        df = analysis.materialize()

        zone_1 = df.filter(pl.col("grid_cell") == 1)
        assert zone_1["value"][0] == pytest.approx(30)


class TestAnalysisCompression:
    """G6 -- `Compression` is bound as its own query model on `/v1/analysis.{format}`
    (POST and GET-dotted), which previously hardcoded `Compression()` regardless of what
    a caller sent. Exercised through the real ASGI routes (`api/__init__.py::post_analysis`,
    `get_analysis`), since this is about query binding and routing, not `create_analysis`.
    """

    RASTER_PIPELINE = {
        "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
        "steps": [
            {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
            {"op": "group", "by": [], "aggregate": "max"},
        ],
        "output": {"format": "tif"},
    }

    VECTOR_PIPELINE = {
        "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
        "steps": [{"op": "project", "input": "eco", "to": "ecoregion_v001"}],
        "output": {"format": "csv"},
    }

    def test_post_compression_reaches_the_raster_profile(self, client, base_path, monkeypatch):
        """A non-default `compression_algorithm`/`compression_level` given on the POST
        form actually reaches the written GeoTIFF's profile -- the same proof
        `test_responses.py::TestRasterioStreamingResponse.test_compression_reaches_the_
        output_profile` uses for `RasterioStreamingResponse` directly, run here through
        the full route so the query binding itself is exercised.
        """
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        response = client.post(
            "/v1/analysis.tif",
            json=self.RASTER_PIPELINE,
            params={"compression_algorithm": "deflate", "compression_level": 9},
        )
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            assert src.profile["compress"].lower() == "deflate"

    def test_get_dotted_compression_reaches_the_raster_profile(self, client, base_path, monkeypatch):
        """Same proof as above, through the GET dotted-query form."""
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        query = pipeline_to_query(parse_pipeline(self.RASTER_PIPELINE))
        query.update({"compression_algorithm": "deflate", "compression_level": "9"})
        response = client.get("/v1/analysis.tif", params=query)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            assert src.profile["compress"].lower() == "deflate"

    def test_get_dotted_pipeline_ignores_compression_keys(self, client, base_path, monkeypatch):
        """`compression_algorithm`/`compression_level` must be stripped from the raw
        query dict before it reaches `pipeline_from_query` -- neither is `input.*`/
        `step.*`/`output.*`, so left in place they would 400 as an "unknown top-level
        parameter group" (see `pipeline.pipeline_from_query`'s docstring) rather than
        being consumed as compression parameters.
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        query = pipeline_to_query(parse_pipeline(self.VECTOR_PIPELINE))
        query["compression_algorithm"] = "zstd"
        query["compression_level"] = "3"
        response = client.get("/v1/analysis.csv", params=query)
        assert response.status_code == 200, response.text

    def test_post_mismatched_compression_and_format_is_400(self, client, base_path, monkeypatch):
        """A raster-only compression algorithm requested against a tabular format is
        rejected by `validate_compression` -- proof the analysis routes now actually
        validate the caller's compression choice instead of silently ignoring it behind
        the old hardcoded `Compression()`.
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        response = client.post(
            "/v1/analysis.csv", json=self.VECTOR_PIPELINE, params={"compression_algorithm": "lzw"}
        )
        assert response.status_code == 400
        assert "Compression algorithm" in response.json()["detail"]

    def test_get_dotted_mismatched_compression_and_format_is_400(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        query = pipeline_to_query(parse_pipeline(self.VECTOR_PIPELINE))
        query["compression_algorithm"] = "lzw"
        response = client.get("/v1/analysis.csv", params=query)
        assert response.status_code == 400
        assert "Compression algorithm" in response.json()["detail"]

    def test_get_compression_defaults_when_unspecified(self, client, base_path, monkeypatch):
        """No compression params given -> falls back to `Compression()`'s own field
        defaults, so the new binding doesn't regress the previously-hardcoded behaviour
        for callers that don't care about compression."""
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        query = pipeline_to_query(parse_pipeline(self.RASTER_PIPELINE))
        response = client.get("/v1/analysis.tif", params=query)
        assert response.status_code == 200, response.text
