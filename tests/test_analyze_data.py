"""Tests for `analyze_data`, the library entry point that wraps `cpra.mp.data.analysis`
(both its flat `create_analysis` surface and its `parse_pipeline`/`compile_pipeline`
surface) so a notebook or script can run an analysis without HTTP and without touching
the underlying `BaseAnalysis`/`PairedAnalysis` objects directly. `read_data`'s own tests
live in `tests/test_init.py`; this module covers the new function only.

Follows `tests/api/test_analysis_entrypoint.py`'s convention: real variables/grids from
`tests/factories.py`, `tmp_path`-backed fixtures, and `async def test_...` (the suite runs
under `asyncio_mode = "auto"`, so no explicit marker is needed).
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import rasterio as rio
from polars.testing import assert_frame_equal

from cpra.mp.data import analyze_data
from cpra.mp.data.analysis import AnalysisError
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
def raster_data(base_path):
    """Two single-band COGs (2030 uniformly filled with 2030, 2040 with 2040)."""
    write_raster_dataset(base_path, years=(2030, 2040))
    return base_path


@pytest.fixture
def scenario_raster_data(base_path):
    """One raster, uniformly filled 30, so a zonal mean has an obvious hand-computed
    answer."""
    write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _year: 30, scenario_id=7)
    return base_path


@pytest.fixture
def zone_crosswalk(crosswalk_path):
    """Split the raster grid into two zones by row: zone 1 on top, zone 2 below."""
    zones = np.zeros((RASTER_GRID.height, RASTER_GRID.width), dtype="int32")
    zones[: RASTER_GRID.height // 2, :] = 1
    zones[RASTER_GRID.height // 2 :, :] = 2
    return write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)


class TestFlatVector:
    async def test_dimension_filter_where_and_group_aggregate(self, two_scenario_annual_data):
        """scenario_id=7 (scalar) excludes the scenario-8 rows entirely; `where` then
        drops calendar_year=2030, leaving one (2040) row per cell for `group_aggregate`
        to collapse -- so the result is just that cell's 2040 value."""
        df = await analyze_data(
            variable=ANNUAL_KWARGS["variable"],
            time_unit=ANNUAL_KWARGS["time_unit"],
            grid=ANNUAL_KWARGS["grid"],
            scenario_id=7,
            where="calendar_year > 2035",
            group_aggregate="mean",
            base_path=two_scenario_annual_data,
        )

        assert isinstance(df, pl.DataFrame)
        assert df.height == 32
        cell_1 = df.filter(pl.col("grid_cell") == 1)
        assert cell_1["value"][0] == pytest.approx(2040 + 1 / 100)


class TestFlatRaster:
    async def test_summarize_returns_a_frame(self, scenario_raster_data, zone_crosswalk):
        """Summarizing the uniform fill=30 raster over zone 1 must average back to 30 --
        and, being narrowed by `summarize_by`, the result is a frame, not a raster."""
        result = await analyze_data(
            variable=RASTER_KWARGS["variable"],
            time_unit=RASTER_KWARGS["time_unit"],
            grid=RASTER_KWARGS["grid"],
            scenario_id=7,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="mean",
            base_path=scenario_raster_data,
        )

        assert isinstance(result, pl.DataFrame)
        zone_1 = result.filter(pl.col("grid_cell") == 1)
        assert zone_1["value"][0] == pytest.approx(30)

    async def test_raster_shaped_without_output_path_raises(self, raster_data):
        with pytest.raises(AnalysisError) as excinfo:
            await analyze_data(
                variable=RASTER_KWARGS["variable"],
                time_unit=RASTER_KWARGS["time_unit"],
                grid=RASTER_KWARGS["grid"],
                base_path=raster_data,
            )
        assert "output_path" in excinfo.value.detail

    async def test_raster_shaped_with_output_path_writes_a_tif(self, raster_data, tmp_path):
        out_path = tmp_path / "out.tif"
        result = await analyze_data(
            variable=RASTER_KWARGS["variable"],
            time_unit=RASTER_KWARGS["time_unit"],
            grid=RASTER_KWARGS["grid"],
            base_path=raster_data,
            output_path=out_path,
        )

        assert result == out_path
        assert isinstance(result, Path)
        assert out_path.exists()
        with rio.open(out_path) as src:
            assert src.count == 2
            assert src.descriptions == ("calendar_year=2030", "calendar_year=2040")
            assert np.all(src.read(1) == 2030)
            assert np.all(src.read(2) == 2040)

    async def test_output_path_for_a_tabular_result_raises(self, scenario_raster_data, zone_crosswalk, tmp_path):
        with pytest.raises(AnalysisError):
            await analyze_data(
                variable=RASTER_KWARGS["variable"],
                time_unit=RASTER_KWARGS["time_unit"],
                grid=RASTER_KWARGS["grid"],
                scenario_id=7,
                summarize_by="hydro_compartment_v001",
                summarize_aggregate="mean",
                base_path=scenario_raster_data,
                output_path=tmp_path / "should-not-be-written.tif",
            )


class TestPipelineSurface:
    async def test_single_input_pipeline_returns_a_frame(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [{"op": "project", "input": "eco", "to": "ecoregion_v001"}],
            "output": {"format": "csv"},
        }

        df = await analyze_data(pipeline=pipeline, base_path=base_path)

        assert isinstance(df, pl.DataFrame)
        assert df.height == 32

    async def test_pipeline_and_flat_params_together_raises(self):
        with pytest.raises(AnalysisError, match="mutually exclusive"):
            await analyze_data(
                pipeline={"inputs": {}, "steps": [], "output": {}},
                variable=ANNUAL_KWARGS["variable"],
            )


class TestFlatPipelineEquivalence:
    async def test_flat_call_and_its_pipeline_twin_agree(self, two_scenario_annual_data):
        """The same request, expressed the two supported ways, must materialize
        identically -- the pipeline surface is a second REQUEST GRAMMAR over the same
        underlying machinery, not a second implementation of it."""
        flat_df = await analyze_data(
            variable=ANNUAL_KWARGS["variable"],
            time_unit=ANNUAL_KWARGS["time_unit"],
            grid=ANNUAL_KWARGS["grid"],
            scenario_id=7,
            where="calendar_year > 2035",
            group_aggregate="mean",
            base_path=two_scenario_annual_data,
        )

        pipeline_df = await analyze_data(
            pipeline={
                "inputs": {
                    "eco": {
                        "variable": ANNUAL_KWARGS["variable"],
                        "time_unit": ANNUAL_KWARGS["time_unit"],
                        "grid": ANNUAL_KWARGS["grid"],
                        "scenario_id": 7,
                    }
                },
                "steps": [
                    {"op": "project", "input": "eco", "to": ANNUAL_KWARGS["grid"]},
                    {"op": "filter", "where": "calendar_year > 2035"},
                    {"op": "group", "by": [], "aggregate": "mean"},
                ],
                "output": {"format": "csv"},
            },
            base_path=two_scenario_annual_data,
        )

        assert_frame_equal(flat_df.sort("grid_cell"), pipeline_df.sort("grid_cell"))


class TestCalculate:
    """The generic calculate entries on the library surface: the flat `calculate=` list
    (the HTTP endpoints' `calculate.N.*` parameters) and the pipeline `calculate` op,
    which compiles onto the same machinery."""

    async def test_flat_calculate_list(self, two_scenario_annual_data):
        """The confidence-weighting idiom: two selections each scaled by their own
        coefficient, collapsed with group_aggregate=sum."""
        df = await analyze_data(
            variable=ANNUAL_KWARGS["variable"],
            time_unit=ANNUAL_KWARGS["time_unit"],
            grid=ANNUAL_KWARGS["grid"],
            calculate=[
                {"expression": "value * 0.5", "where": "scenario_id = 7"},
                {"expression": "value * 0.25", "where": "scenario_id = 8"},
            ],
            group_by="calendar_year",
            group_aggregate="sum",
            base_path=two_scenario_annual_data,
        )
        cell_1 = df.filter((pl.col("grid_cell") == 1) & (pl.col("calendar_year") == 2030))
        assert cell_1["value"][0] == pytest.approx(2030.01 * 0.75, rel=1e-5)

    async def test_pipeline_calculate_op_matches_the_flat_spelling(self, two_scenario_annual_data):
        flat_df = await analyze_data(
            variable=ANNUAL_KWARGS["variable"],
            time_unit=ANNUAL_KWARGS["time_unit"],
            grid=ANNUAL_KWARGS["grid"],
            scenario_id=7,
            calculate=[{"expression": "value * 0.5", "where": "calendar_year = 2030"}],
            group_aggregate="sum",
            base_path=two_scenario_annual_data,
        )
        pipeline_df = await analyze_data(
            pipeline={
                "inputs": {
                    "eco": {
                        "variable": ANNUAL_KWARGS["variable"],
                        "time_unit": ANNUAL_KWARGS["time_unit"],
                        "grid": ANNUAL_KWARGS["grid"],
                        "scenario_id": 7,
                    }
                },
                "steps": [
                    {"op": "project", "input": "eco", "to": ANNUAL_KWARGS["grid"]},
                    {"op": "calculate", "expression": "eco * 0.5", "where": "calendar_year = 2030"},
                    {"op": "group", "by": [], "aggregate": "sum"},
                ],
                "output": {"format": "csv"},
            },
            base_path=two_scenario_annual_data,
        )
        assert_frame_equal(flat_df.sort("grid_cell"), pipeline_df.sort("grid_cell"))

    async def test_mixing_legacy_and_generic_raises(self, two_scenario_annual_data):
        with pytest.raises(AnalysisError, match="ambiguous"):
            await analyze_data(
                variable=ANNUAL_KWARGS["variable"],
                time_unit=ANNUAL_KWARGS["time_unit"],
                grid=ANNUAL_KWARGS["grid"],
                negate="calendar_year = 2030",
                calculate=[{"expression": "value * 2"}],
                base_path=two_scenario_annual_data,
            )


class TestLazy:
    async def test_lazy_true_returns_a_lazyframe_matching_the_eager_result(self, two_scenario_annual_data):
        eager = await analyze_data(
            variable=ANNUAL_KWARGS["variable"],
            time_unit=ANNUAL_KWARGS["time_unit"],
            grid=ANNUAL_KWARGS["grid"],
            scenario_id=7,
            base_path=two_scenario_annual_data,
        )
        lazy = await analyze_data(
            variable=ANNUAL_KWARGS["variable"],
            time_unit=ANNUAL_KWARGS["time_unit"],
            grid=ANNUAL_KWARGS["grid"],
            scenario_id=7,
            base_path=two_scenario_annual_data,
            lazy=True,
        )

        assert isinstance(lazy, pl.LazyFrame)
        sort_cols = ["grid_cell", "calendar_year"]
        assert_frame_equal(lazy.collect().sort(sort_cols), eager.sort(sort_cols))

    async def test_lazy_true_with_a_raster_shaped_result_raises(self, raster_data):
        with pytest.raises(AnalysisError):
            await analyze_data(
                variable=RASTER_KWARGS["variable"],
                time_unit=RASTER_KWARGS["time_unit"],
                grid=RASTER_KWARGS["grid"],
                base_path=raster_data,
                lazy=True,
            )


class TestUnknownKeyword:
    async def test_unknown_keyword_is_rejected_and_named(self):
        with pytest.raises(AnalysisError) as excinfo:
            await analyze_data(
                variable=ANNUAL_KWARGS["variable"],
                time_unit=ANNUAL_KWARGS["time_unit"],
                grid=ANNUAL_KWARGS["grid"],
                scenario_id=7,
                scenario_idd=7,  # typo
            )
        assert "scenario_idd" in excinfo.value.detail
