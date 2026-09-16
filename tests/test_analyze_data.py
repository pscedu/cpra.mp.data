"""Tests for `analyze_data`, the pipeline-only library entry point that wraps
`cpra.mp.data.analysis`'s `parse_pipeline`/`compile_pipeline` surface so a notebook or
script can run an analysis without HTTP and without touching the underlying
`BaseAnalysis`/`PairedAnalysis` objects directly -- the `/v1/analysis` equivalent
(`read_data` is `/v1/datasets`'s; its own tests, including its new `grid_cell`/`where`/
`sort`/`select` selection parameters, live in `tests/test_init.py`).

`analyze_data` has no flat/keyword calling surface at all -- `pipeline` is
its one required, positional-first argument, with no `**params` left to catch a stray
keyword (`TestPipelineOnlySurface::test_flat_keywords_are_rejected_by_python_itself` below
proves what a caller still spelling a request the old way gets: Python's own `TypeError`,
raised before `analyze_data`'s body ever runs). Every test below that used to be a flat
`analyze_data(variable=..., ...)` call is rewritten as either a `read_data` call (plain
dimension filtering, no aggregation -- `tests/test_init.py`) or a pipeline body here.
`cpra.mp.data.analysis.create_analysis`'s own flat/plain-value calling convention is
unaffected by this change and is exercised directly, not through `analyze_data`, by
`tests/api/test_analysis_entrypoint.py`.

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
    """Two years x two scenarios of ssf/ecoregion_v001, so a `scenario_id` input filter
    has something to actually exclude rather than trivially matching everything written."""
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


class TestPipelineVector:
    async def test_project_filter_and_group_aggregate(self, two_scenario_annual_data):
        """scenario_id=7 (an input dimension filter) excludes the scenario-8 rows
        entirely; `filter` then drops calendar_year=2030, leaving one (2040) row per cell
        for `group` to collapse -- so the result is just that cell's 2040 value. The
        pipeline-shaped replacement for what used to be a flat
        `analyze_data(..., where=..., group_aggregate=...)` call -- that flat convention
        itself still exists and is still tested directly, as `create_analysis`, by
        `tests/api/test_analysis_entrypoint.py::TestVectorPlainValues`."""
        pipeline = {
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
        }

        df = await analyze_data(pipeline, base_path=two_scenario_annual_data)

        assert isinstance(df, pl.DataFrame)
        assert df.height == 32
        cell_1 = df.filter(pl.col("grid_cell") == 1)
        assert cell_1["value"][0] == pytest.approx(2040 + 1 / 100)


class TestPipelineRaster:
    async def test_project_to_crosswalk_returns_a_frame(self, scenario_raster_data, zone_crosswalk):
        """Projecting the uniform fill=30 raster onto the zone crosswalk with
        `aggregate=mean` must average back to 30 -- and, being narrowed to a vector grid,
        the result is a frame, not a raster."""
        pipeline = {
            "inputs": {
                "vct": {
                    "variable": RASTER_KWARGS["variable"],
                    "time_unit": RASTER_KWARGS["time_unit"],
                    "grid": RASTER_KWARGS["grid"],
                    "scenario_id": 7,
                }
            },
            "steps": [{"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"}],
            "output": {"format": "csv"},
        }

        result = await analyze_data(pipeline, base_path=scenario_raster_data)

        assert isinstance(result, pl.DataFrame)
        zone_1 = result.filter(pl.col("grid_cell") == 1)
        assert zone_1["value"][0] == pytest.approx(30)

    async def test_raster_shaped_without_output_path_raises(self, raster_data):
        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [{"op": "project", "input": "vct", "to": RASTER_KWARGS["grid"]}],
            "output": {"format": "tif"},
        }
        with pytest.raises(AnalysisError) as excinfo:
            await analyze_data(pipeline, base_path=raster_data)
        assert "output_path" in excinfo.value.detail

    async def test_raster_shaped_with_output_path_writes_a_tif(self, raster_data, tmp_path):
        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [{"op": "project", "input": "vct", "to": RASTER_KWARGS["grid"]}],
            "output": {"format": "tif"},
        }
        out_path = tmp_path / "out.tif"

        result = await analyze_data(pipeline, base_path=raster_data, output_path=out_path)

        assert result == out_path
        assert isinstance(result, Path)
        assert out_path.exists()
        with rio.open(out_path) as src:
            assert src.count == 2
            assert src.descriptions == ("calendar_year=2030", "calendar_year=2040")
            assert np.all(src.read(1) == 2030)
            assert np.all(src.read(2) == 2040)

    async def test_output_path_for_a_tabular_result_raises(self, scenario_raster_data, zone_crosswalk, tmp_path):
        pipeline = {
            "inputs": {
                "vct": {
                    "variable": RASTER_KWARGS["variable"],
                    "time_unit": RASTER_KWARGS["time_unit"],
                    "grid": RASTER_KWARGS["grid"],
                    "scenario_id": 7,
                }
            },
            "steps": [{"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"}],
            "output": {"format": "csv"},
        }
        with pytest.raises(AnalysisError):
            await analyze_data(
                pipeline,
                base_path=scenario_raster_data,
                output_path=tmp_path / "should-not-be-written.tif",
            )


class TestPipelineOnlySurface:
    async def test_single_input_pipeline_returns_a_frame(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [{"op": "project", "input": "eco", "to": "ecoregion_v001"}],
            "output": {"format": "csv"},
        }

        df = await analyze_data(pipeline, base_path=base_path)

        assert isinstance(df, pl.DataFrame)
        assert df.height == 32

    async def test_flat_keywords_are_rejected_by_python_itself(self):
        """`analyze_data`'s flat/keyword calling surface is gone entirely:
        no mutual-exclusivity check, no `**params` -- `pipeline` is the only argument, and
        it is required and positional-first. A caller still spelling a request the old
        way (`variable=`, `where=`, `group_aggregate=`, ...) gets Python's own `TypeError`
        for an unexpected keyword argument, raised before `analyze_data`'s body ever
        runs -- immediate and clearly named, not a silent no-op or a wrong answer.
        `cpra.mp.data.read_data` (dimension filters plus `grid_cell`/`where`/`sort`/
        `select`) or `cpra.mp.data.analysis.create_analysis` (the full flat surface,
        including `group_by`/`group_aggregate`/`summarize_*`) are today's replacements."""
        with pytest.raises(TypeError, match="variable"):
            await analyze_data(
                variable=ANNUAL_KWARGS["variable"],
                time_unit=ANNUAL_KWARGS["time_unit"],
                grid=ANNUAL_KWARGS["grid"],
                scenario_id=7,
                where="calendar_year > 2035",
                group_aggregate="mean",
            )


class TestCalculate:
    """The generic calculate entries on the pipeline surface -- the pipeline `calculate`
    op, which compiles onto the same machinery the (now `create_analysis`-only) flat
    `calculate=` list did."""

    async def test_pipeline_calculate_op(self, two_scenario_annual_data):
        pipeline = {
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
        }

        df = await analyze_data(pipeline, base_path=two_scenario_annual_data)

        # calculate halves only the calendar_year=2030 row (2030.01 -> 1015.005);
        # calendar_year=2040 (2040.01) passes through unchanged. group sums both rows for
        # grid_cell=1 across the collapsed calendar_year axis.
        cell_1 = df.filter(pl.col("grid_cell") == 1)
        assert cell_1["value"][0] == pytest.approx(2030.01 * 0.5 + 2040.01, rel=1e-5)


class TestLazy:
    async def test_lazy_true_returns_a_lazyframe_matching_the_eager_result(self, two_scenario_annual_data):
        pipeline = {
            "inputs": {
                "eco": {
                    "variable": ANNUAL_KWARGS["variable"],
                    "time_unit": ANNUAL_KWARGS["time_unit"],
                    "grid": ANNUAL_KWARGS["grid"],
                    "scenario_id": 7,
                }
            },
            "steps": [{"op": "project", "input": "eco", "to": ANNUAL_KWARGS["grid"]}],
            "output": {"format": "csv"},
        }

        eager = await analyze_data(pipeline, base_path=two_scenario_annual_data)
        lazy = await analyze_data(pipeline, base_path=two_scenario_annual_data, lazy=True)

        assert isinstance(lazy, pl.LazyFrame)
        sort_cols = ["grid_cell", "calendar_year"]
        assert_frame_equal(lazy.collect().sort(sort_cols), eager.sort(sort_cols))

    async def test_lazy_true_with_a_raster_shaped_result_raises(self, raster_data):
        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [{"op": "project", "input": "vct", "to": RASTER_KWARGS["grid"]}],
            "output": {"format": "tif"},
        }
        with pytest.raises(AnalysisError):
            await analyze_data(pipeline, base_path=raster_data, lazy=True)
