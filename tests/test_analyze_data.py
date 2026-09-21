"""Tests for `analyze_data`, the pipeline-only library entry point that runs the same
four stages an HTTP response does -- `parse_pipeline`, `validate_shapes`, `compile_plan`,
then `execute`/`run_to_path` -- so a notebook or script can run an analysis without
FastAPI in the way. It is `/v1/analysis`'s counterpart; `read_data` is `/v1/datasets`'s,
and its own tests (including its `grid_cell`/`where`/`sort`/`select` parameters) live in
`tests/test_init.py`.

What is tested HERE is only what `analyze_data` itself decides -- the return contract.
The pipeline grammar, the shape rules, and the execution semantics all belong to the
engine and are tested against it directly; a pipeline body below is the smallest one that
reaches the branch under test, not a survey of what the grammar can express.

Three things the rewrite onto that engine changed about how a request is spelled, all
visible below:

* **An input selects a dataset and nothing else.** `variable`/`time_unit`/`grid` are the
  only keys `InputSelection` accepts (`extra="forbid"`); a dimension filter that used to
  ride along inside `inputs` is a leading `filter` STEP now, which also makes its
  position in the chain explicit rather than implicit.
* **There is no raster-to-raster `project`.** A raster input is already band-wide the
  moment it is read, so a GeoTIFF request is the input plus whatever narrows it -- no
  projection onto the grid it is already on (there is no crosswalk between two raster
  lattices; see `request.ops.PROJECT_LOWERING_TABLE`).
* **The raster/tabular decision is `output.format`, not a re-derived shape.**
  `validate_shapes` has already forced the format and the final frame's shape to agree
  before `analyze_data` reads the format, so the two can never disagree here.

`analyze_data` is still a coroutine even though nothing in it awaits any more, so these
stay `async def test_...` (the suite runs under `asyncio_mode = "auto"`, so no marker is
needed). Real variables/grids come from `tests/factories.py` against `tmp_path`-backed
fixtures.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import rasterio as rio
from polars.testing import assert_frame_equal

from cpra.mp.data import analyze_data
from cpra.mp.data.analysis.request.errors import NotFoundError, ShapeError
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

# The vector/raster input selections these pipelines are built from. Each is exactly the
# three keys `InputSelection` accepts -- spelled once here so a stray fourth key cannot be
# reintroduced test by test.
ECO_INPUT = {k: ANNUAL_KWARGS[k] for k in ("variable", "time_unit", "grid")}
VCT_INPUT = {k: RASTER_KWARGS[k] for k in ("variable", "time_unit", "grid")}


def eco_pipeline(*steps, **output):
    """One `ssf`/`ecoregion_v001` input named `eco`, `steps`, and a CSV output by default."""
    return {"inputs": {"eco": ECO_INPUT}, "steps": list(steps), "output": {"format": "csv", **output}}


def vct_pipeline(*steps, **output):
    """One `vct`/`veg_grid_cell_v001` input named `vct`, `steps`, and a GeoTIFF output by
    default -- a raster input is band-wide from the start, so an unnarrowed one already
    satisfies `tif`'s "needs a raster surface" rule with no `project` in the way."""
    return {"inputs": {"vct": VCT_INPUT}, "steps": list(steps), "output": {"format": "tif", **output}}


@pytest.fixture
def two_scenario_annual_data(base_path):
    """Two years x two scenarios of ssf/ecoregion_v001, so a `scenario_id` filter step has
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


class TestPipelineVector:
    async def test_filter_and_group_aggregate(self, two_scenario_annual_data):
        """`scenario_id = 7` excludes the scenario-8 rows and `calendar_year > 2035` the
        2030 ones, leaving one row per cell for `group` to collapse -- so each cell's
        result is just its own 2040 value. Both conjuncts ride on the leading `filter`
        step: an input selects a dataset, never a slice of one."""
        pipeline = eco_pipeline(
            {"op": "filter", "input": "eco", "where": "scenario_id = 7 AND calendar_year > 2035"},
            {"op": "group", "by": [], "aggregate": "mean"},
        )

        df = await analyze_data(pipeline, base_path=two_scenario_annual_data)

        assert isinstance(df, pl.DataFrame)
        assert df.height == 32
        cell_1 = df.filter(pl.col("grid_cell") == 1)
        assert cell_1["value"][0] == pytest.approx(2040 + 1 / 100)

    async def test_output_sort_and_select_are_applied(self, two_scenario_annual_data):
        """`output.sort`/`output.select` are `execution.sink.apply_output_shaping`'s job,
        and a frame handed back from here has been through it -- otherwise the same
        pipeline body would mean one thing over HTTP and a quietly unshaped other thing
        from a notebook."""
        pipeline = eco_pipeline(
            {"op": "filter", "input": "eco", "where": "scenario_id = 7"},
            {"op": "group", "by": [], "aggregate": "mean"},
            sort="-grid_cell",
            select="grid_cell,value",
        )

        df = await analyze_data(pipeline, base_path=two_scenario_annual_data)

        assert df.columns == ["grid_cell", "value"]
        assert df["grid_cell"].to_list() == sorted(df["grid_cell"].to_list(), reverse=True)


class TestPipelineRaster:
    async def test_project_to_crosswalk_returns_a_frame(self, scenario_raster_data, zone_crosswalk):
        """Projecting the uniform fill=30 raster onto the zone crosswalk with
        `aggregate=mean` must average back to 30 -- and, being narrowed to a vector grid,
        the result is a frame, not a raster, which is why this one asks for `csv`."""
        pipeline = vct_pipeline(
            {"op": "filter", "input": "vct", "where": "scenario_id = 7"},
            {"op": "project", "to": "hydro_compartment_v001", "aggregate": "mean"},
            format="csv",
        )

        result = await analyze_data(pipeline, base_path=scenario_raster_data)

        assert isinstance(result, pl.DataFrame)
        zone_1 = result.filter(pl.col("grid_cell") == 1)
        assert zone_1["value"][0] == pytest.approx(30)

    async def test_tif_without_output_path_raises(self, raster_data):
        with pytest.raises(ShapeError) as excinfo:
            await analyze_data(vct_pipeline(), base_path=raster_data)
        assert "output_path" in str(excinfo.value)

    async def test_tif_with_output_path_writes_a_tif(self, raster_data, tmp_path):
        out_path = tmp_path / "out.tif"

        result = await analyze_data(vct_pipeline(), base_path=raster_data, output_path=out_path)

        assert result == out_path
        assert isinstance(result, Path)
        assert out_path.exists()
        with rio.open(out_path) as src:
            assert src.count == 2
            assert np.all(src.read(1) == 2030)
            assert np.all(src.read(2) == 2040)
            # Band order is not the only record of which band is which: each band says so
            # itself, in the same `"<dimension>=<value>"` spelling `cpra.mp.data.raster`
            # writes and parses back out, so a client holding only these bytes can still
            # tell 2030 from 2040. Every dimension the band resolved from is spelled, not
            # just the calendar one -- the three that name the dataset are redundant for a
            # single-input request like this one and are exactly what distinguishes the
            # bands once a `join` has put two datasets in one file.
            assert src.descriptions == (
                "variable=vct, time_unit=annual, grid=veg_grid_cell_v001, "
                "model_group_id=500, scenario_id=7, calendar_year=2030",
                "variable=vct, time_unit=annual, grid=veg_grid_cell_v001, "
                "model_group_id=500, scenario_id=7, calendar_year=2040",
            )

    async def test_a_tif_selection_matching_nothing_raises_rather_than_returning_a_missing_path(
        self, raster_data, tmp_path
    ):
        """`run_to_path` returns `False` (and writes no file) for a band-wide result with
        no bands: there is no band count to declare a raster from. Handing back the path
        anyway would point a caller at a file that does not exist, so this is the one
        place `analyze_data` turns a sink's `False` into an error."""
        out_path = tmp_path / "empty.tif"
        pipeline = vct_pipeline({"op": "filter", "input": "vct", "where": "calendar_year = 1999"})

        with pytest.raises(NotFoundError) as excinfo:
            await analyze_data(pipeline, base_path=raster_data, output_path=out_path)

        assert "no bands" in str(excinfo.value)
        assert not out_path.exists()

    async def test_output_path_for_a_tabular_result_raises(self, scenario_raster_data, zone_crosswalk, tmp_path):
        pipeline = vct_pipeline(
            {"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"},
            format="csv",
        )
        with pytest.raises(ShapeError):
            await analyze_data(
                pipeline,
                base_path=scenario_raster_data,
                output_path=tmp_path / "should-not-be-written.tif",
            )


class TestTileFormats:
    """`png`/`jpg` parse (they are real `Output.format` members, and the tile routes use
    them) but cannot be answered here: each needs a tile's `x`/`y`/`z` and pixel size, and
    this signature has nowhere to put them. Rejected explicitly rather than left to fail
    somewhere downstream with a confusing message about sinks."""

    @pytest.mark.parametrize("format", ["png", "jpg"])
    async def test_a_tile_format_is_rejected(self, raster_data, tmp_path, format):
        pipeline = vct_pipeline(format=format)
        with pytest.raises(ShapeError) as excinfo:
            await analyze_data(pipeline, base_path=raster_data, output_path=tmp_path / f"out.{format}")
        assert "TILE" in str(excinfo.value)


class TestPipelineOnlySurface:
    async def test_a_zero_step_pipeline_returns_the_whole_input(self, base_path):
        """The smallest legal request: one input, no steps at all. `validate_pipeline`
        allows it precisely because a single input needs nothing chosen between."""
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)

        df = await analyze_data(eco_pipeline(), base_path=base_path)

        assert isinstance(df, pl.DataFrame)
        assert df.height == 32

    async def test_flat_keywords_are_rejected_by_python_itself(self):
        """`analyze_data` has no flat/keyword calling surface: no mutual-exclusivity
        check, no `**params` -- `pipeline` is the only argument, and it is required and
        positional-first. A caller still spelling a request the old way (`variable=`,
        `where=`, `group_aggregate=`, ...) gets Python's own `TypeError` for an unexpected
        keyword argument, raised before `analyze_data`'s body ever runs -- immediate and
        clearly named, not a silent no-op or a wrong answer. `cpra.mp.data.read_data`
        (dimension filters plus `grid_cell`/`where`/`sort`/`select`) is the replacement
        for the plain, unaggregated half; everything else is a pipeline body."""
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
    async def test_pipeline_calculate_op(self, two_scenario_annual_data):
        """`calculate` rewrites the running `value` column wherever its `where` matches and
        passes every other row through untouched -- never dropping data, unlike `filter`."""
        pipeline = eco_pipeline(
            {"op": "filter", "input": "eco", "where": "scenario_id = 7"},
            {"op": "calculate", "expression": "value * 0.5", "where": "calendar_year = 2030"},
            {"op": "group", "by": [], "aggregate": "sum"},
        )

        df = await analyze_data(pipeline, base_path=two_scenario_annual_data)

        # calculate halves only the calendar_year=2030 row (2030.01 -> 1015.005);
        # calendar_year=2040 (2040.01) passes through unchanged. group sums both rows for
        # grid_cell=1 across the collapsed calendar_year axis.
        cell_1 = df.filter(pl.col("grid_cell") == 1)
        assert cell_1["value"][0] == pytest.approx(2030.01 * 0.5 + 2040.01, rel=1e-5)


class TestLazy:
    async def test_lazy_true_returns_a_lazyframe_matching_the_eager_result(self, two_scenario_annual_data):
        pipeline = eco_pipeline({"op": "filter", "input": "eco", "where": "scenario_id = 7"})

        eager = await analyze_data(pipeline, base_path=two_scenario_annual_data)
        lazy = await analyze_data(pipeline, base_path=two_scenario_annual_data, lazy=True)

        assert isinstance(lazy, pl.LazyFrame)
        sort_cols = ["grid_cell", "calendar_year"]
        assert_frame_equal(lazy.collect().sort(sort_cols), eager.sort(sort_cols))

    async def test_lazy_true_with_a_tif_output_raises(self, raster_data):
        with pytest.raises(ShapeError):
            await analyze_data(vct_pipeline(), base_path=raster_data, lazy=True)
