"""Tests for `_write_paired_notebook_raster`'s port onto `scan_paired` +
`collect_batches` (see `cpra/mp/data/analysis/paired.py`'s module docstring for the
port's motivation and scope).

Two things this file exists to prove, neither covered elsewhere:

* The ported write path actually EXECUTES `scan_paired` -- not merely that its output
  happens to look right, which a bug that silently fell back to some other code path
  could also produce (`TestPortUsesScanPaired`).
* The ported path's actual pixel output is IDENTICAL, byte-for-byte, to the pre-port
  hand-rolled collapse (`_accumulate_paired_window`/`_collapse_paired_window_wide`, both
  retained unmodified in `paired.py` specifically to serve as this differential test's
  oracle -- see their own docstrings) across every supported aggregate, including
  `count`'s zero-vs-nodata split and its extra out-of-domain (pairing-miss) case
  (`TestPortMatchesReferenceImplementation`).

`tests/api/test_pipeline_exec.py` already pins exact hand-computed numeric expectations
for these aggregates through the real ASGI routes (`TestPairedAccumulatorEdgeCases`,
`TestPairedAnalysisCountConsistency`) -- those values are not re-derived here, and this
file does not import anything from that one (mirroring that module's own "reproduced
here rather than imported" convention for NOTEBOOK_QUERY, so the two files stay
independently editable). This file instead proves a different, more direct thing for a
pure refactor: the new code produces exactly what the old code already produced, on the
exact same input.
"""

from contextlib import ExitStack

import numpy as np
import polars as pl
import pytest
import rasterio as rio

from cpra.mp.data import config
from cpra.mp.data.analysis import paired as paired_module
from cpra.mp.data.analysis.paired import (
    _accumulate_paired_window,
    _collapse_paired_window_wide,
)
from cpra.mp.data.analysis.pipeline import parse_pipeline
from cpra.mp.data.analysis.planner import compile_pipeline
from cpra.mp.data.analysis.sources import _batched_windows
from cpra.mp.data.analysis.utils import raster_blocks_per_batch, raster_gdal_env
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.raster import write_cog
from cpra.mp.data.tabular import write_parquet
from tests.factories import write_raster_crosswalk

# Real variable/grid names, matching test_pipeline_exec.py's NOTEBOOK_QUERY fixtures
# byte-for-byte (this is the one combination `tests/factories.py` has no ready-made
# builder for and the spec's own exit-criterion example uses) -- reproduced here rather
# than imported, see the module docstring.
FINE_KWARGS = dict(
    variable="ffibs_coverage_ratio",
    grid="veg_grid_cell_v001",
    time_unit="annual",
    model_group_id=500,
    scenario_id=7,
    ffibs_type="pl_fm",
    model="icm_v2023.0.0",
)

COARSE_KWARGS = dict(
    variable="sal",
    grid="hydro_compartment_v001",
    time_unit="annual",
    model_group_id=500,
    scenario_id=7,
    aggregate_type="2_wk_max",
    model="icm_v2023.0.0",
)


def _pipeline_query(aggregate):
    """The notebook-shape query (project/join/filter/group), varying only the trailing
    `group` step's aggregate. `count` has no aggregand (`PairedAnalysis._prepare`
    rejects a non-default `of` alongside it), so `of` is omitted entirely for it, not
    just set to something harmless.
    """
    group_step = {"op": "group", "by": [], "aggregate": aggregate}
    if aggregate != "count":
        group_step["of"] = "calendar_year"
    return {
        "inputs": {
            "marsh": {k: FINE_KWARGS[k] for k in ("variable", "grid", "time_unit", "model_group_id", "scenario_id", "ffibs_type")},
            "salinity": {
                k: COARSE_KWARGS[k]
                for k in ("variable", "grid", "time_unit", "model_group_id", "scenario_id", "aggregate_type")
            },
        },
        "steps": [
            {"op": "project", "input": "salinity", "to": "veg_grid_cell_v001", "as": "sal"},
            {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
            {"op": "filter", "where": "marsh > 0.5 AND sal > 5.5"},
            group_step,
        ],
        "output": {"format": "tif"},
    }


@pytest.fixture
def paired_port_data(base_path, crosswalk_path, monkeypatch):
    """Real fine (raster)/coarse (vector) datasets with five equal row bands, each
    isolating one thing that matters to at least one aggregate -- `sal` (zone 1) is a
    uniform 6.0 in both years (always satisfies the query's `sal > 5.5`), so every
    band's outcome below is driven purely by `marsh`/fine-nodata/pairing, mirroring
    `test_pipeline_exec.py`'s `accumulator_edge_case_data`/`count_edge_case_data`
    fixtures (reproduced here, not imported -- see the module docstring):

    * rows [0, h/5): `marsh` is 0.3 in both years -- never satisfies `marsh > 0.5`, in
      the pairing crosswalk's domain, zero contributing bands.
    * rows [h/5, 2h/5): `marsh` is `write_cog`'s own fine-nodata sentinel (-9999) in
      both years -- in-domain, zero contributing bands via fine nodata rather than a
      failed `where`.
    * rows [2h/5, 3h/5): `marsh` satisfies `where` in 2030 only (0.7 vs 0.2) -- exactly
      one contributing band.
    * rows [3h/5, 4h/5): `marsh` satisfies `where` in both years (0.6, 0.6) -- two
      contributing bands.
    * rows [4h/5, h): would satisfy `where` in both years (0.6, 0.6) exactly like the
      band above, but the crosswalk's own nodata sentinel covers this band -- OUTSIDE
      the pairing crosswalk's domain entirely, proving the exclusion comes from the
      crosswalk, not from `marsh` itself.
    """
    grid = GRIDS["veg_grid_cell_v001"]
    fifth = grid.height // 5
    marsh_2030 = np.empty((grid.height, grid.width), dtype="float32")
    marsh_2040 = np.empty((grid.height, grid.width), dtype="float32")

    marsh_2030[:fifth, :] = 0.3
    marsh_2040[:fifth, :] = 0.3
    marsh_2030[fifth : 2 * fifth, :] = -9999.0
    marsh_2040[fifth : 2 * fifth, :] = -9999.0
    marsh_2030[2 * fifth : 3 * fifth, :] = 0.7
    marsh_2040[2 * fifth : 3 * fifth, :] = 0.2
    marsh_2030[3 * fifth : 4 * fifth, :] = 0.6
    marsh_2040[3 * fifth : 4 * fifth, :] = 0.6
    marsh_2030[4 * fifth :, :] = 0.6
    marsh_2040[4 * fifth :, :] = 0.6

    for year, data in ((2030, marsh_2030), (2040, marsh_2040)):
        write_cog(data.astype("float32"), base_path=base_path, calendar_year=year, **FINE_KWARGS)

    years = (2030, 2040)
    df = pl.DataFrame(
        {
            "variable": pl.Series([COARSE_KWARGS["variable"]] * len(years), dtype=pl.String),
            "grid": pl.Series([COARSE_KWARGS["grid"]] * len(years), dtype=pl.String),
            "time_unit": pl.Series([COARSE_KWARGS["time_unit"]] * len(years), dtype=pl.String),
            "model_group_id": pl.Series([COARSE_KWARGS["model_group_id"]] * len(years), dtype=pl.Int32),
            "scenario_id": pl.Series([COARSE_KWARGS["scenario_id"]] * len(years), dtype=pl.Int32),
            "aggregate_type": pl.Series([COARSE_KWARGS["aggregate_type"]] * len(years), dtype=pl.String),
            "calendar_year": pl.Series(list(years), dtype=pl.Int32),
            "model": pl.Series([COARSE_KWARGS["model"]] * len(years), dtype=pl.String),
            "1": pl.Series([6.0, 6.0], dtype=pl.Float32),
        }
    )
    write_parquet(df, base_path=base_path, validate=False)
    monkeypatch.setattr(config, "BASE_PATH", base_path)

    zones = np.ones((grid.height, grid.width), dtype="int32")
    zones[4 * fifth :, :] = 0  # write_raster_crosswalk's own nodata sentinel
    write_raster_crosswalk(crosswalk_path, "veg_grid_cell_v001", "hydro_compartment_v001", zones)
    return base_path, fifth


def _reference_write_paired_raster(plan, out_path):
    """A verbatim copy of `_write_paired_notebook_raster`'s pre-Phase-4-step-5 body:
    the hand-rolled rasterio block-window loop dispatching directly onto
    `_accumulate_paired_window`/`_collapse_paired_window_wide` (both retained,
    unmodified, in `paired.py` for exactly this purpose), reading `fine_srcs`/
    `pairing_src` itself rather than through `scan_paired`. This is the "old" side of
    this file's differential test -- see the module docstring. Looks up
    `paired_module._STREAMING_AGGREGATES` (not a locally-bound copy) on every call so a
    test can monkeypatch it to force the wide-frame fallback, exactly as
    `test_pipeline_exec.py::TestPairedAnalysisCountConsistency.
    test_accumulator_matches_wide_fallback` does against the real routes.
    """
    with ExitStack() as stack, raster_gdal_env():
        ref_path, _ref_band = plan.band_sources[0]
        ref_src = stack.enter_context(rio.open(ref_path))
        profile = ref_src.profile.copy()
        block_height, block_width = ref_src.block_shapes[0]
        itemsize = np.dtype(plan.out_dtype).itemsize
        n_units = max(len(plan.band_sources) * 2, 1)
        blocks_per_batch = raster_blocks_per_batch(n_units, block_height, block_width, itemsize)
        window_list = list(_batched_windows(ref_src, blocks_per_batch))

        fine_srcs = {path: stack.enter_context(rio.open(path)) for path in {p for p, _ in plan.band_sources}}
        pairing_src = stack.enter_context(rio.open(str(plan.pairing_crosswalk_path)))

        profile.update(count=1, dtype=plan.out_dtype, nodata=plan.nodata_value, compress="deflate")
        dst = stack.enter_context(rio.open(str(out_path), "w", **profile))

        streaming = plan.group_step.aggregate in paired_module._STREAMING_AGGREGATES
        collapse_expr = None if streaming else plan._collapse_expr()

        for window in window_list:
            pairing_raw = pairing_src.read(1, window=window)
            shape = pairing_raw.shape
            pairing_ids = pairing_raw.ravel().astype("int64")
            pairing_nodata = pairing_src.nodata
            if pairing_nodata is not None:
                pairing_valid = pairing_ids != int(pairing_nodata)
            else:
                pairing_valid = np.ones(pairing_ids.shape, dtype=bool)

            if streaming:
                values = _accumulate_paired_window(plan, fine_srcs, window, pairing_ids, pairing_valid, shape)
            else:
                values = _collapse_paired_window_wide(
                    plan, fine_srcs, window, pairing_ids, pairing_valid, shape, collapse_expr
                )
            dst.write(values.astype(plan.out_dtype), 1, window=window)


class TestPortUsesScanPaired:
    """Proves the ported write path actually drives `scan_paired`, not merely that its
    output happens to be correct (which a bug that silently kept the old code path
    alive alongside the new one could also produce).
    """

    async def test_write_raster_calls_scan_paired(self, paired_port_data, monkeypatch, tmp_path):
        base_path, _fifth = paired_port_data
        pipeline = parse_pipeline(_pipeline_query("sum"))
        plan = compile_pipeline(pipeline, base_path=base_path)

        calls = []
        real_scan_paired = paired_module.scan_paired

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real_scan_paired(*args, **kwargs)

        monkeypatch.setattr(paired_module, "scan_paired", spy)

        out_path = tmp_path / "ported.tif"
        result = await plan.write_raster(str(out_path))
        assert result is True
        assert len(calls) == 1, "the ported write path must call scan_paired exactly once per write"

        grid = GRIDS["veg_grid_cell_v001"]
        with rio.open(out_path) as src:
            data = src.read(1)
        assert data.shape == (grid.height, grid.width)

    async def test_get_tile_does_not_call_scan_paired(self, paired_port_data, monkeypatch):
        """The tile path is explicitly NOT ported (blocked on `WarpedVRT` reprojection
        -- see the module docstring and `_paired_notebook_tile`'s own) -- confirm it
        stays that way: rendering a tile must never reach `scan_paired` either.
        """
        base_path, _fifth = paired_port_data
        pipeline = parse_pipeline(_pipeline_query("sum"))
        plan = compile_pipeline(pipeline, base_path=base_path)

        calls = []
        real_scan_paired = paired_module.scan_paired

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real_scan_paired(*args, **kwargs)

        monkeypatch.setattr(paired_module, "scan_paired", spy)

        # A tile over the grid's own bounds -- z=0 always covers the whole world, so it
        # always overlaps a real dataset regardless of grid location.
        image = await plan.get_tile(0, 0, 0, "png", tile_size=256)
        assert image is not None
        assert calls == []


class TestPortMatchesReferenceImplementation:
    """Differential test: the ported write path's actual COG output must be
    byte-identical to the pre-port hand-rolled collapse's output, on the same input,
    for every supported aggregate and for both of the old implementation's two
    branches (`_accumulate_paired_window` streaming and `_collapse_paired_window_wide`
    wide-frame fallback).
    """

    @pytest.mark.parametrize("aggregate", ["min", "max", "sum", "mean", "count"])
    @pytest.mark.parametrize("force_wide_reference", [False, True], ids=["streaming", "wide"])
    async def test_port_matches_old_implementation(
        self, paired_port_data, tmp_path, aggregate, force_wide_reference, monkeypatch
    ):
        base_path, _fifth = paired_port_data
        if force_wide_reference:
            monkeypatch.setattr(paired_module, "_STREAMING_AGGREGATES", frozenset())

        pipeline = parse_pipeline(_pipeline_query(aggregate))
        plan = compile_pipeline(pipeline, base_path=base_path)

        ported_path = tmp_path / f"ported_{aggregate}_{force_wide_reference}.tif"
        reference_path = tmp_path / f"reference_{aggregate}_{force_wide_reference}.tif"

        result = await plan.write_raster(str(ported_path))
        assert result is True
        _reference_write_paired_raster(plan, reference_path)

        with rio.open(ported_path) as src:
            ported_data = src.read(1)
            ported_nodata = src.nodata
            ported_dtype = src.dtypes[0]
        with rio.open(reference_path) as src:
            reference_data = src.read(1)
            reference_nodata = src.nodata
            reference_dtype = src.dtypes[0]

        assert ported_dtype == reference_dtype
        assert ported_nodata == reference_nodata
        assert np.array_equal(ported_data, reference_data), (
            f"ported vs reference mismatch for aggregate={aggregate!r} "
            f"force_wide_reference={force_wide_reference!r}"
        )
