"""Tests for the Polars IO-plugin raster source (api.raster_source.scan_rasters).

`scan_rasters` backs `Dataset._summarize_zone_stats_plugin` -- see
`test_dataset_raster.py`'s `TestPluginDifferential` for the numpy-agreement tests -- but
is exercised directly here for mechanics only a hand-built LazyFrame source has: dense
vs. zoned reads, projection pushdown, and running under Polars' streaming engine
specifically.
"""

import numpy as np
import polars as pl
import pytest
import rasterio as rio
from polars.io.plugins import register_io_source

from cpra.mp.data.api import utils as api_utils
from cpra.mp.data.api.raster_source import _batched_windows, guarded_aggregate, scan_rasters
from cpra.mp.data.api.utils import raster_blocks_per_batch
from cpra.mp.data.grids import GRIDS

from tests.factories import RASTER_KWARGS, write_raster_crosswalk, write_raster_dataset

GRID = GRIDS[RASTER_KWARGS["grid"]]


@pytest.fixture
def raster_data(base_path):
    """Two COGs: 2030 filled with 30, 2040 filled with 40."""
    write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
    return base_path


def _paths_by_year(base_path):
    """Sort `raster_data`'s two files by calendar year, not by filename.

    `write_cog` names every file with a UUID and never with the calendar dimension (see
    its docstring: "the file itself is named with a UUID rather than by dimension
    value") -- so `sorted(base_path.rglob("*.tif"))` sorts by random UUID and, roughly
    half the time, puts the 2040 file (filled with 40) ahead of the 2030 file (filled
    with 30). Tests below assign `b0`/`b1` to `paths[0]`/`paths[1]` and assert on the
    fill values 30/40, so they need year order, not glob order -- read back from each
    file's `calendar_year=` band description, which is where `write_cog` actually
    records it.
    """
    def year_of(path):
        with rio.open(path) as src:
            return int(src.descriptions[0].split("=")[1])

    return sorted(base_path.rglob("*.tif"), key=year_of)


@pytest.fixture
def zone_crosswalk_path(crosswalk_path):
    """Split the grid into two zones by row: zone 1 on top, zone 2 below."""
    zones = np.zeros((GRID.height, GRID.width), dtype="int32")
    zones[: GRID.height // 2, :] = 1
    zones[GRID.height // 2 :, :] = 2
    write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
    return crosswalk_path / f"{RASTER_KWARGS['grid']}__hydro_compartment_v001.tif"


class TestScanRasters:
    def test_dense_mode_reads_every_pixel_in_raster_order(self, raster_data):
        """zone_ids=None is the one-row-per-pixel shape: no crosswalk read, no
        grid_cell column, every pixel in raster order."""
        paths = _paths_by_year(raster_data)
        with rio.open(paths[0]) as src:
            expected = src.read(1).ravel()

        lf = scan_rasters([paths[0]], [[0]], [None])
        out = lf.collect()

        assert out.columns == ["b0"]
        assert np.array_equal(out["b0"].to_numpy(), expected)

    def test_zoned_mode_matches_manual_grouping(self, raster_data, zone_crosswalk_path):
        paths = _paths_by_year(raster_data)
        zone_ids = np.array([1, 2], dtype="int64")

        lf = scan_rasters(paths, [[0], [1]], [None, None], zones_path=zone_crosswalk_path, zone_ids=zone_ids)
        out = (
            lf.group_by("grid_cell")
            .agg(guarded_aggregate("mean", "b0"), guarded_aggregate("mean", "b1"))
            .sort("grid_cell")
            .collect()
        )

        assert out["grid_cell"].to_list() == [1, 2]
        assert out["b0"].to_list() == pytest.approx([30.0, 30.0])
        assert out["b1"].to_list() == pytest.approx([40.0, 40.0])

    def test_projection_pushdown_skips_unopened_files(self, raster_data, zone_crosswalk_path, monkeypatch):
        """A file backing a column the query never selects is never opened."""
        paths = _paths_by_year(raster_data)
        opened = []
        original_open = rio.open

        def spy_open(path, *args, **kwargs):
            opened.append(str(path))
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(rio, "open", spy_open)
        zone_ids = np.array([1, 2], dtype="int64")

        lf = scan_rasters(paths, [[0], [1]], [None, None], zones_path=zone_crosswalk_path, zone_ids=zone_ids)
        lf.select("grid_cell", "b0").collect()

        assert not any(str(paths[1]) == p for p in opened)

    def test_streaming_engine_matches_in_memory_engine_across_many_batches(
        self, raster_data, zone_crosswalk_path, monkeypatch
    ):
        """Forces more than one batch (RASTER_BLOCKS_PER_BATCH=1 against this grid's 3
        column blocks -- see test_dataset_raster.py's TestBatchedBlockWindows for the same
        grid geometry) and collects the *same* LazyFrame under both engines.

        This is the test that exercises the hazard documented at every `yield` in
        `raster_source.source`: Polars' streaming engine may resume the generator from a
        different execution context between batches, and both `rio.Env` and rasterio
        dataset handles are thread-local. Forcing >1 batch is what makes that resume
        actually happen at least once; a single-batch collect would never touch it. This
        test can't independently prove *which* thread each batch ran on (streaming's
        threading is an internal implementation detail, not part of the public API) --
        what it proves is that the source survives the resume and produces results
        identical to the (single-context) in-memory engine, which is the only thing a
        caller ultimately depends on. `TestStreamingEngineHazard` below independently
        confirms the resume-on-a-different-context failure mode is real on this Polars
        version, by reproducing it with the naive (broken) pattern this source avoids.
        """
        monkeypatch.setattr(api_utils, "RASTER_BLOCKS_PER_BATCH", 1)
        paths = _paths_by_year(raster_data)
        zone_ids = np.array([1, 2], dtype="int64")

        lf = scan_rasters(paths, [[0], [1]], [None, None], zones_path=zone_crosswalk_path, zone_ids=zone_ids)
        grouped = (
            lf.group_by("grid_cell")
            .agg(guarded_aggregate("mean", "b0"), guarded_aggregate("mean", "b1"))
            .sort("grid_cell")
        )

        # Sanity-check that forcing RASTER_BLOCKS_PER_BATCH=1 really does yield more than
        # one batch for this grid, so this test isn't silently degenerating into the
        # single-batch case both engines would handle identically.
        with rio.open(paths[0]) as src:
            block_height, block_width = src.block_shapes[0]
            itemsize = np.dtype(src.dtypes[0]).itemsize
            blocks_per_batch = raster_blocks_per_batch(2, block_height, block_width, itemsize)
            n_batches = sum(1 for _ in _batched_windows(src, blocks_per_batch))
        assert n_batches > 1

        in_memory = grouped.collect()
        streaming = grouped.collect(engine="streaming")

        assert in_memory.equals(streaming)
        assert streaming["b0"].to_list() == pytest.approx([30.0, 30.0])
        assert streaming["b1"].to_list() == pytest.approx([40.0, 40.0])


class TestStreamingEngineHazard:
    """Documents, as an executable check, the failure mode `scan_rasters` is built
    around (see its module docstring and the comment at every `yield` in `source`):
    holding a `rio.Env` and a rasterio dataset handle open across a `yield` breaks under
    the streaming engine, because both are thread-local and Polars may resume the
    generator on a different execution context than the one that entered them.

    If a future Polars version guarantees a generator source resumes on the same thread
    it was entered from, this test starts failing -- which is exactly the signal that
    `scan_rasters`'s per-batch reopen discipline could be relaxed. Until then, this is
    the reproduction of the hazard, independent of whether `scan_rasters` itself gets it
    right (that's `test_streaming_engine_matches_in_memory_engine_across_many_batches`).
    """

    def test_holding_env_and_handle_across_yield_breaks_under_streaming(self, raster_data, monkeypatch):
        monkeypatch.setattr(api_utils, "RASTER_BLOCKS_PER_BATCH", 1)
        paths = _paths_by_year(raster_data)
        schema = pl.Schema({"b0": pl.Int32})

        def naive_source(with_columns, predicate, n_rows, batch_size):
            with rio.Env(), rio.open(paths[0]) as src:
                block_height, block_width = src.block_shapes[0]
                itemsize = np.dtype(src.dtypes[0]).itemsize
                blocks_per_batch = raster_blocks_per_batch(1, block_height, block_width, itemsize)
                for window in _batched_windows(src, blocks_per_batch):
                    yield pl.DataFrame({"b0": src.read(1, window=window).ravel()})

        lf = register_io_source(naive_source, schema=schema)
        with pytest.raises(pl.exceptions.ComputeError, match="GDAL environment"):
            lf.collect(engine="streaming")
