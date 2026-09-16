"""Tests for the Polars IO-plugin raster source (api.raster_source.scan_rasters).

`scan_rasters` backs `Dataset._summarize_zone_stats_plugin` -- see
`test_dataset_raster.py`'s `TestSummarizeCalculateAndGroupAggregate` and
`TestGroupAggregateNodataExclusion` for value-level coverage of that path -- but is
exercised directly here for mechanics only a hand-built LazyFrame source has: dense
vs. zoned reads, projection pushdown, and running under Polars' streaming engine
specifically.
"""

import numpy as np
import polars as pl
import pytest
import rasterio as rio
from polars.io.plugins import register_io_source

from cpra.mp.data.analysis import sources as raster_source
from cpra.mp.data.analysis import utils as api_utils
from cpra.mp.data.analysis.sources import (
    PairedCrosswalk,
    _batched_windows,
    guarded_aggregate,
    masked_value_predicate_exprs,
    scan_broadcast,
    scan_paired,
    scan_rasters,
)
from cpra.mp.data.analysis.utils import raster_blocks_per_batch
from cpra.mp.data.grids import GRIDS
from tests.factories import RASTER_KWARGS, write_raster_crosswalk, write_raster_dataset

GRID = GRIDS[RASTER_KWARGS["grid"]]


def _gradient(offset=0):
    """Every pixel a distinct, non-negative int32 value, offset per caller.

    `raster_data` (fixed fill per file) can't exercise a pixel-level predicate at all --
    every pixel in a file agrees, so a filter either keeps everything or nothing. This is
    the fixture the predicate-pushdown tests need instead: a gradient, so a comparison
    against a threshold actually discriminates between pixels, and an `offset` per file so
    a two-band predicate (`b0` vs `b1`) discriminates differently per column too.
    """
    return (np.arange(GRID.height * GRID.width, dtype="int32") + offset).reshape(GRID.height, GRID.width)


@pytest.fixture
def varied_raster_data(base_path):
    """Two COGs (2030, 2040), each a distinct gradient -- see `_gradient`."""
    write_raster_dataset(
        base_path, years=(2030, 2040), data_fn=lambda y: _gradient(offset=(0 if y == 2030 else 1_000_000))
    )
    return base_path


@pytest.fixture
def nodata_raster_data(base_path):
    """One COG whose first 10 pixels are the nodata sentinel every `write_cog` profile
    bakes in (-9999 -- see `raster.py:_cog_profile`), planted inside a gradient that
    never legitimately goes negative, so any sentinel that leaks through a predicate is
    unambiguous rather than indistinguishable from a real value (the same ambiguity the
    real elevation sentinel has).
    """
    def data_fn(year):
        arr = _gradient()
        arr.reshape(-1)[:10] = -9999
        return arr

    write_raster_dataset(base_path, years=(2030,), data_fn=data_fn)
    return base_path


def _spy_register_io_source(monkeypatch, captured):
    """Wrap `raster_source.register_io_source` so every batch's `predicate` argument is
    recorded, while still registering the real source underneath -- so a test can observe
    what Polars decided to push down without changing what the query actually returns.
    """
    real_register = register_io_source

    def spy(source, schema):
        def wrapped(with_columns, predicate, n_rows, batch_size):
            captured.append(predicate)
            yield from source(with_columns, predicate, n_rows, batch_size)

        return real_register(wrapped, schema=schema)

    monkeypatch.setattr(raster_source, "register_io_source", spy)


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


class TestPredicatePushdownReachesTheSource:
    """`source`'s `predicate` parameter (raster_source.py, applied at lines ~353-355,
    after nodata masking) had never been exercised by any query before this test existed
    -- so it was unverified whether Polars ever pushes a filter down into this particular
    IO plugin at all, as opposed to always collecting everything and filtering
    afterwards. Both engines are checked because of exactly this kind of divergence:
    the production summarize path (`_summarize_zone_stats_plugin`) collects
    under ``engine="streaming"``, and pushdown behaviour is not guaranteed to be the same
    between engines.
    """

    @pytest.mark.parametrize("engine", ["in-memory", "streaming"])
    def test_predicate_reaches_the_source(self, raster_data, monkeypatch, engine):
        captured = []
        _spy_register_io_source(monkeypatch, captured)
        paths = _paths_by_year(raster_data)

        lf = scan_rasters([paths[0]], [[0]], [None])
        lf.filter(pl.col("b0") > 15).collect(engine=engine)

        assert captured, "the source was never called at all"
        assert any(p is not None for p in captured), (
            f"predicate never reached the source under engine={engine!r} -- "
            "every batch was collected and filtered post-hoc instead"
        )


class TestPredicateCorrectness:
    """A pushed predicate is only a placement optimization: filtering inside the source
    must return exactly the rows collecting everything and filtering afterwards would.
    Run under both engines, since a bug that only manifests when the source resumes on a
    different thread (the streaming hazard `TestStreamingEngineHazard` documents) would
    otherwise only show up in production.
    """

    @pytest.mark.parametrize("engine", ["in-memory", "streaming"])
    def test_simple_comparison(self, varied_raster_data, engine):
        paths = _paths_by_year(varied_raster_data)
        lf = scan_rasters([paths[0]], [[0]], [None])
        predicate = pl.col("b0") > 200_000

        pushed = lf.filter(predicate).collect(engine=engine).sort("b0")
        posthoc = lf.collect(engine=engine).filter(predicate).sort("b0")

        assert pushed.equals(posthoc)
        assert 0 < pushed.height < GRID.height * GRID.width

    def test_compound_and_of_two_band_comparisons(self, varied_raster_data):
        """`b0` and `b1` come from different files (`varied_raster_data` offsets the
        second file's gradient by 1,000,000), so this also exercises a predicate spanning
        columns backed by different underlying rasters."""
        paths = _paths_by_year(varied_raster_data)
        lf = scan_rasters(paths, [[0], [1]], [None, None])
        predicate = (pl.col("b0") > 100_000) & (pl.col("b1") < 1_300_000)

        pushed = lf.filter(predicate).collect().sort(["b0", "b1"])
        posthoc = lf.collect().filter(predicate).sort(["b0", "b1"])

        assert pushed.equals(posthoc)
        assert pushed.height > 0

    def test_grid_cell_filter_combined_with_band_comparison(self, varied_raster_data, zone_crosswalk_path):
        paths = _paths_by_year(varied_raster_data)
        zone_ids = np.array([1, 2], dtype="int64")
        lf = scan_rasters([paths[0]], [[0]], [None], zones_path=zone_crosswalk_path, zone_ids=zone_ids)
        predicate = (pl.col("grid_cell") == 1) & (pl.col("b0") > 100_000)

        pushed = lf.filter(predicate).collect().sort("b0")
        posthoc = lf.collect().filter(predicate).sort("b0")

        assert pushed.equals(posthoc)
        assert pushed.height > 0
        assert (pushed["grid_cell"] == 1).all()

    def test_filter_combined_with_column_projection(self, varied_raster_data):
        """A `select` of a subset of bands alongside a filter -- projection pushdown and
        predicate pushdown exercised together, not just each in isolation."""
        paths = _paths_by_year(varied_raster_data)
        lf = scan_rasters(paths, [[0], [1]], [None, None])
        predicate = pl.col("b0") > 200_000

        pushed = lf.select("b0").filter(predicate).collect().sort("b0")
        posthoc = lf.select("b0").collect().filter(predicate).sort("b0")

        assert pushed.equals(posthoc)
        assert pushed.columns == ["b0"]
        assert pushed.height > 0


class TestNodataPredicatePushdownInteraction:
    """Masking (raster_source.py's `mask_nodata` block) runs BEFORE the predicate is
    applied, so a masked pixel is null by the time any filter sees it; `null > x`
    evaluates to null, and Polars' `filter` drops null results the same way it drops
    `False` ones. If masking and pushdown were ever reordered, a nodata sentinel could
    satisfy a predicate it has no business satisfying (e.g. `-9999 > -10000` is true on
    the raw value) and silently reappear in a pushed-predicate result.
    """

    def test_nodata_pixels_never_satisfy_a_predicate_they_would_pass_unmasked(self, nodata_raster_data):
        paths = _paths_by_year(nodata_raster_data)
        lf = scan_rasters([paths[0]], [[0]], [None])
        # True of the raw -9999 sentinel; must be false (i.e. absent, not merely False)
        # once masking has turned those 10 pixels to null.
        predicate = pl.col("b0") > -10_000

        pushed = lf.filter(predicate).collect().sort("b0")
        posthoc = lf.collect().filter(predicate).sort("b0")

        assert pushed.equals(posthoc)
        assert pushed.height == GRID.height * GRID.width - 10
        assert pushed["b0"].min() >= 0


class TestValueFilterPixelCountPreservation:
    """Characterizes what `filter` means per shape at the one primitive both lowerings
    share -- the pixel-count-preservation floor the ops that depend on it are built on.

    `scan_rasters`' dense mode (`zone_ids=None`) is the position-preserving, one-row-per-
    pixel shape a raster/tile OUTPUT needs -- row order is raster-raveled pixel order, so
    a caller writing those rows back into a grid depends on every pixel still being
    present. Its zoned mode (`zone_ids` given) is the TABULAR/zonal shape production
    already uses (`Dataset._summarize_zone_stats_plugin`): one row per (pixel,
    destination zone), immediately collapsed by a `group_by`, where a dropped row only
    means one fewer contributor to its zone's aggregate -- nothing downstream depends on
    row COUNT, only on which zone each surviving row belongs to.

    A value `filter` is correct for the zonal shape applied as a row drop (that's what
    `TestPredicateCorrectness` above already establishes, and `test_zoned_mode...` below
    restates against a pixel-count framing). It is NOT correct for the raster/tile shape:
    `scan_rasters`' `predicate` parameter (sources.py:522-524) drops rows unconditionally,
    with no awareness of which shape its caller intends: the plugin cannot know which
    it is in, so the compiler must. `masked_value_predicate_exprs` is the
    mask lowering that already exists and is already used correctly, but only by the
    zonal path (`raster.py`'s `_summarize_zone_stats_plugin`, against zoned mode, where
    masking and dropping are equally valid -- it happens to use the mask form because
    it's shared with nodata handling, not because dropping would be wrong there). Nothing
    today applies that same mask lowering to the DENSE shape, and nothing stops a caller
    from reaching for `.filter()` on a dense-mode scan instead and getting a silently
    misaligned pixel grid out the other end. Gating that choice is the value-filter
    gate's job, and no production caller does this today (`scan_rasters`
    has exactly two real call sites, `raster.py`'s `_summarize_zone_stats_plugin` and
    `sources.py`'s `count_by_value`, both always zoned) -- so these tests characterize the
    primitive's mechanics as they exist now, not an aspiration wired through a compiler.
    """

    def test_mask_preserves_pixel_count_across_a_value_filter(self, varied_raster_data):
        """The correct raster/tile-output lowering: `masked_value_predicate_exprs` via
        `.with_columns()`, applied against dense mode instead of the zoned mode
        `_summarize_zone_stats_plugin` uses it against -- a raster/tile write has no
        destination zone to group by, but the same mask expression works unchanged
        against `b0` either way. Pixel count survives the filter completely; only the
        failing pixels' values change, to null: the pixel survives, its value becomes
        null.
        """
        paths = _paths_by_year(varied_raster_data)
        lf = scan_rasters([paths[0]], [[0]], [None])  # dense mode: zone_ids=None
        threshold = 200_000
        total_pixels = GRID.height * GRID.width

        raw = lf.collect()
        failing = raw.filter(pl.col("b0") <= threshold).height
        passing = raw.filter(pl.col("b0") > threshold).height
        assert 0 < failing < total_pixels, "the threshold must actually discriminate, or this proves nothing"

        masked = lf.with_columns(masked_value_predicate_exprs(["b0"], f"value > {threshold}")).collect()

        assert masked.height == total_pixels, "masking must never change the row (pixel) count"
        assert masked["b0"].null_count() == failing
        kept = masked.filter(pl.col("b0").is_not_null())
        assert kept.height == passing
        # Surviving pixels keep their original value -- masking must not rewrite
        # anything it doesn't null out.
        assert (kept["b0"] > threshold).all()

    def test_row_filter_does_not_preserve_pixel_count_in_dense_mode(self, varied_raster_data):
        """The WRONG raster/tile-output lowering, characterized rather than endorsed:
        `.filter()` on a dense-mode scan reaches `scan_rasters`' `predicate` parameter
        (sources.py:522-524) and drops the failing pixels as ROWS -- the same mechanics
        `TestPredicateCorrectness.test_simple_comparison` above already exercises on this
        exact scan, just never framed against the raster-output invariant. Restated here:
        `filtered.height` is strictly less than the full pixel count, so writing
        `filtered` back into a grid would leave holes and shift every later pixel out of
        its position -- the silent-misalignment hazard the value-filter gate exists for.
        Nothing in `scan_rasters` or Polars prevents a caller from doing this today; no
        production caller does (see class docstring), so this is a hazard this suite pins
        down at the primitive level, not a regression already reachable through a
        compiled pipeline (no pipeline routes a value filter through dense mode yet).
        """
        paths = _paths_by_year(varied_raster_data)
        lf = scan_rasters([paths[0]], [[0]], [None])
        threshold = 200_000
        total_pixels = GRID.height * GRID.width

        filtered = lf.filter(pl.col("b0") > threshold).collect()

        assert filtered.height < total_pixels, (
            "a row filter on the dense (position-preserving) shape drops pixels outright -- "
            "demonstrating the hazard, not a regression: nothing wires a raster-output query "
            "through this path yet, so this pins the primitive's mechanics "
            "down ahead of that work rather than reporting a live bug"
        )

    def test_zoned_mode_row_filter_is_the_correct_lowering_for_a_tabular_output(
        self, varied_raster_data, zone_crosswalk_path
    ):
        """The complement: for the TABULAR/zonal shape, dropping rows via `.filter()` is
        correct, not merely tolerated -- a zonal `group_by` never needs pixel COUNT
        preserved, only correct group membership, and immediately collapses whatever rows
        survive. Pins this down opposite the two tests above so that "`filter` means
        different things per shape" is asserted on both sides, not just the raster-shape
        half of it.
        """
        paths = _paths_by_year(varied_raster_data)
        zone_ids = np.array([1, 2], dtype="int64")
        lf = scan_rasters([paths[0]], [[0]], [None], zones_path=zone_crosswalk_path, zone_ids=zone_ids)
        threshold = 200_000

        unfiltered = lf.group_by("grid_cell").agg(guarded_aggregate("count", "b0")).sort("grid_cell").collect()
        filtered = (
            lf.filter(pl.col("b0") > threshold)
            .group_by("grid_cell")
            .agg(guarded_aggregate("count", "b0"))
            .sort("grid_cell")
            .collect()
        )

        # Fewer contributing pixels per zone after filtering -- a dropped row, not a
        # preserved-but-nulled one -- which is the right answer for a per-zone count: a
        # zone with no pixel satisfying the predicate should report 0, not NaN (see
        # `guarded_aggregate`'s own `count` exemption for the same reasoning).
        assert filtered["b0"].to_list() != unfiltered["b0"].to_list()
        assert sum(filtered["b0"].to_list()) < sum(unfiltered["b0"].to_list())


class TestWhichPredicateFormsPushDown:
    """Records, as an executable fact rather than a claim in a comment, which predicate
    shapes this Polars version (1.41.2 as of writing) actually pushes into the source
    versus keeps for a post-scan filter. Both are legitimate outcomes for correctness --
    a predicate Polars keeps post-scan still produces the right rows, just without the
    pushdown's memory/speed benefit (measured at 94x on a real query) -- so this class
    only pins down *which* forms currently push down, not
    that they must.
    """

    @pytest.mark.parametrize(
        "label, expr",
        [
            ("is_in", pl.col("b0").is_in([1, 2, 3])),
            ("is_null", pl.col("b0").is_null()),
            ("or_compound", (pl.col("b0") > 300_000) | (pl.col("b0") < 2)),
        ],
    )
    def test_forms_that_push_down(self, raster_data, monkeypatch, label, expr):
        """`is_in`, `is_null`, and an OR of two plain comparisons all arrive at the
        source as a non-None predicate on this Polars version -- confirmed empirically,
        not assumed."""
        captured = []
        _spy_register_io_source(monkeypatch, captured)
        paths = _paths_by_year(raster_data)
        lf = scan_rasters([paths[0]], [[0]], [None])

        lf.filter(expr).collect()

        assert any(p is not None for p in captured), f"{label} did not push down as expected"

    def test_aggregate_inside_predicate_does_not_push_down(self, raster_data, monkeypatch):
        """The one form that is NOT pushed: an aggregate inside the filter forces Polars
        to see every row before it can evaluate the comparison, so nothing survives to
        hand the source a row-level predicate -- the mechanism behind the measured 94x
        memory cliff. Confirmed empirically here (`captured` holds only `None`s) rather
        than assumed.
        """
        captured = []
        _spy_register_io_source(monkeypatch, captured)
        paths = _paths_by_year(raster_data)
        lf = scan_rasters([paths[0]], [[0]], [None])

        lf.filter(pl.col("b0") == pl.col("b0").min()).collect()

        assert captured, "the source was never called at all"
        assert all(p is None for p in captured), "an aggregate-bearing predicate unexpectedly pushed down"


class TestScanBroadcast:
    """`scan_broadcast`: the cross-grid broadcast primitive that projects a small
    coarse-grid selection onto a fine grid's pixels via two co-registered crosswalk
    rasters (see its own docstring). It currently has no API caller -- these tests
    exercise it directly (no `Dataset` involved) as the retained executor for a planned
    pipeline grid-projection operation, pinning its masking mechanics: crosswalk nodata,
    and coarse ids absent from the lookup.
    """

    @pytest.fixture
    def broadcast_crosswalk_path(self, crosswalk_path):
        """Three equal thirds by row: rows [0, H/3) -> zone 1, [H/3, 2H/3) -> nodata (0,
        `write_raster_crosswalk`'s hardcoded background value), [2H/3, H) -> zone 2 --
        the middle band exists specifically to pin crosswalk-nodata masking against a
        region that is neither zone 1 nor zone 2, distinct from "a real zone id that
        happens to be missing from the coarse frame" (`missing_id_broadcast_ids` below).
        """
        third = GRID.height // 3
        zones = np.zeros((GRID.height, GRID.width), dtype="int32")
        zones[:third, :] = 1
        zones[2 * third :, :] = 2
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
        return crosswalk_path / f"{RASTER_KWARGS['grid']}__hydro_compartment_v001.tif"

    def test_identity_case_matches_hand_computed_zone_means(self, broadcast_crosswalk_path):
        """`output_zones_path == broadcast_crosswalk_path` -- summarizing broadcast
        values back onto the same zones they came from, the identity check
        `scan_broadcast`'s own docstring calls out. zone 1's pixels should broadcast
        value 10.0; zone 2's, 20.0; the nodata middle band contributes to neither zone
        (dropped, not merely nulled -- see the row-count assertion)."""
        broadcast_ids = np.array([1, 2], dtype="int64")
        broadcast_values = np.array([10.0, 20.0], dtype="float64")
        output_zone_ids = np.array([1, 2], dtype="int64")

        lf = scan_broadcast(
            broadcast_crosswalk_path, broadcast_ids, broadcast_values, broadcast_crosswalk_path, output_zone_ids
        )
        out = lf.group_by("grid_cell").agg(guarded_aggregate("mean", "value")).sort("grid_cell").collect()

        assert out["grid_cell"].to_list() == [1, 2]
        assert out["value"].to_list() == pytest.approx([10.0, 20.0])

        # Contributing rows = zone 1's band (`third` rows) + zone 2's band (`height -
        # 2 * third` rows) -- the nodata middle band (`third` rows) is dropped entirely,
        # so the total is `height - third`, not `2 * third` (those only coincide when
        # `height` is an exact multiple of 3, which 365 is not).
        third = GRID.height // 3
        expected_rows = (GRID.height - third) * GRID.width
        raw = lf.collect()
        assert raw.height == expected_rows
        assert 0 not in raw["grid_cell"].to_list()

    def test_missing_id_pixels_are_dropped_not_defaulted(self, broadcast_crosswalk_path):
        """The coarse frame supplies a value for zone 1 only -- zone 2 is a real,
        present zone id in the crosswalk raster, but absent from `broadcast_ids`/
        `broadcast_values` (e.g. the coarse dataset had no row for that zone). Every
        zone-2 pixel must be dropped, matching the notebook's
        `np.where(ids == nodata, nan, ...)` masking treating an out-of-frame id no
        differently from nodata -- not defaulted to 0, NaN-as-a-value, or any other
        placeholder that would still contribute a (masked) row.
        """
        broadcast_ids = np.array([1], dtype="int64")
        broadcast_values = np.array([10.0], dtype="float64")
        output_zone_ids = np.array([1, 2], dtype="int64")

        lf = scan_broadcast(
            broadcast_crosswalk_path, broadcast_ids, broadcast_values, broadcast_crosswalk_path, output_zone_ids
        )
        out = lf.collect()

        assert out["grid_cell"].n_unique() == 1
        assert out["grid_cell"].unique().to_list() == [1]
        assert out["value"].to_list() == pytest.approx([10.0] * out.height)

    def test_output_zone_set_can_differ_from_the_broadcast_zone_set(self, crosswalk_path, broadcast_crosswalk_path):
        """Zonal stats of broadcast values over an entirely different zone set than the
        one that supplied the coarse value: a SEPARATE output crosswalk groups the same
        broadcast values by a different zone layout (halves, at H//2, rather than the
        broadcast crosswalk's thirds), pinned here at the primitive level, independent
        of `Dataset`.
        """
        half = GRID.height // 2
        third = GRID.height // 3
        output_zones = np.zeros((GRID.height, GRID.width), dtype="int32")
        output_zones[:half, :] = 1
        output_zones[half:, :] = 2
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "ecoregion_v001", output_zones)
        output_crosswalk_path = crosswalk_path / f"{RASTER_KWARGS['grid']}__ecoregion_v001.tif"

        broadcast_ids = np.array([1, 2], dtype="int64")
        broadcast_values = np.array([10.0, 20.0], dtype="float64")
        output_zone_ids = np.array([1, 2], dtype="int64")

        lf = scan_broadcast(
            broadcast_crosswalk_path, broadcast_ids, broadcast_values, output_crosswalk_path, output_zone_ids
        )
        out = lf.group_by("grid_cell").agg(guarded_aggregate("mean", "value")).sort("grid_cell").collect()

        # Output zone 1 (rows [0, half)) straddles the broadcast raster's zone 1 band
        # (rows [0, third)) and its nodata band (rows [third, half)) -- but the nodata
        # pixels are DROPPED, not merely nulled, so they contribute nothing to the
        # group's mean at all, leaving a pure 10.0 (every surviving pixel is a zone-1
        # broadcast pixel).
        assert out.filter(pl.col("grid_cell") == 1)["value"][0] == pytest.approx(10.0)
        # Output zone 2 (rows [half, H)) straddles the tail of the broadcast raster's
        # nodata band (rows [half, 2*third)) and its zone 2 band (rows [2*third, H)) --
        # by the same reasoning, the surviving pixels are all zone-2 broadcast pixels,
        # leaving a pure 20.0 mean.
        assert out.filter(pl.col("grid_cell") == 2)["value"][0] == pytest.approx(20.0)


def _halves_zones():
    """Top half -> 1, bottom half -> 2. Shared by `TestScanPaired`'s pairing-crosswalk
    fixtures below -- factored out so tests can recompute expected row counts from the
    exact same geometry rather than re-deriving pixel counts by hand."""
    zones = np.zeros((GRID.height, GRID.width), dtype="int32")
    half = GRID.height // 2
    zones[:half, :] = 1
    zones[half:, :] = 2
    return zones


def _thirds_zones():
    """Rows [0, H/3) -> 1, [H/3, 2H/3) -> 2, [2H/3, H) -> 99 -- the OUTPUT zone geometry,
    deliberately different from `_halves_zones`'s two-way split (mirroring the salvaged
    two-crosswalk fixture's "pairing crosswalk != output crosswalk" shape)."""
    zones = np.zeros((GRID.height, GRID.width), dtype="int32")
    third = GRID.height // 3
    zones[:third, :] = 1
    zones[third : 2 * third, :] = 2
    zones[2 * third :, :] = 99
    return zones


def _thirds_with_nodata_middle_zones():
    """Rows [0, H/3) -> 1, [H/3, 2H/3) -> 0 (background/nodata), [2H/3, H) -> 2 -- a
    pairing crosswalk with a real nodata band in the middle, distinct from "a real zone
    id simply absent from the coarse frame" (see `TestScanBroadcast`'s identical shape).
    """
    zones = np.zeros((GRID.height, GRID.width), dtype="int32")
    third = GRID.height // 3
    zones[:third, :] = 1
    zones[2 * third :, :] = 2
    return zones


class TestScanPaired:
    """`scan_paired`: the pipeline `join` operation's executor for pairing a fine
    dataset's own pixel values with a coarse dataset's per-calendar-step values, aligned
    per pixel through a PAIRING crosswalk that may differ from the OUTPUT zone
    crosswalk -- the same two-crosswalk shape as `TestScanBroadcast` above, generalized
    across a calendar axis (one coarse lookup per fine band instead of one for the
    whole call).

    Every fixture below uses two deliberately different crosswalks, mirroring the
    earlier `TestBroadcastStage` fixture: `pairing_crosswalk_path` (fine ->
    hydro_compartment_v001, a halves split) supplies the coarse VALUE; `output_
    crosswalk_path` (fine -> ecoregion_v001, a thirds split) supplies the OUTPUT zone
    set results are grouped by.
    """

    @pytest.fixture
    def pairing_crosswalk_path(self, crosswalk_path):
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", _halves_zones())
        return crosswalk_path / f"{RASTER_KWARGS['grid']}__hydro_compartment_v001.tif"

    @pytest.fixture
    def nodata_pairing_crosswalk_path(self, crosswalk_path):
        write_raster_crosswalk(
            crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", _thirds_with_nodata_middle_zones()
        )
        return crosswalk_path / f"{RASTER_KWARGS['grid']}__hydro_compartment_v001.tif"

    @pytest.fixture
    def output_crosswalk_path(self, crosswalk_path):
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "ecoregion_v001", _thirds_zones())
        return crosswalk_path / f"{RASTER_KWARGS['grid']}__ecoregion_v001.tif"

    @pytest.fixture
    def uniform_output_crosswalk_path(self, crosswalk_path):
        """A single output zone covering the whole grid -- used by tests that only care
        about the pairing side and want every pixel to survive the output-zone filter
        untouched."""
        zones = np.ones((GRID.height, GRID.width), dtype="int32")
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "ecoregion_v001", zones)
        return crosswalk_path / f"{RASTER_KWARGS['grid']}__ecoregion_v001.tif"

    def test_hand_computed_fine_and_paired_values_single_step(
        self, base_path, pairing_crosswalk_path, uniform_output_crosswalk_path
    ):
        """One calendar step, one fine band, uniform fill: `b0` must equal the fine
        fill everywhere, and `p0` must equal the pairing crosswalk's own half (100.0
        top, 200.0 bottom) -- the base case every other test in this class builds on.
        """
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 42)
        paths = _paths_by_year(base_path)
        paired_lookups = {2030: (np.array([1, 2], dtype="int64"), np.array([100.0, 200.0], dtype="float64"))}

        lf = scan_paired(
            [paths[0]],
            [[0]],
            [None],
            crosswalks=[PairedCrosswalk("p", pairing_crosswalk_path, paired_lookups)],
            calendar_steps=[2030],
            zones_path=uniform_output_crosswalk_path,
            zone_ids=np.array([1], dtype="int64"),
        )
        out = lf.collect()

        half = GRID.height // 2
        assert out.height == GRID.height * GRID.width
        assert (out["grid_cell"] == 1).all()
        assert (out["b0"] == 42).all()
        assert out["p0"].n_unique() == 2
        assert out.filter(pl.col("p0") == 100.0).height == half * GRID.width
        assert out.filter(pl.col("p0") == 200.0).height == (GRID.height - half) * GRID.width

    def test_calendar_alignment_across_two_steps(
        self, base_path, pairing_crosswalk_path, uniform_output_crosswalk_path
    ):
        """Two fine bands, two different calendar steps: `p0` (2030's band) must pull
        from the 2030 lookup and `p1` (2040's band) from the 2040 lookup -- never
        cross-mixed, even though both bands share the same pairing crosswalk read."""
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: 10 if y == 2030 else 20)
        paths = _paths_by_year(base_path)
        ids = np.array([1, 2], dtype="int64")
        paired_lookups = {
            2030: (ids, np.array([100.0, 200.0], dtype="float64")),
            2040: (ids, np.array([300.0, 400.0], dtype="float64")),
        }

        lf = scan_paired(
            paths,
            [[0], [1]],
            [None, None],
            crosswalks=[PairedCrosswalk("p", pairing_crosswalk_path, paired_lookups)],
            calendar_steps=[2030, 2040],
            zones_path=uniform_output_crosswalk_path,
            zone_ids=np.array([1], dtype="int64"),
        )
        out = lf.collect()

        half = GRID.height // 2
        assert (out["b0"] == 10).all()
        assert (out["b1"] == 20).all()
        assert out.filter(pl.col("p0") == 100.0).height == half * GRID.width
        assert out.filter(pl.col("p0") == 200.0).height == (GRID.height - half) * GRID.width
        assert out.filter(pl.col("p1") == 300.0).height == half * GRID.width
        assert out.filter(pl.col("p1") == 400.0).height == (GRID.height - half) * GRID.width

    def test_fine_nodata_yields_null_b_but_row_survives_with_paired_value(
        self, base_path, pairing_crosswalk_path, uniform_output_crosswalk_path
    ):
        """The first 10 pixels (row 0, top half) are the nodata sentinel -- `b0` must be
        null there, but the row must SURVIVE with its paired value intact, unlike a
        zone-miss drop."""

        def data_fn(_year):
            from tests.factories import raster_array

            arr = raster_array(fill=42)
            arr.reshape(-1)[:10] = -9999
            return arr

        write_raster_dataset(base_path, years=(2030,), data_fn=data_fn)
        paths = _paths_by_year(base_path)
        paired_lookups = {2030: (np.array([1, 2], dtype="int64"), np.array([100.0, 200.0], dtype="float64"))}

        lf = scan_paired(
            [paths[0]],
            [[0]],
            [None],
            crosswalks=[PairedCrosswalk("p", pairing_crosswalk_path, paired_lookups)],
            calendar_steps=[2030],
            zones_path=uniform_output_crosswalk_path,
            zone_ids=np.array([1], dtype="int64"),
        )
        out = lf.collect()

        assert out.height == GRID.height * GRID.width
        nodata_rows = out.head(10)
        assert nodata_rows["b0"].null_count() == 10
        assert nodata_rows["p0"].null_count() == 0
        assert nodata_rows["p0"].to_list() == pytest.approx([100.0] * 10)

    def test_pairing_crosswalk_nodata_yields_null_paired_but_row_survives(
        self, base_path, nodata_pairing_crosswalk_path, uniform_output_crosswalk_path
    ):
        """The pairing crosswalk's own nodata band (the middle third, background value
        0) must null `p0` there -- but must NOT drop the row, the deliberate divergence
        from `scan_broadcast`'s drop-on-miss rule."""
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 7)
        paths = _paths_by_year(base_path)
        paired_lookups = {2030: (np.array([1, 2], dtype="int64"), np.array([100.0, 200.0], dtype="float64"))}

        lf = scan_paired(
            [paths[0]],
            [[0]],
            [None],
            crosswalks=[PairedCrosswalk("p", nodata_pairing_crosswalk_path, paired_lookups)],
            calendar_steps=[2030],
            zones_path=uniform_output_crosswalk_path,
            zone_ids=np.array([1], dtype="int64"),
        )
        out = lf.collect()

        third = GRID.height // 3
        expected_middle_rows = (2 * third - third) * GRID.width

        assert out.height == GRID.height * GRID.width  # nothing dropped
        assert (out["b0"] == 7).all()  # fine value present everywhere
        assert out["p0"].null_count() == expected_middle_rows

    def test_pairing_id_absent_from_coarse_frame_yields_null_not_drop(
        self, base_path, pairing_crosswalk_path, uniform_output_crosswalk_path
    ):
        """Pairing zone 2 (bottom half) is a real id in the crosswalk raster but absent
        from `paired_lookups`' own ids -- contrast with `TestScanBroadcast`'s
        `test_missing_id_pixels_are_dropped_not_defaulted`, which DROPS the analogous
        pixel; here the row must survive with `p0` null instead."""
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 7)
        paths = _paths_by_year(base_path)
        paired_lookups = {2030: (np.array([1], dtype="int64"), np.array([100.0], dtype="float64"))}

        lf = scan_paired(
            [paths[0]],
            [[0]],
            [None],
            crosswalks=[PairedCrosswalk("p", pairing_crosswalk_path, paired_lookups)],
            calendar_steps=[2030],
            zones_path=uniform_output_crosswalk_path,
            zone_ids=np.array([1], dtype="int64"),
        )
        out = lf.collect()

        half = GRID.height // 2
        assert out.height == GRID.height * GRID.width
        assert (out["b0"] == 7).all()
        assert out.filter(pl.col("p0") == 100.0).height == half * GRID.width
        assert out.filter(pl.col("p0").is_null()).height == (GRID.height - half) * GRID.width

    def test_calendar_step_missing_entirely_yields_null_for_whole_band(
        self, base_path, pairing_crosswalk_path, uniform_output_crosswalk_path
    ):
        """`paired_lookups` has no entry at all for 2040 -- every pixel's `p1` must be
        null, while `b1` (the fine value) stays present, and 2030's `p0` is unaffected."""
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: 10 if y == 2030 else 20)
        paths = _paths_by_year(base_path)
        ids = np.array([1, 2], dtype="int64")
        paired_lookups = {2030: (ids, np.array([100.0, 200.0], dtype="float64"))}

        lf = scan_paired(
            paths,
            [[0], [1]],
            [None, None],
            crosswalks=[PairedCrosswalk("p", pairing_crosswalk_path, paired_lookups)],
            calendar_steps=[2030, 2040],
            zones_path=uniform_output_crosswalk_path,
            zone_ids=np.array([1], dtype="int64"),
        )
        out = lf.collect()

        assert out["p1"].null_count() == out.height
        assert (out["b1"] == 20).all()
        assert out["p0"].null_count() == 0

    def test_output_zone_miss_drops_the_row_entirely(
        self, base_path, pairing_crosswalk_path, output_crosswalk_path
    ):
        """Requesting `zone_ids=[1, 2]` against the thirds-split output crosswalk must
        drop every zone-99 pixel entirely -- not merely null one of its columns."""
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 7)
        paths = _paths_by_year(base_path)
        paired_lookups = {2030: (np.array([1, 2], dtype="int64"), np.array([100.0, 200.0], dtype="float64"))}

        lf = scan_paired(
            [paths[0]],
            [[0]],
            [None],
            crosswalks=[PairedCrosswalk("p", pairing_crosswalk_path, paired_lookups)],
            calendar_steps=[2030],
            zones_path=output_crosswalk_path,
            zone_ids=np.array([1, 2], dtype="int64"),
        )
        out = lf.collect()

        third = GRID.height // 3
        expected_rows = 2 * third * GRID.width
        assert out.height == expected_rows
        assert 99 not in out["grid_cell"].to_list()

    def test_projection_pushdown_skips_pairing_crosswalk_when_no_paired_column_wanted(
        self, base_path, pairing_crosswalk_path, uniform_output_crosswalk_path, monkeypatch
    ):
        # Built BEFORE the spy is installed -- `write_raster_dataset` opens its own
        # files (for writing) via `rio.open`, and those opens are not what this test is
        # about; mirrors `TestScanRasters.test_projection_pushdown_skips_unopened_files`,
        # where the equivalent setup happens through a fixture for the same reason.
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 7)
        paths = _paths_by_year(base_path)
        paired_lookups = {2030: (np.array([1, 2], dtype="int64"), np.array([100.0, 200.0], dtype="float64"))}

        opened = []
        original_open = rio.open

        def spy_open(path, *args, **kwargs):
            opened.append(str(path))
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(rio, "open", spy_open)

        lf = scan_paired(
            [paths[0]],
            [[0]],
            [None],
            crosswalks=[PairedCrosswalk("p", pairing_crosswalk_path, paired_lookups)],
            calendar_steps=[2030],
            zones_path=uniform_output_crosswalk_path,
            zone_ids=np.array([1], dtype="int64"),
        )
        lf.select("grid_cell", "b0").collect()

        assert not any(str(pairing_crosswalk_path) == p for p in opened)

    def test_projection_pushdown_skips_fine_file_when_only_paired_column_wanted(
        self, base_path, pairing_crosswalk_path, uniform_output_crosswalk_path, monkeypatch
    ):
        """Requesting `p0` alone must never open `b0`'s backing file at all -- a paired
        column's value comes entirely from the pairing crosswalk and the precomputed
        lookup array, never from the fine dataset's own pixels. The second file (`b1`),
        entirely unreferenced, is also never opened."""
        # Built BEFORE the spy is installed -- see the identical note in
        # `test_projection_pushdown_skips_pairing_crosswalk_when_no_paired_column_wanted`.
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: 10 if y == 2030 else 20)
        paths = _paths_by_year(base_path)
        ids = np.array([1, 2], dtype="int64")
        paired_lookups = {
            2030: (ids, np.array([100.0, 200.0], dtype="float64")),
            2040: (ids, np.array([300.0, 400.0], dtype="float64")),
        }

        opened = []
        original_open = rio.open

        def spy_open(path, *args, **kwargs):
            opened.append(str(path))
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(rio, "open", spy_open)

        lf = scan_paired(
            paths,
            [[0], [1]],
            [None, None],
            crosswalks=[PairedCrosswalk("p", pairing_crosswalk_path, paired_lookups)],
            calendar_steps=[2030, 2040],
            zones_path=uniform_output_crosswalk_path,
            zone_ids=np.array([1], dtype="int64"),
            # Supplied explicitly so schema construction never has to sniff a fine
            # file's dtype -- that one-time, outside-the-generator open is orthogonal
            # to per-window projection pushdown and would otherwise be indistinguishable
            # from the per-window read this test exists to rule out.
            value_dtype=pl.Int32,
        )
        out = lf.select("grid_cell", "p0").collect()

        assert not any(str(paths[0]) == p for p in opened)
        assert not any(str(paths[1]) == p for p in opened)
        assert out["p0"].n_unique() == 2

    def test_streaming_engine_matches_in_memory_engine(
        self, base_path, pairing_crosswalk_path, uniform_output_crosswalk_path, monkeypatch
    ):
        """The same GDAL/threading hazard `scan_rasters` documents applies here: forces
        more than one batch and checks the streaming engine agrees with the in-memory
        one, rather than trusting the per-`yield` re-entry discipline by inspection
        alone."""
        monkeypatch.setattr(api_utils, "RASTER_BLOCKS_PER_BATCH", 1)
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 7)
        paths = _paths_by_year(base_path)
        paired_lookups = {2030: (np.array([1, 2], dtype="int64"), np.array([100.0, 200.0], dtype="float64"))}

        lf = scan_paired(
            [paths[0]],
            [[0]],
            [None],
            crosswalks=[PairedCrosswalk("p", pairing_crosswalk_path, paired_lookups)],
            calendar_steps=[2030],
            zones_path=uniform_output_crosswalk_path,
            zone_ids=np.array([1], dtype="int64"),
        )

        in_memory = lf.collect().sort(["b0", "p0"])
        streaming = lf.collect(engine="streaming").sort(["b0", "p0"])

        assert in_memory.equals(streaming)

    def test_px_is_row_major_pixel_index_over_the_full_extent(
        self, base_path, pairing_crosswalk_path, output_crosswalk_path
    ):
        """`_px` is the row-major pixel index over the FULL raster
        extent, not window- or filtered-subset-relative -- pinned against a zone filter
        that keeps only the top third of rows, so a wrong (window-relative, or
        compacted-after-filtering) index would show up as values outside
        `[0, third * width)`."""
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 7)
        paths = _paths_by_year(base_path)
        paired_lookups = {2030: (np.array([1, 2], dtype="int64"), np.array([100.0, 200.0], dtype="float64"))}

        lf = scan_paired(
            [paths[0]],
            [[0]],
            [None],
            crosswalks=[PairedCrosswalk("p", pairing_crosswalk_path, paired_lookups)],
            calendar_steps=[2030],
            zones_path=output_crosswalk_path,
            zone_ids=np.array([1], dtype="int64"),
        )
        out = lf.collect()

        third = GRID.height // 3
        expected = {r * GRID.width + c for r in range(third) for c in range(GRID.width)}
        assert out.height == len(expected)
        assert set(out["_px"].to_list()) == expected

    def test_dense_mode_with_no_zones_path_keeps_every_pixel_and_has_no_grid_cell(
        self, base_path, pairing_crosswalk_path
    ):
        """`zones_path=None`/`zone_ids=None` -- the raster-output/write shape --
        attaches no `grid_cell` at all and drops nothing, unlike
        every other mode in this class. `_px` alone identifies each row's position,
        which is exactly what makes this shape usable to drive a write or tile."""
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 7)
        paths = _paths_by_year(base_path)
        paired_lookups = {2030: (np.array([1, 2], dtype="int64"), np.array([100.0, 200.0], dtype="float64"))}

        lf = scan_paired(
            [paths[0]],
            [[0]],
            [None],
            crosswalks=[PairedCrosswalk("p", pairing_crosswalk_path, paired_lookups)],
            calendar_steps=[2030],
        )
        out = lf.collect()

        assert "grid_cell" not in out.columns
        assert out.height == GRID.height * GRID.width
        assert set(out["_px"].to_list()) == set(range(GRID.height * GRID.width))
        half = GRID.height // 2
        assert out.filter(pl.col("p0") == 100.0).height == half * GRID.width
        assert out.filter(pl.col("p0") == 200.0).height == (GRID.height - half) * GRID.width

    def test_attach_only_mode_attaches_zone_without_filtering(
        self, base_path, pairing_crosswalk_path, output_crosswalk_path
    ):
        """`zones_path` given, `zone_ids=None` -- mirroring `scan_rasters`' own
        attach-only split -- attaches every pixel's own output
        zone id, including zone 99 (which a `zone_ids=[1]` filter would drop), without
        dropping a single row."""
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 7)
        paths = _paths_by_year(base_path)
        paired_lookups = {2030: (np.array([1, 2], dtype="int64"), np.array([100.0, 200.0], dtype="float64"))}

        lf = scan_paired(
            [paths[0]],
            [[0]],
            [None],
            crosswalks=[PairedCrosswalk("p", pairing_crosswalk_path, paired_lookups)],
            calendar_steps=[2030],
            zones_path=output_crosswalk_path,
        )
        out = lf.collect()

        third = GRID.height // 3
        assert out.height == GRID.height * GRID.width
        assert set(out["grid_cell"].unique().to_list()) == {1, 2, 99}
        assert out.filter(pl.col("grid_cell") == 99).height == (GRID.height - 2 * third) * GRID.width

    def test_two_named_crosswalks_emit_independent_columns_and_pushdown_independently(
        self, base_path, crosswalk_path, pairing_crosswalk_path, uniform_output_crosswalk_path, monkeypatch
    ):
        """`crosswalks` is a LIST -- two different named
        `PairedCrosswalk`s over two different rasters emit two independently correct
        columns (`x0`/`y0`), and requesting only one must never open the other's
        backing raster, the same projection-pushdown guarantee the single-crosswalk
        tests above pin for the fine/paired split. Names deliberately avoid the `b`
        fine-column prefix (see the reserved-name guard `scan_paired` itself raises on)."""
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "second_pairing_v001", _thirds_zones())
        second_crosswalk_path = crosswalk_path / f"{RASTER_KWARGS['grid']}__second_pairing_v001.tif"

        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 7)
        paths = _paths_by_year(base_path)
        lookups_x = {2030: (np.array([1, 2], dtype="int64"), np.array([100.0, 200.0], dtype="float64"))}
        lookups_y = {2030: (np.array([1, 2, 99], dtype="int64"), np.array([10.0, 20.0, 30.0], dtype="float64"))}

        opened = []
        original_open = rio.open

        def spy_open(path, *args, **kwargs):
            opened.append(str(path))
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(rio, "open", spy_open)

        lf = scan_paired(
            [paths[0]],
            [[0]],
            [None],
            crosswalks=[
                PairedCrosswalk("x", pairing_crosswalk_path, lookups_x),
                PairedCrosswalk("y", second_crosswalk_path, lookups_y),
            ],
            calendar_steps=[2030],
            zones_path=uniform_output_crosswalk_path,
            zone_ids=np.array([1], dtype="int64"),
        )
        lf.select("grid_cell", "y0").collect()

        assert not any(str(pairing_crosswalk_path) == p for p in opened)
        assert any(str(second_crosswalk_path) == p for p in opened)

        full = lf.collect()
        half = GRID.height // 2
        third = GRID.height // 3
        assert (full["b0"] == 7).all()
        assert full.filter(pl.col("x0") == 100.0).height == half * GRID.width
        assert full.filter(pl.col("x0") == 200.0).height == (GRID.height - half) * GRID.width
        assert full.filter(pl.col("y0") == 10.0).height == third * GRID.width
        assert full.filter(pl.col("y0") == 20.0).height == third * GRID.width
        assert full.filter(pl.col("y0") == 30.0).height == (GRID.height - 2 * third) * GRID.width

    def test_duplicate_crosswalk_names_are_rejected(self, base_path, pairing_crosswalk_path, uniform_output_crosswalk_path):
        """Two `PairedCrosswalk`s sharing a `name` would silently overwrite one
        another's column in the per-window `dict` merge -- rejected eagerly instead,
        at scan-construction time, rather than left to surface as a confusing shape
        mismatch or a quietly wrong column later."""
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 7)
        paths = _paths_by_year(base_path)
        paired_lookups = {2030: (np.array([1, 2], dtype="int64"), np.array([100.0, 200.0], dtype="float64"))}

        with pytest.raises(ValueError, match="unique"):
            scan_paired(
                [paths[0]],
                [[0]],
                [None],
                crosswalks=[
                    PairedCrosswalk("p", pairing_crosswalk_path, paired_lookups),
                    PairedCrosswalk("p", pairing_crosswalk_path, paired_lookups),
                ],
                calendar_steps=[2030],
                zones_path=uniform_output_crosswalk_path,
                zone_ids=np.array([1], dtype="int64"),
            )

    def test_crosswalk_name_colliding_with_fine_column_namespace_is_rejected(
        self, base_path, pairing_crosswalk_path, uniform_output_crosswalk_path
    ):
        """A crosswalk named `"b"` would emit `"b0"`, indistinguishable from the fine
        column `scan_rasters`'/`scan_paired`'s own `f"b{i}"` naming -- rejected eagerly
        rather than silently overwriting the fine column."""
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 7)
        paths = _paths_by_year(base_path)
        paired_lookups = {2030: (np.array([1, 2], dtype="int64"), np.array([100.0, 200.0], dtype="float64"))}

        with pytest.raises(ValueError, match="reserved"):
            scan_paired(
                [paths[0]],
                [[0]],
                [None],
                crosswalks=[PairedCrosswalk("b", pairing_crosswalk_path, paired_lookups)],
                calendar_steps=[2030],
                zones_path=uniform_output_crosswalk_path,
                zone_ids=np.array([1], dtype="int64"),
            )

    def test_zone_ids_without_zones_path_is_rejected(self, base_path, pairing_crosswalk_path):
        """Mirrors `scan_rasters`' identical guard: `zone_ids` names a set to filter
        against, and there is no crosswalk to source a zone id from without
        `zones_path`."""
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 7)
        paths = _paths_by_year(base_path)
        paired_lookups = {2030: (np.array([1, 2], dtype="int64"), np.array([100.0, 200.0], dtype="float64"))}

        with pytest.raises(ValueError, match="zones_path"):
            scan_paired(
                [paths[0]],
                [[0]],
                [None],
                crosswalks=[PairedCrosswalk("p", pairing_crosswalk_path, paired_lookups)],
                calendar_steps=[2030],
                zone_ids=np.array([1], dtype="int64"),
            )


class TestScanPairedNotebookShape:
    """The notebook exit-criterion query, at library level: mask
    ``fine > a AND paired > b``, then take the per-pixel ``min(calendar_year)`` over
    surviving steps -- a horizontal min over per-year ``when(cond).then(year)``
    columns. Hand-computed on a small fixture, expressed purely with the primitives
    `scan_paired` exposes plus ordinary Polars expressions (no new executor needed for
    the notebook's `filter`/`group` steps -- this class is evidence `scan_paired`
    alone is sufficient plumbing for the exit criterion).
    """

    def test_first_qualifying_year_matches_hand_computed_expectation(self, base_path, crosswalk_path):
        """Fine fill is uniform per year (5 in 2030, 6 in 2040) so `fine > 4` is always
        true and the surviving-year decision is driven entirely by the paired
        threshold `paired > 5`, which differs by pairing half and by year:

        - top half:    2030 paired 4.0 (fails), 2040 paired 7.0 (passes) -> year 2040
        - bottom half: 2030 paired 6.0 (passes), 2040 paired 3.0 (fails) -> year 2030

        Both years contribute the qualifying answer for a different half, which is
        what actually exercises `min_horizontal` over the per-year `when` columns
        rather than degenerating into "always pick the smaller year".
        """
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", _halves_zones())
        pairing_crosswalk_path = crosswalk_path / f"{RASTER_KWARGS['grid']}__hydro_compartment_v001.tif"
        zones = np.ones((GRID.height, GRID.width), dtype="int32")
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "ecoregion_v001", zones)
        output_crosswalk_path = crosswalk_path / f"{RASTER_KWARGS['grid']}__ecoregion_v001.tif"

        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: 5 if y == 2030 else 6)
        paths = _paths_by_year(base_path)
        ids = np.array([1, 2], dtype="int64")
        paired_lookups = {
            2030: (ids, np.array([4.0, 6.0], dtype="float64")),
            2040: (ids, np.array([7.0, 3.0], dtype="float64")),
        }

        lf = scan_paired(
            paths,
            [[0], [1]],
            [None, None],
            crosswalks=[PairedCrosswalk("p", pairing_crosswalk_path, paired_lookups)],
            calendar_steps=[2030, 2040],
            zones_path=output_crosswalk_path,
            zone_ids=np.array([1], dtype="int64"),
        )

        cond_2030 = (pl.col("b0") > 4) & (pl.col("p0") > 5)
        cond_2040 = (pl.col("b1") > 4) & (pl.col("p1") > 5)
        out = lf.with_columns(
            min_year=pl.min_horizontal(
                pl.when(cond_2030).then(2030),
                pl.when(cond_2040).then(2040),
            )
        ).collect()

        # Grouped by the pairing crosswalk's own half (identified via `p0`, whose
        # correctness is pinned independently by `TestScanPaired`) rather than by
        # raster row position, so this test does not depend on the source's row order.
        top_half = out.filter(pl.col("p0") == 4.0)
        bottom_half = out.filter(pl.col("p0") == 6.0)

        half = GRID.height // 2
        assert top_half.height == half * GRID.width
        assert bottom_half.height == (GRID.height - half) * GRID.width
        assert (top_half["min_year"] == 2040).all()
        assert (bottom_half["min_year"] == 2030).all()
