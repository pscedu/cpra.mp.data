"""Direct tests for the band-wide op lowerings in `analysis.raster`
(`apply_value_filter_band_wide`, `apply_dimension_filter_band_wide`,
`apply_calculate_band_wide`, `apply_group_band_wide`, plus the `band_dimension_keep`/
`band_group_row_idxs` helpers they share) and for the widened third mode of
`analysis.sources.scan_rasters` they build on top of.

These functions are the
band-wide analogue of `vector.py`'s `apply_group_by`/`apply_project`, each the re-entrant
core of an existing `RasterGridAnalysis` method pulled out to take explicit arguments
instead of `self`. `apply_value_filter_band_wide` and `band_dimension_keep` ARE wired into
`RasterGridAnalysis` (`_write_raster_windowed`'s value-filter gate and
`_band_applicability`'s delegation, respectively) but are still exercised directly here
too, at the primitive level; `apply_dimension_filter_band_wide`, `apply_calculate_band_wide`,
`band_group_row_idxs`, and `apply_group_band_wide` are NOT wired into any live request
path -- each is retained specifically because a still-blocked raster pipeline-ordering
rejection in `planner.py` needs it (see `raster.py`'s module comment block for exactly
which rejection needs which function), not merely "kept just in case" -- so these are
exercised directly, the same way `sources.scan_rasters` itself is exercised directly in
`test_raster_source.py` rather than only through `_summarize_zone_stats_plugin`.

A sixth function, `apply_project_band_wide`, used to be tested here too; it was deleted
once no rejection turned out to need it -- see the comment at the end of this file and
`raster.py`'s module comment block.
"""

import numpy as np
import polars as pl
import pytest

from cpra.mp.data.analysis.expressions import ExpressionError, compile_calculate_entries
from cpra.mp.data.analysis.raster import (
    apply_calculate_band_wide,
    apply_dimension_filter_band_wide,
    apply_group_band_wide,
    apply_value_filter_band_wide,
    band_dimension_keep,
    band_group_row_idxs,
)
from cpra.mp.data.analysis.sources import scan_rasters
from cpra.mp.data.grids import GRIDS
from tests.factories import RASTER_KWARGS, write_raster_crosswalk, write_raster_dataset

GRID = GRIDS[RASTER_KWARGS["grid"]]


def _gradient(offset=0):
    """Every pixel a distinct, non-negative int32 value -- see test_raster_source.py's
    identical helper; duplicated rather than imported since the two files intentionally
    don't share fixtures (each raster test file builds its own, per that file's own
    convention)."""
    return (np.arange(GRID.height * GRID.width, dtype="int32") + offset).reshape(GRID.height, GRID.width)


@pytest.fixture
def zone_crosswalk_path(crosswalk_path):
    """Split the grid into two zones by row: zone 1 on top, zone 2 below."""
    zones = np.zeros((GRID.height, GRID.width), dtype="int32")
    zones[: GRID.height // 2, :] = 1
    zones[GRID.height // 2 :, :] = 2
    write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
    return crosswalk_path / f"{RASTER_KWARGS['grid']}__hydro_compartment_v001.tif"


class TestScanRastersAttachWithoutFilter:
    """`scan_rasters`' third mode: `zones_path` given, `zone_ids=None` -- attach every
    pixel's own zone id without dropping any. This is what
    `apply_project_band_wide` is built to consume; the two PRE-EXISTING modes (dense,
    zoned/filtered) are covered by `test_raster_source.py::TestScanRasters` and are not
    re-litigated here beyond confirming this mode doesn't disturb them.
    """

    def test_attaches_grid_cell_without_dropping_any_pixel(self, base_path, zone_crosswalk_path):
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _: 7)
        paths = list(base_path.rglob("*.tif"))

        lf = scan_rasters(paths, [[0]], [None], zones_path=zone_crosswalk_path)
        out = lf.collect()

        assert out.columns == ["grid_cell", "b0"]
        assert out.height == GRID.height * GRID.width
        assert set(out["grid_cell"].unique().to_list()) == {1, 2}
        assert out["b0"].to_list() == [7] * out.height

    def test_grid_cell_values_match_the_crosswalk_raster_positionally(self, base_path, zone_crosswalk_path):
        """Row order out of `scan_rasters` is raster (window) order, so the attached
        `grid_cell` column must equal the crosswalk raster's own pixel values read in
        that same order -- not just "some partition into two group sizes"."""
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _: _gradient())
        paths = list(base_path.rglob("*.tif"))

        import rasterio as rio

        with rio.open(zone_crosswalk_path) as zsrc:
            expected_zones = zsrc.read(1).ravel()

        lf = scan_rasters(paths, [[0]], [None], zones_path=zone_crosswalk_path)
        out = lf.collect()

        assert np.array_equal(out["grid_cell"].to_numpy(), expected_zones)

    def test_zone_ids_without_zones_path_is_rejected(self, base_path):
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _: 1)
        paths = list(base_path.rglob("*.tif"))

        with pytest.raises(ValueError, match="zone_ids requires zones_path"):
            scan_rasters(paths, [[0]], [None], zone_ids=np.array([1], dtype="int64"))


class TestApplyValueFilterBandWide:
    """`ops.LOWERING_TABLE[("filter (value)", Shape.band_wide)]`: a per-cell MASK, never
    a row drop -- reuses `sources.masked_value_predicate_exprs` unchanged."""

    def test_masks_failing_cells_without_dropping_rows(self):
        lf = pl.LazyFrame({"b0": [1, 20, 3], "b1": [10, 2, 30]})

        out = apply_value_filter_band_wide(lf, ["b0", "b1"], "value > 5").collect()

        assert out.height == 3  # pixel count preserved -- the whole point of masking
        assert out["b0"].to_list() == [None, 20, None]
        assert out["b1"].to_list() == [10, None, 30]

    def test_falsy_where_is_a_no_op(self):
        lf = pl.LazyFrame({"b0": [1, 2, 3]})

        out = apply_value_filter_band_wide(lf, ["b0"], None).collect()

        assert out["b0"].to_list() == [1, 2, 3]

    def test_extra_columns_stay_available_to_the_predicate(self):
        """A grid_cell column already attached by an earlier `project` can be referenced
        by a value predicate scoped to it (e.g. "value > 5 and grid_cell == 1"),
        mirroring `masked_value_predicate_exprs`'s own `extra_columns` contract."""
        lf = pl.LazyFrame({"grid_cell": [1, 1, 2], "b0": [10, 10, 10]})

        out = apply_value_filter_band_wide(
            lf, ["b0"], "value > 5 and grid_cell == 1", extra_columns=["grid_cell"]
        ).collect()

        assert out["b0"].to_list() == [10, 10, None]


class TestBandDimensionKeep:
    def test_keeps_matching_rows_only(self):
        df = pl.DataFrame({"calendar_year": [2030, 2040, 2050]})

        assert band_dimension_keep(df, "calendar_year >= 2040") == [False, True, True]

    def test_falsy_where_keeps_every_band(self):
        df = pl.DataFrame({"calendar_year": [2030, 2040]})

        assert band_dimension_keep(df, None) == [True, True]
        assert band_dimension_keep(df, "") == [True, True]

    def test_unevaluable_where_raises_expression_error(self):
        df = pl.DataFrame({"calendar_year": [2030]})

        with pytest.raises(ExpressionError):
            band_dimension_keep(df, "not_a_real_column == 1")


class TestApplyDimensionFilterBandWide:
    """`ops.LOWERING_TABLE[("filter (dimension)", Shape.band_wide)]`: a COLUMN SELECTION
    (projection pushdown), since a dimension lives in column identity, not per-cell."""

    def test_keeps_only_columns_whose_band_matches(self):
        lf = pl.LazyFrame({"b0": [1, 2], "b1": [3, 4], "b2": [5, 6]})
        band_metadata = pl.DataFrame({"calendar_year": [2030, 2040, 2050]})

        out, kept = apply_dimension_filter_band_wide(
            lf, band_metadata, ["b0", "b1", "b2"], "calendar_year >= 2040"
        )

        assert kept == ["b1", "b2"]
        assert out.collect().columns == ["b1", "b2"]

    def test_extra_columns_survive_the_selection(self):
        lf = pl.LazyFrame({"grid_cell": [1, 2], "b0": [1, 2], "b1": [3, 4]})
        band_metadata = pl.DataFrame({"calendar_year": [2030, 2040]})

        out, kept = apply_dimension_filter_band_wide(
            lf, band_metadata, ["b0", "b1"], "calendar_year == 2040", extra_columns=["grid_cell"]
        )

        assert kept == ["b1"]
        assert out.collect().columns == ["grid_cell", "b1"]

    def test_falsy_where_is_a_no_op(self):
        lf = pl.LazyFrame({"b0": [1], "b1": [2]})
        band_metadata = pl.DataFrame({"calendar_year": [2030, 2040]})

        out, kept = apply_dimension_filter_band_wide(lf, band_metadata, ["b0", "b1"], None)

        assert kept == ["b0", "b1"]
        assert out.collect().columns == ["b0", "b1"]


class TestApplyCalculateBandWide:
    """`ops.LOWERING_TABLE[("calculate", Shape.band_wide)]`: `with_columns` across band
    columns -- reuses `sources.band_calculate_passes` unchanged, applying each plan only
    to the bands its dimension `where` selects."""

    def test_applies_expression_to_every_band_with_no_where(self):
        lf = pl.LazyFrame({"b0": [1, 2], "b1": [3, 4]})
        band_metadata = pl.DataFrame({"calendar_year": [2030, 2040]})
        plans = compile_calculate_entries([("calculate.1", "value * 2", None)])

        out = apply_calculate_band_wide(lf, band_metadata, ["b0", "b1"], plans).collect()

        assert out["b0"].to_list() == [2, 4]
        assert out["b1"].to_list() == [6, 8]

    def test_where_restricts_to_matching_bands(self):
        lf = pl.LazyFrame({"b0": [1, 2], "b1": [3, 4]})
        band_metadata = pl.DataFrame({"calendar_year": [2030, 2040]})
        plans = compile_calculate_entries([("calculate.1", "value * 10", "calendar_year == 2040")])

        out = apply_calculate_band_wide(lf, band_metadata, ["b0", "b1"], plans).collect()

        assert out["b0"].to_list() == [1, 2]  # calendar_year 2030 -- untouched
        assert out["b1"].to_list() == [30, 40]  # calendar_year 2040 -- rewritten

    def test_sequential_passes_compose(self):
        """Entry N+1 must see entry N's own rewrite -- `band_calculate_passes`'
        documented sequential-composition contract."""
        lf = pl.LazyFrame({"b0": [1]})
        band_metadata = pl.DataFrame({"calendar_year": [2030]})
        plans = compile_calculate_entries(
            [("calculate.1", "value + 1", None), ("calculate.2", "value * 10", None)]
        )

        out = apply_calculate_band_wide(lf, band_metadata, ["b0"], plans).collect()

        assert out["b0"].to_list() == [20]  # (1 + 1) * 10

    def test_empty_plans_is_a_no_op(self):
        lf = pl.LazyFrame({"b0": [1, 2]})
        band_metadata = pl.DataFrame({"calendar_year": [2030, 2040]})

        out = apply_calculate_band_wide(lf, band_metadata, ["b0"], []).collect()

        assert out["b0"].to_list() == [1, 2]


class TestBandGroupRowIdxs:
    def test_groups_by_column(self):
        df = pl.DataFrame({"model_group_id": [500, 500, 600]})

        groups = sorted(band_group_row_idxs(df, ["model_group_id"]))

        assert groups == [[0, 1], [2]]

    def test_empty_by_collapses_to_one_group(self):
        df = pl.DataFrame({"model_group_id": [500, 600, 700]})

        assert band_group_row_idxs(df, []) == [[0, 1, 2]]


class TestApplyGroupBandWide:
    """`ops.LOWERING_TABLE[("group", Shape.band_wide)]`: a horizontal reduce across band
    columns -- reuses `sources.band_collapse_exprs` unchanged, with `calculate_passes=[]`
    since a preceding `CalculateOp` already ran as its own op in this interpreter."""

    def test_mean_collapses_two_groups(self):
        lf = pl.LazyFrame({"b0": [10, 20], "b1": [30, 40], "b2": [1, 2]})
        # group 0 = {b0, b1}, group 1 = {b2} alone
        group_row_idxs = [[0, 1], [2]]

        out, out_names = apply_group_band_wide(lf, ["b0", "b1", "b2"], group_row_idxs, "mean", "int32")

        collected = out.collect()
        assert out_names == ["out0", "out1"]
        assert collected["out0"].to_list() == [20.0, 30.0]  # mean(10,30), mean(20,40)
        assert collected["out1"].to_list() == [1, 2]

    def test_nodata_masked_pixel_drops_out_of_the_collapse(self):
        """A band's own nodata sentinel is excluded from the group before reducing --
        the same contract `_write_raster_windowed`/`_collapse_tile_bands` depend on."""
        lf = pl.LazyFrame({"b0": [-9999, 5], "b1": [10, -9999]})

        out, out_names = apply_group_band_wide(lf, ["b0", "b1"], [[0, 1]], "sum", "int32", nodata=-9999)

        collected = out.collect()
        # row 0: only b1 contributes (10); row 1: only b0 contributes (5)
        assert collected[out_names[0]].to_list() == [10, 5]

    def test_no_contributor_falls_back_to_nodata_sentinel(self):
        out, out_names = apply_group_band_wide(
            pl.LazyFrame({"b0": [-9999], "b1": [-9999]}),
            ["b0", "b1"],
            [[0, 1]],
            "mean",
            "int32",
            nodata=-9999,
        )

        assert out.collect()[out_names[0]].to_list() == [-9999]

    def test_count_aggregate_is_rejected(self):
        """`RasterGridAnalysis._collapse_group` rejects `group_aggregate="count"` with
        this exact message before any pixel-read stage runs -- this function has no
        `self` to raise an `AnalysisError` off of, so it raises `ExpressionError`
        instead, with the identical text, rather than letting `band_collapse_exprs`
        raise a bare `KeyError` out of `HORIZONTAL_RASTER_AGGREGATES` for a caller that
        reaches this function without going through `_collapse_group` first."""
        lf = pl.LazyFrame({"b0": [10, 20], "b1": [30, 40]})

        with pytest.raises(ExpressionError, match="group_aggregate 'count' is not supported for raster grids"):
            apply_group_band_wide(lf, ["b0", "b1"], [[0, 1]], "count", "int32")


# `apply_project_band_wide` and its `TestApplyProjectBandWide` tests (formerly here) were
# DELETED -- see `raster.py`'s module comment block for the full
# reasoning: no rejection needed it, unlike the other four
# functions in that module, which are retained specifically because a still-blocked
# rejection needs each of them. The `scan_rasters` zone-attach-only scan mode
# `apply_project_band_wide` consumed is untouched and still exercised directly by
# `TestScanRastersAttachWithoutFilter`, above.
