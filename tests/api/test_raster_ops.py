"""Direct tests for the band-wide op lowerings in `analysis.raster` and for the widened
third mode of `analysis.sources.scan_rasters` they build on top of.

Every op there has the same signature -- `(band_wide_lf, metadata_df) -> (band_wide_lf,
metadata_df)` -- and the tests here exercise each one at that primitive level: the pixel
half AND the metadata half, since the metadata frame a step returns is the column
dictionary the NEXT step reads its band names out of. `RasterGridAnalysis` wires all of
them through `apply_band_ops`, which the request-level tests in `test_dataset_raster.py`,
`test_write_raster.py`, and `test_pipeline_exec.py` cover end to end; this file is the
per-primitive layer underneath, the same way `sources.scan_rasters` is exercised directly
in `test_raster_source.py` rather than only through `_summarize_zone_stats_plugin`.

A sixth function, `apply_project_band_wide`, used to be tested here too; it was deleted
unused -- see the comment at the end of this file.
"""

import numpy as np
import polars as pl
import pytest

from cpra.mp.data.analysis.expressions import ExpressionError, compile_calculate_entries
from cpra.mp.data.analysis.raster import (
    BAND_INDEX,
    apply_band_ops,
    apply_calculate_band_wide,
    apply_dimension_filter_band_wide,
    apply_group_band_wide,
    apply_value_filter_band_wide,
    band_column_names,
    band_dimension_keep,
    band_group_row_idxs,
    band_metadata_filter,
    band_metadata_group,
    band_metadata_scan_filter,
    band_ops_for_pixel_read,
    crosswalk_value_column_names,
    pivot_vector_to_band_wide,
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
    pixel's own zone id without dropping any. No live path consumes it -- the function
    written to consume it, `apply_project_band_wide`, was deleted (see the comment at the
    end of this file); the two PRE-EXISTING modes (dense, zoned/filtered) are covered by
    `test_raster_source.py::TestScanRasters` and are not re-litigated here beyond
    confirming this mode doesn't disturb them.
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


class TestBandColumnNames:
    """The band-wide invariant: column `b<i>` is the band described by metadata row `i`."""

    def test_names_one_column_per_metadata_row(self):
        assert band_column_names(pl.DataFrame({"calendar_year": [2030, 2040, 2050]})) == ["b0", "b1", "b2"]

    def test_an_empty_metadata_frame_names_no_columns(self):
        assert band_column_names(pl.DataFrame({"calendar_year": []})) == []


class TestApplyValueFilterBandWide:
    """`ops.LOWERING_TABLE[("filter (value)", Shape.band_wide)]`: a per-cell MASK, never
    a row drop -- reuses `sources.masked_value_predicate_exprs` unchanged."""

    def test_masks_failing_cells_without_dropping_rows(self):
        lf = pl.LazyFrame({"b0": [1, 20, 3], "b1": [10, 2, 30]})
        metadata = pl.DataFrame({"calendar_year": [2030, 2040]})

        out, out_metadata = apply_value_filter_band_wide(lf, metadata, "value > 5")

        collected = out.collect()
        assert collected.height == 3  # pixel count preserved -- the whole point of masking
        assert collected["b0"].to_list() == [None, 20, None]
        assert collected["b1"].to_list() == [10, None, 30]
        # A mask preserves band identity, so the column dictionary is unchanged.
        assert out_metadata.equals(metadata)

    def test_falsy_where_is_a_no_op(self):
        lf = pl.LazyFrame({"b0": [1, 2, 3]})
        metadata = pl.DataFrame({"calendar_year": [2030]})

        out, out_metadata = apply_value_filter_band_wide(lf, metadata, None)

        assert out.collect()["b0"].to_list() == [1, 2, 3]
        assert out_metadata.equals(metadata)

    def test_extra_columns_stay_available_to_the_predicate(self):
        """A grid_cell column attached by a zoned scan can be referenced by a value
        predicate scoped to it (e.g. "value > 5 and grid_cell == 1"), mirroring
        `masked_value_predicate_exprs`'s own `extra_columns` contract."""
        lf = pl.LazyFrame({"grid_cell": [1, 1, 2], "b0": [10, 10, 10]})
        metadata = pl.DataFrame({"calendar_year": [2030]})

        out, _ = apply_value_filter_band_wide(
            lf, metadata, "value > 5 and grid_cell == 1", extra_columns=["grid_cell"]
        )

        assert out.collect()["b0"].to_list() == [10, 10, None]


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


class TestBandMetadataFilter:
    """The metadata half of a dimension filter, shared by the in-flight narrowing
    `RasterGridAnalysis.apply_steps` does and the replay `apply_band_ops` does."""

    def test_narrows_rows_and_reports_which_survived(self):
        metadata = pl.DataFrame({"calendar_year": [2030, 2040, 2050]})

        out, keep = band_metadata_filter(metadata, "calendar_year >= 2040")

        assert keep == [False, True, True]
        assert out["calendar_year"].to_list() == [2040, 2050]
        # The surviving bands are renamed by their NEW positions, not their old ones.
        assert band_column_names(out) == ["b0", "b1"]


class TestBandMetadataScanFilter:
    """The other metadata-side lowering of a dimension filter: push the predicate into the
    SCAN, so the rejected bands are never opened. Legal only over a chain prefix where
    every preceding op preserves band identity; `band_metadata_filter` above takes over
    once a `group` has run."""

    def test_narrows_the_scan_lazily(self):
        lf = pl.LazyFrame({"calendar_year": [2030, 2040, 2050], "band": [1, 2, 3]})

        out = band_metadata_scan_filter(lf, "calendar_year >= 2040")

        assert out.collect()["band"].to_list() == [2, 3]

    def test_a_malformed_predicate_raises_expression_error(self):
        """Polars' SQL parser rejects a syntactically invalid `where` at `lf.sql()` itself,
        eagerly. That is retyped here so `RasterGridAnalysis.apply_steps`' single error
        boundary sees the same exception every other lowering raises, rather than a raw
        `SQLInterfaceError` escaping as a 500."""
        lf = pl.LazyFrame({"calendar_year": [2030]})

        with pytest.raises(ExpressionError):
            band_metadata_scan_filter(lf, "calendar_year >=")

    def test_an_unknown_column_is_deferred_to_collect_not_caught_here(self):
        """The limit of the retyping above, asserted rather than left to be discovered.
        `lf.sql()` only PLANS, so name resolution happens at `collect()` -- which is where
        `BaseAnalysis.materialize`'s own `ColumnNotFoundError` handler already turns it into
        the same 400, so nothing is lost by this function not catching it.

        That path is genuinely used: `planner._validate_filter_where` deliberately does not
        namespace-restrict a dimension conjunct at plan time (matching the flat `where=`
        surface, which has no such check either), so an unknown dimension name in a `filter`
        step is EXPECTED to reach execution rather than being rejected at compile time.
        """
        lf = pl.LazyFrame({"calendar_year": [2030]})

        out = band_metadata_scan_filter(lf, "not_a_real_column == 1")  # no raise here

        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            out.collect()


class TestApplyDimensionFilterBandWide:
    """`ops.LOWERING_TABLE[("filter (dimension)", Shape.band_wide)]`: a COLUMN SELECTION
    (projection pushdown), since a dimension lives in column identity, not per-cell."""

    def test_keeps_only_columns_whose_band_matches_and_renumbers_them(self):
        lf = pl.LazyFrame({"b0": [1, 2], "b1": [3, 4], "b2": [5, 6]})
        metadata = pl.DataFrame({"calendar_year": [2030, 2040, 2050]})

        out, out_metadata = apply_dimension_filter_band_wide(lf, metadata, "calendar_year >= 2040")

        collected = out.collect()
        # b1/b2 survive, renamed to b0/b1 so `band_column_names(out_metadata)` stays true
        # for whatever op runs next -- that renumbering is the invariant, not cosmetics.
        assert collected.columns == ["b0", "b1"]
        assert band_column_names(out_metadata) == ["b0", "b1"]
        assert collected["b0"].to_list() == [3, 4]
        assert collected["b1"].to_list() == [5, 6]
        assert out_metadata["calendar_year"].to_list() == [2040, 2050]

    def test_extra_columns_survive_the_selection(self):
        lf = pl.LazyFrame({"grid_cell": [1, 2], "b0": [1, 2], "b1": [3, 4]})
        metadata = pl.DataFrame({"calendar_year": [2030, 2040]})

        out, out_metadata = apply_dimension_filter_band_wide(
            lf, metadata, "calendar_year == 2040", extra_columns=["grid_cell"]
        )

        collected = out.collect()
        assert collected.columns == ["grid_cell", "b0"]
        assert collected["b0"].to_list() == [3, 4]
        assert out_metadata.height == 1

    def test_falsy_where_is_a_no_op(self):
        lf = pl.LazyFrame({"b0": [1], "b1": [2]})
        metadata = pl.DataFrame({"calendar_year": [2030, 2040]})

        out, out_metadata = apply_dimension_filter_band_wide(lf, metadata, None)

        assert out.collect().columns == ["b0", "b1"]
        assert out_metadata.equals(metadata)


class TestApplyCalculateBandWide:
    """`ops.LOWERING_TABLE[("calculate", Shape.band_wide)]`: `with_columns` across band
    columns -- reuses `sources.band_calculate_passes` unchanged, applying each plan only
    to the bands its dimension `where` selects."""

    def test_applies_expression_to_every_band_with_no_where(self):
        lf = pl.LazyFrame({"b0": [1, 2], "b1": [3, 4]})
        metadata = pl.DataFrame({"calendar_year": [2030, 2040]})
        plans = compile_calculate_entries([("calculate.1", "value * 2", None)])

        out, out_metadata = apply_calculate_band_wide(lf, metadata, plans)

        collected = out.collect()
        assert collected["b0"].to_list() == [2, 4]
        assert collected["b1"].to_list() == [6, 8]
        # A rewrite preserves band identity, so the column dictionary is unchanged.
        assert out_metadata.equals(metadata)

    def test_where_restricts_to_matching_bands(self):
        lf = pl.LazyFrame({"b0": [1, 2], "b1": [3, 4]})
        metadata = pl.DataFrame({"calendar_year": [2030, 2040]})
        plans = compile_calculate_entries([("calculate.1", "value * 10", "calendar_year == 2040")])

        out, _ = apply_calculate_band_wide(lf, metadata, plans)

        collected = out.collect()
        assert collected["b0"].to_list() == [1, 2]  # calendar_year 2030 -- untouched
        assert collected["b1"].to_list() == [30, 40]  # calendar_year 2040 -- rewritten

    def test_where_is_resolved_against_the_metadata_it_is_handed(self):
        """The same plan, the same band-wide frame, DIFFERENT metadata -- and therefore a
        different set of bands rewritten. This is the property a stored row index cannot
        have: applicability is a question about the frame this call was handed, answered
        now, not one answered once at scan time and carried forward."""
        lf = pl.LazyFrame({"b0": [1], "b1": [2]})
        plans = compile_calculate_entries([("calculate.1", "value * 10", "calendar_year == 2040")])

        first, _ = apply_calculate_band_wide(lf, pl.DataFrame({"calendar_year": [2030, 2040]}), plans)
        second, _ = apply_calculate_band_wide(lf, pl.DataFrame({"calendar_year": [2040, 2030]}), plans)

        assert first.collect().row(0) == (1, 20)
        assert second.collect().row(0) == (10, 2)

    def test_sequential_passes_compose(self):
        """Entry N+1 must see entry N's own rewrite -- `band_calculate_passes`'
        documented sequential-composition contract."""
        lf = pl.LazyFrame({"b0": [1]})
        metadata = pl.DataFrame({"calendar_year": [2030]})
        plans = compile_calculate_entries(
            [("calculate.1", "value + 1", None), ("calculate.2", "value * 10", None)]
        )

        out, _ = apply_calculate_band_wide(lf, metadata, plans)

        assert out.collect()["b0"].to_list() == [20]  # (1 + 1) * 10

    def test_empty_plans_is_a_no_op(self):
        lf = pl.LazyFrame({"b0": [1, 2]})
        metadata = pl.DataFrame({"calendar_year": [2030]})

        out, out_metadata = apply_calculate_band_wide(lf, metadata, [])

        assert out.collect()["b0"].to_list() == [1, 2]
        assert out_metadata.equals(metadata)

    def test_an_unevaluable_where_names_the_offending_entry(self):
        """The plan's own label is prefixed onto the failure, which is what lets a caller
        turn it into a 400 that names `calculate.1` rather than an anonymous error."""
        lf = pl.LazyFrame({"b0": [1]})
        metadata = pl.DataFrame({"calendar_year": [2030]})
        plans = compile_calculate_entries([("calculate.1", "value * 2", "aep = 0.2")])

        with pytest.raises(ExpressionError, match="calculate.1"):
            apply_calculate_band_wide(lf, metadata, plans)


class TestBandGroupRowIdxs:
    def test_groups_by_column(self):
        df = pl.DataFrame({"model_group_id": [500, 500, 600]})

        assert band_group_row_idxs(df, ["model_group_id"]) == [[0, 1], [2]]

    def test_group_order_is_first_occurrence_order(self):
        """Group `k` becomes output band `b<k>`, so the output band order has to be a
        property of the data rather than of Polars' hash-partition scheduling."""
        df = pl.DataFrame({"model_group_id": [600, 500, 600, 500]})

        assert band_group_row_idxs(df, ["model_group_id"]) == [[0, 2], [1, 3]]

    def test_empty_by_collapses_to_one_group(self):
        df = pl.DataFrame({"model_group_id": [500, 600, 700]})

        assert band_group_row_idxs(df, []) == [[0, 1, 2]]


class TestBandMetadataGroup:
    """The metadata half of a group, shared by `RasterGridAnalysis._collapse_group` and
    the `apply_band_ops` replay."""

    def test_keeps_one_row_per_group_and_drops_collapsed_dimensions(self):
        metadata = pl.DataFrame({"calendar_year": [2030, 2030, 2040], "model_group_id": [500, 600, 500]})

        out, group_row_idxs = band_metadata_group(metadata, ["calendar_year"])

        assert group_row_idxs == [[0, 1], [2]]
        assert out["calendar_year"].to_list() == [2030, 2040]
        # model_group_id was collapsed across, so it no longer has a single value to
        # report -- and therefore no longer has a column.
        assert "model_group_id" not in out.columns
        assert band_column_names(out) == ["b0", "b1"]

    def test_an_empty_by_still_reports_one_band(self):
        """`group_aggregate` with no `group_by` collapses away every dimension column, and
        Polars cannot represent "zero columns, N rows" -- `BAND_INDEX` is what keeps the
        resulting frame's height meaningful."""
        metadata = pl.DataFrame({"calendar_year": [2030, 2040]})

        out, group_row_idxs = band_metadata_group(metadata, [])

        assert group_row_idxs == [[0, 1]]
        assert out.columns == [BAND_INDEX]
        assert out.height == 1
        assert band_column_names(out) == ["b0"]


class TestApplyGroupBandWide:
    """`ops.LOWERING_TABLE[("group", Shape.band_wide)]`: a horizontal reduce across band
    columns, keyed by the group's own `by` columns in the metadata it is handed."""

    def test_mean_collapses_two_groups(self):
        lf = pl.LazyFrame({"b0": [10, 20], "b1": [30, 40], "b2": [1, 2]})
        # group 0 = {b0, b1} (model_group_id 500), group 1 = {b2} alone (600)
        metadata = pl.DataFrame({"model_group_id": [500, 500, 600]})

        out, out_metadata = apply_group_band_wide(lf, metadata, ["model_group_id"], "mean")

        collected = out.collect()
        assert band_column_names(out_metadata) == ["b0", "b1"]
        assert collected.columns == ["b0", "b1"]
        assert collected["b0"].to_list() == [20.0, 30.0]  # mean(10,30), mean(20,40)
        assert collected["b1"].to_list() == [1, 2]

    def test_nodata_masked_pixel_drops_out_of_the_collapse(self):
        """A null (a masked nodata sentinel, or a rejected value predicate) is excluded
        from the group before reducing -- the contract every pixel-read path depends on."""
        lf = pl.LazyFrame({"b0": [None, 5], "b1": [10, None]})
        metadata = pl.DataFrame({"calendar_year": [2030, 2040]})

        out, _ = apply_group_band_wide(lf, metadata, [], "sum")

        # row 0: only b1 contributes (10); row 1: only b0 contributes (5)
        assert out.collect()["b0"].to_list() == [10, 5]

    def test_a_group_with_no_contributor_stays_null(self):
        """Not a sentinel: a later op in the same chain has to be able to tell "nothing
        here" from a real value, so the refill happens once, at the end of
        `apply_band_ops`, never inside the group itself."""
        lf = pl.LazyFrame({"b0": [None], "b1": [None]})
        metadata = pl.DataFrame({"calendar_year": [2030, 2040]})

        out, _ = apply_group_band_wide(lf, metadata, [], "mean")

        assert out.collect()["b0"].to_list() == [None]

    def test_extra_columns_survive_the_collapse(self):
        lf = pl.LazyFrame({"grid_cell": [1, 2], "b0": [10, 20], "b1": [30, 40]})
        metadata = pl.DataFrame({"calendar_year": [2030, 2040]})

        out, _ = apply_group_band_wide(lf, metadata, [], "sum", extra_columns=["grid_cell"])

        assert out.collect().columns == ["grid_cell", "b0"]

    def test_count_aggregate_is_rejected(self):
        """`RasterGridAnalysis._collapse_group` rejects `group_aggregate="count"` with
        this exact message before any pixel-read stage runs -- this function has no
        `self` to raise an `AnalysisError` off of, so it raises `ExpressionError`
        instead, with the identical text, rather than letting a bare `KeyError` out of
        `HORIZONTAL_RASTER_AGGREGATES` reach a caller that got here without going through
        `_collapse_group` first."""
        lf = pl.LazyFrame({"b0": [10, 20], "b1": [30, 40]})
        metadata = pl.DataFrame({"calendar_year": [2030, 2040]})

        with pytest.raises(ExpressionError, match="group_aggregate 'count' is not supported for raster grids"):
            apply_group_band_wide(lf, metadata, [], "count")


class TestBandWideRowAlignment:
    """The property that makes `ops.Shape.band_wide` a sufficient type on its own.

    A band-wide frame is not merely "one column per band" -- it is POSITIONALLY ALIGNED to
    the raster: row `i` is pixel `i` of the window being read, in the dense row-major order
    `sources.scan_rasters` produces. `RasterGridAnalysis`'s pixel-read path depends on that
    directly (`_take_window_rows` raises on a short read rather than silently writing a
    misaligned band), and so does every window-at-a-time write.

    Rather than carry alignment as a second type field beside the shape -- a field whose
    only reachable value would be `True`, since every band-wide frame in this codebase
    originates from that same dense scan -- it is an invariant of the lowerings themselves:
    no band-wide lowering may add, drop, or reorder rows. This class is what pins that.
    Each lowering gets the same 5-row frame and must hand back 5 rows in the same order,
    including the cases where the op's LONG-shape counterpart would legitimately drop rows.

    There is exactly one exception, and it gets its own test below rather than being left
    implicit: a dimension filter that matches no band at all takes the height to 0 with the
    columns, because a Polars frame with no columns has no height to keep.
    """

    @staticmethod
    def _frame():
        # Five pixels, three bands. Row order is the alignment being asserted, so the
        # values are distinct per row and deliberately not sorted.
        return (
            pl.LazyFrame(
                {
                    "b0": [50, 10, 40, 20, 30],
                    "b1": [1, 2, 3, 4, 5],
                    "b2": [9, 8, 7, 6, 5],
                }
            ),
            pl.DataFrame({"calendar_year": [2030, 2040, 2050]}),
        )

    def test_a_value_filter_masks_rather_than_dropping(self):
        """The load-bearing case. In the long shape the same conjunct is a row drop; here
        a drop would shift every later pixel up by one and misplace the whole raster."""
        lf, metadata = self._frame()

        out, out_metadata = apply_value_filter_band_wide(lf, metadata, "value > 25")

        collected = out.collect()
        assert collected.height == 5
        assert collected["b0"].to_list() == [50, None, 40, None, 30]
        assert out_metadata.height == 3

    def test_a_value_filter_that_nothing_passes_still_returns_every_row(self):
        """The degenerate end of the same rule: an all-reject predicate yields five
        all-null pixels, not an empty frame. `apply_band_ops` refills those with the
        band's nodata at the very end; a zero-row frame would have nothing to refill."""
        lf, metadata = self._frame()

        out, _ = apply_value_filter_band_wide(lf, metadata, "value > 1000")

        collected = out.collect()
        assert collected.height == 5
        assert collected["b0"].to_list() == [None] * 5

    def test_a_dimension_filter_drops_columns_not_rows(self):
        lf, metadata = self._frame()

        out, out_metadata = apply_dimension_filter_band_wide(lf, metadata, "calendar_year == 2040")

        collected = out.collect()
        assert collected.height == 5
        assert band_column_names(out_metadata) == ["b0"]
        assert collected["b0"].to_list() == [1, 2, 3, 4, 5]  # the old b1, renumbered

    def test_the_one_exception_is_a_filter_that_leaves_no_bands_at_all(self):
        """The single documented exception to the invariant above, asserted so it stays
        documented. A dimension filter matching NO band leaves a frame with no band
        columns, and Polars cannot represent "zero columns, five rows" -- the height goes
        to 0 with them. (`BAND_INDEX` solves exactly this for the METADATA frame, where
        `band_metadata_group` needs a zero-dimension group to still count rows; there is no
        equivalent placeholder on the pixel side, and adding one would change the column
        dictionary every downstream read is written against.)

        It is harmless because it is unreachable as a write: a band-wide chain narrows its
        band metadata by the same predicate, and `_apply_transform_params_impl` rejects an
        empty band frame with a 404 before any pixel is read. So there is no surviving
        raster whose rows could be misaligned -- there is no raster at all.
        """
        lf, metadata = self._frame()

        out, out_metadata = apply_dimension_filter_band_wide(lf, metadata, "calendar_year == 1999")

        assert out_metadata.height == 0, "the 404-triggering condition"
        assert out.collect().height == 0

    def test_an_extra_column_holds_the_rows_even_with_no_bands(self):
        """And where a non-band column IS present -- the `extra_columns` metadata contract,
        e.g. a `grid_cell` attached by a zoned scan -- even the zero-band case keeps its
        rows, because the frame still has a column to have a height."""
        lf, metadata = self._frame()
        lf = lf.with_columns(pl.Series("grid_cell", [1, 1, 2, 2, 2]))

        out, _ = apply_dimension_filter_band_wide(
            lf, metadata, "calendar_year == 1999", extra_columns=["grid_cell"]
        )

        collected = out.collect()
        assert collected.height == 5
        assert collected.columns == ["grid_cell"]

    def test_a_calculate_rewrites_in_place(self):
        lf, metadata = self._frame()
        plans = compile_calculate_entries([("calculate.1", "value * 2", None)])

        out, _ = apply_calculate_band_wide(lf, metadata, plans)

        collected = out.collect()
        assert collected.height == 5
        assert collected["b0"].to_list() == [100, 20, 80, 40, 60]

    def test_a_group_reduces_across_columns_not_down_rows(self):
        """The op whose long-shape counterpart collapses rows outright. Horizontally it
        collapses the band AXIS instead: three columns become one, five pixels stay five.
        """
        lf, metadata = self._frame()

        out, out_metadata = apply_group_band_wide(lf, metadata, [], "sum")

        collected = out.collect()
        assert collected.height == 5
        assert band_column_names(out_metadata) == ["b0"]
        assert collected["b0"].to_list() == [60, 20, 50, 30, 40]

    def test_a_full_replay_of_all_four_preserves_the_rows(self):
        """End to end through `apply_band_ops`, the path an actual pixel read takes: the
        invariant has to hold for the composition, not just op by op."""
        lf, metadata = self._frame()
        band_ops = [
            ("calculate", compile_calculate_entries([("calculate.1", "value + 1", None)])),
            ("filter_value", "value > 25"),
            ("filter", "calendar_year <= 2040"),
            ("group", [], "mean"),
        ]

        out, out_metadata = apply_band_ops(lf.collect(), metadata, band_ops)

        assert out.height == 5
        assert band_column_names(out_metadata) == ["b0"]


class TestBandOpsForPixelRead:
    """How a standalone value `where` -- one with no declared position, because the flat
    request surface has no positions -- is folded into a recorded op list."""

    def test_a_standalone_value_where_is_prepended(self):
        assert band_ops_for_pixel_read([("group", [], "mean")], "value > 5") == [
            ("filter_value", "value > 5"),
            ("group", [], "mean"),
        ]

    def test_an_already_positioned_value_filter_is_left_where_it_was(self):
        """A pipeline request records the conjunct at its declared position AND combines
        it onto `_value_where` (which validation still reads), so prepending it again
        would apply the same mask twice, once in the wrong place."""
        band_ops = [("calculate", []), ("filter_value", "value > 5")]

        assert band_ops_for_pixel_read(band_ops, "value > 5") == band_ops

    def test_no_value_where_at_all_changes_nothing(self):
        assert band_ops_for_pixel_read([("group", [], "mean")], None) == [("group", [], "mean")]


class TestApplyBandOps:
    """The replay: every op in declared order, metadata recomputed at each step."""

    @staticmethod
    def _frame():
        # Two bands, one pixel row: b0 = calendar_year 2030, b1 = calendar_year 2040.
        return (
            pl.DataFrame({"b0": [2], "b1": [4]}),
            pl.DataFrame({"calendar_year": [2030, 2040]}),
        )

    def test_a_calculate_after_a_group_sees_the_grouped_band(self):
        """mean(x) ** 2 over bands [2, 4] is 3 ** 2 = 9. A hardcoded
        calculate-before-collapse order computes mean(x ** 2) = (4 + 16) / 2 = 10
        instead, which is why the declared position has to be honored."""
        lf, metadata = self._frame()
        band_ops = [
            ("group", [], "mean"),
            ("calculate", compile_calculate_entries([("calculate.1", "value * value", None)])),
        ]

        out, out_metadata = apply_band_ops(lf, metadata, band_ops)

        assert band_column_names(out_metadata) == ["b0"]
        assert out["b0"].to_list() == [9]

    def test_a_calculate_before_a_group_still_collapses_after_rewriting(self):
        """The same two ops in the other declared order: mean(x ** 2) = 10."""
        lf, metadata = self._frame()
        band_ops = [
            ("calculate", compile_calculate_entries([("calculate.1", "value * value", None)])),
            ("group", [], "mean"),
        ]

        out, _ = apply_band_ops(lf, metadata, band_ops)

        assert out["b0"].to_list() == [10]

    def test_a_dimension_filter_after_a_group_keys_off_the_grouped_metadata(self):
        """The filter names `calendar_year`, a column that only still exists because the
        group grouped BY it -- and it indexes the collapsed bands, not the raw ones."""
        lf = pl.DataFrame({"b0": [1], "b1": [2], "b2": [4]})
        metadata = pl.DataFrame({"calendar_year": [2030, 2040, 2040]})
        band_ops = [("group", ["calendar_year"], "sum"), ("filter", "calendar_year = 2040")]

        out, out_metadata = apply_band_ops(lf, metadata, band_ops)

        assert out.columns == ["b0"]
        assert out["b0"].to_list() == [6]  # 2 + 4, the 2040 group
        assert out_metadata["calendar_year"].to_list() == [2040]

    def test_a_second_group_composes_with_the_first(self):
        """The second group has to reduce the FIRST group's outputs. Bands
        (2030, 500) = 1, (2030, 600) = 2, (2040, 500) = 4: group by both, then by
        calendar_year alone, giving sums 3 and 4 -- not a re-derivation against the raw
        bands that would discard the first group entirely."""
        lf = pl.DataFrame({"b0": [1], "b1": [2], "b2": [4]})
        metadata = pl.DataFrame(
            {"calendar_year": [2030, 2030, 2040], "model_group_id": [500, 600, 500]}
        )
        band_ops = [
            ("group", ["calendar_year", "model_group_id"], "sum"),
            ("group", ["calendar_year"], "sum"),
        ]

        out, out_metadata = apply_band_ops(lf, metadata, band_ops)

        assert out_metadata["calendar_year"].to_list() == [2030, 2040]
        assert out["b0"].to_list() == [3]
        assert out["b1"].to_list() == [4]

    def test_a_value_filter_after_a_calculate_masks_the_rewritten_value(self):
        """Declared order decides which number the predicate tests: `value * 10` then
        `value > 25` keeps 30 (from 3) and drops 20 (from 2)."""
        lf = pl.DataFrame({"b0": [2], "b1": [3]})
        metadata = pl.DataFrame({"calendar_year": [2030, 2040]})
        band_ops = [
            ("calculate", compile_calculate_entries([("calculate.1", "value * 10", None)])),
            ("filter_value", "value > 25"),
        ]

        out, _ = apply_band_ops(lf, metadata, band_ops)

        assert out["b0"].to_list() == [None]
        assert out["b1"].to_list() == [30]

    def test_dtype_masks_nodata_first_and_restores_the_sentinel_last(self):
        lf = pl.DataFrame({"b0": [-9999, 5], "b1": [-9999, 7]})
        metadata = pl.DataFrame({"calendar_year": [2030, 2040]})

        out, _ = apply_band_ops(
            lf, metadata, [("group", [], "sum")], dtype="int32", nodata=-9999
        )

        # Row 0: every band was nodata, so nothing contributed -- the sentinel comes back.
        # Row 1: a real sum, with no sentinel anywhere near the arithmetic.
        assert out["b0"].to_list() == [-9999, 12]

    def test_no_dtype_leaves_a_no_contributor_group_null(self):
        """What the summarize path needs: a genuine null propagates into the vertical
        `guarded_aggregate` instead of a fabricated sentinel entering its denominator."""
        lf = pl.DataFrame({"b0": [None], "b1": [None]})
        metadata = pl.DataFrame({"calendar_year": [2030, 2040]})

        out, _ = apply_band_ops(lf, metadata, [("group", [], "sum")])

        assert out["b0"].to_list() == [None]

    def test_an_empty_op_list_passes_the_frame_through_untouched(self):
        lf = pl.DataFrame({"b0": [1], "b1": [2]})
        metadata = pl.DataFrame({"calendar_year": [2030, 2040]})

        out, out_metadata = apply_band_ops(lf, metadata, [])

        assert out.row(0) == (1, 2)
        assert out_metadata.equals(metadata)

    def test_an_unknown_entry_kind_is_rejected(self):
        with pytest.raises(ValueError, match="unknown band op kind"):
            apply_band_ops(pl.DataFrame({"b0": [1]}), pl.DataFrame({"calendar_year": [2030]}), [("nope",)])


# `apply_project_band_wide` and its `TestApplyProjectBandWide` tests (formerly here) were
# DELETED. The other band-wide ops in `raster.py` are each wired into `apply_band_ops`,
# which every pixel-read path replays; a band-wide `project` never joined them, because a
# summarizing `project` reaches the crosswalk through `_apply_transform_params_impl` ->
# `_summarize_zone_stats_plugin` (the zoned/filtered `scan_rasters` mode) instead. The
# zone-attach-only scan mode it would have consumed is untouched and still exercised
# directly by `TestScanRastersAttachWithoutFilter`, above.


class TestPivotVectorToBandWide:
    """`pivot_vector_to_band_wide`: the long-to-band-wide half of projecting a vector
    dataset onto a raster grid.

    The vector dataset arrives LONG -- one row per `(zone, calendar step)`. Turning that
    into one column per band is the whole job, and the reason it cannot be skipped is a
    CORRECTNESS one, not a speed one: keyed by zone alone, a long frame has no single value
    to give a pixel, and any spelling that tried would either multiply a frame's height by
    the number of steps (a join) or pick an arbitrary step (a lookup). A raster-output write
    reads rows back BY POSITION, so a height change there is a wrong answer, not a slow one.
    """

    @staticmethod
    def _metadata(n):
        return pl.DataFrame({BAND_INDEX: pl.Series(range(n), dtype=pl.UInt32)})

    @staticmethod
    def _long(rows):
        return pl.DataFrame(
            rows,
            schema={"grid_cell": pl.Int32, "_step": pl.Int32, "value": pl.Float64},
            orient="row",
        )

    def test_one_row_per_zone_not_one_per_zone_and_step(self):
        """THE gate this function exists for. Two zones x two calendar steps is FOUR long
        rows; the pivoted table must be TWO rows (one per zone) with two value columns, not
        four. Consumed positionally by a raster scan, a four-row table would pair each
        pixel with whichever step happened to be found first -- and expressed as a join
        instead, it would double the pixel count outright.
        """
        metadata_df = self._metadata(2)
        long = self._long([(1, 2020, 10.0), (1, 2050, 11.0), (2, 2020, 20.0), (2, 2050, 21.0)])
        assert long.height == 4

        wide = pivot_vector_to_band_wide(metadata_df, long, [2020, 2050], step_column="_step", prefix="v")

        assert wide.height == 2, "one row per ZONE -- not one per (zone, step)"
        assert wide.columns == ["v", "v0", "v1"]
        assert wide.sort("v").rows() == [(1, 10.0, 11.0), (2, 20.0, 21.0)]

    def test_column_names_match_the_shared_invariant(self):
        metadata_df = self._metadata(3)
        long = self._long([(1, 2020, 1.0)])
        wide = pivot_vector_to_band_wide(metadata_df, long, [2020, 2030, 2040], step_column="_step", prefix="p")
        assert wide.columns[1:] == crosswalk_value_column_names(metadata_df, "p")

    def test_two_bands_sharing_a_step_both_get_that_steps_value(self):
        """Band identity, not step identity, decides how many value columns there are: two
        bands at the same calendar step each get their own column holding the same value,
        so the scan can look each band up by its own row position with no step bookkeeping.
        """
        metadata_df = self._metadata(3)
        long = self._long([(1, 2020, 5.0), (1, 2050, 9.0)])
        wide = pivot_vector_to_band_wide(metadata_df, long, [2020, 2020, 2050], step_column="_step", prefix="v")
        assert wide.rows() == [(1, 5.0, 5.0, 9.0)]

    def test_step_with_no_vector_rows_yields_an_all_null_column(self):
        """A band whose calendar step is absent from the vector frame entirely gets a
        column of nulls -- the same outcome as a zone with no row, so a downstream miss
        needs no branch of its own."""
        metadata_df = self._metadata(2)
        long = self._long([(1, 2020, 5.0), (2, 2020, 6.0)])
        wide = pivot_vector_to_band_wide(metadata_df, long, [2020, 2050], step_column="_step", prefix="v")
        assert wide.sort("v").rows() == [(1, 5.0, None), (2, 6.0, None)]

    def test_zone_missing_from_one_step_is_null_in_that_column_alone(self):
        metadata_df = self._metadata(2)
        long = self._long([(1, 2020, 5.0), (1, 2050, 7.0), (2, 2020, 6.0)])
        wide = pivot_vector_to_band_wide(metadata_df, long, [2020, 2050], step_column="_step", prefix="v")
        assert wide.sort("v").rows() == [(1, 5.0, 7.0), (2, 6.0, None)]

    def test_duplicate_zone_ids_resolve_last_wins(self):
        """Matched to NumPy's own rule for the fancy-index assignment in
        `sources._dense_lookup_array`, which consumes this table -- deliberately identical
        so the two cannot disagree about a duplicate."""
        metadata_df = self._metadata(1)
        long = self._long([(1, 2020, 5.0), (1, 2020, 99.0)])
        wide = pivot_vector_to_band_wide(metadata_df, long, [2020], step_column="_step", prefix="v")
        assert wide.rows() == [(1, 99.0)]

    def test_rows_with_a_null_zone_or_an_unused_step_are_dropped(self):
        """Neither can ever be selected by any band, and `_dense_lookup_array` cannot index
        a null -- so they are dropped here rather than carried as unreachable rows."""
        metadata_df = self._metadata(1)
        long = self._long([(1, 2020, 5.0), (None, 2020, 6.0), (2, 1999, 7.0)])
        wide = pivot_vector_to_band_wide(metadata_df, long, [2020], step_column="_step", prefix="v")
        assert wide.rows() == [(1, 5.0)]

    def test_empty_vector_frame_yields_an_empty_but_correctly_shaped_table(self):
        """`DataFrame.pivot` on an empty frame produces no value columns at all, so the
        shape has to be built directly. An empty table is a complete answer -- every lookup
        misses and every band's column is null."""
        metadata_df = self._metadata(2)
        wide = pivot_vector_to_band_wide(metadata_df, self._long([]), [2020, 2050], step_column="_step", prefix="v")
        assert wide.height == 0
        assert wide.columns == ["v", "v0", "v1"]
        assert wide.schema["v0"] == pl.Float64

    def test_step_count_must_match_band_count(self):
        metadata_df = self._metadata(2)
        long = self._long([(1, 2020, 5.0)])
        with pytest.raises(ValueError, match="one step per band column"):
            pivot_vector_to_band_wide(metadata_df, long, [2020], step_column="_step", prefix="v")

    def test_non_numeric_step_values_work(self):
        """Steps are numbered internally and the pivot keys on that numbering, never on the
        step value itself -- so a string or date step needs no round-trippable spelling."""
        metadata_df = self._metadata(2)
        long = pl.DataFrame(
            [("a", 1, 5.0), ("b", 1, 7.0)],
            schema={"_step": pl.String, "grid_cell": pl.Int32, "value": pl.Float64},
            orient="row",
        )
        wide = pivot_vector_to_band_wide(metadata_df, long, ["a", "b"], step_column="_step", prefix="v")
        assert wide.rows() == [(1, 5.0, 7.0)]
