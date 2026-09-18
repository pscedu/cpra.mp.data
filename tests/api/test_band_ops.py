"""Tests for `execution/band_ops.py`'s six band-wide lowerings.

Every fixture below is a `BandWideFrame` built by hand -- `pl.LazyFrame` +
`pl.DataFrame` + `ReadWindow` -- rather than read off a real raster, so each test's
expected numbers are computed by hand in the test itself, not cross-checked against a
second implementation. `_window()` builds a `ReadWindow` with sensible defaults; a test
overrides only the field its scenario cares about.
"""

import dataclasses

import polars as pl
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine

from cpra.mp.data.analysis.execution.band_ops import (
    JOIN_OPERAND_COLUMN,
    calculate_band_wide,
    filter_band_wide,
    group_band_wide,
    join_band_wide,
    mask_value_band_wide,
    sample_grid_cells_band_wide,
    select_dimension_band_wide,
)
from cpra.mp.data.analysis.execution.frame import RESERVED_BAND_INDEX_COLUMN, ReadWindow, row_major_grid_cell
from cpra.mp.data.analysis.request.ops import CalculateOp, GroupOp


def _window(**overrides):
    defaults = dict(
        source=("fixture.tif",),
        col_off=0,
        row_off=0,
        width=1,
        height=3,
        transform=Affine.identity(),
        crs=CRS.from_epsg(4326),
        decimation=1,
    )
    defaults.update(overrides)
    return ReadWindow(**defaults)


def _two_band_fixture():
    """3 pixels, 2 bands, `calendar_year` the only dimension -- the disagreeing-predicate
    fixture `mask_value_band_wide`'s test needs: at pixel 2, b0 satisfies `value > 4` and
    b1 does not, so the two bands must disagree on which of that row's values survive.
    """
    lf = pl.LazyFrame({"b0": [1, 5, 10], "b1": [2, 20, 3]})
    metadata_df = pl.DataFrame({"calendar_year": [2020, 2021]})
    return lf, metadata_df, _window()


def _three_band_fixture():
    """3 pixels, 3 bands, `regime` grouping bands 0-1 into "a" and band 2 into "b" --
    the fixture `group_band_wide`'s tests use.
    """
    lf = pl.LazyFrame(
        {
            "b0": [1, None, 3],
            "b1": [10, 20, None],
            "b2": [100, 200, 300],
        }
    )
    metadata_df = pl.DataFrame({"regime": ["a", "a", "b"]})
    return lf, metadata_df, _window()


class TestMaskValueBandWide:
    def test_masks_not_drops_and_bands_disagree(self):
        band_wide = _two_band_fixture()
        out_lf, out_metadata, out_window = mask_value_band_wide(band_wide, "value > 4", extra_columns=[])

        result = out_lf.collect()
        # Row count is untouched -- masking, never dropping.
        assert result.height == 3
        assert result["b0"].to_list() == [None, 5, 10]
        assert result["b1"].to_list() == [None, 20, None]
        # Pixel 2 is exactly the disagreement: b0's value (10) satisfies the predicate,
        # b1's value (3) does not, and both survive as a row -- neither band's outcome
        # forced the other's.
        assert result["b0"][2] == 10
        assert result["b1"][2] is None
        assert out_metadata.equals(band_wide[1])
        assert out_window == band_wide[2]

    def test_extra_columns_carried_through_and_in_value_namespace(self):
        lf = pl.LazyFrame({"b0": [1, 10], "zone": [7, 7]})
        metadata_df = pl.DataFrame({"calendar_year": [2020]})
        band_wide = (lf, metadata_df, _window(height=2))
        out_lf, _metadata, _window_out = mask_value_band_wide(
            band_wide, "value > 5 and zone = 7", extra_columns=["zone"]
        )
        result = out_lf.collect()
        assert result["b0"].to_list() == [None, 10]
        assert result["zone"].to_list() == [7, 7]


class TestSelectDimensionBandWide:
    def test_keeps_and_renumbers_matching_bands(self):
        lf = pl.LazyFrame({"b0": [1, 2], "b1": [3, 4], "b2": [5, 6]})
        metadata_df = pl.DataFrame({"calendar_year": [2020, 2021, 2022]})
        band_wide = (lf, metadata_df, _window(height=2))

        out_lf, out_metadata, out_window = select_dimension_band_wide(
            band_wide, "calendar_year > 2020", extra_columns=[]
        )
        result = out_lf.collect()

        assert out_metadata["calendar_year"].to_list() == [2021, 2022]
        assert list(result.columns) == ["b0", "b1"]
        assert result["b0"].to_list() == [3, 4]  # old b1
        assert result["b1"].to_list() == [5, 6]  # old b2
        assert out_window == band_wide[2]

    def test_no_matching_band_returns_empty_band_wide(self):
        lf = pl.LazyFrame({"b0": [1, 2]})
        metadata_df = pl.DataFrame({"calendar_year": [2020]})
        band_wide = (lf, metadata_df, _window(height=2))

        out_lf, out_metadata, out_window = select_dimension_band_wide(
            band_wide, "calendar_year > 3000", extra_columns=[]
        )

        assert out_lf.collect().is_empty()
        assert out_lf.collect_schema().names() == []
        assert out_metadata.is_empty()
        assert out_metadata.columns == []
        assert out_window == band_wide[2]

    def test_metadata_and_pixels_come_from_one_call(self):
        """A single call returns both halves -- there is no second scan or second call
        needed to reconcile which metadata row now describes which pixel column."""
        lf = pl.LazyFrame({"b0": [1, 2], "b1": [3, 4]})
        metadata_df = pl.DataFrame({"label": ["keep", "drop"]})
        band_wide = (lf, metadata_df, _window(height=2))

        out_lf, out_metadata, _window_out = select_dimension_band_wide(
            band_wide, "label = 'keep'", extra_columns=[]
        )

        assert out_metadata["label"].to_list() == ["keep"]
        assert out_lf.collect()["b0"].to_list() == [1, 2]


class TestHeightAndWindowPreserved:
    """Every band-wide-shape-preserving lowering keeps both its row count and its window
    unchanged. No lowering asserts this on itself -- `assert_height_preserved` collects
    both frames, so an inline call would be two full raster reads per op (see its
    docstring) -- which makes this class the only place the invariant is checked at all,
    for value filter, dimension filter (the non-empty case), calculate, group (the
    non-empty case), and join. The dimension filter's empty-result case and the spatial
    filter are the two lowerings this invariant does not apply to -- see
    `TestSelectDimensionBandWide.test_no_matching_band_returns_empty_band_wide` and
    `TestSampleGridCellsBandWide`.
    """

    def test_value_filter(self):
        band_wide = _two_band_fixture()
        out_lf, _metadata, window = mask_value_band_wide(band_wide, "value > 4", extra_columns=[])
        assert out_lf.collect().height == 3
        assert window == band_wide[2]

    def test_dimension_filter(self):
        lf = pl.LazyFrame({"b0": [1, 2], "b1": [3, 4]})
        metadata_df = pl.DataFrame({"calendar_year": [2020, 2021]})
        band_wide = (lf, metadata_df, _window(height=2))
        out_lf, _metadata, window = select_dimension_band_wide(band_wide, "calendar_year = 2021", extra_columns=[])
        assert out_lf.collect().height == 2
        assert window == band_wide[2]

    def test_calculate(self):
        band_wide = _two_band_fixture()
        op = CalculateOp(step_no=1, input="x", expression="value * 2", output="y")
        out_lf, _metadata, window = calculate_band_wide(band_wide, op, extra_columns=[])
        assert out_lf.collect().height == 3
        assert window == band_wide[2]

    def test_group_non_empty(self):
        band_wide = _three_band_fixture()
        op = GroupOp(step_no=1, input="x", by=["regime"], aggregate="sum", output="y")
        out_lf, _metadata, window = group_band_wide(band_wide, op, extra_columns=[])
        assert out_lf.collect().height == 3
        assert window == band_wide[2]

    def test_join(self):
        window = _window(height=3)
        a = (pl.LazyFrame({"b0": [1, 2, 3]}), pl.DataFrame({"label": ["a"]}), window)
        b = (pl.LazyFrame({"b0": [10, 20, 30]}), pl.DataFrame({"label": ["b"]}), window)
        out_lf, _metadata, out_window = join_band_wide([a, b], ["a", "b"], extra_columns=[])
        assert out_lf.collect().height == 3
        assert out_window == window


class TestSampleGridCellsBandWide:
    def test_returns_long_with_row_major_grid_cell(self):
        lf = pl.LazyFrame({"b0": [1, 2, 3], "b1": [10, 20, 30]})
        metadata_df = pl.DataFrame({"calendar_year": [2020, 2021]})
        band_wide = (lf, metadata_df, _window(height=3))

        result = sample_grid_cells_band_wide(band_wide, "grid_cell < 2", extra_columns=[]).collect()

        # 2 surviving pixels (grid_cell 0 and 1) x 2 bands = 4 long rows.
        assert result.height == 4
        assert set(result["grid_cell"].to_list()) == {0, 1}
        rows = {
            (row["grid_cell"], row["calendar_year"]): row["value"]
            for row in result.to_dicts()
        }
        assert rows[(0, 2020)] == 1
        assert rows[(0, 2021)] == 10
        assert rows[(1, 2020)] == 2
        assert rows[(1, 2021)] == 20

    def test_mixed_kind_where_applies_every_conjunct(self):
        lf = pl.LazyFrame({"b0": [1, 5], "b1": [10, 50]})
        metadata_df = pl.DataFrame({"calendar_year": [2020, 2021]})
        band_wide = (lf, metadata_df, _window(height=2))

        result = sample_grid_cells_band_wide(
            band_wide, "grid_cell = 1 and calendar_year = 2021 and value > 20", extra_columns=[]
        ).collect()

        # Only (grid_cell=1, calendar_year=2021) survives the spatial+dimension conjuncts,
        # and its value (50) survives the value conjunct too.
        assert result.height == 1
        assert result["grid_cell"].to_list() == [1]
        assert result["calendar_year"].to_list() == [2021]
        assert result["value"].to_list() == [50]


class TestFilterBandWide:
    def test_dispatches_value_only_where_to_mask(self):
        band_wide = _two_band_fixture()
        out_lf, out_metadata, out_window = filter_band_wide(band_wide, "value > 4", extra_columns=[])
        assert out_lf.collect()["b0"].to_list() == [None, 5, 10]
        assert out_metadata.equals(band_wide[1])
        assert out_window == band_wide[2]

    def test_dispatches_spatial_where_to_long(self):
        band_wide = _two_band_fixture()
        result = filter_band_wide(band_wide, "grid_cell = 0", extra_columns=[])
        collected = result.collect()
        assert collected["grid_cell"].to_list() == [0, 0]

    def test_dimension_then_value_order(self):
        lf = pl.LazyFrame({"b0": [1, 10], "b1": [2, 20], "b2": [3, 30]})
        metadata_df = pl.DataFrame({"calendar_year": [2020, 2021, 2022]})
        band_wide = (lf, metadata_df, _window(height=2))

        out_lf, out_metadata, _window_out = filter_band_wide(
            band_wide, "calendar_year > 2020 and value > 15", extra_columns=[]
        )
        result = out_lf.collect()
        # Dimension half keeps bands for 2021/2022 (old b1, b2), renumbered b0/b1; value
        # half then masks each surviving band independently.
        assert out_metadata["calendar_year"].to_list() == [2021, 2022]
        assert result["b0"].to_list() == [None, 20]
        assert result["b1"].to_list() == [None, 30]

    def test_a_carried_column_in_a_where_is_classified_as_a_dimension_and_fails_loudly(self):
        """Pins the limitation `band_ops`' module docstring names: predicate classification
        is by name, and every name that is not `value`/`grid_cell` is a dimension column,
        so a conjunct naming a carried per-pixel column is routed to the metadata side and
        raises there. Pinned because the failure is the honest one (it names the columns
        the metadata actually has) and because whoever teaches the classifier about
        carried columns should have to come here and say so.
        """
        lf = pl.LazyFrame({"b0": [1, 10], "zone": [7, 7]})
        metadata_df = pl.DataFrame({"calendar_year": [2020]})
        band_wide = (lf, metadata_df, _window(height=2))

        with pytest.raises(pl.exceptions.ColumnNotFoundError, match="zone"):
            filter_band_wide(band_wide, "zone = 7", extra_columns=["zone"])

    def test_dimension_half_empty_short_circuits_before_value_mask(self):
        lf = pl.LazyFrame({"b0": [1, 2]})
        metadata_df = pl.DataFrame({"calendar_year": [2020]})
        band_wide = (lf, metadata_df, _window(height=2))
        out_lf, out_metadata, _window_out = filter_band_wide(
            band_wide, "calendar_year > 3000 and value > 0", extra_columns=[]
        )
        assert out_lf.collect().is_empty()
        assert out_metadata.is_empty()


class TestCalculateBandWide:
    def test_applies_where_matched_bands_only_and_passes_through_others(self):
        lf = pl.LazyFrame({"b0": [1, 2], "b1": [3, 4]})
        metadata_df = pl.DataFrame({"calendar_year": [2020, 2021]})
        band_wide = (lf, metadata_df, _window(height=2))
        op = CalculateOp(
            step_no=1, input="x", expression="value * 10", where="calendar_year = 2021", output="y"
        )

        out_lf, out_metadata, _window_out = calculate_band_wide(band_wide, op, extra_columns=[])
        result = out_lf.collect()

        assert result["b0"].to_list() == [1, 2]  # band for 2020: entry does not apply, unchanged
        assert result["b1"].to_list() == [30, 40]  # band for 2021: rewritten
        assert out_metadata.equals(metadata_df)

    def test_value_where_masks_within_an_applicable_band(self):
        lf = pl.LazyFrame({"b0": [1, 10]})
        metadata_df = pl.DataFrame({"calendar_year": [2020]})
        band_wide = (lf, metadata_df, _window(height=2))
        op = CalculateOp(
            step_no=1, input="x", expression="value * 10", where="value > 5", output="y"
        )
        out_lf, _metadata, _window_out = calculate_band_wide(band_wide, op, extra_columns=[])
        result = out_lf.collect()
        assert result["b0"].to_list() == [1, 100]  # only the pixel satisfying value > 5 is rewritten

    def test_extra_columns_reachable_from_the_expression_not_only_the_mask(self):
        # The mask and the rewritten value share one namespace; a version that widened
        # only the mask's would raise here rather than produce a number.
        lf = pl.LazyFrame({"b0": [1, 2], "zone_factor": [10, 100]})
        metadata_df = pl.DataFrame({"calendar_year": [2020]})
        band_wide = (lf, metadata_df, _window(height=2))
        op = CalculateOp(step_no=1, input="x", expression="value * zone_factor", output="y")
        out_lf, _metadata, _window_out = calculate_band_wide(band_wide, op, extra_columns=["zone_factor"])
        assert out_lf.collect()["b0"].to_list() == [10, 200]

    def test_nodata_stays_nodata(self):
        lf = pl.LazyFrame({"b0": [1, None]})
        metadata_df = pl.DataFrame({"calendar_year": [2020]})
        band_wide = (lf, metadata_df, _window(height=2))
        op = CalculateOp(step_no=1, input="x", expression="value + 1", output="y")
        out_lf, _metadata, _window_out = calculate_band_wide(band_wide, op, extra_columns=[])
        assert out_lf.collect()["b0"].to_list() == [2, None]


class TestGroupBandWide:
    def test_sum_reduces_across_grouped_bands(self):
        band_wide = _three_band_fixture()
        op = GroupOp(step_no=1, input="x", by=["regime"], aggregate="sum", output="y")
        out_lf, out_metadata, _window_out = group_band_wide(band_wide, op, extra_columns=[])
        result = out_lf.collect()

        assert out_metadata["regime"].to_list() == ["a", "b"]
        # group "a" = b0 + b1 (sum_horizontal skips nulls: pixel1's b0=None contributes 0)
        assert result["b0"].to_list() == [11, 20, 3]
        # group "b" = b2 passthrough
        assert result["b1"].to_list() == [100, 200, 300]

    def test_count_counts_non_null_contributors(self):
        band_wide = _three_band_fixture()
        op = GroupOp(step_no=1, input="x", by=["regime"], aggregate="count", output="y")
        out_lf, _metadata, _window_out = group_band_wide(band_wide, op, extra_columns=[])
        result = out_lf.collect()
        # group "a" (b0, b1): pixel0 both non-null -> 2; pixel1 one null -> 1; pixel2 one null -> 1
        assert result["b0"].to_list() == [2, 1, 1]

    def test_by_empty_uses_reserved_band_index(self):
        band_wide = _three_band_fixture()
        op = GroupOp(step_no=1, input="x", by=[], aggregate="sum", output="y")
        out_lf, out_metadata, _window_out = group_band_wide(band_wide, op, extra_columns=[])
        result = out_lf.collect()

        assert out_metadata.columns == [RESERVED_BAND_INDEX_COLUMN]
        assert out_metadata[RESERVED_BAND_INDEX_COLUMN].to_list() == [0]
        assert list(result.columns) == ["b0"]
        # every band collapses into the one group: 1+10+100, None+20+200, 3+None+300
        assert result["b0"].to_list() == [111, 220, 303]

    def test_no_bands_returns_empty(self):
        window = _window(height=2)
        band_wide = (pl.LazyFrame(), pl.DataFrame(), window)
        op = GroupOp(step_no=1, input="x", by=[], aggregate="sum", output="y")
        out_lf, out_metadata, out_window = group_band_wide(band_wide, op, extra_columns=[])
        assert out_lf.collect().is_empty()
        assert out_metadata.is_empty()
        assert out_window == window


class TestJoinBandWide:
    def test_renumbers_bands_and_names_operand_provenance(self):
        window = _window(height=3)
        a = (
            pl.LazyFrame({"b0": [1, 2, 3]}),
            pl.DataFrame({"variable": ["marsh"]}),
            window,
        )
        b = (
            pl.LazyFrame({"b0": [10, 20, 30]}),
            pl.DataFrame({"variable": ["sal"]}),
            window,
        )

        out_lf, out_metadata, out_window = join_band_wide([a, b], ["a", "b"], extra_columns=[])
        result = out_lf.collect()

        assert list(result.columns) == ["b0", "b1"]
        assert result["b0"].to_list() == [1, 2, 3]
        assert result["b1"].to_list() == [10, 20, 30]
        assert out_metadata[JOIN_OPERAND_COLUMN].to_list() == ["a", "b"]
        assert out_metadata["variable"].to_list() == ["marsh", "sal"]
        assert out_window == window

    def test_extra_columns_taken_from_first_operand(self):
        window = _window(height=2)
        a = (
            pl.LazyFrame({"b0": [1, 2], "zone": [7, 8]}),
            pl.DataFrame({"variable": ["marsh"]}),
            window,
        )
        b = (
            pl.LazyFrame({"b0": [10, 20], "zone": [7, 8]}),
            pl.DataFrame({"variable": ["sal"]}),
            window,
        )
        out_lf, _metadata, _window_out = join_band_wide([a, b], ["a", "b"], extra_columns=["zone"])
        result = out_lf.collect()
        assert result["zone"].to_list() == [7, 8]

    def test_multi_band_operands_concatenate_and_renumber_globally(self):
        window = _window(height=2)
        a = (
            pl.LazyFrame({"b0": [1, 2], "b1": [3, 4]}),
            pl.DataFrame({"calendar_year": [2020, 2021]}),
            window,
        )
        b = (
            pl.LazyFrame({"b0": [100, 200]}),
            pl.DataFrame({"calendar_year": [2020]}),
            window,
        )
        out_lf, out_metadata, _window_out = join_band_wide([a, b], ["a", "b"], extra_columns=[])
        result = out_lf.collect()
        assert list(result.columns) == ["b0", "b1", "b2"]
        assert result["b2"].to_list() == [100, 200]
        assert out_metadata.height == 3
        assert out_metadata[JOIN_OPERAND_COLUMN].to_list() == ["a", "a", "b"]

    def test_operands_read_from_different_sources_join_on_a_shared_pixel_grid(self):
        """The case a real band-wide join is always in: two different rasters brought onto
        one window. Nothing but `source` differs, so the rows still line up pixel for
        pixel -- a precondition written as plain `ReadWindow` equality would reject this
        and leave the lowering satisfiable only by joining a frame to itself.
        """
        a = (
            pl.LazyFrame({"b0": [1, 2, 3]}),
            pl.DataFrame({"variable": ["marsh"]}),
            _window(source=("marsh.tif",), height=3),
        )
        b = (
            pl.LazyFrame({"b0": [10, 20, 30]}),
            pl.DataFrame({"variable": ["sal"]}),
            _window(source=("sal.tif",), height=3),
        )
        assert a[2] != b[2]

        out_lf, out_metadata, out_window = join_band_wide([a, b], ["a", "b"], extra_columns=[])
        result = out_lf.collect()

        assert result["b0"].to_list() == [1, 2, 3]
        assert result["b1"].to_list() == [10, 20, 30]
        assert out_metadata["variable"].to_list() == ["marsh", "sal"]
        assert out_window == a[2]  # operand 0's window, source and all

    def test_pixel_grid_mismatch_assertion_fires(self):
        a = (pl.LazyFrame({"b0": [1, 2, 3]}), pl.DataFrame({"label": ["a"]}), _window(height=3))
        b = (pl.LazyFrame({"b0": [10, 20]}), pl.DataFrame({"label": ["b"]}), _window(height=2))

        with pytest.raises(AssertionError):
            join_band_wide([a, b], ["a", "b"], extra_columns=[])

    def test_rows_stay_in_pixel_order_at_a_size_a_keyed_join_could_reorder(self):
        """Row position IS pixel identity for a band-wide frame, so this lowering combines
        columns positionally rather than joining on a materialized row index -- a keyed
        join's output order is not guaranteed. Sized past the point where Polars starts
        partitioning, since a small frame can come back in order by luck.
        """
        n = 200_000
        window = _window(height=n)
        a = (pl.LazyFrame({"b0": range(n)}), pl.DataFrame({"label": ["a"]}), window)
        b = (pl.LazyFrame({"b0": range(n, 2 * n)}), pl.DataFrame({"label": ["b"]}), window)

        result = join_band_wide([a, b], ["a", "b"], extra_columns=[])[0].collect()

        assert result["b0"].to_list() == list(range(n))
        # Every row still pairs pixel i of one operand with pixel i of the other.
        assert (result["b1"] - result["b0"]).unique().to_list() == [n]

    def test_requires_at_least_two_operands(self):
        a = (pl.LazyFrame({"b0": [1]}), pl.DataFrame({"label": ["a"]}), _window(height=1))
        with pytest.raises(ValueError):
            join_band_wide([a], ["a"], extra_columns=[])

    def test_requires_matching_names_and_operands(self):
        window = _window(height=1)
        a = (pl.LazyFrame({"b0": [1]}), pl.DataFrame({"label": ["a"]}), window)
        b = (pl.LazyFrame({"b0": [2]}), pl.DataFrame({"label": ["b"]}), window)
        with pytest.raises(ValueError):
            join_band_wide([a, b], ["a"], extra_columns=[])


class TestSpatialFilterCommutesWithCalculateAndGroup:
    """A spatial (`grid_cell`) conjunct, sitting after a `calculate` and a `group` in the
    same band-wide chain, is a candidate for a compile-time pushdown marker that names the
    conjunct's own step rather than one nearer the read -- but only if restricting the
    frame to one pixel BEFORE running the `calculate`/`group` in front of it would produce
    the identical answer running them over the whole window and picking that pixel out
    afterward does. `calculate_band_wide` acts one pixel at a time (nothing about its
    `pl.when(...).then(...)` reads across rows) and `group_band_wide` reduces HORIZONTALLY
    across a row's own bands (nothing about `sum_horizontal` reads across rows either) --
    so neither op has any way to make one pixel's output depend on another pixel's input,
    which is exactly the property this test checks with concrete, hand-computed numbers
    rather than assuming it from the two lowerings' docstrings alone.

    2 pixels, 2 bands, no nulls: `b0 = [10, 100]`, `b1 = [1, 5]`. `calculate` adds 1 to
    every band unconditionally (`value + 1`); `group` collapses both bands into one via an
    unweighted horizontal sum (`by=[]`, `aggregate="sum"`). Worked by hand for pixel 1
    (`grid_cell = 1`, the second row): post-calculate `b0 = 101`, `b1 = 6`; post-group
    `b0 = 101 + 6 = 107`.
    """

    def _fixture(self):
        lf = pl.LazyFrame({"b0": [10, 100], "b1": [1, 5]})
        metadata_df = pl.DataFrame({"calendar_year": [2020, 2021]})
        return lf, metadata_df, _window(height=2)

    def test_full_window_then_pick_one_pixel_matches_hand_computed_value(self):
        band_wide = self._fixture()
        calculate_op = CalculateOp(step_no=1, input="x", expression="value + 1", output="c")
        group_op = GroupOp(step_no=2, input="c", by=[], aggregate="sum", output="g")

        calculated = calculate_band_wide(band_wide, calculate_op, extra_columns=[])
        grouped = group_band_wide(calculated, group_op, extra_columns=[])
        long_result = sample_grid_cells_band_wide(grouped, "grid_cell = 1", extra_columns=[]).collect()

        assert long_result.height == 1
        assert long_result["value"].to_list() == [107]

    def test_picking_the_pixel_first_matches_the_same_hand_computed_value(self):
        """The pushdown candidate itself: restrict to `grid_cell = 1` FIRST -- as compile-
        time pushdown would, if it read only that one pixel from disk -- then run the
        identical `calculate`/`group` over just that one-row frame. No executor wires this
        up (this module composes the lowerings by hand, exactly as a real one eventually
        would), but the two results below being equal is the concrete fact that justifies
        marking the spatial conjunct's own step as pushable rather than forcing the read
        back to the chain's start.
        """
        lf, metadata_df, window = self._fixture()
        one_pixel_lf = row_major_grid_cell(lf, window).filter(pl.col("grid_cell") == 1).drop("grid_cell")
        one_pixel = (one_pixel_lf, metadata_df, dataclasses.replace(window, height=1))

        calculate_op = CalculateOp(step_no=1, input="x", expression="value + 1", output="c")
        group_op = GroupOp(step_no=2, input="c", by=[], aggregate="sum", output="g")

        calculated = calculate_band_wide(one_pixel, calculate_op, extra_columns=[])
        grouped = group_band_wide(calculated, group_op, extra_columns=[])
        result = grouped[0].collect()

        assert result["b0"].to_list() == [107]
