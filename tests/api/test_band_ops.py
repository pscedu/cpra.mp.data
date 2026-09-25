"""Tests for `execution/band_ops.py`'s band-wide lowerings.

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
    case_band_wide,
    filter_band_wide,
    group_band_wide,
    join_band_wide,
    mask_value_band_wide,
    sample_grid_cells_band_wide,
    select_dimension_band_wide,
    unify_metadata_parts,
)
from cpra.mp.data.analysis.execution.frame import (
    DOMAIN_COLUMN,
    RESERVED_BAND_INDEX_COLUMN,
    ReadWindow,
    row_major_grid_cell,
)
from cpra.mp.data.analysis.execution.long_ops import filter_long, join_long
from cpra.mp.data.analysis.request.expressions import ExpressionError
from cpra.mp.data.analysis.request.ops import CaseOp, GroupOp, JoinOp
from cpra.mp.data.analysis.request.pipeline import CaseWhen


def _window(**overrides):
    defaults = dict(
        source=("fixture.tif",),
        col_off=0,
        row_off=0,
        width=1,
        height=3,
        transform=Affine.identity(),
        crs=CRS.from_epsg(4326),
        decimation=(1, 1),
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


def _joined_fixture():
    """Two operands over one pixel grid, sharing `calendar_year` and disagreeing about
    `aggregate_type` -- the shape `lnd-type-plus` actually has, where `stg` declares
    `aggregate_type` and the rasters it joins against do not.

    `elev` has a 2045 band `stg` has no counterpart for, so the fixture also covers the
    unpaired band `join_band_wide` keeps and `join_long` would not.
    """
    window = _window(height=2)
    elev = (
        pl.LazyFrame({"b0": [10, 11], "b1": [20, 21], "b2": [30, 31]}),
        pl.DataFrame({"variable": ["elev"] * 3, "calendar_year": [2025, 2035, 2045]}),
        window,
    )
    stg = (
        pl.LazyFrame({"b0": [1, 2], "b1": [3, 4], "b2": [5, 6]}),
        pl.DataFrame(
            {
                "variable": ["stg"] * 3,
                "calendar_year": [2025, 2035, 2045],
                "aggregate_type": ["mean", "mean", "max"],
            }
        ),
        window,
    )
    return elev, stg


def _long_equivalent_survivors(where: str) -> set[tuple[str, int]]:
    """The same two datasets as `_joined_fixture`, joined and filtered in LONG shape, as a
    set of surviving `(variable, calendar_year)` pairs -- the ground truth the band-wide
    lowering has to reproduce. Built with the real `join_long`/`filter_long`, not a
    hand-written expectation, so the two shapes are compared against each other rather
    than both against my reading of the semantics.
    """
    elev = pl.LazyFrame(
        {"grid_cell": [1, 1, 1], "calendar_year": [2025, 2035, 2045], "value": [10.0, 20.0, 30.0]}
    )
    stg = pl.LazyFrame(
        {
            "grid_cell": [1, 1, 1],
            "calendar_year": [2025, 2035, 2045],
            "aggregate_type": ["mean", "mean", "max"],
            "value": [1.0, 3.0, 5.0],
        }
    )
    joined = join_long(
        [elev, stg], JoinOp(step_no=0, type="spatial", inputs=["elev", "stg"], output="joined")
    )
    rows = filter_long(joined, where).collect()
    survivors: set[tuple[str, int]] = set()
    for year in rows["calendar_year"].to_list():
        survivors.add(("elev", year))
        survivors.add(("stg", year))
    return survivors


class TestUnifiedMetadataGrowth:
    """How big the unified frame gets, since nothing bounds it. `ANALYSIS_BAND_CEILING`
    limits the bands a plan may READ; the pairing this builds is a product over the
    dimensions the operands do NOT share, and no ceiling counts those rows.

    That is survivable only because the shared set is never empty in practice -- every
    input carries `model_group_id` and `scenario_id`, and operands agreeing on most of
    their dimensions pair near `max(band counts)` rather than near the product. These two
    tests are what keeps that a measured fact rather than an assumption; the second one is
    the shape `lnd-type-plus` actually has, at its real band counts.
    """

    @staticmethod
    def _part(column, dimensions):
        """One origin's contribution: the position column `_dimension_survivors` names it
        by, plus the dimensions it carries."""
        height = len(next(iter(dimensions.values())))
        return pl.DataFrame({column: list(range(height)), **dimensions})

    def test_operands_agreeing_on_their_dimensions_pair_near_the_larger_band_count(self):
        years = list(range(2025, 2025 + 178))
        elev = self._part("__pos0", {"calendar_year": years})
        land = self._part("__pos1", {"calendar_year": [*years, 2203]})

        unified = unify_metadata_parts([elev, land])

        # 178 + 179 = 357 bands, and 179 pairing rows -- `max`, plus nothing, because the
        # one dimension they share pairs them off. 178 * 179 = 31862 is what a cross join
        # would have cost for the same two reads.
        assert elev.height + land.height == 357
        assert unified.height == 179

    def test_an_empty_shared_set_is_the_cross_join_it_has_to_be(self):
        # The degenerate case: with no dimension in common there is no pairing to make, so
        # every combination IS a combination. Correct, and the growth this class exists to
        # bound -- which is why "the shared set is never empty" is a property of the
        # datasets rather than something enforced here.
        left = self._part("__pos0", {"calendar_year": [2025, 2035]})
        right = self._part("__pos1", {"aggregate_type": ["mean", "max", "min"]})

        unified = unify_metadata_parts([left, right])

        assert unified.height == 6
        assert sorted(unified.columns) == ["__pos0", "__pos1", "aggregate_type", "calendar_year"]


class TestSelectDimensionBandWideOverAJoin:
    """A dimension filter after a `join` must mean what it means in long shape. The
    metadata `join_band_wide` hands over is a VERTICAL concat, so a dimension only one
    operand carries is null on the other's rows for reasons having nothing to do with the
    predicate; filtering it directly discarded that operand wholesale.
    """

    def test_non_shared_dimension_narrows_both_operands(self):
        elev, stg = _joined_fixture()
        joined = join_band_wide([elev, stg], ["elev", "stg"], extra_columns=[])

        _out_lf, out_metadata, _window_out = select_dimension_band_wide(
            joined, "aggregate_type = 'mean'", extra_columns=[]
        )

        survivors = set(zip(out_metadata["variable"].to_list(), out_metadata["calendar_year"].to_list()))
        # `elev` carries no `aggregate_type` at all, and keeps exactly the years whose
        # `stg` counterpart survived -- it is narrowed THROUGH the join, not discarded.
        assert survivors == {("elev", 2025), ("elev", 2035), ("stg", 2025), ("stg", 2035)}
        assert survivors == _long_equivalent_survivors("aggregate_type = 'mean'")

    def test_surviving_bands_carry_their_own_operands_values(self):
        elev, stg = _joined_fixture()
        joined = join_band_wide([elev, stg], ["elev", "stg"], extra_columns=[])

        out_lf, _metadata, _window_out = select_dimension_band_wide(
            joined, "aggregate_type = 'mean'", extra_columns=[]
        )
        result = out_lf.collect()

        # Renumbered contiguously, and each surviving band still holds the pixels of the
        # band it came from -- elev's 2025/2035, then stg's.
        assert list(result.columns) == ["b0", "b1", "b2", "b3"]
        assert result["b0"].to_list() == [10, 11]
        assert result["b1"].to_list() == [20, 21]
        assert result["b2"].to_list() == [1, 2]
        assert result["b3"].to_list() == [3, 4]

    def test_shared_dimension_narrows_both_operands(self):
        elev, stg = _joined_fixture()
        joined = join_band_wide([elev, stg], ["elev", "stg"], extra_columns=[])

        _out_lf, out_metadata, _window_out = select_dimension_band_wide(
            joined, "calendar_year = 2035", extra_columns=[]
        )

        survivors = set(zip(out_metadata["variable"].to_list(), out_metadata["calendar_year"].to_list()))
        assert survivors == {("elev", 2035), ("stg", 2035)}
        assert survivors == _long_equivalent_survivors("calendar_year = 2035")

    def test_trivially_true_predicate_keeps_every_band(self):
        """A filter removes what its predicate excludes and nothing else. The unified
        frame is a FULL outer join for exactly this: an inner one would drop `elev`'s
        unpaired 2045 band here, making the filter quietly finish a job `join_band_wide`
        did not do.
        """
        elev, stg = _joined_fixture()
        joined = join_band_wide([elev, stg], ["elev", "stg"], extra_columns=[])

        _out_lf, out_metadata, _window_out = select_dimension_band_wide(
            joined, "calendar_year > 1900", extra_columns=[]
        )

        assert out_metadata.height == joined[1].height == 6

    def test_no_surviving_combination_returns_empty_band_wide(self):
        elev, stg = _joined_fixture()
        joined = join_band_wide([elev, stg], ["elev", "stg"], extra_columns=[])

        out_lf, out_metadata, out_window = select_dimension_band_wide(
            joined, "aggregate_type = 'nonesuch'", extra_columns=[]
        )

        assert out_lf.collect().is_empty()
        assert out_metadata.is_empty()
        assert out_window == joined[2]

    def test_per_band_identity_column_is_rejected(self):
        elev, stg = _joined_fixture()
        joined = join_band_wide([elev, stg], ["elev", "stg"], extra_columns=[])

        with pytest.raises(ExpressionError, match="per-band identity column"):
            select_dimension_band_wide(joined, "variable = 'elev'", extra_columns=[])

    def test_three_operands_narrow_together(self):
        elev, stg = _joined_fixture()
        lnd = (
            pl.LazyFrame({"b0": [7, 8], "b1": [9, 10], "b2": [11, 12]}),
            pl.DataFrame({"variable": ["lnd_type"] * 3, "calendar_year": [2025, 2035, 2045]}),
            _window(height=2),
        )
        joined = join_band_wide([elev, stg, lnd], ["elev", "stg", "lnd"], extra_columns=[])

        _out_lf, out_metadata, _window_out = select_dimension_band_wide(
            joined, "aggregate_type = 'mean'", extra_columns=[]
        )

        survivors = set(zip(out_metadata["variable"].to_list(), out_metadata["calendar_year"].to_list()))
        assert survivors == {
            ("elev", 2025),
            ("elev", 2035),
            ("stg", 2025),
            ("stg", 2035),
            ("lnd_type", 2025),
            ("lnd_type", 2035),
        }

    def test_operand_contributing_no_bands_does_not_block_the_others(self):
        """`join_band_wide` guards the zero-band operand by contributing zero metadata
        rows, so it is simply absent from the reconstruction rather than joining nothing
        to everything.
        """
        elev, _stg = _joined_fixture()
        empty = (pl.LazyFrame({}), pl.DataFrame(), _window(height=2))
        joined = join_band_wide([elev, empty], ["elev", "empty"], extra_columns=[])

        _out_lf, out_metadata, _window_out = select_dimension_band_wide(
            joined, "calendar_year >= 2035", extra_columns=[]
        )

        assert out_metadata["calendar_year"].to_list() == [2035, 2045]


class TestHeightAndWindowPreserved:
    """Every band-wide-shape-preserving lowering keeps both its row count and its window
    unchanged. No lowering asserts this on itself -- `assert_height_preserved` collects
    both frames, so an inline call would be two full raster reads per op (see its
    docstring) -- which makes this class the only place the invariant is checked at all,
    for value filter, dimension filter (the non-empty case), case, group (the
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

    def test_case(self):
        band_wide = _two_band_fixture()
        op = CaseOp(step_no=1, input="x", cases=[], else_="value * 2", output="y")
        out_lf, _metadata, window = case_band_wide(band_wide, op, extra_columns=[])
        assert out_lf.collect().height == 3
        assert window == band_wide[2]

    def test_case_whose_every_band_compiles_to_a_scalar(self):
        """The one shape that can lose the row count without anyone noticing: a `when` that
        is purely dimensional and a `then`/`else` that are constants leave an expression
        with no column reference in it, whose natural length is one. `_select_band_exprs`
        broadcasts it back to the window's height; a plain `select` would answer one row.
        """
        band_wide = _two_band_fixture()
        op = _case_op(("calendar_year = 2035", "6"), else_="0")
        out_lf, _metadata, window = case_band_wide(band_wide, op, extra_columns=[])
        assert out_lf.collect().height == 3
        assert window == band_wide[2]

    def test_case_with_no_matching_band_and_no_else(self):
        # The same hazard at its emptiest: every arm dropped out and there is no `else`, so
        # the expression is a bare `pl.lit(None)`.
        band_wide = _two_band_fixture()
        out_lf, _metadata, window = case_band_wide(
            band_wide, _case_op(("calendar_year = 1900", "6")), extra_columns=[]
        )
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

        result = sample_grid_cells_band_wide(
            band_wide, "grid_cell < 2", extra_columns=[], grid_width=band_wide[2].width
        ).collect()

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
            band_wide,
            "grid_cell = 1 and calendar_year = 2021 and value > 20",
            extra_columns=[],
            grid_width=band_wide[2].width,
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
        result = filter_band_wide(band_wide, "grid_cell = 0", extra_columns=[], grid_width=band_wide[2].width)
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

    def test_a_carried_column_in_a_where_is_a_per_pixel_mask_not_a_band_selection(self):
        """A conjunct naming a carried per-pixel column classifies as a VALUE conjunct, so
        it masks each band's own pixel where it fails rather than being evaluated once per
        band against a `metadata_df` that has no such column.

        The band set is untouched -- `zone` says nothing about which bands exist -- and the
        carried column itself rides through, same as it does past every other band-wide
        lowering.
        """
        lf = pl.LazyFrame({"b0": [1, 10], "b1": [2, 20], "__zone__": [7, 9]})
        metadata_df = pl.DataFrame({"calendar_year": [2020, 2021]})
        band_wide = (lf, metadata_df, _window(height=2))

        out_lf, out_metadata, _window_out = filter_band_wide(
            band_wide, "__zone__ = 7", extra_columns=["__zone__"]
        )
        result = out_lf.collect()
        assert out_metadata["calendar_year"].to_list() == [2020, 2021]
        assert result["b0"].to_list() == [1, None]
        assert result["b1"].to_list() == [2, None]
        assert result["__zone__"].to_list() == [7, 9]

    def test_a_carried_column_and_a_dimension_conjunct_still_split_by_kind(self):
        """The two halves route independently: `calendar_year` still selects bands off the
        metadata, `__zone__` still masks pixels on what survives. Pins that widening the
        classifier's namespace did not swallow the dimension side with it.
        """
        lf = pl.LazyFrame({"b0": [1, 10], "b1": [2, 20], "__zone__": [7, 9]})
        metadata_df = pl.DataFrame({"calendar_year": [2020, 2021]})
        band_wide = (lf, metadata_df, _window(height=2))

        out_lf, out_metadata, _window_out = filter_band_wide(
            band_wide, "calendar_year = 2021 and __zone__ = 7", extra_columns=["__zone__"]
        )
        result = out_lf.collect()
        assert out_metadata["calendar_year"].to_list() == [2021]
        assert result["b0"].to_list() == [2, None]

    def test_dimension_half_empty_short_circuits_before_value_mask(self):
        lf = pl.LazyFrame({"b0": [1, 2]})
        metadata_df = pl.DataFrame({"calendar_year": [2020]})
        band_wide = (lf, metadata_df, _window(height=2))
        out_lf, out_metadata, _window_out = filter_band_wide(
            band_wide, "calendar_year > 3000 and value > 0", extra_columns=[]
        )
        assert out_lf.collect().is_empty()
        assert out_metadata.is_empty()


class TestLoweredCalculateBandWide:
    """What `compile_ops` turns a `calculate` into band-wide: a `case` with no arms, whose
    `else` is the expression. Every band gets it, unconditionally -- there is no `where`
    to make it selective, which is the whole of the difference from what `calculate`
    used to be.
    """

    def _calculate(self, expression):
        return CaseOp(step_no=1, input="x", cases=[], else_=expression, output="y")

    def test_applies_to_every_band(self):
        lf = pl.LazyFrame({"b0": [1, 2], "b1": [3, 4]})
        metadata_df = pl.DataFrame({"calendar_year": [2020, 2021]})
        band_wide = (lf, metadata_df, _window(height=2))

        out_lf, out_metadata, _window_out = case_band_wide(
            band_wide, self._calculate("value * 10"), extra_columns=[]
        )
        result = out_lf.collect()

        assert result["b0"].to_list() == [10, 20]
        assert result["b1"].to_list() == [30, 40]
        assert out_metadata.equals(metadata_df)

    def test_extra_columns_reachable_from_the_expression(self):
        lf = pl.LazyFrame({"b0": [1, 2], "zone_factor": [10, 100]})
        metadata_df = pl.DataFrame({"calendar_year": [2020]})
        band_wide = (lf, metadata_df, _window(height=2))
        out_lf, _metadata, _window_out = case_band_wide(
            band_wide, self._calculate("value * zone_factor"), extra_columns=["zone_factor"]
        )
        assert out_lf.collect()["b0"].to_list() == [10, 200]

    def test_nodata_stays_nodata(self):
        lf = pl.LazyFrame({"b0": [1, None]})
        metadata_df = pl.DataFrame({"calendar_year": [2020]})
        band_wide = (lf, metadata_df, _window(height=2))
        out_lf, _metadata, _window_out = case_band_wide(
            band_wide, self._calculate("value + 1"), extra_columns=[]
        )
        assert out_lf.collect()["b0"].to_list() == [2, None]


def _case_op(*arms, else_=None):
    return CaseOp(
        step_no=1,
        input="x",
        cases=[CaseWhen(when=when, then=then) for when, then in arms],
        else_=else_,
        output="y",
    )


class TestCaseBandWideUnpaired:
    """One value column, so a `case` is a MAP: every band is reclassified independently
    and the band count is unchanged. `value` in an arm means whichever band is being
    written, exactly as it does in a `calculate`.
    """

    def test_every_band_is_reclassified_independently(self):
        lf = pl.LazyFrame({"b0": [1.0, 2.0], "b1": [3.0, 4.0]})
        metadata_df = pl.DataFrame({"calendar_year": [2020, 2021]})
        op = _case_op(("value > 3", "100"), ("value > 1", "10"))

        out_lf, out_metadata, out_window = case_band_wide(
            (lf, metadata_df, _window(height=2)), op, extra_columns=[]
        )
        result = out_lf.collect()

        assert result["b0"].to_list() == [None, 10.0]
        assert result["b1"].to_list() == [10.0, 100.0]
        # A map, not a reduce: the band axis is untouched, metadata included.
        assert out_metadata.equals(metadata_df)
        assert out_window == _window(height=2)

    def test_an_else_reaches_nodata_and_an_arm_can_take_it_back(self):
        # No guard wraps the chain, so `else 0` paints every nodata pixel a real class --
        # band-wide, a whole raster of them. That is the stated cost of leaving nodata to
        # the author, and the second half is the author stating it: an arm that tests
        # nullity is reachable now, and one conjunct restores the masking.
        lf = pl.LazyFrame({"b0": [1.0, None]})
        metadata_df = pl.DataFrame({"calendar_year": [2020]})
        op = _case_op(("value > 5", "1"), else_="0")
        out_lf, _metadata, _window_out = case_band_wide(
            (lf, metadata_df, _window(height=2)), op, extra_columns=[]
        )
        assert out_lf.collect()["b0"].to_list() == [0.0, 0.0]

        guarded = _case_op(("value IS NULL", "value"), ("value > 5", "1"), else_="0")
        out_lf, _metadata, _window_out = case_band_wide(
            (lf, metadata_df, _window(height=2)), guarded, extra_columns=[]
        )
        assert out_lf.collect()["b0"].to_list() == [0.0, None]

    def test_extra_columns_are_reachable_from_both_the_when_and_the_then(self):
        lf = pl.LazyFrame({"b0": [1.0, 2.0], "zone_factor": [10.0, 100.0]})
        metadata_df = pl.DataFrame({"calendar_year": [2020]})
        op = _case_op(("value * zone_factor > 50", "value * zone_factor"), else_="0")
        out_lf, _metadata, _window_out = case_band_wide(
            (lf, metadata_df, _window(height=2)), op, extra_columns=["zone_factor"]
        )
        result = out_lf.collect()
        assert result["b0"].to_list() == [0.0, 200.0]
        assert result["zone_factor"].to_list() == [10.0, 100.0]

    def test_no_bands_is_the_frame_unchanged(self):
        window = _window(height=2)
        band_wide = (pl.LazyFrame(), pl.DataFrame(), window)
        assert case_band_wide(band_wide, _case_op(("value > 0", "1")), extra_columns=[]) is band_wide


class TestCaseAgreesWithTheRewriteChainItReplaced:
    """`lnd-type-plus` stated its depth reclassification as a value `filter` plus four
    successive conditional rewrites, the middle two writing NEGATIVE codes so no later
    entry's domain could re-match a cell an earlier one had claimed, and the fourth
    flipping them back. That is now one `case`. Both formulations are built here and run
    against the same band, so this test independently confirms they agree -- sign trick
    included -- rather than pinning whatever the new one happens to produce.

    The chain is spelled as four one-arm `case`s (`else: "value"` for the pass-through
    each rewrite left implicit), which is the only spelling left for it: a `calculate` is
    unconditional now, so a chain of `calculate`s could not state this at all.
    """

    # The depths either side of each class bound, plus the domain floor and a nodata
    # pixel. `DEPTH_CLASSES` is `((0.46, 6), (1.0, 7), (None, 0))`, every interval closed
    # at its upper bound.
    DEPTHS = [0.01, 0.46, 0.47, 1.0, 1.01, 12.5, 0.0, -3.2, None]

    def _band_wide(self):
        lf = pl.LazyFrame({"b0": self.DEPTHS}, schema={"b0": pl.Float64})
        metadata_df = pl.DataFrame({"calendar_year": [2030]})
        return lf, metadata_df, _window(height=len(self.DEPTHS))

    def _old(self):
        band_wide = filter_band_wide(self._band_wide(), "value > 0", extra_columns=[])
        entries = [
            ("-6", "value <= 0.46"),
            ("-7", "value > 0.46 AND value <= 1.0"),
            ("0", "value > 1.0"),
            ("value * -1", "value < 0"),
        ]
        for step_no, (expression, where) in enumerate(entries, start=2):
            op = CaseOp(
                step_no=step_no,
                input="x",
                cases=[CaseWhen(when=where, then=expression)],
                else_="value",
                output="y",
            )
            band_wide = case_band_wide(band_wide, op, extra_columns=[])
        return band_wide[0].collect()["b0"].to_list()

    def _new(self):
        op = _case_op(("value > 1.0", "0"), ("value > 0.46", "7"), ("value > 0", "6"))
        out_lf, _metadata, _window_out = case_band_wide(self._band_wide(), op, extra_columns=[])
        return out_lf.collect()["b0"].to_list()

    def test_the_two_formulations_produce_the_same_band(self):
        assert self._new() == self._old()

    def test_and_that_band_is_the_one_the_class_table_names(self):
        # Stated independently of both implementations, so the test above cannot pass by
        # the two agreeing on a wrong answer.
        assert self._new() == [6.0, 6.0, 7.0, 7.0, 0.0, 0.0, None, None, None]


class TestCaseBandWideDimensionArms:
    """The band-wide reading of a dimension conjunct in a `when`, which is where it means
    something a value conjunct cannot: it selects which BANDS the arm applies to. An arm a
    band's metadata excludes is not in that band's chain at all -- the band falls to
    whichever later arm does apply, or to the `else`. So the arms may disagree about which
    bands they claim, which no per-pixel mask could express, since band identity is
    metadata and not a column of the data.
    """

    def _fixture(self):
        # b0 is 2020's band, b1 is 2021's. Two pixels each.
        lf = pl.LazyFrame({"b0": [1.0, 2.0], "b1": [3.0, 4.0]})
        metadata_df = pl.DataFrame({"calendar_year": [2020, 2021]})
        return lf, metadata_df, _window(height=2)

    def test_a_dimension_only_arm_claims_its_bands_entirely(self):
        # Nothing per-pixel is left to test once the band is selected, so the arm matches
        # the whole band -- and 2020's band, which the arm excludes, takes the `else`.
        op = _case_op(("calendar_year = 2021", "value * 10"), else_="value")
        out_lf, out_metadata, _window_out = case_band_wide(self._fixture(), op, extra_columns=[])
        result = out_lf.collect()
        assert result["b0"].to_list() == [1.0, 2.0]
        assert result["b1"].to_list() == [30.0, 40.0]
        # A `case` is a map here, not a reduce: the band count and the metadata are the
        # frame's own, unchanged.
        assert out_metadata["calendar_year"].to_list() == [2020, 2021]

    def test_an_excluded_arm_leaves_a_later_arm_reachable(self):
        """The behavior an arm-level rule buys that a band-level one could not. `filter`'s
        band selection drops the BAND; an arm's drops only the arm, so 2020's band goes on
        to the second arm rather than skipping the step.
        """
        op = _case_op(("calendar_year = 2021", "1"), ("value > 1", "2"), else_="0")
        result = case_band_wide(self._fixture(), op, extra_columns=[])[0].collect()
        assert result["b0"].to_list() == [0.0, 2.0]  # 2020: arm 1 excluded, arm 2 applies
        assert result["b1"].to_list() == [1.0, 1.0]  # 2021: arm 1 claims the whole band

    def test_a_mixed_arm_ands_the_band_selection_with_the_pixel_mask(self):
        op = _case_op(("calendar_year = 2021 AND value > 3", "1"), else_="0")
        result = case_band_wide(self._fixture(), op, extra_columns=[])[0].collect()
        assert result["b0"].to_list() == [0.0, 0.0]
        assert result["b1"].to_list() == [0.0, 1.0]

    def test_a_band_every_arm_excludes_falls_to_the_else(self):
        op = _case_op(("calendar_year = 2021", "1"))
        result = case_band_wide(self._fixture(), op, extra_columns=[])[0].collect()
        # No `else` either, so what is left is the null an unmatched cell always gets.
        assert result["b0"].to_list() == [None, None]
        assert result["b1"].to_list() == [1.0, 1.0]

    def test_arms_partition_the_bands(self):
        """The ensemble-weighting shape (`analyses/land_type_confidence.py`): one arm per
        member, each naming only its own dimension value, and no `else` because between
        them the arms name every band the frame has.
        """
        op = _case_op(("calendar_year = 2020", "value * 0.25"), ("calendar_year = 2021", "value * 0.75"))
        result = case_band_wide(self._fixture(), op, extra_columns=[])[0].collect()
        assert result["b0"].to_list() == [0.25, 0.5]
        assert result["b1"].to_list() == [2.25, 3.0]

    def test_a_band_that_does_not_carry_the_dimension_fails_the_arm(self):
        """A null in the metadata is `fill_null(False)`, so the arm does not apply --
        rather than the comparison going null and the band matching by accident.
        """
        lf = pl.LazyFrame({"b0": [1.0], "b1": [2.0]})
        metadata_df = pl.DataFrame({"calendar_year": [2020, None]}, schema={"calendar_year": pl.Int64})
        op = _case_op(("calendar_year = 2020", "10"), else_="0")
        result = case_band_wide((lf, metadata_df, _window(height=1)), op, extra_columns=[])[0].collect()
        assert result["b0"].to_list() == [10.0]
        assert result["b1"].to_list() == [0.0]

    def test_a_dimension_the_metadata_does_not_carry_is_an_error(self):
        op = _case_op(("scenario_id = 34", "1"))
        with pytest.raises(Exception, match="scenario_id"):
            case_band_wide(self._fixture(), op, extra_columns=[])[0].collect()


class TestCaseBandWidePairedDimensionArms:
    """A dimension arm over a PAIRED frame. The pairing has already reduced 2N bands to N,
    so the arm is evaluated against one metadata row per PAIR -- built by coalescing the
    pair's own operand rows, since a band-wide pair has no single metadata row of its own.
    """

    def _fixture(self):
        # marsh b0/b1 and sal b2/b3, paired on calendar_year. `variable` differs per
        # operand and so is not a pairing key.
        lf = pl.LazyFrame({"b0": [1.0], "b1": [2.0], "b2": [10.0], "b3": [20.0]})
        metadata_df = pl.DataFrame(
            {
                "calendar_year": [2030, 2040, 2030, 2040],
                "variable": ["marsh_var", "marsh_var", "sal_var", "sal_var"],
                JOIN_OPERAND_COLUMN: ["marsh", "marsh", "sal", "sal"],
            }
        )
        return lf, metadata_df, _window(height=1)

    def test_a_pairing_key_selects_the_pair_it_names(self):
        op = _case_op(("calendar_year = 2030", "marsh + sal"), else_="marsh")
        out_lf, out_metadata, _window_out = case_band_wide(
            self._fixture(), op, extra_columns=[], value_columns=("marsh", "sal")
        )
        result = out_lf.collect()
        assert out_metadata["calendar_year"].to_list() == [2030, 2040]
        assert result["b0"].to_list() == [11.0]  # 2030's pair: the arm applies
        assert result["b1"].to_list() == [2.0]  # 2040's pair: the `else`, marsh alone

    def test_a_non_key_column_is_coalesced_from_whichever_operand_carries_it(self):
        """`variable` is not a pairing key -- the operands disagree on it by construction.
        An arm may still name it when only ONE operand's row carries it, since then there
        is exactly one value for the pair to have.
        """
        lf = pl.LazyFrame({"b0": [1.0], "b1": [10.0]})
        metadata_df = pl.DataFrame(
            {
                "calendar_year": [2030, 2030],
                "aggregate_type": ["mean", None],
                JOIN_OPERAND_COLUMN: ["marsh", "sal"],
            }
        )
        op = _case_op(("aggregate_type = 'mean'", "marsh + sal"), else_="marsh")
        out_lf, _metadata, _window_out = case_band_wide(
            (lf, metadata_df, _window(height=1)), op, extra_columns=[], value_columns=("marsh", "sal")
        )
        assert out_lf.collect()["b0"].to_list() == [11.0]

    def test_operands_disagreeing_on_a_non_key_column_is_an_error(self):
        """Three operands, two of which carry `aggregate_type` DIFFERENTLY: the pair has no
        single value, so an arm naming it would mean two things at once. Refused rather
        than resolved by operand order, which would be an answer that depended on the
        join's argument order.
        """
        lf = pl.LazyFrame({"b0": [1.0], "b1": [10.0], "b2": [100.0]})
        metadata_df = pl.DataFrame(
            {
                "calendar_year": [2030, 2030, 2030],
                "aggregate_type": ["mean", "max", None],
                JOIN_OPERAND_COLUMN: ["marsh", "sal", "elev"],
            }
        )
        op = _case_op(("aggregate_type = 'mean'", "marsh"), else_="sal")
        with pytest.raises(ExpressionError, match="conflicting values.*aggregate_type|aggregate_type.*conflicting"):
            case_band_wide(
                (lf, metadata_df, _window(height=1)),
                op,
                extra_columns=[],
                value_columns=("marsh", "sal", "elev"),
            )

    def test_a_pair_every_arm_excludes_still_produces_its_band(self):
        # Arm-level, not band-level: the pair survives the reduce and takes the `else`,
        # rather than being dropped the way a `filter` would drop it.
        op = _case_op(("calendar_year = 2050", "marsh + sal"), else_="sal")
        out_lf, out_metadata, _window_out = case_band_wide(
            self._fixture(), op, extra_columns=[], value_columns=("marsh", "sal")
        )
        assert out_metadata["calendar_year"].to_list() == [2030, 2040]
        assert out_lf.collect().row(0) == (10.0, 20.0)


class TestCaseBandWidePaired:
    """More than one value column, so a `case` is a REDUCE, not a map: each operand's band
    is paired with its counterpart on the dimensions they share, and 2N bands come back as
    N. This is the only lowering that collapses a band-wide pairing without also collapsing
    the bands -- `filter` used to do it too, and no longer does.
    """

    def _paired_fixture(self):
        # The same layout `join_band_wide` produces: all of `marsh`'s bands, then all of
        # `sal`'s. Two years, two operands, two pixels. `variable` differs per operand and
        # must NOT be paired on -- a spatial join's whole premise is that its operands are
        # different datasets -- so it drops out of the paired metadata entirely.
        lf = pl.LazyFrame({"b0": [1.0, 2.0], "b1": [3.0, 4.0], "b2": [10.0, 20.0], "b3": [30.0, 40.0]})
        metadata_df = pl.DataFrame(
            {
                "calendar_year": [2030, 2040, 2030, 2040],
                "variable": ["marsh_var", "marsh_var", "sal_var", "sal_var"],
                JOIN_OPERAND_COLUMN: ["marsh", "marsh", "sal", "sal"],
            }
        )
        return lf, metadata_df, _window(height=2)

    def test_arms_naming_both_operands_reduce_each_pair_to_one_band(self):
        op = _case_op(("sal > 25", "marsh * 2"), ("marsh > 1.5", "marsh"), else_="0")
        out_lf, out_metadata, _window_out = case_band_wide(
            self._paired_fixture(), op, extra_columns=[], value_columns=("marsh", "sal")
        )
        result = out_lf.collect()

        # One band per calendar_year, down from four, and the operand axis is gone.
        assert out_metadata.columns == ["calendar_year"]
        assert out_metadata["calendar_year"].to_list() == [2030, 2040]
        assert list(result.columns) == ["b0", "b1"]
        # 2030 pairs marsh b0 (1.0, 2.0) with sal b2 (10.0, 20.0): neither pixel's sal is
        # over 25, pixel0's marsh fails `> 1.5` and falls to the `else`, pixel1's passes.
        assert result["b0"].to_list() == [0.0, 2.0]
        # 2040 pairs marsh b1 (3.0, 4.0) with sal b3 (30.0, 40.0): both sals clear 25, so
        # the first arm claims both pixels.
        assert result["b1"].to_list() == [6.0, 8.0]

    def test_an_operand_no_arm_names_does_not_null_the_pair(self):
        """Pixel 1 has no `sal`, and no part of this `case` mentions `sal`: the arm reads
        `marsh`, which is present, and `2.0` is its answer. The pair being half present is
        not by itself a reason to withhold one -- the arms decide, and an author who wants
        the whole pair says `AND sal IS NOT NULL`.
        """
        lf = pl.LazyFrame({"b0": [1.0, 2.0], "b1": [10.0, None]})
        metadata_df = pl.DataFrame(
            {"calendar_year": [2030, 2030], JOIN_OPERAND_COLUMN: ["marsh", "sal"]}
        )
        op = _case_op(("marsh > 0", "marsh"), else_="0")
        out_lf, _metadata, _window_out = case_band_wide(
            (lf, metadata_df, _window(height=2)), op, extra_columns=[], value_columns=("marsh", "sal")
        )
        assert out_lf.collect()["b0"].to_list() == [1.0, 2.0]

        both = _case_op(("marsh > 0 AND sal IS NOT NULL", "marsh"))
        out_lf, _metadata, _window_out = case_band_wide(
            (lf, metadata_df, _window(height=2)), both, extra_columns=[], value_columns=("marsh", "sal")
        )
        assert out_lf.collect()["b0"].to_list() == [1.0, None]

    def test_a_band_with_no_counterpart_is_dropped(self):
        """An incomplete pair has no answer to give: the arms name both operands, and one
        of them has nothing at this dimension. Dropping the band is the only honest option
        -- evaluating it would silently answer as if the missing operand had been read.
        """
        lf = pl.LazyFrame({"b0": [1.0], "b1": [2.0], "b2": [3.0]})
        metadata_df = pl.DataFrame(
            {
                "calendar_year": [2030, 2040, 2030],
                JOIN_OPERAND_COLUMN: ["marsh", "marsh", "sal"],
            }
        )
        out_lf, out_metadata, _window_out = case_band_wide(
            (lf, metadata_df, _window(height=1)),
            _case_op(("marsh > sal", "marsh"), else_="sal"),
            extra_columns=[],
            value_columns=("marsh", "sal"),
        )
        assert out_metadata["calendar_year"].to_list() == [2030]
        assert out_lf.collect().columns == ["b0"]

    def test_no_pair_survives_is_an_empty_frame(self):
        """A pairing group holding the SAME operand twice is not a pair either -- which of
        the two would `marsh` mean? Distinguished from the missing-counterpart case above
        because the group is the right SIZE here, just not the right shape.
        """
        lf = pl.LazyFrame({"b0": [1.0], "b1": [2.0]})
        metadata_df = pl.DataFrame(
            {"calendar_year": [2030, 2030], JOIN_OPERAND_COLUMN: ["marsh", "marsh"]}
        )
        out_lf, out_metadata, out_window = case_band_wide(
            (lf, metadata_df, _window(height=1)),
            _case_op(("marsh > sal", "marsh")),
            extra_columns=[],
            value_columns=("marsh", "sal"),
        )
        assert out_metadata.is_empty()
        assert out_lf.collect().is_empty()
        assert out_window == _window(height=1)

    def test_no_shared_dimension_pairs_every_band_into_one_group(self):
        """When the operands share no non-null dimension at all there is nothing to key on,
        so the whole frame is one pairing group. It still has to be a well-formed pair, and
        the surviving band is indexed by `RESERVED_BAND_INDEX_COLUMN` -- the same fallback
        `group_band_wide` uses for `by == []`.
        """
        lf = pl.LazyFrame({"b0": [5.0], "b1": [50.0]})
        metadata_df = pl.DataFrame(
            {
                # Every pairable column is null in at least one row, so none qualifies.
                "calendar_year": [2030, None],
                JOIN_OPERAND_COLUMN: ["marsh", "sal"],
            }
        )
        out_lf, out_metadata, _window_out = case_band_wide(
            (lf, metadata_df, _window(height=1)),
            _case_op(("marsh > 1 AND sal > 1", "marsh")),
            extra_columns=[],
            value_columns=("marsh", "sal"),
        )

        assert out_metadata.columns == [RESERVED_BAND_INDEX_COLUMN]
        assert out_metadata[RESERVED_BAND_INDEX_COLUMN].to_list() == [0]
        assert out_lf.collect()["b0"].to_list() == [5.0]

    def test_no_bands_returns_empty(self):
        window = _window(height=2)
        out_lf, out_metadata, out_window = case_band_wide(
            (pl.LazyFrame(), pl.DataFrame(), window),
            _case_op(("marsh > 0 AND sal > 0", "marsh")),
            extra_columns=[],
            value_columns=("marsh", "sal"),
        )
        assert out_lf.collect().is_empty()
        assert out_metadata.is_empty()
        assert out_window == window


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

    def test_of_a_metadata_column_contributes_that_columns_value_per_band(self):
        """`of="regime"` makes each band contribute its own metadata scalar wherever the
        band is non-null, so `min` over group "a" answers "which regime got here first"
        rather than reducing the pixel values. `max` on the same fixture would be
        indistinguishable from `min` if the lowering silently ignored `of` and reduced
        the bands, so both are checked.
        """
        lf = pl.LazyFrame({"b0": [1, None, 3], "b1": [10, 20, None]})
        metadata_df = pl.DataFrame({"regime": ["a", "a"], "rank": [7, 4]})
        band_wide = (lf, metadata_df, _window())

        op = GroupOp(step_no=1, input="x", by=["regime"], aggregate="min", of="rank", output="y")
        result = group_band_wide(band_wide, op, extra_columns=[])[0].collect()
        # pixel0: both bands contribute -> min(7, 4); pixel1: only b1 -> 4; pixel2: only b0 -> 7.
        assert result["b0"].to_list() == [4, 4, 7]

        op = GroupOp(step_no=1, input="x", by=["regime"], aggregate="max", of="rank", output="y")
        result = group_band_wide(band_wide, op, extra_columns=[])[0].collect()
        assert result["b0"].to_list() == [7, 4, 7]

    def test_of_naming_neither_value_nor_a_metadata_column_raises(self):
        band_wide = _three_band_fixture()
        op = GroupOp(step_no=1, input="x", by=[], aggregate="min", of="calendar_year", output="y")
        with pytest.raises(ValueError, match="names neither"):
            group_band_wide(band_wide, op, extra_columns=[])

    def test_count_is_nodata_outside_the_domain_and_zero_inside_it(self):
        """The one place `DOMAIN_COLUMN` is readable from the outside: every band is null
        at both pixels below, so `count` can only tell them apart by the carried zone.
        Pixel 0 is in the crosswalk's domain and genuinely counted zero contributors;
        pixel 1 was never mapped at all, and must come back null (which the sink then
        writes as its nodata sentinel), not as a confident 0.
        """
        lf = pl.LazyFrame({"b0": [None, None], "b1": [None, None], DOMAIN_COLUMN: [5, None]})
        metadata_df = pl.DataFrame({"regime": ["a", "a"]})
        band_wide = (lf, metadata_df, _window(height=2))

        op = GroupOp(step_no=1, input="x", by=[], aggregate="count", output="y")
        result = group_band_wide(band_wide, op, extra_columns=[DOMAIN_COLUMN])[0].collect()

        assert result["b0"].to_list() == [0, None]


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


class TestSpatialFilterCommutesWithCaseAndGroup:
    """A spatial (`grid_cell`) conjunct, sitting after a `case` and a `group` in the
    same band-wide chain, is a candidate for a compile-time pushdown marker that names the
    conjunct's own step rather than one nearer the read -- but only if restricting the
    frame to one pixel BEFORE running the `case`/`group` in front of it would produce
    the identical answer running them over the whole window and picking that pixel out
    afterward does. `case_band_wide` acts one pixel at a time (nothing about its
    `pl.when(...).then(...)` reads across rows) and `group_band_wide` reduces HORIZONTALLY
    across a row's own bands (nothing about `sum_horizontal` reads across rows either) --
    so neither op has any way to make one pixel's output depend on another pixel's input,
    which is exactly the property this test checks with concrete, hand-computed numbers
    rather than assuming it from the two lowerings' docstrings alone.

    2 pixels, 2 bands, no nulls: `b0 = [10, 100]`, `b1 = [1, 5]`. The `case` is a lowered
    `calculate` -- no arms, `else` `value + 1` -- so it adds 1 to every band
    unconditionally; `group` collapses both bands into one via an unweighted horizontal
    sum (`by=[]`, `aggregate="sum"`). Worked by hand for pixel 1 (`grid_cell = 1`, the
    second row): post-`case` `b0 = 101`, `b1 = 6`; post-group
    `b0 = 101 + 6 = 107`.
    """

    def _fixture(self):
        lf = pl.LazyFrame({"b0": [10, 100], "b1": [1, 5]})
        metadata_df = pl.DataFrame({"calendar_year": [2020, 2021]})
        return lf, metadata_df, _window(height=2)

    def test_full_window_then_pick_one_pixel_matches_hand_computed_value(self):
        band_wide = self._fixture()
        calculate_op = CaseOp(step_no=1, input="x", cases=[], else_="value + 1", output="c")
        group_op = GroupOp(step_no=2, input="c", by=[], aggregate="sum", output="g")

        calculated = case_band_wide(band_wide, calculate_op, extra_columns=[])
        grouped = group_band_wide(calculated, group_op, extra_columns=[])
        long_result = sample_grid_cells_band_wide(
            grouped, "grid_cell = 1", extra_columns=[], grid_width=grouped[2].width
        ).collect()

        assert long_result.height == 1
        assert long_result["value"].to_list() == [107]

    def test_picking_the_pixel_first_matches_the_same_hand_computed_value(self):
        """The pushdown candidate itself: restrict to `grid_cell = 1` FIRST -- as compile-
        time pushdown would, if it read only that one pixel from disk -- then run the
        identical `case`/`group` over just that one-row frame. No executor wires this
        up (this module composes the lowerings by hand, exactly as a real one eventually
        would), but the two results below being equal is the concrete fact that justifies
        marking the spatial conjunct's own step as pushable rather than forcing the read
        back to the chain's start.
        """
        lf, metadata_df, window = self._fixture()
        one_pixel_lf = row_major_grid_cell(lf, window).filter(pl.col("grid_cell") == 1).drop("grid_cell")
        one_pixel = (one_pixel_lf, metadata_df, dataclasses.replace(window, height=1))

        calculate_op = CaseOp(step_no=1, input="x", cases=[], else_="value + 1", output="c")
        group_op = GroupOp(step_no=2, input="c", by=[], aggregate="sum", output="g")

        calculated = case_band_wide(one_pixel, calculate_op, extra_columns=[])
        grouped = group_band_wide(calculated, group_op, extra_columns=[])
        result = grouped[0].collect()

        assert result["b0"].to_list() == [107]
