"""Tests for the band-wide frame contract (`cpra.mp.data.analysis.execution.frame`):
the band-naming invariant, the two degenerate Polars shapes, the reserved band-index
column never surviving into exposed metadata, and the cheap invariant assertions every
band-wide lowering is expected to call on itself.

No file, raster, or request is ever touched here -- everything is built in memory, which
is the whole point of this module being pure Polars.
"""

import dataclasses

import polars as pl
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine

from cpra.mp.data.analysis.execution import frame
from cpra.mp.data.analysis.request.pipeline import BAND_NAME_RE, RESERVED_BAND_INDEX_COLUMN


def make_window(**overrides) -> frame.ReadWindow:
    defaults = dict(
        source="veg_grid_cell_v001/elev.tif",
        col_off=0,
        row_off=0,
        width=10,
        height=10,
        transform=Affine.identity(),
        crs=CRS.from_epsg(6478),
        decimation=1,
    )
    defaults.update(overrides)
    return frame.ReadWindow(**defaults)


# ----------------------------------------------------------------------------------
# The band-naming invariant
# ----------------------------------------------------------------------------------


class TestBandColumnNames:
    def test_matches_the_positional_pattern_for_any_height(self):
        for height in (0, 1, 2, 7):
            metadata_df = pl.DataFrame({"scenario": ["s"] * height})
            assert frame.band_column_names(metadata_df) == [f"b{i}" for i in range(height)]

    def test_recomputed_from_whatever_metadata_is_in_hand_not_cached(self):
        wide_metadata = pl.DataFrame({"scenario": ["a", "b", "c"]})
        names_before = frame.band_column_names(wide_metadata)

        narrow_metadata = wide_metadata.filter(pl.col("scenario") == "b")
        names_after = frame.band_column_names(narrow_metadata)

        # A correct recomputation shrinks; a "cached from first call" bug would not --
        # a wrong implementation must give a visibly different answer here, not just
        # the same one twice.
        assert names_before == ["b0", "b1", "b2"]
        assert names_after == ["b0"]

    def test_every_band_name_matches_the_shared_pattern(self):
        metadata_df = pl.DataFrame({"scenario": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]})
        for name in frame.band_column_names(metadata_df):
            assert BAND_NAME_RE.match(name)


# ----------------------------------------------------------------------------------
# extra_columns validation (spec 4.3's positional-name rejection)
# ----------------------------------------------------------------------------------


class TestValidateExtraColumns:
    @pytest.mark.parametrize("name", ["b0", "b1", "b23", RESERVED_BAND_INDEX_COLUMN])
    def test_rejects_a_name_that_looks_positional(self, name):
        with pytest.raises(ValueError):
            frame.validate_extra_columns([name])

    @pytest.mark.parametrize("name", ["crosswalk", "grid_cell", "b", "b0x", "band0", "_bands"])
    def test_accepts_an_ordinary_name(self, name):
        frame.validate_extra_columns([name])  # must not raise

    def test_rejects_within_a_larger_list_not_just_a_singleton(self):
        with pytest.raises(ValueError):
            frame.validate_extra_columns(["crosswalk", "b7", "another_ok_name"])

    def test_empty_is_always_fine(self):
        frame.validate_extra_columns([])


# ----------------------------------------------------------------------------------
# Degenerate shape 1: a dimension filter matching no band
# ----------------------------------------------------------------------------------


class TestNoBandDegenerateShape:
    def test_polars_select_of_zero_columns_collapses_height_the_footgun_this_exists_for(self):
        # Demonstrates the actual Polars behavior this module's docstring describes,
        # rather than asserting it by prose: selecting away every column silently loses
        # the row count, even though the frame started with real rows.
        lf = pl.LazyFrame({"b0": [1, 2, 3], "b1": [4, 5, 6]})
        collapsed = lf.select([]).collect()
        assert collapsed.height == 0
        assert collapsed.width == 0

    def test_has_no_bands_true_only_when_metadata_height_is_zero(self):
        assert frame.has_no_bands(pl.DataFrame({"scenario": []}, schema={"scenario": pl.Utf8}))
        assert not frame.has_no_bands(pl.DataFrame({"scenario": ["a"]}))

    def test_empty_band_wide_returns_zero_rows_zero_columns_and_the_same_window(self):
        window = make_window()
        lf, metadata_df, returned_window = frame.empty_band_wide(window)

        collected = lf.collect()
        assert collected.height == 0
        assert collected.width == 0
        assert metadata_df.height == 0
        assert metadata_df.width == 0
        assert returned_window is window

    def test_empty_band_wide_is_unconditional_on_extra_columns_a_caller_might_have_had(self):
        # A lowering that had, say, a crosswalk extra column before its dimension filter
        # dropped every band does not get to keep that column here -- the empty result
        # is total, not partial. Pinned directly since this is a real design choice, not
        # an accident: a wrong ("preserve extras") implementation would return a frame
        # with the crosswalk column still populated at the original height, a distinctly
        # different, wrong-per-this-module's-contract answer.
        window = make_window()
        lf, metadata_df, _ = frame.empty_band_wide(window)
        assert lf.collect().columns == []
        assert metadata_df.columns == []


# ----------------------------------------------------------------------------------
# Degenerate shape 2: group with by=[] collapsing every dimension column
# ----------------------------------------------------------------------------------


class TestReservedBandIndexDegenerateShape:
    def test_naive_collapse_loses_height_the_footgun_this_column_exists_for(self):
        metadata_df = pl.DataFrame({"scenario": ["a", "b", "c"]})
        naively_collapsed = metadata_df.select(pl.exclude("scenario"))
        assert naively_collapsed.height == 0  # every dimension column gone -> height gone too

    def test_add_reserved_band_index_survives_the_same_collapse(self):
        metadata_df = pl.DataFrame({"scenario": ["a", "b", "c"]})
        indexed = frame.add_reserved_band_index(metadata_df)

        assert indexed.columns[0] == RESERVED_BAND_INDEX_COLUMN
        assert indexed[RESERVED_BAND_INDEX_COLUMN].to_list() == [0, 1, 2]

        collapsed = indexed.select(pl.exclude("scenario"))
        # Distinguishing fixture: without the reserved column, this is 0; with it, the
        # height a by=[] group needs to preserve survives the identical operation.
        assert collapsed.height == 3
        assert collapsed.columns == [RESERVED_BAND_INDEX_COLUMN]

    def test_add_reserved_band_index_does_not_mutate_its_input(self):
        # `DataFrame.insert_column` mutates its receiver in place and returns it, not a
        # copy -- an easy mistake to reach for since Polars is otherwise consistently
        # non-mutating. A caller that still holds `metadata_df` after this call must see
        # it unchanged.
        metadata_df = pl.DataFrame({"scenario": ["a", "b", "c"]})
        before = metadata_df.clone()

        frame.add_reserved_band_index(metadata_df)

        assert metadata_df.equals(before)

    def test_add_reserved_band_index_rejects_a_metadata_frame_that_already_has_one(self):
        metadata_df = pl.DataFrame({RESERVED_BAND_INDEX_COLUMN: [0, 1]})
        with pytest.raises(ValueError):
            frame.add_reserved_band_index(metadata_df)

    def test_has_reserved_band_index(self):
        metadata_df = pl.DataFrame({"scenario": ["a", "b"]})
        assert not frame.has_reserved_band_index(metadata_df)
        assert frame.has_reserved_band_index(frame.add_reserved_band_index(metadata_df))

    def test_drop_reserved_band_index_removes_it(self):
        metadata_df = pl.DataFrame({"scenario": ["a", "b", "c"]})
        indexed = frame.add_reserved_band_index(metadata_df)

        dropped = frame.drop_reserved_band_index(indexed)

        assert not frame.has_reserved_band_index(dropped)
        assert dropped.columns == ["scenario"]
        assert dropped.height == 3

    def test_drop_reserved_band_index_is_a_no_op_when_absent(self):
        metadata_df = pl.DataFrame({"scenario": ["a", "b"]})
        dropped = frame.drop_reserved_band_index(metadata_df)
        assert dropped.equals(metadata_df)

    def test_round_trip_never_leaves_the_index_behind(self):
        # Stands in for the resume contract's "no API exposes a stored band index":
        # whatever internal shape a by=[] group produces, the metadata this function
        # hands back before exposure carries none of it.
        metadata_df = pl.DataFrame({"scenario": ["a", "b", "c", "d"]})
        indexed = frame.add_reserved_band_index(metadata_df)
        exposed = frame.drop_reserved_band_index(indexed)

        assert RESERVED_BAND_INDEX_COLUMN not in exposed.columns
        assert exposed.equals(metadata_df)


# ----------------------------------------------------------------------------------
# Cheap invariant assertions
# ----------------------------------------------------------------------------------


class TestAssertWindowUnchanged:
    def test_passes_when_the_window_is_identical(self):
        window = make_window()
        frame.assert_window_unchanged(window, window)  # must not raise

    def test_passes_for_an_equal_but_distinct_object(self):
        frame.assert_window_unchanged(make_window(), make_window())  # must not raise

    @pytest.mark.parametrize(
        "override",
        [
            {"col_off": 5},
            {"row_off": 5},
            {"width": 999},
            {"height": 999},
            {"decimation": 2},
            {"source": "a-different-raster.tif"},
            {"crs": CRS.from_epsg(4326)},
            {"transform": Affine.translation(10, 10)},
        ],
    )
    def test_fires_when_any_single_field_differs(self, override):
        handed = make_window()
        returned = make_window(**override)
        with pytest.raises(AssertionError):
            frame.assert_window_unchanged(handed, returned)


class TestAssertHeightPreserved:
    def test_passes_when_height_is_unchanged_even_if_the_data_is_not(self):
        before = pl.LazyFrame({"b0": [1, 2, 3]})
        # A wrong ("just compare the frames") implementation would fail this case even
        # though height preservation is exactly what should pass -- the values changed,
        # the row count did not.
        after = pl.LazyFrame({"b0": [10, 20, 30]})
        frame.assert_height_preserved(before, after)  # must not raise

    def test_passes_when_column_count_changes_but_row_count_does_not(self):
        before = pl.LazyFrame({"b0": [1, 2, 3], "b1": [4, 5, 6]})
        after = pl.LazyFrame({"b0": [5, 7, 9]})  # a group_by collapsing two bands to one
        frame.assert_height_preserved(before, after)  # must not raise

    def test_fires_when_rows_are_dropped(self):
        before = pl.LazyFrame({"b0": [1, 2, 3]})
        after = pl.LazyFrame({"b0": [1, 2]})
        with pytest.raises(AssertionError):
            frame.assert_height_preserved(before, after)

    def test_fires_when_rows_are_added(self):
        before = pl.LazyFrame({"b0": [1, 2]})
        after = pl.LazyFrame({"b0": [1, 2, 3]})
        with pytest.raises(AssertionError):
            frame.assert_height_preserved(before, after)


# ----------------------------------------------------------------------------------
# ReadWindow itself
# ----------------------------------------------------------------------------------


class TestReadWindow:
    def test_is_frozen(self):
        window = make_window()
        with pytest.raises(dataclasses.FrozenInstanceError):
            window.col_off = 1

    def test_equality_is_structural_not_identity(self):
        assert make_window() == make_window()
        assert make_window() is not make_window()

    def test_a_changed_field_breaks_equality(self):
        assert make_window() != make_window(decimation=2)


# ----------------------------------------------------------------------------------
# The two frame-shape type aliases exist and mean what the module says they mean
# ----------------------------------------------------------------------------------


def test_long_frame_alias_is_lazyframe():
    assert frame.LongFrame is pl.LazyFrame


def test_band_wide_frame_alias_matches_the_documented_triple():
    lf, metadata_df, window = pl.LazyFrame(), pl.DataFrame(), make_window()
    value: frame.BandWideFrame = (lf, metadata_df, window)
    assert value == (lf, metadata_df, window)
