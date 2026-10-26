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
from cpra.mp.data.analysis.request.errors import ShapeError
from cpra.mp.data.analysis.request.pipeline import BAND_NAME_RE, RESERVED_BAND_INDEX_COLUMN


def make_window(**overrides) -> frame.ReadWindow:
    defaults = {
        "source": "veg_grid_cell_v001/elev.tif",
        "col_off": 0,
        "row_off": 0,
        "width": 10,
        "height": 10,
        "transform": Affine.identity(),
        "crs": CRS.from_epsg(6478),
        "decimation": (1, 1),
    }
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
# extra_columns validation (a non-band column may not take a positional band name)
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

    def test_dropping_a_sole_reserved_index_destroys_the_count_it_was_added_to_carry(self):
        # The exact hole `released_band_metadata` exists for, pinned on its own so the two
        # tests below are read against a demonstrated fact rather than an asserted one: a
        # `by=[]` group leaves the reserved column as the ONLY column, and `drop` of a
        # frame's last column reports height 0 no matter how many rows went in.
        collapsed = pl.DataFrame({"scenario": ["a"]}).select([])
        sole = frame.add_reserved_band_index(pl.DataFrame({"scenario": ["a"]})).select(
            RESERVED_BAND_INDEX_COLUMN
        )

        assert collapsed.shape == (0, 0)
        assert sole.height == 1
        assert frame.drop_reserved_band_index(sole).height == 0

    def test_released_band_metadata_keeps_a_sole_reserved_index_so_the_band_count_survives(self):
        # One band in, one band out. Dropping here would report zero bands for a result
        # that has one, which a raster sink reads as "this request matched no data".
        sole = frame.add_reserved_band_index(pl.DataFrame({"scenario": ["a"]})).select(
            RESERVED_BAND_INDEX_COLUMN
        )

        released = frame.released_band_metadata(sole)

        assert released.height == 1
        assert not frame.has_no_bands(released)
        assert frame.band_column_names(released) == ["b0"]

    def test_released_band_metadata_still_drops_the_index_beside_a_real_column(self):
        # The ordinary case is unchanged: where a dimension column carries the height, the
        # position has nothing left to carry and goes, exactly as before.
        indexed = frame.add_reserved_band_index(pl.DataFrame({"scenario": ["a", "b", "c"]}))

        released = frame.released_band_metadata(indexed)

        assert released.columns == ["scenario"]
        assert released.height == 3

    def test_released_band_metadata_is_a_no_op_when_the_index_is_absent(self):
        metadata_df = pl.DataFrame({"scenario": ["a", "b"]})
        assert frame.released_band_metadata(metadata_df).equals(metadata_df)

    def test_released_band_metadata_leaves_a_genuinely_empty_frame_empty(self):
        # A dimension filter that kept no band is still zero bands after release -- the
        # sole-column carve-out must not manufacture one out of an empty frame.
        empty = pl.DataFrame({RESERVED_BAND_INDEX_COLUMN: pl.Series([], dtype=pl.UInt32)})

        released = frame.released_band_metadata(empty)

        assert released.height == 0
        assert frame.has_no_bands(released)


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
            {"decimation": (2, 2)},
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


class TestAssertSamePixelGrid:
    def test_passes_for_two_reads_of_different_sources_over_the_same_grid(self):
        # The case this function exists for, and the one plain `ReadWindow` equality gets
        # wrong: two operands of a band-wide join are by definition two different reads.
        # A `==`-based check would reject this, leaving a join satisfiable only against
        # itself -- so a wrong implementation must visibly fail here, not merely agree.
        one = make_window(source="veg_grid_cell_v001/elev.tif")
        other = make_window(source="veg_grid_cell_v001/salinity.tif")
        assert one != other
        frame.assert_same_pixel_grid(one, other)  # must not raise

    def test_passes_when_the_window_is_identical(self):
        frame.assert_same_pixel_grid(make_window(), make_window())  # must not raise

    @pytest.mark.parametrize(
        "override",
        [
            {"col_off": 5},
            {"row_off": 5},
            {"width": 999},
            {"height": 999},
            {"decimation": (2, 2)},
            {"crs": CRS.from_epsg(4326)},
            {"transform": Affine.translation(10, 10)},
            {"batch_layout": ((0, 0, 2, 4), (2, 0, 2, 4))},
        ],
    )
    def test_fires_when_any_geometric_field_differs(self, override):
        with pytest.raises(AssertionError):
            frame.assert_same_pixel_grid(make_window(), make_window(**override))

    def test_fires_for_two_reads_that_cover_the_same_ground_in_different_orders(self):
        """The case that makes `batch_layout` a compared field rather than a bookkeeping
        one: two reads of the identical window, chunked differently (which is what happens
        when they have different band counts), agree on every geometric measure and still
        disagree about which pixel row *i* is.
        """
        one = make_window(batch_layout=((0, 0, 4, 4),))
        other = make_window(batch_layout=((0, 0, 2, 4), (2, 0, 2, 4)))

        with pytest.raises(AssertionError, match="batch_layout"):
            frame.assert_same_pixel_grid(one, other)

    def test_names_the_fields_that_differ(self):
        with pytest.raises(AssertionError, match="col_off, width"):
            frame.assert_same_pixel_grid(make_window(), make_window(col_off=5, width=999))


class TestRowMajorGridCell:
    """`grid_cell` is a pixel's row-major position within the window, which is the frame's
    own row index only when the read came back in one batch.
    """

    def test_a_single_batch_read_is_the_plain_row_index(self):
        window = make_window(width=2, height=3, batch_layout=((0, 0, 2, 3),))
        lf = pl.LazyFrame({"b0": [10, 11, 12, 13, 14, 15]})

        out = frame.row_major_grid_cell(lf, window).collect()

        assert out["grid_cell"].to_list() == [0, 1, 2, 3, 4, 5]
        assert out["b0"].to_list() == [10, 11, 12, 13, 14, 15]

    def test_an_empty_batch_layout_is_treated_as_one_batch(self):
        """A hand-built `ReadWindow` (a test's, or any window not produced by a batched
        scan) carries no layout; "the whole window, once" is what that means."""
        window = make_window(width=2, height=2, batch_layout=())
        lf = pl.LazyFrame({"b0": [1, 2, 3, 4]})

        assert frame.row_major_grid_cell(lf, window).collect()["grid_cell"].to_list() == [0, 1, 2, 3]

    def test_two_column_strips_are_de_interleaved(self):
        """A 4x2 window read as two 2-wide strips emits columns 0-1 of both rows, then
        columns 2-3 of both rows. Row order 0..7 therefore means cells 0,1,4,5,2,3,6,7 --
        a plain row index would put half the pixels in the wrong place.
        """
        window = make_window(width=4, height=2, batch_layout=((0, 0, 2, 2), (2, 0, 2, 2)))
        lf = pl.LazyFrame({"b0": ["a", "b", "e", "f", "c", "d", "g", "h"]})

        out = frame.row_major_grid_cell(lf, window).collect().sort("grid_cell")

        assert out["grid_cell"].to_list() == [0, 1, 2, 3, 4, 5, 6, 7]
        assert out["b0"].to_list() == ["a", "b", "c", "d", "e", "f", "g", "h"]

    def test_batches_of_differing_shape_are_decoded_by_their_own_widths(self):
        """The last batch of a block row is narrower than the rest whenever the width is
        not a multiple of the batch width, so each batch's offset must be split by ITS own
        width, not by a single shared one.
        """
        window = make_window(width=3, height=2, batch_layout=((0, 0, 2, 2), (2, 0, 1, 2)))
        lf = pl.LazyFrame({"b0": [0, 1, 3, 4, 2, 5]})

        out = frame.row_major_grid_cell(lf, window).collect()

        assert out["grid_cell"].to_list() == [0, 1, 3, 4, 2, 5]
        assert out.sort("grid_cell")["b0"].to_list() == [0, 1, 2, 3, 4, 5]

    def test_offsets_are_relative_to_the_window_not_the_raster(self):
        """A windowed read's batches carry absolute pixel offsets, but cell 0 is the
        window's own first pixel."""
        window = make_window(col_off=10, row_off=20, width=2, height=2, batch_layout=((10, 20, 2, 2),))
        lf = pl.LazyFrame({"b0": [1, 2, 3, 4]})

        assert frame.row_major_grid_cell(lf, window).collect()["grid_cell"].to_list() == [0, 1, 2, 3]

    def test_leaves_no_working_columns_behind(self):
        window = make_window(width=2, height=2, batch_layout=((0, 0, 2, 2),))
        lf = pl.LazyFrame({"b0": [1, 2, 3, 4]})

        assert frame.row_major_grid_cell(lf, window).collect().columns == ["b0", "grid_cell"]

    def test_a_decimated_window_is_numbered_over_the_pixels_actually_read(self):
        """A 4x4 window read at (2, 2) comes back as a 2x2 array, so the ids run 0..3 over
        THAT array. Numbering it over the window's 16 source pixels instead -- which
        `window.width` alone would do -- gives four rows the ids 0, 1, 4, 5, and the sink's
        reshape then scatters them across a raster four times too large.
        """
        window = make_window(width=4, height=4, decimation=(2, 2), batch_layout=())
        lf = pl.LazyFrame({"b0": [1, 2, 3, 4]})

        assert frame.row_major_grid_cell(lf, window).collect()["grid_cell"].to_list() == [0, 1, 2, 3]

    def test_a_decimated_window_refuses_absolute_grid_cell_ids(self):
        """The mode that would lie. A decimated row stands for a block of source pixels,
        one of which GDAL picked, so there is no absolute id over the full grid that names
        it honestly -- and an absolute id is exactly what a caller filters against.
        """
        window = make_window(width=4, height=4, decimation=(2, 2))
        lf = pl.LazyFrame({"b0": [1, 2, 3, 4]})

        with pytest.raises(ShapeError, match="decimated"):
            frame.row_major_grid_cell(lf, window, grid_width=100)

    def test_an_undecimated_window_still_allows_absolute_ids(self):
        # The refusal above is conditioned on decimation, not on the mode: a full-resolution
        # window must keep answering the absolute question it has always answered.
        window = make_window(col_off=1, row_off=1, width=2, height=2, batch_layout=((1, 1, 2, 2),))
        lf = pl.LazyFrame({"b0": [1, 2, 3, 4]})

        out = frame.row_major_grid_cell(lf, window, grid_width=10).collect()

        assert out["grid_cell"].to_list() == [11, 12, 21, 22]

    def test_a_decimated_window_carrying_a_batch_layout_is_a_contract_violation(self):
        """Mixed units: a batch layout is in source pixels and a decimated array is not, so
        a window claiming both cannot be decoded. `io.raster.scan_raster` never produces
        one (it records an empty layout for a decimated read, and refuses to decimate a
        batched full read at all); this is the guard that says so out loud.
        """
        window = make_window(width=4, height=4, decimation=(2, 2), batch_layout=((0, 0, 4, 4),))
        lf = pl.LazyFrame({"b0": [1, 2, 3, 4]})

        with pytest.raises(AssertionError, match="batch layout"):
            frame.row_major_grid_cell(lf, window)


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
        assert make_window() != make_window(decimation=(2, 2))

    def test_read_shape_is_the_source_extent_when_undecimated(self):
        assert make_window(width=10, height=7).read_shape == (7, 10)

    def test_read_shape_floor_divides_per_axis(self):
        # Per axis and floor-divided, both: 7 // 4 == 1 on the row axis while the column
        # axis is untouched, and neither rounds up. Ceil-division here would ask GDAL for
        # an effective ratio just under the factor, which makes it serve the read off the
        # next FINER overview level and undoes the saving the factor was chosen for.
        assert make_window(width=10, height=7, decimation=(4, 1)).read_shape == (1, 10)

    def test_read_shape_never_goes_below_one_pixel(self):
        # A sliver window under a large factor floor-divides to zero, which is not a shape
        # any read can return; the floor is 1 on both axes.
        assert make_window(width=3, height=1, decimation=(8, 8)).read_shape == (1, 1)

    def test_read_shape_is_not_a_field_so_it_cannot_drift_from_decimation(self):
        # Derived, not stored: rebuilding the same window with a different factor changes
        # the shape, and there is no way to construct one whose shape contradicts it.
        assert "read_shape" not in {field.name for field in dataclasses.fields(make_window())}
        assert make_window(width=8, height=8, decimation=(2, 2)).read_shape == (4, 4)


# ----------------------------------------------------------------------------------
# The two frame-shape type aliases exist and mean what the module says they mean
# ----------------------------------------------------------------------------------


def test_long_frame_alias_is_lazyframe():
    assert frame.LongFrame is pl.LazyFrame


def test_band_wide_frame_alias_matches_the_documented_triple():
    lf, metadata_df, window = pl.LazyFrame(), pl.DataFrame(), make_window()
    value: frame.BandWideFrame = (lf, metadata_df, window)
    assert value == (lf, metadata_df, window)
