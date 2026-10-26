"""Tests for `cpra.mp.data.analysis.io.raster_points.sample_raster_points` -- the raster
POINT reader, the long-shaped counterpart to `io/raster.py`'s band-wide `scan_raster`.

Fixtures are synthetic GeoTIFFs written directly via rasterio, but on a REAL grid's
geometry (`veg_grid_cell_v001`, 365x1052) rather than the 2x2 toys `test_raster_scan.py`
uses for the band-wide scan. That difference is forced, not stylistic: this reader derives
its sample coordinates from `GRIDS[grid_name]`'s own declared bounds -- that is the whole
point, since the `grid_cell` ids it decodes were computed against the same geometry -- so a
file whose transform does not match a real grid's has no meaning here.

The equivalence class below is the load-bearing one. Everything else this reader does is
only correct insofar as its output is interchangeable with the band-wide reader's, because
`execution/execute.py` substitutes one for the other and hands the result to the same
lowering either way.
"""

import numpy as np
import polars as pl
import pytest
import rasterio as rio
from rasterio.transform import from_bounds

from cpra.mp.data.analysis.execution.band_ops import sample_grid_cells_band_wide
from cpra.mp.data.analysis.io.raster import scan_raster
from cpra.mp.data.analysis.io.raster_points import sample_raster_points
from cpra.mp.data.grids import GRIDS

GRID = "veg_grid_cell_v001"

assert GRIDS[GRID].grid_type == "raster"


def _write_grid_tif(path, bands, nodata=None, dtype="int32"):
    """A GeoTIFF covering exactly `GRID`'s extent, one band per entry in `bands`.

    Each entry is a `{(row, col): value}` dict of pixels to plant over a zero fill -- the
    whole array is 383,980 pixels, and naming only the handful a test actually samples is
    what keeps the expected values readable.
    """
    grid = GRIDS[GRID]
    data = np.zeros((len(bands), grid.height, grid.width), dtype=dtype)
    for index, pixels in enumerate(bands):
        for (row, col), value in pixels.items():
            data[index, row, col] = value
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": len(bands),
        "height": grid.height,
        "width": grid.width,
        "crs": grid.crs,
        "transform": from_bounds(*grid.bounds, width=grid.width, height=grid.height),
        # TILED, because the reader groups sampled coordinates by block (see
        # `io.raster_io._sample_raster`) and an untiled GeoTIFF's blocks are whole rows --
        # which makes every block group one-dimensional and hides a row/column mix-up that
        # a real 2D block would catch. Production COGs are tiled 512x512; 128 keeps that
        # shape while still giving this 365x1052 grid several blocks on both axes.
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def _metadata(rows, years=None):
    """`(path, band)` pairs in band order, plus a `calendar_year` dimension column -- the
    shape `io/catalog.py::resolve_raster_bands` returns, minus the dimensions neither
    reader reads.
    """
    frame = pl.DataFrame(
        {"path": [str(p) for p, _ in rows], "band": [b for _, b in rows]},
        schema={"path": pl.String, "band": pl.Int64},
    )
    if years is not None:
        frame = frame.with_columns(pl.Series("calendar_year", years, dtype=pl.Int32))
    return frame


def _cell(row, col):
    return row * GRIDS[GRID].width + col


class TestEquivalenceWithTheBandWideReader:
    """The substitution `execution/execute.py` makes: for a gated origin it opens
    `sample_raster_points` where it would otherwise have opened `scan_raster` and unpivoted
    the result through `band_ops.sample_grid_cells_band_wide`. Those two have to produce the
    same frame -- not merely the same numbers, the same COLUMNS in the same order with the
    same dtypes -- because the walk then applies the identical `where` to whichever one it
    got, through the identical long lowering.
    """

    def test_the_two_readers_produce_the_same_frame(self, tmp_path):
        # Four pixels, all far from the raster's own origin so that a row/col transposition
        # or a window-relative id would land somewhere else and show up as a wrong VALUE.
        # Two of them -- (200, 800) and (205, 810) -- share a 128x128 block, and share it at
        # DIFFERENT row and column offsets within it, which is what makes the pair sensitive
        # to how the reader indexes back into a block it read once for both (see
        # `io.raster_io._sample_raster`); the other two land on blocks of their own.
        # Two files with two bands each, so the comparison also covers band ordering across
        # a file boundary -- the one place the point reader's per-file grouping could
        # scramble what the band-wide reader's flat VRT source list keeps in order.
        targets = {(200, 800): 11, (205, 810): 7, (11, 1051): 22, (364, 0): 33}
        path_a = _write_grid_tif(tmp_path / "a.tif", [targets, {pixel: value * 2 for pixel, value in targets.items()}])
        path_b = _write_grid_tif(tmp_path / "b.tif", [{pixel: value * 3 for pixel, value in targets.items()}, {}])
        metadata_df = _metadata([(path_a, 1), (path_a, 2), (path_b, 1), (path_b, 2)], years=[2030, 2040, 2050, 2060])
        cells = [_cell(row, col) for row, col in targets]
        where = f"grid_cell IN ({', '.join(str(cell) for cell in cells)})"

        band_wide = sample_grid_cells_band_wide(scan_raster(metadata_df), where, (), grid_width=GRIDS[GRID].width)
        # No `where` applied to the point reader's own output: it read exactly those pixels
        # and nothing else, which is the whole claim. The walk applies the `where` to both
        # in production; applying it here would only hide a reader that over-read.
        points = sample_raster_points(metadata_df, GRID, cells)

        expected = band_wide.collect().sort("grid_cell", "calendar_year")
        actual = points.collect().sort("grid_cell", "calendar_year")

        assert actual.columns == expected.columns
        assert actual.schema == expected.schema
        assert actual.equals(expected)
        # Not a vacuous comparison: the fixture really did put distinguishable numbers at
        # each (pixel, band), so an empty or constant frame could not pass the check above.
        assert expected.height == len(cells) * metadata_df.height
        assert sorted(expected["value"].unique().to_list()) == [0, 7, 11, 14, 21, 22, 33, 44, 66, 99]

    def test_the_reader_reads_only_the_named_pixels(self, tmp_path):
        # The reason the substitution exists. `scan_raster` over the same metadata yields
        # 383,980 rows (every pixel of the grid) before anything filters it; this reader
        # yields one per (named pixel, band) and never materializes the rest.
        path = _write_grid_tif(tmp_path / "a.tif", [{(200, 800): 7}])
        metadata_df = _metadata([(path, 1)], years=[2030])

        band_wide_lf, _, _ = scan_raster(metadata_df)
        assert band_wide_lf.select(pl.len()).collect().item() == GRIDS[GRID].height * GRIDS[GRID].width

        points = sample_raster_points(metadata_df, GRID, [_cell(200, 800)]).collect()
        assert points.height == 1
        assert points["value"].to_list() == [7]


class TestSampledValues:
    """What comes back, and in what order."""

    def test_band_order_follows_metadata_row_order_across_files(self, tmp_path):
        # The band-naming invariant, checked through a permutation rather than a count: the
        # same two files in the opposite metadata order must swap which `calendar_year` each
        # value is reported under. A reader that grouped by file and then forgot to map each
        # group's results back to its original position would pass a count assertion and
        # fail this one.
        path_a = _write_grid_tif(tmp_path / "a.tif", [{(5, 5): 11}])
        path_b = _write_grid_tif(tmp_path / "b.tif", [{(5, 5): 22}])
        cell = _cell(5, 5)

        forward = sample_raster_points(
            _metadata([(path_a, 1), (path_b, 1)], years=[2030, 2040]), GRID, [cell]
        ).collect()
        assert dict(zip(forward["calendar_year"], forward["value"])) == {2030: 11, 2040: 22}

        reversed_out = sample_raster_points(
            _metadata([(path_b, 1), (path_a, 1)], years=[2030, 2040]), GRID, [cell]
        ).collect()
        assert dict(zip(reversed_out["calendar_year"], reversed_out["value"])) == {2030: 22, 2040: 11}

    def test_repeated_and_unordered_ids_are_returned_as_given(self, tmp_path):
        # Two coordinates in a request can resolve to the SAME pixel
        # (`io.geography.resolve_raster_point_labels` maps one id to several labels for
        # exactly this case), and nothing sorts the id list on the way here. The reader
        # samples positionally, so a duplicate must come back twice and the order must be
        # the caller's, not ascending.
        path = _write_grid_tif(tmp_path / "a.tif", [{(5, 5): 11, (300, 900): 22}])
        low, high = _cell(5, 5), _cell(300, 900)
        cells = [high, low, high]

        out = sample_raster_points(_metadata([(path, 1)], years=[2030]), GRID, cells).collect()

        assert out["grid_cell"].to_list() == cells
        assert out["value"].to_list() == [22, 11, 22]

    def test_grid_cell_is_int64(self, tmp_path):
        # `io.geography.relabel_raster_grid_cell` joins a point query's result against an
        # `Int64` `grid_cell` column to swap the integer ids back for the caller's own
        # coordinate strings. A narrower integer type here would fail that join rather than
        # silently produce a wrong answer, but it would fail it deep inside the sink.
        path = _write_grid_tif(tmp_path / "a.tif", [{(5, 5): 11}])
        out = sample_raster_points(_metadata([(path, 1)], years=[2030]), GRID, [_cell(5, 5)])
        assert out.collect_schema()["grid_cell"] == pl.Int64

    def test_dtype_follows_the_band_rather_than_being_widened(self, tmp_path):
        # Each band reads back as the Polars dtype `io.raster.NUMPY_TO_POLARS` maps its
        # storage type to -- the same table `scan_raster` builds its own schema from, and
        # imported rather than copied so the two cannot drift.
        path = _write_grid_tif(tmp_path / "a.tif", [{(5, 5): 1.5}], dtype="float32")
        out = sample_raster_points(_metadata([(path, 1)], years=[2030]), GRID, [_cell(5, 5)]).collect()
        assert out.schema["value"] == pl.Float32
        assert out["value"].to_list() == pytest.approx([1.5])


class TestNodata:
    """`mask_nodata` -- the same two-state behaviour `scan_raster` has for a data band, read
    off the same declared `nodatavals`.
    """

    def test_a_declared_nodata_value_becomes_null(self, tmp_path):
        path = _write_grid_tif(tmp_path / "a.tif", [{(5, 5): -9999, (6, 6): 4}], nodata=-9999)
        metadata_df = _metadata([(path, 1)], years=[2030])
        cells = [_cell(5, 5), _cell(6, 6)]

        masked = sample_raster_points(metadata_df, GRID, cells).collect()
        assert masked["value"].to_list() == [None, 4]

        unmasked = sample_raster_points(metadata_df, GRID, cells, mask_nodata=False).collect()
        assert unmasked["value"].to_list() == [-9999, 4]

    def test_a_band_declaring_no_nodata_is_left_alone(self, tmp_path):
        # The sentinel is only a sentinel because the file says so. A band with no declared
        # nodata holds -9999 as an ordinary measurement, and masking it would invent a null.
        path = _write_grid_tif(tmp_path / "a.tif", [{(5, 5): -9999}])
        out = sample_raster_points(_metadata([(path, 1)], years=[2030]), GRID, [_cell(5, 5)]).collect()
        assert out["value"].to_list() == [-9999]

    def test_each_band_uses_its_own_declared_value(self, tmp_path):
        # Two files, two different sentinels. A reader that read one file's `nodata` and
        # applied it to every band would mask the wrong pixel here -- 0 is a legitimate
        # value in `a.tif` and the sentinel in `b.tif`.
        path_a = _write_grid_tif(tmp_path / "a.tif", [{(5, 5): -1}], nodata=-1)
        path_b = _write_grid_tif(tmp_path / "b.tif", [{(5, 5): 0}], nodata=0)
        metadata_df = _metadata([(path_a, 1), (path_b, 1)], years=[2030, 2040])

        out = sample_raster_points(metadata_df, GRID, [_cell(5, 5)]).collect()

        assert dict(zip(out["calendar_year"], out["value"])) == {2030: None, 2040: None}


class TestRejections:
    """Preconditions this reader states rather than guesses at."""

    def test_zero_bands_raises(self):
        with pytest.raises(ValueError, match="at least one band"):
            sample_raster_points(pl.DataFrame(), GRID, [0])

    def test_missing_path_or_band_column_raises(self):
        frame = pl.DataFrame({"path": ["/nowhere.tif"]})
        with pytest.raises(ValueError, match="'path' and 'band'"):
            sample_raster_points(frame, GRID, [0])

    def test_no_grid_cells_raises(self, tmp_path):
        path = _write_grid_tif(tmp_path / "a.tif", [{(5, 5): 11}])
        with pytest.raises(ValueError, match="at least one grid_cell"):
            sample_raster_points(_metadata([(path, 1)], years=[2030]), GRID, [])

    def test_an_id_off_the_grid_raises(self, tmp_path):
        # `execution.execute`'s gate range-checks ids before it picks this reader at all, so
        # reaching here with one means the two decodes disagree -- a bug, and a plain
        # `ValueError` rather than one of the request layer's errors.
        path = _write_grid_tif(tmp_path / "a.tif", [{(5, 5): 11}])
        grid = GRIDS[GRID]
        with pytest.raises(ValueError, match="does not name a pixel"):
            sample_raster_points(_metadata([(path, 1)], years=[2030]), GRID, [grid.height * grid.width])
