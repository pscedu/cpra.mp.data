"""The tile path's geometry: read the tile's footprint in the source CRS, do the per-pixel
work there, warp the result onto the tile grid last.

The tile path used to warp every source band onto the tile's pixel grid FIRST (a
`WarpedVRT` per source) and collapse the bands afterwards, on tile pixels. It now does it
the other way round, through `utils.tile_source_window` + `utils.warp_window_to_tile`.
This file exists to prove that change moved no pixel, and it has to do that with a
differential rather than an assertion about the code, because the failure mode is a
fractional-pixel offset that renders as a picture that still looks right.

Three things are measured separately here, and conflating them would hide which defect is
present:

* DISPLACEMENT -- both implementations have data at a pixel and the values differ. That is
  a shifted grid, the defect this whole file is about. It must be zero.
* EDGE GAP -- one has data where the other has nodata. Milder (a one-pixel erosion at the
  data boundary), but it is what decides `tile_source_window`'s clip-vs-pad design, so it
  is measured rather than assumed.
* The CONTROL -- the same new code with one deliberately wrong argument. Without it a
  zero-displacement result proves nothing: a differential over tiles that happen to be
  empty, or over a constant-valued raster, passes no matter what the code does. The source
  here therefore has a unique value in every pixel, the sweep asserts it actually compared
  a large number of valid pixels, and the control asserts the comparison can fail.
"""

import math
from contextlib import ExitStack

import mercantile
import numpy as np
import polars as pl
import pytest
import rasterio as rio
from rasterio import windows
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from cpra.mp.data.analysis.execution.band_ops import group_band_wide
from cpra.mp.data.analysis.io import raster_io as utils_module
from cpra.mp.data.analysis.io.catalog import resolve_raster_bands
from cpra.mp.data.analysis.io.raster import scan_raster
from cpra.mp.data.analysis.io.raster_io import (
    TILE_CRS,
    TILE_READ_BUFFER_PX,
    TILE_READ_OVERSAMPLE,
    _axis_decimation_factor,
    _warp_array_to_tile,
    raster_grid_geometry,
    resampling_buffer_px,
    tile_decimation_factors,
    tile_disjoint_from_source,
    tile_read_shape,
    tile_source_window,
    warp_window_to_tile,
)
from cpra.mp.data.analysis.io.window import ReadWindow
from cpra.mp.data.analysis.request.ops import GroupOp
from cpra.mp.data.analysis.request.pipeline import InputSelection
from cpra.mp.data.grids import GRIDS
from tests.factories import RASTER_KWARGS, raster_array, write_grid_geometry, write_raster_dataset

DIMENSIONS = {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}
GRID = GRIDS[RASTER_KWARGS["grid"]]
TILE_SIZE = 256
NODATA = -9999


@pytest.fixture
def unique_pixel_raster(base_path):
    """One COG whose every pixel holds a different value.

    `raster_array()` with no fill is `arange` over the grid, so any displacement at all
    shows up as a value mismatch. A uniform raster -- what most of this suite's fixtures
    use -- would make a shifted tile compare equal to an unshifted one everywhere except
    the data boundary, which is exactly the blind spot the control below guards against.
    """
    write_raster_dataset(base_path, years=(2030,), data_fn=lambda _year: raster_array())
    return next(base_path.rglob("*.tif"))


def _tile_centre():
    """Lon/lat at the middle of the raster grid's own extent."""
    west, south, east, north = transform_bounds(GRID.crs, "EPSG:4326", *GRID.bounds)
    return (west + east) / 2, (south + north) / 2


def _tile_bounds(tile):
    return BoundingBox(*mercantile.xy_bounds(tile))


def _warp_first(src, tile_bounds):
    """The pre-port geometry: warp the source straight onto the tile's pixel grid.

    A verbatim reproduction of what `get_tile` did before this change -- a `WarpedVRT` over
    the source dataset itself, with the tile's transform and size -- kept here as the "old"
    side of the differential rather than in production, where nothing calls it any more.
    """
    tile_transform = from_bounds(*tile_bounds, TILE_SIZE, TILE_SIZE)
    with WarpedVRT(
        src,
        crs=TILE_CRS,
        transform=tile_transform,
        width=TILE_SIZE,
        height=TILE_SIZE,
        nodata=NODATA,
        src_nodata=NODATA,
        add_alpha=False,
        dtype=src.dtypes[0],
        resampling=Resampling.nearest,
    ) as vrt:
        return vrt.read(1)


def _window_then_warp(src, tile_bounds):
    """The ported geometry, through the production helpers and nothing else."""
    window = tile_source_window(src, tile_bounds)
    if window is None:
        return np.full((TILE_SIZE, TILE_SIZE), NODATA, dtype=src.dtypes[0])
    data = src.read(1, window=window)
    return warp_window_to_tile(
        data[np.newaxis, :, :],
        window=window,
        src_transform=src.transform,
        src_crs=src.crs,
        tile_bounds=tile_bounds,
        tile_size=TILE_SIZE,
        nodata=NODATA,
    )[0]


def _window_then_warp_decimated(src, tile_bounds):
    """The production geometry INCLUDING the decimated read: `tile_read_shape` picks the
    shape, the read is decimated to it (GDAL serves that from an overview), and
    `warp_window_to_tile` is told what it was decimated to so it can scale the
    georeferencing to the array it is actually handed.

    Identical to `_window_then_warp` wherever `tile_read_shape` returns None.
    """
    window = tile_source_window(src, tile_bounds)
    if window is None:
        return np.full((TILE_SIZE, TILE_SIZE), NODATA, dtype=src.dtypes[0])
    out_shape = tile_read_shape(window, TILE_SIZE)
    data = src.read(1, window=window, out_shape=out_shape)
    return warp_window_to_tile(
        data[np.newaxis, :, :],
        window=window,
        src_transform=src.transform,
        src_crs=src.crs,
        tile_bounds=tile_bounds,
        tile_size=TILE_SIZE,
        nodata=NODATA,
        out_shape=out_shape,
    )[0]


def _window_then_warp_decimated_unscaled_transform(src, tile_bounds):
    """THE CONTROL FOR THE DECIMATED PATH. Identical to `_window_then_warp_decimated`
    except that the georeferencing handed to the warp is the WINDOW's transform -- the
    source's own resolution -- rather than one scaled to the decimated array.

    This is the specific mistake a decimated read makes available: the array is smaller
    than the window while covering the same ground, so the window's transform describes it
    at the wrong scale and the tile renders from the wrong corner of the source. Like the
    unrounded-transform control above it has no symptom of its own -- real values, right
    dtype, a plausible picture -- so without it a zero-displacement result for the decimated
    path would prove nothing.

    It reaches `_warp_array_to_tile` directly, because `warp_window_to_tile` derives the
    transform from the window AND the decimated shape and so cannot express this mistake.
    """
    window = tile_source_window(src, tile_bounds)
    if window is None:
        return np.full((TILE_SIZE, TILE_SIZE), NODATA, dtype=src.dtypes[0])
    out_shape = tile_read_shape(window, TILE_SIZE)
    data = src.read(1, window=window, out_shape=out_shape)
    return _warp_array_to_tile(
        data[np.newaxis, :, :],
        windows.transform(window, src.transform),
        src.crs,
        tile_bounds,
        TILE_SIZE,
        NODATA,
    )[0]


def _window_then_warp_unrounded_transform(src, tile_bounds):
    """THE CONTROL. Identical to `_window_then_warp` except that the georeferencing handed
    to the warp comes from the UNROUNDED float window rather than the whole-pixel window
    the data was actually read from.

    This is a one-argument difference with no symptom of its own -- the tile still renders,
    the values are all real source values, the extent is right. It must diverge, or the
    differential above is measuring nothing.

    It has to reach `_warp_array_to_tile` directly, because `warp_window_to_tile` derives
    the transform from the window itself and so cannot express this mistake.
    """
    src_bounds = transform_bounds(TILE_CRS, src.crs, *tile_bounds, densify_pts=21)
    float_window = windows.from_bounds(*src_bounds, transform=src.transform)
    window = tile_source_window(src, tile_bounds)
    if window is None:
        return np.full((TILE_SIZE, TILE_SIZE), NODATA, dtype=src.dtypes[0])
    data = src.read(1, window=window)
    return _warp_array_to_tile(
        data[np.newaxis, :, :],
        windows.transform(float_window, src.transform),
        src.crs,
        tile_bounds,
        TILE_SIZE,
        NODATA,
    )[0]


def _window_then_warp_padded(src, tile_bounds, buffer_px=TILE_READ_BUFFER_PX):
    """The variant `tile_source_window` deliberately does NOT implement: the window is
    rounded outward and buffered but never intersected with the source extent, and the read
    is `boundless=True` so GDAL pads past the edge with the nodata sentinel.

    Kept to measure the cost of that choice rather than assert it from a comment.
    """
    src_bounds = transform_bounds(TILE_CRS, src.crs, *tile_bounds, densify_pts=21)
    float_window = windows.from_bounds(*src_bounds, transform=src.transform)
    window = windows.Window(
        col_off=math.floor(float_window.col_off) - buffer_px,
        row_off=math.floor(float_window.row_off) - buffer_px,
        width=math.ceil(float_window.width) + 2 * buffer_px,
        height=math.ceil(float_window.height) + 2 * buffer_px,
    )
    data = src.read(1, window=window, boundless=True, fill_value=NODATA)
    return warp_window_to_tile(
        data[np.newaxis, :, :],
        window=window,
        src_transform=src.transform,
        src_crs=src.crs,
        tile_bounds=tile_bounds,
        tile_size=TILE_SIZE,
        nodata=NODATA,
    )[0]


def _score(reference, candidate):
    """`(displaced, edge_gaps, reference_valid)` for one tile -- the two failure modes kept
    apart, plus how many pixels the comparison actually had to work with."""
    has_reference = reference != NODATA
    has_candidate = candidate != NODATA
    both = has_reference & has_candidate
    displaced = int(((reference != candidate) & both).sum())
    gaps = int((has_reference ^ has_candidate).sum())
    return displaced, gaps, int(has_reference.sum())


def _sweep(src, zooms=range(9, 15), centre_lon_lat=None):
    """Every tile in a 3x3 neighbourhood of the grid's centre, at each zoom, that the
    pre-port implementation renders any data into at all.

    `centre_lon_lat` overrides the grid registry's centre, for a source that is not on one
    of the registered grids at all (`wide_unique_pixel_raster`).
    """
    lon, lat = centre_lon_lat if centre_lon_lat is not None else _tile_centre()
    for zoom in zooms:
        centre = mercantile.tile(lon, lat, zoom)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                bounds = _tile_bounds(mercantile.Tile(centre.x + dx, centre.y + dy, zoom))
                reference = _warp_first(src, bounds)
                if not (reference != NODATA).any():
                    continue
                yield bounds, reference


class TestPortedTileGeometryMatchesTheWarpFirstOrder:
    """The differential proper: window-then-warp against warp-first, on real pixel values,
    over every tile in a 3x3 neighbourhood of the grid centre at six zoom levels."""

    def test_no_pixel_is_displaced_and_no_edge_pixel_is_lost(self, unique_pixel_raster):
        displaced = gaps = valid = tiles = 0
        with rio.open(unique_pixel_raster) as src:
            for bounds, reference in _sweep(src):
                d, g, v = _score(reference, _window_then_warp(src, bounds))
                displaced += d
                gaps += g
                valid += v
                tiles += 1

        # The harness must have had something to compare. A sweep that silently skipped
        # every tile would otherwise report a perfect score.
        assert tiles >= 20, f"only {tiles} tiles overlapped the raster -- nothing was measured"
        assert valid > 100_000, f"only {valid} valid pixels compared -- nothing was measured"

        assert displaced == 0, f"{displaced} of {valid} pixels moved -- the tile grid is shifted"
        assert gaps == 0, f"{gaps} pixels gained or lost data at the boundary"

    def test_the_decimated_read_changes_nothing_on_a_grid_it_never_decimates(self, unique_pixel_raster):
        """The decimated variant's row of the same table.

        `veg_grid_cell_v001` is 1052x365, smaller on its short axis than the 528 pixels
        `tile_read_shape` needs before it will decimate anything, so every tile in this
        sweep reads its window whole and the production path is byte-for-byte what it was.
        That is the row worth pinning here -- decimation must not leak into the zooms it was
        never meant to touch. `TestDecimatedTileReads` measures the zooms it does.
        """
        displaced = gaps = valid = tiles = 0
        with rio.open(unique_pixel_raster) as src:
            for bounds, reference in _sweep(src):
                window = tile_source_window(src, bounds)
                assert tile_read_shape(window, TILE_SIZE) is None
                d, g, v = _score(reference, _window_then_warp_decimated(src, bounds))
                displaced += d
                gaps += g
                valid += v
                tiles += 1

        assert tiles >= 20 and valid > 100_000
        assert displaced == 0 and gaps == 0

    def test_the_unrounded_transform_control_diverges(self, unique_pixel_raster):
        """Without this the test above proves nothing.

        Using the float window's origin for a buffer read from the rounded window is a
        sub-pixel shift, so essentially every pixel that has data in both lands on a
        different source pixel.
        """
        displaced = valid = tiles = 0
        with rio.open(unique_pixel_raster) as src:
            for bounds, reference in _sweep(src):
                d, _gaps, v = _score(reference, _window_then_warp_unrounded_transform(src, bounds))
                displaced += d
                valid += v
                tiles += 1

        assert tiles >= 20
        assert displaced > valid * 0.9, (
            f"the control displaced only {displaced} of {valid} pixels -- this differential "
            "cannot detect a shifted grid, so the result above is not evidence"
        )

    def test_padding_past_the_source_extent_erodes_the_data_boundary(self, unique_pixel_raster):
        """Why `tile_source_window` intersects with the source extent instead of reading
        `boundless=True`.

        Padding introduces no displacement either -- the grid phase is the same -- but GDAL
        snaps the warp differently where the buffer it is handed says "nodata" rather than
        "nothing here", and the tiles that straddle the raster's edge lose pixels to it.
        """
        padded_gaps = clipped_gaps = padded_displaced = 0
        with rio.open(unique_pixel_raster) as src:
            for bounds, reference in _sweep(src):
                d, g, _v = _score(reference, _window_then_warp_padded(src, bounds))
                padded_displaced += d
                padded_gaps += g
                _d, g2, _v2 = _score(reference, _window_then_warp(src, bounds))
                clipped_gaps += g2

        assert padded_displaced == 0, "padding is an edge-gap question, not a displacement one"
        assert padded_gaps > 0, (
            "padding past the source extent was expected to erode the data boundary; if it "
            "no longer does, the clip in tile_source_window is no longer buying anything"
        )
        assert clipped_gaps == 0
        assert padded_gaps > clipped_gaps


#: Side, in pixels, of `wide_unique_pixel_raster`. Big enough that a tile covering the whole
#: raster decimates by 4 under `tile_read_shape` (which needs
#: `4 * TILE_READ_OVERSAMPLE * (256 + 2 * TILE_READ_BUFFER_PX)` = 2112 pixels on the shorter
#: axis), and a multiple of the 256-pixel block size so every overview level is whole.
WIDE_SIZE = 2176

#: Ground resolution of `wide_unique_pixel_raster`, in metres. Only its ratio to a tile's
#: own ground size matters here; 10 m over 2176 pixels puts the whole raster inside one tile
#: at z=7 and roughly one tile's own footprint at z=11, so one sweep over z=7..14 crosses
#: every decimation factor the rule produces for this raster -- 4, then 2, then 1.
WIDE_RES = 10.0

#: The zooms `wide_unique_pixel_raster` is swept over: wide enough to reach a factor of 4 at
#: the top and to be a no-op for several zooms at the bottom.
WIDE_ZOOMS = range(7, 15)


@pytest.fixture(scope="module")
def wide_unique_pixel_raster(tmp_path_factory):
    """A unique-per-pixel raster big enough that `tile_read_shape` actually decimates, with
    NEAREST overviews like the production COGs carry.

    `unique_pixel_raster` above cannot measure decimation at all: `veg_grid_cell_v001` is
    1052x365 pixels, so a tile's source window never reaches the 528 pixels per axis
    `tile_read_shape` requires before it will decimate anything, at any zoom. The production
    grid does -- `morph_pixel_v001` is 20674x8285 and a z=6 tile's window is 13107x8285 --
    and that is the case this file otherwise has no fixture for.

    The overviews matter as much as the size: `raster._cog_profile` writes ``resampling
    NEAREST``, so an overview pixel is a real source pixel rather than a blend, and that is
    what lets the offsets measured below be read as "which source pixel was picked" instead
    of "how much was it smoothed". Values are `row * WIDE_SIZE + col`, so every rendered
    pixel decodes straight back to the source pixel it came from.
    """
    path = tmp_path_factory.mktemp("wide") / "wide_unique.tif"
    centre_x = (GRID.bounds[0] + GRID.bounds[2]) / 2
    centre_y = (GRID.bounds[1] + GRID.bounds[3]) / 2
    half = WIDE_SIZE * WIDE_RES / 2
    with rio.open(
        path,
        "w",
        driver="GTiff",
        width=WIDE_SIZE,
        height=WIDE_SIZE,
        count=1,
        dtype="int32",
        crs=GRID.crs,
        transform=from_bounds(centre_x - half, centre_y - half, centre_x + half, centre_y + half, WIDE_SIZE, WIDE_SIZE),
        nodata=NODATA,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
    ) as dst:
        dst.write(np.arange(WIDE_SIZE * WIDE_SIZE, dtype="int32").reshape(WIDE_SIZE, WIDE_SIZE), 1)
        dst.build_overviews([2, 4, 8], Resampling.nearest)
    return path


def _wide_centre(src):
    """Lon/lat at the middle of a source's own extent."""
    west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return (west + east) / 2, (south + north) / 2


def _source_pixel_offsets(reference, candidate):
    """``(|drow|, |dcol|)`` in SOURCE pixels, per pixel both tiles rendered.

    Only meaningful on a raster whose values encode their own position, which is exactly
    what `wide_unique_pixel_raster` is for. A decimated read cannot be scored by value
    equality the way `_score` scores the full-resolution paths -- each tile pixel takes one
    DECIMATED source pixel, which is a different (equally real) source pixel from the one a
    full-resolution read would pick -- so what has to be bounded is how far away it is, not
    whether it is the same one.
    """
    both = (reference != NODATA) & (candidate != NODATA)
    reference_row, reference_col = np.divmod(reference[both].astype("int64"), WIDE_SIZE)
    candidate_row, candidate_col = np.divmod(candidate[both].astype("int64"), WIDE_SIZE)
    return np.abs(candidate_row - reference_row), np.abs(candidate_col - reference_col)


class TestDecimatedTileReads:
    """The read volume a tile costs, and what decimating it does to the picture.

    Both tile paths used to read their source window at FULL resolution and warp afterwards
    -- 434 MB per band for the 65,536 pixels of one z=6 tile on `morph_pixel_v001`, and on
    the paired path the whole Polars collapse over every one of those pixels. `tile_read_shape`
    decimates that read onto an overview instead. These tests pin the volume property that
    change exists for, and bound the geometric cost of getting it.
    """

    def test_the_read_is_bounded_by_the_tile_rather_than_the_zoom(self, wide_unique_pixel_raster):
        """The whole point of the fix, and the one property nothing else here pins: what a
        tile reads must not grow without limit as the zoom goes out.

        The bound is a constant -- `tile_read_shape` never returns more than twice its own
        target per axis, because the factor is the largest power of two at or below the
        exact ratio -- so it holds at any zoom, on any source, for any window.
        """
        limit = 2 * TILE_READ_OVERSAMPLE * (TILE_SIZE + 2 * TILE_READ_BUFFER_PX)
        best_saving = 0
        with rio.open(wide_unique_pixel_raster) as src:
            lon, lat = _wide_centre(src)
            for zoom in WIDE_ZOOMS:
                centre = mercantile.tile(lon, lat, zoom)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        tile = mercantile.Tile(centre.x + dx, centre.y + dy, zoom)
                        window = tile_source_window(src, _tile_bounds(tile))
                        if window is None:
                            continue
                        out_shape = tile_read_shape(window, TILE_SIZE)
                        read_height, read_width = out_shape if out_shape else (window.height, window.width)
                        assert read_height <= limit and read_width <= limit, (
                            f"z={zoom} reads {read_height}x{read_width}, past the {limit}px bound -- "
                            "a tile read is growing with the zoom again"
                        )
                        best_saving = max(best_saving, (window.height * window.width) / (read_height * read_width))

        # ... and it is actually buying something at the outermost zoom, rather than the
        # bound above holding vacuously because nothing was ever big enough to decimate.
        assert best_saving >= 8, f"the widest window read 1/{best_saving:.1f} of its pixels -- nothing was saved"

    def test_no_rendered_pixel_moves_further_than_the_decimation_factor(self, wide_unique_pixel_raster):
        """Displacement, in the units it is actually bounded in.

        A decimated tile pixel is a decimated source pixel, so it is NOT the same pixel a
        full-resolution read picks and value equality (what `_score` measures) is expected to
        differ. What must hold is that it comes from within the decimation factor's own
        neighbourhood -- i.e. that the georeferencing tracked the decimation rather than a
        grid shift creeping in with it.
        """
        worst = 0
        decimating_tiles = compared = 0
        with rio.open(wide_unique_pixel_raster) as src:
            for bounds, reference in _sweep(src, zooms=WIDE_ZOOMS, centre_lon_lat=_wide_centre(src)):
                window = tile_source_window(src, bounds)
                out_shape = tile_read_shape(window, TILE_SIZE)
                if out_shape is None:
                    continue
                decimating_tiles += 1
                # Each axis is decimated by its own factor, so each axis is bounded by its
                # own -- a sliver window decimated 4x vertically and not at all horizontally
                # may not move a pixel sideways.
                height_factor = round(window.height / out_shape[0])
                width_factor = round(window.width / out_shape[1])
                rows, cols = _source_pixel_offsets(reference, _window_then_warp_decimated(src, bounds))
                compared += rows.size
                worst = max(worst, int(rows.max()), int(cols.max()))
                assert rows.max() <= 2 * height_factor and cols.max() <= 2 * width_factor, (
                    f"a pixel came from ({rows.max()}, {cols.max()}) source pixels away at "
                    f"decimation ({height_factor}, {width_factor}) -- further than decimating "
                    "alone accounts for"
                )

        assert decimating_tiles >= 4, f"only {decimating_tiles} tiles decimated -- nothing was measured"
        assert compared > 100_000, f"only {compared} pixels compared -- nothing was measured"
        # Measured: 4 source pixels, at a factor of 4, which at these zooms is a small
        # fraction of one tile pixel. `TILE_READ_OVERSAMPLE` is what keeps it there.
        assert worst <= 4

    def test_the_unscaled_transform_control_diverges(self, wide_unique_pixel_raster):
        """Without this the test above proves nothing.

        Handing the warp the WINDOW's transform for a decimated array is a whole-grid scale
        error, so the source pixel each tile pixel lands on is hundreds of pixels from the
        right one -- two orders of magnitude past the bound asserted above.
        """
        worst = 0
        decimating_tiles = 0
        with rio.open(wide_unique_pixel_raster) as src:
            for bounds, reference in _sweep(src, zooms=WIDE_ZOOMS, centre_lon_lat=_wide_centre(src)):
                window = tile_source_window(src, bounds)
                if tile_read_shape(window, TILE_SIZE) is None:
                    continue
                decimating_tiles += 1
                rows, cols = _source_pixel_offsets(
                    reference, _window_then_warp_decimated_unscaled_transform(src, bounds)
                )
                worst = max(worst, int(max(rows.max(), cols.max())))

        assert decimating_tiles >= 4
        assert worst > 100, (
            f"the control moved a pixel only {worst} source pixels -- this differential cannot "
            "detect a mis-scaled transform, so the bound above is not evidence"
        )

    def test_where_it_does_not_decimate_it_is_bit_identical(self, wide_unique_pixel_raster):
        """The other half: at the zooms whose windows are already at tile resolution
        (z >~ 12, the majority of real traffic) the decimated path must not merely be close
        to the current one, it must be the same bytes."""
        identical_tiles = 0
        with rio.open(wide_unique_pixel_raster) as src:
            for bounds, _reference in _sweep(src, zooms=WIDE_ZOOMS, centre_lon_lat=_wide_centre(src)):
                window = tile_source_window(src, bounds)
                if tile_read_shape(window, TILE_SIZE) is not None:
                    continue
                identical_tiles += 1
                assert np.array_equal(_window_then_warp_decimated(src, bounds), _window_then_warp(src, bounds))

        assert identical_tiles >= 20, f"only {identical_tiles} tiles read whole -- nothing was compared"

    def test_a_sliver_window_decimates_only_its_long_axis(self):
        """A hand-computed regression for the property `tile_read_shape`'s own docstring
        names directly: "a single factor taken from the shorter axis would leave the read
        unbounded on the longer one". A window thin on one axis and long on the other --
        exactly what a tile straddling the source's own edge produces, per that same
        docstring -- must decimate only the long axis; nothing in `wide_unique_pixel_raster`
        (square, so both axes are always the same length) can exercise that asymmetry at
        all, which is why this is a direct, fixture-free unit test on the function instead.

        `target = TILE_READ_OVERSAMPLE * (TILE_SIZE + 2 * TILE_READ_BUFFER_PX)` = ``2 * (256
        + 8)`` = 528 pixels. The short axis (100px) is already under that, so
        `_axis_decimation_factor` returns 1 (read whole) for it. The long axis (3000px)
        gives ``ratio = 3000 // 528 = 5``, and the largest power of two at or below 5 is 4
        (``1 << (5.bit_length() - 1)`` = ``1 << 2``) -- so it decimates by 4, to
        ``3000 // 4 = 750``.
        """
        window = windows.Window(col_off=0, row_off=0, width=3000, height=100)

        assert _axis_decimation_factor(window.height, 528) == 1
        assert _axis_decimation_factor(window.width, 528) == 4
        assert tile_decimation_factors(window, TILE_SIZE) == (1, 4)
        assert tile_read_shape(window, TILE_SIZE) == (100, 750)

    @pytest.mark.parametrize("extent", [(297, 226), (3000, 100), (20674, 8285), (100, 100), (1, 1)])
    @pytest.mark.parametrize("tile_size", [16, TILE_SIZE])
    def test_a_read_windows_shape_agrees_with_the_rule_that_sized_it(self, extent, tile_size):
        """The two halves of the decimation contract, computed in two modules: the shape
        `tile_read_shape` says a window will come back as, and the shape
        `io.window.ReadWindow` later claims those same factors produced. They are what the
        scan's `out_shape` and the sink's reshape are respectively derived from, so a
        one-pixel disagreement is a reshape failure at best and silently rotated
        georeferencing at worst.
        """
        width, height = extent
        window = windows.Window(0, 0, width, height)
        factors = tile_decimation_factors(window, tile_size)
        read_window = ReadWindow(
            source=(),
            col_off=0,
            row_off=0,
            width=width,
            height=height,
            transform=Affine.identity(),
            crs=TILE_CRS,
            decimation=factors,
        )

        expected = tile_read_shape(window, tile_size)
        if expected is None:
            assert factors == (1, 1)
            assert read_window.read_shape == (height, width)
        else:
            assert read_window.read_shape == expected


class TestWarpWindowToTileRejectsAMismatchedWindow:
    """`warp_window_to_tile` derives the warp's georeferencing from the window it is given,
    so a window that does not describe the array is a wrong answer rather than an error --
    unless it is caught. It is caught."""

    def test_wrong_shape_raises(self, unique_pixel_raster):
        with rio.open(unique_pixel_raster) as src:
            bounds = _tile_bounds(mercantile.tile(*_tile_centre(), 11))
            window = tile_source_window(src, bounds)
            data = src.read(1, window=window)[np.newaxis, :, :]
            bigger = windows.Window(window.col_off, window.row_off, window.width + 1, window.height)
            with pytest.raises(ValueError, match="must be exactly the window that was read"):
                warp_window_to_tile(
                    data,
                    window=bigger,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    tile_bounds=bounds,
                    tile_size=TILE_SIZE,
                    nodata=NODATA,
                )


class TestTileSourceWindow:
    def test_a_tile_far_from_the_raster_has_no_window(self, unique_pixel_raster):
        """A tile over Antarctica: the reprojected footprint is finite but misses every
        source pixel, so there is no window to read."""
        with rio.open(unique_pixel_raster) as src:
            assert tile_source_window(src, _tile_bounds(mercantile.Tile(1, 3, 2))) is None

    def test_a_whole_world_tile_falls_back_to_the_full_extent(self, unique_pixel_raster):
        """z=0 covers the whole planet, and the window still has to come back as the
        source's own extent rather than something larger or an error.

        It gets there by CLIPPING, not by the non-finite fallback, and the distinction is
        worth stating because the name suggests otherwise: this PROJ reprojects the whole
        Web Mercator world into `epsg:26915` as finite (if enormous) bounds -- it clamps
        out-of-domain coordinates rather than returning an infinity -- so the window is
        simply bigger than the source on every side and the clip to `src.width`/`src.height`
        is what produces the assertion below. `tile_source_window`'s `math.isfinite` guard is
        a belt-and-braces path for a CRS pair or PROJ build where that clamping does not
        happen; nothing in this suite reaches it, and it is deliberately kept rather than
        deleted on the strength of one PROJ version's behaviour.
        """
        with rio.open(unique_pixel_raster) as src:
            window = tile_source_window(src, _tile_bounds(mercantile.Tile(0, 0, 0)))
            assert (window.col_off, window.row_off) == (0, 0)
            assert (window.width, window.height) == (src.width, src.height)

    def test_the_window_is_whole_pixels_inside_the_source(self, unique_pixel_raster):
        with rio.open(unique_pixel_raster) as src:
            window = tile_source_window(src, _tile_bounds(mercantile.tile(*_tile_centre(), 12)))
            for value in (window.col_off, window.row_off, window.width, window.height):
                assert value == int(value)
            assert window.col_off >= 0 and window.row_off >= 0
            assert window.col_off + window.width <= src.width
            assert window.row_off + window.height <= src.height


class TestRasterGridGeometryStandsInForAnOpenedRaster:
    """`raster_grid_geometry` lets `execution.plan.compile_plan` size a tile's read window
    without opening a raster, which is only sound if the grid's own geometry and a real
    raster on that grid are the same four numbers. That is an assumption about how this
    repo's rasters are written, not a law, so it is measured here rather than asserted in a
    docstring: every window `compile_plan` hands the reader is wrong by however much these
    two disagree.
    """

    def test_the_four_properties_match_a_real_raster_on_that_grid(self, unique_pixel_raster):
        geometry = raster_grid_geometry(RASTER_KWARGS["grid"])
        with rio.open(unique_pixel_raster) as src:
            assert geometry.crs == src.crs
            assert geometry.transform == src.transform
            assert (geometry.width, geometry.height) == (src.width, src.height)

    def test_it_produces_the_same_window_as_the_opened_raster(self, unique_pixel_raster):
        # The property the compiler actually depends on, one level up from the four fields:
        # sizing a window off the grid and sizing it off the file agree exactly -- including
        # the buffer and clip arithmetic, which is what would amplify a small disagreement
        # in the transform into a differently-offset read.
        geometry = raster_grid_geometry(RASTER_KWARGS["grid"])
        for zoom in (6, 9, 12):
            bounds = _tile_bounds(mercantile.tile(*_tile_centre(), zoom))
            with rio.open(unique_pixel_raster) as src:
                assert tile_source_window(geometry, bounds) == tile_source_window(src, bounds)


class TestTileDisjointFromSource:
    """The cheap "nowhere near" test, which `tile_source_window` cannot make on its own
    because it reprojects the other way (see `tile_disjoint_from_source`'s docstring)."""

    def test_a_tile_over_the_raster_is_not_disjoint(self, unique_pixel_raster):
        bounds = _tile_bounds(mercantile.tile(*_tile_centre(), 12))
        with rio.open(unique_pixel_raster) as src:
            assert tile_disjoint_from_source(src, bounds) is False

    def test_a_low_zoom_tile_on_the_far_side_of_the_world_is_disjoint(self, unique_pixel_raster):
        # The case this function exists for, and the one `tile_source_window` gets wrong on
        # its own: the eastern/southern quadrant of the z=1 split is nowhere near a Gulf
        # Coast raster, but reprojecting that footprint INTO epsg:26915 does not come back
        # with usable bounds, so `tile_source_window` falls back to the source's full extent
        # -- a read of every pixel for a tile that needs none of them.
        bounds = _tile_bounds(mercantile.Tile(1, 1, 1))
        with rio.open(unique_pixel_raster) as src:
            assert tile_disjoint_from_source(src, bounds) is True
            fallback = tile_source_window(src, bounds)
            assert (fallback.width, fallback.height) == (src.width, src.height)

    def test_the_grid_geometry_gives_the_same_answer_as_the_file(self, unique_pixel_raster):
        geometry = raster_grid_geometry(RASTER_KWARGS["grid"])
        with rio.open(unique_pixel_raster) as src:
            for tile in (mercantile.tile(*_tile_centre(), 12), mercantile.Tile(1, 1, 1)):
                bounds = _tile_bounds(tile)
                assert tile_disjoint_from_source(geometry, bounds) == tile_disjoint_from_source(src, bounds)


class TestStraddlingFeatureWidensTheWindow:
    """A raster feeding a raster-to-vector `project` has to read past its tile's own
    footprint when a polygon on the OUTPUT grid straddles the tile edge -- the join needs
    every pixel under that polygon, including the part on the far side of the tile
    boundary, or those pixels are silently missing from the aggregate. `compile_plan`
    decides whether a given raster origin feeds such a project (`TestOverviewEligibility`'s
    sibling detection pass, in `test_execution_plan.py`, covers that decision); this class
    covers what `tile_source_window` does once told the answer is yes -- three CRSs
    involved, named explicitly rather than left implicit: #1 the tile's own bounds, always
    Web Mercator; #2 the raster source's CRS; #3 the vector grid's own CRS, which is a
    DIFFERENT grid than the raster source's (`hydro_compartment_v001`, not
    `veg_grid_cell_v001`) even though the current registry happens to give every grid the
    same crs (epsg:26915) -- so this test cannot distinguish "reprojected correctly" from
    "never reprojected at all". `TestTileSourceWindow` and `TestResamplingBufferSizing`
    above already cover CRS #1 -> #2 on its own; nothing here re-proves that step, only the
    union straddling adds on top of it.
    """

    def test_a_feature_extending_past_the_tile_widens_the_window(self, unique_pixel_raster, geometry_path):
        from shapely.geometry import box

        with rio.open(unique_pixel_raster) as src:
            bounds = _tile_bounds(mercantile.tile(*_tile_centre(), 12))
            plain_window = tile_source_window(src, bounds, resampling=Resampling.nearest)

            # Independently re-derive the tile's footprint in the raster's own CRS (CRS #1
            # -> CRS #2) -- the same reprojection `tile_source_window` itself performs, but
            # recomputed here directly rather than trusted, so this test's expectation
            # doesn't depend on that step's own correctness.
            src_bounds = transform_bounds(TILE_CRS, src.crs, *bounds, densify_pts=21)

            # veg_grid_cell_v001's own pixels are exactly 480m square ((909670-404710) /
            # 1052 == (3374680-3199480) / 365 == 480), so a feature bounding box built from
            # a whole multiple of 480m lands on an exact pixel boundary -- the widened
            # window's edge is then a single, exactly-predictable integer pixel count away
            # from the plain one, not merely "some amount wider".
            pixel_size = 480
            extension_px = 3
            extension = extension_px * pixel_size
            feature = box(src_bounds[0] - extension, src_bounds[1], src_bounds[2], src_bounds[3])
            write_grid_geometry(geometry_path, "hydro_compartment_v001", [(1, feature)])

            widened_window = tile_source_window(
                src, bounds, resampling=Resampling.nearest, straddling_vector_grid="hydro_compartment_v001"
            )

            # The feature only extends the WEST edge, so only the column offset/width move;
            # the east edge and both row bounds are untouched, and move by exactly the
            # extension in whole pixels -- not "wider", exactly 3 pixels wider.
            assert widened_window.col_off == plain_window.col_off - extension_px
            assert widened_window.width == plain_window.width + extension_px
            assert widened_window.row_off == plain_window.row_off
            assert widened_window.height == plain_window.height

    def test_a_feature_nowhere_near_the_tile_leaves_the_window_unchanged(self, unique_pixel_raster, geometry_path):
        """The control: a feature written to the same grid, but far from this tile, must
        contribute nothing -- without this, a widened-window result above would not
        distinguish "the union logic works" from "any feature at all widens the window"."""
        from shapely.geometry import box

        with rio.open(unique_pixel_raster) as src:
            bounds = _tile_bounds(mercantile.tile(*_tile_centre(), 12))
            plain_window = tile_source_window(src, bounds, resampling=Resampling.nearest)

            west, south, _east, _north = GRID.bounds
            far_feature = box(west, south, west + 480, south + 480)
            write_grid_geometry(geometry_path, "hydro_compartment_v001", [(1, far_feature)])

            unchanged_window = tile_source_window(
                src, bounds, resampling=Resampling.nearest, straddling_vector_grid="hydro_compartment_v001"
            )

            assert unchanged_window == plain_window

    def test_no_grid_given_is_unaffected_by_any_geometry_file_present(self, unique_pixel_raster, geometry_path):
        """`straddling_vector_grid=None` (the default -- every caller that never mentions
        it, matching `resampling`'s own default-unchanged property above) must not even
        look at a geometry file, regardless of what it contains."""
        from shapely.geometry import box

        with rio.open(unique_pixel_raster) as src:
            bounds = _tile_bounds(mercantile.tile(*_tile_centre(), 12))
            src_bounds = transform_bounds(TILE_CRS, src.crs, *bounds, densify_pts=21)
            feature = box(src_bounds[0] - 1440, src_bounds[1], src_bounds[2], src_bounds[3])
            write_grid_geometry(geometry_path, "hydro_compartment_v001", [(1, feature)])

            no_grid_window = tile_source_window(src, bounds, resampling=Resampling.nearest)
            plain_window = tile_source_window(src, bounds, resampling=Resampling.nearest, straddling_vector_grid=None)

            assert no_grid_window == plain_window


def _window_then_warp_resampled(src, tile_bounds, resampling, buffer_px=None, decimate=False):
    """`_window_then_warp`, with `resampling`/`buffer_px` threaded through instead of
    hardcoding nearest -- the ported geometry as `resampling` becoming a real parameter
    left it, through the production helpers and nothing else.

    `decimate=True` additionally runs the read through `tile_read_shape` the way
    `_window_then_warp_decimated` does, which is what a low-zoom tile actually gets; it is
    off by default so the buffer differentials above stay full-resolution comparisons.
    """
    window = tile_source_window(src, tile_bounds, buffer_px=buffer_px, resampling=resampling)
    if window is None:
        return np.full((TILE_SIZE, TILE_SIZE), NODATA, dtype=src.dtypes[0])
    out_shape = tile_read_shape(window, TILE_SIZE) if decimate else None
    data = src.read(1, window=window, out_shape=out_shape)
    return warp_window_to_tile(
        data[np.newaxis, :, :],
        window=window,
        src_transform=src.transform,
        src_crs=src.crs,
        tile_bounds=tile_bounds,
        tile_size=TILE_SIZE,
        nodata=NODATA,
        resampling=resampling,
        out_shape=out_shape,
    )[0]


class TestResamplingBufferSizing:
    """resampling as a parameter, not a hardcoded `Resampling.nearest`: `tile_source_window`
    has to read a wider buffer around a tile's footprint for a resampling method whose
    kernel samples more than one pixel, or the warp runs off the edge of the array it was
    given -- silently, since rasterio simply has no neighbour there to sample and the
    output is a value, not an error.
    """

    def test_nearest_reproduces_the_original_buffer_exactly(self):
        # The property that keeps every caller that never mentions `resampling` at all --
        # every other test in this file among them -- byte-for-byte unchanged by this
        # parameter existing.
        assert resampling_buffer_px(Resampling.nearest) == TILE_READ_BUFFER_PX

    @pytest.mark.parametrize(
        "resampling,kernel_radius",
        [
            (Resampling.bilinear, 1),
            (Resampling.cubic, 2),
            (Resampling.cubic_spline, 2),
            (Resampling.lanczos, 3),
        ],
    )
    def test_every_other_resampling_adds_its_kernel_radius_on_top_of_the_original_margin(
        self, resampling, kernel_radius
    ):
        assert resampling_buffer_px(resampling) == TILE_READ_BUFFER_PX + kernel_radius

    def test_an_unsupported_resampling_method_is_rejected_rather_than_silently_underbuffered(self):
        with pytest.raises(ValueError, match="unsupported resampling"):
            resampling_buffer_px(Resampling.average)

    def test_a_wider_kernel_widens_tile_source_windows_own_buffer(self, unique_pixel_raster):
        """`tile_source_window`'s own contribution, one level below the warp: the window it
        hands back for `Resampling.cubic` must be exactly `nearest`'s window widened by
        cubic's own extra kernel radius (2px) on every side -- not merely "some amount
        wider", which a bug that widened the wrong axis or only one side could still pass.
        """
        with rio.open(unique_pixel_raster) as src:
            bounds = _tile_bounds(mercantile.tile(*_tile_centre(), 12))
            nearest_window = tile_source_window(src, bounds, resampling=Resampling.nearest)
            cubic_window = tile_source_window(src, bounds, resampling=Resampling.cubic)

            # Neither window may be touching the source's own extent, or the clip in
            # `tile_source_window` (rather than the buffer difference under test) could be
            # what's actually bounding one side -- `test_the_window_is_whole_pixels_inside_
            # the_source` above already established z=12 at the grid centre is interior;
            # this re-checks it explicitly rather than assuming it silently carries over.
            assert nearest_window.col_off > 0 and nearest_window.row_off > 0
            assert nearest_window.col_off + nearest_window.width < src.width
            assert nearest_window.row_off + nearest_window.height < src.height

            delta = resampling_buffer_px(Resampling.cubic) - resampling_buffer_px(Resampling.nearest)
            assert delta == 2
            assert cubic_window.col_off == nearest_window.col_off - delta
            assert cubic_window.row_off == nearest_window.row_off - delta
            assert cubic_window.width == nearest_window.width + 2 * delta
            assert cubic_window.height == nearest_window.height + 2 * delta

    def test_an_underbuffered_bilinear_read_diverges_from_a_generously_buffered_one_but_the_derived_buffer_does_not(
        self, unique_pixel_raster
    ):
        """The differential proper, on rendered pixel VALUES rather than the window alone --
        `Resampling.bilinear` blends a tile-edge pixel from its source neighbours, so a read
        whose buffer is too narrow to contain them samples the wrong (or, at the array's own
        edge, a nonexistent) neighbour instead, a sub-pixel value error nearest's own
        differential above (`TestPortedTileGeometryMatchesTheWarpFirstOrder`) cannot see --
        that one is byte-equality against a NEAREST reference, and bilinear disagrees with
        nearest everywhere it blends, not just where the buffer runs short.

        Measured empirically before this test was written (a 6-zoom, 3x3-tile sweep of
        `unique_pixel_raster`, matching `_sweep`'s own footprint): a buffer of 0 disagreed
        with a generously-buffered (20px) reference at 59,334 pixels; `resampling_buffer_px`'s
        own derived buffer (5px for bilinear) disagreed at exactly 0.

        SCOPE, and it is narrower than "the derived buffer is sufficient": every zoom
        `_sweep` covers UPSAMPLES here. `veg_grid_cell_v001` is 480 m, and zooms 9 through 14
        put a tile pixel at 265.6 m down to 8.3 m, all finer than the source, so this fixture
        cannot reach the downsampling case at any zoom. The derived buffer is NOT sufficient
        there -- `test_the_derived_buffer_is_not_sufficient_where_the_warp_downsamples` below
        measures that boundary on a fixture that does reach it.
        """
        generous_disagreement = derived_disagreement = 0
        with rio.open(unique_pixel_raster) as src:
            for bounds, _reference in _sweep(src):
                generous = _window_then_warp_resampled(src, bounds, Resampling.bilinear, buffer_px=20)
                zero_buffered = _window_then_warp_resampled(src, bounds, Resampling.bilinear, buffer_px=0)
                derived = _window_then_warp_resampled(src, bounds, Resampling.bilinear)  # buffer_px=None

                both_zero = (generous != NODATA) & (zero_buffered != NODATA)
                both_derived = (generous != NODATA) & (derived != NODATA)
                generous_disagreement += int(((generous != zero_buffered) & both_zero).sum())
                derived_disagreement += int(((generous != derived) & both_derived).sum())

        # The control: without this, a derived_disagreement of 0 would prove nothing -- a
        # differential that can never disagree with its own reference detects nothing.
        assert generous_disagreement > 0, (
            f"the control (buffer_px=0) disagreed with the generous reference at only "
            f"{generous_disagreement} pixels -- this differential cannot detect an "
            "under-sized buffer, so the result below is not evidence"
        )
        assert derived_disagreement == 0, (
            f"{derived_disagreement} pixels rendered under resampling_buffer_px's own "
            "derived buffer still disagree with the generously-buffered reference"
        )

    def test_the_derived_buffer_is_not_sufficient_where_the_warp_downsamples(self, wide_unique_pixel_raster):
        """The boundary of the property above, pinned as a measurement rather than left to
        be discovered by whoever first renders a non-nearest tile at low zoom.

        `resampling_buffer_px` sizes its kernel radius in SOURCE pixels, so it covers a warp
        that upsamples and undercounts one that downsamples by `r` -- each destination pixel
        then draws on roughly `r` source pixels per kernel tap. Two adjacent zooms on either
        side of that line make the case without any appeal to the mechanism, since the only
        thing that changes between them is the ratio:

        * z=14 puts a tile pixel at 8.3 m against this fixture's 10 m source (0.8x, still
          upsampling) -- the derived buffer agrees with a generous one exactly.
        * z=13 is 16.6 m (1.7x, downsampling) -- essentially every rendered pixel moves.

        `Resampling.nearest` over those same windows, at that same zoom and the same
        decimation factor, is the control: radius 0 has nothing to sample around, so it must
        be unaffected. It is, which is what makes the kernel -- rather than the window, the
        decimation, or the georeferencing, all of which z=13 changes identically for both --
        the thing being measured. That is also why `Resampling.nearest`, the only value the
        tile path uses today, is correct at every zoom.
        """
        counts = {}
        with rio.open(wide_unique_pixel_raster) as src:
            for zoom in (13, 14):
                for resampling in (Resampling.nearest, Resampling.bilinear):
                    disagreement = compared = 0
                    for bounds, _reference in _sweep(src, zooms=[zoom], centre_lon_lat=_wide_centre(src)):
                        derived = _window_then_warp_resampled(src, bounds, resampling, decimate=True)
                        generous = _window_then_warp_resampled(src, bounds, resampling, buffer_px=20, decimate=True)
                        both = (derived != NODATA) & (generous != NODATA)
                        disagreement += int(((derived != generous) & both).sum())
                        compared += int(both.sum())
                    counts[(zoom, resampling)] = (disagreement, compared)

        for (zoom, resampling), (_disagreement, compared) in counts.items():
            assert compared > 100_000, (
                f"z={zoom} {resampling.name} compared only {compared} pixels -- too few for "
                "the counts below to mean anything"
            )

        # The control, both sides of the line: nearest has no kernel to run off the end of.
        assert counts[(13, Resampling.nearest)][0] == 0
        assert counts[(14, Resampling.nearest)][0] == 0

        # Upsampling: the derived buffer is sufficient, matching the differential above.
        assert counts[(14, Resampling.bilinear)][0] == 0

        # Downsampling: it is not, and not marginally -- this is the known limitation
        # `resampling_buffer_px`'s own docstring records, not a tolerance to be widened.
        downsampled_disagreement, downsampled_compared = counts[(13, Resampling.bilinear)]
        assert downsampled_disagreement > downsampled_compared // 2, (
            f"only {downsampled_disagreement} of {downsampled_compared} pixels moved -- if "
            "the derived buffer has become sufficient under a downsampling warp, this test "
            "and resampling_buffer_px's docstring both need rewriting, not relaxing"
        )


def _resolve_metadata(dimensions, base_path):
    """`InputSelection` -> `metadata_df`, the first new-module step any real tile read
    starts from -- `io.catalog.resolve_raster_bands` against the sandboxed `base_path`,
    never the real data tree.
    """
    return resolve_raster_bands(InputSelection(**dimensions), base_path=base_path)


def _render_tile(dimensions, base_path, tile_bounds, *, tile_size=TILE_SIZE, aggregate=None):
    """The new-module tile path end to end: resolve the dataset's bands, compute the
    source window (via `utils_module.tile_source_window`, called by attribute so a spy
    patched onto that module still takes effect here -- see the module-qualified-lookup
    rule this suite relies on), scan exactly that window, optionally collapse it across
    bands on SOURCE pixels, then warp the single surviving band onto the tile grid (via
    `utils_module.warp_window_to_tile`, same reason).

    Composed entirely from `request`/`io`/`execution` primitives, so that what this
    file measures is the engine's own tile geometry and not a route's framing of it.
    """
    metadata_df = _resolve_metadata(dimensions, base_path)
    with rio.open(metadata_df["path"][0]) as src:
        window = utils_module.tile_source_window(src, tile_bounds)
        if window is None:
            return np.full((tile_size, tile_size), NODATA, dtype=np.int32)
        src_transform = src.transform
        src_crs = src.crs

    band_wide = scan_raster(metadata_df, window=window)
    if aggregate is not None:
        band_wide = group_band_wide(
            band_wide,
            GroupOp(step_no=1, input="raster", by=[], aggregate=aggregate, output="collapsed"),
            extra_columns=[],
        )
    lf, _metadata_out, read_window = band_wide
    array = (
        lf.select("b0")
        .collect()
        .get_column("b0")
        .fill_null(NODATA)
        .to_numpy()
        .reshape(read_window.height, read_window.width)
    )
    return utils_module.warp_window_to_tile(
        array[np.newaxis, :, :],
        window=window,
        src_transform=src_transform,
        src_crs=src_crs,
        tile_bounds=tile_bounds,
        tile_size=tile_size,
        nodata=NODATA,
    )[0]


def _warp_first_then_collapse(metadata_df, tile_bounds, tile_size, aggregate):
    """The OTHER order, as the reference side of the differential: warp every band onto
    the tile grid FIRST (one `WarpedVRT` per source, `_warp_first`'s single-band shape
    done once per band), and only then collapse across bands -- on TILE pixels rather than
    source pixels, through the same `group_band_wide` lowering `_render_tile` uses, so the
    aggregate itself is not what a mismatch would be measuring.
    """
    paths = metadata_df["path"].to_list()
    tile_transform = from_bounds(*tile_bounds, tile_size, tile_size)
    arrays = []
    with ExitStack() as stack:
        for path in paths:
            source = stack.enter_context(rio.open(path))
            vrt = stack.enter_context(
                WarpedVRT(
                    source,
                    crs=TILE_CRS,
                    transform=tile_transform,
                    width=tile_size,
                    height=tile_size,
                    nodata=NODATA,
                    src_nodata=NODATA,
                    add_alpha=False,
                    dtype=source.dtypes[0],
                    resampling=Resampling.nearest,
                )
            )
            arrays.append(vrt.read(1))

    columns = {f"b{i}": arr.ravel() for i, arr in enumerate(arrays)}
    lf = (
        pl.DataFrame(columns)
        .lazy()
        .with_columns(
            [pl.when(pl.col(name) == NODATA).then(None).otherwise(pl.col(name)).alias(name) for name in columns]
        )
    )
    fake_window = ReadWindow(
        source=tuple(paths),
        col_off=0,
        row_off=0,
        width=tile_size,
        height=tile_size,
        transform=tile_transform,
        crs=TILE_CRS,
        decimation=(1, 1),
        batch_layout=(),
    )
    collapsed_lf, _metadata_out, _window = group_band_wide(
        (lf, metadata_df, fake_window),
        GroupOp(step_no=1, input="raster", by=[], aggregate=aggregate, output="collapsed"),
        extra_columns=[],
    )
    return (
        collapsed_lf.select("b0").collect().get_column("b0").fill_null(NODATA).to_numpy().reshape(tile_size, tile_size)
    )


class TestGetTileRunsTheWindowThenWarpPath:
    """Gate: the new path actually executes, and the old one is gone rather than bypassed."""

    def test_get_tile_calls_the_window_and_warp_helpers(self, unique_pixel_raster, base_path, monkeypatch):
        calls = []
        real_window = utils_module.tile_source_window
        real_warp = utils_module.warp_window_to_tile

        def spy_window(*args, **kwargs):
            calls.append("window")
            return real_window(*args, **kwargs)

        def spy_warp(*args, **kwargs):
            calls.append("warp")
            return real_warp(*args, **kwargs)

        monkeypatch.setattr(utils_module, "tile_source_window", spy_window)
        monkeypatch.setattr(utils_module, "warp_window_to_tile", spy_warp)

        tile = mercantile.tile(*_tile_centre(), 11)
        bounds = _tile_bounds(tile)
        rendered = _render_tile(DIMENSIONS, base_path, bounds)

        assert rendered is not None
        assert calls == ["window", "warp"]

    def test_warping_lives_only_in_the_io_layer(self):
        """Complementary to the spy: `io/raster_io.py` is the one place that reaches for
        `WarpedVRT`, and only ever against an in-memory dataset it just built.
        """
        assert hasattr(utils_module, "WarpedVRT")


class TestCollapseThenWarpMatchesWarpThenCollapse:
    """The other half of the reordering: the band collapse now runs on source pixels, not
    on tile pixels.

    `Resampling.nearest` is what makes the two orders agree -- each tile pixel takes one
    whole source pixel -- so this is a real differential on real arrays, not a restatement
    of the code. The fixture mixes nodata into one band so the collapse has something to
    mask, which is where a difference between the two orders would show up first.
    """

    @pytest.fixture
    def two_band_raster_with_nodata(self, base_path):
        """2030 unique per pixel with a nodata stripe; 2040 unique per pixel throughout."""
        band_2030 = raster_array()
        band_2030[: GRID.height // 3, :] = NODATA
        band_2040 = raster_array() + 7
        write_raster_dataset(
            base_path,
            years=(2030, 2040),
            data_fn=lambda year: band_2030 if year == 2030 else band_2040,
        )
        return base_path

    @pytest.mark.parametrize("aggregate", ["mean", "max", "sum"])
    @pytest.mark.parametrize("zoom", [10, 12, 14])
    def test_grouped_tile_pixels_are_identical(self, two_band_raster_with_nodata, aggregate, zoom):
        tile = mercantile.tile(*_tile_centre(), zoom)
        bounds = _tile_bounds(tile)

        ported = _render_tile(DIMENSIONS, two_band_raster_with_nodata, bounds, aggregate=aggregate)
        metadata_df = _resolve_metadata(DIMENSIONS, two_band_raster_with_nodata)
        reference = _warp_first_then_collapse(metadata_df, bounds, TILE_SIZE, aggregate)

        assert reference.shape == (TILE_SIZE, TILE_SIZE)
        assert (reference != NODATA).sum() > 1000, "the reference tile is empty -- nothing compared"
        assert np.array_equal(ported, reference), (
            f"aggregate={aggregate} z={zoom}: collapse-then-warp differs from warp-then-collapse"
        )
