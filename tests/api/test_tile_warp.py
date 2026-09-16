"""The tile path's geometry: read the tile's footprint in the source CRS, do the per-pixel
work there, warp the result onto the tile grid last.

`RasterGridAnalysis.get_tile` and `paired._paired_notebook_tile` used to warp every source
band onto the tile's pixel grid FIRST (a `WarpedVRT` per source) and collapse the bands
afterwards, on tile pixels. They now do it the other way round, through
`utils.tile_source_window` + `utils.warp_window_to_tile`. This file exists to prove that
change moved no pixel, and it has to do that with a differential rather than an assertion
about the code, because the failure mode is a fractional-pixel offset that renders as a
picture that still looks right.

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
from io import BytesIO

import mercantile
import numpy as np
import pytest
import rasterio as rio
from PIL import Image
from rasterio import windows
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from cpra.mp.data.analysis import raster as raster_module
from cpra.mp.data.analysis import utils as utils_module
from cpra.mp.data.analysis.utils import (
    TILE_CRS,
    TILE_READ_BUFFER_PX,
    _warp_array_to_tile,
    group_rows_by_path,
    tile_source_window,
    warp_window_to_tile,
)
from cpra.mp.data.grids import GRIDS
from tests.factories import RASTER_KWARGS, raster_array, write_raster_dataset

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


async def _warp_first_then_collapse(dataset, tile_bounds, tile_size):
    """The pre-port band order: one `WarpedVRT` per source onto the tile's pixel grid, then
    `_collapse_tile_bands` over the warped TILE pixels.

    The collapse itself is the production method -- only the order changes -- so a
    difference between this and `get_tile` can only come from where the reprojection sits.
    """
    df = dataset._band_frame()
    unique_paths, row_indices, bands = group_rows_by_path(df)
    tile_transform = from_bounds(*tile_bounds, tile_size, tile_size)
    with ExitStack() as stack:
        inputs = [stack.enter_context(rio.open(path)) for path in unique_paths]
        nodata = inputs[0].nodata if inputs[0].nodata is not None else NODATA
        vrts = [
            stack.enter_context(
                WarpedVRT(
                    source,
                    crs=TILE_CRS,
                    transform=tile_transform,
                    width=tile_size,
                    height=tile_size,
                    nodata=nodata,
                    add_alpha=False,
                    src_nodata=nodata,
                    dtype=source.dtypes[0],
                    resampling=Resampling.nearest,
                )
            )
            for source in inputs
        ]
        data = await dataset._read_raster_bands(df, vrts, row_indices, bands, None)
        return dataset._collapse_tile_bands(df, data, nodata)


def _score(reference, candidate):
    """`(displaced, edge_gaps, reference_valid)` for one tile -- the two failure modes kept
    apart, plus how many pixels the comparison actually had to work with."""
    has_reference = reference != NODATA
    has_candidate = candidate != NODATA
    both = has_reference & has_candidate
    displaced = int(((reference != candidate) & both).sum())
    gaps = int((has_reference ^ has_candidate).sum())
    return displaced, gaps, int(has_reference.sum())


def _sweep(src, zooms=range(9, 15)):
    """Every tile in a 3x3 neighbourhood of the grid's centre, at each zoom, that the
    pre-port implementation renders any data into at all."""
    lon, lat = _tile_centre()
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
        """z=0 covers the whole planet, most of which has no finite position in a UTM zone.
        The fallback reads everything rather than failing on an infinite bound."""
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


class TestGetTileRunsTheWindowThenWarpPath:
    """Gate: the new path actually executes, and the old one is gone rather than bypassed."""

    async def test_get_tile_calls_the_window_and_warp_helpers(
        self, dataset_factory, unique_pixel_raster, base_path, monkeypatch
    ):
        calls = []
        real_window = raster_module.tile_source_window
        real_warp = raster_module.warp_window_to_tile

        def spy_window(*args, **kwargs):
            calls.append("window")
            return real_window(*args, **kwargs)

        def spy_warp(*args, **kwargs):
            calls.append("warp")
            return real_warp(*args, **kwargs)

        monkeypatch.setattr(raster_module, "tile_source_window", spy_window)
        monkeypatch.setattr(raster_module, "warp_window_to_tile", spy_warp)

        dataset = dataset_factory(DIMENSIONS)
        await dataset.apply_params()
        tile = mercantile.tile(*_tile_centre(), 11)
        body = await dataset.get_tile(tile.x, tile.y, tile.z, "png")

        assert body is not None
        assert calls == ["window", "warp"]

    def test_the_source_warping_import_is_gone_from_the_raster_module(self):
        """Complementary to the spy: `raster.py` no longer has `WarpedVRT` bound at all, so
        no stale caller can still warp a source dataset straight onto a tile grid from
        there. `utils.py` is the one place that reaches for it now, and only ever against
        an in-memory dataset it just built.
        """
        assert not hasattr(raster_module, "WarpedVRT")
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

    @pytest.fixture
    def capture_ported_tile(self, monkeypatch):
        """Hand back the exact array `get_tile` hands the legend.

        Comparing rendered PNGs instead would compare colors, and a legend maps whole
        ranges of values onto one color -- which is precisely the resolution at which a
        mis-collapsed pixel disappears. `test_the_rendered_png_is_unchanged` covers the
        rendering; this covers the numbers.
        """
        captured = []
        real_warp = raster_module.warp_window_to_tile

        def capture(*args, **kwargs):
            result = real_warp(*args, **kwargs)
            captured.append(result)
            return result

        monkeypatch.setattr(raster_module, "warp_window_to_tile", capture)
        return captured

    @pytest.mark.parametrize("aggregate", ["mean", "max", "sum"])
    @pytest.mark.parametrize("zoom", [10, 12, 14])
    async def test_grouped_tile_pixels_are_identical(
        self, dataset_factory, two_band_raster_with_nodata, capture_ported_tile, aggregate, zoom
    ):
        tile = mercantile.tile(*_tile_centre(), zoom)
        bounds = _tile_bounds(tile)

        dataset = dataset_factory(DIMENSIONS, group_aggregate=aggregate)
        await dataset.apply_params()
        assert await dataset.get_tile(tile.x, tile.y, tile.z, "png") is not None
        (ported,) = capture_ported_tile

        reference_dataset = dataset_factory(DIMENSIONS, group_aggregate=aggregate)
        await reference_dataset.apply_params()
        reference = await _warp_first_then_collapse(reference_dataset, bounds, TILE_SIZE)

        assert reference.shape == (1, TILE_SIZE, TILE_SIZE)
        assert (reference != NODATA).sum() > 1000, "the reference tile is empty -- nothing compared"
        assert np.array_equal(ported, reference), (
            f"aggregate={aggregate} z={zoom}: collapse-then-warp differs from "
            "warp-then-collapse"
        )

    async def test_the_rendered_png_is_unchanged(
        self, dataset_factory, two_band_raster_with_nodata
    ):
        """End to end, on the bytes a client actually receives."""
        tile = mercantile.tile(*_tile_centre(), 11)
        bounds = _tile_bounds(tile)

        dataset = dataset_factory(DIMENSIONS, group_aggregate="mean")
        await dataset.apply_params()
        body = await dataset.get_tile(tile.x, tile.y, tile.z, "png")
        with Image.open(BytesIO(body)) as image:
            ported_pixels = np.array(image)

        reference_dataset = dataset_factory(DIMENSIONS, group_aggregate="mean")
        await reference_dataset.apply_params()
        collapsed = await _warp_first_then_collapse(reference_dataset, bounds, TILE_SIZE)
        reference_pixels = reference_dataset.legend.colorize_pixels(collapsed)

        assert np.array_equal(ported_pixels, reference_pixels)
