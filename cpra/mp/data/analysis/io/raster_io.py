"""Analysis-layer helpers (the former api/utils.py, HTTP-free half). Nothing here may import FastAPI."""
import asyncio
import dataclasses
import itertools
import math
from enum import Enum
from functools import wraps
from time import time

import rasterio as rio
from rasterio import windows
from rasterio.coords import BoundingBox, disjoint_bounds
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import Affine, array_bounds, from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from cpra.mp.data.analysis.io.geography import straddling_feature_bounds
from cpra.mp.data.analysis.request.errors import RequestError
from cpra.mp.data.config import (
    DEBUG,
    RASTER_BATCH_BUFFER_MB,
    RASTER_BLOCKS_PER_BATCH,
    RASTER_CACHE_MAX_MB,
    RASTER_NUM_THREADS,
)
from cpra.mp.data.grids import GRIDS


def unpack_values(values_str, param_name, choices=None, separator=","):
    """Unpack values from a comma-separated list."""
    values = [val.strip() for val in values_str.split(separator)]

    # Validate choices.
    if choices:
        invalid_values = [val for val in values if val not in choices]
        if invalid_values:
            msg = f"Invalid values for {param_name}: {', '.join(invalid_values)}"
            raise RequestError(msg)

    return values


def enum_value(inst):
    """Get the value from an enum."""
    return inst.value if isinstance(inst, Enum) else inst


def timing(func):
    """Time a synchronous function call."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        if DEBUG:
            print(f"func:{func.__name__!r} args:[{args!r}, {kwargs!r}] took: {time() - start:2.4f} sec")
        return result

    return wrapped


def raster_gdal_env():
    """GDAL configuration for reading and writing rasters.

    ``GDAL_NUM_THREADS`` parallelizes tile decompression, which dominates every read of
    a multiband COG. Setting it measured 2.2-3.7x faster across band counts with
    byte-identical output -- but only for a dataset *opened* with it in effect (measured on
    real data: opening under this env then reading plainly took ~5s for a 10-band window
    vs ~15s with no env anywhere; entering the env only around the later ``.read()`` call,
    on an already-opened dataset, recovered almost none of that, ~12-14s). GDAL_NUM_THREADS
    is effectively latched in at open time for the GeoTIFF/COG driver, not re-read per call
    -- see the callers below, which enter this env around ``rio.open()`` for exactly that
    reason, narrowly (not held open for the whole read loop after).

    CORRECTED: an earlier version of this docstring claimed that selecting a
    subset of bands "still decompresses each tile's full band payload, so a 1-band read
    costs nearly as much as a 52-band one." Re-measured against real long-format data and
    found false: I/O-only reads scale with the number of bands actually read, not flat (on
    this machine: 1.22s for 2 bands vs 2.42s for 11 from one file; 1.11s for 2 bands vs
    4.81s for 21 of 52 from two files -- a large fixed per-read overhead means it isn't
    perfectly linear, but a narrow band selection is genuinely cheaper, not merely
    memory-light). The windowed-vs-whole-raster routing keys on the windowed path's
    buffer *size* (bands x block height x width) rather than on any such time assumption,
    so it needed no change once this claim was checked -- confirmed by reading it, not
    merely assumed.

    Capping ``GDAL_CACHEMAX`` is what keeps the windowed path's peak memory down, and unlike
    ``GDAL_NUM_THREADS`` it must be in effect at *read* time to matter, not open time
    (measured on real data, holding ``GDAL_NUM_THREADS`` fixed: capping ``GDAL_CACHEMAX``
    only at open produced no reduction in peak RSS at all -- 3706 MB either way -- while
    capping it at read time took the same read down to 491 MB). Reading one band of a
    52-band float32 COG this way went from 4226 MB to 1027 MB peak RSS, at no measurable
    cost in time. Above a handful of bands the path's own NumPy buffers dominate instead and
    the cap can't help on its own -- see ``raster_blocks_per_batch``, which bounds that
    buffer directly rather than relying on the GDAL cache to do it. Being process-global,
    the cap also bounds cache growth across concurrent requests rather than leaving it at
    GDAL's default 5% of RAM.

    ``GDAL_CACHEMAX`` is passed in BYTES, which is why the megabyte-denominated config
    constant is converted here rather than handed over as it stands. rasterio does not treat
    this option the way it treats the others: ``rasterio._env.set_gdal_config`` special-cases
    the name and calls ``GDALSetCacheMax64`` with the value directly, and that entry point
    takes a byte count -- it never reaches the GDAL config parser that would have read a bare
    ``2048`` as megabytes. Unconverted, ``RASTER_CACHE_MAX_MB``'s default asked for a 2 KB
    cache, which is no cache at all.

    A missing cache is invisible on a full-resolution windowed read, which touches each source
    block once, and ruinous on one GDAL serves out of an overview, which revisits blocks and
    re-decompresses every revisit. Measured on one real ``elev`` COG, a 1149x1144 window
    decimated to half resolution: 0.08s with a 64 MB cache or anything larger, 11.7s with the
    2 KB one, against 0.1s either way for that same window read at full resolution. Eleven
    bands of it decimated took 147s. That asymmetry is why the units went unnoticed for as
    long as they did -- until decimated reads started coming through here, nothing that
    entered this env re-read a block.

    ``rio.Env`` is thread-local storage (``rasterio.env.ThreadEnv`` subclasses
    ``threading.local``), but the GDAL config option it sets is not always thread-scoped:
    rasterio's ``_env.pyx`` uses ``CPLSetConfigOption`` (process-global) when the *calling*
    thread is the main thread, and ``CPLSetThreadLocalConfigOption`` otherwise, and every
    thread falls back to the main thread's global value unless it has set its own override.
    Confirmed empirically (a main-thread-entered env's config option was visible from a
    ``loop.run_in_executor`` worker thread without that thread entering anything itself).
    That means entering this env once in the calling coroutine *would* reach worker-thread
    reads dispatched via ``run_in_executor``/``to_thread`` -- but it would just as easily
    reach any *other* concurrent request's reads sharing the same default thread pool for as
    long as the env stayed open, since the option is genuinely process-global with no
    per-request scoping. Confirmed empirically too: an unrelated concurrent "tile" read that
    never touched this env observed ``GDAL_NUM_THREADS`` leaked in from a second, overlapping
    "summarize" coroutine's still-open env. Entering the env inside the worker-thread
    function that performs the actual read (see ``_read_window``'s ``use_gdal_env``
    argument) avoids that leak (that thread is not the main thread, so it goes through the
    thread-local path and reverts on exit) while entering it narrowly around a ``rio.open()``
    call in the calling coroutine (see callers) is an acceptably small, bounded exposure
    given how fast an open is relative to the read/write loop that follows it.

    Applied to tile requests too, which it once was not. A tile reads a window decimated to
    roughly the tile's own resolution, served out of the source's overviews, and threading
    measured no benefit on a read that small -- so the older reader left tile reads outside
    this env entirely. ``io/raster.py``'s scan has one read path rather than two and enters
    this env around every read, tile or not, which is what the cache paragraph above is about:
    on a decimated read the cap is not a memory nicety, it is the difference between a tile in
    half a second and a tile in half a minute.
    """

    return rio.Env(GDAL_NUM_THREADS=RASTER_NUM_THREADS, GDAL_CACHEMAX=RASTER_CACHE_MAX_MB * 1024 * 1024)


def raster_blocks_per_batch(n_bands, block_height, block_width, itemsize):
    """Number of adjacent raster blocks to union into one summarize-loop batch.

    The summarize loop used to union a whole block row per iteration regardless of how
    many blocks that row contained, so one iteration's read buffer (``n_bands`` bands x
    ``block_height`` x the row's full width) grew with the raster's width and the
    request's band count alike -- 20674 px wide block rows meant
    ~10M pixels x N bands per iteration no matter how small the actual query. Chunking the
    row into groups of ``K`` blocks bounds that buffer directly (measured 4 block rows, 52
    bands: whole row 12.0s/4143MB -> 8 blocks 10.6s/2243MB -> 4 blocks 9.8s/1753MB -> 2
    blocks 9.1s/1363MB -- chunking is both faster and lighter, so there's no tradeoff to
    tune against).

    ``RASTER_BLOCKS_PER_BATCH``, if set, overrides the computed value outright. Otherwise
    ``K`` is derived from ``RASTER_BATCH_BUFFER_MB`` (itself defaulted from
    ``RASTER_CACHE_MAX_MB``) and the actual band count and block shape of the request, so a
    wide band selection or a grid with a different block/dtype size doesn't inherit a
    constant tuned for 52 int32 bands on 512x512 blocks.
    """
    if RASTER_BLOCKS_PER_BATCH is not None:
        return RASTER_BLOCKS_PER_BATCH
    per_block_bytes = max(n_bands, 1) * block_height * block_width * itemsize
    if per_block_bytes <= 0:
        return 1
    budget_bytes = RASTER_BATCH_BUFFER_MB * 1024 * 1024
    return max(1, budget_bytes // per_block_bytes)


def batched_windows(src, blocks_per_batch):
    """Window unions over one block row at a time, chunked to `blocks_per_batch` blocks.

    This is where the VRT-based raster scan (`io/raster.py`) reads its batch boundaries
    from, so a request over a large raster bounds its per-batch buffer rather than
    unioning a whole block row (tens of thousands of pixels wide regardless of how much
    data a request needs) into one batch.

    `src.block_windows(1)` reads band 1's own block layout as the batching reference --
    every band of a co-registered raster covers the same ground, so which band's block
    grid is walked does not change which pixels a batch covers, only the chunk boundaries
    it happens to fall on.
    """
    groups = itertools.groupby(src.block_windows(1), key=lambda bw: bw[0][0])
    for _row, group in groups:
        row_windows = [w for _ji, w in group]
        for i in range(0, len(row_windows), blocks_per_batch):
            yield windows.union(*row_windows[i : i + blocks_per_batch])


#: The CRS every map tile is rendered in.
TILE_CRS = "EPSG:3857"

#: Whole source pixels read around a tile's own footprint, so the warp onto the tile grid
#: has neighbouring pixels to sample at the tile's edge rather than running off the end of
#: the buffer it was handed. Four is what `tile_source_window`'s callers were measured
#: against; with `Resampling.nearest` (the only resampling either tile path uses) a single
#: pixel would do, and a larger margin costs only the extra pixels read.
TILE_READ_BUFFER_PX = 4

#: Half-width, in source pixels, that each resampling kernel actually samples around a
#: source location -- 0 for nearest (the single pixel the rounded coordinate lands on), 1
#: for a 2x2 bilinear kernel, 2 for a 4x4 cubic/cubic-spline kernel, 3 for lanczos's
#: windowed-sinc kernel. These are GDAL's own kernel definitions, not this module's choice.
_RESAMPLING_KERNEL_RADIUS_PX = {
    Resampling.nearest: 0,
    Resampling.bilinear: 1,
    Resampling.cubic: 2,
    Resampling.cubic_spline: 2,
    Resampling.lanczos: 3,
}


def resampling_buffer_px(resampling):
    """Source pixels `tile_source_window` should read around a tile's footprint for
    `resampling`, so the later warp has whole neighbouring pixels to sample at the tile's
    edge rather than running off the end of the buffer it was handed.

    `TILE_READ_BUFFER_PX` (4) is this module's original, `Resampling.nearest`-only buffer,
    measured against `tile_source_window`'s callers before `resampling` existed as a
    parameter -- nearest only ever needs the single pixel a rounded coordinate lands on, so
    all 4 of those pixels are rounding margin rather than kernel radius. Every other
    resampling method needs that same rounding margin PLUS its own kernel's half-width, so
    this adds the kernel radius on top of the margin instead of replacing it:
    `resampling_buffer_px(Resampling.nearest) == TILE_READ_BUFFER_PX` exactly, which is what
    keeps a `tile_source_window` call that never mentions `resampling` at all unchanged from
    before this function existed.

    **The radius is in SOURCE pixels, so it is only sufficient where the warp does not
    DOWNSAMPLE**, and that limit is measured rather than assumed -- a 2176x2176 10 m source
    swept over a 3x3 tile neighbourhood, derived buffer against a generous 20 px one, is
    byte-identical at a tile resolution of 8.3 m (0.8x, upsampling: 0 of 589,824 pixels
    differ) and disagrees at essentially every pixel one zoom lower, at 16.6 m (1.7x,
    downsampling: 583,033 of 589,824) -- while `Resampling.nearest` over those same windows,
    at that same zoom and decimation factor, disagrees at 0. Only the kernel differs between
    those two numbers, which is what identifies the cause: a warp that downsamples by `r`
    draws each destination pixel from roughly `r` source pixels per kernel tap, so the margin
    an `r`-times-coarser destination needs is about `radius * r` source pixels, not `radius`.

    Nothing corrects for that here, because this function is not given the ratio -- it sees
    neither the tile's pixel size nor `tile_size`, only `resampling`. Supplying it is what a
    caller wanting a non-nearest tile would have to add, and the ratio is not incidental:
    `TILE_READ_OVERSAMPLE` deliberately holds a decimated read about 2x finer than the tile
    grid, so every decimated tile warps at `r` near 2 by construction. `Resampling.nearest`
    is unaffected at any ratio (radius 0, nothing to sample around), which is why the tile
    path is correct today at the one value it actually uses.
    """
    try:
        radius = _RESAMPLING_KERNEL_RADIUS_PX[resampling]
    except KeyError:
        raise ValueError(f"resampling_buffer_px: unsupported resampling method {resampling!r}")
    return TILE_READ_BUFFER_PX + radius


@dataclasses.dataclass(frozen=True, slots=True)
class SourceGeometry:
    """The four properties `tile_source_window` reads off a raster: where its pixels sit on
    the ground and how many there are.

    An open `rasterio` dataset already has all four under these names, and is what
    `tile_source_window` was originally written against. This exists so a caller that must
    NOT open a file can supply them from somewhere else -- `raster_grid_geometry` builds one
    from a `cpra.mp.data.grids.RasterGrid`, which is what lets the compiler size a tile's
    read window (`execution.plan`) while keeping its promise to open no data files.

    That substitution is only sound because a raster dataset on grid *G* is co-registered
    with *G* exactly: same CRS, same extent, same pixel count, therefore the same transform.
    `execution.execute._grid_read_window` already rests on the same fact, and says it was
    confirmed byte-identical against a real scan.
    """

    crs: CRS
    transform: Affine
    width: int
    height: int


def raster_grid_geometry(grid_name: str) -> SourceGeometry:
    """`grid_name`'s own geometry as a `SourceGeometry`, opening nothing.

    The transform is derived from the grid's bounds and pixel count exactly as
    `io.raster.scan_raster` derives it for a full-extent read of a real raster on that grid.
    """
    grid = GRIDS[grid_name]
    return SourceGeometry(
        crs=CRS.from_user_input(grid.crs),
        transform=from_bounds(*grid.bounds, width=grid.width, height=grid.height),
        width=grid.width,
        height=grid.height,
    )


def tile_disjoint_from_source(src, tile_bounds) -> bool:
    """Whether `tile_bounds` (Web Mercator, `TILE_CRS`) shares no ground at all with `src`'s
    own extent -- the cheap "this tile is nowhere near this raster" test, answered from the
    same four properties `tile_source_window` reads (so a `SourceGeometry` serves as well as
    an open dataset).

    Separate from `tile_source_window` returning `None`, and not redundant with it, because
    the two reproject in OPPOSITE directions. `tile_source_window` carries the tile's
    footprint into the source's CRS, where a footprint far enough outside the projection's
    domain of validity has no finite bounds at all; that function then falls back to the
    source's full extent, since a window that certainly contains the footprint is the safe
    answer to "which pixels might this tile need". This one carries the source's extent out
    to Web Mercator, which is always finite (the source's own pixels are, by construction,
    inside its projection's domain), and so stays decisive exactly where the other one
    cannot be. A low-zoom tile covering a whole hemisphere of empty ocean is the case: the
    inbound reprojection goes non-finite and yields the full raster, while this returns
    `True` and the caller skips the read entirely.
    """
    source_bounds = transform_bounds(
        src.crs, TILE_CRS, *array_bounds(src.height, src.width, src.transform), densify_pts=21
    )
    return disjoint_bounds(BoundingBox(*source_bounds), BoundingBox(*tile_bounds))


def tile_source_window(
    src, tile_bounds, buffer_px=None, resampling=Resampling.nearest, straddling_vector_grid=None
):
    """The window of `src` (in its OWN CRS) covering the Web Mercator `tile_bounds`, as
    whole source pixels clipped to the source's extent -- or `None` when the tile's
    footprint lands entirely outside the source.

    `src` is anything carrying `crs`/`transform`/`width`/`height`: an open `rasterio`
    dataset, or a `SourceGeometry` built from a grid by `raster_grid_geometry` when the
    caller must not open a file. Those four are all this function reads.

    Two properties of the returned window are what let a caller read it, run per-pixel work
    over the result in the SOURCE CRS, and only then warp onto the tile grid (see
    `warp_window_to_tile`):

    * It is a whole-pixel window: every offset and extent is an integer, so
      `rasterio.windows.transform(window, src.transform)` describes the returned buffer
      exactly -- same resolution and same grid phase as `src` itself, offset by an integer
      number of pixels.
    * It is INTERSECTED with the source extent rather than read `boundless=True` with
      nodata padding. GDAL treats "outside the source extent" and "inside it but nodata"
      differently when it snaps the warp at the data boundary, and padding measured one
      pixel of erosion along that boundary on tiles that straddle it. Clipping instead
      measured none. `tests/api/test_tile_warp.py` pins both halves of that.

    `transform_bounds` densifies the tile's edges before reprojecting (a straight
    Web-Mercator edge is curved in a projected source CRS, so its corners alone understate
    the footprint). A tile large enough that its reprojection is not finite at all -- a
    whole-world tile at low zoom, say -- falls back to the source's full extent, which is
    the smallest window that certainly contains the footprint.

    `buffer_px`, when not given explicitly, is derived from `resampling`
    (`resampling_buffer_px`) rather than defaulting to `TILE_READ_BUFFER_PX` outright -- see
    that function for why the two agree exactly when `resampling` is left at its own
    `Resampling.nearest` default.

    `straddling_vector_grid`, when given, is the name of a vector grid that `execution.plan.
    compile_plan` has determined this source feeds a raster-to-vector `project` step onto --
    this function does not derive that itself, only acts on the answer. When given, the
    window is widened to also cover every feature of that grid the tile touches at all (not
    only the part of each feature inside the tile), via `io.geography.
    straddling_feature_bounds`. Three CRSs are in play for that widening: the tile's own
    Web Mercator grid (`TILE_CRS`), `straddling_vector_grid`'s OWN crs (what
    `straddling_feature_bounds` returns its bounds in), and `src.crs` (what this function's
    return value is ultimately expressed in) -- a raster-to-vector `project` step joins each
    pixel to whichever polygon it falls inside on the OUTPUT grid, and that polygon can
    extend past the tile's own edge, so a read clipped to the tile's footprint alone would
    silently drop the pixels on the far side of a straddling polygon before the join ever
    sees them.
    """
    # CRS #1 (this tile's own grid, Web Mercator, `TILE_CRS`) -> CRS #2 (`src`'s own crs):
    # the tile's pixel-accurate footprint in the source's own georeferencing.
    src_bounds = transform_bounds(TILE_CRS, src.crs, *tile_bounds, densify_pts=21)
    if not all(math.isfinite(value) for value in src_bounds):
        return windows.Window(0, 0, src.width, src.height)

    if straddling_vector_grid is not None:
        # CRS #1 -> CRS #3 (`straddling_vector_grid`'s OWN crs, read inside
        # `straddling_feature_bounds` itself): the whole footprint of every feature the tile
        # touches at all, not just the part already inside the tile.
        feature_bounds = straddling_feature_bounds(straddling_vector_grid, tile_bounds, TILE_CRS)
        if feature_bounds is not None:
            # CRS #3 -> CRS #2: reproject that footprint into the same source crs
            # `src_bounds` above is already expressed in, so the two can be unioned.
            feature_src_bounds = transform_bounds(
                GRIDS[straddling_vector_grid].crs, src.crs, *feature_bounds, densify_pts=21
            )
            if all(math.isfinite(value) for value in feature_src_bounds):
                src_bounds = (
                    min(src_bounds[0], feature_src_bounds[0]),
                    min(src_bounds[1], feature_src_bounds[1]),
                    max(src_bounds[2], feature_src_bounds[2]),
                    max(src_bounds[3], feature_src_bounds[3]),
                )

    if buffer_px is None:
        buffer_px = resampling_buffer_px(resampling)

    window = windows.from_bounds(*src_bounds, transform=src.transform)
    col_off = max(0, math.floor(window.col_off) - buffer_px)
    row_off = max(0, math.floor(window.row_off) - buffer_px)
    col_end = min(src.width, math.ceil(window.col_off + window.width) + buffer_px)
    row_end = min(src.height, math.ceil(window.row_off + window.height) + buffer_px)
    if col_end <= col_off or row_end <= row_off:
        return None
    return windows.Window(col_off, row_off, col_end - col_off, row_end - row_off)


#: How much finer than the tile's own pixel grid a decimated tile read is kept.
#:
#: A tile pixel takes ONE source pixel under `Resampling.nearest`, so if the decimated grid
#: were at the tile's own resolution the pixel it picks could be up to ~1.5 decimated pixels
#: -- i.e. more than a whole tile pixel -- away from the one a full-resolution read would
#: have picked. Measured over 38 tiles (z=7..14, 1,741,449 rendered pixels compared) of the
#: 2176x2176 unique-per-pixel fixture `tests/api/test_tile_warp.py` builds: reading at the
#: tile's own resolution displaces a rendered pixel by up to 0.75 tile pixels, reading twice
#: as fine by up to 0.15. The cost is 4x the pixels of a 1:1 rule, which at low zoom is still
#: two orders of magnitude below a full-resolution read (434 MB -> 3.4 MB per band at z=6 on
#: `morph_pixel_v001`, versus 0.8 MB at 1:1).
#:
#: It is a margin, not a guarantee: a decimated read still takes a different source pixel
#: than a full-resolution one, which is what `TestDecimatedTileReads` bounds rather than
#: asserts away.
TILE_READ_OVERSAMPLE = 2


def _axis_decimation_factor(extent, target):
    """The largest power of two by which `extent` source pixels can be decimated and still
    leave at least `target` of them -- 1 (read this axis whole) when it cannot be halved."""
    ratio = int(extent // target)
    if ratio < 2:
        return 1
    return 1 << (ratio.bit_length() - 1)


def tile_decimation_factors(window, tile_size, buffer_px=TILE_READ_BUFFER_PX, oversample=TILE_READ_OVERSAMPLE):
    """The ``(row_factor, col_factor)`` to decimate `window` by while reading it for a
    tile -- ``(1, 1)`` when it is already small enough to read whole.

    The factors themselves, rather than the shape they produce: this is what
    `execution.plan.compile_plan` records on a `ResolvedInput` and what
    `io.raster.scan_raster` is handed, since a factor survives being carried around a
    compile step while a shape is only meaningful beside the window it was derived from.
    `tile_read_shape` is the same decision expressed as a shape, derived from this so the
    two cannot disagree; see it for how the factors are chosen and why per axis.
    """
    target = oversample * (tile_size + 2 * buffer_px)
    return (_axis_decimation_factor(window.height, target), _axis_decimation_factor(window.width, target))


def tile_read_shape(window, tile_size, buffer_px=TILE_READ_BUFFER_PX, oversample=TILE_READ_OVERSAMPLE):
    """The ``(height, width)`` to decimate `window` to while reading it for a tile, or None
    when `window` is already small enough to read whole.

    A tile's source window is only as small as the tile's ground footprint, which at low
    zoom is most of the raster: `tile_source_window` on a 20674x8285 source returns a
    13107x8285 window at z=6, 434 MB per band for the 65,536 pixels a 256x256 tile actually
    renders. Reading that window into a smaller buffer makes GDAL serve it from the source's
    overviews (rasterio passes ``out_shape`` straight through to GDAL's own overview
    selection), so the decompressed volume drops with the square of the factor.

    The factor is chosen so that:

    * it never decimates below `oversample` times the tile's own resolution -- the target is
      `tile_size` plus `tile_source_window`'s buffer on both sides, so the returned shape
      still covers the same ground as the window at a resolution finer than the tile grid
      it is about to be warped onto (see `TILE_READ_OVERSAMPLE` for what the margin buys);
    * it never UPSAMPLES -- an axis already at or below that size keeps every one of its
      pixels, and a window small enough on BOTH axes returns None, which the callers read
      exactly as they did before this existed;
    * it is a power of two, so it lands on a real overview level (the COGs this reads carry
      ``[2, 4, 8, 16, 32, 64]``, built with NEAREST -- see `raster._cog_profile` -- which is
      what keeps every decimated pixel a real source value rather than a blend) rather than
      making GDAL resample down from a finer one;
    * it is chosen PER AXIS. A single factor taken from the shorter axis would leave the
      read unbounded on the longer one: a tile straddling the source's edge gets a window
      clipped to a sliver -- 2176x817 on the test fixture, 20674 x a few hundred on
      morph_pixel -- whose short axis forbids decimating at all while its long axis is still
      the whole raster. Per axis, the returned shape is under
      ``2 * oversample * (tile_size + 2 * buffer_px)`` on BOTH axes for every window there
      is, which is the bound that makes a tile's cost a function of the tile rather than of
      the zoom (`tests/api/test_tile_warp.py::TestDecimatedTileReads` pins it). When the two
      factors differ GDAL can serve only one axis straight off the matching overview level
      and decimates the other itself; the array handed back is bounded either way, and under
      `Resampling.nearest` both axes stay whole source values.

    The returned shape is `window` floor-divided by those factors, not ceil-divided: the
    effective ratio has to be at least the factor for GDAL to pick that overview level
    rather than the next finer one. The shape still describes the WHOLE window -- it is the
    same ground extent at a coarser resolution -- which is what `warp_window_to_tile` needs
    to scale the georeferencing by.

    The same arithmetic lives on `io.window.ReadWindow.read_shape`, which answers it for a
    window that has already been read and is carrying its own factors; both derive the
    shape from `tile_decimation_factors`' answer the same way, so a decimated scan's frame
    and this function agree on the pixel count by construction.
    """
    height_factor, width_factor = tile_decimation_factors(window, tile_size, buffer_px, oversample)
    if height_factor == 1 and width_factor == 1:
        return None
    return (max(1, int(window.height) // height_factor), max(1, int(window.width) // width_factor))


def _warp_array_to_tile(data, array_transform, src_crs, tile_bounds, tile_size, nodata, resampling=Resampling.nearest):
    """Warp `data` (a ``(bands, height, width)`` array georeferenced by `array_transform`
    in `src_crs`) onto a tile's own Web Mercator pixel grid, through an in-memory dataset.

    `array_transform` must describe the array PASSED IN. `warp_window_to_tile` is the entry
    point that guarantees that; this function takes the transform as a bare argument so a
    test can hand it a deliberately wrong one and measure what that costs
    (`tests/api/test_tile_warp.py`'s control).

    The warp targets the tile grid directly -- explicit `transform`/`width`/`height`, no
    intermediate output grid. `resampling` defaults to `Resampling.nearest`, which keeps
    every rendered pixel a real source pixel value rather than a blend of several; a caller
    that passes anything else is opting into a blend, and is the one responsible for
    widening the read (`resampling_buffer_px`, via `tile_source_window`) to match.
    """
    tile_transform = from_bounds(*tile_bounds, tile_size, tile_size)
    count, height, width = data.shape
    dtype = data.dtype.name
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=dtype,
            crs=src_crs,
            transform=array_transform,
            nodata=nodata,
        ) as mem:
            mem.write(data)
        # Reopened read-only: rasterio deprecates warping a dataset that is still open for
        # writing, and the written bytes are already complete once the block above exits.
        with (
            memfile.open() as mem,
            WarpedVRT(
                mem,
                crs=TILE_CRS,
                transform=tile_transform,
                width=tile_size,
                height=tile_size,
                nodata=nodata,
                src_nodata=nodata,
                add_alpha=False,
                dtype=dtype,
                resampling=resampling,
            ) as vrt,
        ):
            return vrt.read()


def warp_window_to_tile(
    data,
    *,
    window,
    src_transform,
    src_crs,
    tile_bounds,
    tile_size,
    nodata,
    out_shape=None,
    resampling=Resampling.nearest,
):
    """Warp one window's worth of per-pixel results onto a tile's pixel grid.

    `data` is whatever a caller computed over the pixels of `window`, still in the source
    CRS and still on the source's own grid, and `window`/`src_transform` are the window that
    was READ and the transform of the dataset it was read from. `out_shape` is the
    ``(height, width)`` that window was DECIMATED to on the way in (`tile_read_shape`), or
    None when it was read whole -- `data` is ``(bands, *out_shape)`` in the first case and
    ``(bands, window.height, window.width)`` in the second.

    The georeferencing handed to the warp is derived here, from those three, rather than
    accepted from the caller: a transform built from anything other than the array actually
    handed to the warp is a sub-pixel origin error that displaces every rendered pixel, with
    no symptom short of a differential against another implementation (see
    `tests/api/test_tile_warp.py`, which measures exactly that). A decimated read is where
    that is easiest to get wrong -- the array is smaller than the window while covering the
    same ground, so the window's own transform describes it at the wrong resolution -- which
    is why the scaling lives here too and not at either call site.

    The shape check below is the mechanical half of the same guarantee -- an array that
    matches neither the window nor the shape it was decimated to cannot reach the warp.

    `resampling` defaults to `Resampling.nearest`, forwarded to `_warp_array_to_tile`
    unchanged -- the buffer `window` was read with has to have been sized for whatever
    `resampling` is passed here (`resampling_buffer_px`, via `tile_source_window`), which is
    the caller's responsibility, not this function's; it only carries the choice through.
    """
    expected = tuple(out_shape) if out_shape is not None else (window.height, window.width)
    if tuple(data.shape[-2:]) != expected:
        raise ValueError(
            f"warp_window_to_tile: data is {data.shape[-2:]} but window is "
            f"{expected} -- the array must be exactly the window "
            "that was read, or its georeferencing is wrong"
        )
    array_transform = windows.transform(window, src_transform)
    if out_shape is not None:
        array_transform *= Affine.scale(window.width / expected[1], window.height / expected[0])
    return _warp_array_to_tile(data, array_transform, src_crs, tile_bounds, tile_size, nodata, resampling=resampling)


def warp_array_to_tile(
    data, *, array_transform, src_crs, tile_bounds, tile_size, nodata, resampling=Resampling.nearest
):
    """Warp `data` onto a tile's pixel grid when its transform is ALREADY known to describe
    it -- the sibling of `warp_window_to_tile` for a caller holding a georeferenced array
    rather than a window and a dataset transform.

    That is the case for anything read through `io.raster.scan_raster`: the `ReadWindow` it
    returns carries the transform of the window it actually read, so a result computed over
    those pixels is georeferenced by `ReadWindow.transform` directly, with nothing left to
    derive. `warp_window_to_tile` exists to compute that transform for a caller who has only
    the source's own; handing it a window and transform just to have it recompute a value
    already in hand would be the longer way round, and would need the decimation scaling
    reapplied by hand on top.

    The sub-pixel warning on `warp_window_to_tile` applies here with nothing between the
    caller and the mistake: `array_transform` MUST describe `data` itself. Prefer
    `warp_window_to_tile` where the transform still has to be derived.
    """
    return _warp_array_to_tile(data, array_transform, src_crs, tile_bounds, tile_size, nodata, resampling=resampling)


def _read_window(dataset, window, out_shape=None, resampling=Resampling.nearest, indexes=None, use_gdal_env=False):
    """Read a window of data from a single raster (optionally a subset of bands).

    ``use_gdal_env`` enters ``raster_gdal_env()`` around the read itself, which is what
    keeps ``GDAL_CACHEMAX`` bounding memory for *this* read specifically (it must be live
    at read time -- see ``raster_gdal_env``'s docstring) without leaving a process-global
    GDAL config change sitting open for the whole duration of an unrelated concurrent
    request that happens to dispatch onto the same worker thread pool. Kept as an opt-in
    argument rather than always-on because tile requests deliberately skip it (see
    ``raster_gdal_env``).
    """
    if use_gdal_env:
        with raster_gdal_env():
            return dataset.read(indexes=indexes, window=window, out_shape=out_shape, resampling=resampling)
    return dataset.read(indexes=indexes, window=window, out_shape=out_shape, resampling=resampling)


async def parallel_read_rasters(
    datasets, window=None, out_shape=None, resampling=Resampling.nearest, bands=None, use_gdal_env=False
):
    """Read several rasters in multiple threads.

    ``bands`` optionally selects, per dataset, the list of band numbers (1-based) to
    read; each dataset's selected bands are read together as one ``(n, height, width)``
    array. When ``bands`` is None every band of each dataset is read.

    ``use_gdal_env`` is forwarded to ``_read_window``, entering ``raster_gdal_env()`` on
    each worker thread that performs a read, scoped to just that read -- see that
    function's and ``raster_gdal_env``'s docstrings for why that scoping (rather than
    entering the env once in the calling coroutine) matters here.
    """
    loop = asyncio.get_running_loop()
    if bands is None:
        bands = [None] * len(datasets)
    return await asyncio.gather(
        *[
            loop.run_in_executor(None, _read_window, dataset, window, out_shape, resampling, band, use_gdal_env)
            for dataset, band in zip(datasets, bands)
        ]
    )


def _sample_raster(dataset, coordinates, bands=None):
    """Sample a single raster at the given coordinates (optionally a subset of bands)."""
    return list(dataset.sample(coordinates, indexes=bands))


async def parallel_sample_rasters(datasets, coordinates, bands=None):
    """Sample several rasters at the given coordinates in multiple threads.

    ``bands`` optionally selects, per dataset, the list of band numbers (1-based) to
    sample; each dataset's selected bands are sampled together in one call. When
    ``bands`` is None every band of each dataset is sampled.
    """
    loop = asyncio.get_running_loop()
    if bands is None:
        bands = [None] * len(datasets)
    return await asyncio.gather(
        *[
            loop.run_in_executor(None, _sample_raster, dataset, coordinates, band)
            for dataset, band in zip(datasets, bands)
        ]
    )
