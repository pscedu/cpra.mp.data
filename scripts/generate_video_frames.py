#!/usr/bin/env python3
"""
Export each layer of a video frame separately as an RGBA PNG with transparency, all on
one fixed pixel grid. The layers are overlaid to build the composite frame downstream,
outside this script.

Each group of layers is emitted only when the input it needs is given, so a run can
export the basemap, the data layers, or both.

Basemap layers (--basemap-style):
  1. Split the MapLibre style into an "underlay" style (background .. water-not-in-icm)
     and an "overlay" style (aeroway-taxiway .. label).
  2. Render each style with headless Chromium + maplibre-gl-js against the live
     basemap-vector/basemap-raster tile sources, and georeference the screenshots
     from the browser's own reported view bounds.
  3. Reproject both renders onto the exact target grid (CRS/origin/pixel size/size
     given below), preserving alpha.
  4. Write basemap_underlay.png and basemap_overlay.png.

Data layers (--historic-data, the historic persistent-change raster):
  1. Color the categorical raster at its own 15 m resolution, then resample the finished
     image onto the target grid with bicubic interpolation, so each video pixel blends the
     colors of everything it covers instead of taking one source pixel's class.
  2. Write historic_1932.png, the state at the start of the record: everything that later
     changed is drawn in its pale "hasn't happened yet" color, land where the record shows
     loss and water where it shows gain.
  3. For each year group in HISTORIC_GROUPS, write historic_<group>.png. Color follows a
     period's role in the frame, not which period it is: that group is the current period
     and takes the saturated loss/gain pair, the groups before it take the pale pair, the
     groups after it stay the "hasn't happened yet" gray, and everything else is
     transparent.
  4. The last group ends in 2024, past the historic raster's 2015 end date, so with
     --land-water-data and --baseline-land-water-data it is written from a comparison
     against a 2024 land/water raster instead of from the historic raster alone: both the
     group's own historic classes and the pixels that flipped land/water state by 2024 are
     drawn as the current period. The 2015 state is the baseline raster with the historic
     change classes burned in, so ground the historic raster never classified still gets
     compared. The comparison runs at the historic raster's own 15 m resolution and only
     then gets colored and resampled to the video grid, so the inputs are compared where
     they describe the same ground. Without that data step 3 writes the group from the
     historic record alone, which stops in 2015.

Confidence layers (--confidence-data, a directory of lnd_conf_chng rasters):
  1. Symbolize each raster at its own 30 m resolution by classifying the signed value
     against CONFIDENCE_BREAKS. An lnd_conf_chng value is a plain difference against the
     base year, so its sign is the direction of change and the ramps read as "moved toward
     water" (blue) and "moved toward land" (green), each running pale where the pixel has
     barely moved to dark where it has flipped outright. The two bands either side of zero
     are transparent, so the layer covers only ground whose confidence has actually moved.
  2. Resample that image onto the target grid with bicubic interpolation, the same way the
     categorical layers are, and write confidence_<year>.png per raster.

With --save-full, every data layer also gets a <layer>_full.tif: the same colorized
image, georeferenced, at the resolution it was colored on rather than the video grid. That
is the input the bicubic resample works from, so it is what to look at to tell whether
something missing from a frame was lost in the resample or was never in the data.

Stack the results in order — basemap_underlay, historic_<group> or confidence_<year>,
basemap_overlay — to get one frame.

Requires (installed in .venv): rasterio, numpy, Pillow, pyproj, playwright
(+ `playwright install chromium` once). The render page (maplibre-gl loaded from
a CDN) is embedded in this script and written out to render/render.html at runtime,
so no local maplibre-gl / node_modules install is needed.
"""

import http.server
import json
import re
import threading
from pathlib import Path

import click
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import reproject

PROJECT_DIR = Path(__file__).resolve().parent
RENDER_DIR = PROJECT_DIR / "render"

# --- Target frame grid (from the gdalinfo dump for frame.tif) ---------------------
TARGET_CRS = "EPSG:26915"
TARGET_TRANSFORM = Affine(
    276.487678933628899,
    0.0,
    302182.296454079449177,
    # top-left y shifted up ~150px (150 * 289.290718890394999) so the data layer's
    # top edge isn't cut off
    0.0,
    -289.290718890394999,
    3452805.148774370176109,
)
TARGET_WIDTH = 2520
TARGET_HEIGHT = 1080

UNDERLAY_LAST_LAYER = "water-not-in-icm"
OVERLAY_FIRST_LAYER = "aeroway-taxiway"

# --- Historic land change classification -----------------------------------------
# In sim_3381_persistent_change_spatial_data_092616.img (see historic_data_legend.png
# and scripts/convert_historic_land_change.py) class values 1..19 are the "persistent
# land loss" observation periods in chronological order, and 20..38 are the matching
# "persistent land gain" periods. 0 is "no persistent change".
# Loss period v spans HISTORIC_PERIOD_YEARS[v - 1] .. HISTORIC_PERIOD_YEARS[v]; the
# corresponding gain period is class value GAIN_VALUE_OFFSET + v.
HISTORIC_PERIOD_YEARS = [
    1932, 1956, 1973, 1975, 1977, 1985, 1988, 1990, 1995, 1998,
    1999, 2002, 2004, 2006, 2008, 2009, 2010, 2013, 2014, 2015,
]  # fmt: skip
N_HISTORIC_PERIODS = len(HISTORIC_PERIOD_YEARS) - 1  # 19
GAIN_VALUE_OFFSET = N_HISTORIC_PERIODS

# The year groups to emit one PNG per. The int is the index of the last loss period in
# the group, so each group covers the periods after the previous group's through its own.
# The last group runs past the end of the historic record: its label ends in 2024 but the
# raster's periods stop at 2015, so the 2015-2024 part of it comes from the land/water
# comparison below and the frame is only complete with --land-water-data.
HISTORIC_GROUPS = [
    ("1932_1956", 1),
    ("1956_1985", 5),
    ("1985_2004", 12),
    ("2004_2024", 19),
]

_group_ends = [last for _, last in HISTORIC_GROUPS]
assert (
    _group_ends == sorted(set(_group_ends)) and _group_ends[-1] == N_HISTORIC_PERIODS
), f"HISTORIC_GROUPS must partition periods 1..{N_HISTORIC_PERIODS} in order, got {_group_ends}"

# Color depends on a period's role in the frame being drawn rather than on which period it
# is (see historical_colors_revised.png): the group the frame is "on" is the current
# period, the groups before it are previous, and the groups after it haven't happened yet
# and are drawn in a pale gray.
CURRENT_LOSS_COLOR = "#256B84"
CURRENT_GAIN_COLOR = "#4D7141"
PREVIOUS_LOSS_COLOR = "#A0C7D4"
PREVIOUS_GAIN_COLOR = "#ABCB9F"
FUTURE_LOSS_COLOR = "#E4E4E4"
FUTURE_GAIN_COLOR = "#F1F1F1"

# The series opens on the state at the start of the record, before any group's change has
# happened: every classified pixel is drawn in its "future" color.
NO_GROUP_SELECTED = -1
INITIAL_STATE_LABEL = str(HISTORIC_PERIOD_YEARS[0])

# --- 2015-2024 land/water comparison ---------------------------------------------
# The historic raster's period of observation ends in 2015, so the last group is
# extended by comparing it against a 2024 land/water raster. That raster uses the
# lnd_type classification from cpra/mp/data/structure/legends.csv: 1 Vegetated Wetland,
# 2 Open Water, 3 Bare/Unvegetated Wetland, 4 Upland/Developed Areas, 5 Flotant Marsh.
# Only class 2 is water; every other valid class is some form of land.
LND_TYPE_OPEN_WATER = 2

# The comparison finishes the last group rather than adding one of its own, so it writes
# that group's frame: the historic classes and the 2015-2024 transitions are the same
# current period and belong in the same PNG.
LAND_WATER_GROUP_LABEL = HISTORIC_GROUPS[-1][0]

# Land/water state codes the lnd_type classes are reduced to before comparing.
LW_UNKNOWN, LW_LAND, LW_WATER = 0, 1, 2

# Synthetic class values appended to the historic classification to record the outcome of
# the 2015-2024 comparison. Keeping the result a single categorical raster means it can be
# colored and resampled to the video grid in one pass, the same way the historic classes
# are.
LAND_TO_WATER_CLASS = 39
WATER_TO_LAND_CLASS = 40

# --- Color helpers ----------------------------------------------------------------


def hex_to_rgb(color: str):
    h = color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def hex_to_rgba(color: str):
    return (*hex_to_rgb(color), 255)


# --- Land-change confidence layers -----------------------------------------------
# Each confidence raster holds lnd_conf_chng (see scripts/calculate_land_change_confidence.py):
# a year's land/water confidence minus the base year's, with no offset applied to either
# side, so the value is a plain difference on [-1, 1] and its sign is the direction of
# change:
#
#   -1  confidently land in the base year, confidently water now   (land lost)
#    0  confidence unmoved since the base year                     (unchanged)
#   +1  confidently water in the base year, confidently land now   (land gained)
#
# The bins classify that signed value, so the ramps read as direction of change: blue for
# pixels that have moved toward water, green for pixels that have moved toward land, each
# running pale where the pixel has barely moved to dark where it has flipped outright.
# Each band runs from its own break up to the next, so the breaks are closed lower bounds
# and the last band has no upper bound.
# Opt3
# CONFIDENCE_BREAKS = [-1, -0.67, -0.42, -0.09, 0, 0.17, 0.50, 0.75]
# Opt4
CONFIDENCE_BREAKS = [-1, -0.75, -0.42, -0.09, 0, 0.17, 0.50, 0.83]

# The 3-stop ramps from the earlier symbology (data-input/confidence_groups.png), pale to
# dark, used as given rather than resampled to a longer ramp.
CONFIDENCE_LOSS_RAMP = ["#A0C7D4", "#4999B6", "#256B84"]
CONFIDENCE_GAIN_RAMP = ["#ABCB9F", "#658859", "#1F4314"]

# The two bands either side of zero, [-0.09, 0) and [0, 0.17). A pixel this close to its
# base-year confidence has not meaningfully moved, so it is left out of the layer rather
# than given a color of its own.
TRANSPARENT_BAND = None

# One color per band, in break order. The loss ramp is reversed because the bands run from
# the most negative value up: the first band is the far "now water" end and so takes the
# darkest blue, and the third is the edge of the transparent middle and takes the palest.
CONFIDENCE_BAND_COLORS = CONFIDENCE_LOSS_RAMP[::-1] + [TRANSPARENT_BAND, TRANSPARENT_BAND] + CONFIDENCE_GAIN_RAMP

assert len(CONFIDENCE_BAND_COLORS) == len(CONFIDENCE_BREAKS), (
    f"expected one color per band, got {len(CONFIDENCE_BAND_COLORS)} colors " f"for {len(CONFIDENCE_BREAKS)} breaks"
)

# Confidence outputs are labeled by the year their filename ends in, so a directory of
# calendar_year=2025.tif ... calendar_year=2077.tif becomes confidence_2025.png ... etc.
CONFIDENCE_YEAR_PATTERN = re.compile(r"(\d{4})$")

# Pin to the version this project was developed against (see package.json).
MAPLIBRE_VERSION = "5.24.0"
MAPLIBRE_JS = f"https://unpkg.com/maplibre-gl@{MAPLIBRE_VERSION}/dist/maplibre-gl.js"
MAPLIBRE_CSS = f"https://unpkg.com/maplibre-gl@{MAPLIBRE_VERSION}/dist/maplibre-gl.css"

RENDER_HTML = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<link rel="stylesheet" href="{MAPLIBRE_CSS}" />
<style>
  html, body {{ margin: 0; padding: 0; background: transparent; }}
  #map {{ position: absolute; top: 0; left: 0; }}
</style>
<script src="{MAPLIBRE_JS}"></script>
</head>
<body>
<div id="map"></div>
<script>
  window.renderDone = false;
  window.renderError = null;

  async function run() {{
    const params = new URLSearchParams(window.location.search);
    const styleUrl = params.get('style');
    const w = parseInt(params.get('w'), 10);
    const h = parseInt(params.get('h'), 10);
    const minLon = parseFloat(params.get('minLon'));
    const minLat = parseFloat(params.get('minLat'));
    const maxLon = parseFloat(params.get('maxLon'));
    const maxLat = parseFloat(params.get('maxLat'));

    document.getElementById('map').style.width = w + 'px';
    document.getElementById('map').style.height = h + 'px';

    const styleResp = await fetch(styleUrl);
    const style = await styleResp.json();

    const map = new maplibregl.Map({{
      container: 'map',
      style: style,
      center: [0, 0],
      zoom: 0,
      interactive: false,
      attributionControl: false,
      fadeDuration: 0,
      preserveDrawingBuffer: true,
      canvasContextAttributes: {{ alpha: true, premultipliedAlpha: false }}
    }});

    map.on('error', (e) => {{
      window.renderError = String(e && e.error && e.error.message || e);
      console.error('maplibre error', window.renderError);
    }});

    map.once('load', () => {{
      map.fitBounds(
        [[minLon, minLat], [maxLon, maxLat]],
        {{ padding: 0, animate: false, linear: true }}
      );
      map.once('idle', () => {{
        const b = map.getBounds();
        window.actualBounds = {{
          west: b.getWest(), south: b.getSouth(),
          east: b.getEast(), north: b.getNorth()
        }};
        window.renderDone = true;
      }});
    }});
  }}

  run().catch(e => {{ window.renderError = String(e); }});
</script>
</body>
</html>
"""


def split_style(style_path: Path):
    style = json.loads(style_path.read_text())
    layers = style["layers"]
    ids = [l["id"] for l in layers]
    under_idx = ids.index(UNDERLAY_LAST_LAYER)
    over_idx = ids.index(OVERLAY_FIRST_LAYER)
    assert over_idx == under_idx + 1, (
        f"expected '{OVERLAY_FIRST_LAYER}' to immediately follow "
        f"'{UNDERLAY_LAST_LAYER}', found layers in between: "
        f"{ids[under_idx + 1:over_idx]}"
    )

    underlay_style = {**style, "layers": layers[: under_idx + 1]}
    overlay_style = {**style, "layers": layers[over_idx:]}
    return underlay_style, overlay_style


def lonlat_envelope_of_target(buffer_frac=0.02):
    """Sample the boundary of the target UTM extent densely and return a padded
    lon/lat envelope that fully contains it (UTM rectangle edges are not aligned
    to meridians/parallels, so corner-only sampling can clip the middle of an edge)."""
    to_wgs84 = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)

    x0, y0 = TARGET_TRANSFORM.c, TARGET_TRANSFORM.f
    x1 = x0 + TARGET_TRANSFORM.a * TARGET_WIDTH
    y1 = y0 + TARGET_TRANSFORM.e * TARGET_HEIGHT

    xs, ys = [], []
    n = 64
    for i in range(n + 1):
        t = i / n
        # top and bottom edges
        xs += [x0 + (x1 - x0) * t, x0 + (x1 - x0) * t]
        ys += [y0, y1]
        # left and right edges
        xs += [x0, x1]
        ys += [y0 + (y1 - y0) * t, y0 + (y1 - y0) * t]

    lons, lats = to_wgs84.transform(xs, ys)
    minlon, maxlon = min(lons), max(lons)
    minlat, maxlat = min(lats), max(lats)

    dlon = (maxlon - minlon) * buffer_frac
    dlat = (maxlat - minlat) * buffer_frac
    return minlon - dlon, minlat - dlat, maxlon + dlon, maxlat + dlat


def serve_project_dir(directory: Path):
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=str(directory), **kw)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def render_style(page, base_url, style_rel_url, bounds, width, height):
    minlon, minlat, maxlon, maxlat = bounds
    url = (
        f"{base_url}/render/render.html?style={style_rel_url}"
        f"&w={width}&h={height}"
        f"&minLon={minlon}&minLat={minlat}&maxLon={maxlon}&maxLat={maxlat}"
    )
    page.goto(url)
    page.wait_for_function("window.renderDone === true || window.renderError !== null", timeout=60000)
    err = page.evaluate("window.renderError")
    if err:
        raise RuntimeError(f"maplibre render failed for {style_rel_url}: {err}")
    actual_bounds = page.evaluate("window.actualBounds")
    png_bytes = page.screenshot(omit_background=True)
    return png_bytes, actual_bounds


def georeference_render(png_bytes, actual_bounds):
    """Return (rgba uint8 array [4,H,W], crs, transform) for a rendered screenshot,
    using the browser-reported view bounds (which are exact for the flat/no-tilt
    Mercator camera we use) to build the Web Mercator affine transform."""
    img = Image.open(__import__("io").BytesIO(png_bytes)).convert("RGBA")
    arr = np.asarray(img)  # H, W, 4
    h, w = arr.shape[:2]

    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    minx, maxy = to_merc.transform(actual_bounds["west"], actual_bounds["north"])
    maxx, miny = to_merc.transform(actual_bounds["east"], actual_bounds["south"])

    pixel_w = (maxx - minx) / w
    pixel_h = (maxy - miny) / h
    transform = Affine(pixel_w, 0.0, minx, 0.0, -pixel_h, maxy)

    bands = np.moveaxis(arr, 2, 0)  # 4, H, W
    return bands, "EPSG:3857", transform, w, h


def reproject_to_target(src_bands, src_crs, src_transform, resampling):
    n = src_bands.shape[0]
    dst = np.zeros((n, TARGET_HEIGHT, TARGET_WIDTH), dtype=src_bands.dtype)
    reproject(
        source=src_bands,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=TARGET_TRANSFORM,
        dst_crs=TARGET_CRS,
        resampling=resampling,
        num_threads=4,
    )
    return dst


def unmultiply_alpha(bands):
    """Divide premultiplied RGBA float bands back out to straight alpha as uint8, which
    is what PNG stores. Interpolating resamplers can overshoot past the endpoints of the
    values they blend, so both alpha and the recovered color are clipped back into range.
    """
    a = np.clip(bands[3].astype(np.float32), 0.0, 255.0)
    unmul = np.where(a > 0.0, 255.0 / np.maximum(a, 1e-6), 0.0)
    rgb = np.clip(bands[:3].astype(np.float32) * unmul, 0.0, 255.0)
    return np.concatenate([rgb, a[None]], axis=0).round().astype(np.uint8)


def reproject_rgba_to_target(src_bands, src_crs, src_transform, resampling):
    """Reproject an RGBA render onto the target grid. Alpha is premultiplied first (and
    divided back out afterwards) so that interpolating resamplers don't bleed the color
    of fully transparent pixels into partially transparent edges as dark fringing.
    Work is done in float32 to avoid quantizing twice."""
    a = src_bands[3].astype(np.float32) / 255.0
    premul = np.concatenate([src_bands[:3].astype(np.float32) * a, src_bands[3:].astype(np.float32)], axis=0)

    return unmultiply_alpha(reproject_to_target(premul, src_crs, src_transform, resampling))


def resample_image_to_target(bands, src_crs, src_transform):
    """Resample a colored data layer, held at its source resolution as premultiplied RGBA
    bands, onto the target grid with bicubic interpolation.

    Coloring first and resampling the image is what lets a video pixel show a blend of
    everything inside it. Resampling the classes first can't: the values are class ids, so
    the only valid resamplers pick one winning class per target pixel, and with each target
    pixel covering ~340 historic pixels that means either point-sampling (thin, scattered
    change survives, but as speckle) or a majority vote (coherent, but change covering less
    than half a target pixel disappears entirely). Bicubic over the image instead
    downscales it the way any photograph is downscaled, weighting every source pixel in the
    footprint — GDAL widens the kernel to the target pixel size when downsampling, so no
    data is skipped.

    Both colorizers emit premultiplied RGBA for free, since every color they assign is
    either fully opaque or fully transparent and black, so transparent ground contributes
    no color to a blend. Alpha, being 0 or 255 at the source, comes out fractional exactly
    where a target pixel is partly covered, which is what antialiases the layer's edges.

    Target pixels the source doesn't reach keep the destination's transparent zeros.
    """
    return unmultiply_alpha(reproject_to_target(bands, src_crs, src_transform, Resampling.cubic))


def colorize(classes: np.ndarray, lut: np.ndarray):
    """Apply a (256,4) RGBA lookup table to a categorical raster, returning (4,H,W) bands.

    Band-major so the result can be resampled without a copy, and one band at a time into
    a preallocated output because the full-resolution historic raster is ~850 M pixels: at
    that size, every extra RGBA copy of it is another 3.4 GB.
    """
    bands = np.empty((4,) + classes.shape, dtype=np.uint8)
    for band in range(4):
        np.take(lut[:, band], classes, out=bands[band])
    return bands


def read_classes(data_path: Path):
    """Read a categorical single-band raster at its own resolution, with its grid."""
    with rasterio.open(data_path) as ds:
        return ds.read(1), ds.crs, ds.transform


def read_values(data_path: Path):
    """Read a continuous single-band raster at its own resolution, with its grid. Nodata
    becomes NaN, which symbolizes transparent (every comparison against NaN is False)."""
    with rasterio.open(data_path) as ds:
        values = ds.read(1).astype(np.float32)
        nodata = ds.nodatavals[0]
        if nodata is not None:
            values[values == nodata] = np.nan
        return values, ds.crs, ds.transform


def historic_group_lut(selected: int):
    """Build a (256,4) uint8 RGBA lookup table mapping historic class values to colors for
    a frame drawn on the year group at index `selected`. That group is the current period
    and gets the current loss/gain colors; groups before it are previous and groups after
    it are future. Class values with no group (0, and 39..255) are left fully transparent.

    Two indexes outside the range of groups are meaningful. NO_GROUP_SELECTED puts every
    group in the future, which is the state at the start of the record: nothing has changed
    yet, so every loss class is still land and every gain class is still water. An index
    past the last group makes every group previous, for a frame whose current period comes
    after the historic record ends.
    """
    lut = np.zeros((256, 4), dtype=np.uint8)
    first = 1
    for group_idx, (_, last) in enumerate(HISTORIC_GROUPS):
        if group_idx == selected:
            loss_color, gain_color = CURRENT_LOSS_COLOR, CURRENT_GAIN_COLOR
        elif group_idx < selected:
            loss_color, gain_color = PREVIOUS_LOSS_COLOR, PREVIOUS_GAIN_COLOR
        else:
            loss_color, gain_color = FUTURE_LOSS_COLOR, FUTURE_GAIN_COLOR
        for period in range(first, last + 1):
            lut[period] = hex_to_rgba(loss_color)
            lut[GAIN_VALUE_OFFSET + period] = hex_to_rgba(gain_color)
        first = last + 1
    return lut


def write_historic_group_pngs(
    classes: np.ndarray, src_crs, src_transform, output_dir: Path, save_full: bool, skip_last_group: bool
):
    """Color and resample one frame per year group, plus the frame the series opens on.

    skip_last_group leaves the last group to write_land_water_group_png. That group ends
    after the historic record does, so this raster holds only the first half of it; the
    frame written here would be missing the 2015-2024 change and overwritten anyway.
    """
    groups = HISTORIC_GROUPS[:-1] if skip_last_group else HISTORIC_GROUPS
    selections = [(INITIAL_STATE_LABEL, NO_GROUP_SELECTED)]
    selections += [(label, group_idx) for group_idx, (label, _) in enumerate(groups)]
    for label, selected in selections:
        lut = historic_group_lut(selected)
        write_historic_group_png(output_dir, label, lut, classes, src_crs, src_transform, save_full)


def write_historic_group_png(
    output_dir: Path, label: str, lut: np.ndarray, classes: np.ndarray, src_crs, src_transform, save_full: bool
):
    print(f"Coloring historic_{label} at source resolution, then resampling to target grid (cubic)...")
    bands = colorize(classes, lut)
    if save_full:
        write_full_tif(output_dir / f"historic_{label}_full.tif", bands, src_crs, src_transform)
    write_historic_png(output_dir, label, resample_image_to_target(bands, src_crs, src_transform))


def read_land_water_state(land_water_path: Path):
    """Read a lnd_type raster at its full resolution, reduced to LW_UNKNOWN / LW_LAND /
    LW_WATER. Only class 2 is water; every other valid class is a form of land, and
    anything below class 1 (the legend's "No Data", and the raster's nodata) is unknown."""
    with rasterio.open(land_water_path) as ds:
        values = ds.read(1)
        state = np.where(values == LND_TYPE_OPEN_WATER, LW_WATER, LW_LAND).astype(np.uint8)
        state[values < 1] = LW_UNKNOWN
        return state, ds.crs, ds.transform


def read_land_water_state_at_historic_grid(land_water_path: Path, historic_ds):
    """read_land_water_state, upsampled onto the historic raster's grid. Nearest is the
    right choice going from 30 m to 15 m: each historic pixel simply takes the state of
    the land/water pixel containing it."""
    state, state_crs, state_transform = read_land_water_state(land_water_path)
    dst = np.zeros((historic_ds.height, historic_ds.width), dtype=np.uint8)
    reproject(
        source=state,
        destination=dst,
        src_crs=state_crs,
        src_transform=state_transform,
        src_nodata=LW_UNKNOWN,
        dst_crs=historic_ds.crs,
        dst_transform=historic_ds.transform,
        dst_nodata=LW_UNKNOWN,
        resampling=Resampling.nearest,
        num_threads=4,
    )
    return dst


def compare_land_water_at_historic_grid(historic_path: Path, baseline_path: Path, land_water_path: Path):
    """Compare a 2024 land/water raster against the 2015 land/water state, on the historic
    raster's own 15 m grid. Returns (classes, crs, transform) for that grid, with the
    converted pixels replaced by LAND_TO_WATER_CLASS / WATER_TO_LAND_CLASS.

    The 2015 state is built from the baseline land/water raster (2018, the closest in time
    available) with the historic raster's change classes burned in on top: each historic
    class records the last transition that persisted through the end of observation in
    2015, so a loss class is water in 2015 and a gain class is land, regardless of what the
    2018 baseline says. Pixels the historic raster leaves unclassified had no persistent
    change on record between 1932 and 2015, so they keep their 2018 state. Using a baseline
    is what lets the comparison see change on ground the historic raster never classified —
    which is where most new land loss actually happens.

    The comparison has to happen at this resolution rather than at the video grid. Each
    video pixel covers ~340 historic pixels and ~90 land/water pixels, so downsampling the
    inputs first leaves point samples that describe different patches of ground, and they
    disagree more often than the underlying data does.
    """
    with rasterio.open(historic_path) as ds:
        classes = ds.read(1)
        src_crs, src_transform = ds.crs, ds.transform
        state_2015 = read_land_water_state_at_historic_grid(baseline_path, ds)
        state_2024 = read_land_water_state_at_historic_grid(land_water_path, ds)

    loss = (classes >= 1) & (classes <= N_HISTORIC_PERIODS)
    gain = (classes > GAIN_VALUE_OFFSET) & (classes <= GAIN_VALUE_OFFSET + N_HISTORIC_PERIODS)
    state_2015[loss] = LW_WATER
    state_2015[gain] = LW_LAND

    # A pixel unclassified on either side yields no comparison, so it keeps its class.
    classes[(state_2015 == LW_LAND) & (state_2024 == LW_WATER)] = LAND_TO_WATER_CLASS
    classes[(state_2015 == LW_WATER) & (state_2024 == LW_LAND)] = WATER_TO_LAND_CLASS
    return classes, src_crs, src_transform


def land_water_group_lut():
    """The last historic group as the current period, the groups before it as previous,
    plus the two 2015-2024 transition classes. This frame is the last group, which runs
    from 2004 past the end of the historic record to 2024, so both halves of it are
    current: the record's own 2004-2015 loss/gain classes and the transitions the
    comparison found on top of them."""
    lut = historic_group_lut(len(HISTORIC_GROUPS) - 1)
    lut[LAND_TO_WATER_CLASS] = hex_to_rgba(CURRENT_LOSS_COLOR)
    lut[WATER_TO_LAND_CLASS] = hex_to_rgba(CURRENT_GAIN_COLOR)
    return lut


def write_land_water_group_png(
    historic_path: Path, baseline_path: Path, land_water_path: Path, output_dir: Path, save_full: bool
):
    """Extend the historic series past the end of its 2015 period of observation: compare
    the 2024 land/water raster against the 2015 state at the historic raster's own
    resolution, then color that result and resample it to the video grid."""
    print(f"Comparing {land_water_path.name} against {baseline_path.name} + {historic_path.name} at 15 m...")
    classes, src_crs, src_transform = compare_land_water_at_historic_grid(historic_path, baseline_path, land_water_path)
    write_historic_group_png(
        output_dir, LAND_WATER_GROUP_LABEL, land_water_group_lut(), classes, src_crs, src_transform, save_full
    )


def confidence_rgba(values: np.ndarray):
    """Symbolize an lnd_conf_chng array against CONFIDENCE_BREAKS, as (4,H,W) RGBA bands to
    match colorize().

    Classifies the signed value, so each band takes one color from CONFIDENCE_BAND_COLORS.
    Each band includes its lower break and excludes the next. The two bands either side of
    zero are TRANSPARENT_BAND and are simply left at the array's transparent zeros, so the
    layer drops out over ground whose confidence has barely moved and the frame beneath it
    shows through there.

    The first band's lower bound is left open rather than closed at CONFIDENCE_BREAKS[0], so
    that a value sitting exactly on the -1 floor cannot fall out of the classification on a
    float comparison. Nodata arrives as NaN and stays transparent, since every comparison
    against NaN is False.
    """
    rgba = np.zeros((4,) + values.shape, dtype=np.uint8)
    lowers = [-np.inf] + CONFIDENCE_BREAKS[1:]
    uppers = CONFIDENCE_BREAKS[1:] + [np.inf]
    for lower, upper, color in zip(lowers, uppers, CONFIDENCE_BAND_COLORS):
        if color is TRANSPARENT_BAND:
            continue
        band = (values >= lower) & (values < upper)
        rgba[:, band] = np.array(hex_to_rgba(color), dtype=np.uint8)[:, None]
    return rgba


def confidence_label(path: Path):
    """The output label for a confidence raster: the year its name ends in, or, failing
    that, the whole file name."""
    match = CONFIDENCE_YEAR_PATTERN.search(path.stem)
    return match.group(1) if match else path.stem


def write_confidence_pngs(confidence_dir: Path, output_dir: Path, save_full: bool):
    paths = sorted(confidence_dir.glob("*.tif"))
    if not paths:
        raise click.UsageError(f"no .tif files found in {confidence_dir}")
    for path in paths:
        label = confidence_label(path)
        print(f"Symbolizing {path.name} at source resolution and resampling to target grid (cubic)...")
        values, src_crs, src_transform = read_values(path)
        bands = confidence_rgba(values)
        if save_full:
            write_full_tif(output_dir / f"confidence_{label}_full.tif", bands, src_crs, src_transform)
        output_path = output_dir / f"confidence_{label}.png"
        write_png(output_path, resample_image_to_target(bands, src_crs, src_transform))
        print(f"Wrote {output_path}")


def write_historic_png(output_dir: Path, label: str, rgba_bands: np.ndarray):
    output_path = output_dir / f"historic_{label}.png"
    write_png(output_path, rgba_bands)
    print(f"Wrote {output_path}")


def write_png(path: Path, rgba_bands: np.ndarray):
    """Write a (4,H,W) uint8 RGBA array as a PNG with transparency."""
    Image.fromarray(np.moveaxis(rgba_bands, 0, 2)).save(path, optimize=True)


def write_full_tif(path: Path, rgba_bands: np.ndarray, src_crs, src_transform):
    """Write a colorized data layer at the resolution it was colored on, as a GeoTIFF.

    A PNG is not usable at this size — the historic layer is 42889x19763, which is past
    what most viewers, and Pillow's own decompression-bomb guard, will open — and it would
    drop the georeferencing that makes the image comparable against the rasters it came
    from. Tiled and compressed so it stays a browsable fraction of its 3.4 GB uncompressed
    size.
    """
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=rgba_bands.shape[2],
        height=rgba_bands.shape[1],
        count=4,
        dtype=rgba_bands.dtype,
        crs=src_crs,
        transform=src_transform,
        photometric="RGB",
        alpha="YES",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="deflate",
        predictor=2,
        num_threads="all_cpus",
        bigtiff="IF_SAFER",
    ) as ds:
        ds.write(rgba_bands)
    print(f"Wrote {path}")


@click.command(help=__doc__)
@click.option(
    "--basemap-style",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="path to MapLibre style JSON; emits basemap_underlay.png and basemap_overlay.png. "
    "Omit to skip the basemap layers.",
)
@click.option(
    "--historic-data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="path to the historic persistent-change raster; emits one historic_<group>.png per year group. "
    "Omit to skip the data layers.",
)
@click.option(
    "--land-water-data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=f"path to the 2024 lnd_type land/water raster; finishes the last year group by comparing it "
    f"against the 2015 state, and writes historic_{LAND_WATER_GROUP_LABEL}.png in place of the "
    "historic-only frame for that group. Requires --historic-data and --baseline-land-water-data.",
)
@click.option(
    "--baseline-land-water-data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="path to a lnd_type land/water raster from near 2015; the historic change classes are burned "
    "into it to build the 2015 state that --land-water-data is compared against.",
)
@click.option(
    "--confidence-data",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="path to a directory of lnd_conf_chng rasters, e.g. the scenario_id=0 directory written by "
    "scripts/calculate_land_change_confidence.py; emits one confidence_<year>.png per .tif in it. "
    "Omit to skip the confidence layers.",
)
@click.option("--output", help="output directory for the layer PNGs", required=True)
@click.option(
    "--save-full",
    is_flag=True,
    help="also write each data layer's colorized image at the resolution it was colored on, before the "
    "resample to the video grid, as <layer>_full.tif. GeoTIFF rather than PNG because these are much "
    "too large for one (the historic layers are 42889x19763) and because the georeferencing is what "
    "makes them comparable against the rasters they came from. Costs about 12 s and 12 MB per historic "
    "layer, which is a handful of flat colors on mostly transparent ground and so compresses hard",
)
@click.option("--scale", type=float, default=2.0, show_default=True, help="device scale factor for supersampling")
@click.option("--keep-intermediate", is_flag=True)
def main(
    basemap_style,
    historic_data,
    land_water_data,
    baseline_land_water_data,
    confidence_data,
    output,
    save_full,
    scale,
    keep_intermediate,
):
    if land_water_data is not None and (historic_data is None or baseline_land_water_data is None):
        raise click.UsageError(
            "--land-water-data needs both --historic-data and --baseline-land-water-data: the 2015 "
            "land/water state it compares against is the baseline raster with the historic change "
            "classes burned in"
        )
    if baseline_land_water_data is not None and land_water_data is None:
        raise click.UsageError("--baseline-land-water-data does nothing without --land-water-data")
    if basemap_style is None and historic_data is None and confidence_data is None:
        raise click.UsageError("nothing to do: pass --basemap-style, --historic-data and/or --confidence-data")
    if save_full and historic_data is None and confidence_data is None:
        raise click.UsageError(
            "--save-full only applies to the data layers: pass --historic-data and/or --confidence-data"
        )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if historic_data is not None:
        print(f"Reading {historic_data.name} at source resolution...")
        classes, src_crs, src_transform = read_classes(historic_data)
        write_historic_group_pngs(
            classes, src_crs, src_transform, output_dir, save_full, skip_last_group=land_water_data is not None
        )
        if land_water_data is not None:
            write_land_water_group_png(historic_data, baseline_land_water_data, land_water_data, output_dir, save_full)

    if confidence_data is not None:
        write_confidence_pngs(confidence_data, output_dir, save_full)

    if basemap_style is None:
        return

    from playwright.sync_api import sync_playwright

    style_path = basemap_style
    underlay_style, overlay_style = split_style(style_path)
    RENDER_DIR.mkdir(exist_ok=True)
    (RENDER_DIR / "underlay_style.json").write_text(json.dumps(underlay_style))
    (RENDER_DIR / "overlay_style.json").write_text(json.dumps(overlay_style))
    (RENDER_DIR / "render.html").write_text(RENDER_HTML)

    bounds = lonlat_envelope_of_target()

    httpd, port = serve_project_dir(PROJECT_DIR)
    base_url = f"http://127.0.0.1:{port}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"],
            )
            context = browser.new_context(
                viewport={"width": TARGET_WIDTH, "height": TARGET_HEIGHT},
                device_scale_factor=scale,
            )
            page = context.new_page()

            print("Rendering underlay layers...")
            under_png, under_bounds = render_style(
                page, base_url, "/render/underlay_style.json", bounds, TARGET_WIDTH, TARGET_HEIGHT
            )
            print("Rendering overlay layers...")
            over_png, over_bounds = render_style(
                page, base_url, "/render/overlay_style.json", bounds, TARGET_WIDTH, TARGET_HEIGHT
            )
            browser.close()
    finally:
        httpd.shutdown()

    under_bands, under_crs, under_tf, uw, uh = georeference_render(under_png, under_bounds)
    over_bands, over_crs, over_tf, ow, oh = georeference_render(over_png, over_bounds)

    if keep_intermediate:
        Image.fromarray(np.moveaxis(under_bands, 0, 2)).save(PROJECT_DIR / "_underlay_raw.png")
        Image.fromarray(np.moveaxis(over_bands, 0, 2)).save(PROJECT_DIR / "_overlay_raw.png")

    print("Reprojecting underlay to target grid...")
    underlay_t = reproject_rgba_to_target(under_bands, under_crs, under_tf, Resampling.bilinear)
    print("Reprojecting overlay to target grid...")
    overlay_t = reproject_rgba_to_target(over_bands, over_crs, over_tf, Resampling.bilinear)

    # The data layer belongs between these two; it is composited downstream.
    for name, bands in (("basemap_underlay.png", underlay_t), ("basemap_overlay.png", overlay_t)):
        output_path = output_dir / name
        write_png(output_path, bands)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
