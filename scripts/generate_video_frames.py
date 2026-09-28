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
  1. Warp the categorical raster onto the same target grid, nearest neighbor by default
     or a majority downsample with --majority.
  2. Write historic_1932.png, the state at the start of the record: everything that later
     changed is drawn in its pale "hasn't happened yet" color, land where the record shows
     loss and water where it shows gain.
  3. For each year group in HISTORIC_GROUPS, write historic_<group>.png. Color follows a
     period's role in the frame, not which period it is: that group is the current period
     and takes the saturated loss/gain pair, the groups before it take the pale pair, the
     groups after it stay the "hasn't happened yet" gray, and everything else is
     transparent.
  4. With --land-water-data and --baseline-land-water-data, extend the series one group
     past the historic raster's 2015 end date: write historic_2015_2024.png, where every
     historic group is a previous period and the pixels that flipped land/water state by
     2024 are the current one. The 2015 state is the baseline raster with the historic
     change classes burned in, so ground the historic raster never classified still gets
     compared. The
     comparison runs at the historic raster's own 15 m resolution and only then gets
     downsampled to the video grid, so the inputs are compared where they describe the
     same ground.

Confidence layers (--confidence-data, a directory of confidence rasters):
  1. Warp each raster onto the target grid, averaging source pixels, since unlike the
     categorical layers these hold a continuous confidence difference on [-1, 1].
  2. Write confidence_<year>.png per raster, binning by magnitude per
     data-input/confidence_groups.png: a transparent band around zero, then three
     progressively darker bands, green where confidence in land is growing and blue where
     confidence in water is.

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
HISTORIC_GROUPS = [
    ("1932_1956", 1),
    ("1956_1975", 3),
    ("1975_1985", 5),
    ("1985_1995", 8),
    ("1995_2004", 12),
    ("2004_2006", 13),
    ("2006_2015", 19),
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
LAND_WATER_GROUP_LABEL = "2015_2024"

# Land/water state codes the lnd_type classes are reduced to before comparing.
LW_UNKNOWN, LW_LAND, LW_WATER = 0, 1, 2

# Synthetic class values appended to the historic classification to record the outcome of
# the 2015-2024 comparison. Keeping the result a single categorical raster means it can be
# downsampled to the video grid in one pass, the same way the historic classes are.
LAND_TO_WATER_CLASS = 39
WATER_TO_LAND_CLASS = 40

# --- Land-change confidence layers -----------------------------------------------
# Each confidence raster holds a projected year's land/water confidence minus the base
# year's, on [-1, 1]: positive means growing confidence in land, negative in water.
# data-input/confidence_groups.png bins these by magnitude, with a transparent band
# straddling zero, and picks the palette by sign. Each band runs from its own break up to
# the next, so the breaks below are lower bounds and the last band has no upper bound.
CONFIDENCE_BREAKS = [0.1, 0.3, 0.7]
CONFIDENCE_GAIN_COLORS = ["#ABCB9F", "#658859", "#1F4314"]
CONFIDENCE_LOSS_COLORS = ["#A0C7D4", "#4999B6", "#256B84"]

# Confidence outputs are labeled by the year their filename ends in, so a directory of
# test_land_change_confidence_2025.tif ... _2070.tif becomes confidence_2025.png ... etc.
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


def reproject_rgba_to_target(src_bands, src_crs, src_transform, resampling):
    """Reproject an RGBA render onto the target grid. Alpha is premultiplied first (and
    divided back out afterwards) so that interpolating resamplers don't bleed the color
    of fully transparent pixels into partially transparent edges as dark fringing.
    Work is done in float32 to avoid quantizing twice."""
    a = src_bands[3].astype(np.float32) / 255.0
    premul = np.concatenate([src_bands[:3].astype(np.float32) * a, src_bands[3:].astype(np.float32)], axis=0)

    dst = reproject_to_target(premul, src_crs, src_transform, resampling)

    dst_a = np.clip(dst[3], 0.0, 255.0)
    unmul = np.where(dst_a > 0.0, 255.0 / np.maximum(dst_a, 1e-6), 0.0)
    rgb = np.clip(dst[:3] * unmul, 0.0, 255.0)
    return np.concatenate([rgb, dst_a[None]], axis=0).round().astype(np.uint8)


def reproject_classes_to_target(data_path: Path, resampling):
    """Warp a categorical single-band raster onto the target grid. Only class-preserving
    resamplers are valid here — the values are class ids, so an averaging resampler
    would invent classes that don't exist. The source is 15 m and the target is
    ~277x289 m, so each target pixel covers ~340 source pixels: nearest takes whichever
    class the single sampled point lands on (keeps thin, scattered change, at the cost
    of speckle), while mode takes the class that dominates the footprint (coherent, but
    drops change that covers less than half a target pixel, since class 0 "no persistent
    change" wins the vote). Class 0 also doubles as the fill for target pixels the
    source doesn't cover."""
    with rasterio.open(data_path) as ds:
        nodata = ds.nodatavals[0]
        dst = np.full((TARGET_HEIGHT, TARGET_WIDTH), 0 if nodata is None else nodata, dtype=ds.dtypes[0])
        reproject(
            source=rasterio.band(ds, 1),
            destination=dst,
            dst_transform=TARGET_TRANSFORM,
            dst_crs=TARGET_CRS,
            src_nodata=nodata,
            dst_nodata=nodata,
            resampling=resampling,
            num_threads=4,
        )
    return dst


def reproject_class_array_to_target(classes, src_crs, src_transform, resampling):
    """Same as reproject_classes_to_target, for a categorical array already in memory."""
    dst = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=classes.dtype)
    reproject(
        source=classes,
        destination=dst,
        src_crs=src_crs,
        src_transform=src_transform,
        dst_transform=TARGET_TRANSFORM,
        dst_crs=TARGET_CRS,
        resampling=resampling,
        num_threads=4,
    )
    return dst


def reproject_values_to_target(data_path: Path):
    """Warp a continuous single-band raster onto the target grid, averaging the source
    pixels that fall in each target pixel.

    The categorical layers can only be resampled by picking a winning class, but these
    values are confidence fractions, so averaging them is meaningful — and at roughly 9x
    downsampling it uses all of the source data instead of point-sampling 1 pixel in 85.
    Nodata becomes NaN, which is excluded from the averages and symbolized transparent.
    """
    with rasterio.open(data_path) as ds:
        values = ds.read(1).astype(np.float32)
        nodata = ds.nodatavals[0]
        if nodata is not None:
            values[values == nodata] = np.nan
        dst = np.full((TARGET_HEIGHT, TARGET_WIDTH), np.nan, dtype=np.float32)
        reproject(
            source=values,
            destination=dst,
            src_crs=ds.crs,
            src_transform=ds.transform,
            src_nodata=np.nan,
            dst_transform=TARGET_TRANSFORM,
            dst_crs=TARGET_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.average,
            num_threads=4,
        )
    return dst


def hex_to_rgba(color: str):
    h = color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)


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


def write_historic_group_pngs(classes: np.ndarray, output_dir: Path):
    write_historic_png(output_dir, INITIAL_STATE_LABEL, historic_group_lut(NO_GROUP_SELECTED)[classes])
    for group_idx, (label, _) in enumerate(HISTORIC_GROUPS):
        rgba = historic_group_lut(group_idx)[classes]  # H, W, 4
        write_historic_png(output_dir, label, rgba)


def read_land_water_state(land_water_path: Path):
    """Read a lnd_type raster at its native resolution, reduced to LW_UNKNOWN / LW_LAND /
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
    raster's own 15 m grid, and return its classes with the converted pixels replaced by
    LAND_TO_WATER_CLASS / WATER_TO_LAND_CLASS.

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
        state_2015 = read_land_water_state_at_historic_grid(baseline_path, ds)
        state_2024 = read_land_water_state_at_historic_grid(land_water_path, ds)

    loss = (classes >= 1) & (classes <= N_HISTORIC_PERIODS)
    gain = (classes > GAIN_VALUE_OFFSET) & (classes <= GAIN_VALUE_OFFSET + N_HISTORIC_PERIODS)
    state_2015[loss] = LW_WATER
    state_2015[gain] = LW_LAND

    # A pixel unclassified on either side yields no comparison, so it keeps its class.
    classes[(state_2015 == LW_LAND) & (state_2024 == LW_WATER)] = LAND_TO_WATER_CLASS
    classes[(state_2015 == LW_WATER) & (state_2024 == LW_LAND)] = WATER_TO_LAND_CLASS
    return classes


def land_water_group_lut():
    """Every historic group as a previous period, plus the two 2015-2024 transition
    classes. 2015-2024 is this frame's current period, so it takes the current loss/gain
    colors and no group in the historic record is current — including the 2006-2015 one
    that was current in the frame before this."""
    lut = historic_group_lut(len(HISTORIC_GROUPS))
    lut[LAND_TO_WATER_CLASS] = hex_to_rgba(CURRENT_LOSS_COLOR)
    lut[WATER_TO_LAND_CLASS] = hex_to_rgba(CURRENT_GAIN_COLOR)
    return lut


def write_land_water_group_png(
    historic_path: Path, baseline_path: Path, land_water_path: Path, output_dir: Path, resampling
):
    """Extend the historic series past the end of its 2015 period of observation: compare
    the 2024 land/water raster against the 2015 state at the historic raster's own
    resolution, then downsample that result to the video grid."""
    print(f"Comparing {land_water_path.name} against {baseline_path.name} + {historic_path.name} at 15 m...")
    classes = compare_land_water_at_historic_grid(historic_path, baseline_path, land_water_path)

    print(f"Downsampling the comparison to target grid ({resampling.name})...")
    with rasterio.open(historic_path) as ds:
        downsampled = reproject_class_array_to_target(classes, ds.crs, ds.transform, resampling)

    write_historic_png(output_dir, LAND_WATER_GROUP_LABEL, land_water_group_lut()[downsampled])


def confidence_rgba(values: np.ndarray):
    """Symbolize a confidence difference array as RGBA per confidence_groups.png.

    Bins on magnitude, so the bands are symmetric about zero, and picks the green ramp
    where the value is positive and the blue ramp where it is negative. Each band includes
    its lower break and excludes the next one. Anything below the first break — including
    exactly zero — stays transparent, as does NaN, since every comparison against NaN is
    False.
    """
    rgba = np.zeros(values.shape + (4,), dtype=np.uint8)
    magnitude = np.abs(values)
    uppers = CONFIDENCE_BREAKS[1:] + [np.inf]
    for lower, upper, gain, loss in zip(CONFIDENCE_BREAKS, uppers, CONFIDENCE_GAIN_COLORS, CONFIDENCE_LOSS_COLORS):
        band = (magnitude >= lower) & (magnitude < upper)
        rgba[band & (values > 0)] = hex_to_rgba(gain)
        rgba[band & (values < 0)] = hex_to_rgba(loss)
    return rgba


def confidence_label(path: Path):
    """The output label for a confidence raster: the year its name ends in, or, failing
    that, the whole file name."""
    match = CONFIDENCE_YEAR_PATTERN.search(path.stem)
    return match.group(1) if match else path.stem


def write_confidence_pngs(confidence_dir: Path, output_dir: Path):
    paths = sorted(confidence_dir.glob("*.tif"))
    if not paths:
        raise click.UsageError(f"no .tif files found in {confidence_dir}")
    for path in paths:
        label = confidence_label(path)
        print(f"Reprojecting {path.name} to target grid (average)...")
        output_path = output_dir / f"confidence_{label}.png"
        write_png(output_path, np.moveaxis(confidence_rgba(reproject_values_to_target(path)), 2, 0))
        print(f"Wrote {output_path}")


def write_historic_png(output_dir: Path, label: str, rgba: np.ndarray):
    output_path = output_dir / f"historic_{label}.png"
    write_png(output_path, np.moveaxis(rgba, 2, 0))
    print(f"Wrote {output_path}")


def write_png(path: Path, rgba_bands: np.ndarray):
    """Write a (4,H,W) uint8 RGBA array as a PNG with transparency."""
    Image.fromarray(np.moveaxis(rgba_bands, 0, 2)).save(path, optimize=True)


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
    help=f"path to the 2024 lnd_type land/water raster; emits historic_{LAND_WATER_GROUP_LABEL}.png "
    "by comparing it against the 2015 state. Requires --historic-data and --baseline-land-water-data.",
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
    help="path to a directory of lnd_chng confidence rasters; emits one confidence_<year>.png per "
    ".tif in it. Omit to skip the confidence layers.",
)
@click.option("--output", help="output directory for the layer PNGs", required=True)
@click.option(
    "--majority",
    is_flag=True,
    help="downsample the categorical data layers with a majority (mode) vote instead of nearest "
    "neighbor: coherent rather than speckled, but drops change covering less than half a target "
    "pixel. Does not affect the confidence layers, which are continuous and always averaged",
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
    majority,
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

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    resampling = Resampling.mode if majority else Resampling.nearest

    if historic_data is not None:
        print(f"Reprojecting {historic_data.name} to target grid ({resampling.name})...")
        classes = reproject_classes_to_target(historic_data, resampling)
        write_historic_group_pngs(classes, output_dir)
        if land_water_data is not None:
            write_land_water_group_png(historic_data, baseline_land_water_data, land_water_data, output_dir, resampling)

    if confidence_data is not None:
        write_confidence_pngs(confidence_data, output_dir)

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
