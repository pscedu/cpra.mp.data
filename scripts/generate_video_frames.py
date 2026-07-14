#!/usr/bin/env python3
"""
Insert a georeferenced data raster (test_land_type_confidence_2019.tif) between the
"initial-conditions-water" and "levee" layers of a MapLibre style, and flatten the
whole stack (basemap-below, data, basemap-above) into a single georeferenced GeoTIFF
on a fixed pixel grid.

Pipeline:
  1. Split the MapLibre style into an "underlay" style (background .. initial-conditions-water)
     and an "overlay" style (levee .. label).
  2. Render each style with headless Chromium + maplibre-gl-js against the live
     basemap-vector/basemap-raster tile sources, and georeference the screenshots
     from the browser's own reported view bounds.
  3. Reproject the underlay render, the data GeoTIFF, and the overlay render onto the
     exact target grid (CRS/origin/pixel size/size given below) and alpha-composite
     them in order: underlay -> data -> overlay.
  4. Write the result as a 3-band (RGB) GeoTIFF.

Requires (installed in .venv): rasterio, numpy, Pillow, pyproj, playwright
(+ `playwright install chromium` once). The render page (maplibre-gl loaded from
a CDN) is embedded in this script and written out to render/render.html at runtime,
so no local maplibre-gl / node_modules install is needed.
"""

import http.server
import json
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

UNDERLAY_LAST_LAYER = "initial-conditions-water"
OVERLAY_FIRST_LAYER = "levee"

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
    dst = np.zeros((n, TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
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


def reproject_data_tif(data_path: Path, resampling):
    with rasterio.open(data_path) as ds:
        n = ds.count
        dst = np.zeros((n, TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
        for b in range(1, n + 1):
            reproject(
                source=rasterio.band(ds, b),
                destination=dst[b - 1],
                dst_transform=TARGET_TRANSFORM,
                dst_crs=TARGET_CRS,
                resampling=resampling,
                num_threads=4,
            )
    if n == 3:
        alpha = np.full((1, TARGET_HEIGHT, TARGET_WIDTH), 255, dtype=np.uint8)
        dst = np.concatenate([dst, alpha], axis=0)
    return dst


def alpha_over(fg_rgba, bg_rgb):
    """Composite fg (4,H,W uint8, RGBA) over bg (3,H,W uint8, RGB, assumed opaque)."""
    fg_rgb = fg_rgba[:3].astype(np.float32)
    fg_a = (fg_rgba[3].astype(np.float32) / 255.0)[None, :, :]
    out = fg_rgb * fg_a + bg_rgb.astype(np.float32) * (1.0 - fg_a)
    return np.clip(out, 0, 255).astype(np.uint8)


def write_geotiff(path: Path, rgb_bands: np.ndarray):
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 3,
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "crs": TARGET_CRS,
        "transform": TARGET_TRANSFORM,
        "photometric": "RGB",
        "interleave": "pixel",
        "compress": "deflate",
        "predictor": 2,
        "zlevel": 9,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(rgb_bands)
        dst.colorinterp = [
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
        ]


DATA_EXTENSIONS = {".tif", ".tiff"}


def resolve_data_output_pairs(data_arg: Path, output_arg: Path):
    """If data_arg is a directory, convert every raster in it and mirror the
    filenames into the output directory. Otherwise treat both as single files."""
    if data_arg.is_dir():
        output_arg.mkdir(parents=True, exist_ok=True)
        data_paths = sorted(p for p in data_arg.iterdir() if p.is_file() and p.suffix.lower() in DATA_EXTENSIONS)
        if not data_paths:
            raise FileNotFoundError(f"no {sorted(DATA_EXTENSIONS)} files found in {data_arg}")
        return [(p, output_arg / p.name) for p in data_paths]
    return [(data_arg, output_arg)]


@click.command(help=__doc__)
@click.option("--style", help="path to MapLibre style JSON", required=True)
@click.option("--data", help="path to a single GeoTIFF, or a directory of GeoTIFFs to convert", required=True)
@click.option("--output", help="output GeoTIFF path, or output directory when --data is a directory", required=True)
@click.option("--scale", type=float, default=2.0, show_default=True, help="device scale factor for supersampling")
@click.option("--keep-intermediate", is_flag=True)
def main(style, data, output, scale, keep_intermediate):
    data_output_pairs = resolve_data_output_pairs(Path(data), Path(output))

    from playwright.sync_api import sync_playwright

    style_path = Path(style)
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
    underlay_t = reproject_to_target(under_bands, under_crs, under_tf, Resampling.bilinear)
    print("Reprojecting overlay to target grid...")
    overlay_t = reproject_to_target(over_bands, over_crs, over_tf, Resampling.bilinear)
    base_rgb = underlay_t[:3]

    if keep_intermediate:
        write_geotiff(PROJECT_DIR / "_underlay.tif", base_rgb)
        write_geotiff(PROJECT_DIR / "_overlay.tif", overlay_t[:3])

    for data_path, output_path in data_output_pairs:
        print(f"Reprojecting data raster {data_path} to target grid...")
        data_t = reproject_data_tif(data_path, Resampling.average)

        with_data = alpha_over(data_t, base_rgb)
        final_rgb = alpha_over(overlay_t, with_data)

        if keep_intermediate:
            write_geotiff(PROJECT_DIR / f"_data_{data_path.stem}.tif", data_t[:3])

        write_geotiff(output_path, final_rgb)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
