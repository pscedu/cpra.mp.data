import re
from pathlib import Path

import click
import numpy as np
import rasterio
from rasterio.windows import Window

NODATA = -9999
TILE_SIZE = 512
BASE_YEAR = 2019

YEAR_RE = re.compile(r"(\d{4})\.tif$")

# Historic (<= 2019): discrete -1/0/1 diff values.
HISTORIC_NEGATIVE = (15, 101, 136)  # #0F6588
HISTORIC_POSITIVE = (47, 90, 30)  # #2F5A1E

# Future (> 2019): binned diff values.
FUTURE_BINS = [
    (68, 179, 225),  # #44B3E1, -1 to < -0.67
    (131, 204, 235),  # #83CCEB, -0.67 to < -0.33
    (192, 230, 245),  # #C0E6F5, -0.33 to < 0
    (218, 242, 208),  # #DAF2D0, > 0 to < 0.33
    (142, 217, 115),  # #8ED973, 0.33 to < 0.67
    (60, 125, 34),  # #3C7D22, 0.67 to 1
]


def colorize_historic(arr):
    """Map float32 diff values in {-1, 0, 1} to (4, H, W) RGBA uint8."""
    valid = arr != NODATA
    r = np.zeros(arr.shape, dtype=np.uint8)
    g = np.zeros(arr.shape, dtype=np.uint8)
    b = np.zeros(arr.shape, dtype=np.uint8)
    a = np.where(valid, 255, 0).astype(np.uint8)

    negative = valid & (arr < 0)
    positive = valid & (arr > 0)
    zero = valid & (arr == 0)

    r[negative], g[negative], b[negative] = HISTORIC_NEGATIVE
    r[positive], g[positive], b[positive] = HISTORIC_POSITIVE
    a[zero] = 0

    return np.stack([r, g, b, a])


def colorize_future(arr):
    """Map float32 diff values in [-1, 1] to (4, H, W) RGBA uint8 using binned colors."""
    valid = arr != NODATA
    r = np.zeros(arr.shape, dtype=np.uint8)
    g = np.zeros(arr.shape, dtype=np.uint8)
    b = np.zeros(arr.shape, dtype=np.uint8)
    a = np.where(valid, 255, 0).astype(np.uint8)

    masks = [
        valid & (arr < -0.67),
        valid & (arr >= -0.67) & (arr < -0.33),
        valid & (arr >= -0.33) & (arr < 0),
        valid & (arr > 0) & (arr < 0.33),
        valid & (arr >= 0.33) & (arr < 0.67),
        valid & (arr >= 0.67),
    ]
    for mask, (cr, cg, cb) in zip(masks, FUTURE_BINS):
        r[mask] = cr
        g[mask] = cg
        b[mask] = cb

    zero = valid & (arr == 0)
    a[zero] = 0

    return np.stack([r, g, b, a])


def year_from_path(path):
    match = YEAR_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not determine year from filename: {path.name}")
    return int(match.group(1))


@click.command()
@click.argument("input_dir", type=click.Path(file_okay=False, exists=True))
@click.argument("output_dir", type=click.Path(file_okay=False))
def main(input_dir, output_dir):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for in_path in sorted(in_dir.glob("*.tif")):
        out_path = out_dir / in_path.name
        if out_path.exists():
            print(f"Skipping {in_path.name}, already exists.")
            continue

        year = year_from_path(in_path)
        colorize = colorize_historic if year <= BASE_YEAR else colorize_future
        print(f"Colorizing {in_path.name} ({'historic' if year <= BASE_YEAR else 'future'})...")

        with rasterio.open(in_path) as src:
            profile = src.profile.copy()
            profile.update(dtype="uint8", count=4, nodata=None, compress="zstd")

            with rasterio.open(out_path, "w", **profile) as dst:
                for row_off in range(0, src.height, TILE_SIZE):
                    tile_h = min(TILE_SIZE, src.height - row_off)
                    for col_off in range(0, src.width, TILE_SIZE):
                        tile_w = min(TILE_SIZE, src.width - col_off)
                        window = Window(col_off, row_off, tile_w, tile_h)
                        tile = src.read(1, window=window)
                        dst.write(colorize(tile), window=window)


if __name__ == "__main__":
    main()
