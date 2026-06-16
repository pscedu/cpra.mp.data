from pathlib import Path

import click
import numpy as np
import rasterio
from rasterio.windows import Window

NODATA = -9999
TILE_SIZE = 512

# Gradient breakpoints: (value, (R, G, B))
# 0.0 → #6495ed (water), 0.5 → #ffffbf (mixed), 1.0 → #008000 (land)
STOPS = np.array([0.0, 0.5, 1.0])
COLORS = np.array([
    [100, 149, 237],  # #6495ed
    [255, 255, 191],  # #ffffbf
    [  0, 128,   0],  # #008000
], dtype=np.float32)


def colorize(arr):
    """Map float32 confidence values to (4, H, W) RGBA uint8."""
    valid = arr != NODATA
    r = np.interp(arr, STOPS, COLORS[:, 0]).astype(np.uint8)
    g = np.interp(arr, STOPS, COLORS[:, 1]).astype(np.uint8)
    b = np.interp(arr, STOPS, COLORS[:, 2]).astype(np.uint8)
    a = np.where(valid, 255, 0).astype(np.uint8)
    return np.stack([r, g, b, a])


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
        print(f"Colorizing {in_path.name}...")

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
