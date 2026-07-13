#!/usr/bin/env python3
"""Stack per-year elevation COGs into a single multi-band COG.

Bands are ordered chronologically (band 1 = earliest year) and named after
the source file's dimension value (e.g. "calendar_year=2019"). Output uses
INTERLEAVE=PIXEL so all year values for each tile pixel are adjacent in
memory, letting ZSTD exploit temporal correlation.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import click
import rasterio

DIMENSION_RE = re.compile(r"^(?P<key>[^=]+)=(?P<value>.+)$")


def path_dimension_tags(input_dir: Path) -> dict[str, str]:
    """Build dataset tags from key=value path components (e.g. variable=elev)."""

    tags = {}
    for part in input_dir.parts:
        match = DIMENSION_RE.match(part)
        if match:
            tags[match["key"]] = match["value"]
    return tags


def stack_annual_cogs(input_dir: Path, output_path: Path) -> None:
    tifs = sorted(input_dir.glob("calendar_year=*.tif"))
    if not tifs:
        raise click.ClickException(f"No calendar_year=*.tif files found in {input_dir}")

    click.echo(f"Stacking {len(tifs)} files: {tifs[0].name} → {tifs[-1].name}")

    with tempfile.TemporaryDirectory() as tmp:
        vrt_path = Path(tmp) / "stack.vrt"
        subprocess.run(
            ["gdalbuildvrt", "-separate", str(vrt_path)] + [str(t) for t in tifs],
            check=True,
        )

        with rasterio.open(vrt_path, "r+") as ds:
            ds.update_tags(**path_dimension_tags(input_dir))
            for band_idx, tif in enumerate(tifs, start=1):
                ds.set_band_description(band_idx, tif.stem)

        subprocess.run(
            [
                "gdal_translate",
                "-of",
                "COG",
                "-co",
                "BIGTIFF=YES",
                "-co",
                "COMPRESS=ZSTD",
                "-co",
                "LEVEL=6",
                "-co",
                "PREDICTOR=YES",
                "-co",
                "INTERLEAVE=PIXEL",
                "-co",
                "RESAMPLING=NEAREST",
                "-co",
                "OVERVIEW_RESAMPLING=NEAREST",
                "-co",
                "WARP_RESAMPLING=NEAREST",
                "-co",
                "NUM_THREADS=ALL_CPUS",
                str(vrt_path),
                str(output_path),
            ],
            check=True,
        )

    click.echo(f"Written: {output_path}")


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
def main(input_dir: Path, output: Path) -> None:
    """Stack per-year COGs in INPUT_DIR into a single multi-band COG at OUTPUT."""

    stack_annual_cogs(input_dir, output)


if __name__ == "__main__":
    main()
