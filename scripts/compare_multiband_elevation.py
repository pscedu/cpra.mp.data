"""Verify a migrated multiband elevation COG against its original single-band sources.

Elevation uses LERC_ZSTD compression (see ``Variable.max_z_error`` in variables.py), which
is lossy: this confirms that loss never exceeds three-decimal precision by comparing every
band of a multiband COG (written by ``compact_cog``) to its pre-migration single-band
source file, both rounded to the same number of decimal places.

Each source file is located by exact filename match against the multiband band's
description (e.g. band description ``"calendar_year=2032"`` -> ``calendar_year=2032.tif``
in ``source_dir``), the naming convention used by the pre-migration single-band tree.

Usage:

\b
    python scripts/compare_multiband_elevation.py \\
        data-long/variable=elev/time_unit=annual/grid=morph_pixel_v001/model_group_id=520/scenario_id=7/c98c362e-....tif \\
        data/variable=elev/grid=morph_pixel_v001/time_unit=annual/model_group_id=520/scenario_id=7
"""

import sys
from pathlib import Path

import click
import numpy as np
import rasterio as rio

# Add the project root to sys.path so the package is importable when the script is run
# from any working directory.
sys.path.insert(0, str(Path(__file__).parent.parent))


def _source_path(source_dir, description):
    """Locate the single-band source file matching a multiband band's description."""

    path = source_dir / f"{description}.tif"
    if not path.exists():
        msg = f"no source file for band description {description!r} (expected {path})"
        raise FileNotFoundError(msg)
    return path


def _check_alignment(multiband_profile, source_profile, source_path):
    mismatched = [
        key
        for key in ("width", "height", "crs", "transform", "nodata")
        if multiband_profile[key] != source_profile[key]
    ]
    if mismatched:
        details = ", ".join(f"{key}: {multiband_profile[key]!r} != {source_profile[key]!r}" for key in mismatched)
        msg = f"{source_path} is not aligned with the multiband COG ({details})"
        raise ValueError(msg)


@click.command()
@click.argument("multiband_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--decimals", default=3, show_default=True, help="Decimal places to round both rasters to before comparing.")
@click.option("--max-examples", default=5, show_default=True, help="Mismatching pixels to print per band.")
def compare(multiband_path, source_dir, decimals, max_examples):
    """Compare every band of MULTIBAND_PATH to its single-band source file in SOURCE_DIR."""

    total_mismatches = 0
    with rio.open(multiband_path) as multiband_ds:
        multiband_profile = multiband_ds.profile
        band_count = multiband_ds.count
        for band_idx, description in enumerate(multiband_ds.descriptions, start=1):
            source_path = _source_path(source_dir, description)
            with rio.open(source_path) as source_ds:
                _check_alignment(multiband_profile, source_ds.profile, source_path)
                source = source_ds.read(1)

            multiband = multiband_ds.read(band_idx)
            source_rounded = np.round(source, decimals)
            multiband_rounded = np.round(multiband, decimals)

            mismatch = source_rounded != multiband_rounded
            count = int(np.count_nonzero(mismatch))
            total_mismatches += count

            if count:
                rows, cols = np.nonzero(mismatch)
                max_abs_diff = float(np.max(np.abs(source[mismatch] - multiband[mismatch])))
                click.echo(
                    f"MISMATCH {description}: {count} pixel(s) differ after rounding "
                    f"(max raw abs diff {max_abs_diff:.6f})"
                )
                for row, col in zip(rows[:max_examples], cols[:max_examples]):
                    click.echo(
                        f"    ({row}, {col}): source={source[row, col]!r} multiband={multiband[row, col]!r}"
                    )
            else:
                click.echo(f"OK       {description}: {source.size} pixels match")

    if total_mismatches:
        click.echo(f"\nFAILED: {total_mismatches} mismatching pixel(s) across {band_count} bands")
        sys.exit(1)
    click.echo(f"\nPASSED: all bands match at {decimals} decimal place(s)")


if __name__ == "__main__":
    compare()
