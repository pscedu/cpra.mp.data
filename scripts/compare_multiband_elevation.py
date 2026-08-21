"""Verify a migrated multiband elevation COG against its original single-band sources.

The migration converts elevation from Float32 meters to Int32 millimeters (see
``_elev_mm_vrt`` in migrate_to_long_format.py), so every band of a multiband COG written by
``compact_cog`` is compared against its pre-migration single-band source scaled by
``--scale-factor``.  Both sides are quantized to whole millimeters -- the multiband already
is, and the source is rounded -- so the comparison is exact rather than tolerance-based:
the conversion loses nothing above the millimeter the output is stored in.  The scaling is
done in float64 to match what GDAL does inside the VRT; float32 would put ties like
2.0005 m on the wrong side of the rounding boundary and report spurious mismatches.

Nodata pixels are compared as nodata rather than scaled, since the VRT's ``<NODATA>``
excludes them from the multiply and leaves the sentinel intact.

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


def _round_half_away(values):
    """Round to whole numbers the way GDAL does on a float-to-integer write.

    ``np.round`` breaks ties to even, GDAL breaks them away from zero, and exact ties do
    occur here: any source value that is a small negative power of two scales to one
    (0.0625 m -> 62.5 mm), which numpy would call 62 and GDAL calls 63.  Using numpy's
    rule would report those as migration failures when the migration is correct.
    """

    return np.copysign(np.floor(np.abs(values) + 0.5), values)


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
@click.option("--scale-factor", default=1000, show_default=True, help="Storage units per source unit (meters to millimeters).")
@click.option("--max-examples", default=5, show_default=True, help="Mismatching pixels to print per band.")
def compare(multiband_path, source_dir, scale_factor, max_examples):
    """Compare every band of MULTIBAND_PATH to its single-band source file in SOURCE_DIR."""

    total_mismatches = 0
    with rio.open(multiband_path) as multiband_ds:
        multiband_profile = multiband_ds.profile
        band_count = multiband_ds.count
        nodata = multiband_profile["nodata"]
        for band_idx, description in enumerate(multiband_ds.descriptions, start=1):
            source_path = _source_path(source_dir, description)
            with rio.open(source_path) as source_ds:
                _check_alignment(multiband_profile, source_ds.profile, source_path)
                source = source_ds.read(1)

            multiband = multiband_ds.read(band_idx)
            # float64 both to match GDAL's arithmetic and to hold the sentinel and the
            # scaled values in one array without an integer overflow to worry about.
            is_nodata = source == nodata
            expected = np.where(is_nodata, nodata, _round_half_away(source.astype("float64") * scale_factor))

            mismatch = expected != multiband
            count = int(np.count_nonzero(mismatch))
            total_mismatches += count

            if count:
                rows, cols = np.nonzero(mismatch)
                max_abs_diff = float(np.max(np.abs(expected[mismatch] - multiband[mismatch])))
                click.echo(
                    f"MISMATCH {description}: {count} pixel(s) differ "
                    f"(max abs diff {max_abs_diff:.0f})"
                )
                for row, col in zip(rows[:max_examples], cols[:max_examples]):
                    click.echo(
                        f"    ({row}, {col}): source={source[row, col]!r} "
                        f"expected={expected[row, col]!r} multiband={multiband[row, col]!r}"
                    )
            else:
                click.echo(f"OK       {description}: {source.size} pixels match")

    if total_mismatches:
        click.echo(f"\nFAILED: {total_mismatches} mismatching pixel(s) across {band_count} bands")
        sys.exit(1)
    click.echo(f"\nPASSED: all {band_count} band(s) match exactly after scaling by {scale_factor}")


if __name__ == "__main__":
    compare()
