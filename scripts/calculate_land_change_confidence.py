"""Calculate land/water confidence and land change confidence from lnd_type ensembles.

For each year, the five environmental scenarios are collapsed into a single confidence
value: lnd_type is reduced to land (1) / water (0), and the scenarios are combined as a
weighted mean using the CPRA scenario probabilities. The result is written as `lnd_conf`
under scenario_id=0 -- scenario is the dimension being collapsed, so the output is no
longer attributable to any one of them.

`lnd_conf_chng` is that confidence measured against the first year's: a plain difference
between a year's confidence and the base year's, with no offset applied to either side. It
runs [-1, 1] and its sign is the direction of change -- negative where the pixel has moved
toward water since the base year, positive where it has moved toward land, and 0 where its
confidence has not moved at all. Magnitude is how far it has moved, so a pixel that has
gone from confidently land to confidently water reads -1 whatever its absolute confidence
was.
"""

import click
import numpy as np
import polars as pl
import rasterio

from cpra.mp.data import read_data, write_data
from cpra.mp.data.grids import GRIDS

# Maps lnd_type values to land/water. 1/3/4 = land-like, 2/5 = water-like. Anything else
# -- including the -9999 nodata -- is unmapped and drops out of the weighted mean.
VALUE_MAP = {
    1: 1,
    2: 0,
    3: 1,
    4: 1,
    5: 0,
}

# CPRA environmental scenario probabilities. These sum to 1 across the full set; where a
# year is missing a scenario the weights present are renormalized, so the output stays a
# weighted mean of what exists rather than decaying toward 0 as scenarios drop out.
SCENARIO_WEIGHTS = {
    34: 0.33,
    35: 0.085,
    36: 0.085,
    37: 0.25,
    38: 0.25,
}

SOURCE_VARIABLE = "lnd_type"
CONFIDENCE_VARIABLE = "lnd_conf"
CHANGE_VARIABLE = "lnd_conf_chng"

GRID = "morph_pixel_v002"
TIME_UNIT = "annual"
MODEL = "icm_v26.0.0"

# The scenario dimension is collapsed by the weighted mean, so the output carries the
# reserved "no single scenario" id rather than one of the inputs'.
OUTPUT_SCENARIO_ID = 0

NODATA = -9999


def to_land_water(arr):
    """Map lnd_type to 0.0/1.0, using NaN for unmapped and nodata values."""
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    for in_val, out_val in VALUE_MAP.items():
        out[arr == in_val] = float(out_val)
    return out


def confidence_array(scenario_paths, grid):
    """Weighted-mean land/water confidence across scenarios, NaN where no scenario is valid.

    `scenario_paths` maps scenario_id to a raster path. The weight denominator is
    accumulated per pixel rather than taken as a constant so that a pixel masked out in
    one scenario is averaged over the remaining ones, the same way a year missing a whole
    scenario is.
    """
    total = np.zeros((grid.height, grid.width), dtype=np.float32)
    weight = np.zeros((grid.height, grid.width), dtype=np.float32)

    for scenario_id, path in sorted(scenario_paths.items()):
        with rasterio.open(path) as src:
            mapped = to_land_water(src.read(1))

        valid = ~np.isnan(mapped)
        scenario_weight = np.float32(SCENARIO_WEIGHTS[scenario_id])
        total += np.where(valid, mapped, 0.0) * scenario_weight
        weight += valid * scenario_weight

    conf = np.full(total.shape, np.nan, dtype=np.float32)
    np.divide(total, weight, out=conf, where=weight > 0)
    return conf


def to_raster(arr):
    """Fill NaN with the nodata value and cast for writing."""
    return np.where(np.isnan(arr), NODATA, arr).astype(np.float32)


def scenario_paths_by_year(model_group_id, base_path):
    """Map each calendar year to its {scenario_id: path}, sorted by year."""
    df = (
        read_data(
            variable=SOURCE_VARIABLE,
            grid=GRID,
            time_unit=TIME_UNIT,
            model_group_id=model_group_id,
            base_path=base_path,
        )
        .collect()
        .filter(pl.col("scenario_id").is_in(list(SCENARIO_WEIGHTS)))
        .group_by("calendar_year")
        .agg(pl.col("scenario_id"), pl.col("path"))
        .sort("calendar_year")
    )

    return {
        row["calendar_year"]: dict(zip(row["scenario_id"], row["path"]))
        for row in df.iter_rows(named=True)
    }


@click.command()
@click.option("--model-group-id", type=int, default=705, show_default=True)
@click.option(
    "--base-path",
    type=click.Path(file_okay=False),
    default=None,
    help="Data directory to read and write. Defaults to the configured base path.",
)
def main(model_group_id, base_path):
    grid = GRIDS[GRID]

    by_year = scenario_paths_by_year(model_group_id, base_path)
    if not by_year:
        raise click.ClickException(
            f"No {SOURCE_VARIABLE} rasters found for model_group_id={model_group_id} on {GRID}."
        )

    years = sorted(by_year)
    partial = {year: sorted(by_year[year]) for year in years if len(by_year[year]) < len(SCENARIO_WEIGHTS)}
    if partial:
        click.echo(
            f"{len(partial)} of {len(years)} years are missing scenarios and will be averaged "
            f"over the scenarios present: "
            + ", ".join(f"{year} ({', '.join(str(s) for s in scenarios)})" for year, scenarios in partial.items())
        )

    # The baseline is the base year's confidence as-is. A pixel with no valid baseline is
    # NaN here and stays NaN through the subtraction, so it propagates as nodata into the
    # change raster for every year.
    base_year = years[0]
    click.echo(f"Computing baseline from {base_year} ({len(by_year[base_year])} scenarios)...")
    baseline = confidence_array(by_year[base_year], grid)

    output_dimensions = {
        "grid": GRID,
        "time_unit": TIME_UNIT,
        "model_group_id": model_group_id,
        "scenario_id": OUTPUT_SCENARIO_ID,
        "model": MODEL,
        "base_path": base_path,
    }

    for year in years:
        click.echo(f"Processing {year} ({len(by_year[year])} scenarios)...")
        conf = confidence_array(by_year[year], grid)

        write_data(
            to_raster(conf),
            variable=CONFIDENCE_VARIABLE,
            calendar_year=year,
            **output_dimensions,
        )
        write_data(
            to_raster(conf - baseline),
            variable=CHANGE_VARIABLE,
            calendar_year=year,
            **output_dimensions,
        )


if __name__ == "__main__":
    main()
