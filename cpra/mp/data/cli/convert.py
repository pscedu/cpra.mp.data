import click
import polars as pl
import re
from cpra.mp.data import write_data
from cpra.mp.data.cli import cli
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.variables import VARIABLES

DIMENSION_MAP = {
    "CalendarDay": "calendar_day",
    "ModelGroup": "model_group_id",
    "Scenario": "scenario_id",
}
GEOGRAPHY_MAP = {
    "HydroCompID": "hydro_compartment",
}
TIME_UNIT_MAP = {
    "CalendarDay": "daily",
}
VARIABLE_MAP = {
    "MeanWaterLevel_m": "stg",
    "Salinity_ppt": "sal",
    "Temperature_C": "tmp",
    "TidalRange_m": "trg",
    "TotalSuspendedSolids_mgperL": "tss",
}


def get_columns(lf_columns, map_dict, obj_dict=None):
    """Map LazyFrame columns to their corresponding names or objects."""
    return [
        (k, obj_dict[v] if obj_dict else v)
        for k, v in map_dict.items()
        if k in lf_columns
    ]


def convert_parquet(dump_path, plan_version):
    """
    Write variables to the new Parquet data format from a Parquet dump of a PDD table.
    """

    lf = pl.scan_parquet(dump_path)
    lf_columns = lf.collect_schema().names()

    variables = get_columns(lf_columns, VARIABLE_MAP, VARIABLES)
    dimensions = get_columns(
        lf_columns, DIMENSION_MAP, {dim.name: dim for dim in DIMENSIONS.values()}
    )
    geographies = get_columns(lf_columns, GEOGRAPHY_MAP)
    time_units = get_columns(lf_columns, TIME_UNIT_MAP)

    if len(geographies) != 1:
        msg = f"Expected exactly one geography column, found {', '.join(geographies)}"
        raise ValueError(msg)

    if len(time_units) != 1:
        msg = f"Expected exactly one time unit column, found {', '.join(time_units)}"
        raise ValueError(msg)

    for value_column, variable in variables:
        print(f"Reading {variable.label} from {dump_path}...")
        df = lf.select(
            **{dim.name: col_name for col_name, dim in dimensions},
            geography=pl.col(geographies[0][0]),
            value=pl.col(value_column).cast(variable.dtype),
        ).collect()

        print(f"Writing {variable.label}...")
        write_data(
            df,
            plan_version=plan_version,
            variable=variable.name,
            geographic_unit=geographies[0][1],
            time_unit=time_units[0][1],
        )


def validate_plan_version(ctx, param, value):
    """Validate the provided plan version."""
    if re.match(r"^\d{4}\.\d+\.\d+$", value) is None:
        raise click.BadParameter("format must be YYYY.minor.patch")
    return value


def validate_filename(ctx, param, value):
    """Validate a filename for conversion."""
    if not (value.endswith(".parquet") or value.endswith(".xyz.b")):
        raise click.BadParameter("filename must end with .parquet or .xyz.b")
    return value


@cli.command()
@click.option(
    "--plan-version",
    default="2023.0.0",
    help="Plan version",
    callback=validate_plan_version,
)
@click.argument(
    "filename", type=click.Path(exists=True), nargs=1, callback=validate_filename
)
def convert(plan_version, filename):
    """Convert MP2023 data to the MP2029 data format."""

    if filename.endswith(".parquet"):
        convert_parquet(filename, plan_version)
    else:
        msg = "Raster conversion is not yet implemented."
        raise NotImplementedError(msg)
