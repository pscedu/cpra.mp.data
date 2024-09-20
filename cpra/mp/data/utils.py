import polars as pl
from cpra.mp.data.dimensions import DATASET_DIMENSIONS, DIMENSIONS
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.grids import get_grid
from cpra.mp.data.variables import VARIABLES
from pathlib import Path
from polars.exceptions import ColumnNotFoundError
from scipy.io import FortranFile


def partition_columns(df, include_dataset=True):
    """Get a list of partition columns based on the values in the data frame."""

    return [
        col.name
        for col in df.select(
            **{
                dim.name: dim.partition & dim.required
                for dim in DIMENSIONS.values()
                if include_dataset or dim.partition is not True
            }
        )
        if col.any()
    ]


def required_columns(df):
    """Get a list of required columns based on the values in the data frame."""

    return [
        col.name
        for col in df.select(**{dim.name: dim.required for dim in DIMENSIONS.values()})
        if col.any()
    ]


def validate_data_frame(df, required_cols=None):
    """Ensure that the provided data frame complies with the validation conditions."""

    if required_cols is None:
        required_cols = required_columns(df)

    # Raise an exception if required columns are missing.
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValidationError(
            f"Required columns are missing: {', '.join(missing_cols)}"
        )

    expressions = {}

    # Validate required columns, and add dimension validation expressions.
    for col in required_cols:
        dim = DIMENSIONS[col]
        dim.validate_dtype(df[col])
        expressions.update(dim.validation_expressions)

    # Add validation expressions for variable-geographic_unit and variable-time_unit.
    expressions["variable_matches_geographic_unit"] = pl.any_horizontal(
        *[
            (
                (pl.col("variable") == n)
                & pl.col("geographic_unit").is_in(v.geographic_units)
            )
            for n, v in VARIABLES.items()
        ]
    )
    expressions["variable_matches_time_unit"] = pl.any_horizontal(
        *[
            ((pl.col("variable") == n) & pl.col("time_unit").is_in(v.time_units))
            for n, v in VARIABLES.items()
        ]
    )

    results = df.select(**expressions).with_columns(
        valid=pl.all_horizontal(pl.col("*"))
    )

    if not results["valid"].all():
        failed_row_count = results["valid"].len() - results["valid"].sum()
        failed_cols = [
            series.name
            for series in results
            if not series.all() and not series.name == "valid"
        ]
        msg = f"{failed_row_count} rows failed validation: {", ".join(failed_cols)}"
        raise ValidationError(msg)


def extract_dimension_value(column, data=None, **kwargs):
    """Extract a dimension value from keyword arguments or a data source."""

    if column in kwargs:
        return kwargs[column]

    if isinstance(data, dict) and column in data:
        return data[column]

    try:
        return data[column][0]
    except (KeyError, IndexError, ValueError, ColumnNotFoundError):
        return None


def dataset_path(base_path, **kwargs):
    """Generate a dataset directory path from keyword arguments."""

    dataset_path = Path(base_path)
    for dim in DATASET_DIMENSIONS.values():
        dataset_path = dataset_path / f"{dim.name}={str(kwargs[dim.name])}"

    return dataset_path


def validate_grid(data=None, **kwargs):
    """Validate that there is a grid for the given geographic unit and plan version."""

    geographic_unit = extract_dimension_value("geographic_unit", data, **kwargs)
    plan_version = extract_dimension_value("plan_version", data, **kwargs)

    if geographic_unit is None:
        msg = "Geographic unit is required."
        raise ValidationError(msg)

    if plan_version is None:
        msg = "Plan version is required."
        raise ValidationError(msg)

    grid = get_grid(geographic_unit, plan_version)
    if grid is None:
        msg = f"No grid for geographic unit {geographic_unit} and plan version {plan_version}."
        raise ValidationError(msg)

    return grid


def cast_columns(df):
    """Cast all dimension columns in the given dataframe to the correct dtype."""

    return df.with_columns(
        **{
            dim.name: pl.col(dim.name).cast(dim.dtype)
            for dim in DIMENSIONS.values()
            if dim.name in df.columns
        }
    )


def read_fortran_array(filename, dtype=None, variable=None):
    """Read a Fortran binary array file."""

    if dtype is None:
        if variable is None:
            msg = "Either variable or dtype must be provided"
            raise ValueError(msg)
        spec = VARIABLES.get(variable)

        if spec is None:
            msg = f"Invalid variable: {variable}"
            raise ValueError(msg)

        # Convert Polars dtype to Numpy dtype.
        dtype = pl.Series(dtype=spec.dtype).to_numpy().dtype

    with FortranFile(filename, "r") as array_file:
        return array_file.read_record(dtype)
