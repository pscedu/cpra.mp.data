import polars as pl
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.grids import get_grid
from cpra.mp.data.variables import VARIABLES
from scipy.io import FortranFile


def required_columns(df):
    """Get a list of required columns based on the values in the data frame."""

    return [
        col.name
        for col in df.select(**{dim.column: dim.required for dim in DIMENSIONS})
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

    # Validate required columns.
    for dim in DIMENSIONS:
        if dim.column in required_cols:
            dim.validate_series(df.get_column(dim.column))


def extract_dimension_value(column, data, **kwargs):
    """Extract a dimension value from keyword arguments or a data source."""

    if column in kwargs:
        return kwargs[column]

    if isinstance(data, dict) and column in data:
        return data[column]

    if isinstance(data, pl.DataFrame) and column in data and len(data[column]):
        return data[column][0]

    return None


def validate_grid(data, **kwargs):
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
            dim.column: pl.col(dim.column).cast(dim.dtype)
            for dim in DIMENSIONS
            if dim.column in df.columns
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
