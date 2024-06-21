import numpy as np
import polars as pl
from cpra.mp.data.exceptions import SchemaValidationError
from cpra.mp.data.schema import SCHEMA
from cpra.mp.data.variables import VARIABLES
from scipy.io import FortranFile


def required_columns(df):
    """Get a list of required columns based on the values in the data frame."""

    return [
        col.name
        for col in df.select(**{level.column: level.required for level in SCHEMA})
        if col.any()
    ]


def validate_data_frame(df, required_cols=None):
    """Ensure that the provided data frame complies with the schema."""

    if required_cols is None:
        required_cols = required_columns(df)

    # Raise an exception if required columns are missing.
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise SchemaValidationError(
            f"Required columns are missing: {', '.join(missing_cols)}"
        )

    # Validate required columns.
    for level in SCHEMA:
        if level.column in required_cols:
            level.validate_series(df.get_column(level.column))


def cast_columns(df):
    """Cast all schema columns in the given dataframe to the correct dtype."""

    return df.with_columns(
        **{
            level.column: pl.col(level.column).cast(level.dtype)
            for level in SCHEMA
            if level.column in df.columns
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
