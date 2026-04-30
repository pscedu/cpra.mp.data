from pathlib import Path

import polars as pl
from polars.exceptions import ColumnNotFoundError
from scipy.io import FortranFile

from cpra.mp.data.config import BASE_PATH
from cpra.mp.data.dimensions import DATASET_DIMENSIONS, DIMENSIONS, METADATA_DIMENSIONS
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.variables import VARIABLES


def partition_columns(df, include_dataset=True):
    """Get a list of partition columns based on the values in the data frame."""

    return [
        col.name
        for col in df.select(
            **{
                dim.name: dim.partition & dim.required
                for dim in DIMENSIONS.values()
                if (include_dataset or dim.name not in DATASET_DIMENSIONS)
            }
        )
        if col.any()
    ]


def required_columns(
    df,
    include_dataset=True,
    include_metadata=True,
    include_geography=False,
    include_value=False,
):
    """Get a list of required columns based on the values in the data frame."""

    return (
        [
            col.name
            for col in df.select(**{dim.name: dim.required for dim in DIMENSIONS.values()})
            if col.any()
            and (include_dataset or col.name not in DATASET_DIMENSIONS)
            and (include_metadata or col.name not in METADATA_DIMENSIONS)
        ]
        + (["geography"] if include_geography else [])
        + (["value"] if include_value else [])
    )


def validate_data_frame(df, required_cols=None):
    """Ensure that the provided data frame complies with the validation conditions."""

    if required_cols is None:
        required_cols = required_columns(df)

    # Raise an exception if required columns are missing.
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValidationError(f"Required columns are missing: {', '.join(missing_cols)}")

    expressions = {}

    # Validate required columns, and add dimension validation expressions.
    for col in required_cols:
        dim = DIMENSIONS[col]
        dim.validate_dtype(df[col])
        expressions.update(dim.validation_expressions)

    # Add validation expressions for variable-geographic_unit and variable-time_unit.
    expressions["variable_matches_grid"] = pl.any_horizontal(
        *[
            ((pl.col("variable") == n) & pl.col("grid").str.replace(r"_v[0-9.]+", "").is_in(v.geographic_units))
            for n, v in VARIABLES.items()
        ]
    )
    expressions["variable_matches_time_unit"] = pl.any_horizontal(
        *[((pl.col("variable") == n) & pl.col("time_unit").is_in(v.time_units)) for n, v in VARIABLES.items()]
    )

    results = df.select(**expressions).with_columns(valid=pl.all_horizontal(pl.col("*")))

    if not results["valid"].all():
        failed_row_count = results["valid"].len() - results["valid"].sum()
        failed_cols = [series.name for series in results if not series.all() and series.name != "valid"]
        msg = f"{failed_row_count} rows failed validation: {', '.join(failed_cols)}"
        raise ValidationError(msg)


def extract_dimension_value(column, data=None, **kwargs):
    """Extract a dimension value from keyword arguments or a data source."""

    if column in kwargs:
        return kwargs[column]

    if isinstance(data, dict) and column in data:
        return data[column]

    try:
        return data[column][0]
    except (KeyError, IndexError, TypeError, ValueError, ColumnNotFoundError):
        return None


def dataset_path(base_path, **kwargs):
    """Generate a dataset directory path from keyword arguments."""

    dataset_path = Path(base_path)
    for dim in DATASET_DIMENSIONS.values():
        dataset_path = dataset_path / f"{dim.name}={kwargs[dim.name]!s}"

    return dataset_path


def validate_grid(data=None, **kwargs):
    """Validate that there is a grid for the given geographic unit and plan version."""

    grid_name = extract_dimension_value("grid", data, **kwargs)

    if grid_name is None:
        msg = "Grid is required."
        raise ValidationError(msg)

    grid = GRIDS.get(grid_name, None)
    if grid is None:
        msg = f"{grid_name} is not a valid grid."
        raise ValidationError(msg)

    return grid


def cast_columns(df):
    """Cast all dimension columns in the given dataframe to the correct dtype."""

    return df.with_columns(
        **{dim.name: pl.col(dim.name).cast(dim.dtype) for dim in DIMENSIONS.values() if dim.name in df.columns}
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


def build_glob_pattern(required_cols, extension=None, **kwargs):
    """Build a glob pattern for selecting files from a dataset directory."""
    # The values of the keyword arguments can be lists or tuples that contain
    # multiple values, but for the glob pattern, we can only use single values.
    single_values = {
        k: v[0] if isinstance(v, (list, tuple)) else v
        for k, v in kwargs.items()
        if not isinstance(v, (list, tuple)) or len(v) == 1
    }
    dimension_patterns = [f"{col}={str(single_values[col]) if col in single_values else '*'}" for col in required_cols]
    return "/".join(dimension_patterns) + (f".{extension}" if extension else "")


def glob_lazy_frame(required_cols, base_path=BASE_PATH, extension=None, include_file_paths="path", **kwargs):
    """Create a LazyFrame with information about matched paths."""

    glob_pattern = build_glob_pattern(required_cols=required_cols, extension=extension, **kwargs)
    suffix = f"\\.{extension}" if extension else "$"

    return (
        pl.DataFrame({"path": [str(p) for p in base_path.glob(glob_pattern)]})
        .with_columns(path=pl.col("path").cast(pl.String))
        .select(
            [
                pl.col("path").str.extract(f"{dim.name}=([^/]+)(/|{suffix})", 1).cast(dim.dtype).alias(dim.name)
                for dim in DIMENSIONS.values()
                if dim.name in required_cols
            ]
            + ([pl.col("path").alias(include_file_paths)] if include_file_paths else [])
        )
        .sort(*required_cols)
        .lazy()
    )


def get_input_data_frame(**kwargs):
    """Create an input data frame from keyword arguments."""
    df = None
    for key, value in kwargs.items():
        if df is None:
            df = pl.DataFrame({key: value})
        else:
            df = df.join(pl.DataFrame({key: value}), how="cross")

    return df


def get_filter_expressions(**kwargs):
    """Generate a list of Polars filter expressions."""
    return [
        pl.col(k).is_in(v) if isinstance(v, (list, tuple)) else pl.col(k) == v
        for k, v in kwargs.items()
        if k not in DATASET_DIMENSIONS
    ]
