import polars as pl
import polars.selectors as cs
from cpra.mp.data.config import BASE_PATH
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.grids import get_grid
from cpra.mp.data.utils import required_columns, validate_data_frame
from cpra.mp.data.variables import VARIABLES
from pathlib import Path


def read_parquet(base_path=BASE_PATH, lazy=True, filters=None, cache=True, **kwargs):
    """Read tabular data into a data frame from a Parquet file."""

    filter_kwargs = kwargs.copy()
    dataset_path = Path(base_path)
    dataset_cols = []
    dataset_exprs = []

    # Iterate over dataset dimensions and update the dataset path.
    for dimension in DIMENSIONS:
        if dimension.partition is not True:
            break

        dimension_value = kwargs.get(dimension.column)
        if dimension_value is None:
            msg = f"Required dataset dimension {dimension.column} is missing"
            raise ValueError(msg)

        dataset_path = dataset_path / f"{dimension.column}={str(dimension_value)}"
        dataset_cols.append(dimension.column)
        dataset_exprs.append(
            pl.lit(dimension_value).cast(dimension.dtype).alias(dimension.column)
        )
        del filter_kwargs[dimension.column]

    # TODO: Provide Hive partitioning schema since this is known in advance.
    lf = pl.scan_parquet(dataset_path, cache=cache)

    # Filter based on keyword arguments.
    if filter_kwargs:
        lf = lf.filter(**filter_kwargs)

    # TODO: Handle advanced filters.
    if filters is not None:
        msg = "Advanced filters are not yet implemented"
        raise NotImplementedError(msg)

    # Sort columns and add dataset values.
    required_cols = required_columns(pl.DataFrame(kwargs))
    lf = lf.select(
        *dataset_exprs,
        *[pl.col(col) for col in required_cols if col not in dataset_cols],
        cs.digit(),
    )

    return lf if lazy else lf.collect()


def write_parquet(
    data,
    base_path=BASE_PATH,
    validate=True,
    sort=True,
    compression="zstd",
    compression_level=6,
    row_group_size=5000,
    use_dictionary=False,
    **kwargs,
):
    """Write data from a data frame to a Parquet file."""

    df = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)

    # Add keyword arguments to the dataframe as columns.
    df = df.with_columns(**{k: pl.lit(v) for k, v in kwargs.items()})

    # If this is long-format data (i.e., with a geography column), unmelt it.
    if "geography" in df and "value" in df:
        df = df.pivot("geography", values="value")

    # Validate and sort the data.
    if validate or sort:
        # Check to make sure that all required columns are present.
        required_cols = required_columns(df)

        if validate:
            validate_data_frame(df, required_cols=required_cols)

        # Validate geography column names.
        grid = get_grid(df["geographic_unit"][0], df["plan_version"][0])
        geography_cols = [str(geo) for geo in sorted(grid.geographies)]
        all_cols = required_cols + geography_cols

        invalid_cols = set(all_cols) - set(df.columns)
        if invalid_cols:
            msg = f"Invalid columns: {', '.join(list(invalid_cols))}"
            raise ValidationError(msg)

        missing_cols = set(df.columns) - set(all_cols)
        if missing_cols:
            msg = f"Missing columns: {', '.join(list(missing_cols))}"
            raise ValidationError(msg)

        # Validate geometry column data types.
        variable = VARIABLES[df["variable"][0]]
        for col in geography_cols:
            if df[col].dtype != variable.dtype:
                msg = f"Series {col} has dtype {df[col].dtype!s}, expected {variable.dtype!s}"
                raise ValidationError(msg)

        # Sort columns and rows.
        if sort:
            df = df.select(*all_cols).sort(*required_cols)

    # Get partition columns.
    partition_cols = [
        col.name
        for col in df.select(
            **{dim.column: dim.partition & dim.required for dim in DIMENSIONS}
        )
        if col.any()
    ]

    # Write the partitioned file.
    df.write_parquet(
        base_path,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        use_pyarrow=True,
        pyarrow_options=dict(
            use_dictionary=use_dictionary, partition_cols=partition_cols
        ),
    )
