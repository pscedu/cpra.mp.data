import re
from pathlib import Path

import polars as pl
import polars.selectors as cs

from cpra.mp.data.config import BASE_PATH
from cpra.mp.data.dimensions import DATASET_DIMENSIONS, DIMENSIONS
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.utils import (
    build_glob_pattern,
    dataset_path,
    get_filter_expressions,
    get_input_data_frame,
    partition_columns,
    required_columns,
    validate_data_frame,
    validate_grid,
)
from cpra.mp.data.variables import VARIABLES

ALPHA_COLUMN = re.compile(r"^[a-z]")
NUMERIC_COLUMN = re.compile(r"^[0-9]")


def read_parquet(base_path=BASE_PATH, cache=True, include_file_paths=None, **kwargs):
    """Read tabular data into a data frame from a Parquet file."""

    # Validate the input keyword arguments.
    input_df = get_input_data_frame(**kwargs)
    validate_data_frame(input_df, required_cols=DATASET_DIMENSIONS.keys())

    # Calculate the dataset directory path and expressions.
    dataset_dir = dataset_path(base_path, **kwargs)
    dataset_exprs = [pl.lit(kwargs[dim.name]).cast(dim.dtype).alias(dim.name) for dim in DATASET_DIMENSIONS.values()]
    filter_args = get_filter_expressions(**kwargs)

    # Default to an un-partitioned dataset.
    hive_partitioning = False
    hive_schema = None

    # Provide Hive partitioning schema since this is known in advance.
    partition_cols = partition_columns(input_df, include_dataset=False)
    if partition_cols:
        hive_partitioning = True
        hive_schema = {dim.name: dim.dtype for dim in DIMENSIONS.values() if dim.name in partition_cols}

    lf = pl.scan_parquet(
        dataset_dir,
        cache=cache,
        hive_partitioning=hive_partitioning,
        hive_schema=hive_schema,
        include_file_paths=include_file_paths,
    )

    # Filter based on keyword arguments.
    if filter_args:
        lf = lf.filter(*filter_args)

    # Sort columns and add dataset values.
    required_cols = required_columns(input_df)
    lf = lf.select(
        *dataset_exprs,
        *[pl.col(col) for col in required_cols if col not in DATASET_DIMENSIONS],
        cs.digit(),
        *([pl.col(include_file_paths)] if include_file_paths else []),
    )

    return lf


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

    # If this is long-format data (i.e., with a geography column), pivot it.
    if "geography" in df and "value" in df:
        df = df.pivot("geography", values="value")

    # Validate and sort the data.
    if validate or sort:
        # Check to make sure that all required columns are present.
        required_cols = required_columns(df)

        if validate:
            validate_data_frame(df, required_cols=required_cols)

        # Validate column names.
        grid = validate_grid(df, **kwargs)

        alpha_cols = [col for col in df.columns if ALPHA_COLUMN.match(col)]
        numeric_cols = [col for col in df.columns if NUMERIC_COLUMN.match(col)]

        invalid_cols = set(alpha_cols) - set(required_cols)
        if invalid_cols:
            msg = f"Invalid columns: {', '.join(list(invalid_cols))}"
            raise ValidationError(msg)

        missing_cols = set(required_cols) - set(alpha_cols)
        if missing_cols:
            msg = f"Missing columns: {', '.join(list(missing_cols))}"
            raise ValidationError(msg)

        if grid.id_count is not None and len(numeric_cols) != grid.id_count:
            msg = "Wrong number of geography columns: " f"expected {grid.id_count}, found {len(numeric_cols)}"
            raise ValidationError(msg)

        # Validate geometry column data type and min/max ID.
        variable = VARIABLES[df["variable"][0]]
        for col in numeric_cols:
            if (grid.id_min is not None and int(col) < grid.id_min) or (
                grid.id_max is not None and int(col) > grid.id_max
            ):
                msg = f"Invalid geography ID for grid {grid.name}: {col}"
                raise ValidationError(msg)
            if df[col].dtype != variable.dtype:
                msg = f"Series {col} has dtype {df[col].dtype!s}, expected {variable.dtype!s}"
                raise ValidationError(msg)

        # Sort columns and rows.
        if sort:
            all_cols = required_cols + list(map(str, sorted(map(int, numeric_cols))))
            df = df.select(*all_cols).sort(*required_cols)

    # Get partition columns.
    partition_cols = partition_columns(df)

    # Write the partitioned file.
    df.write_parquet(
        base_path,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        use_pyarrow=True,
        pyarrow_options=dict(use_dictionary=use_dictionary, partition_cols=partition_cols),
    )


def compact_parquet(
    base_path=BASE_PATH,
    sort=True,
    compression="zstd",
    compression_level=6,
    row_group_size=5000,
    use_dictionary=False,
    **kwargs,
):
    """Rewrite Parquet datasets with one file per dataset."""

    # Validate the input keyword arguments.
    input_df = pl.DataFrame(kwargs)
    validate_data_frame(input_df, required_cols=DATASET_DIMENSIONS)

    # Calculate the dataset directory path.
    dataset_dir = dataset_path(base_path, **kwargs)

    # Construct the glob pattern.
    required_cols = required_columns(input_df, include_dataset=False, include_metadata=False)
    glob_pattern = build_glob_pattern(required_cols=required_cols, extension="parquet", **kwargs)
    paths = list(dataset_dir.glob(glob_pattern))

    # Group by directory, and extract dimension values for directories with
    # more than one file.
    partitions_df = (
        pl.DataFrame(
            {
                "directory": [str(p.parent) for p in paths],
                "file": [str(p.name) for p in paths],
            }
        )
        .group_by(pl.col("directory"))
        .agg(pl.col("file"))
        .filter(pl.col("file").list.len() > 1)
        .with_columns(
            [
                pl.col("directory").str.extract(f"{dim.name}=([^/]+)(/|$)", 1).cast(dim.dtype).alias(dim.name)
                for dim in DIMENSIONS.values()
                if dim.name in partition_columns(input_df)
            ]
        )
    )

    # Iterate over the directories, read the data, and rewrite it
    # as a single file.
    for partition in partitions_df.iter_rows(named=True):
        read_kwargs = partition.copy()
        del read_kwargs["directory"]
        del read_kwargs["file"]

        write_parquet(
            read_parquet(**read_kwargs).collect(),
            validate=False,
            sort=sort,
            compression=compression,
            compression_level=compression_level,
            row_group_size=row_group_size,
            use_dictionary=use_dictionary,
        )

        # Delete the original files from the partition directory.
        directory = Path(partition["directory"])
        for file in partition["file"]:
            (directory / file).unlink()
