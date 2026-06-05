from pathlib import Path

import polars as pl

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
    validate_grids,
)
from cpra.mp.data.variables import VARIABLES


def read_parquet(base_path=BASE_PATH, cache=True, include_file_paths=None, **kwargs):
    """Read tabular data into a data frame from a Parquet file."""

    # Validate the input keyword arguments.
    input_df = get_input_data_frame(**kwargs)
    validate_data_frame(input_df, required_cols=DATASET_DIMENSIONS.keys())

    # Calculate the dataset directory path and expressions.
    dataset_dir = dataset_path(base_path, **kwargs)
    dataset_exprs = [pl.lit(kwargs[dim.name]).cast(dim.dtype).alias(dim.name) for dim in DATASET_DIMENSIONS.values()]
    filter_args = get_filter_expressions(**kwargs)
    if "calendar_day" in kwargs:
        days = kwargs["calendar_day"]
        if not isinstance(days, (list, tuple)):
            days = [days]
        years = list({d.year for d in days})
        filter_args.append(pl.col("calendar_year").is_in(years))

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
        pl.col("grid_cell"),
        pl.col("value"),
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
    row_group_size=500_000,
    **kwargs,
):
    """Write data from a data frame to a Parquet file."""

    df = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)

    # Add keyword arguments to the dataframe as columns.
    df = df.with_columns(**{k: pl.lit(v) for k, v in kwargs.items()})

    # Derive calendar_year from calendar_day if not already present.
    if "calendar_year" not in df.columns and "calendar_day" in df.columns:
        df = df.with_columns(
            pl.col("calendar_day").dt.year().cast(pl.Int32).alias("calendar_year")
        )

    # Validate and sort the data.
    if validate or sort:
        # Check to make sure that all required columns are present.
        required_cols = required_columns(df)

        if validate:
            for col in ("grid_cell", "value"):
                if col not in df.columns:
                    raise ValidationError(f"Required column missing: {col}")
            validate_data_frame(df, required_cols=required_cols)

        # Validate grids and geography IDs.
        validate_grids(df)

        # Validate value dtype against the variable spec.
        variable = VARIABLES[df["variable"][0]]
        if df["value"].dtype != variable.dtype:
            msg = f"Series value has dtype {df['value'].dtype!s}, expected {variable.dtype!s}"
            raise ValidationError(msg)

        # Sort columns and rows. calendar_day is placed after grid_cell so rows
        # are ordered cell-first within each partition (all days for cell N, then
        # all days for cell N+1), matching the access pattern for spatial queries.
        if sort:
            day_cols = [c for c in ["calendar_day"] if c in df.columns]
            non_day_required = [c for c in required_cols if c not in day_cols]
            sort_keys = non_day_required + ["grid_cell"] + day_cols
            df = df.select(*required_cols, "grid_cell", "value").sort(*sort_keys)

    # Get partition columns.
    partition_cols = partition_columns(df)

    # Write the partitioned file.
    df.write_parquet(
        base_path,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        use_pyarrow=True,
        pyarrow_options=dict(partition_cols=partition_cols),
    )


def compact_parquet(
    base_path=BASE_PATH,
    sort=True,
    compression="zstd",
    compression_level=6,
    row_group_size=500_000,
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
        )

        # Delete the original files from the partition directory.
        directory = Path(partition["directory"])
        for file in partition["file"]:
            (directory / file).unlink()
