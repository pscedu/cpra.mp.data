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

# Days/months per year multiplied by a safety margin to ensure one row group
# covers at most one calendar year for each time unit.
_ROWS_PER_YEAR = {"daily": 400, "monthly": 15, "annual": 1, "decadal": 1}
_MAX_ROW_GROUP_SIZE = 5_000_000


def _row_group_size(grid, time_unit):
    """Return a row group size that aligns row group boundaries to calendar years.

    For daily/monthly data the target is one row group per year, so each row
    group's calendar_day min/max spans exactly one calendar year and Polars
    predicate pushdown can skip entire row groups for year-filtered queries.
    """
    if grid is None or grid.id_count is None:
        return 500_000
    rows_per_year = _ROWS_PER_YEAR.get(time_unit, 1)
    return min(max(grid.id_count * rows_per_year, 500_000), _MAX_ROW_GROUP_SIZE)


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
    row_group_size=None,
    use_dictionary=False,
    **kwargs,
):
    """Write data from a data frame to a Parquet file."""

    df = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)

    # Add keyword arguments to the dataframe as columns.
    df = df.with_columns(**{k: pl.lit(v) for k, v in kwargs.items()})

    # If this is wide-format data (grid_cell IDs as column names), unpivot to long.
    numeric_cols = [col for col in df.columns if col[:1].isdigit()]
    if numeric_cols:
        alpha_cols = [col for col in df.columns if not col[:1].isdigit()]
        df = df.unpivot(index=alpha_cols, variable_name="grid_cell", value_name="value").with_columns(
            pl.col("grid_cell").cast(pl.Int32)
        )

    grid = validate_grid(df, **kwargs)
    time_unit = df["time_unit"][0] if "time_unit" in df.columns else kwargs.get("time_unit")

    # Validate and sort the data.
    if validate or sort:
        required_cols = required_columns(df)

        if validate:
            validate_data_frame(df, required_cols=required_cols)

            # Validate grid_cell ID range.
            if grid.id_min is not None and df["grid_cell"].min() < grid.id_min:
                msg = f"Geography ID below minimum for grid {grid.name}: {df['grid_cell'].min()}"
                raise ValidationError(msg)
            if grid.id_max is not None and df["grid_cell"].max() > grid.id_max:
                msg = f"Geography ID above maximum for grid {grid.name}: {df['grid_cell'].max()}"
                raise ValidationError(msg)

            # Validate value column dtype.
            variable = VARIABLES[df["variable"][0]]
            if df["value"].dtype != variable.dtype:
                msg = f"Column 'value' has dtype {df['value'].dtype!s}, expected {variable.dtype!s}"
                raise ValidationError(msg)

        if sort:
            if time_unit in ("daily", "monthly"):
                # Sort so each row group covers one calendar year: year → grid_cell → date.
                sort_exprs = [pl.col("calendar_day").dt.year(), pl.col("grid_cell"), pl.col("calendar_day")]
            else:
                sort_exprs = [pl.col("grid_cell"), pl.col("calendar_year")]

            df = df.select(*required_cols, "grid_cell", "value").sort(*sort_exprs)

    # Compute row group size from grid and time_unit if not provided explicitly.
    if row_group_size is None:
        row_group_size = _row_group_size(grid, time_unit)

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
    row_group_size=None,
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
