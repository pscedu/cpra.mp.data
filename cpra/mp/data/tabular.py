from pathlib import Path

import polars as pl

from cpra.mp.data.config import BASE_PATH
from cpra.mp.data.dimensions import DATASET_DIMENSIONS, DIMENSIONS
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.utils import (
    build_glob_pattern,
    dataset_path,
    extract_dimension_value,
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
    **kwargs,
):
    """Write data from a data frame to a Parquet file.

    Accepts a Polars DataFrame, LazyFrame, or any object that can be passed to
    pl.DataFrame().  All writes go through sink_parquet so the output always
    uses the Polars engine; DataFrames are wrapped with .lazy() before writing.
    A single metadata row is collected upfront to drive grid/sort/partition
    decisions.
    """

    lf = data if isinstance(data, pl.LazyFrame) else pl.DataFrame(data).lazy()

    # Add keyword arguments as literal columns.
    lf = lf.with_columns(**{k: pl.lit(v) for k, v in kwargs.items()})

    # If this is wide-format data (grid_cell IDs as column names), unpivot to long.
    _cols = lf.collect_schema().names()
    numeric_cols = [col for col in _cols if col[:1].isdigit()]
    if numeric_cols:
        alpha_cols = [col for col in _cols if not col[:1].isdigit()]
        lf = lf.unpivot(index=alpha_cols, variable_name="grid_cell", value_name="value").with_columns(
            pl.col("grid_cell").cast(pl.Int32)
        )

    # Collect one metadata row to drive decisions that need concrete values.
    meta = lf.head(1).collect()

    grid = validate_grid(meta, **kwargs)
    time_unit = extract_dimension_value("time_unit", meta, **kwargs)
    required_cols = required_columns(meta)

    # Validate and sort the data.
    if validate or sort:
        if validate:
            validate_data_frame(lf, required_cols=required_cols)

            # Validate grid_cell ID range.
            if grid.id_min is not None or grid.id_max is not None:
                gc_stats = lf.select(
                    pl.col("grid_cell").min().alias("min"),
                    pl.col("grid_cell").max().alias("max"),
                ).collect()
                gc_min, gc_max = gc_stats["min"][0], gc_stats["max"][0]
                if grid.id_min is not None and gc_min < grid.id_min:
                    msg = f"Geography ID below minimum for grid {grid.name}: {gc_min}"
                    raise ValidationError(msg)
                if grid.id_max is not None and gc_max > grid.id_max:
                    msg = f"Geography ID above maximum for grid {grid.name}: {gc_max}"
                    raise ValidationError(msg)

            # Validate value column dtype.
            variable = VARIABLES[meta["variable"][0]]
            if lf.schema["value"] != variable.dtype:
                msg = f"Column 'value' has dtype {lf.schema['value']!s}, expected {variable.dtype!s}"
                raise ValidationError(msg)

        if sort:
            sort_exprs = [pl.col(col) for col in required_cols if not col.startswith("calendar_")]
            if time_unit in ("daily", "monthly"):
                # Sort so each row group covers one calendar year: year → grid_cell → date.
                sort_exprs.extend([pl.col("calendar_day").dt.year(), pl.col("grid_cell"), pl.col("calendar_day")])
            else:
                sort_exprs.extend([pl.col("grid_cell"), pl.col("calendar_year")])

            lf = lf.select(*required_cols, "grid_cell", "value").sort(*sort_exprs)

    # Compute row group size from grid and time_unit if not provided explicitly.
    if row_group_size is None:
        row_group_size = _row_group_size(grid, time_unit)

    # Derive partition columns from the metadata row.
    partition_cols = partition_columns(meta)

    sink_args = [pl.PartitionBy(base_path, key=partition_cols, include_key=False)] if partition_cols else []
    sink_kwargs = dict(
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
    )
    lf.sink_parquet(*sink_args, **sink_kwargs)


def compact_parquet(
    base_path=BASE_PATH,
    sort=True,
    compression="zstd",
    compression_level=6,
    row_group_size=None,
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
