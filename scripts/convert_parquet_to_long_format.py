import polars as pl
import polars.selectors as cs
from pathlib import Path
from polars.exceptions import ColumnNotFoundError
from cpra.mp.data import read_data
from cpra.mp.data.dimensions import DATASET_DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.utils import (
    glob_lazy_frame,
    partition_columns,
    required_columns,
)

DIMENSION_COLUMNS = cs.matches(r"^[a-z]")
LONG_BASE_PATH = Path("/home/mjyoder/cpra.mp.data/data-long")


def write_long(
    df,
    base_path=LONG_BASE_PATH,
    compression="zstd",
    compression_level=12,
    row_group_size=500000,
    use_dictionary=False,
):
    """Write a data frame in long format."""

    partition_cols = partition_columns(df)
    required_cols = required_columns(df)
    all_cols = required_cols + ["geography", "value"]

    (
        df.unpivot(index=DIMENSION_COLUMNS, variable_name="geography")
        .with_columns(pl.col("geography").cast(pl.Int32))
        .select(*all_cols)
        .sort(*all_cols)
        .write_parquet(
            base_path,
            compression=compression,
            compression_level=compression_level,
            row_group_size=row_group_size,
            use_pyarrow=True,
            pyarrow_options=dict(
                use_dictionary=use_dictionary,
                partition_cols=partition_cols,
            ),
        )
    )


def convert_to_long_format():
    """Convert Parquet datasets to long format."""

    datasets = glob_lazy_frame(required_cols=DATASET_DIMENSIONS).collect()

    for dataset_dict in datasets.iter_rows(named=True):
        src_path = Path(dataset_dict["path"])
        del dataset_dict["path"]
        dest_path = (
            LONG_BASE_PATH
            / f"variable={dataset_dict['variable']}"
            / f"grid={dataset_dict['grid']}"
            / f"time_unit={dataset_dict['time_unit']}"
        )

        if dest_path.is_dir() or dest_path.is_symlink():
            continue

        print(f"Converting {str(src_path)}...")

        # Symlink raster datasets to the original dataset directory.
        grid = GRIDS[dataset_dict["grid"]]
        if grid.grid_type == "raster":
            dest_path.symlink_to(src_path)
            continue

        if dataset_dict["time_unit"] == "daily":
            # Find all Parquet files and convert them one by one.
            input_df = pl.DataFrame(dataset_dict)
            dataset_dimensions = (
                glob_lazy_frame(
                    required_cols=partition_columns(input_df),
                    **dataset_dict,
                )
                .select(cs.exclude("path"))
                .collect()
            )
            for row in dataset_dimensions.iter_rows(named=True):
                df = read_data(**row).collect()
                write_long(df)
        else:
            try:
                df = read_data(**dataset_dict).collect()
            except ColumnNotFoundError:
                # Handle monthly datasets that are still partitioned.
                df = pl.read_parquet(src_path).with_columns(
                    **{k: pl.lit(v) for k, v in dataset_dict.items()}
                )
            write_long(df)


if __name__ == "__main__":
    convert_to_long_format()
