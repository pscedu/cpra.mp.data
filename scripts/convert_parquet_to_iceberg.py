import polars as pl
import polars.selectors as cs
from pathlib import Path
from polars.exceptions import ColumnNotFoundError
from pyiceberg.catalog.sql import SqlCatalog
from cpra.mp.data import read_data
from cpra.mp.data.dimensions import DATASET_DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.utils import (
    glob_lazy_frame,
    partition_columns,
    required_columns,
)

DIMENSION_COLUMNS = cs.matches(r"^[a-z]")
WAREHOUSE_PATH = Path("/home/mjyoder/cpra.mp.data/iceberg")
WAREHOUSE_PATH.mkdir(parents=True, exist_ok=True)
CATALOG = SqlCatalog(
    "default",
    **{
        "uri": f"sqlite:///{WAREHOUSE_PATH}/catalog.db",
        "warehouse": f"file://{WAREHOUSE_PATH}",
    },
)
CATALOG.create_namespace_if_not_exists("default")


def write_long(df):
    """Write a data frame in long format."""

    required_cols = required_columns(df)
    all_cols = required_cols + ["geography", "value"]

    arrow_df = (
        df.unpivot(index=DIMENSION_COLUMNS, variable_name="geography")
        .with_columns(pl.col("geography").cast(pl.Int32))
        .select(*all_cols)
        .sort(*all_cols)
        .to_arrow()
    )

    table_name = f"default.{df['variable'][0]}_{df['time_unit'][0]}"
    table = CATALOG.create_table_if_not_exists(table_name, schema=arrow_df.schema)
    table.append(arrow_df)


def convert_to_long_format():
    """Convert Parquet datasets to long format."""

    datasets = glob_lazy_frame(required_cols=DATASET_DIMENSIONS).collect()

    for dataset_dict in datasets.iter_rows(named=True):
        src_path = Path(dataset_dict["path"])
        del dataset_dict["path"]

        print(f"Converting {str(src_path)}...")

        # Skip raster datasets.
        grid = GRIDS[dataset_dict["grid"]]
        if grid.grid_type == "raster":
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
