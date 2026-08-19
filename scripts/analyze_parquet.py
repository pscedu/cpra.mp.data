import sys
from collections import defaultdict

import pyarrow.parquet as pq


def get_parquet_column_sizes(file_path):
    # Open only the metadata footer of the parquet file
    parquet_file = pq.ParquetFile(file_path)
    metadata = parquet_file.metadata

    # Track sizes across all row groups
    column_sizes = defaultdict(lambda: {"compressed_bytes": 0, "uncompressed_bytes": 0})

    # Loop through each row group and aggregate column chunk sizes
    for rg_idx in range(metadata.num_row_groups):
        row_group = metadata.row_group(rg_idx)
        for col_idx in range(row_group.num_columns):
            column = row_group.column(col_idx)
            col_name = (
                ".".join(column.path_in_schema)
                if isinstance(column.path_in_schema, (list, tuple))
                else column.path_in_schema
            )

            column_sizes[col_name]["compressed_bytes"] += column.total_compressed_size
            column_sizes[col_name]["uncompressed_bytes"] += column.total_uncompressed_size

    # Print the formatted results
    print(f"{'Column Name':<30} | {'Compressed Size':<15} | {'Uncompressed Size':<15}")
    print("-" * 68)
    for col, sizes in column_sizes.items():
        print(f"{col:<30} | {sizes['compressed_bytes']:<15,} | {sizes['uncompressed_bytes']:<15,}")


if __name__ == "__main__":
    get_parquet_column_sizes(sys.argv[1])
