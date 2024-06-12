import polars as pl
import polars.selectors as cs

from cpra.mp.data.exceptions import SchemaValidationError
from cpra.mp.data.schema import SCHEMA

BASE_PATH = "/home/mjyoder/cpra.mp.data/data"


def write_data(
    df,
    base_path=BASE_PATH,
    validate=True,
    sort=True,
    compression="zstd",
    compression_level=6,
    row_group_size=5000,
    use_dictionary=False,
    **kwargs,
):
    """Write model data to disk."""

    # Add keyword arguments to the dataframe as columns.
    df = df.with_columns(**{k: pl.lit(v) for k, v in kwargs.items()})

    # If this is long-format data (i.e., with a geography column), unmelt it.
    df = df.with_columns(
        geography=pl.lit("g") + pl.col("geography").cast(pl.String)
    ).pivot(
        None,
        index=cs.exclude("geography", "variable_value"),
        columns=["geography"],
    )

    # Validate and sort the data.
    if validate or sort:
        # Check to make sure that all required columns are present.
        required_cols = [
            col.name
            for col in df.select(**{level.column: level.required for level in SCHEMA})
            if col.any()
        ]

        if validate:
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

        # TODO: Validate geography column names and dtypes.
        geogarphy_cols = [col for col in df.columns if col not in required_cols]

        # Sort columns and rows.
        if sort:
            df = df.select(*(required_cols + geogarphy_cols)).sort(*required_cols)

    # Get partition columns.
    partition_cols = [
        col.name
        for col in df.select(
            **{level.column: level.partition & level.required for level in SCHEMA}
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
