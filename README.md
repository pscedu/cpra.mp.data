# CPRA Master Plan Data

Tools for reading and writing data from [Louisiana's Coastal Master Plan](https://coastal.la.gov/our-plan/2023-coastal-master-plan/).

## Writing Data

To write data using the library, prepare it as a data frame in the wide format, with dates/years as rows, and geographies as columns:

```python
import polars as pl
from datetime import date
from cpra.mp.data import read_data, write_data


# Define a Polars data frame containing the data to write. Pandas data frames are also supported as input data.
df1 = pl.DataFrame({
    "calendar_date": [date(2030, 1, 1), date(2030, 1, 2), date(2030, 1, 3)],
    "g1": [0.4, 0.4, 0.5],
    "g2": [2.1, 2.2, 2.3],
    ...
    "g1786": [1.8, 1.8, 1.8],
    "g1787": [6.4, 6.6, 6.5],
})

# Write the data to a Parquet file. Keyword arguments passed to write_data are added to the input
# data frame as columns.
write_data(
    df1,
    master_plan=2029,
    variable="salinity",
    geographic_unit="hydro_compartment",
    aggregate="mean",
    time_unit="daily",
    model_group=500,
    scenario=7
)
```

## Reading Data

To read the data, use the same keyword arguments used when it was written:

```python
>>> df2 = read_data(master_plan=2029, variable="salinity", geographic_unit="hydro_compartment", aggregate="mean", time_unit="daily", model_group=500, scenario=7)
>>> df2
shape: (3, 1795)
┌─────────────┬──────────┬───────────────────┬───────────┬───┬───────┬───────┬───────┬───────┐
│ master_plan ┆ variable ┆ geographic_unit   ┆ aggregate ┆ … ┆ g1784 ┆ g1785 ┆ g1786 ┆ g1787 │
│ ---         ┆ ---      ┆ ---               ┆ ---       ┆   ┆ ---   ┆ ---   ┆ ---   ┆ ---   │
│ i32         ┆ str      ┆ str               ┆ str       ┆   ┆ f64   ┆ f64   ┆ f64   ┆ f64   │
╞═════════════╪══════════╪═══════════════════╪═══════════╪═══╪═══════╪═══════╪═══════╪═══════╡
│ 2029        ┆ salinity ┆ hydro_compartment ┆ mean      ┆ … ┆ 10.1  ┆ 7.9   ┆ 1.8   ┆ 6.4   │
│ 2029        ┆ salinity ┆ hydro_compartment ┆ mean      ┆ … ┆ 10.2  ┆ 8.0   ┆ 1.8   ┆ 6.6   │
│ 2029        ┆ salinity ┆ hydro_compartment ┆ mean      ┆ … ┆ 10.0  ┆ 8.0   ┆ 1.8   ┆ 6.5   │
└─────────────┴──────────┴───────────────────┴───────────┴───┴───────┴───────┴───────┴───────┘
```

For more advanced Parquet filtering, you can also provide a `filters` argument in the same [format used by `pyarrow`](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html):

```python
# In the filters argument, elements in the inner lists are joined with AND,
# while elements in the outer list are joined with OR. All keyword arguments
# passed to read_data are added to the inner lists (i.e., joined to the 
# explicit filter arguments using AND).
df3 = read_data(
    master_plan=2029,
    variable="salinity",
    geographic_unit="hydro_compartment",
    aggregate="mean",
    time_unit="daily",
    scenario=7,
    filters=[
        [("model_group", "==", 520), ("calendar_date", "<", "2040-01-01")],
        [("model_group", "==", 521), ("calendar_date", ">=", "2040-01-01")],
    ]
)
```

