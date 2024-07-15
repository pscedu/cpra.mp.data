# CPRA Master Plan Data

Tools for reading and writing data from [Louisiana's Coastal Master Plan](https://coastal.la.gov/our-plan/2023-coastal-master-plan/).

## Data Structure

See the [data structure documentation](docs/index.md) for details about variables, dimensions, and validation conditions.

## Writing Data

To write tabular data using the library, prepare it as a data frame in the wide format, with dates/years as rows, and geographies as columns:

```python
import polars as pl
from datetime import date
from cpra.mp.data import read_data, write_data


# Define a Polars data frame containing the data to write. Pandas data frames are also supported as input data.
df1 = pl.DataFrame({
    "calendar_day": [date(2030, 1, 1), date(2030, 1, 2), date(2030, 1, 3)],
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
    variable="salinity",
    geographic_unit="hydro_compartment",
    aggregate="mean",
    time_unit="daily",
    plan_version="2023.0.0",
    model_group_id=500,
    scenario_id=7
)
```

Raster data can be written to a COG using values from a NumPy array:

```python
from cpra.mp.data import read_fortran_array, write_data

# Read a Fortran binary dump into a NumPy array. The variable keyword argument is used
# to determine the data type of the array.
raster_array = read_fortran_array(
    "MP2023_S07_G520_C000_U00_V00_SLA_N_52_52_W_lndtyp30.xyz.b",
    variable="land_type",
)

# Write the data to a COG.
write_data(
    raster_array,
    variable="land_type",
    geographic_unit="icm_raster",
    time_unit="annual",
    plan_version="2023.0.0",
    model_group_id=520,
    scenario_id=7,
    calendar_year=2070,
)
```

## Reading Data

To read the data, use the same keyword arguments used when it was written. For tabular data, `read_data` returns a Polars [`LazyFame`](https://docs.pola.rs/api/python/stable/reference/lazyframe/index.html). To materialize the data frame, use the `.collect()` method.:

```python
>>> from cpra.mp.data import read_data
>>> lf = read_data(variable="salinity", geographic_unit="hydro_compartment", aggregate="mean", time_unit="daily", plan_version="2023.0.0", model_group_id=500, scenario_id=7)
>>> lf.collect()
shape: (3, 1795)
┌──────────┬───────────────────┬───────────┬───────────┬───┬───────┬───────┬───────┬───────┐
│ variable ┆ geographic_unit   ┆ aggregate ┆ time_unit ┆ … ┆ g1784 ┆ g1785 ┆ g1786 ┆ g1787 │
│ ---      ┆ ---               ┆ ---       ┆ ---       ┆   ┆ ---   ┆ ---   ┆ ---   ┆ ---   │
│ str      ┆ str               ┆ str       ┆ str       ┆   ┆ f64   ┆ f64   ┆ f64   ┆ f64   │
╞══════════╪═══════════════════╪═══════════╪═══════════╪═══╪═══════╪═══════╪═══════╪═══════╡
│ salinity ┆ hydro_compartment ┆ mean      ┆ daily     ┆ … ┆ 10.1  ┆ 7.9   ┆ 1.8   ┆ 6.4   │
│ salinity ┆ hydro_compartment ┆ mean      ┆ daily     ┆ … ┆ 10.2  ┆ 8.0   ┆ 1.8   ┆ 6.6   │
│ salinity ┆ hydro_compartment ┆ mean      ┆ daily     ┆ … ┆ 10.0  ┆ 8.0   ┆ 1.8   ┆ 6.5   │
└──────────┴───────────────────┴───────────┴───────────┴───┴───────┴───────┴───────┴───────┘
```

Since the `LazyFrame` returned by `read_data` has not yet been materialized, additional Polars filters expressions can be used to achieve advanced filtering logic (using logical operators, etc.) without a performance penalty:

```python
import polars as pl
from cpra.mp.data import read_data
from datetime import date
g520_ip1 = p.col("model_group_id") == 520 & pl.col("calendar_day") < date(2040, 1, 1)
g521_ip2 = p.col("model_group_id") == 521 & pl.col("calendar_day") >= date(2040, 1, 1)
lf2 = read_data(
    variable="salinity",
    geographic_unit="hydro_compartment",
    aggregate="mean",
    time_unit="daily",
    plan_version="2029.0.0",
    scenario_id=7,
).filter(g520_ip1 | g521_ip2)
```

Reading raster data is a two-step process. Calling `read_data` for a raster-grid variable returns a DataFrame containing information about matching COGs. The `path` column contains the path to the COGs, and the other columns contain dimension values for the COGs. The dimension columns can be used to perform advanced filtering using the Polars API.

The second step is to use a library like [Rasterio](https://rasterio.readthedocs.io/en/stable/) to open the COG paths and access the image data. A common access pattern is to create a [VRT dataset](https://gdal.org/drivers/raster/vrt.html) that merges several single-band COGs into a virtual multiband image. The `vrt` helper function takes an iterable containing COG paths and returns an XML string describing a VRT dataset that can be opened using GDAL or Rasterio.

```python
import rasterio as rio
from cpra.mp.data import vrt, read_data

# Generate a data frame containing information about matching COGs.
cog_df = read_data(
    variable="land_type",
    geographic_unit="icm_raster",
    time_unit="annual",
    plan_version="2023.0.0",
    model_group_id=520,
    scenario_id=7,
)

# Create a virtual dataset (VRT) from the COG paths so that they
# can be accessed as a single multiband image.
with rio.open(vrt(cog_df["path"])) as multiband:
    print(multiband.profile)
```
