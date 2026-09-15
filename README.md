# CPRA Master Plan Data

Tools for reading and writing data from [Louisiana's Coastal Master Plan](https://coastal.la.gov/our-plan/2023-coastal-master-plan/).

## Data Structure

See the [data structure documentation](docs/index.md) for details about variables, dimensions, and validation conditions.

## APIs

Several APIs that depend on the Python library are implemented in the `cpra.mp.data.api` submodule:

* [Data API](docs/data-api.md)
* [Analysis API](docs/analysis-api.md)
* [Named Analyses (Registry) API](docs/analyses-api.md)

## Writing Data

To write tabular data using the library, prepare it as a data frame in the wide format, with dates/years as rows, and geographies as columns:

```python
import polars as pl
from datetime import date
from cpra.mp.data import read_data, write_data


# Define a Polars data frame containing the data to write. Pandas data frames are also supported as input data.
df1 = pl.DataFrame({
    "calendar_day": [date(2030, 1, 1), date(2030, 1, 2), date(2030, 1, 3)],
    "1": [0.4, 0.4, 0.5],
    "2": [2.1, 2.2, 2.3],
    ...
    "1786": [1.8, 1.8, 1.8],
    "1787": [6.4, 6.6, 6.5],
})

# Write the data to a Parquet file. Keyword arguments passed to write_data are added to the input
# data frame as columns.
write_data(
    df1,
    variable="sal",
    grid="hydro_compartment_v001",
    time_unit="daily",
    model_group_id=500,
    scenario_id=7,
    aggregate="mean",
    model="icm_v2023.0.0",
)
```

Raster data can be written to a COG using values from a NumPy array:

```python
from cpra.mp.data import read_fortran_array, write_data

# Read a Fortran binary dump into a NumPy array. The variable keyword argument is used
# to determine the data type of the array.
raster_array = read_fortran_array(
    "MP2023_S07_G520_C000_U00_V00_SLA_N_52_52_W_lndtyp30.xyz.b",
    variable="lnd_type",
)

# Write the data to a COG.
write_data(
    raster_array,
    variable="lnd_type",
    grid="morph_pixel_v001",
    time_unit="annual",
    model_group_id=520,
    scenario_id=7,
    calendar_year=2070,
    model="icm_v2023.0.0",
)
```

## Reading Data

To read the data, use the same keyword arguments used when it was written. For tabular data, `read_data` returns a Polars [`LazyFame`](https://docs.pola.rs/api/python/stable/reference/lazyframe/index.html). To materialize the data frame, use the `.collect()` method.:

```python
>>> from cpra.mp.data import read_data
>>> lf = read_data(variable="sal", grid="hydro_compartment_v001", time_unit="daily", aggregate="mean", model_group_id=500, scenario_id=7)
>>> lf.collect()
shape: (3, 1795)
┌──────────┬────────────────────────┬───────────┬───┬───────┬───────┬───────┬───────┐
│ variable ┆ grid                   ┆ time_unit ┆ … ┆ 1784  ┆ 1785  ┆ 1786  ┆ 1787  │
│ ---      ┆ ---                    ┆ ---       ┆   ┆ ---   ┆  ---  ┆ ---   ┆ ---   │
│ str      ┆ str                    ┆ str       ┆   ┆ f64   ┆ f64   ┆ f64   ┆ f64   │
╞══════════╪════════════════════════╪═══════════╪═══╪═══════╪═══════╪═══════╪═══════╡
│ sal      ┆ hydro_compartment_v001 ┆ daily     ┆ … ┆ 10.1  ┆ 7.9   ┆ 1.8   ┆ 6.4   │
│ sal      ┆ hydro_compartment_v001 ┆ daily     ┆ … ┆ 10.2  ┆ 8.0   ┆ 1.8   ┆ 6.6   │
│ sal      ┆ hydro_compartment_v001 ┆ daily     ┆ … ┆ 10.0  ┆ 8.0   ┆ 1.8   ┆ 6.5   │
└──────────┴────────────────────────┴───────────┴───┴───────┴───────┴───────┴───────┘
```

> [!TIP]
> Calling `LazyFrame.collect()` when the underlying Parquet files do not exist (e.g., filtering on a model group that has not yet been run) will raise `FileNotFoundError`. Applications where the filter parameters are user-supplied should handle this exception.

Since the `LazyFrame` returned by `read_data` has not yet been materialized, additional Polars filters expressions can be used to achieve advanced filtering logic (using logical operators, etc.) without a performance penalty:

```python
import polars as pl
from cpra.mp.data import read_data
from datetime import date
g520_ip1 = p.col("model_group_id") == 520 & pl.col("calendar_day") < date(2040, 1, 1)
g521_ip2 = p.col("model_group_id") == 521 & pl.col("calendar_day") >= date(2040, 1, 1)
lf2 = read_data(
    variable="sal",
    grid="hydro_compartment_v001",
    time_unit="daily",
    aggregate="mean",
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
    variable="lnd_type",
    grid="hydro_compartment_v001",
    time_unit="annual",
    model_group_id=520,
    scenario_id=7,
)

# Create a virtual dataset (VRT) from the COG paths so that they
# can be accessed as a single multiband image.
with rio.open(vrt(cog_df["path"])) as multiband:
    print(multiband.profile)
```

## Analyzing Data

Two functions cover everything beyond a raw, single-dataset read:

* `read_data` selects and narrows one dataset's stored values — the library form of the
  [Data API](docs/data-api.md).
* `analyze_data` runs a pipeline — filtering, calculating, aggregating, projecting onto a
  different grid, spatially joining across datasets — the library form of the
  [Analysis API](docs/analysis-api.md).

Neither makes an HTTP request; both read Parquet/COG files directly.

### `read_data`

```python
def read_data(*, variable, time_unit, grid, base_path=None, **selection)
```

`variable`, `time_unit`, and `grid` select the dataset. Every other keyword goes into
`**selection`: a dimension filter (a scalar or a list of values, e.g. `scenario_id=7` or
`scenario_id=[7, 8]`), or one of the reserved selection keys `grid_cell`, `where`, `sort`,
`select` — the same parameters and semantics documented for the
[Data API](docs/data-api.md#filter)'s query string. `read_data` only ever selects and
narrows a single dataset; it never transforms, aggregates, or joins.

### `analyze_data`

```python
async def analyze_data(pipeline, *, base_path=None, output_path=None, lazy=False)
```

`pipeline` is required, and is a `dict`/JSON document in exactly the shape the
[Analysis API](docs/analysis-api.md#request-shape) accepts (`inputs`/`steps`/`output`).
There is no separate flat-keyword-argument form — every transformation, including a
single-dataset filter/calculate/group, is expressed as a (possibly one-input) pipeline.
`analyze_data` is `async`; in a notebook, just `await` it, or wrap the call with
`asyncio.run` in a plain script.

The return value depends on the pipeline's own `output.format`:

* **Tabular** output (`csv`/`ndjson`/`parquet`) returns a Polars `DataFrame` — or a
  `LazyFrame` if `lazy=True`.
* **Raster** output (`tif`/tile formats) requires `output_path=` (there is no in-memory
  raster return) and returns the written file's `pathlib.Path`.

```python
from cpra.mp.data import analyze_data

# Count the days each hydro compartment's salinity exceeded 5.5 in one scenario.
df = await analyze_data({
    "inputs": {
        "sal": {
            "variable": "sal", "time_unit": "daily", "grid": "hydro_compartment_v001",
            "model_group_id": 500, "scenario_id": 7, "aggregate_type": "mean",
        }
    },
    "steps": [
        {"op": "project", "input": "sal", "to": "hydro_compartment_v001"},
        {"op": "filter", "where": "value > 5.5"},
        {"op": "group", "by": [], "aggregate": "count"},
    ],
    "output": {"format": "csv"},
})
# → one row per grid_cell (an empty group.by collapses every day into one count).
```

### Raster Analyses

A pipeline that reaggregates raster pixels onto a coarser vector zone (a many-to-one
`project`) still returns a table — surviving dimensions become columns, so there is no
band bookkeeping:

```python
# Mean elevation per hydro compartment, per year, averaged across two scenarios.
df = await analyze_data({
    "inputs": {
        "elev": {
            "variable": "elev", "time_unit": "annual", "grid": "morph_pixel_v001",
            "model_group_id": 500, "scenario_id": [7, 8],
        }
    },
    "steps": [
        {"op": "project", "input": "elev", "to": "hydro_compartment_v001", "aggregate": "mean"},
        {"op": "group", "by": ["calendar_year"], "aggregate": "mean"},
    ],
    "output": {"format": "csv"},
})
# columns: grid_cell, calendar_year, value — one row per (compartment, year)
```

A pipeline whose result is a full-resolution raster image (a band collapse with no
zonal `project`) has no sensible tabular form, so it requires `output_path` and writes a
GeoTIFF instead, returning the path:

```python
path = await analyze_data(
    {
        "inputs": {
            "elev": {
                "variable": "elev", "time_unit": "annual", "grid": "morph_pixel_v001",
                "model_group_id": 500, "scenario_id": [7, 8],
            }
        },
        "steps": [
            {"op": "project", "input": "elev", "to": "morph_pixel_v001"},
            {"op": "group", "by": ["calendar_year"], "aggregate": "mean"},
        ],
        "output": {"format": "tif"},
    },
    output_path="mean_elev_by_year.tif",
)
```

### Multi-Dataset Pipelines

For analyses spanning several datasets — projecting one grid onto another, spatially
joining, filtering across both — the pipeline names more than one entry under `inputs`.
The pipeline document is identical to the one the API's `/v1/analysis` endpoints accept
(see the [Analysis API documentation](docs/analysis-api.md#request-shape) for the full
request shape and operations):

```python
# Earliest year each vegetation pixel has marsh coverage > 0.5 AND salinity > 5.5.
path = await analyze_data(
    {
        "inputs": {
            "marsh": {
                "variable": "ffibs_coverage_ratio", "time_unit": "annual",
                "grid": "veg_grid_cell_v001",
                "model_group_id": 500, "scenario_id": 7, "ffibs_type": "pl_fm",
            },
            "salinity": {
                "variable": "sal", "time_unit": "annual",
                "grid": "hydro_compartment_v001",
                "model_group_id": 500, "scenario_id": 7, "aggregate_type": "2_wk_max",
            },
        },
        "steps": [
            {"op": "project", "input": "salinity", "to": "veg_grid_cell_v001", "as": "sal"},
            {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
            {"op": "filter", "where": "marsh > 0.5 AND sal > 5.5"},
            {"op": "group", "by": [], "aggregate": "min", "of": "calendar_year"},
        ],
        "output": {"format": "tif"},
    },
    output_path="first_exceedance_year.tif",
)
```

A small set of pre-built pipelines — a difference between two scenarios, a land-type
concatenation, this same earliest-exceedance-year query — is also available by name with
no hand-authored pipeline document at all; see the
[named-analysis registry documentation](docs/analyses-api.md).
