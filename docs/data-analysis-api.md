# Data and Analysis API

The data and analysis API provides a RESTful interface for accessing and transforming model data stored in Parquet and COG files. Data from Parquet files can be returned in several tabular formats. Data from COGs can be returned as GeoTIFFs, raster map tiles, or — when summarized or extracted at points — in any of the tabular formats. Vector-grid data can also be rendered as vector map tiles.

All of the data endpoints use the general format:

```
/v1/datasets/<variable>/<time_unit>/<grid>/data.<format>
```

Example URIs for vector-grid (Parquet) data are:

```
/v1/datasets/sal/daily/hydro_compartment_v001/data.csv
/v1/datasets/sal/daily/hydro_compartment_v001/data.ndjson
/v1/datasets/sal/daily/hydro_compartment_v001/data.parquet
```

Additional example URIs for raster-grid (COG) data and map tiles are:

```
/v1/datasets/elev/annual/morph_pixel_v001/data.tif
/v1/datasets/elev/annual/morph_pixel_v001/data/{z}/{x}/{y}.png
/v1/datasets/damage/decadal/clara_grid_cell_v001/tiles/{z}/{x}/{y}.pbf
```

Tabular formats are `csv`, `ndjson`, and `parquet`. Raster output is `tif` (GeoTIFF); raster map tiles are `png` or `jpg`; vector map tiles are Mapbox vector tiles (`pbf`). Tabular and GeoTIFF responses accept `compression_algorithm` and `compression_level` parameters (the accepted algorithms differ by format; `zstd` level 6 is the default).

## Query Operations

Without any query string parameters, each endpoint returns the full dataset by default. To obtain a subset of the data or to perform transformation and analysis operations, query string parameters can be added. There are five query operations, which are always performed in the same order:

| Order | Operation | Vector-Grid Status | Raster-Grid Status |
|---|---|---|---|
| 1 | [Filter](#filter) | ✅ Implemented | ✅ Implemented |
| 2 | [Calculate](#calculate) | ✅ Implemented | ✅ Implemented |
| 3 | [Group and Aggregate](#group-and-aggregate) | ✅ Implemented | ✅ Implemented |
| 4 | [Transform and Extract](#transform-and-extract) | ✅ Implemented | ✅ Implemented |
| 5 | [Sort and Select](#sort-and-select) | ✅ Implemented | ✅ Implemented |

### Filter

Filter parameters limit the rows, columns, or bands of the dataset to those matching specific criteria. Any dimension from the partition and non-partition dimensions can be used as a filter parameter. Multiple values can be provided in a comma-separated list. Note that special characters in SQL-style parameters should be [percent-encoded](https://developer.mozilla.org/en-US/docs/Glossary/Percent-encoding) (e.g., using [`encodeURIComponent`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/encodeURIComponent)).

For vector-grid data, the `grid_cell` parameter accepts a comma-separated list of source grid cell IDs. For raster-grid data, it accepts a comma-separated list of coordinate pairs (in the format `x,y`) at which values are extracted from the COGs. When specifying coordinate pairs for a raster-grid dataset, the result is a tabular dataset that can be transformed in the same way as Parquet data (e.g., using the `unpivot` parameter below).

Vector-grid data also accepts coordinate pairs, so the same `grid_cell` value can be used against any grid without knowing its type. The list is read as grid cell IDs only when every value is a plain unsigned integer; otherwise it is read as coordinate pairs, and each pair is resolved to the grid cell whose boundary contains it. Results are always reported under the grid cell ID, never the coordinate pair that selected it — as a column name, or as a `grid_cell` value once unpivoted. A coordinate pair that falls outside every cell in the grid is omitted from the results, and several pairs that fall inside the same cell report it once.

For raster-grid data, when the `grid_cell` parameter is not provided, the full images are used. Each COG band matching the filters becomes a band in the resulting image. These bands can be aggregated using the `group_by` and `group_aggregate` parameters below.

A single SQL-style `where` parameter accepts two kinds of conditions, joined with `AND`:

- **Dimension** conditions, against dimension columns, using SQL where-clause syntax without restriction on the expression (aside from the aggregate/subquery rule below).
- **Value** conditions, against the data values themselves, written using the placeholder column `value` (e.g. `value > 3`). A value condition is parsed against a restricted whitelist of comparison operators and elementwise functions. A non-matching value is treated as missing (excluded from aggregates and counts) rather than dropping the row when the request is aggregating (`group_by`/`group_aggregate` or `summarize_by`/`summarize_aggregate`) — which is what lets a value condition compose with every aggregate, including `count`, making "count pixels where value > x" queries possible. For a non-aggregating (plain tabular or tile) request, a row failing a value condition is dropped instead.

Every `AND`-joined term in `where` must be entirely one kind or the other — a single comparison cannot mix a dimension column and `value` (e.g. `value > calendar_year` is rejected; write them as two separate `AND`-ed conditions instead, like `calendar_year > 2040 AND value > 3`). Whichever kind a term is, an aggregate function call or a subquery inside it is rejected before any data is read, because either would force the whole dataset to be materialized to evaluate what was supposed to be a per-row test.

| Parameter | Description | Vector-Grid | Raster-Grid | Example |
|---|---|---|---|---|
| Partition and Non-Partition Dimensions | Any dimension from the partition and non-partition tables above | ✅ | ✅ | `aep=0.2` |
| `grid_cell` | Grid cells to include, as IDs (vector grids only) or as `x,y` coordinate pairs | ✅ | ✅ | `grid_cell=24,81` or `grid_cell=-90.49,30.01` |
| `where` | Dimension and/or value filters in SQL where-clause format | ✅ | ✅ (value conditions: summarize path only) | `where=calendar_year%3E2040` or `where=value%20%3E%203` |

### Calculate

Calculate parameters change the values of individual table cells or raster pixels prior to aggregation. They are intended to support comparison operations, such as calculating the difference between two scenarios or concatenating digits to highlight changes. See the examples below for specific applications of these parameters.

| Parameter | Description | Vector-Grid | Raster-Grid | Example |
|---|---|---|---|---|
| `negate` | SQL expression for rows/bands where the value should be multiplied by -1 | ✅ | ✅ | `negate=scenario_id%3D8` |
| `multiply10` | SQL expression for rows/bands where the value should be multiplied by 10 | ✅ | ✅ | `multiply10=scenario_id%3D8%20AND%20calendar_year%3D2030` |
| `multiply100` | SQL expression for rows/bands where the value should be multiplied by 100 | ✅ | ✅ | `multiply100=scenario_id%3D8%20AND%20calendar_year%3D2040` |

### Group and Aggregate

Group and aggregate parameters allow multiple rows or bands to be summarized as a single row or band. Unlike the geographic aggregations in the transformation step below, these aggregations operate on a single vector grid cell or raster pixel. For differences, select both values of the differing dimension, use `negate` on one of them, and collapse with `group_aggregate=sum`. Note one limitation of that idiom: a cell or pixel present in only one of the two selections contributes just its own value to the "difference", indistinguishable from a true difference against zero. All currently served grids are dense, so this does not arise today.

| Parameter | Description | Vector-Grid | Raster-Grid | Example |
|---|---|---|---|---|
| `group_by` | Group by dimensions | ✅ | ✅ | `group_by=model_group_id,scenario_id,aep` |
| `group_aggregate` | Aggregate function: one of `sum`, `max`, `min`, `mean`, or `count` (`count` is vector-grid only — "how many bands contributed" has no per-pixel meaning; `count_by_value` is not valid here on either grid type) | ✅ | ✅ | `group_aggregate=sum` |

### Transform and Extract

Transform and extract parameters change the shape or format of the resulting dataset, summarize data at other geographic levels, project data from one grid onto another, or apply color maps to raster-grid output.

`summarize_aggregate` accepts `sum`, `max`, `min`, `mean`, `count`, and `count_by_value`. Aggregates skip missing values (raster NoData, or values masked by a `where` value condition); a summary zone with no contributing values at all reports a missing value rather than being omitted. `count` counts contributing cells or pixels, which combined with a `where` value condition yields conditional counts. `count_by_value` produces a different output shape from the other aggregates — one row per (zone, distinct value) with the count of cells or pixels carrying that value — and is intended for bounded categorical data such as land-type codes; a selection with more distinct values than the configured ceiling is rejected.

| Parameter | Description | Vector-Grid | Raster-Grid | Example |
|---|---|---|---|---|
| `color_map` | Apply the given color map to raster output | ❌ | ✅ | `color_map=ffibs` |
| `unpivot` | Change the data shape from wide to long, giving the resulting table a grid_cell column and a value column instead of a separate column per grid cell | ✅ | ✅ (once raster output is tabular, via `grid_cell` or `summarize_by`) | `unpivot=true` |
| `summarize_by` | Summarize by geographic unit | ✅ | ✅ | `summarize_by=hydro_compartment_v001` |
| `summarize_aggregate` | Aggregate function: one of `sum`, `max`, `min`, `mean`, `count`, or `count_by_value` | ✅ | ✅ | `summarize_aggregate=max` |
| `summarize_grid_cell` | Limit the summary zones to the provided IDs | ✅ | ✅ | `summarize_grid_cell=659,680,700` |

### Sort and Select

The `sort` parameter sorts the final results. It accepts multiple dimensions in a comma-separated list. Dimensions are sorted in ascending order by default unless prefixed with a minus sign (`-`) for descending order. The `select` parameter limits the returned columns to the provided comma-separated list.

| Parameter | Description | Vector-Grid | Raster-Grid | Example |
|---|---|---|---|---|
| `sort` | Sort order | ✅ | ✅ | `sort=scenario_id,model_group_id,calendar_day` |
| `select` | Columns to return | ✅ | ✅ (tabular output) | `select=grid_cell,calendar_year,value` |

## Examples

The following examples illustrate common patterns of use for the data and analysis API.

### Vector-Grid Data

#### Basic Map Query

This request returns daily mean salinity data for all grid cells on a specific date as CSV.

```
GET /v1/datasets/sal/daily/hydro_compartment_v001/data.csv?model_group_id=520&scenario_id=7&calendar_day=2030-01-01
```

#### Map Query Comparing Two Scenarios

This request returns the difference in daily mean salinity between two scenarios on the same date (scenario 7 minus scenario 8), using the `negate` + `group_aggregate=sum` difference idiom.

```
GET /v1/datasets/sal/daily/hydro_compartment_v001/data.csv?model_group_id=520&scenario_id=7,8&calendar_day=2030-01-01&negate=scenario_id%3D8&group_by=calendar_day&group_aggregate=sum
```

#### Basic Time Series Query

This request returns daily time series data for three grid cells as newline-delimited JSON.

```
GET /v1/datasets/sal/daily/hydro_compartment_v001/data.ndjson?model_group_id=520&scenario_id=7&grid_cell=79,842,1003&sort=calendar_day
```

#### Query with Additional Dimensions

This request returns decadal damage for a single grid cell as newline-delimited JSON. Since the `aep`, `asset_type`, and `percentile` parameters are not provided, the query matches all available values for those dimensions. The results are summed across `asset_type` to provide the total damage for each AEP-percentile combination.

```
GET /v1/datasets/damage/decadal/clara_grid_cell_v001/data.ndjson?model_group_id=500&scenario_id=7&pumping_id=0&fragility_scenario=0&grid_cell=12&group_by=aep,percentile&group_aggregate=sum&sort=aep,percentile
```

#### Conditional Count

This request counts, per grid cell, how many decadal damage values exceed a threshold.

```
GET /v1/datasets/damage/decadal/clara_grid_cell_v001/data.csv?model_group_id=500&scenario_id=7&where=value%20%3E%201000000&group_aggregate=count
```

### Raster-Grid Data

#### Basic Map Query

This request returns colorized elevation data for a single year as PNG map tiles.

```
GET /v1/datasets/elev/annual/morph_pixel_v001/data/{z}/{x}/{y}.png?model_group_id=500&scenario_id=7&calendar_year=2045&color_map=elevation
```

#### Map Query Comparing Two Years

This request returns the colorized elevation difference between two years (2050 minus 2045) as PNG map tiles, using the `negate` + `group_aggregate=sum` difference idiom.

```
GET /v1/datasets/elev/annual/morph_pixel_v001/data/{z}/{x}/{y}.png?model_group_id=500&scenario_id=7&calendar_year=2045,2050&negate=calendar_year%3D2045&group_by=true&group_aggregate=sum&color_map=elevation_difference
```

#### Time Series Query at Specific Points

This request returns an elevation time series at three longitude-latitude pairs as an unpivoted (long format) CSV.

```
GET /v1/datasets/elev/annual/morph_pixel_v001/data.csv?model_group_id=500&scenario_id=7&grid_cell=-90.4914181,30.0144111,-90.8628217,29.4254702,-91.3641637,29.5319325&unpivot=true&sort=calendar_year,grid_cell
```

#### Single Year Query Summarized by Hydro Compartment

This request returns elevation for a single year summarized by hydro compartment as an unpivoted (long format) Parquet file.

```
GET /v1/datasets/elev/annual/morph_pixel_v001/data.parquet?model_group_id=500&scenario_id=7&calendar_year=2050&summarize_by=hydro_compartment_v001&summarize_aggregate=mean&unpivot=true&sort=calendar_year,grid_cell
```

#### Conditional Pixel Count per Zone

This request counts, per hydro compartment, the pixels whose elevation exceeds a threshold in a single year.

```
GET /v1/datasets/elev/annual/morph_pixel_v001/data.csv?model_group_id=500&scenario_id=7&calendar_year=2050&where=value%20%3E%200&summarize_by=hydro_compartment_v001&summarize_aggregate=count
```

#### Categorical Count per Zone

This request counts, per hydro compartment, the pixels carrying each distinct land-type value in a single year. The output has one row per (zone, value) pair.

```
GET /v1/datasets/lndtyp/annual/morph_pixel_v001/data.csv?model_group_id=500&scenario_id=7&calendar_year=2050&summarize_by=hydro_compartment_v001&summarize_aggregate=count_by_value
```
