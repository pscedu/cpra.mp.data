# Data and Analysis API

The data and analysis API provides a RESTful interface for accessing and transforming model data stored in Parquet and COG files. Data from Parquet files can be returned in several tabular formats. Data from COGs can be returned as GeoTIFFs or raster map tiles. COG data can also be summarized geographic unit (e.g., hydro compartment or ecoregion) and returned in any of the formats used for the tabular data.

All of the API endpoints use the general format:

```
/<api_version>/datasets/<variable>/<geographic_unit>/<time_unit>/data.<format>
```

Example URIs for vector-grid (Parquet) data are:

```
/v1/datasets/sal/hydro_compartment/daily/data.arrow
/v1/datasets/sal/hydro_compartment/daily/data.csv
/v1/datasets/sal/hydro_compartment/daily/data.ndjson
/v1/datasets/sal/hydro_compartment/daily/data.parquet
```

Additional example URIs for raster-grid (COG) data are:

```
/v1/datasets/inun/morph_pixel/annual/data.tif
/v1/datasets/inun/morph_pixel/annual/{z}/{x}/{y}/data.png
```

## Query Operations

Without any query string parameters, each endpoint returns the full dataset by default. To obtain a subset of the data or to perform transformation and analysis operations, query string parameters can be added. There are six query operations, which are always performed in the same order:

| Order | Operation | Vector-Grid Status | Raster-Grid Status |
|---|---|---|---|
| 1 | [Filter](#filter) | ✅ Implemented | ✅ Implemented |
| 2 | [Calculate](#calculate) | ✅ Implemented | ❌ Not Implemented |
| 3 | [Group and Aggregate](#group-and-aggregate) | ✅ Implemented | ❌ Not Implemented |
| 4 | [Transform and Extract](#transform-and-extract) | ✅ Implemented | ❌ Not Implemented |
| 5 | [Sort](#sort) | ✅ Implemented | ❌ Not Implemented |

### Filter

Filter parameters are used to limit the rows, columns, or bands of the dataset to those matching specific criteria. Any dimension from the partition and non-partition dimensions can be used as a filter parameter. Multiple values can be provided in a comma-separated list. For advanced filtering of vector-grid data, a `where` parameter can be used to specify complex filtering logic in SQL where-clause format. Note that special characters in the where clause should be [percent-encoded](https://developer.mozilla.org/en-US/docs/Glossary/Percent-encoding) (e.g., using [`encodeURIComponent`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/encodeURIComponent)).

For vector-grid data, the `geography` parameter accepts a comma-separated list of source geography IDs. For raster-grid data, it accepts a comma-separated list of coordinate pairs (in the format `x,y`) at which values are extracted from the COGs. When specifying coordinate pairs for a raster-grid dataset, the result is a tabular dataset that can be transformed in the same way as Parquet data (e.g., using the `melt` parameter below).

For raster-grid data, when the `geography` parameter is not provided, the full images are used. Each COG becomes a band in the resulting image. These bands can be aggregated using the `group_by` and `group_aggregate` parameters below. Due to memory constrains, a maximum of 10 bands can be loaded from COGs. Additional matches beyond the first 10 COGs are ignored.

| Parameter | Description | Vector-Grid | Raster-Grid | Example |
|---|---|---|---|---|
| Partition and Non-Partition Dimensions | Any dimension from the partition and non-partition tables above | ✅ | ✅ | `aep=0.2` |
| `geography` | Geographies to include | ✅ | ✅ | `geography=24,81` |
| `where` | Advanced filters in SQL where-clause format | ✅ | ✅ | `where=calendar_year%3E2040` |

### Calculate

Calculate parameters change the values of individual table cells or raster pixels prior to aggregation. They are intended to support comparison operations, such as calculating the difference between two scenarios or concatenating digits to highlight changes. See the examples below for specific applications of these parameters.

| Parameter | Description | Vector-Grid | Raster-Grid | Example |
|---|---|---|---|---|
| `negate` | SQL expression for rows/bands where the value should be multiplied by -1 | ✅ | ✅ | `negate=scenario_id%3D8` |
| `multiply10` | SQL expression for rows/bands where the value should be multiplied by 10 | ✅ | ✅ | `multiply10=scenario_id%3D8%20AND%20calendar_year%3D2030` |
| `multiply100` | SQL expression for rows/bands where the value should be multiplied by 100 | ✅ | ✅ | `multiply100=scenario_id%3D8%20AND%20calendar_year%3D2040` |

### Group and Aggregate

Group and aggregate parameters allow multiple rows or bands to be summarized as a single row or band. Unlike the geographic aggregations in the transformation step below, these aggregations operate on a single vector geography or raster pixel. For differences, use `group_aggregate=sum` with the `negate` parameter.

| Parameter | Description | Vector-Grid | Raster-Grid | Example |
|---|---|---|---|---|
| `group_by` | Group by dimensions | ✅ | ✅ | `group_by=model_group,scenario,aep` |
| `group_aggregate` | Aggregate function: one of `sum`, `max`, `min`, or `mean` | ✅ | ✅ | `group_aggregate=sum` |


### Transform and Extract

Transform and extract parameters can be used to change the shape or format of the resulting dataset, summarize data at higher geographic levels, or apply color maps to raster-grid data.

| Parameter | Description | Vector-Grid | Raster-Grid | Example |
|---|---|---|---|---|
| `color_map` | Apply the given color map to raster data | ❌ | ✅ | `color_map=ffibs` |
| `melt` | Melt results so that the resulting table has a geography column and a value column instead of a separate column for each geography | ✅ | ❌ | `melt=true` |
| `summarize_by` | Summarize by geographic unit | ✅ | ✅ | `summarize_by=hydro_compartment` |
| `summarize_aggregate` | Aggregate function: one of `sum`, `difference`, `max`, `min`, or `mean` | ✅ | ✅ | `summarize_aggregate=max` |
| `summarize_geography` | Limit the summary geographies to the provided IDs | ✅ | ✅ | `summarize_geography=659,680,700` |

### Sort

The sort parameter sorts the final results. The parameter can accept multiple dimensions in a comma-separated list. Dimensions are sorted in ascending order by default unless prefixed with a minus sign (`-`) for descending order.

| Parameter | Description | Vector-Grid | Raster-Grid | Example |
|---|---|---|---|---|
| `sort` | Sort order | ✅ | ✅ | `sort=scenario,model_group,calendar_date` |


## Examples

The following examples illustrate common patterns of use for the data and analysis API.

### Vector-Grid Data

#### Basic Map Query

This request returns daily mean salinity data for all extraction points on a specific date in the Arrow IPC format.

```
GET /data/v1/salinity_mean/extraction_point/daily.arrow?model_group=500&scenario=7&calendar_date=2030-01-01
```

#### Map Query Comparing Two Model Groups

This request returns the difference in daily mean salinity for two model groups on the same date in the Arrow IPC format.

```
GET /data/v1/salinity_mean/extraction_point/daily.arrow?scenario_id=7&calendar_day=2030-01-01&where=model_group_id%20in%20(500%2C%20501)&negate=model_group_id=500&group_by=calendar_date&group_aggregate=sum
```

#### Basic Time Series Query

This request returns daily time series data for three extraction points in the arrow IPC format.

```
GET /v1/datasets/salinity_mean/extraction_point/daily/data.arrow?model_group=500&scenario=7&geography=79,842,1003&sort=calendar_date
```

#### Query with Additional Dimensions

This request returns annual exceedance damage, dollars (AEDD) for a single decade as newline-delimited JSON. Since the `aep`, `asset_type`, and `percentile` parameters are not provided, the query matches all available values for these dimensions. The results are summed across `asset_type` to provide the total damage for each AEP-percentile combination.

```
GET /v1/datasets/annual_exceedance_damages_dollars/clara/decadal/data.ndjson?model_group=500&scenario=7&pumping_id=0&fragility_scenario=0geography=12&group_by=aep,percentile&group_aggregate=sum&sort=aep,percentile
```

### Raster-Grid Data

#### Basic Map Query

This request returns colorized FFIBS data for a single year as PNG map tiles.

```
GET /v1/datasets/vegetation_type_ffibs/icm_raster/annual/{z}/{x}/{y}/data.png?model_group=500&scenario=7&calendar_year=2045&color_map=ffibs
```

#### Map Query Comparing Two Years

This request returns the colorized flood depth difference between two years as PNG map tiles.

```
GET /v1/datasets/flood_depth/icm_raster/annual/{z}/{x}/{y}/data.png?model_group=500&scenario=7&calendar_year=2045,2050&presort=-calendar_year&group_by=true&group_aggregate=difference&color_map=flood_depth_difference
```

#### Time Series Query at Specific Points

This request returns a flood depth time series at three longitude-latitude pairs as a melted (long format) CSV.

```
GET /v1/datasets/flood_depth/icm_raster/annual/data.csv?model_group=500&scenario=7&geography=-90.4914181,30.0144111,-90.8628217,29.4254702,-91.3641637,29.5319325&melt=true&sort=calendar_year,geography
```

#### Single Year Query Summarized by Hydro Compartment

This request returns a flood depths for a single year summarized by hydro compartment as a melted (long format) Parquet file.

```
GET /v1/datasets/flood_depth/icm_raster/annual/data.parquet?model_group=500&scenario=7&calendar_year=2050&summarize_by=hydro_compartment&summarize_aggregate=mean&melt=true&sort=calendar_year,geography
```
