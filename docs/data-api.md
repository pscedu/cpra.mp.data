# Data API

The data API provides a RESTful interface for accessing (never transforming) model data
stored in Parquet and COG files. Data from Parquet files can be returned in several
tabular formats. Data from COGs can be returned as GeoTIFFs or raster map tiles.
Vector-grid data can also be rendered as vector map tiles.

A request against these endpoints only ever **selects and narrows** stored values --
filtering to a subset, sorting/selecting columns, and (for raster) applying a color map
for rendering. It never transforms, aggregates, or joins data across datasets; for that,
see the [Analysis API](analysis-api.md).

The same engine is available as a Python library for code running with direct filesystem
access to the data (notebooks, scripts): `read_data` accepts the same filter parameter
names documented here, with no HTTP involved. See the
[README's Analyzing Data section](../README.md#analyzing-data) for usage.

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

Tabular formats are `csv`, `ndjson`, and `parquet`. Raster output is `tif` (GeoTIFF);
raster map tiles are `png` or `jpg`; vector map tiles are Mapbox vector tiles (`pbf`).
Tabular and GeoTIFF responses accept `compression_algorithm` and `compression_level`
parameters (the accepted algorithms differ by format; `zstd` level 6 is the default).

## Filter

Filter parameters limit the rows, columns, or bands of the dataset to those matching
specific criteria. Any dimension from the partition and non-partition dimensions can be
used as a filter parameter. Multiple values can be provided in a comma-separated list.
Note that special characters in SQL-style parameters should be
[percent-encoded](https://developer.mozilla.org/en-US/docs/Glossary/Percent-encoding)
(e.g., using
[`encodeURIComponent`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/encodeURIComponent)).

For vector-grid data, the `grid_cell` parameter accepts a comma-separated list of source
grid cell IDs. For raster-grid data, it accepts a comma-separated list of coordinate pairs
(in the format `x,y`) at which values are extracted from the COGs, producing a tabular
result.

Vector-grid data also accepts coordinate pairs, so the same `grid_cell` value can be used
against any grid without knowing its type. The list is read as grid cell IDs only when
every value is a plain unsigned integer; otherwise it is read as coordinate pairs, and
each pair is resolved to the grid cell whose boundary contains it. Results are always
reported under the grid cell ID, never the coordinate pair that selected it. A coordinate
pair that falls outside every cell in the grid is omitted from the results, and several
pairs that fall inside the same cell report it once.

For raster-grid data, when the `grid_cell` parameter is not provided, the full images are
used: each COG band matching the filters becomes a band in the resulting image.

A single SQL-style `where` parameter accepts two kinds of conditions, joined with `AND`:

- **Dimension** conditions, against dimension columns, using SQL where-clause syntax
  without restriction on the expression (aside from the aggregate/subquery rule below).
- **Value** conditions, against the data values themselves, written using the placeholder
  column `value` (e.g. `value > 3`). A value condition is parsed against a restricted
  whitelist of comparison operators and elementwise functions. What a failing value
  condition *does* depends on the grid: on a **vector** grid the row is dropped, since a
  row carries exactly one measurement and there is nothing else in it to preserve; on a
  **raster** grid the pixel is masked to nodata **in that band alone**, because dropping
  its row would drop the same pixel from every other band too, including the bands the
  condition holds for.

Every `AND`-joined term in `where` must be entirely one kind or the other -- a single
comparison cannot mix a dimension column and `value` (e.g. `value > calendar_year` is
rejected; write them as two separate `AND`-ed conditions instead, like
`calendar_year > 2040 AND value > 3`). Whichever kind a term is, an aggregate function
call or a subquery inside it is rejected before any data is read, because either would
force the whole dataset to be materialized to evaluate what was supposed to be a per-row
test.

| Parameter | Description | Vector-Grid | Raster-Grid | Example |
|---|---|---|---|---|
| Partition and Non-Partition Dimensions | Any dimension from the partition and non-partition tables above | ✅ | ✅ | `aep=0.2` |
| `grid_cell` | Grid cells to include, as IDs (vector grids only) or as `x,y` coordinate pairs | ✅ | ✅ | `grid_cell=24,81` or `grid_cell=-90.49,30.01` |
| `where` | Dimension and/or value conditions, in SQL where-clause format; a failing value condition drops the row on a vector grid and masks the pixel to nodata on a raster grid | ✅ | ✅ | `where=calendar_year%3E2040` or `where=value%20%3E%203` |

## Sort, Select, and Color Map

The `sort` parameter sorts the final results. It accepts multiple dimensions in a
comma-separated list. Dimensions are sorted in ascending order by default unless prefixed
with a minus sign (`-`) for descending order. The `select` parameter limits the returned
columns to the provided comma-separated list. The `color_map` parameter applies a named
color map to rendered raster output (tiles or GeoTIFF).

| Parameter | Description | Vector-Grid | Raster-Grid | Example |
|---|---|---|---|---|
| `sort` | Sort order | ✅ | ✅ (tabular output) | `sort=scenario_id,model_group_id,calendar_day` |
| `select` | Columns to return | ✅ | ✅ (tabular output) | `select=grid_cell,calendar_year,value` |
| `color_map` | Apply the given color map to raster output | ❌ | ✅ | `color_map=elevation` |

## Examples

### Vector-Grid Data

#### Basic Map Query

This request returns daily mean salinity data for all grid cells on a specific date as
CSV.

```
GET /v1/datasets/sal/daily/hydro_compartment_v001/data.csv?model_group_id=520&scenario_id=7&calendar_day=2030-01-01
```

#### Basic Time Series Query

This request returns daily time series data for three grid cells as newline-delimited
JSON.

```
GET /v1/datasets/sal/daily/hydro_compartment_v001/data.ndjson?model_group_id=520&scenario_id=7&grid_cell=79,842,1003&sort=calendar_day
```

### Raster-Grid Data

#### Basic Map Query

This request returns colorized elevation data for a single year as PNG map tiles.

```
GET /v1/datasets/elev/annual/morph_pixel_v001/data/{z}/{x}/{y}.png?model_group_id=500&scenario_id=7&calendar_year=2045&color_map=elevation
```

#### Time Series Query at Specific Points

This request returns an elevation time series at three longitude-latitude pairs as CSV.
Data is already served in long format (one `grid_cell` column and one `value` column, not
one column per grid cell), so no `unpivot` parameter is needed:

```
GET /v1/datasets/elev/annual/morph_pixel_v001/data.csv?model_group_id=500&scenario_id=7&grid_cell=-90.4914181,30.0144111,-90.8628217,29.4254702,-91.3641637,29.5319325&sort=calendar_year,grid_cell
```

## Migration from Removed Parameters

Several parameters that used to describe how a request *transforms* data --
`calculate`/`negate`/`multiply10`/`multiply100`, `group_by`/`group_aggregate`,
`summarize_by`/`summarize_aggregate`/`summarize_grid_cell`, and the already-deprecated
`value_where` -- are no longer accepted on these data endpoints. Sending one is rejected
with a 400 naming the exact parameter and its replacement, rather than being silently
ignored. `unpivot` is also rejected, though it was already a no-op (data is served in long
format regardless).

Every one of these has a pipeline equivalent on the [Analysis API](analysis-api.md):

| Removed Parameter | Replacement |
|---|---|
| `value_where` | Fold the condition into `where`, which now accepts both dimension and value conditions in one string (e.g. `calendar_year > 2040 AND value > 3`) |
| `negate` | A `difference` named analysis (see [the registry doc](analyses-api.md)), or a pipeline `case` step directly: `negate=X` is one arm `{"when": X, "then": "value * -1"}` with `"else": "value"` for the rows it doesn't match |
| `multiply10` | A pipeline `case` step: `multiply10=X` is one arm `{"when": X, "then": "value * 10"}` with `"else": "value"` |
| `multiply100` | A pipeline `case` step: `multiply100=X` is one arm `{"when": X, "then": "value * 100"}` with `"else": "value"` |
| `calculate.N.expression` / `calculate.N.where` | A pipeline `calculate` step for the unconditional form; where a `where` was given, one `case` step whose arms are the N `where`/`expression` pairs in order, plus `"else": "value"` |
| `group_by` / `group_aggregate` | A pipeline `group` step |
| `summarize_by` / `summarize_aggregate` | A pipeline `project` step's `to`/`aggregate` fields |
| `summarize_grid_cell` | A pipeline `filter` step naming `grid_cell`, placed after the `project` that produced the zones (a `grid_cell` filter before the `project` selects source cells instead) |
| `unpivot` | Nothing -- data is already served in long format (`grid_cell`/`value` columns); this was already a documented no-op |

See [Analysis API](analysis-api.md) for the full pipeline grammar, and
[the named-analysis registry](analyses-api.md) for pre-built pipelines like `difference`
that need no hand-authored pipeline JSON at all.
