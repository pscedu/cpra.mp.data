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

## Analysis Pipelines

The data endpoints above operate on one dataset at a time. Analysis pipelines let a request name **several** datasets, move them onto a common grid, join them, and filter/aggregate the result -- e.g. "wherever marsh coverage exceeds a threshold *and* salinity exceeds a threshold, what is the earliest year that became true?" A pipeline request is a small, named-step program rather than a set of query parameters, since expressing that kind of cross-dataset condition as flat parameters on a single dataset's endpoint has no natural spelling.

Three endpoints accept a pipeline, sharing one request model:

```
POST /v1/analysis.<format>
GET  /v1/analysis.<format>
GET  /v1/analysis/{z}/{x}/{y}.<png|jpg>
```

`POST` takes the pipeline as a JSON body. `GET` (both the flat and the tile form) takes the identical pipeline flattened into a dotted query string, so a pipeline can be embedded in a map tile URL template. In every case, `<format>` in the URL must agree with the pipeline's own `output.format` -- a mismatch (e.g. a `.csv` URL for a pipeline whose `output.format` is `"tif"`) is rejected with a 400 before anything is read.

### Request Shape

A pipeline has three parts:

```json
{
  "inputs": {
    "<name>": {"variable": "...", "time_unit": "...", "grid": "...", "<dimension>": "..."}
  },
  "steps": [ {"op": "...", ...} ],
  "output": {"format": "...", "color_map": "..."}
}
```

**`inputs`** names one or more dataset selections. Each is the same `variable`/`time_unit`/`grid` triple the flat data endpoints take, plus any number of dimension filters (e.g. `model_group_id`, `scenario_id`) as extra keys -- a scalar or a comma-separated/JSON-list of values, exactly as on the flat endpoints.

**`steps`** is an ordered list of operations, each consuming a named input or a previous step's own output:

| Op | Fields | Description |
|---|---|---|
| `project` | `input` (optional, chains from the previous step if omitted), `to` (target grid), `aggregate` (required moving to a coarser grid, forbidden moving to a finer one), `as` | Move a named input onto a different grid via a crosswalk. |
| `join` | `type` (`"spatial"`), `inputs` (two or more names), `as` | Combine named inputs/step outputs on shared dimensions and `grid_cell`. |
| `filter` | `where` | A SQL-style predicate over the in-scope named columns (e.g. `marsh > 0.5 AND sal > 5.5`), always over the previous step's output. |
| `group` | `by` (dimensions to group by; `[]` collapses to one value per `grid_cell`), `aggregate`, `of` (defaults to `"value"`; a numeric dimension such as `calendar_year` is also valid, for "first year a condition held") | Collapse rows/bands, always over the previous step's output. |

`filter` and `group` never take an explicit input -- they always consume whatever the immediately preceding step produced -- so the **first** step of every pipeline must be a `project` (naming its input explicitly) or a `join` (naming two or more inputs explicitly). A single-input pipeline that changes no grid at all still opens with an identity `project` (`to` equal to the input's own grid) purely to satisfy this rule.

**`output`** gives the response `format` (any tabular/`tif` format, or a tile format for the tile route) and an optional `color_map` for rendered raster output.

### What Compiles Today

Not every step sequence a pipeline could describe is executable yet. Three shapes are supported; anything else is rejected with a `PipelineError`-derived 400 naming the specific unsupported shape and why, rather than a generic failure. The response's `detail` text is intended to be actionable, e.g. it names the exact rule that failed (a cardinality direction, a row ceiling, a missing `group` step) rather than just "invalid pipeline."

1. **Single-input chains equivalent to a flat dataset request**: one input, an opening `project` (identity, or many-to-one onto a coarser grid with `aggregate`), then optionally `filter` step(s) and at most one trailing `group`, to any tabular or `tif`/tile format. This compiles onto the same machinery the flat data endpoints use (`project`'s `to`/`aggregate` become `summarize_by`/`summarize_aggregate`; `filter.where` and `group` become `where`/`group_by`/`group_aggregate`) -- a single-input pipeline and the equivalent flat request return the same data.
2. **Project-only reaggregation**: the same shape as (1), specialized to a many-to-one `project` (e.g. raster pixels reaggregated onto a coarser vector zone) with a tabular output -- no `filter`/`group` needed.
3. **The paired (spatially-joined) shape**: exactly two inputs -- a coarser one broadcast (one-to-many `project`, no `aggregate`) onto a finer *raster* grid, immediately `join`ed against that raster input, then any number of `filter` steps and exactly one trailing `group` with `by: []` (a per-pixel collapse across bands, e.g. across years -- not a zonal collapse) and a numeric `of` (`calendar_year`, or the running `value`) for every aggregate except `count`, which has none (see below). Because this shape's result is inherently one raster surface, only `tif`/`png`/`jpg` output is implemented for it; a tabular paired output is rejected outright (materializing every joined pixel row into a table is exactly the memory cost this shape's execution plan avoids -- see "Memory Behavior" below). This shape's `group.aggregate` supports `min`, `max`, `sum`, `mean`, and `count`: `count` reports, per pixel, how many bands (e.g. calendar steps) satisfy the preceding `filter` step(s)' `where` clause (or, with no `filter` at all, how many bands have real fine+paired data) -- it has no aggregand, so `of` must be left at its default (omitted) rather than naming a column. A pixel with zero satisfying bands reports a real `0`, not a missing value (consistent with `count` everywhere else in this API); nodata is reserved for a pixel entirely outside the pairing crosswalk's domain.

A `join`'s two sides must agree on every dimension they share (present on all joined inputs and filtered to the same value(s)), and a dimension present on only one side must be pinned to a single value there, since there is no row on the other side to align it against. The coarse/zone side of a `project`/`join` is also capped at a configured row ceiling (`ANALYSIS_JOIN_MAX_ROWS`), checked before that side is materialized, so an unbounded selection is rejected rather than silently read in full.

Everything else -- three or more inputs, a `join` not immediately preceded by its broadcasting `project`, a tabular output for the paired shape, `filter`/`group` in an order the compiler doesn't recognize, and so on -- is rejected informatively rather than silently misexecuted.

### Memory Behavior

The paired shape (3) exists specifically to avoid materializing a full joined table: naively broadcasting a coarse dataset onto every pixel of a fine raster grid, across every year, and then grouping, means holding one column per year for every pixel in memory at once before the aggregate can drop most of it. Instead, the compiled plan reads both inputs' small ROW metadata up front (file paths/bands for the raster input; the already row-ceiling-bounded coarse table), then executes as a single pass over the raster's own block windows: for each window, it reads that window from every year's band plus the pairing crosswalk, looks up each pixel's paired coarse value, applies the `filter`/`group` collapse for just that window, and writes the collapsed window straight back at its own position -- never holding a full-extent array of any per-pixel intermediate.

### Examples

#### The Exit-Criterion Query: Earliest Year Marsh and Salinity Both Cross a Threshold

This pipeline broadcasts annual salinity (by hydro compartment) onto the vegetation grid, pairs it against marsh coverage there, keeps only pixel-years where marsh coverage exceeds 0.5 *and* paired salinity exceeds 5.5, and reports the earliest (`min`) such `calendar_year` per pixel, as a GeoTIFF:

```json
POST /v1/analysis.tif
{
  "inputs": {
    "marsh": {
      "variable": "ffibs_coverage_ratio", "time_unit": "annual", "grid": "veg_grid_cell_v001",
      "model_group_id": 500, "scenario_id": 7, "ffibs_type": "pl_fm"
    },
    "salinity": {
      "variable": "sal", "time_unit": "annual", "grid": "hydro_compartment_v001",
      "model_group_id": 500, "scenario_id": 7, "aggregate_type": "2_wk_max"
    }
  },
  "steps": [
    {"op": "project", "input": "salinity", "to": "veg_grid_cell_v001", "as": "sal"},
    {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
    {"op": "filter", "where": "marsh > 0.5 AND sal > 5.5"},
    {"op": "group", "by": [], "aggregate": "min", "of": "calendar_year"}
  ],
  "output": {"format": "tif"}
}
```

The same pipeline as a `GET` request, using the dotted query encoding (dot = nesting, an integer segment = a 1-based list position, comma = a list value -- never inside `where`, which is one opaque string):

```
GET /v1/analysis.tif?input.marsh.variable=ffibs_coverage_ratio&input.marsh.time_unit=annual&input.marsh.grid=veg_grid_cell_v001&input.marsh.model_group_id=500&input.marsh.scenario_id=7&input.marsh.ffibs_type=pl_fm&input.salinity.variable=sal&input.salinity.time_unit=annual&input.salinity.grid=hydro_compartment_v001&input.salinity.model_group_id=500&input.salinity.scenario_id=7&input.salinity.aggregate_type=2_wk_max&step.1.op=project&step.1.input=salinity&step.1.to=veg_grid_cell_v001&step.1.as=sal&step.2.op=join&step.2.type=spatial&step.2.inputs=marsh,sal&step.2.as=paired&step.3.op=filter&step.3.where=marsh%20%3E%200.5%20AND%20sal%20%3E%205.5&step.4.op=group&step.4.by=&step.4.aggregate=min&step.4.of=calendar_year&output.format=tif
```

And as map tiles, replacing `output.format=tif` with `output.format=png`:

```
GET /v1/analysis/{z}/{x}/{y}.png?...same query as above, with output.format=png
```

#### Single-Input Pipeline Equivalent to a Flat Request

An identity `project` (no grid change), a `filter`, and a `group`, equivalent to the same query and aggregate on the flat `sal`/`hydro_compartment_v001` endpoint:

```json
POST /v1/analysis.csv
{
  "inputs": {
    "sal": {"variable": "sal", "time_unit": "daily", "grid": "hydro_compartment_v001", "model_group_id": 520, "scenario_id": 7}
  },
  "steps": [
    {"op": "project", "input": "sal", "to": "hydro_compartment_v001"},
    {"op": "filter", "where": "sal > 5"},
    {"op": "group", "by": ["calendar_day"], "aggregate": "max"}
  ],
  "output": {"format": "csv"}
}
```

#### Project-Only Reaggregation

Reaggregate raster elevation onto hydro compartments as a zonal mean, with no `filter`/`group` needed:

```json
POST /v1/analysis.csv
{
  "inputs": {
    "elev": {"variable": "elev", "time_unit": "annual", "grid": "morph_pixel_v001", "model_group_id": 500, "scenario_id": 7, "calendar_year": 2050}
  },
  "steps": [
    {"op": "project", "input": "elev", "to": "hydro_compartment_v001", "aggregate": "mean"}
  ],
  "output": {"format": "csv"}
}
```
