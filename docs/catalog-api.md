# Catalog API

The catalog API provides routes for querying what datasets and combinations of dimensions are available in the data and analysis API. It uses a hierarchy of dimensions to allow cascading selection of datasets, as in the MP-DAP. This hierarchy follows the order of dataset, partition, and non-partition dimensions outlined above.

Available choices come from a materialized catalog side-table (`cpra.mp.data.catalog`) rather than a walk of the data tree, so they cover tabular and raster datasets alike and cost one partition read per dataset. The side-table holds one row per distinct dimension combination × contiguous calendar run, and `write_parquet`/`write_cog` keep it in sync on every write.

Calendar dimensions are reported as `[start, end]` pairs per contiguous run rather than as every individual date; a run of one is reported as a bare value.

## Examples

### Get Available Datasets

```
GET /v1/datasets.csv
```

```json
plan_version,variable,geographic_unit,time_unit
2023.0.0,elev,morph_pixel,annual
2023.0.0,sal,hydro_compartment,daily
2023.0.0,stg,hydro_compartment,daily
2023.0.0,tmp,hydro_compartment,daily
2023.0.0,trg,hydro_compartment,daily
2023.0.0,tss,hydro_compartment,daily
...
```

### Convert Geographies Between Grids

```
GET /v1/grids/<source_grid>/crosswalk.<format>?grid_cell=<grid_cells>&to_grid=<grid>
```

Converts one or more geographies from a source grid into the equivalent geographies in one or more destination grids, using the same crosswalk files a pipeline `project` step reads. The result is a table in any of the tabular formats above (`csv`, `ndjson`, or `parquet`), always with the same columns and the same column types:

| Column | Type | Description |
|---|---|---|
| `src_id` | integer | The source geography's cell ID, for a vector source grid |
| `src_lng`, `src_lat` | float | The source geography's coordinate, for a raster or `wgs84` source grid |
| `dst_id` | integer | The destination geography's cell ID, for a vector destination grid |
| `dst_lng`, `dst_lat` | float | The destination geography's coordinate, for a raster destination grid |
| `scale_factor` | float | The share of the source geography that falls in the destination geography |
| `dst_grid` | string | The destination grid the row converts to |

A vector grid names a geography by ID and a raster grid names one by location, so each side of a row has a column for both spellings and only the one its grid uses is populated; the other is null. A row from a vector grid to a raster grid therefore fills in `src_id`, `dst_lng`, and `dst_lat`, and leaves `src_lng`, `src_lat`, and `dst_id` empty. Coordinates are WGS84 (EPSG:4326) on the way in and on the way out, as everywhere else in the API — never the grid's own projection. Either way, a value from the table can be passed straight back to the data API as a `grid_cell` parameter for the grid that names it.

| Parameter | Description | Example |
|---|---|---|
| `grid_cell` | Grid cells to convert, as IDs or as `lng,lat` coordinate pairs. Required for raster and `wgs84` source grids; for vector source grids, the whole crosswalk table is returned when it is omitted | `grid_cell=24,81` or `grid_cell=-90.49,30.01` |
| `to_grid` | Destination grid, repeatable. Defaults to every grid with a crosswalk file from the source grid, plus every other raster grid when the source grid is a raster grid, or every grid with geometry plus every raster grid for `wgs84` | `to_grid=ecoregion_v001&to_grid=parish_v2020` |

How a conversion is made depends on the two grids:

* **Vector to vector** uses the crosswalk Parquet table, returned as stored. A source geography split across several destinations keeps one row per destination, with the table's own scale factors.
* **Raster to vector** samples the crosswalk GeoTIFF at each coordinate.
* **Vector to raster** returns a point on the surface of the source geography's boundary, which is guaranteed to lie inside it even when the geography is concave or multi-part.
* **Raster to raster** returns the coordinates unchanged: a raster geography is the location itself, which every raster grid covers at its own resolution.
* **WGS84 to vector** finds the polygon containing each coordinate in the destination grid's geometry, and **WGS84 to raster** returns the coordinates unchanged, as raster to raster does.

Geographies with no counterpart in a destination grid are left out of the results rather than reported as empty rows, so the table can be shorter than the request. Asking for a destination grid that has no crosswalk from the source at all is an error.

```
GET /v1/grids/hydro_compartment_v001/crosswalk.csv?grid_cell=1,2&to_grid=ecoregion_v001&to_grid=morph_pixel_v001
```

```
src_id,src_lng,src_lat,dst_id,dst_lng,dst_lat,scale_factor,dst_grid
1,,,10,,,0.5,ecoregion_v001
1,,,11,,,0.5,ecoregion_v001
2,,,11,,,1.0,ecoregion_v001
1,,,,-90.70719027235607,30.04624414612512,1.0,morph_pixel_v001
2,,,,-90.80656477108946,30.105857483129597,1.0,morph_pixel_v001
```

#### Converting Coordinates

`wgs84` is a source grid that isn't a grid: it stands for the coordinates themselves, so converting from it converts the coordinates given rather than the geographies of some grid containing them. It is a source only — a destination coordinate is what a raster destination already reports — and it requires a `grid_cell`.

Its use is finding the cell of a vector grid that a location falls in, which it does with the same point-in-polygon lookup the data API's `grid_cell` filter makes. A raster source answers that question too, by sampling the crosswalk GeoTIFF, but from a rasterization of the destination grid's polygons: it places the boundaries within a pixel of where the polygons put them, and reports nothing at all where the raster doesn't reach. Converting from `wgs84` instead means a coordinate crosswalks to the same geography whose data the API selects for it.

Every coordinate gets a row of its own, naming the coordinate it converts in `src_lng` and `src_lat`, so coordinates sharing a geography are reported once each rather than merged. Coordinates outside every geography are left out, as unconverted geographies are everywhere else.

```
GET /v1/grids/wgs84/crosswalk.csv?grid_cell=-91.90375,29.7798,-91.90275,29.7798&to_grid=hydro_compartment_v001
```

```
src_id,src_lng,src_lat,dst_id,dst_lng,dst_lat,scale_factor,dst_grid
,-91.90375,29.7798,844,,,1.0,hydro_compartment_v001
,-91.90275,29.7798,844,,,1.0,hydro_compartment_v001
```

### Query a Request's Catalog

`GET /v1/catalog.json` takes the same parameter set as the data routes — `analysis=`, dotted pipeline parameters, dimension selections, `grid_cell`/`where` — and resolves the same pipeline they would. It answers two questions about that one request: what the caller can still steer (`choices`/`results`/`exact`), and what the request would produce (`output`).

```
GET /v1/catalog.json?variable=sal&time_unit=daily&grid=hydro_compartment_v001&model_group_id=500&scenario_id=7&calendar_day=2030-01-01
```

```json
{
  "choices": {
	"model_group_id": [500, 521],
	"scenario_id": [7, 8],
	"aggregate_type": ["mean"],
	"model": ["icm_v2023.0.0"],
	"calendar_day": [["2025-01-01", "2044-12-31"], ["2050-01-01", "2070-12-31"]]
  },
  "results": {
	"model_group_id": [500],
	"scenario_id": [7],
	"aggregate_type": ["mean"],
	"model": ["icm_v2023.0.0"],
	"calendar_day": ["2030-01-01"]
  },
  "exact": true,
  "output": {
	"grid": "hydro_compartment_v001",
	"grid_type": "vector",
	"shape": "long",
	"time_unit": "daily",
	"time_column": "calendar_day",
	"value_columns": ["value"],
	"variables": ["sal"],
	"label": "Salinity",
	"legend": "sal",
	"result_kind": "value",
	"collapsed_dimensions": []
  }
}
```

`choices` lists every value available for each dimension, `results` what the request selected (defaulting to the first choice where it selected nothing, or selected something unavailable), and `exact` is false when the two differ for any selectable dimension. Consecutive calendar values are compressed into `[start, end]` pairs.

The dataset triple is not reported in `choices`/`results`. It names *which* data to read rather than which slice of it, and a request that reads several datasets has no single triple to report. What a client needs from it — which grid to draw on, which time unit to step by, which legend and units to label with — is a fact about the result, and lives in `output`.

#### With an Analysis

```
GET /v1/catalog.json?variable=lnd_type&time_unit=annual&grid=morph_pixel_v002&analysis=lnd_chng&calendar_year=2050&model_group_id=500&model_group_id=520
```

`output` describes what the analysis produces rather than what it reads: `lnd_chng` outputs land-change categories, so `legend`, `label`, and `result_kind` are the analysis's, not the input variable's.

`collapsed_dimensions` names what the pipeline consumed — here `model_group_id`, whose two selected values this analysis's three-image mode folds into one code. A collapsed dimension is **absent** from `choices` and `results` rather than reported with a value the caller cannot change, and defaulting it does not make `exact` false. It is per-request rather than per-analysis: drop one of the two `model_group_id`s above and the same analysis collapses nothing, because two-image mode compares within a single model group.

`output.value_columns` and `output.variables` are parallel: zip them, and each pair is a column of the result and the variable whose units, legend and number format label it (null where none could be resolved). `output.folded_into` says how to read that first list. It is null for every pipeline without an [`unpivot`](analysis-api.md) step, and the names really are columns. When it is set — to the name of the label column that step folded into — the names are the **values** that column takes instead: the result carries one `value` column with the measurements stacked, one row per name. A client keying display metadata per measurement does the same thing either way, looking the measurement up by column when `folded_into` is null and by that column's value when it is not.

When a request reads several datasets, each dimension's choices are the values **common to all of them**: an analysis joins its inputs, so a `model_group_id` only one of them has would yield an empty result. A dimension with no value common to every input is a 404, as is a request naming a dataset nothing has been written for.
