# Catalog API

The catalog API provides routes for querying what datasets and combinations of dimensions are available in the data and analysis API. It uses a hierarchy of dimensions to allow cascading selection of datasets, as in the MP-DAP. This hierarchy follows the order of dataset, partition, and non-partition dimensions outlined above.

For dataset and partition dimensions, the catalog uses the data directory tree to determine the available choices for each dimension. For non-partition dimensions, it use the Parquet data and COG file names to determine the available choices.

🚧 **TODO:** We still need to finalize how to represent `calendar_date` ranges in the response without specifying each individual date. The syntax in the example below is one possibility.

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
GET /v1/grids/<source_grid>/crosswalk.<format>?geography=<geographies>&to_grid=<grid>
```

Converts one or more geographies from a source grid into the equivalent geographies in one or more destination grids, using the crosswalk files that back the data API's `summarize_by` parameter. The result is a table in any of the tabular formats above (`csv`, `ndjson`, or `parquet`), always with the same columns and the same column types:

| Column | Type | Description |
|---|---|---|
| `src_id` | integer | The source geography's cell ID, for a vector source grid |
| `src_lng`, `src_lat` | float | The source geography's coordinate, for a raster or `wgs84` source grid |
| `dst_id` | integer | The destination geography's cell ID, for a vector destination grid |
| `dst_lng`, `dst_lat` | float | The destination geography's coordinate, for a raster destination grid |
| `scale_factor` | float | The share of the source geography that falls in the destination geography |
| `dst_grid` | string | The destination grid the row converts to |

A vector grid names a geography by ID and a raster grid names one by location, so each side of a row has a column for both spellings and only the one its grid uses is populated; the other is null. A row from a vector grid to a raster grid therefore fills in `src_id`, `dst_lng`, and `dst_lat`, and leaves `src_lng`, `src_lat`, and `dst_id` empty. Coordinates are WGS84 (EPSG:4326) on the way in and on the way out, as everywhere else in the API — never the grid's own projection. Either way, a value from the table can be passed straight back to the data API as a `geography` parameter for the grid that names it.

| Parameter | Description | Example |
|---|---|---|
| `geography` | Geographies to convert, as IDs or as `lng,lat` coordinate pairs. Required for raster and `wgs84` source grids; for vector source grids, the whole crosswalk table is returned when it is omitted | `geography=24,81` or `geography=-90.49,30.01` |
| `to_grid` | Destination grid, repeatable. Defaults to every grid with a crosswalk file from the source grid, plus every other raster grid when the source grid is a raster grid, or every grid with geometry plus every raster grid for `wgs84` | `to_grid=ecoregion_v001&to_grid=parish_v2020` |

How a conversion is made depends on the two grids:

* **Vector to vector** uses the crosswalk Parquet table, returned as stored. A source geography split across several destinations keeps one row per destination, with the table's own scale factors.
* **Raster to vector** samples the crosswalk GeoTIFF at each coordinate.
* **Vector to raster** returns a point on the surface of the source geography's boundary, which is guaranteed to lie inside it even when the geography is concave or multi-part.
* **Raster to raster** returns the coordinates unchanged: a raster geography is the location itself, which every raster grid covers at its own resolution.
* **WGS84 to vector** finds the polygon containing each coordinate in the destination grid's geometry, and **WGS84 to raster** returns the coordinates unchanged, as raster to raster does.

Geographies with no counterpart in a destination grid are left out of the results rather than reported as empty rows, so the table can be shorter than the request. Asking for a destination grid that has no crosswalk from the source at all is an error.

```
GET /v1/grids/hydro_compartment_v001/crosswalk.csv?geography=1,2&to_grid=ecoregion_v001&to_grid=morph_pixel_v001
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

`wgs84` is a source grid that isn't a grid: it stands for the coordinates themselves, so converting from it converts the coordinates given rather than the geographies of some grid containing them. It is a source only — a destination coordinate is what a raster destination already reports — and it requires a `geography`.

Its use is finding the cell of a vector grid that a location falls in, which it does with the same point-in-polygon lookup the data API's `geography` filter makes. A raster source answers that question too, by sampling the crosswalk GeoTIFF, but from a rasterization of the destination grid's polygons: it places the boundaries within a pixel of where the polygons put them, and reports nothing at all where the raster doesn't reach. Converting from `wgs84` instead means a coordinate crosswalks to the same geography whose data the API selects for it.

Every coordinate gets a row of its own, naming the coordinate it converts in `src_lng` and `src_lat`, so coordinates sharing a geography are reported once each rather than merged. Coordinates outside every geography are left out, as unconverted geographies are everywhere else.

```
GET /v1/grids/wgs84/crosswalk.csv?geography=-91.90375,29.7798,-91.90275,29.7798&to_grid=hydro_compartment_v001
```

```
src_id,src_lng,src_lat,dst_id,dst_lng,dst_lat,scale_factor,dst_grid
,-91.90375,29.7798,844,,,1.0,hydro_compartment_v001
,-91.90275,29.7798,844,,,1.0,hydro_compartment_v001
```

### Query a Dataset Catalog

```
GET /v1/datasets/2029.1.0/sal/hydro_compartment/daily/catalog.json?model_group_id=500&scenario_id=7&calendar_date=2030-01-01
```

```json
{
  "choices": {
	"plan_version": ["2023.0.2", "2029.1.0"],
	"variable": ["aedd", "eadd", ...],
	"geographic_unit": ["hydro_compartment"],
	"time_unit": ["annual", "daily"],
	"model_group": [500, 521],
	"scenario": [7, 8],
	"calendar_date": [["2025-01-01", "2044-12-31"], ["2050-01-01", "2070-12-31"]]
  },
  "dataset": {
	"metric": "sal",
	"geographic_unit": "hydro_compartment",
	"time_unit": "daily",
	"model_group": 500,
	"scenario": 7,
	"calendar_date": "2030-01-01"
  }
}
```
