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

### Query a Dataset Catalog

```
GET /v1/datasets/2029.1.0/sal/hydro_compartment/daily/catalog.json?model_group_id=500&scenario_id=7&calendar_date=2030-01-01
```

```json
{
  "choices": {
	"variable": ["annual_exceedance_damages_dollars", "expected_annual_damages_dollars", ...],
	"geographic_unit": ["extraction_point", "hydro_compartment"],
	"time_unit": ["annual", "daily"],
    "plan_version": ["2023.0.2", "2029.1.0"],
	"model_group": [500, 521],
	"scenario": [7, 8],
	"calendar_date": [["2025-01-01", "2044-12-31"], ["2050-01-01", "2070-12-31"]]
  },
  "dataset": {
	"metric": "salinity_mean",
	"geographic_unit": "hydro_compartment",
	"time_unit": "daily",
	"model_group": 500,
	"scenario": 7,
	"calendar_date": "2030-01-01"
  },
  "exactMatch": true
}
```
