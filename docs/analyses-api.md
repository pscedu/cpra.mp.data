# Named Analyses (Registry API)

A named analysis is a pre-built [pipeline](analysis-api.md) with its own request-shaped
parameters -- a `difference` between two scenarios, a `concatenate` land-type-change
code, a `salinity-veg-investigation` earliest-crossing-year surface -- registered under a
short name so a caller never has to hand-author the equivalent pipeline JSON. Every named
analysis is **sugar over the same pipeline grammar** documented in the
[Analysis API](analysis-api.md): resolving one always produces a real `Pipeline`, run
through exactly the same compiler any hand-authored pipeline uses. Nothing is reachable
only through a named analysis, and `pipeline.json` (below) lets a caller see the exact
pipeline any request would run.

## Catalog

```
GET /v1/analyses.json
```

Returns the full list of registered analyses, each with:

| Field | Description |
|---|---|
| `name` | The analysis's route name (e.g. `difference`, `concatenate`, `salinity-veg-investigation`) |
| `label` | A human-readable display label |
| `type` | `"dataset"` (operates on one caller-chosen dataset) or `"custom"` (its inputs are fixed by the analysis itself) |
| `variables` | The variable(s) it's meaningful for, or `null` for any |
| `legend` | A named legend/color-map hint for rendered output, or `null` |
| `result_kind` | `"value"`, `"difference"`, or `"category"` -- a presentation hint for how to label the result |

As of this writing, four analyses are registered: `difference`, `concatenate` and
`land-type-confidence` (`type="dataset"`), and `salinity-veg-investigation`
(`type="custom"`).

## Dataset-Scoped vs. Custom Routes

A `type="dataset"` analysis (`difference`, `concatenate`) runs against a caller-chosen
dataset, so its routes are nested under that dataset the same way the flat Data API's
routes are:

```
GET /v1/datasets/{variable}/{time_unit}/{grid}/analyses/{analysis_name}/data.{format}
GET /v1/datasets/{variable}/{time_unit}/{grid}/analyses/{analysis_name}/data/{z}/{x}/{y}.{format}
GET /v1/datasets/{variable}/{time_unit}/{grid}/analyses/{analysis_name}/tiles/{z}/{x}/{y}.pbf
GET /v1/datasets/{variable}/{time_unit}/{grid}/analyses/{analysis_name}/pipeline.json
```

A `type="custom"` analysis (`salinity-veg-investigation`) fixes its own inputs inside
its `pipeline()` method -- it isn't scoped to any one caller-chosen dataset, so its
routes have no `/v1/datasets/...` prefix at all:

```
GET /v1/analyses/{analysis_name}/data.{format}
GET /v1/analyses/{analysis_name}/data/{z}/{x}/{y}.{format}
GET /v1/analyses/{analysis_name}/tiles/{z}/{x}/{y}.pbf
GET /v1/analyses/{analysis_name}/pipeline.json
```

Requesting a `type="dataset"` analysis through the custom route (or vice versa) is
rejected with a 400 naming the mismatch -- e.g. `GET /v1/analyses/difference/data.csv`
fails, since `difference` requires a dataset and this route supplies none.

Every route above accepts:

- The dataset's own dimension filters (for the dataset-scoped routes) plus the reserved
  `grid_cell` and `where` keys, exactly as the flat Data API accepts them -- forwarded to
  the analysis as its pipeline input's own filters.
- The analysis's own declared parameters (e.g. `dimension`/`order` for `difference`,
  `model_group_id`/`scenario_id`/`marsh_threshold`/`salinity_threshold` for
  `salinity-veg-investigation`) as additional query parameters -- there is no fixed
  parameter set at the route level, since it differs per analysis; whatever isn't one
  of the route's own known fields is handed to the analysis's `Parameters` model for
  validation.
- `format`, `color_map`, `sort`, and `select` as request-level overrides on the *resolved*
  pipeline's `output` -- these are never part of the registry entry itself (the same
  value can be rendered with different color maps by different callers), and always win
  over whatever the analysis's own `pipeline()` method set. `data.{format}` routes also
  accept the tabular/raster `compression_algorithm`/`compression_level` parameters, the
  same as the flat Data API.

## Registered Analyses

### `difference`

`type="dataset"`. Computes the difference between two selected values of one dimension:
negates whichever of the two is later (`order=asc`) or earlier (`order=desc`), then sums
across it, collapsing that dimension out of the result.

| Parameter | Description |
|---|---|
| `dimension` | The dimension to difference across (must be a real, filterable dimension) |
| `order` | `"asc"` or `"desc"` -- which of the two selected values gets negated (`asc` negates the later value, `desc` negates the earlier) |

**Cardinality rule**: the dataset's own filter on `dimension` must select exactly 2
distinct values -- fewer or more is rejected with a 400 before a pipeline is even built.

```
GET /v1/datasets/sal/annual/hydro_compartment_v001/analyses/difference/data.csv?model_group_id=520&scenario_id=7,8&aggregate_type=2_wk_max&calendar_year=2030&dimension=scenario_id&order=asc
```

### `concatenate`

`type="dataset"`. Produces a categorical land-type-change code (`variables=["lnd_type"]`,
`legend="lnd_chng"`) from two selected values of one dimension. Two modes, selected by how
many `model_group_id` values are also selected (not by `dimension`'s own cardinality,
which is always exactly 2 for both modes):

- **Two-image** (`model_group_id` absent, or selected to exactly 1 value): the earlier
  selected value is multiplied by 10, then summed with the later.
- **Three-image** (`model_group_id` selected to exactly 2 values): a redundant
  model-group/dimension corner is dropped, the earlier dimension value is multiplied by
  100, the later dimension value (for the lower of the two model groups) is multiplied by
  10, then all three are summed.

| Parameter | Description |
|---|---|
| `dimension` | The dimension to concatenate across (must select exactly 2 distinct values) |

**Cardinality rule**: `dimension`'s own filter must select exactly 2 distinct values;
`model_group_id`, if given, must select exactly 1 (two-image) or exactly 2 (three-image)
-- any other count is rejected with a 400.

`lnd_type` lives only on the `morph_pixel` pixel grid, so the result is a raster surface
and a tabular `data.{format}` is rejected -- ask for `tif` (or `png`/`jpg` on the tile
routes).

```
GET /v1/datasets/lnd_type/annual/morph_pixel_v001/analyses/concatenate/data.tif?model_group_id=500,510&scenario_id=7&calendar_year=2040,2050&dimension=calendar_year
```

### `land-type-confidence`

`type="dataset"`. A confidence-weighted land/water surface over the five-member land-type
ensemble (`variables=["lnd_type"]`). Each member (`scenario_id` 34-38) is reduced to a land
(`1.0`) / water (`0.0`) indicator, then the five are combined as a weighted sum:

| `scenario_id` | Weight |
|---|---|
| 34 | 0.085 |
| 35 | 0.25 |
| 36 | 0.33 |
| 37 | 0.25 |
| 38 | 0.085 |

The weights sum to exactly `1.0`, so the result is the weighted FRACTION of the ensemble
calling a cell land: `0.0` = every member says water, `1.0` = every member says land,
anything between = the weight of the members that disagree. Land is `lnd_type` `1`, `3`
and `4`; water is `2` and `5`; anything else -- including nodata -- is dropped rather
than mapped, so an unrecognized code never reaches the sum as itself.

This analysis declares **no parameters of its own**: the ensemble, the weights and the
mapping are all fixed, and a request is dimension filters alone (typically
`model_group_id` and `calendar_year`, plus the usual `grid_cell`/`where`).

**Selection rule**: `scenario_id` may not be filtered by the caller -- the analysis
supplies the full five-member set itself, and the weights are only a weighted mean over
all of it, so a subset would silently return a partial sum that reads like a confidence
but isn't. Supplying one is rejected with a 400.

Every other dimension is kept in the `group`, so omitting `calendar_year` returns one
result per year rather than summing years together. On a pixel grid that means one raster
band per year, which for the full 44-year ensemble is a very large read -- filter to the
year(s) you want.

A cell that is nodata in only SOME members is summed over the members it has, so its
weights no longer total `1.0` and its confidence reads low rather than being rescaled to
the members that actually reported.

```
GET /v1/datasets/lnd_type/annual/morph_pixel_v002/analyses/land-type-confidence/data.tif?model_group_id=705&calendar_year=2050
GET /v1/datasets/lnd_type/annual/morph_pixel_v002/analyses/land-type-confidence/data/{z}/{x}/{y}.png?model_group_id=705&calendar_year=2050
```

### `salinity-veg-investigation`

`type="custom"`. The earliest calendar year at which marsh coverage
(`ffibs_coverage_ratio`/annual/`veg_grid_cell_v001`, fixed `ffibs_type=pl_fm`) and paired
salinity (`sal`/annual/`hydro_compartment_v001`, fixed `aggregate_type=2_wk_max`) both
cross their thresholds, at every vegetation-grid pixel -- the paired (`project` + spatial
`join`) pipeline shape shown in full in the
[Analysis API's salinity-veg-investigation example](analysis-api.md#the-salinity-veg-investigation-earliest-year-marsh-and-salinity-both-cross-a-threshold).
Because it's `type="custom"`, its two inputs are fixed by its own `pipeline()` method --
there is no `/v1/datasets/.../` dataset to scope it to.

| Parameter | Description |
|---|---|
| `model_group_id` | Selects both inputs' `model_group_id` (required, no default) |
| `scenario_id` | Selects both inputs' `scenario_id` (required, no default) |
| `marsh_threshold` | Marsh coverage ratio threshold, `0.0`-`1.0` (default `0.5`) |
| `salinity_threshold` | Salinity threshold, `>= 0.0` (default `5.5`) |

Output is always a raster (`tif`, or `png`/`jpg` on the tile routes) -- a tabular output is
rejected the same way any paired-shape pipeline's tabular output would be, since the
result is inherently one raster surface.

```
GET /v1/analyses/salinity-veg-investigation/data.tif?model_group_id=500&scenario_id=7&marsh_threshold=0.5&salinity_threshold=5.5
GET /v1/analyses/salinity-veg-investigation/data/{z}/{x}/{y}.png?model_group_id=500&scenario_id=7
```

## Seeing the Expanded Pipeline: `pipeline.json`

Every named analysis's `pipeline.json` route returns the exact `Pipeline` JSON its
parameters and filters resolve to -- the same shape a hand-authored `POST
/v1/analysis.json` request would submit -- with no `output` overrides beyond what the
request itself supplied. For example:

```
GET /v1/datasets/sal/annual/hydro_compartment_v001/analyses/difference/pipeline.json?model_group_id=520&scenario_id=7,8&aggregate_type=2_wk_max&calendar_year=2030&dimension=scenario_id&order=asc
```

returns:

```json
{
  "inputs": {
    "x": {"variable": "sal", "time_unit": "annual", "grid": "hydro_compartment_v001"}
  },
  "steps": [
    {"op": "filter", "input": "x", "where": "aggregate_type = '2_wk_max' AND calendar_year = 2030 AND model_group_id = 520 AND scenario_id IN (7, 8)"},
    {"op": "calculate", "expression": "value * -1", "where": "scenario_id = 8"},
    {"op": "group", "by": ["model_group_id", "aggregate_type", "calendar_year"], "aggregate": "sum", "of": "value"}
  ],
  "output": {"format": "csv"}
}
```

**About `output.format` here.** A `Pipeline`'s `output.format` is required, so every
analysis's `pipeline()` sets one; on the `data.{format}`/tile routes that value is always
overwritten by the format in the URL, which is why the same analysis can be fetched as
`csv`, `tif`, `png` or `.pbf` without declaring any of them. `pipeline.json` is the one
route with no format segment to source an override from, so it returns whatever the
analysis itself set -- and that value is therefore the format the published pipeline is
meant to be resubmittable under. The dataset-scoped analyses derive it from the selected
grid's kind: `tif` on a pixel grid, whose result is a raster surface, and `csv` on a vector
one, whose result is a table.

The URL's dataset filters land in a `filter` **step**, not on the input: an input names a
dataset and nothing else (see [Analysis API](analysis-api.md#request-shape)). The
registry builds that step in sorted conjunct order, which is why the `where` above reads
`aggregate_type ... AND calendar_year ... AND model_group_id ... AND scenario_id ...`
rather than in the order the query string listed them.

which is exactly the hand-authored "Map Query Comparing Two Scenarios" pipeline shown in
the [Analysis API](analysis-api.md#map-query-comparing-two-scenarios) -- proving the named
analysis is sugar over the same pipeline grammar, never a second engine. This holds for
every registered analysis: `.../analyses/salinity-veg-investigation/pipeline.json?...` likewise
returns exactly the explicit paired-shape `Pipeline` shown in the
[salinity-veg-investigation example](analysis-api.md#the-salinity-veg-investigation-earliest-year-marsh-and-salinity-both-cross-a-threshold).
