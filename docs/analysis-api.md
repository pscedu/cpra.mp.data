# Analysis API

The [Data API](data-api.md) operates on one dataset at a time and only ever selects or
narrows stored values. Analysis pipelines let a request name **several** datasets, move
them onto a common grid, join them, and filter/calculate/aggregate the result -- e.g.
"wherever marsh coverage exceeds a threshold *and* salinity exceeds a threshold, what is
the earliest year that became true?" A pipeline request is a small, named-step program
rather than a set of query parameters, since expressing that kind of cross-dataset
condition as flat parameters on a single dataset's endpoint has no natural spelling. A
single-dataset transformation -- filtering, calculating, grouping, or zonal summary of
just one dataset -- is also expressed as a (one-input) pipeline; see
[Migration from Removed Parameters](data-api.md#migration-from-removed-parameters) for
how each of the old flat transformation parameters maps onto a pipeline step.

The same engine is available as a Python library for code running with direct filesystem
access to the data (notebooks, scripts): `analyze_data` runs a pipeline with no HTTP
round trip. See the
[README's Analyzing Data section](../README.md#analyzing-data) for usage.

A small, fixed set of named, pre-built pipelines (`difference`, `concatenate`,
`exit-criterion`) is also available without hand-authoring pipeline JSON at all -- see
[the named-analysis registry](analyses-api.md).

Three endpoints accept a pipeline, sharing one request model:

```
POST /v1/analysis.<format>
GET  /v1/analysis.<format>
GET  /v1/analysis/{z}/{x}/{y}.<png|jpg>
```

`POST` takes the pipeline as a JSON body. `GET` (both the flat and the tile form) takes
the identical pipeline flattened into a dotted query string, so a pipeline can be
embedded in a map tile URL template. In every case, `<format>` in the URL must agree with
the pipeline's own `output.format` -- a mismatch (e.g. a `.csv` URL for a pipeline whose
`output.format` is `"tif"`) is rejected with a 400 before anything is read.

## Request Shape

A pipeline has three parts:

```json
{
  "inputs": {
    "<name>": {"variable": "...", "time_unit": "...", "grid": "...", "<dimension>": "..."}
  },
  "steps": [ {"op": "...", ...} ],
  "output": {"format": "...", "color_map": "...", "sort": "...", "select": "..."}
}
```

**`inputs`** names one or more dataset selections. Each is the same `variable`/
`time_unit`/`grid` triple the flat data endpoints take, plus any number of dimension
filters (e.g. `model_group_id`, `scenario_id`) as extra keys -- a scalar or a
comma-separated/JSON-list of values, exactly as on the flat endpoints -- plus the
reserved `grid_cell` key, filtering that input's own source grid exactly like the flat
`grid_cell=` parameter.

**`steps`** is an ordered list of operations, each consuming a named input or a previous
step's own output:

| Op | Fields | Description |
|---|---|---|
| `project` | `input` (optional, chains from the previous step if omitted), `to` (target grid), `aggregate` (required moving to a coarser grid, forbidden moving to a finer one; may be `sum`/`max`/`min`/`mean`/`count`/`count_by_value`), `as` | Move a named input onto a different grid via a crosswalk. |
| `join` | `type` (`"spatial"`), `inputs` (two or more names), `as` | Combine named inputs/step outputs on shared dimensions and `grid_cell`. |
| `filter` | `where` | A SQL-style predicate over the in-scope named columns (e.g. `marsh > 0.5 AND sal > 5.5`), always over the previous step's output. |
| `calculate` | `expression`, `where` (optional) | Conditionally rewrite the running value: one expression/where pair per step (several pairs = repeated steps, applied sequentially -- step N+1 sees step N's output), always over the previous step's output. Rows/bands not matching `where` flow through unchanged; `calculate` never drops data. |
| `group` | `by` (dimensions to group by; `[]` collapses to one value per `grid_cell`), `aggregate` (`sum`/`min`/`max`/`count`/`mean` -- the decomposable aggregates only; `count_by_value` is never valid here), `of` (defaults to `"value"`; a numeric dimension such as `calendar_year` is also valid, for "first year a condition held") | Collapse rows/bands, always over the previous step's output. |

`filter`, `calculate`, and `group` never take an explicit input -- they always consume
whatever the immediately preceding step produced -- so the **first** step of every
pipeline must be a `project` (naming its input explicitly) or a `join` (naming two or
more inputs explicitly). A single-input pipeline that changes no grid at all still opens
with an identity `project` (`to` equal to the input's own grid) purely to satisfy this
rule.

A pipeline `calculate`/`filter` expression references the running value **by its
in-scope column name** -- the input or step name (e.g.
`{"op": "calculate", "expression": "sal * 0.6", "where": "model_group_id = 500"}` after an
input named `sal`); the placeholder name `value` is also accepted, since both spellings
resolve to the same running value in a single-input chain.

**`output`** gives the response `format` (any tabular/`tif` format, or a tile format for
the tile route), an optional `color_map` for rendered raster output, and optional
`sort`/`select`, with exactly the same semantics as the flat data endpoints' `sort`/
`select` parameters.

## What Compiles Today

Not every step sequence a pipeline could describe is executable yet. Three shapes are
supported; anything else is rejected with a `PipelineError`-derived 400 naming the
specific unsupported shape and why, rather than a generic failure.

1. **Single-input chains equivalent to a flat dataset request**: one input, an opening
   `project` (identity, or many-to-one onto a coarser grid with `aggregate`), then
   optionally `filter` and/or `calculate` step(s) and at most one trailing `group`, to any
   tabular or `tif`/tile format. One ordering caveat falls out of this: a `filter` step
   ordered *after* a `calculate` step is accepted only when its `where` is dimension-only
   (the two commute); a value condition there would test the pre-calculate value, so it
   is rejected with a 400 suggesting the filter be moved before the calculate (or folded
   into the calculate entry's own `where`). A `filter` naming `grid_cell` right after a
   many-to-one `project` is understood as filtering the *summarized zones*, not the
   source grid's own cells -- only one such filter is accepted per pipeline, and it must
   be a clean literal-ID selection (`grid_cell = 1` or `grid_cell IN (1, 2)`), not an
   arbitrary predicate.
2. **Project-only reaggregation**: the same shape as (1), specialized to a many-to-one
   `project` (e.g. raster pixels reaggregated onto a coarser vector zone) with a tabular
   output -- no `filter`/`group` needed. `output.sort`/`output.select`/`output.color_map`
   still apply to this degenerate case.
3. **The paired (spatially-joined) shape** (`calculate` steps are not supported in this
   shape and are rejected with a `PipelineError`-derived 400 naming it -- apply the
   rewrite on the flat endpoint or a single-input pipeline feeding the analysis instead):
   exactly two inputs -- a coarser one broadcast (one-to-many `project`, no `aggregate`)
   onto a finer *raster* grid, immediately `join`ed against that raster input, then any
   number of `filter` steps and exactly one trailing `group` with `by: []` (a per-pixel
   collapse across bands, e.g. across years -- not a zonal collapse) and a numeric `of`
   (`calendar_year`, or the running `value`) for every aggregate except `count`, which has
   none. Because this shape's result is inherently one raster surface, only
   `tif`/`png`/`jpg` output is implemented for it; a tabular paired output is rejected
   outright. This shape rejects `grid_cell` on either input, and rejects `output.sort`/
   `output.select` outright (they would silently no-op against a per-pixel raster
   result).

A `join`'s two sides must agree on every dimension they share (present on all joined
inputs and filtered to the same value(s)), and a dimension present on only one side must
be pinned to a single value there, since there is no row on the other side to align it
against. The coarse/zone side of a `project`/`join` is also capped at a configured row
ceiling (`ANALYSIS_JOIN_MAX_ROWS`), checked before that side is materialized, so an
unbounded selection is rejected rather than silently read in full.

Everything else -- three or more inputs, a `join` not immediately preceded by its
broadcasting `project`, a tabular output for the paired shape, `filter`/`group` in an
order the compiler doesn't recognize, and so on -- is rejected informatively rather than
silently misexecuted.

## Memory Behavior

The paired shape (3) exists specifically to avoid materializing a full joined table:
naively broadcasting a coarse dataset onto every pixel of a fine raster grid, across
every year, and then grouping, means holding one column per year for every pixel in
memory at once before the aggregate can drop most of it. Instead, the compiled plan reads
both inputs' small ROW metadata up front (file paths/bands for the raster input; the
already row-ceiling-bounded coarse table), then executes as a single pass over the
raster's own block windows: for each window, it reads that window from every year's band
plus the pairing crosswalk, looks up each pixel's paired coarse value, applies the
`filter`/`group` collapse for just that window, and writes the collapsed window straight
back at its own position -- never holding a full-extent array of any per-pixel
intermediate.

## Examples

### Vector-Grid Pipelines

#### Map Query Comparing Two Scenarios

This pipeline returns the difference in annual salinity (two-week-max aggregate) between
two scenarios in one year (scenario 7 minus scenario 8): an identity `project`, a
`calculate` step negating scenario 8's rows, then a `group` collapsing the two scenarios
into one value per grid cell. `model_group_id`, `aggregate_type`, and `calendar_year` are
each pinned to one value here (so listing them in `group.by` alongside the truly-varying
`scenario_id` is a no-op on this particular request), because `sal` always carries an
`aggregate_type` dimension at every time granularity -- pin it explicitly rather than
leaving it to match every stored value.

```json
POST /v1/analysis.csv
{
  "inputs": {
    "sal": {
      "variable": "sal", "time_unit": "annual", "grid": "hydro_compartment_v001",
      "model_group_id": 520, "scenario_id": [7, 8], "aggregate_type": "2_wk_max",
      "calendar_year": 2030
    }
  },
  "steps": [
    {"op": "project", "input": "sal", "to": "hydro_compartment_v001"},
    {"op": "calculate", "expression": "value * -1", "where": "scenario_id = 8"},
    {"op": "group", "by": ["model_group_id", "aggregate_type", "calendar_year"], "aggregate": "sum"}
  ],
  "output": {"format": "csv"}
}
```

This exact `calculate`(negate)+`group`(sum) idiom is also available as the `difference`
named analysis (`dimension=scenario_id`, `order=asc` negates the later of the two
selected values, here `8`) -- see [the registry doc](analyses-api.md#difference) for the
equivalent request with no hand-authored pipeline. The named analysis derives its own
`group.by` as "every required dimension of the dataset except the one being differenced,"
which is why it lists `model_group_id`/`aggregate_type`/`calendar_year` explicitly rather
than assuming they're already pinned to one value each.

#### Query with Additional Dimensions

This pipeline returns decadal damage for a single grid cell. Since `aep`, `asset_type`,
and `percentile` are not filtered on the input, the query matches all available values for
those dimensions; the results are summed across `asset_type` (the one dimension not
listed in `group.by`) to give the total damage for each AEP-percentile combination.

```json
POST /v1/analysis.ndjson
{
  "inputs": {
    "damage": {
      "variable": "damage", "time_unit": "decadal", "grid": "clara_grid_cell_v001",
      "model_group_id": 500, "scenario_id": 7, "pumping_id": 0, "fragility_scenario": 0,
      "grid_cell": "12"
    }
  },
  "steps": [
    {"op": "project", "input": "damage", "to": "clara_grid_cell_v001"},
    {"op": "group", "by": ["aep", "percentile"], "aggregate": "sum"}
  ],
  "output": {"format": "ndjson", "sort": "aep,percentile"}
}
```

#### Conditional Count

This pipeline counts, per grid cell, how many decadal damage values exceed a threshold: a
`filter` step masks non-matching values (since a `group` step downstream makes this an
aggregating request), and `group`'s `count` aggregate then counts only what wasn't
masked.

```json
POST /v1/analysis.csv
{
  "inputs": {
    "damage": {
      "variable": "damage", "time_unit": "decadal", "grid": "clara_grid_cell_v001",
      "model_group_id": 500, "scenario_id": 7
    }
  },
  "steps": [
    {"op": "project", "input": "damage", "to": "clara_grid_cell_v001"},
    {"op": "filter", "where": "value > 1000000"},
    {"op": "group", "by": [], "aggregate": "count"}
  ],
  "output": {"format": "csv"}
}
```

#### Single-Input Pipeline Equivalent to a Flat Request

An identity `project` (no grid change), a `filter`, and a `group`:

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

#### Single-Input Pipeline with Calculate Steps: a Confidence-Weighted Collapse

Two model groups weighted 0.6/0.4 via two sequential `calculate` steps (each selecting
its model group's rows with a dimension `where`), collapsed with `group`/`sum`. The
expression names the running value by the input's own name (`sal`), the same way
`filter.where` would:

```json
POST /v1/analysis.csv
{
  "inputs": {
    "sal": {"variable": "sal", "time_unit": "annual", "grid": "hydro_compartment_v001", "model_group_id": [500, 510], "scenario_id": 7, "aggregate_type": "2_wk_max"}
  },
  "steps": [
    {"op": "project", "input": "sal", "to": "hydro_compartment_v001"},
    {"op": "calculate", "expression": "sal * 0.6", "where": "model_group_id = 500"},
    {"op": "calculate", "expression": "sal * 0.4", "where": "model_group_id = 510"},
    {"op": "group", "by": ["calendar_year"], "aggregate": "sum"}
  ],
  "output": {"format": "csv"}
}
```

The same pipeline in the dotted `GET` encoding spells each step as
`step.2.op=calculate&step.2.expression=sal%20*%200.6&step.2.where=model_group_id%20%3D%20500`,
and works on the tile route the same way.

### Raster-Grid Pipelines

#### Basic Map Query

For a plain raster selection with no transformation, use the [Data API](data-api.md)
directly -- no pipeline is needed:

```
GET /v1/datasets/elev/annual/morph_pixel_v001/data/{z}/{x}/{y}.png?model_group_id=500&scenario_id=7&calendar_year=2045&color_map=elevation
```

#### Map Query Comparing Two Years

This pipeline returns the colorized elevation difference between two years (2050 minus
2045): an identity `project`, a `calculate` step negating 2045's band, then a `group`
collapsing the two years into one band per pixel, rendered on the analysis tile route.

```json
{
  "inputs": {
    "elev": {
      "variable": "elev", "time_unit": "annual", "grid": "morph_pixel_v001",
      "model_group_id": 500, "scenario_id": 7, "calendar_year": [2045, 2050]
    }
  },
  "steps": [
    {"op": "project", "input": "elev", "to": "morph_pixel_v001"},
    {"op": "calculate", "expression": "elev * -1", "where": "calendar_year = 2045"},
    {"op": "group", "by": [], "aggregate": "sum"}
  ],
  "output": {"format": "png", "color_map": "elevation_difference"}
}
```

As a `GET` request to `/v1/analysis/{z}/{x}/{y}.png`, this pipeline is flattened into the
same dotted query encoding shown in the exit-criterion example below
(`input.elev.variable=elev&...&step.1.op=project&...&output.format=png&output.color_map=elevation_difference`);
there is no `POST` form for the tile route.

#### Confidence-Weighted Surface Across Model Groups

This pipeline builds a confidence-weighted elevation surface: three model groups are each
multiplied by their own confidence coefficient (0.6, 0.3, 0.1) via three `calculate`
steps -- a dimension condition in each step's `where` selecting that model group's band
-- then collapsed to one band per pixel with `group`/`sum`. The steps are independent
(each matches a different model group), so this is a per-band weighting; a pixel that is
NoData in one group's band simply contributes nothing from that group.

```json
{
  "inputs": {
    "elev": {
      "variable": "elev", "time_unit": "annual", "grid": "morph_pixel_v001",
      "model_group_id": [500, 510, 520], "scenario_id": 7, "calendar_year": 2050
    }
  },
  "steps": [
    {"op": "project", "input": "elev", "to": "morph_pixel_v001"},
    {"op": "calculate", "expression": "elev * 0.6", "where": "model_group_id = 500"},
    {"op": "calculate", "expression": "elev * 0.3", "where": "model_group_id = 510"},
    {"op": "calculate", "expression": "elev * 0.1", "where": "model_group_id = 520"},
    {"op": "group", "by": [], "aggregate": "sum"}
  ],
  "output": {"format": "png", "color_map": "elevation"}
}
```

The same pipeline works on the `.tif` output too (drop the tile route, `POST` this to
`/v1/analysis.tif` with `output.format` set to `"tif"`), and can `project` onto a vector
zone instead of collapsing to a full-resolution raster, for a tabular result.

#### Single Year Query Summarized by Hydro Compartment

This pipeline returns elevation for a single year summarized by hydro compartment as a
Parquet file -- a many-to-one `project` with no `filter`/`group` needed at all
(`output.sort` still applies to this degenerate, project-only shape):

```json
POST /v1/analysis.parquet
{
  "inputs": {
    "elev": {
      "variable": "elev", "time_unit": "annual", "grid": "morph_pixel_v001",
      "model_group_id": 500, "scenario_id": 7, "calendar_year": 2050
    }
  },
  "steps": [
    {"op": "project", "input": "elev", "to": "hydro_compartment_v001", "aggregate": "mean"}
  ],
  "output": {"format": "parquet", "sort": "calendar_year,grid_cell"}
}
```

#### Conditional Pixel Count per Zone

This pipeline counts, per hydro compartment, the pixels whose elevation exceeds a
threshold in a single year: a many-to-one `project` with `aggregate: "count"`, plus a
`filter` step whose value condition masks non-matching pixels before the count (since the
project's own `count` aggregate makes this an aggregating request).

```json
POST /v1/analysis.csv
{
  "inputs": {
    "elev": {
      "variable": "elev", "time_unit": "annual", "grid": "morph_pixel_v001",
      "model_group_id": 500, "scenario_id": 7, "calendar_year": 2050
    }
  },
  "steps": [
    {"op": "project", "input": "elev", "to": "hydro_compartment_v001", "aggregate": "count"},
    {"op": "filter", "where": "value > 0"}
  ],
  "output": {"format": "csv"}
}
```

#### Categorical Count per Zone

This pipeline counts, per hydro compartment, the pixels carrying each distinct land-type
value in a single year, using `count_by_value` as the `project` aggregate. The output has
one row per (zone, value) pair.

```json
POST /v1/analysis.csv
{
  "inputs": {
    "lnd_type": {
      "variable": "lnd_type", "time_unit": "annual", "grid": "morph_pixel_v001",
      "model_group_id": 500, "scenario_id": 7, "calendar_year": 2050
    }
  },
  "steps": [
    {"op": "project", "input": "lnd_type", "to": "hydro_compartment_v001", "aggregate": "count_by_value"}
  ],
  "output": {"format": "csv"}
}
```

#### Project-Only Reaggregation

Reaggregate raster elevation onto hydro compartments as a zonal mean, with no
`filter`/`group` needed:

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

### Multi-Input (Paired) Pipelines

#### The Exit-Criterion Query: Earliest Year Marsh and Salinity Both Cross a Threshold

This pipeline broadcasts annual salinity (by hydro compartment) onto the vegetation grid,
pairs it against marsh coverage there, keeps only pixel-years where marsh coverage
exceeds 0.5 *and* paired salinity exceeds 5.5, and reports the earliest (`min`) such
`calendar_year` per pixel, as a GeoTIFF.

**As an explicit pipeline:**

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

The same pipeline as a `GET` request, using the dotted query encoding (dot = nesting, an
integer segment = a 1-based list position, comma = a list value -- never inside `where`,
which is one opaque string):

```
GET /v1/analysis.tif?input.marsh.variable=ffibs_coverage_ratio&input.marsh.time_unit=annual&input.marsh.grid=veg_grid_cell_v001&input.marsh.model_group_id=500&input.marsh.scenario_id=7&input.marsh.ffibs_type=pl_fm&input.salinity.variable=sal&input.salinity.time_unit=annual&input.salinity.grid=hydro_compartment_v001&input.salinity.model_group_id=500&input.salinity.scenario_id=7&input.salinity.aggregate_type=2_wk_max&step.1.op=project&step.1.input=salinity&step.1.to=veg_grid_cell_v001&step.1.as=sal&step.2.op=join&step.2.type=spatial&step.2.inputs=marsh,sal&step.2.as=paired&step.3.op=filter&step.3.where=marsh%20%3E%200.5%20AND%20sal%20%3E%205.5&step.4.op=group&step.4.by=&step.4.aggregate=min&step.4.of=calendar_year&output.format=tif
```

And as map tiles, replacing `output.format=tif` with `output.format=png`:

```
GET /v1/analysis/{z}/{x}/{y}.png?...same query as above, with output.format=png
```

**As the `exit-criterion` named analysis** -- the same request, with the `Pipeline` above
built by the registry rather than hand-authored:

```
GET /v1/analyses/exit-criterion/data.tif?model_group_id=500&scenario_id=7&marsh_threshold=0.5&salinity_threshold=5.5
```

`exit-criterion` is a `type="custom"` analysis (its two inputs are fixed by its own
`pipeline()` method, not supplied by a dataset in the URL -- see
[the registry doc](analyses-api.md#exit-criterion)), so this route has no
`/v1/datasets/{variable}/{time_unit}/{grid}/` prefix. Fetching
`GET /v1/analyses/exit-criterion/pipeline.json?model_group_id=500&scenario_id=7&marsh_threshold=0.5&salinity_threshold=5.5`
returns exactly the explicit `Pipeline` JSON shown at the top of this section (with
`model_group_id`/`scenario_id` substituted from the request) -- proving the named
analysis is sugar over the same pipeline grammar, never a second engine.
