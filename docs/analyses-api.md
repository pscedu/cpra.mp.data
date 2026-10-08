# Named Analyses (Registry API)

A named analysis is a pre-built [pipeline](analysis-api.md) with its own request-shaped
parameters -- a `diff` between two scenarios, a `lnd_chng` land-type-change
code, a `sal_veg_investigation` earliest-crossing-year surface -- registered under a
short name so a caller never has to hand-author the equivalent pipeline JSON. Every named
analysis is **sugar over the same pipeline grammar** documented in the
[Analysis API](analysis-api.md): resolving one always produces a real `Pipeline`, run
through exactly the same compiler any hand-authored pipeline uses. Nothing is reachable
only through a named analysis, and `/v1/analyses/{analysis}.json` (below) lets a caller see
the exact pipeline any request would run.

## Asking for an Analysis

An analysis has no routes of its own. It is the `analysis=` **query parameter** on the
ordinary data routes:

```
GET /v1/data.{format}?analysis={name}&...
GET /v1/data/{z}/{x}/{y}.{format}?analysis={name}&...
GET /v1/catalog.json?analysis={name}&...
GET /v1/analyses/{analysis}.json?...
```

There used to be four routes per analysis, and two sets of them: a `/v1/datasets/{variable}/
{time_unit}/{grid}/analyses/{name}/...` family for an analysis that read the caller's
dataset and a `/v1/analyses/{name}/...` family for one that fixed its own inputs. Both are
gone. The URL's shape said nothing about the request that the request could not say about
itself, and the split meant the two families could not be combined -- there was no way to
write "this dataset, this analysis, and these extra pipeline steps" at all. See the
[Data API](data-api.md) for the consolidated route set.

`/v1/analyses/{analysis}.json` is the one place an analysis name still appears in a path,
because there the name is *what is being asked about* rather than how the request is
spelled.

### Chaining

`analysis=` takes a comma-separated chain, resolved left to right:

```
GET /v1/data.tif?variable=lnd_type&time_unit=annual&grid=morph_pixel_v002&analysis=lnd_conf,diff&dimension=calendar_year&order=asc&model_group_id=705&calendar_year=2030&calendar_year=2050
```

A multi-value selection is spelled by **repeating the key**, as above -- `calendar_year`
twice, not `calendar_year=2030,2050`. The comma is the chain separator in `analysis=` and
nothing else.

Stage 1 resolves as it would alone; **every later stage must be `kind="dimension"`**
(below), and contributes its steps on top of the running pipeline. Putting a
`kind="variable"` or `kind="internal"` analysis anywhere but first is a 400:

```json
{"detail": "analysis 'lnd_conf' is kind='variable', but only a kind='dimension' analysis may follow the first stage of a chain"}
```

Two further chain rules, both 400s: a later stage's `dimension` may not name a dimension
the chain has already collapsed (tracked from stage 1's reported
`collapsed_dimensions`, so an analysis that overrides them rather than deriving them
structurally is still accounted for), and no two stages may declare a parameter of the same
name -- which today can only happen by naming the same analysis twice.

### Whether to send a dataset

An analysis either reads the caller's dataset or fixes its own inputs, and this decides
whether the request may carry a `variable`/`time_unit`/`grid` triple:

- `diff`, `lnd_chng`, `lnd_conf`, `sediment_dynamics` **require** one. Omitting it is a 400.
- `lnd_type_plus`, `sal_veg_investigation` **reject** one -- they name their own inputs,
  across grids the triple could not describe. Sending one is a 400.

`sediment_dynamics` is the odd one: it requires a triple but reads five variables, taking
only the `grid` and `time_unit` from it. "Requires a dataset" means the analysis cannot
answer without one, not that it reads exactly that one.

This is not declared in the catalog. It lives in each analysis's own `pipeline()`, as a
call to `require_dataset()` or `reject_dataset()`, and the catalog reports its
*consequence* instead -- see `output` below. (The retired `Meta.type` field declared it,
and existed only to pick between the two route families that no longer exist.)

## Catalog

```
GET /v1/analyses.json
```

Returns the full list of registered analyses, each with:

| Field | Description |
|---|---|
| `name` | The analysis's registered name (e.g. `diff`, `lnd_chng`, `sal_veg_investigation`) |
| `label` | A human-readable display label |
| `kind` | `"variable"`, `"dimension"` or `"internal"` -- its presentation role; see below |
| `variables` | The variable(s) it's meaningful for, or `null` for any (and for one with fixed inputs, which has no caller-chosen dataset to qualify) |
| `legend` | A named legend/color-map hint for rendered output, or `null` |
| `result_kind` | `"value"`, `"difference"`, or `"category"` -- a presentation hint for how to label the result |
| `subsumed_dimensions` | Dimensions this analysis removes from the caller's view, so a picker should stop offering them |
| `output` | What this analysis produces, or `null`; see below |

`kind` is the analysis's **presentation class**, and is a different partition from the one
the old `type` field drew:

- `"variable"` stands in for an input variable in a client's variable picker
  (`lnd_conf`, `lnd_chng`, `lnd_type_plus`);
- `"dimension"` operates across one dimension of whatever it is given -- a plain dataset,
  or the output of a `"variable"` analysis it is chained after (`diff`). **Only this kind
  may appear as a non-leading stage of a chain.**
- `"internal"` is neither: no picker role and nothing to chain onto it
  (`sal_veg_investigation`, `sediment_dynamics` -- the latter is reached through the chart
  of the same name rather than through a variable picker).

### `output`

`output` is a [`/v1/catalog.json`](catalog-api.md) output block -- the same descriptor,
field for field, that route reports for a request -- and it is **non-null exactly for an
analysis that fixes its own inputs**:

```json
{
  "name": "lnd_type_plus",
  "label": "Land Type Plus",
  "kind": "variable",
  "variables": null,
  "legend": "lnd_type_plus",
  "result_kind": "category",
  "subsumed_dimensions": ["aggregate_type"],
  "output": {
    "grid": "morph_pixel_v002",
    "grid_type": "raster",
    "shape": "band_wide",
    "time_unit": "annual",
    "time_column": "calendar_year",
    "value_columns": ["value"],
    "variables": [null],
    "label": "Land Type Plus",
    "legend": "lnd_type_plus",
    "result_kind": "category",
    "collapsed_dimensions": ["aggregate_type"]
  }
}
```

For such an analysis the result is a fact about the analysis rather than about any
request -- `lnd_type_plus` reads three datasets across two grids and outputs on a third no
matter what a caller sends -- so it can be answered before a request exists. For every
other entry the result depends on a dataset the caller has not named yet, and `output` is
`null`; ask `/v1/catalog.json` once you have named one.

The server computes this by resolving the analysis with nothing supplied at all and
reporting what came out, so it cannot drift from what the analysis actually does. A client
needing a grid and time unit for an analysis *before* its first request -- to offer it in a
variable picker at all -- reads them from here.

## Parameters

Every route above accepts, in one flat query string:

- **`analysis=`** -- the name or chain.
- **The dataset triple** `variable`/`time_unit`/`grid`, for an analysis that reads one.
- **Dimension filters**, plus the reserved `grid_cell` and `where` keys, exactly as the
  [Data API](data-api.md) accepts them. For an analysis with several inputs these are
  stated once and pushed down to every input's read.
- **The analysis's own declared parameters** (e.g. `dimension`/`order` for `diff`,
  `marsh_threshold`/`salinity_threshold` for `sal_veg_investigation`). There is no fixed
  parameter set at the route level, since it differs per analysis. A name no stage of the
  chain declares is a 400 -- including when no analysis is named at all, so a misspelled
  `calender_year=2050` is now rejected rather than silently ignored.
- **Dotted pipeline parameters** -- `input.*`, `step.*`, `output.*` in the
  [dotted codec](analysis-api.md). Inputs merge into the analysis's own; steps append
  after them. This is what lets an analysis be extended in place rather than replaced.
- **Output overrides** -- `color_map`, `sort`, `select`, applied to the *resolved*
  pipeline's `output` and always winning over whatever the analysis's own `pipeline()` set.
  `/v1/data.{format}` additionally accepts `compression_algorithm`/`compression_level`; the
  tile and JSON routes reject them by name.

  There is no bare `format=` parameter -- the format is the URL's own extension, and on the
  one route without one it is the dotted `output.format=`. A bare `format=` is a leftover,
  and lands in whatever error an undeclared parameter lands in.

**A dimension name is always a filter.** No analysis may declare a parameter named after a
dimension, so there is no case where the same key means one thing on one request and
another on the next. `lnd_type_plus` and `sal_veg_investigation` used to declare
`model_group_id`/`scenario_id` as required *parameters*, because under the old split routes
a request with no dataset in its URL had nowhere to put a filter; both are ordinary filters
now.

## Registered Analyses

### `diff`

`kind="dimension"`, reads the caller's dataset. Computes the difference between two
selected values of one dimension: negates whichever of the two is later (`order=asc`) or
earlier (`order=desc`), then sums across it, collapsing that dimension out of the result.

| Parameter | Description |
|---|---|
| `dimension` | The dimension to difference across (must be a real, filterable dimension) |
| `order` | `"asc"` or `"desc"` -- which of the two selected values gets negated (`asc` negates the later value, `desc` negates the earlier) |

**Cardinality rule**: the filter on `dimension` must select exactly 2 distinct values --
fewer or more is rejected with a 400 before a pipeline is even built.

The only analysis that may appear after the first stage of a chain.

```
GET /v1/data.csv?variable=sal&time_unit=annual&grid=hydro_compartment_v002&analysis=diff&dimension=scenario_id&order=asc&model_group_id=700&scenario_id=34&scenario_id=35&aggregate_type=mean&calendar_year=2030
```

### `lnd_chng`

`kind="variable"`, reads the caller's dataset. Produces a categorical land-type-change code
(`variables=["lnd_type"]`, `legend="lnd_chng"`) between an automatically-derived baseline
year and the caller's selected comparison years. It declares no parameters of its own;
the comparison years and the mode below both come from ordinary filters.

The baseline year is derived from the selected `model_group_id`: values below `700`
belong to an earlier model era whose baseline year is `2019`; at or above it, the
baseline year is `2025`.

**The baseline broadcasts.** The pipeline reads the dataset twice: a baseline branch
filtered to the derived year and then `group`ed over every other required dimension,
which drops its `calendar_year` column, and a comparison branch carrying every year the
caller selected. The `join` keys on `grid_cell` plus the columns both branches carry, so
the baseline -- which no longer carries a year -- fans out across all of them. Each
comparison year is therefore compared against the *same* baseline and comes back as its
own row, so `calendar_year` survives into the result and a land-change series over time is
an ordinary year selection. That is why `calendar_year` is **not** in
`subsumed_dimensions` or `collapsed_dimensions`.

There are two modes, selected by how many `model_group_id` values are selected. In both,
each land-type value is placed in its own decimal digit:

- **Two-image** (`model_group_id` selected to exactly 1 value): `baseline * 10 + comparison`.
- **Three-image** (`model_group_id` selected to exactly 2 values, whichever is lower
  decides the baseline era): the baseline is the *lower* model group at the baseline year
  alone; the comparison branch keeps both model groups, multiplying the lower one's value
  by 10 and summing them. Result: `baseline * 100 + comparison`. `model_group_id` is
  collapsed in this mode, since both of its values are folded into one code.

**Cardinality rules**: `model_group_id` must select exactly 1 (two-image) or exactly 2
(three-image) -- any other count is rejected with a 400. `calendar_year` has no rule; as
everywhere else, selecting none of its values means all of them.

A **raster** output additionally requires exactly 1 comparison year, because one image
holds one. This is checked where the grid is known rather than alongside the rule above,
and a `grid_cell` selection lifts it -- that lowers the request to a tabular shape, which
has a row per year to put the series in.

`lnd_type` lives only on the `morph_pixel` pixel grid, so without a `grid_cell` the result
is a raster surface and a tabular format is rejected -- ask for `tif` (or `png`/`jpg` on
the tile route).

```
# One image: exactly one comparison year.
GET /v1/data.tif?variable=lnd_type&time_unit=annual&grid=morph_pixel_v002&analysis=lnd_chng&model_group_id=705&scenario_id=34&calendar_year=2050

# A series at one pixel: every year against the derived baseline.
GET /v1/data.csv?variable=lnd_type&time_unit=annual&grid=morph_pixel_v002&analysis=lnd_chng&model_group_id=705&scenario_id=34&grid_cell=-92.21787,29.71833
```

### `lnd_conf`

`kind="variable"`, reads the caller's dataset. A confidence-weighted land/water surface over
the five-member land-type ensemble (`variables=["lnd_type"]`). Each member (`scenario_id`
34-38) is reduced to a land (`1.0`) / water (`0.0`) indicator, then the five are combined as
a weighted sum:

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
but isn't. Supplying one is rejected with a 400. It is reported in
`subsumed_dimensions`.

Every other dimension is kept in the `group`, so omitting `calendar_year` returns one
result per year rather than summing years together. On a pixel grid that means one raster
band per year, which for the full 44-year ensemble is a very large read -- filter to the
year(s) you want.

A cell that is nodata in only SOME members is summed over the members it has, so its
weights no longer total `1.0` and its confidence reads low rather than being rescaled to
the members that actually reported.

```
GET /v1/data.tif?variable=lnd_type&time_unit=annual&grid=morph_pixel_v002&analysis=lnd_conf&model_group_id=705&calendar_year=2050
GET /v1/data/{z}/{x}/{y}.png?variable=lnd_type&time_unit=annual&grid=morph_pixel_v002&analysis=lnd_conf&model_group_id=705&calendar_year=2050
```

### `sal_veg_investigation`

`kind="internal"`, fixes its own inputs. The earliest calendar year at which marsh coverage
(`ffibs_coverage_ratio`/annual/`veg_grid_cell_v001`, fixed `ffibs_type=pl_fm`) and paired
salinity (`sal`/annual/`hydro_compartment_v001`, fixed `aggregate_type=2_wk_max`) both
cross their thresholds, at every vegetation-grid pixel -- the paired (`project` + spatial
`join`) pipeline shape shown in full in the
[Analysis API's sal_veg_investigation example](analysis-api.md#the-salinity-veg-investigation-earliest-year-marsh-and-salinity-both-cross-a-threshold).
Its two inputs are fixed by its own `pipeline()`, so a request must not carry a dataset
triple.

| Parameter | Description |
|---|---|
| `marsh_threshold` | Marsh coverage ratio threshold, `0.0`-`1.0` (default `0.5`) |
| `salinity_threshold` | Salinity threshold, `>= 0.0` (default `5.5`) |

`model_group_id` and `scenario_id` are ordinary dimension filters, stated once and pushed
down to both inputs' reads.

Output is always a raster (`tif`, or `png`/`jpg` on the tile route): the result is one
raster surface, so its final frame is band-wide and a tabular sink is rejected with the
same shape-checker message `lnd_type_plus` gives below.

It declares no `validate_selection`. The one it used to have rejected any filter at all,
which was true only while a fixed-input analysis had no way to receive one.

```
GET /v1/data.tif?analysis=sal_veg_investigation&model_group_id=500&scenario_id=7&marsh_threshold=0.5&salinity_threshold=5.5
GET /v1/data/{z}/{x}/{y}.png?analysis=sal_veg_investigation&model_group_id=500&scenario_id=7
```

### `lnd_type_plus`

`kind="variable"`, fixes its own inputs. The `lnd_type` categorical surface extended with
two shallow-inundation classes the land types don't themselves distinguish.

Inundation depth is **computed**, not read. Mean annual stage
(`stg`/annual/`hydro_compartment_v002`, with `aggregate_type=mean` fixed by the analysis) is
projected onto `morph_pixel_v002`, and topobathymetric elevation
(`elev`/annual/`morph_pixel_v002`, stored in NAVD88 millimetres) is converted to metres and
subtracted from it:

```
depth = stage - elevation / 1000
```

Stage, elevation and annual land type (`lnd_type`/annual/`morph_pixel_v002`) arrive on **one
spatial join**, and a single `case` step both computes the depth in its arms and
reclassifies it:

| Inundation depth (m), over `lnd_type = 2` | Code |
|---|---|
| up to and including `0.46` | `6` |
| deeper than `0.46`, up to and including `1.0` | `7` |
| deeper than `1.0` | `2` (plain open water) |
| anything else | the pixel's own `lnd_type`, unchanged |

Each interval excludes its lower bound and includes its upper one, so a bound belongs to
the class below it. The shallowest class has **no lower bound**: open water is open water,
so a pixel `lnd_type` calls water reads as `6` however little depth the surface finds there
-- zero or negative. The two surfaces are independently modelled and disagree at the
coastline, and where they do it is the land type that is authoritative about what is water.

Deep water has no code of its own. The result's value domain is `1`-`7`, `2` is labelled
"Water - Deep", and water deeper than 1.0 m is simply water: only the two shallow classes
say something `lnd_type` could not. The deepest arm still earns its place, because the arms
are first-match-wins in descending order of bound -- it is what stops a 1.5 m depth from
falling through to the `> 0.46` arm and reporting `7`.

**Only open water is reclassified.** The depth classes subdivide `lnd_type = 2`; they never
overwrite land. That is enforced rather than assumed -- every arm carries an
`AND land_type = 2` conjunct, so a pixel where the independently-modelled elevation and
land-type surfaces disagree reads as its land type rather than as shallow inundation over
marsh or upland. A nodata land type is not water either, and it is also what the `else`
would hand back, so such a pixel stays nodata.

**Where the stage surface does not reach, the result is nodata.** `morph_pixel_v002` carries
land types past the last compartment `hydro_compartment_v002` covers, and for those pixels
no depth exists in either direction. Handing back the bare land type there would publish a
surface over ground this analysis never evaluated, so a leading `case` arm writes null
instead. This is why the analysis's extent is the hydro-compartment crosswalk's rather than
the morphology grid's.

The analysis's `result_kind` is `"category"` and its `legend` is `lnd_type_plus`, an entry
of its own in `structure/legends.csv` (rendered in full under "Legends" in the [data
documentation](index.md)) rather than `lnd_type`'s, whose ramp stops at `5`:

| Value | Label | Color |
|---|---|---|
| nodata | No Data | `#00000000` |
| `1` | Vegetated Wetland | `#008000` |
| `2` | Water - Deep | `#6495ed` |
| `3` | Bare/Unvegetated Wetland | `#8b4513` |
| `4` | Upland/Developed Areas | `#999999` |
| `5` | Flotant Marsh | `#afeeee` |
| `6` | Water - Shallow Intertidal | `#d1dffa` |
| `7` | Water - Shallow Subtidal | `#93b5f2` |

Codes `1`-`5` carry `lnd_type`'s own colors, so a pixel this analysis leaves alone renders
the same as it does under plain `lnd_type`; `2` is relabelled "Water - Deep" because that is
what it means here -- open water that is either not evaluated by the depth surface or
inundated by more than 1.0 m.

This analysis declares **no parameters of its own**. `model_group_id`, `scenario_id` and
`calendar_year` are ordinary dimension filters; omitting `calendar_year` means every year,
one output band each.

`aggregate_type` is not selectable: the class bounds only mean something against one
reading of stage, so the analysis pins `mean` itself. It is reported in both
`subsumed_dimensions` and the output's `collapsed_dimensions`.

Output is always a raster (`tif`, or `png`/`jpg` on the tile route) -- the one `project`
lands **on** the pixel grid and nothing projects off it, so the final frame is band-wide and
a tabular sink is rejected:

```json
{"detail": "pipeline's final frame is band-wide (raster), but output.format='csv' needs a tabular sink -- add a project to a vector grid, or a spatial filter, before the sink"}
```

**Selection rule**: `grid_cell` and `where` are accepted like any other selection -- they
compose into the same conjunct the dimension filters do -- but both lower the frame to long,
and this analysis's sink is a raster one, so a `tif`/`png` request carrying either is
rejected by the shape checker rather than by a rule of the analysis's own:

```json
{"detail": "step 4 (filter): conjunct(s) ['grid_cell = 60812006'] sample the raster at a set of grid_cell points, which makes the result tabular (long), but output.format='tif' needs a raster surface. ..."}
```

```
GET /v1/data.tif?analysis=lnd_type_plus&model_group_id=705&scenario_id=34&calendar_year=2050
GET /v1/data/{z}/{x}/{y}.png?analysis=lnd_type_plus&model_group_id=705&scenario_id=34&calendar_year=2050
```

### `sediment_dynamics`

`kind="internal"`, requires a dataset but reads five. The five morphology surfaces that
describe how one pixel's ground is built and drowned -- topobathymetric elevation (`elev`),
inundation depth (`inun`), land type (`lnd_type`) and the two accretion components
(`test_mnrl_accr`, `test_org_accr`) -- sampled at **one** pixel and stacked into a single
tabular series. One row per (year, variable):

| `calendar_year` | `variable` | `value` |
|---|---|---|
| 2040 | `elev` | ... |
| 2040 | `inun` | ... |
| 2040 | `lnd_type` | ... |
| 2040 | `test_mnrl_accr` | ... |
| 2040 | `test_org_accr` | ... |
| 2050 | `elev` | ... |

The dataset triple contributes only its `grid` and `time_unit`; `variable` names which of
the five the caller was looking at and does **not** choose the inputs -- the same request
with `variable=elev` and with `variable=test_org_accr` expands to byte-identical pipelines.
All five are read on the named grid, which is `morph_pixel_v001` or `_v002` in practice;
`/v1/catalog.json`'s input-intersected choices are the authority on which grids carry all
five.

The pipeline is `join` -> spatial `filter` -> `unpivot`, and the order is forced. A spatial
join pairs the five rasters band-for-band and leaves five operand-named value columns on a
band-wide frame; an `unpivot` is long-only, and a spatial conjunct is the one thing that
lowers a band-wide frame to long. So the pixel selection is not merely narrowing -- it is
what makes the fold expressible. Without one the request is refused up front:

```json
{"detail": "analysis 'sediment_dynamics' reads five variables at ONE pixel -- name it with grid_cell=lng,lat. Without a pixel the five surfaces stay band-wide and there is nothing to stack them into."}
```

The operands are named after their variables on purpose: a join's output value columns take
the operand names, and the `unpivot` folds those names into the `variable` column. The
output descriptor reports the five in `value_columns` paired with their `Variable`s in
`variables`, which is how a client keys per-variable units, number formats and
`display_scale_factor` off one source.

`legend` is `null`, and has to be: a legend names one value domain and this result carries
five in three units (NAVD88 metres, metres, centimetres, plus a categorical land type). For
the same reason there is no sole presentation variable to fall back to, so
`/v1/catalog.json`'s `output` reports the five in `value_columns`/`variables`,
`folded_into: "variable"`, and `legend: null`.

This analysis declares **no parameters of its own**. `model_group_id`, `scenario_id` and
`calendar_year` are ordinary dimension filters; `grid_cell` is the required pixel.

Output is always tabular (`csv`/`ndjson`/`parquet`) -- the fold is long-only, so a raster
sink is rejected by the shape checker whatever grid the caller named.

```
GET /v1/data.csv?variable=elev&time_unit=annual&grid=morph_pixel_v002&analysis=sediment_dynamics&model_group_id=705&scenario_id=34&grid_cell=60812006
GET /v1/data.csv?variable=elev&time_unit=annual&grid=morph_pixel_v002&analysis=sediment_dynamics&model_group_id=705&scenario_id=34&grid_cell=-91.2,29.5
```

## What Can Still Be Selected: `/v1/catalog.json`

```
GET /v1/catalog.json?analysis={name}&...
```

Takes the same parameter set as the data routes and resolves the same pipeline, so "the
catalog for this request" and "the data for this request" cannot describe different things.
It answers two questions at once: what the caller may still choose (`choices`/`results`,
strictly intersected across every input the analysis reads) and what they would get
(`output`). See the [Catalog API](catalog-api.md).

This is what an analysis with fixed inputs previously had no way to ask: the catalog was
keyed on a dataset triple such an analysis could not state.

```
GET /v1/catalog.json?analysis=lnd_type_plus&model_group_id=705
```

## Seeing the Expanded Pipeline: `/v1/analyses/{analysis}.json`

```
GET /v1/analyses/{analysis}.json?...
```

Returns the exact `Pipeline` JSON a request's parameters and filters resolve to -- the same
shape a `POST /v1/data.{format}` body would submit. The `{analysis}` segment takes a single
name or a comma-separated chain, and every other parameter means exactly what it means on
`/v1/data.{format}`, which is the point: this route answers "what would
`/v1/data.csv?analysis=...&...` actually run?", and the answer is only useful if the two
decode the same request the same way. For example:

```
GET /v1/analyses/diff.json?variable=sal&time_unit=annual&grid=hydro_compartment_v002&model_group_id=700&scenario_id=34&scenario_id=35&aggregate_type=mean&calendar_year=2030&dimension=scenario_id&order=asc
```

returns (null-valued optional fields elided):

```json
{
  "inputs": {
    "x": {"variable": "sal", "time_unit": "annual", "grid": "hydro_compartment_v002"}
  },
  "steps": [
    {"op": "filter", "input": "x", "where": "aggregate_type = 'mean' AND calendar_year = 2030 AND model_group_id = 700 AND scenario_id IN (34, 35)"},
    {"op": "case", "cases": [{"when": "scenario_id = 35", "then": "value * -1"}], "else": "value"},
    {"op": "group", "by": ["model_group_id", "aggregate_type", "calendar_year"], "aggregate": "sum", "of": "value"}
  ],
  "output": {"format": "csv"}
}
```

which is exactly the hand-authored "Map Query Comparing Two Scenarios" pipeline shown in
the [Analysis API](analysis-api.md#map-query-comparing-two-scenarios) -- proving the named
analysis is sugar over the same pipeline grammar, never a second engine. This holds for
every registered analysis: `/v1/analyses/sal_veg_investigation.json?...` likewise returns
exactly the explicit paired-shape `Pipeline` shown in the
[sal_veg_investigation example](analysis-api.md#the-salinity-veg-investigation-earliest-year-marsh-and-salinity-both-cross-a-threshold).

**About `output.format` here.** A `Pipeline`'s `output.format` is required, so every
analysis's `pipeline()` sets one; on the data and tile routes that value is always
overwritten by the format in the URL, which is why the same analysis can be fetched as
`csv`, `tif`, `png` or `pbf` without declaring any of them. This route is the one with no
format segment to source an override from, so it returns the caller's own `output.format=`
if they sent one, and failing that whatever the resolved analysis declared. An analysis that
fixes a raster sink of its own (`lnd_type_plus`, `sal_veg_investigation`) therefore reports
`tif` rather than a `csv` that its own `validate_shapes` would reject.

The dataset filters land in a `filter` **step**, not on the input: an input names a
dataset and nothing else (see [Analysis API](analysis-api.md#request-shape)). The
registry builds that step in sorted conjunct order, which is why the `where` above reads
`aggregate_type ... AND calendar_year ... AND model_group_id ... AND scenario_id ...`
rather than in the order the query string listed them.

`analysis=download` -- the decoder's sentinel for "no analysis at all" -- resolves here
too, to the plain dataset read the name describes. It is not a registered analysis and does
not appear in `/v1/analyses.json`; it is the absence of a pick.
