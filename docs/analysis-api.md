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

A small, fixed set of named, pre-built pipelines (`diff`, `lnd_chng`,
`sal_veg_investigation`) is also available without hand-authoring pipeline JSON at
all -- see [the named-analysis registry](analyses-api.md).

Four endpoints accept a pipeline, sharing one request model:

```
POST /v1/analysis.<csv|ndjson|parquet|tif>
GET  /v1/analysis.<csv|ndjson|parquet|tif>
GET  /v1/analysis/{z}/{x}/{y}.<png|jpg>
GET  /v1/analysis/{z}/{x}/{y}.pbf
```

`POST` takes the pipeline as a JSON body. `GET` (the flat form and both tile forms) takes
the identical pipeline flattened into a dotted query string, so a pipeline can be
embedded in a map tile URL template. On the `.{format}` routes, `<format>` in the URL must
agree with the pipeline's own `output.format` -- a mismatch (e.g. a `.csv` URL for a
pipeline whose `output.format` is `"tif"`) is rejected with a 400 before anything is read.
The `.pbf` vector-tile route is the exception: `pbf` is not a spellable `output.format`,
so there is nothing to cross-check; that route instead requires the pipeline's final
frame to be a table on a vector grid, and rejects anything else with a 400.

## Request Shape

A pipeline has three parts:

```json
{
  "inputs": {"<name>": {"variable": "...", "time_unit": "...", "grid": "..."}},
  "steps": [ {"op": "...", ...} ],
  "output": {"format": "...", "color_map": "...", "sort": "...", "select": "..."}
}
```

**`inputs`** names one or more dataset selections. An input names a dataset and nothing
else: exactly the `variable`/`time_unit`/`grid` triple, with no other keys accepted. There
is no input-level filtering of any kind -- every narrowing, whether by dimension
(`scenario_id = 7`), by value (`value > 5`), or by cell (`grid_cell IN (1, 2)`), is a
`filter` **step**, so a request narrows its result the same way no matter where in the
chain the narrowing happens. (This is a change: earlier versions accepted dimension
filters and a `grid_cell` key directly on an input. Those keys are now rejected outright
rather than silently ignored; move each one into a `filter` step.)

**`steps`** is an ordered list of operations, each consuming a named input or a previous
step's output:

| Op | Fields | Description |
|---|---|---|
| `project` | `input` (optional, chains from the previous step if omitted), `to` (target grid), `aggregate` (required whenever the target may draw from several source cells -- there is no default; forbidden only when broadcasting a vector grid onto a raster one; `sum`/`max`/`min`/`mean`/`count`/`count_by_value`), `as` | Move a frame onto a different grid via a crosswalk. |
| `join` | `type` (`"spatial"`), `inputs` (two or more names), `as` | Combine frames cell-for-cell on a shared grid, naming each operand's measurement after the operand. |
| `filter` | `input` (optional), `where`, `as` | A SQL-style predicate -- over dimensions, over the value, or over `grid_cell`. |
| `calculate` | `input` (optional), `expression`, `as` | **Unconditionally** rewrite the running value: `expression` is applied to every row/band. It carries no `where` of its own -- it is exactly a `case` with only an `else`, and is lowered to one. A rewrite that should apply only *sometimes* is a `case` with an arm. |
| `case` | `input` (optional), `cases` (a non-empty list of `{when, then}` arms), `else` (optional), `as` | Rewrite the running value by the **first matching** arm, SQL `CASE WHEN` style. Every `when` is read against the *incoming* value, so the arms are independent of one another and each row/pixel is written exactly once. Anything matching no arm takes `else`, which defaults to null -- i.e. masked, so **a pass-through is `else: "value"`**, written out rather than assumed. **Nodata is yours to state:** a null matches no `when` (every predicate reads null as false), so with no `else` it stays null, but `else 0` really does paint every nodata row/pixel `0`. Arms may test nullity directly (`x IS NULL`, `x IS NOT NULL`), and `coalesce` is available in a `then`/`else`. An arm's `when` may also be a **dimension** predicate, or mix dimension and value conjuncts (`scenario_id = 8 AND value > 0`); the dimension half picks which *bands* (or rows) the arm applies to, and an arm excluded from a band simply isn't in that band's chain, leaving the remaining arms and the `else` to write it. |
| `group` | `input` (optional), `by` (dimensions to group by; `[]` collapses to one value per `grid_cell`), `aggregate` (`sum`/`min`/`max`/`count`/`mean` -- the decomposable aggregates only; `count_by_value` is never valid here), `of` (defaults to `"value"`; a numeric dimension such as `calendar_year` is also valid, for "first year a condition held"), `as` | Collapse rows/bands. |

Every op may name its input explicitly; when it doesn't, it consumes whatever the
immediately preceding step produced (`join` is the exception -- its `inputs` is required,
so it never chains). The **first** step must name its input, whichever op it is: a
single-input pipeline that changes no grid opens with a `filter` naming the input, or
with a `group` naming it if collapsing is the only thing it does. There is no identity
`project` to open with: a `project` onto the frame's own grid is not a no-op, and a
vector-to-vector `project` with no `aggregate` is rejected outright, since a target cell
may draw from any number of source cells and there is no default for that.

**A pipeline may have zero steps.** A plain selection -- nothing to compute -- is a
pipeline with an empty `steps` list, exactly one input, and an `output`.

### Naming and the Column Namespace

Input names and step `as` names must match `[a-z_][a-z0-9_]*`, must not collide with each
other, and must not be `value`, `grid_cell`, `value_category`, a positional band name
(`b0`, `b1`, ...), or any dimension the data structure already declares.

The running measurement is spelled **`value`** in every `filter.where`,
`calculate.expression`, and `case` arm, in every single-input chain. The one exception is a frame
produced by a `join`: there, each operand's measurement is named after the operand, so a
`join` of `marsh` and `sal` is filtered with `marsh > 0.5 AND sal > 5.5`. Referring to a
single-input chain's measurement by the input's own name does *not* work -- that name is
classified as a dimension, and fails against the real dimension set.

A conjunct that references `grid_cell` is a spatial predicate; one that references a value
column is a value predicate; anything else is a dimension predicate. A single conjunct may
not mix `grid_cell` with a value column. Otherwise a `where` can freely combine all three
kinds in one string (`calendar_year > 2040 AND value > 3`).

### Output

**`output`** gives the response `format`, an optional `color_map` for rendered raster
output, and optional `sort`/`select`, with exactly the same semantics as the flat data
endpoints' `sort`/`select` parameters. `sort`/`select` are checked for syntax at parse
time (comma-separated `[-]column`, no empty component, no bare `-`, no `-` prefix in
`select`, no repeats); whether a named column exists on the final frame surfaces when the
result is collected.

Compression is **not** part of the pipeline body -- it is a pair of query parameters,
`compression_algorithm` (default `zstd`) and `compression_level` (default `6`), accepted
on both `POST /v1/analysis.{format}` and `GET /v1/analysis.{format}`. For `tif` output the
algorithm must be one of `deflate`/`lzw`/`lzma`/`zstd`/`none`; for tabular output, one of
`gzip`/`lz4`/`zstd`/`none`. There is no `output.compression` field -- an earlier version
accepted one and then ignored it; it is now rejected like any other unknown output field.

GeoTIFF output carries a band description per band, spelled `"<dimension>=<value>"` and
comma-joined, so a downloaded file records which band is which without relying on band
order.

## Shapes: What the Compiler Accepts

Every frame in a pipeline has one of two shapes, and the shape is what decides which
lowering each op gets:

- **long** -- a table, one row per (`grid_cell`, dimension combination), with a `value`
  column. This is the shape of every vector-grid input and of every tabular output.
- **band-wide** -- a raster surface, one band per surviving dimension combination, with
  no rows. This is the shape of every raster-grid input and of every `tif`/`png`/`jpg`
  output.

The rules, in full:

| Op | Effect on shape |
|---|---|
| `project` vector -> vector | long -> long (`aggregate` required) |
| `project` raster -> vector | band-wide -> long (`aggregate` required) |
| `project` vector -> raster | long -> band-wide (a broadcast; `aggregate` forbidden) |
| `project` raster -> raster | **never legal** -- there is no crosswalk between two raster lattices |
| `filter` on a dimension or a value | shape unchanged (and value columns unchanged -- a `filter` never changes a frame's pairing) |
| `filter` on `grid_cell` (spatial) | band-wide -> long; long -> long |
| `calculate` | shape unchanged (it lowers to a `case`, so the row below applies to it too) |
| `case` | shape unchanged (and always one value column out, so it collapses a pairing) |
| `group` | shape unchanged |
| `join` | shape unchanged; operands must agree on shape **and** grid |

Two consequences of the spatial-filter row are worth calling out. A `grid_cell` filter
over a raster frame is what turns it into a table, which is how a raster dataset reaches a
CSV without a zonal `project`. And because that leaves the frame still addressed against
the raster lattice, a later `project` from it is still raster-to-something.

There is no longer any restriction on step *ordering* beyond the chaining rule: a `filter`
after a `calculate` sees the post-calculate value, a `grid_cell` filter may appear anywhere
and as many times as you like, and `filter`/`calculate`/`case`/`group` compose in any
sequence.
Every combination this table permits has an executor -- nothing is accepted at parse time
and then refused as unimplemented at run time.

### What Is Rejected, and Why

Each of these is a 400 naming the specific problem:

- **A raster-to-raster `project`** (above).
- **A `project` with no `aggregate`** where the target may draw from several source cells,
  or **with** one where it may not (the vector-to-raster broadcast).
- **A second `count_by_value` in one chain** -- the first already occupies the
  `value_category` column. `count_by_value` is a `project` aggregate only; it is never a
  valid `group.aggregate`, because it changes the output's row count rather than folding.
- **A `join` mixing a long operand with a band-wide one**, or **whose operands sit on two
  different grids** -- `grid_cell` identifies a cell *within* one grid, so two frames on
  different grids have no cell in common. Insert a `project` first.
- **A `join` over an operand that is itself an uncollapsed `join`'s output** -- a join
  names each operand's measurement after the operand, and an operand carrying several
  measurements has no single one for that name to mean. `join(a, b)` is fine;
  `join(join(a, b), c)` is not, until something collapses the pairing in between.
- **A value `filter` naming an operand of a paired band-wide frame** -- a band-wide frame's
  bands are positional, so there is no per-operand column to name. Evaluating
  `marsh > 0.5 AND sal > 5.5` over positional
  bands means pairing them and emitting one band per pair, which reduces the frame and has
  to pick which operand's value survives -- a choice the predicate never states. A
  `case`/`calculate` states it, in `then`/`expression`. (A value `filter` naming only a
  *carried* column, such as `__domain__ > 0`, is fine over a paired frame: it is a
  per-pixel mask each band takes independently, so nothing pairs and nothing is picked.)
- **A `case` arm whose `when` is a `grid_cell` predicate** -- an arm rewrites the value at
  a row/pixel it is already looking at, while a spatial predicate selects which cells there
  are at all. State that as a `filter`, before or after the `case`. (A *dimension*
  predicate in a `when` is fine -- see the `case` row of the ops table.)
- **More than one value column reaching the sink** -- every output writes exactly one
  measured quantity. A `join`'s paired frame must therefore be collapsed before the
  output, by a `case`, a `calculate`, or a `group`.
- **A long final frame under a raster output format** (`tif`/`png`/`jpg`), or **a
  band-wide final frame under a tabular one** (`csv`/`ndjson`/`parquet`). Add a `project`,
  or (from band-wide) a spatial `filter`.
- **`join.type: "attribute"`** -- reserved for a future release; only `"spatial"` is
  implemented.
- **A first step that names no input** (any op with `input` omitted), or any step
  referencing a name that isn't yet defined.
- **A zero-step pipeline with more than one input** -- nothing selects among several
  inputs with no step to combine or choose between them.

Two further rejections need information only available once compilation has read the
catalog, so they surface slightly later but are still 400s: a `project` from a raster
frame that an earlier spatial `filter` already turned long, and one raster input fanning
out into two raster-to-vector `project`s with different targets.

### Limits

The vector (zone) side of a `project`/spatial `join` is capped at a configured row ceiling
(`ANALYSIS_JOIN_MAX_ROWS`, default 200,000), checked before that side is materialized, so
an unbounded selection is rejected rather than silently read in full. `count_by_value` is
capped at `COUNT_BY_VALUE_MAX_DISTINCT` (default 1,000) distinct values, which catches a
caller pointing it at a continuous column.

## Memory Behavior

The paired shape -- a vector dataset broadcast onto a raster grid and joined against a
raster dataset there -- exists specifically to avoid materializing a full joined table:
naively broadcasting a vector dataset onto every pixel of a raster grid, across every
year, and then grouping, means holding one column per year for every pixel in memory at
once before the aggregate can drop most of it. Instead, the compiled plan reads both
inputs' small metadata up front (file paths/bands for the raster input; the already
row-ceiling-bounded vector table), then executes as a single pass over the raster's own
block windows: for each window, it reads that window from every year's band plus the
pairing crosswalk, looks up each pixel's paired vector value, applies the `case`/`group`
collapse for just that window, and writes the collapsed window straight back at its own
position -- never holding a full-extent array of any per-pixel intermediate.

## Examples

### Vector-Grid Pipelines

#### Map Query Comparing Two Scenarios

This pipeline returns the difference in annual salinity (two-week-max aggregate) between
two scenarios in one year (scenario 7 minus scenario 8): a `filter` selecting the two
scenarios, a `case` step negating scenario 8's rows, then a `group` collapsing the
two scenarios into one value per grid cell. The `case`'s `else` is `"value"` -- scenario
7's rows have to reach the `sum` as themselves, and a `case` nulls whatever no arm matched
unless the pass-through is written out. `model_group_id`, `aggregate_type`, and
`calendar_year` are each pinned to one value here (so listing them in `group.by` alongside
the truly-varying `scenario_id` is a no-op on this particular request), because `sal`
always carries an `aggregate_type` dimension at every time granularity -- pin it
explicitly rather than leaving it to match every stored value.

```json
POST /v1/analysis.csv
{
  "inputs": {
    "sal": {"variable": "sal", "time_unit": "annual", "grid": "hydro_compartment_v001"}
  },
  "steps": [
    {"op": "filter", "input": "sal", "where": "model_group_id = 520 AND scenario_id IN (7, 8) AND aggregate_type = '2_wk_max' AND calendar_year = 2030"},
    {"op": "case", "cases": [{"when": "scenario_id = 8", "then": "value * -1"}], "else": "value"},
    {"op": "group", "by": ["model_group_id", "aggregate_type", "calendar_year"], "aggregate": "sum"}
  ],
  "output": {"format": "csv"}
}
```

This exact `case`(negate)+`group`(sum) idiom is also available as the `diff`
named analysis (`dimension=scenario_id`, `order=asc` negates the later of the two
selected values, here `8`) -- see [the registry doc](analyses-api.md#diff) for the
equivalent request with no hand-authored pipeline. The named analysis derives its own
`group.by` as "every required dimension of the dataset except the one being differenced,"
which is why it lists `model_group_id`/`aggregate_type`/`calendar_year` explicitly rather
than assuming they're already pinned to one value each.

#### Query with Additional Dimensions

This pipeline returns decadal damage for a single grid cell. Since `aep`, `asset_type`,
and `percentile` are not filtered, the query matches all available values for those
dimensions; the results are summed across `asset_type` (the one dimension not listed in
`group.by`) to give the total damage for each AEP-percentile combination. The cell
restriction is an ordinary `filter` on `grid_cell`, which may be folded into the dimension
filter or kept separate as here.

```json
POST /v1/analysis.ndjson
{
  "inputs": {
    "damage": {"variable": "damage", "time_unit": "decadal", "grid": "clara_grid_cell_v001"}
  },
  "steps": [
    {"op": "filter", "input": "damage", "where": "model_group_id = 500 AND scenario_id = 7 AND pumping_id = 0 AND fragility_scenario = 0"},
    {"op": "filter", "where": "grid_cell = 12"},
    {"op": "group", "by": ["aep", "percentile"], "aggregate": "sum"}
  ],
  "output": {"format": "ndjson", "sort": "aep,percentile"}
}
```

#### Conditional Count

This pipeline counts, per grid cell, how many decadal damage values exceed a threshold: a
value `filter` drops non-matching rows, and `group`'s `count` aggregate counts what's
left.

```json
POST /v1/analysis.csv
{
  "inputs": {
    "damage": {"variable": "damage", "time_unit": "decadal", "grid": "clara_grid_cell_v001"}
  },
  "steps": [
    {"op": "filter", "input": "damage", "where": "model_group_id = 500 AND scenario_id = 7"},
    {"op": "filter", "where": "value > 1000000"},
    {"op": "group", "by": [], "aggregate": "count"}
  ],
  "output": {"format": "csv"}
}
```

#### Single-Input Pipeline Equivalent to a Flat Request

One `filter` mixing dimension and value conditions, then a `group`:

```json
POST /v1/analysis.csv
{
  "inputs": {
    "sal": {"variable": "sal", "time_unit": "daily", "grid": "hydro_compartment_v001"}
  },
  "steps": [
    {"op": "filter", "input": "sal", "where": "model_group_id = 520 AND scenario_id = 7 AND value > 5"},
    {"op": "group", "by": ["calendar_day"], "aggregate": "max"}
  ],
  "output": {"format": "csv"}
}
```

A plain selection with no transformation at all is a pipeline with no steps:

```json
POST /v1/analysis.csv
{
  "inputs": {
    "sal": {"variable": "sal", "time_unit": "annual", "grid": "hydro_compartment_v001"}
  },
  "steps": [],
  "output": {"format": "csv", "sort": "calendar_year,grid_cell"}
}
```

#### Single-Input Pipeline with a Weighting Case: a Confidence-Weighted Collapse

Two model groups weighted 0.6/0.4 by one `case` -- an arm apiece, each `when` a dimension
predicate picking out that group's rows -- collapsed with `group`/`sum`. One step rather
than two sequential rewrites: the arms all read the *incoming* value, so each row is
written exactly once and there is no ordering to get wrong. No `else`, because the leading
`filter` already pinned `model_group_id` to exactly these two and nothing reaches it:

```json
POST /v1/analysis.csv
{
  "inputs": {
    "sal": {"variable": "sal", "time_unit": "annual", "grid": "hydro_compartment_v001"}
  },
  "steps": [
    {"op": "filter", "input": "sal", "where": "model_group_id IN (500, 510) AND scenario_id = 7 AND aggregate_type = '2_wk_max'"},
    {"op": "case", "cases": [
      {"when": "model_group_id = 500", "then": "value * 0.6"},
      {"when": "model_group_id = 510", "then": "value * 0.4"}
    ]},
    {"op": "group", "by": ["calendar_year"], "aggregate": "sum"}
  ],
  "output": {"format": "csv"}
}
```

The same pipeline in the dotted `GET` encoding spells the arms positionally --
`step.2.op=case&step.2.cases.1.when=model_group_id%20%3D%20500&step.2.cases.1.then=value%20*%200.6`
-- and works on the tile route the same way.

### Raster-Grid Pipelines

#### Basic Map Query

For a plain raster selection with no transformation, use the [Data API](data-api.md)
directly -- no pipeline is needed:

```
GET /v1/datasets/elev/annual/morph_pixel_v001/data/{z}/{x}/{y}.png?model_group_id=500&scenario_id=7&calendar_year=2045&color_map=elevation
```

#### Map Query Comparing Two Years

This pipeline returns the colorized elevation difference between two years (2050 minus
2045): a `filter` selecting the two years, a `case` step negating 2045's band, then a
`group` collapsing the two bands into one per pixel, rendered on the analysis tile route.
Each step here leaves the frame band-wide, which is what the `png` output requires. The
arm's `when` is purely dimensional, so it picks out the 2045 band and negates all of it;
`else: "value"` leaves 2050's band as itself, which is what the `sum` needs to subtract
from.

```json
{
  "inputs": {
    "elev": {"variable": "elev", "time_unit": "annual", "grid": "morph_pixel_v001"}
  },
  "steps": [
    {"op": "filter", "input": "elev", "where": "model_group_id = 500 AND scenario_id = 7 AND calendar_year IN (2045, 2050)"},
    {"op": "case", "cases": [{"when": "calendar_year = 2045", "then": "value * -1"}], "else": "value"},
    {"op": "group", "by": [], "aggregate": "sum"}
  ],
  "output": {"format": "png", "color_map": "elevation_difference"}
}
```

As a `GET` request to `/v1/analysis/{z}/{x}/{y}.png`, this pipeline is flattened into the
same dotted query encoding shown in the sal_veg_investigation example below
(`input.elev.variable=elev&...&step.1.op=filter&...&output.format=png&output.color_map=elevation_difference`);
there is no `POST` form for the tile route.

#### Confidence-Weighted Surface Across Model Groups

This pipeline builds a confidence-weighted elevation surface: three model groups are each
multiplied by their own confidence coefficient (0.6, 0.3, 0.1) by one `case` with three
arms -- each `when` a dimension predicate selecting that model group's band -- then
collapsed to one band per pixel with `group`/`sum`. The arms are independent (each matches
a different model group), so this is a per-band weighting; a pixel that is NoData in one
group's band simply contributes nothing from that group. No `else`: the leading `filter`
pinned `model_group_id` to exactly these three, so there is no unweighted band to pass
through.

```json
{
  "inputs": {
    "elev": {"variable": "elev", "time_unit": "annual", "grid": "morph_pixel_v001"}
  },
  "steps": [
    {"op": "filter", "input": "elev", "where": "model_group_id IN (500, 510, 520) AND scenario_id = 7 AND calendar_year = 2050"},
    {"op": "case", "cases": [
      {"when": "model_group_id = 500", "then": "value * 0.6"},
      {"when": "model_group_id = 510", "then": "value * 0.3"},
      {"when": "model_group_id = 520", "then": "value * 0.1"}
    ]},
    {"op": "group", "by": [], "aggregate": "sum"}
  ],
  "output": {"format": "png", "color_map": "elevation"}
}
```

The same pipeline works on `.tif` output too (drop the tile route, `POST` this to
`/v1/analysis.tif` with `output.format` set to `"tif"`).

#### Single Year Summarized by Hydro Compartment

Elevation for a single year summarized by hydro compartment as a Parquet file -- the
`project` is raster-to-vector, so it both aggregates and turns the frame into a table,
which is what makes a tabular output legal here:

```json
POST /v1/analysis.parquet
{
  "inputs": {
    "elev": {"variable": "elev", "time_unit": "annual", "grid": "morph_pixel_v001"}
  },
  "steps": [
    {"op": "filter", "input": "elev", "where": "model_group_id = 500 AND scenario_id = 7 AND calendar_year = 2050"},
    {"op": "project", "to": "hydro_compartment_v001", "aggregate": "mean"}
  ],
  "output": {"format": "parquet", "sort": "calendar_year,grid_cell"}
}
```

#### Conditional Pixel Count per Zone

This pipeline counts, per hydro compartment, the pixels whose elevation exceeds a
threshold in a single year: a value `filter` masks the non-matching pixels, then a
`project` with `aggregate: "count"` counts what survives in each zone. The filter must
precede the project -- after it, `value` is the already-counted zone total.

```json
POST /v1/analysis.csv
{
  "inputs": {
    "elev": {"variable": "elev", "time_unit": "annual", "grid": "morph_pixel_v001"}
  },
  "steps": [
    {"op": "filter", "input": "elev", "where": "model_group_id = 500 AND scenario_id = 7 AND calendar_year = 2050"},
    {"op": "filter", "where": "value > 2"},
    {"op": "project", "to": "hydro_compartment_v001", "aggregate": "count"}
  ],
  "output": {"format": "csv"}
}
```

#### Categorical Count per Zone

This pipeline counts, per hydro compartment, the pixels carrying each distinct land-type
value in a single year, using `count_by_value` as the `project` aggregate. The output has
one row per (zone, value) pair, with the category in a `value_category` column -- which a
later `group` may itself group by.

```json
POST /v1/analysis.csv
{
  "inputs": {
    "lnd_type": {"variable": "lnd_type", "time_unit": "annual", "grid": "morph_pixel_v001"}
  },
  "steps": [
    {"op": "filter", "input": "lnd_type", "where": "model_group_id = 500 AND scenario_id = 7 AND calendar_year = 2050"},
    {"op": "project", "to": "hydro_compartment_v001", "aggregate": "count_by_value"}
  ],
  "output": {"format": "csv", "sort": "grid_cell,value_category"}
}
```

#### Raster Values for Specific Pixels

A spatial `filter` is the other way out of band-wide shape: it turns the raster frame into
a table of just the named cells, with no zonal aggregation at all.

```json
POST /v1/analysis.csv
{
  "inputs": {
    "elev": {"variable": "elev", "time_unit": "annual", "grid": "morph_pixel_v001"}
  },
  "steps": [
    {"op": "filter", "input": "elev", "where": "model_group_id = 500 AND scenario_id = 7 AND calendar_year = 2050"},
    {"op": "filter", "where": "grid_cell IN (1, 2, 3)"}
  ],
  "output": {"format": "csv"}
}
```

#### Broadcasting a Vector Dataset onto a Raster Grid

The reverse direction: render a hydro-compartment salinity value as a full-resolution
raster surface on the vegetation grid. No `aggregate` is given (nor allowed) -- each
target pixel draws from exactly one source zone.

```json
POST /v1/analysis.tif
{
  "inputs": {
    "salinity": {"variable": "sal", "time_unit": "annual", "grid": "hydro_compartment_v001"}
  },
  "steps": [
    {"op": "filter", "input": "salinity", "where": "aggregate_type = '2_wk_max' AND model_group_id = 500 AND scenario_id = 7 AND calendar_year = 2050"},
    {"op": "project", "to": "veg_grid_cell_v001"}
  ],
  "output": {"format": "tif"}
}
```

### Multi-Input (Paired) Pipelines

#### The Salinity-Veg Investigation: Earliest Year Marsh and Salinity Both Cross a Threshold

This pipeline broadcasts annual salinity (by hydro compartment) onto the vegetation grid,
pairs it against marsh coverage there, keeps only pixel-years where marsh coverage
exceeds 0.5 *and* paired salinity exceeds 5.5, and reports the earliest (`min`) such
`calendar_year` per pixel, as a GeoTIFF.

Note the shape of the chain: each input is narrowed by its own `filter` (named with `as`
so the `join` can reference it), the vector input is projected onto the raster grid, the
`join` pairs the two -- leaving a frame with two value columns, `marsh` and `sal` -- and
the single-armed `case` naming both is what collapses that pairing back to a single
measured quantity, which the output requires. It has no `else`, so a pixel-year failing
either threshold comes out null and the `group` below skips it; its `then` keeps `marsh`,
naming which of the two operands the surviving band carries. Nothing here reads that
value: `min of=calendar_year` asks only *whether* the band is present, and `of` supplies
*which* year to report.

**As an explicit pipeline:**

```json
POST /v1/analysis.tif
{
  "inputs": {
    "coverage": {"variable": "ffibs_coverage_ratio", "time_unit": "annual", "grid": "veg_grid_cell_v001"},
    "salinity": {"variable": "sal", "time_unit": "annual", "grid": "hydro_compartment_v001"}
  },
  "steps": [
    {"op": "filter", "input": "coverage", "where": "ffibs_type = 'pl_fm' AND model_group_id = 500 AND scenario_id = 7", "as": "marsh"},
    {"op": "filter", "input": "salinity", "where": "aggregate_type = '2_wk_max' AND model_group_id = 500 AND scenario_id = 7", "as": "selected_salinity"},
    {"op": "project", "input": "selected_salinity", "to": "veg_grid_cell_v001", "as": "sal"},
    {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
    {"op": "case", "cases": [{"when": "marsh > 0.5 AND sal > 5.5", "then": "marsh"}]},
    {"op": "group", "by": [], "aggregate": "min", "of": "calendar_year"}
  ],
  "output": {"format": "tif"}
}
```

The same pipeline as a `GET` request, using the dotted query encoding (dot = nesting, an
integer segment = a 1-based list position, comma = a list value -- never inside `where`,
which is one opaque string). A list of objects rather than of scalars -- `case.cases` is
the only one -- is spelled positionally instead of comma-joined, for the same reason:
`step.3.cases.1.when=value%20%3E%201.0&step.3.cases.1.then=0&step.3.cases.2.when=...`.

```
GET /v1/analysis.tif?input.coverage.variable=ffibs_coverage_ratio&input.coverage.time_unit=annual&input.coverage.grid=veg_grid_cell_v001&input.salinity.variable=sal&input.salinity.time_unit=annual&input.salinity.grid=hydro_compartment_v001&step.1.op=filter&step.1.input=coverage&step.1.where=ffibs_type%20%3D%20%27pl_fm%27%20AND%20model_group_id%20%3D%20500%20AND%20scenario_id%20%3D%207&step.1.as=marsh&step.2.op=filter&step.2.input=salinity&step.2.where=aggregate_type%20%3D%20%272_wk_max%27%20AND%20model_group_id%20%3D%20500%20AND%20scenario_id%20%3D%207&step.2.as=selected_salinity&step.3.op=project&step.3.input=selected_salinity&step.3.to=veg_grid_cell_v001&step.3.as=sal&step.4.op=join&step.4.type=spatial&step.4.inputs=marsh,sal&step.4.as=paired&step.5.op=case&step.5.cases.1.when=marsh%20%3E%200.5%20AND%20sal%20%3E%205.5&step.5.cases.1.then=marsh&step.6.op=group&step.6.by=&step.6.aggregate=min&step.6.of=calendar_year&output.format=tif
```

And as map tiles, replacing `output.format=tif` with `output.format=png`:

```
GET /v1/analysis/{z}/{x}/{y}.png?...same query as above, with output.format=png
```

**As the `sal_veg_investigation` named analysis** -- the same request, with the
`Pipeline` above built by the registry rather than hand-authored:

```
GET /v1/analyses/sal_veg_investigation/data.tif?model_group_id=500&scenario_id=7&marsh_threshold=0.5&salinity_threshold=5.5
```

`sal_veg_investigation` is a `type="custom"` analysis (its two inputs are fixed by its own
`pipeline()` method, not supplied by a dataset in the URL -- see
[the registry doc](analyses-api.md#sal_veg_investigation)), so this route has no
`/v1/datasets/{variable}/{time_unit}/{grid}/` prefix. Fetching
`GET /v1/analyses/sal_veg_investigation/pipeline.json?model_group_id=500&scenario_id=7&marsh_threshold=0.5&salinity_threshold=5.5`
returns exactly the explicit `Pipeline` JSON shown at the top of this section (with
`model_group_id`/`scenario_id` substituted from the request) -- proving the named
analysis is sugar over the same pipeline grammar, never a second engine.
