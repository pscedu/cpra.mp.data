# Exporting a view to GIS (QGIS layer files and OGC links)

A user looking at a map in the Master Plan Data Portal can open **the same data, with the
same colors,** in QGIS. The portal does this by sending the user a link built from the
view's own query parameters. This page is for the code that builds that link.

Nothing is stored on the server: the link *is* the view. Open it later, or send it to a
colleague, and it shows the same thing.

## The short version

When the user clicks **Export**, point the browser at:

```
GET https://<api-host>/v1/view.qlr?<data request>
```

The response is a small XML file (`Content-Disposition: attachment; filename="master_plan.qlr"`).
The user opens it in QGIS (double-click it, or *Layer ▸ Add from Layer Definition File…*)
and gets one layer, already styled like the portal map.

For example, for the portal view
`#aggregate_type=mean&calendar_day=2019-01-01&grid=hydro_compartment_v001&map=8.00/29.75/-91.50&model_group_id=500&scenario_id=7&time_unit=daily&variable=sal`:

```
https://<api-host>/v1/view.qlr?variable=sal&time_unit=daily&grid=hydro_compartment_v001&model_group_id=500&scenario_id=7&aggregate_type=mean&calendar_day=2019-01-01
```

An ordinary link works (`<a href="…" download>`, or `window.location = …`). There is no need
to `fetch` it first.

## The data request

The query string after `view.qlr?` is **the same data request the portal already sends to
`/v1/data.<format>`** -- `variable`, `time_unit` and `grid` together, plus the dimension
selections (`model_group_id`, `scenario_id`, `aggregate_type`, `calendar_day` or
`calendar_year`, ...) and, for a difference or other analysis, `analysis=` and its
parameters. See the [Data API](data-api.md) and [named analyses](analyses-api.md) for what
each parameter means. For a map view that means one value per dimension.

**Forward only data parameters.** The API rejects any parameter it does not recognize with a
`400`, so view-state keys must be removed first. The portal's `map=<zoom>/<lat>/<lng>` is the
usual one:

```
GET /v1/view.qlr?variable=sal&…&map=8.00/29.75/-91.50
→ 400 {"detail": "unknown query parameter(s): ['map'] -- …"}
```

The portal's URL fragment can be turned into the request like this:

```ts
// Keys that describe the portal's own view, not the data.
const VIEW_STATE_KEYS = new Set(["map"]);

function qlrUrl(apiBase: string, fragment: string): string {
  const params = new URLSearchParams(fragment.replace(/^#/, ""));
  for (const key of VIEW_STATE_KEYS) params.delete(key);
  return `${apiBase}/v1/view.qlr?${params}`;
}
```

Repeated keys are how a multi-value selection is written (`calendar_year=2019&calendar_year=2050`),
and `URLSearchParams` preserves them.

### Differences and other analyses

A difference is requested exactly as it is for the data API:

```
…/v1/view.qlr?variable=sal&time_unit=daily&grid=hydro_compartment_v001&scenario_id=7&aggregate_type=mean&calendar_day=2019-01-01&analysis=diff&dimension=model_group_id&order=asc&model_group_id=500&model_group_id=514
```

`diff` is the analysis that has been tested for export. Other named analyses may work when
they resolve to a vector grid, but they have not been tried; any request that resolves to a
raster grid is rejected (see below), which includes `lnd_conf` and `lnd_type_plus`
because they produce raster output.

## What the layer file contains

- **The data**: an absolute URL to this API's OGC items endpoint carrying the same data
  request. QGIS fetches it each time the layer loads or refreshes, so the layer reflects the
  data as it is then, not as it was when the file was made.
- **The style**: a graduated renderer on the cell value, using the legend the portal's map
  would use -- the analysis's own legend if it has one, a red/white/blue ramp for a
  difference without one, otherwise the variable's legend.
- **A name**: for example `Salinity (model_group_id=500, scenario_id=7, aggregate_type=mean, calendar_day=2019-01-01)`.

The embedded URL is built from the `PYGEOAPI_SERVER_URL` setting, so that must be the API's
**public** base URL ending in `/v1/ogc` in every deployment, or exported files will point at
the wrong host.

## Errors

The export is validated when it is requested, the same way the data routes validate a
request, so most bad links fail at the Export click instead of inside QGIS. Failures are a
`400` with `{"detail": "<message>"}`; show the message to the user.

| Cause | Result |
| --- | --- |
| No `variable`/`time_unit`/`grid` and no `analysis=` | `400`, "no dataset selected" |
| An unrecognized parameter (for example the portal's `map`) | `400`, "unknown query parameter(s)" |
| A raster grid (`veg_grid_cell_*`, `morph_pixel_*`, ...) | `400`; only **vector** grids can be exported |
| An invalid dimension value or analysis parameter | `400` with the engine's message |

**Known gap:** a selection with *several* values for a dimension (for example two
`model_group_id`s with no `analysis=`) is accepted by the export but fails when QGIS loads
it, because a map layer holds one value per grid cell. Use `analysis=diff` to compare values,
or send one value per dimension.

## Linking to the data without a layer file

For a user who wants to add the data themselves, the same view is available as GeoJSON:

```
https://<api-host>/v1/ogc/collections/master_plan_data/items?f=json&limit=100000&<data request>
```

In QGIS: *Layer ▸ Data Source Manager ▸ Vector*, *Source Type* **Protocol: HTTP(S), cloud,
etc.**, paste the URL into *URI*. The layer arrives in one flat color; the user styles it by
`value` themselves. `limit` must be at least as large as the grid's feature count, otherwise
the layer is silently truncated (the default is 50), so always send `limit=100000`. Prefer
the `.qlr` link in the portal.

## What is not supported

- **Browsing the API from QGIS's own OGC dialogs** (*WFS / OGC API - Features*, or ArcGIS's
  OGC API connector). Those expect a catalog of collections; this API serves one view per
  request. Connecting to it directly returns a `400` explaining that the user should use
  Export in the portal.
- **Raster grids.**
- **Live styling updates**: if the portal's legends change, files already exported keep the
  legend they were made with.

## Payload size

The items response carries full-resolution cell geometry, and the geometry is the same for
every view of a grid. For `hydro_compartment_v001` (1,657 cells) that is about **23 MB**,
uncompressed. The `.qlr` itself is a few kilobytes. Expect a few seconds to load on a slow
connection.

## Related

- Implementation: `cpra/mp/data/api/qlr.py` (the file builder),
  `cpra/mp/data/api/ogc_provider.py` (the items endpoint), and the `/v1/view.qlr` route in
  `cpra/mp/data/api/__init__.py`.
- OGC server configuration: `cpra/mp/data/ogc_config.yaml`.
