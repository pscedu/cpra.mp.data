"""The flat data routes' one piece of request-to-`Pipeline` work that
`request.data_query.build_data_query_pipeline` deliberately cannot do for itself:
resolving a coordinate-valued `grid_cell` against real grid geometry, which needs `io/`
and therefore cannot live under `request/` (see that module's docstring, and
`tests/api/test_analysis_layering.py`).

`build_data_query` is what the three `/v1/datasets/{variable}/{time_unit}/{grid}/...`
routes call instead of the old engine's `create_analysis`: it resolves `grid_cell`, hands
the resulting conjunct to `build_data_query_pipeline`, and returns the `Pipeline` together
with the two extras a route needs to plumb into its response but cannot recover from the
pipeline alone.

**Why `grid_cell` resolution is grid-kind-dependent, and exactly how.** The two grid kinds
have always disagreed about what a `grid_cell` value MEANS, and this module reproduces that
split rather than unifying it -- unifying it would silently change what existing URLs
select:

- A VECTOR grid accepts either form. Every token a plain (unsigned, non-decimal) digit
  string means literal polygon ids; anything else is parsed as coordinate pairs and
  resolved by point-in-polygon lookup. That is `analysis.vector.apply_grid_cell_selection`'s
  own rule, kept token-for-token. The digit case is not resolved here at all -- it is left
  to `build_data_query_pipeline`'s own `_grid_cell_conjunct`, so "these are literal ids"
  has exactly one implementation.
- A RASTER grid accepts ONLY coordinates, with no digit check of any kind:
  `analysis.raster.sample_band_ops_to_long` calls `parse_coordinate_pairs` on the raw
  tokens unconditionally, so `grid_cell=1,2` on a raster has always meant the point at
  lng=1, lat=2 -- a degenerate coordinate pair, not pixel id 1 and pixel id 2. Reading
  digits as pixel ids here (as a vector grid would) would reinterpret such a URL rather
  than merely extend it, so the raster branch resolves every value as a coordinate.

A pipeline request (`/v1/analysis...`) is the place to name raster pixel ids directly: a
`filter` step's `where` takes the absolute `grid_cell` ids `io.geography.
raster_grid_cell_id` computes, with no coordinate convention in the way.

**Out-of-extent points are dropped, not rejected**, for both grid kinds -- a coordinate
outside every polygon (vector) or outside the raster's extent (raster) contributes no id,
and a request where EVERY point resolves to nothing becomes `grid_cell_id_where`'s
"matched nothing" conjunct and answers an empty result. That is `resolve_vector_grid_cell`'s
long-standing convention, extended to the raster side; see `io.geography.
resolve_raster_point_labels`'s docstring. A coordinate that is not numeric, not an even
count, not valid WGS84, or not projectable into the grid's CRS is still a 400 -- those are
rejected by `io.geography.parse_coordinate_pairs`/`project_coordinates` before either
grid kind's lookup begins.
"""

from typing import NamedTuple

from cpra.mp.data.analysis.io.geography import (
    grid_cell_id_where,
    parse_coordinate_pairs,
    resolve_raster_point_labels,
    resolve_vector_grid_cell,
)
from cpra.mp.data.analysis.request.data_query import build_data_query_pipeline
from cpra.mp.data.analysis.request.errors import NotFoundError
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.variables import VARIABLES


class ResolvedDataQuery(NamedTuple):
    """A flat data query, lowered: the `Pipeline` itself plus the two things a route has to
    carry separately into its response because the `Pipeline` does not record them.

    `relabel` is `io.geography.resolve_raster_point_labels`'s absolute-id -> coordinate-
    string map for a raster point query, and `None` for everything else. It cannot live on
    the `Pipeline`: the pipeline's `where` names integer ids (that is what the executor
    filters on), and the caller's own coordinate strings are a presentation concern applied
    to the RESULT -- see `execution.execute.run_to_path`'s `relabel` argument.

    `float_precision` is the query's one variable's declared CSV rounding, which a
    `Pipeline` has no field for either (`request.pipeline.Output` carries `format`/
    `color_map`/`sort`/`select`/`compression` and nothing about number formatting). A flat
    data query always has exactly one variable, so unlike a general pipeline it always has
    exactly one answer here -- see `api.pipeline.PipelineTabularSource`.

    Both are `None`-by-default plumbing rather than behavior: a route that ignores them
    (the two tile routes, neither of which produces text floats or a `grid_cell` column)
    is not skipping a step.
    """

    pipeline: object
    relabel: dict[int, list[str]] | None
    float_precision: int | None


async def resolve_grid_cell(grid_name: str, grid_cell: str | None):
    """Resolve a flat route's raw `grid_cell` query value against `grid_name`'s geometry,
    returning `(grid_cell_where, relabel)` for `build_data_query_pipeline`/`run_to_path` --
    see this module's docstring for the per-grid-kind rules this implements.

    A `grid_cell_where` of `None` means "this function resolved nothing; leave
    `options.grid_cell` to `build_data_query_pipeline`" -- returned both when the caller
    set no `grid_cell` at all and for a vector grid's digit-id form, which that function
    already handles and which nothing here could improve on.

    Async only for the vector branch: `resolve_vector_grid_cell` runs one windowed
    FlatGeoBuf read per point in an executor. The raster branch opens no file at all
    (`io.geography.resolve_raster_pixel`) and is synchronous inside this coroutine.
    """
    if not grid_cell:
        return None, None

    grid = GRIDS.get(grid_name)
    if grid is None:
        raise NotFoundError(f"unknown grid: {grid_name!r}")

    tokens = [token.strip() for token in grid_cell.split(",")]

    if grid.grid_type == "raster":
        labels = resolve_raster_point_labels(grid_name, tokens)
        return grid_cell_id_where(list(labels)), labels

    if all(token.isdigit() for token in tokens):
        return None, None

    lngs, lats = parse_coordinate_pairs(tokens)
    resolved = await resolve_vector_grid_cell(grid_name, lngs, lats)
    return grid_cell_id_where([cell_id for cell_id in resolved if cell_id is not None]), None


async def build_data_query(
    dataset_dimensions, filter_dimensions, options, format: str
) -> ResolvedDataQuery:
    """Lower one flat data/tile request onto a `Pipeline`, resolving `grid_cell` first.

    Runs only `build_data_query_pipeline`'s own STRUCTURAL validation (see that function)
    -- a caller still runs `request.ops.validate_shapes` on the returned pipeline before
    handing it to a responder, exactly as the `/v1/analysis...` routes do for a pipeline
    they parsed. That ordering is deliberate rather than incidental: resolving `grid_cell`
    is what decides whether a raster request's final frame is band-wide or long, and
    `validate_shapes` is what turns that shape into the right rejection for the requested
    format (a raster grid asked for `csv` with no `grid_cell`; a point query asked for
    `tif`) -- it cannot answer either question before this function has run.

    `format` is the string the URL's own extension resolves to, used verbatim as
    `output.format`.
    """
    grid_cell_where, relabel = await resolve_grid_cell(dataset_dimensions.grid, options.grid_cell)
    pipeline = build_data_query_pipeline(
        dataset_dimensions, filter_dimensions, options, format, grid_cell_where=grid_cell_where
    )
    variable = VARIABLES.get(dataset_dimensions.variable)
    return ResolvedDataQuery(
        pipeline=pipeline,
        relabel=relabel,
        float_precision=variable.float_precision if variable is not None else None,
    )
