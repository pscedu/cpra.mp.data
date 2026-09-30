"""The decoder's one piece of request-to-`Pipeline` work that neither
`request.unified.partition_query` nor the named-analysis registry may do for itself:
resolving a coordinate-valued `grid_cell` against real grid geometry, which needs `io/`
and therefore cannot live under `request/` (see that module's docstring, and
`tests/api/test_analysis_layering.py`).

The two `resolve_grid_cell` forms below are the whole module. `api.unified.decode_request`
calls the async one (and `decode_request_sync` the blocking one, for a visualization
building its spec outside the event loop), once it has probed the resolved pipeline for the
grid its output is addressed against. There is no `build_data_query` wrapper any more: it
existed to compose this with `build_data_query_pipeline` for the three
`/v1/datasets/{variable}/{time_unit}/{grid}/...` routes, all of which are now
`GET /v1/data.{format}` going through the unified decoder.

**Why `grid_cell` resolution is grid-kind-dependent, and exactly how.** The two grid kinds
have always disagreed about what a `grid_cell` value MEANS, and this module reproduces that
split rather than unifying it -- unifying it would silently change what existing URLs
select:

- A VECTOR grid accepts either form. Every token a plain (unsigned, non-decimal) digit
  string means literal polygon ids; anything else is parsed as coordinate pairs and
  resolved by point-in-polygon lookup. That rule is kept token-for-token from the flat
  routes' long-standing behavior. The digit case is not resolved here at all -- it is
  left to `request.data_query.grid_cell_conjunct`, so "these are literal ids" has exactly
  one implementation.
- A RASTER grid accepts ONLY coordinates, with no digit check of any kind: the raw tokens
  go straight to `parse_coordinate_pairs`, so `grid_cell=1,2` on a raster has always
  meant the point at lng=1, lat=2 -- a degenerate coordinate pair, not pixel id 1 and
  pixel id 2. Reading digits as pixel ids here (as a vector grid would) would reinterpret
  such a URL rather than merely extend it, so the raster branch resolves every value as a
  coordinate.

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
    resolve_vector_grid_cell_sync,
)
from cpra.mp.data.analysis.request.errors import NotFoundError
from cpra.mp.data.grids import GRIDS


class _VectorPoints(NamedTuple):
    """The one case `_resolve_without_geometry` below cannot finish on its own: WGS84
    pairs still needing a point-in-polygon lookup, which is the only part of `grid_cell`
    resolution that differs between the async and sync forms.
    """

    lngs: list[float]
    lats: list[float]


def _resolve_without_geometry(grid_name: str, grid_cell: str | None):
    """Everything both `resolve_grid_cell` forms below share: the two "nothing to resolve"
    early outs, the unknown-grid 404, and the whole raster branch -- which opens no file
    (`io.geography.resolve_raster_pixel` works off the grid's declared transform alone)
    and so needs no async form.

    Returns the finished `(grid_cell_where, relabel)` pair, or a `_VectorPoints` when a
    polygon lookup is still owed.
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

    return _VectorPoints(*parse_coordinate_pairs(tokens))


def _vector_point_where(resolved: list[int | None]):
    """A `(grid_cell_where, relabel)` pair from a vector point lookup's ids. `relabel` is
    always `None`: a vector result is keyed by polygon id, which is what the caller asked
    about -- only a raster's synthetic pixel ids need renaming back to coordinates.
    """
    return grid_cell_id_where([cell_id for cell_id in resolved if cell_id is not None]), None


async def resolve_grid_cell(grid_name: str, grid_cell: str | None):
    """Resolve a flat route's raw `grid_cell` query value against `grid_name`'s geometry,
    returning `(grid_cell_where, relabel)` for `build_data_query_pipeline`/`run_to_path` --
    see this module's docstring for the per-grid-kind rules this implements.

    A `grid_cell_where` of `None` means "this function resolved nothing; leave
    `options.grid_cell` to `build_data_query_pipeline`" -- returned both when the caller
    set no `grid_cell` at all and for a vector grid's digit-id form, which that function
    already handles and which nothing here could improve on.

    Async only for the vector branch: `resolve_vector_grid_cell` runs one windowed
    FlatGeoBuf read per point in an executor.
    """
    resolved = _resolve_without_geometry(grid_name, grid_cell)
    if not isinstance(resolved, _VectorPoints):
        return resolved

    return _vector_point_where(
        await resolve_vector_grid_cell(grid_name, resolved.lngs, resolved.lats)
    )


def resolve_grid_cell_sync(grid_name: str, grid_cell: str | None):
    """Blocking form of `resolve_grid_cell`, for a caller outside the event loop.

    A visualization builds its spec from synchronous cached properties but still has to
    resolve a coordinate selection before it can hand the result to
    `analyses.resolve_analysis(grid_cell_where=...)` -- the registry cannot do it itself
    (resolving a coordinate needs `io/`, which nothing under `request/` or in `analyses/`
    may import), and there is no event loop to await the form above on. Same trade
    `io.geography.resolve_vector_grid_cell_sync` exists for, one layer down.
    """
    resolved = _resolve_without_geometry(grid_name, grid_cell)
    if not isinstance(resolved, _VectorPoints):
        return resolved

    return _vector_point_where(
        resolve_vector_grid_cell_sync(grid_name, resolved.lngs, resolved.lats)
    )
