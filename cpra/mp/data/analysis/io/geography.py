"""Convert between grid cell IDs and lat/lng coordinates using grid geometry, for both
grid kinds: a vector grid's polygon ids (`resolve_vector_grid_cell`, backed by a `.fgb`
geometry file) and a raster grid's pixel positions (`resolve_raster_pixel`/
`raster_grid_cell_id`, backed by nothing but `cpra.mp.data.grids.RasterGrid`'s own
declared extent -- no file is opened to answer a raster coordinate lookup).
`grid_cell_id_where` builds the `where` conjunct either grid kind's resolved ids feed into
`request.data_query.build_data_query_pipeline(..., grid_cell_where=...)`;
`resolve_raster_point_labels`/`relabel_raster_grid_cell` are the raster-only pair that
additionally relabels an executed raster point query's output back to the caller's own
coordinate strings, which a vector `grid_cell` query never needs (it reports the polygon id
itself -- see `resolve_raster_point_labels`'s docstring on the asymmetry).

Formerly `api/geography.py`; moved here because `cpra.mp.data.analysis` may not import
FastAPI (see this package's other modules, and `tests/api/test_analysis_boundary.py`), so
nothing here raises an `HTTPException`. Every rejection is a `request.errors.RequestError`
member instead, which `api/boundary.py` maps to a status code at exactly one place: a
malformed or unprojectable coordinate and a missing geometry file are `RequestError`
itself (400), and a grid name that is not a raster grid is `NotFoundError` (404).
"""

import asyncio
import math

import polars as pl
import pyogrio

# Rasterio doesn't re-export its GDAL error classes from `rasterio.errors`, and this is
# the only handle on the failure PROJ reports for an unprojectable point.
from rasterio._err import CPLE_BaseError
from rasterio.crs import CRS
from rasterio.transform import from_bounds, rowcol
from rasterio.warp import transform, transform_bounds
from shapely.geometry import Point

from cpra.mp.data import config
from cpra.mp.data.analysis.request.errors import NotFoundError, RequestError
from cpra.mp.data.grids import GRIDS


def parse_coordinate_pairs(values):
    """Parse a flat list of strings into (lngs, lats), validating that every value is
    numeric, that there's an even count, and that each pair is a valid WGS84 coordinate.

    The bounds are written as chained comparisons rather than as a magnitude test so that
    they also reject a NaN, which `float` accepts and every comparison against reports
    False — and which projects to infinity instead of failing.
    """
    try:
        coords = [float(value) for value in values]
    except ValueError:
        raise RequestError("grid_cell contains non-numeric coordinates")

    if len(coords) % 2:
        raise RequestError("grid_cell contain an even number of coordinates")

    lngs = []
    lats = []
    for start_idx in range(0, len(coords), 2):
        lng, lat = coords[start_idx : start_idx + 2]
        if not -180 <= lng <= 180 or not -90 <= lat <= 90:
            msg = f"grid_cell contains invalid coordinate pair: {lng}, {lat}"
            raise RequestError(msg)
        lngs.append(lng)
        lats.append(lat)

    return lngs, lats


def project_coordinates(lngs: list[float], lats: list[float], dst_crs) -> tuple[list, list]:
    """Project WGS84 (lng, lat) pairs into a grid's CRS, as a bad request if they don't fit.

    A coordinate can be valid WGS84 and still have nowhere to go: PROJ rejects points
    outside the target projection's domain, which for the coastal grids' UTM zone includes
    the origin at (0, 0). It reports that for the batch rather than for the pair that
    caused it, so the offending pairs are hunted down one at a time — but only once the
    batch has already failed, keeping the cost off the path where every coordinate is fine.
    """
    try:
        xs, ys = transform(CRS.from_epsg(4326), dst_crs, lngs, lats)
    except CPLE_BaseError:
        pairs = [f"{lng}, {lat}" for lng, lat in zip(lngs, lats) if _is_unprojectable(lng, lat, dst_crs)]
        msg = f"grid_cell contains coordinate pairs outside the grid's projection: {'; '.join(pairs)}"
        raise RequestError(msg)

    return list(xs), list(ys)


def _is_unprojectable(lng: float, lat: float, dst_crs) -> bool:
    """Check a single pair against the projection that rejected the batch it came in."""
    try:
        transform(CRS.from_epsg(4326), dst_crs, [lng], [lat])
    except CPLE_BaseError:
        return True
    return False


def _resolve_point(fgb_path, x, y):
    """Find the polygon containing a single point, via a windowed FlatGeoBuf read.

    FlatGeoBuf stores a packed R-tree alongside the features, so a bbox read is a cheap
    subset scan regardless of file size — no need to load the whole file or cache it. A
    zero-area (point) bbox still returns every feature whose bounding box contains the
    point; the exact `contains` check below narrows that candidate set down to the one
    (if any) whose actual polygon contains it.
    """
    candidates = pyogrio.read_dataframe(fgb_path, bbox=(x, y, x, y))
    if candidates.empty:
        return None
    point = Point(x, y)
    match = candidates[candidates.contains(point)]
    return int(match["id"].iloc[0]) if not match.empty else None


def _geometry_path(grid_name: str):
    """Locate a vector grid's geometry file, reporting a missing one as a 400."""
    fgb_path = config.GRIDS_PATH / f"{grid_name}.fgb"
    if not fgb_path.exists():
        raise RequestError(f"No geometry file for grid: {grid_name}")
    return fgb_path


def straddling_feature_bounds(grid_name: str, tile_bounds, tile_crs) -> tuple[float, float, float, float] | None:
    """Bounding box, in `grid_name`'s OWN crs (`cpra.mp.data.grids.GRIDS[grid_name].crs`),
    of every feature in that vector grid's `.fgb` whose bounding box the tile footprint
    (`tile_bounds`, expressed in `tile_crs` -- Web Mercator for every map tile) overlaps at
    all, not just the part of each feature that falls inside the tile.

    Called from `io.raster_io.tile_source_window` when `execution.plan.compile_plan` has
    determined that the raster being read feeds a raster-to-vector `project` step onto
    `grid_name`: that step joins each raster pixel to whichever polygon it falls inside on
    the OUTPUT grid, and a polygon can extend past a tile's own edge, so a raster read
    clipped to the tile's footprint alone would silently drop the pixels on the far side of
    a straddling polygon before the join ever sees them. Returning the union of every
    touched feature's own bounding box -- not an exact per-feature geometry test -- is
    deliberately generous: `tile_source_window` reads a few extra source pixels around a
    feature whose bounding box merely brushes the tile without the polygon itself doing so,
    which costs nothing but a slightly larger read, rather than risking the opposite.

    `bbox=` is pyogrio's own R-tree prefilter (the same mechanism `_resolve_point` uses for
    a single point) -- a bounding-box overlap test, which is exactly the union this function
    returns, so no further exact-geometry check narrows the candidate set the way
    `_resolve_point`'s `.contains()` does for a point-in-polygon answer.

    Returns `None` when the tile's footprint does not reproject to a finite bounding box in
    `grid_name`'s crs, or when no feature's bounding box overlaps it -- the same "nothing to
    add" convention `tile_source_window` uses for its own window.
    """
    fgb_path = _geometry_path(grid_name)
    grid_crs = GRIDS[grid_name].crs
    grid_bounds = transform_bounds(tile_crs, grid_crs, *tile_bounds, densify_pts=21)
    if not all(math.isfinite(value) for value in grid_bounds):
        return None

    features = pyogrio.read_dataframe(fgb_path, bbox=grid_bounds, columns=["id"])
    if features.empty:
        return None
    minx, miny, maxx, maxy = features.total_bounds
    return (float(minx), float(miny), float(maxx), float(maxy))


def _projected_points(grid_name: str, lngs: list[float], lats: list[float]):
    """Locate a vector grid's geometry file and project WGS84 pairs into its CRS."""
    fgb_path = _geometry_path(grid_name)
    fgb_crs = pyogrio.read_info(str(fgb_path))["crs"]
    xs, ys = project_coordinates(lngs, lats, CRS.from_user_input(fgb_crs))
    return fgb_path, list(zip(xs, ys))


async def resolve_vector_grid_cell(grid_name: str, lngs: list[float], lats: list[float]) -> list[int | None]:
    """Resolve WGS84 (lng, lat) pairs to grid cell IDs for a vector grid.

    Returns one ID per input pair, in order, with `None` for points that don't fall
    inside any polygon in the grid.
    """
    fgb_path, points = _projected_points(grid_name, lngs, lats)

    loop = asyncio.get_running_loop()
    return await asyncio.gather(
        *[loop.run_in_executor(None, _resolve_point, fgb_path, x, y) for x, y in points]
    )


def resolve_raster_pixel(grid_name: str, lngs: list[float], lats: list[float]) -> list[tuple[int, int] | None]:
    """Resolve WGS84 (lng, lat) pairs to ABSOLUTE (row, col) pixel positions on
    `grid_name`'s own full pixel grid -- the raster counterpart of
    `resolve_vector_grid_cell`, for NEW-engine code (see `raster_grid_cell_id` below for
    the ABSOLUTE `grid_cell` id a spatial `filter` actually tests against, which this
    function deliberately does NOT compute).

    Returns one entry per input pair, in order, `None` for a point that falls outside the
    raster's own extent -- the same per-point, independent-resolution contract
    `resolve_vector_grid_cell` already keeps for a point outside every polygon. A request
    where EVERY point resolves to `None` is not special-cased here: it is exactly what
    "none of these points are inside this grid" means, resolved one point at a time same
    as any other outcome. What that should mean for a spatial filter downstream (an empty
    result vs. a 400) is a decision for a CALLER building a `where` conjunct out of this
    list, not for a lookup that has no opinion about how its answer gets used -- mirrored
    by `resolve_vector_grid_cell`'s own docstring making the identical distinction.

    Opens no data file: `cpra.mp.data.grids.RasterGrid` already carries everything a pixel
    lookup needs (`bounds`, `height`, `width`, `crs`) -- the same "the grid's own geometry
    is enough, no raster need be opened" property `io.raster_io.raster_grid_geometry` and
    `execution.plan.compile_plan` already rely on for a raster origin's window/extent (only
    resolving which BANDS exist touches the catalog). `NotFoundError` -- a 404, unlike the
    plain `RequestError` every other rejection here raises -- is used for a name that isn't
    a raster grid at all: nothing about the request is malformed, the thing it named simply
    does not exist. Contrast `_geometry_path`'s missing-file 400 on the vector side, which
    is a real grid whose geometry artifact has not been built.
    """
    grid = GRIDS.get(grid_name)
    if grid is None or grid.grid_type != "raster":
        raise NotFoundError(f"{grid_name!r} is not a raster grid")

    xs, ys = project_coordinates(lngs, lats, CRS.from_user_input(grid.crs))
    grid_transform = from_bounds(*grid.bounds, width=grid.width, height=grid.height)
    rows, cols = rowcol(grid_transform, xs, ys)

    resolved: list[tuple[int, int] | None] = []
    for row, col in zip(rows, cols):
        if 0 <= row < grid.height and 0 <= col < grid.width:
            resolved.append((int(row), int(col)))
        else:
            resolved.append(None)
    return resolved


def raster_grid_cell_id(grid_name: str, pixel: tuple[int, int]) -> int:
    """The ABSOLUTE row-major `grid_cell` id a pixel at ABSOLUTE `(row, col)` (as
    `resolve_raster_pixel` returns) would be assigned by `execution.band_ops.
    sample_grid_cells_band_wide` -- computed forward from a known pixel position, rather
    than decoded from a row a real read produced.

    `row * grid.width + col`, over `grid_name`'s own full pixel width -- no window
    involved at all. That is deliberate, not an omission: `execution.frame.
    row_major_grid_cell`'s absolute mode (what `sample_grid_cells_band_wide` always uses to
    label its output now) numbers every pixel from the RASTER's own origin, never from
    whatever window a particular read happened to cover, so this id means the same pixel
    regardless of whether -- or how -- `execution.execute`'s point-sample narrowing chose
    to read it. A caller building a `grid_cell` filter from a resolved coordinate calls this
    once per pixel and uses the result directly; there is no window to keep in sync with it,
    and no "recompute every id together" step the way the window-relative id this function
    used to return once required.
    """
    row, col = pixel
    return row * GRIDS[grid_name].width + col


def grid_cell_id_where(grid_cell_ids: list[int]) -> str:
    """The `where` conjunct for an already-resolved, plain list of int `grid_cell` ids --
    shared by both grid kinds' non-digit `grid_cell` handling, since by the time either
    `resolve_vector_grid_cell` or `resolve_raster_point_labels`/`raster_grid_cell_id` has
    run, a coordinate has already become exactly this: a list of ints, with no grid-kind-
    specific shape left to it.

    `= N` for one id, `IN (...)` for several, and `"grid_cell != grid_cell"` -- the
    unsatisfiable-for-any-real-value clause `request.data_query`'s own module docstring
    names as the convention for "matched nothing" -- for zero ids, so a request whose every
    point resolved to nothing (see `resolve_raster_point_labels`'s docstring, or a vector
    coordinate landing outside every polygon) still builds a STRUCTURALLY valid `Pipeline`
    (one that runs and returns an empty result), rather than a `where` string with an empty
    `IN ()`, which is not valid SQL.
    """
    ids = sorted(set(grid_cell_ids))
    if not ids:
        return "grid_cell != grid_cell"
    if len(ids) == 1:
        return f"grid_cell = {ids[0]}"
    return f"grid_cell IN ({', '.join(str(cell_id) for cell_id in ids)})"


def resolve_raster_point_labels(grid_name: str, tokens: list[str]) -> dict[int, list[str]]:
    """Resolve a raster point query's raw, comma-flattened `grid_cell` tokens (e.g.
    `options.grid_cell.split(",")` -- alternating lng, lat, lng, lat, ..., exactly as the
    caller wrote them and not yet paired up) to the ABSOLUTE ids `grid_cell_id_where` turns
    into a `where` conjunct, and to the id -> [original label] map a caller uses afterward
    to relabel the executed result back to coordinate strings (`relabel_raster_grid_cell`,
    below).

    Each returned map value is that point's own two tokens rejoined with a literal comma --
    `f"{lng_token},{lat_token}"` -- never a value round-tripped through `float()` and back.
    This is a deliberate departure from the long-standing raster point-query convention
    of relabeling columns as `f"{x!s},{y!s}"` over the PARSED floats, which reformats e.g.
    `"-90.10"` down to `"-90.1"`: a caller joining the response back to its own request
    needs the EXACT string it sent, not a value that has been through a float and back. A caller that names the same point twice, or two points that resolve to
    the same pixel, gets both of their original label strings back in that one id's list --
    `relabel_raster_grid_cell` fans them back out into one row apiece. Nothing here
    deduplicates by pixel the way a vector grid's ID-labeled output does, where two
    coordinates landing in the same polygon collapse to that one id: the raster
    convention this mirrors was never ID-labeled to begin with, so there is no id-level
    dedup to preserve going into it.

    A token pair that resolves to no pixel at all (`resolve_raster_pixel`'s own `None`, for
    a point outside `grid_name`'s raster extent) is simply ABSENT from the returned map --
    the same "outside every polygon is dropped, not an error" convention
    `resolve_vector_grid_cell` already keeps for a vector grid, extended here to mean: a
    request naming a point outside the raster gets no row back for that point (fewer rows
    than points requested, or none at all), never a 400. `grid_cell_id_where` turns "every
    point resolved to nothing" into the pipeline's own "matched nothing" `where`, which is
    what makes that choice hold end-to-end rather than just at this one function.
    """
    lngs, lats = parse_coordinate_pairs(tokens)
    pixels = resolve_raster_pixel(grid_name, lngs, lats)

    id_to_labels: dict[int, list[str]] = {}
    for point_idx, pixel in enumerate(pixels):
        if pixel is None:
            continue
        label = f"{tokens[2 * point_idx]},{tokens[2 * point_idx + 1]}"
        cell_id = raster_grid_cell_id(grid_name, pixel)
        id_to_labels.setdefault(cell_id, []).append(label)
    return id_to_labels


def relabel_raster_grid_cell(lf: pl.LazyFrame, id_to_labels: dict[int, list[str]]) -> pl.LazyFrame:
    """Replace an executed raster point query's ABSOLUTE integer `grid_cell` column with
    the original coordinate-string label(s) that resolved to it -- the raster point-query
    output convention this engine keeps (see `resolve_raster_point_labels`'s docstring for
    why the label is the caller's own token pair, verbatim, never reformatted).

    An inner join, not a dict lookup: `id_to_labels` (built by `resolve_raster_point_labels`)
    can map one id to MORE than one label, when two requested points share a pixel --
    joining `lf` against a flattened `(grid_cell, label)` table fans a single matching data
    row back out into one row per original label, which a `.replace()`-style substitution
    could not do; a plain Python dict lookup only ever produces one output per input row. A
    `grid_cell` value with no entry in `id_to_labels` at all (impossible if `lf` was itself
    produced by filtering on `grid_cell_id_where(list(id_to_labels))`, but not assumed here)
    simply drops out of the join, the same as any other row a `where` would exclude.
    """
    ids: list[int] = []
    labels: list[str] = []
    for cell_id, point_labels in id_to_labels.items():
        for label in point_labels:
            ids.append(cell_id)
            labels.append(label)

    labels_frame = pl.LazyFrame(
        {"grid_cell": ids, "__grid_cell_label": labels},
        schema={"grid_cell": pl.Int64, "__grid_cell_label": pl.Utf8},
    )
    return (
        lf.join(labels_frame, on="grid_cell", how="inner")
        .drop("grid_cell")
        .rename({"__grid_cell_label": "grid_cell"})
    )


def representative_points(grid_name: str, grid_cell_ids: list[int] | None = None) -> dict[int, tuple]:
    """Get a representative WGS84 (lng, lat) point for each grid cell in a vector grid.

    The point is PostGIS's `ST_PointOnSurface`, so it is guaranteed to lie inside the
    polygon it names — unlike a centroid, which a concave or multi-part grid cell can
    place outside itself, and which would then be a coordinate that doesn't select the
    grid cell it came from.

    Grid cells are read by attribute filter rather than by scanning the whole file, and
    IDs with no polygon are simply absent from the result.
    """
    fgb_path = _geometry_path(grid_name)
    where = None
    if grid_cell_ids is not None:
        if not grid_cell_ids:
            return {}
        where = f"id IN ({','.join(str(int(grid_cell_id)) for grid_cell_id in grid_cell_ids)})"

    gdf = pyogrio.read_dataframe(fgb_path, columns=["id"], where=where)
    if gdf.empty:
        return {}

    points = gdf.geometry.representative_point()
    lngs, lats = transform(CRS.from_user_input(gdf.crs), CRS.from_epsg(4326), list(points.x), list(points.y))
    return {int(grid_cell_id): point for grid_cell_id, point in zip(gdf["id"], zip(lngs, lats))}


def resolve_vector_grid_cell_sync(grid_name: str, lngs: list[float], lats: list[float]) -> list[int | None]:
    """Blocking form of `resolve_vector_grid_cell`, for callers outside the event loop.

    A visualization builds its spec from synchronous properties but still needs the cell
    IDs behind a coordinate selection to label itself — the same trade `map.py` makes when
    it collects its feature values there.
    """
    fgb_path, points = _projected_points(grid_name, lngs, lats)
    return [_resolve_point(fgb_path, x, y) for x, y in points]
