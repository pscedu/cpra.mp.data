"""Convert between vector grid cell IDs and lat/lng coordinates using grid geometry.

Formerly `api/geography.py`; moved here because `vector.apply_grid_cell_selection`
(called by `VectorGridAnalysis.apply_steps`'s terminal `grid_cell` filter) resolves
`grid_cell` coordinates through this module directly, and `cpra.mp.data.analysis` may not
import FastAPI (see this package's other modules, and
`tests/api/test_analysis_boundary.py`) -- so every `HTTPException` this module used to
raise is an `AnalysisError` instead.
"""

import asyncio
import math

import pyogrio

# Rasterio doesn't re-export its GDAL error classes from `rasterio.errors`, and this is
# the only handle on the failure PROJ reports for an unprojectable point.
from rasterio._err import CPLE_BaseError
from rasterio.crs import CRS
from rasterio.warp import transform, transform_bounds
from shapely.geometry import Point

from cpra.mp.data import config
from cpra.mp.data.analysis.base import AnalysisError
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
        raise AnalysisError("grid_cell contains non-numeric coordinates", status_code=400)

    if len(coords) % 2:
        raise AnalysisError("grid_cell contain an even number of coordinates", status_code=400)

    lngs = []
    lats = []
    for start_idx in range(0, len(coords), 2):
        lng, lat = coords[start_idx : start_idx + 2]
        if not -180 <= lng <= 180 or not -90 <= lat <= 90:
            msg = f"grid_cell contains invalid coordinate pair: {lng}, {lat}"
            raise AnalysisError(msg, status_code=400)
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
        raise AnalysisError(msg, status_code=400)

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
        raise AnalysisError(f"No geometry file for grid: {grid_name}", status_code=400)
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
