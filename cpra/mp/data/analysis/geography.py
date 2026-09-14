"""Convert between vector grid cell IDs and lat/lng coordinates using grid geometry.

Formerly `api/geography.py`; moved here because `VectorGridAnalysis._apply_grid_cell`
(the vector half of the old `Dataset._apply_grid_cell_vector`) resolves `grid_cell`
coordinates through this module directly, and `cpra.mp.data.analysis` may not import
FastAPI (see this package's other modules, and `tests/api/test_analysis_boundary.py`) --
so every `HTTPException` this module used to raise is an `AnalysisError` instead.
"""

import asyncio

import pyogrio

# Rasterio doesn't re-export its GDAL error classes from `rasterio.errors`, and this is
# the only handle on the failure PROJ reports for an unprojectable point.
from rasterio._err import CPLE_BaseError
from rasterio.crs import CRS
from rasterio.warp import transform
from shapely.geometry import Point

from cpra.mp.data import config
from cpra.mp.data.analysis.base import AnalysisError


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
