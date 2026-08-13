"""Resolve lat/lng coordinates to vector grid cell IDs via point-in-polygon lookup."""

import asyncio

import pyogrio
from fastapi import HTTPException
from rasterio.crs import CRS
from rasterio.warp import transform
from shapely.geometry import Point

from cpra.mp.data import config


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


def _projected_points(grid_name: str, lngs: list[float], lats: list[float]):
    """Locate a vector grid's geometry file and project WGS84 pairs into its CRS."""
    fgb_path = config.GEOMETRY_PATH / f"{grid_name}.fgb"
    if not fgb_path.exists():
        raise HTTPException(status_code=400, detail=f"No geometry file for grid: {grid_name}")

    fgb_crs = pyogrio.read_info(str(fgb_path))["crs"]
    xs, ys = transform(CRS.from_epsg(4326), CRS.from_user_input(fgb_crs), lngs, lats)
    return fgb_path, list(zip(xs, ys))


async def resolve_vector_geography(grid_name: str, lngs: list[float], lats: list[float]) -> list[int | None]:
    """Resolve WGS84 (lng, lat) pairs to grid cell IDs for a vector grid.

    Returns one ID per input pair, in order, with `None` for points that don't fall
    inside any polygon in the grid.
    """
    fgb_path, points = _projected_points(grid_name, lngs, lats)

    loop = asyncio.get_running_loop()
    return await asyncio.gather(
        *[loop.run_in_executor(None, _resolve_point, fgb_path, x, y) for x, y in points]
    )


def resolve_vector_geography_sync(grid_name: str, lngs: list[float], lats: list[float]) -> list[int | None]:
    """Blocking form of `resolve_vector_geography`, for callers outside the event loop.

    A visualization builds its spec from synchronous properties but still needs the cell
    IDs behind a coordinate selection to label itself — the same trade `map.py` makes when
    it collects its feature values there.
    """
    fgb_path, points = _projected_points(grid_name, lngs, lats)
    return [_resolve_point(fgb_path, x, y) for x, y in points]
