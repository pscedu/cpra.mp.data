import geopandas as gpd
import mapbox_vector_tile
import mercantile
import polars as pl
import polars.selectors as cs
import pyproj
import shapely
from fastapi import HTTPException
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from cpra.mp.data.config import GEOMETRY_PATH

TILE_EXTENT = 4096


def _tile_bbox_in_crs(tile: mercantile.Tile, crs: str) -> tuple[float, float, float, float]:
    bounds = mercantile.bounds(tile)
    if crs.upper() == "EPSG:4326":
        return (bounds.west, bounds.south, bounds.east, bounds.north)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    west, south = transformer.transform(bounds.west, bounds.south)
    east, north = transformer.transform(bounds.east, bounds.north)
    return (west, south, east, north)


def read_tile_geometries(
    grid_name: str, tile: mercantile.Tile, crs: str = "EPSG:26915"
) -> gpd.GeoDataFrame | None:
    """Read FGB features intersecting a tile's bbox, reprojected to EPSG:3857."""
    fgb_path = GEOMETRY_PATH / f"{grid_name}.fgb"
    if not fgb_path.exists():
        return None
    bbox = _tile_bbox_in_crs(tile, crs)
    gdf = gpd.read_file(str(fgb_path), bbox=bbox, engine="pyogrio")
    if gdf.empty:
        return None
    return gdf.to_crs("EPSG:3857")


def _data_to_id_value(df: pl.DataFrame) -> dict[int, float]:
    """Extract {geography_id: value} from a wide-format DataFrame (first row used)."""
    value_cols = df.select(cs.matches(r"^[0-9\-]")).columns
    if not value_cols or df.is_empty():
        return {}
    row = df.select(value_cols).row(0, named=True)
    return {int(col): val for col, val in row.items() if val is not None}


def encode_mvt(
    gdf: gpd.GeoDataFrame, df: pl.DataFrame, tile: mercantile.Tile
) -> bytes | None:
    """Clip geometries to tile extent, join dataset values, and encode as MVT/PBF."""
    id_values = _data_to_id_value(df)
    if not id_values:
        return None

    xy_bounds = mercantile.xy_bounds(tile)
    tile_box = shapely.box(xy_bounds.left, xy_bounds.bottom, xy_bounds.right, xy_bounds.top)

    # Zoom-dependent simplification to reduce tile size at low zoom levels.
    gdf = gdf.copy()
    if tile.z < 12:
        tile_width = xy_bounds.right - xy_bounds.left
        tolerance = tile_width / TILE_EXTENT
        gdf["geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=True)

    gdf["geometry"] = gdf.geometry.intersection(tile_box)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    if gdf.empty:
        return None

    quantize_bounds = (xy_bounds.left, xy_bounds.bottom, xy_bounds.right, xy_bounds.top)

    ids = gdf["id"].to_numpy().astype(int)
    labels = gdf["label"].to_numpy()
    geoms = gdf.geometry.to_numpy()
    centroids = gdf.geometry.centroid.to_numpy()

    boundary_features = []
    label_features = []
    for fid, label, geom, centroid in zip(ids, labels, geoms, centroids):
        fid = int(fid)
        props: dict = {"id": fid, "label": str(label)}
        value = id_values.get(fid)
        if value is not None:
            props["value"] = value
        boundary_features.append({"geometry": geom.__geo_interface__, "properties": props})
        if not centroid.is_empty:
            label_features.append(
                {
                    "geometry": centroid.__geo_interface__,
                    "properties": {"id": fid, "label": str(label)},
                }
            )

    if not boundary_features:
        return None

    layers = [{"name": "boundary", "features": boundary_features}]
    if label_features:
        layers.append({"name": "label", "features": label_features})

    return mapbox_vector_tile.encode(
        layers,
        default_options={"extents": TILE_EXTENT, "quantize_bounds": quantize_bounds},
    )


class VectorTileResponse(Response):
    background = None
    media_type = "application/vnd.mapbox-vector-tile"
    status_code = 200

    def __init__(self, dataset, x: int, y: int, z: int) -> None:
        self.dataset = dataset
        self.x = x
        self.y = y
        self.z = z
        self.init_headers()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.body = await self.dataset.get_vector_tile(self.x, self.y, self.z)
        if self.body is None:
            raise HTTPException(status_code=404, detail="Tile does not exist")
        await super().__call__(scope, receive, send)
