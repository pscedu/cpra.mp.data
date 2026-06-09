import gzip

import mercantile
import polars as pl
from fastapi import HTTPException
from mapbox_vector_tile.Mapbox.vector_tile_pb2 import tile as VectorTile
from pmtiles.reader import MmapSource, Reader
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from cpra.mp.data.config import TILES_PATH


def read_pmtile(grid_name: str, tile: mercantile.Tile) -> bytes | None:
    """Return the raw (still gzip-compressed) MVT bytes for a tile, or None."""
    path = TILES_PATH / f"{grid_name}.pmtiles"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = Reader(MmapSource(f)).get(tile.z, tile.x, tile.y)
    return data  # may be None if tile not in archive


def _data_to_id_value(df: pl.DataFrame) -> dict[int, float]:
    """Extract {grid_cell_id: value} from a long-format DataFrame."""
    if df.is_empty() or "grid_cell" not in df.columns:
        return {}
    return {
        cell: val
        for cell, val in zip(df["grid_cell"].to_list(), df["value"].to_list())
        if val is not None
    }


def parse_tile(compressed: bytes) -> tuple[VectorTile, list[int]]:
    """Decompress and parse a gzip-compressed MVT tile once.

    Returns the parsed VectorTile object and the feature IDs from the boundary
    layer. Callers should pass the VectorTile to inject_and_serialize rather than
    re-parsing from bytes.
    """
    vt = VectorTile()
    vt.ParseFromString(gzip.decompress(compressed))
    ids = []
    for layer in vt.layers:
        if layer.name == "boundary":
            ids = [f.id for f in layer.features]
            break
    return vt, ids


def inject_and_serialize(vt: VectorTile, id_values: dict[int, float]) -> bytes:
    """Inject a 'value' property into each boundary feature and serialize.

    Geometry bytes are never decoded — only keys/values/tags are modified — so
    this is O(features) string/int work rather than O(coordinates) float work.
    """
    for layer in vt.layers:
        if layer.name != "boundary":
            continue
        key_idx = len(layer.keys)
        layer.keys.append("value")
        for feature in layer.features:
            v = id_values.get(feature.id)
            if v is not None:
                val = layer.values.add()
                val.double_value = v
                feature.tags.extend([key_idx, len(layer.values) - 1])

    return vt.SerializeToString()


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
