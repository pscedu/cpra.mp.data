from fastapi import HTTPException
from starlette.responses import Response
from starlette.types import Receive, Scope, Send


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
