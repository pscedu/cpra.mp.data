from cpra.mp.data.api.query import CatalogQuery
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from typing import Annotated

catalog = APIRouter(tags=["Catalog"])


@catalog.get("/")
async def catalog_query(
    query: Annotated[CatalogQuery, Query()],
):
    return JSONResponse(content=query.execute())
