import polars.selectors as cs
from cpra.mp.data.api.catalog import Catalog
from cpra.mp.data.api.dataset import Dataset
from cpra.mp.data.api.models import Dimensions, Options
from cpra.mp.data.api.raster import RasterioStreamingResponse, RasterioTileResponse
from cpra.mp.data.api.sources import LazyFrameSource
from cpra.mp.data.api.tabular import PolarsStreamingResponse
from cpra.mp.data.api.types import DataFormat, OutputType, TileFormat
from cpra.mp.data.dimensions import DATASET_DIMENSIONS
from cpra.mp.data.utils import glob_lazy_frame
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Annotated


app = FastAPI(
    title="CPRA Master Plan APIs",
    description="Access model data from [Louisiana's Coastal Master Plan](https://coastal.la.gov/our-plan/2023-coastal-master-plan/).",
    prefix="/v1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/datasets.{format}", tags=["Catalog"])
async def catalog_datasets(
    format: DataFormat,
):
    lf = glob_lazy_frame(required_cols=DATASET_DIMENSIONS).select(cs.exclude("path"))
    source = LazyFrameSource(lf)
    return PolarsStreamingResponse(source, format)


@app.get(
    "/v1/datasets/{plan_version}/{variable}/{geographic_unit}/{time_unit}/catalog.json",
    tags=["Catalog"],
)
async def catalog_query(
    dimensions: Annotated[Dimensions, Query()],
):
    catalog = Catalog(dimensions)
    return JSONResponse(content=catalog.execute())


@app.get(
    "/v1/datasets/{plan_version}/{variable}/{geographic_unit}/{time_unit}/data.{format}",
    tags=["Data and Analysis"],
)
async def get_model_data(
    format: DataFormat,
    dimensions: Annotated[Dimensions, Query()],
    options: Annotated[Options, Query()],
):
    dataset = Dataset(dimensions, options)
    await dataset.apply_params()
    if dataset.output_type == OutputType.raster:
        return RasterioStreamingResponse(dataset, format)
    return PolarsStreamingResponse(dataset, format)


@app.get(
    "/v1/datasets/{plan_version}/{variable}/{geographic_unit}/{time_unit}/data/{z}/{x}/{y}.{format}",
    tags=["Data and Analysis"],
)
async def get_model_tile(
    format: TileFormat,
    z: int,
    x: int,
    y: int,
    dimensions: Annotated[Dimensions, Query()],
    options: Annotated[Options, Query()],
):
    dataset = Dataset(dimensions, options)
    await dataset.apply_params()
    return RasterioTileResponse(dataset, x, y, z, format)
