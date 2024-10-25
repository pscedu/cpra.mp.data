import polars.selectors as cs
from cpra.mp.data.api.catalog import CatalogQuery
from cpra.mp.data.api.dataset import Dataset
from cpra.mp.data.api.models import (
    DatasetDimensions,
    FilterDimensions,
    FilterDimensionsAndOptions,
)
from cpra.mp.data.api.raster import RasterioStreamingResponse, RasterioTileResponse
from cpra.mp.data.api.sources import LazyFrameSource
from cpra.mp.data.api.tabular import PolarsStreamingResponse
from cpra.mp.data.api.types import DataFormat, OutputType, TileFormat
from cpra.mp.data.dimensions import DATASET_DIMENSIONS
from cpra.mp.data.utils import glob_lazy_frame
from fastapi import Depends, FastAPI, Query
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
    filter_dimensions: Annotated[FilterDimensions, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
):
    catalog_query = CatalogQuery(dataset_dimensions, filter_dimensions)
    return JSONResponse(content=catalog_query.execute())


@app.get(
    "/v1/datasets/{plan_version}/{variable}/{geographic_unit}/{time_unit}/data.{format}",
    tags=["Data and Analysis"],
)
async def get_model_data(
    format: DataFormat,
    query: Annotated[FilterDimensionsAndOptions, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
):
    dataset = Dataset(dataset_dimensions, query.filter_dimensions, query.options)
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
    query: Annotated[FilterDimensionsAndOptions, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
):
    dataset = Dataset(dataset_dimensions, query.filter_dimensions, query.options)
    await dataset.apply_params()
    return RasterioTileResponse(dataset, x, y, z, format)
