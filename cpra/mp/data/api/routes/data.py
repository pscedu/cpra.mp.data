from cpra.mp.data.api.dataset import Dataset
from cpra.mp.data.api.raster import RasterioStreamingResponse, RasterioTileResponse
from cpra.mp.data.api.tabular import PolarsStreamingResponse
from cpra.mp.data.api.types import DataFormat, OutputType, TileFormat
from fastapi import APIRouter, Depends


data = APIRouter(tags=["Data and Analysis"])


@data.get("/{plan_version}/{variable}/{geographic_unit}/{time_unit}.{format}")
async def get_model_data(
    format: DataFormat,
    dataset: Dataset = Depends(),
):
    await dataset.apply_params()
    if dataset.output_type == OutputType.raster:
        return RasterioStreamingResponse(dataset, format)
    return PolarsStreamingResponse(dataset, format)


@data.get(
    "/{plan_version}/{variable}/{geographic_unit}/{time_unit}/{z}/{x}/{y}.{format}"
)
async def get_model_tile(
    format: TileFormat,
    z: int,
    x: int,
    y: int,
    dataset: Dataset = Depends(),
):
    await dataset.apply_params()
    return RasterioTileResponse(dataset, x, y, z, format)
