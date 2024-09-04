from cpra.mp.data.api.dataset import Dataset
from cpra.mp.data.api.raster import RasterioStreamingResponse, RasterioTileResponse
from cpra.mp.data.api.tabular import PolarsStreamingResponse
from cpra.mp.data.api.types import DataFormat, TileFormat
from fastapi import APIRouter, Depends


data = APIRouter(tags=["Data and Analysis"])


@data.get("/{plan_version}/{variable}/{geographic_unit}/{time_unit}.{format}")
async def get_model_data(
    format: DataFormat,
    dataset: Dataset = Depends(),
):
    # The presence of a path column indicates that this is a raster.
    if dataset.grid.grid_type == "raster":
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
    return RasterioTileResponse(dataset, x, y, z, format)
