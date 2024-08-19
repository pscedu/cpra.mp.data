from cpra.mp.data.api.dataset import Dataset
from cpra.mp.data.api.raster import RasterioStreamingResponse
from cpra.mp.data.api.tabular import PolarsStreamingResponse
from cpra.mp.data.api.types import DataFormat
from fastapi import APIRouter, Depends


data = APIRouter(tags=["Data and Analysis"])


@data.get("/{plan_version}/{variable}/{geographic_unit}/{time_unit}.{format}")
async def get_model_data(
    format: DataFormat,
    dataset: Dataset = Depends(),
):
    # The presence of a path column indicates that this is a raster.
    if "path" in dataset.lf.columns:
        return RasterioStreamingResponse(dataset.lf, format)
    return PolarsStreamingResponse(dataset.lf, format)
