import polars.selectors as cs
from cpra.mp.data.api.catalog import CatalogQuery
from cpra.mp.data.api.dataset import Dataset
from cpra.mp.data.api.models import (
    Compression,
    DataQuery,
    DatasetDimensions,
    Dimension,
    FilterDimensions,
    Grid,
    Legend,
    TileQuery,
    Variable,
)
from cpra.mp.data.api.raster import RasterioStreamingResponse, RasterioTileResponse
from cpra.mp.data.api.sources import LazyFrameSource
from cpra.mp.data.api.tabular import PolarsStreamingResponse
from cpra.mp.data.api.types import DataFormat, OutputType, TileFormat
from cpra.mp.data.api.utils import validate_compression, validate_format
from cpra.mp.data.dimensions import DATASET_DIMENSIONS, DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.legends import LEGENDS
from cpra.mp.data.variables import VARIABLES
from cpra.mp.data.visualizations import VISUALIZATIONS
from cpra.mp.data.utils import glob_lazy_frame
from fastapi import Depends, FastAPI, Query, Request
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


@app.get("/v1/variables.json", tags=["Catalog"])
def variables() -> list[Variable]:
    return [
        Variable(
            name=var.name,
            label=var.label,
            dtype=str(var.dtype),
            legend=(var.legend.name if var.legend else None),
            model=var.model,
            subroutine=var.subroutine,
            unit=var.unit,
            geographic_units=var.geographic_units,
            time_units=var.time_units,
            required_dimensions=var.required_dimensions,
        )
        for var in VARIABLES.values()
    ]


@app.get("/v1/dimensions.json", tags=["Catalog"])
def dimensions() -> list[Dimension]:
    return [
        Dimension(
            name=dim.name,
            label=dim.label,
            type=dim.type,
            dtype=str(dim.dtype),
            required=(True if dim.required is True else False),
            null=dim.null,
            min_value=dim.min_value,
            max_value=dim.max_value,
            values=(
                dim.values.select("value", "label").iter_rows(named=True)
                if dim.values is not None
                else None
            ),
        )
        for dim in DIMENSIONS.values()
    ]


@app.get("/v1/grids.json", tags=["Catalog"])
def grids() -> list[Grid]:
    # TODO: Compress geography IDs into ranges.
    return [
        Grid(
            name=grid.name,
            label=grid.label,
            grid_type=grid.grid_type,
            crs=grid.crs,
            id_min=getattr(grid, "id_min", None),
            id_max=getattr(grid, "id_max", None),
            id_count=getattr(grid, "id_count", None),
            bounds=getattr(grid, "bounds", None),
            height=getattr(grid, "height", None),
            width=getattr(grid, "width", None),
        )
        for grid in GRIDS.values()
    ]


@app.get("/v1/legends.json", tags=["Catalog"])
def legends() -> list[Legend]:
    return [
        Legend(
            name=legend.name,
            break_points=legend.break_points.iter_rows(named=True),
        )
        for legend in LEGENDS.values()
    ]


@app.get("/v1/datasets.{format}", tags=["Catalog"])
async def datasets(format: DataFormat, compression: Annotated[Compression, Query()]):
    validate_compression(compression, format)
    lf = glob_lazy_frame(required_cols=DATASET_DIMENSIONS).select(cs.exclude("path"))
    source = LazyFrameSource(lf)
    return PolarsStreamingResponse(source, format, compression)


@app.get(
    "/v1/datasets/{variable}/{grid}/{time_unit}/catalog.json",
    tags=["Catalog"],
)
async def query_catalog(
    filter_dimensions: Annotated[FilterDimensions, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
):
    catalog_query = CatalogQuery(dataset_dimensions, filter_dimensions)
    return JSONResponse(content=catalog_query.execute())


@app.get(
    "/v1/datasets/{variable}/{grid}/{time_unit}/data.{format}",
    tags=["Data and Analysis"],
)
async def get_model_data(
    format: DataFormat,
    query: Annotated[DataQuery, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
):
    compression = query.compression
    validate_compression(compression, format)
    dataset = Dataset(dataset_dimensions, query.filter_dimensions, query.options)
    validate_format(format, dataset.output_type)
    await dataset.apply_params()
    if dataset.output_type == OutputType.raster:
        return RasterioStreamingResponse(dataset, format, compression)
    return PolarsStreamingResponse(dataset, format, compression)


@app.get(
    "/v1/datasets/{variable}/{grid}/{time_unit}/data/{z}/{x}/{y}.{format}",
    tags=["Data and Analysis"],
)
async def get_model_tile(
    format: TileFormat,
    z: int,
    x: int,
    y: int,
    query: Annotated[TileQuery, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
):
    dataset = Dataset(dataset_dimensions, query.filter_dimensions, query.options)
    await dataset.apply_params()
    dataset.validate_tile()
    return RasterioTileResponse(dataset, x, y, z, format)


@app.get(
    "/v1/datasets/{variable}/{grid}/{time_unit}/visualization/{visualization_type}.json",
    tags=["Visualization"],
)
async def get_visualization(
    request: Request,
    visualization_type,
    query: Annotated[DataQuery, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
):
    VisualizationClass = VISUALIZATIONS[visualization_type]
    visualization = VisualizationClass(
        variable=dataset_dimensions.variable,
        grid=dataset_dimensions.grid,
        time_unit=dataset_dimensions.time_unit,
        request=request,
        **dict(query),
    )
    return JSONResponse(visualization.spec)
