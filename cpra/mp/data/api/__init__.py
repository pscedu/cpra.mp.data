from collections import defaultdict
from typing import Annotated

import polars.selectors as cs
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlmodel import Session, create_engine, select

from cpra.mp.data.api.dataset import Dataset, TabularDataset
from cpra.mp.data.api.duckdb import get_limiter
from cpra.mp.data.api.models import (
    Compression,
    DataQuery,
    DatasetDimensions,
    Dimension,
    DimensionValue,
    FilterDimensions,
    Grid,
    Legend,
    MPDDimensionValue,
    TileQuery,
    Variable,
    Visualization,
    VisualizationQuery,
)
from cpra.mp.data.api.raster import RasterioTileResponse
from cpra.mp.data.api.sources import LazyFrameSource
from cpra.mp.data.api.tabular import PolarsStreamingResponse
from cpra.mp.data.api.types import DataFormat, OutputType, TileFormat
from cpra.mp.data.api.utils import (
    get_visualization_class,
    validate_compression,
    validate_format,
)
from cpra.mp.data.api.vector import VectorTileResponse
from cpra.mp.data.catalog import CatalogQuery
from cpra.mp.data.config import MPD_URI
from cpra.mp.data.dimensions import DATASET_DIMENSIONS, DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.legends import LEGENDS
from cpra.mp.data.utils import glob_lazy_frame
from cpra.mp.data.variables import VARIABLES
from cpra.mp.data.visualizations import VISUALIZATIONS

MPD_ENGINE = create_engine(MPD_URI)


def get_mpd_session():
    with Session(MPD_ENGINE) as session:
        yield session


MPDSession = Annotated[Session, Depends(get_mpd_session)]

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
            number_format=var.number_format,
            geographic_units=var.geographic_units,
            time_units=var.time_units,
            required_dimensions=var.required_dimensions,
            difference_legend_limit=var.difference_legend_limit,
        )
        for var in VARIABLES.values()
    ]


@app.get("/v1/dimensions.json", tags=["Catalog"])
def dimensions(session: MPDSession) -> list[Dimension]:
    mpd_dimension_values = defaultdict(list)
    for dv in session.exec(select(MPDDimensionValue)).all():
        mpd_dimension_values[dv.dimension].append(
            DimensionValue(value=dv.value_str if dv.value_int is None else dv.value_int, label=dv.label)
        )
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
                else mpd_dimension_values.get(dim.name, None)
            ),
        )
        for dim in DIMENSIONS.values()
    ]


@app.get("/v1/grids.json", tags=["Catalog"])
def grids() -> list[Grid]:
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
    "/v1/datasets/{variable}/{time_unit}/{grid}/catalog.json",
    tags=["Catalog"],
)
async def query_catalog(
    filter_dimensions: Annotated[FilterDimensions, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
):
    catalog_query = CatalogQuery(dataset_dimensions, filter_dimensions)
    try:
        result = catalog_query.execute()
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return JSONResponse(content=result)


@app.get(
    "/v1/datasets/{variable}/{time_unit}/{grid}/data.{format}",
    tags=["Data and Analysis"],
)
async def get_model_data(
    format: DataFormat,
    query: Annotated[DataQuery, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
):
    """
    Download model data in the requested format.

    The `sql` parameter accepts a full SQL SELECT statement that operates on a
    pre-registered view called ``data``.  When `summarize_by` is set a crosswalk
    view named ``xw`` (columns: src_id, dst_id, scale_factor) is also available.

    Each query runs in an isolated DuckDB connection with a configurable memory
    limit.  Queries that exceed the limit receive a 507 response.  If the server
    is at maximum concurrent query capacity the endpoint returns 503.
    """
    limiter = get_limiter()
    if limiter.available_tokens < 1:
        raise HTTPException(status_code=503, detail="Server is at query capacity — retry shortly")

    compression = query.compression
    validate_compression(compression, format)

    async with limiter:
        dataset = TabularDataset(dataset_dimensions, query)
        await dataset.apply_params()

        if dataset.output_type == OutputType.raster:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Raster grid data without grid_cell cannot be downloaded as tabular. "
                    "Use the tile endpoint (/data/{z}/{x}/{y}.{format}) to retrieve raster data, "
                    "or add grid_cell coordinates to extract point values."
                ),
            )

        validate_format(format, dataset.output_type)
        return PolarsStreamingResponse(dataset.source, format, compression)


@app.get(
    "/v1/datasets/{variable}/{time_unit}/{grid}/data/{z}/{x}/{y}.{format}",
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
    dataset = Dataset(dataset_dimensions, query.filter_dimensions, query)
    await dataset.apply_params()
    dataset.validate_tile()
    return RasterioTileResponse(dataset, x, y, z, format)


@app.get(
    "/v1/datasets/{variable}/{time_unit}/{grid}/tiles/{z}/{x}/{y}.pbf",
    tags=["Data and Analysis"],
)
async def get_vector_tile(
    z: int,
    x: int,
    y: int,
    query: Annotated[TileQuery, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
):
    dataset = Dataset(dataset_dimensions, query.filter_dimensions, query)
    await dataset.apply_params()
    dataset.validate_vector_tile()
    return VectorTileResponse(dataset, x, y, z)


@app.get("/v1/visualizations.json", tags=["Visualization"])
def visualizations() -> list[Visualization]:
    return [viz.Meta.__dict__ for viz in VISUALIZATIONS.values()]


@app.get(
    "/v1/datasets/{variable}/{time_unit}/{grid}/visualization/{visualization_name}.json",
    tags=["Visualization"],
)
async def get_dataset_visualization(
    request: Request,
    visualization_name,
    query: Annotated[VisualizationQuery, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
):
    VisualizationClass = get_visualization_class(visualization_name, "dataset")
    query_dict = {k: v for k, v in dict(query).items() if k != "grid"}
    visualization = VisualizationClass(
        variable=dataset_dimensions.variable,
        grid=dataset_dimensions.grid,
        time_unit=dataset_dimensions.time_unit,
        request=request,
        **query_dict,
    )
    return JSONResponse(visualization.spec)


@app.get(
    "/v1/visualizations/{visualization_name}.json",
    tags=["Visualization"],
)
async def get_visualization(
    request: Request,
    visualization_name,
    query: Annotated[VisualizationQuery, Query()],
):
    VisualizationClass = get_visualization_class(visualization_name, "custom")
    visualization = VisualizationClass(
        request=request,
        **dict(query),
    )
    return JSONResponse(visualization.spec)
