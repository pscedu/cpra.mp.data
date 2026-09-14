from collections import defaultdict
from functools import lru_cache
from typing import Annotated, Any

import polars.selectors as cs
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlmodel import Session, create_engine, select

from cpra.mp.data.api.crosswalk import Crosswalk
from cpra.mp.data.api.dataset import Dataset
from cpra.mp.data.api.models import (
    Compression,
    CrosswalkQuery,
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
from cpra.mp.data.api.pipeline import PipelineError, parse_pipeline, pipeline_from_query
from cpra.mp.data.api.pipeline_exec import compile_pipeline
from cpra.mp.data.api.raster import RasterioStreamingResponse, RasterioTileResponse
from cpra.mp.data.api.sources import LazyFrameSource
from cpra.mp.data.api.tabular import PolarsStreamingResponse
from cpra.mp.data.api.types import CrosswalkGrid, DataFormat, OutputType, TileFormat
from cpra.mp.data.api.types import Grid as GridName
from cpra.mp.data.api.utils import (
    get_visualization_class,
    validate_compression,
    validate_format,
)
from cpra.mp.data.api.vector import VectorTileResponse
from cpra.mp.data.catalog import CatalogQuery
from cpra.mp.data.config import MPD_URI, STREAM_THRESHOLD
from cpra.mp.data.dimensions import DATASET_DIMENSIONS, DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.legends import LEGENDS
from cpra.mp.data.utils import glob_lazy_frame
from cpra.mp.data.variables import VARIABLES
from cpra.mp.data.visualizations import VISUALIZATIONS


@lru_cache(maxsize=1)
def get_mpd_engine():
    """Create the MPD database engine.

    Built lazily so that importing the app doesn't require a database: MPD_URI
    defaults to an empty string, which create_engine rejects, and only the
    dimensions endpoint actually reads from the MPD.
    """
    return create_engine(MPD_URI)


def get_mpd_session():
    with Session(get_mpd_engine()) as session:
        yield session


MPDSession = Annotated[Session, Depends(get_mpd_session)]

# `value_where` was retired in favor of a single `where` parameter that accepts both
# dimension and value conditions (see `Options.where`'s comment and
# `api.expressions.classify_where`). It is no longer a field on `Options`/`DataQuery`/
# `TileQuery`/`VisualizationQuery`, and FastAPI/pydantic silently ignore an unrecognized
# query parameter by default (no `model_config = ConfigDict(extra=...)` is set anywhere
# in models.py) -- confirmed empirically: a request with a bogus query param returns 200
# with the rest of the query answered normally. Left alone, a caller still sending
# `value_where` would have their filter silently vanish rather than fail loudly, so this
# dependency checks the raw query string directly (the only place the retired name is
# still visible at all) and rejects it explicitly.
def reject_retired_value_where(request: Request) -> None:
    if "value_where" in request.query_params:
        raise HTTPException(
            status_code=400,
            detail=(
                "'value_where' has been retired -- use 'where' instead, which now accepts both "
                "dimension and value conditions (e.g. 'calendar_year > 2040 AND value > 3')"
            ),
        )


RejectRetiredValueWhere = Depends(reject_retired_value_where)

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
            display_unit=var.display_unit,
            display_scale_factor=var.display_scale_factor,
            display_number_format=var.display_number_format,
            float_precision=var.float_precision,
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
                dim.values.select("value", "label", "color").iter_rows(named=True)
                if dim.values is not None
                else mpd_dimension_values.get(dim.name, None)
            ),
        )
        for dim in DIMENSIONS.values()
    ]


@app.get("/v1/grids.json", tags=["Catalog"])
def grids() -> list[Grid]:
    # TODO: Compress grid_cell IDs into ranges.
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


@app.get("/v1/grids/{source_grid}/crosswalk.{format}", tags=["Catalog"])
async def get_crosswalk(
    source_grid: CrosswalkGrid,
    format: DataFormat,
    query: Annotated[CrosswalkQuery, Query()],
):
    compression = query.compression
    validate_compression(compression, format)
    validate_format(format, OutputType.vector)
    crosswalk = Crosswalk(source_grid.value, query.grid_cell, query.to_grid)
    source = LazyFrameSource((await crosswalk.execute()).lazy())
    return PolarsStreamingResponse(source, format, compression, collect_in_memory=True)


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
    _reject_value_where: None = RejectRetiredValueWhere,
):
    compression = query.compression
    validate_compression(compression, format)
    dataset = Dataset(dataset_dimensions, query.filter_dimensions, query.options)
    validate_format(format, dataset.output_type)
    await dataset.apply_params()
    if dataset.output_type == OutputType.raster:
        return RasterioStreamingResponse(dataset, format, compression)

    data_size = dataset.estimate_data_size()
    collect_in_memory = data_size is not None and data_size <= (STREAM_THRESHOLD * pow(10, 6))
    return PolarsStreamingResponse(dataset, format, compression, collect_in_memory)


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
    _reject_value_where: None = RejectRetiredValueWhere,
):
    dataset = Dataset(dataset_dimensions, query.filter_dimensions, query.options)
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
    _reject_value_where: None = RejectRetiredValueWhere,
):
    dataset = Dataset(dataset_dimensions, query.filter_dimensions, query.options)
    await dataset.apply_params()
    dataset.validate_vector_tile()
    return VectorTileResponse(dataset, x, y, z)


def _check_output_format_agreement(pipeline, url_format):
    """`{format}`/`.png`|`.jpg` in the URL and the pipeline's own `output.format` are two
    spellings of the same thing (the tile route's z/x/y segments leave no other place to
    put it, and the flat data route mirrors that rather than special-casing one route to
    trust the URL and the other to trust the body). Both are always present (`Output.format`
    has no default), so this rejects outright rather than silently preferring one.
    """
    if pipeline.output.format != url_format:
        raise HTTPException(
            status_code=400,
            detail=(
                f"URL format {url_format.value!r} does not match the pipeline's own "
                f"output.format {pipeline.output.format!r}"
            ),
        )


async def _respond_to_analysis(pipeline, format):
    try:
        compiled = compile_pipeline(pipeline)
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    validate_format(format, compiled.output_type)
    await compiled.apply_params()

    if compiled.output_type == OutputType.raster:
        return RasterioStreamingResponse(compiled, format, Compression())

    data_size = compiled.estimate_data_size()
    collect_in_memory = data_size is not None and data_size <= (STREAM_THRESHOLD * pow(10, 6))
    return PolarsStreamingResponse(compiled, format, Compression(), collect_in_memory)


@app.post("/v1/analysis.{format}", tags=["Data and Analysis"])
async def post_analysis(format: DataFormat, body: dict[str, Any] = Body(...)):
    try:
        pipeline = parse_pipeline(body)
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _check_output_format_agreement(pipeline, format)
    return await _respond_to_analysis(pipeline, format)


@app.get("/v1/analysis.{format}", tags=["Data and Analysis"])
async def get_analysis(format: DataFormat, request: Request):
    try:
        pipeline = pipeline_from_query(dict(request.query_params))
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _check_output_format_agreement(pipeline, format)
    return await _respond_to_analysis(pipeline, format)


@app.get("/v1/analysis/{z}/{x}/{y}.{format}", tags=["Data and Analysis"])
async def get_analysis_tile(z: int, x: int, y: int, format: TileFormat, request: Request):
    try:
        pipeline = pipeline_from_query(dict(request.query_params))
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _check_output_format_agreement(pipeline, format)

    try:
        compiled = compile_pipeline(pipeline)
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await compiled.apply_params()
    compiled.validate_tile()
    return RasterioTileResponse(compiled, x, y, z, format)


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
    _reject_value_where: None = RejectRetiredValueWhere,
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
    _reject_value_where: None = RejectRetiredValueWhere,
):
    VisualizationClass = get_visualization_class(visualization_name, "custom")
    visualization = VisualizationClass(
        request=request,
        **dict(query),
    )
    return JSONResponse(visualization.spec)
