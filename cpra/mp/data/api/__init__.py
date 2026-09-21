from collections import defaultdict
from functools import lru_cache
from typing import Annotated, Any

import polars.selectors as cs
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlmodel import Session, create_engine, select

from cpra.mp.data.analysis.request.ops import validate_shapes
from cpra.mp.data.analysis.request.pipeline import (
    parse_pipeline,
    pipeline_from_query,
)
from cpra.mp.data.analysis.request.types import (
    CrosswalkGrid,
    DataFormat,
    OutputType,
    TileFormat,
)
from cpra.mp.data.api.boundary import install_analysis_boundary
from cpra.mp.data.api.crosswalk import Crosswalk
from cpra.mp.data.api.data_query import build_data_query, resolve_grid_cell
from cpra.mp.data.api.models import (
    Analysis,
    AnalysisDataQuery,
    AnalysisQuery,
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
from cpra.mp.data.api.pipeline import (
    respond_to_analysis as respond_to_analysis_pipeline,
    respond_to_analysis_tile as respond_to_analysis_pipeline_tile,
    respond_to_analysis_vector_tile as respond_to_analysis_pipeline_vector_tile,
)
from cpra.mp.data.api.sources import LazyFrameSource
from cpra.mp.data.api.tabular import PolarsStreamingResponse
from cpra.mp.data.api.utils import (
    ANALYSES,
    AnalysisRegistryError,
    DatasetRef,
    UnknownAnalysisError,
    get_analysis_class,
    get_visualization_class,
    resolve_analysis,
    validate_compression,
    validate_format,
)
from cpra.mp.data.catalog import CatalogQuery
from cpra.mp.data.config import MPD_URI
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

# The flat data/tile/visualization endpoints only ever accept
# `DataOptions`'s fields (dimension filters, `grid_cell`/`where`/`sort`/`select`/
# `color_map`, plus compression) -- every parameter that used to describe how a compiled
# pipeline *executes* (`negate`/`multiply10`/`multiply100`, the indexed
# `calculate.N.expression`/`calculate.N.where`, `group_by`/`group_aggregate`,
# `summarize_by`/`summarize_aggregate`/`summarize_grid_cell`, `unpivot`) is retired from
# the HTTP surface, along with the already-retired `value_where`. Each of those now has a
# pipeline STEP as its only spelling (`group`, `calculate`, `project`); they simply have
# no flat HTTP spelling anymore, on either the plain data endpoints or the pipeline
# surface's own query-string binding.
#
# FastAPI/pydantic silently ignore an unrecognized query parameter by default (no
# `model_config = ConfigDict(extra=...)` is set anywhere in models.py) -- confirmed
# empirically: a request with a bogus query param returns 200 with the rest of the query
# answered normally. Left alone, a caller still sending one of these would have it
# silently vanish rather than fail loudly, so this dependency checks the raw query
# string directly (the only place a retired name is still visible at all) and rejects it
# explicitly, naming the pipeline equivalent that replaces it.
_RETIRED_PARAMS = {
    "value_where": "'where', which now accepts both dimension and value conditions (e.g. 'calendar_year > 2040 AND value > 3')",
    "negate": "a 'difference' named analysis (see /v1/analyses.json), or the 'calculate' pipeline step directly (negate=X is calculate expression 'value * -1' with where=X)",
    "multiply10": "the 'calculate' pipeline step directly (multiply10=X is calculate expression 'value * 10' with where=X)",
    "multiply100": "the 'calculate' pipeline step directly (multiply100=X is calculate expression 'value * 100' with where=X)",
    "group_by": "the 'group' pipeline step (see /v1/analysis.json or a named analysis)",
    "group_aggregate": "the 'group' pipeline step (see /v1/analysis.json or a named analysis)",
    # There is no `summarize` op in the pipeline grammar -- the ops are project/join/
    # filter/calculate/group. A `project` step's `to`/`aggregate` is what `summarize_by`/
    # `summarize_aggregate` used to spell, and a `grid_cell` `filter` placed AFTER that
    # project is what `summarize_grid_cell` used to spell. Name those, not a step that
    # does not exist -- a 400 that points at nothing is worse than no hint.
    "summarize_by": "a 'project' step's `to` (the grid to summarize onto; see /v1/analysis.json or a named analysis)",
    "summarize_aggregate": "a 'project' step's `aggregate` (see /v1/analysis.json or a named analysis)",
    "summarize_grid_cell": "a 'filter' step naming `grid_cell`, placed after the 'project' step (see /v1/analysis.json or a named analysis)",
    "unpivot": "nothing -- data is already served in long format (grid_cell/value columns); the parameter was already a documented no-op",
    # The generic calculate entries used to arrive as indexed, dotted query parameters
    # (`calculate.1.expression=...&calculate.1.where=...`), parsed off the raw query
    # string since FastAPI's query models can't bind dotted names. There is no flat HTTP
    # spelling for `calculate` at all anymore -- checked below via a startswith/exact-key
    # match rather than a plain dict lookup, since the retired name is a whole family of
    # keys (`calculate`, `calculate.1.expression`, `calculate.2.where`, ...), not one.
    "calculate": "the 'calculate' pipeline step directly (see /v1/analysis.json, or a named analysis's own pipeline)",
}


def reject_retired_params(request: Request) -> None:
    found = [key for key in _RETIRED_PARAMS if key in request.query_params]
    if any(key == "calculate" or key.startswith("calculate.") for key in request.query_params):
        if "calculate" not in found:
            found.append("calculate")

    if found:
        details = "; ".join(f"{name!r} has been retired -- use {_RETIRED_PARAMS[name]}" for name in found)
        raise HTTPException(status_code=400, detail=details)


RejectRetiredParams = Depends(reject_retired_params)


# Compression is bound as its own query model (`Query(Compression)`) on the flat analysis
# routes below rather than riding inside the pipeline -- it's a transport concern, and
# `pipeline.pipeline_from_query` rejects any top-level query key that isn't
# `input.*`/`step.*`/`output.*` (see that function's docstring), so `compression_algorithm`/
# `compression_level` have to be stripped out of the raw query dict before it reaches the
# pipeline decoder. This mirrors how `get_dataset_visualization` strips `grid` before handing
# its own query dict onward.
_COMPRESSION_QUERY_FIELDS = frozenset(Compression.model_fields)

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


# The translation point for `cpra.mp.data.analyses.resolve_analysis` --
# a distinct, FastAPI-free exception family from `request.errors.RequestError` (see
# `cpra/mp/data/analyses/base.py`'s `AnalysisRegistryError`, a plain `ValueError`
# subclass), raised by `resolve_analysis` itself for the one case `get_analysis_class`
# cannot pre-empt: `InvalidAnalysisParametersError`, when an analysis's own `Parameters`
# validation or `validate_selection` cardinality rule (or the `Pipeline` its `pipeline()`
# builds) rejects otherwise-well-formed request params/filters. Without this handler that
# case would propagate as a bare 500 instead of the 400 its docstring calls for.
# `UnknownAnalysisError`/`AnalysisTypeMismatchError` are already fully pre-empted by every
# route's `get_analysis_class(name, type)` call before `resolve_analysis` is ever reached
# (same registry, same `Meta.type` check), so this handler is unreachable for those two in
# practice -- registered anyway so the mapping is correct if that ever changes.
@app.exception_handler(AnalysisRegistryError)
async def analysis_registry_error_handler(request: Request, exc: AnalysisRegistryError) -> JSONResponse:
    status_code = 404 if isinstance(exc, UnknownAnalysisError) else 400
    return JSONResponse(status_code=status_code, content={"detail": str(exc)})


# The analysis engine's own counterpart to the handler above -- covers `request.errors.
# RequestError` and its members, `request.expressions.ExpressionError`, `execution.
# execute.LoweringNotImplementedError`, and Polars' own `ColumnNotFoundError`, for every
# route that runs a pipeline. See `api.boundary`'s own module docstring for the full
# mapping and why it is a separate, independent set of handlers from the one above rather
# than a unification of them.
install_analysis_boundary(app)


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
    tags=["Data"],
)
async def get_model_data(
    format: DataFormat,
    query: Annotated[DataQuery, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
    _reject_retired: None = RejectRetiredParams,
):
    # `validate_shapes` replaces the old `validate_format(format, dataset.output_type)`
    # entirely, and covers strictly more: `output_type` was "vector unless this is a raster
    # grid with no grid_cell/summarize_by" (`analysis.output.resolve_output_type`), i.e. a
    # restatement of the final frame's shape derived from the two options that can change
    # it, where `validate_shapes` types the frame through the ops themselves. Both reject
    # the same two requests -- a raster grid asked for a tabular format, and a point query
    # asked for `tif` -- with a different message; nothing that used to pass now fails.
    #
    # `estimate_data_size`/`STREAM_THRESHOLD`/`collect_in_memory` have no counterpart here:
    # `PipelineTabularResponse` always streams (see its docstring), because estimating a
    # compiled plan's result size ahead of running it is not implemented. The old
    # size-based collect-in-memory shortcut for a small result is the one thing this
    # rebind drops outright.
    compression = query.compression
    validate_compression(compression, format)
    resolved = await build_data_query(
        dataset_dimensions, query.filter_dimensions, query.options, format.value
    )
    validate_shapes(resolved.pipeline)
    return await respond_to_analysis_pipeline(
        resolved.pipeline,
        format.value,
        compression,
        relabel=resolved.relabel,
        float_precision=resolved.float_precision,
    )


@app.get(
    "/v1/datasets/{variable}/{time_unit}/{grid}/data/{z}/{x}/{y}.{format}",
    tags=["Data"],
)
async def get_model_tile(
    format: TileFormat,
    z: int,
    x: int,
    y: int,
    query: Annotated[TileQuery, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
    _reject_retired: None = RejectRetiredParams,
):
    # `dataset.validate_tile()`'s two rules, both preserved but now enforced in two
    # different places. "Not a tabular result" is `validate_shapes` below: `png`/`jpg` are
    # `RASTER_OUTPUT_FORMATS` members, so a long final frame -- which a `grid_cell` point
    # query on a raster grid produces, and which a vector grid produces unconditionally --
    # is a 400 there, the same rejection `resolve_output_type`-plus-`validate_raster_tile`
    # made. "Exactly one band" is `execution.sink.render_tile`'s own check, which raises
    # `ShapeError` (-> 400) at render time rather than at validate time: it needs the band
    # count, which is only known once the catalog has been read.
    #
    # `resolved.relabel` is deliberately not forwarded: a `grid_cell` that resolves to
    # coordinate labels only arises on a raster grid, and that request cannot reach a
    # renderer at all (the line above rejects it), so there is never a tile response with
    # a `grid_cell` column to relabel.
    resolved = await build_data_query(
        dataset_dimensions, query.filter_dimensions, query.options, format.value
    )
    validate_shapes(resolved.pipeline)
    return await respond_to_analysis_pipeline_tile(resolved.pipeline, x, y, z, format.value)


@app.get(
    "/v1/datasets/{variable}/{time_unit}/{grid}/tiles/{z}/{x}/{y}.pbf",
    tags=["Data"],
)
async def get_vector_tile(
    z: int,
    x: int,
    y: int,
    query: Annotated[TileQuery, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
    _reject_retired: None = RejectRetiredParams,
):
    # `output.format` is a stand-in, not a choice with consequences. `pbf` is not a member
    # of `DataFormat`/`TileFormat` (see `request.pipeline.Output.format`), so a pipeline
    # cannot spell this route's actual output format at all, and nothing here renders from
    # `output.format`: `respond_to_analysis_pipeline_vector_tile` reads the PMTiles
    # boundary layer and injects values into it, never applying `output`'s own shaping
    # (see its docstring). The one thing `output.format` still decides is which half of
    # `validate_shapes`'s sink-shape rule applies, and every TABULAR format picks the same
    # half -- "the final frame must be long" -- which is exactly this route's requirement,
    # so any of them would behave identically here.
    #
    # That check is half of what this route needs: a raster grid with no `grid_cell`
    # leaves a band-wide final frame and is a 400 there. The other half -- the final grid
    # must actually BE a vector grid, which a raster point query's long-on-a-raster-grid
    # frame is not -- is `api.pipeline._require_vector_final_frame`'s `ShapeError`, also a
    # 400. A request that could never have had a PMTiles file is a client error, not a
    # missing file, so neither half answers 404.
    resolved = await build_data_query(
        dataset_dimensions, query.filter_dimensions, query.options, DataFormat.ndjson.value
    )
    validate_shapes(resolved.pipeline)
    return await respond_to_analysis_pipeline_vector_tile(resolved.pipeline, x, y, z)


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


@app.post("/v1/analysis.{format}", tags=["Analysis"])
async def post_analysis(
    format: DataFormat,
    compression: Annotated[Compression, Query()],
    body: dict[str, Any] = Body(...),
):
    # `request.pipeline.parse_pipeline`'s own exceptions (`RequestError` and members,
    # `ExpressionError`) are NOT caught here -- they flow to `api.boundary`'s registered
    # handlers rather than a route-local `except`, so every route maps them identically.
    pipeline = parse_pipeline(body)
    _check_output_format_agreement(pipeline, format)
    validate_compression(compression, format)
    validate_shapes(pipeline)
    return await respond_to_analysis_pipeline(pipeline, format.value, compression)


@app.get("/v1/analysis.{format}", tags=["Analysis"])
async def get_analysis(format: DataFormat, compression: Annotated[Compression, Query()], request: Request):
    # `compression` is bound above as its own query model; strip its fields out of the raw
    # query dict before decoding the rest as a pipeline -- see `_COMPRESSION_QUERY_FIELDS`.
    pipeline_params = {
        k: v for k, v in request.query_params.items() if k not in _COMPRESSION_QUERY_FIELDS
    }
    pipeline = pipeline_from_query(pipeline_params)
    _check_output_format_agreement(pipeline, format)
    validate_compression(compression, format)
    validate_shapes(pipeline)
    return await respond_to_analysis_pipeline(pipeline, format.value, compression)


# The vector-tile counterpart to `get_analysis_tile` below, mirroring `get_vector_tile`
# (the data-route equivalent) as closely as the pipeline surface allows. Unlike the
# `.{format}` tile route, there is no `{format}` path segment to bind and cross-check
# against `pipeline.output.format` here: `pbf` is not a member of `DataFormat`/`TileFormat`
# (see `request.pipeline.Output.format`'s docstring), so a pipeline can never spell its
# output format as `pbf` and `_check_output_format_agreement` has nothing on the URL side
# to compare against. `api.pipeline._require_vector_final_frame` performs the check that
# actually matters instead: it rejects outright a pipeline whose final frame isn't long on
# a vector grid, the same rule `get_vector_tile` enforces on the data route. (Whatever
# format the pipeline's own `output` block declares -- e.g. `csv` -- is irrelevant here;
# it's never used to render the tile.)
#
# Must be declared before `get_analysis_tile`'s `.{format}` route: both match
# `/v1/analysis/{z}/{x}/{y}.pbf` (`{format}` is a wildcard suffix), and FastAPI/Starlette
# dispatches to the first route whose path matches, trying dependency validation only
# after that choice is made -- if `.{format}` were tried first, `"pbf"` would fail to
# coerce into `TileFormat` and this route would never be reached.
@app.get("/v1/analysis/{z}/{x}/{y}.pbf", tags=["Analysis"])
async def get_analysis_vector_tile(z: int, x: int, y: int, request: Request):
    # No `_check_output_format_agreement` call here: `pbf` is not a member of
    # `request.types.DataFormat`/`TileFormat` (see `request.pipeline.Output.format`'s
    # docstring), so there is nothing on the URL side to cross-check
    # `pipeline.output.format` against. `api.pipeline.
    # respond_to_analysis_pipeline_vector_tile`'s own `_require_vector_final_frame` check
    # (raising `ShapeError`, -> 400 via `api.boundary`) is what enforces the vector-grid
    # requirement instead -- see that helper's docstring for why
    # `request.ops.validate_shapes` cannot perform this particular check itself.
    #
    # `validate_shapes` still runs here despite that gap -- it is the only thing that
    # enforces every OTHER sink-shape/aggregate-legality rule (the raster/vector sink
    # cross-check is simply not among them for `pbf`, since `output.format` never says
    # `pbf`), and `execution.plan.compile_plan` documents itself as expecting to be called
    # only after it already has.
    pipeline = pipeline_from_query(dict(request.query_params))
    validate_shapes(pipeline)
    return await respond_to_analysis_pipeline_vector_tile(pipeline, x, y, z)


@app.get("/v1/analysis/{z}/{x}/{y}.{format}", tags=["Analysis"])
async def get_analysis_tile(z: int, x: int, y: int, format: TileFormat, request: Request):
    pipeline = pipeline_from_query(dict(request.query_params))
    _check_output_format_agreement(pipeline, format)
    validate_shapes(pipeline)
    return await respond_to_analysis_pipeline_tile(pipeline, x, y, z, format.value)


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
    _reject_retired: None = RejectRetiredParams,
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
    _reject_retired: None = RejectRetiredParams,
):
    VisualizationClass = get_visualization_class(visualization_name, "custom")
    visualization = VisualizationClass(
        request=request,
        **dict(query),
    )
    return JSONResponse(visualization.spec)


# ----------------------------------------------------------------------------------------
# Named analyses -- a registered pipeline plus parameters, resolved through
# `cpra.mp.data.analyses.resolve_analysis` into a real `Pipeline` and answered through
# EXACTLY the same `api.pipeline` responders the explicit `/v1/analysis.*` routes above
# already use. This is sugar, never a second engine, so nothing here reads a file, opens a
# dataset, or reimplements a compiler rule; it only builds the request-level inputs
# `resolve_analysis` asks for (including resolving `grid_cell` against real geometry, which
# the registry deliberately cannot do -- see `_resolve_dataset_analysis`) and applies the
# request-level `Output` overrides (`format`/`color_map`/`sort`/`select`) that are NEVER the
# registry's concern -- color_map is a request parameter here, not a registry field.
# ----------------------------------------------------------------------------------------


def _analysis_filters(query: AnalysisQuery, *, dimensions: bool = True) -> dict:
    """Dimension filters + `grid_cell` + `where` from the request, forwarded verbatim to
    `resolve_analysis`'s `filters` argument.

    `dimensions=False` for a `type="custom"` analysis, which has no `{variable}/
    {time_unit}/{grid}` URL segment: with no dataset in the request there is nothing for
    a dimension filter to narrow, so a dimension-NAMED query parameter on one of those
    routes is the analysis's own parameter instead (`exit-criterion` declares
    `model_group_id`/`scenario_id`, which are also real dimension names) and is routed to
    `params` by `_analysis_extra_params`'s matching `known` set. Only the two reserved
    keys below survive here.

    `where` is bundled in here rather than spliced into the resolved pipeline's own
    `steps` afterward: the pipeline's shape (how many steps, and which one a selection
    belongs on) is the registry's to know, not this route's, and grafting an extra
    `FilterStep` in from outside risks becoming a second, route-level pipeline engine --
    a `filter` appended after a `group` step, for instance, means something quite
    different from the same `filter` composed into the leading one. Letting
    `resolve_analysis` fold it into its own pipeline construction (see
    `analyses.base.selection_filter_where`) keeps that structural knowledge in one place.

    `grid_cell` is left here in its raw request form. It is only ever USED for a vector
    grid's literal-digit-id spelling -- every other spelling is resolved against real
    geometry first and passed separately as `resolve_analysis(grid_cell_where=...)`,
    which takes precedence over this key. That is the same division of labor
    `api.data_query.build_data_query` and `request.data_query.build_data_query_pipeline`
    already have on the flat data routes, for the same reason: resolving a coordinate
    needs `io/`, which nothing under `request/` (or in the registry) may import.
    """
    filters = {}
    if dimensions:
        filters = {
            k: v
            for k, v in dict(query).items()
            if k in FilterDimensions.model_fields and v is not None
        }
    if query.grid_cell is not None:
        filters["grid_cell"] = query.grid_cell
    if query.where is not None:
        filters["where"] = query.where
    return filters


def _analysis_extra_params(request: Request, known: frozenset) -> dict:
    """Everything in the query string besides `known` (the bound query model's own
    fields) is the named analysis's own declared parameter set -- `dimension`/`order` for
    `difference`, `model_group_id`/`scenario_id`/`marsh_threshold`/`salinity_threshold`
    for `exit-criterion`, etc. There is no fixed field set for these at the route level
    (it varies per analysis), so they are read directly off the raw query string and
    handed to `resolve_analysis(params=...)` for the registry itself to validate --
    the same "no flat spelling a route-level query model can bind" reasoning
    `reject_retired_params` above uses to check the raw query string directly for the
    retired `calculate.*` family.
    """
    return {k: v for k, v in request.query_params.items() if k not in known}


def _apply_output_overrides(pipeline, query: AnalysisQuery, format=None) -> None:
    """Overwrite the resolved pipeline's `Output` with the request's own `format`/
    `color_map`/`sort`/`select` -- never the registry's concern: the same
    value can be drawn with different ramps, and the portal already varies it per map.
    `format` is omitted (left as whatever `resolve_analysis` produced) for `pipeline.json`,
    which has no `{format}`/`.pbf` URL segment to source one from.
    """
    if format is not None:
        pipeline.output.format = format
    if query.color_map is not None:
        pipeline.output.color_map = query.color_map
    if query.sort is not None:
        pipeline.output.sort = query.sort
    if query.select is not None:
        pipeline.output.select = query.select


def _dataset_ref(dataset_dimensions: DatasetDimensions):
    return DatasetRef(
        variable=dataset_dimensions.variable,
        time_unit=dataset_dimensions.time_unit,
        grid=dataset_dimensions.grid,
    )


async def _resolve_dataset_analysis(analysis_name, request, query, dataset_dimensions):
    """The four dataset-scoped named-analysis routes' shared prologue: reject the name up
    front if it is unknown or `type="custom"`, resolve `grid_cell` against the dataset's
    own grid exactly as the flat data routes do, and resolve the analysis itself.

    Returns `(resolved, relabel)`. `relabel` is `api.data_query.resolve_grid_cell`'s
    absolute-id -> coordinate-string map for a raster point query and `None` otherwise;
    only the `data.{format}` route forwards it, for the same reason `get_model_tile`
    doesn't (a tile response has no `grid_cell` column to relabel).
    """
    get_analysis_class(analysis_name, "dataset")
    grid_cell_where, relabel = await resolve_grid_cell(dataset_dimensions.grid, query.grid_cell)
    resolved = resolve_analysis(
        analysis_name,
        dataset=_dataset_ref(dataset_dimensions),
        filters=_analysis_filters(query),
        params=_analysis_extra_params(request, frozenset(type(query).model_fields)),
        grid_cell_where=grid_cell_where,
    )
    return resolved, relabel


def _resolve_custom_analysis(analysis_name, request, query):
    """The three custom named-analysis routes' shared prologue.

    No `grid_cell` resolution counterpart to `_resolve_dataset_analysis`'s: a custom
    analysis names its own inputs (there is no `{grid}` URL segment, and `exit-criterion`
    reads two different grids), so there is no single grid to resolve a coordinate
    against. `grid_cell` on these routes therefore means literal ids, and a coordinate
    value is rejected by `request.data_query.grid_cell_conjunct` with a message saying so.

    The dimension/parameter split is the mirror image of the dataset routes': with no
    dataset in the URL there is nothing for a dimension filter to narrow, so the
    `FilterDimensions` half of the bound query model is subtracted from
    `_analysis_extra_params`'s `known` set and a dimension-named query parameter reaches
    the analysis's own `Parameters` instead. `exit-criterion` needs exactly that -- it
    declares `model_group_id`/`scenario_id`, both real dimension names, and would
    otherwise see them bound as filters it has no dataset to apply them to.
    """
    get_analysis_class(analysis_name, "custom")
    known = frozenset(type(query).model_fields) - frozenset(FilterDimensions.model_fields)
    return resolve_analysis(
        analysis_name,
        dataset=None,
        filters=_analysis_filters(query, dimensions=False),
        params=_analysis_extra_params(request, known),
    )


@app.get("/v1/analyses.json", tags=["Analyses"])
def analyses_catalog() -> list[Analysis]:
    return [analysis.Meta.__dict__ for analysis in ANALYSES.values()]


@app.get(
    "/v1/datasets/{variable}/{time_unit}/{grid}/analyses/{analysis_name}/data.{format}",
    tags=["Analyses"],
)
async def get_dataset_analysis_data(
    format: DataFormat,
    analysis_name: str,
    request: Request,
    query: Annotated[AnalysisDataQuery, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
):
    compression = query.compression
    validate_compression(compression, format)
    resolved, relabel = await _resolve_dataset_analysis(
        analysis_name, request, query, dataset_dimensions
    )
    pipeline = resolved.pipeline
    _apply_output_overrides(pipeline, query, format=format)
    validate_shapes(pipeline)
    return await respond_to_analysis_pipeline(
        pipeline, format.value, compression, relabel=relabel
    )


@app.get(
    "/v1/datasets/{variable}/{time_unit}/{grid}/analyses/{analysis_name}/data/{z}/{x}/{y}.{format}",
    tags=["Analyses"],
)
async def get_dataset_analysis_tile(
    format: TileFormat,
    analysis_name: str,
    z: int,
    x: int,
    y: int,
    request: Request,
    query: Annotated[AnalysisQuery, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
):
    resolved, _relabel = await _resolve_dataset_analysis(
        analysis_name, request, query, dataset_dimensions
    )
    pipeline = resolved.pipeline
    _apply_output_overrides(pipeline, query, format=format)
    validate_shapes(pipeline)
    return await respond_to_analysis_pipeline_tile(pipeline, x, y, z, format.value)


@app.get(
    "/v1/datasets/{variable}/{time_unit}/{grid}/analyses/{analysis_name}/tiles/{z}/{x}/{y}.pbf",
    tags=["Analyses"],
)
async def get_dataset_analysis_vector_tile(
    analysis_name: str,
    z: int,
    x: int,
    y: int,
    request: Request,
    query: Annotated[AnalysisQuery, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
):
    resolved, _relabel = await _resolve_dataset_analysis(
        analysis_name, request, query, dataset_dimensions
    )
    pipeline = resolved.pipeline
    _apply_output_overrides(pipeline, query)
    validate_shapes(pipeline)
    return await respond_to_analysis_pipeline_vector_tile(pipeline, x, y, z)


@app.get(
    "/v1/datasets/{variable}/{time_unit}/{grid}/analyses/{analysis_name}/pipeline.json",
    tags=["Analyses"],
)
async def get_dataset_analysis_pipeline(
    analysis_name: str,
    request: Request,
    query: Annotated[AnalysisQuery, Query()],
    dataset_dimensions: DatasetDimensions = Depends(),
):
    resolved, _relabel = await _resolve_dataset_analysis(
        analysis_name, request, query, dataset_dimensions
    )
    pipeline = resolved.pipeline
    _apply_output_overrides(pipeline, query)
    return JSONResponse(content=pipeline.model_dump(mode="json"))


@app.get("/v1/analyses/{analysis_name}/data.{format}", tags=["Analyses"])
async def get_custom_analysis_data(
    format: DataFormat,
    analysis_name: str,
    request: Request,
    query: Annotated[AnalysisDataQuery, Query()],
):
    compression = query.compression
    validate_compression(compression, format)
    resolved = _resolve_custom_analysis(analysis_name, request, query)
    pipeline = resolved.pipeline
    _apply_output_overrides(pipeline, query, format=format)
    validate_shapes(pipeline)
    return await respond_to_analysis_pipeline(pipeline, format.value, compression)


@app.get("/v1/analyses/{analysis_name}/data/{z}/{x}/{y}.{format}", tags=["Analyses"])
async def get_custom_analysis_tile(
    format: TileFormat,
    analysis_name: str,
    z: int,
    x: int,
    y: int,
    request: Request,
    query: Annotated[AnalysisQuery, Query()],
):
    resolved = _resolve_custom_analysis(analysis_name, request, query)
    pipeline = resolved.pipeline
    _apply_output_overrides(pipeline, query, format=format)
    validate_shapes(pipeline)
    return await respond_to_analysis_pipeline_tile(pipeline, x, y, z, format.value)


@app.get("/v1/analyses/{analysis_name}/tiles/{z}/{x}/{y}.pbf", tags=["Analyses"])
async def get_custom_analysis_vector_tile(
    analysis_name: str,
    z: int,
    x: int,
    y: int,
    request: Request,
    query: Annotated[AnalysisQuery, Query()],
):
    resolved = _resolve_custom_analysis(analysis_name, request, query)
    pipeline = resolved.pipeline
    _apply_output_overrides(pipeline, query)
    validate_shapes(pipeline)
    return await respond_to_analysis_pipeline_vector_tile(pipeline, x, y, z)


@app.get("/v1/analyses/{analysis_name}/pipeline.json", tags=["Analyses"])
async def get_custom_analysis_pipeline(
    analysis_name: str,
    request: Request,
    query: Annotated[AnalysisQuery, Query()],
):
    resolved = _resolve_custom_analysis(analysis_name, request, query)
    pipeline = resolved.pipeline
    _apply_output_overrides(pipeline, query)
    return JSONResponse(content=pipeline.model_dump(mode="json"))
