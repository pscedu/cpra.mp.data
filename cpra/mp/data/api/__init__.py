from collections import defaultdict
from functools import lru_cache
from typing import Annotated, Any

import polars.selectors as cs
from fastapi import Body, Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlmodel import Session, create_engine, select

from cpra.mp.data.analysis.request.descriptor import output_descriptor
from cpra.mp.data.analysis.request.ops import validate_shapes
from cpra.mp.data.analysis.request.pipeline import parse_pipeline
from cpra.mp.data.analysis.request.types import (
    CrosswalkGrid,
    DataFormat,
    OutputType,
    TileFormat,
)
from cpra.mp.data.api.boundary import install_analysis_boundary
from cpra.mp.data.api.catalog import catalog_body, output_body
from cpra.mp.data.api.crosswalk import Crosswalk
from cpra.mp.data.api.models import (
    RENDER_OPTIONS,
    Analysis,
    CatalogOutput,
    Compression,
    CrosswalkQuery,
    Dimension,
    DimensionValue,
    Grid,
    Legend,
    MPDDimensionValue,
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
from cpra.mp.data.api.ogc_provider import strip_ogc_params
from cpra.mp.data.api.qlr import build_qlr, export_legend, items_url, layer_label
from cpra.mp.data.api.unified import decode_request
from cpra.mp.data.api.utils import (
    ANALYSES,
    AnalysisRegistryError,
    UnknownAnalysisError,
    get_visualization_class,
    resolve_analysis,
    validate_compression,
    validate_format,
)
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

from starlette.middleware.wsgi import WSGIMiddleware  # noqa: E402
from pygeoapi.flask_app import APP as pygeoapi_app  # noqa: E402

app.mount("/v1/ogc", WSGIMiddleware(pygeoapi_app))


# The translation point for `cpra.mp.data.analyses.resolve_analysis` --
# a distinct, FastAPI-free exception family from `request.errors.RequestError` (see
# `cpra/mp/data/analyses/base.py`'s `AnalysisRegistryError`, a plain `ValueError`
# subclass). It is the ONLY thing standing between the registry and a 500 now: with one
# route per shape there is no longer a per-route `get_analysis_class(name, type)` gate
# ahead of `resolve_analysis` to pre-empt the unknown-name case, so every member of the
# family arrives here --
#
# - `UnknownAnalysisError` -> 404, for a name (at any position in a chain) that is not
#   registered. This used to be raised by `get_analysis_class` as an `HTTPException` for
#   the chain's HEAD only and by the registry for every later stage; one route means one
#   answer for every position.
# - everything else -> 400: `AnalysisChainError` (a later stage that is not
#   `kind="dimension"`), `AnalysisDatasetMismatchError` (an analysis that needs the
#   caller's dataset and was given none, or fixes its own inputs and was given one -- what
#   the retired `Meta.type` gate used to answer before the analysis was built at all), and
#   `InvalidAnalysisParametersError` (an analysis's own `Parameters` validation or
#   `validate_selection` rule rejecting otherwise-well-formed request params/filters).
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


# Liveness check for the container. Deliberately a plain `def`, not `async def`: FastAPI
# runs sync routes on anyio's default thread limiter, the same pool every tile and data
# read goes through (`anyio.to_thread.run_sync` in `api.pipeline`). When the data
# filesystem (Lustre) hangs, reads block uninterruptibly and pile up there until the pool
# is exhausted and every data route stops answering -- while async routes such as
# /openapi.json, served straight off the event loop, keep responding. Answering here
# therefore proves that both the event loop and a worker thread are free.
#
# It reads nothing from disk on purpose: one slow read is not a reason to restart, and a
# filesystem that is down outright would not be fixed by a restart loop. Outside /v1 so
# the ingress (which routes /v1 only) doesn't expose it.
@app.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    return {"status": "ok"}


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


@app.get("/v1/catalog.json", tags=["Catalog"])
async def get_catalog(request: Request):
    """What this request can still be steered by, and what it would produce.

    Replaces `query_catalog` (`/v1/datasets/{variable}/{time_unit}/{grid}/catalog.json`),
    the one route outside the four that was keyed on the dataset triple -- and the reason
    the triple could not simply be dropped from the portal along with the data routes. An
    analysis that fixes its own inputs could not state a triple, so it could not ask what it
    was allowed to select; the portal filled the gap by inventing one per derived variable
    and keeping it in agreement with the analysis by hand.

    Decoded exactly as `get_data` would decode it, with two differences that are the same
    two the visualization route has: there is no `{format}` segment for the URL to win with
    (`format=None`, so a plain download's sink is the one its own grid implies), and no
    response body is written, so the sink rules have nothing to be about
    (`validate_sink=False`). `allow_compression=False` because a JSON metadata response
    carries none -- sending one is a 400 rather than a silent no-op.

    `CatalogQuery`'s `ValueError` becomes a 404 as it always did: no catalog entries for a
    requested dataset, or -- newly reachable now that a request may read several -- no value
    of some dimension common to all of them.
    """
    decoded = await decode_request(
        request.query_params.multi_items(), format=None, allow_compression=False, validate_sink=False
    )
    try:
        return JSONResponse(content=catalog_body(decoded))
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@app.get("/v1/view.qlr", tags=["OGC"])
async def get_qgis_layer(request: Request):
    """A QGIS layer file for a data request: the OGC items URL for it plus the legend the
    portal's map would color it with (the analysis's, a difference's ramp, or the variable's).

    Takes the same query string `get_data` does, so the portal's "export to GIS" link is
    its current data request pointed at this path. It is decoded exactly as the OGC
    provider will decode it when QGIS fetches the items, so a request that would fail
    there fails here, at export, and not in a QGIS dialog later. Only a request that
    resolves to a vector grid can be exported: the OGC API serves features, not rasters.
    """
    items = strip_ogc_params(request.query_params.multi_items())
    decoded = await decode_request(items, format="parquet")
    descriptor = decoded.descriptor
    if descriptor.grid_type != "vector":
        raise HTTPException(
            status_code=400,
            detail=f"grid '{descriptor.grid}' is a raster grid and cannot be exported to a QGIS layer",
        )

    selection = ", ".join(f"{key}={value}" for key, value in items if key not in DATASET_DIMENSIONS)
    label = layer_label(decoded)
    layer_name = f"{label} ({selection})" if selection else label
    document = build_qlr(layer_name, items_url(items), export_legend(decoded))
    return Response(
        content=document,
        media_type="application/xml",
        headers={"Content-Disposition": 'attachment; filename="master_plan.qlr"'},
    )


# ----------------------------------------------------------------------------------------
# The consolidated surface: four GET routes plus one POST, all taking one parameter set,
# all decoded by `api.unified.decode_request`.
#
# What used to distinguish 17 routes was the URL's SHAPE -- whether it carried a
# `{variable}/{time_unit}/{grid}` prefix (which decided whether a dimension-named parameter
# was a filter or the analysis's own parameter), whether it carried an `/analyses/{name}`
# segment (which decided which of two resolvers ran), and whether it ended in `data.`,
# `data/{z}/{x}/{y}.` or `tiles/{z}/{x}/{y}.pbf`. None of that was a fact about the
# REQUEST; it was a fact about how the request had been spelled, and it forced the same
# taxonomy to be reproduced in `visualizations/base.py`, in `map.py` and again in the
# portal, each of which had to pick a route NAME before it could supply parameters. What is
# a real fact -- tabular-or-COG vs. tile vs. spec vs. pipeline JSON -- is what the four
# routes below are. Everything else is a query parameter; see `analysis.request.unified`'s
# module docstring for the five categories they sort into.
#
# The URL's `{format}` segment WINS over whatever the pipeline's own `output.format` says,
# on every route that has one -- applied once, inside `request.unified.build_pipeline`,
# rather than per route. This used to be a cross-check (`_check_output_format_agreement`)
# that 400-ed on disagreement, which read well for a hand-written POST body and for nothing
# else: a tile URL's `{format}` is the only place a tile request can state its format at
# all, and refetching one saved pipeline as `.csv` and as `.png` is an ordinary thing for a
# client to do (the portal does exactly that). One rule for every route is also what keeps
# `pipeline_to_query(pipeline)` a stable identity independent of the extension it was
# fetched with.
# ----------------------------------------------------------------------------------------


@app.get("/v1/data.{format}", tags=["Data"])
async def get_data(format: DataFormat, request: Request):
    """Tabular or COG output for any request: a plain dataset read, a named analysis or
    chain, a hand-written dotted pipeline, or any combination of the three.

    Replaces four routes -- `get_model_data`
    (`/v1/datasets/{variable}/{time_unit}/{grid}/data.{format}`), `get_analysis`
    (`/v1/analysis.{format}`), `get_dataset_analysis_data` and `get_custom_analysis_data`.
    Each of their spellings survives as a query string against this one path:
    `?variable=elev&time_unit=annual&grid=...`, `?input.data.variable=elev&...`,
    `?analysis=diff&dimension=calendar_year&...`, and -- the combination none of the four
    could express -- all three at once.

    `validate_compression` stays a route concern rather than a decoder one because it draws
    a different allowed-algorithm line per format (`tif` takes deflate/lzw/lzma/zstd/none,
    the tabular formats gzip/lz4/zstd/none) and the decoder is never told what the URL's
    extension means. `validate_shapes` is deliberately NOT called here: `decode_request`
    runs it as its own last step, before it describes the pipeline it is about to return.
    """
    decoded = await decode_request(request.query_params.multi_items(), format=format.value)
    validate_compression(decoded.compression, format)
    return await respond_to_analysis_pipeline(
        decoded.pipeline,
        format.value,
        decoded.compression,
        relabel=decoded.relabel,
        float_precision=decoded.float_precision,
    )


@app.post("/v1/data.{format}", tags=["Data"])
async def post_data(
    format: DataFormat,
    compression: Annotated[Compression, Query()],
    body: dict[str, Any] = Body(...),
):
    """The same output as `get_data` above, for a `Pipeline` written directly as JSON.

    Deliberately NOT routed through `decode_request`: the body already IS the whole
    pipeline, so there is nothing to partition, no analysis to resolve and no sugar to
    lower. `parse_pipeline` is the JSON codec's own entry point, and the GET route's dotted
    query form decodes to exactly the same object -- the two are the two spellings of one
    IR, which is why they share a path and a format segment. Everything after the decode is
    shared outright: the URL's format wins, compression is validated against it, shapes are
    checked, and the same responder answers.

    `parse_pipeline`'s own exceptions (`RequestError` and members, `ExpressionError`) are
    not caught here -- they flow to `api.boundary`'s registered handlers rather than a
    route-local `except`, so every route maps them identically.
    """
    pipeline = parse_pipeline(body)
    pipeline.output.format = format.value
    validate_compression(compression, format)
    validate_shapes(pipeline)
    return await respond_to_analysis_pipeline(pipeline, format.value, compression)


@app.get("/v1/data/{z}/{x}/{y}.{format}", tags=["Data"])
async def get_data_tile(format: TileFormat, z: int, x: int, y: int, request: Request):
    """One declaration for all three tile formats, raster and vector alike, and for every
    request shape -- replacing `get_model_tile`, `get_vector_tile`, `get_analysis_tile`,
    `get_dataset_analysis_tile`, `get_dataset_analysis_vector_tile`,
    `get_custom_analysis_tile` and `get_custom_analysis_vector_tile`: seven routes that
    differed only in how they reached a `Pipeline`, and in two cases only in whether `pbf`
    was spelled as a `{format}` or as a literal `/tiles/....pbf` suffix.

    It was two declarations until `pbf` became a `TileFormat` member, and the pair had a
    load-bearing declaration ORDER: both matched `.../{z}/{x}/{y}.pbf` (`{format}` is a
    wildcard suffix) and Starlette dispatches to the first path match, validating
    dependencies only afterwards, so `.{format}` declared first would have failed to coerce
    `"pbf"` into `TileFormat` and the vector route would never have been reached. With one
    route there is no order to get wrong.

    The `pbf` branch renders from the grid's PMTiles boundary archive rather than from
    `output`, so `sort`/`select`/`color_map` are not applied to it (see `api.pipeline.
    respond_to_analysis_vector_tile`); `output.format` is still set, because that is what
    `validate_shapes`'s `pbf` rule -- final frame long, on a vector grid -- keys on.

    `allow_compression=False` is this route's one parameter-set difference from `get_data`:
    a tile response carries no compression of its own, so sending one is a 400 here rather
    than the silent no-op it used to be (enforced structurally before, by the retired
    `api.models.TileQuery` simply not binding `Compression` -- which stopped it binding
    without rejecting it). `decoded.relabel` is likewise never forwarded: a `grid_cell` that
    resolves to coordinate labels only arises on a raster grid, and `validate_shapes`
    rejects that request against a raster tile format before a renderer is reached, so
    there is no tile response with a `grid_cell` column left to relabel.
    """
    decoded = await decode_request(
        request.query_params.multi_items(), format=format.value, allow_compression=False
    )
    if format is TileFormat.pbf:
        return await respond_to_analysis_pipeline_vector_tile(decoded.pipeline, x, y, z)
    return await respond_to_analysis_pipeline_tile(decoded.pipeline, x, y, z, format.value)


@app.get("/v1/visualizations.json", tags=["Visualization"])
def visualizations() -> list[Visualization]:
    return [viz.Meta.__dict__ for viz in VISUALIZATIONS.values()]


@app.get("/v1/visualizations/{visualization}.json", tags=["Visualization"])
async def get_visualization(
    visualization: str,
    request: Request,
    query: Annotated[VisualizationQuery, Query()],
):
    """One Vega-Lite spec route, replacing `get_dataset_visualization` and the old
    `get_visualization` -- two routes that differed only in whether the URL carried the
    dataset triple, and that were kept from each other's names by a `Meta.type` string that
    is now deleted.

    There is no fork left at all. The one case that was not a `DatasetVisualization` --
    `SedimentDynamics`, which built five single-variable sources and joined them in the
    browser -- reads the `sediment_dynamics` analysis's stacked output now, so every
    visualization draws a result and every one needs the render context decoded for it.

    `request_items` is the class's say in what gets decoded, and the only thing it is used
    for today is that chart: a visualization that IS the presentation of a named analysis
    declares it (`Meta.analysis`) and the decode is run with it at the head of the chain,
    so a client can ask for the chart by name without also knowing which analysis backs it.

    `query` is still bound, and is still what populates `self.params`: a visualization reads
    `self.height` and `self.params["show_water"]` as ordinary attributes carrying their
    DECLARED DEFAULTS, which a raw query string does not supply. What it is no longer is the
    thing that VALIDATES the request -- `decode_request` is, and it accepts strictly more
    than this model can declare (a dotted `step.2.op=`, an analysis's own `dimension=`).
    Pydantic query models ignore an unrecognized parameter rather than rejecting it, which
    is exactly what is wanted here now that something else is doing the rejecting.

    The render context is DERIVED from the resolved pipeline rather than read off the URL,
    and that is what retires the portal's `CANONICAL_DERIVED_DATASETS`: an analysis that
    fixes its own inputs has no honest triple to put in a URL, so the portal invented one
    and the API had to trust it. The deriving happens in `DatasetVisualization.descriptor`
    rather than here, so that a notebook-constructed visualization -- which has no route to
    have done it -- reaches the same answer. All this route passes is the decode itself.
    """
    VisualizationClass = get_visualization_class(visualization)
    kwargs = dict(query)
    kwargs["decoded"] = await decode_request(
        VisualizationClass.request_items(request.query_params.multi_items()),
        format=DataFormat.csv.value,
        render_fields=RENDER_OPTIONS,
        validate_sink=False,
    )

    return JSONResponse(VisualizationClass(request=request, **kwargs).spec)


def unrequested_output(name: str) -> CatalogOutput | None:
    """What the named analysis produces when it is asked with NOTHING -- no dataset, no
    selection, no parameters -- or `None` when that question has no answer.

    It has one exactly for an analysis that fixes its own inputs: `lnd_type_plus` reads
    three named datasets across two grids and outputs on a third, and nothing a caller sends
    can change that, so `morph_pixel_v002`/`annual` is a fact about the analysis. For every
    other entry the result is a fact about the dataset the caller will name, and there is
    nothing to report until they name one.

    The test is the resolve itself rather than a flag on `Meta`, because the resolve is
    where the answer actually lives -- `pipeline()` calls either `require_dataset` or
    `reject_dataset`, and a declared flag would be a second copy of that choice, free to
    drift from it. So: run the real `resolve_analysis` with empty everything and see whether
    it objects. `AnalysisRegistryError` covers every way it can (a missing dataset, a
    required parameter, a selection an analysis checks the cardinality of), and every one of
    them means the same thing here -- this analysis needs a request before it can say.

    Only `/v1/analyses.json` calls this, once per registered analysis per request. The
    resolve is pure pipeline construction -- no read, no file touched -- so it costs the
    same whether or not any data exists.
    """
    try:
        resolved = resolve_analysis(name, dataset=None, filters={}, params={})
    except AnalysisRegistryError:
        return None
    return output_body(output_descriptor(resolved.pipeline), resolved)


@app.get("/v1/analyses.json", tags=["Analyses"])
def analyses_catalog() -> list[Analysis]:
    # `model_dump()`, not `Meta.__dict__` (which is what the visualization catalog above
    # still does, since those `Meta`s are bare classes): every analysis `Meta` is an
    # `AnalysisMeta` instance, so this is its declared fields and nothing else -- no
    # `__module__`/`__qualname__` for `Analysis`'s `extra="ignore"` to quietly drop, and
    # no way for a field to be absent, since the model required it at import.
    #
    # `output` is the one field NOT in the `Meta` dump: it is derived from the analysis's
    # own `pipeline()` rather than declared beside it, which is the whole point (see
    # `unrequested_output`).
    return [
        Analysis(**analysis.Meta.model_dump(), output=unrequested_output(name))
        for name, analysis in ANALYSES.items()
    ]


# A single registered name, or a comma-separated chain (e.g. `lnd_conf,diff`) -- see
# `analyses.resolve_analysis_chain`. The path segment accepts either form unchanged, since
# a comma is not `/` and needs no extra URL-encoding, and the decoder splits it.
AnalysisName = Annotated[str, Path(description="A registered analysis name, or a comma-separated chain of them (e.g. `lnd_conf,diff`).")]


@app.get("/v1/analyses/{analysis}.json", tags=["Analyses"])
async def get_analysis_pipeline(analysis: AnalysisName, request: Request):
    """The `Pipeline` a named analysis resolves to, as JSON -- what
    `get_dataset_analysis_pipeline` and `get_custom_analysis_pipeline` each served for
    their half of the registry.

    This is the one route where the analysis is named in the PATH rather than as
    `analysis=`, and the reason is that the name is what is being asked about: the URL
    identifies a resolved pipeline the way `/v1/analyses.json` identifies the catalog it
    came from. The path segment is fed to the decoder as the `analysis` parameter and wins
    over an `analysis=` in the query string, so the two spellings cannot disagree. Every
    other parameter means exactly what it means on `get_data` -- which is the point: this
    route answers "what would `/v1/data.csv?analysis=...&...` actually run?", and the
    answer is only useful if the two decode the same request the same way.

    There is no `{format}` segment to source `output.format` from, so this is the one place
    `build_pipeline`'s "the URL's format always wins" rule has no URL to win from, and it is
    passed `format=None` to say so: the caller's own `output.format=` is honored, and
    failing that whatever the resolved analysis declared. The `csv` this used to assume when
    the caller wrote none was wrong for an analysis that fixes a RASTER sink of its own
    (`lnd_type_plus`, `sal_veg_investigation`): writing `csv` over it produced a pipeline
    whose final frame is band-wide under a tabular sink, which `validate_shapes` then
    rejected -- a 400 on a route that was asked only to describe the analysis.

    `analysis=download` resolves here too, to the plain dataset read the sentinel names.
    That is not a special case worth suppressing: it is a legitimate question (what does
    `?variable=elev&time_unit=annual&grid=...` compile to?) and it is how a caller inspects
    the pipeline behind a download link.
    """
    items = [(key, value) for key, value in request.query_params.multi_items() if key != "analysis"]
    decoded = await decode_request([("analysis", analysis), *items], format=None)
    return JSONResponse(content=decoded.pipeline.model_dump(mode="json"))
