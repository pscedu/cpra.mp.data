"""Route-facing adapters for the NEW analysis engine's `/v1/analysis...` pipeline surface:
turning an already-parsed, already-shape-validated `request.pipeline.Pipeline` into a real
HTTP response, for each of the four response shapes the four routes in `api/__init__.py`
need (tabular, `tif`, `png`/`jpg` tile, `pbf` tile).

**What is NOT this module's job.** Parsing the request body/query string into a `Pipeline`
(`request.pipeline.parse_pipeline`/`pipeline_from_query`), validating its shape
(`request.ops.validate_shapes`), and matching the URL's `{format}` segment against
`pipeline.output.format` (`api.__init__._check_output_format_agreement`) all happen in the
route, before any function here is called -- everything here assumes it already has a
structurally- and shape-valid `Pipeline` in hand. Every exception this module's own code
raises, or lets escape from `execution`/`request`, is caught by `api.boundary`'s handlers,
not by anything in here.

**Why `sink.render_tile`/`sink.sink_vector_tile` are called directly instead of through
`execution.execute.run_to_path`.** `run_to_path` is the composition helper for the two
"whole result, at a path" sinks (`sink_tabular`, `write_raster`); a tile needs per-request
parameters (`x`/`y`/`z`, pixel size, a legend) that call has nowhere to put -- see
`run_to_path`'s own docstring, which names this module's tile responders as the intended
caller for `render_tile`/`sink_vector_tile` directly.

**Reuse vs. purpose-built, by response shape:**

- Tabular (csv/ndjson/parquet/arrow): `PipelineTabularResponse` below reuses `api.tabular
  .StreamingThread` verbatim (the temp-file-plus-background-thread machinery,
  unchanged) but is NOT `api.tabular.PolarsStreamingResponse` itself, despite looking
  almost identical to it. The one load-bearing difference is documented on the class.
- `tif`: `PipelineRasterResponse` below -- a temp file, `FileResponse` on success, and
  `HTTPException(404)` when `run_to_path` reports no bands. `run_to_path` is synchronous,
  so it is driven off the event loop via `anyio.to_thread.run_sync`.
- `png`/`jpg`: purpose-built (`respond_to_analysis_tile`), specifically NOT
  `HTTPException` on a tile miss -- see `execution.sink.render_tile`'s and `empty_tile`'s
  docstrings: a 404 on an image URL still needs a decodable image body, which
  `HTTPException` would discard in favor of a JSON error document.
- `pbf`: purpose-built (`respond_to_analysis_vector_tile`), and again not
  `HTTPException` -- a miss here is a bodiless 204 (`execution.sink.sink_vector_tile`'s
  own docstring), which `HTTPException` cannot express either (it always carries a JSON
  body).
"""

import tempfile
from functools import partial

import anyio.to_thread
from fastapi import HTTPException
from starlette.responses import FileResponse, Response
from starlette.types import Receive, Scope, Send

from cpra.mp.data.analysis.execution.execute import execute, run_to_path
from cpra.mp.data.analysis.execution.plan import TileRequest, compile_plan
from cpra.mp.data.analysis.execution import sink
from cpra.mp.data.analysis.io import output
from cpra.mp.data.analysis.request.errors import ShapeError
from cpra.mp.data.analysis.request.ops import Shape
from cpra.mp.data.analysis.request.types import CompressionAlgorithm
from cpra.mp.data.api.boundary import resolve_detail, resolve_status
from cpra.mp.data.api.tabular import StreamingThread
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.variables import VARIABLES

# The rasterio profile key each compression algorithm spells its level with. Only the
# three that HAVE a level appear -- `lzw`/`none` are absent rather than mapped to `None`,
# so `.get()` returning nothing is what "this algorithm takes no level" means (see
# `PipelineRasterResponse.__call__`, the only reader).
COMPRESSION_LEVEL_OPTIONS = {
    CompressionAlgorithm.deflate: "zlevel",
    CompressionAlgorithm.lzma: "lzma_preset",
    CompressionAlgorithm.zstd: "zstd_level",
}

# The four tabular formats' MIME types -- copied from `api.tabular.PolarsStreamingResponse
# .mime_types` rather than imported from it, so this module's own response class does not
# depend on that one's internals beyond `StreamingThread`, which it genuinely reuses.
_TABULAR_MIME_TYPES = {
    "arrow": "application/vnd.apache.arrow.file",
    "csv": "text/csv",
    "ndjson": "application/x-ndjson",
    "parquet": "application/vnd.apache.parquet",
}

_TILE_MIME_TYPES = {"jpg": "image/jpeg", "png": "image/png"}

_VECTOR_TILE_MEDIA_TYPE = "application/vnd.mapbox-vector-tile"


class PipelineTabularSource:
    """The duck-typed source `api.tabular.stream_or_write_to_file` (called by
    `StreamingThread`'s target) needs: `.tabular_kwargs(format)`,
    `.stream_tabular(out_path, format, **kwargs)`. Backed by a `Pipeline` and
    `execution.execute.run_to_path` rather than a plain `LazyFrame` the way
    `api.sources.LazyFrameSource` is.

    No `write_tabular` method: `PipelineTabularResponse` always streams (`collect_in_memory
    =False`, hardcoded -- see that class's own docstring), so `stream_or_write_to_file`
    never calls the collect-first branch for this source, and there is nothing here to
    satisfy that duck type for.

    `float_precision` is the one `api.sources.LazyFrameSource` behavior this source can
    only take on when a caller hands it the value: that class's `tabular_kwargs` rounds CSV
    output to its single `Variable`'s own `float_precision`, and a pipeline's final frame
    can blend more than one input, so there is no value this could resolve for itself.
    `api.data_query.build_data_query` DOES have one -- a flat data query is a single-input
    pipeline built from one `variable` by construction -- and passes it, which is what
    keeps `/v1/datasets/{variable}/.../data.csv` rounding exactly as it did before that
    route was rebound onto this engine. Left at `None` by the `/v1/analysis...` routes,
    whose pipelines have no single variable to round to: that surface does not round, and
    this default is what keeps it that way rather than having it silently start.

    `relabel` is forwarded to `run_to_path` (see that function's docstring): the raster
    point query's absolute-id -> coordinate-string map, or `None` for every request that
    did not point-sample a raster, which is all of them on the `/v1/analysis...` surface
    (a pipeline spells a spatial filter in `grid_cell` ids directly, so there are no
    caller-supplied coordinate strings to relabel back to).

    `stream_tabular` forwards its `**kwargs` straight into `run_to_path` -- `compression`/
    `compression_level`, computed by `stream_or_write_to_file` from the request's own
    `compression` query parameter, land there as plain values (never the `api.models.
    Compression` object itself), the same way `PipelineRasterResponse` forwards
    `_raster_profile_options`'s plain dict rather than a `Compression` instance:
    `execution.execute.run_to_path` and everything below it live under `analysis/`, which
    may not import from `api/` (see `tests/api/test_analysis_boundary.py`).
    """

    def __init__(self, pipeline, *, base_path=None, relabel=None, float_precision=None):
        self.pipeline = pipeline
        self.base_path = base_path
        self.relabel = relabel
        self.float_precision = float_precision

    def tabular_kwargs(self, format):
        return output.tabular_kwargs(format, self.float_precision)

    def stream_tabular(self, out_path, format, **kwargs):
        run_to_path(self.pipeline, out_path, base_path=self.base_path, relabel=self.relabel, **kwargs)


class PipelineTabularResponse(Response):
    """The tabular (csv/ndjson/parquet/arrow) response for the pipeline surface: reuses
    `api.tabular.StreamingThread` and its temp-file-plus-chunked-read loop verbatim, but is
    a separate class from `api.tabular.PolarsStreamingResponse` rather than that class
    reused outright, for one load-bearing reason.

    `PolarsStreamingResponse` computes the response's status code as `getattr(err,
    "status_code", 500)` once the background thread's target raises -- correct for the OLD
    engine, whose `api.sources.LazyFrameSource.materialize` catches `ColumnNotFoundError`/
    `FileNotFoundError` INSIDE the thread and re-raises them as `HTTPException` (which
    carries `.status_code`), so by the time `StreamingThread.exception` is set, it is
    already the right type for that lookup to work. Nothing in `execution.execute.
    run_to_path`'s call chain does that translation -- `request.errors.RequestError`/
    `request.expressions.ExpressionError`/`polars.exceptions.ColumnNotFoundError` all reach
    `StreamingThread.exception` as themselves, with no `.status_code` attribute, so
    `getattr(err, "status_code", 500)` would wrongly answer 500 for every one of them
    (concretely: an `output.sort`/`output.select` column that passed `request.pipeline`'s
    syntax check but does not exist on the final frame -- the deliberate other half of that
    split, see `execution.sink.apply_output_shaping` -- must reach a caller as 400, not
    500, and this is the seam where that would otherwise silently fail).

    This class exists to fix exactly that: `api.boundary.resolve_status`, the same lookup
    `install_analysis_boundary`'s registered handlers are built from, is consulted FIRST,
    falling back to `PolarsStreamingResponse`'s own `getattr(..., "status_code", 500)`
    rule only for whatever `resolve_status` does not recognize -- so this route and the
    boundary's own exception handlers can never disagree about what status a given
    exception type means, without hand-maintaining that mapping a second time here.

    Always streams (`collect_in_memory=False`): there is no size estimate for a compiled
    `ExecutionPlan` to compare against `config.STREAM_THRESHOLD` -- estimating a
    pipeline's own result size ahead of running it is not implemented. Streaming is the
    safe default either way (`sink.sink_tabular`'s lazy branch never materializes the
    whole frame), just not the size-based optimization a small result could take.
    """

    background = None
    chunk_size = 64 * 1024

    def __init__(self, source: PipelineTabularSource, format: str, compression) -> None:
        self.source = source
        self.format = format
        self.compression = compression
        self.media_type = _TABULAR_MIME_TYPES[format]
        self.init_headers()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["method"].upper() == "HEAD":
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return

        with tempfile.NamedTemporaryFile() as fp:
            write_thread = StreamingThread(
                self.source, fp.name, self.format, self.compression, collect_in_memory=False
            )
            write_thread.start()
            sent_headers = False
            async with await anyio.open_file(fp.name, mode="rb") as file:
                more_body = True
                while more_body:
                    is_alive = write_thread.is_alive()
                    chunk = await file.read(self.chunk_size)
                    chunk_has_data = len(chunk) > 0
                    more_body = chunk_has_data or is_alive
                    if chunk_has_data or not more_body:
                        if not sent_headers:
                            sent_headers = True
                            err = write_thread.exception
                            status = 200
                            if err is not None:
                                status = resolve_status(err)
                                if status is None:
                                    status = getattr(err, "status_code", 500)
                            await send(
                                {
                                    "type": "http.response.start",
                                    "status": status,
                                    "headers": self.raw_headers,
                                }
                            )
                            if err:
                                chunk = resolve_detail(err).encode("utf-8")
                                more_body = False
                        await send(
                            {
                                "type": "http.response.body",
                                "body": chunk,
                                "more_body": more_body,
                            }
                        )


def _raster_profile_options(compression) -> dict:
    """`compression`'s algorithm/level, in `rasterio`'s write-profile spelling."""
    profile_options = {"compress": compression.compression_algorithm.value.upper()}
    level_option = COMPRESSION_LEVEL_OPTIONS.get(compression.compression_algorithm)
    if level_option is not None:
        profile_options[level_option] = compression.compression_level
    return profile_options


class PipelineRasterResponse(FileResponse):
    """The `tif` response: `run_to_path` off the event loop (`anyio.to_thread.run_sync`),
    then `FileResponse` on success.

    The temp file is opened inside `__call__` rather than in the route function, so that it
    is still open when `FileResponse` streams it -- a `with` block in the route would have
    closed it the moment the route returned, before a byte was sent.
    """

    background = None
    chunk_size = 64 * 1024
    media_type = "image/tiff; application=geotiff"

    def __init__(self, pipeline, compression, *, base_path=None) -> None:
        self.pipeline = pipeline
        self.compression = compression
        self.base_path = base_path
        self.init_headers()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        with tempfile.NamedTemporaryFile() as fp:
            profile_options = _raster_profile_options(self.compression)
            ok = await anyio.to_thread.run_sync(
                partial(run_to_path, self.pipeline, fp.name, base_path=self.base_path, **profile_options)
            )
            if ok:
                self.path = fp.name
                self.stat_result = None
                self.status_code = 200
                await super().__call__(scope, receive, send)
            else:
                raise HTTPException(status_code=404, detail="source data do not exist")


async def respond_to_analysis(
    pipeline, format: str, compression, *, base_path=None, relabel=None, float_precision=None
) -> Response:
    """Build the response for `POST|GET /v1/analysis.{format}` -- and, since it was rebound
    onto this engine, for `GET /v1/datasets/{variable}/{time_unit}/{grid}/data.{format}`
    too (`api.__init__.get_model_data`, via `api.data_query.build_data_query`): `tif` gets
    `PipelineRasterResponse`, every other (tabular) format gets `PipelineTabularResponse`.
    Called only after the caller has already run its own parse/`validate_shapes`/
    format-agreement/`validate_compression` checks -- this function does none of them
    itself.

    `relabel`/`float_precision` are the two flat-data-route-only extras, both forwarded to
    `PipelineTabularSource` and both meaningless for a `tif` (a raster point query's long
    result can never reach the raster branch -- `request.ops.validate_shapes` rejects
    `output.format="tif"` on a long final frame before this function is called -- and a
    GeoTIFF has no text floats to round). See that class's docstring for each.
    """
    if format == "tif":
        return PipelineRasterResponse(pipeline, compression, base_path=base_path)
    source = PipelineTabularSource(
        pipeline, base_path=base_path, relabel=relabel, float_precision=float_precision
    )
    return PipelineTabularResponse(source, format, compression)


def _first_input_variable_legend(pipeline):
    """`render_tile`'s `variable_legend` argument: the declared `Variable.legend` of
    `pipeline`'s FIRST input (dict order, i.e. declaration order in `inputs`), or `None`
    if that variable declares none. A pipeline may join several inputs across several
    variables; `render_tile` (`io.output.resolve_legend`) only ever wants ONE fallback
    legend to use when the request's own `color_map` is absent, and there is no principled
    way to pick among several inputs' legends beyond "the one the caller listed first" --
    documented here, not derived, because a different rule (the LAST input, the input that
    ends up as the sink's grid, ...) would be equally defensible and this is simply the one
    chosen. An explicit `color_map` on the request overrides this value entirely
    (`io.output.resolve_legend`'s own precedence: a named/custom color map beats the
    variable's own legend), so this choice only matters when the caller supplied none.
    """
    first_input = next(iter(pipeline.inputs.values()))
    variable = VARIABLES.get(first_input.variable)
    return variable.legend if variable is not None else None


async def respond_to_analysis_tile(pipeline, x: int, y: int, z: int, format: str, *, base_path=None) -> Response:
    """Build the response for `GET /v1/analysis/{z}/{x}/{y}.{format}` (`png`/`jpg`).

    Compiles the plan for this one tile (`compile_plan(pipeline, tile=...)`), executes it,
    and renders it (`sink.render_tile`) -- all off the event loop in one `anyio.to_thread.
    run_sync` call, since every step does real file I/O. A `None` render (nothing to show:
    see `render_tile`'s own docstring for the two cases this covers) is a 404 whose BODY is
    `sink.empty_tile(format, tile_size)` -- a decodable, correctly-sized, transparent image
    of the requested format -- returned directly as this function's `Response`, never as a
    raised `HTTPException`: raising one here would replace that body with a JSON error
    document and the image `Content-Type` with `application/json`, exactly the anti-pattern
    `execution.sink.empty_tile`'s own docstring warns against.
    """
    tile = TileRequest(x=x, y=y, z=z)

    def _render() -> bytes | None:
        plan = compile_plan(pipeline, base_path=base_path, tile=tile)
        band_wide = execute(plan, base_path=base_path)
        variable_legend = _first_input_variable_legend(pipeline)
        return sink.render_tile(
            band_wide, tile, format, color_map=pipeline.output.color_map, variable_legend=variable_legend
        )

    body = await anyio.to_thread.run_sync(_render)
    media_type = _TILE_MIME_TYPES[format]
    if body is None:
        return Response(content=sink.empty_tile(format, tile.tile_size), status_code=404, media_type=media_type)
    return Response(content=body, status_code=200, media_type=media_type)


def _final_frame_name(plan) -> str:
    """The name (an input's own key, or a step's `as`/synthesized name) the plan's final
    frame is bound to -- the same "last op's output, else the lone input" rule `execution.
    execute.execute` itself uses (see that function's docstring), needed here only to look
    the final frame's `FrameType` up in `plan.frame_types`.
    """
    if plan.ops:
        return plan.ops[-1].output
    (name,) = plan.inputs.keys()
    return name


def _require_vector_final_frame(plan) -> str:
    """The final frame's grid name, once confirmed to be a long frame addressed against a
    VECTOR grid -- what a `.pbf` vector tile needs, and the one check `request.ops.
    validate_shapes` structurally cannot perform itself: `pbf` is deliberately not a member
    of `request.types.DataFormat`/`TileFormat` (see `request.pipeline.Output.format`'s
    docstring), so `validate_shapes`'s sink-shape rule -- keyed on `pipeline.output.format`
    -- never sees `pbf` and has nothing to reject a raster-final pipeline against on this
    route's behalf. So this check is performed at the route level instead, against
    `ExecutionPlan.frame_types`.

    Raises `ShapeError` (-> 400 via `api.boundary`) rather than deferring to `sink.
    sink_vector_tile`'s own `None` return: that function cannot tell "this grid genuinely
    has no PMTiles archive" apart from "this pipeline's final grid was never going to have
    one, because it addresses raster pixels, not vector polygons" -- both would otherwise
    read as the same 204 "nothing to render", collapsing a client error into a case meant
    for "no data at this location."
    """
    frame_type = plan.frame_types[_final_frame_name(plan)]
    if frame_type.shape != Shape.long or GRIDS[frame_type.grid].grid_type != "vector":
        raise ShapeError(
            f"pipeline's final frame (grid={frame_type.grid!r}, shape={frame_type.shape.value!r}) "
            "is not a long frame on a vector grid -- a .pbf vector tile needs the final "
            "result addressed against a vector grid's polygons, not a raster's pixels"
        )
    return frame_type.grid


async def respond_to_analysis_vector_tile(pipeline, x: int, y: int, z: int, *, base_path=None) -> Response:
    """Build the response for `GET /v1/analysis/{z}/{x}/{y}.pbf`.

    Unlike the `png`/`jpg` tile route, this compiles `pipeline` WITHOUT a `TileRequest`:
    nothing about a vector tile needs a raster read window narrowed to one tile's bounds --
    the tile's own `x`/`y`/`z` only select which PMTiles boundary layer entry `sink.
    sink_vector_tile` reads, after the ordinary (full-extent) op walk has already produced
    a long frame keyed by `grid_cell`. `pipeline.output`'s `sort`/`select`/`format` are
    never applied here either (see `sink.sink_vector_tile`'s own docstring on this point,
    and `api.__init__`'s existing vector-tile route docstring, which makes the same point
    about the OLD engine's equivalent route) -- `apply_output_shaping` is a tabular-sink
    concern, and a `select` that dropped `grid_cell`/`value` would break the injection this
    function calls into.

    A `None` result (see `sink.sink_vector_tile`'s docstring for the several cases that
    collapses) is a bodiless 204, not a 404: an MVT client already treats a 204 as an empty
    tile, so there is nothing to encode as a body the way an image tile needs one.
    """

    def _render() -> bytes | None:
        plan = compile_plan(pipeline, base_path=base_path)
        grid_name = _require_vector_final_frame(plan)
        result = execute(plan, base_path=base_path)
        return sink.sink_vector_tile(result, grid_name, (x, y, z))

    body = await anyio.to_thread.run_sync(_render)
    if body is None:
        return Response(status_code=204)
    return Response(content=body, status_code=200, media_type=_VECTOR_TILE_MEDIA_TYPE)
