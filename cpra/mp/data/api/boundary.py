"""The one translation point between the NEW analysis engine's transport-agnostic
exception families (`request.errors.RequestError` and its members, `request.expressions.
ExpressionError`, `execution.execute.LoweringNotImplementedError`, plus Polars' own
`ColumnNotFoundError` and the builtin `FileNotFoundError`) and real HTTP responses, for
the `/v1/analysis...` pipeline surface (`api/pipeline.py`) and the four routes in
`api/__init__.py` that use it.

This is the NEW engine's counterpart to `api/__init__.py`'s three existing handlers
(`analysis_error_handler`, `analysis_registry_error_handler`, `pipeline_error_handler`),
which keep translating the OLD, unrelated `analysis.base.AnalysisError`/`analysis.pipeline.
PipelineError`/`analyses.base.AnalysisRegistryError` families exactly as they already do --
see `request.errors`'s module docstring for why the two exception worlds are independent
by design, not by oversight. This module never touches those three.

**One handler per exception type, not one dispatcher.** `RequestError` is registered on
its own: FastAPI/Starlette's `add_exception_handler` looks up the handler for a raised
exception by walking its MRO and using the most specific type that has one registered
(confirmed empirically -- see `tests/api/test_analysis_boundary_module.py`'s MRO test),
so registering the BASE class once is enough to cover every member
(`ShapeError`/`AggregateError`/`NotFoundError`/`BandLimitError`, and any future member)
without a handler per subclass -- provided no subclass ever needs a DIFFERENT status code
than the family default, in which case that subclass gets its own, more specific,
registration instead (none currently does).

**Response body.** Every handler here returns `{"detail": <text>}`, matching
`api/__init__.py`'s `analysis_error_handler` shape exactly, so a caller sees no difference
in envelope between the old and new exception worlds -- only the status code and the
message differ per exception type. The text is `resolve_detail`'s, which is `str(exc)` for
everything except the few types `_DETAIL_BY_TYPE` overrides because their message is not
ours to disclose.

**Status code mapping**, each chosen by what the exception's own docstring says it means:

- `request.errors.NotFoundError` -> 404 (a lookup that resolved to nothing).
- `request.errors.BandLimitError` -> 400 (a well-formed request whose resolved band count
  exceeds the configured ceiling -- the caller's request is what needs narrowing).
- `request.errors.RequestError` (the base, covering every other member by MRO) -> 400.
- `request.expressions.ExpressionError` -> 400. A separate `ValueError` subclass, NOT a
  `RequestError` member (see that class's own docstring) -- registered here in its own
  right rather than assumed covered by the `RequestError` registration, which would not
  reach it.
- `execution.execute.LoweringNotImplementedError` -> 501. Deliberately not a
  `RequestError` member (see its own docstring): the pipeline is well-formed, this
  engine's executor coverage is what is incomplete, so a well-formed-but-unimplemented
  request is a server capability gap, not a bad request.
- `polars.exceptions.ColumnNotFoundError` -> 400. Raised when `output.sort`/`output.
  select` names a column that passed `request.pipeline`'s SYNTAX check
  (`_validate_output_sort_or_select`) but does not exist on the final frame -- the other
  half of that deliberate split; see `execution.sink.apply_output_shaping`'s docstring.
- `FileNotFoundError` -> 404. Raised by Polars/rasterio's own readers when a real
  variable/grid/time_unit combo simply has no file on disk yet -- the NEW engine's
  counterpart to the OLD engine's `BaseAnalysis.materialize`/`api.sources.LazyFrameSource
  .materialize`, both of which catch this and re-raise as a 404 for the same reason (see
  `execution.sink`'s `_materialize_closure` docstring, which explicitly leaves this
  translation to "the error boundary a route installs" -- this entry is that boundary).
  Not a member of any `request.errors` family: it is raised by the underlying I/O layer
  (`analysis.io.vector.scan_vector` -> `tabular.read_parquet` -> `pl.scan_parquet(...)
  .collect()`), not by anything under `request/`, so it is registered here directly
  rather than wrapped in a `RequestError` subclass. Its body text is overridden
  (`_DETAIL_BY_TYPE`) rather than passed through: the reader's own message names the
  absolute path it could not open. Note this entry is app-wide, so a missing file that is
  a server MISCONFIGURATION (an absent crosswalk or grid file) also answers 404 rather
  than 500 -- the same conflation the old engine's two readers make, kept deliberately
  because narrowing it means catching at each read site, of which the new engine has
  several (tabular sink, vector tile, raster scan).

**`resolve_status`, and why it exists alongside the registered handlers rather than being
replaced by them.** Both `install_analysis_boundary`'s handlers AND `resolve_status` below
are built from the same `_STATUS_BY_TYPE` table -- one mapping, maintained once -- but they
serve two different call paths. The registered handlers fire when an exception propagates
through FastAPI/Starlette's own ASGI exception dispatch, which is true for everything on
the `png`/`jpg`/`pbf` tile routes and the `tif` route (all driven through `anyio.to_thread.
run_sync`, whose exceptions re-raise normally in the awaiting coroutine). It is NOT true for
the tabular route: `api.pipeline.PipelineTabularResponse` runs the actual sink work inside
`api.tabular.StreamingThread`, a plain background `Thread` whose raised exception is caught
and stored (`StreamingThread.exception`), never propagated -- so FastAPI's dispatch, and
therefore these registered handlers, never see it. `resolve_status` lets that response
class ask the same "what status code does this exception type mean" question directly,
without hand-maintaining a second copy of the mapping.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from polars.exceptions import ColumnNotFoundError

from cpra.mp.data.analysis.execution.execute import LoweringNotImplementedError
from cpra.mp.data.analysis.request.errors import BandLimitError, NotFoundError, RequestError
from cpra.mp.data.analysis.request.expressions import ExpressionError

# One mapping, used two ways below: to register `app.exception_handler`s (for exceptions
# that reach FastAPI/Starlette's own ASGI dispatch normally) and, via `resolve_status`, to
# answer the same question directly for `api.pipeline.PipelineTabularResponse`'s
# background-thread-swallowed exceptions, which never reach that dispatch at all -- see
# this module's docstring for why both paths are needed. Order matters: it is the MRO
# fallback order `resolve_status` walks, most specific first, so a subclass with its own
# entry (e.g. `NotFoundError`/`BandLimitError` before the `RequestError` base they both
# derive from) is found before the family default it would otherwise fall back to.
_STATUS_BY_TYPE: dict[type[Exception], int] = {
    NotFoundError: 404,
    BandLimitError: 400,
    RequestError: 400,
    ExpressionError: 400,
    LoweringNotImplementedError: 501,
    ColumnNotFoundError: 400,
    FileNotFoundError: 404,
}


def resolve_status(exc: Exception) -> int | None:
    """The HTTP status `_STATUS_BY_TYPE` maps `type(exc)` (or its closest registered
    ancestor) to, walking `type(exc).__mro__` the same way FastAPI/Starlette's own
    `add_exception_handler` lookup does -- so this function and the handlers `
    install_analysis_boundary` registers from the same table can never disagree about a
    given exception. Returns `None` for a type this module has no mapping for at all
    (`_STATUS_BY_TYPE` does not have `object`/`Exception` as a catch-all entry on purpose:
    callers of this function -- currently only `api.pipeline.PipelineTabularResponse` --
    decide their own fallback for "not a type this module recognizes," rather than this
    function silently picking one on their behalf).
    """
    for klass in type(exc).__mro__:
        status = _STATUS_BY_TYPE.get(klass)
        if status is not None:
            return status
    return None


# The types whose own message must NOT reach the caller, and what to say instead.
# Everything absent from here answers with `str(exc)`, which is this package's own message
# text. `FileNotFoundError`'s is the reader's, and it carries the absolute server path of
# the file that was missing ("No such file or directory (os error 2): /ocean/projects/
# .../variable=marsh/....parquet") -- a filesystem layout the caller has no business
# seeing. The old engine's two readers (`api.sources.LazyFrameSource.materialize` and
# `analysis.base.BaseAnalysis.materialize`) both answer this exact case with a fixed
# string for that reason, while passing `ColumnNotFoundError`'s message straight through;
# that split is deliberate and is reproduced here so moving a route onto the new engine
# cannot start disclosing paths the old one withheld.
_DETAIL_BY_TYPE: dict[type[Exception], str] = {
    FileNotFoundError: "Dataset does not exist",
}


def resolve_detail(exc: Exception) -> str:
    """The response body text for `exc`: the fixed string for a type `_DETAIL_BY_TYPE`
    overrides, else the exception's own `detail` when it carries one (an `HTTPException`
    raised before this boundary was reached), else `str(exc)`.

    Walks `type(exc).__mro__` exactly as `resolve_status` does, so a SUBCLASS of an
    overridden type cannot route around the override and leak the message its parent was
    listed to suppress.
    """
    for klass in type(exc).__mro__:
        detail = _DETAIL_BY_TYPE.get(klass)
        if detail is not None:
            return detail
    return getattr(exc, "detail", str(exc))


def install_analysis_boundary(app: FastAPI) -> None:
    """Register one exception handler per `_STATUS_BY_TYPE` entry onto `app`. Called
    exactly once, from `api/__init__.py`, alongside (not instead of) the three existing
    `AnalysisError`/`AnalysisRegistryError`/`PipelineError` handlers that keep serving the
    OLD engine's remaining routes.

    Built from `_STATUS_BY_TYPE` in a loop, rather than one `@app.exception_handler(...)`
    per type written out by hand, so that table is the only place the mapping is
    maintained -- `resolve_status` reads the same dict. Registration order does not matter
    to Starlette's own lookup (it walks the raised exception's MRO and picks the most
    specific registered type regardless of the order handlers were added in -- confirmed
    empirically, see `tests/api/test_analysis_boundary_module.py`'s MRO test), so the
    dict's own (insertion) order, chosen for human readability, has no effect here.
    """

    def _make_handler(status: int):
        async def _handler(request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse(status_code=status, content={"detail": resolve_detail(exc)})

        return _handler

    for exc_type, status in _STATUS_BY_TYPE.items():
        app.add_exception_handler(exc_type, _make_handler(status))
