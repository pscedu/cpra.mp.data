"""The request layer's own exception family.

Every function under `request/` raises out of this family -- an ordinary `ValueError`
subclass hierarchy, never `HTTPException`, never a FastAPI import -- so the family stays
usable wherever a `Pipeline` gets parsed or validated, API route or not. A single
boundary -- to be written wherever this package is eventually mounted on a transport, and
deliberately absent here -- maps a caught member of this family onto a transport response,
using the exception's own TYPE to decide how: a caller that already knows a failure's
request-level meaning (a shape mismatch is always a bad request; a lookup with nothing to
find is always "not found") expresses that by raising the matching subclass, rather than
attaching a status code as an afterthought at the point something is raised.

This is a NEW family, unconnected to `cpra.mp.data.analysis.base.AnalysisError` (the
existing flat-API error, an `Exception` subclass carrying `status_code`/`detail`
attributes read directly by that API's own response machinery). That class keeps serving
`io/geography.py` and `io/output.py` -- both still wired into the old code path -- exactly
as before; nothing here re-points them. The two families are independent by design, not
by oversight: unifying them is a wiring decision for whenever a route is actually rebound
to this module, not a fact about the exceptions themselves.
"""


class RequestError(ValueError):
    """The base of this module's exception family: a request failed a plan-time rule --
    a malformed pipeline shape, an unresolvable step reference, an illegal op/shape
    combination, a non-decomposable aggregate, or any other structural or plan-time
    check `request/` performs. An ordinary `ValueError` subclass, in the family of
    `expressions.ExpressionError` -- catchable on its own, or as the one type every more
    specific member below is also an instance of.
    """


class ShapeError(RequestError):
    """A pipeline step is well-formed by itself, but its declared shape does not fit
    where it sits in the chain: a `project` with no crosswalk (raster to raster), a
    `join` mixing a long frame with a band-wide one, a band-wide `join` whose operands
    sit on two different grids, a long final frame under an output format that needs a
    raster surface, or a final frame carrying more than one value column. Every one of
    these is decidable from the declared ops and each input's grid kind alone, with no
    file opened and no crosswalk read -- see `request/ops.py`'s shape-typing walk, which
    is what raises this.
    """


class AggregateError(RequestError):
    """An aggregate can't do what a step asked of it: `count_by_value` where the step
    doesn't allow it, an aggregate a fold can't decompose where one is required, a
    project with no aggregate where one is required (or one supplied where none is
    allowed), or `value` used as a grouping key. Split out from `ShapeError`/
    `RequestError` so a caller that only cares about aggregate-vocabulary problems can
    catch just this.
    """


class NotFoundError(RequestError):
    """A request named something that turned out not to exist -- a crosswalk between two
    grids with no artifact built for that pair, a named analysis that isn't registered,
    and similar lookups performed in `io/catalog.py`. Nothing in `request/` itself
    raises this today, since nothing here touches a crosswalk, a catalog, or a file; it
    is declared alongside the rest of the family now because the family is one thing,
    and a caller that wants to catch "any request-layer failure" should be able to catch
    `RequestError` once and mean it, even before every member has a raiser.
    """


class BandLimitError(RequestError):
    """A raster input resolves to more bands than `cpra.mp.data.config.
    ANALYSIS_BAND_CEILING` allows, once compile-time band selection has narrowed it as
    far as the declared pipeline lets it -- raised by `execution.plan.compile_plan`. For
    a single raster input this is its own resolved (and possibly dimension-filter-
    narrowed) band count; for a band-wide `join`, both operands are read through one
    shared VRT, so this is raised against the UNION of the operands' band counts instead
    of either one alone (see `ANALYSIS_BAND_CEILING`'s own comment in `config.py`). Always
    decidable from the catalog's band listing alone -- no pixel is read to raise this.
    """
