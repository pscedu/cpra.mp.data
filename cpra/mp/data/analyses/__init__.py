"""Named-analysis registry: mirrors `cpra.mp.data.visualizations`'s `VISUALIZATIONS`
pattern (`ANALYSES = {cls.Meta.name: cls for cls in [...]}`) for the small set of
multi-step, non-visual analyses a route can resolve by NAME rather than requiring a caller
to hand-author the equivalent `analysis.request.pipeline.Pipeline` JSON directly --
replacing the portal's own client-side-only `subtract-asc`/`subtract-desc`/`concatenate`
derivation logic and its un-enforced cardinality rules. The registry-drift test
(`tests/test_analyses.py`) enforces the constraint that keeps this honest: every entry's
`pipeline()` must return a real `Pipeline`, run through the exact same
`execution.plan.compile_plan` any hand-authored pipeline does -- nothing is reachable only
through a named analysis.

`ANALYSES`, `DatasetRef`, and `resolve_analysis` are the exact three names
`cpra/mp/data/api/utils.py` imports -- see that module's `get_analysis_class`, which
resolves `ANALYSES[name].Meta.type` directly for its own 404/400 (unknown name / wrong
route kind) translation, independent of the exceptions `resolve_analysis` itself raises
below for the parameter-validation/pipeline-construction half of the same job.
"""

from pydantic import ValidationError

from cpra.mp.data.analyses.base import (
    Analysis,
    AnalysisRegistryError,
    AnalysisTypeMismatchError,
    DatasetRef,
    InvalidAnalysisParametersError,
    ResolvedAnalysis,
    UnknownAnalysisError,
)
from cpra.mp.data.analyses.concatenate import Concatenate
from cpra.mp.data.analyses.difference import Difference
from cpra.mp.data.analyses.land_type_confidence import LandTypeConfidence
from cpra.mp.data.analyses.lnd_type_plus import LndTypePlus
from cpra.mp.data.analyses.salinity_veg_investigation import SalinityVegInvestigation
from cpra.mp.data.analysis.request.errors import RequestError

# Landed in this order (`salinity-veg-investigation` first -- the only entry exercising
# the `type="custom"` path, multiple named inputs, and a `project` + spatial `join`, and
# the only one with no existing consumer waiting on it), but `ANALYSES` itself is a `name
# -> class` dict, so registration order has no behavioral effect.
ANALYSES = {
    cls.Meta.name: cls
    for cls in [SalinityVegInvestigation, Difference, Concatenate, LandTypeConfidence, LndTypePlus]
}

__all__ = [
    "ANALYSES",
    "Analysis",
    "AnalysisRegistryError",
    "AnalysisTypeMismatchError",
    "DatasetRef",
    "InvalidAnalysisParametersError",
    "ResolvedAnalysis",
    "UnknownAnalysisError",
    "resolve_analysis",
]


def resolve_analysis(
    name: str,
    *,
    dataset: DatasetRef | None = None,
    filters: dict | None = None,
    params: dict | None = None,
    grid_cell_where: str | None = None,
) -> ResolvedAnalysis:
    """Resolve a named analysis into a `ResolvedAnalysis` (a real, `compile_plan`-ready
    `Pipeline` plus presentation metadata): look `name` up in `ANALYSES`, check its
    `Meta.type` against whether `dataset` was given (a `type="dataset"` analysis requires
    one; a `type="custom"` analysis requires `dataset=None`), validate `params` against
    its `Parameters` model, run its own `validate_selection` cardinality/consistency rule
    against `filters`, then build and return its `pipeline()`.

    `dataset` is `None` for a `type="custom"` analysis (its inputs are hardcoded in its
    own `pipeline()` method -- see `salinity_veg_investigation.py`); `filters` is the dataset's
    dimension filters plus the two reserved keys `grid_cell` and `where`, which the
    analysis composes into its pipeline's leading `filter` STEP (see
    `base.selection_filter_where`) rather than onto its input selection -- an
    `InputSelection` names a dataset and carries no filtering at all. (`request.sort`/
    `.select`/`.color_map`/compression parameters are NOT part of `filters` -- those apply
    to the RESOLVED `Pipeline`'s `.output`/response at the route layer, after this
    function returns, the same way they would for a hand-authored pipeline.) `params` is
    the analysis's own declared parameters (`dimension`/`order` for `difference`,
    `dimension` for `concatenate`, `model_group_id`/`scenario_id`/`marsh_threshold`/
    `salinity_threshold` for `salinity-veg-investigation`, none at all for
    `land-type-confidence`, whose selection is entirely `filters`), raw and unvalidated
    until this call.

    `grid_cell_where` is an already-resolved spatial conjunct (`"grid_cell IN (1, 2)"`)
    for a caller that had to turn coordinates into IDs itself -- resolving coordinates
    needs the catalog, which is `io/`'s job, not this package's. When given it entirely
    replaces `filters["grid_cell"]`'s own digit-ID handling, exactly as the same-named
    argument does for `request.data_query.build_data_query_pipeline`.

    Raises exactly one of three `AnalysisRegistryError` subclasses -- never a bare
    `RequestError`, even though `pipeline()` itself calls `parse_pipeline` and could raise
    one; any `RequestError` a pipeline-construction step (from valid-shaped, but together
    nonsensical, parameters/filters) raises is caught and re-raised as
    `InvalidAnalysisParametersError` below, so a caller only ever needs to catch this
    module's three exception types:

    - `UnknownAnalysisError` -- `name` is not in `ANALYSES` (maps to 404);
    - `AnalysisTypeMismatchError` -- `name` resolves to an analysis whose `Meta.type`
      doesn't match whether `dataset` was given (maps to 400);
    - `InvalidAnalysisParametersError` -- `params` failed `Parameters` validation,
      `filters` failed the analysis's `validate_selection` cardinality rule, or the
      resulting `Pipeline` itself failed `parse_pipeline`'s validation (maps to 400).
    """
    filters = dict(filters or {})
    params = dict(params or {})

    analysis_class = ANALYSES.get(name)
    if analysis_class is None:
        raise UnknownAnalysisError(f"unknown analysis name {name!r} -- known: {sorted(ANALYSES)}")

    expected_type = "custom" if dataset is None else "dataset"
    if analysis_class.Meta.type != expected_type:
        given = "no dataset given" if dataset is None else f"dataset given: {dataset!r}"
        raise AnalysisTypeMismatchError(
            f"analysis {name!r} is type={analysis_class.Meta.type!r}, but was resolved "
            f"as if it were type={expected_type!r} ({given})"
        )

    try:
        parameters = analysis_class.Parameters.model_validate(params)
    except ValidationError as exc:
        raise InvalidAnalysisParametersError(f"analysis {name!r} parameters: {exc}") from exc

    analysis_class.validate_selection(parameters, filters)

    analysis = analysis_class(
        dataset=dataset, filters=filters, parameters=parameters, grid_cell_where=grid_cell_where
    )
    try:
        pipeline = analysis.pipeline()
    except RequestError as exc:
        raise InvalidAnalysisParametersError(
            f"analysis {name!r} could not build a pipeline from the given parameters/"
            f"filters: {exc}"
        ) from exc

    return ResolvedAnalysis(
        pipeline=pipeline,
        meta=analysis_class.Meta,
        legend=analysis.legend,
        result_kind=analysis.result_kind,
        collapsed_dimensions=analysis.collapsed_dimensions,
        title_suffix=analysis.title_suffix,
    )
