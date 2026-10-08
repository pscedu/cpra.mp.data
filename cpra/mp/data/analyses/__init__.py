"""Named-analysis registry: mirrors `cpra.mp.data.visualizations`'s `VISUALIZATIONS`
pattern (`ANALYSES = {cls.Meta.name: cls for cls in [...]}`, though each entry's `Meta` is
an `AnalysisMeta` INSTANCE here rather than the bare `class Meta:` that module still uses)
for the small set of
multi-step, non-visual analyses a route can resolve by NAME rather than requiring a caller
to hand-author the equivalent `analysis.request.pipeline.Pipeline` JSON directly --
replacing the portal's own client-side-only `subtract-asc`/`subtract-desc`/`concatenate`
derivation logic and its un-enforced cardinality rules. The registry-drift test
(`tests/test_analyses.py`) enforces the constraint that keeps this honest: every entry's
`pipeline()` must return a real `Pipeline`, run through the exact same
`execution.plan.compile_plan` any hand-authored pipeline does -- nothing is reachable only
through a named analysis.

`cpra/mp/data/api/utils.py` re-exports this package's names for the API layer, and does no
resolving of its own: there is no `get_analysis_class` pre-flight gate any more, so an
unknown name, a dataset mismatch and a bad parameter all surface as the three
`AnalysisRegistryError` subclasses `resolve_analysis` raises below, mapped to their 404/400
by one handler in `api/__init__.py`.
"""

from pydantic import ValidationError

from cpra.mp.data.analyses.base import (
    Analysis,
    AnalysisMeta,
    AnalysisChainError,
    AnalysisDatasetMismatchError,
    AnalysisRegistryError,
    DatasetRef,
    InvalidAnalysisParametersError,
    ResolvedAnalysis,
    UnknownAnalysisError,
    default_collapsed_dimensions,
    is_filterable_dimension,
    pipeline_input_dimensions,
)
from cpra.mp.data.analyses.diff import Difference
from cpra.mp.data.analyses.lnd_chng import LandChange
from cpra.mp.data.analyses.lnd_conf import LandConfidence
from cpra.mp.data.analyses.lnd_type_plus import LndTypePlus
from cpra.mp.data.analyses.sal_veg_investigation import SalinityVegInvestigation
from cpra.mp.data.analyses.sediment_dynamics import SedimentDynamics
from cpra.mp.data.analysis.request.errors import RequestError
from cpra.mp.data.analysis.request.pipeline import append_steps

# Landed in this order (`sal_veg_investigation` first -- the only entry exercising
# the fixed-input path, multiple named inputs, and a `project` + spatial `join`, and
# the only one with no existing consumer waiting on it), but `ANALYSES` itself is a `name
# -> class` dict, so registration order has no behavioral effect.
_REGISTERED = [
    SalinityVegInvestigation,
    Difference,
    LandChange,
    LandConfidence,
    LndTypePlus,
    SedimentDynamics,
]


def _register(classes: list[type[Analysis]]) -> dict[str, type[Analysis]]:
    """Build the `name -> class` registry, rejecting an entry whose `Meta` is not an
    `AnalysisMeta` instance or whose `Parameters` names a dimension.

    The `Meta` check is what makes `AnalysisMeta`'s validation binding. A subclass that
    keeps the old bare `class Meta:` would otherwise register fine -- `cls.Meta.name` reads
    the same off either -- and would then be missing exactly the guarantees the model exists
    to give: no forgotten `kind`, no misspelled `result_kind`, no field this model has since
    dropped.

    The `Parameters` check is what makes the decoder's partition unambiguous. A request's
    parameters are split into dimension FILTERS and the analysis's own PARAMETERS by name
    (`request.unified.partition_query`), and the rule it applies is the simplest one there
    is: a dimension name is always a filter. That rule is only sound while no analysis
    declares a dimension-named parameter -- `lnd_type_plus` and `sal_veg_investigation` both
    did, back when a fixed-input analysis had no route that could carry a filter -- and a
    reintroduction would be the highest-severity failure the design has AND invisible to
    every response-level test: the caller's `model_group_id` lands in `filters`,
    `Parameters` takes its default, and the request 200s for the wrong model group. So the
    possibility is removed rather than policed.

    Both raise at the import of this module, which is the import every consumer of
    `ANALYSES` goes through.
    """
    for cls in classes:
        meta = cls.__dict__.get("Meta")
        if not isinstance(meta, AnalysisMeta):
            raise AnalysisRegistryError(
                f"{cls.__name__} must declare `Meta = AnalysisMeta(...)`; got "
                f"{type(meta).__name__ if meta is not None else 'no Meta of its own'}"
            )
        dimension_named = sorted(
            field for field in cls.Parameters.model_fields if is_filterable_dimension(field)
        )
        if dimension_named:
            raise AnalysisRegistryError(
                f"{cls.__name__}.Parameters declares dimension-named field(s) "
                f"{dimension_named} -- a dimension name is always a FILTER, never an "
                f"analysis parameter, so declare it nowhere and read it off `self.filters`"
            )
    return {cls.Meta.name: cls for cls in classes}


ANALYSES = _register(_REGISTERED)

__all__ = [
    "ANALYSES",
    "Analysis",
    "AnalysisChainError",
    "AnalysisDatasetMismatchError",
    "AnalysisRegistryError",
    "DatasetRef",
    "InvalidAnalysisParametersError",
    "ResolvedAnalysis",
    "UnknownAnalysisError",
    "resolve_analysis",
    "resolve_analysis_chain",
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
    `Pipeline` plus presentation metadata): look `name` up in `ANALYSES`, validate
    `params` against its `Parameters` model, run its own `validate_selection`
    cardinality/consistency rule against `filters`, then build and return its `pipeline()`.

    `dataset` is `None` when the caller named no `variable`/`time_unit`/`grid`, which is
    what an analysis with fixed inputs requires (they are hardcoded in its own
    `pipeline()` method -- see `sal_veg_investigation.py`); `filters` is the dataset's
    dimension filters plus the two reserved keys `grid_cell` and `where`, which the
    analysis composes into its pipeline's leading `filter` STEP (see
    `base.selection_filter_where`) rather than onto its input selection -- an
    `InputSelection` names a dataset and carries no filtering at all. (`request.sort`/
    `.select`/`.color_map`/compression parameters are NOT part of `filters` -- those apply
    to the RESOLVED `Pipeline`'s `.output`/response at the route layer, after this
    function returns, the same way they would for a hand-authored pipeline.) `params` is
    the analysis's own declared parameters (`dimension`/`order` for `diff`,
    `model_group_id`/`scenario_id`/`marsh_threshold`/`salinity_threshold` for
    `sal_veg_investigation`, none at all for `lnd_conf` or `lnd_chng`,
    whose selections are entirely `filters`), raw and unvalidated until this call.

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
    - `AnalysisDatasetMismatchError` -- `name` resolves to an analysis that needs a
      `dataset` and was given none, or fixes its own inputs and was given one (400);
    - `InvalidAnalysisParametersError` -- `params` failed `Parameters` validation,
      `filters` failed the analysis's `validate_selection` cardinality rule, or the
      resulting `Pipeline` itself failed `parse_pipeline`'s validation (maps to 400).
    """
    filters = dict(filters or {})
    params = dict(params or {})

    analysis_class = ANALYSES.get(name)
    if analysis_class is None:
        raise UnknownAnalysisError(f"unknown analysis name {name!r} -- known: {sorted(ANALYSES)}")

    # No `Meta.type` gate here any more. It compared `dataset is None` against a declared
    # route kind BEFORE the analysis existed; each `pipeline()` now states the same
    # requirement itself (`Analysis.require_dataset`/`reject_dataset`), which is both
    # closer to the code that reads `self.dataset` and one fewer place for the two to
    # disagree. `AnalysisDatasetMismatchError` still names the mistake, and still 400s --
    # it is raised a few lines lower, out of `analysis.pipeline()`.
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


def resolve_analysis_chain(
    names: list[str],
    *,
    dataset: DatasetRef | None = None,
    filters: dict | None = None,
    params: dict | None = None,
    grid_cell_where: str | None = None,
) -> ResolvedAnalysis:
    """Resolve a comma-separated chain of named analyses, e.g.
    `["lnd_conf", "diff"]`: stage 1 resolves exactly as `resolve_analysis`
    would alone; each later stage must be `kind="dimension"` (checked against `Meta.kind`
    before anything else runs) and contributes its own steps (`_dimension_steps`), appended
    to the running pipeline via `analysis.request.pipeline.append_steps` -- the same
    stored-pipeline-plus-extra-steps mechanism a narrowing request already uses, so a chain
    is re-validated as an ordinary `Pipeline` rather than trusted to have skipped a check a
    standalone one would have been subject to.

    `len(names) == 1` delegates straight to `resolve_analysis` -- every existing
    single-name call site (`pipeline.json`'s resubmission contract, every pre-chain test)
    behaves identically.

    A later stage's `dimension` parameter must name a dimension the chain has not already
    collapsed. "Already collapsed" is tracked in a running `collapsed` set seeded from
    stage 1's own AUTHORITATIVE `resolved.collapsed_dimensions` -- NOT recomputed
    structurally from the intermediate pipeline via `default_collapsed_dimensions` at each
    step, because that would be wrong for a chain led by an analysis (e.g. `lnd_type_plus`)
    whose `collapsed_dimensions` is a hand-written override rather than the structural
    default: `default_collapsed_dimensions` reports `frozenset()` whenever the pipeline's
    last step isn't a `group`, which is exactly `lnd_type_plus`'s own case (it ends in a
    `case`), so recomputing structurally after stage 1 would silently forget that
    `aggregate_type` is already gone. The FINAL published `collapsed_dimensions` (below)
    doesn't have this problem and is recomputed structurally on purpose: the combined
    pipeline by then always ends in a real trailing `group` (every `_dimension_steps`
    appends one), so the union of everything every stage subsumed falls out on its own.

    The published `legend` is resolved per stage, most specific first: the stage's own
    `Meta.legend`, then -- for a stage that produces a DIFFERENCE -- the LEADING analysis's
    `Meta.difference_legend`, then whatever the chain already had. That middle rung is what
    lets `lnd_conf,diff` come back as `lnd_conf_diff`: `diff` cannot name a legend, since it
    knows nothing about what it was chained onto, and `lnd_conf`'s own `legend` classifies
    an undifferenced fraction on `[0, 1]` and would be the wrong key for a signed change in
    it. See `analyses/lnd_conf.py`, and `visualizations.map._uses_difference_ramp`, which is
    where this decides whether the map draws a named legend or the synthesized diverging
    ramp.

    Every stage's `Parameters` fields are checked pairwise for a name collision before any
    parameter is partitioned out of `params` -- today's registry has exactly one
    `kind="dimension"` analysis (`diff`), so this never actually fires yet, but it
    turns "two stages silently overwrite each other's parameter" into an explicit 400
    naming the field, for the day a second one is added or the same one is named twice.

    Raises the same three `AnalysisRegistryError` subclasses `resolve_analysis` does (see
    its docstring), plus `AnalysisChainError` -- an unknown name anywhere in `names`, a
    wrong-`kind` analysis in a chain position, a parameter collision, an
    unrecognized parameter, a `dimension` that names an already-collapsed dimension, or a
    `Pipeline`-construction failure at any stage.
    """
    if len(names) == 1:
        return resolve_analysis(
            names[0], dataset=dataset, filters=filters, params=params, grid_cell_where=grid_cell_where
        )

    classes = []
    for name in names:
        cls = ANALYSES.get(name)
        if cls is None:
            raise UnknownAnalysisError(f"unknown analysis name {name!r} -- known: {sorted(ANALYSES)}")
        classes.append(cls)
    for cls in classes[1:]:
        if cls.Meta.kind != "dimension":
            raise AnalysisChainError(
                f"analysis {cls.Meta.name!r} is kind={cls.Meta.kind!r}, but only a "
                "kind='dimension' analysis may follow the first stage of a chain"
            )

    # Pairwise collision across every stage's Parameters fields, not just stage 1 vs. 2 --
    # a 3+-stage chain can collide on stage 2 vs. stage 3 just as easily.
    field_sets = [set(cls.Parameters.model_fields) for cls in classes]
    for i in range(len(field_sets)):
        for j in range(i + 1, len(field_sets)):
            overlap = field_sets[i] & field_sets[j]
            if overlap:
                raise InvalidAnalysisParametersError(
                    f"{names[i]!r} and {names[j]!r} both declare parameter(s) "
                    f"{sorted(overlap)} -- this chain is ambiguous and is not supported"
                )

    filters = dict(filters or {})
    params = dict(params or {})
    partitioned = [{k: v for k, v in params.items() if k in fields} for fields in field_sets]
    claimed = {k for stage_params in partitioned for k in stage_params}
    unknown = set(params) - claimed
    if unknown:
        raise InvalidAnalysisParametersError(f"unrecognized parameter(s): {sorted(unknown)}")

    resolved = resolve_analysis(
        names[0], dataset=dataset, filters=filters, params=partitioned[0], grid_cell_where=grid_cell_where
    )
    pipeline = resolved.pipeline
    title_suffix = resolved.title_suffix
    legend = resolved.legend
    result_kind = resolved.result_kind
    collapsed = set(resolved.collapsed_dimensions)
    all_dims = pipeline_input_dimensions(pipeline)

    for cls, own_params in zip(classes[1:], partitioned[1:]):
        try:
            parameters = cls.Parameters.model_validate(own_params)
        except ValidationError as exc:
            raise InvalidAnalysisParametersError(f"analysis {cls.Meta.name!r} parameters: {exc}") from exc
        cls.validate_selection(parameters, filters)

        remaining = [d for d in all_dims if d not in collapsed]
        dim = parameters.dimension
        if dim not in remaining:
            raise InvalidAnalysisParametersError(
                f"{cls.Meta.name!r} cannot operate over {dim!r}: it is already subsumed "
                f"upstream (remaining dimensions: {remaining})"
            )
        group_by = [d for d in remaining if d != dim]

        stage = cls(dataset=None, filters=filters, parameters=parameters, grid_cell_where=None)
        appended_steps = stage._dimension_steps(group_by)
        try:
            pipeline = append_steps(pipeline, appended_steps, pipeline.output)
        except RequestError as exc:
            raise InvalidAnalysisParametersError(
                f"chained analysis could not build a pipeline at stage {cls.Meta.name!r}: {exc}"
            ) from exc
        collapsed.add(dim)
        title_suffix += stage.title_suffix
        # Three-way, most specific first. The STAGE's own `legend` wins where it declares
        # one, because a stage that names a legend is naming the reading of its own output
        # whatever it was given (none does today -- `diff` declares `None`). Failing that,
        # a stage that produces a DIFFERENCE hands over to the leading analysis's
        # `difference_legend`, which is where an analysis says how its own output reads once
        # differenced: `lnd_conf` classifies a fraction on `[0, 1]` and `lnd_conf_diff` a
        # signed change in that fraction, and only the analysis itself can pair the two.
        # Failing both, the running legend carries through unchanged, as it always did.
        #
        # `resolved.meta` is stage 1's, not the running one, on purpose: `difference_legend`
        # is a fact about what THIS analysis's values mean, and the leading stage is the
        # only one that produced values of its own -- every later stage rewrites them.
        legend = (
            stage.legend
            or (resolved.meta.difference_legend if stage.result_kind == "difference" else None)
            or legend
        )
        result_kind = stage.result_kind

    return ResolvedAnalysis(
        pipeline=pipeline,
        meta=resolved.meta,
        legend=legend,
        result_kind=result_kind,
        collapsed_dimensions=default_collapsed_dimensions(pipeline),
        title_suffix=title_suffix,
    )
