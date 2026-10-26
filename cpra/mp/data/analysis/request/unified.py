"""The **unified decoder**: one query string, one `Pipeline`, for every data, tile,
visualization, analysis and catalog route.

Before this, a caller's intent was spread across the URL's *shape*: whether it carried a
`{variable}/{time_unit}/{grid}` prefix decided whether dimension-named parameters were
filters or the analysis's own parameters; whether it carried an `/analyses/{name}` segment
decided which of two resolvers ran; and `analysis=` had two spellings (a path segment and a
query key) that could not both be sent. This module replaces all of that with one
declared partition of the query string, applied identically no matter which route the
request arrived on.

**The five categories** (`partition_query`), in the order they are peeled off:

1. `analysis=` -- a name, or a comma-separated chain. Defaults to `DOWNLOAD`, a decoder
   sentinel meaning "no analysis"; see that constant.
2. **Dotted pipeline parameters** -- `input.*`, `step.*`, `output.*`, in the existing
   dotted codec (`pipeline.pipeline_from_query`). Inputs merge into the analysis's own
   inputs; steps append after the analysis's own steps.
3. **Selection sugar** -- the dataset triple `variable`/`time_unit`/`grid`, every other
   filterable dimension, and `grid_cell`/`where`.
4. **Output** -- `sort`, `select`, `color_map`, plus `compression_algorithm`/
   `compression_level` where the route allows them.
5. **Render options** -- per visualization class, passed in by the caller
   (`render_fields`), because the set varies per class the way an analysis's `Parameters`
   do. `projects_model_group_id` is the reason this has to be a declared category rather
   than an inferred one: it is dimension-NAMED and is emphatically not a dimension filter.

Whatever is left is the named analysis's own declared parameters (`dimension`, `order`,
`marsh_threshold`, ...), handed to `analyses.resolve_analysis_chain`, which rejects a name
no stage declares. Under the `DOWNLOAD` sentinel there is no `Parameters` model to reject
against, so a leftover is a 400 here instead -- which is the one behaviour change a caller
can actually see: today FastAPI silently ignores an unrecognized query parameter, so
`?variable=elev&…&calender_year=2050` answers 200 with the whole unfiltered dataset.

**What this module does NOT do**, and why it is not the whole decoder:

- It never resolves `analysis=`. That needs `cpra.mp.data.analyses`, which imports this
  package -- reaching back the other way would invert the dependency. The orchestrator
  passes the resolved pipeline in as `base`.
- It never resolves a coordinate-valued `grid_cell`, never opens a file, and never knows a
  grid's kind. That needs `io/`, which `request/` may not import -- the same split
  `request.data_query.build_data_query_pipeline`/`api.data_query.build_data_query` already
  has, and for the same reason.
- It never composes the selection `where`. That is `analyses.base.selection_filter_where`,
  which lives with the registry because a dimension literal's rendering is shared with
  every analysis's own `pipeline()`.

`api/unified.py` is the orchestrator that supplies those three and runs the six steps in
order. This module owns the partition and the pipeline ASSEMBLY -- the parts that are pure
query-to-`Pipeline` mechanics.

**Assembly happens at the QUERY level, not the `Pipeline` level** (see `build_pipeline`):
the base pipeline is re-encoded with `pipeline_to_query`, the caller's dotted parameters
are overlaid on top of it, and the result is decoded once by `pipeline_from_query`. Those
two are already proven inverses and already own every merge rule worth having -- field-wise
overlay of a nested path, list-index contiguity, the per-op field whitelist. Merging
`Pipeline` objects directly would mean a second implementation of all of it.
"""

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

from cpra.mp.data.analysis.request.data_query import DEFAULT_INPUT_NAME
from cpra.mp.data.analysis.request.errors import RequestError
from cpra.mp.data.analysis.request.parameters import (
    BaseDatasetDimensions,
    BaseFilterDimensions,
    Compression,
    DataOptions,
)
from cpra.mp.data.analysis.request.pipeline import (
    FilterStep,
    Pipeline,
    append_steps,
    pipeline_from_query,
    pipeline_to_query,
)

# The `analysis=` value meaning "no analysis at all": read the selected dataset, filter it,
# write it. A DECODER SENTINEL, deliberately not a registered `Analysis`.
#
# Registering it was the obvious alternative and is worse in three ways. `Pipeline.inputs`
# is `Field(min_length=1)` and `validate_pipeline` requires "zero steps implies exactly one
# input", so a `download` analysis's `pipeline()` could not return the empty pipeline the
# name describes without weakening the IR globally to serve one entry. `resolve_analysis`
# would still need a special case, since the sentinel has no dataset of its own to check
# `Meta.type` against. And `/v1/analyses.json` would advertise it as a thing to pick, which
# it is not -- it is the absence of a pick.
#
# As a sentinel the empty pipeline is never constructed: `download` plus the dataset triple
# is exactly one input, and `download` with nothing at all is rejected by the orchestrator
# before any `Pipeline` is built.
DOWNLOAD = "download"

# The sink to assume where nothing else states one. Only `build_pipeline(format=None)`
# reads it -- the pipeline-JSON route, whose URL carries no extension -- and only when
# neither the resolved analysis nor the caller declared an `output.format` of their own.
# `Output.format` is a required field, so something has to be chosen; `csv` is the one every
# shape can be asked for, and the one the retired `.../pipeline.json` routes assumed.
DEFAULT_FORMAT = "csv"

# The top-level groups of the dotted codec (`pipeline.pipeline_from_query`). A parameter
# whose first dotted segment is one of these is category 2 and is passed through untouched;
# the decoder itself rejects an unknown group, an unknown step op and an unknown field, so
# nothing here needs to know the grammar below the first segment.
PIPELINE_GROUPS = frozenset({"input", "step", "output"})

# The two `DataOptions` fields that describe WHICH ROWS, not what to do with them. Named
# rather than derived because the split is semantic, not structural: all five fields sit on
# one model. Mirrors `analyses.base.RESERVED_FILTER_KEYS`, which is the same pair seen from
# the registry's side -- the two names a `filters` dict may carry that are not dimensions.
SELECTION_OPTIONS = frozenset({"grid_cell", "where"})

# Everything else `DataOptions` declares -- `sort`, `select`, `color_map` -- read off the
# model rather than listed, so a field added there lands in the right category without a
# second edit here.
OUTPUT_OPTIONS = frozenset(DataOptions.model_fields) - SELECTION_OPTIONS

COMPRESSION_OPTIONS = frozenset(Compression.model_fields)

DATASET_SUGAR = frozenset(BaseDatasetDimensions.model_fields)
FILTER_SUGAR = frozenset(BaseFilterDimensions.model_fields)


# Every flat parameter that used to describe how a request EXECUTED, and the pipeline step
# that is now its only spelling. Each is rejected by name, because the alternative is
# silence: an unrecognized query parameter is ignored by FastAPI's own binding, so a caller
# still sending `group_by=` would have it vanish and get a 200 describing something else.
#
# Lives here, rather than beside the route that first checked it, because the check now
# runs for every route through `partition_query` -- it used to be a `Depends` on 5 of the
# 17 routes, so the same `group_by=` was a targeted 400 on some URLs and silently dropped
# on others.
RETIRED_PARAMS = {
    "value_where": "'where', which now accepts both dimension and value conditions (e.g. 'calendar_year > 2040 AND value > 3')",
    # `negate`/`multiply10`/`multiply100` were all CONDITIONAL rewrites -- scale where X,
    # pass through everywhere else -- so their replacement is a `case` arm, not a
    # `calculate`: `calculate` applies its expression to everything. The pass-through the
    # old parameters left implicit is the `else: "value"` these hints spell out.
    "negate": "a 'diff' named analysis (see /v1/analyses.json), or a 'case' pipeline step directly (negate=X is a case with one arm when=X then='value * -1' and else='value')",
    "multiply10": "a 'case' pipeline step directly (multiply10=X is a case with one arm when=X then='value * 10' and else='value')",
    "multiply100": "a 'case' pipeline step directly (multiply100=X is a case with one arm when=X then='value * 100' and else='value')",
    "group_by": "the 'group' pipeline step (e.g. step.1.op=group&step.1.by=grid_cell)",
    "group_aggregate": "the 'group' pipeline step (e.g. step.1.op=group&step.1.aggregate=mean)",
    # There is no `summarize` op in the pipeline grammar -- the steps are project/join/
    # filter/calculate/case/group. A `project` step's `to`/`aggregate` is what `summarize_by`/
    # `summarize_aggregate` used to spell, and a `grid_cell` `filter` placed AFTER that
    # project is what `summarize_grid_cell` used to spell. Name those, not a step that
    # does not exist -- a 400 that points at nothing is worse than no hint.
    "summarize_by": "a 'project' step's `to` (the grid to summarize onto)",
    "summarize_aggregate": "a 'project' step's `aggregate`",
    "summarize_grid_cell": "a 'filter' step naming `grid_cell`, placed after the 'project' step",
    "unpivot": "nothing -- data is already served in long format (grid_cell/value columns); the parameter was already a documented no-op",
    # The generic calculate entries used to arrive as indexed, dotted query parameters
    # (`calculate.1.expression=...`). There is no flat spelling for `calculate` at all
    # anymore -- and note that `step.1.calculate...` is NOT one of these: the check below
    # matches a top-level `calculate` key only, so the dotted codec's own `calculate` step
    # is untouched.
    "calculate": "the 'calculate' pipeline step directly (e.g. step.1.op=calculate&step.1.expression=value * 2)",
}


def retired_params(keys: Iterable[str]) -> list[str]:
    """Which `RETIRED_PARAMS` names `keys` contains, in `RETIRED_PARAMS` order.

    `calculate` is matched as a PREFIX as well as a name, because the retired spelling was
    a whole family of keys (`calculate.1.expression`, `calculate.2.where`, ...) rather than
    one. Public so that a caller holding a raw query string can ask the question without
    going through the whole partition; `partition_query` below is the only caller that
    needs it today, the FastAPI dependency that used to bind it having gone with the routes
    it guarded.
    """
    keys = list(keys)
    found = [name for name in RETIRED_PARAMS if name in keys]
    if "calculate" not in found and any(key.startswith("calculate.") for key in keys):
        found.append("calculate")
    return found


def retired_param_message(found: Iterable[str]) -> str:
    return "; ".join(f"{name!r} has been retired -- use {RETIRED_PARAMS[name]}" for name in found)


@dataclasses.dataclass(frozen=True)
class QueryPartition:
    """One query string, sorted into the five declared categories plus the leftovers.

    `analysis` is the chain, already split on commas -- `()` for the `DOWNLOAD` sentinel,
    so a caller tests emptiness rather than comparing against the sentinel string.

    `dataset` holds only the triple keys actually given, so "absent" and "given" stay
    distinguishable; `filters` holds every other selected dimension as a LIST of raw
    strings (a dimension is legitimately repeated -- `calendar_year=2030&calendar_year=2040`
    is how the portal spells a two-value selection) plus `grid_cell`/`where` as scalars.
    That is exactly the shape `analyses.base.selection_filter_where` takes, which is where
    it goes next; the raw strings are cast per dimension there
    (`normalized_filter_values`), not here.

    `params` is everything left over -- the named analysis's own declared parameters. It is
    NOT validated here: which names are legal varies per analysis, and
    `resolve_analysis_chain` is what knows.
    """

    analysis: tuple[str, ...]
    dataset: dict[str, str]
    filters: dict[str, Any]
    pipeline_params: dict[str, str]
    output: dict[str, str]
    compression: dict[str, str]
    render: dict[str, str]
    params: dict[str, str]

    @property
    def is_download(self) -> bool:
        return not self.analysis


def partition_query(
    items: Iterable[tuple[str, str]],
    *,
    render_fields: Iterable[str] = (),
    allow_compression: bool = True,
) -> QueryPartition:
    """Sort a raw query string into `QueryPartition`'s categories.

    `items` is the request's `(key, value)` pairs WITH REPEATS PRESERVED -- Starlette's
    `QueryParams.multi_items()`, not `.items()`. A repeat is meaningful for a dimension and
    is the portal's own spelling for a multi-value selection; for every other category the
    last value wins, which is what a scalar query parameter has always meant.

    `render_fields` are the visualization class's own render options (category 5), which
    vary per class and so are supplied rather than declared here. `allow_compression=False`
    is the tile routes, where compression is not a thing a response can carry: passing one
    there is a 400 rather than the silent no-op it would otherwise be (today that is
    enforced structurally, by `api.models.TileQuery` simply not subclassing `Compression`,
    which stops binding it but does not reject it).

    Raises `RequestError` for a retired parameter and for a leftover under the `DOWNLOAD`
    sentinel; a leftover under a named analysis is left for `resolve_analysis_chain` to
    reject against the stage that would have had to declare it.
    """
    render_fields = frozenset(render_fields)

    pairs = [(key, value) for key, value in items]
    found = retired_params(key for key, _ in pairs)
    if found:
        raise RequestError(retired_param_message(found))

    partition: dict[str, dict] = {
        "dataset": {},
        "filters": {},
        "pipeline_params": {},
        "output": {},
        "compression": {},
        "render": {},
        "params": {},
    }
    analysis_raw: str | None = None

    for key, value in pairs:
        if key == "analysis":
            analysis_raw = value
        elif key.split(".", 1)[0] in PIPELINE_GROUPS and "." in key:
            partition["pipeline_params"][key] = value
        elif key in DATASET_SUGAR:
            partition["dataset"][key] = value
        elif key in SELECTION_OPTIONS:
            partition["filters"][key] = value
        elif key in FILTER_SUGAR:
            # The one category that accumulates rather than overwrites -- see the docstring.
            partition["filters"].setdefault(key, []).append(value)
        elif key in COMPRESSION_OPTIONS:
            if not allow_compression:
                raise RequestError(
                    f"{key!r} is not accepted on a tile route -- a tile response carries no compression of its own"
                )
            partition["compression"][key] = value
        elif key in OUTPUT_OPTIONS:
            partition["output"][key] = value
        elif key in render_fields:
            partition["render"][key] = value
        else:
            partition["params"][key] = value

    analysis = ()
    if analysis_raw is not None and analysis_raw != DOWNLOAD:
        names = [name.strip() for name in analysis_raw.split(",")]
        if not all(names):
            raise RequestError(
                f"analysis={analysis_raw!r} has an empty chain stage -- write a chain as "
                "comma-separated names, e.g. analysis=lnd_conf,diff"
            )
        analysis = tuple(names)

    if not analysis and partition["params"]:
        raise RequestError(
            f"unknown query parameter(s): {sorted(partition['params'])} -- with no "
            "analysis= selected there is nothing that could declare them"
        )

    return QueryPartition(analysis=analysis, **partition)


def download_pipeline(dataset: Mapping[str, str], format: str) -> Pipeline:
    """The `DOWNLOAD` sentinel's base pipeline: the dataset triple as this request's one
    input, no steps at all.

    Named `DEFAULT_INPUT_NAME` ("data") rather than something derived from the request,
    because that is the name `pipeline_to_query` already renders a flat request's input
    under -- so a saved link and a `pipeline.json` response spell the same input the same
    way whether the caller sent `variable=elev` or `input.data.variable=elev`. The whole
    point of the sugar is that the two resolve to one pipeline.

    Raises `RequestError` naming the missing keys if the triple is incomplete: a partial
    triple is a caller who meant to select a dataset, and reporting "missing grid" beats
    `InputSelection`'s own field-level error about a model they never mentioned.
    """
    missing = sorted(DATASET_SUGAR - set(dataset))
    if missing:
        raise RequestError(
            f"incomplete dataset selection: missing {missing} -- a dataset is named by all "
            f"of {sorted(DATASET_SUGAR)} together"
        )
    return pipeline_from_query(
        {
            **{f"input.{DEFAULT_INPUT_NAME}.{field}": dataset[field] for field in dataset},
            "output.format": format,
        }
    )


def _renumber_steps(dotted: Mapping[str, str], offset: int) -> dict[str, str]:
    """Shift every `step.N.*` key in `dotted` up by `offset`, leaving `input.*`/`output.*`
    alone.

    This is what "the caller's steps append after the analysis's steps" means concretely.
    The caller always numbers from 1 -- they are writing about their own request and have
    no way to know how many steps the analysis they named contributes, which may change
    with the analysis's parameters (`diff` contributes two, `lnd_chng` five or seven).
    """
    shifted: dict[str, str] = {}
    for key, value in dotted.items():
        head, _, rest = key.partition(".")
        if head != "step":
            shifted[key] = value
            continue
        index, _, tail = rest.partition(".")
        if not index.isdigit() or not tail:
            raise RequestError(
                f"{key!r} is not a step parameter -- steps are 1-based indices with a field, e.g. step.1.op=filter"
            )
        shifted[f"step.{int(index) + offset}.{tail}"] = value
    return shifted


def build_pipeline(
    base: Pipeline | None,
    pipeline_params: Mapping[str, str],
    output: Mapping[str, str] = {},
    *,
    format: str | None,
) -> Pipeline:
    """Combine `base` -- the resolved analysis's pipeline, or the `DOWNLOAD` sentinel's
    one-input pipeline, or `None` when the caller's dotted parameters are the whole
    request -- with the caller's own dotted parameters and flat output options, and fix
    `output.format` from the URL.

    Merging is done on the ENCODED form (see the module docstring): `base` is re-encoded
    with `pipeline_to_query`, the caller's keys are written over the result, and
    `pipeline_from_query` decodes the union. An `input.<name>.<field>` therefore merges
    field-wise into an input the analysis already declared and adds a new input when the
    name is new, both for free, and a malformed step index or an unknown field is caught by
    the same decoder that catches it for a hand-authored request.

    `output` is the FLAT spelling of the same three fields (`sort`/`select`/`color_map`),
    written in as `output.<field>` rather than `setattr`-ed onto the built pipeline
    afterward -- which is what the route-level `_apply_output_overrides` it replaces did,
    and which skipped `validate_pipeline`'s own `sort`/`select` column-name checks entirely.
    A key the caller did not send leaves whatever the analysis declared, so an analysis can
    still ship a default.

    `format` always wins over whatever `base` or the caller declared, on every route that
    has one. The URL's extension is the only place a tile request can state a format at
    all, and refetching one saved pipeline as `.csv` and as `.png` is an ordinary thing for
    a client to do -- the portal does exactly that. One rule for every route is also what
    keeps `pipeline_to_query(pipeline)` a stable identity independent of the extension it
    was fetched with.

    `format=None` is the one route with no extension to win from --
    `/v1/analyses/{analysis}.json`, which serves a pipeline rather than running one. There,
    whatever `base` or the caller declared STANDS: an analysis that fixes its own raster
    sink (`lnd_type_plus`, `sal_veg_investigation`) publishes the `tif` it actually
    resolves to, instead of having a `csv` written over it that the same pipeline's own
    `validate_shapes` would then reject. `DEFAULT_FORMAT` covers the case where neither
    declared one -- a bare dotted-step request with no `output.format=`, which has to
    produce SOME sink for `Output`'s required field.
    """
    merged = pipeline_to_query(base) if base is not None else {}
    merged.update(_renumber_steps(pipeline_params, len(base.steps) if base is not None else 0))
    merged.update({f"output.{field}": value for field, value in output.items()})
    if format is not None:
        merged["output.format"] = format
    else:
        merged.setdefault("output.format", DEFAULT_FORMAT)
    return pipeline_from_query(merged)


def append_selection(pipeline: Pipeline, where: str | None) -> Pipeline:
    """Append the caller's selection to `pipeline` as ONE trailing `filter` step, or return
    `pipeline` unchanged when there is nothing selected.

    Trailing, always, with no special case for "an analysis was named" -- the caller's
    selection is against the pipeline's OUTPUT frame, which is the intuitive reading and
    the only one that is right for a chain. It costs nothing at the read: `execution.plan`
    pushes a dimension conjunct declared anywhere down to every origin's read, across joins
    and past `project`, so a trailing dimension filter reads the same bytes a leading one
    would.

    The step names its input only when `pipeline` has no steps of its own to chain from --
    `validate_pipeline`'s ordinary rule, which `append_steps` re-runs over the combined
    list. `pipeline.output` is carried through unchanged; this changes which rows are
    written, not what is written or how.
    """
    if where is None:
        return pipeline
    input_name = None if pipeline.steps else next(iter(pipeline.inputs))
    return append_steps(pipeline, [FilterStep(input=input_name, where=where)], pipeline.output)
