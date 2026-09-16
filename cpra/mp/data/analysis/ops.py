"""The op IR: a small typed `Op` per pipeline step -- `ProjectOp`/`JoinOp`/`FilterOp`/
`CalculateOp`/`GroupOp`, exactly the five ops `pipeline.py`'s `Step` union already names
(there is no sixth "summarize" op -- `project`'s own `aggregate` field IS the
summarize/reaggregate case; see `ProjectStep`'s docstring) -- plus `compile_ops`,
which turns an already-validated `Pipeline` into `list[Op]` **in declared order**.

**Shape-agnostic at the interface.** An `Op` carries only what the pipeline DECLARED: a
`project`'s target grid, a `filter`'s predicate, a `calculate`'s expression, a `group`'s
aggregate -- never whether the frame it will eventually run against is `long` (vector: one
row per (dimensions, `grid_cell`, `value`) observation) or `band_wide` (raster: one row per
pixel, one column per band). `LOWERING_TABLE` below is the ONE place shape is consulted,
reproduced as data rather than as code that runs. Whether `filter`'s predicate is a value
condition or a dimension condition also matters to the lowering -- that is a property of
the predicate's own text (see `expressions.classify_where`), not something `FilterOp`
stores twice, so `LOWERING_TABLE` keys `filter` by that split directly rather than adding
a field to `FilterOp` no other op needs.

**This module only compiles the IR; `planner.compile_pipeline` still lowers a `Pipeline`
onto `Options`/`PairedAnalysis` for the shapes that have no interpreter of their own.**
`compile_ops` here does not run `planner._compile_single_input_chain`'s SUMMARIZING-PROJECT
ORDER GUARD (the rejection of a `calculate`/value-`filter` declared after a many-to-one
`project`, needed because that pair actually executes out of declared order once lowered
onto `Options` -- see that guard's own comment in `planner.py`): a `Pipeline` shaped that
way compiles here just fine, in its declared order, because emitting the IR needs no
grid-type or crosswalk knowledge at all (contrast `planner._validate_project_cardinality`,
which does, and is exactly why that module -- not this one -- owns the guard). That is
deliberate, not an oversight: preserving declared order for exactly the pipelines the flat
`Options` engine gets wrong is this module's whole point (see `tests/api/test_ops.py`).

`vector.VectorGridAnalysis.apply_steps` and `raster.RasterGridAnalysis.apply_steps` are the
two interpreters that actually walk `list[Op]` and apply `LOWERING_TABLE`'s row for each op
against a real LazyFrame, one shape each.
"""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from cpra.mp.data.analysis.pipeline import (
    CalculateStep,
    FilterStep,
    GroupStep,
    JoinStep,
    Pipeline,
    PipelineError,
    ProjectStep,
    _step_explicit_inputs,
)
from cpra.mp.data.analysis.types import AggregateFunction, Grid


class ProjectOp(BaseModel):
    """Move `input`'s rows onto `to`'s grid, aggregating with `aggregate` when the move is
    many-to-one (a `to` matching the resolved input's own grid, or `aggregate is None`, is
    identity/one-to-many -- which of those it is needs crosswalk/grid-type knowledge this
    op deliberately does not carry; see `ProjectStep`'s own docstring, which this mirrors
    field-for-field). `long` lowers this onto a crosswalk-table join + `group_by`; band-wide
    onto a zonal join against a zone-ID crosswalk raster + `group_by` -- see
    `LOWERING_TABLE`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    op: Literal["project"] = "project"
    step_no: int
    input: str
    to: Grid
    aggregate: AggregateFunction | None = None
    output: str


class JoinOp(BaseModel):
    """Combine `inputs` (always exactly the names `JoinStep.inputs` gave, in that order --
    `join` is the one op with no implicit-chain default; see `pipeline.py`'s chaining rule)
    on shared dimensions plus `grid_cell`. `long` lowers this onto an ordinary Polars join
    keyed on `grid_cell` plus every shared dimension (`sources.scan_rasters`' crosswalk
    docstring); band-wide onto an O(1) dense lookup array over a zone-ID raster
    (`sources._dense_lookup_array`) rather than a generic join, since that array beats a
    hash join for a small, dense key range and is worth preserving as-is -- see
    `LOWERING_TABLE`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["join"] = "join"
    step_no: int
    type: Literal["spatial", "attribute"]
    inputs: list[str]
    output: str


class FilterOp(BaseModel):
    """A predicate (`where`) over `input`'s in-scope columns. The ONE op whose lowering
    depends on more than its own type: whether `where` tests `value` or a dimension column
    changes both shapes' behavior (long: a row filter either way; band-wide: a per-cell
    mask for a value condition, since dropping a pixel's ROW would drop it from every
    band, versus a column selection -- projection pushdown -- for a dimension condition,
    since a dimension lives in column identity, not a cell, in band-wide shape). See
    `LOWERING_TABLE`, keyed by predicate kind, not by a field here -- `expressions.
    classify_where` is what tells the two apart, and it needs no new field on this op to
    keep doing that.

    `mask_value` is the ONE thing this op records that a pipeline step never declares: for
    a LONG-shape value conjunct, whether a non-matching row is masked to null
    (`mask_value=True`) or dropped outright (`False`, the default and the only value
    `compile_ops` ever emits). It exists because the flat request surface has no declared
    order at all -- `stages.lower_flat_options` emits its single `where` FIRST,
    unconditionally, ahead of whatever `group_by`/`summarize_by` the same request asked
    for -- so a flat `where` on an AGGREGATING request is by construction hoisted above the
    aggregate it was written against, and masking is what lets it compose with that
    aggregate's own null-skipping behavior instead of changing the aggregate's denominator
    or erasing a group entirely (see `stages.lower_flat_options`, which sets it, and
    `vector.apply_long_op`, which reads it). A pipeline `filter` step runs at its own
    declared position, so there is no downstream null for an aggregate to skip and a drop
    is the only coherent lowering -- hence the default.

    The band-wide lowering ignores this field: a value conjunct there is ALWAYS a per-cell
    mask, because that shape must stay positionally aligned to the raster (see
    `raster.apply_value_filter_band_wide`), which is a property of the shape rather than of
    the request.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["filter"] = "filter"
    step_no: int
    input: str
    where: str
    output: str
    mask_value: bool = False


class CalculateOp(BaseModel):
    """Conditionally (`where`, optional) rewrite `input`'s running value by `expression`.
    `long` lowers this onto `with_columns` over the `value` column; band-wide onto
    `with_columns` across every band column (reusing `sources.band_calculate_passes`'s
    existing vocabulary, not a second implementation) -- see `LOWERING_TABLE`.

    `label` is what a 400 out of `expressions.compile_calculate_entries` calls this entry.
    It is diagnostic only -- nothing dispatches on it -- and defaults to `None`, meaning
    "name me after my step", which is what a pipeline step wants (`step.3`). The flat
    request surface sets it explicitly to `calculate.N`, the name its own `calculate.N.
    expression=`/`calculate.N.where=` query parameters already use (`api/__init__.py`
    parses exactly those keys), so a caller's error message names the parameter they
    actually wrote. `calculate_label` below is the one place the two spellings are
    resolved.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: Literal["calculate"] = "calculate"
    step_no: int
    input: str
    expression: str
    where: str | None = None
    output: str
    label: str | None = None


class GroupOp(BaseModel):
    """Collapse `by` (empty means "one row per `grid_cell`") with `aggregate` applied to
    `of`. `long` lowers this onto `group_by(...).agg(...)` (or `analysis.fold`'s
    cliff-shaped fold, for a non-decomposable-but-foldable aggregate); band-wide onto a
    HORIZONTAL reduce across band columns (again `sources.HORIZONTAL_RASTER_AGGREGATES`'s
    existing vocabulary) -- see `LOWERING_TABLE`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    op: Literal["group"] = "group"
    step_no: int
    input: str
    by: list[str]
    aggregate: AggregateFunction
    of: str = "value"
    output: str


Op = Annotated[
    Union[ProjectOp, JoinOp, FilterOp, CalculateOp, GroupOp],
    Field(discriminator="op"),
]


def calculate_label(op: CalculateOp) -> str:
    """What a 400 raised while compiling `op`'s expression should call it: the explicit
    `CalculateOp.label` a flat request set (`calculate.1`, `calculate.2`, ... -- the query
    parameter the caller actually wrote), else `step.<step_no>`, the pipeline step's own
    position.

    One function rather than the same `or` twice, because both lowerings of `calculate`
    need it -- `vector.apply_long_op` and `raster.apply_band_wide_op` -- and a label that
    disagreed between the two would make the SAME request name its offending entry
    differently depending on which shape it happened to run against.
    """
    return op.label if op.label is not None else f"step.{op.step_no}"


class Shape(str, Enum):
    """The two data shapes an op lowers against. `long` is the vector row model --
    dimension columns + `grid_cell` + `value`, one row per observation. `band_wide` is the
    raster row model -- one row per pixel, one column per band. The paired
    (spatially-joined) shape adds no THIRD entry here: its output rows are still one of
    these two -- `long` for a tabular zonal pairing, `band_wide` for a pixel-local one
    driving a raster write/tile -- `join` is just an ordinary op like any other, lowered
    per `LOWERING_TABLE`'s `join` row rather than matched as a template; see `JoinOp`'s
    docstring.
    """

    long = "long"
    band_wide = "band_wide"


# The lowering table itself, reproduced as DATA (never executed) rather than as five
# function bodies, so that which op, which shape, which strategy is written down and
# checked (`tests/api/test_ops.py::test_lowering_table_covers_every_op_and_shape`) in
# exactly one place, per the module docstring's "shape-agnostic at the interface" framing.
#
# `filter` has TWO rows, not one, because its lowering depends on what its predicate
# tests, not on the op alone -- see `FilterOp`'s docstring. Every other op has a single
# row: `project`'s row already covers a many-to-one reaggregation (there is no separate
# "summarize" op -- see the module docstring), and `join`'s row states the band-wide
# lowering in `JoinOp`'s own docstring: an O(1) dense lookup array over a zone-ID raster,
# not a generic join.
LOWERING_TABLE: dict[tuple[str, Shape], str] = {
    ("filter (value)", Shape.long): "row filter",
    ("filter (value)", Shape.band_wide): "per-cell mask",
    ("filter (dimension)", Shape.long): "row filter",
    ("filter (dimension)", Shape.band_wide): "column selection (projection pushdown)",
    ("calculate", Shape.long): "with_columns on value",
    ("calculate", Shape.band_wide): "with_columns across band columns",
    ("group", Shape.long): "group_by(...).agg(...) / fold",
    ("group", Shape.band_wide): "horizontal reduce across band columns",
    ("project", Shape.long): "crosswalk-table join + group_by",
    ("project", Shape.band_wide): "zonal join on zone-ID raster + group_by",
    ("join", Shape.long): "ordinary join on grid_cell + shared dimensions",
    (
        "join",
        Shape.band_wide,
    ): "O(1) dense lookup array over a zone-ID raster (sources._dense_lookup_array) -- not a generic join",
}


def _default_output_name(step_no: int) -> str:
    """The output binding synthesized for a step that gave no `as_`. Every `Op` needs a
    non-`None` `output` so a later `Op`'s `input`/`inputs` can always be "the name of some
    op's output" uniformly, whether that name came from an explicit `as_` or an implicit
    "chain from the previous step" default (see `compile_ops`) -- there is no
    residual positional/`None` case once the IR exists.

    A bare digit string can never collide with a real `as_`/input name: both are required
    (`pipeline._validate_name`'s `_NAME_RE`) to start with a lowercase letter or
    underscore, so `str(step_no)` -- which starts with a digit -- is disjoint from every
    legal name by construction. No registry or uniqueness check is needed for that reason;
    distinct `step_no` values are already distinct strings.
    """
    return str(step_no)


def compile_ops(pipeline: Pipeline) -> list[Op]:
    """Turn an already-validated `Pipeline` -- one that passed
    `pipeline.parse_pipeline`/`validate_pipeline`, whose chaining/`as`/aggregate-
    decomposability rules this function assumes rather than re-checks -- into `list[Op]`,
    one entry per step, **in declared order**. Unlike `planner._compile_single_input_chain`/
    `_compile_paired_notebook_shape`, this needs no `base_path`, no catalog, no crosswalk,
    and no grid-type lookup: it never opens a dataset and never decides a `project`'s
    cardinality direction, because an `Op` records only what the pipeline DECLARED (see
    the module docstring's "shape-agnostic at the interface").

    Every `Op`'s `input` (or `inputs`, for `join`) is fully resolved here, never left as an
    implicit "chain from the previous step" default: `pipeline._step_explicit_inputs` says
    only whether a step named its input explicitly, and `validate_pipeline`'s chaining rule
    checks but does not store what an omitted one resolves to -- this function is what
    actually resolves it, to the immediately preceding step's own `output` name, so a
    downstream consumer of `list[Op]` never has to re-derive "the previous op" positionally;
    every reference is an explicit string, the same uniform shape whether it came from an
    explicit `input`/`join.inputs` or an implicit chain.
    """
    ops: list[Op] = []
    previous_output: str | None = None
    for step_no, step in enumerate(pipeline.steps, start=1):
        as_name = getattr(step, "as_", None)
        output = as_name if as_name is not None else _default_output_name(step_no)

        if isinstance(step, JoinStep):
            op: Op = JoinOp(step_no=step_no, type=step.type, inputs=list(step.inputs), output=output)
            ops.append(op)
            previous_output = output
            continue

        explicit = _step_explicit_inputs(step)
        input_name = explicit[0] if explicit else previous_output
        if input_name is None:
            # Unreachable for a `Pipeline` built through `parse_pipeline`: its chaining
            # rule already rejects a step 1 that names no explicit input ("step 1 (...)
            # has no input to chain from" -- pipeline.py's `validate_pipeline`). Guarded
            # here rather than silently emitting an `Op` with input=None, so a `Pipeline`
            # constructed directly (bypassing `parse_pipeline` -- see this module's own
            # docstring on why a test might do exactly that, to reach an IR shape
            # `planner.py`'s SUMMARIZING-PROJECT ORDER GUARD would otherwise reject first)
            # fails loudly instead of producing an IR no lowering could ever run.
            raise PipelineError(
                f"step {step_no} ({step.op}) has no input to chain from -- this Pipeline "
                "did not pass validate_pipeline's chaining rule"
            )

        if isinstance(step, ProjectStep):
            op = ProjectOp(
                step_no=step_no, input=input_name, to=step.to, aggregate=step.aggregate, output=output
            )
        elif isinstance(step, FilterStep):
            op = FilterOp(step_no=step_no, input=input_name, where=step.where, output=output)
        elif isinstance(step, CalculateStep):
            op = CalculateOp(
                step_no=step_no, input=input_name, expression=step.expression, where=step.where, output=output
            )
        elif isinstance(step, GroupStep):
            op = GroupOp(
                step_no=step_no,
                input=input_name,
                by=list(step.by),
                aggregate=step.aggregate,
                of=step.of,
                output=output,
            )
        else:  # pragma: no cover -- exhaustive over the Step union
            raise PipelineError(f"step {step_no} has unrecognized op {getattr(step, 'op', None)!r}")

        ops.append(op)
        previous_output = output

    return ops
