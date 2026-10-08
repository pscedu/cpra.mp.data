"""The **output descriptor**: what a resolved `Pipeline` would produce, described well
enough to render it -- the grid it is addressed against, the shape it comes out in, the
time unit its rows are keyed by, and the display metadata of the variable behind each of
its value columns.

One function, two kinds of consumer. `GET /v1/catalog.json` serves this alongside the
dimension choices, and every visualization takes its render context from it instead of
from `variable`/`grid`/`time_unit` constructor arguments. They are the same object on
purpose: what the catalog advertises about a pipeline's output and what a chart needs in
order to draw that output are the same facts, and two derivations of them would drift.

**Derived, not declared.** Everything here was previously passed in by whoever built the
visualization, which meant a caller with no dataset of its own -- an analysis that fixes
its own inputs, like `lnd_type_plus` -- had to INVENT a plausible triple and keep it in
agreement with what the analysis actually does (the portal's `CANONICAL_DERIVED_DATASETS`).
Reading it off the pipeline removes both the invention and the agreement:
`lnd_type_plus` projects onto `morph_pixel_v002` and `output_frame_type` says so.

**Nothing here opens a file.** `ops.output_frame_type` needs only the declared ops and
each input's grid kind, and the rest is read straight off `pipeline.inputs`, so a
descriptor is available synchronously at spec-build time -- before the pipeline is
executed, and long before any crosswalk or raster is touched.

**Why a module of its own, rather than more of `ops.py`.** `ops.py` is the op IR and the
shape-typing rules, and is deliberately ignorant of presentation: it consults
`grids.GRIDS` only for a grid's KIND, because that is what picks an op's lowering.
Variables, legends, display units and time columns decide none of that. This module is
the one that layers them on, in one direction -- it imports `ops`, and `ops` does not
import it.

**The analysis's own `Meta` is NOT folded in here.** An analysis may override the
presentation of its result (`Meta.label`/`legend`/`result_kind`), and that override wins
over anything derivable from the pipeline -- but `analyses/` imports `request/`, so this
module cannot read a `Meta` without inverting that. The overlay therefore stays with the
consumer, which is where it already lives: `visualizations.map` has always written
`LEGENDS.get(self._legend_name()) or self.variable.legend or ...`, and this changes only
where the second operand comes from.
"""

import dataclasses

from cpra.mp.data.analysis.request.ops import FrameType, output_frame_type
from cpra.mp.data.analysis.request.pipeline import Pipeline
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.variables import VARIABLES, Variable

# The time units whose rows are keyed by a whole year rather than by a date. Mirrors
# `visualizations.base.DatasetVisualization.get_time_column`, which is where the resulting
# column name is consumed; the mapping is written down once here so that a chart, a CSV
# parse spec and a catalog entry cannot disagree about which column carries time.
_YEAR_KEYED_TIME_UNITS = frozenset({"annual", "decadal"})


@dataclasses.dataclass(frozen=True)
class OutputDescriptor:
    """What `pipeline` would write, in the terms a renderer needs.

    `frame_type` is the whole of `ops.output_frame_type`'s answer, kept rather than
    unpacked so a consumer that needs a fact this class does not name (`has_count_by_value`,
    `extra_columns`) has it without a second walk. `grid`/`grid_type`/`shape` are lifted
    out of it because they are what almost every consumer actually asks for:

    - `grid_type` is the raster-vs-vector fork that decides whether a `grid_cell` is a
      numeric cell id or a `"lng,lat"` coordinate pair, whether a map source is a tile
      layer or a feature layer, and whether a CSV's `grid_cell` column parses as a number;
    - `shape` says whether the result is a surface or a table, which is what the two tile
      renderers fork on.

    `time_unit` is `None` when the pipeline's inputs do not agree on one -- possible in
    principle for a multi-input analysis, and a consumer that needs the literal word (a
    chart title) should fall back to the analysis's `Meta.label` rather than picking one.
    `time_column` follows it and is `None` for the same reason.

    `variables` is parallel to `value_columns` and holds the resolved presentation
    metadata for each -- `None` where none could be resolved, which `visualizations.
    display.display_value_expr` already reads as "no scaling". See `output_descriptor`
    for the resolution rule.

    `folded_into` is what tells a consumer how to read `value_columns`, and is `None` for
    every pipeline with no `unpivot` in it -- which is all of them but
    `sediment_dynamics`. When it is set, the names in `value_columns` are not columns of
    the result at all: they are the VALUES its `folded_into` column takes, one per stacked
    measurement, and the result's only real value column is the `value` every folded row
    shares. The pairing with `variables` is unchanged either way, which is the point of
    describing it like this -- "which variable is this number" is answered by position in
    the same tuple whether the answer lives in a column name or in a row's label.
    """

    frame_type: FrameType
    grid: str
    grid_type: str
    shape: str
    value_columns: tuple[str, ...]
    variables: tuple[Variable | None, ...]
    time_unit: str | None
    time_column: str | None
    folded_into: str | None = None

    @property
    def variable(self) -> Variable | None:
        """The one variable this output is about, or `None` when it carries several.

        `validate_shapes` allows exactly one value column at the sink, so today this is
        `variables[0]` for every pipeline that can actually be written. It stays a property
        rather than the only field because an `unpivot`ed result folds several variables
        into one long `value` column and is described per folded name, not by one of them.
        """
        if len(self.variables) != 1:
            return None
        return self.variables[0]


def output_descriptor(pipeline: Pipeline) -> OutputDescriptor:
    """Describe what `pipeline` would produce. `pipeline` must already be structurally
    valid (`pipeline.parse_pipeline`); this raises the same per-op `ShapeError`s
    `ops.validate_shapes` does, since `output_frame_type` runs the same walk.

    The time unit comes from the inputs: `InputSelection.time_unit` is the only place a
    pipeline says how its rows are keyed in time, no step changes it, and a `group` that
    collapses the time dimension removes the column without changing what the remaining
    rows were measured in. Inputs that disagree yield `None` rather than a guess.

    A value column's variable is resolved by NAME first, then by the pipeline's inputs:

    1. `VARIABLES[column_name]`. A join names each operand's value column after the
       operand, and an operand is named for the variable it reads, so a multi-variable
       result answers per column by name alone. This is the trick
       `visualizations.base.source_variable` used to play on data-source names, moved onto
       value-column names, where the naming is actually guaranteed -- and the reason that
       function is gone: `SedimentDynamics` was its only multi-source caller.
    2. Otherwise, the pipeline's one input variable. The default value column is just
       `"value"` -- named for the field it fills, not for what fills it -- so a single-input
       pipeline (every dataset download, every `diff`, every chart today) is answered here.
       A multi-input pipeline whose columns did not resolve by name gets `None`, and the
       consumer falls back to the analysis's own `Meta`; an analysis that collapses several
       variables and declares no presentation is rejected at registration rather than
       guessed at here.

    **An `unpivot`ed pipeline is described by its FOLDED names, not by `value`.** The fold
    leaves one value column called `value`, which resolves to no variable by name and to
    the sole input variable only when there is one -- neither of which describes a result
    that is five variables stacked. So `value_columns` reports `FrameType.folded_columns`
    instead, `variables` resolves those by the same rule 1 above (a join named each
    operand after its variable, and the fold turned those names into row labels without
    changing them), and `folded_into` names the column the labels live in so a consumer
    can tell the two readings apart. `OutputDescriptor.variable` then correctly answers
    `None`: there is no one variable this output is about.
    """
    frame_type = output_frame_type(pipeline)

    time_units = {selection.time_unit for selection in pipeline.inputs.values()}
    time_unit = time_units.pop() if len(time_units) == 1 else None

    input_variables = {selection.variable for selection in pipeline.inputs.values()}
    sole_input_variable = VARIABLES.get(next(iter(input_variables))) if len(input_variables) == 1 else None

    described = frame_type.folded_columns or frame_type.value_columns
    value_columns = tuple(sorted(described))
    return OutputDescriptor(
        frame_type=frame_type,
        grid=frame_type.grid,
        grid_type=GRIDS[frame_type.grid].grid_type,
        shape=frame_type.shape.value,
        value_columns=value_columns,
        variables=tuple(VARIABLES.get(column) or sole_input_variable for column in value_columns),
        time_unit=time_unit,
        time_column=time_column_for(time_unit),
        folded_into=frame_type.folded_into,
    )


def time_column_for(time_unit: str | None) -> str | None:
    """The column a dataset of `time_unit` keys its rows by in time, or `None` for an
    unknown/absent unit.

    A pure function of the unit, exposed so that the several consumers that need the column
    without needing a whole descriptor (a CSV parse spec, an x-scale type, a groupby key)
    read the same mapping rather than each repeating the two-name comparison.
    """
    if time_unit is None:
        return None
    return "calendar_year" if time_unit in _YEAR_KEYED_TIME_UNITS else "calendar_day"
