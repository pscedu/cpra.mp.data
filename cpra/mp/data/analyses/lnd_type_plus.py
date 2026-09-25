"""`lnd-type-plus` -- a `type="custom"` analysis that extends the `lnd_type` categorical
surface with two SHALLOW-INUNDATION classes the morphology model's land types do not
themselves distinguish. Annual inundation depth is COMPUTED here rather than read: mean
annual stage (`stg`/annual/`hydro_compartment_v002`, `aggregate_type=mean`) is projected
onto `morph_pixel_v002`, spatially joined against topobathymetric elevation
(`elev`/annual/`morph_pixel_v002`, stored in millimetres) and annual land type
(`lnd_type`/annual/`morph_pixel_v002`), and the difference of the first two is reclassified
where the third says open water: `6` (inundated, no deeper than 0.46 m), `7` (deeper than
0.46 m, no deeper than 1.0 m), and `2` -- plain open water -- deeper than that. Everything
the reclassification does not claim is the land type, unchanged.

The whole analysis is five steps and one join -- filter stage to its aggregate type,
project it onto the pixel grid, join all three inputs, state the caller's selection, and
reclassify with a single `case`. There is no reduce: the `case` is total, its `else` is
`land_type`, and the result's domain is `lnd_type`'s own `1`-`5` plus the two new codes.

**Why deep water has no code of its own.** The output domain is what
`structure/legends.csv` publishes for the `lnd_type_plus` legend: values `1`-`7`, with `2`
labelled "Water - Deep". Deep water is not reclassified at all -- it is simply water, and
only the two shallow classes say something `lnd_type` could not. So the deepest class's
`then` IS `WATER_LAND_TYPE`, and its arm returns exactly what the `else` would have. It is
still load-bearing, and not because of its value: the arms are first-match-wins in
DESCENDING order of bound, so the `> 1.0` arm is what stops a 1.5 m depth from falling
through to the `> 0.46` arm and reporting `7`. It claims deep water in order to withhold it
from the shallow classes, and hands back the land type when it does.

**Why the shallow classes are confined to water.** The rule this analysis implements
subdivides OPEN WATER (`lnd_type = 2`) by depth; it does not reclassify land. That the two
should coincide -- a positive depth means the water level stands above the ground, which is
water -- is a property of the two input surfaces agreeing, and they are independently
modelled, so at the coastline they disagree. A pixel where they do would otherwise report
shallow inundation over marsh or upland. So the restriction is stated rather than assumed --
as an `AND land_type = 2` conjunct on EVERY arm of the one `case` below, which is why
`land_type` is an operand of the same join the depth's two operands arrive on. No arm
matches off water, and the `else` hands back the land type, so a non-water pixel reads as
its own land type no matter how deep the depth surface says it is.

**Why the depth is computed inside the `case` arms.** `stage_m - elevation` could be a
`calculate` of its own over the joined operands -- a `calculate` is an arm-less `case`, and
both name a joined frame's operands directly, because over such a frame the value namespace
IS the operand names (band-wide, each group's names are rewritten to that group's own band
columns before the expression is compiled -- `execution.band_ops.case_band_wide`). What
that step would cost is the rest of the join: any step that writes the running value
collapses the frame to ONE value column, so a depth computed separately arrives with
`land_type` already gone, and the water restriction below would have nothing left to test.
Computing the depth inside the arms keeps all three operands live until the single step
that consumes them. Millimetres to metres rides along in the same expression
(`elevation / 1000`), which is why there is no separate conversion on elevation's own
branch either: one shared `depth` string says it once for all three arms, with no step and
no intermediate name to carry it.

Neither the join nor the `case` is limited to two operands: `pipeline.JoinStep.inputs` is
`min_length=2`, and `band_ops._pair_band_groups` keys on the dimensions the operands share
and requires exactly one band per operand per group, for any number of them. That is what
lets stage, elevation and land type arrive on ONE join and be reclassified by ONE `case`,
instead of pairing the depth's two operands, collapsing them, and joining the land type in
afterwards to restrict on it.

**The presence requirement is carried by the arms, and costs no step.** `case` adds no
nodata guard of its own (see `execution.long_ops.case_expression` for why it no longer
does), so what a pixel missing an operand does here is whatever the arms and the `else`
SAY. Missing stage or elevation makes the depth null, a null predicate is `fill_null(False)`
(`expressions.value_predicate_mask`), so no arm matches and the `else` reads the land type
alone -- the right answer, since an unknown depth is not evidence of inundation. Missing
LAND TYPE nulls both sides at once, the conjunct and the `else`, so such a pixel stays
nodata. An older formulation said the first half with a `join` + `IS NOT NULL` filter +
`join` + `{"aggregate": "sum"}`, and said it incompletely: a horizontal `sum` SKIPS nulls
rather than propagating them, so a pixel with elevation but no stage summed to the bare
negated elevation and reclassified as real inundation wherever the ground sits below the
NAVD88 datum. Four steps became none, and none of them is stricter.

**Why the selection is stated once, at the end.** `model_group_id`/`scenario_id`/
`calendar_year` are one `filter` on the fully joined frame, just before the `case`, rather
than a leading `filter` repeated per input. Every step between the reads and that filter
works one (model group, scenario, year) triple at a time -- the join groups band-for-band on
exactly those three dimensions and the `case` collapses a pairing that agrees on them -- so
narrowing at the end and narrowing at the front compute the same thing, and stating it once
is the honest way to say it. It sits BEFORE the `case` rather than after because a dimension
filter leaves the pairing standing while the `case` is what collapses it.

What makes it cost the same is `execution.plan`'s two dimension passes: a dimension
conjunct reaches every origin's READ no matter where in the pipeline it was declared,
across joins and past the `project`, so all three inputs narrow to the selected bands
before anything is opened. Without that this would be a catastrophic read -- `elev` and
`lnd_type` are 178 and 179 bands across every model group and scenario on disk -- which is
why `tests/api/test_analyses_registry.py`'s `TestLndTypePlusNarrowsItsReads` compiles this
pipeline against two years on disk and asserts the band counts, rather than trusting the
property: no RESULT would change if the pushdown stopped working, only the read.
`aggregate_type` is the one filter that stays on its own input: it is definitional rather
than a selection, and the pairing does not carry it (it is the one dimension the three
inputs do not share), so there is no later frame to state it on.

**Why the reclassification is one `case` step.** The target codes are not disjoint from the
source domains in any order -- `6` and `7` both fall in the `> 1.0` domain -- so a chain of
successive rewrites cannot state this at all without an encoding trick, since each one
rewrites the RUNNING value and the next one's test then sees what it wrote. A `case` has no such
hazard: every arm's `when` is evaluated against the INCOMING operands, the first match wins,
and each pixel is written exactly once. The arms are therefore stated in DESCENDING order of
bound, each as a single `>` test of the depth against the previous class's bound, `AND`ed
with the water conjunct the paragraph above explains. The last one, `... > 0`, is the domain
floor, and the `else` is what catches everything at or below it -- an elevation standing at
or above the water level, which is dry land rather than any inundation class, and which
therefore reads as its own land type.

Nothing here filters `lnd_type` to its `1`-`5` value domain, and nothing needs to: every
variable declares a nodata sentinel (`variables.py`'s `Variable.nodata`, `-9999` unless
`variables.csv` names another -- `elev`'s `-9999000` is the one exception, its own sentinel
in millimetres), `raster.write_cog` writes it into the COG profile, and the read path masks
it to null before any step here sees it. So the `else` hands back a null rather than a
`-9999`, and a sentinel can never be mistaken for a land-type code.
"""

from typing import Any

from cpra.mp.data.analyses.base import (
    Analysis,
    DatasetRef,
    InvalidAnalysisParametersError,
    required_dataset_dimensions,
    selection_filter_where,
)
from cpra.mp.data.analysis.request.pipeline import parse_pipeline

# The three fixed inputs. `elev` and `lnd_type` are already `morph_pixel_v002` rasters;
# `stg` is a hydro-compartment POLYGON series, and is brought onto the pixel grid by the
# `project` step below -- the one grid pair this analysis needs a crosswalk for.
STAGE_INPUT = DatasetRef(variable="stg", time_unit="annual", grid="hydro_compartment_v002")
ELEVATION_INPUT = DatasetRef(variable="elev", time_unit="annual", grid="morph_pixel_v002")
LAND_TYPE_INPUT = DatasetRef(variable="lnd_type", time_unit="annual", grid="morph_pixel_v002")

# `stg` declares `aggregate_type` required, and the choice is DEFINITIONAL rather than a
# selection: the class bounds below only mean something against one reading of stage, so
# fixing it here is the same call `salinity-veg-investigation` makes for its own
# `aggregate_type=2_wk_max`. Mean annual stage is the reading that pairs with an annual
# morphology elevation surface.
STAGE_AGGREGATE_TYPE = "mean"

# `elev` is stored as Int32 NAVD88 MILLIMETRES (`variables.csv`'s `unit`), and everything
# else here is metres. Folded into the negation below rather than applied separately, so
# the conversion and the sign flip are one expression instead of two steps.
ELEVATION_MM_PER_M = 1000

# The one `lnd_type` code the reclassification is allowed to overwrite -- Open Water, per
# `structure/legends.csv`. See "Why the shallow classes are confined to water" in the module
# docstring for why this is enforced rather than assumed.
WATER_LAND_TYPE = 2

# The inundation-depth reclassification, as (upper bound in metres, code) in ascending
# order of bound, with `None` standing for "no upper bound". Each interval is open at its
# lower end and CLOSED at its upper one, with zero itself excluded by the last `case` arm
# below: `(0, 0.46] -> 6`, `(0.46, 1.0] -> 7`, `(1.0, inf) -> 2`.
#
# The deepest class reports the OPEN-WATER LAND TYPE rather than a code of its own, which
# is what `structure/legends.csv` publishes: `lnd_type_plus` legend values run 1-7, and `2`
# is labelled "Water - Deep". Deep water is not reclassified at all -- it is simply water,
# and only the two shallow classes say something `lnd_type` could not.
DEPTH_CLASSES: tuple[tuple[float | None, int], ...] = ((0.46, 6), (1.0, 7), (None, WATER_LAND_TYPE))


def _ordered_intersection(*sequences: list[str]) -> list[str]:
    """The sequences' INTERSECTION, in first-seen order -- the dimensions every operand
    named carries, which is what a pairing over a joined frame keys on
    (`execution.band_ops._pair_band_groups`). A dimension only one operand carries
    (`aggregate_type`, which `stg` declares and the two rasters do not) cannot survive that
    pairing, which is what `collapsed_dimensions` below reads this for.

    Order matters only in that it should not depend on set iteration: `collapsed_dimensions`
    is published, and the same request must always render the same answer.
    """
    shared = set.intersection(*(set(sequence) for sequence in sequences))
    return [name for name in sequences[0] if name in shared]


class LndTypePlus(Analysis):
    class Meta:
        name = "lnd-type-plus"
        label = "Land Type Plus"
        type = "custom"
        # `None`, not a variable list: this field says which dataset a `type="dataset"`
        # analysis is meaningful for, and a custom analysis has no caller-chosen dataset
        # to qualify -- its inputs are fixed in `pipeline()` below. Same reading as
        # `salinity-veg-investigation`'s.
        variables = None
        # `structure/legends.csv`'s own `lnd_type_plus` entry, which covers the full `1`-`7`
        # result domain. NOT `lnd_type`: its ramp stops at `5`, so both shallow classes
        # would render off the end of it. The two legends share their `1`-`5` colors, so a
        # pixel this analysis leaves alone draws the same as it does under plain `lnd_type`;
        # the entry's own `2` is relabelled "Water - Deep" because that is what a `2` here
        # means -- open water that either carries no depth or is deeper than 1.0 m.
        legend = "lnd_type_plus"
        result_kind = "category"
        kind = "variable"
        subsumed_dimensions = ["aggregate_type"]

    class Parameters(Analysis.Parameters):
        # Required, and for the same reason `salinity-veg-investigation`'s are: with no
        # dataset in the URL these arrive as the analysis's own parameters rather than as
        # dimension filters (see `api.__init__._resolve_custom_analysis`), and leaving
        # either unpinned would read every model group / scenario on disk into one
        # band-wide join, whose operand union is what `config.ANALYSIS_BAND_CEILING` is
        # checked against.
        model_group_id: int
        scenario_id: int
        # Optional: omitted means every year, one output band per year (the `case`'s
        # pairing keeps `calendar_year`), which is a legitimate multi-band request and is
        # also a large read -- three inputs' worth here. A single year is the common case.
        calendar_year: int | None = None

    @classmethod
    def validate_selection(cls, parameters: "LndTypePlus.Parameters", filters: dict[str, Any]) -> None:
        """Reject a selection outright rather than accepting one `pipeline()` silently
        drops -- the same contract `salinity-veg-investigation` has, and now for the same
        reason as well: with stage on `hydro_compartment_v002` and the other two inputs on
        `morph_pixel_v002`, a `grid_cell` names a cell in a lattice the request does not
        say which of.

        A caller-written `where` has the same problem in the value direction: it would
        have to be applied to one of three inputs, and nothing in the request says which.
        And a spatial predicate would additionally turn a band-wide frame long
        (`request.ops.FILTER_LOWERING_TABLE`), making the raster `output.format` this
        analysis declares invalid rather than merely narrowing the result. Both are
        rejected rather than half-supported; the dimension values this analysis does take
        are `Parameters`.
        """
        if filters:
            raise InvalidAnalysisParametersError(
                f"analysis 'lnd-type-plus' takes no selection, but got: "
                f"{', '.join(sorted(filters))} -- it names its own three inputs, across "
                f"two different grids, and joins them as raster surfaces, so a "
                f"grid_cell/where has no single frame to narrow and would change the "
                f"result's shape rather than its extent; its dimension values are "
                f"parameters (model_group_id, scenario_id, calendar_year) instead"
            )

    @property
    def collapsed_dimensions(self) -> frozenset[str]:
        """Stated here rather than derived, because `base.default_collapsed_dimensions`
        reads a trailing `group.by` and this pipeline ends on a `case`.

        The collapse is no less structural for that -- it is just carried by a different
        mechanism. A paired `case` reduces each pairing GROUP to one band, and the groups
        are keyed on the dimensions every operand shares
        (`execution.band_ops._pair_band_groups`), so a dimension only one input declares
        cannot survive it. Here that is exactly `aggregate_type`: `stg` requires it, neither
        raster has it, and this analysis pins it to `mean` itself, so a caller can neither
        see nor vary it -- which is what `collapsed_dimensions` exists to tell the
        visualization layer. The same intersection the old trailing `group.by` was built
        from, read for the same reason.
        """
        shared = set(
            _ordered_intersection(
                required_dataset_dimensions(ELEVATION_INPUT),
                required_dataset_dimensions(STAGE_INPUT),
                required_dataset_dimensions(LAND_TYPE_INPUT),
            )
        )
        declared = {
            dimension
            for dataset in (STAGE_INPUT, ELEVATION_INPUT, LAND_TYPE_INPUT)
            for dimension in required_dataset_dimensions(dataset)
        }
        return frozenset(declared - shared)

    def pipeline(self):
        assert self.dataset is None, "lnd-type-plus is a type='custom' analysis"
        p = self.parameters

        selection: dict[str, Any] = {
            "model_group_id": p.model_group_id,
            "scenario_id": p.scenario_id,
        }
        if p.calendar_year is not None:
            selection["calendar_year"] = p.calendar_year
        # The caller's selection, written ONCE, at the end -- see "Why the selection is
        # stated once, at the end" in the module docstring. `stage_where` is not a selection
        # at all: `aggregate_type` is this analysis's own definitional choice, and it has to
        # be applied on stage's own branch, because nothing downstream pairs or groups on a
        # dimension only one input declares. Left unfiltered, stage would bring one band per
        # aggregate type into a pairing that expects exactly one band per operand, and the
        # whole year would pair-miss and drop.
        selection_where = selection_filter_where(selection)
        stage_where = selection_filter_where({"aggregate_type": STAGE_AGGREGATE_TYPE})

        steps: list[dict] = [
            {"op": "filter", "input": "stage", "where": stage_where, "as": "selected_stage"},
            # Vector-to-raster, so no `aggregate`: a project ONTO a pixel grid broadcasts
            # each polygon's value to its pixels rather than reducing many into one
            # (`request.ops._type_project`'s `direction_takes_aggregate`).
            {"op": "project", "input": "selected_stage", "to": ELEVATION_INPUT.grid, "as": "stage_m"},
            # All three inputs at once -- a `join` takes N operands, not two
            # (`pipeline.JoinStep.inputs` is `min_length=2`), and a `case` pairs N of them
            # (`execution.band_ops._pair_band_groups` keys on the dimensions they share and
            # requires exactly one band per operand per group, for any N). `elevation` and
            # `land_type` are the INPUTS, read directly: nothing stands between either read
            # and the join, and the operand names are what the arms below read -- see "Why
            # the depth is computed inside the `case` arms" in the module docstring.
            {
                "op": "join",
                "type": "spatial",
                "inputs": ["stage_m", "elevation", "land_type"],
                "as": "paired",
            },
            # The caller's selection, and the only one this pipeline states. Declared past
            # the join, it is a dimension conjunct over a frame all three origins feed --
            # exactly the case `execution.plan`'s two dimension passes exist for, so it
            # narrows every origin's READ rather than only the frame it is written on. See
            # the module docstring. It sits before the `case` because a dimension filter
            # leaves the pairing standing and the `case` is what collapses it; nothing may
            # follow that collapse which would want the pair back.
            {"op": "filter", "where": selection_where, "as": "selected"},
        ]
        # The depth, its reclassification, the water restriction AND the land type showing
        # through, in one step -- see "Why the reclassification is one `case` step" in the
        # module docstring. `DEPTH_CLASSES` is stated ascending by bound, each interval open
        # at its lower end and closed at its upper one, so an arm's own lower bound is the
        # PREVIOUS class's upper one and the first class's is the domain floor, zero.
        # Reversed, those lower bounds become first-match-wins `>` tests. Nothing states the
        # upper ends: an earlier arm already claimed everything above them. Every arm
        # additionally carries the `land_type = 2` conjunct, which is the restriction stated
        # rather than assumed.
        #
        # Millimetres to metres happens here rather than in a step of its own: `elev`
        # is stored as NAVD88 millimetres and every arm needs the same metres-denominated
        # depth, so one conversion written into the shared `depth` string says it once for
        # all three without a step and an intermediate name to carry it.
        depth = f"stage_m - elevation / {ELEVATION_MM_PER_M}"
        lower_bounds: list[float] = [0, *(upper for upper, _ in DEPTH_CLASSES[:-1])]
        steps.append(
            {
                "op": "case",
                "cases": [
                    {"when": f"land_type = {WATER_LAND_TYPE} AND {depth} > {lower}", "then": str(code)}
                    for lower, (_, code) in reversed(list(zip(lower_bounds, DEPTH_CLASSES)))
                ],
                # Everything the arms do not claim is the land type, unchanged: dry land,
                # water standing at or below the ground, and any pixel missing stage or
                # elevation (a null term makes the depth null and a null predicate false).
                # A pixel whose LAND TYPE is nodata comes out nodata, because `land_type`
                # is then null on both sides -- the arms' conjunct and this `else`.
                "else": "land_type",
            }
        )

        data = {
            "inputs": {
                name: {"variable": ref.variable, "time_unit": ref.time_unit, "grid": ref.grid}
                for name, ref in (
                    ("stage", STAGE_INPUT),
                    ("elevation", ELEVATION_INPUT),
                    # Named `land_type` because the join above references it DIRECTLY --
                    # nothing stands between this read and the join -- and a join operand's
                    # name is the value column name the `land_type = 2` conjunct and the
                    # `else` both read.
                    ("land_type", LAND_TYPE_INPUT),
                )
            },
            "steps": steps,
            # Always raster: the final frame is band-wide on `morph_pixel_v002`, so a
            # tabular sink would be rejected by `request.ops.validate_shapes`. Only
            # `pipeline.json` ever shows this value -- the `data.{format}`/tile routes
            # overwrite it from their own URL segment -- but what `pipeline.json`
            # publishes has to still validate if a caller resubmits it verbatim.
            "output": {"format": "tif"},
        }
        return parse_pipeline(data)
