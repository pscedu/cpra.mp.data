"""`lnd-type-plus` -- a `type="custom"` analysis that extends the `lnd_type` categorical
surface with two SHALLOW-INUNDATION classes the morphology model's land types do not
themselves distinguish. Annual inundation depth is COMPUTED here rather than read: mean
annual stage (`stg`/annual/`hydro_compartment_v002`, `aggregate_type=mean`) is projected
onto `morph_pixel_v002` and topobathymetric elevation (`elev`/annual/`morph_pixel_v002`,
stored in millimetres) is subtracted from it. That depth is reclassified to `6` (inundated,
no deeper than 0.46 m), `7` (deeper than 0.46 m, no deeper than 1.0 m) or `0` (deeper than
1.0 m), spatially joined against annual land type (`lnd_type`/annual/`morph_pixel_v002`),
and the two are reduced to the MAX of the pair at every pixel.

**Why a max, and why `0` for deep water.** `lnd_type`'s own codes are `1`-`5` (see
`structure/legends.csv`: vegetated wetland, open water, bare/unvegetated wetland,
upland/developed, flotant marsh), so the two new codes sit ABOVE all of them and a
`max` lets either shallow class override the land type the cell carries. The deep
class is `0`, which is below every real land-type code and is therefore the identity
element of that same `max`: at a cell inundated by a metre or more, the land type shows
through unchanged. One aggregate expresses both halves of the rule because the code
values were chosen to make it do so -- this analysis does not get to choose them, and
picking a different reduction would silently change what `6`/`7`/`0` mean.

**Why the shallow classes are confined to water.** The rule this analysis implements
subdivides OPEN WATER (`lnd_type = 2`) by depth; it does not reclassify land. That the two
should coincide -- a positive depth means the water level stands above the ground, which is
water -- is a property of the two input surfaces agreeing, not something the `max` enforces:
`max` takes `6` over any land-type code, so a single pixel where the elevation and land-type
surfaces disagree (independently modelled, and at the coastline they do) would silently
report shallow inundation over marsh or upland. So the restriction is stated rather than
assumed. `depth_class` is joined against `land_type` once and masked by `land_type = 2`
before the real join -- the same paired-filter idiom as the presence guard above, collapsing
to the first-named operand (`depth_class`) masked wherever the land type underneath is
anything else. A masked depth contributes nothing to the `max`, so a non-water pixel reads
as its own land type no matter how deep the depth surface says it is.

One consequence worth naming: `0` can no longer reach the output. It survives the mask only
over water, where `max(0, 2)` is `2` -- so deep water reads as plain open water and the
result's domain is `1`-`7`, not `0`-`7`. The deep class is still what keeps deep water OUT
of `6`/`7` rather than a code that appears in its own right -- which is why the
`lnd_type_plus` legend this analysis declares labels `2` "Water - Deep" and has no entry
for `0` at all.

**Why the depth is a `join` + `group sum` rather than one subtraction.** A `calculate`
over a paired band-wide frame is rejected outright (a band-wide frame's bands are
positional, so an expression has no per-operand column to name -- see
`request.ops._type_calculate`), so `stage - elevation` cannot be written as an expression
over the joined pair. The subtraction is instead the shape `difference` uses: NEGATE one
operand with a `calculate` on its own branch (`value / -1000`, which converts millimetres
to metres and flips the sign in one pass), then `join` and reduce with `{"aggregate":
"sum"}`, whose band-wide lowering (`execution.band_ops.group_band_wide`) is a horizontal
add across each group's bands. The same lowering, with `max`, is what the trailing group
does for the land-type pairing.

**The presence guard before that sum.** A horizontal `sum` SKIPS nulls rather than
propagating them -- it emits its `no_contributor` null only where EVERY contributing band
is null. Left alone, a pixel carrying stage but no elevation would sum to the bare stage
(a NAVD88 water level, typically a few tenths of a metre) and reclassify as a real shallow
class, painting inundation across whatever the elevation surface does not cover. So the
two operands are joined once before the arithmetic and filtered on `IS NOT NULL` on both
sides: a value filter over a paired band-wide frame pairs each operand's band with its
counterpart and masks the pair (`execution.band_ops.pair_and_mask_band_wide`), collapsing
to the FIRST-named operand's value -- stage, here, masked wherever elevation is missing.
That masked stage is what the real join then adds the negated elevation to, so a pixel
missing either input contributes nothing and stays nodata.

**Why the reclassification is four `calculate` steps and not three.** A `calculate`
rewrites the RUNNING value, so a later entry's `where` sees whatever an earlier one
already wrote -- and the three target codes are not disjoint from the three source
domains in any order: `6` and `7` both fall in the `> 1.0` domain, and `0` falls in the
`<= 0.46` one. So the two positive classes are first written as their own NEGATIVES
(`-6`/`-7`), which no later rule's domain can match, and a fourth step flips them back in
one pass. What makes the negative encoding safe is the leading `value > 0` filter: it
masks every non-positive depth -- an elevation standing at or above the water level, which
is dry land rather than any inundation class -- before any rewrite happens, so a negative
value at step four can only ever be one this analysis itself wrote.

A cell masked out that way contributes nothing to the `max` rather than a `0`, so a dry
cell reads as its land type alone, and a cell with no depth and no land type stays nodata
rather than becoming `0`. Nothing here filters `lnd_type` to its `1`-`5` value domain:
every variable declares a nodata sentinel (`variables.py`'s `Variable.nodata`, `-9999`
unless `variables.csv` names another -- `elev`'s `-9999000` is the one exception, its own
sentinel in millimetres), `raster.write_cog` writes it into the COG profile, and the read
path masks it to null before any step here sees it.
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
# the conversion and the sign flip are one `calculate` instead of two.
ELEVATION_MM_PER_M = 1000

# The one `lnd_type` code the reclassification is allowed to overwrite -- Open Water, per
# `structure/legends.csv`. See "Why the shallow classes are confined to water" in the module
# docstring for why this is enforced rather than assumed.
WATER_LAND_TYPE = 2

# The inundation-depth reclassification, as (upper bound in metres, code) in ascending
# order of bound, with `None` standing for "no upper bound". Each interval is open at its
# lower end and CLOSED at its upper one, with zero itself excluded by the domain filter
# below: `(0, 0.46] -> 6`, `(0.46, 1.0] -> 7`, `(1.0, inf) -> 0`.
DEPTH_CLASSES: tuple[tuple[float | None, int], ...] = ((0.46, 6), (1.0, 7), (None, 0))


def _ordered_intersection(*sequences: list[str]) -> list[str]:
    """The sequences' INTERSECTION, in first-seen order -- the dimensions every operand
    named carries, which is what a `group.by` over a joined frame has to be. A dimension
    only one operand carries (`aggregate_type`, which `stg` declares and the two rasters
    do not) would split the pair into two one-band groups that never reduce together.

    Order matters only in that it should not depend on set iteration: the published
    `pipeline.json` is a contract a caller may resubmit verbatim, so the same request must
    always render the same `by`.
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

    class Parameters(Analysis.Parameters):
        # Required, and for the same reason `salinity-veg-investigation`'s are: with no
        # dataset in the URL these arrive as the analysis's own parameters rather than as
        # dimension filters (see `api.__init__._resolve_custom_analysis`), and leaving
        # either unpinned would read every model group / scenario on disk into one
        # band-wide join, whose operand union is what `config.ANALYSIS_BAND_CEILING` is
        # checked against.
        model_group_id: int
        scenario_id: int
        # Optional: omitted means every year, one output band per year (the `group.by`
        # below keeps `calendar_year`), which is a legitimate multi-band request and is
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

    def pipeline(self):
        assert self.dataset is None, "lnd-type-plus is a type='custom' analysis"
        p = self.parameters

        selection: dict[str, Any] = {
            "model_group_id": p.model_group_id,
            "scenario_id": p.scenario_id,
        }
        if p.calendar_year is not None:
            selection["calendar_year"] = p.calendar_year
        # The two raster inputs carry the same dimensions and take the same selection, so
        # one rendering serves both -- `selection_filter_where` sorts its conjuncts, so the
        # two `where`s are identical strings rather than merely equivalent ones. Stage adds
        # the one dimension only it declares.
        selection_where = selection_filter_where(selection)
        stage_where = selection_filter_where(
            {**selection, "aggregate_type": STAGE_AGGREGATE_TYPE}
        )

        # Every operand of both joins is grouped by the dimensions all three inputs share;
        # `aggregate_type` is deliberately not among them (see `_ordered_intersection`).
        group_by = _ordered_intersection(
            required_dataset_dimensions(ELEVATION_INPUT),
            required_dataset_dimensions(STAGE_INPUT),
            required_dataset_dimensions(LAND_TYPE_INPUT),
        )

        steps: list[dict] = [
            {"op": "filter", "input": "stage", "where": stage_where, "as": "selected_stage"},
            # Vector-to-raster, so no `aggregate`: a project ONTO a pixel grid broadcasts
            # each polygon's value to its pixels rather than reducing many into one
            # (`request.ops._type_project`'s `direction_takes_aggregate`).
            {"op": "project", "input": "selected_stage", "to": ELEVATION_INPUT.grid, "as": "stage_m"},
            {"op": "filter", "input": "elevation", "where": selection_where, "as": "selected_elevation"},
            # Millimetres to metres AND the sign flip the sum below needs, in one pass.
            {
                "op": "calculate",
                "expression": f"value / -{ELEVATION_MM_PER_M}",
                "as": "neg_elevation_m",
            },
            # The presence guard -- see the module docstring. Collapses to `stage_m`'s own
            # value (the first-named operand), masked wherever either side is missing.
            {
                "op": "join",
                "type": "spatial",
                "inputs": ["stage_m", "neg_elevation_m"],
                "as": "paired_inputs",
            },
            {
                "op": "filter",
                "where": "stage_m IS NOT NULL AND neg_elevation_m IS NOT NULL",
                "as": "stage_present",
            },
            # The subtraction itself: stage + (-elevation), as a horizontal add.
            {
                "op": "join",
                "type": "spatial",
                "inputs": ["stage_present", "neg_elevation_m"],
                "as": "paired_depth",
            },
            {"op": "group", "by": group_by, "aggregate": "sum"},
            # The running value is now inundation depth in metres. Mask everything at or
            # below zero -- elevation standing at or above the water level, which is dry
            # land rather than any inundation class -- both because no class covers it and
            # because the negative encoding below depends on it. A band-wide value filter
            # masks the pixel rather than dropping its row, so a masked pixel stays null
            # through every `calculate` (a null never matches a `where`) and contributes
            # nothing to the trailing `max`.
            {"op": "filter", "where": "value > 0"},
        ]
        # Each bound is only a `<=` test: the previous class's bound is the lower one and
        # is EXCLUSIVE, and the `value > 0` filter above supplies the first. Written as the
        # NEGATED code for the two positive classes so no later entry's domain can re-match
        # a cell an earlier one already claimed -- see the module docstring.
        lower: float | None = None
        for upper, code in DEPTH_CLASSES:
            conjuncts = []
            if lower is not None:
                conjuncts.append(f"value > {lower}")
            if upper is not None:
                conjuncts.append(f"value <= {upper}")
            steps.append(
                {
                    "op": "calculate",
                    "expression": str(-code if code > 0 else code),
                    "where": " AND ".join(conjuncts),
                }
            )
            lower = upper
        # Flip the encoded classes back. `0` (the deep class) is untouched, and no depth
        # can be negative here, so this matches exactly the cells the two entries above
        # wrote.
        steps.append(
            {"op": "calculate", "expression": "value * -1", "where": "value < 0", "as": "depth_class"}
        )

        steps.extend(
            [
                {"op": "filter", "input": "land", "where": selection_where, "as": "land_type"},
                # The water restriction -- the same paired-filter idiom as the presence
                # guard above, and the one place either join's operand value columns are
                # named. Collapses to `depth_class` (the first-named operand), masked
                # wherever the land type underneath is anything but open water.
                {
                    "op": "join",
                    "type": "spatial",
                    "inputs": ["depth_class", "land_type"],
                    "as": "paired_water",
                },
                {
                    "op": "filter",
                    "where": f"land_type = {WATER_LAND_TYPE}",
                    "as": "water_depth_class",
                },
                # The operands' own names become the joined frame's value column names
                # (`ops._type_join`), but nothing downstream reads them: the `group` below
                # collapses the pairing without naming either, which is the one way a
                # paired BAND-WIDE frame can be collapsed other than a value filter.
                {
                    "op": "join",
                    "type": "spatial",
                    "inputs": ["water_depth_class", "land_type"],
                    "as": "paired",
                },
                # Band-wide, this groups the joined frame's band METADATA and reduces each
                # group's bands horizontally at every pixel: one group per (model group,
                # scenario, year), holding that year's reclassified depth band and its
                # land-type counterpart, reduced to their max. Grouping by every dimension
                # the inputs share is what keeps the pairing one-year-to-one-year; the
                # axis being collapsed is the operand axis, which is not a dimension.
                {"op": "group", "by": group_by, "aggregate": "max"},
            ]
        )

        data = {
            "inputs": {
                name: {"variable": ref.variable, "time_unit": ref.time_unit, "grid": ref.grid}
                for name, ref in (
                    ("stage", STAGE_INPUT),
                    ("elevation", ELEVATION_INPUT),
                    ("land", LAND_TYPE_INPUT),
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
