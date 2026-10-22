"""`lnd_conf` -- a dataset-scoped analysis producing a confidence-weighted
land/water surface from the `lnd_type` categorical variable: each of the five ensemble
members (`scenario_id` 34-38) is reduced to a land (`1.0`, codes `1`/`3`/`4`/`5` -- flotant
marsh counts as land) / water (`0.0`, code `2`) indicator, then
the five are combined as a weighted sum whose weights (`SCENARIO_WEIGHTS`) sum to exactly
`1.0`. The result is therefore the weighted FRACTION of the ensemble calling a cell land:
`0.0` = every member says water, `1.0` = every member says land, anything between = the
weight of the members that disagree.

The mapping and the member set are fixed by the analysis rather than exposed as
parameters -- the weights are only a weighted mean over the full five-member ensemble, so
a caller selecting a subset would silently get a partial sum that reads like a confidence
but isn't. `validate_selection` rejects a caller-supplied `scenario_id` filter for that
reason; every other dimension filter (`model_group_id`, `calendar_year`, `grid_cell`,
`where`) is the caller's, and composes into the leading `filter` step exactly as it does
for any other dataset-scoped analysis.

The land/water mapping is one `case` step, whose two arms are read against the INCOMING
code rather than the running value -- so the mapping states itself directly, with no
question of one branch re-matching a cell the other already claimed. The leading
value-domain `filter` drops anything outside `1`-`5` (nodata included) so an unmapped code
can never reach the weighted sum as itself; the `case` has no `else` for the same reason,
and the two agree on which codes mean anything here.

The weighting is a second `case`, one arm per member, each arm's `when` naming only that
member's `scenario_id`. A dimension-only `when` selects which bands (or rows) the arm
applies to, so the five arms partition the frame the way `SCENARIO_WEIGHTS` does and the
whole ensemble is scaled in one step; it too has no `else`, since the leading `filter`
pinned `scenario_id` to exactly these five and there is no sixth member to pass through.

A cell that is nodata in only SOME members is summed over the members it has, so its
weights no longer total `1.0` and its confidence reads low rather than being rescaled to
the members that actually reported. That is the arithmetic `group`/`sum` gives; a
mean-over-valid-members would need a matching count, which this shape has no way to
carry.

`legend = "lnd_conf"` colorizes that fraction directly: five even bands over `[0, 1]`,
diverging about the `0.4`-`0.6` band that straddles an evenly-split ensemble, water-colored
below it and land-colored above, under a transparent `No Data` band for everything below
`0.0`. Nothing this analysis outputs is below `0.0` -- the fraction's domain starts there --
so that band catches exactly one thing: the nodata sentinel
(`execution.sink.DEFAULT_NODATA`) a null pixel is filled with before the tile warp.
`Legend.colorize_pixels` digitizes every pixel it is given, nodata included, so a legend
with no under-band paints the whole out-of-domain area in its FIRST color; here that was
the darkest water color, which reads as a confident all-water answer where there is no
answer at all.

A DIFFERENCE of this analysis (`lnd_conf,diff`) has no named legend: the fraction's
legend classifies absolute values on `[0, 1]`, not a signed change in them, so
`analyses.resolve_analysis_chain` drops it for a differenced `result_kind="value"` result and
`visualizations.map` draws the synthesized red-white-blue ramp off `lnd_type`'s
`difference_legend_limit` (`1`), exactly as it does for any other differenced variable.

`analyses/lnd_conf_chng.py` reuses this analysis's steps (`confidence_steps`) to classify
the fraction in two years and pair the classes into a transition code.
"""

from typing import Any

from cpra.mp.data.analyses.base import (
    Analysis,
    AnalysisMeta,
    InvalidAnalysisParametersError,
    dataset_input_selection,
    dataset_output_format,
    format_dimension_literal,
    required_dataset_dimensions,
    selection_filter_where,
)
from cpra.mp.data.analysis.request.pipeline import parse_pipeline

# The ensemble members this analysis combines, and each one's weight. The weights sum to
# exactly 1.0, which is what makes the trailing `group`/`sum` a weighted MEAN of the
# members' land/water indicators rather than an unnormalized total. These are the MP29
# confidence weighting factors, by ESLR percentile: 36 (5th) and 35 (95th) 0.085, 37 (17th)
# and 38 (83rd) 0.25, 34 (50th) 0.33.
SCENARIO_WEIGHTS = {34: 0.33, 35: 0.085, 36: 0.085, 37: 0.25, 38: 0.25}

# `lnd_type` codes that count as land, and as water. Everything else -- including nodata --
# is dropped rather than mapped, since it has no land/water reading to contribute. Flotant
# marsh (`5`) is land.
LAND_VALUES = (1, 3, 4, 5)
WATER_VALUES = (2,)


def _value_in(values: tuple[int, ...]) -> str:
    """`value` matching any of `values` (ascending), as a disjunction. Written out as
    `value = a OR value = b ...` rather than `value IN (a, b, ...)`: a VALUE expression's
    function-call check reads the `IN (` as a call to an unknown function named `in` and
    rejects it, while the same form is accepted in a DIMENSION conjunct -- so the
    disjunction is the portable spelling for a condition over `value` itself.
    """
    return " OR ".join(f"value = {v}" for v in sorted(values))


def confidence_steps(filters: dict[str, Any], grid_cell_where: str | None, group_by: list[str]) -> list[dict]:
    """The steps that reduce a `lnd_type` selection to the weighted land fraction, ending
    on the `group` over `group_by` (which must not include `scenario_id`). The first step
    names no `input`; the caller supplies it. Shared with `lnd_conf_chng`, which runs it
    once per year it compares.
    """
    # Fold the fixed member set in as though the caller had selected it, so it renders
    # through the same sorted-conjunct assembly every other filter does rather than
    # being appended out of order. `validate_selection` has already rejected a
    # caller-supplied `scenario_id`, so nothing is being overwritten here -- and the fixed
    # conjunct alone guarantees `selection_filter_where` returns a string.
    filters = {**filters, "scenario_id": sorted(SCENARIO_WEIGHTS)}

    return [
        {"op": "filter", "where": selection_filter_where(filters, grid_cell_where=grid_cell_where)},
        {"op": "filter", "where": _value_in(LAND_VALUES + WATER_VALUES)},
        # The land/water indicator, in one pass over the incoming code -- see the
        # module docstring. No `else`: the filter above has already dropped everything
        # neither arm names.
        {
            "op": "case",
            "cases": [
                {"when": _value_in(LAND_VALUES), "then": "1.0"},
                {"when": _value_in(WATER_VALUES), "then": "0.0"},
            ],
        },
        # The ensemble weights, one arm per member: each arm's `when` is purely dimensional,
        # so it picks out that member's bands (or rows) and scales all of them. One `case`
        # rather than N successive rewrites because the members partition the frame -- the
        # leading `filter` pinned `scenario_id` to exactly `SCENARIO_WEIGHTS`' keys -- and
        # arms say that directly, where a chain would only be correct because each step's
        # `where` happened to miss what the ones before it had already written. No `else`
        # for the same reason: nothing reaches it, and writing a pass-through would be
        # claiming there is an unweighted member.
        {
            "op": "case",
            "cases": [
                {
                    "when": f"scenario_id = {format_dimension_literal('scenario_id', scenario_id)}",
                    "then": f"value * {weight}",
                }
                for scenario_id, weight in sorted(SCENARIO_WEIGHTS.items())
            ],
        },
        {"op": "group", "by": group_by, "aggregate": "sum"},
    ]


def reject_scenario_selection(name: str, filters: dict[str, Any]) -> None:
    """400 a caller-supplied `scenario_id`: the weights are a mean over the full ensemble
    only, so a subset would be a partial sum that reads like a confidence but isn't."""
    if filters.get("scenario_id") is not None:
        raise InvalidAnalysisParametersError(
            f"{name} combines a fixed ensemble "
            f"(scenario_id {', '.join(str(s) for s in sorted(SCENARIO_WEIGHTS))}) whose "
            "weights sum to 1.0, so it cannot be scoped to a caller-selected "
            f"scenario_id (got {filters['scenario_id']!r})"
        )


class LandConfidence(Analysis):
    Meta = AnalysisMeta(
        name="lnd_conf",
        label="Land Confidence",
        variables=["lnd_type"],
        legend="lnd_conf",
        result_kind="value",
        kind="variable",
        subsumed_dimensions=["scenario_id"],
    )

    class Parameters(Analysis.Parameters):
        pass

    @classmethod
    def validate_selection(cls, parameters: "LandConfidence.Parameters", filters: dict[str, Any]) -> None:
        reject_scenario_selection(cls.Meta.name, filters)

    def pipeline(self):
        dataset = self.require_dataset()
        group_by = [d for d in required_dataset_dimensions(dataset) if d != "scenario_id"]
        steps = confidence_steps(self.filters, self.grid_cell_where, group_by)
        # A pipeline's first step must name its input explicitly; every later step chains
        # from the one before it.
        steps[0]["input"] = "x"

        data = {
            "inputs": {"x": dataset_input_selection(self.dataset)},
            "steps": steps,
            # Derived from the dataset's grid kind: this analysis never changes shape, and
            # only `pipeline.json` ever shows the value, where it has to be a sink the
            # published pipeline would actually run under. `lnd_type` lives only on
            # `morph_pixel` (a raster grid), so in practice that is always `tif`.
            "output": {"format": dataset_output_format(self.dataset)},
        }
        return parse_pipeline(data)

    @property
    def title_suffix(self) -> str:
        return " Confidence"
