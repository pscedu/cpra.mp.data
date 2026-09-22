"""`land-type-confidence` -- a dataset-scoped analysis producing a confidence-weighted
land/water surface from the `lnd_type` categorical variable: each of the five ensemble
members (`scenario_id` 34-38) is reduced to a land (`1.0`) / water (`0.0`) indicator, then
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

The land/water mapping is expressed as two `calculate` steps rather than one conditional
expression, because a `calculate` expression is a pure value expression -- the branch
belongs in its `where`. The two are safe in either order: once `1/3/4` have become `1.0`,
no remapped cell equals `2` or `5`, so the water branch cannot re-match a cell the land
branch already claimed. The leading value-domain `filter` drops anything outside
`1`-`5` (nodata included) so an unmapped code can never reach the weighted sum as itself.

A cell that is nodata in only SOME members is summed over the members it has, so its
weights no longer total `1.0` and its confidence reads low rather than being rescaled to
the members that actually reported. That is the arithmetic `group`/`sum` gives; a
mean-over-valid-members would need a matching count, which this shape has no way to
carry.
"""

from typing import Any

from cpra.mp.data.analyses.base import (
    Analysis,
    InvalidAnalysisParametersError,
    dataset_input_selection,
    format_dimension_literal,
    required_dataset_dimensions,
    selection_filter_where,
)
from cpra.mp.data.analysis.request.pipeline import parse_pipeline

# The ensemble members this analysis combines, and each one's weight. The weights sum to
# exactly 1.0, which is what makes the trailing `group`/`sum` a weighted MEAN of the
# members' land/water indicators rather than an unnormalized total.
SCENARIO_WEIGHTS = {34: 0.085, 35: 0.25, 36: 0.33, 37: 0.25, 38: 0.085}

# `lnd_type` codes that count as land, and as water. Everything else -- including nodata --
# is dropped rather than mapped, since it has no land/water reading to contribute.
LAND_VALUES = (1, 3, 4)
WATER_VALUES = (2, 5)


def _value_in(values: tuple[int, ...]) -> str:
    """`value` matching any of `values` (ascending), as a disjunction. Written out as
    `value = a OR value = b ...` rather than `value IN (a, b, ...)`: a VALUE expression's
    function-call check reads the `IN (` as a call to an unknown function named `in` and
    rejects it, while the same form is accepted in a DIMENSION conjunct -- so the
    disjunction is the portable spelling for a condition over `value` itself.
    """
    return " OR ".join(f"value = {v}" for v in sorted(values))


class LandTypeConfidence(Analysis):
    class Meta:
        name = "land-type-confidence"
        label = "Land Type Confidence"
        type = "dataset"
        variables = ["lnd_type"]
        legend = None
        result_kind = "value"

    class Parameters(Analysis.Parameters):
        pass

    @classmethod
    def validate_selection(
        cls, parameters: "LandTypeConfidence.Parameters", filters: dict[str, Any]
    ) -> None:
        if filters.get("scenario_id") is not None:
            raise InvalidAnalysisParametersError(
                "land-type-confidence combines a fixed ensemble "
                f"(scenario_id {', '.join(str(s) for s in sorted(SCENARIO_WEIGHTS))}) whose "
                "weights sum to 1.0, so it cannot be scoped to a caller-selected "
                f"scenario_id (got {filters['scenario_id']!r})"
            )

    def pipeline(self):
        assert self.dataset is not None, "land-type-confidence is a type='dataset' analysis"
        required = required_dataset_dimensions(self.dataset)
        group_by = [d for d in required if d != "scenario_id"]

        # Fold the fixed member set in as though the caller had selected it, so it renders
        # through the same sorted-conjunct assembly every other filter does rather than
        # being appended out of order. `validate_selection` has already rejected a
        # caller-supplied `scenario_id`, so nothing is being overwritten here.
        filters = {**self.filters, "scenario_id": sorted(SCENARIO_WEIGHTS)}

        steps: list[dict] = [
            {"op": "filter", "where": selection_filter_where(filters, grid_cell_where=self.grid_cell_where)},
            {"op": "filter", "where": _value_in(LAND_VALUES + WATER_VALUES)},
            {"op": "calculate", "expression": "1.0", "where": _value_in(LAND_VALUES)},
            {"op": "calculate", "expression": "0.0", "where": _value_in(WATER_VALUES)},
        ]
        steps.extend(
            {
                "op": "calculate",
                "expression": f"value * {weight}",
                "where": f"scenario_id = {format_dimension_literal('scenario_id', scenario_id)}",
            }
            for scenario_id, weight in sorted(SCENARIO_WEIGHTS.items())
        )
        steps.append({"op": "group", "by": group_by, "aggregate": "sum"})
        # A pipeline's first step must name its input explicitly; every later step chains
        # from the one before it. The leading `filter` is always present here -- the fixed
        # `scenario_id` conjunct alone guarantees `selection_filter_where` returns a string.
        steps[0]["input"] = "x"

        data = {
            "inputs": {"x": dataset_input_selection(self.dataset)},
            "steps": steps,
            # `Output.format` is required, and every `data.{format}`/tile route overwrites
            # it from its own URL segment -- so the value set here is only ever observable
            # on `pipeline.json`, which has no format segment to override it with. It has
            # to be a format the published pipeline would actually RUN under, or
            # `pipeline.json` hands a caller something `validate_shapes` rejects on
            # resubmission. `lnd_type` lives only on `morph_pixel` (a raster grid), so a
            # valid selection for this analysis always ends band-wide, and `tif` is the
            # only tabular-vs-raster sink that fits it.
            "output": {"format": "tif"},
        }
        return parse_pipeline(data)

    @property
    def title_suffix(self) -> str:
        return " Confidence"
