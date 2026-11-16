"""`lnd_conf_chng` (Confidence-Weighted Land Change) -- a dataset-scoped,
`kind="variable"` analysis over `lnd_type`: `lnd_conf`'s weighted land fraction in each
selected comparison year, minus the median member's (`scenario_id` 34's) land/water state
in the initial-conditions year:

    lnd_conf_chng = current weighted land fraction - S34 initial land (1) / water (0)

so a cell that starts as water reads its fraction directly (`0` to `1`, land gain) and one
that starts as land reads it less one (`-1` to `0`, land loss). `0` is "highest
confidence no change" from either side, `+1`/`-1` are every member agreeing on gain/loss,
and the further from `+-0.5` the more the members agree. The result is float32, the same
way and for the same reason `lnd_conf`'s is: it can only take the 17 subset sums of
`SCENARIO_WEIGHTS`, or those less one, all at least `0.005` apart.

The initial-conditions year is `lnd_chng.baseline_year` for the dataset's grid -- `2029`
for the MP2029 (`_v002`) grids the ensemble lives on -- and the comparison year(s) are the
caller's `calendar_year` selection. Everything else follows `lnd_chng`'s two-image shape:
two reads of the same dataset, the initial-year branch grouped without `calendar_year` so
its one value per grid cell broadcasts across every comparison year at the `join`, and the
same "band-wide output takes exactly one comparison year" rule (see `lnd_chng`'s module
docstring). The comparison branch is `lnd_conf.confidence_steps`; the initial branch is the
same land/water mapping over S34 alone, unweighted. Like `lnd_conf`, the analysis rejects a
caller `scenario_id`: both branches fix their own members.

A cell null on either side (S34 nodata in the initial year, or no member reporting in the
comparison year) stays null through the subtraction rather than reading as a change.

`legend = "lnd_conf_chng"` splits `0` into two adjacent transparent classes, `[-0.04, 0)` "stays
land" and `[0, 0.04)` "stays water", so the portal's layout can show each at the start of
its own strip (`legend_layouts.csv`). Which of the two an exact `0` lands in is arbitrary
-- both are transparent, so the tiles draw the same thing either way.
"""

from typing import Any

from cpra.mp.data.analyses.base import (
    Analysis,
    AnalysisMeta,
    InvalidAnalysisParametersError,
    dataset_input_selection,
    dataset_output_format,
    normalized_filter_values,
    required_dataset_dimensions,
    selection_filter_where,
)
from cpra.mp.data.analyses.lnd_chng import baseline_year
from cpra.mp.data.analyses.lnd_conf import (
    LAND_VALUES,
    WATER_VALUES,
    _value_in,
    confidence_steps,
    reject_scenario_selection,
)
from cpra.mp.data.analysis.request.pipeline import parse_pipeline
from cpra.mp.data.grids import GRIDS

# The ensemble member whose initial-year land/water state the comparison is measured from.
INITIAL_SCENARIO_ID = 34


def initial_steps(filters: dict[str, Any], grid_cell_where: str | None, group_by: list[str]) -> list[dict]:
    """S34's land (`1.0`) / water (`0.0`) state, ending on a `group` over `group_by` that
    folds away the single `scenario_id`. The same value filter and mapping as
    `lnd_conf.confidence_steps`, without the weights. The first step names no `input`; the
    caller supplies it."""
    filters = {**filters, "scenario_id": [INITIAL_SCENARIO_ID]}
    return [
        {"op": "filter", "where": selection_filter_where(filters, grid_cell_where=grid_cell_where)},
        {"op": "filter", "where": _value_in(LAND_VALUES + WATER_VALUES)},
        {
            "op": "case",
            "cases": [
                {"when": _value_in(LAND_VALUES), "then": "1.0"},
                {"when": _value_in(WATER_VALUES), "then": "0.0"},
            ],
        },
        {"op": "group", "by": group_by, "aggregate": "sum"},
    ]


class LandConfidenceChange(Analysis):
    Meta = AnalysisMeta(
        name="lnd_conf_chng",
        label="Confidence-Weighted Land Change",
        kind="variable",
        variables=["lnd_type"],
        legend="lnd_conf_chng",
        result_kind="value",
        subsumed_dimensions=["scenario_id"],
    )

    class Parameters(Analysis.Parameters):
        pass

    @classmethod
    def validate_selection(cls, parameters: "LandConfidenceChange.Parameters", filters: dict[str, Any]) -> None:
        reject_scenario_selection(cls.Meta.name, filters)

    @property
    def collapsed_dimensions(self) -> frozenset[str]:
        """Stated rather than derived: the pipeline ends on a `join` (then `calculate`s),
        which `base.default_collapsed_dimensions` reads as collapsing nothing. The ensemble is
        folded away; `calendar_year` is not -- the result varies over the comparison
        years the caller selected."""
        return frozenset({"scenario_id"})

    def pipeline(self):
        dataset = self.require_dataset()
        dim = "calendar_year"
        raw_years = self.filters.get(dim)
        years = normalized_filter_values(dim, raw_years) if raw_years is not None else []
        t0 = baseline_year(dataset.grid)

        # Band-wide, the join pairs one initial-year band against one comparison band; see
        # `lnd_chng.pipeline`, which has the same rule for the same reason.
        spatial = self.grid_cell_where is not None or bool(self.filters.get("grid_cell"))
        if len(years) != 1 and not spatial and GRIDS[dataset.grid].grid_type == "raster":
            got = f"got {len(years)}: {years!r}" if years else "none selected, which is all of them"
            raise InvalidAnalysisParametersError(
                f"{self.Meta.name} draws one image per comparison year and a raster grid "
                f"({dataset.grid!r}) holds one, so select exactly 1 calendar_year -- {got}. "
                "Several years are a series rather than an image; select a grid_cell to ask "
                "for one."
            )

        required = [d for d in required_dataset_dimensions(dataset) if d != "scenario_id"]
        # Dropping `calendar_year` from the initial-year `group` is what lets it broadcast
        # across every comparison year at the join.
        initial = initial_steps({**self.filters, dim: [t0]}, self.grid_cell_where, [d for d in required if d != dim])
        initial[0]["input"] = "initial_year"
        initial[-1]["as"] = "initial"
        current = confidence_steps({**self.filters, dim: years}, self.grid_cell_where, required)
        current[0]["input"] = "current_year"
        current[-1]["as"] = "current"

        steps = [
            *initial,
            *current,
            {"op": "join", "type": "spatial", "inputs": ["initial", "current"], "as": "paired"},
            {"op": "calculate", "expression": "current - initial"},
            # Float32 -- see the module docstring. `real`, not `float`: Polars' SQL reads
            # `float` as Float64.
            {"op": "calculate", "expression": "cast(value as real)"},
        ]

        data = {
            "inputs": {
                "initial_year": dataset_input_selection(dataset),
                "current_year": dataset_input_selection(dataset),
            },
            "steps": steps,
            "output": {"format": dataset_output_format(dataset)},
        }
        return parse_pipeline(data)
