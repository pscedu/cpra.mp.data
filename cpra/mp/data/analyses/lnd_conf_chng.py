"""`lnd_conf_chng` -- a dataset-scoped, `kind="variable"` analysis producing a categorical
land-confidence TRANSITION code from `lnd_type`: `lnd_conf`'s weighted land fraction is
computed for the initial-conditions year and for each selected comparison year, each is
classified into one of four confidence classes, and the two class digits are concatenated
(`initial * 10 + current`), so `14` reads "every member water in the initial year, every
member land now".

| class | `lnd_conf` fraction |
|-------|---------------------|
| `1`   | `0`                 |
| `2`   | `> 0` to `< 0.5`    |
| `3`   | `0.5` to `< 1`      |
| `4`   | `1`                 |

The initial-conditions year is `lnd_chng.baseline_year` for the dataset's grid -- `2029`
for the MP2029 (`_v002`) grids the `lnd_conf` ensemble lives on -- and the comparison
year(s) are the caller's `calendar_year` selection. Everything else follows `lnd_chng`'s
two-image shape: two reads of the same dataset, the initial-year branch grouped without
`calendar_year` so its one value per grid cell broadcasts across every comparison year at
the `join`, and the same "band-wide output takes exactly one comparison year" rule (see
`lnd_chng`'s module docstring). Each branch is `lnd_conf.confidence_steps` followed by the
classifying `case`, and like `lnd_conf` the analysis rejects a caller `scenario_id`.

**The class boundaries carry a tolerance (`_EPSILON`).** The fraction is a float sum of
the five weights, and depending on summation order `0.33 + 0.085 + 0.085` -- or all five
-- need not land exactly on `0.5` or `1.0`. Every fraction the ensemble can produce is a
subset sum of `SCENARIO_WEIGHTS`, and none other than `0`, `0.5`, and `1` comes closer
than `0.08` to a boundary, so comparing against `boundary - _EPSILON` classifies exactly the
values the table above names and nothing else.

The classifying `case` has no `else`: a null fraction (no member reported) matches no arm
and stays null, rather than falling into a class.
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
)
from cpra.mp.data.analyses.lnd_chng import baseline_year
from cpra.mp.data.analyses.lnd_conf import confidence_steps, reject_scenario_selection
from cpra.mp.data.analysis.request.pipeline import parse_pipeline
from cpra.mp.data.grids import GRIDS

# Float slack at each class boundary -- see the module docstring.
_EPSILON = 0.001

# The confidence classes, as ordered `case` arms (first match wins, so each arm only needs
# its upper bound). Fixed-point formatting keeps the literals out of scientific notation.
CONFIDENCE_CLASSES = [
    (f"value < {_EPSILON:.3f}", "1"),
    (f"value < {0.5 - _EPSILON:.3f}", "2"),
    (f"value < {1 - _EPSILON:.3f}", "3"),
    (f"value >= {1 - _EPSILON:.3f}", "4"),
]


def _classify(name: str) -> dict:
    return {
        "op": "case",
        "cases": [{"when": when, "then": then} for when, then in CONFIDENCE_CLASSES],
        "as": name,
    }


class LandConfidenceChange(Analysis):
    Meta = AnalysisMeta(
        name="lnd_conf_chng",
        label="Land Confidence Change",
        kind="variable",
        variables=["lnd_type"],
        legend="lnd_conf_chng",
        result_kind="category",
        subsumed_dimensions=["scenario_id"],
    )

    class Parameters(Analysis.Parameters):
        pass

    @classmethod
    def validate_selection(cls, parameters: "LandConfidenceChange.Parameters", filters: dict[str, Any]) -> None:
        reject_scenario_selection(cls.Meta.name, filters)

    @property
    def collapsed_dimensions(self) -> frozenset[str]:
        """Stated rather than derived: the pipeline ends on a `calculate`, which
        `base.default_collapsed_dimensions` reads as collapsing nothing. The ensemble is
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
                f"lnd_conf_chng draws one image per comparison year and a raster grid "
                f"({dataset.grid!r}) holds one, so select exactly 1 calendar_year -- {got}. "
                "Several years are a series rather than an image; select a grid_cell to ask "
                "for one."
            )

        required = [d for d in required_dataset_dimensions(dataset) if d != "scenario_id"]
        # Dropping `calendar_year` from the initial-year `group` is what lets it broadcast
        # across every comparison year at the join.
        initial = confidence_steps({**self.filters, dim: [t0]}, self.grid_cell_where, [d for d in required if d != dim])
        initial[0]["input"] = "initial_year"
        current = confidence_steps({**self.filters, dim: years}, self.grid_cell_where, required)
        current[0]["input"] = "current_year"

        steps = [
            *initial,
            _classify("initial"),
            *current,
            _classify("current"),
            {"op": "join", "type": "spatial", "inputs": ["initial", "current"], "as": "paired"},
            # A null class on either side carries through as null: no transition to report.
            {"op": "calculate", "expression": "initial * 10 + current"},
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

    @property
    def title_suffix(self) -> str:
        return " Confidence Change"
