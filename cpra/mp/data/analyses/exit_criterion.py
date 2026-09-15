"""`exit-criterion` -- the one `type="custom"` v1 entry: the earliest calendar year at
which marsh coverage (`ffibs_coverage_ratio`/annual/`veg_grid_cell_v001`, `ffibs_type=
pl_fm`) and paired salinity (`sal`/annual/`hydro_compartment_v001`,
`aggregate_type=2_wk_max`) both cross their thresholds, at every vegetation-grid pixel --
the paired (project + spatial join) pipeline shape documented at
`docs/data-analysis-api.md`'s "The Exit-Criterion Query" example, landed here as a named
analysis rather than requiring a caller to hand-author that exact JSON.

Lands FIRST among the three v1 entries (plan §8): it is the only one exercising the
`type="custom"` path, multiple named inputs, and a `project` + spatial `join`, and has no
existing consumer (unlike `difference`/`concatenate`, which replace real portal catalog
entries).
"""

from pydantic import Field

from cpra.mp.data.analyses.base import Analysis
from cpra.mp.data.analysis.pipeline import parse_pipeline


class ExitCriterion(Analysis):
    class Meta:
        name = "exit-criterion"
        label = "Exit Criterion"
        type = "custom"
        variables = None
        legend = None
        result_kind = "value"

    class Parameters(Analysis.Parameters):
        model_group_id: int
        scenario_id: int
        # Bounded, per the task's own callout ("be careful about quoting/validation
        # there, it is the point of this entry"): both thresholds are interpolated
        # directly into a pipeline `filter.where` string below. Pydantic's `float` +
        # `Field(ge=..., le=...)` guarantees a real, range-bound Python float reaches
        # that f-string -- never a caller-supplied string, so there is no SQL-injection
        # surface here to escape, only a range to bound. `marsh_threshold` is a coverage
        # RATIO (see `ffibs_coverage_ratio`'s own name), so it is bounded to [0, 1];
        # `salinity_threshold` has no natural upper bound, only a lower one (a negative
        # salinity threshold is meaningless).
        marsh_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
        salinity_threshold: float = Field(default=5.5, ge=0.0)

    def pipeline(self):
        assert self.dataset is None, "exit-criterion is a type='custom' analysis"
        p = self.parameters
        where = f"marsh > {p.marsh_threshold} AND sal > {p.salinity_threshold}"
        data = {
            "inputs": {
                "marsh": {
                    "variable": "ffibs_coverage_ratio",
                    "time_unit": "annual",
                    "grid": "veg_grid_cell_v001",
                    "model_group_id": p.model_group_id,
                    "scenario_id": p.scenario_id,
                    "ffibs_type": "pl_fm",
                },
                "salinity": {
                    "variable": "sal",
                    "time_unit": "annual",
                    "grid": "hydro_compartment_v001",
                    "model_group_id": p.model_group_id,
                    "scenario_id": p.scenario_id,
                    "aggregate_type": "2_wk_max",
                },
            },
            "steps": [
                {"op": "project", "input": "salinity", "to": "veg_grid_cell_v001", "as": "sal"},
                {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
                {"op": "filter", "where": where},
                {"op": "group", "by": [], "aggregate": "min", "of": "calendar_year"},
            ],
            "output": {"format": "tif"},
        }
        return parse_pipeline(data)
