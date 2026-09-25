"""`salinity-veg-investigation` -- the one `type="custom"` v1 entry: the earliest calendar
year at which marsh coverage (`ffibs_coverage_ratio`/annual/`veg_grid_cell_v001`,
`ffibs_type=pl_fm`) and paired salinity (`sal`/annual/`hydro_compartment_v001`,
`aggregate_type=2_wk_max`) both cross their thresholds, at every vegetation-grid pixel --
the paired (project + spatial join) pipeline shape documented at `docs/analysis-api.md`'s
"The Salinity-Veg Investigation" example, landed here as a named analysis rather than
requiring a caller to hand-author that exact JSON.

This is the salinity/freshwater-vegetation QA-QC investigation the team's template
notebook works through block by block ("the first year each freshwater marsh raster cell
exceeds the given salinity threshold"), served as one request instead.

The only entry exercising the `type="custom"` path, multiple named inputs, and a
`project` + spatial `join`, and the only one with no existing consumer (unlike
`difference`/`land-change`, which replace real portal catalog entries).

**Why the inputs are named `coverage`/`salinity` and the join operands `marsh`/`sal`.**
A `join`'s output value columns are named after its OPERANDS (see `ops.validate_shapes`),
so the threshold test -- `marsh > ... AND sal > ...` -- can only work if the two frames
reaching the join carry exactly those names. An `as` may not collide with an input name,
so the inputs take the two longer names and the per-input `filter`/`project` steps rename
their outputs down to the two the threshold test reads.

**Why the threshold test is a `case` rather than a `filter`.** Band-wide, the joined frame
holds both operands' bands side by side in one flat positional list, and the question being
asked is about the PAIR. A `case` over a paired frame reconstitutes the pairs and reduces
each to a single band (`execution.band_ops.case_band_wide`) -- 2N bands in, N out -- which
is what leaves one band per year for the `group` to reduce over. A `filter` cannot: it
restricts observations, and there is no observation here to drop that would not also drop
the other operand's band at the same year.

Nothing reads the surviving band's VALUE. The `group` is `min of=calendar_year`, so each
band contributes its own year wherever it is non-null and withholds it where it is not --
the band carries WHETHER, and `of` supplies WHICH. The arm names `marsh` anyway, because a
band has to hold something and an arm has to say what: the analysis's answer is the year,
and the value under it is the marsh coverage that cleared the threshold.
"""

from pydantic import Field

from cpra.mp.data.analyses.base import (
    Analysis,
    InvalidAnalysisParametersError,
    selection_filter_where,
)
from cpra.mp.data.analysis.request.pipeline import parse_pipeline


class SalinityVegInvestigation(Analysis):
    class Meta:
        name = "salinity-veg-investigation"
        label = "Salinity-Veg Investigation"
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

    @classmethod
    def validate_selection(cls, parameters, filters):
        """Reject a selection outright rather than accepting one `pipeline()` below would
        silently drop.

        A `type="dataset"` analysis composes `filters` into its leading `filter` step via
        `selection_filter_where(self.filters)`; this one cannot. It has no dataset, and
        its two inputs sit on DIFFERENT grids (`veg_grid_cell_v001` pixels and
        `hydro_compartment_v001` polygons), so neither a `grid_cell` selection nor a bare
        `where` has one unambiguous frame to apply to -- and the caller-facing dimension
        values it does accept (`model_group_id`/`scenario_id`) are declared `Parameters`,
        applied to both inputs by `pipeline()` itself. Silently ignoring a `grid_cell` the
        caller sent would look like a whole-extent answer to a point query.
        """
        if filters:
            raise InvalidAnalysisParametersError(
                f"analysis 'salinity-veg-investigation' takes no selection, but got: "
                f"{', '.join(sorted(filters))} -- it names its own two inputs, on two "
                f"different grids, so there is no single frame for a grid_cell/where to "
                f"narrow; its dimension values are parameters "
                f"(model_group_id, scenario_id) instead"
            )

    def pipeline(self):
        assert self.dataset is None, "salinity-veg-investigation is a type='custom' analysis"
        p = self.parameters
        coverage_where = selection_filter_where(
            {
                "model_group_id": p.model_group_id,
                "scenario_id": p.scenario_id,
                "ffibs_type": "pl_fm",
            }
        )
        salinity_where = selection_filter_where(
            {
                "model_group_id": p.model_group_id,
                "scenario_id": p.scenario_id,
                "aggregate_type": "2_wk_max",
            }
        )
        data = {
            "inputs": {
                "coverage": {
                    "variable": "ffibs_coverage_ratio",
                    "time_unit": "annual",
                    "grid": "veg_grid_cell_v001",
                },
                "salinity": {
                    "variable": "sal",
                    "time_unit": "annual",
                    "grid": "hydro_compartment_v001",
                },
            },
            "steps": [
                {"op": "filter", "input": "coverage", "where": coverage_where, "as": "marsh"},
                {
                    "op": "filter",
                    "input": "salinity",
                    "where": salinity_where,
                    "as": "selected_salinity",
                },
                {
                    "op": "project",
                    "input": "selected_salinity",
                    "to": "veg_grid_cell_v001",
                    "as": "sal",
                },
                {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
                # The threshold test, as a single-armed `case` over the pair -- see "Why
                # the threshold test is a `case`" in the module docstring. No `else`, so a
                # year failing either threshold comes out null and the `group` below skips
                # it; `then` keeps `marsh`, the only determinate value on offer.
                {
                    "op": "case",
                    "cases": [
                        {
                            "when": f"marsh > {p.marsh_threshold} AND sal > {p.salinity_threshold}",
                            "then": "marsh",
                        }
                    ],
                },
                {"op": "group", "by": [], "aggregate": "min", "of": "calendar_year"},
            ],
            "output": {"format": "tif"},
        }
        return parse_pipeline(data)
