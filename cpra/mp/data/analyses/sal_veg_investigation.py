"""`sal_veg_investigation` -- the one v1 entry with FIXED inputs: the earliest calendar
year at which marsh coverage (`ffibs_coverage_ratio`/annual/`veg_grid_cell_v001`,
`ffibs_type=pl_fm`) and paired salinity (`sal`/annual/`hydro_compartment_v001`,
`aggregate_type=2_wk_max`) both cross their thresholds, at every vegetation-grid pixel --
the paired (project + spatial join) pipeline shape documented at `docs/analysis-api.md`'s
"The Salinity-Veg Investigation" example, landed here as a named analysis rather than
requiring a caller to hand-author that exact JSON.

This is the salinity/freshwater-vegetation QA-QC investigation the team's template
notebook works through block by block ("the first year each freshwater marsh raster cell
exceeds the given salinity threshold"), served as one request instead.

The only entry exercising the fixed-input path, multiple named inputs, and a
`project` + spatial `join`, and the only one with no existing consumer (unlike
`diff`/`lnd_chng`, which replace real portal catalog entries).

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
    AnalysisMeta,
    selection_filter_where,
)
from cpra.mp.data.analysis.request.pipeline import parse_pipeline


class SalinityVegInvestigation(Analysis):
    Meta = AnalysisMeta(
        name="sal_veg_investigation",
        label="Salinity-Veg Investigation",
        variables=None,
        legend=None,
        result_kind="value",
        # Neither a variable the portal offers nor a stage anything may chain onto: this
        # entry exists to exercise the two-input, two-grid path end to end. It declared no
        # `kind` at all until `AnalysisMeta` made the field required -- which read as
        # `None` at `resolve_analysis_chain`'s gate and at `map.py`'s title hook, i.e.
        # indistinguishable from an author who forgot one.
        kind="internal",
    )

    class Parameters(Analysis.Parameters):
        # `model_group_id`/`scenario_id` were declared here too, required, back when a
        # fixed-input analysis could not RECEIVE a dimension filter. They are ordinary
        # filters now, composed into the trailing `filter` step `pipeline()` writes over the
        # joined frame; see `lnd_type_plus.Parameters` for the whole argument, including why
        # nothing replaces the `required`. What is left is the two THRESHOLDS, which were
        # never dimensions -- they are this analysis's own numbers, and `analyses._register`'s
        # dimension-name check passes precisely because neither names one.
        #
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

    # No `validate_selection`. It rejected any `filters` at all, on the reading that a
    # fixed-input analysis has no single frame for a selection to narrow -- true of a
    # `grid_cell`, which names a cell in a lattice this request does not say which of, and
    # NOT true of the dimension filters that made up every real use of it. The engine now
    # makes that distinction itself: a dimension conjunct on the joined frame pushes down to
    # both origins' reads (`execution.plan`), while a spatial or value predicate lowers the
    # frame to long (`request.ops.FILTER_LOWERING_TABLE`) and is then rejected against this
    # analysis's raster `output.format` by `validate_shapes`, with a message that says which
    # of the two problems the caller actually has.

    def pipeline(self):
        self.reject_dataset()
        p = self.parameters
        # The caller's selection, stated ONCE on the joined frame below rather than folded
        # into each input's own `filter`. `execution.plan`'s dimension passes push a
        # conjunct declared anywhere down to every origin's read, across the join and past
        # the `project`, so the two read the same bytes either way -- and stating it once
        # keeps it from having to be spelled twice, in two dicts that could drift.
        selection_where = selection_filter_where(self.filters, grid_cell_where=self.grid_cell_where)
        # What stays per-input is DEFINITIONAL rather than selected: `ffibs_type=pl_fm` is
        # the freshwater marsh this investigation is about, and `aggregate_type=2_wk_max` is
        # the one reading of salinity its threshold means anything against. Neither is a
        # dimension the pairing carries -- each is declared by exactly one of the two
        # inputs -- so there is no later frame to state either on.
        coverage_where = selection_filter_where({"ffibs_type": "pl_fm"})
        salinity_where = selection_filter_where({"aggregate_type": "2_wk_max"})
        # Between the join and the threshold `case`, for the same reason `lnd_type_plus`
        # puts its own there: a dimension filter leaves the pairing standing, and the `case`
        # is what collapses it. `None` when the caller selected nothing, which is now legal.
        selection_steps = [] if selection_where is None else [{"op": "filter", "where": selection_where}]
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
                *selection_steps,
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
