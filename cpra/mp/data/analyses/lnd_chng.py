"""`lnd_chng` -- a dataset-scoped, `kind="variable"` analysis producing a categorical
land-type-change code (`variables=["lnd_type"]`, `legend="lnd_chng"`) between an
automatically-derived baseline year and the caller-selected comparison year(s). Renamed
from `concatenate`, whose caller-facing shape it replaces: rather than the caller supplying
any filterable `dimension` and exactly 2 values of it, `lnd_chng` fixes the concatenation
axis to `calendar_year` and derives the earlier (baseline) year itself from the selected
`model_group_id`, so the caller supplies only the later year(s) -- the same shape selecting
`calendar_year` already has for every other variable. This is the concrete sense in which
`lnd_chng` "behaves like a variable" rather than a special multi-select mode.

**The baseline broadcasts; it is not one half of a pair.** Each comparison year is compared
against the SAME derived baseline and comes back as its own row/band, so a land-type change
series over time is an ordinary `calendar_year` selection -- several values, or (as a chart
request sends it) none at all, which means every year. That is why `calendar_year` is not in
`Meta.subsumed_dimensions` and why this pipeline ends on a `calculate` rather than a
`group`: the caller still selects years, the result still varies over them, and nothing
here folds the axis away.

The broadcast is the `join`. The baseline branch is filtered to the derived year and then
`group`ed over every OTHER required dimension, which drops its `calendar_year` column; the
comparison branch keeps every selected year. `execution.long_ops.join_long` keys on
`grid_cell` plus the columns both operands carry, so a column only one of them carries
rides along unchanged -- one baseline value per grid cell, fanned out across every
comparison year. Band-wide, the same missing column is excluded from
`execution.band_ops._pair_band_groups`'s pairing key, which pairs the baseline band with
the comparison band for any single year.

**Band-wide output takes exactly one comparison year**, and `pipeline()` -- not
`validate_selection`, which cannot see the grid -- 400s on any other count. That pairing
requires exactly one band per operand per group, and with `calendar_year` out of the key a
one-band baseline against N comparison bands is a single group holding N+1 bands: a pairing
MISS, which drops silently and surfaces as an empty result rather than as anything a caller
could act on. A raster selection lowers to long when (and only when) it carries a spatial
conjunct (`request.ops.FILTER_LOWERING_TABLE`), which is what a per-grid-cell chart request
always does, so the series a chart asks for is unaffected -- as is every vector dataset,
which is long throughout.

The baseline-year rule is a business rule, not derived from anything else in this
codebase: `model_group_id` values below `BASELINE_ERA_BREAKPOINT` belong to an earlier
model era whose baseline year is `EARLY_BASELINE_YEAR`; at or above it, the baseline year
is `LATE_BASELINE_YEAR`. `BASELINE_YEAR_OVERRIDES` names the model groups the breakpoint
does not describe at all, and wins over it where it has an entry -- `model_group_id` 0
compares against 2025, not the 2019 its position below the breakpoint would otherwise give
it. That override is not a tweak to the era rule but a statement that the group is outside
the eras: the breakpoint sorts MODEL eras, and a group whose data is not one of them has
its baseline given directly rather than derived. It was added because the derived 2019
simply has no data for that group, which surfaced as a 500 from a deployed portal (see
`tests/api/test_analyses_endpoints.py::TestLandChangeMissingBaselineRoute`, which still
pins the answer for the general case of a baseline year no dataset carries).

In three-image mode (two selected `model_group_id` values), the LOWER of the two decides
the baseline regardless of where the other one falls -- e.g. `model_group_id` 650 and 900
selected together still resolve to the 2019 baseline, since 650 is the lower one, and 0
selected alongside anything resolves to its own 2025 for the same reason.

Two modes, selected by how many `model_group_id` values are selected (mirrors
`concatenate`'s own two-mode split; see that history for the portal-side derivation this
replaces, `cpra-mp-ui/src/helpers/catalog.ts`'s `threeImage` reading). Both place each
land-type value in its own decimal digit and add them:

* **two-image** (`model_group_id` selected to exactly 1 value): `baseline * 10 +
  comparison`, where `baseline` is the derived year's land type and `comparison` is each
  selected year's, unchanged.
* **three-image** (`model_group_id` selected to exactly 2 values, `mg0 < mg1`):
  `baseline * 100 + comparison`, where `baseline` is `mg0`'s land type in the derived year
  and `comparison` is the two model groups' land types in the selected year already packed
  into two digits -- `mg0`'s multiplied by 10 and `mg1`'s passed through, summed by the
  `group` that folds `model_group_id` away. The 2x2 (model group x year) grid's fourth
  corner, `mg1` in the baseline year, is simply never read: the baseline branch is filtered
  to `mg0`. It used to be excluded by a `NOT (...)` conjunct on a single shared read, which
  is the same thing said negatively because there was one branch to say it on.

The three-image `case`'s `else` is `value`, not omitted: `mg1`'s land type is the ones
digit, and it has to arrive at the `sum` unchanged rather than null.
"""

from cpra.mp.data.analyses.base import (
    Analysis,
    AnalysisMeta,
    InvalidAnalysisParametersError,
    dataset_input_selection,
    dataset_output_format,
    format_dimension_literal,
    normalized_filter_values,
    required_dataset_dimensions,
    selection_filter_where,
)
from cpra.mp.data.analysis.request.pipeline import parse_pipeline
from cpra.mp.data.grids import GRIDS

# The model_group_id boundary between the two baseline eras. Given directly by the
# maintainer -- no existing convention in either repo names this boundary or either
# baseline year, so the rationale for why 700 is the era boundary belongs in a follow-up
# to this docstring rather than being invented here.
BASELINE_ERA_BREAKPOINT = 700
EARLY_BASELINE_YEAR = 2019
LATE_BASELINE_YEAR = 2025

# The model groups the breakpoint above does not describe, mapped to the baseline year each
# one compares against. Also given directly by the maintainer, and a TABLE rather than a
# third named constant because these are exceptions to the era rule rather than a third
# era: an entry says "this group's baseline is not derivable from where it sits relative to
# `BASELINE_ERA_BREAKPOINT`", and the next such group needs a row here, not another branch.
# Group 0's year happens to equal `LATE_BASELINE_YEAR` today; it is spelled as a literal on
# purpose, so moving the late era's baseline does not silently move group 0's with it.
BASELINE_YEAR_OVERRIDES = {0: 2025}


def baseline_year(model_group_id: int) -> int:
    """The `calendar_year` `model_group_id`'s land types are compared AGAINST -- the whole
    of the derivation the caller does not get to make (see the module docstring's
    baseline-year paragraph).

    Module-level rather than a method so a test, and any future caller that needs to know
    which year a selection will actually read, can ask without building an `Analysis`.
    """
    if model_group_id in BASELINE_YEAR_OVERRIDES:
        return BASELINE_YEAR_OVERRIDES[model_group_id]
    return EARLY_BASELINE_YEAR if model_group_id < BASELINE_ERA_BREAKPOINT else LATE_BASELINE_YEAR


class LandChange(Analysis):
    Meta = AnalysisMeta(
        name="lnd_chng",
        label="Land Change",
        kind="variable",
        variables=["lnd_type"],
        legend="lnd_chng",
        result_kind="category",
        # Nothing. This field is what the portal hides from the caller
        # (`Structure.subsumedFor`), and `calendar_year` is the one dimension this analysis
        # most needs a caller to keep: the comparison years are selected, the result varies
        # over them, and a chart plots them. It DID list `calendar_year`, back when the two
        # years were fused into one code by a trailing `group`/`sum` -- see the module
        # docstring's second paragraph for what replaced that.
        #
        # `model_group_id`'s subsumption (in three-image mode only) is not declared here
        # either, and for the reason it never was: it isn't static. The runtime
        # `collapsed_dimensions` below reports it correctly per-request.
        subsumed_dimensions=[],
    )

    class Parameters(Analysis.Parameters):
        pass  # no fields -- comparison years and image count both come from ordinary filters

    @classmethod
    def validate_selection(cls, parameters: "LandChange.Parameters", filters: dict) -> None:
        """`calendar_year` has no rule here at all any more, and its absence is the
        interesting half: an unselected dimension means "every value of it" everywhere else
        in this system, and it means that here too -- every year against the derived
        baseline, which is the series a chart asks for. It used to be "exactly 1", which is
        a rule only a raster IMAGE needs; `pipeline()` states that one where the grid kind
        is known (see the module docstring's third paragraph), so a chart is not held to it.
        """
        mg_raw = filters.get("model_group_id")
        mg_values = normalized_filter_values("model_group_id", mg_raw) if mg_raw is not None else []
        if len(mg_values) not in (1, 2):
            raise InvalidAnalysisParametersError(
                "lnd_chng requires exactly 1 selected model_group_id (two-image mode) "
                f"or exactly 2 (three-image mode), got {len(mg_values)}: {mg_raw!r}"
            )

    @property
    def _three_image(self) -> bool:
        """Two selected `model_group_id` values, the three-image mode. Read by
        `pipeline()` and by `collapsed_dimensions`, which must agree about it without one
        having to build the other.
        """
        raw = self.filters.get("model_group_id")
        return len(normalized_filter_values("model_group_id", raw) if raw is not None else []) == 2

    @property
    def collapsed_dimensions(self) -> frozenset[str]:
        """Stated here rather than derived, because `base.default_collapsed_dimensions`
        reads a trailing `group.by` and this pipeline ends on a `calculate`.

        The answer is also shorter than it used to be. `calendar_year` is no longer in it:
        the result carries one value per (grid cell, selected comparison year), so the
        caller's own year selection still varies the output and the visualization layer
        must keep the column. `model_group_id` is, in three-image mode only, because the
        two selected model groups are packed into two digits of a single code and the
        `group` that packs them drops the column.
        """
        return frozenset({"model_group_id"}) if self._three_image else frozenset()

    def pipeline(self):
        dataset = self.require_dataset()
        dim = "calendar_year"
        # An empty list is "every year", the ordinary meaning of an unselected dimension --
        # `selection_filter_where` writes no conjunct for it and the comparison branch reads
        # the lot. See `validate_selection`.
        raw_years = self.filters.get(dim)
        years = normalized_filter_values(dim, raw_years) if raw_years is not None else []
        mg_values = normalized_filter_values("model_group_id", self.filters["model_group_id"])
        three_image = len(mg_values) == 2
        # `min` in three-image mode: the lower model group decides, and it is also the one
        # the baseline branch is filtered to below, so the year and the group it is read
        # from cannot disagree.
        t0 = baseline_year(min(mg_values))

        # A raster selection stays band-wide unless something lowers it, and only a SPATIAL
        # conjunct does (`request.ops.FILTER_LOWERING_TABLE`) -- a caller's value `where`
        # does not. Band-wide, the join below pairs one baseline band against one comparison
        # band and N of them pair-MISS, which drops silently; say so instead. See the module
        # docstring's third paragraph.
        spatial = self.grid_cell_where is not None or bool(self.filters.get("grid_cell"))
        if len(years) != 1 and not spatial and GRIDS[dataset.grid].grid_type == "raster":
            got = f"got {len(years)}: {years!r}" if years else "none selected, which is all of them"
            raise InvalidAnalysisParametersError(
                f"lnd_chng draws one image per comparison year and a raster grid "
                f"({dataset.grid!r}) holds one, so select exactly 1 calendar_year -- {got}. "
                "Several years are a series rather than an image; select a grid_cell to ask "
                "for one."
            )

        # t0 is never one of the caller's own filter values -- fold it in the same way
        # `lnd_conf` folds in its fixed `scenario_id` set. Each branch states the WHOLE
        # selection, its own year(s) included, so both narrow to their own bands at the
        # read: `execution.plan`'s dimension passes push a conjunct to the origins that
        # actually feed the frame it was written on, not to every origin in the pipeline.
        baseline_filters = {**self.filters, dim: [t0]}
        comparison_filters = {**self.filters, dim: years}
        if three_image:
            # The 2x2 grid's fourth corner (`mg1` in the baseline year) is redundant, and
            # here it is simply not read: the baseline is `mg0`'s.
            mg0, mg1 = sorted(mg_values)
            baseline_filters["model_group_id"] = [mg0]

        required = required_dataset_dimensions(dataset)
        # Dropping `calendar_year` from the baseline's `group.by` is the whole point of the
        # step: with the column gone, the join has nothing to match it against and the one
        # baseline value per grid cell broadcasts across every comparison year. The
        # aggregate itself is an identity -- one year, one model group, one row per cell.
        baseline_by = [d for d in required if d != dim]
        comparison_by = list(required)
        if three_image:
            baseline_by = [d for d in baseline_by if d != "model_group_id"]
            comparison_by = [d for d in comparison_by if d != "model_group_id"]

        # Both branches' `where` is unconditional: folding the year(s) into `filters` above
        # guarantees `selection_filter_where` returns a conjunct (at minimum the
        # `calendar_year` one), unlike `diff`, which may see an empty `filters` on a direct
        # constructor call.
        steps: list[dict] = [
            {
                "op": "filter",
                "input": "base",
                "where": selection_filter_where(
                    baseline_filters, grid_cell_where=self.grid_cell_where
                ),
                "as": "baseline_year",
            },
            {"op": "group", "by": baseline_by, "aggregate": "sum", "as": "baseline"},
            {
                "op": "filter",
                "input": "years",
                "where": selection_filter_where(
                    comparison_filters, grid_cell_where=self.grid_cell_where
                ),
                # A join operand's name IS the value column name the `calculate` below
                # reads, so in two-image mode this filter's output is the operand itself.
                "as": "comparison_years" if three_image else "comparison",
            },
        ]
        if three_image:
            # The tens and ones digits, packed on the comparison branch alone so the
            # baseline is free to broadcast: `mg0`'s land type is multiplied by 10, `mg1`'s
            # passes through, and the `group` below sums the pair per (grid cell, year).
            # That `group` is also what drops `model_group_id`, without which the two model
            # groups' bands would be keyed apart and never meet.
            steps.append(
                {
                    "op": "case",
                    "cases": [
                        {
                            "when": f"model_group_id = {format_dimension_literal('model_group_id', mg0)}",
                            "then": "value * 10",
                        }
                    ],
                    "else": "value",
                    "as": "scaled",
                }
            )
            steps.append(
                {"op": "group", "by": comparison_by, "aggregate": "sum", "as": "comparison"}
            )
        steps.append(
            {"op": "join", "type": "spatial", "inputs": ["baseline", "comparison"], "as": "paired"}
        )
        # The hundreds digit (three-image) or the tens digit (two-image), and the single
        # value column the sink rule requires: a `calculate` always produces exactly one,
        # whatever the join handed it. A null on either operand carries through as a null,
        # which is the right answer -- a grid cell with no baseline has no change to report.
        steps.append(
            {"op": "calculate", "expression": f"baseline * {100 if three_image else 10} + comparison"}
        )

        data = {
            # The same dataset twice, once per branch. `execution.plan` narrows each origin
            # from the conjuncts on the frames it feeds, so the baseline read is one year
            # wide and the comparison read is the caller's selection -- not both reading
            # both.
            "inputs": {
                "base": dataset_input_selection(dataset),
                "years": dataset_input_selection(dataset),
            },
            "steps": steps,
            # Derived from the dataset's grid kind: this analysis neither projects nor
            # otherwise changes shape, so a vector selection ends tabular and a raster one
            # ends band-wide, and `validate_shapes` rejects the mismatched sink in either
            # direction. Only `pipeline.json` ever shows this value -- the
            # `data.{format}`/tile routes overwrite it from their own URL segment -- but
            # what `pipeline.json` publishes has to be resubmittable, and this analysis's
            # own `Meta.variables` (`lnd_type`) lives only on a raster grid.
            "output": {"format": dataset_output_format(dataset)},
        }
        return parse_pipeline(data)
