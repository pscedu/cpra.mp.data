"""`land-change` -- a dataset-scoped, `kind="variable"` analysis producing a categorical
land-type-change code (`variables=["lnd_type"]`, `legend="lnd_chng"`) between an
automatically-derived baseline year and a caller-selected comparison year. Renamed from
`concatenate`, whose caller-facing shape it replaces: rather than the caller supplying any
filterable `dimension` and exactly 2 values of it, `land-change` fixes the concatenation
axis to `calendar_year` and derives the earlier (baseline) year itself from the selected
`model_group_id`, so the caller supplies only the later (comparison) year -- the same
single-value shape selecting `calendar_year` already has for every other variable. This is
the concrete sense in which `land-change` "behaves like a variable" rather than a special
multi-select mode.

The baseline-year rule is a business rule, not derived from anything else in this
codebase: `model_group_id` values below `BASELINE_ERA_BREAKPOINT` belong to an earlier
model era whose baseline year is `EARLY_BASELINE_YEAR`; at or above it, the baseline year
is `LATE_BASELINE_YEAR`. In three-image mode (two selected `model_group_id` values), the
LOWER of the two decides the era regardless of where the other one falls -- e.g.
`model_group_id` 650 and 900 selected together still resolve to the 2019 baseline, since
650 is the lower one.

Two modes, selected by how many `model_group_id` values are selected (mirrors
`concatenate`'s own two-mode split; see that history for the portal-side derivation this
replaces, `cpra-mp-ui/src/helpers/catalog.ts`'s `threeImage` reading):

* **two-image** (`model_group_id` selected to exactly 1 value): a `case` whose one arm is
  `value * 10` when `calendar_year = t0` (the derived baseline year), with `else` `value`,
  then `group`/`sum` over every other required dimension.
* **three-image** (`model_group_id` selected to exactly 2 values, `mg0 < mg1`): a `filter`
  dropping `model_group_id = mg1 AND calendar_year = t0` (the redundant corner), then a
  `case` whose arms are `value * 100` when `calendar_year = t0` and `value * 10` when
  `model_group_id = mg0 AND calendar_year = t1`, with `else` `value`, then `group`/`sum`
  over every other required dimension (excluding `model_group_id`, which the `case` above
  has folded away).

Both modes' `else` is `value`, not omitted: the cell the arms do not name is the ones
digit, and it has to arrive at the `sum` unchanged rather than null.
"""

from cpra.mp.data.analyses.base import (
    Analysis,
    InvalidAnalysisParametersError,
    dataset_input_selection,
    dataset_output_format,
    format_dimension_literal,
    normalized_filter_values,
    required_dataset_dimensions,
    selection_filter_where,
)
from cpra.mp.data.analysis.request.pipeline import parse_pipeline

# The model_group_id boundary between the two baseline eras. Given directly by the
# maintainer -- no existing convention in either repo names this boundary or either
# baseline year, so the rationale for why 700 is the era boundary belongs in a follow-up
# to this docstring rather than being invented here.
BASELINE_ERA_BREAKPOINT = 700
EARLY_BASELINE_YEAR = 2019
LATE_BASELINE_YEAR = 2025


class LandChange(Analysis):
    class Meta:
        name = "land-change"
        label = "Land Change"
        type = "dataset"
        kind = "variable"
        variables = ["lnd_type"]
        legend = "lnd_chng"
        result_kind = "category"
        # Unconditional: `lnd_type` only exists at annual grain, so `calendar_year` is
        # always the concatenation axis. `model_group_id`'s OTHER subsumption (in
        # three-image mode only) is not declared here at all, because it isn't static --
        # the runtime `collapsed_dimensions` (derived structurally by
        # `default_collapsed_dimensions`, since this pipeline always ends in an ordinary
        # `group`) already reports it correctly per-request.
        subsumed_dimensions = ["calendar_year"]

    class Parameters(Analysis.Parameters):
        pass  # no fields -- comparison year and image count both come from ordinary filters

    @classmethod
    def validate_selection(cls, parameters: "LandChange.Parameters", filters: dict) -> None:
        raw = filters.get("calendar_year")
        values = normalized_filter_values("calendar_year", raw) if raw is not None else []
        if len(values) != 1:
            raise InvalidAnalysisParametersError(
                "land-change requires exactly 1 selected calendar_year (the comparison "
                f"year -- the baseline year is derived automatically), got {raw!r}"
            )
        mg_raw = filters.get("model_group_id")
        mg_values = normalized_filter_values("model_group_id", mg_raw) if mg_raw is not None else []
        if len(mg_values) not in (1, 2):
            raise InvalidAnalysisParametersError(
                "land-change requires exactly 1 selected model_group_id (two-image mode) "
                f"or exactly 2 (three-image mode), got {len(mg_values)}: {mg_raw!r}"
            )

    def pipeline(self):
        assert self.dataset is not None, "land-change is a type='dataset' analysis"
        dim = "calendar_year"
        t1 = normalized_filter_values(dim, self.filters[dim])[0]
        mg_values = normalized_filter_values("model_group_id", self.filters["model_group_id"])
        three_image = len(mg_values) == 2
        t0 = EARLY_BASELINE_YEAR if min(mg_values) < BASELINE_ERA_BREAKPOINT else LATE_BASELINE_YEAR

        # t0 is never one of the caller's own filter values -- fold it in the same way
        # `land-type-confidence` folds in its fixed `scenario_id` set, so BOTH years reach
        # the leading filter step rather than just the caller-selected one.
        filters = {**self.filters, dim: [t0, t1]}

        required = required_dataset_dimensions(self.dataset)
        group_by = [d for d in required if d != dim]
        if three_image:
            group_by = [d for d in group_by if d != "model_group_id"]
            mg0, mg1 = sorted(mg_values)

        # Folding t0/t1 into `filters` above guarantees `selection_filter_where` always
        # returns a conjunct (at minimum the `calendar_year IN (...)` one), unlike
        # `difference`/the old `concatenate`, which may see an empty `filters` on a direct
        # constructor call -- so the leading `filter` step here is unconditional.
        conjuncts = [selection_filter_where(filters, grid_cell_where=self.grid_cell_where)]
        if three_image:
            # Drop the redundant corner of the 2x2 (model_group_id, calendar_year) grid,
            # so the three surviving cells are what the `case` below encodes.
            conjuncts.append(
                f"NOT (model_group_id = {format_dimension_literal('model_group_id', mg1)} "
                f"AND {dim} = {format_dimension_literal(dim, t0)})"
            )
        steps: list[dict] = [{"op": "filter", "where": " AND ".join(c for c in conjuncts if c)}]

        # Place each surviving cell in its own decimal digit, then let `group sum` pack
        # them into one number -- see `concatenate`'s own history for why these are ARMS
        # rather than successive rewrites (`case` tests every `when` against the value the
        # frame arrived with, so the `* 10` arm cannot see what the `* 100` arm wrote).
        if three_image:
            cases = [
                {"when": f"{dim} = {format_dimension_literal(dim, t0)}", "then": "value * 100"},
                {
                    "when": (
                        f"model_group_id = {format_dimension_literal('model_group_id', mg0)} "
                        f"AND {dim} = {format_dimension_literal(dim, t1)}"
                    ),
                    "then": "value * 10",
                },
            ]
        else:
            cases = [{"when": f"{dim} = {format_dimension_literal(dim, t0)}", "then": "value * 10"}]
        steps.append({"op": "case", "cases": cases, "else": "value"})
        steps.append({"op": "group", "by": group_by, "aggregate": "sum"})
        # A pipeline's first step must name its input explicitly; every later step chains
        # from the one before it.
        steps[0]["input"] = "x"

        data = {
            "inputs": {"x": dataset_input_selection(self.dataset)},
            "steps": steps,
            # Derived from the dataset's grid kind: this analysis neither projects nor
            # otherwise changes shape, so a vector selection ends tabular and a raster one
            # ends band-wide, and `validate_shapes` rejects the mismatched sink in either
            # direction. Only `pipeline.json` ever shows this value -- the
            # `data.{format}`/tile routes overwrite it from their own URL segment -- but
            # what `pipeline.json` publishes has to be resubmittable, and this analysis's
            # own `Meta.variables` (`lnd_type`) lives only on a raster grid.
            "output": {"format": dataset_output_format(self.dataset)},
        }
        return parse_pipeline(data)
