"""`concatenate` -- a dataset-scoped analysis over two selected values of one dimension,
producing a categorical land-type-change code (`variables=["lnd_type"]`,
`legend="lnd_chng"`). Replaces the portal's `concatenate` catalog entry (see
`cpra-mp-ui/src/helpers/catalog.ts`'s real derivation, lines 95-138).

Two modes, selected by how many `model_group_id` values are selected alongside
`dimension`'s own two values (see `catalog.ts`'s `threeImage = modelGroupValues.length ===
2`, NOT `dimension`'s own cardinality -- `dimension` is always exactly 2 for both modes;
see this package's final report for why the "concatenate takes 2-3" phrasing some of this
package's own design notes used is an imprecise paraphrase of that real logic):

* **two-image** (`model_group_id` absent, or selected to exactly 1 value): `calculate`
  `value * 10` where `dimension = t0` (the earlier of the two selected values), then
  `group`/`sum` over every other required dimension.
* **three-image** (`model_group_id` selected to exactly 2 values, `mg0 < mg1`): a `filter`
  dropping `model_group_id = mg1 AND dimension = t0` (the redundant corner), then
  `calculate` `value * 100` where `dimension = t0`, then `calculate` `value * 10` where
  `model_group_id = mg0 AND dimension = t1`, then `group`/`sum` over every other required
  dimension (excluding `model_group_id`, which the calculates above have folded away).

The `filter` step is emitted BEFORE both `calculate` steps (matching the pipeline's own
step order) -- `analysis.planner._compile_single_input_chain` accepts a `filter` ordered
after a `calculate` only when the `where` is dimension-only (the two commute); this
`filter`'s `where` IS dimension-only (neither conjunct names `value`), so placing it first
is not strictly required to compile, but is the clearer -- and, per this package's task
instructions, the mandated -- ordering.

**On the filter's exact spelling.** The redundant corner is naturally "NOT (model_group_id
= mg1 AND dimension = t0)", but `analysis.planner._validate_filter_where` (the pipeline
compiler's own compile-time check on every `filter` step's `where`, run regardless of
whether the conjunct is dimension- or value-shaped) validates each top-level AND-ed
conjunct through `expressions.parse_value_expr`'s restricted grammar, which parses `not(`
as an attempted function call and rejects it (`not` is not in `ALLOWED_FUNCTIONS`) --
`where`'s restricted grammar has no boolean NOT/OR-of-AND-groups spelling at all. The
logically-equivalent De Morgan's form -- "(model_group_id != mg1) OR (dimension != t0)" --
uses only `!=`/`OR` at the top level (no parenthesized call, so nothing matches
`parse_value_expr`'s function-call check), and does compile.
"""

from cpra.mp.data.analyses.base import (
    Analysis,
    InvalidAnalysisParametersError,
    dataset_input_selection,
    format_dimension_literal,
    is_filterable_dimension,
    normalized_filter_values,
    required_dataset_dimensions,
)
from cpra.mp.data.analysis.pipeline import parse_pipeline
from pydantic import field_validator


class Concatenate(Analysis):
    class Meta:
        name = "concatenate"
        label = "Concatenate"
        type = "dataset"
        variables = ["lnd_type"]
        legend = "lnd_chng"
        result_kind = "category"

    class Parameters(Analysis.Parameters):
        dimension: str

        @field_validator("dimension")
        @classmethod
        def _dimension_must_be_filterable(cls, value: str) -> str:
            if not is_filterable_dimension(value):
                raise ValueError(f"{value!r} is not a filterable dimension")
            return value

    @classmethod
    def validate_selection(cls, parameters: "Concatenate.Parameters", filters: dict) -> None:
        dim = parameters.dimension
        raw = filters.get(dim)
        if raw is None:
            raise InvalidAnalysisParametersError(
                f"concatenate on dimension {dim!r} requires a {dim!r} filter selecting "
                "exactly 2 distinct values, but none was given"
            )
        values = normalized_filter_values(dim, raw)
        if len(values) != 2:
            raise InvalidAnalysisParametersError(
                f"concatenate on dimension {dim!r} requires exactly 2 distinct selected "
                f"values, got {len(values)}: {raw!r}"
            )

        mg_raw = filters.get("model_group_id")
        if mg_raw is not None:
            mg_values = normalized_filter_values("model_group_id", mg_raw)
            if len(mg_values) not in (1, 2):
                raise InvalidAnalysisParametersError(
                    "concatenate requires exactly 1 selected model_group_id (two-image "
                    f"mode) or exactly 2 (three-image mode), got {len(mg_values)}: "
                    f"{mg_raw!r}"
                )

    def pipeline(self):
        assert self.dataset is not None, "concatenate is a type='dataset' analysis"
        dim = self.parameters.dimension
        values = normalized_filter_values(dim, self.filters[dim])
        if len(values) != 2:
            raise InvalidAnalysisParametersError(
                f"concatenate on dimension {dim!r} requires exactly 2 distinct selected "
                f"values, got {len(values)}: {self.filters[dim]!r}"
            )
        t0, t1 = sorted(values)

        mg_raw = self.filters.get("model_group_id")
        mg_values = normalized_filter_values("model_group_id", mg_raw) if mg_raw is not None else []
        if mg_values and len(mg_values) not in (1, 2):
            raise InvalidAnalysisParametersError(
                "concatenate requires exactly 1 selected model_group_id (two-image mode) "
                f"or exactly 2 (three-image mode), got {len(mg_values)}: {mg_raw!r}"
            )
        three_image = len(mg_values) == 2

        required = required_dataset_dimensions(self.dataset)
        group_by = [d for d in required if d != dim]
        if three_image:
            group_by = [d for d in group_by if d != "model_group_id"]
            mg0, mg1 = sorted(mg_values)

        steps: list[dict] = [{"op": "project", "input": "x", "to": self.dataset.grid}]
        if three_image:
            steps.append(
                {
                    "op": "filter",
                    # De Morgan's form of "not (model_group_id = mg1 AND dim = t0)" --
                    # see this module's own docstring for why the direct NOT/AND-group
                    # spelling doesn't compile.
                    "where": (
                        f"model_group_id != {format_dimension_literal('model_group_id', mg1)} "
                        f"OR {dim} != {format_dimension_literal(dim, t0)}"
                    ),
                }
            )
            steps.append(
                {
                    "op": "calculate",
                    "expression": "value * 100",
                    "where": f"{dim} = {format_dimension_literal(dim, t0)}",
                }
            )
            steps.append(
                {
                    "op": "calculate",
                    "expression": "value * 10",
                    "where": (
                        f"model_group_id = {format_dimension_literal('model_group_id', mg0)} "
                        f"AND {dim} = {format_dimension_literal(dim, t1)}"
                    ),
                }
            )
        else:
            steps.append(
                {
                    "op": "calculate",
                    "expression": "value * 10",
                    "where": f"{dim} = {format_dimension_literal(dim, t0)}",
                }
            )
        steps.append({"op": "group", "by": group_by, "aggregate": "sum"})

        data = {
            "inputs": {"x": dataset_input_selection(self.dataset, self.filters)},
            "steps": steps,
            "output": {"format": "csv"},
        }
        return parse_pipeline(data)
