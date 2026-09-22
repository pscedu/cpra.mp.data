"""`concatenate` -- a dataset-scoped analysis over two selected values of one dimension,
producing a categorical land-type-change code (`variables=["lnd_type"]`,
`legend="lnd_chng"`). Replaces the portal's `concatenate` catalog entry (see
`cpra-mp-ui/src/helpers/catalog.ts`'s real derivation, lines 95-138).

Two modes, selected by how many `model_group_id` values are selected alongside
`dimension`'s own two values (see `catalog.ts`'s `threeImage = modelGroupValues.length ===
2`, NOT `dimension`'s own cardinality -- `dimension` is always exactly 2 for both modes):

* **two-image** (`model_group_id` absent, or selected to exactly 1 value): `calculate`
  `value * 10` where `dimension = t0` (the earlier of the two selected values), then
  `group`/`sum` over every other required dimension.
* **three-image** (`model_group_id` selected to exactly 2 values, `mg0 < mg1`): a `filter`
  dropping `model_group_id = mg1 AND dimension = t0` (the redundant corner), then
  `calculate` `value * 100` where `dimension = t0`, then `calculate` `value * 10` where
  `model_group_id = mg0 AND dimension = t1`, then `group`/`sum` over every other required
  dimension (excluding `model_group_id`, which the calculates above have folded away).

Both modes' leading `filter` step is the one the caller's own dimension filters/
`grid_cell`/`where` compose into (`base.selection_filter_where`); in three-image mode the
redundant-corner conjunct is AND-ed onto that same step rather than emitted as a second
one, since the two are both dimension conditions applied at the same point in the chain.
The corner is written directly as `NOT (model_group_id = mg1 AND dimension = t0)` --
`expressions.classify_where_kinds` reads a `NOT (...)` group's root names the same way it
reads any other conjunct, so this needs no De Morgan's rewrite to be accepted.
"""

from cpra.mp.data.analyses.base import (
    Analysis,
    InvalidAnalysisParametersError,
    dataset_input_selection,
    dataset_output_format,
    format_dimension_literal,
    is_filterable_dimension,
    normalized_filter_values,
    required_dataset_dimensions,
    selection_filter_where,
)
from cpra.mp.data.analysis.request.pipeline import parse_pipeline
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

        conjuncts = []
        selection_where = selection_filter_where(self.filters, grid_cell_where=self.grid_cell_where)
        if selection_where is not None:
            conjuncts.append(selection_where)
        if three_image:
            # Drop the redundant corner of the 2x2 (model_group_id, dim) grid, so the
            # three surviving cells are what the two calculates below encode.
            conjuncts.append(
                f"NOT (model_group_id = {format_dimension_literal('model_group_id', mg1)} "
                f"AND {dim} = {format_dimension_literal(dim, t0)})"
            )

        steps: list[dict] = []
        if conjuncts:
            steps.append({"op": "filter", "where": " AND ".join(conjuncts)})
        if three_image:
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
        # A pipeline's first step must name its input explicitly; every later step chains
        # from the one before it. That is normally the leading `filter`, but falls to the
        # first `calculate` when there was nothing to filter on.
        steps[0]["input"] = "x"

        data = {
            "inputs": {"x": dataset_input_selection(self.dataset)},
            "steps": steps,
            # Derived from the dataset's grid kind rather than fixed: this analysis neither
            # projects nor otherwise changes shape, so a vector selection ends tabular and
            # a raster one ends band-wide, and `validate_shapes` rejects the mismatched
            # sink in either direction. Only `pipeline.json` ever shows this value -- the
            # `data.{format}`/tile routes overwrite it from their own URL segment -- but
            # what `pipeline.json` publishes has to be resubmittable, and this analysis's
            # own `Meta.variables` (`lnd_type`) lives only on a raster grid.
            "output": {"format": dataset_output_format(self.dataset)},
        }
        return parse_pipeline(data)
