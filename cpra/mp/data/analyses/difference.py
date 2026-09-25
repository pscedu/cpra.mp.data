"""`difference` -- a dataset-scoped analysis over two selected values of one dimension:
negate whichever of the two is later (`order="asc"`) or earlier (`order="desc"`), then sum
across it. Replaces the portal's `subtract-asc`/`subtract-desc` catalog entries (see
`cpra-mp-ui/src/helpers/catalog.ts`'s real derivation, lines 95-138) with a named pipeline
the server itself validates and compiles, rather than a client-side-only convention.
"""

from typing import Literal

from pydantic import field_validator

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


class Difference(Analysis):
    class Meta:
        name = "difference"
        label = "Difference"
        type = "dataset"
        variables = None
        legend = None
        result_kind = "difference"

    class Parameters(Analysis.Parameters):
        dimension: str
        order: Literal["asc", "desc"]

        @field_validator("dimension")
        @classmethod
        def _dimension_must_be_filterable(cls, value: str) -> str:
            if not is_filterable_dimension(value):
                raise ValueError(f"{value!r} is not a filterable dimension")
            return value

    @classmethod
    def validate_selection(cls, parameters: "Difference.Parameters", filters: dict) -> None:
        dim = parameters.dimension
        raw = filters.get(dim)
        if raw is None:
            raise InvalidAnalysisParametersError(
                f"difference on dimension {dim!r} requires a {dim!r} filter selecting "
                "exactly 2 distinct values, but none was given"
            )
        values = normalized_filter_values(dim, raw)
        if len(values) != 2:
            raise InvalidAnalysisParametersError(
                f"difference on dimension {dim!r} requires exactly 2 distinct selected "
                f"values, got {len(values)}: {raw!r}"
            )

    def pipeline(self):
        assert self.dataset is not None, "difference is a type='dataset' analysis"
        dim = self.parameters.dimension
        # `validate_selection` (called by `resolve_analysis` before `pipeline()` is ever
        # reached) already guarantees `filters[dim]` normalizes to exactly 2 distinct
        # values -- re-derive here rather than trust a caller who skipped that step, since
        # a direct `Difference(...).pipeline()` call (e.g. a test) has no such guarantee.
        values = normalized_filter_values(dim, self.filters[dim])
        if len(values) != 2:
            raise InvalidAnalysisParametersError(
                f"difference on dimension {dim!r} requires exactly 2 distinct selected "
                f"values, got {len(values)}: {self.filters[dim]!r}"
            )
        earlier, later = sorted(values)
        negate_value = later if self.parameters.order == "asc" else earlier

        required = required_dataset_dimensions(self.dataset)
        group_by = [d for d in required if d != dim]

        steps: list[dict] = []
        where = selection_filter_where(self.filters, grid_cell_where=self.grid_cell_where)
        if where is not None:
            steps.append({"op": "filter", "where": where})
        # Negate one of the two values and sum: `later - earlier` said as `(-later) +
        # earlier`, which is what lets the subtraction ride on `group`'s existing `sum`.
        # `else: "value"` is the pass-through the other value needs -- without it the
        # non-negated rows would go null and the sum would have nothing to add.
        steps.append(
            {
                "op": "case",
                "cases": [
                    {
                        "when": f"{dim} = {format_dimension_literal(dim, negate_value)}",
                        "then": "value * -1",
                    }
                ],
                "else": "value",
            }
        )
        steps.append({"op": "group", "by": group_by, "aggregate": "sum"})
        # A pipeline's FIRST step must name its input explicitly; every later step chains
        # from the one before it. The leading `filter` normally carries that name, but a
        # caller invoking `pipeline()` directly with no filters at all (the API path
        # always supplies the two-value `dimension` filter `validate_selection` demands)
        # leaves the `case` first, so the name goes there instead.
        steps[0]["input"] = "x"

        data = {
            "inputs": {"x": dataset_input_selection(self.dataset)},
            "steps": steps,
            # Derived from the dataset's grid kind rather than fixed: `Meta.variables` is
            # `None` here, so this analysis really does run on both kinds, and it neither
            # projects nor otherwise changes shape -- a vector selection ends tabular, a
            # raster one band-wide, and `validate_shapes` rejects the mismatched sink
            # either way. Only `pipeline.json` ever shows the value, since the
            # `data.{format}`/tile routes overwrite it from their own URL segment, but what
            # `pipeline.json` publishes has to still validate if a caller resubmits it.
            "output": {"format": dataset_output_format(self.dataset)},
        }
        return parse_pipeline(data)

    @property
    def title_suffix(self) -> str:
        return " Difference"
