import datetime as dt
import itertools
import json
from typing import final

import altair as alt
import polars as pl

from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.visualizations.base import DatasetVisualization
from cpra.mp.data.visualizations.display import display_number_format


class TimeSeriesLine(DatasetVisualization):
    """Time series line chart for a dataset."""

    @final
    class Meta:
        name = "time-series-line"
        label = "Time Series Line"
        type = "dataset"
        max_grid_cells = 10

    def chart(self):
        """Build the visualization using Altair."""

        start_date = self.params.get("start_date", None) or dt.date(2025, 1, 1)
        end_date = self.params.get("end_date", None) or dt.date(2026, 1, 1)
        initial_date_range = (start_date, end_date)
        is_daily = self.time_column == "calendar_day"

        # Which dimensions the resolved analysis's own pipeline already collapsed away
        # (e.g. `diff`'s two selected values of its own `dimension` param). A
        # collapsed dimension has nothing left to compare across; with no analysis
        # selected, nothing is collapsed.
        resolved = self.resolved_analysis
        excluded_dimensions = set(resolved.collapsed_dimensions) if resolved is not None else set()

        compare_dimensions = {
            d: self.params[d]
            for d in DIMENSIONS
            if isinstance(self.params.get(d), (list, tuple))
            and len(self.params.get(d)) > 1
            and d not in excluded_dimensions
        }
        # Limit compare dimensions to the number of available channels.
        # TODO: Adjust params to remove additional compare dimensions from the data.
        compare_channels = [alt.StrokeDash, alt.Opacity, alt.StrokeWidth]
        compare_dimensions = {
            k: compare_dimensions[k] for k in list(compare_dimensions.keys())[: len(compare_channels)]
        }
        compare_choices = list(itertools.product(*compare_dimensions.values()))
        compare_choice_indices = list(itertools.product(*[range(len(v)) for v in compare_dimensions.values()]))

        # `size_to_container` (see base.py) reinterprets width/height as the total
        # container size and lets Vega-Lite's autosize give the leftover to the plot area
        # after measuring real axis/title/legend chrome, so no chrome allowance is guessed
        # here. The daily/vconcat branch is the exception: autosize only fits width for a
        # vconcat spec, so the summary strip's 60px plus a fixed allowance still has to be
        # subtracted from height by hand.
        chart_width = self.width
        chart_height = max(self.height - 222, 50) if is_daily else self.height

        nearest = alt.selection_point("hover", nearest=True, on="pointerover", fields=["date"], empty=False)
        when_near = alt.when(nearest)

        # The data is keyed by whatever the API reports as `grid_cell`, but the legend shows
        # `grid_cell_label` — a coordinate selection carries the resolved grid cell ID, which
        # is what actually identifies the values being drawn. The row index doubles as the
        # tooltip's column key below, so both label lists line up with it.
        grid_cell_label_df = pl.DataFrame(
            {"grid_cell": self.grid_cells, "grid_cell_label": self.grid_cell_labels}
        ).with_row_index(name="grid_cell_label_idx")

        main_line_encodings = [
            alt.X("date:T", axis=alt.Axis(grid=True)).title(self.time_dimension.label),
            # `self.data_source.name`, not `self.variable.name`: the expression has to name
            # the Vega dataset this chart actually reads, and `get_data_source` names it
            # `variable or "value"` -- a result that folds several variables (or fixes its
            # own inputs, like `lnd_type_plus`) has no single variable to be named after, so
            # `self.variable` is `None` there and the source is called "value". Re-deriving
            # the name from the variable both crashed on that `None` and would have pointed
            # `data()` at a dataset that does not exist whenever the two disagreed.
            alt.Y("value:Q", axis=alt.Axis(grid=True))
            .title(self.variable_label)
            .scale(domainMax=alt.expr(f"extent(pluck(data('{self.data_source.name}'), 'value'))[1]")),
            # Each series is colored as the map colors the grid cell it draws, rather than by
            # its position in the legend — see `grid_cell_colors`, which is why the palette
            # isn't handed over whole.
            alt.Color("grid_cell_label:N")
            .title(self.grid.label)
            .scale(domain=self.grid_cell_labels, range=self.grid_cell_colors),
        ] + [
            compare_channels[i](f"{dimension}:N").title(DIMENSIONS[dimension].label)
            for i, dimension in enumerate(compare_dimensions.keys())
        ]

        main_line = (
            self.get_base_chart(self.data_source, self.time_column)
            .transform_lookup(
                "grid_cell",
                from_=alt.LookupData(grid_cell_label_df, "grid_cell", ["grid_cell_label"]),
            )
            .mark_line()
            .encode(*main_line_encodings)
            .properties(title=self.title, width=chart_width)
        )

        points = main_line.mark_point().encode(opacity=when_near.then(alt.value(1)).otherwise(alt.value(0)))

        grid_cell_df = pl.DataFrame({"grid_cell": self.grid_cells}).with_row_index(name="grid_cell_idx")
        tooltips = [self.get_time_tooltip(self.time_column)] + [
            alt.Tooltip(
                "_".join([str(c) for c in [i] + list(idx_choices)]),
                type="quantitative",
                title=f"{label}: {', '.join([str(c) for c in val_choices])}",
                format=display_number_format(self.variable),
            )
            # Short labels here: a tooltip carries a row per grid cell, so a full label
            # would push the numbers being read off to the right.
            for i, label in enumerate(self.grid_cell_short_labels)
            for val_choices, idx_choices in zip(compare_choices, compare_choice_indices)
        ]

        # Build the rules chart. Use calculate transforms with loose equality (==)
        # rather than lookup to map dimension values to indices. Vega's transform_lookup
        # uses String() coercion on the key, so the number 0 (from JSON's 0.0) becomes
        # "0" and fails to match the CSV string "0.0". Loose == coerces correctly.
        rules = self.get_base_chart(self.data_source, self.time_column).transform_lookup(
            "grid_cell",
            from_=alt.LookupData(grid_cell_df, "grid_cell", ["grid_cell_idx"]),
        )
        for dim_name, dim_values in compare_dimensions.items():
            idx_expr = (
                " : ".join([f"datum.{dim_name} == {json.dumps(v)} ? {i}" for i, v in enumerate(dim_values)]) + " : null"
            )
            rules = rules.transform_calculate(**{f"{dim_name}_idx": idx_expr})
        pivot_key_parts = ["grid_cell_idx"] + [f"{d}_idx" for d in compare_dimensions]
        rules = (
            rules.transform_calculate(tooltip=" + '_' + ".join([f"datum.{col}" for col in pivot_key_parts]))
            .transform_pivot("tooltip", value="value", groupby=["date", self.time_column])
            .mark_rule(color="gray")
            .encode(
                x="date:T",
                opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
                tooltip=tooltips,
            )
            .add_params(nearest)
        )

        layer = [main_line, points, rules]

        # Add a horizontal rule at zero for difference charts.
        if resolved is not None and resolved.result_kind == "difference":
            layer.insert(
                0,
                self.get_base_chart(self.data_source, self.time_column)
                .mark_rule(color="lightGray", size=2)
                .encode(y=alt.datum(0)),
            )

        viz = (alt.layer(*layer)).properties(width=chart_width, height=chart_height)

        if is_daily:
            brush = alt.selection_interval("date_range", encodings=["x"], value={"x": initial_date_range})
            viz = viz.transform_filter(brush)
            summary_line = (
                main_line.encode(
                    alt.X("yearmonth(calendar_day):T").title("Month"),
                    alt.Y("mean(value):Q").title(self.variable_label),
                )
                .properties(title=f"Monthly {self.title}", height=60, width=chart_width)
                .add_params(brush)
            )
            viz = viz & summary_line

        return self.size_to_container(viz, is_concat=is_daily)
