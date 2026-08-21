import datetime as dt
import itertools
import json
import re
from typing import final

import altair as alt

from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.visualizations.base import DatasetVisualization
from cpra.mp.data.visualizations.display import display_number_format


class StackedBar(DatasetVisualization):
    """Stacked bar chart for a dataset, colored/shaded by up to two compared dimensions."""

    @final
    class Meta:
        name = "stacked-bar"
        label = "Stacked Bar"
        type = "dataset"
        max_grid_cells = 1

    def chart(self):
        """Build the visualization using Altair."""

        start_date = self.params.get("start_date", None) or dt.date(2025, 1, 1)
        end_date = self.params.get("end_date", None) or dt.date(2026, 1, 1)
        initial_date_range = (start_date, end_date)
        is_daily = self.time_column == "calendar_day"

        negate_param = self.params.get("negate")
        negate_dimension = re.match(r"\w+", negate_param).group() if negate_param else None

        # Dimensions with more than one selected value (excluding the negate
        # dimension) are compared. Unlike TimeSeriesLine (3 channels), this chart
        # supports at most 2, mapped onto Color and Opacity — Vega-Lite stacks bar
        # segments automatically by the combination of discrete channels present.
        compare_dimensions = {
            d: self.params[d]
            for d in DIMENSIONS.keys()
            if isinstance(self.params.get(d), (list, tuple)) and len(self.params.get(d)) > 1 and d != negate_dimension
        }
        compare_channels = [alt.Color, alt.Opacity]
        compare_dimensions = {
            k: compare_dimensions[k] for k in list(compare_dimensions.keys())[: len(compare_channels)]
        }
        compare_choices = list(itertools.product(*compare_dimensions.values())) or [()]
        compare_choice_indices = list(itertools.product(*[range(len(v)) for v in compare_dimensions.values()])) or [()]

        # `size_to_container` (see base.py) reinterprets width/height as the total
        # container size and lets Vega-Lite's autosize give the leftover to the plot area
        # after measuring real axis/title/legend chrome, so no chrome allowance is guessed
        # here. The daily/vconcat branch is the exception: autosize only fits width for a
        # vconcat spec, so the summary strip's 60px plus a fixed allowance still has to be
        # subtracted from height by hand.
        chart_width = self.width
        chart_height = max(self.height - 222, 50) if is_daily else self.height

        # For daily data, bars are keyed on the continuous `date:T` field (same as
        # TimeSeriesLine) so it can be brushed/filtered below; annual/decadal data
        # is keyed on the raw time column ordinally, with zero band padding, so
        # bars touch.
        time_field = "date" if is_daily else self.time_column

        base = self.get_base_chart(self.data_source, self.time_column)

        # Map each dimension value to its index using loose equality (==) rather than
        # transform_lookup. Vega's transform_lookup coerces the key with String(),
        # which breaks numeric-vs-string matching (e.g. 0 vs "0.0"); loose == does
        # not. (Same workaround TimeSeriesLine uses for its rules/tooltip pivot.)
        for dim_name, dim_values in compare_dimensions.items():
            idx_expr = (
                " : ".join(f"datum.{dim_name} == {json.dumps(v)} ? {i}" for i, v in enumerate(dim_values)) + " : null"
            )
            base = base.transform_calculate(**{f"{dim_name}_idx": idx_expr})

        # Combine per-dimension indices into a single stack/legend order key, so bar
        # segments stack in the declared dimension-value order rather than whatever
        # default order Vega-Lite would otherwise pick.
        sizes = [len(v) for v in compare_dimensions.values()]
        idx_terms = [f"datum.{d}_idx" for d in compare_dimensions]
        order_idx_expr = idx_terms[0] if idx_terms else "0"
        for i in range(1, len(idx_terms)):
            order_idx_expr = f"({order_idx_expr}) * {sizes[i]} + {idx_terms[i]}"
        base = base.transform_calculate(order_idx=order_idx_expr)

        x = alt.X(f"{time_field}:{'T' if is_daily else 'O'}", axis=alt.Axis(grid=True)).title(
            self.time_dimension.label
        )
        if not is_daily:
            x = x.scale(paddingInner=0, paddingOuter=0)

        main_bar_encodings = [
            x,
            alt.Y("value:Q").title(self.variable_label).stack("zero"),
            alt.Order("order_idx:Q"),
        ] + [
            compare_channels[i](f"{dimension}:N").title(DIMENSIONS[dimension].label)
            for i, dimension in enumerate(compare_dimensions.keys())
        ]

        bars = base.mark_bar().encode(*main_bar_encodings).properties(title=self.title, width=chart_width)

        # A hover rule per time period, pivoting every compared dimension
        # combination's value into its own column so the tooltip shows the whole
        # bar's stack at once, not just the segment under the pointer — mirrors
        # TimeSeriesLine's hover-rule/pivot pattern.
        nearest = alt.selection_point("hover", nearest=True, on="pointerover", fields=[time_field], empty=False)
        when_near = alt.when(nearest)

        pivot_key_parts = [f"{d}_idx" for d in compare_dimensions]
        tooltip_key_expr = " + '_' + ".join(f"datum.{col}" for col in pivot_key_parts) if pivot_key_parts else "'value'"

        tooltips = [self.get_time_tooltip(self.time_column)] + [
            alt.Tooltip(
                "_".join(str(i) for i in idx_choice) if idx_choice else "value",
                type="quantitative",
                title=", ".join(str(v) for v in val_choice) if val_choice else self.variable_label,
                format=display_number_format(self.variable),
            )
            for val_choice, idx_choice in zip(compare_choices, compare_choice_indices)
        ]

        # groupby must include self.time_column too (not just time_field): the
        # time tooltip below references the raw time column, and transform_pivot
        # drops every field not in groupby — for daily data time_field is the
        # separate "date" field, so without this the tooltip's field disappears.
        pivot_groupby = list(dict.fromkeys([time_field, self.time_column]))
        rules = (
            base.transform_calculate(tooltip=tooltip_key_expr)
            .transform_pivot("tooltip", value="value", groupby=pivot_groupby)
            .mark_rule(color="gray")
            .encode(
                x=f"{time_field}:{'T' if is_daily else 'O'}",
                opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
                tooltip=tooltips,
            )
            .add_params(nearest)
        )

        layer = [bars, rules]
        # Difference charts can have negative values; Vega-Lite stacks negative bar
        # segments below zero automatically, so add a zero-reference rule for
        # visual parity with TimeSeriesLine's negate handling.
        if self.params.get("negate"):
            layer.insert(0, base.mark_rule(color="lightGray", size=2).encode(y=alt.datum(0)))

        viz = alt.layer(*layer).properties(width=chart_width, height=chart_height)

        if is_daily:
            # Show only the brushed date range in the main chart above, with a
            # compact monthly-mean overview strip below to select the range —
            # same pattern as TimeSeriesLine's daily brush/summary chart.
            brush = alt.selection_interval("date_range", encodings=["x"], value={"x": initial_date_range})
            viz = viz.transform_filter(brush)
            summary_bars = (
                bars.encode(
                    alt.X(f"yearmonth({self.time_column}):T").title("Month"),
                    alt.Y("mean(value):Q").title(self.variable_label).stack("zero"),
                )
                .properties(title=f"Monthly {self.title}", height=60, width=chart_width)
                .add_params(brush)
            )
            viz = viz & summary_bars

        return self.size_to_container(viz, is_concat=is_daily)
