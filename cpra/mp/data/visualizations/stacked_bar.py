import datetime as dt
import itertools
import json
import re
from functools import cached_property
from typing import final

import altair as alt

from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.visualizations.base import DatasetVisualization
from cpra.mp.data.visualizations.display import display_number_format

# Discrete channels a compared dimension can be mapped onto, in the order they're used.
# Vega-Lite stacks bar segments by the combination of discrete channels present, so the
# length of this list is also the number of dimensions this chart can compare at once
# (unlike TimeSeriesLine, which has three).
COMPARE_CHANNELS = [alt.Color, alt.Opacity]


class StackedBar(DatasetVisualization):
    """Stacked bar chart for a dataset, colored/shaded by up to two compared dimensions.

    `chart` is the skeleton every stacked bar shares: the time axis (ordinal, or continuous
    and brushable for daily data), the stacked value axis, a hover rule per period showing
    the whole stack, the zero reference a difference chart needs, and the container sizing.
    What a subclass changes is what the *series* are, which is four seams -- what defines
    them (`series_transforms`), how they're distinguished (`series_channels`), what order
    they stack in (`order_channel`), and how the hover rule reports them
    (`hover_transforms`, `tooltip_encoding`). `SpeciesStackedBar` replaces exactly those and
    inherits the rest.
    """

    @final
    class Meta:
        name = "stacked-bar"
        label = "Stacked Bar"
        type = "dataset"
        max_geographies = 1

    @cached_property
    def is_daily(self):
        """Daily data gets a continuous, brushable axis and a summary strip to brush on."""
        return self.time_column == "calendar_day"

    @cached_property
    def time_field(self):
        """The field the bars are keyed on.

        For daily data that's the continuous `date:T` field (same as TimeSeriesLine) so it
        can be brushed and filtered; annual/decadal data is keyed on the raw time column
        ordinally, with zero band padding, so the bars touch.
        """
        return "date" if self.is_daily else self.time_column

    @cached_property
    def negate_dimension(self):
        """The dimension a difference chart is taking its difference across, if any."""
        negate_param = self.params.get("negate")
        return re.match(r"\w+", negate_param).group() if negate_param else None

    @cached_property
    def compare_dimensions(self):
        """Selected dimensions with more than one value, mapped onto `COMPARE_CHANNELS`.

        The negate dimension is excluded: its two values are the difference itself, not two
        series to draw. Anything past the available channels is dropped rather than silently
        collapsed into an unreadable stack.
        """
        compared = {
            d: self.params[d]
            for d in DIMENSIONS.keys()
            if isinstance(self.params.get(d), (list, tuple))
            and len(self.params.get(d)) > 1
            and d != self.negate_dimension
        }
        return {k: compared[k] for k in list(compared.keys())[: len(COMPARE_CHANNELS)]}

    @cached_property
    def compare_choices(self):
        """Every combination of compared dimension values, as labels."""
        return list(itertools.product(*self.compare_dimensions.values())) or [()]

    @cached_property
    def compare_choice_indices(self):
        """The same combinations as indices, which is how the pivot names its columns."""
        return list(itertools.product(*[range(len(v)) for v in self.compare_dimensions.values()])) or [()]

    @cached_property
    def chart_width(self):
        return self.width

    @cached_property
    def chart_height(self):
        """`size_to_container` (see base.py) reinterprets width/height as the total container
        size and lets Vega-Lite's autosize give the leftover to the plot area after measuring
        real axis/title/legend chrome, so no chrome allowance is guessed here. The
        daily/vconcat branch is the exception: autosize only fits width for a vconcat spec,
        so the summary strip's 60px plus a fixed allowance still has to be subtracted by
        hand."""
        return max(self.height - 222, 50) if self.is_daily else self.height

    @cached_property
    def x_encoding(self):
        """The time axis, continuous for daily data and ordinal otherwise."""
        x = alt.X(f"{self.time_field}:{'T' if self.is_daily else 'O'}", axis=alt.Axis(grid=True)).title(
            self.time_dimension.label
        )
        return x if self.is_daily else x.scale(paddingInner=0, paddingOuter=0)

    def series_transforms(self, base):
        """Attach the transforms that turn rows into series.

        Each compared dimension value becomes an index, and the indices combine into a single
        `order_idx`, so segments stack in the declared dimension-value order rather than
        whatever default order Vega-Lite would otherwise pick.

        The indices are calculated with loose equality (==) rather than `transform_lookup`,
        whose Vega implementation coerces the key with String() and so breaks numeric-vs-
        string matching (e.g. 0 vs "0.0"). (Same workaround TimeSeriesLine uses.)
        """
        for dim_name, dim_values in self.compare_dimensions.items():
            idx_expr = (
                " : ".join(f"datum.{dim_name} == {json.dumps(v)} ? {i}" for i, v in enumerate(dim_values)) + " : null"
            )
            base = base.transform_calculate(**{f"{dim_name}_idx": idx_expr})

        sizes = [len(v) for v in self.compare_dimensions.values()]
        idx_terms = [f"datum.{d}_idx" for d in self.compare_dimensions]
        order_idx_expr = idx_terms[0] if idx_terms else "0"
        for i in range(1, len(idx_terms)):
            order_idx_expr = f"({order_idx_expr}) * {sizes[i]} + {idx_terms[i]}"
        return base.transform_calculate(order_idx=order_idx_expr)

    @cached_property
    def series_channels(self):
        """The channels that tell one series from another."""
        return [
            COMPARE_CHANNELS[i](f"{dimension}:N").title(DIMENSIONS[dimension].label)
            for i, dimension in enumerate(self.compare_dimensions.keys())
        ]

    @cached_property
    def order_channel(self):
        """What the segments stack in order of, from the baseline up."""
        return alt.Order("order_idx:Q")

    def hover_transforms(self, base):
        """Pivot each series into its own column, so the hover tooltip can show the whole
        bar's stack at once rather than just the segment under the pointer -- mirrors
        TimeSeriesLine's hover-rule/pivot pattern.

        groupby must include `self.time_column` as well as `time_field`: the time tooltip
        references the raw time column and `transform_pivot` drops every field not in
        groupby, so for daily data (where `time_field` is the separate "date" field) the
        tooltip's own field would otherwise disappear.
        """
        pivot_key_parts = [f"{d}_idx" for d in self.compare_dimensions]
        tooltip_key_expr = " + '_' + ".join(f"datum.{col}" for col in pivot_key_parts) if pivot_key_parts else "'value'"
        groupby = list(dict.fromkeys([self.time_field, self.time_column]))
        return base.transform_calculate(tooltip=tooltip_key_expr).transform_pivot(
            "tooltip", value="value", groupby=groupby
        )

    @cached_property
    def tooltip_encoding(self):
        """One row per compared dimension combination, which this chart knows up front."""
        return [self.get_time_tooltip(self.time_column)] + [
            alt.Tooltip(
                "_".join(str(i) for i in idx_choice) if idx_choice else "value",
                type="quantitative",
                title=", ".join(str(v) for v in val_choice) if val_choice else self.variable_label,
                format=display_number_format(self.variable),
            )
            for val_choice, idx_choice in zip(self.compare_choices, self.compare_choice_indices)
        ]

    def hover_rules(self, base, when_near):
        """A rule per time period, visible only when the pointer is nearest to it."""
        return (
            self.hover_transforms(base)
            .mark_rule(color="gray")
            .encode(
                x=f"{self.time_field}:{'T' if self.is_daily else 'O'}",
                opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
                tooltip=self.tooltip_encoding,
            )
        )

    def chart(self):
        """Build the visualization using Altair."""

        base = self.series_transforms(self.get_base_chart(self.data_source, self.time_column))

        bars = (
            base.mark_bar()
            .encode(
                self.x_encoding,
                alt.Y("value:Q").title(self.variable_label).stack("zero"),
                self.order_channel,
                *self.series_channels,
            )
            .properties(title=self.title, width=self.chart_width)
        )

        nearest = alt.selection_point(
            "hover", nearest=True, on="pointerover", fields=[self.time_field], empty=False
        )
        rules = self.hover_rules(base, alt.when(nearest)).add_params(nearest)

        layer = [bars, rules]
        # Difference charts can have negative values; Vega-Lite stacks negative bar
        # segments below zero automatically, so add a zero-reference rule for
        # visual parity with TimeSeriesLine's negate handling.
        if self.params.get("negate"):
            layer.insert(0, base.mark_rule(color="lightGray", size=2).encode(y=alt.datum(0)))

        viz = alt.layer(*layer).properties(width=self.chart_width, height=self.chart_height)

        if self.is_daily:
            # Show only the brushed date range in the main chart above, with a
            # compact monthly-mean overview strip below to select the range —
            # same pattern as TimeSeriesLine's daily brush/summary chart.
            start_date = self.params.get("start_date", None) or dt.date(2025, 1, 1)
            end_date = self.params.get("end_date", None) or dt.date(2026, 1, 1)
            brush = alt.selection_interval("date_range", encodings=["x"], value={"x": (start_date, end_date)})
            viz = viz.transform_filter(brush)
            summary_bars = (
                bars.encode(
                    alt.X(f"yearmonth({self.time_column}):T").title("Month"),
                    alt.Y("mean(value):Q").title(self.variable_label).stack("zero"),
                )
                .properties(title=f"Monthly {self.title}", height=60, width=self.chart_width)
                .add_params(brush)
            )
            viz = viz & summary_bars

        return self.size_to_container(viz, is_concat=self.is_daily)
