import datetime as dt
import itertools
import json
import re
from typing import final

import altair as alt
import polars as pl

from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.visualizations.base import DatasetVisualization
from cpra.mp.data.visualizations.colors import MAP_SELECTION


class TimeSeriesLine(DatasetVisualization):
    """Time series line chart for a dataset."""

    @final
    class Meta:
        name = "time-series-line"
        label = "Time Series Line"
        type = "dataset"
        max_geographies = 10

    def chart(self):
        """Build the visualization using Altair."""

        start_date = self.params.get("start_date", None) or dt.date(2019, 1, 1)
        end_date = self.params.get("end_date", None) or dt.date(2020, 1, 1)
        initial_date_range = (start_date, end_date)
        is_daily = self.time_column == "calendar_day"

        negate_param = self.params.get("negate")
        negate_dimension = re.match(r"\w+", negate_param).group() if negate_param else None

        compare_dimensions = {
            d: self.params[d]
            for d in DIMENSIONS.keys()
            if isinstance(self.params.get(d), (list, tuple)) and len(self.params.get(d)) > 1 and d != negate_dimension
        }
        # Limit compare dimensions to the number of available channels.
        # TODO: Adjust params to remove additional compare dimensions from the data.
        compare_channels = [alt.StrokeDash, alt.Opacity, alt.StrokeWidth]
        compare_dimensions = {
            k: compare_dimensions[k] for k in list(compare_dimensions.keys())[: len(compare_channels)]
        }
        compare_choices = list(itertools.product(*compare_dimensions.values()))
        compare_choice_indices = list(itertools.product(*[range(len(v)) for v in compare_dimensions.values()]))

        chart_height = max(self.height - (222 if is_daily else 86), 50)
        chart_width = max(self.width - 200, 50)

        nearest = alt.selection_point("hover", nearest=True, on="pointerover", fields=["date"], empty=False)
        when_near = alt.when(nearest)

        main_line_encodings = [
            alt.X("date:T", axis=alt.Axis(grid=True)).title(self.time_dimension.label),
            alt.Y("value:Q", axis=alt.Axis(grid=True))
            .title(self.variable_label)
            .scale(domainMax=alt.expr(f"extent(pluck(data('{self.variable.name}'), 'value'))[1]")),
            alt.Color("geography:N")
            .title(self.grid.label)
            .scale(domain=[geo for geo in self.geographies], range=MAP_SELECTION),
        ] + [
            compare_channels[i](f"{dimension}:N").title(DIMENSIONS[dimension].label)
            for i, dimension in enumerate(compare_dimensions.keys())
        ]

        main_line = (
            self.get_base_chart(self.data_source, self.time_column)
            .mark_line()
            .encode(*main_line_encodings)
            .properties(title=self.title, width=chart_width)
        )

        points = main_line.mark_point().encode(opacity=when_near.then(alt.value(1)).otherwise(alt.value(0)))

        geography_df = pl.DataFrame(dict(geography=self.geographies)).with_row_index(name="geography_idx")
        tooltips = [self.get_time_tooltip(self.time_column)] + [
            alt.Tooltip(
                "_".join([str(c) for c in [i] + list(idx_choices)]),
                type="quantitative",
                title=f"{str(geo)}: {', '.join([str(c) for c in val_choices])}",
                format=self.variable.number_format or "",
            )
            for (i, geo) in geography_df.iter_rows()
            for val_choices, idx_choices in zip(compare_choices, compare_choice_indices)
        ]

        # Build the rules chart. Use calculate transforms with loose equality (==)
        # rather than lookup to map dimension values to indices. Vega's transform_lookup
        # uses String() coercion on the key, so the number 0 (from JSON's 0.0) becomes
        # "0" and fails to match the CSV string "0.0". Loose == coerces correctly.
        rules = self.get_base_chart(self.data_source, self.time_column).transform_lookup(
            "geography",
            from_=alt.LookupData(geography_df, "geography", ["geography_idx"]),
        )
        for dim_name, dim_values in compare_dimensions.items():
            idx_expr = " : ".join([
                f"datum.{dim_name} == {json.dumps(v)} ? {i}"
                for i, v in enumerate(dim_values)
            ]) + " : null"
            rules = rules.transform_calculate(**{f"{dim_name}_idx": idx_expr})
        pivot_key_parts = ["geography_idx"] + [f"{d}_idx" for d in compare_dimensions.keys()]
        rules = (
            rules
            .transform_calculate(
                tooltip=" + '_' + ".join([f"datum.{col}" for col in pivot_key_parts])
            )
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
        if self.params.get("negate"):
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

        return viz
