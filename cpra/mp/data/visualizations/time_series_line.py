import datetime as dt
import itertools
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

        compare_dimensions = {
            d: self.params[d]
            for d in DIMENSIONS.keys()
            if isinstance(self.params.get(d), (list, tuple)) and len(self.params.get(d)) > 1
        }
        # Limit compare dimensions to the number of available channels.
        # TODO: Adjust params to remove additional compare dimensions from the data.
        compare_channels = [alt.StrokeDash, alt.Opacity, alt.StrokeWidth]
        compare_dimensions = {
            k: compare_dimensions[k] for k in list(compare_dimensions.keys())[: len(compare_channels)]
        }
        compare_choices = list(itertools.product(*compare_dimensions.values()))

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
                "_".join([str(c) for c in [i] + list(choices)]),
                type="quantitative",
                title=f"{str(geo)}: {', '.join([str(c) for c in choices])}",
                format=self.variable.number_format or "",
            )
            for (i, geo) in geography_df.iter_rows()
            for choices in compare_choices
        ]

        rules = (
            self.get_base_chart(self.data_source, self.time_column)
            .transform_lookup(  # Transform geography to index to support coordinates
                "geography",
                from_=alt.LookupData(
                    geography_df,
                    "geography",
                    ["geography_idx"],
                ),
            )
            .transform_calculate(
                tooltip=" + '_' + ".join(
                    [f"datum.{col}" for col in ["geography_idx"] + list(compare_dimensions.keys())]
                )
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
