import datetime as dt
from typing import final

import altair as alt
import polars as pl

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

        chart_height = max(self.height - (222 if is_daily else 86), 50)
        chart_width = max(self.width - 200, 50)

        nearest = alt.selection_point("hover", nearest=True, on="pointerover", fields=["date"], empty=False)
        when_near = alt.when(nearest)

        main_line = (
            self.get_base_chart(self.data_source, self.time_column)
            .mark_line()
            .encode(
                alt.X("date:T", axis=alt.Axis(grid=True)).title(self.time_dimension.label),
                alt.Y("value:Q", axis=alt.Axis(grid=True))
                .title(self.variable_label)
                .scale(domainMax=alt.expr(f"extent(pluck(data('{self.variable.name}'), 'value'))[1]")),
                alt.Color("geography:N")
                .title(self.grid.label)
                .scale(domain=[geo for geo in self.geographies], range=MAP_SELECTION),
            )
            .properties(title=self.title, width=chart_width)
        )

        points = main_line.mark_point().encode(opacity=when_near.then(alt.value(1)).otherwise(alt.value(0)))

        geography_df = pl.DataFrame(dict(geography=self.geographies)).with_row_index(name="geography_idx")

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
            .transform_pivot("geography_idx", value="value", groupby=["date", self.time_column])
            .mark_rule(color="gray")
            .encode(
                x="date:T",
                opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
                tooltip=[self.get_time_tooltip(self.time_column)]
                + [
                    alt.Tooltip(str(i), type="quantitative", title=str(geo), format=self.variable.number_format or "")
                    for (i, geo) in geography_df.iter_rows()
                ],
            )
            .add_params(nearest)
        )

        layer = [main_line, points, rules]

        # Add a horizontal rule at zero for difference charts.
        if self.params["negate"]:
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
