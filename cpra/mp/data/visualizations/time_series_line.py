import datetime as dt

import altair as alt

from cpra.mp.data.visualizations.base import DatasetVisualization
from cpra.mp.data.visualizations.colors import MAP_SELECTION


class TimeSeriesLine(DatasetVisualization):
    name = "time-series-line"
    label = "Time Series Line"

    def chart(self):
        """Build the visualization using Altair."""

        start_date = self.params.get("start_date", dt.date(2019, 1, 1))
        end_date = self.params.get("end_date", dt.date(2020, 1, 1))
        initial_date_range = (start_date, end_date)
        is_daily = self.time_column == "calendar_day"

        chart_height = self.height - (222 if is_daily else 86)
        chart_width = self.width - 200

        nearest = alt.selection_point(nearest=True, on="pointerover", fields=[self.time_column], empty=False)

        main_line = (
            alt.Chart(self.data_source)
            .mark_line()
            .encode(
                alt.X(f"{self.time_column}:T").title("Date"),
                alt.Y("value:Q").title(self.variable_label),
                alt.Color("geography:N")
                .title(self.grid.label)
                .scale(domain=[str(geo) for geo in self.geographies], range=MAP_SELECTION),
            )
            .properties(title=f"{self.time_unit.title()} {self.title}", width=chart_width)
        )

        when_near = alt.when(nearest)
        points = main_line.mark_point().encode(opacity=when_near.then(alt.value(1)).otherwise(alt.value(0)))

        rules = (
            alt.Chart(self.data_source)
            .transform_pivot("geography", value="value", groupby=[self.time_column])
            .mark_rule(color="gray")
            .encode(
                x=f"{self.time_column}:T",
                opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
                tooltip=[alt.Tooltip(self.time_column, type="temporal")]
                + [alt.Tooltip(str(c), type="quantitative") for c in self.geographies],
            )
            .add_params(nearest)
        )

        viz = (alt.layer(main_line, points, rules)).properties(width=chart_width, height=chart_height)

        if is_daily:
            brush = alt.selection_interval(encodings=["x"], value={"x": initial_date_range})
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
