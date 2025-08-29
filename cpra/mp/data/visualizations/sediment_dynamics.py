import altair as alt
import polars as pl

from cpra.mp.data.legends import LEGENDS
from cpra.mp.data.variables import VARIABLES
from cpra.mp.data.visualizations.base import BaseVisualization


class SedimentDynamics(BaseVisualization):
    """A chart showing five sediment dynamics variables."""

    class Meta:
        name = "sediment-dynamics"
        label = "Sediment Dynamics"
        type = "custom"
        max_geographies = 1
        variables = ["elev", "inun", "lnd_type", "test_mnrl_accr", "test_org_accr"]

    def __init__(self, request=None, height=400, width=600, **kwargs):
        super().__init__(request, height, width, **kwargs)
        self.meter_variables = ["elev", "inun"]
        self.centimeter_variables = []
        self.line_variables = self.meter_variables + self.centimeter_variables
        self.variables = ["lnd_type"] + self.line_variables
        self.data_sources = {
            variable_name: self.get_data_source(
                variable_name,
                "morph_pixel_v001",
                "annual",
                **(self.params | dict(select=",".join(["calendar_year", "value"]))),
            )
            for variable_name in self.variables
        }

    def _base_chart(self, variables, long=False):
        chart: alt.Chart = None
        for variable_name in variables:
            data_source = self.data_sources[variable_name]
            if chart is None:
                chart = alt.Chart(data_source).transform_calculate(
                    date="toDate(toString(datum.calendar_year))", **{variable_name: "datum.value"}
                )
            else:
                chart = chart.transform_lookup(
                    "calendar_year",
                    as_=[variable_name],
                    from_=alt.LookupData(data_source, "calendar_year", ["value"]),
                )

        if long:
            chart = chart.transform_fold(variables, as_=["variable", "value"])
            variables_df = pl.DataFrame([{"name": v, "label": VARIABLES[v].label} for v in variables])
            chart = chart.transform_lookup(
                "variable", as_=["variable_label"], from_=alt.LookupData(variables_df, "name", ["label"])
            )

        return chart

    def _lnd_type_chart(self):
        lnd_type_legend = (
            LEGENDS["lnd_type"]
            .break_points.with_columns(pl.col("upper_bound").shift(1).floor().alias("lower_bound"))
            .filter(pl.col("lower_bound").is_not_null())
        )
        return (
            self._base_chart(["lnd_type"])
            .mark_area(interpolate="step-after", opacity=0.3)
            .transform_lookup(
                "value",
                from_=alt.LookupData(lnd_type_legend.select("label", "lower_bound"), "lower_bound", ["label", "color"]),
            )
            .transform_calculate(y="1")
            .encode(
                alt.X("date:T"),
                alt.Y("y:Q", axis=None),
                alt.Color("label:N")
                .title("Land Type")
                .scale(domain=lnd_type_legend["label"], range=lnd_type_legend["color"]),
            )
        )

    def _line_chart(self, variable_names):
        variables = [VARIABLES[v] for v in variable_names]
        return (
            self._base_chart(variable_names, long=True)
            .mark_line()
            .encode(
                alt.X("date:T").title("Year"),
                alt.Y("value:Q").title(", ".join([f"{v.label} ({v.unit})" for v in variables])),
                alt.Color("variable_label:N").title("Variable"),
            )
        )

    def chart(self):
        """Build the visualization using Altair."""
        lnd_type_chart = self._lnd_type_chart()
        meter_chart = self._line_chart(self.meter_variables)

        nearest = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False)
        when_near = alt.when(nearest)
        points_chart = (
            self._base_chart(self.line_variables, long=True)
            .mark_point(color="black")
            .encode(alt.X("date:T"), alt.Y("value:Q"), opacity=when_near.then(alt.value(1)).otherwise(alt.value(0)))
        )
        rule_chart = (
            self._base_chart(self.variables)
            .mark_rule(color="gray")
            .encode(
                x="date:T",
                opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
                tooltip=[alt.Tooltip("calendar_year", type="quantitative", title="Year")]
                + [
                    alt.Tooltip(VARIABLES[v].name, type="quantitative", title=VARIABLES[v].label)
                    for v in self.variables
                ],
            )
            .add_params(nearest)
        )

        return (
            alt.layer(lnd_type_chart, alt.layer(meter_chart, points_chart, rule_chart))
            .resolve_scale(y="independent", color="independent")
            .properties(title="Sediment Dynamics", width=1200, height=600)
        )
