"""The Sediment Dynamics chart: five morphology surfaces at one pixel, over time.

An ordinary `DatasetVisualization` over ONE data source, which is the whole change here.
This used to be the only visualization that built its own data sources -- five of them, one
per variable, each with its own `/v1/data.csv` URL -- and then glued them back together in
the browser with four `transform_lookup`s keyed on `calendar_year` and a `transform_fold`
to stack the result. The `sediment_dynamics` named analysis does that server-side now
(`analyses/sediment_dynamics.py`), so the source arrives already stacked: one row per
(year, variable), with the variable name in a `variable` column and the measurement in
`value`.

What that retires, beyond the five requests: the client-side join needed a `grid_cell`
column in every source to key on, and it needed the time column named `calendar_year`
outright -- a lookup key cannot be chosen per request -- so this chart was annual-only by
construction. Both are gone. `get_time_column(time_unit)` answers the time column now, and
the grid and time unit come from the descriptor rather than from the `morph_pixel_v001` /
`annual` pair this file used to hardcode.

**The analysis is implied, not asked for.** `Meta.analysis` names it, and
`DatasetVisualization.request_items`/`analysis_chain` put it at the head of the chain for
both the route's decode and a notebook's -- so a client that simply asks for this chart over
one
of the five variables gets it, without having to know that a named analysis backs it. A
caller who names an analysis of their own keeps it, appended after this one (`analysis=
sediment_dynamics,diff` draws the five surfaces' differences).

**Three unit families, one source, so the split moved onto the `variable` column.** Metres
(`elev`, `inun`) and centimetres (`test_mnrl_accr`, `test_org_accr`) get a line each on two
independent y-scales, and `lnd_type` is a categorical band behind them. Those were five
named sources before, which is what let `get_base_chart` apply each variable's own
`display_scale_factor` on the way in. One source has one `value` column and no such hook,
so the scaling is a `transform_calculate` conditional on `datum.variable` (`_value_expr`),
and each layer selects its rows with a `transform_filter` on the same column.

**The rule layer pivots back.** A hover tooltip shows all five measurements for one year,
which is one ROW with five fields -- the shape the fold undid. Vega-Lite's `pivot`
transform rebuilds it in the browser, on the one layer that needs it, from the same source
every other layer reads. That is not a step backwards to the client-side join it replaced:
a pivot has no second source to fetch and no key to agree on, and it runs AFTER the display
scaling, which is what fixes the tooltip's long-standing wrong `elev`.
"""

import altair as alt
import polars as pl

from cpra.mp.data.legends import LEGENDS
from cpra.mp.data.variables import VARIABLES
from cpra.mp.data.visualizations.base import DatasetVisualization, analysis_chain
from cpra.mp.data.visualizations.display import display_number_format, display_unit, display_value_expr

# The variable whose values are drawn as a categorical band rather than as a line, and the
# label column the analysis's `unpivot` writes variable names into
# (`analyses.sediment_dynamics.LABEL_COLUMN` -- the two have to agree, and this is the side
# that reads it).
LND_TYPE = "lnd_type"
LABEL_COLUMN = "variable"


class SedimentDynamics(DatasetVisualization):
    """A chart showing five sediment dynamics variables."""

    class Meta:
        name = "sediment-dynamics"
        label = "Sediment Dynamics"
        # `type = "dataset"` now: this reads the caller's dataset (for its grid and time
        # unit) like every other chart, and `"custom"` described a route family that no
        # longer exists. `/v1/visualizations.json` still serves the field.
        type = "dataset"
        # One pixel. It is what the chart is FOR -- five surfaces at a point -- and it is
        # also what makes the analysis expressible at all: a spatial selection is what
        # lowers the joined band-wide frame so the `unpivot` can run.
        max_grid_cells = 1
        # The five the analysis reads, and the five datasets a client should offer this
        # chart for. Same list as `analyses.sediment_dynamics.VARIABLE_NAMES`.
        variables = ["elev", "inun", LND_TYPE, "test_mnrl_accr", "test_org_accr"]
        # The named analysis this chart is the presentation of. Read by
        # `BaseVisualization.request_items` (the route's decode) and by `__init__` below
        # (the data URLs this spec emits); no other visualization declares one.
        analysis = "sediment_dynamics"

    def __init__(self, request=None, **kwargs):
        # Forced into `self.params`, not merely defaulted, because `forward_query` is what
        # puts `analysis=` on this spec's own `/v1/data.csv` URL -- a source built without
        # it would fetch a plain single-variable read and every layer below would filter it
        # down to nothing. `analysis_chain` keeps a caller's own chain, after this one.
        kwargs["analysis"] = analysis_chain(self.Meta.analysis, kwargs.get("analysis"))
        super().__init__(request, **kwargs)
        self.meter_variables = ["elev", "inun"]
        self.meter_colors = ["#284D36", "#ED6631"]
        self.centimeter_variables = ["test_mnrl_accr", "test_org_accr"]
        self.centimeter_colors = ["#E39F17", "#98C92F"]
        self.line_variables = self.meter_variables + self.centimeter_variables
        self.variables = [LND_TYPE] + self.line_variables

    @property
    def _value_expr(self):
        """`value` in display units, per variable.

        `get_base_chart` scales a source by ITS variable's `display_scale_factor`, which a
        folded source has none of -- `descriptor.variable` is `None` for it, so the base
        chart leaves `value` alone and this puts the conversion back, one arm per variable
        that needs one. Today that is `elev` alone (stored in NAVD88 millimetres, read in
        metres); the arms are generated rather than written out so that a scale factor
        added to `variables.csv` reaches this chart without an edit here.
        """
        expr = "datum.value"
        for name in reversed(self.variables):
            scaled = display_value_expr(VARIABLES[name])
            if scaled != "datum.value":
                expr = f"datum.{LABEL_COLUMN} === '{name}' ? {scaled} : {expr}"
        return expr

    @property
    def _label_expr(self):
        """Each row's variable label, for a color legend to key on.

        This is the `transform_lookup` against a one-column `LookupData` frame that used to
        do it, written as the expression it always was: five names, five labels, known at
        spec-build time and needing no second dataset in the spec to carry them.
        """
        expr = f"datum.{LABEL_COLUMN}"
        for name in reversed(self.variables):
            expr = f"datum.{LABEL_COLUMN} === '{name}' ? '{VARIABLES[name].label}' : {expr}"
        return expr

    def _base_chart(self):
        """The one source, in display units, with a label per row -- what every layer below
        starts from and then filters down to its own variables."""
        chart = self.get_base_chart(self.data_source, self.time_column)
        calculate_exprs = {f"{LABEL_COLUMN}_label": self._label_expr}
        value_expr = self._value_expr
        if value_expr != "datum.value":
            calculate_exprs["value"] = value_expr
        return chart.transform_calculate(**calculate_exprs)

    def _lnd_type_chart(self):
        lnd_type_legend = (
            LEGENDS[LND_TYPE]
            .break_points.with_columns(pl.col("upper_bound").shift(1).floor().alias("lower_bound"))
            .filter(pl.col("lower_bound").is_not_null())
        )
        return (
            self._base_chart()
            .transform_filter(alt.datum[LABEL_COLUMN] == LND_TYPE)
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

    def _variables_chart(self, variable_names):
        """The base chart narrowed to `variable_names` -- one row per (time, variable) for
        those, and nothing else."""
        return self._base_chart().transform_filter(
            alt.FieldOneOfPredicate(field=LABEL_COLUMN, oneOf=list(variable_names))
        )

    def _line_chart(self, variable_names, colors, show_label=True):
        variables = [VARIABLES[v] for v in variable_names]
        return (
            self._variables_chart(variable_names)
            .mark_line()
            .encode(
                alt.X("date:T").title("Year"),
                alt.Y("value:Q").title(", ".join([f"{v.label} ({display_unit(v)})" for v in variables])),
                alt.Color(f"{LABEL_COLUMN}_label:N")
                .title("Variable" if show_label else None)
                .scale(domain=[VARIABLES[v].label for v in variable_names], range=colors),
            )
        )

    def _points_chart(self, variable_names, when_near):
        return (
            self._variables_chart(variable_names)
            .mark_point(color="black")
            .encode(
                alt.X("date:T"),
                alt.Y("value:Q"),
                opacity=when_near.then(alt.value(1)).otherwise(alt.value(0)),
            )
        )

    def _rule_chart(self, nearest, when_near):
        """The hover rule, and the one layer that needs all five variables on one row.

        `transform_pivot` turns the folded rows back into a column per variable, keyed by
        time, so the tooltip can list all five for the year under the pointer -- the shape
        the old five-source `transform_lookup` chain produced and this is the cheap
        browser-side equivalent of. `groupby` carries `date` through with the time column
        because the pivot drops every field it is not told to keep, and `date` is what the
        x-encoding and the `nearest` selection are both stated in.
        """
        return (
            self._base_chart()
            .transform_pivot(LABEL_COLUMN, value="value", groupby=[self.time_column, "date"])
            .mark_rule(color="gray")
            .encode(
                x="date:T",
                opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
                tooltip=[self.get_time_tooltip(self.time_column)]
                + [
                    alt.Tooltip(
                        VARIABLES[v].name,
                        type="quantitative",
                        title=VARIABLES[v].label,
                        format=display_number_format(VARIABLES[v]),
                    )
                    for v in self.variables
                ],
            )
            .add_params(nearest)
        )

    def chart(self):
        """Build the visualization using Altair."""
        # `size_to_container` (see base.py) reinterprets width/height as the total
        # container size and lets Vega-Lite's autosize give the leftover to the plot area
        # after measuring real axis/title/legend chrome, so no chrome allowance is guessed
        # here — this chart is a single layered view, so the full container size is passed
        # straight through.
        chart_width = self.width
        chart_height = self.height

        lnd_type_chart = self._lnd_type_chart()
        meter_chart = self._line_chart(self.meter_variables, self.meter_colors)
        centimeter_chart = self._line_chart(self.centimeter_variables, self.centimeter_colors, False)

        nearest = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False)
        when_near = alt.when(nearest)
        meter_points_chart = self._points_chart(self.meter_variables, when_near)
        centimeter_points_chart = self._points_chart(self.centimeter_variables, when_near)
        rule_chart = self._rule_chart(nearest, when_near)

        chart = (
            alt.layer(
                lnd_type_chart,
                alt.layer(meter_chart, meter_points_chart, rule_chart),
                alt.layer(centimeter_chart, centimeter_points_chart),
            )
            .resolve_scale(y="independent", color="independent")
            .properties(title=self.Meta.label, width=chart_width, height=chart_height)
        )
        return self.size_to_container(chart)
