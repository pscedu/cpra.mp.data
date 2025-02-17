import altair as alt
import datetime as dt

geographies = ["1", "2", "3"]
data_url = f"https://api.mpdp.coastal.la.gov/v1/datasets/sal/hydro_compartment_v001/daily/data.csv?scenario_id=7&model_group_id=500&aggregate_type=mean&geography={','.join(geographies)}&melt=true"
source = alt.UrlData(data_url, format=alt.CsvDataFormat(type="csv"), name="salinity")

initial_date_range = (dt.date(2019, 1, 1), dt.date(2020, 1, 1))

nearest = alt.selection_point(
    nearest=True, on="pointerover", fields=["calendar_day"], empty=False
)
brush = alt.selection_interval(encodings=["x"], value={"x": initial_date_range})

daily_line = (
    alt.Chart(source)
    .mark_line()
    .encode(
        alt.X("calendar_day:T").title("Date"),
        alt.Y("value:Q").title("Salinity (PPM)"),
        alt.Color("geography:N").title("Hydro Compartment"),
    )
    .properties(title="Daily Mean Salinity", width=800)
)

annual_line = (
    daily_line.encode(
        alt.X("yearmonth(calendar_day):T").title("Month"),
        alt.Y("mean(value):Q").title("Salinity (PPM)"),
    )
    .properties(title="Monthly Mean Salinity", height=60)
    .add_params(brush)
)

when_near = alt.when(nearest)
points = daily_line.mark_point().encode(
    opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
)

rules = (
    alt.Chart(source)
    .transform_pivot("geography", value="value", groupby=["calendar_day"])
    .mark_rule(color="gray")
    .encode(
        x="calendar_day:T",
        opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
        tooltip=[alt.Tooltip("calendar_day", type="temporal")]
        + [alt.Tooltip(c, type="quantitative") for c in geographies],
    )
    .add_params(nearest)
)

layer = (
    (alt.layer(daily_line, points, rules))
    .transform_filter(brush)
    .properties(width=800, height=500)
)
viz = layer & annual_line
viz.save("altair_chart.html")
