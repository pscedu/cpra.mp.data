from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    RangeTool,
    PanTool,
    SaveTool,
)
from bokeh.plotting import figure, show

from cpra.mp.data import read_data

lf = read_data(
    variable="sal",
    grid="hydro_compartment_v001",
    time_unit="daily",
    scenario_id=7,
    model_group_id=500,
    aggregate_type="mean",
)

df = lf.select("calendar_day", "1").collect()
source = ColumnDataSource(
    data={"calendar_day": df["calendar_day"].to_numpy(), "1": df["1"].to_numpy()}
)

# adapter = CustomJS(
#     code="""
#     console.log(cb_data.response);
# """
# )
# source = WebDataSource(
#     data_url="https://api.mpdp.coastal.la.gov/v1/datasets/sal/hydro_compartment_v001/annual/data.ndjson?scenario_id=7&model_group_id=500&aggregate_type=mean",
#     method="GET",
#     mode="replace",
#     adapter=adapter,
# )

hover_tool = HoverTool(
    tooltips=[
        ("Date", "@calendar_day{%F}"),
        ("1", "@{1}"),
    ],
    formatters={"@calendar_day": "datetime"},
    mode="vline",
)

p = figure(
    title="Simple line example",
    x_axis_label="Date",
    y_axis_label="Salinity",
    sizing_mode="scale_width",
    x_axis_type="datetime",
    tools=[PanTool(), hover_tool, SaveTool()],
)
p.toolbar.logo = None
p.toolbar.autohide = True
p.line("calendar_day", "1", legend_label="Salinity", line_width=1, source=source)


select = figure(
    title="Drag the middle and edges of the selection box to change the range above.",
    height=130,
    width=800,
    y_range=p.y_range,
    x_axis_type="datetime",
    y_axis_type=None,
    tools="",
    toolbar_location=None,
    background_fill_color="#efefef",
)

range_tool = RangeTool(x_range=p.x_range, start_gesture="pan")
range_tool.overlay.fill_color = "navy"
range_tool.overlay.fill_alpha = 0.2

select.line("calendar_day", "1", source=source)
select.ygrid.grid_line_color = None
select.add_tools(range_tool)

show(column(p, select))
