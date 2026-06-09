from functools import cached_property
from urllib.parse import urlencode

import altair as alt
import polars as pl

from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.utils import required_columns
from cpra.mp.data.variables import VARIABLES


class BaseVisualization:
    def __init__(self, request=None, height=400, width=600, **kwargs):
        self.request = request
        self.height = height
        self.width = width
        self.params = kwargs

    def chart(self):
        """Build the visualization using Altair."""

        raise NotImplementedError

    def get_base_chart(
        self,
        data_sources: alt.UrlData | dict[str, alt.UrlData],
        time_column: str,
        join_column: str | None = None,
        date_column: str | None = "date",
    ) -> alt.Chart:
        """Generate a base chart from the given data source and options."""
        chart: alt.Chart | None = None

        if isinstance(data_sources, alt.UrlData):
            data_sources = {"value": data_sources}

        for source_name, source in data_sources.items():
            # If this is the first data source, create the chart.
            if chart is None:
                chart = alt.Chart(source)

                # Collect and apply calculate expressions.
                calculate_exprs = {}
                if source_name != "value":
                    calculate_exprs[source_name] = "datum.value"

                if date_column:
                    if time_column == "calendar_year":
                        calculate_exprs[date_column] = "toDate(toString(datum.calendar_year))"
                    else:
                        calculate_exprs[date_column] = "datum.calendar_day"

                if calculate_exprs:
                    chart = chart.transform_calculate(**calculate_exprs)
            # Otherwise, join this data source to the previous chart.
            else:
                col = join_column or time_column
                chart = chart.transform_lookup(
                    col,
                    as_=[source_name],
                    from_=alt.LookupData(source, col, ["value"]),
                )

        return chart

    def get_data_uri(self, variable, grid, time_unit, **kwargs):
        """Get an API data URI."""

        required_cols = required_columns(pl.DataFrame({"variable": variable, "time_unit": time_unit}))
        select = [self.get_time_column(time_unit), "grid_cell", "value"]

        # Force unpivot to be true.
        kwargs["unpivot"] = True

        params = [("grid", grid)] if grid else []
        for param_name, param_values in kwargs.items():
            if isinstance(param_values, (list, tuple)):
                if param_name == "grid_cell":
                    params.append(("grid_cell", ",".join([str(geo) for geo in param_values])))
                else:
                    for value in param_values:
                        params.append((param_name, value))
                if param_name in required_cols and len(param_values) > 1:
                    select.append(param_name)
            elif param_values is not None:
                params.append((param_name, param_values))

        # Set select param to limit columns returned.
        if "select" not in kwargs:
            params.append(("select", ",".join(select)))

        if self.request:
            dataset_url = self.request.url_for(
                "get_model_data",
                variable=variable,
                time_unit=time_unit,
                format="csv",
            )
        else:
            dataset_url = "https://api.mpdp.coastal.la.gov/v1/datasets" f"/{variable}/{time_unit}/data.csv"

        return f"{dataset_url}?{urlencode(params)}"

    def get_data_source(self, variable, grid, time_unit, name=None, parse=None, **kwargs):
        """Get an Altair data source for the given dataset and query parameters."""
        data_uri = self.get_data_uri(
            variable,
            grid,
            time_unit,
            **{k: v for k, v in kwargs.items() if v is not None},
        )

        if parse is None:
            parse = {"value": "number"}
            if GRIDS[grid].grid_type == "vector":
                parse["grid_cell"] = "number"
            time_column = self.get_time_column(time_unit)
            parse[time_column] = "date:%Y-%m-%d" if time_column == "calendar_day" else "number"

        return alt.UrlData(data_uri, format=alt.CsvDataFormat(type="csv", parse=parse), name=(name or variable))

    def get_time_column(self, time_unit):
        """Get the name of the time column."""
        return "calendar_year" if time_unit in ("decadal", "annual") else "calendar_day"

    def get_time_tooltip(self, time_column: str):
        """Create a tooltip for the given time column."""
        return alt.Tooltip(
            time_column,
            type="quantitative" if time_column == "calendar_year" else "temporal",
            title=DIMENSIONS[time_column].label,
        )

    @property
    def spec(self):
        """Get the Vega-Lite specification."""
        chart = self.chart()
        return chart.to_dict()


class DatasetVisualization(BaseVisualization):
    """Visualization for a single dataset."""

    def __init__(self, variable, grid, time_unit, **kwargs):
        super().__init__(**kwargs)
        self.variable = VARIABLES[variable]
        self.grid = GRIDS[grid]
        self.time_unit = time_unit

    @cached_property
    def data_source(self):
        """Get the Altair data source."""
        return self.get_data_source(self.variable.name, self.grid.name, self.time_unit, **self.params)

    @cached_property
    def difference_suffix(self):
        """Suffix for the chart title and y-axis if this is a difference chart."""
        return "" if self.params.get("negate") is None else " Difference"

    @cached_property
    def geographies(self):
        """Get the selected geographies."""
        raster = self.grid.grid_type == "raster"
        coord_type = float if raster else int
        geos = self.params["grid_cell"]
        coords = list(map(coord_type, geos.split(",") if isinstance(geos, str) else geos))

        if not raster:
            return coords

        return [",".join([str(coord) for coord in coords[i : i + 2]]) for i in range(0, len(coords), 2)]

    @cached_property
    def time_column(self):
        """Get the name of the time column."""
        return self.get_time_column(self.time_unit)

    @cached_property
    def time_dimension(self):
        """Get the temporal dimension."""
        return DIMENSIONS[self.time_column]

    @cached_property
    def title(self):
        """Get the chart title."""
        title = self.variable.label
        aggregate_type = self.params.get("aggregate_type", None)
        if aggregate_type and len(aggregate_type) == 1:
            aggregate_label = DIMENSIONS["aggregate_type"].values.filter(value=aggregate_type[0])["label"][0]
            title = f"{aggregate_label} {title}"
        return f"{self.time_unit.title()} {title}{self.difference_suffix}"

    @cached_property
    def variable_label(self):
        """Get the variable label and unit."""
        label = f"{self.variable.label}{self.difference_suffix}"
        if self.variable.unit:
            label += f" ({self.variable.unit})"
        return label
