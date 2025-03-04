import altair as alt
from functools import cached_property
from urllib.parse import urlencode
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.variables import VARIABLES


class BaseVisualization:
    request = None

    def chart(self):
        """Build the visualization using Altair."""

        raise NotImplementedError

    def get_data_uri(self, variable, grid, time_unit, **kwargs):
        """Get an API data URI."""

        params = []
        for param_name, param_values in kwargs.items():
            if isinstance(param_values, (list, tuple)):
                for value in param_values:
                    params.append((param_name, value))
            elif param_values is not None:
                params.append((param_name, param_values))

        # TODO: Set select param to limit columns returned.

        if self.request:
            dataset_url = self.request.url_for(
                "get_model_data",
                variable=variable,
                grid=grid,
                time_unit=time_unit,
                format="csv",
            )
        else:
            dataset_url = (
                "https://api.mpdp.coastal.la.gov/v1/datasets"
                f"/{variable}/{grid}/{time_unit}/data.csv"
            )

        return f"{dataset_url}?{urlencode(params)}"

    @property
    def spec(self):
        """Get the Vega-Lite specification."""
        chart = self.chart()
        return chart.to_dict()


class DatasetVisualization(BaseVisualization):
    """Visualization for a single dataset."""

    def __init__(self, variable, grid, time_unit, request=None, **kwargs):
        self.variable = VARIABLES[variable]
        self.grid = GRIDS[grid]
        self.time_unit = time_unit
        self.request = request
        self.params = kwargs

    @cached_property
    def data_source(self):
        """Get the Altair data source."""

        data_uri = self.get_data_uri(
            self.variable.name,
            self.grid.name,
            self.time_unit,
            **{k: v for k, v in self.params.items() if v is not None},
        )
        return alt.UrlData(
            data_uri, format=alt.CsvDataFormat(type="csv"), name=self.variable.name
        )

    @cached_property
    def geographies(self):
        """Get the selected geographies."""
        return map(int, self.params["geography"].split(","))

    @cached_property
    def time_column(self):
        """Get the name of the time column."""
        return (
            "calendar_year"
            if self.time_unit in ("decadal", "annual")
            else "calendar_day"
        )

    @cached_property
    def title(self):
        """Get the chart title."""
        title = self.variable.label
        aggregate_type = self.params.get("aggregate_type", None)
        if aggregate_type and len(aggregate_type) == 1:
            aggregate_label = DIMENSIONS["aggregate_type"].values.filter(
                value=aggregate_type[0]
            )["label"][0]
            return f"{aggregate_label} {title}"
        return title

    @cached_property
    def variable_label(self):
        """Get the variable label and unit."""
        label = self.variable.label
        if self.variable.unit:
            label += f" ({self.variable.unit})"
        return label
