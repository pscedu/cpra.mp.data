from functools import cached_property
from urllib.parse import urlencode

import altair as alt
import polars as pl

from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.utils import required_columns
from cpra.mp.data.variables import VARIABLES


def geography_tokens(geography):
    """Split a `geography` parameter into stripped string tokens.

    Accepts either the comma-separated string the API binds from the query string or the
    plain sequence a notebook is more likely to pass.
    """
    values = geography.split(",") if isinstance(geography, str) else geography
    return [str(value).strip() for value in values]


def is_geography_ids(geography):
    """Are these geography values grid cell IDs rather than lat/lng coordinate pairs?

    Mirrors the rule in `Dataset._apply_geography_vector`: values are IDs only when every
    token is a plain unsigned digit string, so anything carrying a decimal point or a sign
    is a coordinate. Only meaningful for a vector grid, since raster geography is always
    coordinates. An absent geography selects the whole grid, which is keyed by ID.
    """
    return geography is None or all(token.isdigit() for token in geography_tokens(geography))


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

        required_cols = required_columns(pl.DataFrame({"variable": variable, "grid": grid, "time_unit": time_unit}))
        select = [self.get_time_column(time_unit), "geography", "value"]

        # Force unpivot to be true.
        kwargs["unpivot"] = True

        params = []
        for param_name, param_values in kwargs.items():
            if isinstance(param_values, (list, tuple)):
                if param_name == "geography":
                    params.append(("geography", ",".join([str(geo) for geo in param_values])))
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
                grid=grid,
                time_unit=time_unit,
                format="csv",
            )
        else:
            dataset_url = "https://api.mpdp.coastal.la.gov/v1/datasets" f"/{variable}/{grid}/{time_unit}/data.csv"

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
            # A vector grid reports geography as numeric grid cell IDs, except when the
            # caller selected cells by coordinate — then the API labels them with the
            # original "lng,lat" strings, which have to stay strings here to match.
            if GRIDS[grid].grid_type == "vector" and is_geography_ids(kwargs.get("geography")):
                parse["geography"] = "number"
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
        """Get the selected geographies as the API keys them in its own output.

        A vector grid takes either grid cell IDs or lat/lng coordinate pairs; a raster
        grid only takes coordinates. These are what a `geography` value in the response
        will actually equal, so they're what a scale domain or a lookup key has to match —
        see `geography_labels` for what to show the reader instead.
        """
        return [geography for geography, _, _ in self._geographies]

    @cached_property
    def geography_labels(self):
        """Get a display label per selected geography, in the same order as `geographies`.

        Identifies a geography in full, for somewhere with room for it — a legend. A
        coordinate is shown parenthesized, and on a vector grid it's prefixed with the ID of
        the cell it landed in, since the cell — not the point — is what the values describe:
        `24`, `24 (-90.49, 30.01)`, or `(-90.49, 30.01)` on a raster grid.
        """
        return [label for _, label, _ in self._geographies]

    @cached_property
    def geography_short_labels(self):
        """Get a compact display label per selected geography, in the same order.

        For somewhere a full label would crowd out the numbers being read, like a tooltip
        with one row per geography. Where a grid cell ID is available it stands alone, since
        it's what the values describe and the coordinate that selected it only repeats what
        the legend already says. A raster pixel has no ID, so the coordinate is all there is
        to show. Two coordinates inside one cell therefore share a short label; the legend is
        what tells them apart.
        """
        return [short_label for _, _, short_label in self._geographies]

    @cached_property
    def _geographies(self):
        """One `(API geography key, label, short label)` triple per selected geography.

        The API reports a coordinate-selected geography under a `"lng,lat"` string built
        from the parsed floats, so round-trip the tokens through `float` here to match it
        (`-91` in the request becomes `-91.0` in the response). Labeling a vector-grid
        coordinate means resolving it to a cell the same way the data endpoint does; a
        coordinate outside the grid is dropped, as it is there, so no legend entry or
        tooltip slot is left with nothing behind it.
        """
        geography = self.params["geography"]

        if self.grid.grid_type == "vector" and is_geography_ids(geography):
            ids = [int(token) for token in geography_tokens(geography)]
            return [(geo_id, str(geo_id), str(geo_id)) for geo_id in ids]

        coords = [float(token) for token in geography_tokens(geography)]
        pairs = list(zip(coords[::2], coords[1::2]))
        keys = [f"{lng!s},{lat!s}" for lng, lat in pairs]

        if self.grid.grid_type != "vector":
            return [(key, f"({lng}, {lat})", f"({lng}, {lat})") for key, (lng, lat) in zip(keys, pairs)]

        # Imported lazily: the api package pulls in fastapi and the geospatial stack, and
        # a notebook that only builds specs shouldn't need either. Same reason map.py
        # imports Dataset where it uses it.
        from cpra.mp.data.api.geography import resolve_vector_geography_sync

        ids = resolve_vector_geography_sync(self.grid.name, [p[0] for p in pairs], [p[1] for p in pairs])
        return [
            (key, f"{geo_id} ({lng}, {lat})", str(geo_id))
            for key, (lng, lat), geo_id in zip(keys, pairs, ids)
            if geo_id is not None
        ]

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
