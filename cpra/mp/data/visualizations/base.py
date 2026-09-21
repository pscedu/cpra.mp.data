from functools import cached_property
from urllib.parse import urlencode

import altair as alt
import polars as pl

from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.utils import required_columns
from cpra.mp.data.variables import VARIABLES
from cpra.mp.data.visualizations.colors import selection_color
from cpra.mp.data.visualizations.display import display_unit, display_value_expr


def grid_cell_tokens(grid_cell):
    """Split a `grid_cell` parameter into stripped string tokens.

    Accepts either the comma-separated string the API binds from the query string or the
    plain sequence a notebook is more likely to pass.
    """
    values = grid_cell.split(",") if isinstance(grid_cell, str) else grid_cell
    return [str(value).strip() for value in values]


def is_grid_cell_ids(grid_cell):
    """Are these grid_cell values grid cell IDs rather than lat/lng coordinate pairs?

    Mirrors the rule in `Dataset._apply_grid_cell_vector`: values are IDs only when every
    token is a plain unsigned digit string, so anything carrying a decimal point or a sign
    is a coordinate. Only meaningful for a vector grid, since a raster grid_cell is always
    coordinates. An absent grid_cell selects the whole grid, which is keyed by ID.
    """
    return grid_cell is None or all(token.isdigit() for token in grid_cell_tokens(grid_cell))


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
            # The API serves storage units; every chart downstream reads display units, so
            # the conversion happens here, once, on the way in -- see visualizations.display.
            value_expr = display_value_expr(self.source_variable(source_name))

            # If this is the first data source, create the chart.
            if chart is None:
                chart = alt.Chart(source)

                # Collect and apply calculate expressions.
                calculate_exprs = {}
                # Both entries read `datum.value` directly rather than one reading the
                # other, so they don't depend on which calculate transform Vega runs first.
                if value_expr != "datum.value":
                    calculate_exprs["value"] = value_expr
                if source_name != "value":
                    calculate_exprs[source_name] = value_expr

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
                # The lookup pulls the joined source's raw `value` across, so this source's
                # own scaling has to be applied to the field it landed in.
                joined_field = f"datum['{source_name}']"
                joined_expr = display_value_expr(self.source_variable(source_name), joined_field)
                if joined_expr != joined_field:
                    chart = chart.transform_calculate(**{source_name: joined_expr})

                # The lookup pulls the joined source's raw `value` across, so this source's
                # own scaling has to be applied to the field it landed in.
                joined_field = f"datum['{source_name}']"
                joined_expr = display_value_expr(self.source_variable(source_name), joined_field)
                if joined_expr != joined_field:
                    chart = chart.transform_calculate(**{source_name: joined_expr})

        return chart

    def source_variable(self, source_name):
        """The variable whose values the ``get_base_chart`` data source ``source_name`` carries.

        A source is named for the variable it holds (``get_data_source``'s ``name or
        variable``), so a multi-variable chart like ``SedimentDynamics`` is answered by
        name alone.  The exception is a ``DatasetVisualization``, whose single source is
        named ``"value"`` after the field it fills; that falls back to the one variable the
        visualization is about.  Returns ``None`` when neither applies, which
        ``display_value_expr`` treats as "no scaling".
        """

        return VARIABLES.get(source_name) or getattr(self, "variable", None)

    def get_data_uri(self, variable, grid, time_unit, **kwargs):
        """Get an API data URI.

        When a named analysis is selected (`resolved_analysis` -- a `DatasetVisualization`
        property, absent/`None` on a visualization that doesn't have one, e.g.
        `SedimentDynamics`), points at that analysis's own `.../analyses/{name}/data.csv`
        route instead of the plain `.../data.csv` route. With no analysis selected, this
        is unchanged -- there's no reason to route an unanalyzed filtered read through an
        analysis; either way the data returned is already long format (grid_cell/value
        columns), so there is no `unpivot` param to forward on either route.
        """

        required_cols = required_columns(pl.DataFrame({"variable": variable, "time_unit": time_unit, "grid": grid}))
        select = [self.get_time_column(time_unit), "grid_cell", "value"]

        resolved = getattr(self, "resolved_analysis", None)
        analysis_name = kwargs.pop("analysis", None)

        if resolved is not None:
            # Only the analysis route's own known fields, plus whatever extra parameters
            # the named analysis itself declares (`dimension`/`order`/...), have any
            # meaning here -- every OTHER `VisualizationQuery` field (the map-only display
            # params, ...) would otherwise reach the analysis route as an unrecognized
            # query parameter and be rejected outright, since its `Parameters` model is
            # `extra="forbid"` (see `analyses/base.py`).
            from cpra.mp.data.api.models import AnalysisDataQuery

            # `kwargs` alone is missing a named analysis's own parameters (`dimension`/
            # `order`/...) for a request-driven visualization: `VisualizationQuery` can't
            # declare them (each analysis declares its own), so FastAPI's query-model
            # binding never puts them in `self.params` -- and `kwargs` here traces back to
            # exactly that dict (via `get_data_source`) -- only `_analysis_params`'s own
            # extra scan of the raw query string recovers them (see its docstring above).
            # Folding it in is what actually lets those parameters reach the emitted
            # data.csv URL; it never overlaps `kwargs`'s own `VisualizationQuery`-declared
            # keys, so this can't clobber anything already there.
            extra_analysis_params = getattr(self, "_analysis_params", {})
            allowed = set(AnalysisDataQuery.model_fields) | set(extra_analysis_params)
            kwargs = {k: v for k, v in {**kwargs, **extra_analysis_params}.items() if k in allowed}

        # A caller-supplied `select` (`SpeciesStackedBar.data_source`) replaces the computed
        # one outright, but still has to be filtered below, so it is taken out of `kwargs`
        # here and re-emitted as the single `select` parameter at the end rather than being
        # passed straight through by the loop.
        caller_select = kwargs.pop("select", None)

        params = []
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

        if caller_select is not None:
            select = caller_select.split(",") if isinstance(caller_select, str) else list(caller_select)

        # A dimension the analysis COLLAPSES is gone from its output frame -- `difference`
        # over `model_group_id` returns one value per grid cell per day, not one per model
        # group -- so naming it in `select` asks for a column that no longer exists, and the
        # analysis routes fail the whole read ("unable to find column"). The two values are
        # exactly what a `difference` needs as FILTERS and still have to be SENT, so this
        # drops only the `select` entry, never the parameter itself.
        #
        # Filtering one finished list rather than guarding each place a column is added is
        # deliberate: a collapsed dimension reaches `select` three ways, and the multi-value
        # branch above is only one of them. The time column goes in unconditionally at the
        # top (so a `difference` over `calendar_day` hit the identical failure), and
        # `caller_select` bypasses both (so `SpeciesStackedBar` did too, over `calendar_year`).
        # `TimeSeriesLine`/`StackedBar` already exclude collapsed dimensions from their own
        # compare-channel choices, for the same underlying reason -- a collapsed dimension
        # has nothing left to compare across -- so this keeps the request in step with the
        # chart that consumes it. With no analysis selected, nothing is collapsed.
        if resolved is not None:
            select = [column for column in select if column not in resolved.collapsed_dimensions]

        # Set select param to limit columns returned.
        params.append(("select", ",".join(select)))

        if resolved is not None:
            if self.request:
                dataset_url = self.request.url_for(
                    "get_dataset_analysis_data",
                    variable=variable,
                    time_unit=time_unit,
                    grid=grid,
                    analysis_name=analysis_name,
                    format="csv",
                )
            else:
                dataset_url = (
                    f"https://api.mpdp.coastal.la.gov/v1/datasets/{variable}/{time_unit}/{grid}"
                    f"/analyses/{analysis_name}/data.csv"
                )
        elif self.request:
            dataset_url = self.request.url_for(
                "get_model_data",
                variable=variable,
                time_unit=time_unit,
                grid=grid,
                format="csv",
            )
        else:
            dataset_url = f"https://api.mpdp.coastal.la.gov/v1/datasets/{variable}/{time_unit}/{grid}/data.csv"

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
            # A vector grid reports grid_cell as numeric grid cell IDs however the caller
            # asked for it; a raster grid reports the "lng,lat" string it was given.
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

    def size_to_container(self, chart, is_concat: bool = False):
        """Size a finished chart to exactly fill `self.width` x `self.height` (the frontend's
        container), whatever room the axes/title/legend end up taking.

        In Vega-Lite, a mark's `width`/`height` describe the plotting area only — the
        rendered SVG is that area plus the surrounding chrome, so a naive `.properties(width=,
        height=)` on the container size leaves the chrome to overflow it. `autosize`'s `fit`
        mode reinterprets `width`/`height` as the *total* size instead and shrinks the
        plotting area to make room, which is exactly what's wanted here since it measures real
        text metrics rather than guessing at legend width the way this code used to.

        `fit` only works for single or layered views, though — Vega-Lite rejects it on a
        vconcat spec (warns and silently falls back to `pad`, which overflows again exactly
        like the bug this is fixing). `fit-x` is the documented workaround: it fits the width
        only, which vconcat does support, and is enough on its own because concat's overflow
        here only ever came from the legend pushing the width out — the height side of a
        concat spec still has to be sized by hand (see the `- 222` arithmetic in
        `TimeSeriesLine`/`StackedBar` for the daily/vconcat branch). Don't "simplify" this back
        to `fit` for a concat chart; it will silently stop being honored.

        A top-level vconcat spec has no `width`/`height` property at all in the Vega-Lite
        schema — sizing there comes entirely from each concatenated child's own `width`/
        `height` (set where the child is built, e.g. `chart_width`/`chart_height` in
        `TimeSeriesLine`/`StackedBar`). So for a concat chart this method only sets
        `autosize`; `fit-x` then stretches each child's plotting area so the child's own
        declared width is honored as a total rather than a plot-area-only size, the same
        way `fit` does for a single/layered chart's top-level width.

        Apply this to the final, top-level chart object only (e.g. after a `&` concat) — a
        nested/intermediate layer's `properties(width=, height=)` is what actually gets fit,
        and putting autosize somewhere other than the top level has no effect.
        """
        autosize = alt.AutoSizeParams(type="fit-x" if is_concat else "fit", contains="padding")
        if is_concat:
            return chart.properties(autosize=autosize)
        return chart.properties(width=self.width, height=self.height, autosize=autosize)

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
    def _analysis_name(self):
        """The `analysis=` request parameter, or `None` when not given."""
        return self.params.get("analysis")

    @cached_property
    def _analysis_filters(self):
        """Dimension filters + `grid_cell`/`where`, in the exact shape `resolve_analysis`'s
        `filters` argument expects -- mirrors `api.__init__._analysis_filters`."""
        from cpra.mp.data.api.models import FilterDimensions

        filter_fields = set(FilterDimensions.model_fields)
        filters = {k: v for k, v in self.params.items() if k in filter_fields and v is not None}
        for extra_key in ("grid_cell", "where"):
            value = self.params.get(extra_key)
            if value is not None:
                filters[extra_key] = value
        return filters

    @cached_property
    def _analysis_params(self):
        """The named analysis's own declared parameters (`dimension`/`order`/...), read off
        whatever isn't a `VisualizationQuery` field -- mirrors
        `api.__init__._analysis_extra_params`. A request-driven visualization only has
        these in `self.params` when `self.request`'s raw query string carries them (a
        route handler's `dict(query)` only ever carries `VisualizationQuery`'s own
        declared fields, since the analysis's parameters have no fixed field set of their
        own); a notebook caller that builds this class directly passes them as ordinary
        keyword arguments instead, which already land in `self.params`.
        """
        from cpra.mp.data.api.models import VisualizationQuery

        known = set(VisualizationQuery.model_fields)
        extra_params = {k: v for k, v in self.params.items() if k not in known and v is not None}
        if self.request is not None:
            for key, value in self.request.query_params.items():
                if key not in known and key not in extra_params:
                    extra_params[key] = value
        return extra_params

    @cached_property
    def resolved_analysis(self):
        """The `analysis=` request parameter, resolved into a real, ready-to-
        compile `Pipeline` plus presentation metadata (`.legend`/`.result_kind`/
        `.collapsed_dimensions`/`.title_suffix`) -- or `None` when no analysis was
        selected, the signal every migrated hook in this module (and in `map.py`/
        `time_series_line.py`/`stacked_bar.py`) falls back to its pre-existing legacy
        behavior on.

        `grid_cell` is resolved against real geometry FIRST and passed separately as
        `grid_cell_where`, exactly as `api.__init__._resolve_dataset_analysis` does for
        the HTTP routes and for the same reason: the registry cannot parse a coordinate
        (doing so needs `io/`), and a visualization's `grid_cell` is a coordinate as often
        as it is an id -- `_grid_cells` above already resolves the same value the same
        way. A `None` back from `resolve_grid_cell_sync` means the value was a vector
        grid's digit-id form (or absent), which `_analysis_filters`'s own raw `grid_cell`
        key covers.
        """
        name = self._analysis_name
        if name is None:
            return None

        # Imported lazily: `cpra.mp.data.api`'s `__init__.py` imports `VISUALIZATIONS`
        # from this package at module level, so importing it back from here at module
        # level would be a circular import -- see `map.py`'s `feature_colors`, which
        # already does the same thing for its own model imports.
        from cpra.mp.data.analyses import DatasetRef, resolve_analysis
        from cpra.mp.data.api.data_query import resolve_grid_cell_sync

        grid_cell_where, _relabel = resolve_grid_cell_sync(self.grid.name, self.params.get("grid_cell"))
        dataset = DatasetRef(variable=self.variable.name, time_unit=self.time_unit, grid=self.grid.name)
        return resolve_analysis(
            name,
            dataset=dataset,
            filters=self._analysis_filters,
            params=self._analysis_params,
            grid_cell_where=grid_cell_where,
        )

    @cached_property
    def difference_suffix(self):
        """Suffix for the chart title and y-axis if this is a difference chart."""
        resolved = self.resolved_analysis
        if resolved is not None:
            return resolved.title_suffix
        return ""

    @cached_property
    def grid_cells(self):
        """Get the selected grid cells as the API keys them in its own output.

        A vector grid takes either grid cell IDs or lat/lng coordinate pairs; a raster
        grid only takes coordinates. These are what a `grid_cell` value in the response
        will actually equal, so they're what a scale domain or a lookup key has to match —
        see `grid_cell_labels` for what to show the reader instead.
        """
        return [grid_cell for grid_cell, _, _, _ in self._grid_cells]

    @cached_property
    def grid_cell_labels(self):
        """Get a display label per selected grid cell, in the same order as `grid_cells`.

        Identifies a grid cell in full, for somewhere with room for it — a legend. A
        coordinate is shown parenthesized, and on a vector grid it's prefixed with the ID of
        the cell it landed in, since the cell — not the point — is what the values describe:
        `24`, `24 (-90.49, 30.01)`, or `(-90.49, 30.01)` on a raster grid.
        """
        return [label for _, label, _, _ in self._grid_cells]

    @cached_property
    def grid_cell_short_labels(self):
        """Get a compact display label per selected grid cell, in the same order.

        For somewhere a full label would crowd out the numbers being read, like a tooltip
        with one row per grid cell. Where a grid cell ID is available it stands alone, since
        it's what the values describe and the coordinate that selected it only repeats what
        the legend already says. A raster pixel has no ID, so the coordinate is all there is
        to show. Two coordinates inside one cell therefore share a short label; the legend is
        what tells them apart.
        """
        return [short_label for _, _, short_label, _ in self._grid_cells]

    @cached_property
    def grid_cell_colors(self):
        """Get the color the map draws each selected grid cell with, in the same order.

        A chart is read beside the map, so a grid cell has to carry the same color in both.
        The map's color comes from the grid cell's place in the *selection*, which is not its
        place in this list: a coordinate that fell outside the grid, or inside a cell an
        earlier coordinate already took, is left out here and takes its color out with it.
        Pairing the whole palette with this list positionally would instead shift every
        series after such a coordinate onto the color of the pin before it.
        """
        return [color for _, _, _, color in self._grid_cells]

    @cached_property
    def _grid_cells(self):
        """One `(API grid_cell key, label, short label, color)` tuple per selected grid cell.

        A raster grid is keyed by the coordinate itself, reported as a `"lng,lat"` string
        built from the parsed floats — so round-trip the tokens through `float` to match
        it (`-91` in the request becomes `-91.0` in the response). A vector grid is always
        keyed by grid cell ID, whichever way the caller asked for it, so a coordinate has
        to be resolved to a cell the same way the data endpoint does. That also means a
        coordinate outside the grid drops out, as it does there, and coordinates sharing a
        cell collapse into the one series the response actually carries.

        The color is the map's, and so is taken from the grid cell's place in the selection
        rather than from its place in this list, which dropping and collapsing shorten.
        """
        grid_cell = self.params["grid_cell"]

        if self.grid.grid_type == "vector" and is_grid_cell_ids(grid_cell):
            ids = [int(token) for token in grid_cell_tokens(grid_cell)]
            return [
                (grid_cell_id, str(grid_cell_id), str(grid_cell_id), selection_color(i))
                for i, grid_cell_id in enumerate(ids)
            ]

        coords = [float(token) for token in grid_cell_tokens(grid_cell)]
        pairs = list(zip(coords[::2], coords[1::2]))

        if self.grid.grid_type != "vector":
            return [
                (f"{lng!s},{lat!s}", f"({lng}, {lat})", f"({lng}, {lat})", selection_color(i))
                for i, (lng, lat) in enumerate(pairs)
            ]

        # Imported lazily: this pulls in the geospatial stack, and a notebook that only
        # builds specs shouldn't need it. Same reason map.py imports the analysis engine
        # where it's used rather than at module scope.
        from cpra.mp.data.analysis.io.geography import resolve_vector_grid_cell_sync

        ids = resolve_vector_grid_cell_sync(self.grid.name, [p[0] for p in pairs], [p[1] for p in pairs])

        grid_cells = {}
        for i, ((lng, lat), grid_cell_id) in enumerate(zip(pairs, ids)):
            # First coordinate wins a shared cell: the response has one series for it, so a
            # second entry would be a legend key the data never matches. Its color is the
            # first coordinate's too, which is how the map draws the cell's outline.
            if grid_cell_id is not None and grid_cell_id not in grid_cells:
                grid_cells[grid_cell_id] = (
                    grid_cell_id,
                    f"{grid_cell_id} ({lng}, {lat})",
                    str(grid_cell_id),
                    selection_color(i),
                )
        return list(grid_cells.values())

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
        """Get the variable label and unit.

        The unit shown is the display unit, matching the scaling `get_base_chart` applies
        to the values this labels.
        """
        label = f"{self.variable.label}{self.difference_suffix}"
        unit = display_unit(self.variable)
        if unit:
            label += f" ({unit})"
        return label
