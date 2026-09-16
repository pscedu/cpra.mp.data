from functools import cached_property
from typing import Any, final
from urllib.parse import quote, urlencode

from cpra.mp.data.config import GLYPHS_BASE_URL, SPRITES_BASE_URL, STATIC_TILES_BASE_URL
from cpra.mp.data.legends import LEGENDS, Legend
from cpra.mp.data.variables import Variable
from cpra.mp.data.visualizations.base import DatasetVisualization
from cpra.mp.data.visualizations.display import (
    display_number_format,
    display_unit,
    to_display,
)

# Used when feature IDs are too sparse for array indexing.
COLOR_OBJECT_EXPR = ["coalesce", ["get", ["to-string", ["id"]], ["global-state", "colors"]], "transparent"]
# Used when feature IDs are compact integers (1..N); avoids string conversion and hash lookup.
COLOR_ARRAY_EXPR = ["coalesce", ["at", ["id"], ["global-state", "colors"]], "transparent"]
SELECTION_EXPR = ["get", ["to-string", ["id"]], ["global-state", "selectionColors"]]


class Map(DatasetVisualization):
    """Map visualization for a dataset using MapLibre GL style specification."""

    DEFAULT_BOUNDS = [
        -94.00080091962352,
        28.70426878930118,
        -87.51148588666453,
        31.048795426051505,
    ]

    @final
    class Meta:
        name = "map"
        label = "Map Visualization"
        type = "dataset"
        max_grid_cells = 10

    @cached_property
    def feature_colors(self) -> dict[str, str]:
        """Pre-compute hex colors for all vector grid features.

        Applies legend classification (step or linear interpolation) in Python
        so the map style only needs a single global-state lookup per feature
        rather than a full color expression evaluation per feature per frame.

        Reads the resolved analysis first when `analysis=` is selected -- run through
        EXACTLY the same `compile_pipeline` -> `apply_params` -> `materialize` machinery
        the named-analysis HTTP routes use (see `api.__init__`'s `_compile_analysis_
        pipeline`) -- falling back to a plain `create_analysis` + the same
        `apply_params()` orchestration otherwise, so a plain (unanalyzed) map still
        honors `grid_cell`/`where` -- reaching past `apply_params` to apply only some of
        the options by hand, as this used to, silently skipped them.

        Bug this replaces: the storage format is long (one row per grid cell, after
        grouping/aggregation collapses every other dimension), not the wide,
        one-numeric-named-column-per-grid-cell shape a `cs.matches(r"^[0-9]")` selector
        used to assume -- which matched nothing here and silently returned `{}` for every
        caller, analysis or not.

        Returns:
            A dictionary mapping string grid cell IDs to hex color strings.
        """
        if self.grid.grid_type != "vector":
            return {}

        resolved = self.resolved_analysis
        if resolved is not None:
            import asyncio

            from cpra.mp.data.analysis.planner import compile_pipeline

            compiled = compile_pipeline(resolved.pipeline)
            asyncio.run(compiled.apply_params())
            df = compiled.materialize()
        else:
            import asyncio

            from cpra.mp.data.analysis import create_analysis
            from cpra.mp.data.api.models import DatasetDimensions, FilterDimensions, Options

            dataset_dimensions = DatasetDimensions(
                variable=self.variable.name,
                grid=self.grid.name,
                time_unit=self.time_unit,
            )

            filter_fields = set(FilterDimensions.model_fields.keys())
            filter_dimensions = FilterDimensions(
                **{k: v for k, v in self.params.items() if k in filter_fields and v is not None}
            )

            options_fields = set(Options.model_fields.keys())
            options = Options(**{k: v for k, v in self.params.items() if k in options_fields and v is not None})

            dataset = create_analysis(dataset_dimensions, filter_dimensions, options)
            asyncio.run(dataset.apply_params())
            df = dataset.materialize()

        if df.is_empty() or "grid_cell" not in df.columns or "value" not in df.columns:
            return {}

        values = {
            int(row["grid_cell"]): row["value"]
            for row in df.select("grid_cell", "value").iter_rows(named=True)
            if row["value"] is not None
        }
        if not values:
            return {}

        if self._is_difference():
            stops = self._difference_breaks(self.variable)
            return {str(fid): self._interpolate_stop_color(val, stops) for fid, val in values.items()}

        legend = self._get_legend()
        if legend is None:
            return {}
        return {str(fid): self._classify_step_color(val, legend) for fid, val in values.items()}

    def _classify_step_color(self, value: float, legend: Legend) -> str:
        """Classify a value into a color bucket matching the step expression behavior."""
        ramp = []
        found_bound = False
        for row in legend.break_points.iter_rows(named=True):
            color = row["color"]
            upper_bound = row["upper_bound"]
            found_bound = found_bound or upper_bound is not None
            if not found_bound:
                continue
            ramp.append(color)
            if upper_bound is not None:
                ramp.append(upper_bound)

        if not ramp:
            return "#000000"

        result = ramp[0]
        i = 1
        while i < len(ramp) - 1:
            if value < ramp[i]:
                return result
            result = ramp[i + 1]
            i += 2
        return result

    def _interpolate_stop_color(self, value: float, stops: list) -> str:
        """Linearly interpolate a hex color between value-color stops.

        Args:
            value: The numeric value to colorize.
            stops: Flat list of alternating [v0, c0, v1, c1, ...] pairs sorted by value.
        """
        pairs = list(zip(stops[::2], stops[1::2]))
        if value <= pairs[0][0]:
            return pairs[0][1]
        if value >= pairs[-1][0]:
            return pairs[-1][1]
        for i in range(len(pairs) - 1):
            v0, c0 = pairs[i]
            v1, c1 = pairs[i + 1]
            if v0 <= value <= v1:
                t = (value - v0) / (v1 - v0)
                return self._lerp_hex(c0, c1, t)
        return pairs[-1][1]

    def _lerp_hex(self, c0: str, c1: str, t: float) -> str:
        """Linearly interpolate between two hex colors."""
        r0, g0, b0 = int(c0[1:3], 16), int(c0[3:5], 16), int(c0[5:7], 16)
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r = round(r0 + t * (r1 - r0))
        g = round(g0 + t * (g1 - g0))
        b = round(b0 + t * (b1 - b0))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _id_step_expr(self, color_array: list[str | None]) -> list | None:
        """Build a MapLibre step expression over feature ID from a pre-computed color array.

        Compresses runs of the same color so sequential IDs with identical colors
        are represented by a single entry rather than repeated array elements.
        When used, the colors are baked into the style expression and no global-state
        lookup is needed at render time — only a binary search through run boundaries.

        Returns None if run compression is insufficient to justify the switch from
        the array approach (i.e. step would be at least as large as the array).
        """
        first_id = next((i for i, c in enumerate(color_array) if c is not None), None)
        if first_id is None:
            return None

        runs: list[tuple[int, str]] = []
        for i in range(first_id, len(color_array)):
            c = color_array[i] or "transparent"
            if not runs or c != runs[-1][1]:
                runs.append((i, c))

        # Each additional run adds 2 elements (threshold + color) to the step expression.
        # Only use step if it's meaningfully smaller than the array.
        if len(runs) * 2 >= len(color_array):
            return None

        step: list[Any] = ["step", ["id"], runs[0][1]]
        for start_id, color in runs[1:]:
            step.append(start_id)
            step.append(color)
        return step

    def linear_zoom(self, min_val, max_val) -> list[Any]:
        """Generate a linear zoom interpolation expression.

        Args:
            min_val: Value at zoom level 6.
            max_val: Value at zoom level 12.

        Returns:
            A MapLibre style expression for linear zoom interpolation.
        """
        return ["interpolate", ["linear"], ["zoom"], 6, min_val, 12, max_val]

    def raster_source(
        self,
        bounds: list[float] | None = None,
        max_zoom: int = 13,
        tile_size: int = 256,
        tile_urls: list[str] | None = None,
        color_map: str = "",
    ) -> dict[str, Any]:
        """Generate a raster source definition.

        Args:
            bounds: Bounding box [west, south, east, north].
            max_zoom: Maximum zoom level.
            tile_size: Size of tiles in pixels.
            tile_urls: List of tile URL templates.
            color_map: Color map parameter to append to URLs.

        Returns:
            A MapLibre raster source definition.
        """
        if bounds is None:
            bounds = self.DEFAULT_BOUNDS
        if tile_urls is None:
            tile_urls = []

        tiles = [f"{url}&color_map={quote(color_map)}" if color_map else url for url in tile_urls]

        return {
            "bounds": bounds,
            "maxzoom": max_zoom,
            "type": "raster",
            "tiles": tiles,
            "tileSize": tile_size,
        }

    def difference_raster_source(self, **kwargs) -> dict[str, Any]:
        """Generate a raster source for difference visualization.

        Args:
            **kwargs: Additional arguments passed to raster_source.

        Returns:
            A MapLibre raster source definition with difference color map.
        """
        breaks = self._difference_breaks(self.variable, "#00000000")
        color_map = f"continuous:{','.join(str(b) for b in breaks)}"
        return self.raster_source(color_map=color_map, **kwargs)

    def _difference_breaks(self, variable: Variable, zero_color: str = "#ffffff") -> list:
        """Return flat [value, color, ...] stops for difference visualization."""
        limit = variable.difference_legend_limit or 1
        if limit > 0:
            return [-limit, "#d53e4f", 0, zero_color, limit, "#3288bd"]
        return [limit, "#3288bd", 0, zero_color, -limit, "#d53e4f"]

    def vector_source(self, grid_name: str) -> dict[str, Any]:
        """Generate a vector source definition.

        When a request context is available, points to the dynamic MVT endpoint
        so tile geometry is joined to dataset values server-side. Falls back to
        the static PMTiles URL for offline/notebook use.

        Args:
            grid_name: Name of the grid for the tile URL.

        Returns:
            A MapLibre vector source definition.
        """
        if self.request:
            return {
                "type": "vector",
                "tiles": [self._get_vector_tile_url_template()],
                "minzoom": 0,
                "maxzoom": 14,
            }
        return {
            "type": "vector",
            "url": f"{STATIC_TILES_BASE_URL}/{grid_name}",
        }

    def _get_vector_tile_url_template(self) -> str:
        """Generate the tile URL template for the dynamic MVT endpoint.

        Routes to the named analysis's own vector-tile endpoint when `analysis=` is
        selected. Builds the query string from a known field ALLOWLIST -- the target
        route's own query model fields, plus (once an analysis is selected) whatever
        extra parameters that analysis itself declares (`dimension`/`order`/...) -- rather
        than a denylist of fields to strip: a denylist silently forwards any newly-added
        `VisualizationQuery` field to a route that, once an analysis is selected, rejects
        anything it doesn't recognize outright (its `Parameters` model is `extra=
        "forbid"`; see `analyses/base.py`).
        """
        from cpra.mp.data.api.models import AnalysisQuery, TileQuery

        resolved = self.resolved_analysis

        if resolved is not None:
            base_url = str(
                self.request.url_for(
                    "get_dataset_analysis_vector_tile",
                    variable=self.variable.name,
                    grid=self.grid.name,
                    time_unit=self.time_unit,
                    analysis_name=self._analysis_name,
                    z="{z}",
                    x="{x}",
                    y="{y}",
                )
            )
            allowed = set(AnalysisQuery.model_fields) | set(self._analysis_params)
        else:
            base_url = str(
                self.request.url_for(
                    "get_vector_tile",
                    variable=self.variable.name,
                    grid=self.grid.name,
                    time_unit=self.time_unit,
                    z="{z}",
                    x="{x}",
                    y="{y}",
                )
            )
            allowed = set(TileQuery.model_fields)

        # `self.params` alone is missing a named analysis's own parameters
        # (`dimension`/`order`/...) for a request-driven visualization: `VisualizationQuery`
        # doesn't declare them (it can't -- each analysis declares its own), so FastAPI's
        # query-model binding never puts them in `self.params` at all -- only
        # `_analysis_params`'s own extra scan of the raw query string recovers them (see
        # its docstring in `base.py`). Merging it in here is what actually lets those
        # parameters reach the follow-up tile request; `_analysis_params` never overlaps
        # `self.params`'s own `VisualizationQuery`-declared keys, so this can't clobber
        # anything already there.
        params_source = {**self.params, **self._analysis_params} if resolved is not None else self.params

        query_params = []
        for key, value in params_source.items():
            if value is None or key not in allowed:
                continue
            if isinstance(value, (list, tuple)):
                for v in value:
                    query_params.append((key, v))
            else:
                query_params.append((key, value))
        if query_params:
            return f"{base_url}?{urlencode(query_params)}"
        return base_url

    def _legend_fill_expr(self, is_difference: bool) -> list:
        """Build a MapLibre color expression from the legend using the 'value' property.

        Returns a 'case' expression that shows 'transparent' for features with
        no data, and applies a step (or interpolate for difference maps) otherwise.
        """
        if is_difference:
            stops = self._difference_breaks(self.variable)
            fill_expr: list = ["interpolate", ["linear"], ["get", "value"]] + stops
        else:
            legend = self._get_legend()
            if legend is None:
                return "transparent"

            ramp: list = []
            found_bound = False
            for row in legend.break_points.iter_rows(named=True):
                color = row["color"]
                upper_bound = row["upper_bound"]
                found_bound = found_bound or upper_bound is not None
                if not found_bound:
                    continue
                ramp.append(color)
                if upper_bound is not None:
                    ramp.append(upper_bound)

            if not ramp:
                return "transparent"

            fill_expr = ["step", ["get", "value"]] + ramp

        return ["case", ["has", "value"], fill_expr, "transparent"]

    def projects_source(self, model_group_ids: list[int]) -> dict[str, Any]:
        """Generate a projects source definition.

        Args:
            model_group_ids: Model group ID(s) for project filtering.

        Returns:
            A MapLibre vector source definition for projects.
        """
        params = urlencode({"model_group_id": model_group_ids}, doseq=True)
        return {
            "type": "vector",
            "url": f"{STATIC_TILES_BASE_URL}/f_generate_tile_element_project?{params}",
        }

    def get_base_style(self) -> dict[str, Any]:
        """Generate the base MapLibre style specification.

        Returns:
            A complete MapLibre style specification dictionary.
        """
        return {
            "version": 8,
            "glyphs": f"{GLYPHS_BASE_URL}/{{fontstack}}/{{range}}.pbf",
            "sprite": f"{SPRITES_BASE_URL}/sprite",
            "sources": {
                "basemap-raster": {
                    "type": "raster",
                    "url": f"{STATIC_TILES_BASE_URL}/basemap-raster",
                },
                "basemap-vector": {
                    "type": "vector",
                    "url": f"{STATIC_TILES_BASE_URL}/basemap-vector",
                },
            },
            "layers": [
                {
                    "id": "background",
                    "type": "background",
                    "paint": {"background-color": "#ffffff"},
                },
                {
                    "id": "state",
                    "type": "fill",
                    "source": "basemap-vector",
                    "source-layer": "state",
                    "paint": {"fill-color": "#d6d6d6"},
                },
                {
                    "id": "state-screen",
                    "type": "fill",
                    "source": "basemap-vector",
                    "source-layer": "state",
                    "filter": ["!=", ["get", "abbr"], "LA"],
                    "paint": {"fill-color": "#ffffff", "fill-opacity": 0.4},
                },
                {
                    "id": "water-not-in-icm",
                    "type": "fill",
                    "source": "basemap-vector",
                    "source-layer": "water",
                    "paint": {"fill-color": "#ffffff"},
                },
                {
                    "id": "initial-conditions-water",
                    "type": "raster",
                    "source": "basemap-raster",
                },
                {
                    "id": "levee",
                    "type": "line",
                    "source": "basemap-vector",
                    "source-layer": "levee",
                    "paint": {
                        "line-color": "#939598",
                        "line-opacity": 0.5,
                        "line-width": 0.667,
                    },
                },
                {
                    "id": "road-secondary",
                    "type": "line",
                    "source": "basemap-vector",
                    "source-layer": "road",
                    "filter": ["==", ["get", "element_type_code"], "S"],
                    "paint": {
                        "line-color": "#a4a4a4",
                        "line-opacity": 0.5,
                        "line-width": 1.333,
                    },
                },
                {
                    "id": "road-primary",
                    "type": "line",
                    "source": "basemap-vector",
                    "source-layer": "road",
                    "filter": ["==", ["get", "element_type_code"], "P"],
                    "paint": {"line-color": "#a4a4a4", "line-width": 1.333},
                },
                {
                    "id": "label",
                    "type": "symbol",
                    "source": "basemap-vector",
                    "source-layer": "label",
                    "layout": {
                        "text-field": ["get", "text"],
                        "text-size": [
                            "case",
                            ["==", "community", ["get", "element_type_code"]],
                            16,
                            13,
                        ],
                        "text-font": [
                            "case",
                            ["==", "community", ["get", "element_type_code"]],
                            ["literal", ["Franklin Gothic Demi Regular"]],
                            ["literal", ["Franklin Gothic Book Regular"]],
                        ],
                    },
                    "paint": {"text-color": "#444444"},
                },
            ],
            "transition": {"duration": 0, "delay": 0},
        }

    def raster_layers(self, layer_id: str, source_id: str) -> list[dict[str, Any]]:
        """Generate raster layer definitions.

        Args:
            layer_id: The ID for the raster layer.
            source_id: The source ID for the raster data.

        Returns:
            A list containing the raster layer definition.
        """
        return [{"type": "raster", "id": layer_id, "source": source_id}]

    def vector_layers(
        self,
        layer_id: str,
        source_id: str,
        fill_color: list = COLOR_OBJECT_EXPR,
        line_colors: tuple[str, str] = ("#DDDDDD", "#666666"),
        line_widths: tuple[float, float] = (0.5, 3),
        show_grid_labels: bool = False,
        show_grid_lines: bool = False,
    ) -> list[dict[str, Any]]:
        """Generate vector layer definitions for a grid source.

        Args:
            layer_id: Base ID for the layers.
            source_id: The source ID for the vector data.
            fill_color: MapLibre expression for the fill color.
            line_colors: Tuple of (min_zoom_color, max_zoom_color).
            line_widths: Tuple of (min_zoom_width, max_zoom_width).
            show_grid_labels: Whether to show grid labels.
            show_grid_lines: Whether to show grid lines.

        Returns:
            A list of layer definitions for fill, line, selected, and label.
        """
        return [
            {
                "id": f"{layer_id}-fill",
                "type": "fill",
                "source": source_id,
                "source-layer": "boundary",
                "paint": {
                    "fill-color": fill_color,
                },
            },
            {
                "id": f"{layer_id}-line",
                "type": "line",
                "source": source_id,
                "source-layer": "boundary",
                "layout": {"visibility": "visible" if show_grid_lines else "none"},
                "paint": {
                    "line-color": self.linear_zoom(*line_colors),
                    "line-width": self.linear_zoom(*line_widths),
                },
            },
            {
                "id": f"{layer_id}-selected",
                "type": "line",
                "source": source_id,
                "source-layer": "boundary",
                "paint": {
                    "line-color": [
                        "coalesce",
                        SELECTION_EXPR,
                        "transparent",
                    ],
                    "line-offset": 1.5,
                    "line-width": 3,
                },
            },
            {
                "id": f"{layer_id}-label",
                "type": "symbol",
                "source": source_id,
                "source-layer": "label",
                "minzoom": 9,
                "layout": {
                    "text-field": ["get", "label"],
                    "text-offset": [0, 0],
                    "text-size": 12,
                    "text-font": ["literal", ["Franklin Gothic Demi Regular"]],
                    "visibility": "visible" if show_grid_labels else "none",
                },
                "paint": {
                    "text-halo-blur": 0.5,
                    "text-color": "#000000",
                    "text-halo-color": "#ffffff",
                    "text-halo-width": 2,
                },
            },
        ]

    def project_layers(self, year: int, is_overlay: bool, suffix: str = "") -> list[dict[str, Any]]:
        """Generate project layer definitions.

        Args:
            year: The year filter for projects.
            is_overlay: Whether to render as overlay style.
            suffix: Optional suffix for layer IDs.

        Returns:
            A list of layer definitions for various project types.
        """
        year_filter = ["<=", ["get", "start_year"], year]

        def overlay(yes, no):
            return yes if is_overlay else no

        def project_layer(gis_type: str, project_type_code: str, layer_type: str):
            return {
                "id": f"project-{gis_type.lower()}-{project_type_code.lower()}-{layer_type}{suffix}",
                "type": layer_type,
                "source": "projects",
                "source-layer": "project_element",
                "filter": [
                    "all",
                    ["==", ["get", "gis_type"], gis_type],
                    ["==", ["get", "project_type_code"], project_type_code],
                    year_filter,
                ],
            }

        overlay_stroke_color = self.linear_zoom("rgba(0, 0, 0, 0.1)", "rgba(0, 0, 0, 0.25)")
        overlay_stroke_width = self.linear_zoom(1, 2)
        overlay_fill_color = self.linear_zoom("rgba(0, 0, 0, 0.07)", "rgba(0, 0, 0, 0.15)")

        return [
            # Diversion polygon fill
            {
                **project_layer("Polygon", "DI", "fill"),
                "paint": {
                    "fill-color": overlay(overlay_fill_color, "#000000"),
                    "fill-opacity": overlay(1, 0),
                },
            },
            # Diversion polygon line
            {
                **project_layer("Polygon", "DI", "line"),
                "paint": {
                    "line-color": overlay(overlay_stroke_color, "#ed6731"),
                    "line-dasharray": [2, 1],
                    "line-width": overlay(overlay_stroke_width, self.linear_zoom(1, 3)),
                },
            },
            # Landbridge polygon fill
            {
                **project_layer("Polygon", "LB", "fill"),
                "paint": {"fill-color": overlay(overlay_fill_color, "#5C913A")},
            },
            # Landbridge polygon line
            {
                **project_layer("Polygon", "LB", "line"),
                "paint": {
                    "line-color": overlay_stroke_color,
                    "line-width": overlay_stroke_width,
                },
                "layout": {"visibility": overlay("visible", "none")},
            },
            # Marsh creation polygon fill
            {
                **project_layer("Polygon", "MC", "fill"),
                "paint": overlay(
                    {"fill-color": overlay_fill_color},
                    {"fill-pattern": "green-dots"},
                ),
            },
            # Marsh creation polygon line
            {
                **project_layer("Polygon", "MC", "line"),
                "paint": {
                    "line-color": overlay_stroke_color,
                    "line-width": overlay_stroke_width,
                },
                "layout": {"visibility": overlay("visible", "none")},
            },
            # Bank stabilization line
            {
                **project_layer("Line", "BS", "line"),
                "paint": {
                    "line-color": overlay(overlay_stroke_color, "#D9A73B"),
                    "line-width": overlay(overlay_stroke_width, self.linear_zoom(2, 5)),
                },
            },
            # Structural risk reduction line
            {
                **project_layer("Line", "SR", "line"),
                "paint": {
                    "line-color": "#404042",
                    "line-width": self.linear_zoom(2, 5),
                },
            },
            # Diversion line
            {
                **project_layer("Line", "DI", "line"),
                "paint": {
                    "line-color": overlay(overlay_stroke_color, "#ED6731"),
                    "line-dasharray": [2, 1],
                    "line-width": overlay(overlay_stroke_width, self.linear_zoom(2, 6.75)),
                },
            },
            # Hydrologic restoration line
            {
                **project_layer("Line", "HR", "line"),
                "paint": {
                    "line-color": overlay(overlay_stroke_color, "#1DBEF0"),
                    "line-dasharray": [2, 1],
                    "line-width": overlay(overlay_stroke_width, self.linear_zoom(2, 6.75)),
                },
            },
            # Ridge restoration line
            {
                **project_layer("Line", "RR", "line"),
                "paint": {
                    "line-color": overlay(overlay_stroke_color, "#1C4E7F"),
                    "line-width": overlay(overlay_stroke_width, self.linear_zoom(1.5, 2.5)),
                },
            },
            # Shoreline protection line
            {
                **project_layer("Line", "SP", "line"),
                "paint": {
                    "line-color": overlay(overlay_stroke_color, "#19A598"),
                    "line-width": overlay(overlay_stroke_width, self.linear_zoom(3, 4.5)),
                },
            },
            # Diversion point
            {
                **project_layer("Point", "DI", "circle"),
                "paint": {
                    "circle-color": overlay(overlay_fill_color, "#FFFFFF"),
                    "circle-radius": overlay(self.linear_zoom(1, 3), self.linear_zoom(2, 5)),
                    "circle-stroke-color": overlay(overlay_stroke_color, "#ED6731"),
                    "circle-stroke-width": overlay(overlay_stroke_width, self.linear_zoom(1.5, 4)),
                },
            },
            # Hydrologic restoration point
            {
                **project_layer("Point", "HR", "circle"),
                "paint": {
                    "circle-color": overlay(overlay_fill_color, "#FFFFFF"),
                    "circle-radius": overlay(self.linear_zoom(1, 3), self.linear_zoom(2, 5)),
                    "circle-stroke-color": overlay(overlay_stroke_color, "#1DBEF0"),
                    "circle-stroke-width": overlay(overlay_stroke_width, self.linear_zoom(1.5, 4)),
                },
            },
            # Structural risk reduction point
            {
                **project_layer("Point", "SR", "symbol"),
                "paint": {"icon-opacity": 1},
                "layout": {
                    "icon-allow-overlap": True,
                    "icon-image": "square",
                    "icon-size": self.linear_zoom(0.1, 1),
                },
            },
            # Project labels
            {
                "id": f"project-label-id{suffix}",
                "type": "symbol",
                "source": "projects",
                "source-layer": "project_label",
                "filter": [
                    "all",
                    ["!=", ["get", "project_type_code"], "OR"],
                    year_filter,
                ],
                "maxzoom": 10,
                "minzoom": 8,
                "layout": {
                    "text-anchor": "top",
                    "text-field": ["get", "portal_id"],
                    "text-offset": ["literal", [0, 0.75]],
                    "text-size": 14,
                    "text-font": ["literal", ["Franklin Gothic Demi Regular"]],
                },
                "paint": {
                    "text-color": "#444444",
                    "text-halo-blur": 1,
                    "text-halo-color": "#ffffff",
                    "text-halo-width": 2,
                },
            },
            {
                "id": f"project-label{suffix}",
                "type": "symbol",
                "source": "projects",
                "source-layer": "project_label",
                "filter": [
                    "all",
                    ["!=", ["get", "project_type_code"], "OR"],
                    year_filter,
                ],
                "minzoom": 10,
                "layout": {
                    "text-anchor": "top",
                    "text-field": ["concat", ["get", "project_name"], " (", ["get", "portal_id"], ")"],
                    "text-offset": ["literal", [0, 0.75]],
                    "text-size": 14,
                    "text-font": ["literal", ["Franklin Gothic Demi Regular"]],
                },
                "paint": {
                    "text-color": "#444444",
                    "text-halo-blur": 1,
                    "text-halo-color": "#ffffff",
                    "text-halo-width": 2,
                },
            },
            {
                "id": f"project-icon-dot{suffix}",
                "type": "circle",
                "source": "projects",
                "source-layer": "project_label",
                "filter": ["all", year_filter],
                "minzoom": 8,
                "paint": {
                    "circle-color": [
                        "match",
                        ["get", "project_type_code"],
                        "BH",
                        "#a2e5e0",
                        "BS",
                        "#e7a62c",
                        "DI",
                        "#ea6833",
                        "HR",
                        "#30bded",
                        "LB",
                        "#5c913e",
                        "MC",
                        "#d3df3a",
                        "NS",
                        "#64668b",
                        "OR",
                        "#eca583",
                        "RR",
                        "#204e7e",
                        "SP",
                        "#18a899",
                        "SR",
                        "#414143",
                        "#c8c8c8",
                    ],
                    "circle-radius": self.linear_zoom(2, 6),
                    "circle-stroke-color": "#FFFFFF",
                    "circle-stroke-width": 1,
                },
            },
        ]

    def _get_tile_url_template(self) -> str:
        """Generate the tile URL template for raster data.

        Routes to the named analysis's own tile endpoint when `analysis=` is selected,
        forwarding only that route's own known fields plus the analysis's own declared
        parameters (`dimension`/`order`/...) -- see `_get_vector_tile_url_template`'s
        docstring for why forwarding the full `VisualizationQuery` field set would
        otherwise trip the analysis route's strict (`extra="forbid"`) parameter
        validation. Unchanged (the plain per-dataset tile endpoint) when no analysis was
        selected.

        Returns:
            A tile URL template with {z}/{x}/{y} placeholders.
        """
        resolved = self.resolved_analysis
        analysis_name = self._analysis_name

        if self.request:
            if resolved is not None:
                base_url = str(
                    self.request.url_for(
                        "get_dataset_analysis_tile",
                        variable=self.variable.name,
                        grid=self.grid.name,
                        time_unit=self.time_unit,
                        analysis_name=analysis_name,
                        z="{z}",
                        x="{x}",
                        y="{y}",
                        format="png",
                    )
                )
            else:
                base_url = str(
                    self.request.url_for(
                        "get_model_tile",
                        variable=self.variable.name,
                        grid=self.grid.name,
                        time_unit=self.time_unit,
                        z="{z}",
                        x="{x}",
                        y="{y}",
                        format="png",
                    )
                )
        else:
            base = (
                f"https://api.mpdp.coastal.la.gov/v1/datasets"
                f"/{self.variable.name}/{self.grid.name}/{self.time_unit}"
            )
            base_url = (
                f"{base}/analyses/{analysis_name}/data/{{z}}/{{x}}/{{y}}.png"
                if resolved is not None
                else f"{base}/data/{{z}}/{{x}}/{{y}}.png"
            )

        if resolved is not None:
            from cpra.mp.data.api.models import AnalysisQuery

            allowed = set(AnalysisQuery.model_fields) | set(self._analysis_params)
            # See `_get_vector_tile_url_template`'s identical merge for why
            # `self._analysis_params` has to be folded in here: a named analysis's own
            # parameters (`dimension`/`order`/...) never land in `self.params` itself for a
            # request-driven visualization, since `VisualizationQuery` has no fixed field
            # set for them.
            params_source = {
                k: v for k, v in {**self.params, **self._analysis_params}.items() if k in allowed
            }
        else:
            # Build query string from filter dimensions and options
            exclude = {"height", "width", "start_date", "end_date", "request"}
            params_source = {k: v for k, v in self.params.items() if k not in exclude}

        query_params = []
        for key, value in params_source.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                for v in value:
                    query_params.append((key, v))
            else:
                query_params.append((key, value))

        if query_params:
            return f"{base_url}?{urlencode(query_params)}"
        return base_url

    def chart(self) -> dict[str, Any]:
        """Build the map visualization.

        Returns:
            A MapLibre style specification dictionary with data layers
            and optional project/water overlays.
        """

        style = self.get_base_style()
        style["state"] = {}

        show_projects = self.params.get("show_projects", False)
        show_water = self.params.get("show_water", False)
        show_grid_labels = self.params.get("show_grid_labels", False)
        show_grid_lines = self.params.get("show_grid_lines", False)

        calendar_year = self.params.get("calendar_year", [])
        calendar_day = self.params.get("calendar_day", [])
        model_group_id = self.params.get("model_group_id", [])

        is_difference = self._is_difference()
        is_overlay = self.params.get("is_overlay", True)

        # Insert data layers after "initial-conditions-water" so they render
        # below levees, roads, and labels.
        insert_idx = next(
            (i + 1 for i, layer in enumerate(style["layers"]) if layer["id"] == "initial-conditions-water"),
            len(style["layers"]),
        )

        if self.grid.grid_type == "raster":
            tile_url = self._get_tile_url_template()
            style["sources"]["data"] = (
                self.difference_raster_source(tile_urls=[tile_url])
                if is_difference
                else self.raster_source(tile_urls=[tile_url], color_map=self._legend_name())
            )
            data_layers = self.raster_layers("data-layer", "data")
        else:
            style["sources"]["data"] = self.vector_source(self.grid.name)
            if self.request:
                # Dynamic MVT endpoint: colors derived from the 'value' property
                # embedded in each tile, no global-state lookup needed.
                fill_color = self._legend_fill_expr(is_difference)
            else:
                # Static PMTiles fallback: pre-compute colors server-side and
                # pass via global state for the browser to look up per feature.
                colors = self.feature_colors
                max_id = max((int(k) for k in colors), default=0)
                if colors and max_id < len(colors) * 10:
                    color_array: list[str | None] = [None] * (max_id + 1)
                    for k, v in colors.items():
                        color_array[int(k)] = v
                    step_expr = self._id_step_expr(color_array)
                    if step_expr is not None:
                        fill_color = step_expr
                    else:
                        style["state"]["colors"] = {"default": color_array}
                        fill_color = COLOR_ARRAY_EXPR
                else:
                    style["state"]["colors"] = {"default": colors}
                    fill_color = COLOR_OBJECT_EXPR
            data_layers = self.vector_layers(
                layer_id="data-layer",
                source_id="data",
                fill_color=fill_color,
                show_grid_labels=show_grid_labels,
                show_grid_lines=show_grid_lines,
            )

        # Insert data layers at the correct z-position
        for i, layer in enumerate(data_layers):
            style["layers"].insert(insert_idx + i, layer)

        # Add project overlay if requested (above data, below labels)
        projects_model_group_id = self.params.get("projects_model_group_id") or model_group_id
        if show_projects and projects_model_group_id and (calendar_year or calendar_day):
            style["sources"]["projects"] = self.projects_source(projects_model_group_id)
            project_insert_idx = insert_idx + len(data_layers)
            project_year = calendar_year[0] if calendar_year else calendar_day[0].year
            for i, layer in enumerate(self.project_layers(project_year, is_overlay)):
                style["layers"].insert(project_insert_idx + i, layer)

        # If the water overlay is requested, move the initial conditions water layer to
        # just above data layers.
        if show_water:
            water_insert_idx = insert_idx + len(data_layers)
            style["layers"].insert(water_insert_idx, style["layers"].pop(insert_idx - 1))

        # Include legend metadata for UI rendering
        style["state"]["legend"] = {"default": {"entries": [self._get_legend_metadata(is_difference)]}}

        # Set an empty object as the default value for selection colors.
        # TODO: Add support for setting initial selection colors based on the value
        # of the grid_cell parameter.
        style["state"]["selectionColors"] = {"default": {}}

        return style

    def _is_difference(self) -> bool:
        """Is this a difference result?

        Reads the resolved analysis's own `result_kind` when `analysis=` is selected
        (`difference`'s `Meta.result_kind = "difference"`); a map with no analysis
        selected is never a difference.

        Returns:
            Boolean indicating whether this is a difference result.
        """
        resolved = self.resolved_analysis
        return resolved is not None and resolved.result_kind == "difference"

    def _legend_name(self) -> str | None:
        """The named legend (a `cpra.mp.data.legends.LEGENDS` key) this map's coloring
        should use, or `None` for "no override -- fall back to the variable's own
        default legend" (see `_get_legend`, which does exactly that).

        Reads the resolved analysis's own declared `legend` when `analysis=` is selected
        (`concatenate`'s `Meta.legend = "lnd_chng"`; `difference` declares none); a map
        with no analysis selected has no override.
        """
        resolved = self.resolved_analysis
        return resolved.legend if resolved is not None else None

    def _get_legend(self) -> Legend | None:
        """Get the current legend.

        Returns:
            A Legend instance for the current map.
        """
        return LEGENDS.get(self._legend_name()) or self.variable.legend or LEGENDS.get("default")

    def _get_legend_metadata(self, difference: bool = False) -> dict[str, Any]:
        """Generate legend metadata for the style response.

        Args:
            difference: Whether this is a difference visualization.

        Returns:
            A dictionary with legend title, unit, type, and color steps.
        """
        var = self.variable

        if difference:
            # The client renders these as tick labels, so unlike `_difference_breaks` --
            # which colorizes storage-unit pixels server-side and stays in storage units --
            # the limit is converted for display here.
            limit = to_display(var, var.difference_legend_limit or 1)
            steps = sorted(
                [
                    {"color": "#d53e4f", "value": -1 * limit},
                    {"color": "#ffffff", "value": 0},
                    {"color": "#3288bd", "value": limit},
                ],
                key=lambda s: s["value"],
            )
            return {
                "title": f"{var.label} Difference",
                "subtitle": display_unit(var),
                "type": "continuous",
                "numberFormat": display_number_format(var),
                "steps": steps,
            }

        legend = self._get_legend()
        if legend is None:
            return {}

        steps = []
        for row in legend.break_points.iter_rows(named=True):
            step = {"color": row["color"]}
            if "label" in row and row["label"] is not None:
                step["label"] = row["label"]
            if row["upper_bound"] is not None:
                # Break points are stored in storage units (they colorize pixels); every
                # value in this block is display units, matching `subtitle` below.
                step["value"] = to_display(var, row["upper_bound"])
            steps.append(step)

        return {
            "title": "Land Change" if self._legend_name() == "lnd_chng" else var.label,
            "subtitle": display_unit(var),
            "type": legend.type,
            "steps": steps,
        }

    @property
    def spec(self):
        """Get the map style specification."""
        return self.chart()
