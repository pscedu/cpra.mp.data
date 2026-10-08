from functools import cached_property
from typing import Any, final
from urllib.parse import quote, urlencode

from cpra.mp.data.config import GLYPHS_BASE_URL, SPRITE_NAME, SPRITES_BASE_URL, STATIC_TILES_BASE_URL
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
        # A VISUALIZATION's `Meta.type`, which is not the retired `AnalysisMeta.type`: it is
        # still served by `/v1/visualizations.json` (`api.models.Visualization`). Route
        # selection no longer reads it -- `get_visualization_class` lost its `type` argument,
        # and the one visualization route tells a dataset visualization apart by
        # `issubclass(..., DatasetVisualization)` -- but the catalog entry stays.
        type = "dataset"
        max_grid_cells = 10
        # The only visualization here that does NOT plot time: a map draws one grid cell per
        # pixel at whatever instant the filters pinned, so it is the one that can present an
        # analysis whose output has no time column left (a `diff` over `calendar_year`).
        # See `DatasetVisualization.validate_time_axis`, which rejects the
        # combination everywhere else, and `api.models.Visualization`, which serves this so a
        # client can offer the same chart types the server will accept.
        time_axis = False

    @cached_property
    def feature_colors(self) -> dict[str, str]:
        """Pre-compute hex colors for all vector grid features.

        Applies legend classification (step or linear interpolation) in Python
        so the map style only needs a single global-state lookup per feature
        rather than a full color expression evaluation per feature per frame.

        Reads `self.decoded.pipeline` -- the SAME `Pipeline` `/v1/data.csv` would run for
        this request, analysis or not, and the same one the tile URLs below re-request. There
        is no fork here any more: the no-`analysis=` branch used to lower this map's own
        dataset and filters onto a pipeline by hand (`_data_query_pipeline`, via
        `request.data_query.build_data_query_pipeline`), which was a second place a filter
        had to mean the same thing. The result then goes through EXACTLY the
        `validate_shapes` -> `compile_plan` -> `execute` path every HTTP response does --
        this reads data, so it must not be a second, quieter engine.

        Synchronous throughout, unlike the routes: a visualization builds its spec from
        cached properties with no event loop to await on. `execute` itself is sync (it
        returns a lazy frame), and `decoded` resolves `grid_cell` through
        `api.unified.decode_request_sync`.

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

        from cpra.mp.data.analysis.execution.execute import execute
        from cpra.mp.data.analysis.execution.plan import compile_plan
        from cpra.mp.data.analysis.request.ops import validate_shapes

        pipeline = self.decoded.pipeline

        # A vector grid's final frame is always long (a `project` INTO a raster grid is
        # the only way to reach band-wide, and nothing here builds one), so `execute`
        # returns a `LongFrame` -- a plain `pl.LazyFrame`, not the band-wide 3-tuple.
        validate_shapes(pipeline)
        df = execute(compile_plan(pipeline)).collect()

        if df.is_empty() or "grid_cell" not in df.columns or "value" not in df.columns:
            return {}

        values = {
            int(row["grid_cell"]): row["value"]
            for row in df.select("grid_cell", "value").iter_rows(named=True)
            if row["value"] is not None
        }
        if not values:
            return {}

        if self._uses_difference_ramp():
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

    def _difference_breaks(self, variable: Variable | None, zero_color: str = "#ffffff") -> list:
        """Return flat [value, color, ...] stops for difference visualization.

        `variable` is `None` for a result that carries no single presentation variable, which
        has no declared limit of its own; the symmetric +/-1 default covers it, as it already
        did for a variable that declares none.
        """
        limit = (variable.difference_legend_limit if variable is not None else None) or 1
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
                "tiles": [self._tile_url_template("pbf")],
                "minzoom": 0,
                "maxzoom": 14,
            }
        return {
            "type": "vector",
            "url": f"{STATIC_TILES_BASE_URL}/{grid_name}",
        }

    def _tile_url_template(self, format: str) -> str:
        """The `/v1/data/{z}/{x}/{y}.{format}` template this map's tiles are fetched from.

        One builder for both tile kinds now, and one route under both: `format="pbf"` is the
        vector source's MVT endpoint and `format="png"` the raster source's. The four routes
        this replaced (plain/analysis x vector/raster) differed only in which segments of the
        request they carried in the path, and the whole request is in the query string here.

        No field ALLOWLIST either. It existed because a follow-up request's target route
        bound a DIFFERENT query model than the visualization route did, so the extra fields
        had to be named and dropped; the two routes now run the same partition, and the only
        category that is this route's alone is the render options `forward_query` subtracts.
        Compression goes with them: a tile response carries none, and
        `partition_query(allow_compression=False)` rejects one outright rather than ignoring
        it -- while `Compression`'s fields have non-`None` defaults and so are always bound.
        """
        if self.request:
            base_url = str(
                self.request.url_for("get_data_tile", z="{z}", x="{x}", y="{y}", format=format)
            )
        else:
            base_url = f"https://api.mpdp.coastal.la.gov/v1/data/{{z}}/{{x}}/{{y}}.{format}"

        query_params = [*self.dataset_params, *self.forward_query(compression=False)]

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
            "glyphs": f"{GLYPHS_BASE_URL}/font/{{fontstack}}/{{range}}",
            "sprite": f"{SPRITES_BASE_URL}/sprite/{SPRITE_NAME}",
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

        uses_ramp = self._uses_difference_ramp()
        is_overlay = self.params.get("is_overlay", True)

        # Insert data layers after "initial-conditions-water" so they render
        # below levees, roads, and labels.
        insert_idx = next(
            (i + 1 for i, layer in enumerate(style["layers"]) if layer["id"] == "initial-conditions-water"),
            len(style["layers"]),
        )

        if self.grid.grid_type == "raster":
            tile_url = self._tile_url_template("png")
            style["sources"]["data"] = (
                self.difference_raster_source(tile_urls=[tile_url])
                if uses_ramp
                else self.raster_source(tile_urls=[tile_url], color_map=self._legend_name())
            )
            data_layers = self.raster_layers("data-layer", "data")
        else:
            style["sources"]["data"] = self.vector_source(self.grid.name)
            if self.request:
                # Dynamic MVT endpoint: colors derived from the 'value' property
                # embedded in each tile, no global-state lookup needed.
                fill_color = self._legend_fill_expr(uses_ramp)
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
        style["state"]["legend"] = {"default": {"entries": [self._get_legend_metadata(uses_ramp)]}}

        # Set an empty object as the default value for selection colors.
        # TODO: Add support for setting initial selection colors based on the value
        # of the grid_cell parameter.
        style["state"]["selectionColors"] = {"default": {}}

        return style

    def _is_difference(self) -> bool:
        """Is this a difference result?

        Reads the resolved analysis's own `result_kind` when `analysis=` is selected
        (`diff`'s `Meta.result_kind = "difference"`); a map with no analysis
        selected is never a difference.

        Returns:
            Boolean indicating whether this is a difference result.
        """
        resolved = self.resolved_analysis
        return resolved is not None and resolved.result_kind == "difference"

    def _uses_difference_ramp(self) -> bool:
        """Should this map's coloring be the SYNTHESIZED diverging ramp
        (`_difference_breaks`: red at `-limit`, white/transparent at `0`, blue at `+limit`,
        off the variable's `difference_legend_limit`) rather than a named legend?

        A difference with no named legend, which is every difference there was until
        `lnd_conf` declared a `Meta.difference_legend`. The ramp exists because a
        difference of an arbitrary variable has no classification anyone has written down:
        it is derived from the variable's own scale, and is the honest default for that.
        An analysis that HAS written one down (`lnd_conf,diff` ->
        `lnd_conf_diff`, whose classes are "likely land gain"/"high confidence no
        change"/... rather than a symmetric ramp) says something the limit cannot, so it
        wins -- and this is the one place that has to be decided, since all four coloring
        branches (`feature_colors`, the raster source, `_legend_fill_expr`, and the legend
        metadata) read it.

        Distinct from `_is_difference`, which stays the plain question about the RESULT and
        still answers `True` here -- `difference_suffix` and the title still say
        "Difference" for a difference that happens to have a legend of its own.
        """
        return self._is_difference() and self._legend_name() is None

    def _legend_name(self) -> str | None:
        """The named legend (a `cpra.mp.data.legends.LEGENDS` key) this map's coloring
        should use, or `None` for "no override -- fall back to the variable's own
        default legend" (see `_get_legend`, which does exactly that).

        Reads the resolved analysis's own declared `legend` when `analysis=` is selected
        (`lnd_chng`'s `Meta.legend = "lnd_chng"`; `diff` declares none); a map
        with no analysis selected has no override. For a CHAIN this is what
        `resolve_analysis_chain` settled on, which is where a leading analysis's
        `Meta.difference_legend` takes over once a stage has differenced its output
        (`lnd_conf,diff` -> `lnd_conf_diff`) -- nothing about that resolution is redone
        here.
        """
        resolved = self.resolved_analysis
        return resolved.legend if resolved is not None else None

    def _get_legend(self) -> Legend | None:
        """Get the current legend.

        Returns:
            A Legend instance for the current map.
        """
        variable_legend = self.variable.legend if self.variable is not None else None
        return LEGENDS.get(self._legend_name()) or variable_legend or LEGENDS.get("default")

    def _get_legend_metadata(self, difference: bool = False) -> dict[str, Any]:
        """Generate legend metadata for the style response.

        Args:
            difference: Whether this map is colored by the synthesized diverging ramp --
                `_uses_difference_ramp`, not `_is_difference`. A difference that names a
                legend of its own (`lnd_conf,diff`) takes the branch below, so that the
                key the client renders matches the colors the tiles actually came back in.

        Returns:
            A dictionary with legend title, unit, type, and color steps.
        """
        # `None` for a result carrying no single presentation variable (`lnd_type_plus`).
        # Every `display.*` helper below is the identity for it, and `result_label` falls
        # back to the analysis's own `Meta.label` for the title.
        var = self.variable

        if difference:
            # The client renders these as tick labels, so unlike `_difference_breaks` --
            # which colorizes storage-unit pixels server-side and stays in storage units --
            # the limit is converted for display here.
            limit = to_display(var, (var.difference_legend_limit if var is not None else None) or 1)
            steps = sorted(
                [
                    {"color": "#d53e4f", "value": -1 * limit},
                    {"color": "#ffffff", "value": 0},
                    {"color": "#3288bd", "value": limit},
                ],
                key=lambda s: s["value"],
            )
            return {
                "title": f"{self.result_label} Difference",
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

        resolved = self.resolved_analysis
        title = (
            resolved.meta.label
            if resolved is not None and resolved.meta.kind == "variable"
            else self.result_label
        )
        # A difference reaching THIS branch is one with a legend of its own, and its title
        # still has to say what the reader is looking at. `" Difference"` verbatim rather
        # than `difference_suffix`: that is the chain's whole `title_suffix`, which for
        # `lnd_conf,diff` is `" Confidence Difference"` and would repeat the word already
        # inside `Meta.label` ("Land Confidence").
        if self._is_difference():
            title = f"{title} Difference"
        return {
            "title": title,
            "subtitle": display_unit(var),
            "type": legend.type,
            "steps": steps,
        }

    @property
    def spec(self):
        """Get the map style specification."""
        return self.chart()
