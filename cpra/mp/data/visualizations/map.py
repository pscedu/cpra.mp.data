from functools import cached_property
from typing import Any, final
from urllib.parse import urlencode

from cpra.mp.data.config import GLYPHS_BASE_URL, SPRITES_BASE_URL, STATIC_TILES_BASE_URL
from cpra.mp.data.legends import LEGENDS, Legend
from cpra.mp.data.variables import Variable
from cpra.mp.data.visualizations.base import DatasetVisualization


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
        max_geographies = 10

    @cached_property
    def feature_values(self) -> dict[int, float]:
        """Load feature values for vector grids.

        Uses the Dataset class to apply filter dimensions, calculation
        parameters (negate, multiply10, multiply100), and group/aggregate
        operations, consistent with the API data pipeline.

        Returns:
            A dictionary mapping geography IDs to their values.
        """
        if self.grid.grid_type != "vector":
            return {}

        import polars.selectors as cs

        from cpra.mp.data.api.dataset import Dataset
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
        options = Options(
            **{k: v for k, v in self.params.items() if k in options_fields and v is not None}
        )

        dataset = Dataset(dataset_dimensions, filter_dimensions, options)
        dataset._apply_calculate_params()
        dataset._apply_group_params()

        df = dataset.lf.collect()

        if df.is_empty():
            return {}

        value_cols = df.select(cs.matches(r"^[0-9]")).columns
        if not value_cols:
            return {}

        # Take the first row of values (single time point after grouping)
        row = df.select(value_cols).row(0, named=True)
        return {int(col): val for col, val in row.items() if val is not None}

    def color_ramp(self, legend: Legend) -> list[Any]:
        """Generate a step color ramp expression from a legend.

        Args:
            legend: A Legend object with break_points.

        Returns:
            A MapLibre style step expression for color ramp.
        """
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
        return ["step", ["get", ["get", "label"], ["global-state", "values"]], *ramp]

    def difference_breaks_colors(
        self, variable: Variable, zero_color: str = "#ffffff"
    ) -> list:
        """Generate color breaks for difference visualization.

        Args:
            variable: A Variable object with difference_legend_limit.
            zero_color: The color to use for zero values.

        Returns:
            A list of [value, color] pairs for interpolation.
        """
        limit = variable.difference_legend_limit or 1
        if limit > 0:
            return [-1 * limit, "#d53e4f", 0, zero_color, limit, "#3288bd"]
        return [limit, "#3288bd", 0, zero_color, limit * -1, "#d53e4f"]

    def difference_color_ramp(self, variable: Variable) -> list[Any]:
        """Generate an interpolated color ramp for difference visualization.

        Args:
            variable: A Variable object with difference_legend_limit.

        Returns:
            A MapLibre style expression for linear color interpolation.
        """
        return [
            "interpolate",
            ["linear"],
            ["get", ["get", "label"], ["global-state", "values"]],
            *self.difference_breaks_colors(variable),
        ]

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

        tiles = [f"{url}&color_map={color_map}" if color_map else url for url in tile_urls]

        return {
            "bounds": bounds,
            "maxzoom": max_zoom,
            "type": "raster",
            "tiles": tiles,
            "tileSize": tile_size,
        }

    def difference_raster_source(
        self, variable: Variable, **kwargs
    ) -> dict[str, Any]:
        """Generate a raster source for difference visualization.

        Args:
            variable: A Variable object with difference_legend_limit.
            **kwargs: Additional arguments passed to raster_source.

        Returns:
            A MapLibre raster source definition with difference color map.
        """
        breaks = self.difference_breaks_colors(variable, "#00000000")
        color_map = f"continuous:{','.join(str(b) for b in breaks)}"
        return self.raster_source(color_map=color_map, **kwargs)

    def vector_source(self, grid_name: str) -> dict[str, Any]:
        """Generate a vector source definition.

        Args:
            grid_name: Name of the grid for the tile URL.

        Returns:
            A MapLibre vector source definition.
        """
        return {
            "type": "vector",
            "url": f"{STATIC_TILES_BASE_URL}/{grid_name}",
        }

    def projects_source(self, model_group_id: int) -> dict[str, Any]:
        """Generate a projects source definition.

        Args:
            model_group_id: Model group ID for project filtering.

        Returns:
            A MapLibre vector source definition for projects.
        """
        params = urlencode({"model_group_id": model_group_id})
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
        difference: bool = False,
        line_colors: tuple[str, str] = ("#DDDDDD", "#666666"),
        line_widths: tuple[float, float] = (0.5, 3),
        show_grid_labels: bool = False,
        show_grid_lines: bool = False,
        variable: Variable | None = None,
    ) -> list[dict[str, Any]]:
        """Generate vector layer definitions with embedded feature values.

        For vector grids, this method loads the data and pre-computes colors
        for each geography ID, embedding them directly in the style using
        match expressions.

        Args:
            layer_id: Base ID for the layers.
            source_id: The source ID for the vector data.
            difference: Whether to use difference color ramp.
            line_colors: Tuple of (min_zoom_color, max_zoom_color).
            line_widths: Tuple of (min_zoom_width, max_zoom_width).
            show_grid_labels: Whether to show grid labels.
            show_grid_lines: Whether to show grid lines.
            variable: Variable object for color ramp (uses self.variable if None).

        Returns:
            A list of layer definitions for fill, line, selected, and label.
        """
        var = variable or self.variable

        legend = var.legend or LEGENDS.get("default")
        ramp = self.difference_color_ramp(var) if difference else self.color_ramp(legend)

        return [
            {
                "id": f"{layer_id}-fill",
                "type": "fill",
                "source": source_id,
                "source-layer": "boundary",
                "paint": {
                    "fill-color": [
                        "case",
                        ["==", ["get", ["get", "label"], ["global-state", "values"]], None],
                        "transparent",
                        ramp
                    ],
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
                        ["feature-state", "selected"],
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

    def project_layers(
        self, year: int, is_overlay: bool, suffix: str = ""
    ) -> list[dict[str, Any]]:
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

        overlay_stroke_color = self.linear_zoom(
            "rgba(0, 0, 0, 0.1)", "rgba(0, 0, 0, 0.25)"
        )
        overlay_stroke_width = self.linear_zoom(1, 2)
        overlay_fill_color = self.linear_zoom(
            "rgba(0, 0, 0, 0.07)", "rgba(0, 0, 0, 0.15)"
        )

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
                    "line-width": overlay(
                        overlay_stroke_width, self.linear_zoom(2, 6.75)
                    ),
                },
            },
            # Hydrologic restoration line
            {
                **project_layer("Line", "HR", "line"),
                "paint": {
                    "line-color": overlay(overlay_stroke_color, "#1DBEF0"),
                    "line-dasharray": [2, 1],
                    "line-width": overlay(
                        overlay_stroke_width, self.linear_zoom(2, 6.75)
                    ),
                },
            },
            # Ridge restoration line
            {
                **project_layer("Line", "RR", "line"),
                "paint": {
                    "line-color": overlay(overlay_stroke_color, "#1C4E7F"),
                    "line-width": overlay(
                        overlay_stroke_width, self.linear_zoom(1.5, 2.5)
                    ),
                },
            },
            # Shoreline protection line
            {
                **project_layer("Line", "SP", "line"),
                "paint": {
                    "line-color": overlay(overlay_stroke_color, "#19A598"),
                    "line-width": overlay(
                        overlay_stroke_width, self.linear_zoom(3, 4.5)
                    ),
                },
            },
            # Diversion point
            {
                **project_layer("Point", "DI", "circle"),
                "paint": {
                    "circle-color": overlay(overlay_fill_color, "#FFFFFF"),
                    "circle-radius": overlay(
                        self.linear_zoom(1, 3), self.linear_zoom(2, 5)
                    ),
                    "circle-stroke-color": overlay(overlay_stroke_color, "#ED6731"),
                    "circle-stroke-width": overlay(
                        overlay_stroke_width, self.linear_zoom(1.5, 4)
                    ),
                },
            },
            # Hydrologic restoration point
            {
                **project_layer("Point", "HR", "circle"),
                "paint": {
                    "circle-color": overlay(overlay_fill_color, "#FFFFFF"),
                    "circle-radius": overlay(
                        self.linear_zoom(1, 3), self.linear_zoom(2, 5)
                    ),
                    "circle-stroke-color": overlay(overlay_stroke_color, "#1DBEF0"),
                    "circle-stroke-width": overlay(
                        overlay_stroke_width, self.linear_zoom(1.5, 4)
                    ),
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
                "id": f"project-label{suffix}",
                "type": "symbol",
                "source": "projects",
                "source-layer": "project_label",
                "filter": [
                    "all",
                    ["!=", ["get", "project_type_code"], "OR"],
                    year_filter,
                ],
                "minzoom": 9,
                "layout": {
                    "text-anchor": "top",
                    "text-field": ["get", "project_name"],
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
                        "BH", "#a2e5e0",
                        "BS", "#e7a62c",
                        "DI", "#ea6833",
                        "HR", "#30bded",
                        "LB", "#5c913e",
                        "MC", "#d3df3a",
                        "NS", "#64668b",
                        "OR", "#eca583",
                        "RR", "#204e7e",
                        "SP", "#18a899",
                        "SR", "#414143",
                        "#c8c8c8",
                    ],
                    "circle-radius": self.linear_zoom(2, 6),
                    "circle-stroke-color": "#FFFFFF",
                    "circle-stroke-width": 1,
                },
            },
        ]

    def water_overlay_layers(self, source_id: str = "basemap-vector") -> list[dict[str, Any]]:
        """Generate water overlay layer definitions.

        Args:
            source_id: The source ID for the water overlay.

        Returns:
            A list of layer definitions for water overlay.
        """
        return [
            {
                "id": "water-overlay",
                "type": "fill",
                "source": source_id,
                "source-layer": "water",
                "paint": {
                    "fill-color": "#a6cee3",
                    "fill-opacity": 0.6,
                },
            }
        ]

    def _get_tile_url_template(self) -> str:
        """Generate the tile URL template for raster data.

        Returns:
            A tile URL template with {z}/{x}/{y} placeholders.
        """
        if self.request:
            base_url = str(self.request.url_for(
                "get_model_tile",
                variable=self.variable.name,
                grid=self.grid.name,
                time_unit=self.time_unit,
                z="{z}",
                x="{x}",
                y="{y}",
                format="png",
            ))
        else:
            base_url = (
                f"https://api.mpdp.coastal.la.gov/v1/datasets"
                f"/{self.variable.name}/{self.grid.name}/{self.time_unit}"
                f"/data/{{z}}/{{x}}/{{y}}.png"
            )

        # Build query string from filter dimensions and options
        query_params = []
        for key, value in self.params.items():
            if value is None:
                continue
            if key in ("height", "width", "start_date", "end_date", "request"):
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

        is_overlay = self.params.get("is_overlay", True)

        # Insert data layers after "initial-conditions-water" so they render
        # below levees, roads, and labels.
        insert_idx = next(
            (i + 1 for i, layer in enumerate(style["layers"]) if layer["id"] == "initial-conditions-water"),
            len(style["layers"]),
        )

        if self.grid.grid_type == "raster":
            tile_url = self._get_tile_url_template()
            style["sources"]["data"] = self.raster_source(tile_urls=[tile_url])
            data_layers = self.raster_layers("data-layer", "data")
        else:
            style["sources"]["data"] = self.vector_source(self.grid.name)
            style["state"]["values"] = {"default": self.feature_values}
            data_layers = self.vector_layers(
                layer_id="data-layer",
                source_id="data",
                difference=self.params.get("negate") is not None,
                show_grid_labels=show_grid_labels,
                show_grid_lines=show_grid_lines,
            )

        # Insert data layers at the correct z-position
        for i, layer in enumerate(data_layers):
            style["layers"].insert(insert_idx + i, layer)

        # Add project overlay if requested (above data, below labels)
        if show_projects and model_group_id and (calendar_year or calendar_day):
            style["sources"]["projects"] = self.projects_source(model_group_id[0])
            project_insert_idx = insert_idx + len(data_layers)
            project_year = calendar_year[0] if calendar_year else calendar_day[0].year
            for i, layer in enumerate(self.project_layers(project_year, is_overlay)):
                style["layers"].insert(project_insert_idx + i, layer)

        # Add water overlay if requested (just above data layers)
        if show_water:
            water_insert_idx = insert_idx + len(data_layers)
            for i, layer in enumerate(self.water_overlay_layers()):
                style["layers"].insert(water_insert_idx + i, layer)

        # Include legend metadata for UI rendering
        difference = self.params.get("negate") is not None
        style["state"]["legend"] = {"default": {"entries": [self._get_legend_metadata(difference)]}}

        return style

    def _get_legend_metadata(self, difference: bool = False) -> dict[str, Any]:
        """Generate legend metadata for the style response.

        Args:
            difference: Whether this is a difference visualization.

        Returns:
            A dictionary with legend title, unit, type, and color steps.
        """
        var = self.variable

        if difference:
            limit = var.difference_legend_limit or 1
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
                "unit": var.unit,
                "type": "continuous",
                "steps": steps,
            }

        legend = var.legend or LEGENDS.get("default")
        if legend is None:
            return {}

        steps = []
        for row in legend.break_points.iter_rows(named=True):
            step = {"color": row["color"]}
            if "label" in row and row["label"] is not None:
                step["label"] = row["label"]
            if row["upper_bound"] is not None:
                step["value"] = row["upper_bound"]
            steps.append(step)

        return {
            "title": var.label,
            "unit": var.unit,
            "type": legend.type,
            "steps": steps,
        }
