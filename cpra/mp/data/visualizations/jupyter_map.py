import json
from urllib.parse import urlencode
from urllib.request import urlopen

from cpra.mp.data.config import MPD_URI

_FALLBACK_BASE = "https://api.mpdp.coastal.la.gov"

# Default center and zoom for Louisiana
_DEFAULT_CENTER = [-90.756, 29.877]
_DEFAULT_ZOOM = 7

try:
    from maplibre import MapOptions
    from maplibre.ipywidget import MapWidget
except ImportError:
    MapOptions = None  # type: ignore[assignment,misc]
    MapWidget = None  # type: ignore[assignment,misc]


def _resolve_global_state(obj, values: dict, selection_colors: dict):
    if isinstance(obj, list):
        if len(obj) == 2 and obj[0] == "global-state":
            return ["literal", values if obj[1] == "values" else selection_colors]
        return [_resolve_global_state(item, values, selection_colors) for item in obj]
    if isinstance(obj, dict):
        return {k: _resolve_global_state(v, values, selection_colors) for k, v in obj.items()}
    return obj


class JupyterMap:
    """Jupyter widget that renders a MapLibre GL map from the map visualization API.

    Accepts the same dataset dimensions and filter kwargs as the server-side Map
    visualization. Builds a URL pointing to the map.json endpoint and delegates
    rendering to py-maplibregl's MapWidget.

    Usage::

        from cpra.mp.data.visualizations import JupyterMap

        m = JupyterMap("elev", "morph_pixel_v001", "annual", calendar_year=2030)
        m  # renders in Jupyter cell
    """

    def __init__(self, variable: str, grid: str, time_unit: str, **kwargs):
        # base = MPD_URI or _FALLBACK_BASE
        base = _FALLBACK_BASE
        path = f"{base}/v1/datasets/{variable}/{grid}/{time_unit}/visualization/map.json"

        params = []
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple)):
                for v in value:
                    params.append((key, v))
            else:
                params.append((key, value))

        query = urlencode(params)
        self.url = f"{path}?{query}" if query else path

    def _make_widget(self) -> "MapWidget":
        with urlopen(self.url) as resp:
            style = json.loads(resp.read())

        state = style.get("state", {})
        values = (state.get("values") or {}).get("default", {})
        selection_colors = (state.get("selectionColors") or {}).get("default", {})

        style = _resolve_global_state(style, values, selection_colors)

        return MapWidget(
            MapOptions(
                style=style,
                center=_DEFAULT_CENTER,
                zoom=_DEFAULT_ZOOM,
            )
        )

    def _repr_mimebundle_(self, **kwargs):
        """Jupyter auto-display: renders the map as an ipywidget.

        Called automatically when a Jupyter cell returns a JupyterMap instance.
        MapWidget uses _repr_mimebundle_ (not _repr_html_) as its display protocol.
        """
        if MapWidget is None or MapOptions is None:
            raise ImportError(
                "maplibre is required for Jupyter rendering. "
                "Install it with: pip install 'cpra.mp.data[jupyter]'"
            )
        return self._make_widget()._repr_mimebundle_(**kwargs)

    def show(self):
        """Explicitly display the map widget.

        Use this when rendering inside a function or loop where Jupyter does not
        automatically call _repr_mimebundle_ (i.e. the map is not the last
        expression in a cell).
        """
        if MapWidget is None or MapOptions is None:
            raise ImportError(
                "maplibre is required for Jupyter rendering. "
                "Install it with: pip install 'cpra.mp.data[jupyter]'"
            )
        from IPython.display import display

        display(self._make_widget())
