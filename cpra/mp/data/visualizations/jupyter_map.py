import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

# Default center and zoom for Louisiana
_DEFAULT_CENTER = [-90.756, 29.877]
_DEFAULT_ZOOM = 7

try:
    from maplibre import MapOptions
    from maplibre.ipywidget import MapWidget
except ImportError:
    MapOptions = None  # type: ignore[assignment,misc]
    MapWidget = None  # type: ignore[assignment,misc]

def _ensure_legend_env() -> None:
    """Point legend env vars at bundled legend.js/css if not already set.

    The fork reads os.path.isfile(val) → reads content → sends inline to JS.
    JS creates a blob URL for import() and injects CSS as a <style> tag,
    so no browser URL resolution or CORS is needed.
    """
    os.environ.setdefault("PYMAPLIBREGL_LEGEND_JS", str((Path(__file__).parent / "jupyter_assets/legend.js").resolve()))
    os.environ.setdefault("PYMAPLIBREGL_LEGEND_CSS", str((Path(__file__).parent / "jupyter_assets/legend.css").resolve()))


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
        base = "https://api.mpdp.coastal.la.gov"
        if "base_url" in kwargs:
            base = kwargs.pop("base_url")
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
        _ensure_legend_env()

        with urlopen(self.url) as resp:
            style = json.loads(resp.read())

        state = style.get("state", {})
        values = (state.get("values") or {}).get("default", {})
        selection_colors = (state.get("selectionColors") or {}).get("default", {})
        legend = (state.get("legend") or {}).get("default", {})

        widget = MapWidget(
            MapOptions(
                style=style,
                center=_DEFAULT_CENTER,
                zoom=_DEFAULT_ZOOM,
            )
        )
        widget.set_global_state_property("values", values)
        widget.set_global_state_property("selectionColors", selection_colors)
        widget.set_global_state_property("legend", legend)

        return widget

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
