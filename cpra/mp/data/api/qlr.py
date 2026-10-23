"""QGIS layer definition (.qlr) files for the OGC API.

A QLR is the one thing a user can open in QGIS that carries both the data source and its
styling. The OGC items URL alone loads the right data, but QGIS then draws it in one flat
color; this adds the portal's legend for the variable as a graduated renderer, so the layer
matches the map the link was exported from.
"""

import os
import uuid
from urllib.parse import urlencode
from xml.sax.saxutils import escape, quoteattr

import polars as pl

# Imported for its side effect: it gives PYGEOAPI_SERVER_URL its default.
from cpra.mp.data import config  # noqa: F401
from cpra.mp.data.legends import LEGENDS, Legend
from cpra.mp.data.visualizations.display import to_display

# The first and last classes are open-ended in a legend (the last has no upper bound, and
# the first needs to catch values below its upper bound), but QGIS ranges are closed.
_LOWER_LIMIT = -1e9
_UPPER_LIMIT = 1e9

# What one items request asks for. A QGIS GeoJSON source has no paging, so this has to
# cover the largest vector grid in one response; `ogc_config.yaml`'s `max_items` must be at
# least this (tests/api/test_qlr.py checks).
ITEMS_LIMIT = 100000

OUTLINE_COLOR = "35,35,35,255"
COLLECTION = "master_plan_data"


def items_url(items):
    """The OGC items URL for a data request, as a public URL a QGIS install can reach.

    `items` is the data request's `(key, value)` query pairs. The base is the same
    `PYGEOAPI_SERVER_URL` the OGC API builds its own links from, so an export lands on the
    host that serves it.
    """
    base = os.environ["PYGEOAPI_SERVER_URL"].rstrip("/")
    query = urlencode([("f", "json"), ("limit", ITEMS_LIMIT), *items])
    return f"{base}/collections/{COLLECTION}/items?{query}"


# The synthesized difference ramp's colors and how finely it is stepped. The colors are
# `visualizations.map`'s `_difference_breaks`; QGIS has no per-feature interpolation for a
# categorized layer, so the ramp is drawn as steps, nine so that one of them is centered on 0.
_DIFFERENCE_LOW, _DIFFERENCE_MID, _DIFFERENCE_HIGH = (213, 62, 79), (255, 255, 255), (50, 136, 189)
_DIFFERENCE_STEPS = 9


def _hex(rgb):
    return "#" + "".join(f"{round(c):02x}" for c in rgb)


def _format_bound(value):
    """A legend bound to three significant figures, with thousands separators once it is large."""
    rounded = float(f"{value:.3g}")
    return f"{rounded:,.0f}" if abs(rounded) >= 1000 else f"{rounded:g}"


def difference_legend(variable):
    """The diverging legend a difference with no legend of its own is drawn with: red at
    `-limit`, white at 0, blue at `+limit`, off the variable's `difference_legend_limit`.

    Mirrors `visualizations.map`'s synthesized ramp (`_difference_breaks`), including its
    flipped colors for a negative limit and its default limit of 1. Class bounds are in
    storage units, like every legend's, since they are compared with the stored `value`;
    labels are in display units.
    """
    limit = (variable.difference_legend_limit if variable is not None else None) or 1
    low, high = (_DIFFERENCE_LOW, _DIFFERENCE_HIGH) if limit > 0 else (_DIFFERENCE_HIGH, _DIFFERENCE_LOW)
    reach = abs(limit)
    half = _DIFFERENCE_STEPS // 2
    colors = []
    for i in range(_DIFFERENCE_STEPS):
        t = (i - half) / half
        end = low if t < 0 else high
        colors.append(_hex(tuple(m + (e - m) * abs(t) for m, e in zip(_DIFFERENCE_MID, end))))

    bounds = [reach * (2 * i - _DIFFERENCE_STEPS + 2) / _DIFFERENCE_STEPS for i in range(_DIFFERENCE_STEPS - 1)]
    shown = [_format_bound(to_display(variable, b)) for b in bounds]
    labels = [f"< {shown[0]}"]
    labels += [f"{lo} to < {hi}" for lo, hi in zip(shown, shown[1:])]
    labels.append(f">= {shown[-1]}")
    return Legend(
        "difference",
        pl.DataFrame({"label": labels, "color": colors, "upper_bound": [*bounds, None]}),
    )


def export_legend(decoded):
    """The legend a request's layer is styled with, chosen the way the portal's map chooses
    it: the analysis's own named legend, else a difference's synthesized ramp, else the
    variable's legend, else the `default` one.

    `decoded` is a `UnifiedRequest`. A difference that still names a legend (a category
    result) keeps it; only one with none gets the ramp.
    """
    analysis, variable = decoded.analysis, decoded.descriptor.variable
    named = LEGENDS.get(analysis.legend) if analysis is not None and analysis.legend else None
    if named is None and analysis is not None and analysis.result_kind == "difference":
        return difference_legend(variable)
    return named or (variable.legend if variable is not None else None) or LEGENDS.get("default")


def layer_label(decoded):
    """What the layer is called, ahead of the selection: an analysis that names its own
    result (`kind="variable"`) is called that, anything else by its variable plus the
    analysis's title suffix (" Difference")."""
    analysis, variable = decoded.analysis, decoded.descriptor.variable
    if analysis is not None and analysis.meta.kind == "variable":
        return analysis.meta.label
    label = variable.label if variable is not None else "Master Plan data"
    return label + (analysis.title_suffix if analysis is not None else "")


def _rgba(color):
    """`#rrggbb` or `#rrggbbaa` as QGIS's `r,g,b,a` color property."""
    color = color.lstrip("#")
    alpha = int(color[6:8], 16) if len(color) == 8 else 255
    red, green, blue = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return f"{red},{green},{blue},{alpha}"


def _symbol(index, color):
    return f"""<symbol type="fill" name="{index}" alpha="1" clip_to_extent="1" force_rhr="0">
          <layer class="SimpleFill" enabled="1" locked="0" pass="0">
            <Option type="Map">
              <Option name="color" type="QString" value="{_rgba(color)}"/>
              <Option name="outline_color" type="QString" value="{OUTLINE_COLOR}"/>
              <Option name="outline_width" type="QString" value="0.1"/>
              <Option name="outline_style" type="QString" value="solid"/>
              <Option name="style" type="QString" value="solid"/>
            </Option>
          </layer>
        </symbol>"""


def _renderer(legend):
    """A graduated renderer on `value` from a legend's classes.

    A legend row's `upper_bound` closes its class and the previous row's closes the one
    before, which is how `Legend.colorize_pixels` reads them too. A continuous legend is
    drawn as the same steps, since QGIS has no per-feature interpolation for a categorized
    layer.
    """
    ranges, symbols = [], []
    lower = _LOWER_LIMIT
    for index, row in enumerate(legend.break_points.to_dicts()):
        upper = _UPPER_LIMIT if row["upper_bound"] is None else row["upper_bound"]
        ranges.append(
            f'<range render="true" symbol="{index}" lower="{lower}" upper="{upper}" '
            f"label={quoteattr(row['label'])}/>"
        )
        symbols.append(_symbol(index, row["color"]))
        lower = upper

    return f"""<renderer-v2 type="graduatedSymbol" attr="value" graduatedMethod="GraduatedColor" symbollevels="0" enableorderby="0" forceraster="0">
        <ranges>
          {"".join(ranges)}
        </ranges>
        <symbols>
          {"".join(symbols)}
        </symbols>
        <classificationMethod id="Custom"/>
      </renderer-v2>"""


def build_qlr(layer_name, source_url, legend=None):
    """The QLR document for one OGC items layer.

    `legend` is a `legends.Legend`, or `None` for a variable with none, in which case QGIS
    keeps its default single-color style.
    """
    layer_id = f"mpdp_{uuid.uuid4().hex}"
    source = escape(source_url)
    renderer = _renderer(legend) if legend is not None else ""
    return f"""<!DOCTYPE qgis-layer-definition>
<qlr>
  <layer-tree-group name="" checked="Qt::Checked" expanded="1">
    <layer-tree-layer id="{layer_id}" name={quoteattr(layer_name)} providerKey="ogr" source="{source}" checked="Qt::Checked" expanded="1"/>
  </layer-tree-group>
  <maplayers>
    <maplayer type="vector" geometry="Polygon" hasScaleBasedVisibilityFlag="0">
      <id>{layer_id}</id>
      <datasource>{source}</datasource>
      <layername>{escape(layer_name)}</layername>
      <srs>
        <spatialrefsys nativeFormat="Wkt">
          <authid>EPSG:4326</authid>
        </spatialrefsys>
      </srs>
      <provider encoding="UTF-8">ogr</provider>
      {renderer}
    </maplayer>
  </maplayers>
</qlr>
"""
