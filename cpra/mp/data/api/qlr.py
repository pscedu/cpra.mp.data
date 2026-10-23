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

# Imported for its side effect: it gives PYGEOAPI_SERVER_URL its default.
from cpra.mp.data import config  # noqa: F401

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
