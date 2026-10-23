"""The QGIS layer file: its builder and the /v1/view.qlr route."""

from pathlib import Path
from urllib.parse import parse_qsl, urlparse
from xml.etree import ElementTree

import polars as pl
import yaml

from cpra.mp.data.api.qlr import ITEMS_LIMIT, build_qlr, items_url
from cpra.mp.data.legends import Legend

SALINITY = (
    "variable=sal&time_unit=daily&grid=hydro_compartment_v001&model_group_id=500"
    "&scenario_id=7&aggregate_type=mean&calendar_day=2019-01-01"
)
OGC_CONFIG = Path(__file__).parents[2] / "cpra/mp/data/ogc_config.yaml"


def legend():
    return Legend(
        "test",
        pl.DataFrame(
            {
                "label": ["< 10", "10 to < 20", ">= 20"],
                "color": ["#ff0000", "#00ff00", "#0000ff80"],
                "upper_bound": [10.0, 20.0, None],
            }
        ),
    )


def layer_of(document):
    return ElementTree.fromstring(document.split("\n", 1)[1]).find("maplayers/maplayer")


class TestBuildQlr:
    def test_graduated_renderer_has_one_range_per_legend_class(self):
        renderer = layer_of(build_qlr("Test", "http://x/items", legend())).find("renderer-v2")
        assert renderer.get("type") == "graduatedSymbol"
        assert renderer.get("attr") == "value"
        ranges = renderer.findall("ranges/range")
        assert [r.get("label") for r in ranges] == ["< 10", "10 to < 20", ">= 20"]

    def test_ranges_are_contiguous_and_open_at_both_ends(self):
        """The first class has to catch values below its upper bound (elevation goes
        negative), and the last has no upper bound in a legend at all."""
        ranges = layer_of(build_qlr("Test", "http://x/items", legend())).findall(
            "renderer-v2/ranges/range"
        )
        bounds = [(float(r.get("lower")), float(r.get("upper"))) for r in ranges]
        assert bounds == [(-1e9, 10.0), (10.0, 20.0), (20.0, 1e9)]

    def test_symbol_colors_are_rgba_and_keep_an_alpha_channel(self):
        layer = layer_of(build_qlr("Test", "http://x/items", legend()))
        colors = [
            option.get("value")
            for option in layer.findall("renderer-v2/symbols/symbol/layer/Option/Option")
            if option.get("name") == "color"
        ]
        assert colors == ["255,0,0,255", "0,255,0,255", "0,0,255,128"]

    def test_range_symbols_refer_to_defined_symbols(self):
        layer = layer_of(build_qlr("Test", "http://x/items", legend()))
        defined = {s.get("name") for s in layer.findall("renderer-v2/symbols/symbol")}
        assert {r.get("symbol") for r in layer.findall("renderer-v2/ranges/range")} == defined

    def test_source_and_name_are_xml_escaped(self):
        layer = layer_of(build_qlr("A & B <c>", "http://x/items?a=1&b=2", legend()))
        assert layer.find("datasource").text == "http://x/items?a=1&b=2"
        assert layer.find("layername").text == "A & B <c>"

    def test_no_legend_leaves_qgis_its_default_style(self):
        layer = layer_of(build_qlr("Test", "http://x/items"))
        assert layer.find("renderer-v2") is None


class TestItemsUrl:
    def test_carries_the_data_request_and_a_limit_that_covers_a_whole_grid(self, monkeypatch):
        monkeypatch.setenv("PYGEOAPI_SERVER_URL", "https://example.org/v1/ogc/")
        parsed = urlparse(items_url([("variable", "sal"), ("model_group_id", "500")]))
        assert parsed.geturl().startswith("https://example.org/v1/ogc/collections/master_plan_data/items?")
        assert parse_qsl(parsed.query) == [
            ("f", "json"),
            ("limit", str(ITEMS_LIMIT)),
            ("variable", "sal"),
            ("model_group_id", "500"),
        ]

    def test_repeated_dimension_keys_survive(self, monkeypatch):
        monkeypatch.setenv("PYGEOAPI_SERVER_URL", "https://example.org/v1/ogc")
        query = urlparse(items_url([("model_group_id", "500"), ("model_group_id", "520")])).query
        assert [v for k, v in parse_qsl(query) if k == "model_group_id"] == ["500", "520"]

    def test_limit_does_not_exceed_what_the_server_will_serve(self):
        max_items = yaml.safe_load(OGC_CONFIG.read_text())["server"]["limits"]["max_items"]
        assert ITEMS_LIMIT <= max_items


class TestViewQlrRoute:
    def test_returns_a_downloadable_layer_pointing_at_the_items_url(self, client):
        response = client.get(f"/v1/view.qlr?{SALINITY}")
        assert response.status_code == 200
        assert response.headers["content-disposition"].startswith("attachment")
        layer = layer_of(response.text)
        query = dict(parse_qsl(urlparse(layer.find("datasource").text).query))
        assert query["variable"] == "sal"
        assert query["calendar_day"] == "2019-01-01"
        assert query["f"] == "json"

    def test_styles_the_layer_with_the_variables_legend(self, client):
        layer = layer_of(client.get(f"/v1/view.qlr?{SALINITY}").text)
        labels = [r.get("label") for r in layer.findall("renderer-v2/ranges/range")]
        assert labels[0] == "0 to < 0.1"
        assert labels[-1] == "20.0 to 35.0"

    def test_names_the_layer_for_the_selection(self, client):
        name = layer_of(client.get(f"/v1/view.qlr?{SALINITY}").text).find("layername").text
        assert name.startswith("Salinity")
        assert "calendar_day=2019-01-01" in name
        assert "grid=" not in name

    def test_ogc_params_in_the_request_are_not_duplicated(self, client):
        layer = layer_of(client.get(f"/v1/view.qlr?{SALINITY}&f=html&limit=5").text)
        pairs = parse_qsl(urlparse(layer.find("datasource").text).query)
        assert [v for k, v in pairs if k == "f"] == ["json"]
        assert [v for k, v in pairs if k == "limit"] == [str(ITEMS_LIMIT)]

    def test_raster_grid_is_rejected(self, client):
        response = client.get("/v1/view.qlr?variable=vct&time_unit=annual&grid=veg_grid_cell_v001")
        assert response.status_code == 400
        assert "raster" in response.json()["detail"]

    def test_unusable_request_fails_at_export(self, client):
        assert client.get("/v1/view.qlr").status_code == 400
        assert client.get("/v1/view.qlr?variable=nope&time_unit=daily&grid=hydro_compartment_v001").status_code >= 400
