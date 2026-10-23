"""The QGIS layer file: its builder and the /v1/view.qlr route."""

from pathlib import Path
from urllib.parse import parse_qsl, urlparse
from xml.etree import ElementTree

from types import SimpleNamespace

import polars as pl
import pytest
import yaml

from cpra.mp.data.api.qlr import (
    ITEMS_LIMIT,
    build_qlr,
    difference_legend,
    export_legend,
    items_url,
    layer_label,
)
from cpra.mp.data.legends import LEGENDS, Legend
from cpra.mp.data.variables import VARIABLES

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


def decoded(variable=None, analysis=None):
    """The two parts of a `UnifiedRequest` the legend and label choices read."""
    return SimpleNamespace(descriptor=SimpleNamespace(variable=variable), analysis=analysis)


def analysis(legend=None, result_kind="value", kind="analysis", label="Analysis", title_suffix=""):
    return SimpleNamespace(
        legend=legend,
        result_kind=result_kind,
        title_suffix=title_suffix,
        meta=SimpleNamespace(kind=kind, label=label),
    )


class TestExportLegend:
    """The same precedence the portal's map colors by."""

    def test_the_variables_own_legend_by_default(self):
        assert export_legend(decoded(VARIABLES["sal"])) is VARIABLES["sal"].legend

    def test_an_analysis_legend_wins_over_the_variables(self):
        assert export_legend(decoded(VARIABLES["sal"], analysis(legend="elev"))) is LEGENDS["elev"]

    def test_a_difference_without_a_legend_gets_the_synthesized_ramp(self):
        legend = export_legend(decoded(VARIABLES["sal"], analysis(result_kind="difference")))
        assert legend.name == "difference"

    def test_a_difference_with_a_legend_of_its_own_keeps_it(self):
        legend = export_legend(decoded(VARIABLES["sal"], analysis(legend="elev", result_kind="difference")))
        assert legend is LEGENDS["elev"]

    def test_falls_back_to_the_default_legend_with_no_variable(self):
        assert export_legend(decoded()) is LEGENDS["default"]


class TestDifferenceLegend:
    def test_ramp_is_symmetric_about_zero_with_a_white_middle_class(self):
        legend = difference_legend(VARIABLES["damage"])
        rows = legend.break_points.to_dicts()
        bounds = [r["upper_bound"] for r in rows[:-1]]
        assert rows[-1]["upper_bound"] is None
        assert bounds == pytest.approx([-b for b in reversed(bounds)])
        assert rows[len(rows) // 2]["color"] == "#ffffff"

    def test_a_positive_limit_runs_red_to_blue_and_a_negative_one_blue_to_red(self):
        positive = SimpleNamespace(difference_legend_limit=5, display_scale_factor=None)
        negative = SimpleNamespace(difference_legend_limit=-5, display_scale_factor=None)
        assert difference_legend(positive).break_points["color"][0] == "#d53e4f"
        assert difference_legend(negative).break_points["color"][0] == "#3288bd"

    def test_a_variable_with_no_limit_defaults_to_one(self):
        legend = difference_legend(SimpleNamespace(difference_legend_limit=None, display_scale_factor=None))
        assert max(b for b in legend.break_points["upper_bound"] if b is not None) == pytest.approx(7 / 9)

    def test_labels_are_in_display_units_and_short(self):
        variable = SimpleNamespace(difference_legend_limit=9000, display_scale_factor=1000)
        labels = difference_legend(variable).break_points["label"].to_list()
        assert labels[0] == "< -7"
        assert labels[-1] == ">= 7"

    def test_large_bounds_get_separators_not_exponents(self):
        labels = difference_legend(VARIABLES["damage"]).break_points["label"].to_list()
        assert labels[0] == "< -7,780,000"
        assert not any("e+" in label for label in labels)


class TestLayerLabel:
    def test_the_variable_label_with_no_analysis(self):
        assert layer_label(decoded(VARIABLES["sal"])) == "Salinity"

    def test_an_analysis_that_names_its_own_result_is_called_that(self):
        assert layer_label(decoded(VARIABLES["sal"], analysis(kind="variable", label="Land Type"))) == "Land Type"

    def test_otherwise_the_variable_plus_the_title_suffix(self):
        assert layer_label(decoded(VARIABLES["sal"], analysis(title_suffix=" Difference"))) == "Salinity Difference"

    def test_a_generic_label_when_no_single_variable_resolves(self):
        assert layer_label(decoded()) == "Master Plan data"


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

    def test_a_difference_is_styled_with_the_diverging_ramp_and_named_for_it(self, client):
        query = (
            f"{SALINITY.rsplit('&calendar_day', 1)[0].replace('&model_group_id=500', '')}"
            "&analysis=diff&dimension=model_group_id&order=asc&model_group_id=500&model_group_id=521"
        )
        response = client.get(f"/v1/view.qlr?{query}")
        assert response.status_code == 200, response.text
        layer = layer_of(response.text)
        assert layer.find("layername").text.startswith("Salinity Difference")
        labels = [r.get("label") for r in layer.findall("renderer-v2/ranges/range")]
        assert len(labels) == 9 and labels[0].startswith("< -")

    def test_raster_grid_is_rejected(self, client):
        response = client.get("/v1/view.qlr?variable=vct&time_unit=annual&grid=veg_grid_cell_v001")
        assert response.status_code == 400
        assert "raster" in response.json()["detail"]

    def test_unusable_request_fails_at_export(self, client):
        assert client.get("/v1/view.qlr").status_code == 400
        assert client.get("/v1/view.qlr?variable=nope&time_unit=daily&grid=hydro_compartment_v001").status_code >= 400
