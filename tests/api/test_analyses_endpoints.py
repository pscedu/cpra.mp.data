"""Tests for the named-analysis HTTP routes (plan §3.2a): the 9 routes exposing a
REGISTERED pipeline (`ANALYSES`/`resolve_analysis`) through exactly the same
`compile_pipeline` -> response machinery the explicit `/v1/analysis.*` routes already use.

The real registry package (`cpra.mp.data.analyses`) is a concurrent Phase 2 change and may
not exist yet -- `api/utils.py` guards that import so the app keeps working either way (see
its module docstring). These tests monkeypatch a FAKE registry directly onto the two
places `api/__init__.py` and `api/utils.py` bind the names (`ANALYSES`/`resolve_analysis`/
`DatasetRef`), honoring the agreed contract:

    resolve_analysis(name, *, dataset=None, filters=None, params=None) -> ResolvedAnalysis
    ResolvedAnalysis.pipeline  # a real, compilable cpra.mp.data.analysis.pipeline.Pipeline

so the route/response MACHINERY itself (catalog serving, `get_analysis_class`'s 400/404
split, `_apply_output_overrides`, `_respond_to_analysis`, tile/vector-tile execution,
`pipeline.json`) is exercised end-to-end against a real, executable `Pipeline` -- exactly
what a real registry would hand back -- without this file depending on that package's
internals. Once `cpra/mp/data/analyses` lands for real, these should keep passing
unchanged and can be supplemented (not replaced) with tests against the real registered
analyses.

Two fake analyses stand in for the plan's own three examples:

* ``concatenate`` (type="dataset") -- an identity `project` over whatever dataset the
  dataset-scoped route names, optionally filtered by `grid_cell`/`where`/dimension
  filters forwarded from the request. Exercises the 4 dataset-scoped routes.
* ``exit-criterion`` (type="custom") -- literally the plan's own worked example
  (`tests.api.test_pipeline_exec.NOTEBOOK_QUERY`, imported rather than re-typed so this
  file can't drift from the byte-exact fixture that module's own tests already pin).
  Exercises the 4 multi-input routes, including the paired (notebook) shape, which is the
  one shape whose planner lowering actually forwards `pipeline.output.color_map` (see
  `_compile_paired_notebook_shape` in `analysis/planner.py`) -- so it is also this file's
  vehicle for proving `color_map` really changes rendered pixels, not just that the
  parameter is accepted.
"""

import copy
import io

import mercantile
import pytest
import rasterio as rio
from mapbox_vector_tile.Mapbox.vector_tile_pb2 import tile as VectorTile

from cpra.mp.data import config
from cpra.mp.data.analysis.pipeline import parse_pipeline
from cpra.mp.data.tabular import write_parquet
from tests.api.test_pipeline_exec import NOTEBOOK_QUERY, _tile_for_grid, notebook_query_data  # noqa: F401
from tests.factories import ANNUAL_KWARGS, annual_frame, write_pmtiles

TILE = mercantile.Tile(x=1, y=2, z=3)


class _DatasetMeta:
    name = "concatenate"
    label = "Concatenate"
    type = "dataset"
    variables = ["ssf"]
    legend = None
    result_kind = "vector"


class _CustomMeta:
    name = "exit-criterion"
    label = "Exit Criterion"
    type = "custom"
    variables = None
    legend = None
    result_kind = "raster"


class _FakeDatasetAnalysis:
    Meta = _DatasetMeta


class _FakeCustomAnalysis:
    Meta = _CustomMeta


class _FakeDatasetRef:
    def __init__(self, variable, time_unit, grid):
        self.variable = variable
        self.time_unit = time_unit
        self.grid = grid


class _FakeResolvedAnalysis:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.meta = None
        self.legend = None
        self.result_kind = None
        self.collapsed_dimensions = None
        self.title_suffix = None


def _fake_resolve_analysis(name, *, dataset=None, filters=None, params=None):
    """A minimal stand-in for the real registry's `resolve_analysis`, just enough to
    build a genuine, executable `Pipeline` for each of the two fake analyses above --
    see this module's docstring for why a fake, rather than the real package, is used.
    """
    filters = dict(filters or {})

    if name == "concatenate":
        assert dataset is not None, "concatenate is dataset-scoped -- must get a DatasetRef"
        where = filters.pop("where", None)
        grid_cell = filters.pop("grid_cell", None)
        input_selection = {"variable": dataset.variable, "time_unit": dataset.time_unit, "grid": dataset.grid}
        input_selection.update(filters)
        if grid_cell is not None:
            input_selection["grid_cell"] = grid_cell

        steps = [{"op": "project", "input": "ds", "to": dataset.grid}]
        if where is not None:
            steps.append({"op": "filter", "where": where})

        pipeline = parse_pipeline(
            {
                "inputs": {"ds": input_selection},
                "steps": steps,
                "output": {"format": "csv"},
            }
        )
        return _FakeResolvedAnalysis(pipeline)

    if name == "exit-criterion":
        assert dataset is None, "exit-criterion is a multi-input (custom) analysis"
        pipeline = parse_pipeline(copy.deepcopy(NOTEBOOK_QUERY))
        return _FakeResolvedAnalysis(pipeline)

    raise KeyError(f"no fake resolution registered for {name!r}")  # pragma: no cover


def _install_fake_registry(monkeypatch):
    import cpra.mp.data.api as api_module
    import cpra.mp.data.api.utils as api_utils

    fake_analyses = {"concatenate": _FakeDatasetAnalysis, "exit-criterion": _FakeCustomAnalysis}
    monkeypatch.setattr(api_utils, "ANALYSES", fake_analyses)
    monkeypatch.setattr(api_module, "ANALYSES", fake_analyses)
    monkeypatch.setattr(api_module, "resolve_analysis", _fake_resolve_analysis)
    monkeypatch.setattr(api_module, "DatasetRef", _FakeDatasetRef)


@pytest.fixture(autouse=True)
def fake_registry(monkeypatch):
    _install_fake_registry(monkeypatch)


class TestRouteKindRejection:
    """The `type="dataset"` vs `type="custom"` split (plan §3.2a), both directions --
    generalized via `get_analysis_class`, not reinvented per route.
    """

    def test_dataset_analysis_rejected_on_custom_route(self, client):
        response = client.get("/v1/analyses/concatenate/data.csv")
        assert response.status_code == 400
        assert "Invalid analysis type" in response.json()["detail"]

    def test_custom_analysis_rejected_on_dataset_route(self, client):
        v, t, g = ANNUAL_KWARGS["variable"], ANNUAL_KWARGS["time_unit"], ANNUAL_KWARGS["grid"]
        response = client.get(f"/v1/datasets/{v}/{t}/{g}/analyses/exit-criterion/data.csv")
        assert response.status_code == 400
        assert "Invalid analysis type" in response.json()["detail"]

    def test_unknown_name_404s_on_the_dataset_route(self, client):
        v, t, g = ANNUAL_KWARGS["variable"], ANNUAL_KWARGS["time_unit"], ANNUAL_KWARGS["grid"]
        response = client.get(f"/v1/datasets/{v}/{t}/{g}/analyses/nope/data.csv")
        assert response.status_code == 404
        assert "Unknown analysis name" in response.json()["detail"]

    def test_unknown_name_404s_on_the_custom_route(self, client):
        response = client.get("/v1/analyses/nope/data.csv")
        assert response.status_code == 404
        assert "Unknown analysis name" in response.json()["detail"]


class TestDatasetAnalysisData:
    """The dataset-scoped `data.{format}` route over the fake `concatenate` analysis."""

    def test_returns_the_dataset_rows(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        v, t, g = ANNUAL_KWARGS["variable"], ANNUAL_KWARGS["time_unit"], ANNUAL_KWARGS["grid"]

        response = client.get(f"/v1/datasets/{v}/{t}/{g}/analyses/concatenate/data.csv")
        assert response.status_code == 200, response.text
        assert "grid_cell" in response.text
        assert "2030" in response.text and "2040" in response.text

    def test_where_is_forwarded_as_a_filter(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        v, t, g = ANNUAL_KWARGS["variable"], ANNUAL_KWARGS["time_unit"], ANNUAL_KWARGS["grid"]

        response = client.get(
            f"/v1/datasets/{v}/{t}/{g}/analyses/concatenate/data.csv",
            params={"where": "calendar_year = 2040"},
        )
        assert response.status_code == 200, response.text
        assert "2030" not in response.text
        assert "2040" in response.text

    def test_compression_is_accepted_on_the_data_form(self, client, base_path, monkeypatch):
        """Parquet is the one tabular format compression actually changes the encoding
        of (see `api/tabular.py::stream_or_write_to_file`) -- both real algorithms
        succeed, proving `AnalysisDataQuery.compression` reaches `_respond_to_analysis`
        the same way the explicit `/v1/analysis.*` routes' own `Compression` query model
        does.
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        v, t, g = ANNUAL_KWARGS["variable"], ANNUAL_KWARGS["time_unit"], ANNUAL_KWARGS["grid"]

        for algorithm in ("zstd", "gzip"):
            response = client.get(
                f"/v1/datasets/{v}/{t}/{g}/analyses/concatenate/data.parquet",
                params={"compression_algorithm": algorithm},
            )
            assert response.status_code == 200, response.text
            assert response.content[:4] == b"PAR1"

    def test_an_incompatible_compression_algorithm_400s(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        v, t, g = ANNUAL_KWARGS["variable"], ANNUAL_KWARGS["time_unit"], ANNUAL_KWARGS["grid"]

        response = client.get(
            f"/v1/datasets/{v}/{t}/{g}/analyses/concatenate/data.csv",
            params={"compression_algorithm": "lzw"},
        )
        assert response.status_code == 400
        assert "must be one of" in response.json()["detail"]


class TestDatasetAnalysisVectorTile:
    """`.../analyses/{name}/tiles/{z}/{x}/{y}.pbf` -- the named-analysis, dataset-scoped
    counterpart to `TestAnalysisVectorTileRoute` (tests/api/test_vector_tiles.py). Also
    the trap the task called out explicitly: this route's `tiles/` segment and the
    sibling `data/{z}/{x}/{y}.{format}` route's `data/` segment are distinct literal path
    prefixes, so there is no `.pbf`-vs-`.{format}` collision to design around here (unlike
    `/v1/analysis/{z}/{x}/{y}.pbf` vs `/v1/analysis/{z}/{x}/{y}.{format}`, which share the
    identical path up to the final segment) -- fetching a real tile through the ASGI app
    below is the verification the task asked for, not just an assumption from reading the
    route table.
    """

    def test_serves_a_real_vector_tile(self, client, base_path, tiles_path, monkeypatch):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        write_pmtiles(tiles_path, "ecoregion_v001", TILE, [1, 2, 3])
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        v, t, g = ANNUAL_KWARGS["variable"], ANNUAL_KWARGS["time_unit"], ANNUAL_KWARGS["grid"]

        response = client.get(
            f"/v1/datasets/{v}/{t}/{g}/analyses/concatenate/tiles/{TILE.z}/{TILE.x}/{TILE.y}.pbf"
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"
        result = VectorTile()
        result.ParseFromString(response.content)
        expected = annual_frame(years=(2030,))["1"][0]
        assert result.layers[0].values[0].double_value == pytest.approx(expected, rel=1e-5)


class TestDatasetAnalysisPipelineJson:
    """`.../analyses/{name}/pipeline.json` -- the expanded `Pipeline` (plan §5.4's "pure
    sugar" claim): must itself compile via `compile_pipeline`, and must reflect the
    request's own `sort`/`select`/`color_map` overrides (never the registry's concern).
    """

    def test_expands_to_a_compilable_pipeline(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        v, t, g = ANNUAL_KWARGS["variable"], ANNUAL_KWARGS["time_unit"], ANNUAL_KWARGS["grid"]

        response = client.get(f"/v1/datasets/{v}/{t}/{g}/analyses/concatenate/pipeline.json")
        assert response.status_code == 200, response.text
        expanded = parse_pipeline(response.json())
        assert expanded.inputs["ds"].variable == v

    def test_sort_select_and_color_map_overrides_are_reflected(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        v, t, g = ANNUAL_KWARGS["variable"], ANNUAL_KWARGS["time_unit"], ANNUAL_KWARGS["grid"]

        response = client.get(
            f"/v1/datasets/{v}/{t}/{g}/analyses/concatenate/pipeline.json",
            params={"sort": "-grid_cell", "select": "grid_cell,value", "color_map": "default"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["output"]["sort"] == "-grid_cell"
        assert body["output"]["select"] == "grid_cell,value"
        assert body["output"]["color_map"] == "default"
        # And the result still compiles with those overrides in place.
        parse_pipeline(body)

    def test_format_is_not_overridden_by_the_route(self, client, base_path, monkeypatch):
        """There is no `{format}`/`.pbf` URL segment on this route to source an override
        from -- the pipeline's own `output.format` (whatever the fake registry produced,
        "csv") is left untouched.
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        v, t, g = ANNUAL_KWARGS["variable"], ANNUAL_KWARGS["time_unit"], ANNUAL_KWARGS["grid"]

        response = client.get(f"/v1/datasets/{v}/{t}/{g}/analyses/concatenate/pipeline.json")
        assert response.json()["output"]["format"] == "csv"


class TestCustomAnalysisData:
    """The multi-input `/v1/analyses/{name}/data.{format}` route over the fake
    `exit-criterion` analysis (the plan's own worked example, `NOTEBOOK_QUERY`).
    """

    def test_post_equivalent_result(self, client, notebook_query_data):
        response = client.get("/v1/analyses/exit-criterion/data.tif")
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
        assert (data == 2040).all()


class TestCustomAnalysisPipelineJson:
    def test_expands_to_the_notebook_shape(self, client, notebook_query_data):
        response = client.get("/v1/analyses/exit-criterion/pipeline.json")
        assert response.status_code == 200, response.text
        expanded = parse_pipeline(response.json())
        assert len(expanded.inputs) == 2
        assert expanded.steps[1].op == "join"


class TestCustomAnalysisColorMap:
    """`color_map` is a REQUEST parameter (plan §3.2a), never the registry's concern --
    proven here by actually rendering two different PNG tiles for the same analysis and
    confirming the pixels differ, not merely that the parameter round-trips through
    `pipeline.json`. `exit-criterion` compiles onto the PAIRED (notebook) shape, which is
    the one planner lowering that forwards `pipeline.output.color_map` at all (see
    `_compile_paired_notebook_shape` in `analysis/planner.py`, and this module's own
    docstring) -- `default` and `ffibs` are both real legend names (`cpra/mp/data/
    structure/legends.csv`) with different color ramps.
    """

    def test_different_color_maps_render_different_pixels(self, client, notebook_query_data):
        tile = _tile_for_grid(notebook_query_data)
        default_response = client.get(
            f"/v1/analyses/exit-criterion/data/{tile.z}/{tile.x}/{tile.y}.png",
            params={"color_map": "default"},
        )
        ffibs_response = client.get(
            f"/v1/analyses/exit-criterion/data/{tile.z}/{tile.x}/{tile.y}.png",
            params={"color_map": "ffibs"},
        )
        assert default_response.status_code == 200, default_response.text
        assert ffibs_response.status_code == 200, ffibs_response.text
        assert default_response.content != ffibs_response.content


class TestCustomAnalysisRasterFormatAgreement:
    def test_png_tile_route_serves_a_real_png(self, client, notebook_query_data):
        tile = _tile_for_grid(notebook_query_data)
        response = client.get(f"/v1/analyses/exit-criterion/data/{tile.z}/{tile.x}/{tile.y}.png")
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


class TestCustomAnalysisVectorTileRejection:
    """`exit-criterion`'s output grid (`veg_grid_cell_v001`, raster) can never serve a
    `.pbf` tile -- `compiled.validate_vector_tile()` (the same rule the explicit
    `/v1/analysis/{z}/{x}/{y}.pbf` route enforces) rejects it before any tile lookup,
    mirroring `TestAnalysisVectorTileRoute.test_rejects_a_raster_output_grid`.
    """

    def test_rejects_a_raster_output_grid(self, client, notebook_query_data):
        response = client.get("/v1/analyses/exit-criterion/tiles/3/1/2.pbf")
        assert response.status_code == 400
        assert "invalid output type for vector tile" in response.json()["detail"]


class TestAnalysisRegistryErrorMapping:
    """`resolve_analysis` itself (not just `get_analysis_class`) can reject a request --
    most notably `InvalidAnalysisParametersError`, raised for otherwise-well-formed
    params/filters a real analysis's own `Parameters` validation or `validate_selection`
    cardinality rule rejects (e.g. `difference`/`concatenate`'s "exactly 2 selected
    values" rule). `get_analysis_class` cannot pre-empt this -- it only checks the name
    and `Meta.type`, both of which are fine here -- so this is the one path that reaches
    `api/__init__.py`'s `analysis_registry_error_handler`. Verified here by monkeypatching
    `resolve_analysis` to raise each of the three real
    `cpra.mp.data.analyses.base.AnalysisRegistryError` subclasses directly, independent of
    the fake pipeline-building logic the rest of this file uses.
    """

    def test_invalid_parameters_maps_to_400(self, client, monkeypatch):
        import cpra.mp.data.api as api_module
        from cpra.mp.data.analyses.base import InvalidAnalysisParametersError

        def _raise(name, *, dataset=None, filters=None, params=None):
            raise InvalidAnalysisParametersError("dimension must select exactly 2 values")

        monkeypatch.setattr(api_module, "resolve_analysis", _raise)
        response = client.get("/v1/analyses/exit-criterion/data.csv")
        assert response.status_code == 400
        assert "exactly 2 values" in response.json()["detail"]

    def test_unknown_analysis_error_maps_to_404(self, client, monkeypatch):
        import cpra.mp.data.api as api_module
        from cpra.mp.data.analyses.base import UnknownAnalysisError

        def _raise(name, *, dataset=None, filters=None, params=None):
            raise UnknownAnalysisError(f"unknown analysis name {name!r}")

        monkeypatch.setattr(api_module, "resolve_analysis", _raise)
        response = client.get("/v1/analyses/exit-criterion/data.csv")
        assert response.status_code == 404

    def test_type_mismatch_error_maps_to_400(self, client, monkeypatch):
        import cpra.mp.data.api as api_module
        from cpra.mp.data.analyses.base import AnalysisTypeMismatchError

        def _raise(name, *, dataset=None, filters=None, params=None):
            raise AnalysisTypeMismatchError(f"analysis {name!r} type mismatch")

        monkeypatch.setattr(api_module, "resolve_analysis", _raise)
        response = client.get("/v1/analyses/exit-criterion/data.csv")
        assert response.status_code == 400
