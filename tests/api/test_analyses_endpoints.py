"""Tests for the named-analysis HTTP routes: the 8 routes exposing a REGISTERED pipeline
(`cpra.mp.data.analyses`'s `ANALYSES`/`resolve_analysis`) through exactly the same
`api.pipeline` responders the explicit `/v1/analysis.*` routes already use.

These drive the REAL registry end to end -- no fake `resolve_analysis`. The route layer's
whole claim is that a named analysis is pure sugar: it builds `resolve_analysis`'s inputs,
applies the request's own `Output` overrides, and hands the result to the same responder a
hand-authored pipeline reaches. A fake registry can only prove the first and third of
those, and it cannot notice the one thing most likely to break -- the registry and the
route disagreeing about what `filters`/`grid_cell`/`params` mean.

**Which registered analysis stands in for what.**

* `difference` (`type="dataset"`) carries the four dataset-scoped routes. It is exercised
  over BOTH grid kinds: `tests/factories.py`'s `ssf`/`ecoregion_v001` (vector) for the
  tabular and `.pbf` routes, and its `vct`/`veg_grid_cell_v001` (raster) for the `png`
  tile route, which needs a band-wide final frame. Substituting these for
  `Meta.variables`' real-world datasets follows that module's own documented convention
  (the engine cares about grid TYPES and SHAPES, not variable semantics).
* `exit-criterion` (`type="custom"`) carries the custom routes. It is the one registered
  analysis that is band-wide end to end -- a vector-to-raster `project` feeding a
  band-wide `join`, then a `filter` naming that join's operand value columns -- so the
  two routes that execute it are also this module's only coverage of that shape reaching
  a client. What the shape itself does is pinned at the engine level
  (`tests/api/test_analyses_registry.py::TestExitCriterionExecution`); what these pin is
  that the custom routes deliver its bytes.

`grid_cell` is the one request parameter the registry deliberately cannot handle alone:
resolving a coordinate against real geometry needs `io/`, which nothing under `request/`
(or in `analyses/`) may import. The dataset routes therefore resolve it first and pass it
as `resolve_analysis(grid_cell_where=...)`; `TestDatasetAnalysisGridCell` covers both
spellings and the custom routes' deliberate lack of the same treatment.
"""

import io

import mercantile
import pytest
import rasterio as rio
from mapbox_vector_tile.Mapbox.vector_tile_pb2 import tile as VectorTile
from rasterio.crs import CRS
from rasterio.warp import transform, transform_bounds
from shapely.geometry import box

from cpra.mp.data import config
from cpra.mp.data.analysis.request.pipeline import parse_pipeline
from tests.factories import (
    ANNUAL_KWARGS,
    RASTER_KWARGS,
    write_annual_dataset,
    write_grid_geometry,
    write_pmtiles,
    write_raster_dataset,
)

TILE = mercantile.Tile(x=1, y=2, z=3)

V, T, G = ANNUAL_KWARGS["variable"], ANNUAL_KWARGS["time_unit"], ANNUAL_KWARGS["grid"]
RV, RT, RG = RASTER_KWARGS["variable"], RASTER_KWARGS["time_unit"], RASTER_KWARGS["grid"]

DATASET_URL = f"/v1/datasets/{V}/{T}/{G}/analyses/difference"
RASTER_DATASET_URL = f"/v1/datasets/{RV}/{RT}/{RG}/analyses/difference"

# `difference` needs its own `dimension`/`order` parameters plus a two-valued filter on
# that dimension; the remaining three are `ssf`'s other required dimensions, which
# `required_dataset_dimensions` puts in the `group` step's `by` list.
DIFFERENCE_QUERY = {
    "dimension": "calendar_year",
    "order": "asc",
    "calendar_year": [2030, 2040],
    "model_group_id": [ANNUAL_KWARGS["model_group_id"]],
    "scenario_id": [ANNUAL_KWARGS["scenario_id"]],
    "percentile": [ANNUAL_KWARGS["percentile"]],
}

# `vct` declares no `percentile`, so the raster form drops it.
RASTER_DIFFERENCE_QUERY = {
    "dimension": "calendar_year",
    "order": "asc",
    "calendar_year": [2030, 2040],
    "model_group_id": [RASTER_KWARGS["model_group_id"]],
    "scenario_id": [RASTER_KWARGS["scenario_id"]],
}

# `exit-criterion` is `type="custom"`: no dataset and no dimension filters, just its own
# two required parameters.
EXIT_CRITERION_QUERY = {"model_group_id": 500, "scenario_id": 7}


def _lng_lat(x, y):
    """Project an `ecoregion_v001` (EPSG:26915) point to WGS84, as a `grid_cell=` value."""
    lng, lat = [c[0] for c in transform(CRS.from_epsg(26915), CRS.from_epsg(4326), [x], [y])]
    return f"{lng!s},{lat!s}"


def _tile_for_grid(base_path, zoom=8):
    """A tile covering the middle of whatever raster this fixture wrote."""
    with rio.open(next(base_path.rglob("*.tif"))) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return mercantile.tile((west + east) / 2, (south + north) / 2, zoom)


@pytest.fixture
def vector_dataset(base_path, monkeypatch):
    """Two years of `ssf`/`ecoregion_v001`, both holding the SAME value per geography --
    so `difference`'s asc order (negate the later year, then sum) leaves exactly 0
    everywhere, a result no unfiltered passthrough could produce by accident.
    """
    write_annual_dataset(base_path, years=(2030, 2040), value_fn=lambda year, geo_id: 100.0 + geo_id)
    monkeypatch.setattr(config, "BASE_PATH", base_path)
    return base_path


@pytest.fixture
def ecoregion_geometry(geometry_path):
    """A single 1km square at `ecoregion_v001` cell 5, for coordinate lookups."""
    return write_grid_geometry(geometry_path, G, [(5, box(500000, 3300000, 501000, 3301000))])


@pytest.fixture
def raster_dataset(base_path, monkeypatch):
    """Two years of `vct`/`veg_grid_cell_v001`, uniformly filled and 10 apart, so the
    asc difference is a uniform -10 raster.
    """
    write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda year: 10 if year == 2030 else 20)
    monkeypatch.setattr(config, "BASE_PATH", base_path)
    return base_path


class TestRouteKindRejection:
    """The `type="dataset"` vs `type="custom"` split, both directions -- generalized via
    `get_analysis_class`, not reinvented per route.
    """

    def test_dataset_analysis_rejected_on_custom_route(self, client):
        response = client.get("/v1/analyses/difference/data.csv")
        assert response.status_code == 400
        assert "Invalid analysis type" in response.json()["detail"]

    def test_custom_analysis_rejected_on_dataset_route(self, client):
        response = client.get(f"/v1/datasets/{V}/{T}/{G}/analyses/exit-criterion/data.csv")
        assert response.status_code == 400
        assert "Invalid analysis type" in response.json()["detail"]

    def test_unknown_name_404s_on_the_dataset_route(self, client):
        response = client.get(f"/v1/datasets/{V}/{T}/{G}/analyses/nope/data.csv")
        assert response.status_code == 404
        assert "Unknown analysis name" in response.json()["detail"]

    def test_unknown_name_404s_on_the_custom_route(self, client):
        response = client.get("/v1/analyses/nope/data.csv")
        assert response.status_code == 404
        assert "Unknown analysis name" in response.json()["detail"]


class TestDatasetAnalysisData:
    """The dataset-scoped `data.{format}` route, running the real `difference` analysis."""

    def test_returns_the_analysis_result(self, client, vector_dataset):
        response = client.get(f"{DATASET_URL}/data.csv", params=DIFFERENCE_QUERY)
        assert response.status_code == 200, response.text
        rows = [line.split(",") for line in response.text.strip().splitlines()]
        header, body = rows[0], rows[1:]
        assert "grid_cell" in header and "value" in header
        value_at = header.index("value")
        # Both years hold the same value, and asc negates the later one before the sum.
        assert {float(row[value_at]) for row in body} == {0.0}

    def test_the_analysiss_own_parameters_are_honored(self, client, base_path, monkeypatch):
        """`order` is read off the raw query string by `_analysis_extra_params`, not bound
        by `AnalysisQuery` -- so flipping it must really flip which year is negated.
        """
        write_annual_dataset(base_path, years=(2030, 2040), value_fn=lambda year, geo_id: float(year))
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        response = client.get(f"{DATASET_URL}/data.csv", params={**DIFFERENCE_QUERY, "order": "desc"})
        assert response.status_code == 200, response.text
        rows = [line.split(",") for line in response.text.strip().splitlines()]
        value_at = rows[0].index("value")
        # desc negates 2030: -2030 + 2040 == 10.
        assert {float(row[value_at]) for row in rows[1:]} == {10.0}

    def test_where_is_composed_into_the_analysiss_own_filter(self, client, base_path, monkeypatch):
        """A request `where` is not grafted on as an extra step -- `resolve_analysis` folds
        it into the leading `filter` the analysis builds. Dropping 2040 leaves 2030's value
        alone in the sum, un-negated.
        """
        write_annual_dataset(base_path, years=(2030, 2040), value_fn=lambda year, geo_id: float(year))
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        response = client.get(
            f"{DATASET_URL}/data.csv",
            params={**DIFFERENCE_QUERY, "where": "calendar_year = 2030"},
        )
        assert response.status_code == 200, response.text
        rows = [line.split(",") for line in response.text.strip().splitlines()]
        value_at = rows[0].index("value")
        assert {float(row[value_at]) for row in rows[1:]} == {2030.0}

    def test_compression_is_accepted_on_the_data_form(self, client, vector_dataset):
        """Parquet is the one tabular format compression actually changes the encoding of,
        so both real algorithms succeeding proves `AnalysisDataQuery.compression` reaches
        the responder the same way the explicit `/v1/analysis.*` routes' own `Compression`
        query model does.
        """
        for algorithm in ("zstd", "gzip"):
            response = client.get(
                f"{DATASET_URL}/data.parquet",
                params={**DIFFERENCE_QUERY, "compression_algorithm": algorithm},
            )
            assert response.status_code == 200, response.text
            assert response.content[:4] == b"PAR1"

    def test_an_incompatible_compression_algorithm_400s(self, client, vector_dataset):
        response = client.get(
            f"{DATASET_URL}/data.csv",
            params={**DIFFERENCE_QUERY, "compression_algorithm": "lzw"},
        )
        assert response.status_code == 400
        assert "must be one of" in response.json()["detail"]

    def test_a_selection_the_analysis_rejects_400s(self, client, vector_dataset):
        """`difference`'s own `validate_selection` cardinality rule ("exactly 2 selected
        values of `dimension`") is what rejects this, inside `resolve_analysis` -- the one
        rejection path `get_analysis_class` cannot pre-empt, since both the name and the
        `Meta.type` are fine.
        """
        response = client.get(
            f"{DATASET_URL}/data.csv", params={**DIFFERENCE_QUERY, "calendar_year": [2030]}
        )
        assert response.status_code == 400
        assert "exactly 2" in response.json()["detail"]


class TestDatasetAnalysisGridCell:
    """`grid_cell` on a dataset-scoped named analysis, both spellings.

    The route resolves it against the dataset's own grid exactly as the flat data routes
    do (`api.data_query.resolve_grid_cell`) and passes the resulting conjunct as
    `resolve_analysis(grid_cell_where=...)`; only a vector grid's literal-digit-id form is
    left in `filters` for the registry to lower itself. Both paths have to end up
    narrowing the SAME analysis, which is what these check.
    """

    def test_literal_ids_narrow_the_result(self, client, vector_dataset):
        response = client.get(f"{DATASET_URL}/data.csv", params={**DIFFERENCE_QUERY, "grid_cell": "3,5"})
        assert response.status_code == 200, response.text
        rows = [line.split(",") for line in response.text.strip().splitlines()]
        cell_at = rows[0].index("grid_cell")
        assert {row[cell_at] for row in rows[1:]} == {"3", "5"}

    def test_a_coordinate_is_resolved_by_the_route_not_rejected_by_the_registry(
        self, client, vector_dataset, ecoregion_geometry
    ):
        """The registry itself cannot parse a coordinate (`grid_cell_conjunct` rejects one
        outright, by design), so a 200 here is only possible if the ROUTE resolved it to an
        id first -- and the id it resolves to must be the cell the point falls in.
        """
        response = client.get(
            f"{DATASET_URL}/data.csv",
            params={**DIFFERENCE_QUERY, "grid_cell": _lng_lat(500500, 3300500)},
        )
        assert response.status_code == 200, response.text
        rows = [line.split(",") for line in response.text.strip().splitlines()]
        cell_at = rows[0].index("grid_cell")
        assert {row[cell_at] for row in rows[1:]} == {"5"}

    def test_a_grid_cell_on_a_custom_route_is_rejected_rather_than_ignored(self, client):
        """A custom analysis names its own inputs -- two, on two different grids for
        `exit-criterion` -- so the route cannot resolve a coordinate against "the" grid,
        and `ExitCriterion.validate_selection` rejects the whole selection rather than
        letting it be silently dropped. The distinction matters: silently ignoring a
        `grid_cell` would return a whole-extent raster to a point query.
        """
        response = client.get(
            "/v1/analyses/exit-criterion/data.tif",
            params={**EXIT_CRITERION_QUERY, "grid_cell": "-91.5,29.5"},
        )
        assert response.status_code == 400
        assert "takes no selection" in response.json()["detail"]
        assert "grid_cell" in response.json()["detail"]


class TestDatasetAnalysisVectorTile:
    """`.../analyses/{name}/tiles/{z}/{x}/{y}.pbf` -- the named-analysis, dataset-scoped
    counterpart to `TestAnalysisVectorTileRoute` (tests/api/test_vector_tiles.py). Also the
    one route-table trap worth fetching a real tile to rule out: this route's `tiles/`
    segment and the sibling `data/{z}/{x}/{y}.{format}` route's `data/` segment are
    distinct literal path prefixes, so there is no `.pbf`-vs-`.{format}` collision to
    design around here (unlike `/v1/analysis/{z}/{x}/{y}.pbf` vs
    `/v1/analysis/{z}/{x}/{y}.{format}`, which share the identical path up to the final
    segment).
    """

    def test_serves_a_real_vector_tile(self, client, base_path, tiles_path, monkeypatch):
        write_annual_dataset(base_path, years=(2030, 2040), value_fn=lambda year, geo_id: float(year))
        write_pmtiles(tiles_path, G, TILE, [1, 2, 3])
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        response = client.get(f"{DATASET_URL}/tiles/{TILE.z}/{TILE.x}/{TILE.y}.pbf", params=DIFFERENCE_QUERY)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"
        result = VectorTile()
        result.ParseFromString(response.content)
        # asc negates 2040: 2030 - 2040 == -10, the same for every feature in the tile.
        assert result.layers[0].values[0].double_value == pytest.approx(-10.0, rel=1e-5)


class TestDatasetAnalysisRasterTile:
    """The `data/{z}/{x}/{y}.{format}` tile route, over a RASTER dataset -- the only grid
    kind that leaves the band-wide final frame `render_tile` needs. Also this file's
    vehicle for proving `color_map` is a REQUEST parameter that really changes rendered
    pixels, not merely one that round-trips through `pipeline.json`.
    """

    def test_serves_a_real_png(self, client, raster_dataset):
        tile = _tile_for_grid(raster_dataset)
        response = client.get(
            f"{RASTER_DATASET_URL}/data/{tile.z}/{tile.x}/{tile.y}.png", params=RASTER_DIFFERENCE_QUERY
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_different_color_maps_render_different_pixels(self, client, raster_dataset):
        tile = _tile_for_grid(raster_dataset)
        rendered = {}
        for color_map in ("default", "ffibs"):
            response = client.get(
                f"{RASTER_DATASET_URL}/data/{tile.z}/{tile.x}/{tile.y}.png",
                params={**RASTER_DIFFERENCE_QUERY, "color_map": color_map},
            )
            assert response.status_code == 200, response.text
            rendered[color_map] = response.content
        assert rendered["default"] != rendered["ffibs"]

    def test_a_tif_of_the_whole_raster_result(self, client, raster_dataset):
        response = client.get(f"{RASTER_DATASET_URL}/data.tif", params=RASTER_DIFFERENCE_QUERY)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
        # 10 (2030) - 20 (2040), uniform across the grid.
        assert (data == -10).all()


class TestDatasetAnalysisPipelineJson:
    """`.../analyses/{name}/pipeline.json` -- the expanded `Pipeline`, proving the
    named-analysis machinery really is pure sugar over a hand-authored one: it must parse
    back through the same `parse_pipeline` a POSTed body goes through, and it must reflect
    the request's own `sort`/`select`/`color_map` overrides, which are never the registry's
    concern.
    """

    def test_expands_to_a_parsable_pipeline_over_the_url_dataset(self, client, vector_dataset):
        response = client.get(f"{DATASET_URL}/pipeline.json", params=DIFFERENCE_QUERY)
        assert response.status_code == 200, response.text
        expanded = parse_pipeline(response.json())
        selection = next(iter(expanded.inputs.values()))
        assert (selection.variable, selection.time_unit, selection.grid) == (V, T, G)

    def test_the_request_filters_land_on_a_filter_step_not_the_input(self, client, vector_dataset):
        """An `InputSelection` is `extra="forbid"` over `variable`/`time_unit`/`grid` and
        carries no filtering at all, so a dimension filter that survived the round trip can
        only be on a step.
        """
        response = client.get(f"{DATASET_URL}/pipeline.json", params=DIFFERENCE_QUERY)
        body = response.json()
        assert set(next(iter(body["inputs"].values()))) == {"variable", "time_unit", "grid"}
        assert "calendar_year IN (2030, 2040)" in body["steps"][0]["where"]

    def test_sort_select_and_color_map_overrides_are_reflected(self, client, vector_dataset):
        response = client.get(
            f"{DATASET_URL}/pipeline.json",
            params={
                **DIFFERENCE_QUERY,
                "sort": "-grid_cell",
                "select": "grid_cell,value",
                "color_map": "default",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["output"]["sort"] == "-grid_cell"
        assert body["output"]["select"] == "grid_cell,value"
        assert body["output"]["color_map"] == "default"
        # And the result still parses with those overrides in place.
        parse_pipeline(body)

    def test_format_is_not_overridden_by_the_route(self, client, vector_dataset):
        """There is no `{format}`/`.pbf` URL segment on this route to source an override
        from, so the pipeline's own `output.format` (`difference` builds `csv`) is left
        untouched.
        """
        response = client.get(f"{DATASET_URL}/pipeline.json", params=DIFFERENCE_QUERY)
        assert response.json()["output"]["format"] == "csv"


class TestCustomAnalysisPipelineJson:
    """`exit-criterion`'s expansion -- the one registered `type="custom"` entry, and the
    only one with multiple named inputs and a `project` + spatial `join`.
    """

    def test_expands_to_the_paired_shape(self, client):
        response = client.get("/v1/analyses/exit-criterion/pipeline.json", params=EXIT_CRITERION_QUERY)
        assert response.status_code == 200, response.text
        expanded = parse_pipeline(response.json())
        assert len(expanded.inputs) == 2
        assert [step.op for step in expanded.steps] == [
            "filter",
            "filter",
            "project",
            "join",
            "filter",
            "group",
        ]

    def test_color_map_override_is_reflected(self, client):
        response = client.get(
            "/v1/analyses/exit-criterion/pipeline.json",
            params={**EXIT_CRITERION_QUERY, "color_map": "ffibs"},
        )
        assert response.json()["output"]["color_map"] == "ffibs"

    def test_the_analysiss_own_parameters_are_validated_by_the_registry(self, client):
        """`marsh_threshold` is bounded to [0, 1] by `ExitCriterion.Parameters`; the route
        does not know that and does not need to -- `resolve_analysis` raises
        `InvalidAnalysisParametersError`, which `api/__init__` maps to 400.
        """
        response = client.get(
            "/v1/analyses/exit-criterion/pipeline.json",
            params={**EXIT_CRITERION_QUERY, "marsh_threshold": 42},
        )
        assert response.status_code == 400
        assert "marsh_threshold" in response.json()["detail"]


class TestCustomAnalysisVectorTileRejection:
    """`exit-criterion`'s final frame is band-wide on a raster grid, so it can never serve
    a `.pbf` tile. Unaffected by the execution gap the class below pins: a `.pbf` request
    never reaches `execute()` at all -- `api.pipeline._require_vector_final_frame` reads
    the plan's `frame_types` and rejects before any data is read.
    """

    def test_rejects_a_raster_output_grid(self, client):
        response = client.get(
            "/v1/analyses/exit-criterion/tiles/3/1/2.pbf", params=EXIT_CRITERION_QUERY
        )
        assert response.status_code == 400
        assert "not a long frame on a vector grid" in response.json()["detail"]


class TestCustomAnalysisExecution:
    """The two custom routes that actually RUN the analysis.

    `exit_criterion_data` is built so marsh and salinity each cross their threshold only
    in 2040, making the `min of=calendar_year` answer a uniform 2040 raster -- a value
    the route can only produce if the whole band-wide chain (shared scan, crosswalk
    broadcast, operand-named filter, band collapse) survived the trip through HTTP.
    """

    def test_data_tif(self, client, exit_criterion_data):
        response = client.get("/v1/analyses/exit-criterion/data.tif", params=EXIT_CRITERION_QUERY)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
        assert (data == 2040).all()

    def test_png_tile(self, client, exit_criterion_data):
        tile = _tile_for_grid(exit_criterion_data)
        response = client.get(
            f"/v1/analyses/exit-criterion/data/{tile.z}/{tile.x}/{tile.y}.png",
            params=EXIT_CRITERION_QUERY,
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


class TestAnalysisRegistryErrorMapping:
    """`resolve_analysis` raising each of the three `AnalysisRegistryError` subclasses,
    monkeypatched directly so the mapping is checked independently of which real analysis
    happens to raise which -- `api/__init__.py`'s `analysis_registry_error_handler` is the
    single translation point, and the `404` case in particular is unreachable through the
    real registry (`get_analysis_class` catches an unknown name first).
    """

    def _patch(self, monkeypatch, exception):
        import cpra.mp.data.api as api_module

        def _raise(name, *, dataset=None, filters=None, params=None, grid_cell_where=None):
            raise exception

        monkeypatch.setattr(api_module, "resolve_analysis", _raise)

    def test_invalid_parameters_maps_to_400(self, client, monkeypatch):
        from cpra.mp.data.analyses.base import InvalidAnalysisParametersError

        self._patch(monkeypatch, InvalidAnalysisParametersError("dimension must select exactly 2 values"))
        response = client.get("/v1/analyses/exit-criterion/data.csv")
        assert response.status_code == 400
        assert "exactly 2 values" in response.json()["detail"]

    def test_unknown_analysis_error_maps_to_404(self, client, monkeypatch):
        from cpra.mp.data.analyses.base import UnknownAnalysisError

        self._patch(monkeypatch, UnknownAnalysisError("unknown analysis name"))
        response = client.get("/v1/analyses/exit-criterion/data.csv")
        assert response.status_code == 404

    def test_type_mismatch_error_maps_to_400(self, client, monkeypatch):
        from cpra.mp.data.analyses.base import AnalysisTypeMismatchError

        self._patch(monkeypatch, AnalysisTypeMismatchError("type mismatch"))
        response = client.get("/v1/analyses/exit-criterion/data.csv")
        assert response.status_code == 400


class TestAnalysesCatalog:
    def test_lists_every_registered_analysis(self, client):
        response = client.get("/v1/analyses.json")
        assert response.status_code == 200, response.text
        assert {entry["name"] for entry in response.json()} == {
            "difference",
            "concatenate",
            "exit-criterion",
        }
