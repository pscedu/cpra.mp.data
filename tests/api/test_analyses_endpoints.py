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

* `diff` (`type="dataset"`) carries the four dataset-scoped routes. It is exercised
  over BOTH grid kinds: `tests/factories.py`'s `ssf`/`ecoregion_v001` (vector) for the
  tabular and `.pbf` routes, and its `vct`/`veg_grid_cell_v001` (raster) for the `png`
  tile route, which needs a band-wide final frame. Substituting these for
  `Meta.variables`' real-world datasets follows that module's own documented convention
  (the engine cares about grid TYPES and SHAPES, not variable semantics).
* `sal_veg_investigation` (`type="custom"`) carries the custom routes. It is the one
  registered analysis that is band-wide end to end -- a vector-to-raster `project` feeding
  a band-wide `join`, then a `filter` naming that join's operand value columns -- so the
  two routes that execute it are also this module's only coverage of that shape reaching
  a client. What the shape itself does is pinned at the engine level
  (`tests/api/test_analyses_registry.py::TestSalinityVegInvestigationExecution`); what
  these pin is that the custom routes deliver its bytes.
* `lnd_conf` (`type="dataset"`) reuses the routes `diff` already
  carries, and is covered only where it DIFFERS from them: it declares no parameters of
  its own and supplies its own `scenario_id`, so its query string is dimension filters
  alone and a caller-supplied `scenario_id` must 400. It rides the raster stand-in
  (`vct`/`veg_grid_cell_v001`), since its real variable is raster-only and its own
  `pipeline()` sinks to `tif` to match.
* `lnd_type_plus` (`type="custom"`) likewise reuses the custom routes
  `sal_veg_investigation` carries, and is covered where it differs: all three of its
  parameters are real dimension NAMES, which makes it the sharpest test of the custom
  routes' dimension/parameter split, and it rejects any selection at all. It runs on the
  real `morph_pixel_v002`/`hydro_compartment_v002` pair, with the former shrunk to 2x4 by
  `conftest.py`'s `lnd_type_plus_data` -- the analysis names both grids itself, so there is
  no stand-in to substitute.

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
from cpra.mp.data.analyses.lnd_conf import SCENARIO_WEIGHTS
from cpra.mp.data.analysis.request.ops import validate_shapes
from cpra.mp.data.analysis.request.pipeline import parse_pipeline
from tests.factories import (
    ANNUAL_KWARGS,
    LND_TYPE_PLUS_SELECTION,
    RASTER_KWARGS,
    write_annual_dataset,
    write_grid_geometry,
    write_pmtiles,
    write_raster_dataset,
)

TILE = mercantile.Tile(x=1, y=2, z=3)

V, T, G = ANNUAL_KWARGS["variable"], ANNUAL_KWARGS["time_unit"], ANNUAL_KWARGS["grid"]
RV, RT, RG = RASTER_KWARGS["variable"], RASTER_KWARGS["time_unit"], RASTER_KWARGS["grid"]

DATASET_URL = f"/v1/datasets/{V}/{T}/{G}/analyses/diff"
RASTER_DATASET_URL = f"/v1/datasets/{RV}/{RT}/{RG}/analyses/diff"

# `diff` needs its own `dimension`/`order` parameters plus a two-valued filter on
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

# `sal_veg_investigation` is `type="custom"`: no dataset and no dimension filters,
# just its own two required parameters.
SALINITY_VEG_QUERY = {"model_group_id": 500, "scenario_id": 7}

# `lnd_conf` declares no parameters at all, and supplies `scenario_id` itself
# -- so its query is `vct`'s other required dimensions and nothing else. It rides the raster
# stand-in rather than the vector one: its real variable (`lnd_type`) is raster-only, and
# its own `pipeline()` sinks to `tif` accordingly.
LAND_TYPE_CONFIDENCE_URL = f"/v1/datasets/{RV}/{RT}/{RG}/analyses/lnd_conf"
LAND_TYPE_CONFIDENCE_QUERY = {
    "calendar_year": [2050],
    "model_group_id": [RASTER_KWARGS["model_group_id"]],
}

# `lnd_type_plus` is `type="custom"` too, and every one of its three parameters is a real
# dimension NAME -- which on the custom routes is exactly what must reach `Parameters`
# rather than being bound as a dataset filter. See `TestLndTypePlusRoute`.
LND_TYPE_PLUS_QUERY = {**LND_TYPE_PLUS_SELECTION, "calendar_year": 2050}


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
    so `diff`'s asc order (negate the later year, then sum) leaves exactly 0
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
        response = client.get("/v1/analyses/diff/data.csv")
        assert response.status_code == 400
        assert "Invalid analysis type" in response.json()["detail"]

    def test_custom_analysis_rejected_on_dataset_route(self, client):
        response = client.get(f"/v1/datasets/{V}/{T}/{G}/analyses/sal_veg_investigation/data.csv")
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
    """The dataset-scoped `data.{format}` route, running the real `diff` analysis."""

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
        """`diff`'s own `validate_selection` cardinality rule ("exactly 2 selected
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
        `sal_veg_investigation` -- so the route cannot resolve a coordinate against
        "the" grid, and `SalinityVegInvestigation.validate_selection` rejects the whole
        selection rather than letting it be silently dropped. The distinction matters:
        silently ignoring a `grid_cell` would return a whole-extent raster to a point
        query.
        """
        response = client.get(
            "/v1/analyses/sal_veg_investigation/data.tif",
            params={**SALINITY_VEG_QUERY, "grid_cell": "-91.5,29.5"},
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
        from, so the pipeline's own `output.format` -- `csv`, for `diff` over this
        vector dataset -- is left untouched.
        """
        response = client.get(f"{DATASET_URL}/pipeline.json", params=DIFFERENCE_QUERY)
        assert response.json()["output"]["format"] == "csv"

    def test_the_unoverridden_format_matches_the_datasets_grid_kind(self, client, raster_dataset):
        """The same route over a RASTER dataset. Since nothing overrides the format here,
        what `pipeline.json` publishes is what a caller resubmitting it would run -- so it
        has to match the shape the pipeline actually produces, which on a pixel grid is a
        raster surface rather than a table.
        """
        response = client.get(f"{RASTER_DATASET_URL}/pipeline.json", params=RASTER_DIFFERENCE_QUERY)
        assert response.status_code == 200, response.text
        assert response.json()["output"]["format"] == "tif"
        validate_shapes(parse_pipeline(response.json()))  # no raise


class TestCustomAnalysisPipelineJson:
    """`sal_veg_investigation`'s expansion -- the one registered `type="custom"`
    entry, and the only one with multiple named inputs and a `project` + spatial `join`.
    """

    def test_expands_to_the_paired_shape(self, client):
        response = client.get("/v1/analyses/sal_veg_investigation/pipeline.json", params=SALINITY_VEG_QUERY)
        assert response.status_code == 200, response.text
        expanded = parse_pipeline(response.json())
        assert len(expanded.inputs) == 2
        assert [step.op for step in expanded.steps] == [
            "filter",
            "filter",
            "project",
            "join",
            "case",
            "group",
        ]

    def test_color_map_override_is_reflected(self, client):
        response = client.get(
            "/v1/analyses/sal_veg_investigation/pipeline.json",
            params={**SALINITY_VEG_QUERY, "color_map": "ffibs"},
        )
        assert response.json()["output"]["color_map"] == "ffibs"

    def test_the_analysiss_own_parameters_are_validated_by_the_registry(self, client):
        """`marsh_threshold` is bounded to [0, 1] by `SalinityVegInvestigation.Parameters`; the route
        does not know that and does not need to -- `resolve_analysis` raises
        `InvalidAnalysisParametersError`, which `api/__init__` maps to 400.
        """
        response = client.get(
            "/v1/analyses/sal_veg_investigation/pipeline.json",
            params={**SALINITY_VEG_QUERY, "marsh_threshold": 42},
        )
        assert response.status_code == 400
        assert "marsh_threshold" in response.json()["detail"]


class TestCustomAnalysisVectorTileRejection:
    """`sal_veg_investigation`'s final frame is band-wide on a raster grid, so it can
    never serve a `.pbf` tile. Unaffected by the execution gap the class below pins: a
    `.pbf` request never reaches `execute()` at all --
    `api.pipeline._require_vector_final_frame` reads the plan's `frame_types` and rejects
    before any data is read.
    """

    def test_rejects_a_raster_output_grid(self, client):
        response = client.get(
            "/v1/analyses/sal_veg_investigation/tiles/3/1/2.pbf", params=SALINITY_VEG_QUERY
        )
        assert response.status_code == 400
        assert "not a long frame on a vector grid" in response.json()["detail"]


class TestCustomAnalysisExecution:
    """The two custom routes that actually RUN the analysis.

    `salinity_veg_data` is built so marsh and salinity each cross their threshold only
    in 2040, making the `min of=calendar_year` answer a uniform 2040 raster -- a value
    the route can only produce if the whole band-wide chain (shared scan, crosswalk
    broadcast, operand-named filter, band collapse) survived the trip through HTTP.
    """

    def test_data_tif(self, client, salinity_veg_data):
        response = client.get("/v1/analyses/sal_veg_investigation/data.tif", params=SALINITY_VEG_QUERY)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
        assert (data == 2040).all()

    def test_png_tile(self, client, salinity_veg_data):
        tile = _tile_for_grid(salinity_veg_data)
        response = client.get(
            f"/v1/analyses/sal_veg_investigation/data/{tile.z}/{tile.x}/{tile.y}.png",
            params=SALINITY_VEG_QUERY,
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


class TestAnalysisRegistryErrorMapping:
    """`resolve_analysis_chain` raising each of the three `AnalysisRegistryError`
    subclasses, monkeypatched directly so the mapping is checked independently of which
    real analysis happens to raise which -- `api/__init__.py`'s
    `analysis_registry_error_handler` is the single translation point, and the `404` case
    in particular is unreachable through the real registry (`get_analysis_class` catches
    an unknown name first).
    """

    def _patch(self, monkeypatch, exception):
        import cpra.mp.data.api as api_module

        def _raise(names, *, dataset=None, filters=None, params=None, grid_cell_where=None):
            raise exception

        monkeypatch.setattr(api_module, "resolve_analysis_chain", _raise)

    def test_invalid_parameters_maps_to_400(self, client, monkeypatch):
        from cpra.mp.data.analyses.base import InvalidAnalysisParametersError

        self._patch(monkeypatch, InvalidAnalysisParametersError("dimension must select exactly 2 values"))
        response = client.get("/v1/analyses/sal_veg_investigation/data.csv")
        assert response.status_code == 400
        assert "exactly 2 values" in response.json()["detail"]

    def test_unknown_analysis_error_maps_to_404(self, client, monkeypatch):
        from cpra.mp.data.analyses.base import UnknownAnalysisError

        self._patch(monkeypatch, UnknownAnalysisError("unknown analysis name"))
        response = client.get("/v1/analyses/sal_veg_investigation/data.csv")
        assert response.status_code == 404

    def test_type_mismatch_error_maps_to_400(self, client, monkeypatch):
        from cpra.mp.data.analyses.base import AnalysisTypeMismatchError

        self._patch(monkeypatch, AnalysisTypeMismatchError("type mismatch"))
        response = client.get("/v1/analyses/sal_veg_investigation/data.csv")
        assert response.status_code == 400


class TestLandConfidenceRoute:
    """`lnd_conf` on the dataset-scoped routes. The route machinery itself is
    already carried by `diff` above; what these pin is the one thing that differs --
    it declares NO parameters of its own and supplies its own `scenario_id`, so its query
    string is dimension filters alone and a caller-supplied `scenario_id` has to come back
    as a 400 rather than silently narrowing the ensemble. What the pipeline computes is
    pinned at the engine level (`test_analyses_registry.py::TestLandConfidenceExecution`).
    """

    @pytest.fixture
    def ensemble_dataset(self, base_path, monkeypatch):
        """The five ensemble members, every one of them holding a land code, so the
        weighted sum is 1.0 at every pixel.
        """
        for scenario_id in SCENARIO_WEIGHTS:
            write_raster_dataset(base_path, years=(2050,), scenario_id=scenario_id, fill_fn=lambda year: 1)
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        return base_path

    def test_returns_the_confidence_surface(self, client, ensemble_dataset):
        response = client.get(f"{LAND_TYPE_CONFIDENCE_URL}/data.tif", params=LAND_TYPE_CONFIDENCE_QUERY)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            band = src.read(1, masked=True)
        assert band.count()
        assert band.compressed() == pytest.approx(1.0)

    def test_a_caller_supplied_scenario_id_400s(self, client, ensemble_dataset):
        response = client.get(
            f"{LAND_TYPE_CONFIDENCE_URL}/data.tif",
            params={**LAND_TYPE_CONFIDENCE_QUERY, "scenario_id": [34]},
        )
        assert response.status_code == 400
        assert "fixed ensemble" in response.json()["detail"]

    def test_pipeline_json_shows_the_ensemble_the_url_never_named(self, client, ensemble_dataset):
        response = client.get(f"{LAND_TYPE_CONFIDENCE_URL}/pipeline.json", params=LAND_TYPE_CONFIDENCE_QUERY)
        assert response.status_code == 200, response.text
        expanded = parse_pipeline(response.json())
        assert expanded.steps[0].where == (
            "calendar_year = 2050 AND model_group_id = 500 AND scenario_id IN (34, 35, 36, 37, 38)"
        )
        assert expanded.steps[-1].by == ["model_group_id", "calendar_year"]

    def test_pipeline_json_publishes_a_resubmittable_pipeline(self, client, ensemble_dataset):
        """`pipeline.json` is the one route that does NOT override `output.format` from a
        URL segment, so whatever `pipeline()` set is what a caller gets back -- and a caller
        resubmitting it verbatim has to get a pipeline the request layer accepts. A tabular
        format here would not: this analysis's final frame is band-wide.
        """
        response = client.get(f"{LAND_TYPE_CONFIDENCE_URL}/pipeline.json", params=LAND_TYPE_CONFIDENCE_QUERY)
        assert response.json()["output"]["format"] == "tif"
        validate_shapes(parse_pipeline(response.json()))  # no raise


class TestLndTypePlusRoute:
    """`lnd_type_plus` on the custom routes. The custom-route machinery itself is already
    carried by `sal_veg_investigation` above; what these pin is the one thing that
    differs -- ALL THREE of its parameters are real dimension names, so it is the sharpest
    test of `_resolve_custom_analysis`'s dimension/parameter split. On these routes the
    `FilterDimensions` half of the bound query model is subtracted from the known set
    precisely so `model_group_id`/`scenario_id`/`calendar_year` reach the analysis's own
    `Parameters`; were any of them bound as a filter instead, this analysis would reject
    the request outright (it takes no selection at all).

    What the pipeline itself computes is pinned at the engine level
    (`test_analyses_registry.py::TestLndTypePlusExecution`); these pin that the route
    delivers it.
    """

    def test_data_tif(self, client, lnd_type_plus_data):
        response = client.get("/v1/analyses/lnd_type_plus/data.tif", params=LND_TYPE_PLUS_QUERY)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
        # The fixture's first four pixels are one per reclassification rule, by COMPUTED
        # depth, plus the water restriction: 0.20 m over water -> 6, 0.70 m over water -> 7,
        # 1.50 m over water -> 2 (the deep class reports the open-water code itself), and
        # 0.20 m over land type 1 -> 1 (no arm matches off water, so the `else` hands back
        # the land type unchanged). The rest repeat the shallow case.
        assert data.ravel().tolist() == [6, 7, 2, 1, 6, 6, 6, 6]

    def test_png_tile(self, client, lnd_type_plus_data):
        tile = _tile_for_grid(lnd_type_plus_data)
        response = client.get(
            f"/v1/analyses/lnd_type_plus/data/{tile.z}/{tile.x}/{tile.y}.png",
            params=LND_TYPE_PLUS_QUERY,
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_pipeline_json_publishes_a_resubmittable_pipeline(self, client, lnd_type_plus_data):
        """The one `project` lands ON the pixel grid and nothing projects off it, so the
        published `output.format` has to be the raster one -- a caller resubmitting this
        verbatim gets a pipeline the request layer accepts rather than a tabular-sink
        rejection.
        """
        response = client.get("/v1/analyses/lnd_type_plus/pipeline.json", params=LND_TYPE_PLUS_QUERY)
        assert response.status_code == 200, response.text
        expanded = parse_pipeline(response.json())
        assert len(expanded.inputs) == 3
        assert expanded.steps[-1].op == "case"
        assert expanded.steps[-1].else_ == "land_type"
        assert response.json()["output"]["format"] == "tif"
        validate_shapes(expanded)  # no raise

    def test_calendar_year_is_optional(self, client, lnd_type_plus_data):
        """The one optional parameter. Omitted, every year on disk is kept as its own band --
        no `filter` anywhere names `calendar_year`, and nothing downstream reduces across it
        either. There is no `group` in this pipeline at all; the one collapse is the `case`'s
        own pairing, which keys on the dimensions all three operands SHARE
        (`execution.band_ops._pair_band_groups`) and so can only ever pair a year with itself.
        That is what keeps each year reclassified against its OWN counterpart bands.
        """
        params = {k: v for k, v in LND_TYPE_PLUS_QUERY.items() if k != "calendar_year"}
        response = client.get("/v1/analyses/lnd_type_plus/pipeline.json", params=params)
        assert response.status_code == 200, response.text
        expanded = parse_pipeline(response.json())
        assert not any("calendar_year" in (step.where or "") for step in expanded.steps if step.op == "filter")
        assert not any(step.op == "group" for step in expanded.steps)

    def test_a_missing_required_parameter_is_a_400(self, client, lnd_type_plus_data):
        """`model_group_id` is a dimension name, so a route that bound it as a filter would
        see it supplied and the analysis would see it missing -- `Parameters` validation is
        what catches that, and `resolve_analysis` raises the 400.
        """
        params = {k: v for k, v in LND_TYPE_PLUS_QUERY.items() if k != "model_group_id"}
        response = client.get("/v1/analyses/lnd_type_plus/pipeline.json", params=params)
        assert response.status_code == 400
        assert "model_group_id" in response.json()["detail"]

    def test_a_grid_cell_selection_is_rejected(self, client, lnd_type_plus_data):
        """`grid_cell` is one of the two reserved keys that DO reach a custom analysis as a
        filter -- and this analysis rejects any selection, both because its inputs span two
        grids (so a cell ID is ambiguous) and because a spatial predicate would turn its
        band-wide frame long and invalidate the raster output rather than narrow it.
        """
        response = client.get(
            "/v1/analyses/lnd_type_plus/pipeline.json",
            params={**LND_TYPE_PLUS_QUERY, "grid_cell": "1,2"},
        )
        assert response.status_code == 400
        assert "takes no selection" in response.json()["detail"]


class TestAnalysesCatalog:
    def test_lists_every_registered_analysis(self, client):
        response = client.get("/v1/analyses.json")
        assert response.status_code == 200, response.text
        assert {entry["name"] for entry in response.json()} == {
            "diff",
            "lnd_chng",
            "sal_veg_investigation",
            "lnd_conf",
            "lnd_type_plus",
        }
