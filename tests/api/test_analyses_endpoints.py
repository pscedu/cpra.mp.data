"""Tests for a REGISTERED pipeline (`cpra.mp.data.analyses`'s `ANALYSES`/
`resolve_analysis`) reached over HTTP -- `analysis=` on `/v1/data.{format}` and
`/v1/data/{z}/{x}/{y}.{format}`, plus `/v1/analyses/{name}.json` for the resolved pipeline
itself.

Eight routes before, differing only in whether the name sat in the path and whether a
`{variable}/{time_unit}/{grid}` prefix came with it. Now it is one query parameter on the
two data paths, and whether the dataset triple accompanies it is the caller's statement
about what the analysis reads -- checked by the analysis's own `pipeline()`
(`require_dataset`/`reject_dataset`), not by the URL's shape.

These drive the REAL registry end to end -- no fake `resolve_analysis`. The claim under
test is that a named analysis is pure sugar: the decoder builds `resolve_analysis`'s
inputs, applies the request's own `Output` overrides, and hands the result to the same
responder a hand-authored pipeline reaches. A fake registry can only prove the first and
third of those, and it cannot notice the one thing most likely to break -- the registry and
the decoder disagreeing about what `filters`/`grid_cell`/`params` mean.

**Which registered analysis stands in for what.**

* `diff` reads the caller's dataset, and carries the dataset-named cases. It is exercised
  over BOTH grid kinds: `tests/factories.py`'s `ssf`/`ecoregion_v001` (vector) for the
  tabular and `.pbf` cases, and its `vct`/`veg_grid_cell_v001` (raster) for the `png`
  tile case, which needs a band-wide final frame. Substituting these for
  `Meta.variables`' real-world datasets follows that module's own documented convention
  (the engine cares about grid TYPES and SHAPES, not variable semantics).
* `sal_veg_investigation` fixes its own inputs, and carries the no-dataset cases. It is
  the one registered analysis that is band-wide end to end -- a vector-to-raster `project`
  feeding a band-wide `join`, then a `filter` naming that join's operand value columns --
  so the two requests that execute it are also this module's only coverage of that shape
  reaching a client. What the shape itself does is pinned at the engine level
  (`tests/api/test_analyses_registry.py::TestSalinityVegInvestigationExecution`); what
  these pin is that the route delivers its bytes.
* `lnd_conf` also reads the caller's dataset, and is covered only where it DIFFERS from
  `diff`: it declares no parameters of its own and supplies its own `scenario_id`, so its
  query string is dimension filters alone and a caller-supplied `scenario_id` must 400. It
  rides the raster stand-in (`vct`/`veg_grid_cell_v001`), since its real variable is
  raster-only and its own `pipeline()` sinks to `tif` to match.
* `lnd_type_plus` fixes its own inputs too, and is covered where it differs from
  `sal_veg_investigation`: its whole selection is dimension names, which makes it the
  sharpest test of the rule that a dimension value is a FILTER on the analysis's output
  frame and never an analysis parameter. It runs on the real
  `morph_pixel_v002`/`hydro_compartment_v002` pair, with the former shrunk to 2x4 by
  `conftest.py`'s `lnd_type_plus_data` -- the analysis names both grids itself, so there is
  no stand-in to substitute.

`grid_cell` is the one request parameter the registry deliberately cannot handle alone:
resolving a coordinate against real geometry needs `io/`, which nothing under `request/`
(or in `analyses/`) may import. `api.unified.decode_request` therefore resolves it first
and passes the resulting conjunct as `resolve_analysis_chain(grid_cell_where=...)`;
`TestDatasetAnalysisGridCell` covers both spellings, and the no-dataset case where there is
no single grid to resolve against.
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

# The dataset triple, as the three query parameters that used to be three path segments.
DATASET = {"variable": V, "time_unit": T, "grid": G}
RASTER_DATASET = {"variable": RV, "time_unit": RT, "grid": RG}

# There is one data path now, and one tile path; what used to distinguish eight routes is
# entirely in the query string below.
DATA_URL = "/v1/data"


def pipeline_url(analysis):
    """`/v1/analyses/{name}.json` -- the one route that still names the analysis in its
    PATH, because the name is what is being asked ABOUT rather than a parameter of the
    request. Replaces `.../analyses/{name}/pipeline.json` in both its dataset-scoped and
    its custom spelling.
    """
    return f"/v1/analyses/{analysis}.json"


# `diff` needs its own `dimension`/`order` parameters plus a two-valued filter on
# that dimension; the remaining three are `ssf`'s other required dimensions, which
# `required_dataset_dimensions` puts in the `group` step's `by` list. `analysis=` and the
# dataset triple ride along in the same dict -- on one route they are all just parameters.
DIFFERENCE_QUERY = {
    "analysis": "diff",
    **DATASET,
    "dimension": "calendar_year",
    "order": "asc",
    "calendar_year": [2030, 2040],
    "model_group_id": [ANNUAL_KWARGS["model_group_id"]],
    "scenario_id": [ANNUAL_KWARGS["scenario_id"]],
    "percentile": [ANNUAL_KWARGS["percentile"]],
}

# `vct` declares no `percentile`, so the raster form drops it.
RASTER_DIFFERENCE_QUERY = {
    "analysis": "diff",
    **RASTER_DATASET,
    "dimension": "calendar_year",
    "order": "asc",
    "calendar_year": [2030, 2040],
    "model_group_id": [RASTER_KWARGS["model_group_id"]],
    "scenario_id": [RASTER_KWARGS["scenario_id"]],
}

# `sal_veg_investigation` fixes its own two inputs, so it names no dataset -- and its
# selection is dimension FILTERS against the frame those inputs produce, not parameters.
SALINITY_VEG_QUERY = {"analysis": "sal_veg_investigation", "model_group_id": 500, "scenario_id": 7}

# `lnd_conf` declares no parameters at all, and supplies `scenario_id` itself
# -- so its query is `vct`'s other required dimensions and nothing else. It rides the raster
# stand-in rather than the vector one: its real variable (`lnd_type`) is raster-only, and
# its own `pipeline()` sinks to `tif` accordingly.
LAND_TYPE_CONFIDENCE_QUERY = {
    "analysis": "lnd_conf",
    **RASTER_DATASET,
    "calendar_year": [2050],
    "model_group_id": [RASTER_KWARGS["model_group_id"]],
}

# `lnd_type_plus` fixes its own inputs too, and its whole selection is dimension names --
# which is exactly what must reach the trailing selection `filter` rather than the
# analysis's (empty) `Parameters`. See `TestLndTypePlusRoute`.
LND_TYPE_PLUS_QUERY = {
    "analysis": "lnd_type_plus",
    **LND_TYPE_PLUS_SELECTION,
    "calendar_year": 2050,
}


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


class TestDatasetMismatchRejection:
    """Whether an analysis takes the caller's dataset, both directions.

    This used to be the route table's job: two families of URLs, a `Meta.type` on each
    analysis, and `get_analysis_class` refusing the pairings that didn't match. With one
    data route there is no pairing left to refuse -- the question is asked of the analysis
    itself, inside its own `pipeline()`, by `require_dataset`/`reject_dataset`, and both
    answers arrive as `AnalysisDatasetMismatchError` -> 400. The error names the parameters
    to add or drop, which the old "Invalid analysis type" could not.
    """

    def test_an_analysis_that_needs_a_dataset_says_so_when_given_none(self, client):
        response = client.get(
            "/v1/data.csv",
            params={
                "analysis": "diff",
                "dimension": "calendar_year",
                "order": "asc",
                "calendar_year": [2030, 2040],
            },
        )
        assert response.status_code == 400
        assert "reads the dataset you choose" in response.json()["detail"]

    def test_an_analysis_with_fixed_inputs_says_so_when_given_a_dataset(self, client):
        response = client.get(
            "/v1/data.csv", params={"analysis": "sal_veg_investigation", **DATASET}
        )
        assert response.status_code == 400
        assert "takes no dataset" in response.json()["detail"]

    def test_an_unknown_name_is_a_404(self, client):
        response = client.get("/v1/data.csv", params={"analysis": "nope", **DATASET})
        assert response.status_code == 404
        assert "unknown analysis name" in response.json()["detail"]

    def test_an_unknown_name_is_a_404_on_the_pipeline_route_too(self, client):
        response = client.get(pipeline_url("nope"))
        assert response.status_code == 404
        assert "unknown analysis name" in response.json()["detail"]


class TestDatasetAnalysisData:
    """The dataset-scoped `data.{format}` route, running the real `diff` analysis."""

    def test_returns_the_analysis_result(self, client, vector_dataset):
        response = client.get("/v1/data.csv", params=DIFFERENCE_QUERY)
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

        response = client.get("/v1/data.csv", params={**DIFFERENCE_QUERY, "order": "desc"})
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
            "/v1/data.csv",
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
                "/v1/data.parquet",
                params={**DIFFERENCE_QUERY, "compression_algorithm": algorithm},
            )
            assert response.status_code == 200, response.text
            assert response.content[:4] == b"PAR1"

    def test_an_incompatible_compression_algorithm_400s(self, client, vector_dataset):
        response = client.get(
            "/v1/data.csv",
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
            "/v1/data.csv", params={**DIFFERENCE_QUERY, "calendar_year": [2030]}
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
        response = client.get("/v1/data.csv", params={**DIFFERENCE_QUERY, "grid_cell": "3,5"})
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
            "/v1/data.csv",
            params={**DIFFERENCE_QUERY, "grid_cell": _lng_lat(500500, 3300500)},
        )
        assert response.status_code == 200, response.text
        rows = [line.split(",") for line in response.text.strip().splitlines()]
        cell_at = rows[0].index("grid_cell")
        assert {row[cell_at] for row in rows[1:]} == {"5"}

    def test_a_grid_cell_under_a_raster_format_is_rejected_rather_than_ignored(self, client):
        """`sal_veg_investigation` produces a raster surface, and a `grid_cell` SAMPLES a
        raster at a set of points -- which turns the result into rows and makes `tif`
        unwritable. That rejection is `validate_shapes`'s, stated against the shape rather
        than against the analysis: what used to be `SalinityVegInvestigation.
        validate_selection` refusing every selection outright is now a rule about the sink,
        so the same `grid_cell` is perfectly legal under `csv` (the test below).

        The distinction still matters for the same reason it did: silently ignoring a
        `grid_cell` would return a whole-extent raster to a point query.
        """
        response = client.get(
            "/v1/data.tif",
            params={**SALINITY_VEG_QUERY, "grid_cell": "-91.5,29.5"},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "grid_cell" in detail
        assert "ask for a tabular format" in detail

    def test_the_same_grid_cell_is_accepted_under_a_tabular_format(self, client):
        """The other side of the rule above, and the behaviour change B9 bought: a
        fixed-input analysis now takes a spatial selection like any other request, provided
        the sink can hold the rows it produces. No data is written for this fixture, so what
        is asserted is that the request RESOLVES -- it gets past the registry and the shape
        check to a read -- not what it reads.
        """
        response = client.get(
            pipeline_url("sal_veg_investigation"),
            params={**SALINITY_VEG_QUERY, "grid_cell": "-91.5,29.5", "output.format": "csv"},
        )
        assert response.status_code == 200, response.text
        assert any(
            "grid_cell" in (step.get("where") or "") for step in response.json()["steps"]
        )


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

        response = client.get(f"{DATA_URL}/{TILE.z}/{TILE.x}/{TILE.y}.pbf", params=DIFFERENCE_QUERY)
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
            f"{DATA_URL}/{tile.z}/{tile.x}/{tile.y}.png", params=RASTER_DIFFERENCE_QUERY
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_different_color_maps_render_different_pixels(self, client, raster_dataset):
        tile = _tile_for_grid(raster_dataset)
        rendered = {}
        for color_map in ("default", "ffibs"):
            response = client.get(
                f"{DATA_URL}/{tile.z}/{tile.x}/{tile.y}.png",
                params={**RASTER_DIFFERENCE_QUERY, "color_map": color_map},
            )
            assert response.status_code == 200, response.text
            rendered[color_map] = response.content
        assert rendered["default"] != rendered["ffibs"]

    def test_a_tif_of_the_whole_raster_result(self, client, raster_dataset):
        response = client.get("/v1/data.tif", params=RASTER_DIFFERENCE_QUERY)
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
        response = client.get(pipeline_url("diff"), params=DIFFERENCE_QUERY)
        assert response.status_code == 200, response.text
        expanded = parse_pipeline(response.json())
        selection = next(iter(expanded.inputs.values()))
        assert (selection.variable, selection.time_unit, selection.grid) == (V, T, G)

    def test_the_request_filters_land_on_a_filter_step_not_the_input(self, client, vector_dataset):
        """An `InputSelection` is `extra="forbid"` over `variable`/`time_unit`/`grid` and
        carries no filtering at all, so a dimension filter that survived the round trip can
        only be on a step.
        """
        response = client.get(pipeline_url("diff"), params=DIFFERENCE_QUERY)
        body = response.json()
        assert set(next(iter(body["inputs"].values()))) == {"variable", "time_unit", "grid"}
        assert "calendar_year IN (2030, 2040)" in body["steps"][0]["where"]

    def test_sort_select_and_color_map_overrides_are_reflected(self, client, vector_dataset):
        response = client.get(
            pipeline_url("diff"),
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
        response = client.get(pipeline_url("diff"), params=DIFFERENCE_QUERY)
        assert response.json()["output"]["format"] == "csv"

    def test_the_unoverridden_format_matches_the_datasets_grid_kind(self, client, raster_dataset):
        """The same route over a RASTER dataset. Since nothing overrides the format here,
        what `pipeline.json` publishes is what a caller resubmitting it would run -- so it
        has to match the shape the pipeline actually produces, which on a pixel grid is a
        raster surface rather than a table.
        """
        response = client.get(pipeline_url("diff"), params=RASTER_DIFFERENCE_QUERY)
        assert response.status_code == 200, response.text
        assert response.json()["output"]["format"] == "tif"
        validate_shapes(parse_pipeline(response.json()))  # no raise


class TestCustomAnalysisPipelineJson:
    """`sal_veg_investigation`'s expansion -- the one registered `type="custom"`
    entry, and the only one with multiple named inputs and a `project` + spatial `join`.
    """

    def test_expands_to_the_paired_shape(self, client):
        response = client.get(pipeline_url("sal_veg_investigation"), params=SALINITY_VEG_QUERY)
        assert response.status_code == 200, response.text
        expanded = parse_pipeline(response.json())
        assert len(expanded.inputs) == 2
        # The trailing `filter` at index 4 is the caller's own selection -- `analysis=`
        # no longer implies "no selection", so `model_group_id`/`scenario_id` land on the
        # joined frame as an ordinary conjunct rather than as `Parameters`.
        assert [step.op for step in expanded.steps] == [
            "filter",
            "filter",
            "project",
            "join",
            "filter",
            "case",
            "group",
        ]

    def test_color_map_override_is_reflected(self, client):
        response = client.get(
            pipeline_url("sal_veg_investigation"),
            params={**SALINITY_VEG_QUERY, "color_map": "ffibs"},
        )
        assert response.json()["output"]["color_map"] == "ffibs"

    def test_the_analysiss_own_parameters_are_validated_by_the_registry(self, client):
        """`marsh_threshold` is bounded to [0, 1] by `SalinityVegInvestigation.Parameters`; the route
        does not know that and does not need to -- `resolve_analysis` raises
        `InvalidAnalysisParametersError`, which `api/__init__` maps to 400.
        """
        response = client.get(
            pipeline_url("sal_veg_investigation"),
            params={**SALINITY_VEG_QUERY, "marsh_threshold": 42},
        )
        assert response.status_code == 400
        assert "marsh_threshold" in response.json()["detail"]


class TestCustomAnalysisVectorTileRejection:
    """`sal_veg_investigation`'s final frame is band-wide on a raster grid, so it can
    never serve a `.pbf` tile. Unaffected by the execution gap the class below pins: a
    `.pbf` request never reaches `execute()` at all -- the route applies `.pbf` as the
    pipeline's `output.format` and `request.ops.validate_shapes`'s `pbf` rule rejects it
    from the declared ops alone, before any data is read or any plan is compiled.
    """

    def test_rejects_a_raster_output_grid(self, client):
        response = client.get(
            "/v1/data/3/1/2.pbf", params=SALINITY_VEG_QUERY
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
        response = client.get("/v1/data.tif", params=SALINITY_VEG_QUERY)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
        assert (data == 2040).all()

    def test_png_tile(self, client, salinity_veg_data):
        tile = _tile_for_grid(salinity_veg_data)
        response = client.get(
            f"{DATA_URL}/{tile.z}/{tile.x}/{tile.y}.png",
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
        # Patched on `api.unified`, the module that imported it: the resolution moved out
        # of the route bodies and into the one decoder every route shares, so that is the
        # name the running code actually looks up.
        import cpra.mp.data.api.unified as unified_module

        def _raise(names, *, dataset=None, filters=None, params=None, grid_cell_where=None):
            raise exception

        monkeypatch.setattr(unified_module, "resolve_analysis_chain", _raise)

    def test_invalid_parameters_maps_to_400(self, client, monkeypatch):
        from cpra.mp.data.analyses.base import InvalidAnalysisParametersError

        self._patch(monkeypatch, InvalidAnalysisParametersError("dimension must select exactly 2 values"))
        response = client.get("/v1/data.csv", params=SALINITY_VEG_QUERY)
        assert response.status_code == 400
        assert "exactly 2 values" in response.json()["detail"]

    def test_unknown_analysis_error_maps_to_404(self, client, monkeypatch):
        from cpra.mp.data.analyses.base import UnknownAnalysisError

        self._patch(monkeypatch, UnknownAnalysisError("unknown analysis name"))
        response = client.get("/v1/data.csv", params=SALINITY_VEG_QUERY)
        assert response.status_code == 404

    def test_dataset_mismatch_error_maps_to_400(self, client, monkeypatch):
        from cpra.mp.data.analyses.base import AnalysisDatasetMismatchError

        self._patch(monkeypatch, AnalysisDatasetMismatchError("takes no dataset"))
        response = client.get("/v1/data.csv", params=SALINITY_VEG_QUERY)
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
        response = client.get("/v1/data.tif", params=LAND_TYPE_CONFIDENCE_QUERY)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            band = src.read(1, masked=True)
        assert band.count()
        assert band.compressed() == pytest.approx(1.0)

    def test_a_caller_supplied_scenario_id_400s(self, client, ensemble_dataset):
        response = client.get(
            "/v1/data.tif",
            params={**LAND_TYPE_CONFIDENCE_QUERY, "scenario_id": [34]},
        )
        assert response.status_code == 400
        assert "fixed ensemble" in response.json()["detail"]

    def test_pipeline_json_shows_the_ensemble_the_url_never_named(self, client, ensemble_dataset):
        response = client.get(pipeline_url("lnd_conf"), params=LAND_TYPE_CONFIDENCE_QUERY)
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
        response = client.get(pipeline_url("lnd_conf"), params=LAND_TYPE_CONFIDENCE_QUERY)
        assert response.json()["output"]["format"] == "tif"
        validate_shapes(parse_pipeline(response.json()))  # no raise


class TestLandChangeChainRoute:
    """A CHAINED analysis whose leading stage is a paired band-wide reduce -- the portal's
    land-change difference map, and the shape that turns on what a paired reduce's metadata
    keeps.

    `lnd_chng`'s baseline branch groups `calendar_year` away so a single baseline band can
    broadcast across the comparison, which makes the comparison year ONE-SIDED: the
    comparison operand carries it, the baseline operand has no such column, so it is no part
    of the pairing key. `execution.band_ops._paired_metadata` used to keep the key alone, so
    the frame reaching `diff` had `model_group_id`/`scenario_id` and no year at all -- while
    the caller had CHOSEN that year, and `resolve_analysis_chain` therefore (correctly) left
    it in the `remaining` dimensions it builds the chained stage's trailing `by` from
    (`collapsed_dimensions` means "the caller cannot steer it"). That `by` reached polars as
    a 400.

    The year is carried now, the same way `long_ops.join_long` carries a column only one
    operand has, so this chain and the download below both report it. `band_ops.
    _present_group_columns` remains as a backstop for a `by` naming a column no band carries
    at all (`tests/api/test_band_ops.py::TestGroupBandWide`), which this no longer reaches.

    The raster stand-in carries it, per this module's convention: what matters is that the
    grid is raster (so the frame stays band-wide -- the same chain over a vector grid is long
    throughout and never had the defect) and that the values are land-type codes.
    """

    # 705 is at or above `lnd_chng.BASELINE_ERA_BREAKPOINT`, so the derived baseline year is
    # 2025 and 2040 is the one comparison year band-wide output allows.
    QUERY = {
        "analysis": "lnd_chng,diff",
        **RASTER_DATASET,
        "dimension": "scenario_id",
        "order": "asc",
        "calendar_year": 2040,
        "model_group_id": 705,
        "scenario_id": [34, 35],
    }

    @pytest.fixture
    def land_change_dataset(self, base_path, monkeypatch):
        """Two scenarios, each with the derived baseline year and the comparison year.

        Both start as land type 1. Scenario 34 stays 1 (two-image code 1*10 + 1 = 11);
        scenario 35 becomes 2 (1*10 + 2 = 12). Their asc difference is therefore a uniform
        -1 -- a value neither stage could produce alone, so a chain that silently dropped
        either one would not land on it.
        """
        for scenario_id, comparison in ((34, 1), (35, 2)):
            write_raster_dataset(
                base_path,
                years=(2025, 2040),
                model_group_id=705,
                scenario_id=scenario_id,
                fill_fn=lambda year, comparison=comparison: 1 if year == 2025 else comparison,
            )
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        return base_path

    def test_data_tif_differences_two_land_change_surfaces(self, client, land_change_dataset):
        response = client.get("/v1/data.tif", params=self.QUERY)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            assert src.count == 1
            band = src.read(1, masked=True)
        assert band.count()
        assert set(band.compressed().tolist()) == {-1}

    def test_png_tile(self, client, land_change_dataset):
        """The reported URL's route. The failure was in the band-wide lowering rather than
        in the sink, so a tile and a download fail and pass together -- but the tile is what
        the map actually requests.
        """
        tile = _tile_for_grid(land_change_dataset)
        response = client.get(
            f"{DATA_URL}/{tile.z}/{tile.x}/{tile.y}.png",
            params={**self.QUERY, "color_map": "continuous:-1,#d53e4f,0,#00000000,1,#3288bd"},
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"

    def test_the_download_names_the_comparison_year(self, client, land_change_dataset):
        """`lnd_chng` alone, on the same fixture: the year the pairing could not key on is
        still what a reader needs to know which comparison they downloaded.

        It is one-sided rather than shared -- the baseline branch grouped it away so one
        baseline band could broadcast -- so it was absent from the band descriptions while
        the same selection's CSV (which lowers to long) reported it.
        `band_ops._paired_metadata` carries it now, one value per pair.
        """
        response = client.get(
            "/v1/data.tif",
            params={k: v for k, v in self.QUERY.items() if k not in ("dimension", "order")}
            | {"analysis": "lnd_chng"},
        )
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            assert src.descriptions == (
                "model_group_id=705, scenario_id=34, calendar_year=2040",
                "model_group_id=705, scenario_id=35, calendar_year=2040",
            )

    def test_the_chained_by_still_names_the_year_the_pairing_dropped(
        self, client, land_change_dataset
    ):
        """The published pipeline is the other half: the narrowing happens at execution, so
        `by` must still ARRIVE naming `calendar_year`. If it stopped doing so, the fix above
        would be untested by anything and this chain would quietly stop grouping on a
        dimension a long-shape run of it still splits by.
        """
        response = client.get(pipeline_url("lnd_chng,diff"), params=self.QUERY)
        assert response.status_code == 200, response.text
        expanded = parse_pipeline(response.json())
        assert expanded.steps[-1].op == "group"
        assert "calendar_year" in expanded.steps[-1].by


class TestLndTypePlusRoute:
    """`lnd_type_plus` on the consolidated routes. The fixed-input machinery itself is
    already carried by `sal_veg_investigation` above; what these pin is the one thing that
    differs -- every value this analysis is steered by is a real dimension name, so it is
    the sharpest test of the rule that a dimension value is ALWAYS a filter.
    `LndTypePlus.Parameters` is empty: `model_group_id`/`scenario_id`/`calendar_year` never
    reach the analysis as parameters, they land in the trailing `filter` the request layer
    composes against whatever frame the pipeline produced. Nothing about that is
    analysis-specific, which is why this analysis no longer rejects a selection outright.

    What the pipeline itself computes is pinned at the engine level
    (`test_analyses_registry.py::TestLndTypePlusExecution`); these pin that the route
    delivers it.
    """

    def test_data_tif(self, client, lnd_type_plus_data):
        response = client.get("/v1/data.tif", params=LND_TYPE_PLUS_QUERY)
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
            f"{DATA_URL}/{tile.z}/{tile.x}/{tile.y}.png",
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
        response = client.get(pipeline_url("lnd_type_plus"), params=LND_TYPE_PLUS_QUERY)
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
        response = client.get(pipeline_url("lnd_type_plus"), params=params)
        assert response.status_code == 200, response.text
        expanded = parse_pipeline(response.json())
        assert not any("calendar_year" in (step.where or "") for step in expanded.steps if step.op == "filter")
        assert not any(step.op == "group" for step in expanded.steps)

    def test_omitting_a_dimension_widens_the_selection_rather_than_failing(
        self, client, lnd_type_plus_data
    ):
        """There is no required parameter left to omit -- `LndTypePlus.Parameters` is empty,
        and every name this analysis is steered by is a dimension. A dimension left out of
        the query is therefore not a missing argument, it is an unconstrained axis: it drops
        out of the trailing selection conjunct and the remaining ones still narrow.
        """
        params = {k: v for k, v in LND_TYPE_PLUS_QUERY.items() if k != "model_group_id"}
        response = client.get(pipeline_url("lnd_type_plus"), params=params)
        assert response.status_code == 200, response.text
        expanded = parse_pipeline(response.json())
        selection = [step.where for step in expanded.steps if step.op == "filter"][-1]
        assert "model_group_id" not in selection
        assert "calendar_year = 2050" in selection

    def test_a_grid_cell_is_rejected_under_the_raster_format_this_analysis_publishes(
        self, client, lnd_type_plus_data
    ):
        """`grid_cell` no longer meets an analysis-specific refusal -- it is resolved against
        whatever grid the pipeline's OWN output descriptor names (this analysis names no
        dataset, so there is nothing else it could resolve against), and only then does the
        sink rule bite: cells lower a band-wide frame to rows, which a `tif` cannot carry.
        """
        with rio.open(next(lnd_type_plus_data.rglob("*.tif"))) as src:
            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        response = client.get(
            pipeline_url("lnd_type_plus"),
            params={
                **LND_TYPE_PLUS_QUERY,
                "grid_cell": f"{(west + east) / 2},{(south + north) / 2}",
            },
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "grid_cell" in detail
        assert "ask for a tabular format" in detail


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
            "sediment_dynamics",
        }


class TestAnalysesCatalogOutput:
    """`output` on a `/v1/analyses.json` entry -- what the analysis PRODUCES, for the ones
    that can say before a request exists.

    The field is what retired the portal's `CANONICAL_DERIVED_DATASETS`: a client offering
    `lnd_type_plus` in a variable picker needs its grid and time unit at picker-build time,
    before anything has been selected and so before `/v1/catalog.json` could be asked. It
    used to hardcode a plausible dataset triple and keep it in agreement with the analysis
    by hand; these pin that the agreement is now served.
    """

    @pytest.fixture
    def entries(self, client):
        response = client.get("/v1/analyses.json")
        assert response.status_code == 200, response.text
        return {entry["name"]: entry for entry in response.json()}

    def test_an_analysis_that_fixes_its_own_inputs_publishes_its_output(self, entries):
        """`morph_pixel_v002`/`annual` is what `CANONICAL_DERIVED_DATASETS` asserted by
        hand -- and it is not any ONE of this analysis's three inputs' triples: `stg` is
        annual on `hydro_compartment_v002` (a vector grid), `elev` and `lnd_type` are the
        two that share this grid. The presentation overlay is the analysis's own `Meta`,
        resolved by the same `output_body` `/v1/catalog.json` uses, which is why `legend`
        is `lnd_type_plus` rather than either raster input's.
        """
        output = entries["lnd_type_plus"]["output"]
        assert output["grid"] == "morph_pixel_v002"
        assert output["grid_type"] == "raster"
        assert output["time_unit"] == "annual"
        assert output["shape"] == "band_wide"
        assert output["label"] == "Land Type Plus"
        assert output["legend"] == "lnd_type_plus"
        assert output["result_kind"] == "category"
        assert output["collapsed_dimensions"] == ["aggregate_type"]

    def test_the_other_fixed_input_analysis_publishes_its_own_grid_too(self, entries):
        """Not a second copy of the case above: `sal_veg_investigation` outputs on a
        different grid from `lnd_type_plus` and declares no legend, so this pins that the
        field is derived per analysis rather than keyed on anything `lnd_type_plus`-shaped.
        """
        output = entries["sal_veg_investigation"]["output"]
        assert output["grid"] == "veg_grid_cell_v001"
        assert output["grid_type"] == "raster"
        assert output["legend"] is None

    @pytest.mark.parametrize("name", ["diff", "lnd_chng", "lnd_conf", "sediment_dynamics"])
    def test_an_analysis_that_reads_the_callers_dataset_publishes_no_output(self, name, entries):
        """Null rather than a guess. What `diff` produces depends entirely on the dataset
        it is pointed at -- which is the same reason `/v1/catalog.json` exists for the
        question and this route cannot answer it.

        All four land on null for structurally the same reason (the resolve refuses with
        nothing supplied), by four different refusals: `lnd_conf` has no dataset,
        `diff` has no `dimension` parameter, `lnd_chng` needs a selected `model_group_id`
        (its baseline year is derived from which era that is), and `sediment_dynamics`
        needs a `grid_cell` as well as a dataset. They are
        parametrized together because the guarantee is about the FIELD, not about which
        objection got there first.
        """
        assert entries[name]["output"] is None

    def test_it_agrees_with_what_the_catalog_route_reports_for_the_same_analysis(
        self, client, entries, lnd_type_plus_data
    ):
        """The one invariant that matters, since a client builds its picker from this route
        and then draws from the other: both go through `api.catalog.output_body`, so the
        only way they could disagree is if one stopped. `collapsed_dimensions` is excluded
        from the comparison -- `/v1/catalog.json` reports it for a SELECTION, and this
        route has none.
        """
        response = client.get("/v1/catalog.json", params={"analysis": "lnd_type_plus"})
        assert response.status_code == 200, response.text
        catalog_output = response.json()["output"]
        assert {k: v for k, v in entries["lnd_type_plus"]["output"].items() if k != "collapsed_dimensions"} == {
            k: v for k, v in catalog_output.items() if k != "collapsed_dimensions"
        }
