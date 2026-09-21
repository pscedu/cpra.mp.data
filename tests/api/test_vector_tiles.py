"""Tests for the MVT/PMTiles helpers in cpra.mp.data.analysis.vector."""

import gzip

import mercantile
import polars as pl
import pytest
from mapbox_vector_tile.Mapbox.vector_tile_pb2 import tile as VectorTile
from rasterio.crs import CRS
from rasterio.warp import transform

from cpra.mp.data.analysis.vector import (
    _data_to_id_value,
    inject_and_serialize,
    parse_tile,
    read_pmtile,
)
from tests.factories import ANNUAL_KWARGS, annual_frame, write_pmtiles

TILE = mercantile.Tile(x=1, y=2, z=3)


@pytest.fixture
def archive(tiles_path):
    write_pmtiles(tiles_path, "ecoregion_v001", TILE, [1, 2, 3])
    return tiles_path


class TestReadPmtile:
    def test_returns_bytes_for_a_present_tile(self, archive):
        assert read_pmtile("ecoregion_v001", TILE) is not None

    def test_missing_archive_returns_none(self, tiles_path):
        assert read_pmtile("no_such_grid", TILE) is None

    def test_tile_absent_from_the_archive_returns_none(self, archive):
        assert read_pmtile("ecoregion_v001", mercantile.Tile(x=9, y=9, z=9)) is None

    def test_returned_bytes_are_still_gzipped(self, archive):
        raw = read_pmtile("ecoregion_v001", TILE)
        assert gzip.decompress(raw)


class TestParseTile:
    def test_extracts_boundary_feature_ids(self, archive):
        _vt, ids = parse_tile(read_pmtile("ecoregion_v001", TILE))
        assert ids == [1, 2, 3]

    def test_returns_a_parsed_tile(self, archive):
        vt, _ids = parse_tile(read_pmtile("ecoregion_v001", TILE))
        assert [layer.name for layer in vt.layers] == ["boundary"]

    def test_ignores_layers_that_are_not_boundary(self):
        vt = VectorTile()
        layer = vt.layers.add()
        layer.name = "other"
        layer.version = 2
        layer.extent = 4096
        feature = layer.features.add()
        feature.id = 7

        _parsed, ids = parse_tile(gzip.compress(vt.SerializeToString()))
        assert ids == []


class TestDataToIdValue:
    """_data_to_id_value now reads a long-format frame (grid_cell/value columns, one row
    per cell) rather than a wide frame's first row of numeric ID columns."""

    def test_extracts_grid_cell_value_pairs(self):
        df = pl.DataFrame({"calendar_year": [2030, 2030], "grid_cell": [1, 2], "value": [1.5, 2.5]})
        assert _data_to_id_value(df) == {1: 1.5, 2: 2.5}

    def test_skips_null_values(self):
        df = pl.DataFrame(
            {"grid_cell": [1, 2], "value": [1.5, None]}, schema={"grid_cell": pl.Int32, "value": pl.Float64}
        )
        assert _data_to_id_value(df) == {1: 1.5}

    def test_no_grid_cell_column_yields_empty(self):
        assert _data_to_id_value(pl.DataFrame({"calendar_year": [2030]})) == {}

    def test_empty_frame_yields_empty(self):
        df = pl.DataFrame({"grid_cell": [], "value": []}, schema={"grid_cell": pl.Int32, "value": pl.Float64})
        assert _data_to_id_value(df) == {}

    def test_negative_ids_are_matched(self):
        df = pl.DataFrame({"grid_cell": [-1]}, schema={"grid_cell": pl.Int32}).with_columns(
            value=pl.Series([3.5], dtype=pl.Float64)
        )
        assert _data_to_id_value(df) == {-1: 3.5}


class TestInjectAndSerialize:
    def _tile_with_ids(self, ids):
        vt = VectorTile()
        layer = vt.layers.add()
        layer.name = "boundary"
        layer.version = 2
        layer.extent = 4096
        for feature_id in ids:
            feature = layer.features.add()
            feature.id = feature_id
            feature.type = 1
            feature.geometry.extend([9, 10, 10])
        return vt

    def test_adds_a_value_key(self):
        vt = self._tile_with_ids([1, 2])
        result = VectorTile()
        result.ParseFromString(inject_and_serialize(vt, {1: 5.5}))
        assert "value" in result.layers[0].keys

    def test_injects_values_for_matching_features(self):
        vt = self._tile_with_ids([1, 2])
        result = VectorTile()
        result.ParseFromString(inject_and_serialize(vt, {1: 5.5, 2: 6.5}))
        values = [v.double_value for v in result.layers[0].values]
        assert values == [5.5, 6.5]

    def test_features_without_a_value_get_no_tags(self):
        vt = self._tile_with_ids([1, 2])
        result = VectorTile()
        result.ParseFromString(inject_and_serialize(vt, {1: 5.5}))
        by_id = {f.id: f for f in result.layers[0].features}
        assert len(by_id[1].tags) == 2
        assert len(by_id[2].tags) == 0

    def test_geometry_bytes_are_untouched(self):
        """The docstring's core claim: only keys/values/tags change, never geometry."""
        vt = self._tile_with_ids([1, 2])
        before = [list(f.geometry) for f in vt.layers[0].features]

        result = VectorTile()
        result.ParseFromString(inject_and_serialize(vt, {1: 5.5, 2: 6.5}))
        after = [list(f.geometry) for f in result.layers[0].features]
        assert after == before

    def test_non_boundary_layers_are_left_alone(self):
        vt = self._tile_with_ids([1])
        other = vt.layers.add()
        other.name = "other"
        other.version = 2
        other.extent = 4096

        result = VectorTile()
        result.ParseFromString(inject_and_serialize(vt, {1: 5.5}))
        by_name = {layer.name: layer for layer in result.layers}
        assert "value" not in by_name["other"].keys


class TestGetVectorTile:
    @pytest.fixture
    def vector_data(self, base_path):
        from cpra.mp.data.tabular import write_parquet
        from tests.factories import annual_frame

        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        return base_path

    @pytest.fixture
    def ecoregion_tiles(self, tiles_path):
        write_pmtiles(tiles_path, "ecoregion_v001", TILE, [1, 2, 3])
        return tiles_path

    async def test_injects_dataset_values_into_the_tile(
        self, dataset_factory, vector_data, ecoregion_tiles
    ):
        from tests.factories import ANNUAL_KWARGS, annual_frame

        dimensions = {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}
        dataset = dataset_factory(dimensions)
        await dataset.apply_params()

        body = await dataset.get_vector_tile(TILE.x, TILE.y, TILE.z)
        result = VectorTile()
        result.ParseFromString(body)

        expected = annual_frame(years=(2030,))["1"][0]
        assert result.layers[0].values[0].double_value == pytest.approx(expected, rel=1e-5)

    async def test_missing_tile_returns_none(self, dataset_factory, vector_data, tiles_path):
        from tests.factories import ANNUAL_KWARGS

        dimensions = {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}
        dataset = dataset_factory(dimensions)
        await dataset.apply_params()
        assert await dataset.get_vector_tile(TILE.x, TILE.y, TILE.z) is None

    async def test_grid_cell_filter_limits_injected_columns(
        self, dataset_factory, vector_data, ecoregion_tiles
    ):
        from tests.factories import ANNUAL_KWARGS

        dimensions = {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}
        dataset = dataset_factory(dimensions, grid_cell="1")
        await dataset.apply_params()

        body = await dataset.get_vector_tile(TILE.x, TILE.y, TILE.z)
        result = VectorTile()
        result.ParseFromString(body)
        assert len(result.layers[0].values) == 1

    async def test_coordinate_grid_cell_still_matches_features_by_id(
        self, dataset_factory, vector_data, ecoregion_tiles, geometry_path
    ):
        """A coordinate-selected grid_cell is resolved to an ID before it reaches the tile,
        which is what a pmtile feature is keyed by."""
        from shapely.geometry import box

        from tests.factories import ANNUAL_KWARGS, annual_frame, write_grid_geometry

        write_grid_geometry(
            geometry_path, "ecoregion_v001", [(1, box(500000, 3300000, 501000, 3301000))]
        )
        lng, lat = [
            c[0] for c in transform(CRS.from_epsg(26915), CRS.from_epsg(4326), [500500], [3300500])
        ]

        dimensions = {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}
        dataset = dataset_factory(dimensions, grid_cell=f"{lng!s},{lat!s}")
        await dataset.apply_params()

        body = await dataset.get_vector_tile(TILE.x, TILE.y, TILE.z)
        result = VectorTile()
        result.ParseFromString(body)
        expected = annual_frame(years=(2030,))["1"][0]
        assert len(result.layers[0].values) == 1
        assert result.layers[0].values[0].double_value == pytest.approx(expected, rel=1e-5)


class TestAnalysisVectorTileRoute:
    """`/v1/analysis/{z}/{x}/{y}.pbf` (G5) -- the pipeline counterpart to
    `get_vector_tile`/`TestGetVectorTile` above, exercised through the real ASGI app
    (`client`) rather than `BaseAnalysis.get_vector_tile()` directly, since the point of
    this gap is the route itself: it did not exist before.

    Rebound onto the NEW engine (`request.pipeline.parse_pipeline`/
    `pipeline_to_query`, `execution.plan.compile_plan`, `execution.sink.
    sink_vector_tile`) rather than the old `analysis.pipeline`/`planner.compile_pipeline`/
    `VectorTileResponse` stack the three tests below originally exercised -- two of the
    three needed real changes, not just an updated expectation, because they leaned on
    old-engine behavior the new engine's stricter op rules do not reproduce:

    - `VECTOR_PIPELINE` no longer includes a `project ... to: ecoregion_v001` step: the
      old engine apparently tolerated a same-grid project as a no-op, but the new engine's
      `project` lowering table requires an explicit `aggregate` for a vector-to-vector
      project with no default (there is no single obviously-correct way to combine several
      source rows onto one target cell), and this input is already on `ecoregion_v001` --
      the step was never doing anything, so it is simply dropped instead of given an
      aggregate that would change nothing.
    - `test_rejects_a_raster_output_grid`'s pipeline used to project a raster input onto
      the SAME raster grid it started on; the new engine's project-direction table has no
      raster-to-raster entry at all (there is no crosswalk between two raster lattices --
      see `execution.execute`'s raster-to-raster `ShapeError`), so that pipeline now fails
      earlier, with a different message, than the one this test means to exercise. Fixed
      by dropping the project (the input is already raster-typed) and reaching the same
      band-wide-final-frame-on-a-raster-grid condition via a `filter`+`group` chain
      instead -- still the same reason a `.pbf` request on it should be rejected.
    """

    VECTOR_PIPELINE = {
        "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
        "steps": [],
        "output": {"format": "csv"},
    }

    async def test_serves_a_real_vector_tile(self, client, base_path, tiles_path, monkeypatch):
        from cpra.mp.data import config
        from cpra.mp.data.analysis.request.pipeline import parse_pipeline, pipeline_to_query
        from cpra.mp.data.tabular import write_parquet

        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        write_pmtiles(tiles_path, "ecoregion_v001", TILE, [1, 2, 3])
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        query = pipeline_to_query(parse_pipeline(self.VECTOR_PIPELINE))
        response = client.get(f"/v1/analysis/{TILE.z}/{TILE.x}/{TILE.y}.pbf", params=query)

        # A wrong grid/route wiring would either 4xx here or return an empty/garbled MVT
        # payload -- the value assertion below independently recomputes the expected
        # number from `annual_frame` rather than trusting whatever the response says.
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"
        result = VectorTile()
        result.ParseFromString(response.content)
        expected = annual_frame(years=(2030,))["1"][0]
        assert result.layers[0].values[0].double_value == pytest.approx(expected, rel=1e-5)

    async def test_rejects_a_raster_output_grid(self, client, base_path, monkeypatch):
        """A pipeline whose final frame is band-wide (raster), reached via a `filter`+
        `group` chain over a raster input with no project narrowing it to a vector grid,
        400s on `.pbf` -- `api.pipeline._require_vector_final_frame` (the new engine's
        replacement for the old `compiled.validate_vector_tile()`) rejects it outright as
        a `ShapeError`, reached here without ever needing a PMTiles archive on disk (the
        rejection fires before any tile lookup, same as the old engine's own check did).
        A test that instead let this reach `sink_vector_tile` and get a `None` back would
        wrongly conflate "this request can never make a vector tile" with "no data here
        right now" -- the two 400-vs-204 paths this suite must keep telling apart.
        """
        from cpra.mp.data import config
        from cpra.mp.data.analysis.request.pipeline import parse_pipeline, pipeline_to_query
        from tests.factories import RASTER_KWARGS, write_raster_dataset

        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "filter", "input": "vct", "where": "calendar_year > 0"},
                {"op": "group", "by": [], "aggregate": "max"},
            ],
            "output": {"format": "tif"},
        }
        query = pipeline_to_query(parse_pipeline(pipeline))
        response = client.get("/v1/analysis/3/1/2.pbf", params=query)

        assert response.status_code == 400
        assert "not a long frame on a vector grid" in response.json()["detail"]

    async def test_missing_tile_returns_204(self, client, base_path, monkeypatch):
        """Same pipeline as the success case, but with no PMTiles archive written for the
        grid.

        Under the OLD engine this 404ed ("no tile file for grid") -- `sink_vector_tile`'s
        own docstring makes this a DELIBERATE behavior change, not an oversight: "no
        archive for this grid" is exactly as much "nothing to render" as "no feature under
        this pixel" or "every matching row was null," and the new engine answers all of
        them the same bodiless-204 way rather than singling one out as a client-facing
        error. A test still asserting 404 here would be pinning the exact old-engine
        behavior deliverable E is required to update, not a case where the old assertion
        happens to still hold.
        """
        from cpra.mp.data import config
        from cpra.mp.data.analysis.request.pipeline import parse_pipeline, pipeline_to_query
        from cpra.mp.data.tabular import write_parquet

        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        query = pipeline_to_query(parse_pipeline(self.VECTOR_PIPELINE))
        response = client.get(f"/v1/analysis/{TILE.z}/{TILE.x}/{TILE.y}.pbf", params=query)

        # A 204 that still carried a body would violate the HTTP spec for that status and
        # would trip up a client trusting Content-Length; content == b"" is the part a
        # not-actually-empty response would fail.
        assert response.status_code == 204
        assert response.content == b""
