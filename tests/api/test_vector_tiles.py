"""Tests for the MVT/PMTiles helpers in cpra.mp.data.api.vector."""

import gzip

import mercantile
import polars as pl
import pytest
from mapbox_vector_tile.Mapbox.vector_tile_pb2 import tile as VectorTile
from rasterio.crs import CRS
from rasterio.warp import transform

from cpra.mp.data.api.vector import (
    _data_to_id_value,
    inject_and_serialize,
    parse_tile,
    read_pmtile,
)

from tests.factories import write_pmtiles

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
    def test_extracts_numeric_columns_from_the_first_row(self):
        df = pl.DataFrame({"calendar_year": [2030], "1": [1.5], "2": [2.5]})
        assert _data_to_id_value(df) == {1: 1.5, 2: 2.5}

    def test_skips_null_values(self):
        df = pl.DataFrame({"1": [1.5], "2": [None]}, schema={"1": pl.Float64, "2": pl.Float64})
        assert _data_to_id_value(df) == {1: 1.5}

    def test_only_the_first_row_is_used(self):
        df = pl.DataFrame({"1": [1.5, 9.9]})
        assert _data_to_id_value(df) == {1: 1.5}

    def test_no_numeric_columns_yields_empty(self):
        assert _data_to_id_value(pl.DataFrame({"calendar_year": [2030]})) == {}

    def test_empty_frame_yields_empty(self):
        df = pl.DataFrame({"1": []}, schema={"1": pl.Float64})
        assert _data_to_id_value(df) == {}

    def test_negative_ids_are_matched(self):
        """The value-column pattern allows a leading minus sign."""
        df = pl.DataFrame({"-1": [3.5]})
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

    async def test_geography_filter_limits_injected_columns(
        self, dataset_factory, vector_data, ecoregion_tiles
    ):
        from tests.factories import ANNUAL_KWARGS

        dimensions = {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}
        dataset = dataset_factory(dimensions, geography="1")
        await dataset.apply_params()

        body = await dataset.get_vector_tile(TILE.x, TILE.y, TILE.z)
        result = VectorTile()
        result.ParseFromString(body)
        assert len(result.layers[0].values) == 1

    async def test_coordinate_geography_still_matches_features_by_id(
        self, dataset_factory, vector_data, ecoregion_tiles, geometry_path
    ):
        """A coordinate-selected geography is resolved to an ID before it reaches the tile,
        because features are keyed by ID here — the coordinate relabeling that the tabular
        response applies would leave nothing for a feature to match."""
        from shapely.geometry import box

        from tests.factories import ANNUAL_KWARGS, annual_frame, write_grid_geometry

        write_grid_geometry(
            geometry_path, "ecoregion_v001", [(1, box(500000, 3300000, 501000, 3301000))]
        )
        lng, lat = [
            c[0] for c in transform(CRS.from_epsg(26915), CRS.from_epsg(4326), [500500], [3300500])
        ]

        dimensions = {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}
        dataset = dataset_factory(dimensions, geography=f"{lng!s},{lat!s}")
        await dataset.apply_params()

        body = await dataset.get_vector_tile(TILE.x, TILE.y, TILE.z)
        result = VectorTile()
        result.ParseFromString(body)
        expected = annual_frame(years=(2030,))["1"][0]
        assert len(result.layers[0].values) == 1
        assert result.layers[0].values[0].double_value == pytest.approx(expected, rel=1e-5)
