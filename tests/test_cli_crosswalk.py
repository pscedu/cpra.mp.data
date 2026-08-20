"""Tests for `python -m cpra.mp.data.cli crosswalk`.

Importing `cpra.mp.data.cli.crosswalk` (even just for its side effect) registers the
`crosswalk` group on the shared `cli` object exported by `cpra.mp.data.cli` -- that's what
makes `tests/test_cli.py::test_crosswalk_group_is_registered` pass once this module is
collected in the same pytest session.
"""

import pyarrow.parquet as pq
import pytest
import rasterio as rio
from click.testing import CliRunner
from shapely.geometry import box

from cpra.mp.data.cli import cli, crosswalk  # noqa: F401 - side effect: registers `crosswalk`
from cpra.mp.data.grids import GRIDS

from tests.factories import write_grid_geometry

SRC_GRID = "hydro_compartment_v001"
DST_GRID_A = "ecoregion_v001"
DST_GRID_B = "parish_v2020"
DST_GRID_C = "clara_community_v001"
# Registered in GRIDS, but deliberately never given a .fgb file, so it exists as a valid
# grid name yet fails to open -- the tool for exercising the partial-failure path.
MISSING_DST_GRID = "clara_grid_cell_v001"

RASTER_GRID = "veg_grid_cell_v001"
# A second raster grid, used only as an invalid destination -- destinations must be vector.
OTHER_RASTER_GRID = "morph_pixel_v001"
RASTER_DST_A = "ecoregion_v001"
RASTER_DST_B = "parish_v2020"

runner = CliRunner()


def _src_features(extra_id=None):
    """Two small boxes, both inside the covering polygon `_covering_polygon` returns."""
    features = [
        (1, box(500100, 3300100, 500200, 3300200)),
        (2, box(505000, 3305000, 505100, 3305100)),
    ]
    if extra_id is not None:
        features.append((extra_id, box(507000, 3307000, 507100, 3307100)))
    return features


def _covering_polygon(dst_id):
    """A single polygon large enough to contain every centroid `_src_features` produces."""
    return [(dst_id, box(490000, 3290000, 520000, 3320000))]


def _row_count(path):
    return pq.read_table(path).num_rows


@pytest.fixture
def src_geometry(geometry_path):
    return write_grid_geometry(geometry_path, SRC_GRID, _src_features())


@pytest.fixture
def dst_geometries(geometry_path):
    write_grid_geometry(geometry_path, DST_GRID_A, _covering_polygon(100))
    write_grid_geometry(geometry_path, DST_GRID_B, _covering_polygon(200))
    write_grid_geometry(geometry_path, DST_GRID_C, _covering_polygon(300))
    return geometry_path


class TestVector:
    def test_happy_path_defaults_to_every_fgb_but_the_source(self, src_geometry, dst_geometries, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "vector", SRC_GRID])

        assert result.exit_code == 0, result.output
        for dst in (DST_GRID_A, DST_GRID_B, DST_GRID_C):
            out = crosswalk_path / f"{SRC_GRID}__{dst}.parquet"
            assert out.exists()
            assert _row_count(out) == 2
        assert "DONE" in result.output

    def test_to_filters_destinations(self, src_geometry, dst_geometries, crosswalk_path):
        result = runner.invoke(
            cli, ["crosswalk", "vector", SRC_GRID, "--to", DST_GRID_A, "--to", DST_GRID_B]
        )

        assert result.exit_code == 0, result.output
        assert (crosswalk_path / f"{SRC_GRID}__{DST_GRID_A}.parquet").exists()
        assert (crosswalk_path / f"{SRC_GRID}__{DST_GRID_B}.parquet").exists()
        assert not (crosswalk_path / f"{SRC_GRID}__{DST_GRID_C}.parquet").exists()

    def test_no_destinations_found_is_a_usage_error(self, src_geometry, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "vector", SRC_GRID])

        assert result.exit_code != 0
        assert "No vector destination grids" in result.output

    def test_raster_grid_fgb_is_skipped_as_a_default_destination(self, src_geometry, geometry_path, crosswalk_path):
        # `veg_grid_cell_v001` really does ship a .fgb alongside the vector grids, so a bare
        # *.fgb glob would pick it as a destination and emit a crosswalk no consumer can use.
        write_grid_geometry(geometry_path, DST_GRID_A, _covering_polygon(100))
        write_grid_geometry(geometry_path, RASTER_GRID, _covering_polygon(400))

        result = runner.invoke(cli, ["crosswalk", "vector", SRC_GRID])

        assert result.exit_code == 0, result.output
        assert (crosswalk_path / f"{SRC_GRID}__{DST_GRID_A}.parquet").exists()
        assert not (crosswalk_path / f"{SRC_GRID}__{RASTER_GRID}.parquet").exists()
        assert f"SKIP: {RASTER_GRID} (a raster grid)" in result.output

    def test_unregistered_fgb_is_skipped_as_a_default_destination(self, src_geometry, geometry_path, crosswalk_path):
        write_grid_geometry(geometry_path, DST_GRID_A, _covering_polygon(100))
        # Never opened -- the grid-registry check rejects it first, so the contents (and the
        # fact that this isn't valid FlatGeobuf at all) never come into play.
        (geometry_path / "leftover_export.fgb").touch()

        result = runner.invoke(cli, ["crosswalk", "vector", SRC_GRID])

        assert result.exit_code == 0, result.output
        assert (crosswalk_path / f"{SRC_GRID}__{DST_GRID_A}.parquet").exists()
        assert "SKIP: leftover_export (not in the grid registry)" in result.output

    def test_raster_grid_named_by_to_is_rejected(self, src_geometry, dst_geometries, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "vector", SRC_GRID, "--to", RASTER_GRID])

        assert result.exit_code != 0
        assert f"{RASTER_GRID!r} is a 'raster' grid, expected a 'vector' grid" in result.output

    def test_unknown_src_grid_is_rejected(self, geometry_path, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "vector", "not_a_real_grid"])

        assert result.exit_code != 0
        assert "Unknown grid" in result.output

    def test_unknown_to_grid_is_rejected(self, src_geometry, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "vector", SRC_GRID, "--to", "not_a_real_grid"])

        assert result.exit_code != 0
        assert "Unknown grid" in result.output

    def test_wrong_grid_type_is_rejected(self, geometry_path, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "vector", RASTER_GRID])

        assert result.exit_code != 0
        assert "raster" in result.output

    def test_no_overwrite_skips_existing_output(self, src_geometry, dst_geometries, crosswalk_path):
        first = runner.invoke(cli, ["crosswalk", "vector", SRC_GRID, "--to", DST_GRID_A])
        assert first.exit_code == 0, first.output

        second = runner.invoke(cli, ["crosswalk", "vector", SRC_GRID, "--to", DST_GRID_A])
        assert second.exit_code == 0, second.output
        assert "SKIP" in second.output
        assert _row_count(crosswalk_path / f"{SRC_GRID}__{DST_GRID_A}.parquet") == 2

    def test_overwrite_rewrites_existing_output(self, geometry_path, crosswalk_path):
        write_grid_geometry(geometry_path, SRC_GRID, _src_features())
        write_grid_geometry(geometry_path, DST_GRID_A, _covering_polygon(100))
        out = crosswalk_path / f"{SRC_GRID}__{DST_GRID_A}.parquet"

        first = runner.invoke(cli, ["crosswalk", "vector", SRC_GRID, "--to", DST_GRID_A])
        assert first.exit_code == 0, first.output
        assert _row_count(out) == 2

        # Rewrite the source with an extra feature; without --overwrite the stale 2-row
        # file would stick around, so seeing 3 rows proves the rerun actually happened.
        write_grid_geometry(geometry_path, SRC_GRID, _src_features(extra_id=3))
        second = runner.invoke(cli, ["crosswalk", "vector", SRC_GRID, "--to", DST_GRID_A, "--overwrite"])
        assert second.exit_code == 0, second.output
        assert _row_count(out) == 3

    def test_one_failing_destination_does_not_abort_the_rest(self, src_geometry, dst_geometries, crosswalk_path):
        result = runner.invoke(
            cli, ["crosswalk", "vector", SRC_GRID, "--to", DST_GRID_A, "--to", MISSING_DST_GRID]
        )

        assert result.exit_code == 1
        assert (crosswalk_path / f"{SRC_GRID}__{DST_GRID_A}.parquet").exists()
        assert "FAIL" in result.output
        assert "FAILED: 1 of 2" in result.output


def _raster_bounds():
    return GRIDS[RASTER_GRID].bounds


@pytest.fixture
def raster_dst_geometry(geometry_path):
    minx, miny, maxx, maxy = _raster_bounds()
    write_grid_geometry(geometry_path, RASTER_DST_A, [(1, box(minx, miny, maxx, maxy))])
    return geometry_path


@pytest.fixture
def raster_dst_geometries(geometry_path):
    minx, miny, maxx, maxy = _raster_bounds()
    write_grid_geometry(geometry_path, RASTER_DST_A, [(1, box(minx, miny, maxx, maxy))])
    write_grid_geometry(geometry_path, RASTER_DST_B, [(1, box(minx, miny, maxx, maxy))])
    return geometry_path


class TestRaster:
    def test_happy_path(self, raster_dst_geometry, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "raster", RASTER_GRID, "--to", RASTER_DST_A])

        assert result.exit_code == 0, result.output
        out = crosswalk_path / f"{RASTER_GRID}__{RASTER_DST_A}.tif"
        assert out.exists()
        with rio.open(out) as f:
            assert f.tags().get("dst_ids") == "1"
        assert "DONE" in result.output

    def test_to_filters_destinations(self, raster_dst_geometries, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "raster", RASTER_GRID, "--to", RASTER_DST_A])

        assert result.exit_code == 0, result.output
        assert (crosswalk_path / f"{RASTER_GRID}__{RASTER_DST_A}.tif").exists()
        assert not (crosswalk_path / f"{RASTER_GRID}__{RASTER_DST_B}.tif").exists()

    def test_defaults_to_every_fgb_in_grids_path(self, raster_dst_geometries, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "raster", RASTER_GRID])

        assert result.exit_code == 0, result.output
        assert (crosswalk_path / f"{RASTER_GRID}__{RASTER_DST_A}.tif").exists()
        assert (crosswalk_path / f"{RASTER_GRID}__{RASTER_DST_B}.tif").exists()

    def test_raster_grid_fgb_is_skipped_as_a_default_destination(
        self, raster_dst_geometry, geometry_path, crosswalk_path
    ):
        # Same filter as the vector command -- a raster-to-raster crosswalk is meaningless,
        # and `_resolve_destinations` is shared, so both commands get the guard.
        minx, miny, maxx, maxy = _raster_bounds()
        write_grid_geometry(geometry_path, OTHER_RASTER_GRID, [(1, box(minx, miny, maxx, maxy))])

        result = runner.invoke(cli, ["crosswalk", "raster", RASTER_GRID])

        assert result.exit_code == 0, result.output
        assert (crosswalk_path / f"{RASTER_GRID}__{RASTER_DST_A}.tif").exists()
        assert not (crosswalk_path / f"{RASTER_GRID}__{OTHER_RASTER_GRID}.tif").exists()
        assert f"SKIP: {OTHER_RASTER_GRID} (a raster grid)" in result.output

    def test_raster_grid_named_by_to_is_rejected(self, raster_dst_geometry, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "raster", RASTER_GRID, "--to", OTHER_RASTER_GRID])

        assert result.exit_code != 0
        assert f"{OTHER_RASTER_GRID!r} is a 'raster' grid, expected a 'vector' grid" in result.output

    def test_unknown_grid_is_rejected(self, geometry_path, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "raster", "not_a_real_grid"])

        assert result.exit_code != 0
        assert "Unknown grid" in result.output

    def test_wrong_grid_type_is_rejected(self, geometry_path, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "raster", DST_GRID_A])

        assert result.exit_code != 0
        assert "vector" in result.output

    def test_no_overwrite_skips_existing_output(self, raster_dst_geometry, crosswalk_path):
        first = runner.invoke(cli, ["crosswalk", "raster", RASTER_GRID, "--to", RASTER_DST_A])
        assert first.exit_code == 0, first.output

        second = runner.invoke(cli, ["crosswalk", "raster", RASTER_GRID, "--to", RASTER_DST_A])
        assert second.exit_code == 0, second.output
        assert "SKIP" in second.output

    def test_overwrite_rewrites_existing_output(self, geometry_path, crosswalk_path):
        minx, miny, maxx, maxy = _raster_bounds()
        midx = (minx + maxx) / 2
        out = crosswalk_path / f"{RASTER_GRID}__{RASTER_DST_A}.tif"

        write_grid_geometry(geometry_path, RASTER_DST_A, [(1, box(minx, miny, maxx, maxy))])
        first = runner.invoke(cli, ["crosswalk", "raster", RASTER_GRID, "--to", RASTER_DST_A])
        assert first.exit_code == 0, first.output
        with rio.open(out) as f:
            assert f.tags().get("dst_ids") == "1"

        # Rewrite the destination with a second zone; without --overwrite the stale
        # single-zone file would stick around, so seeing both zone IDs proves the rerun
        # actually happened.
        write_grid_geometry(
            geometry_path,
            RASTER_DST_A,
            [(1, box(minx, miny, midx, maxy)), (2, box(midx, miny, maxx, maxy))],
        )
        second = runner.invoke(cli, ["crosswalk", "raster", RASTER_GRID, "--to", RASTER_DST_A, "--overwrite"])
        assert second.exit_code == 0, second.output
        with rio.open(out) as f:
            assert f.tags().get("dst_ids") == "1,2"

    def test_one_failing_destination_does_not_abort_the_rest(self, raster_dst_geometry, crosswalk_path):
        result = runner.invoke(
            cli, ["crosswalk", "raster", RASTER_GRID, "--to", RASTER_DST_A, "--to", MISSING_DST_GRID]
        )

        assert result.exit_code == 1
        assert (crosswalk_path / f"{RASTER_GRID}__{RASTER_DST_A}.tif").exists()
        assert "FAIL" in result.output
        assert "FAILED: 1 of 2" in result.output
