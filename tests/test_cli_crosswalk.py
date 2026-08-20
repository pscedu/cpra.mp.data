"""Tests for `python -m cpra.mp.data.cli crosswalk`.

Importing `cpra.mp.data.cli.crosswalk` (even just for its side effect) registers the
`crosswalk` command on the shared `cli` object exported by `cpra.mp.data.cli` -- that's
what makes `tests/test_cli.py::test_crosswalk_command_is_registered` pass once this module
is collected in the same pytest session.
"""

import pyarrow.parquet as pq
import pytest
import rasterio as rio
from click.testing import CliRunner
from shapely.geometry import box

from cpra.mp.data.cli import cli, crosswalk  # noqa: F401 - side effect: registers `crosswalk`
from cpra.mp.data.grids import GRIDS

from tests.factories import write_grid_geometry

VECTOR_SRC = "hydro_compartment_v001"
VECTOR_SRC_B = "hydro_compartment_v002"
DST_A = "ecoregion_v001"
DST_B = "parish_v2020"
DST_C = "clara_community_v001"

RASTER_SRC = "veg_grid_cell_v001"
RASTER_SRC_B = "veg_grid_cell_v002"
# A raster grid is never a valid destination, and is big enough that accidentally treating
# it as a source in a test would be obvious: 8265 x 20674 pixels.
BIG_RASTER = "morph_pixel_v001"

runner = CliRunner()


def _src_features(extra_id=None):
    """Two small boxes, both inside the polygon `_covering_polygon` returns."""
    features = [
        (1, box(500100, 3300100, 500200, 3300200)),
        (2, box(505000, 3305000, 505100, 3305100)),
    ]
    if extra_id is not None:
        features.append((extra_id, box(507000, 3307000, 507100, 3307100)))
    return features


def _covering_polygon(dst_id):
    """One polygon covering the whole `veg_grid_cell_v001` footprint.

    That footprint also contains every centroid `_src_features` produces, so the same
    destination geometry works for both a vector and a raster source -- which is the point
    of a single command handling both.
    """
    return [(dst_id, box(*GRIDS[RASTER_SRC].bounds))]


def _row_count(path):
    return pq.read_table(path).num_rows


def _zone_ids(path):
    with rio.open(path) as f:
        return f.tags().get("dst_ids")


@pytest.fixture
def vector_src(geometry_path):
    return write_grid_geometry(geometry_path, VECTOR_SRC, _src_features())


@pytest.fixture
def destinations(geometry_path):
    write_grid_geometry(geometry_path, DST_A, _covering_polygon(100))
    write_grid_geometry(geometry_path, DST_B, _covering_polygon(200))
    return geometry_path


@pytest.fixture
def small_registry(monkeypatch):
    """Drop the morphology grids for tests that let `--from` default to the whole registry.

    `morph_pixel_v001` is 8265 x 20674 = 171M pixels and v002 another 97M; rasterizing both
    for a test that only cares about *which* grids get picked would dominate the suite's
    runtime. Both are `raster`, and `veg_grid_cell_v001`/`v002` cover that branch at ~380K
    pixels each.
    """

    for name in (BIG_RASTER, "morph_pixel_v002"):
        monkeypatch.delitem(GRIDS, name)


class TestDispatch:
    def test_vector_source_writes_a_parquet_crosswalk(self, vector_src, destinations, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "--from", VECTOR_SRC, "--to", DST_A])

        assert result.exit_code == 0, result.output
        out = crosswalk_path / f"{VECTOR_SRC}__{DST_A}.parquet"
        assert _row_count(out) == 2
        assert "2 rows" in result.output

    def test_raster_source_writes_a_cog_crosswalk(self, destinations, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "--from", RASTER_SRC, "--to", DST_A])

        assert result.exit_code == 0, result.output
        out = crosswalk_path / f"{RASTER_SRC}__{DST_A}.tif"
        assert _zone_ids(out) == "100"
        assert "windows" in result.output

    def test_mixed_sources_each_take_their_own_path(self, vector_src, destinations, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "--from", VECTOR_SRC, "--from", RASTER_SRC, "--to", DST_A])

        assert result.exit_code == 0, result.output
        assert _row_count(crosswalk_path / f"{VECTOR_SRC}__{DST_A}.parquet") == 2
        assert _zone_ids(crosswalk_path / f"{RASTER_SRC}__{DST_A}.tif") == "100"

    def test_every_from_to_pair_is_built(self, vector_src, destinations, crosswalk_path):
        result = runner.invoke(
            cli,
            ["crosswalk", "--from", VECTOR_SRC, "--from", RASTER_SRC, "--to", DST_A, "--to", DST_B],
        )

        assert result.exit_code == 0, result.output
        assert "Building 4 crosswalk(s): 2 source(s) -> 2 destination(s)" in result.output
        assert (crosswalk_path / f"{VECTOR_SRC}__{DST_A}.parquet").exists()
        assert (crosswalk_path / f"{VECTOR_SRC}__{DST_B}.parquet").exists()
        assert (crosswalk_path / f"{RASTER_SRC}__{DST_A}.tif").exists()
        assert (crosswalk_path / f"{RASTER_SRC}__{DST_B}.tif").exists()

    def test_repeated_grids_are_deduplicated(self, vector_src, destinations, crosswalk_path):
        result = runner.invoke(
            cli, ["crosswalk", "--from", VECTOR_SRC, "--from", VECTOR_SRC, "--to", DST_A, "--to", DST_A]
        )

        assert result.exit_code == 0, result.output
        assert "Building 1 crosswalk(s)" in result.output

    def test_a_grid_is_not_crosswalked_to_itself(self, destinations, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "--from", DST_A, "--from", DST_B, "--to", DST_A])

        assert result.exit_code == 0, result.output
        assert "Building 1 crosswalk(s)" in result.output
        assert (crosswalk_path / f"{DST_B}__{DST_A}.parquet").exists()
        assert not (crosswalk_path / f"{DST_A}__{DST_A}.parquet").exists()

    def test_only_a_self_pair_is_a_usage_error(self, destinations, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "--from", DST_A, "--to", DST_A])

        assert result.exit_code != 0
        assert "No crosswalks to build" in result.output


class TestDefaults:
    def test_from_defaults_to_every_grid_in_the_registry(
        self, small_registry, vector_src, destinations, crosswalk_path
    ):
        result = runner.invoke(cli, ["crosswalk", "--to", DST_A])

        assert result.exit_code == 0, result.output
        # The two vector grids with geometry (one of which is the destination, so it only
        # contributes a dropped self-pair) plus both raster grids, which need none.
        assert (crosswalk_path / f"{VECTOR_SRC}__{DST_A}.parquet").exists()
        assert (crosswalk_path / f"{DST_B}__{DST_A}.parquet").exists()
        assert (crosswalk_path / f"{RASTER_SRC}__{DST_A}.tif").exists()
        assert (crosswalk_path / f"{RASTER_SRC_B}__{DST_A}.tif").exists()
        assert not (crosswalk_path / f"{DST_A}__{DST_A}.parquet").exists()

    def test_to_defaults_to_the_registry_s_vector_grids(self, vector_src, destinations, crosswalk_path):
        # A raster grid can have a .fgb -- `veg_grid_cell_v001` does in the real data -- so
        # the default destination set is filtered by grid type, not by what's on disk.
        write_grid_geometry(geometry_path := destinations, RASTER_SRC, _covering_polygon(300))
        assert (geometry_path / f"{RASTER_SRC}.fgb").exists()

        result = runner.invoke(cli, ["crosswalk", "--from", VECTOR_SRC])

        assert result.exit_code == 0, result.output
        assert (crosswalk_path / f"{VECTOR_SRC}__{DST_A}.parquet").exists()
        assert (crosswalk_path / f"{VECTOR_SRC}__{DST_B}.parquet").exists()
        assert not (crosswalk_path / f"{VECTOR_SRC}__{RASTER_SRC}.parquet").exists()

    def test_defaults_on_both_sides_need_no_options_at_all(
        self, small_registry, vector_src, destinations, crosswalk_path
    ):
        result = runner.invoke(cli, ["crosswalk"])

        assert result.exit_code == 0, result.output
        assert (crosswalk_path / f"{VECTOR_SRC}__{DST_A}.parquet").exists()
        assert (crosswalk_path / f"{RASTER_SRC}__{DST_B}.tif").exists()


class TestMissingGeometry:
    def test_a_vector_grid_without_geometry_is_skipped_with_a_warning(
        self, small_registry, vector_src, destinations, crosswalk_path
    ):
        result = runner.invoke(cli, ["crosswalk", "--to", DST_A])

        assert result.exit_code == 0, result.output
        assert f"WARNING: skipping source grid {DST_C!r}: no geometry file at" in result.output
        assert not (crosswalk_path / f"{DST_C}__{DST_A}.parquet").exists()

    def test_a_raster_grid_without_geometry_is_still_a_usable_source(self, destinations, crosswalk_path):
        """A raster source reads only the destination's geometry -- its own cells come from
        the bounds/width/height in grids.csv, so it needs no .fgb of its own."""

        result = runner.invoke(cli, ["crosswalk", "--from", RASTER_SRC, "--to", DST_A])

        assert result.exit_code == 0, result.output
        assert not (destinations / f"{RASTER_SRC}.fgb").exists()
        assert (crosswalk_path / f"{RASTER_SRC}__{DST_A}.tif").exists()

    def test_a_destination_without_geometry_is_skipped_with_a_warning(
        self, vector_src, destinations, crosswalk_path
    ):
        result = runner.invoke(cli, ["crosswalk", "--from", VECTOR_SRC, "--to", DST_A, "--to", DST_C])

        assert result.exit_code == 0, result.output
        assert f"WARNING: skipping destination grid {DST_C!r}: no geometry file at" in result.output
        assert (crosswalk_path / f"{VECTOR_SRC}__{DST_A}.parquet").exists()
        assert not (crosswalk_path / f"{VECTOR_SRC}__{DST_C}.parquet").exists()

    def test_no_usable_sources_is_a_usage_error(self, geometry_path, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "--from", VECTOR_SRC, "--to", DST_A])

        assert result.exit_code != 0
        assert "No usable source grids" in result.output

    def test_no_usable_destinations_is_a_usage_error(self, geometry_path, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "--from", RASTER_SRC, "--to", DST_A])

        assert result.exit_code != 0
        assert "No usable destination grids" in result.output


class TestValidation:
    def test_unknown_from_grid_is_rejected(self, destinations, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "--from", "not_a_real_grid", "--to", DST_A])

        assert result.exit_code != 0
        assert "Unknown grid" in result.output

    def test_unknown_to_grid_is_rejected(self, vector_src, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "--from", VECTOR_SRC, "--to", "not_a_real_grid"])

        assert result.exit_code != 0
        assert "Unknown grid" in result.output

    def test_raster_to_grid_is_rejected(self, vector_src, destinations, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "--from", VECTOR_SRC, "--to", BIG_RASTER])

        assert result.exit_code != 0
        assert f"{BIG_RASTER!r} is a 'raster' grid, expected a 'vector' grid" in result.output


class TestIgnoredOptions:
    """Each source kind ignores the other kind's tuning option.

    Both tests pass a value that would be unusable if the option were consulted, so a run
    that succeeds proves the option was never reached rather than merely tolerated.
    """

    def test_block_size_is_ignored_for_a_vector_source(self, vector_src, destinations, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "--from", VECTOR_SRC, "--to", DST_A, "--block-size=-1"])

        assert result.exit_code == 0, result.output
        assert _row_count(crosswalk_path / f"{VECTOR_SRC}__{DST_A}.parquet") == 2

    def test_batch_size_is_ignored_for_a_raster_source(self, destinations, crosswalk_path):
        result = runner.invoke(cli, ["crosswalk", "--from", RASTER_SRC, "--to", DST_A, "--batch-size=-1"])

        assert result.exit_code == 0, result.output
        assert _zone_ids(crosswalk_path / f"{RASTER_SRC}__{DST_A}.tif") == "100"


class TestOverwrite:
    def test_existing_output_is_skipped(self, vector_src, destinations, crosswalk_path):
        first = runner.invoke(cli, ["crosswalk", "--from", VECTOR_SRC, "--to", DST_A])
        assert first.exit_code == 0, first.output

        second = runner.invoke(cli, ["crosswalk", "--from", VECTOR_SRC, "--to", DST_A])
        assert second.exit_code == 0, second.output
        assert "SKIP" in second.output
        assert "SUMMARY: 0 written, 1 skipped, 0 failed" in second.output

    def test_an_all_skipped_source_never_builds_a_centroid_index(self, vector_src, destinations, crosswalk_path):
        first = runner.invoke(cli, ["crosswalk", "--from", VECTOR_SRC, "--to", DST_A])
        assert first.exit_code == 0, first.output
        assert "Indexing" in first.output

        # The index costs hundreds of MB on the larger grids, so it must not be built for a
        # source with nothing left to write.
        second = runner.invoke(cli, ["crosswalk", "--from", VECTOR_SRC, "--to", DST_A])
        assert second.exit_code == 0, second.output
        assert "Indexing" not in second.output

    def test_overwrite_rewrites_a_vector_crosswalk(self, geometry_path, crosswalk_path):
        write_grid_geometry(geometry_path, VECTOR_SRC, _src_features())
        write_grid_geometry(geometry_path, DST_A, _covering_polygon(100))
        out = crosswalk_path / f"{VECTOR_SRC}__{DST_A}.parquet"

        first = runner.invoke(cli, ["crosswalk", "--from", VECTOR_SRC, "--to", DST_A])
        assert first.exit_code == 0, first.output
        assert _row_count(out) == 2

        # Rewrite the source with an extra feature; without --overwrite the stale 2-row file
        # would stick around, so seeing 3 rows proves the rerun actually happened.
        write_grid_geometry(geometry_path, VECTOR_SRC, _src_features(extra_id=3))
        second = runner.invoke(cli, ["crosswalk", "--from", VECTOR_SRC, "--to", DST_A, "--overwrite"])
        assert second.exit_code == 0, second.output
        assert _row_count(out) == 3

    def test_overwrite_rewrites_a_raster_crosswalk(self, geometry_path, crosswalk_path):
        minx, miny, maxx, maxy = GRIDS[RASTER_SRC].bounds
        midx = (minx + maxx) / 2
        out = crosswalk_path / f"{RASTER_SRC}__{DST_A}.tif"

        write_grid_geometry(geometry_path, DST_A, _covering_polygon(1))
        first = runner.invoke(cli, ["crosswalk", "--from", RASTER_SRC, "--to", DST_A])
        assert first.exit_code == 0, first.output
        assert _zone_ids(out) == "1"

        write_grid_geometry(
            geometry_path, DST_A, [(1, box(minx, miny, midx, maxy)), (2, box(midx, miny, maxx, maxy))]
        )
        second = runner.invoke(cli, ["crosswalk", "--from", RASTER_SRC, "--to", DST_A, "--overwrite"])
        assert second.exit_code == 0, second.output
        assert _zone_ids(out) == "1,2"


class TestFailures:
    def test_one_failing_pair_does_not_abort_the_rest(self, vector_src, destinations, crosswalk_path):
        # A file that exists (so it isn't skipped as missing) but isn't FlatGeobuf, which is
        # what makes it fail at open time rather than at resolution time.
        (destinations / f"{DST_C}.fgb").write_bytes(b"not a flatgeobuf")

        result = runner.invoke(cli, ["crosswalk", "--from", VECTOR_SRC, "--to", DST_A, "--to", DST_C])

        assert result.exit_code == 1
        assert (crosswalk_path / f"{VECTOR_SRC}__{DST_A}.parquet").exists()
        assert "SUMMARY: 1 written, 0 skipped, 1 failed" in result.output
        assert f"FAILED: 1 of 2 crosswalk(s): {VECTOR_SRC} -> {DST_C}" in result.output

    def test_an_unreadable_source_fails_all_of_its_pairs(self, destinations, crosswalk_path):
        (destinations / f"{VECTOR_SRC}.fgb").write_bytes(b"not a flatgeobuf")

        result = runner.invoke(cli, ["crosswalk", "--from", VECTOR_SRC, "--to", DST_A, "--to", DST_B])

        assert result.exit_code == 1
        assert f"FAIL: {VECTOR_SRC}:" in result.output
        assert "SUMMARY: 0 written, 0 skipped, 2 failed" in result.output

    def test_a_failing_source_does_not_abort_the_others(self, vector_src, destinations, crosswalk_path):
        (destinations / f"{VECTOR_SRC_B}.fgb").write_bytes(b"not a flatgeobuf")

        result = runner.invoke(
            cli, ["crosswalk", "--from", VECTOR_SRC_B, "--from", VECTOR_SRC, "--to", DST_A]
        )

        assert result.exit_code == 1
        assert (crosswalk_path / f"{VECTOR_SRC}__{DST_A}.parquet").exists()
        assert "SUMMARY: 1 written, 0 skipped, 1 failed" in result.output
