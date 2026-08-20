"""Tests for `cpra.mp.data.crosswalks`.

Two properties do most of the work here (see the module docstring in
`cpra/mp/data/crosswalks.py` for the full rationale):

* Windowed rasterization must be pixel-identical to a whole-array `rasterize` call --
  that's the entire correctness argument for writing raster crosswalks window-by-window
  instead of loading the whole grid into memory.
* A window whose bbox matches zero destination features must come back all-NODATA, never
  zone 0 -- this pins a live FlatGeobuf-specific GDAL bug in `GetArrowStream` (see
  `_valid_geometries`'s docstring for the reproduction details).

Most tests use small synthetic grids registered into `cpra.mp.data.grids.GRIDS` (via the
`synthetic_grids` fixture below) rather than the real production grids, so extents, block
alignment, and ID ranges can be constructed exactly instead of hoped for in real data.
`crosswalks.py` itself never reads `cpra.mp.data.config`, so `grids_path`/`crosswalks_path`
are just plain `tmp_path` subdirectories here -- the `geometry_path`/`crosswalk_path`
fixtures are used anyway for consistency with the rest of the suite.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rasterio as rio
import shapely
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box

from cpra.mp.data import crosswalks
from cpra.mp.data.crosswalks import (
    GDAL_CACHE_BYTES,
    MAX_SAFE_ZONE_ID,
    NODATA,
    PAIR_CHUNK,
    ZONE_DTYPES,
    get_zone_dtype,
    read_geometry_tree,
    write_raster_crosswalk,
    write_vector_crosswalk,
)
from cpra.mp.data.grids import GRIDS, RasterGrid, VectorGrid
from tests.factories import write_grid_geometry

SYNTHETIC_CRS = "epsg:26915"

# A small raster grid (50 x 37 px, 1 map unit per pixel) used everywhere the raster tests
# need exact control over extent and block-size alignment. Real raster grids (e.g.
# `veg_grid_cell_v001`, 365 x 1052) work fine functionally but are needlessly slow and
# make it hard to choose block sizes that deliberately don't divide the grid evenly.
RASTER_NAME = "cw_test_raster"
RASTER_BOUNDS = (0, 0, 50, 37)
RASTER_HEIGHT = 37
RASTER_WIDTH = 50

VECTOR_SRC_NAME = "cw_test_vector_src"
VECTOR_DST_NAME = "cw_test_vector_dst"
VECTOR_DST2_NAME = "cw_test_vector_dst2"


@pytest.fixture(autouse=True)
def synthetic_grids(monkeypatch):
    """Register small synthetic grids used throughout this file.

    `write_grid_geometry` (tests/factories.py) looks up `GRIDS[grid_name].crs` and
    `write_raster_crosswalk` looks up `GRIDS[raster_grid_name]` for bounds/height/width/
    crs, so every grid name used below must resolve through the real `GRIDS` registry.
    """
    extra = {
        VECTOR_SRC_NAME: VectorGrid(name=VECTOR_SRC_NAME, label="test src", crs=SYNTHETIC_CRS),
        VECTOR_DST_NAME: VectorGrid(name=VECTOR_DST_NAME, label="test dst", crs=SYNTHETIC_CRS),
        VECTOR_DST2_NAME: VectorGrid(name=VECTOR_DST2_NAME, label="test dst 2", crs=SYNTHETIC_CRS),
        RASTER_NAME: RasterGrid(
            name=RASTER_NAME,
            label="test raster",
            bounds=RASTER_BOUNDS,
            height=RASTER_HEIGHT,
            width=RASTER_WIDTH,
            crs=SYNTHETIC_CRS,
        ),
    }
    for name, grid in extra.items():
        monkeypatch.setitem(GRIDS, name, grid)
    return extra


def _tiles(x_edges, y_edges, start_id):
    """Non-overlapping rectangular tiles covering the box spanned by the given edges.

    Non-overlapping is deliberate: `rasterize` paints later shapes over earlier ones in
    the order it's given, and a windowed bbox read is not guaranteed to preserve the same
    feature order as a whole-file read (FlatGeobuf's packed R-tree orders by spatial
    index, not necessarily by insertion order). Overlapping polygons would make the
    "pixel-identical" comparison depend on an ordering guarantee the module doesn't make
    or need, since real grids' cells don't overlap either.
    """
    features = []
    zone_id = start_id
    for xi in range(len(x_edges) - 1):
        for yi in range(len(y_edges) - 1):
            features.append(
                (np.int32(zone_id), box(x_edges[xi], y_edges[yi], x_edges[xi + 1], y_edges[yi + 1]))
            )
            zone_id += 1
    return features


# Six tiles covering the full 50x37 extent, with boundaries at x=13,29 and y=21 -- none of
# which are multiples of the block sizes exercised below (16, 32), so every block size
# actually cuts through tile boundaries instead of conveniently aligning with them.
FULL_TILES = _tiles([0, 13, 29, 50], [0, 21, 37], start_id=301)
FULL_TILE_IDS = ",".join(str(t[0]) for t in FULL_TILES)


class TestConstants:
    def test_nodata(self):
        assert NODATA == -1

    def test_zone_dtypes_order(self):
        # Narrowest first: get_zone_dtype relies on iterating this in ascending width.
        assert ZONE_DTYPES == ["int16", "int32", "int64"]

    def test_max_safe_zone_id(self):
        assert MAX_SAFE_ZONE_ID == 2**53

    def test_gdal_cache_bytes(self):
        assert GDAL_CACHE_BYTES == 64 * 1024 * 1024


class TestGetZoneDtype:
    def test_selects_int16_for_small_ids(self):
        assert get_zone_dtype([1, 100, 32767]) == "int16"

    def test_int16_upper_boundary(self):
        assert get_zone_dtype([32767]) == "int16"
        assert get_zone_dtype([32768]) == "int32"

    def test_int16_lower_boundary(self):
        assert get_zone_dtype([-32768]) == "int16"
        assert get_zone_dtype([-32769]) == "int32"

    def test_int32_boundary(self):
        assert get_zone_dtype([2147483647]) == "int32"
        assert get_zone_dtype([2147483648]) == "int64"

    def test_twelve_digit_fips_ids_select_int64(self):
        # Block-group / tract style FIPS codes: 12 digits, well past int32 but nowhere
        # near MAX_SAFE_ZONE_ID.
        assert get_zone_dtype([220510001001, 220510001002]) == "int64"

    @pytest.mark.parametrize("dst_ids", [[1], [-5, 5], [220510001001], [32767], [2147483647]])
    def test_nodata_is_always_representable(self, dst_ids):
        """NODATA (-1) is folded into the range via `min(NODATA, min(dst_ids))`, so
        whatever dtype comes back must be able to hold -1 as well as every dst id."""
        dtype = get_zone_dtype(dst_ids)
        info = np.iinfo(dtype)
        assert info.min <= NODATA
        assert info.min <= min(dst_ids) and max(dst_ids) <= info.max

    def test_max_safe_zone_id_guard_raises(self):
        """rasterize() burns zone values through a double, which stops representing
        integers exactly past 2**53; ids at or beyond that must be rejected rather than
        silently rounded to a neighboring value."""
        with pytest.raises(ValueError, match="exceed what rasterize"):
            get_zone_dtype([MAX_SAFE_ZONE_ID])

    def test_max_safe_zone_id_guard_boundary_is_exclusive(self):
        assert get_zone_dtype([MAX_SAFE_ZONE_ID - 1]) == "int64"

    def test_max_safe_zone_id_guard_applies_to_negative_ids_too(self):
        with pytest.raises(ValueError, match="exceed what rasterize"):
            get_zone_dtype([-MAX_SAFE_ZONE_ID])

    def test_ids_that_fit_nothing_raise(self, monkeypatch):
        """Ids too wide for any dtype in ZONE_DTYPES must raise ValueError.

        NOTE: under the real MAX_SAFE_ZONE_ID (2**53 ~= 9.007e15), this branch is
        unreachable through the public API: MAX_SAFE_ZONE_ID is far smaller than
        int64's range (~9.223e18), so anything that survives the MAX_SAFE_ZONE_ID guard
        always fits int64, and the loop's final `raise` can never execute. This is
        flagged separately as a dead-code observation (not a correctness bug -- it never
        produces a wrong result, the guard just fires first). To actually exercise the
        fallback line for coverage, MAX_SAFE_ZONE_ID is patched larger than int64's max
        here so an id can clear the guard while still not fitting any ZONE_DTYPES entry.
        """
        monkeypatch.setattr(crosswalks, "MAX_SAFE_ZONE_ID", 10**30)
        too_big_for_int64 = np.iinfo("int64").max + 1
        with pytest.raises(ValueError, match="do not fit"):
            get_zone_dtype([too_big_for_int64])


class TestReadGeometryTree:
    def test_ids_areas_and_tree_align(self, geometry_path):
        features = [
            (1, box(0, 0, 2, 2)),  # area 4
            (2, box(10, 10, 14, 14)),  # area 16
        ]
        path = write_grid_geometry(geometry_path, VECTOR_SRC_NAME, features)

        tree, ids, areas = read_geometry_tree(path)

        assert sorted(ids.tolist()) == [1, 2]
        assert dict(zip(ids.tolist(), areas.tolist())) == {1: 4.0, 2: 16.0}
        # The tree holds the polygons themselves, not their centroids: a point well away
        # from either centroid but inside one polygon still matches it. Area weighting
        # needs the real geometry, so this is what distinguishes the tree from the centroid
        # index this replaced.
        corner = shapely.Point(10.5, 10.5)
        assert list(tree.query(corner, predicate="intersects")) == [int(np.where(ids == 2)[0][0])]


def _rows(out_path):
    """`{(src_id, dst_id): scale_factor}` for a crosswalk table."""
    return {(row["src_id"], row["dst_id"]): row["scale_factor"] for row in pq.read_table(out_path).to_pylist()}


class TestWriteVectorCrosswalk:
    def test_fully_contained_cells_and_absent_outside_geography(self, geometry_path, crosswalk_path):
        """A source cell wholly inside one destination gets scale_factor 1.0, and one
        overlapping nothing is simply absent -- this is an inner join, so it must not
        appear as a NODATA row or raise."""
        src_features = [
            (1, box(0, 0, 2, 2)),  # wholly inside dst 10
            (2, box(10, 10, 12, 12)),  # wholly inside dst 20
            (3, box(100, 100, 102, 102)),  # overlaps nothing
        ]
        dst_features = [
            (10, box(0, 0, 5, 5)),
            (20, box(5, 5, 15, 15)),
        ]
        write_grid_geometry(geometry_path, VECTOR_SRC_NAME, src_features)
        write_grid_geometry(geometry_path, VECTOR_DST_NAME, dst_features)

        out_path, row_count = write_vector_crosswalk(
            VECTOR_SRC_NAME, VECTOR_DST_NAME, geometry_path, crosswalk_path
        )

        assert row_count == 2
        assert _rows(out_path) == {(1, 10): 1.0, (2, 20): 1.0}

    def test_a_straddling_cell_is_split_by_area(self, geometry_path, crosswalk_path):
        """The core of area weighting: a source cell overlapping two destinations produces
        a row for each, weighted by how much of the source cell's area each one takes.

        Under the centroid join this replaced, cell 1 went whole to whichever destination
        held the point (5, 5) and cell 2 went whole to dst 1 -- the 25/75 split was lost
        entirely.
        """
        src_features = [
            (1, box(0, 0, 10, 10)),  # half in each destination
            (2, box(0, 0, 4, 10)),  # 1 unit of 4 in dst 2, 3 in dst 1
        ]
        dst_features = [
            (1, box(0, 0, 3, 10)),
            (2, box(3, 0, 10, 10)),
        ]
        write_grid_geometry(geometry_path, VECTOR_SRC_NAME, src_features)
        write_grid_geometry(geometry_path, VECTOR_DST_NAME, dst_features)

        out_path, row_count = write_vector_crosswalk(
            VECTOR_SRC_NAME, VECTOR_DST_NAME, geometry_path, crosswalk_path
        )

        assert row_count == 4
        assert _rows(out_path) == pytest.approx({(1, 1): 0.3, (1, 2): 0.7, (2, 1): 0.75, (2, 2): 0.25})

    def test_scale_factors_of_a_covered_cell_sum_to_one(self, geometry_path, crosswalk_path):
        """The invariant `api/dataset.py` depends on: summarizing a source cell that the
        destination grid covers completely must neither lose nor duplicate its value, which
        holds exactly when its scale factors sum to 1."""
        src_features = [(1, box(0, 0, 30, 30))]
        dst_features = _tiles([0, 7, 19, 30], [0, 11, 30], start_id=1)
        write_grid_geometry(geometry_path, VECTOR_SRC_NAME, src_features)
        write_grid_geometry(geometry_path, VECTOR_DST_NAME, dst_features)

        out_path, row_count = write_vector_crosswalk(
            VECTOR_SRC_NAME, VECTOR_DST_NAME, geometry_path, crosswalk_path
        )

        assert row_count == len(dst_features)
        assert sum(_rows(out_path).values()) == pytest.approx(1.0)

    def test_a_partly_covered_cell_sums_to_less_than_one(self, geometry_path, crosswalk_path):
        """The complement of the invariant above: where the destination grid doesn't reach,
        the scale factors deliberately don't add up, because the uncovered part of the
        source cell belongs to no destination."""
        src_features = [(1, box(0, 0, 10, 10))]  # only the left quarter is covered
        dst_features = [(1, box(0, 0, 2.5, 10))]
        write_grid_geometry(geometry_path, VECTOR_SRC_NAME, src_features)
        write_grid_geometry(geometry_path, VECTOR_DST_NAME, dst_features)

        out_path, _row_count = write_vector_crosswalk(
            VECTOR_SRC_NAME, VECTOR_DST_NAME, geometry_path, crosswalk_path
        )

        assert _rows(out_path) == pytest.approx({(1, 1): 0.25})

    def test_boundary_only_contact_is_not_an_overlap(self, geometry_path, crosswalk_path):
        """`intersects` is true for polygons sharing only an edge or a corner, so the
        candidate query alone would emit those pairs with scale_factor 0. They contribute
        nothing to a summary and are dropped instead."""
        src_features = [(1, box(0, 0, 5, 5))]
        dst_features = [
            (1, box(0, 0, 5, 5)),  # identical -- the one real overlap
            (2, box(5, 0, 10, 5)),  # shares the edge x=5
            (3, box(5, 5, 10, 10)),  # shares only the corner (5, 5)
        ]
        write_grid_geometry(geometry_path, VECTOR_SRC_NAME, src_features)
        write_grid_geometry(geometry_path, VECTOR_DST_NAME, dst_features)

        out_path, row_count = write_vector_crosswalk(
            VECTOR_SRC_NAME, VECTOR_DST_NAME, geometry_path, crosswalk_path
        )

        assert row_count == 1
        assert _rows(out_path) == {(1, 1): 1.0}

    def test_a_zero_area_source_cell_is_dropped_not_infinite(self, geometry_path, crosswalk_path):
        """A degenerate source cell has no area to apportion, so dividing by it would write
        inf (or nan) into scale_factor and silently poison every summary using it."""
        src_features = [
            (1, box(0, 0, 0, 5)),  # collapsed to a line: area 0
            (2, box(0, 0, 5, 5)),
        ]
        dst_features = [(10, box(0, 0, 5, 5))]
        write_grid_geometry(geometry_path, VECTOR_SRC_NAME, src_features)
        write_grid_geometry(geometry_path, VECTOR_DST_NAME, dst_features)

        out_path, row_count = write_vector_crosswalk(
            VECTOR_SRC_NAME, VECTOR_DST_NAME, geometry_path, crosswalk_path
        )

        assert row_count == 1
        assert _rows(out_path) == {(2, 10): 1.0}

    def test_pair_chunking_does_not_change_the_result(self, geometry_path, crosswalk_path, tmp_path):
        """`PAIR_CHUNK` bounds how many intersections are computed at once; slicing at a
        size smaller than the pair count must produce the same table, just across more
        record batches."""
        src_features = _tiles(list(range(0, 11)), list(range(0, 11)), start_id=1)  # 100 cells
        dst_features = [(1, box(0, 0, 5.5, 11)), (2, box(5.5, 0, 11, 11))]
        write_grid_geometry(geometry_path, VECTOR_SRC_NAME, src_features)
        write_grid_geometry(geometry_path, VECTOR_DST_NAME, dst_features)

        whole_dir = tmp_path / "whole"
        chunked_dir = tmp_path / "chunked"
        whole_dir.mkdir()
        chunked_dir.mkdir()

        whole_path, whole_count = write_vector_crosswalk(
            VECTOR_SRC_NAME, VECTOR_DST_NAME, geometry_path, whole_dir
        )
        assert whole_count > 7, "test needs more pairs than the chunk size below"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(crosswalks, "PAIR_CHUNK", 7)
            chunked_path, chunked_count = write_vector_crosswalk(
                VECTOR_SRC_NAME, VECTOR_DST_NAME, geometry_path, chunked_dir
            )

        assert PAIR_CHUNK > whole_count, "the unchunked run must fit in one slice"
        assert chunked_count == whole_count
        assert _rows(chunked_path) == _rows(whole_path)
        assert pq.ParquetFile(chunked_path).metadata.num_row_groups > pq.ParquetFile(whole_path).metadata.num_row_groups

    def test_schema_column_names_and_types_for_small_ids(self, geometry_path, crosswalk_path):
        """api/dataset.py and api/crosswalk.py both hard-depend on these exact column
        names; the id columns must match tests/factories.py's canonical schema."""
        write_grid_geometry(
            geometry_path, VECTOR_SRC_NAME, [(np.int32(1), box(0, 0, 2, 2))]
        )
        write_grid_geometry(
            geometry_path, VECTOR_DST_NAME, [(np.int32(10), box(0, 0, 5, 5))]
        )

        out_path, _ = write_vector_crosswalk(VECTOR_SRC_NAME, VECTOR_DST_NAME, geometry_path, crosswalk_path)

        schema = pq.ParquetFile(out_path).schema_arrow
        assert schema.names == ["src_id", "dst_id", "scale_factor"]
        assert schema.field("src_id").type == pa.int32()
        assert schema.field("dst_id").type == pa.int32()
        assert schema.field("scale_factor").type == pa.float64()

    def test_large_source_ids_widen_to_int64(self, geometry_path, crosswalk_path):
        """FIPS-scale source ids (int64 at the OGR layer) must widen src_id to int64,
        independent of the destination grid's own id width."""
        write_grid_geometry(
            geometry_path, VECTOR_SRC_NAME, [(220510001001, box(0, 0, 2, 2))]
        )
        write_grid_geometry(
            geometry_path, VECTOR_DST_NAME, [(np.int32(10), box(0, 0, 5, 5))]
        )

        out_path, row_count = write_vector_crosswalk(
            VECTOR_SRC_NAME, VECTOR_DST_NAME, geometry_path, crosswalk_path
        )

        assert row_count == 1
        schema = pq.ParquetFile(out_path).schema_arrow
        assert schema.field("src_id").type == pa.int64()
        assert schema.field("dst_id").type == pa.int32()
        table = pq.read_table(out_path)
        assert table.column("src_id").to_pylist() == [220510001001]

    def test_large_destination_ids_widen_to_int64(self, geometry_path, crosswalk_path):
        write_grid_geometry(
            geometry_path, VECTOR_SRC_NAME, [(np.int32(1), box(0, 0, 2, 2))]
        )
        write_grid_geometry(
            geometry_path, VECTOR_DST_NAME, [(220510001001, box(0, 0, 5, 5))]
        )

        out_path, _row_count = write_vector_crosswalk(
            VECTOR_SRC_NAME, VECTOR_DST_NAME, geometry_path, crosswalk_path
        )

        schema = pq.ParquetFile(out_path).schema_arrow
        assert schema.field("src_id").type == pa.int32()
        assert schema.field("dst_id").type == pa.int64()

    def test_compression_codec(self, geometry_path, crosswalk_path):
        write_grid_geometry(geometry_path, VECTOR_SRC_NAME, [(1, box(0, 0, 2, 2))])
        write_grid_geometry(geometry_path, VECTOR_DST_NAME, [(10, box(0, 0, 5, 5))])

        out_path, _ = write_vector_crosswalk(
            VECTOR_SRC_NAME, VECTOR_DST_NAME, geometry_path, crosswalk_path, compression="zstd"
        )

        column_meta = pq.ParquetFile(out_path).metadata.row_group(0).column(0)
        assert column_meta.compression.upper() == "ZSTD"

    def test_prebuilt_tree_matches_self_built_tree(self, geometry_path, tmp_path):
        src_features = [
            (1, box(0, 0, 2, 2)),
            (2, box(10, 10, 12, 12)),
            (3, box(100, 100, 102, 102)),
        ]
        dst_features = [(10, box(0, 0, 5, 5)), (20, box(5, 5, 15, 15))]
        write_grid_geometry(geometry_path, VECTOR_SRC_NAME, src_features)
        write_grid_geometry(geometry_path, VECTOR_DST_NAME, dst_features)

        without_tree_dir = tmp_path / "without_tree"
        with_tree_dir = tmp_path / "with_tree"
        without_tree_dir.mkdir()
        with_tree_dir.mkdir()

        out_a, count_a = write_vector_crosswalk(
            VECTOR_SRC_NAME, VECTOR_DST_NAME, geometry_path, without_tree_dir
        )

        prebuilt_tree = read_geometry_tree(geometry_path / f"{VECTOR_SRC_NAME}.fgb")
        out_b, count_b = write_vector_crosswalk(
            VECTOR_SRC_NAME, VECTOR_DST_NAME, geometry_path, with_tree_dir, tree=prebuilt_tree
        )

        assert count_a == count_b
        table_a = pq.read_table(out_a).sort_by([("src_id", "ascending"), ("dst_id", "ascending")])
        table_b = pq.read_table(out_b).sort_by([("src_id", "ascending"), ("dst_id", "ascending")])
        assert table_a.equals(table_b)


def _reference_rasterize(raster_grid, tiles, dtype):
    transform = from_bounds(*raster_grid.bounds, width=raster_grid.width, height=raster_grid.height)
    ids = [t[0] for t in tiles]
    geoms = [t[1] for t in tiles]
    ref = np.full((raster_grid.height, raster_grid.width), NODATA, dtype=dtype)
    rasterize(list(zip(geoms, ids)), out=ref, transform=transform)
    return ref


class TestWriteRasterCrosswalkPixelIdentity:
    """Test A: windowed rasterization must be pixel-identical to a whole-array
    `rasterize` call using the same geometries and transform. This is the correctness
    proof for writing raster crosswalks window-by-window instead of loading the whole
    destination grid and raster array into memory at once (see module docstring)."""

    @pytest.mark.parametrize("block_size", [16, 32])
    def test_matches_whole_array_rasterize(self, geometry_path, crosswalk_path, block_size):
        # GTiff block sizes must be multiples of 16, so 16 and 32 are the smallest two
        # valid choices. Neither divides the grid evenly in either dimension (50/16,
        # 37/16, 50/32, 37/32 are all fractional), so both exercise partial edge windows,
        # not just interior ones.
        write_grid_geometry(geometry_path, VECTOR_DST_NAME, FULL_TILES)
        raster_grid = GRIDS[RASTER_NAME]

        out_path, window_count = write_raster_crosswalk(
            RASTER_NAME, VECTOR_DST_NAME, geometry_path, crosswalk_path, block_size=block_size
        )

        assert window_count > 1
        with rio.open(out_path) as f:
            windowed = f.read(1)
            dtype = f.dtypes[0]

        reference = _reference_rasterize(raster_grid, FULL_TILES, dtype)
        assert np.array_equal(windowed, reference)


class TestWriteRasterCrosswalkEmptyBboxGuard:
    """Test B: pins a live FlatGeobuf-specific GDAL bug (reproduced on GDAL 3.11.4 /
    pyogrio 0.12.1 AND GDAL 3.12.4 / pyogrio 0.13.0 -- confirmed not fixed by upgrading).

    In GDAL's FlatGeobuf `GetArrowStream`, a `bbox` read matching zero features doesn't
    return an empty stream: it returns exactly one bogus batch of `batch_size` rows with
    empty-bytes WKB and `id=0`, and `null_count` is 0 for that batch (so a null check
    doesn't catch it -- only a zero-length-bytes check does, which is what
    `_valid_geometries` does). GPKG and Shapefile are unaffected; every grid this module
    reads is FlatGeobuf. Unguarded, a window whose bbox matches nothing would be burned
    with zone 0 instead of staying NODATA -- silent corruption, not a crash, since 0 looks
    like a legitimate zone id. If a future GDAL version fixes this and someone removes the
    guard in `_valid_geometries`, this test should start failing and point back here.
    """

    def test_windows_outside_destination_coverage_are_nodata_not_zero(self, geometry_path, crosswalk_path):
        # Destination polygon covers only x in [0, 15) of the raster's [0, 50) extent, so
        # windows entirely to the right of x=15 have zero intersecting features.
        partial_tiles = [(np.int32(401), box(0, 0, 15, RASTER_HEIGHT))]
        write_grid_geometry(geometry_path, VECTOR_DST_NAME, partial_tiles)

        # block_size=16 (GTiff block sizes must be multiples of 16): column windows
        # start at x=0,16,32,48. Every window starting at x>=16 is entirely right of the
        # [0, 15) coverage strip, so it has zero intersecting features.
        out_path, _window_count = write_raster_crosswalk(
            RASTER_NAME, VECTOR_DST_NAME, geometry_path, crosswalk_path, block_size=16
        )

        with rio.open(out_path) as f:
            arr = f.read(1)

        # Zone 0 must never appear -- it's the bug's signature, not a real destination id.
        assert not np.any(arr == 0)
        # Columns >= 16 (x >= 16) are entirely outside the covered [0, 15) strip and fall
        # in windows with zero intersecting features: must be all-NODATA.
        assert np.all(arr[:, 16:] == NODATA)
        # Sanity check the covered strip actually got real data, so this isn't trivially
        # passing because nothing was ever rasterized at all.
        assert np.all(arr[:, :15] == 401)


class TestWriteRasterCrosswalkOutputProfile:
    def test_profile(self, geometry_path, crosswalk_path):
        write_grid_geometry(geometry_path, VECTOR_DST_NAME, FULL_TILES)
        raster_grid = GRIDS[RASTER_NAME]

        out_path, window_count = write_raster_crosswalk(
            RASTER_NAME, VECTOR_DST_NAME, geometry_path, crosswalk_path
        )

        # Default block_size (512) is bigger than the whole 50x37 grid: a single window.
        assert window_count == 1

        with rio.open(out_path) as f:
            assert f.tags()["dst_ids"] == FULL_TILE_IDS
            assert f.nodata == NODATA
            assert f.crs.to_epsg() == int(raster_grid.crs.split(":")[1])
            assert f.dtypes[0] == get_zone_dtype([t[0] for t in FULL_TILES])
            assert f.profile["tiled"] is True
            assert f.driver == "GTiff"
            # rio.open(path, "w", driver="COG") would buffer the whole raster in an
            # in-RAM MEM dataset; this module instead writes a tiled GTiff and only
            # turns it into a COG via rasterio.shutil.copy's CreateCopy at the end, so
            # the *file itself* still has to actually be a COG at the end.
            assert f.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"

    def test_dst_ids_tag_parses_as_sorted_ids(self, geometry_path, crosswalk_path):
        """api/dataset.py does crosswalk.tags()["dst_ids"].split(",") and KeyErrors
        without this tag -- and relies on being able to sort the parsed ids."""
        # Deliberately out of numeric order in the source file to prove the tag is
        # actually sorted, not just passed through in file order.
        features = [(np.int32(50), box(0, 0, 25, 37)), (np.int32(5), box(25, 0, 50, 37))]
        write_grid_geometry(geometry_path, VECTOR_DST_NAME, features)

        out_path, _ = write_raster_crosswalk(RASTER_NAME, VECTOR_DST_NAME, geometry_path, crosswalk_path)

        with rio.open(out_path) as f:
            dst_ids = sorted(int(v) for v in f.tags()["dst_ids"].split(","))
        assert dst_ids == [5, 50]

    def test_window_count_matches_block_tiling(self, geometry_path, crosswalk_path):
        write_grid_geometry(geometry_path, VECTOR_DST_NAME, FULL_TILES)

        _out_path, window_count = write_raster_crosswalk(
            RASTER_NAME, VECTOR_DST_NAME, geometry_path, crosswalk_path, block_size=16
        )

        # 50x37 tiled at 16x16: ceil(37/16)=3 row bands, ceil(50/16)=4 col bands.
        assert window_count == 3 * 4

    def test_dst_ids_tag_lost_in_copy_fails_loudly(self, geometry_path, crosswalk_path, monkeypatch):
        """A crosswalk whose `dst_ids` tag didn't survive CreateCopy must raise, not ship.

        `api/dataset.py` does `crosswalk.tags()["dst_ids"]` and KeyErrors (500s) without
        the tag, so a COG missing it is unusable. There is deliberately no repair-in-place
        fallback: GDAL refuses a plain "r+" open of a COG-layout GeoTIFF
        (CPLE_AppDefinedError, directing callers to IGNORE_COG_LAYOUT_BREAK=YES, which
        would break the very layout that makes the file a COG). So the only honest
        outcomes are "tag present" or "loud failure", and this pins the latter.

        The tag does survive on this build, so reaching the check requires forcing a
        mismatch. Corrupting the *intermediate plain GTiff* before the real copy runs is
        what makes `out_path` a genuinely pristine COG carrying a wrong tag -- a plain
        GTiff has no COG-layout restriction, so the setup itself mutates nothing about the
        output. (Corrupting the finished COG instead would break its layout as a side
        effect and the check would pass for the wrong reason.)
        """
        write_grid_geometry(geometry_path, VECTOR_DST_NAME, FULL_TILES)
        real_copy = crosswalks.copy_raster

        def copy_from_a_source_with_the_wrong_tag(src, dst, **kwargs):
            # `src` here is still the intermediate plain GTiff (not yet a COG), so
            # correcting -- er, corrupting -- its tag is an ordinary "r+" open with no
            # COG-layout restriction. The resulting `dst` is then a pristine COG built
            # from a single real CreateCopy, carrying the wrong tag through untouched --
            # exactly the "tag didn't survive" scenario the fallback defends against,
            # with no prior mutation of `dst` itself.
            with rio.open(src, "r+") as corrupted_source:
                corrupted_source.update_tags(dst_ids="corrupted")
            real_copy(src, dst, **kwargs)

        monkeypatch.setattr(crosswalks, "copy_raster", copy_from_a_source_with_the_wrong_tag)

        with pytest.raises(ValueError, match="dropped the dst_ids tag"):
            write_raster_crosswalk(RASTER_NAME, VECTOR_DST_NAME, geometry_path, crosswalk_path)
