"""Tests for `cpra.mp.data.analysis.io.raster.scan_raster` -- the VRT-based raster scan the
execution layer opens a band-wide dataset through -- and
`cpra.mp.data.analysis.io.catalog.resolve_raster_bands`, the one query that resolves a
raster `InputSelection` into the ordered band list a scan needs.

Two fixture styles, deliberately different from each other:

* `resolve_raster_bands` tests go through `tests/factories.py`'s real `vct`/
  `veg_grid_cell_v001` fixtures (`write_raster_dataset`), because that function depends on
  the real `GRIDS`/`DIMENSIONS` structure -- there is no way to exercise it against a fake
  grid.
* `scan_raster` tests write their own small, synthetic GeoTIFFs directly via rasterio
  instead, because that function operates on an already-resolved `metadata_df` and never
  touches `GRIDS` at all -- a real grid's 384k-pixel raster would only make the
  hand-computed expected values below harder to check by eye for no added coverage.
  Exception: the batched-vs-unbatched test, which specifically needs a raster with more
  than one block to force more than one batch, reuses the real, COG-tiled `vct` fixture
  the same way `test_raster_source.py`'s own batching test does.
"""

import dataclasses

import numpy as np
import polars as pl
import pytest
import rasterio as rio
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from cpra.mp.data.analysis.execution.frame import (
    assert_height_preserved,
    assert_window_unchanged,
    band_column_names,
    row_major_grid_cell,
)
from cpra.mp.data.analysis.execution.plan import compile_plan
from cpra.mp.data.analysis.io import raster_io
from cpra.mp.data.analysis.io.catalog import resolve_raster_bands
from cpra.mp.data.analysis.io.raster import scan_raster
from cpra.mp.data.analysis.request.errors import NotFoundError
from cpra.mp.data.analysis.request.ops import validate_shapes
from cpra.mp.data.analysis.request.pipeline import InputSelection, parse_pipeline
from tests.factories import RASTER_KWARGS, raster_array, write_raster_dataset

# Arbitrary but fixed georeferencing -- every synthetic fixture below shares it, since a
# VRT assembled from sources of differing size/origin is not a shape these tests need to
# characterize (a real crosswalk/data pair is always co-registered -- see `io/raster.py`'s
# module docstring).
_TRANSFORM = from_origin(0, 10, 1, 1)
_CRS = "EPSG:4326"

_SELECTION_FIELDS = ("variable", "time_unit", "grid")


def _pipeline(data):
    """`parse_pipeline` then `validate_shapes` -- `compile_plan`'s own precondition."""
    pipeline = parse_pipeline(data)
    validate_shapes(pipeline)
    return pipeline


def _write_tif(path, data, nodata=None):
    """A GeoTIFF holding exactly `data`, nothing else -- no tiling, no compression.

    `data` is `(height, width)` (one band) or `(bands, height, width)`.
    """
    if data.ndim == 2:
        data = data[None, ...]
    count, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "dtype": data.dtype.name,
        "count": count,
        "height": height,
        "width": width,
        "crs": _CRS,
        "transform": _TRANSFORM,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def _metadata(rows):
    """The minimal `scan_raster` input: one `(path, band)` pair per row, in order."""
    return pl.DataFrame(
        {"path": [str(p) for p, _ in rows], "band": [b for _, b in rows]},
        schema={"path": pl.String, "band": pl.Int64},
    )


class TestBandOrderMatchesMetadataRowOrder:
    """The band-naming invariant `execution/frame.py` documents: column `b<i>` is the band
    described by metadata row `i`. Two files with different, distinguishable fill values
    make a row-order bug (e.g. a scan that sorted sources instead of preserving caller
    order) show up as a wrong VALUE, not just a wrong column count.
    """

    def test_permuting_metadata_rows_permutes_the_output_columns(self, tmp_path):
        path_a = _write_tif(tmp_path / "a.tif", np.full((2, 2), 11, dtype="int32"))
        path_b = _write_tif(tmp_path / "b.tif", np.full((2, 2), 22, dtype="int32"))

        forward_lf, _, _ = scan_raster(_metadata([(path_a, 1), (path_b, 1)]))
        forward = forward_lf.collect()
        assert forward["b0"].unique().to_list() == [11]
        assert forward["b1"].unique().to_list() == [22]

        reversed_lf, _, _ = scan_raster(_metadata([(path_b, 1), (path_a, 1)]))
        reversed_out = reversed_lf.collect()
        assert reversed_out["b0"].unique().to_list() == [22]
        assert reversed_out["b1"].unique().to_list() == [11]


class TestCrosswalkIsANonBandColumn:
    """A crosswalk rides along in the same VRT read, but comes back as a NAMED non-band
    column, not as `b<n>` -- see `io/raster.py`'s module docstring. The distinction is not
    cosmetic: `band_column_names(metadata_df)` is what every band-wide lowering reduces
    across, so a crosswalk counted as a band would get folded into the statistic.
    """

    def _paths(self, tmp_path):
        data_path = _write_tif(tmp_path / "data.tif", np.full((2, 2), 5, dtype="int32"))
        crosswalk_path = _write_tif(tmp_path / "xwalk.tif", np.array([[1, 1], [2, 2]], dtype="int32"))
        return data_path, crosswalk_path

    def test_crosswalk_comes_back_under_its_own_name(self, tmp_path):
        data_path, crosswalk_path = self._paths(tmp_path)
        metadata_df = _metadata([(data_path, 1)])

        lf, out_metadata, _ = scan_raster(metadata_df, crosswalks=[("zone", crosswalk_path)])
        out = lf.collect()

        assert list(out.columns) == ["b0", "zone"]
        assert (out["b0"] == 5).all()
        # Row-major raster order: [1, 1, 2, 2] for a 2x2 crosswalk of [[1,1],[2,2]].
        assert out["zone"].to_list() == [1, 1, 2, 2]
        # The metadata frame is untouched -- a crosswalk is not a band, so it gains no row
        # and `band_column_names` keeps naming only the data bands.
        assert out_metadata is metadata_df
        assert band_column_names(out_metadata) == ["b0"]

    def test_attaching_a_crosswalk_does_not_shift_any_band_name(self, tmp_path):
        """Crosswalk sources are appended after every data band, so the `b<i>` a given
        metadata row maps to is the same whether or not a crosswalk rides along."""
        data_path, crosswalk_path = self._paths(tmp_path)
        other_path = _write_tif(tmp_path / "other.tif", np.full((2, 2), 7, dtype="int32"))
        metadata_df = _metadata([(data_path, 1), (other_path, 1)])

        without_lf, _, _ = scan_raster(metadata_df)
        with_lf, _, _ = scan_raster(metadata_df, crosswalks=[("zone", crosswalk_path)])
        without = without_lf.collect()
        with_crosswalk = with_lf.collect()

        assert with_crosswalk.drop("zone").equals(without)

    def test_two_crosswalks_come_back_as_two_columns_in_order(self, tmp_path):
        data_path, crosswalk_path = self._paths(tmp_path)
        second_path = _write_tif(tmp_path / "xwalk2.tif", np.array([[8, 9], [8, 9]], dtype="int32"))

        lf, _, _ = scan_raster(
            _metadata([(data_path, 1)]),
            crosswalks=[("zone", crosswalk_path), ("basin", second_path)],
        )
        out = lf.collect()

        assert list(out.columns) == ["b0", "zone", "basin"]
        assert out["basin"].to_list() == [8, 9, 8, 9]

    @pytest.mark.parametrize("name", ["b0", "b7", "_band"])
    def test_a_crosswalk_name_that_looks_like_a_band_is_rejected(self, tmp_path, name):
        data_path, crosswalk_path = self._paths(tmp_path)
        with pytest.raises(ValueError, match="positional band-name pattern"):
            scan_raster(_metadata([(data_path, 1)]), crosswalks=[(name, crosswalk_path)])

    def test_duplicate_crosswalk_names_are_rejected(self, tmp_path):
        data_path, crosswalk_path = self._paths(tmp_path)
        with pytest.raises(ValueError, match="duplicate"):
            scan_raster(
                _metadata([(data_path, 1)]),
                crosswalks=[("zone", crosswalk_path), ("zone", crosswalk_path)],
            )


class TestWindowedReadEqualsDirect:
    def test_window_matches_direct_rasterio_read(self, tmp_path):
        data = np.arange(100, dtype="int32").reshape(10, 10)
        path = _write_tif(tmp_path / "grad.tif", data)
        window = Window(col_off=2, row_off=3, width=4, height=3)

        lf, _, read_window = scan_raster(_metadata([(path, 1)]), window=window)
        out = lf.collect()

        with rio.open(path) as src:
            expected = src.read(1, window=window)
            raster_transform = src.transform

        assert out["b0"].to_numpy().reshape(3, 4).tolist() == expected.tolist()
        assert (read_window.col_off, read_window.row_off) == (2, 3)
        assert (read_window.width, read_window.height) == (4, 3)
        assert read_window.transform == window_transform(window, raster_transform)
        assert read_window.decimation == (1, 1)
        assert read_window.read_shape == (3, 4)


class TestDecimatedWindowedRead:
    """`decimation` asks GDAL to serve the window out of the source's overview pyramid.
    What the scan owes its caller is that the smaller array is still described correctly:
    the same ground, at a coarser resolution, with the frame's row count and the window's
    transform both saying so.
    """

    def _gradient(self, tmp_path, size=8):
        # Unique per pixel, so a wrong stride or a transposed axis shows up as a wrong
        # VALUE rather than merely a wrong shape.
        data = np.arange(size * size, dtype="int32").reshape(size, size)
        return _write_tif(tmp_path / "grad.tif", data), data

    def test_row_count_and_window_report_the_decimated_shape(self, tmp_path):
        path, _ = self._gradient(tmp_path)
        window = Window(0, 0, 8, 8)

        lf, _, read_window = scan_raster(_metadata([(path, 1)]), window=window, decimation=(2, 2))

        assert lf.collect().height == 16
        assert read_window.read_shape == (4, 4)
        # The SOURCE extent is unchanged: the window still describes the same ground, which
        # is what lets `assert_same_pixel_grid` treat extent and resolution as two separate
        # questions rather than one conflated one.
        assert (read_window.width, read_window.height) == (8, 8)
        assert read_window.decimation == (2, 2)

    def test_values_match_a_direct_decimated_rasterio_read(self, tmp_path):
        path, _ = self._gradient(tmp_path)
        window = Window(0, 0, 8, 8)

        lf, _, _ = scan_raster(_metadata([(path, 1)]), window=window, decimation=(2, 2))

        with rio.open(path) as src:
            expected = src.read(1, window=window, out_shape=(4, 4))
        assert lf.collect()["b0"].to_numpy().reshape(4, 4).tolist() == expected.tolist()

    def test_every_decimated_value_is_a_real_source_value(self, tmp_path):
        """Nearest, not an average -- which is what keeps a band's nodata sentinel intact
        for the masking below to find, and what makes a decimated tile a real sample of the
        data rather than a blend of it.
        """
        path, data = self._gradient(tmp_path)

        lf, _, _ = scan_raster(_metadata([(path, 1)]), window=Window(0, 0, 8, 8), decimation=(2, 2))

        assert set(lf.collect()["b0"].to_list()) <= set(data.ravel().tolist())

    def test_the_transform_is_scaled_so_the_array_covers_the_same_ground(self, tmp_path):
        path, _ = self._gradient(tmp_path)
        window = Window(0, 0, 8, 8)

        _, _, read_window = scan_raster(_metadata([(path, 1)]), window=window, decimation=(2, 2))

        with rio.open(path) as src:
            undecimated = window_transform(window, src.transform)
        # Same origin, pixels twice as large on each axis: the array is half the size and
        # covers the identical extent. A transform left unscaled here is the sub-pixel error
        # `warp_window_to_tile`'s docstring describes -- every rendered pixel displaced,
        # with nothing in the output to say so.
        assert (read_window.transform.c, read_window.transform.f) == (undecimated.c, undecimated.f)
        assert (read_window.transform.a, read_window.transform.e) == (
            undecimated.a * 2,
            undecimated.e * 2,
        )

    def test_a_per_axis_factor_decimates_only_that_axis(self, tmp_path):
        path, _ = self._gradient(tmp_path)

        _, _, read_window = scan_raster(_metadata([(path, 1)]), window=Window(0, 0, 8, 8), decimation=(4, 1))

        assert read_window.read_shape == (2, 8)

    def test_a_crosswalk_rides_the_same_decimated_read(self, tmp_path):
        """One read, so the zone ids stay positionally aligned with the bands for free --
        and stay real zone ids, since nearest never blends two zones into a third.
        """
        path, _ = self._gradient(tmp_path)
        zones = np.repeat(np.arange(4, dtype="int32"), 16).reshape(8, 8)
        crosswalk_path = _write_tif(tmp_path / "xwalk.tif", zones)

        lf, _, read_window = scan_raster(
            _metadata([(path, 1)]),
            crosswalks=[("zone", crosswalk_path)],
            window=Window(0, 0, 8, 8),
            decimation=(2, 2),
        )
        out = lf.collect()

        assert out.height == 16 == read_window.read_shape[0] * read_window.read_shape[1]
        assert set(out["zone"].to_list()) <= {0, 1, 2, 3}

    def test_a_decimated_read_records_no_batch_layout(self, tmp_path):
        """A decimated read is one batch over the whole window. Recording the source-pixel
        extents an undecimated single-batch read records would be a unit mismatch against
        the smaller array the rows actually came from, and `row_major_grid_cell` reads the
        layout in array units.
        """
        path, _ = self._gradient(tmp_path)

        _, _, decimated = scan_raster(_metadata([(path, 1)]), window=Window(0, 0, 8, 8), decimation=(2, 2))
        _, _, whole = scan_raster(_metadata([(path, 1)]), window=Window(0, 0, 8, 8))

        assert decimated.batch_layout == ()
        assert whole.batch_layout == ((0, 0, 8, 8),)

    def test_decimation_without_a_window_is_rejected(self, tmp_path):
        """The one combination that cannot be made correct: a full read is chunked into
        batches whose layout is expressed in source pixels, and floor-dividing each batch
        separately does not tile the decimated array -- so every row index downstream would
        decode to the wrong pixel. Refused outright rather than silently read whole.
        """
        path, _ = self._gradient(tmp_path)

        with pytest.raises(ValueError, match="requires an explicit window"):
            scan_raster(_metadata([(path, 1)]), decimation=(2, 2))

    @pytest.mark.parametrize("decimation", [(0, 1), (1, -2)])
    def test_a_factor_below_one_is_rejected(self, tmp_path, decimation):
        path, _ = self._gradient(tmp_path)

        with pytest.raises(ValueError, match="must be >= 1"):
            scan_raster(_metadata([(path, 1)]), window=Window(0, 0, 8, 8), decimation=decimation)

    def test_the_default_is_full_resolution(self, tmp_path):
        path, data = self._gradient(tmp_path)
        window = Window(0, 0, 8, 8)

        lf, _, read_window = scan_raster(_metadata([(path, 1)]), window=window)

        assert read_window.decimation == (1, 1)
        assert lf.collect()["b0"].to_list() == data.ravel().tolist()


class TestTwoTierNodataMasking:
    """A data band's nodata sentinel masks only when `mask_nodata` says to (default on);
    a crosswalk column's masks unconditionally, regardless of the flag -- see
    `io/raster.py`'s module docstring for why the two are different questions.
    """

    def _build(self, tmp_path):
        data = np.array([[-9999, 1], [2, 3]], dtype="int32")
        data_path = _write_tif(tmp_path / "data.tif", data, nodata=-9999)
        crosswalk = np.array([[0, 1], [2, 0]], dtype="int32")
        crosswalk_path = _write_tif(tmp_path / "xwalk.tif", crosswalk, nodata=0)
        return _metadata([(data_path, 1)]), [("zone", crosswalk_path)]

    def test_data_band_masked_when_flag_on(self, tmp_path):
        metadata_df, crosswalks = self._build(tmp_path)
        lf, _, _ = scan_raster(metadata_df, crosswalks=crosswalks, mask_nodata=True)
        out = lf.collect()
        assert out["b0"].null_count() == 1
        assert out["b0"].drop_nulls().to_list() == [1, 2, 3]

    def test_data_band_not_masked_when_flag_off(self, tmp_path):
        metadata_df, crosswalks = self._build(tmp_path)
        lf, _, _ = scan_raster(metadata_df, crosswalks=crosswalks, mask_nodata=False)
        out = lf.collect()
        assert out["b0"].null_count() == 0
        assert -9999 in out["b0"].to_list()

    @pytest.mark.parametrize("mask_nodata", [True, False])
    def test_crosswalk_column_masked_regardless_of_flag(self, tmp_path, mask_nodata):
        metadata_df, crosswalks = self._build(tmp_path)
        lf, _, _ = scan_raster(metadata_df, crosswalks=crosswalks, mask_nodata=mask_nodata)
        out = lf.collect()
        assert out["zone"].null_count() == 2


class TestMultibandCogSharedByTwoMetadataRows:
    """Several `metadata_df` rows can share one file (a multiband COG) -- band SELECTION
    is by the `band` column, independent of metadata row order.
    """

    def test_band_selection_is_independent_of_metadata_row_order(self, tmp_path):
        band1 = np.full((2, 2), 100, dtype="int32")
        band2 = np.full((2, 2), 200, dtype="int32")
        path = _write_tif(tmp_path / "multi.tif", np.stack([band1, band2]))

        lf, _, _ = scan_raster(_metadata([(path, 2), (path, 1)]))
        out = lf.collect()

        assert out["b0"].unique().to_list() == [200]
        assert out["b1"].unique().to_list() == [100]


class TestZeroBandMetadataRaises:
    def test_raises_value_error(self):
        empty = pl.DataFrame({"path": [], "band": []}, schema={"path": pl.String, "band": pl.Int64})
        with pytest.raises(ValueError):
            scan_raster(empty)


class TestMissingRequiredColumnsRaises:
    def test_missing_band_column_raises_value_error(self):
        malformed = pl.DataFrame({"path": ["a.tif"]}, schema={"path": pl.String})
        with pytest.raises(ValueError, match="'path' and 'band'"):
            scan_raster(malformed)


class TestVrtBandCountMismatchRaises:
    """Defensive guard against a `vrt()` that resolved a different number of bands than
    `metadata_df` named -- not reachable through the real `vrt()` (it always emits
    exactly one output band per source), so exercised here via a routing-knob monkeypatch
    on `io/raster.py`'s own `vrt` name, mirroring `test_project.py`'s
    `TestDirectPathMatchesFoldedPath` precedent for pinning an otherwise-unreachable
    defensive branch.
    """

    def test_mismatched_band_count_raises(self, tmp_path, monkeypatch):
        from cpra.mp.data.analysis.io import raster as raster_module

        path_a = _write_tif(tmp_path / "a.tif", np.full((2, 2), 1, dtype="int32"))
        path_b = _write_tif(tmp_path / "b.tif", np.full((2, 2), 2, dtype="int32"))
        real_vrt = raster_module.vrt
        # Silently drops the second source's band, so the assembled VRT has one band
        # while metadata_df still names two.
        monkeypatch.setattr(raster_module, "vrt", lambda sources: real_vrt(sources[:1]))

        with pytest.raises(ValueError, match="VRT assembled"):
            scan_raster(_metadata([(path_a, 1), (path_b, 1)]))


class TestNRowsTruncation:
    """`n_rows` is the one plugin parameter `io/raster.py`'s generator honors despite the
    "no pushdown" scope fence -- see its module docstring's `source` inline comment: it
    is a hard truncation requirement (`.head`/`.limit`), not an optimization a caller
    could apply just as well downstream.
    """

    def test_head_truncates_to_the_first_n_pixels_in_raster_order(self, tmp_path):
        data = np.arange(16, dtype="int32").reshape(4, 4)
        path = _write_tif(tmp_path / "grad.tif", data)

        lf, _, _ = scan_raster(_metadata([(path, 1)]))
        out = lf.head(3).collect()

        assert out.height == 3
        assert out["b0"].to_list() == [0, 1, 2]

    def test_n_rows_reaches_the_source_across_batches(self, tmp_path, monkeypatch):
        """Forces small batches (RASTER_BLOCKS_PER_BATCH=1) so a `.head()` smaller than
        one batch's worth of pixels still has to stop mid-stream, exercising the
        truncate-within-a-batch path. (The generator's `remaining <= 0` break stays
        uncovered by design: Polars stops pulling once `n_rows` is satisfied, so that
        branch is a backstop against a consumer that keeps asking, not a path this or
        any other realistic read takes.)"""
        monkeypatch.setattr(raster_io, "RASTER_BLOCKS_PER_BATCH", 1)
        data = np.arange(100, dtype="int32").reshape(10, 10)
        path = _write_tif(tmp_path / "grad.tif", data)

        lf, _, _ = scan_raster(_metadata([(path, 1)]))
        out = lf.head(5).collect()

        assert out.height == 5
        assert out["b0"].to_list() == [0, 1, 2, 3, 4]


class TestTripleConformsToBandWideInvariants:
    """Checks the returned triple against `execution/frame.py`'s own conformance
    helpers, which every real band-wide lowering is built to satisfy -- `io/raster.py`
    cannot import that module (layering), so these checks live here, in the test that is
    allowed to depend on both layers.
    """

    def test_columns_match_band_column_names_and_height_equals_pixel_count(self, tmp_path):
        path_a = _write_tif(tmp_path / "a.tif", np.full((2, 2), 1, dtype="int32"))
        path_b = _write_tif(tmp_path / "b.tif", np.full((2, 2), 2, dtype="int32"))
        metadata_df = _metadata([(path_a, 1), (path_b, 1)])

        lf, out_metadata, _ = scan_raster(metadata_df)
        out = lf.collect()

        assert list(out.columns) == band_column_names(out_metadata)
        # 2x2 window: exactly 4 pixel rows, whatever the batching happened to be.
        assert_height_preserved(pl.LazyFrame({"_": range(4)}), lf)

    def test_window_is_deterministic_across_repeated_scans(self, tmp_path):
        path = _write_tif(tmp_path / "a.tif", np.full((2, 2), 1, dtype="int32"))
        metadata_df = _metadata([(path, 1)])

        _, _, window_1 = scan_raster(metadata_df)
        _, _, window_2 = scan_raster(metadata_df)

        # A fresh VRT is built per call (no caching) -- this checks that doing so twice
        # over the identical input still produces the identical window, the property
        # `assert_window_unchanged` exists to check on a real lowering's input/output.
        assert_window_unchanged(window_1, window_2)


class TestResolveRasterBands:
    """`io/catalog.py`'s `resolve_raster_bands` -- the ordered band list a scan needs,
    resolved from an `InputSelection` via real `vct`/`veg_grid_cell_v001` fixtures.
    """

    SELECTION = InputSelection(**{k: RASTER_KWARGS[k] for k in _SELECTION_FIELDS})

    def test_returns_one_row_per_file_band_sorted_by_calendar_year(self, base_path):
        write_raster_dataset(base_path, years=(2040, 2030))  # written out of order
        df = resolve_raster_bands(self.SELECTION, base_path=base_path)

        assert df.height == 2
        assert df["calendar_year"].to_list() == [2030, 2040]
        assert df["band"].to_list() == [1, 1]

    def test_unnamed_partition_dimension_comes_back_unconstrained(self, base_path):
        """`InputSelection` names a dataset, not a model run -- every `scenario_id`
        present on disk for this variable/time_unit/grid comes back, not just one."""
        write_raster_dataset(base_path, years=(2030,), scenario_id=7)
        write_raster_dataset(base_path, years=(2030,), scenario_id=8)

        df = resolve_raster_bands(self.SELECTION, base_path=base_path)

        assert set(df["scenario_id"].to_list()) == {7, 8}

    def test_dataset_with_no_files_yet_is_an_empty_frame_not_an_error(self, base_path):
        df = resolve_raster_bands(self.SELECTION, base_path=base_path)
        assert df.height == 0

    def test_structurally_invalid_combination_raises_not_found(self, base_path):
        # vct's variables.csv row declares time_units=annual only -- daily is a
        # structurally invalid combination, not merely one with no files on disk.
        bad_selection = InputSelection(variable="vct", time_unit="daily", grid="veg_grid_cell_v001")
        with pytest.raises(NotFoundError):
            resolve_raster_bands(bad_selection, base_path=base_path)


class TestPathPredicate:
    """`resolve_raster_bands`' `path_predicate`, and `compile_plan` handing it one.

    A file's calendar values are only known once it is opened, but its on-path dimensions
    are known from the glob alone -- so a request pinned to one model group must open that
    model group's files and no others. `veg_species` (a directory per species code) made
    the difference tens of thousands of opens per map tile.
    """

    SELECTION = TestResolveRasterBands.SELECTION

    @staticmethod
    def _two_model_groups(base_path):
        write_raster_dataset(base_path, years=(2030, 2040), model_group_id=500)
        write_raster_dataset(base_path, years=(2030, 2040), model_group_id=512)

    def test_files_outside_the_pinned_partition_are_never_opened(self, base_path):
        """Proved by corrupting them: if one were opened, the open would raise."""
        self._two_model_groups(base_path)
        for path in base_path.rglob("model_group_id=512/**/*.tif"):
            path.write_bytes(b"not a tiff")

        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {"a": {k: RASTER_KWARGS[k] for k in _SELECTION_FIELDS}},
                    # The calendar conjunct cannot be applied to a path; it must be left to
                    # the band-identity pass rather than break or widen the path predicate.
                    "steps": [{"op": "filter", "input": "a", "where": "model_group_id = 500 AND calendar_year = 2030"}],
                    "output": {"format": "tif"},
                }
            ),
            base_path=base_path,
        )

        metadata = plan.inputs["a"].band_metadata
        assert metadata["model_group_id"].to_list() == [500]
        assert metadata["calendar_year"].to_list() == [2030]

    def test_narrows_the_enumeration_to_matching_files(self, base_path):
        self._two_model_groups(base_path)
        df = resolve_raster_bands(self.SELECTION, base_path=base_path, path_predicate=pl.col("model_group_id") == 512)
        assert df["model_group_id"].to_list() == [512, 512]
        assert df["calendar_year"].to_list() == [2030, 2040]

    def test_a_predicate_matching_no_file_keeps_one_file_rather_than_none(self, base_path):
        """See `resolve_raster_bands`' docstring: zero bands would starve the join-sibling
        floor (`test_execute.py`'s path-dimension floor test is the end-to-end case)."""
        self._two_model_groups(base_path)
        df = resolve_raster_bands(self.SELECTION, base_path=base_path, path_predicate=pl.col("model_group_id") == 999)
        assert df["path"].n_unique() == 1
        assert df.height > 0

    def test_a_dataset_with_no_files_stays_empty(self, base_path):
        df = resolve_raster_bands(self.SELECTION, base_path=base_path, path_predicate=pl.col("model_group_id") == 500)
        assert df.height == 0


class TestDroppedBandsAreNeverOpened:
    """Band selection is a real narrowing of the physical read, not a filter applied to
    rows that were already read.

    Everything else in this file drives `scan_raster` from a hand-built `metadata_df`,
    and every raster test in `test_execution_plan.py` monkeypatches the catalog. This
    one deliberately does neither: it goes through the real `resolve_raster_bands` and
    the real dimension narrowing, and then proves the narrowing reached the file system
    by DELETING the dropped year's file. If the scan still returns 2030's pixels, the
    2040 file was never opened -- which is the only way that can succeed.
    """

    def _dropped_file(self, base_path, where):
        write_raster_dataset(base_path, years=(2030, 2040))
        plan = compile_plan(
            _pipeline(
                {
                    "inputs": {"a": {k: RASTER_KWARGS[k] for k in _SELECTION_FIELDS}},
                    "steps": [{"op": "filter", "input": "a", "where": where}],
                    "output": {"format": "tif"},
                }
            ),
            base_path=base_path,
        )
        metadata = plan.inputs["a"].band_metadata
        assert metadata["calendar_year"].to_list() == [2030]
        return metadata

    def test_the_dropped_years_file_is_never_opened(self, base_path):
        metadata = self._dropped_file(base_path, "calendar_year == 2030")

        surviving = set(metadata["path"].to_list())
        removed = [p for p in base_path.rglob("*.tif") if str(p) not in surviving]
        assert removed, "fixture wrote nothing for the dropped year -- the test would pass vacuously"
        for path in removed:
            path.unlink()

        lf, _, _ = scan_raster(metadata)
        out = lf.collect()

        assert out.height > 0
        assert [c for c in out.columns if c.startswith("b")] == ["b0"]


class TestBatchedReadMatchesUnbatched:
    """Every fixture here writes DISTINGUISHING pixel values (`raster_array()`'s own
    `arange`, through `data_fn`), never the uniform per-year fill `write_raster_dataset`
    defaults to. A uniform raster cannot tell any of these apart: re-chunking a read
    permutes its rows, and every permutation of a constant array is that same array.
    """

    @staticmethod
    def _distinct(base_path):
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda year: raster_array())
        return resolve_raster_bands(TestResolveRasterBands.SELECTION, base_path=base_path)

    def test_forcing_many_small_batches_reads_every_pixel_the_default_batching_does(self, base_path, monkeypatch):
        """Two chunkings of one read see the same values at the same CELLS. That is what
        "batching does not change what was read" means; it is NOT the same claim as "the
        two frames are equal row for row", which is false -- see the next test.
        """
        metadata_df = self._distinct(base_path)
        default_lf, _, default_window = scan_raster(metadata_df)
        default_cells = row_major_grid_cell(default_lf, default_window).collect().sort("grid_cell")

        monkeypatch.setattr(raster_io, "RASTER_BLOCKS_PER_BATCH", 1)
        forced_lf, _, forced_window = scan_raster(metadata_df)
        forced_cells = row_major_grid_cell(forced_lf, forced_window).collect().sort("grid_cell")

        assert len(forced_window.batch_layout) > 1  # the forcing really did take effect
        assert forced_cells.equals(default_cells)

    def test_rechunking_permutes_the_frames_own_row_order(self, base_path, monkeypatch):
        """The fact `row_major_grid_cell` exists for, pinned as a test rather than left as
        a claim in a docstring: a batch covers a sub-rectangle of a block row, not a
        full-width strip, so re-chunking a read reorders its rows. Same values, same
        multiset, different row indices.
        """
        metadata_df = self._distinct(base_path)
        default_out = scan_raster(metadata_df)[0].collect()

        monkeypatch.setattr(raster_io, "RASTER_BLOCKS_PER_BATCH", 1)
        forced_out = scan_raster(metadata_df)[0].collect()

        assert not forced_out.equals(default_out)
        assert forced_out.sort("b0").equals(default_out.sort("b0"))

    def test_the_window_records_the_batching_and_is_otherwise_unchanged(self, base_path, monkeypatch):
        """`batch_layout` is the one field two chunkings of the same read differ in, and it
        has to differ: it is the only record of which pixel a given row index means.
        """
        metadata_df = self._distinct(base_path)
        _, _, default_window = scan_raster(metadata_df)

        monkeypatch.setattr(raster_io, "RASTER_BLOCKS_PER_BATCH", 1)
        _, _, forced_window = scan_raster(metadata_df)

        assert forced_window.batch_layout != default_window.batch_layout
        assert dataclasses.replace(forced_window, batch_layout=()) == dataclasses.replace(
            default_window, batch_layout=()
        )

    def test_batch_layout_covers_the_window_exactly_once(self, base_path, monkeypatch):
        monkeypatch.setattr(raster_io, "RASTER_BLOCKS_PER_BATCH", 1)
        metadata_df = self._distinct(base_path)
        _, _, window = scan_raster(metadata_df)

        covered = {
            (row, col)
            for col_off, row_off, width, height in window.batch_layout
            for row in range(row_off, row_off + height)
            for col in range(col_off, col_off + width)
        }
        assert sum(width * height for _c, _r, width, height in window.batch_layout) == window.width * window.height
        assert len(covered) == window.width * window.height


class TestBandsOfDifferingDtype:
    """A `rasterio` read of several bands at once returns ONE array, so every band it reads
    in that call must share a dtype. The bands of one VRT routinely do not: a crosswalk's
    dtype follows its zone-id range alone, and two variables read together are two
    different measurements stored two different ways. `scan_raster` groups its bands by
    dtype and reads one group per call; asking for all of them at once raises
    `ValueError: more than one 'dtype' found`.
    """

    def test_data_bands_of_two_dtypes_scan_together(self, tmp_path):
        int_path = _write_tif(tmp_path / "ints.tif", np.array([[1, 2], [3, 4]], dtype="int32"))
        float_path = _write_tif(tmp_path / "floats.tif", np.array([[1.5, 2.5], [3.5, 4.5]], dtype="float32"))
        metadata_df = pl.DataFrame({"path": [str(int_path), str(float_path)], "band": [1, 1]})

        out = scan_raster(metadata_df)[0].collect()

        assert out.schema["b0"] == pl.Int32
        assert out.schema["b1"] == pl.Float32
        assert out["b0"].to_list() == [1, 2, 3, 4]
        assert out["b1"].to_list() == [1.5, 2.5, 3.5, 4.5]

    def test_an_integer_crosswalk_rides_along_with_float_data(self, tmp_path):
        """The production shape, and the one the raster-to-vector `project` depends on: a
        float measurement band and an integer zone crosswalk in a single read.
        """
        data_path = _write_tif(tmp_path / "data.tif", np.array([[1.5, 2.5], [3.5, 4.5]], dtype="float32"))
        zone_path = _write_tif(tmp_path / "zones.tif", np.array([[7, 7], [8, -1]], dtype="int16"), nodata=-1)
        metadata_df = pl.DataFrame({"path": [str(data_path)], "band": [1]})

        out = scan_raster(metadata_df, crosswalks=[("zone", str(zone_path))])[0].collect()

        assert out.schema["b0"] == pl.Float32
        assert out.schema["zone"] == pl.Int16
        assert out["b0"].to_list() == [1.5, 2.5, 3.5, 4.5]
        assert out["zone"].to_list() == [7, 7, 8, None]


class TestPushdownIsHonored:
    """Polars REMOVES a `.select`/`.filter` from the plan when it pushes it into an IO
    plugin, and does not re-apply it afterward. A plugin that accepts either and ignores it
    therefore does not merely miss an optimization -- it returns rows the plan says were
    already filtered out, and columns the plan says were already projected away.
    """

    def _two_band_scan(self, tmp_path):
        path = _write_tif(
            tmp_path / "two.tif",
            np.array([[[1, 2], [3, 4]], [[10, 20], [30, 40]]], dtype="int32"),
        )
        return scan_raster(pl.DataFrame({"path": [str(path)] * 2, "band": [1, 2]}))[0]

    def test_a_chained_filter_actually_drops_rows(self, tmp_path):
        lf = self._two_band_scan(tmp_path)

        assert lf.filter(pl.col("b0") > 2).collect()["b0"].to_list() == [3, 4]

    def test_a_chained_select_actually_drops_columns(self, tmp_path):
        lf = self._two_band_scan(tmp_path)

        assert lf.select("b1").collect().columns == ["b1"]

    def test_a_filter_and_a_head_compose(self, tmp_path):
        """`n_rows` means "the first n rows of what this source produces", so it applies
        after the predicate, not before it."""
        lf = self._two_band_scan(tmp_path)

        assert lf.filter(pl.col("b0") > 1).head(2).collect()["b0"].to_list() == [2, 3]


class TestValuePredicateNeverBelongsAsARowFilter:
    """This scan honors whatever `.filter()` a caller chains onto it -- that is what
    `TestPushdownIsHonored` above checks. For a VALUE predicate specifically, honoring it
    this way is exactly the danger `execution.plan.compile_plan` is built to avoid: a row
    is one pixel across every band at once, so dropping a row because one band's value
    failed a predicate throws away every OTHER band's value at that same pixel too, even
    the ones the predicate holds for.

    Reproduces `tests/api/test_band_ops.py`'s `_two_band_fixture()` disagreement (one
    pixel where band 0 satisfies a `value` predicate and band 1 does not) against a real
    two-band GeoTIFF read through this scan, rather than the hand-built frame that
    fixture uses -- so the corruption below is demonstrated at the actual IO boundary, not
    just against a synthetic `pl.LazyFrame`. `execution.band_ops.mask_value_band_wide`'s
    own `test_masks_not_drops_and_bands_disagree` (in `test_band_ops.py`) checks the
    corresponding CORRECT behavior -- null out the failing band only, keep the row -- on
    that same hand-built frame; `compile_plan` never gives a raster `ResolvedInput` a
    `predicate` at all, so a value conjunct against a raster origin has no path to this
    scan's `filter` argument and can only ever reach `mask_value_band_wide` once a real
    executor lowers it.
    """

    def test_a_value_filter_on_a_disagreeing_pixel_drops_the_whole_row(self, tmp_path):
        # Pixel index 2 (row-major): b0 = 1 fails `value > 4`, b1 = 20 satisfies it on ITS
        # OWN band -- the same disagreement `_two_band_fixture()` engineers by hand. A
        # `.filter()` naming one band as a stand-in for "value" has no way to see that: it
        # can only check the band it names.
        band0 = np.array([[1, 5], [1, 1]], dtype="int32")
        band1 = np.array([[2, 1], [20, 1]], dtype="int32")
        path = _write_tif(tmp_path / "two.tif", np.stack([band0, band1]))
        lf, _, _ = scan_raster(_metadata([(path, 1), (path, 2)]))

        filtered = lf.filter(pl.col("b0") > 4).collect()

        # The row-drop corruption: pixel index 2 vanishes ENTIRELY -- including band 1's
        # own value (20), which satisfies the very predicate being applied -- because the
        # filter only ever consulted band 0's value at that pixel.
        assert filtered.height == 1
        assert filtered["b0"].to_list() == [5]
        assert 20 not in filtered["b1"].to_list()
