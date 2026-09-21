"""Tests for `execution/project.py`'s two raster-touching lowerings:
`project_raster_to_vector` and `project_vector_to_raster`. Sibling to `test_project.py`,
which covers `project_vector_to_vector` -- same conventions (hand-computed expected
values written as literals, the `crosswalk_path` fixture, `pytest.approx`), but this file
writes its own small synthetic GeoTIFF crosswalks directly via rasterio, following
`test_raster_scan.py`'s `_write_tif` helper, rather than reusing `tests/factories.py`'s
`write_raster_crosswalk` -- that factory locks its raster's shape to a real grid's full
(hundreds-of-thousands-of-pixel) dimensions, far too large to hand-compute a zonal
statistic against by eye.

`RASTER_GRID`/`VECTOR_GRID` below are real, registered grid names (`Grid` is a pydantic
enum built off the real grid structure declared on disk, so `ProjectOp.to` must name one
that exists) -- nothing under test here ever opens them against `GRIDS`; they are only
used to build the crosswalk artifact's own directed filename via
`resolve_raster_crosswalk_path(RASTER_GRID, VECTOR_GRID)`, exactly like production would.

The large-scale order-preservation test aside, every raster/crosswalk fixture here is
tiny (2x2 or 2x3) by design -- a distinguishing, hand-checkable fixture beats a large,
"probably right" one for a zonal aggregate or a broadcast join.
"""

import numpy as np
import polars as pl
import pytest
import rasterio as rio
from rasterio.transform import from_origin

from cpra.mp.data.analysis.execution import project
from cpra.mp.data.analysis.execution.frame import DOMAIN_COLUMN
from cpra.mp.data.analysis.execution.project import project_raster_to_vector, project_vector_to_raster
from cpra.mp.data.analysis.io.crosswalk_read import resolve_raster_crosswalk_path
from cpra.mp.data.analysis.io.raster import scan_raster
from cpra.mp.data.analysis.request.aggregates import validate_project_aggregate
from cpra.mp.data.analysis.request.errors import AggregateError, NotFoundError
from cpra.mp.data.analysis.request.ops import FrameType, ProjectOp, Shape
from cpra.mp.data.analysis.request.pipeline import RESERVED_BAND_INDEX_COLUMN

RASTER_GRID = "veg_grid_cell_v001"
VECTOR_GRID = "ecoregion_v001"

_TRANSFORM = from_origin(0, 10, 1, 1)
_CRS = "EPSG:4326"


def _write_tif(path, data, nodata=None):
    """Identical in spirit to `test_raster_scan.py`'s own helper of the same name --
    duplicated here rather than imported, since test modules in this suite do not import
    fixtures from one another. `data` is `(height, width)` (one band) or `(bands, height,
    width)`.
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


def _write_crosswalk(crosswalk_path, zones, nodata=-1):
    """The one raster-to-vector crosswalk artifact both new lowerings resolve, via
    `resolve_raster_crosswalk_path(RASTER_GRID, VECTOR_GRID)` -- named
    `{RASTER_GRID}__{VECTOR_GRID}.tif`, the same directed name production's
    `write_raster_crosswalk` would give it. One artifact serves both directions.
    """
    path = crosswalk_path / f"{RASTER_GRID}__{VECTOR_GRID}.tif"
    return _write_tif(path, zones.astype("int32"), nodata=nodata)


def _metadata(rows, **extra_columns):
    """The minimal `scan_raster` input: one `(path, band)` pair per row, in order, plus
    whatever dimension columns a test wants to ride along (e.g. `calendar_year=`)."""
    df = pl.DataFrame(
        {"path": [str(p) for p, _ in rows], "band": [b for _, b in rows]},
        schema={"path": pl.String, "band": pl.Int64},
    )
    if extra_columns:
        df = df.with_columns(**{name: pl.Series(values) for name, values in extra_columns.items()})
    return df


def _open_raster_frame(metadata_df, crosswalk_written=True, input_name="r"):
    """Open `metadata_df` exactly as `execute.py`'s `_open_band_wide_input` would for a
    raster origin whose pixels straddle a `project` -- data bands plus, when
    `crosswalk_written` (the crosswalk this module's own `_write_crosswalk` wrote, resolved
    via the same `resolve_raster_crosswalk_path(RASTER_GRID, VECTOR_GRID)` production
    calls), the crosswalk attached on the SAME `scan_raster` call under `project.
    ZONE_COLUMN` -- see `execute.py`'s and `project.py`'s own module docstrings for why the
    crosswalk always rides on the data bands' own read rather than a second scan.

    Returns the `frames` dict `project_raster_to_vector` itself now reads (its signature
    changed from `(metadata_df, op, frame_types)` to `(frames, op, frame_types)` -- see
    `project.py`'s module docstring, "all three project lowerings share the one uniform
    registry shape").
    """
    crosswalks = (
        [(project.ZONE_COLUMN, str(resolve_raster_crosswalk_path(RASTER_GRID, VECTOR_GRID)))]
        if crosswalk_written
        else []
    )
    return {input_name: scan_raster(metadata_df, crosswalks=crosswalks)}


def _raster_frame_types(input_name="r"):
    return {
        input_name: FrameType(
            shape=Shape.band_wide, grid=RASTER_GRID, has_count_by_value=False, value_columns=frozenset({"value"})
        )
    }


def _vector_frame_types(input_name="v"):
    return {
        input_name: FrameType(
            shape=Shape.long, grid=VECTOR_GRID, has_count_by_value=False, value_columns=frozenset({"value"})
        )
    }


def _raster_op(aggregate="sum", to=VECTOR_GRID, input_name="r"):
    return ProjectOp(step_no=1, input=input_name, to=to, aggregate=aggregate, output="out")


def _vector_op(to=RASTER_GRID, input_name="v"):
    return ProjectOp(step_no=1, input=input_name, to=to, aggregate=None, output="out")


class TestRasterToVectorZonalAggregate:
    """Hand-computed zonal aggregate: a 2x3 single-band raster, zoned by a crosswalk that
    also carries one nodata pixel (row-major order: [[100,100,100],[200,200,-1]]).
    Aggregate reads exclude the nodata pixel; see `TestRasterToVectorNodataExclusion` for
    a fixture built specifically to make an inclusion bug obvious instead.

    zone 100 -- pixels (0,0)=1, (0,1)=5, (0,2)=9: sum=15, mean=5.0, min=1, max=9, count=3.
    zone 200 -- pixels (1,0)=2, (1,1)=100 (pixel (1,1) is really (1,2)'s neighbor; the
    third zone-200 pixel is nodata and excluded): sum=102, mean=51.0, min=2, max=100,
    count=2.
    """

    _DATA = np.array([[1, 5, 9], [2, 100, 999]], dtype="float32")
    _ZONES = np.array([[100, 100, 100], [200, 200, -1]], dtype="int32")

    _EXPECTED = {
        "sum": {100: 15.0, 200: 102.0},
        "mean": {100: 5.0, 200: 51.0},
        "min": {100: 1.0, 200: 2.0},
        "max": {100: 9.0, 200: 100.0},
        "count": {100: 3, 200: 2},
    }

    @pytest.mark.parametrize("aggregate", ["sum", "mean", "min", "max", "count"])
    def test_zonal_aggregate_matches_hand_computed_values(self, tmp_path, crosswalk_path, aggregate):
        _write_crosswalk(crosswalk_path, self._ZONES)
        data_path = _write_tif(tmp_path / "data.tif", self._DATA)
        metadata_df = _metadata([(data_path, 1)])

        frames = _open_raster_frame(metadata_df)
        out = project_raster_to_vector(
            frames, _raster_op(aggregate=aggregate), _raster_frame_types()
        ).collect()

        got = dict(zip(out["grid_cell"].to_list(), out["value"].to_list()))
        assert got == pytest.approx(self._EXPECTED[aggregate])


class TestRasterToVectorDimensionCarrying:
    """Two bands differing only by `calendar_year` must not merge into one destination
    group -- `metadata_df`'s dimension columns ride onto the output via a band-identity
    join, keyed positionally to the aggregated column order.

    zones (2x2, no nodata): [[100,100],[200,200]].
    year 2030 data: [[1,2],[3,4]]  -> zone100 sum=1+2=3, zone200 sum=3+4=7.
    year 2040 data: [[10,20],[30,40]] -> zone100 sum=10+20=30, zone200 sum=30+40=70.
    """

    _ZONES = np.array([[100, 100], [200, 200]], dtype="int32")

    def test_two_years_do_not_merge(self, tmp_path, crosswalk_path):
        _write_crosswalk(crosswalk_path, self._ZONES)
        path_2030 = _write_tif(tmp_path / "y2030.tif", np.array([[1, 2], [3, 4]], dtype="float32"))
        path_2040 = _write_tif(tmp_path / "y2040.tif", np.array([[10, 20], [30, 40]], dtype="float32"))
        metadata_df = _metadata([(path_2030, 1), (path_2040, 1)], calendar_year=[2030, 2040])

        frames = _open_raster_frame(metadata_df)
        out = project_raster_to_vector(frames, _raster_op(aggregate="sum"), _raster_frame_types()).collect()

        rows = {
            (row["grid_cell"], row["calendar_year"]): row["value"]
            for row in out.iter_rows(named=True)
        }
        assert out.height == 4
        assert rows == pytest.approx({(100, 2030): 3.0, (100, 2040): 30.0, (200, 2030): 7.0, (200, 2040): 70.0})


class TestRasterToVectorNodataExclusion:
    """A nodata pixel's data value is deliberately enormous (999999) so that including it
    by mistake would be unmistakable in the result, not a subtle off-by-a-little error.

    zones (2x2): [[100, -1], [100, 100]] -- pixel (0,1) belongs to no destination cell.
    data: [[1000, 999999], [2, 3]]. Zone 100's true sum, excluding the nodata pixel, is
    1000 + 2 + 3 = 1005.
    """

    _ZONES = np.array([[100, -1], [100, 100]], dtype="int32")
    _DATA = np.array([[1000, 999999], [2, 3]], dtype="float32")

    def test_nodata_pixel_excluded_from_its_neighbors_aggregate(self, tmp_path, crosswalk_path):
        _write_crosswalk(crosswalk_path, self._ZONES)
        data_path = _write_tif(tmp_path / "data.tif", self._DATA)
        metadata_df = _metadata([(data_path, 1)])

        frames = _open_raster_frame(metadata_df)
        out = project_raster_to_vector(frames, _raster_op(aggregate="sum"), _raster_frame_types()).collect()

        assert out.height == 1
        assert out["grid_cell"].to_list() == [100]
        assert out["value"].to_list() == pytest.approx([1005.0])

    def test_nodata_zone_never_appears_as_a_destination_cell(self, tmp_path, crosswalk_path):
        _write_crosswalk(crosswalk_path, self._ZONES)
        data_path = _write_tif(tmp_path / "data.tif", self._DATA)
        metadata_df = _metadata([(data_path, 1)])

        frames = _open_raster_frame(metadata_df)
        out = project_raster_to_vector(frames, _raster_op(aggregate="count"), _raster_frame_types()).collect()

        assert None not in out["grid_cell"].to_list()
        assert -1 not in out["grid_cell"].to_list()


class TestRasterToVectorCountByValue:
    """One row per (band, destination cell, distinct value): the category in
    `value_category`, its count in `value` -- the same output shape the vector-to-vector
    direction produces, so a reader does not have to know which direction made the frame.

    zones (2x3): [[100,100,100],[200,200,-1]].
    data:        [[7, 7, 9], [7, nan-as-nodata, 999]] -- the nodata pixel is masked to null
    by the scan, and the pixel outside every zone is dropped before counting.
    Zone 100: 7 twice, 9 once. Zone 200: 7 once, null once.
    """

    _ZONES = np.array([[100, 100, 100], [200, 200, -1]], dtype="int32")
    _DATA = np.array([[7, 7, 9], [7, -1, 999]], dtype="float32")

    def _counts(self, tmp_path, crosswalk_path):
        _write_crosswalk(crosswalk_path, self._ZONES)
        data_path = _write_tif(tmp_path / "data.tif", self._DATA, nodata=-1)
        metadata_df = _metadata([(data_path, 1)])
        frames = _open_raster_frame(metadata_df)
        out = project_raster_to_vector(
            frames, _raster_op(aggregate="count_by_value"), _raster_frame_types()
        ).collect()
        return {(row["grid_cell"], row["value_category"]): row["value"] for row in out.iter_rows(named=True)}

    def test_one_row_per_zone_and_distinct_value(self, tmp_path, crosswalk_path):
        assert self._counts(tmp_path, crosswalk_path) == {
            (100, 7.0): 2,
            (100, 9.0): 1,
            (200, 7.0): 1,
            (200, None): 1,
        }

    def test_null_is_its_own_category_rather_than_a_dropped_row(self, tmp_path, crosswalk_path):
        """A destination whose contributing pixels are nodata should say so, not vanish --
        so the null category is counted, not filtered out ahead of the group.
        """
        counts = self._counts(tmp_path, crosswalk_path)

        assert (200, None) in counts

    def test_a_pixel_outside_every_zone_is_counted_nowhere(self, tmp_path, crosswalk_path):
        """The 999 pixel sits under the crosswalk's own nodata entry -- it belongs to no
        destination cell at all, which is not the same thing as a null VALUE in a zone."""
        counts = self._counts(tmp_path, crosswalk_path)

        assert 999.0 not in {category for _cell, category in counts}

    def test_carries_band_dimensions_like_every_other_aggregate(self, tmp_path, crosswalk_path):
        _write_crosswalk(crosswalk_path, np.array([[100, 100], [200, 200]], dtype="int32"))
        path_2030 = _write_tif(tmp_path / "y2030.tif", np.array([[1, 1], [2, 2]], dtype="float32"))
        path_2040 = _write_tif(tmp_path / "y2040.tif", np.array([[3, 3], [4, 4]], dtype="float32"))
        metadata_df = _metadata([(path_2030, 1), (path_2040, 1)], calendar_year=[2030, 2040])

        frames = _open_raster_frame(metadata_df)
        out = project_raster_to_vector(
            frames, _raster_op(aggregate="count_by_value"), _raster_frame_types()
        ).collect()

        rows = {
            (row["calendar_year"], row["grid_cell"], row["value_category"]): row["value"]
            for row in out.iter_rows(named=True)
        }
        assert rows == {
            (2030, 100, 1.0): 2,
            (2030, 200, 2.0): 2,
            (2040, 100, 3.0): 2,
            (2040, 200, 4.0): 2,
        }

    def test_too_many_distinct_values_is_an_aggregate_error(self, tmp_path, crosswalk_path, monkeypatch):
        _write_crosswalk(crosswalk_path, np.array([[100, 100], [200, 200]], dtype="int32"))
        data_path = _write_tif(tmp_path / "data.tif", np.array([[1, 2], [3, 4]], dtype="float32"))
        metadata_df = _metadata([(data_path, 1)])
        monkeypatch.setattr(project, "COUNT_BY_VALUE_MAX_DISTINCT", 2)
        frames = _open_raster_frame(metadata_df)

        with pytest.raises(AggregateError, match="count_by_value"):
            project_raster_to_vector(
                frames, _raster_op(aggregate="count_by_value"), _raster_frame_types()
            ).collect()


class TestRasterToVectorMissingCrosswalk:
    def test_resolving_a_missing_crosswalk_raises_not_found(self, crosswalk_path):
        """`project_raster_to_vector` no longer resolves the crosswalk itself -- opening a
        raster origin's frame with its crosswalk attached is `execute.py`'s own job now
        (see that module's docstring, "the raster-to-vector crosswalk is attached here"),
        performed before `project_raster_to_vector` ever runs. This pins that the
        resolution `execute.py` (and this file's own `_open_raster_frame` helper) relies on
        still raises `NotFoundError` for a raster/vector grid pair with no crosswalk
        artifact written -- no crosswalk written here, so this never even reaches
        `scan_raster`.
        """
        with pytest.raises(NotFoundError):
            resolve_raster_crosswalk_path(RASTER_GRID, VECTOR_GRID)


class TestRasterToVectorReaddressesGrid:
    def test_grid_column_names_the_destination_grid(self, tmp_path, crosswalk_path):
        _write_crosswalk(crosswalk_path, np.array([[100, 100], [200, 200]], dtype="int32"))
        data_path = _write_tif(tmp_path / "data.tif", np.array([[1, 2], [3, 4]], dtype="float32"))
        metadata_df = _metadata([(data_path, 1)], grid=[RASTER_GRID])

        frames = _open_raster_frame(metadata_df)
        out = project_raster_to_vector(frames, _raster_op(aggregate="sum"), _raster_frame_types()).collect()

        assert set(out["grid"].to_list()) == {VECTOR_GRID}


class TestValidateProjectAggregateUpstream:
    """`project_vector_to_raster` does not re-validate `op.aggregate` -- legality is a
    request-layer concern, already enforced by `request.aggregates.
    validate_project_aggregate` before a lowering ever runs (see that function's own
    docstring). These tests exercise that upstream function directly rather than
    re-implementing its rule inside this lowering's own test.
    """

    def test_vector_to_raster_rejects_a_supplied_aggregate(self):
        with pytest.raises(AggregateError, match="vector-to-raster project takes no aggregate"):
            validate_project_aggregate("sum", direction_takes_aggregate=False)

    def test_vector_to_raster_accepts_none(self):
        validate_project_aggregate(None, direction_takes_aggregate=False)

    def test_every_other_direction_requires_an_aggregate(self):
        with pytest.raises(AggregateError, match="requires an aggregate"):
            validate_project_aggregate(None, direction_takes_aggregate=True)

    def test_every_other_direction_accepts_count_by_value_even_though_not_decomposable(self):
        validate_project_aggregate("count_by_value", direction_takes_aggregate=True)


class TestVectorToRasterBroadcastAndNodata:
    """Two dimension combinations (`calendar_year` 2030/2040) pivoted into two bands,
    broadcast onto a 2x2 crosswalk with one nodata pixel.

    zones: [[10, 20], [-1, 10]] -- pixel (1,0) belongs to no source cell.
    source values: cell 10 -> (2030: 1.0, 2040: 10.0); cell 20 -> (2030: 2.0, 2040: 20.0).
    Expected, row-major: b0 (2030) = [1.0, 2.0, None, 1.0]; b1 (2040) = [10.0, 20.0,
    None, 10.0] -- the nodata pixel is null on EVERY band, not just one.
    """

    _ZONES = np.array([[10, 20], [-1, 10]], dtype="int32")

    def test_broadcast_and_nodata_propagation(self, crosswalk_path):
        _write_crosswalk(crosswalk_path, self._ZONES)
        lf = pl.DataFrame(
            {
                "grid_cell": [10, 10, 20, 20],
                "calendar_year": [2030, 2040, 2030, 2040],
                "value": [1.0, 10.0, 2.0, 20.0],
            }
        ).lazy()
        frames = {"v": lf}

        out_lf, metadata_out, window = project_vector_to_raster(frames, _vector_op(), _vector_frame_types())
        out = out_lf.collect()

        assert metadata_out["calendar_year"].to_list() == [2030, 2040]
        assert out["b0"].to_list() == [1.0, 2.0, None, 1.0]
        assert out["b1"].to_list() == [10.0, 20.0, None, 10.0]


class TestVectorToRasterMissingCrosswalk:
    def test_raises_not_found(self, crosswalk_path):
        lf = pl.DataFrame({"grid_cell": [1], "value": [1.0]}).lazy()
        frames = {"v": lf}

        with pytest.raises(NotFoundError):
            project_vector_to_raster(frames, _vector_op(), _vector_frame_types())


class TestVectorToRasterWindowNamesCrosswalk:
    def test_window_source_names_the_crosswalk_file(self, crosswalk_path):
        crosswalk_file = _write_crosswalk(crosswalk_path, np.array([[1, 2]], dtype="int32"))
        lf = pl.DataFrame({"grid_cell": [1, 2], "value": [1.0, 2.0]}).lazy()
        frames = {"v": lf}

        _, _, window = project_vector_to_raster(frames, _vector_op(), _vector_frame_types())

        assert any(str(crosswalk_file) in entry for entry in window.source)


class TestVectorToRasterZeroPivotDimensions:
    """A frame with no dimension column besides `grid_cell`/`value` still produces a
    valid single-band frame -- `add_reserved_band_index` (not a plain `.select([])`)
    keeps `metadata_out`'s height meaningful, exactly `group_band_wide`'s own precedent
    for `by == []`. The reserved column is left standing in what this lowering returns,
    matching that same precedent (see `execution/band_ops.py`'s module docstring for why
    it is not dropped here)."""

    def test_single_band_and_reserved_index_column(self, crosswalk_path):
        _write_crosswalk(crosswalk_path, np.array([[1, 2]], dtype="int32"))
        lf = pl.DataFrame({"grid_cell": [1, 2], "value": [5.0, 6.0]}).lazy()
        frames = {"v": lf}

        out_lf, metadata_out, _ = project_vector_to_raster(frames, _vector_op(), _vector_frame_types())
        out = out_lf.collect()

        # `DOMAIN_COLUMN` rides alongside the bands out of every vector-to-raster
        # projection -- see its own comment in `execution/frame.py`.
        assert list(out.columns) == ["b0", DOMAIN_COLUMN]
        assert out["b0"].to_list() == [5.0, 6.0]
        assert metadata_out.columns == [RESERVED_BAND_INDEX_COLUMN]
        assert metadata_out[RESERVED_BAND_INDEX_COLUMN].to_list() == [0]


class TestVectorToRasterMetadataDeterminism:
    """Band order is sorted by dimension value, not by whatever order the input frame's
    rows happen to arrive in -- the input here is deliberately scrambled (2040 before
    2030, cell 20 before cell 10) to make an unsorted-band bug show up as a wrong VALUE on
    the wrong band, not just a coincidentally-right answer. Running the lowering twice
    over the identical (already-scrambled) input checks that nothing about band order or
    values is order-of-arrival-dependent.
    """

    _ZONES = np.array([[10, 20]], dtype="int32")

    def _scrambled_lf(self):
        return pl.DataFrame(
            {
                "grid_cell": [20, 10, 10, 20],
                "calendar_year": [2040, 2030, 2040, 2030],
                "value": [240.0, 100.0, 140.0, 200.0],
            }
        ).lazy()

    def test_band_order_and_values_are_deterministic_across_runs(self, crosswalk_path):
        _write_crosswalk(crosswalk_path, self._ZONES)
        frames = {"v": self._scrambled_lf()}

        out_lf_1, metadata_1, _ = project_vector_to_raster(frames, _vector_op(), _vector_frame_types())
        out_1 = out_lf_1.collect()
        out_lf_2, metadata_2, _ = project_vector_to_raster(frames, _vector_op(), _vector_frame_types())
        out_2 = out_lf_2.collect()

        assert metadata_1["calendar_year"].to_list() == [2030, 2040]
        assert metadata_1.equals(metadata_2)
        assert out_1.equals(out_2)
        # pixel 0 = zone 10, pixel 1 = zone 20.
        assert out_1["b0"].to_list() == [100.0, 200.0]  # 2030
        assert out_1["b1"].to_list() == [140.0, 240.0]  # 2040


class TestVectorToRasterOrderPreservationLargeScale:
    """250,000 pixels, each assigned a globally UNIQUE zone id equal to its row-major
    position + 1 -- with every pixel's expected broadcast value distinct
    (`grid_cell * 2.0`), a join that silently reordered rows (e.g. a plain `.join()`
    instead of `maintain_order="left"`) would almost certainly land a value on the wrong
    pixel somewhere in a quarter million rows, rather than merely on a coincidentally
    inert duplicate.
    """

    _HEIGHT, _WIDTH = 500, 500
    _N = _HEIGHT * _WIDTH

    def test_broadcast_preserves_pixel_row_order_at_scale(self, crosswalk_path):
        zones = np.arange(1, self._N + 1, dtype="int32").reshape(self._HEIGHT, self._WIDTH)
        _write_crosswalk(crosswalk_path, zones, nodata=None)

        grid_cells = pl.arange(1, self._N + 1, eager=True)
        lf = pl.DataFrame({"grid_cell": grid_cells, "value": grid_cells.cast(pl.Float64) * 2.0}).lazy()
        frames = {"v": lf}

        out_lf, metadata_out, _ = project_vector_to_raster(frames, _vector_op(), _vector_frame_types())
        out = out_lf.collect()

        assert out.height == self._N
        expected = (np.arange(1, self._N + 1, dtype="float64")) * 2.0
        assert np.array_equal(out["b0"].to_numpy(), expected)
