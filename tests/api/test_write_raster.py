"""Tests for the two raster download paths in analysis.raster.

``write_raster`` either streams a band-separate VRT through GDAL (``_write_raster_vrt``) or
reads windows into NumPy (``_write_raster_windowed``). The two agree pixel for pixel and are
within noise of each other on time, so the dispatcher picks on memory -- the windowed path's
buffer grows with band count while the VRT's footprint is roughly flat. Nothing else guards
that equivalence, so most of these tests drive one path explicitly and compare.

All data is written under the tmp_path-based base_path fixture (see conftest.py) -- never
the real data/ directory (see repo CLAUDE.md).
"""

import asyncio
import xml.etree.ElementTree
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import rasterio
from rasterio.shutil import copy as rio_copy

from cpra.mp.data.analysis.request.expressions import compile_calculate_entries
from cpra.mp.data.analysis.raster import (
    RasterGridAnalysis,
    band_column_names,
    flush_calculate_plans,
)
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.raster import (
    DECADE_GROUP_YEARS,
    PENTAD_GROUP_YEARS,
    _cog_profile,
    _period_subgroups,
    compact_cog,
    scale_vrt_bands,
    vrt,
    write_cog,
)

RASTER_KWARGS = dict(
    variable="hsi",
    grid="veg_grid_cell_v001",
    time_unit="annual",
    model_group_id=500,
    scenario_id=7,
    species_code="allig",
    model="icm_v2023.0.0",
)
YEARS = (2019, 2020, 2021, 2022)


def _band_data(year):
    """Distinct, non-uniform pixels per year, so band mix-ups can't pass unnoticed."""

    grid = GRIDS[RASTER_KWARGS["grid"]]
    ramp = np.arange(grid.height * grid.width, dtype="float32").reshape(grid.height, grid.width)
    return (ramp % 97) + (year - YEARS[0]) * 1000.5


@pytest.fixture
def multiband_cog(base_path):
    """One multiband COG covering YEARS, built the way the migration builds them."""

    for year in YEARS:
        write_cog(_band_data(year), base_path=base_path, catalog_path=None, calendar_year=year, **RASTER_KWARGS)
    outputs = compact_cog(base_path=base_path, catalog_path=None, **RASTER_KWARGS)["outputs"]
    assert len(outputs) == 1, "expected the contiguous run to merge into a single file"
    return outputs[0]


def _index_frame(cog_path, bands):
    """The materialized frame write_raster consumes: one row per output band."""

    with rasterio.open(cog_path) as src:
        years = [int(desc.split("=")[1]) for desc in src.descriptions]
    return pl.DataFrame(
        {
            "path": [str(cog_path)] * len(bands),
            "band": list(bands),
            "calendar_year": [years[band - 1] for band in bands],
        }
    )


def _scale_plans(factor, where=None):
    """Calculate plans multiplying every (or the `where`-selected) band by `factor` --
    the validated form a `calculate` op's lowering records for the equivalent
    request."""

    return compile_calculate_entries([("calculate.1", f"value * {factor}", where)])


def _dataset(df, group_by=None, group_aggregate=None, calculate_plans=None, value_where=None):
    """A RasterGridAnalysis with only the state the write paths touch.

    ``RasterGridAnalysis.__init__`` builds a LazyFrame from on-disk data and a grid definition, none of
    which these paths use -- they read ``lf``/``materialize()``, ``_raw_band_frame``,
    ``_band_ops``, ``_calculate_plans``, ``options.group_aggregate``, and ``_value_where``
    only. ``value_where=None`` matches ``BaseAnalysis.__init__``'s own default, so every
    caller that doesn't pass it gets the no-filter behavior.

    ``group_by`` is a ``GroupOp``'s own key -- a list of ``df`` column names, empty for
    "collapse every band into one", or ``None`` for an ungrouped request. When it is given,
    ``_band_ops`` is built the way ``collapse_band_group`` builds it -- any
    ``calculate_plans`` flushed in first, then the group -- since that ordered list is what
    the write path replays. Ungrouped, the plans stay pending and the write path's own flush
    positions them.
    """

    dataset = object.__new__(RasterGridAnalysis)
    dataset.options = SimpleNamespace(group_aggregate=group_aggregate)
    dataset.lf = df.lazy()
    dataset.materialize = lambda: df
    dataset._raw_band_frame = df
    dataset._band_ops = []
    dataset._calculate_plans = calculate_plans or []
    dataset._value_where = value_where
    if group_by is not None:
        flush_calculate_plans(dataset._band_ops, dataset._calculate_plans)
        dataset._band_ops.append(("group", list(group_by), group_aggregate))
    return dataset


def _write(dataset, out_path, **kwargs):
    """Through the dispatcher, so the path is whichever write_raster chooses."""

    return asyncio.run(dataset.write_raster(str(out_path), **kwargs))


def _write_vrt(dataset, df, out_path, **kwargs):
    """The VRT path directly.

    Called explicitly rather than via the dispatcher because these fixtures are small
    enough that write_raster routes them to the windowed path on memory grounds (see
    RasterGridAnalysis._windowed_read_is_lighter), which would make VRT assertions vacuous.
    """

    return asyncio.run(dataset._write_raster_vrt(df, str(out_path), **kwargs))


def _write_windowed(dataset, df, out_path, **kwargs):
    return asyncio.run(dataset._write_raster_windowed(df, str(out_path), **kwargs))


def _describe(path):
    with rasterio.open(path) as src:
        return dict(
            data=src.read(),
            descriptions=src.descriptions,
            nodata=src.nodata,
            crs=src.crs,
            transform=src.transform,
            count=src.count,
            dtype=src.dtypes[0],
        )


def _assert_same_raster(vrt_path, windowed_path):
    produced = _describe(vrt_path)
    expected = _describe(windowed_path)
    assert produced["count"] == expected["count"]
    assert produced["dtype"] == expected["dtype"]
    assert produced["descriptions"] == expected["descriptions"]
    assert produced["nodata"] == expected["nodata"]
    assert produced["crs"] == expected["crs"]
    assert produced["transform"] == expected["transform"]
    np.testing.assert_array_equal(produced["data"], expected["data"])


@pytest.mark.parametrize("bands", [(1,), (2,), (1, 3), tuple(range(1, len(YEARS) + 1))])
def test_vrt_matches_windowed_path(multiband_cog, tmp_path, bands):
    """Selecting any subset of bands produces identical rasters either way."""

    df = _index_frame(multiband_cog, bands)
    assert _write_vrt(_dataset(df), df, tmp_path / "vrt.tif")
    assert _write_windowed(_dataset(df), df, tmp_path / "windowed.tif")
    _assert_same_raster(tmp_path / "vrt.tif", tmp_path / "windowed.tif")


def test_calculate_entries_apply_on_the_windowed_path(multiband_cog, tmp_path):
    """A calculate entry's rewrite reaches every written pixel (the successor to the old
    per-band VRT ScaleRatio, which cannot express a general expression)."""

    df = _index_frame(multiband_cog, (1, 2))
    out_path = tmp_path / "out.tif"
    assert _write_windowed(_dataset(df, calculate_plans=_scale_plans(10)), df, out_path)

    with rasterio.open(out_path) as src:
        np.testing.assert_array_equal(src.read(1), _band_data(YEARS[0]) * 10)
        np.testing.assert_array_equal(src.read(2), _band_data(YEARS[1]) * 10)


def test_calculate_selection_is_dispatched_to_windowed_path(multiband_cog, tmp_path, monkeypatch):
    """Any calculate entry forces the windowed path: a general expression (and the NoData
    mask-then-restore contract) has no VRT spelling.

    The buffer limit is raised first so the dispatcher would otherwise choose the VRT
    path -- without that this would pass whether or not the calculate routing works.
    """

    monkeypatch.setattr("cpra.mp.data.analysis.raster.RASTER_WINDOW_BUFFER_LIMIT_MB", 0)

    def fail(*args, **kwargs):
        raise AssertionError("a selection with calculate entries must not use the VRT path")

    monkeypatch.setattr(RasterGridAnalysis, "_write_raster_vrt", fail)

    df = _index_frame(multiband_cog, (1, 2))
    out_path = tmp_path / "out.tif"
    assert _write(_dataset(df, calculate_plans=_scale_plans(10)), out_path)
    with rasterio.open(out_path) as src:
        np.testing.assert_array_equal(src.read(1), _band_data(YEARS[0]) * 10)


def test_vrt_applies_compression_options(multiband_cog, tmp_path):
    """Compression options from the request reach the output unchanged."""

    df = _index_frame(multiband_cog, (1, 2))
    assert _write_vrt(_dataset(df), df, tmp_path / "out.tif", compress="deflate", zlevel=9)
    with rasterio.open(tmp_path / "out.tif") as src:
        assert src.profile["compress"] == "deflate"


def test_vrt_writes_no_pam_sidecar(multiband_cog, tmp_path):
    """GDAL_PAM_ENABLED=NO keeps a .aux.xml from being orphaned beside the response file."""

    df = _index_frame(multiband_cog, (1, 2))
    out_path = tmp_path / "out.tif"
    assert _write_vrt(_dataset(df), df, out_path)
    assert not list(tmp_path.glob("*.aux.xml"))


def test_vrt_labels_bands_from_legacy_source(base_path, tmp_path):
    """A legacy COG stores its calendar value as a dataset tag and has no band description,
    so the VRT has nothing to inherit -- the output must still be labeled."""

    grid = GRIDS[RASTER_KWARGS["grid"]]
    directory = base_path / "legacy"
    directory.mkdir(parents=True)
    legacy_path = directory / "legacy.tif"
    with rasterio.open(legacy_path, "w", **_cog_profile(grid, "float32")) as dst:
        dst.update_tags(calendar_year=2019, **{k: v for k, v in RASTER_KWARGS.items() if k != "grid"})
        dst.write(_band_data(2019), 1)

    with rasterio.open(legacy_path) as src:
        assert src.descriptions == (None,), "fixture should have no band description"

    df = pl.DataFrame(
        {
            "path": [str(legacy_path)],
            "band": [1],
            "calendar_year": [2019],
        }
    )
    out_path = tmp_path / "out.tif"
    assert _write_vrt(_dataset(df), df, out_path)
    with rasterio.open(out_path) as src:
        assert src.descriptions == ("calendar_year=2019",)


def _aggregated(df, aggregate, groups=None):
    """An aggregated dataset over ``df``.

    ``groups``, when given, is a list of row-position lists -- one per intended output
    band. A `GroupOp` keys its groups by a DIMENSION COLUMN, not by row position, so an
    arbitrary grouping is expressed here by materializing exactly that column: each row
    gets the index of the group it belongs to, and the group runs on it. ``None`` means the
    whole selection collapses into one band, which is `group_by=[]` -- a real request shape
    (``group_aggregate`` with no ``group_by``), not a synthetic one.
    """
    if groups is None:
        return _dataset(df, group_by=[], group_aggregate=aggregate)
    group_of = {row: index for index, rows in enumerate(groups) for row in rows}
    keyed = df.with_columns(_group=pl.Series([group_of[row] for row in range(len(df))], dtype=pl.UInt32))
    return _dataset(keyed, group_by=["_group"], group_aggregate=aggregate)


def test_aggregated_selection_uses_windowed_path(multiband_cog, tmp_path, monkeypatch):
    """Aggregates must not take the VRT path.

    A VRTDerivedRasterBand re-reads every contributing band per output chunk, which for a
    PIXEL-interleaved multiband source decompresses each tile once per band: 983s vs 80s
    reducing 52 elev bands, at any swath size. GDAL's builtins are the more accurate
    reduction, but not at that cost.
    """

    def fail(*args, **kwargs):
        raise AssertionError("aggregated selections must not use the VRT path")

    monkeypatch.setattr(RasterGridAnalysis, "_write_raster_vrt", fail)

    df = _index_frame(multiband_cog, (1, 2, 3, 4))
    out_path = tmp_path / "out.tif"
    assert _write(_aggregated(df, "mean"), out_path)
    with rasterio.open(out_path) as src:
        assert src.count == 1


@pytest.mark.parametrize("aggregate", ["mean", "sum", "min", "max"])
def test_aggregate_reduces_to_one_band_per_group(multiband_cog, tmp_path, aggregate):
    """Each aggregate reduces its group, accumulating at the input's own precision.

    Mirrors the implementation's float32 accumulation rather than asserting a float64
    result: ``np.mean`` over a float32 array accumulates in float32 and drifts slightly
    with band count. Pinned so a change of accumulator width can't pass unnoticed.
    """

    df = _index_frame(multiband_cog, tuple(range(1, len(YEARS) + 1)))
    out_path = tmp_path / "out.tif"
    assert _write(_aggregated(df, aggregate), out_path)

    stacked = np.stack([_band_data(year) for year in YEARS])
    expected = getattr(np, aggregate)(stacked, axis=0).astype("float32")
    with rasterio.open(out_path) as src:
        assert src.count == 1
        np.testing.assert_array_equal(src.read(1), expected)


def test_aggregate_produces_one_band_per_group_in_order(multiband_cog, tmp_path):
    """Several groups reduce independently, in group order."""

    df = _index_frame(multiband_cog, tuple(range(1, len(YEARS) + 1)))
    groups = [[0, 1], [2, 3]]
    out_path = tmp_path / "out.tif"
    assert _write(_aggregated(df, "sum", groups=groups), out_path)

    with rasterio.open(out_path) as src:
        assert src.count == 2
        for out_band, group in enumerate(groups, start=1):
            expected = np.sum([_band_data(YEARS[i]) for i in group], axis=0).astype("float32")
            np.testing.assert_array_equal(src.read(out_band), expected)


def test_aggregated_bands_are_unlabeled(multiband_cog, tmp_path):
    """An aggregate spans several calendar values, so it gets no single-value label."""

    df = _index_frame(multiband_cog, (1, 2, 3))
    out_path = tmp_path / "out.tif"
    assert _write(_aggregated(df, "mean"), out_path)
    with rasterio.open(out_path) as src:
        assert src.descriptions == (None,)


def test_aggregate_applies_calculate_before_reducing(multiband_cog, tmp_path):
    """Calculate-then-reduce, not reduce-then-calculate."""

    df = _index_frame(multiband_cog, (1, 2))
    out_path = tmp_path / "out.tif"
    dataset = _aggregated(df, "sum")
    dataset._calculate_plans = _scale_plans(10)
    assert _write(dataset, out_path)

    expected = np.sum([_band_data(year) * 10 for year in YEARS[:2]], axis=0).astype("float32")
    with rasterio.open(out_path) as src:
        np.testing.assert_array_equal(src.read(1), expected)


@pytest.fixture
def nodata_cog(base_path):
    """A multiband COG with a fully-nodata pixel and a partially-nodata one.

    Fully-nodata pixels are the realistic case: nodata marks ground outside the model
    domain, which doesn't move between years, so it is nodata in every band.
    """

    grid = GRIDS[RASTER_KWARGS["grid"]]
    for i, year in enumerate(YEARS):
        data = np.full((grid.height, grid.width), 5.0, dtype="float32")
        data[0, 0] = -9999.0  # nodata in every band
        if i == 0:
            data[0, 1] = -9999.0  # nodata in the first band only
        write_cog(data, base_path=base_path, catalog_path=None, calendar_year=year, **RASTER_KWARGS)
    outputs = compact_cog(base_path=base_path, catalog_path=None, **RASTER_KWARGS)["outputs"]
    return outputs[0]


def test_aggregate_excludes_nodata_from_the_reduction(nodata_cog, tmp_path):
    """Nodata pixels are excluded from a group_aggregate collapse, matching the summarize
    path's exclusion semantics (see ``raster_source.horizontal_aggregate_expr``).

    This used to be pinned here as a KNOWN QUIRK the other direction: the windowed writer's
    earlier NumPy reduce collapsed with plain ``np.<fn>`` over the raw window, so -9999 was
    treated as an ordinary value -- a pixel nodata in only some bands produced an arbitrary
    blend with the real bands instead of their mean/sum, and ``sum`` over a fully-nodata
    pixel produced ``-9999 * n`` rather than any kind of no-data marker. The windowed
    writer now masks each band's own raw pixel value against the source's nodata sentinel
    before the multiply/collapse step (mirroring the summarize path's own nodata-to-null
    exclusion, see ``raster_source.scan_rasters``'s ``mask_nodata``), and, when a pixel has
    no contributing band at all, writes the nodata sentinel itself rather than NaN -- an
    integer output band cannot hold NaN the way the summarize path's Float64 result can, so
    this is a deliberate difference from that path's NaN marker, not an inconsistency.
    """

    df = _index_frame(nodata_cog, tuple(range(1, len(YEARS) + 1)))
    n = len(YEARS)

    mean_path = tmp_path / "mean.tif"
    assert _write(_aggregated(df, "mean"), mean_path)
    with rasterio.open(mean_path) as src:
        band = src.read(1)
        assert src.nodata == -9999.0
        # Nodata in every band: no band contributes, so the sentinel is written directly.
        assert band[0, 0] == -9999.0
        # Nodata in one band only: excluded, so the mean is exactly the real bands' mean.
        assert band[0, 1] == pytest.approx(5.0)
        assert band[1, 1] == pytest.approx(5.0)

    sum_path = tmp_path / "sum.tif"
    assert _write(_aggregated(df, "sum"), sum_path)
    with rasterio.open(sum_path) as src:
        band = src.read(1)
        # Fully-nodata pixel: no contributor, so the sentinel is written directly.
        assert band[0, 0] == pytest.approx(-9999.0)
        # Nodata in one band only: excluded from the sum of the remaining n - 1 bands.
        assert band[0, 1] == pytest.approx(5.0 * (n - 1))


def test_non_aggregated_output_preserves_nodata(nodata_cog, tmp_path):
    """Without aggregation, nodata pixels pass through untouched."""

    df = _index_frame(nodata_cog, (1, 2))
    out_path = tmp_path / "out.tif"
    assert _write_vrt(_dataset(df), df, out_path)
    with rasterio.open(out_path) as src:
        assert src.nodata == -9999.0
        assert src.read(1)[0, 0] == -9999.0
        assert src.read(1)[0, 1] == -9999.0, "nodata in band 1 only"
        assert src.read(2)[0, 1] == 5.0, "band 2 has real data at that pixel"


def test_calculate_never_rewrites_nodata_on_the_windowed_path(nodata_cog, tmp_path):
    """NoData in, NoData out: a non-aggregated write with calculate entries masks the
    sentinel before the entries run and restores it afterwards, so `value * 10` cannot
    turn -9999 into -99990 -- while every real pixel is rewritten."""

    df = _index_frame(nodata_cog, (1, 2))
    out_path = tmp_path / "out.tif"
    assert _write_windowed(_dataset(df, calculate_plans=_scale_plans(10)), df, out_path)
    with rasterio.open(out_path) as src:
        assert src.nodata == -9999.0
        assert src.read(1)[0, 0] == -9999.0, "nodata stays the sentinel, not sentinel * 10"
        assert src.read(1)[0, 1] == -9999.0, "nodata in band 1 only"
        assert src.read(1)[1, 1] == 50.0, "real pixels are rewritten"
        assert src.read(2)[0, 1] == 50.0, "band 2 has real data at that pixel"


def test_integer_aggregate_truncates_toward_zero(base_path, tmp_path):
    """KNOWN QUIRK, pinned deliberately: integer aggregates truncate rather than round.

    ``np.mean`` returns a float and rasterio's float->int32 conversion on write truncates
    toward zero, so a mean of exactly 1.5 writes 1. This biases positive integer aggregates
    downward by up to 1 per pixel. (GDAL's pixel functions round half away from zero, but
    the reduction can't move to GDAL for performance reasons -- see
    test_aggregated_selection_uses_windowed_path.)
    """

    grid = GRIDS[RASTER_KWARGS["grid"]]
    for year, value in zip((2019, 2020), (1, 2)):
        write_cog(
            np.full((grid.height, grid.width), value, dtype="int32"),
            base_path=base_path, catalog_path=None, calendar_year=year, **RASTER_KWARGS,
        )
    cog_path = compact_cog(base_path=base_path, catalog_path=None, **RASTER_KWARGS)["outputs"][0]

    df = _index_frame(cog_path, (1, 2))
    out_path = tmp_path / "out.tif"
    assert _write(_aggregated(df, "mean"), out_path)
    with rasterio.open(out_path) as src:
        assert src.dtypes[0] == "int32"
        assert src.read(1)[0, 0] == 1, "mean of 1 and 2 truncates to 1 rather than rounding to 2"


def test_kill_switch_falls_back_to_windowed_path(multiband_cog, tmp_path, monkeypatch):
    """RASTER_USE_VRT_COPY=False restores the previous behavior without a code change.

    The buffer limit is dropped to 0 first, so the dispatcher would otherwise choose the VRT
    path -- without that this would pass whether or not the switch works.
    """

    monkeypatch.setattr("cpra.mp.data.analysis.raster.RASTER_WINDOW_BUFFER_LIMIT_MB", 0)

    def fail(*args, **kwargs):
        raise AssertionError("VRT path should be disabled")

    monkeypatch.setattr(RasterGridAnalysis, "_write_raster_vrt", fail)
    monkeypatch.setattr("cpra.mp.data.analysis.raster.RASTER_USE_VRT_COPY", False)

    df = _index_frame(multiband_cog, (1, 2))
    out_path = tmp_path / "out.tif"
    assert _write(_dataset(df), out_path)
    with rasterio.open(out_path) as src:
        assert src.count == 2


def test_narrow_selection_is_dispatched_to_windowed_path(multiband_cog, tmp_path, monkeypatch):
    """A few bands means a small NumPy buffer, so the windowed path is the lighter choice."""

    def fail(*args, **kwargs):
        raise AssertionError("a narrow selection should not use the VRT path")

    monkeypatch.setattr(RasterGridAnalysis, "_write_raster_vrt", fail)

    df = _index_frame(multiband_cog, (1, 2))
    out_path = tmp_path / "out.tif"
    assert _write(_dataset(df), out_path)
    with rasterio.open(out_path) as src:
        assert src.count == 2


def test_wide_selection_is_dispatched_to_vrt_path(multiband_cog, tmp_path, monkeypatch):
    """Once the windowed buffer would exceed the limit, the VRT copy is the lighter choice."""

    # The fixture grid is tiny, so lower the limit rather than build a huge raster.
    monkeypatch.setattr("cpra.mp.data.analysis.raster.RASTER_WINDOW_BUFFER_LIMIT_MB", 0)

    def fail(*args, **kwargs):
        raise AssertionError("a wide selection should not use the windowed path")

    monkeypatch.setattr(RasterGridAnalysis, "_write_raster_windowed", fail)

    df = _index_frame(multiband_cog, (1, 2))
    out_path = tmp_path / "out.tif"
    assert _write(_dataset(df), out_path)
    with rasterio.open(out_path) as src:
        assert src.count == 2


class TestValueFilterGate:
    """The value-filter gate: the compiler-side decision that a raster-OUTPUT value
    `where` must lower to a per-cell mask, never a row `.filter()`, exercised through the
    REAL write path (`RasterGridAnalysis._write_raster_windowed`) -- not just the
    `apply_value_filter_band_wide` primitive already covered on its own
    (`tests/api/test_raster_ops.py::TestApplyValueFilterBandWide`) or the `scan_rasters`
    mechanics already characterized directly
    (`tests/api/test_raster_source.py::TestValueFilterPixelCountPreservation`).

    No production caller reaches `_value_where` on a raster grid through this path today
    (`_apply_transform_params_impl` rejects a standalone value `where` on a raster grid
    unless `summarize_by`/`summarize_aggregate` are also given) -- these tests drive
    `_write_raster_windowed` directly via `_dataset(..., value_where=...)`, the same
    bypass-``__init__`` pattern every other test in this module already uses for
    `_calculate_plans`/`_band_ops`.
    """

    def test_value_filter_masks_in_place_without_misaligning_the_grid(self, multiband_cog, tmp_path):
        """Every pixel that fails the predicate becomes the nodata sentinel AT ITS OWN
        POSITION; every pixel that passes keeps its original value AT ITS OWN POSITION --
        proven against a hand-computed expectation, not merely "the write didn't crash".
        This is the pixel-count/position-preservation assertion wired to the real write
        path."""

        # `_band_data`'s ramp (0..96) is offset by 1000.5 per later year, so the two bands
        # occupy disjoint value ranges -- a single threshold expression (applied
        # identically to every selected band, since a standalone value `where` has no
        # per-band awareness) can only discriminate within one of them. A single-band
        # selection keeps the discrimination check unambiguous; band 1 (year 2019, ramp
        # 0..96, no offset) is exercised on its own.
        df = _index_frame(multiband_cog, (1,))
        out_path = tmp_path / "out.tif"
        threshold = 48.0
        dataset = _dataset(df, value_where=f"value > {threshold}")
        assert _write_windowed(dataset, df, out_path)

        with rasterio.open(out_path) as src:
            assert src.nodata == -9999.0
            expected = _band_data(YEARS[0])
            written = src.read(1)
            keep = expected > threshold
            assert 0 < keep.sum() < keep.size, "the threshold must actually discriminate"
            np.testing.assert_array_equal(written[keep], expected[keep])
            assert (written[~keep] == -9999.0).all()

    def test_value_filter_composes_with_group_aggregate(self, multiband_cog, tmp_path):
        """The value-filter mask and `group_aggregate`'s horizontal collapse compose
        correctly: a masked-out pixel drops out of the reduction exactly like a nodata
        pixel already does (`horizontal_aggregate_expr`'s `no_contributor` branch), and a
        pixel with no surviving band writes the sentinel rather than NaN. The band
        collapse is exercised together with the value filter to prove the two compose
        rather than only checking each alone."""

        df = _index_frame(multiband_cog, tuple(range(1, len(YEARS) + 1)))
        out_path = tmp_path / "out.tif"
        threshold = 50.0
        dataset = _aggregated(df, "mean")
        dataset._value_where = f"value > {threshold}"
        assert _write(dataset, out_path)

        stacked = np.stack([_band_data(year) for year in YEARS])
        masked = np.where(stacked > threshold, stacked, np.nan)
        no_contributor = np.isnan(masked).all(axis=0)
        with np.errstate(invalid="ignore"):
            expected_mean = np.nanmean(masked, axis=0).astype("float32")
        expected = np.where(no_contributor, np.float32(-9999.0), expected_mean)

        with rasterio.open(out_path) as src:
            assert src.count == 1
            np.testing.assert_array_equal(src.read(1), expected)

    def test_reintroducing_a_bare_filter_fails_loudly_instead_of_misaligning(
        self, multiband_cog, tmp_path, monkeypatch
    ):
        """Guards the gate's design, not just today's code: if the value-filter lowering
        `apply_band_ops` replays for the write path were ever changed from
        `apply_value_filter_band_wide` (mask) to a bare `.filter()`,
        `_take_window_rows`'s rolling flush would come up short for some window -- a
        dropped pixel is a dropped ROW, and every later window needs exactly
        `height * width` rows to reshape into -- and raise `RuntimeError` rather than
        silently writing a misaligned grid. Proves the design fails safe, rather than
        merely observing that today's lowering avoids the mistake."""

        import cpra.mp.data.analysis.raster as raster_module

        def bare_filter(lf, metadata_df, value_where, extra_columns=()):
            return lf.filter(pl.col(band_column_names(metadata_df)[0]) > 0), metadata_df

        monkeypatch.setattr(raster_module, "apply_value_filter_band_wide", bare_filter)

        df = _index_frame(multiband_cog, (1, 2))
        out_path = tmp_path / "out.tif"
        # Pixel 0 of band 1 (year 2019, ramp index 0) is exactly 0 -- `> 0` is False there,
        # so the bare filter above drops at least that one row, guaranteeing a deficit.
        dataset = _dataset(df, value_where="value > 0")
        with pytest.raises(RuntimeError, match="scan_rasters produced only"):
            _write_windowed(dataset, df, out_path)

    def test_dispatcher_routes_a_value_filtered_selection_to_the_windowed_path(self, multiband_cog, tmp_path):
        """The VRT path has no mask hook at all, so `write_raster`'s dispatcher must not
        route a value-filtered selection there -- doing so would silently drop the filter
        instead of applying (or rejecting) it."""

        df = _index_frame(multiband_cog, (1, 2))
        out_path = tmp_path / "out.tif"
        dataset = _dataset(df, value_where="value > 50")

        def fail(*args, **kwargs):
            raise AssertionError("a value-filtered selection must not use the VRT path")

        dataset._write_raster_vrt = fail
        assert _write(dataset, out_path)


@pytest.fixture
def tile_interleaved_cog(base_path, multiband_cog):
    """The multiband COG re-encoded INTERLEAVE=TILE, a COG-driver-only layout."""

    out_path = base_path / "tile_interleaved.tif"
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        rio_copy(
            str(multiband_cog), str(out_path), driver="COG", INTERLEAVE="TILE",
            COMPRESS="ZSTD", LEVEL=6, BLOCKSIZE=256,
        )
    with rasterio.open(out_path) as src:
        assert src.profile["interleave"] == "tile", "fixture should be tile interleaved"
    return out_path


@pytest.mark.parametrize("writer", ["vrt", "windowed"])
def test_tile_interleaved_source_is_downloadable(tile_interleaved_cog, tmp_path, writer):
    """A TILE-interleaved source must not break either write path.

    INTERLEAVE=TILE is accepted by the COG driver but rejected by the GTiff driver used for
    downloads, so inheriting the source's interleave made such a source fail outright with
    "INTERLEAVE=tile unsupported". The output interleave is set explicitly instead.
    """

    df = _index_frame(tile_interleaved_cog, (1, 2))
    out_path = tmp_path / "out.tif"
    write = _write_vrt if writer == "vrt" else _write_windowed
    assert write(_dataset(df), df, out_path)

    with rasterio.open(out_path) as src:
        assert src.count == 2
        assert src.profile["interleave"] == "pixel"
        np.testing.assert_array_equal(src.read(1), _band_data(YEARS[0]))


def test_multiband_output_is_pixel_interleaved(multiband_cog, tmp_path):
    """Pixel interleaving is what keeps a multiband response small, so it is not inherited."""

    df = _index_frame(multiband_cog, (1, 2, 3))
    for writer in (_write_vrt, _write_windowed):
        out_path = tmp_path / f"{writer.__name__}.tif"
        assert writer(_dataset(df), df, out_path)
        with rasterio.open(out_path) as src:
            assert src.profile["interleave"] == "pixel"


def test_empty_selection_returns_false(tmp_path):
    """An empty selection is what drives the endpoint's 404, before either path runs."""

    df = pl.DataFrame(schema={"path": pl.String, "band": pl.Int64, "calendar_year": pl.Int32})
    out_path = tmp_path / "out.tif"
    assert _write(_dataset(df), out_path) is False
    assert not out_path.exists()


def test_scale_vrt_bands_only_promotes_scaled_bands(multiband_cog):
    """ComplexSource is a slower per-pixel path, so unscaled bands stay SimpleSource."""

    sources = [(str(multiband_cog), 1), (str(multiband_cog), 2)]
    root = xml.etree.ElementTree.fromstring(scale_vrt_bands(vrt(sources), [1, 10]))
    band_els = root.findall("VRTRasterBand")
    assert band_els[0].find("SimpleSource") is not None
    assert band_els[0].find("ComplexSource") is None
    assert band_els[1].find("SimpleSource") is None
    assert band_els[1].find("ComplexSource/ScaleRatio").text == "10"


def test_vrt_path_leaves_no_temporary_files(multiband_cog, tmp_path):
    """The throwaway .vrt lives in a TemporaryDirectory, not beside the output."""

    # Its own directory, since tmp_path also holds the base_path fixture's tree.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    df = _index_frame(multiband_cog, (1, 2))
    assert _write_vrt(_dataset(df), df, out_dir / "out.tif")
    assert sorted(p.name for p in out_dir.iterdir()) == ["out.tif"]


def test_index_frame_band_order_is_preserved(multiband_cog, tmp_path):
    """Output band N must come from the Nth selected row, not the file's own band order."""

    bands = (4, 1, 3)
    df = _index_frame(multiband_cog, bands)
    out_path = tmp_path / "out.tif"
    assert _write_vrt(_dataset(df), df, out_path)
    with rasterio.open(out_path) as src:
        assert src.descriptions == tuple(f"calendar_year={YEARS[band - 1]}" for band in bands)
        for out_band, source_band in enumerate(bands, start=1):
            np.testing.assert_array_equal(src.read(out_band), _band_data(YEARS[source_band - 1]))


def test_paths_agree_across_multiple_source_files(base_path, tmp_path):
    """Selections spanning two files (a gap ends a run, so both can exist) must still agree."""

    for year in (2019, 2020, 2030, 2031):
        write_cog(_band_data(year), base_path=base_path, catalog_path=None, calendar_year=year, **RASTER_KWARGS)
    outputs = compact_cog(base_path=base_path, catalog_path=None, **RASTER_KWARGS)["outputs"]
    assert len(outputs) == 2, "a calendar gap should produce two runs"

    rows = []
    for path in sorted(outputs):
        with rasterio.open(path) as src:
            for band, desc in enumerate(src.descriptions, start=1):
                rows.append((str(path), band, int(desc.split("=")[1])))
    df = pl.DataFrame(
        {
            "path": [r[0] for r in rows],
            "band": [r[1] for r in rows],
            "calendar_year": [r[2] for r in rows],
        }
    )

    assert _write_vrt(_dataset(df), df, tmp_path / "vrt.tif")
    assert _write_windowed(_dataset(df), df, tmp_path / "windowed.tif")
    _assert_same_raster(tmp_path / "vrt.tif", tmp_path / "windowed.tif")


# --- compact_cog run-grouping (decade default / pentad / unbounded) -------------------

# A 28-year span straddling three decades, so folding the short edge groups into their
# neighbors (see _period_subgroups) leaves two full-size decade groups rather than one of
# each size -- a case that actually exercises both edges of the fold, unlike a span short
# enough to collapse into a single group.
GROUPING_YEARS = list(range(2016, 2044))


@pytest.mark.parametrize(
    "values,period,expected",
    [
        ([2019, 2020, 2021, 2022], 10, [[0, 1, 2, 3]]),  # shorter than one period: one group
        (list(range(2019, 2029)), 10, [list(range(10))]),  # exactly one period: one group
        # A lone leading year folds forward into its decade rather than standing alone.
        (list(range(2019, 2031)), 10, [list(range(12))]),
        # A lone trailing year folds backward into its decade rather than standing alone.
        (list(range(2020, 2032)), 10, [list(range(12))]),
        # Full interior periods stay their own groups; only the run's own edges fold.
        (
            list(range(2016, 2044)),
            10,
            [list(range(14)), list(range(14, 28))],
        ),
    ],
)
def test_period_subgroups(values, period, expected):
    """Pure boundary/folding cases for the grouping helper, without writing any COGs."""

    assert _period_subgroups(values, period) == expected


def test_compact_cog_defaults_to_decade_grouping(base_path):
    """Without an explicit group_years, a multi-decade run splits into decade-sized COGs."""

    for year in GROUPING_YEARS:
        write_cog(_band_data(year), base_path=base_path, catalog_path=None, calendar_year=year, **RASTER_KWARGS)
    outputs = compact_cog(base_path=base_path, catalog_path=None, **RASTER_KWARGS)["outputs"]

    band_counts = sorted(rasterio.open(p).count for p in outputs)
    assert band_counts == [14, 14], "28 years over three decades should fold into two 14-year groups"

    for path in outputs:
        with rasterio.open(path) as src:
            years = [int(desc.split("=")[1]) for desc in src.descriptions]
            for band, year in enumerate(years, start=1):
                np.testing.assert_array_equal(src.read(band), _band_data(year))


def test_compact_cog_pentad_grouping(base_path):
    """group_years=PENTAD_GROUP_YEARS produces finer, differently-folded groups."""

    for year in GROUPING_YEARS:
        write_cog(_band_data(year), base_path=base_path, catalog_path=None, calendar_year=year, **RASTER_KWARGS)
    outputs = compact_cog(
        base_path=base_path, catalog_path=None, group_years=PENTAD_GROUP_YEARS, **RASTER_KWARGS
    )["outputs"]

    band_counts = sorted(rasterio.open(p).count for p in outputs)
    assert band_counts == [5, 5, 9, 9], "pentad grouping should differ from the decade default"


def test_compact_cog_regroup_at_a_different_period_does_not_lose_data(base_path):
    """Regression: re-grouping already-grouped COGs at a different group_years must not
    drop bands.

    A source file's bands can straddle more than one new subgroup (e.g. a 14-band decade
    file re-split into two pentads), so "does this subgroup already live in exactly one
    file" is not enough to call it optimal -- that file may still owe bands to a
    *different* subgroup. Checking only per-subgroup file count previously mis-classified
    both halves of a split file as already optimal and skipped rewriting either one,
    silently leaving the never-produced pentad file's years unrepresented anywhere.
    """

    for year in GROUPING_YEARS:
        write_cog(_band_data(year), base_path=base_path, catalog_path=None, calendar_year=year, **RASTER_KWARGS)
    decade_outputs = compact_cog(base_path=base_path, catalog_path=None, **RASTER_KWARGS)["outputs"]
    assert len(decade_outputs) == 2

    pentad_outputs = compact_cog(
        base_path=base_path, catalog_path=None, group_years=PENTAD_GROUP_YEARS, **RASTER_KWARGS
    )["outputs"]

    remaining = sorted(base_path.glob("**/*.tif"))
    assert sorted(remaining) == sorted(pentad_outputs), "old decade files must not survive the re-group"

    seen_years = {}
    for path in remaining:
        with rasterio.open(path) as src:
            for band, desc in enumerate(src.descriptions, start=1):
                year = int(desc.split("=")[1])
                assert year not in seen_years, f"year {year} duplicated across output files"
                seen_years[year] = src.read(band)
    assert sorted(seen_years) == GROUPING_YEARS
    for year, band in seen_years.items():
        np.testing.assert_array_equal(band, _band_data(year))


def test_compact_cog_group_years_none_restores_unbounded_run(base_path):
    """group_years=None disables the split, matching the pre-grouping behavior."""

    for year in GROUPING_YEARS:
        write_cog(_band_data(year), base_path=base_path, catalog_path=None, calendar_year=year, **RASTER_KWARGS)
    outputs = compact_cog(base_path=base_path, catalog_path=None, group_years=None, **RASTER_KWARGS)["outputs"]

    assert len(outputs) == 1
    with rasterio.open(outputs[0]) as src:
        assert src.count == len(GROUPING_YEARS)


def test_decade_group_years_constant_is_ten():
    """Pinned so a change to the default grouping period is a deliberate edit, not silent."""

    assert DECADE_GROUP_YEARS == 10
    assert PENTAD_GROUP_YEARS == 5


def test_write_cog_compresses_with_zstd_and_a_predictor(base_path):
    """The codec `_cog_profile` gives every raster write, asserted on the file itself rather
    than on the profile dict: ZSTD with PREDICTOR, which is what makes integer bands delta-
    encode well. Nothing else in this module checks the codec `write_cog` actually lands.
    """

    write_cog(_band_data(2019), base_path=base_path, catalog_path=None, calendar_year=2019, **RASTER_KWARGS)

    (cog_path,) = base_path.glob("**/*.tif")
    with rasterio.open(cog_path) as src:
        assert src.profile["compress"] == "zstd"
        assert src.tags(ns="IMAGE_STRUCTURE").get("PREDICTOR") is not None
