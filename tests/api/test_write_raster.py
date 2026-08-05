"""Tests for the two raster download paths in api.dataset.

``write_raster`` streams a band-separate VRT through GDAL (``_write_raster_vrt``) for
selections that map 1:1 to output bands, and falls back to windowed NumPy reads
(``_write_raster_windowed``) for aggregated ones. The VRT path is several times faster and
lighter, so it handles every non-aggregated download -- these tests pin it to producing
exactly what the windowed path produces, since nothing else guards that equivalence.

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

from cpra.mp.data.api.dataset import Dataset
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.raster import _cog_profile, compact_cog, scale_vrt_bands, vrt, write_cog

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


def _index_frame(cog_path, bands, multiplier=1):
    """The materialized frame write_raster consumes: one row per output band."""

    with rasterio.open(cog_path) as src:
        years = [int(desc.split("=")[1]) for desc in src.descriptions]
    return pl.DataFrame(
        {
            "path": [str(cog_path)] * len(bands),
            "band": list(bands),
            "calendar_year": [years[band - 1] for band in bands],
            "multiplier": pl.Series([multiplier] * len(bands), dtype=pl.Int32),
        }
    )


def _dataset(df, raster_groups=None, group_aggregate=None):
    """A Dataset with only the state the write paths touch.

    ``Dataset.__init__`` builds a LazyFrame from on-disk data and a grid definition, none of
    which these paths use -- they read ``materialize()``, ``raster_groups`` and
    ``options.group_aggregate`` only.
    """

    dataset = object.__new__(Dataset)
    dataset.options = SimpleNamespace(group_aggregate=group_aggregate)
    dataset.materialize = lambda: df
    dataset.raster_groups = raster_groups
    return dataset


def _write(dataset, out_path, **kwargs):
    return asyncio.run(dataset.write_raster(str(out_path), **kwargs))


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
    assert _write(_dataset(df), tmp_path / "vrt.tif")
    dataset = _dataset(df)
    assert asyncio.run(dataset._write_raster_windowed(df, str(tmp_path / "windowed.tif")))
    _assert_same_raster(tmp_path / "vrt.tif", tmp_path / "windowed.tif")


def test_vrt_matches_windowed_path_with_multiplier(multiband_cog, tmp_path):
    """A ScaleRatio in the VRT matches the NumPy multiply it replaces."""

    df = _index_frame(multiband_cog, (1, 2), multiplier=10)
    assert _write(_dataset(df), tmp_path / "vrt.tif")
    dataset = _dataset(df)
    assert asyncio.run(dataset._write_raster_windowed(df, str(tmp_path / "windowed.tif")))
    _assert_same_raster(tmp_path / "vrt.tif", tmp_path / "windowed.tif")

    with rasterio.open(tmp_path / "vrt.tif") as src:
        np.testing.assert_array_equal(src.read(1), _band_data(YEARS[0]) * 10)


def test_vrt_applies_compression_options(multiband_cog, tmp_path):
    """Compression options from the request reach the output unchanged."""

    df = _index_frame(multiband_cog, (1, 2))
    assert _write(_dataset(df), tmp_path / "out.tif", compress="deflate", zlevel=9)
    with rasterio.open(tmp_path / "out.tif") as src:
        assert src.profile["compress"] == "deflate"


def test_vrt_writes_no_pam_sidecar(multiband_cog, tmp_path):
    """GDAL_PAM_ENABLED=NO keeps a .aux.xml from being orphaned beside the response file."""

    df = _index_frame(multiband_cog, (1, 2))
    out_path = tmp_path / "out.tif"
    assert _write(_dataset(df), out_path)
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
            "multiplier": pl.Series([1], dtype=pl.Int32),
        }
    )
    out_path = tmp_path / "out.tif"
    assert _write(_dataset(df), out_path)
    with rasterio.open(out_path) as src:
        assert src.descriptions == ("calendar_year=2019",)


def test_aggregated_selection_uses_windowed_path(multiband_cog, tmp_path, monkeypatch):
    """Aggregation has no bit-identical GDAL equivalent, so it must not take the VRT path."""

    def fail(*args, **kwargs):
        raise AssertionError("aggregated selections must not use the VRT path")

    monkeypatch.setattr(Dataset, "_write_raster_vrt", fail)

    df = _index_frame(multiband_cog, (1, 2, 3, 4))
    raster_groups = pl.LazyFrame({"row_idxs": [list(range(len(df)))]})
    dataset = _dataset(df, raster_groups=raster_groups, group_aggregate="mean")
    out_path = tmp_path / "out.tif"
    assert _write(dataset, out_path)

    with rasterio.open(out_path) as src:
        assert src.count == 1
        expected = np.mean([_band_data(year) for year in YEARS], axis=0)
        np.testing.assert_array_equal(src.read(1), expected.astype("float32"))


def test_kill_switch_falls_back_to_windowed_path(multiband_cog, tmp_path, monkeypatch):
    """RASTER_USE_VRT_COPY=False restores the previous behavior without a code change."""

    def fail(*args, **kwargs):
        raise AssertionError("VRT path should be disabled")

    monkeypatch.setattr(Dataset, "_write_raster_vrt", fail)
    monkeypatch.setattr("cpra.mp.data.api.dataset.RASTER_USE_VRT_COPY", False)

    df = _index_frame(multiband_cog, (1, 2))
    out_path = tmp_path / "out.tif"
    assert _write(_dataset(df), out_path)
    with rasterio.open(out_path) as src:
        assert src.count == 2


def test_empty_selection_returns_false(tmp_path):
    """An empty selection is what drives the endpoint's 404, before either path runs."""

    df = pl.DataFrame(
        schema={"path": pl.String, "band": pl.Int64, "calendar_year": pl.Int32, "multiplier": pl.Int32}
    )
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
    assert _write(_dataset(df), out_dir / "out.tif")
    assert sorted(p.name for p in out_dir.iterdir()) == ["out.tif"]


def test_index_frame_band_order_is_preserved(multiband_cog, tmp_path):
    """Output band N must come from the Nth selected row, not the file's own band order."""

    bands = (4, 1, 3)
    df = _index_frame(multiband_cog, bands)
    out_path = tmp_path / "out.tif"
    assert _write(_dataset(df), out_path)
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
            "multiplier": pl.Series([1] * len(rows), dtype=pl.Int32),
        }
    )

    assert _write(_dataset(df), tmp_path / "vrt.tif")
    dataset = _dataset(df)
    assert asyncio.run(dataset._write_raster_windowed(df, str(tmp_path / "windowed.tif")))
    _assert_same_raster(tmp_path / "vrt.tif", tmp_path / "windowed.tif")
