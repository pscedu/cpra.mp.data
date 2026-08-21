"""Tests for the elev unit conversion the long-format migration applies.

The migration is a one-way pass over tens of gigabytes of elevation rasters that also
changes their storage unit -- Float32 meters to Int32 millimeters -- so the conversion is
pinned here at both levels: the VRT rewrite in isolation, and what GDAL actually writes
when it executes that VRT. The end-to-end case is the one that matters, since the rewrite
only *declares* the conversion and GDAL is what performs it.

`scripts/` is not an importable package, so the module is loaded by path.
"""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import rasterio as rio

from cpra.mp.data.grids import GRIDS
from cpra.mp.data.raster import _cog_profile, cog_path_expression, read_cog

from tests.factories import RASTER_KWARGS


def _load_migrate_module():
    path = Path(__file__).parent.parent / "scripts" / "migrate_to_long_format.py"
    spec = importlib.util.spec_from_file_location("migrate_to_long_format", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migrate = _load_migrate_module()

# elev is only ever written on morph_pixel (write validation enforces the pairing), and
# that grid is 8265x20674 -- 170M pixels per band, far too large to write in a test. The
# `small_grid` fixture shrinks it instead; nothing about the unit conversion depends on the
# raster's size.
ELEV_KWARGS = {**RASTER_KWARGS, "variable": "elev", "grid": "morph_pixel_v001"}
DATASET_KWARGS = {k: ELEV_KWARGS[k] for k in ("variable", "grid", "time_unit")}
NODATA = -9999

# Consecutive years, so the two files form one contiguous run and are actually compacted --
# a run already living in a single file is left untouched (see compact_cog).
YEARS = (2030, 2031)

# Chosen for what each one proves: a plain value, a negative, a value whose float32
# representation sits exactly on a rounding tie that the float64 multiply resolves the
# other way, and a value below the millimeter being stored.
VALUES = [1.5, -0.25, 2.0005, 0.0001]


def _write_float_elev_cog(base_path, calendar_year, values):
    """Write a pre-migration elevation COG: Float32, meters, one band.

    Deliberately not ``write_cog``: the point is to reproduce what sits on disk from
    before elev's dtype changed, independent of what the current write path would do.
    """

    grid = GRIDS[ELEV_KWARGS["grid"]]
    df = pl.DataFrame({k: [v] for k, v in ELEV_KWARGS.items()})
    cog_dir = Path(df.select(p=cog_path_expression(df, base_path))["p"].item())
    cog_dir.mkdir(parents=True, exist_ok=True)

    # Everything past the listed values stays nodata, which is also most of the raster --
    # the sentinel has to survive the conversion untouched.
    data = np.full((grid.height, grid.width), NODATA, dtype="float32")
    data.reshape(-1)[: len(values)] = values

    profile = _cog_profile(grid, np.dtype("float32"))
    with rio.open(cog_dir / f"{calendar_year}.tif", "w", **profile) as dst:
        dst.update_tags(**ELEV_KWARGS)
        dst.set_band_description(1, f"calendar_year={calendar_year}")
        dst.write(data, 1)


def _vrt_of(dtype="Float32", nodata=NODATA):
    return (
        f"<VRTDataset rasterXSize='2' rasterYSize='1'>"
        f"<VRTRasterBand dataType='{dtype}' band='1'>"
        f"<Description>calendar_year=2030</Description>"
        f"<NoDataValue>{nodata}</NoDataValue>"
        f"<SimpleSource><SourceFilename>a.tif</SourceFilename></SimpleSource>"
        f"</VRTRasterBand></VRTDataset>"
    )


class TestVrtTransformFor:
    def test_elev_gets_the_millimeter_conversion(self):
        assert migrate._vrt_transform_for("elev") is migrate._elev_mm_vrt

    def test_every_other_variable_is_copied_unchanged(self):
        """A `None` transform is what leaves compact_cog doing its plain streaming copy."""
        assert migrate._vrt_transform_for("vct") is None
        assert migrate._vrt_transform_for("lnd_type") is None


class TestElevMmVrt:
    def test_retypes_bands_to_int32(self):
        """CreateCopy takes its output type from the VRT, so the band's declared
        `dataType` -- not the write profile -- is what makes the output Int32."""
        out, dtype = migrate._elev_mm_vrt(_vrt_of(), "float32")
        assert dtype == "int32"
        assert ET.fromstring(out).find("VRTRasterBand").get("dataType") == "Int32"

    def test_promotes_the_source_and_scales_it(self):
        """`ScaleRatio` only exists on a ComplexSource, so the promotion is required."""
        out, _ = migrate._elev_mm_vrt(_vrt_of(), "float32")
        band = ET.fromstring(out).find("VRTRasterBand")
        assert band.find("SimpleSource") is None
        assert band.find("ComplexSource").findtext("ScaleRatio") == "1000"

    def test_excludes_the_bands_own_nodata_from_the_scaling(self):
        """Read off the band rather than assumed, so whatever GDAL was told to treat as
        nodata on the way in is what gets excluded."""
        out, _ = migrate._elev_mm_vrt(_vrt_of(nodata=-5555), "float32")
        assert ET.fromstring(out).find("VRTRasterBand").find("ComplexSource").findtext("NODATA") == "-5555"

    def test_preserves_band_descriptions(self):
        """Band descriptions carry the calendar year -- the only record of which year a
        band holds, so dropping them would make the compacted COG unreadable."""
        out, _ = migrate._elev_mm_vrt(_vrt_of(), "float32")
        assert ET.fromstring(out).find("VRTRasterBand").findtext("Description") == "calendar_year=2030"

    def test_a_source_gdal_already_made_complex_is_scaled_in_place(self):
        """gdalbuildvrt emits a ComplexSource itself for sources that need one; it already
        supports ScaleRatio, so it is scaled rather than promoted -- and its existing
        children are replaced rather than duplicated."""
        vrt_text = _vrt_of().replace("SimpleSource", "ComplexSource").replace(
            "</ComplexSource>", "<NODATA>-1</NODATA></ComplexSource>"
        )
        source = ET.fromstring(migrate._elev_mm_vrt(vrt_text, "float32")[0]).find("VRTRasterBand/ComplexSource")
        assert [child.text for child in source.findall("NODATA")] == [str(NODATA)]
        assert [child.text for child in source.findall("ScaleRatio")] == ["1000"]

    @pytest.mark.parametrize("dtype", ["int32", "int16", "uint8"])
    def test_an_integer_source_is_left_alone(self, dtype):
        """Idempotency: a resumed or re-run migration reads already-converted Int32
        rasters, and scaling those again would multiply elevations by a million."""
        vrt_text = _vrt_of(dtype="Int32")
        assert migrate._elev_mm_vrt(vrt_text, dtype) == (vrt_text, dtype)


@pytest.fixture
def small_grid(monkeypatch):
    """Shrink elev's grid to a handful of pixels for the duration of one test.

    Only the pixel dimensions change, so the profile stays internally consistent (the
    transform is derived from the same bounds, height, and width) -- the pixels are just
    absurdly large.
    """

    grid = GRIDS[ELEV_KWARGS["grid"]]
    monkeypatch.setattr(grid, "height", 4)
    monkeypatch.setattr(grid, "width", 8)
    return grid


@pytest.fixture
def migrated(base_path, tmp_path, small_grid):
    """Migrate two Float32-meter elevation COGs and return the multiband COG's path."""

    for year in YEARS:
        _write_float_elev_cog(base_path, year, VALUES)
    output_path = tmp_path / "data-long"

    result, _ = migrate.migrate_raster_dataset(
        {k: ELEV_KWARGS[k] for k in ("variable", "time_unit", "grid", "model_group_id", "scenario_id")},
        base_path=base_path,
        output_path=output_path,
    )
    assert result["files"] == len(YEARS)
    paths = set(read_cog(base_path=output_path, **DATASET_KWARGS).collect()["path"])
    assert len(paths) == 1, f"expected one compacted COG, got {sorted(paths)}"
    return paths.pop()


class TestElevMigration:
    """The conversion as GDAL actually performs it, end to end through the migration."""

    def test_writes_int32(self, migrated):
        with rio.open(migrated) as src:
            assert src.dtypes[0] == "int32"
            assert src.count == len(YEARS)

    def test_converts_meters_to_millimeters(self, migrated):
        with rio.open(migrated) as src:
            band = src.descriptions.index(f"calendar_year={YEARS[0]}") + 1
            written = src.read(band).reshape(-1)[: len(VALUES)]
        # 2.0005 m is 2000.49996... mm in float64 and rounds down; in float32 it would be
        # exactly 2000.5 and round to 2001, so this pins the arithmetic's precision too.
        assert list(written) == [1500, -250, 2000, 0]

    def test_preserves_the_nodata_sentinel(self, migrated):
        """Without `<NODATA>` the sentinel is scaled along with real values, landing at
        -9999000 and no longer matching the nodata the band declares -- so every nodata
        pixel would read back as a real elevation 9.999 km below the datum."""
        with rio.open(migrated) as src:
            assert src.nodata == NODATA
            data = src.read(1).reshape(-1)
        assert (data[len(VALUES) :] == NODATA).all()

    def test_leaves_the_source_rasters_untouched(self, migrated, base_path):
        """The migration writes into a separate tree; until its output is verified, the
        Float32 originals are still the only copy of the data."""
        sources = list(base_path.rglob("*.tif"))
        assert len(sources) == len(YEARS)
        for path in sources:
            with rio.open(path) as src:
                assert src.dtypes[0] == "float32"
                assert src.read(1).reshape(-1)[0] == pytest.approx(VALUES[0])
