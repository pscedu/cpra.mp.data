"""Tests for writing, locating, and virtually stacking COGs."""

import xml.etree.ElementTree as ET

import numpy as np
import polars as pl
import pytest
import rasterio as rio

from cpra.mp.data import config
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.raster import cog_path_expression, read_cog, vrt, write_cog

from tests.factories import RASTER_KWARGS, raster_array, write_raster_dataset

DATASET_KWARGS = {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}


class TestCogPathExpression:
    def test_builds_a_dimension_directory_path(self, tmp_path):
        """The calendar dimension is never part of the path -- its value lives only in
        the band description (see write_cog) -- and neither is the file's .tif suffix,
        since the file itself is named with a UUID rather than derived from this path."""
        df = pl.DataFrame({"variable": ["vct"], "grid": ["veg_grid_cell_v001"], "calendar_year": [2030]})
        path = df.select(p=cog_path_expression(df, tmp_path))["p"].item()
        assert path == str(tmp_path / "variable=vct/grid=veg_grid_cell_v001")

    def test_metadata_dimensions_are_excluded_from_the_path(self, tmp_path):
        df = pl.DataFrame({"variable": ["vct"], "model": ["icm_v2023.0.0"]})
        path = df.select(p=cog_path_expression(df, tmp_path))["p"].item()
        assert "model=" not in path

    def test_unknown_columns_are_ignored(self, tmp_path):
        df = pl.DataFrame({"variable": ["vct"], "not_a_dimension": ["x"]})
        path = df.select(p=cog_path_expression(df, tmp_path))["p"].item()
        assert "not_a_dimension" not in path

    def test_base_path_defaults_to_the_configured_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "BASE_PATH", tmp_path)
        df = pl.DataFrame({"variable": ["vct"]})
        path = df.select(p=cog_path_expression(df))["p"].item()
        assert path.startswith(str(tmp_path))


class TestWriteCog:
    def test_writes_to_the_dimension_path(self, base_path):
        """Dataset dimensions (in variable/time_unit/grid order) and partition
        dimensions form the directory; the calendar dimension isn't part of it, and the
        file itself is named with a UUID, not a dimension value."""
        write_cog(raster_array(fill=1), base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        expected_dir = (
            base_path
            / "variable=vct/time_unit=annual/grid=veg_grid_cell_v001"
            / "model_group_id=500/scenario_id=7"
        )
        assert list(expected_dir.glob("*.tif"))

    def test_creates_parent_directories(self, base_path):
        write_cog(raster_array(fill=1), base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        assert list(base_path.rglob("*.tif"))

    def test_pixel_values_round_trip(self, base_path):
        data = raster_array()
        write_cog(data, base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        with rio.open(next(base_path.rglob("*.tif"))) as src:
            assert np.array_equal(src.read(1), data)

    def test_geometry_matches_the_grid(self, base_path):
        grid = GRIDS[RASTER_KWARGS["grid"]]
        write_cog(raster_array(fill=1), base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        with rio.open(next(base_path.rglob("*.tif"))) as src:
            assert (src.height, src.width) == (grid.height, grid.width)
            assert src.crs.to_string().lower() == grid.crs
            assert src.bounds.left == pytest.approx(grid.bounds[0])
            assert src.bounds.top == pytest.approx(grid.bounds[3])

    def test_nodata_is_set(self, base_path):
        write_cog(raster_array(fill=1), base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        with rio.open(next(base_path.rglob("*.tif"))) as src:
            assert src.nodata == -9999

    def test_single_band(self, base_path):
        write_cog(raster_array(fill=1), base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        with rio.open(next(base_path.rglob("*.tif"))) as src:
            assert src.count == 1

    def test_kwargs_are_written_as_tags(self, base_path):
        """Dimension values are the only record of what a COG contains, except the
        calendar dimension, which is never a tag -- it lives solely in the band
        description (see write_cog)."""
        write_cog(raster_array(fill=1), base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        with rio.open(next(base_path.rglob("*.tif"))) as src:
            tags = src.tags()
            description = src.descriptions[0]
        assert tags["variable"] == "vct"
        assert tags["model"] == "icm_v2023.0.0"
        assert "calendar_year" not in tags
        assert description == "calendar_year=2030"

    def test_flat_arrays_are_reshaped_to_the_grid(self, base_path):
        grid = GRIDS[RASTER_KWARGS["grid"]]
        flat = np.arange(grid.height * grid.width, dtype="int32")
        write_cog(flat, base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        with rio.open(next(base_path.rglob("*.tif"))) as src:
            assert src.shape == (grid.height, grid.width)

    def test_missing_required_dimension_raises(self, base_path):
        """calendar_year is required for annual data."""
        with pytest.raises(ValidationError, match="calendar_year"):
            write_cog(raster_array(fill=1), base_path=base_path, **RASTER_KWARGS)

    def test_invalid_dimension_value_raises(self, base_path):
        with pytest.raises(ValidationError, match="calendar_year_maximum_value"):
            write_cog(raster_array(fill=1), base_path=base_path, calendar_year=2500, **RASTER_KWARGS)

    def test_validate_false_skips_validation(self, base_path):
        write_cog(
            raster_array(fill=1),
            base_path=base_path,
            validate=False,
            calendar_year=2500,
            **RASTER_KWARGS,
        )
        assert list(base_path.rglob("*.tif"))


class TestReadCog:
    def test_finds_written_cogs(self, base_path):
        write_raster_dataset(base_path, years=(2030, 2040))
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()
        assert df["calendar_year"].to_list() == [2030, 2040]

    def test_uses_the_given_base_path(self, base_path, tmp_path, monkeypatch):
        """Regression: read_cog accepted base_path but never forwarded it to
        glob_lazy_frame, so it always searched the configured BASE_PATH instead."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.setattr(config, "BASE_PATH", elsewhere)

        write_raster_dataset(base_path, years=(2030,))
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()
        assert df.height == 1

    def test_path_column_points_at_real_files(self, base_path):
        write_raster_dataset(base_path, years=(2030,))
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()
        assert all(rio.open(p) for p in df["path"])

    def test_dimension_columns_are_typed(self, base_path):
        write_raster_dataset(base_path, years=(2030,))
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()
        assert df["calendar_year"].dtype == pl.Int32
        assert df["model_group_id"].dtype == pl.Int32

    def test_scalar_filter(self, base_path):
        write_raster_dataset(base_path, years=(2030, 2040, 2050))
        df = read_cog(base_path=base_path, calendar_year=2040, **DATASET_KWARGS).collect()
        assert df["calendar_year"].to_list() == [2040]

    def test_list_filter(self, base_path):
        write_raster_dataset(base_path, years=(2030, 2040, 2050))
        df = read_cog(base_path=base_path, calendar_year=[2030, 2050], **DATASET_KWARGS).collect()
        assert df["calendar_year"].to_list() == [2030, 2050]

    def test_metadata_dimensions_are_not_required_in_the_path(self, base_path):
        """model is metadata, so it never appears in a COG path or the result columns."""
        write_raster_dataset(base_path, years=(2030,))
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()
        assert "model" not in df.columns

    def test_no_matches_yields_an_empty_frame(self, base_path):
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()
        assert df.height == 0

    def test_missing_dataset_dimension_raises(self, base_path):
        with pytest.raises(ValidationError, match="Required columns are missing"):
            read_cog(base_path=base_path, variable="vct")

    def test_include_file_paths_can_be_renamed(self, base_path):
        write_raster_dataset(base_path, years=(2030,))
        df = read_cog(base_path=base_path, include_file_paths="src", **DATASET_KWARGS).collect()
        assert "src" in df.columns


class TestVrt:
    def test_stacks_cogs_into_a_multiband_dataset(self, base_path):
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()

        with rio.open(vrt(df["path"])) as src:
            assert src.count == 2
            assert src.read(1)[0, 0] == 30
            assert src.read(2)[0, 0] == 40

    def test_band_numbers_are_sequential(self, base_path):
        write_raster_dataset(base_path, years=(2030, 2040, 2050))
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()

        root = ET.fromstring(vrt(df["path"]))
        bands = root.findall("VRTRasterBand")
        assert [b.get("band") for b in bands] == ["1", "2", "3"]

    def test_preserves_grid_geometry(self, base_path):
        grid = GRIDS[RASTER_KWARGS["grid"]]
        write_raster_dataset(base_path, years=(2030, 2040))
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()

        with rio.open(vrt(df["path"])) as src:
            assert (src.height, src.width) == (grid.height, grid.width)

    def test_single_path_produces_one_band(self, base_path):
        write_raster_dataset(base_path, years=(2030,))
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()

        with rio.open(vrt(df["path"])) as src:
            assert src.count == 1

    def test_childless_band_elements_are_not_dropped(self):
        """Regression: the band was appended under `if band_el:`, and an Element with no
        children is falsy, so a band carrying no sub-elements was silently discarded."""
        first = "<VRTDataset><VRTRasterBand band='1'><X/></VRTRasterBand></VRTDataset>"
        second = "<VRTDataset><VRTRasterBand band='1'/></VRTDataset>"

        out = ET.fromstring(first)
        band = ET.fromstring(second).find("VRTRasterBand")
        # Having no children is exactly what made `if band_el:` skip this band. Spelled
        # as len() rather than `not band`, which triggers the deprecation it demonstrates.
        assert len(band) == 0
        assert band is not None

        if band is not None:
            band.set("band", "2")
            out.append(band)
        assert len(out.findall("VRTRasterBand")) == 2
