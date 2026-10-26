"""Tests for the grid registry built from grids.csv."""

import pytest

from cpra.mp.data.grids import GRIDS, RasterGrid, VectorGrid


class TestRegistry:
    def test_names_match_keys(self):
        for name, grid in GRIDS.items():
            assert grid.name == name

    def test_grid_classes_match_grid_type(self):
        for grid in GRIDS.values():
            expected = RasterGrid if grid.grid_type == "raster" else VectorGrid
            assert isinstance(grid, expected), grid.name

    def test_grid_type_is_a_class_attribute(self):
        """Dispatch in read_data/write_data reads grid_type off the class."""
        assert VectorGrid.grid_type == "vector"
        assert RasterGrid.grid_type == "raster"

    def test_default_crs(self):
        assert all(grid.crs.startswith("epsg:") for grid in GRIDS.values())


class TestVectorGrid:
    def test_hydro_compartment_bounds(self):
        grid = GRIDS["hydro_compartment_v001"]
        assert (grid.id_min, grid.id_max, grid.id_count) == (1, 1657, 1657)

    def test_ecoregion_bounds(self):
        grid = GRIDS["ecoregion_v001"]
        assert (grid.id_min, grid.id_max, grid.id_count) == (1, 32, 32)

    def test_grids_without_id_bounds_parse_as_none(self):
        """Empty CSV cells become None, not 0 -- write_parquet skips those checks."""
        grid = GRIDS["block_group_v2020"]
        assert grid.id_min is None
        assert grid.id_max is None
        assert grid.id_count is None

    def test_vector_grids_have_no_raster_geometry(self):
        grid = GRIDS["ecoregion_v001"]
        assert not hasattr(grid, "bounds")
        assert not hasattr(grid, "height")


class TestRasterGrid:
    def test_bounds_parsed_into_four_ints(self):
        grid = GRIDS["veg_grid_cell_v001"]
        assert grid.bounds == [404710, 3199480, 909670, 3374680]

    def test_dimensions(self):
        grid = GRIDS["veg_grid_cell_v001"]
        assert (grid.height, grid.width) == (365, 1052)

    def test_raster_grids_have_no_id_bounds(self):
        grid = GRIDS["morph_pixel_v001"]
        assert not hasattr(grid, "id_min")
        assert not hasattr(grid, "id_count")

    @pytest.mark.parametrize("name", [n for n, g in GRIDS.items() if g.grid_type == "raster"])
    def test_transform_is_derivable(self, name):
        """write_cog builds a transform from bounds/width/height, so it must be valid."""
        from rasterio.transform import from_bounds

        grid = GRIDS[name]
        transform = from_bounds(*grid.bounds, width=grid.width, height=grid.height)
        assert transform.a > 0  # positive pixel width
        assert transform.e < 0  # north-up rasters have negative pixel height
