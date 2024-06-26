from collections import defaultdict


class BaseGrid:
    """Base class for geographic grids."""

    def __init__(self, geographic_unit, plan_versions, crs="epsg:26915"):
        self.geographic_unit = geographic_unit
        self.plan_versions = plan_versions
        self.crs = crs


class VectorGrid(BaseGrid):
    """Vector grid."""

    grid_type = "vector"


class RasterGrid(BaseGrid):
    """Raster grid."""

    grid_type = "raster"

    def __init__(
        self, geographic_unit, plan_versions, bounds, height, width, crs="epsg:26915"
    ):
        super().__init__(
            geographic_unit=geographic_unit, plan_versions=plan_versions, crs=crs
        )
        self.bounds = bounds
        self.height = height
        self.width = width


GRIDS = defaultdict(dict)


def add_grid(grid):
    """Add a grid to the index."""

    for version in grid.plan_versions:
        GRIDS[grid.geographic_unit][version] = grid


def get_grid(geographic_unit, plan_version):
    """Get the grid for the given geographic unit and plan version."""

    if geographic_unit not in GRIDS:
        return None
    return GRIDS[geographic_unit].get(plan_version)


add_grid(
    RasterGrid(
        geographic_unit="icm_raster",
        plan_versions=["2023.0.0"],
        bounds=(404500, 3187060, 1024720, 3435010),
        height=8285,
        width=20674,
    )
)
add_grid(VectorGrid(geographic_unit="block_group", plan_versions=["2023.0.0"]))
add_grid(VectorGrid(geographic_unit="ecoregion", plan_versions=["2023.0.0"]))
add_grid(VectorGrid(geographic_unit="extraction_point", plan_versions=["2023.0.0"]))
add_grid(VectorGrid(geographic_unit="hydro_compartment", plan_versions=["2023.0.0"]))
add_grid(VectorGrid(geographic_unit="parish", plan_versions=["2023.0.0"]))
