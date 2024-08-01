import polars as pl
from collections import defaultdict
from cpra.mp.data.structure import read_structure_csv


class BaseGrid:
    """Base class for geographic grids."""

    def __init__(self, geographic_unit, plan_versions, crs="epsg:26915"):
        self.geographic_unit = geographic_unit
        self.plan_versions = plan_versions
        self.crs = crs


class VectorGrid(BaseGrid):
    """Vector grid."""

    grid_type = "vector"

    def __init__(
        self, geographic_unit, plan_versions, geographies=None, crs="epsg:26915"
    ):
        super().__init__(
            geographic_unit=geographic_unit, plan_versions=plan_versions, crs=crs
        )
        if isinstance(geographies, int):
            self.geographies = list(range(1, geographies + 1))
        elif geographies:
            self.geographies = list(geographies)
        else:
            self.geographies = []


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


def get_grid(geographic_unit, plan_version):
    """Get the grid for the given geographic unit and plan version."""

    if geographic_unit not in GRIDS:
        return None
    return GRIDS[geographic_unit].get(plan_version)


for row in (
    read_structure_csv("grids")
    .with_columns(
        pl.col("plan_versions").str.split(", "),
        pl.col("bounds").str.split(", ").cast(pl.List(pl.Int32)),
        pl.col("height", "width", "geographies").cast(pl.Int32),
    )
    .iter_rows(named=True)
):
    if row["type"] == "raster":
        grid = RasterGrid(
            geographic_unit=row["geographic_unit"],
            plan_versions=row["plan_versions"],
            bounds=row["bounds"],
            height=row["height"],
            width=row["width"],
            crs=row["crs"],
        )
    else:
        grid = VectorGrid(
            geographic_unit=row["geographic_unit"],
            plan_versions=row["plan_versions"],
            geographies=row["geographies"],
            crs=row["crs"],
        )

    for version in grid.plan_versions:
        GRIDS[grid.geographic_unit][version] = grid
