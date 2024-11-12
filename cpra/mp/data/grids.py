import polars as pl
from collections import defaultdict
from cpra.mp.data.structure import read_structure_csv


class BaseGrid:
    """Base class for geographic grids."""

    def __init__(self, name, crs="epsg:26915"):
        self.name = name
        self.crs = crs


class VectorGrid(BaseGrid):
    """Vector grid."""

    grid_type = "vector"

    def __init__(self, name, geographies=None, crs="epsg:26915"):
        super().__init__(name=name, crs=crs)
        if isinstance(geographies, int):
            self.geographies = list(range(1, geographies + 1))
        elif geographies:
            self.geographies = list(geographies)
        else:
            self.geographies = []


class RasterGrid(BaseGrid):
    """Raster grid."""

    grid_type = "raster"

    def __init__(self, name, bounds, height, width, crs="epsg:26915"):
        super().__init__(name=name, crs=crs)
        self.bounds = bounds
        self.height = height
        self.width = width


GRIDS = {}


for row in (
    read_structure_csv("grids")
    .with_columns(
        pl.col("bounds").str.split(", ").cast(pl.List(pl.Int32)),
        pl.col("height", "width", "geographies").cast(pl.Int32),
    )
    .iter_rows(named=True)
):
    if row["type"] == "raster":
        grid = RasterGrid(
            name=row["name"],
            bounds=row["bounds"],
            height=row["height"],
            width=row["width"],
            crs=row["crs"],
        )
    else:
        grid = VectorGrid(
            name=row["name"],
            geographies=row["geographies"],
            crs=row["crs"],
        )

    GRIDS[grid.name] = grid
