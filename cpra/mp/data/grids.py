import polars as pl
from collections import defaultdict
from cpra.mp.data.structure import read_structure_csv


class BaseGrid:
    """Base class for geographic grids."""

    def __init__(self, name, label, crs="epsg:26915"):
        self.name = name
        self.label = label
        self.crs = crs


class VectorGrid(BaseGrid):
    """Vector grid."""

    grid_type = "vector"

    def __init__(
        self, name, label, id_min=None, id_max=None, id_count=None, crs="epsg:26915"
    ):
        super().__init__(name=name, label=label, crs=crs)
        self.id_min = id_min
        self.id_max = id_max
        self.id_count = id_count


class RasterGrid(BaseGrid):
    """Raster grid."""

    grid_type = "raster"

    def __init__(self, name, label, bounds, height, width, crs="epsg:26915"):
        super().__init__(name=name, label=label, crs=crs)
        self.bounds = bounds
        self.height = height
        self.width = width


GRIDS = {}


for row in (
    read_structure_csv("grids")
    .with_columns(
        pl.col("bounds").str.split(", ").cast(pl.List(pl.Int32)),
        pl.col("height", "width", "id_min", "id_max", "id_count").cast(pl.Int32),
    )
    .iter_rows(named=True)
):
    if row["type"] == "raster":
        grid = RasterGrid(
            name=row["name"],
            label=row["label"],
            bounds=row["bounds"],
            height=row["height"],
            width=row["width"],
            crs=row["crs"],
        )
    else:
        grid = VectorGrid(
            name=row["name"],
            label=row["label"],
            id_min=row["id_min"],
            id_max=row["id_max"],
            id_count=row["id_count"],
            crs=row["crs"],
        )

    GRIDS[grid.name] = grid
