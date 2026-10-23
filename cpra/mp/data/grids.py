
import polars as pl

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


class CoordinateGrid(BaseGrid):
    """A grid whose geographies are coordinates themselves.

    Not one of the model's grids, and so not in `GRIDS`: it divides nothing up, holds no
    data, and has no set of geographies to enumerate. It stands in for a grid where an API
    needs to name where a bare coordinate lives — converting geographies from it means
    converting the coordinates given, rather than the cells of some grid containing them.
    """

    grid_type = "coordinate"


# The coordinates the API speaks everywhere else in, named as a grid so that they can be
# converted from.
WGS84 = CoordinateGrid(name="wgs84", label="Coordinates", crs="epsg:4326")

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
