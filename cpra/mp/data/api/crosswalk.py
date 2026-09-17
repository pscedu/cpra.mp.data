"""Convert grid cells from one grid to another.

The conversions come from the crosswalk files in `config.CROSSWALKS_PATH`: a Parquet table
of `src_id`/`dst_id`/`scale_factor` rows for each vector-to-vector pair, and a zone-ID
GeoTIFF for each raster-to-vector pair (the same files `Dataset`'s `summarize_by` uses).
The two conversions those files don't cover are derived from the grids themselves —
vector to raster from the source polygon's point on surface, and raster to raster from the
coordinate itself, which names a location rather than a cell and so needs no translation.

The `wgs84` pseudo-grid adds a fifth conversion, and is the one source a caller can name that
isn't a grid at all: its grid cells are the request's own coordinates, converted to a vector
grid's cells from that grid's polygons. It answers the same question a raster source does,
but from the polygons rather than from a rasterization of them — so a coordinate crosswalks
to the grid cell whose data the data API selects for it, both having resolved it the same way.

The result keeps the crosswalk tables' own columns, plus a `dst_grid` column naming the
grid each row converts to, since one response can carry several destinations. A grid cell
is spelled by ID on a vector grid and by coordinate on a raster one, so each side gets a
column for both spellings and fills in the one its grid uses — which keeps every column's
type the same no matter which grids a request names.
"""

import asyncio

import polars as pl
import rasterio as rio
from fastapi import HTTPException

from cpra.mp.data import config
from cpra.mp.data.analysis.io.geography import (
    parse_coordinate_pairs,
    project_coordinates,
    representative_points,
    resolve_vector_grid_cell,
)
from cpra.mp.data.analysis.io.raster_io import unpack_values
from cpra.mp.data.grids import GRIDS, WGS84

# Every conversion outside the vector-to-vector crosswalk tables maps a grid cell to
# exactly one other grid cell, which is what the tables spell as a whole scale factor.
WHOLE = 1

# The grid types whose grid cells are named by coordinate rather than by ID: a raster
# grid, whose cells have no IDs to name them by, and the WGS84 pseudo-grid, whose
# grid cells are the coordinates.
COORDINATE_GRID_TYPES = {"raster", WGS84.grid_type}

# The columns naming the two grid cells a row converts between, in output order. A
# conversion populates only the ones its grids use and leaves the rest null.
GRID_CELL_COLUMNS = {
    "src_id": pl.Int64,
    "src_lng": pl.Float64,
    "src_lat": pl.Float64,
    "dst_id": pl.Int64,
    "dst_lng": pl.Float64,
    "dst_lat": pl.Float64,
}


def _columns(rows, width):
    """Transpose rows into columns, keeping the column count when there are no rows."""
    return [list(column) for column in zip(*rows)] if rows else [[] for _ in range(width)]


class Crosswalk:
    """Convert one grid's grid cells into one or more other grids' grid cells."""

    def __init__(self, source_grid: str, grid_cell: str | None = None, to_grid: list[str] | None = None):
        self.grid = WGS84 if source_grid == WGS84.name else GRIDS[source_grid]
        self.grid_cell = grid_cell
        self.destination_grids = self._get_destination_grids(to_grid)

        # Exactly one of these is filled in by `_resolve_source`, according to how the
        # source grid identifies a grid cell: a vector grid by ID, a raster or WGS84 grid
        # by coordinate. A vector grid may also leave `source_ids` None, meaning every
        # grid cell in the grid.
        self.source_ids = None
        self.source_coordinates = None

        # Read once for the whole request, since the points depend only on the source grid.
        self._representative_points = None

    def _get_destination_grids(self, to_grid):
        """Get the grids to convert to, defaulting to every grid reachable from the source.

        A destination is reachable if a crosswalk file names it, and — from a raster source
        grid — if it is any other raster grid, since those conversions need no file.
        """
        if to_grid:
            return list(dict.fromkeys(to_grid))

        names = set()
        for pattern in [f"{self.grid.name}__*.parquet", f"{self.grid.name}__*.tif"]:
            for path in config.CROSSWALKS_PATH.glob(pattern):
                names.add(path.stem.split("__", 1)[1])

        if self.grid.grid_type == "raster":
            names |= {name for name, grid in GRIDS.items() if grid.grid_type == "raster"}

        if self.grid is WGS84:
            # A coordinate reaches every grid that can say where it falls: a vector grid
            # whose polygons are on hand to contain it, and every raster grid, each of
            # which holds a value at the coordinate itself.
            names |= {path.stem for path in config.GRIDS_PATH.glob("*.fgb")}
            names |= {name for name, grid in GRIDS.items() if grid.grid_type == "raster"}

        names &= set(GRIDS)
        names -= {self.grid.name}
        if not names:
            raise HTTPException(status_code=400, detail=f"No crosswalks from grid: {self.grid.name}")

        return sorted(names)

    def _crosswalk_path(self, dst_grid_name, extension):
        """Locate the crosswalk file to a destination grid, reporting a missing one as a
        400 the way `summarize_by` does."""
        crosswalk_path = config.CROSSWALKS_PATH / f"{self.grid.name}__{dst_grid_name}.{extension}"
        if not crosswalk_path.exists():
            msg = f"No crosswalk from {self.grid.name} to {dst_grid_name}"
            raise HTTPException(status_code=400, detail=msg)
        return crosswalk_path

    async def _resolve_source(self):
        """Parse the grid_cell parameter into source grid cell IDs or coordinate pairs.

        A vector grid accepts either, on the same terms as the data API's grid_cell
        filter: IDs only when every value is a plain unsigned integer, otherwise
        coordinate pairs resolved to the grid cells containing them. A raster or WGS84
        grid accepts only coordinate pairs, and requires them — neither has an enumerable
        set of grid cells to convert in their absence.
        """
        if self.grid.grid_type in COORDINATE_GRID_TYPES and not self.grid_cell:
            msg = f"grid_cell is required for {self.grid.grid_type} grid: {self.grid.name}"
            raise HTTPException(status_code=400, detail=msg)

        if not self.grid_cell:
            return

        values = unpack_values(self.grid_cell, "grid_cell")

        if self.grid.grid_type == "vector" and all(value.isdigit() for value in values):
            self.source_ids = list(dict.fromkeys(int(value) for value in values))
            return

        lngs, lats = parse_coordinate_pairs(values)
        if self.grid.grid_type in COORDINATE_GRID_TYPES:
            self.source_coordinates = list(dict.fromkeys(zip(lngs, lats)))
            return

        resolved = await resolve_vector_grid_cell(self.grid.name, lngs, lats)
        self.source_ids = list(
            dict.fromkeys(grid_cell_id for grid_cell_id in resolved if grid_cell_id is not None)
        )

    def _frame(self, **columns):
        """Build a crosswalk frame from the grid cell columns a conversion populates.

        Only the vector-to-vector tables record a scale factor; every other conversion maps
        a grid cell to exactly one counterpart, which is a whole one.
        """
        length = len(next(iter(columns.values())))
        return pl.DataFrame(
            {**columns, "scale_factor": [WHOLE] * length},
            schema_overrides={name: GRID_CELL_COLUMNS[name] for name in columns},
        )

    def _normalize(self, df):
        """Add the grid cell columns a conversion left out, so that every conversion —
        and so every response — has the same columns with the same types.

        The scale factor is widened rather than passed through as stored: crosswalk tables
        size it to the splits they record, whole in some and fractional in others, and a
        float holds either without rounding one to the other.
        """
        missing = [
            pl.lit(None, dtype=dtype).alias(name)
            for name, dtype in GRID_CELL_COLUMNS.items()
            if name not in df.columns
        ]
        return df.with_columns(missing).select(
            *GRID_CELL_COLUMNS, pl.col("scale_factor").cast(pl.Float64)
        )

    def _vector_to_vector(self, dst_grid_name):
        """Convert vector grid cells with a crosswalk table, returned as it is stored.

        A grid cell that the table splits across several destinations keeps one row per
        destination, with the table's own scale factors.
        """
        crosswalk_path = self._crosswalk_path(dst_grid_name, "parquet")
        lf = pl.scan_parquet(str(crosswalk_path))
        if self.source_ids is not None:
            lf = lf.filter(pl.col("src_id").is_in(self.source_ids))

        # Crosswalk tables size their ID columns to the grids they were built from, so
        # widen them to one type before several destinations are stacked together. The
        # scale factor is passed through as stored, whole or fractional.
        df = lf.select(
            pl.col("src_id").cast(pl.Int64),
            pl.col("dst_id").cast(pl.Int64),
            pl.col("scale_factor"),
        ).collect()

        if self.source_ids is None:
            return df.sort("src_id", "dst_id")

        # Report the grid cells in the order they were asked for, which the crosswalk
        # table's own ordering (by destination) doesn't preserve.
        order = pl.DataFrame({"src_id": self.source_ids}, schema={"src_id": pl.Int64})
        return order.with_row_index("src_idx").join(df, on="src_id").sort("src_idx", "dst_id").drop("src_idx")

    def _vector_to_raster(self):
        """Convert vector grid cells to the coordinates of a point on each one's surface.

        A raster grid has no cell IDs, so a vector grid cell converts to a location inside
        it. Which raster grid is being converted to doesn't enter into it: the point comes
        from the source polygon, and every raster grid covers it at a different resolution.
        That's why the points are read once, before the destinations fan out.
        """
        points = self._representative_points
        source_ids = self.source_ids if self.source_ids is not None else sorted(points)
        matched = [
            (grid_cell_id, *points[grid_cell_id]) for grid_cell_id in source_ids if grid_cell_id in points
        ]
        src_ids, lngs, lats = _columns(matched, 3)
        return self._frame(src_id=src_ids, dst_lng=lngs, dst_lat=lats)

    def _raster_to_vector(self, dst_grid_name):
        """Convert coordinates to the grid cells containing them, by sampling the zone-ID
        crosswalk raster at each one. Coordinates outside every zone sample as nodata."""
        crosswalk_path = self._crosswalk_path(dst_grid_name, "tif")
        lngs, lats = zip(*self.source_coordinates)

        with rio.open(crosswalk_path) as crosswalk:
            xs, ys = project_coordinates(list(lngs), list(lats), crosswalk.crs)
            samples = list(crosswalk.sample(zip(xs, ys), indexes=1))
            nodata = crosswalk.nodata

        matched = [
            (lng, lat, int(sample[0]))
            for (lng, lat), sample in zip(self.source_coordinates, samples)
            if sample[0] != nodata
        ]
        src_lngs, src_lats, dst_ids = _columns(matched, 3)
        return self._frame(src_lng=src_lngs, src_lat=src_lats, dst_id=dst_ids)

    async def _wgs84_to_vector(self, dst_grid_name):
        """Convert coordinates to the grid cells containing them, from the destination
        grid's own polygons.

        This is the same point-in-polygon lookup the data API's grid_cell filter makes, so
        a coordinate converts to the grid cell whose data it selects. Sampling a zone
        raster the way a raster source does answers from a rasterization of those polygons
        instead, which puts the boundaries within a pixel of where they are and leaves the
        polygons it doesn't cover out altogether.

        The points are resolved concurrently by `resolve_vector_grid_cell`, so unlike the
        conversions that read a whole file this one is awaited rather than run in a thread.
        """
        lngs, lats = zip(*self.source_coordinates)
        resolved = await resolve_vector_grid_cell(dst_grid_name, list(lngs), list(lats))
        matched = [
            (lng, lat, grid_cell_id)
            for (lng, lat), grid_cell_id in zip(self.source_coordinates, resolved)
            if grid_cell_id is not None
        ]
        src_lngs, src_lats, dst_ids = _columns(matched, 3)
        return self._frame(src_lng=src_lngs, src_lat=src_lats, dst_id=dst_ids)

    def _coordinates_to_raster(self):
        """Convert coordinates to a raster grid, which leaves them unchanged: a raster
        grid cell is the location itself, and every raster grid holds a value there."""
        lngs, lats = _columns(self.source_coordinates, 2)
        return self._frame(src_lng=lngs, src_lat=lats, dst_lng=lngs, dst_lat=lats)

    async def _convert(self, dst_grid_name):
        """Convert the source grid cells to a single destination grid.

        The two conversions that read a crosswalk file do it in a thread, so that a request
        naming several destinations reads their files at once rather than one after the
        other. The rest either work from what is already in memory, and would only pay the
        thread's overhead, or spread their own reads across the executor.
        """
        if self.grid.grid_type in COORDINATE_GRID_TYPES:
            if GRIDS[dst_grid_name].grid_type == "raster":
                return self._coordinates_to_raster()
            if self.grid is WGS84:
                return await self._wgs84_to_vector(dst_grid_name)
            return await asyncio.to_thread(self._raster_to_vector, dst_grid_name)

        if GRIDS[dst_grid_name].grid_type == "raster":
            return self._vector_to_raster()
        return await asyncio.to_thread(self._vector_to_vector, dst_grid_name)

    async def _read_shared_geometry(self):
        """Read what every raster destination will need before they fan out.

        All of them convert a source grid cell to the same point on its surface, so reading
        the geometry file here means reading it once for the request rather than racing
        several conversions to read it apiece.
        """
        if self.grid.grid_type in COORDINATE_GRID_TYPES:
            return
        if not any(GRIDS[name].grid_type == "raster" for name in self.destination_grids):
            return

        self._representative_points = await asyncio.to_thread(
            representative_points, self.grid.name, self.source_ids
        )

    async def execute(self):
        """Convert the source grid cells to every destination grid, as one table.

        Grid cells with no counterpart in a destination grid are left out rather than
        reported as empty rows, so the result can be shorter than the request.
        """
        await self._resolve_source()
        await self._read_shared_geometry()

        # Gathered rather than awaited in turn, since the destinations are independent of
        # each other, and in order, since `gather` returns results in the order given.
        frames = await asyncio.gather(*[self._convert(name) for name in self.destination_grids])

        # A plain vertical concat, since normalizing has already given every frame the same
        # schema — and so this raises rather than quietly widening if one ever doesn't.
        return pl.concat(
            [
                self._normalize(frame).with_columns(dst_grid=pl.lit(name, dtype=pl.String))
                for frame, name in zip(frames, self.destination_grids)
            ]
        )
