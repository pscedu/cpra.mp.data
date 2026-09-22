"""The raster POINT scan: `sample_raster_points`, this module's only export, reads a named
handful of pixels out of one input's bands and returns them directly as a LONG
`pl.LazyFrame` -- dimension columns, `grid_cell`, `value`, one row per (pixel, band)
observation, the same row model `io/vector.py` opens a vector dataset into.

**One reader, one shape.** `io/raster.py`'s `scan_raster` is the band-wide reader: it
returns the `(lf, metadata_df, ReadWindow)` triple, one row per pixel of a window, one
column per band, and everything downstream of it has to carry that window around and
eventually unpivot it. This module is the other reader, and it is deliberately a SEPARATE
one rather than a mode of that one: it has no window (a set of scattered pixels is not a
rectangle), no batching (the answer is a handful of rows, not a memory-bounded stream), no
decimation (the pixel under a coordinate is the whole answer, and a coarser grid hands back
a neighbour's value instead), and no VRT (it samples the underlying files directly). Each
reader therefore has exactly one output shape, fixed at the call, and neither has to
describe the other's.

**When the walk picks this one.** `execution/execute.py` gates it -- see
`_point_sample_reads` there for the full condition. In short: the origin is a raster whose
only consumer is a spatial `filter` naming literal `grid_cell` ids. That is the request the
old path answered by reading the whole BOUNDING BOX of those ids and then throwing away
every pixel but the named ones; three coordinates scattered across a 20674x8265 grid made
that a 4.3-million-row read to produce 156 rows. Everything the gate does not cover still
goes through `scan_raster` exactly as before -- this reader is additive, and the bounding
box remains the fallback.

**Why it emits long directly.** The band-wide reader's output has to become long anyway for
a spatial filter (`execution.band_ops.sample_grid_cells_band_wide`), and that transition is
the only thing a point sample ever wanted. Producing the long frame here skips the
intermediate entirely, and -- because the frame this returns is constructed the same way
that function constructs its own, by the same unpivot against the same band-identity join
-- the walk's ordinary long `filter` lowering can then apply the `where` unchanged. See
`sample_raster_points`'s own docstring for the column-by-column correspondence.

Band order is `metadata_df`'s own row order, exactly, same as `scan_raster`: row *i* becomes
column `b<i>` becomes the long rows carrying metadata row *i*'s dimension values. This
module computes those `b<i>` names itself rather than importing `band_column_names`, for the
same layering reason `io/raster.py` does (`io` may import `request`, never `execution`); the
three must never diverge.

`io` may import `request` but never `execution` -- see `tests/api/test_analysis_layering.py`.
"""

import asyncio
from collections.abc import Sequence

import numpy as np
import polars as pl
import rasterio
from rasterio.transform import from_bounds
from rasterio.transform import xy as transform_xy

from cpra.mp.data.analysis.io.raster import NUMPY_TO_POLARS
from cpra.mp.data.analysis.io.raster_io import parallel_sample_rasters, raster_gdal_env
from cpra.mp.data.grids import GRIDS

__all__ = ["sample_raster_points"]


def _pixel_positions(grid_name: str, grid_cells: Sequence[int]) -> list[tuple[int, int]]:
    """Decode ABSOLUTE row-major `grid_cell` ids back to `(row, col)` on `grid_name`'s own
    full pixel grid -- the exact inverse of `io.geography.raster_grid_cell_id`'s
    `row * grid.width + col`, over the same grid's own width and with no window involved.

    Raises `ValueError` for an id that does not land on the grid. A caller reaching this
    function has already decided these ids name real pixels (`execution.execute`'s gate
    range-checks them before choosing this reader at all), so an out-of-range id here means
    those two decodes disagree, which is a bug rather than a bad request -- hence a plain
    `ValueError` and not one of the request layer's errors.
    """
    grid = GRIDS[grid_name]
    positions: list[tuple[int, int]] = []
    for cell in grid_cells:
        row, col = divmod(int(cell), grid.width)
        if not (0 <= row < grid.height) or not (0 <= col < grid.width):
            raise ValueError(
                f"grid_cell {cell} does not name a pixel of {grid_name!r} "
                f"({grid.height}x{grid.width})"
            )
        positions.append((row, col))
    return positions


def _sample_groups(sources: Sequence[tuple[object, object]]):
    """Split `(path, band)` sources into the parallel sample calls they should become:
    `(paths, band_lists, slots)`, where `slots[i]` is `(group index, offset within that
    group's band list)` for source *i*.

    Grouped by FILE first, which is the whole point -- 52 bands spread over 5 files is 5
    reads, not 52, and each file is opened once instead of once per band the way a VRT
    assembled from the same sources opens it. Then split again by DTYPE within a file,
    because `rasterio`'s `sample` shares `read`'s one-array-per-call constraint and raises
    `ValueError: more than one 'dtype' found` the moment a single call spans two of them --
    the same grouping, and the same reason, as `io.raster.scan_raster`'s `read_groups`.
    That second split is normally a no-op (a COG this package writes has one dtype for all
    its bands), so this is usually exactly one group per file.

    Each group gets its OWN entry, and therefore its own dataset handle at the call site: a
    `rasterio` dataset is not safe to sample from two threads at once, and
    `parallel_sample_rasters` gives each entry its own thread.
    """
    by_path: dict[str, list[tuple[int, int]]] = {}
    for position, (path, band) in enumerate(sources):
        by_path.setdefault(str(path), []).append((position, int(band)))

    paths: list[str] = []
    band_lists: list[list[int]] = []
    slots: dict[int, tuple[int, int]] = {}
    dtype_of: dict[str, dict[int, str]] = {}
    with raster_gdal_env():
        for path, entries in by_path.items():
            with rasterio.open(path) as probe:
                dtype_of[path] = {band: probe.dtypes[band - 1] for _position, band in entries}
    for path, entries in by_path.items():
        groups: dict[str, list[tuple[int, int]]] = {}
        for position, band in entries:
            groups.setdefault(dtype_of[path][band], []).append((position, band))
        for members in groups.values():
            for offset, (position, _band) in enumerate(members):
                slots[position] = (len(paths), offset)
            paths.append(path)
            band_lists.append([band for _position, band in members])
    return paths, band_lists, slots


def sample_raster_points(
    metadata_df: pl.DataFrame,
    grid_name: str,
    grid_cells: Sequence[int],
    *,
    mask_nodata: bool = True,
) -> pl.LazyFrame:
    """Sample `metadata_df`'s bands at exactly the pixels `grid_cells` names, as a long
    frame.

    Columns, in order: `grid_cell` (the ABSOLUTE id, `pl.Int64`, exactly as given -- see
    `io.geography.raster_grid_cell_id` for what the id means and
    `io.geography.relabel_raster_grid_cell` for what a point query turns it back into),
    `value`, then every column of `metadata_df` -- which is to say, column for column and
    dtype for dtype, what `execution.band_ops.sample_grid_cells_band_wide` produces from a
    band-wide read of the same bands, minus its trailing `filter_long` call. That
    correspondence is the contract: the walk applies the same `where` to this frame through
    the same long lowering, so anything this frame carried differently would change the
    answer.

    `grid_cells` may repeat an id and may be in any order; the returned rows follow it. It
    must name at least one pixel, and every id must land on `grid_name` (see
    `_pixel_positions`).

    `metadata_df` must carry `path` and `band` columns and at least one row -- exactly what
    `io/catalog.py::resolve_raster_bands` returns, and the same precondition `scan_raster`
    states. There is no zero-band answer here for the same reason there is none there: zero
    bands is a degenerate frame the walk builds for itself
    (`execution.frame.empty_band_wide`), never something a reader invents.

    Reads through `io.raster_io.parallel_sample_rasters`, one thread per (file, dtype)
    group -- see `_sample_groups`. Pixel coordinates are derived from `grid_name`'s OWN
    geometry (`rasterio.transform.from_bounds` over its declared bounds, then
    `rasterio.transform.xy` at pixel centres), never from an opened file's transform, so the
    id -> pixel -> coordinate chain is closed over one source of truth: the same
    `GRIDS[grid_name]` the ids were computed against in the first place. Sampling at a pixel
    CENTRE is what makes the round trip through world coordinates exact -- `sample` floors
    the coordinate back through the same affine, and the centre is the one point in a pixel
    that cannot land in a neighbour.

    `mask_nodata` (default on) replaces each band's declared nodata value with null, exactly
    as `scan_raster`'s own `mask_nodata` does for a band-wide read, and reading the same
    per-band `nodatavals` off the same files. A band declaring no nodata is left alone.
    """
    if metadata_df.height == 0:
        raise ValueError(
            "sample_raster_points requires at least one band -- zero bands is a degenerate "
            "frame the walk builds for itself (see execution/frame.py's empty_band_wide), "
            "not something this reader can produce"
        )
    if "path" not in metadata_df.columns or "band" not in metadata_df.columns:
        raise ValueError("sample_raster_points requires metadata_df to carry 'path' and 'band' columns")
    cells = [int(cell) for cell in grid_cells]
    if not cells:
        raise ValueError("sample_raster_points requires at least one grid_cell to sample")

    grid = GRIDS[grid_name]
    positions = _pixel_positions(grid_name, cells)
    grid_transform = from_bounds(*grid.bounds, width=grid.width, height=grid.height)
    xs, ys = transform_xy(grid_transform, [row for row, _col in positions], [col for _row, col in positions])
    coordinates = list(zip(xs, ys))

    sources = list(zip(metadata_df["path"].to_list(), metadata_df["band"].to_list()))
    paths, band_lists, slots = _sample_groups(sources)

    # `asyncio.run`, not an `await`: every path that reaches this reader is already OFF the
    # event loop -- `execution.execute` is called from a worker thread in both API routes
    # that can point-sample (`anyio.to_thread.run_sync` for the file sinks, a
    # `StreamingThread` for the streamed tabular response) -- so there is no running loop
    # here to attach to, and making this function async would push that constraint onto
    # every caller of a reader that is otherwise entirely synchronous.
    with raster_gdal_env():
        datasets = [rasterio.open(path) for path in paths]
        try:
            nodatavals = {
                position: datasets[group].nodatavals[band_lists[group][offset] - 1]
                for position, (group, offset) in slots.items()
            }
            sampled = asyncio.run(parallel_sample_rasters(datasets, coordinates, band_lists))
            dtypes = {
                position: datasets[group].dtypes[band_lists[group][offset] - 1]
                for position, (group, offset) in slots.items()
            }
        finally:
            for dataset in datasets:
                dataset.close()

    band_names = [f"b{i}" for i in range(len(sources))]
    schema = pl.Schema({name: NUMPY_TO_POLARS[dtypes[i]] for i, name in enumerate(band_names)})
    columns = {}
    for position, name in enumerate(band_names):
        group, offset = slots[position]
        columns[name] = np.array([sampled[group][point][offset] for point in range(len(cells))])

    df = pl.DataFrame({"grid_cell": pl.Series(cells, dtype=pl.Int64)}).hstack(pl.DataFrame(columns, schema=schema))
    if mask_nodata:
        mask_exprs = [
            pl.when(pl.col(name) == nodatavals[i]).then(None).otherwise(pl.col(name)).alias(name)
            for i, name in enumerate(band_names)
            if nodatavals[i] is not None
        ]
        if mask_exprs:
            df = df.with_columns(mask_exprs)

    long_lf = df.lazy().unpivot(index=["grid_cell"], on=band_names, variable_name="__band", value_name="value")
    band_identity = pl.DataFrame({"__band": pl.Series(band_names, dtype=pl.String)}).hstack(metadata_df)
    return long_lf.join(band_identity.lazy(), on="__band", how="left").drop("__band")
