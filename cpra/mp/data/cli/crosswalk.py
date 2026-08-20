"""`crosswalk` command group: generate vector and raster crosswalk files.

Pure-CLI wrapper around `cpra.mp.data.crosswalks`, itself a pure-Python replacement for
the removed `scripts/generate_crosswalks.py` (vector) and
`scripts/generate_raster_crosswalks.py` (raster). This module owns argument validation,
default-destination discovery, progress reporting, and per-destination failure handling;
the actual work happens in `crosswalks.write_vector_crosswalk` /
`write_raster_crosswalk`.

Usage:

    python -m cpra.mp.data.cli crosswalk vector hydro_compartment_v001
    python -m cpra.mp.data.cli crosswalk raster morph_pixel_v001 --to parish_v2020
"""

import time
from pathlib import Path

import click

from cpra.mp.data import config
from cpra.mp.data.cli import cli
from cpra.mp.data.crosswalks import read_centroid_tree, write_raster_crosswalk, write_vector_crosswalk
from cpra.mp.data.grids import GRIDS


def _require_grid(name, grid_type=None):
    """Look up `name` in `GRIDS`, raising `click.BadParameter` if it's missing or (when
    `grid_type` is given) the wrong kind of grid."""

    grid = GRIDS.get(name)
    if grid is None:
        known = ", ".join(sorted(GRIDS))
        raise click.BadParameter(f"Unknown grid {name!r}. Known grids: {known}")
    if grid_type is not None and grid.grid_type != grid_type:
        raise click.BadParameter(f"{name!r} is a {grid.grid_type!r} grid, expected a {grid_type!r} grid")
    return grid


def _default_destinations(grids_path, src_grid_name):
    """Every other vector grid with a `.fgb` in `grids_path`, sorted, plus what was skipped.

    Mirrors the old scripts' default of globbing `GRIDS_PATH/*.fgb` and skipping the source
    grid itself, but filters to vector grids. A crosswalk destination is always a vector
    grid: `write_vector_crosswalk` needs polygons to assign source centroids to, and
    `write_raster_crosswalk` rasterizes destination polygons into zones.

    The filter matters because a `.fgb` can exist for a grid that isn't a vector grid --
    `veg_grid_cell_v001` is a raster grid in `structure/grids.csv` and has one -- or for no
    registered grid at all. The old scripts globbed blindly and would build a crosswalk to
    such a file anyway, producing output that neither `summarize_by` consumer in `api/` can
    use as a destination.
    """

    stems = sorted(path.stem for path in Path(grids_path).glob("*.fgb") if path.stem != src_grid_name)
    destinations = [stem for stem in stems if getattr(GRIDS.get(stem), "grid_type", None) == "vector"]
    skipped = [stem for stem in stems if stem not in set(destinations)]
    return destinations, skipped


def _resolve_destinations(to_grids, grids_path, src_grid_name):
    """Validate explicit `--to` values, or discover vector destinations in `grids_path`.

    An explicit `--to` naming a non-vector grid is an error rather than a skip: the caller
    asked for that destination by name, so silently dropping it would be a lie.
    """

    if to_grids:
        destinations = list(dict.fromkeys(to_grids))
        for name in destinations:
            _require_grid(name, grid_type="vector")
        return destinations

    destinations, skipped = _default_destinations(grids_path, src_grid_name)
    for stem in skipped:
        grid = GRIDS.get(stem)
        reason = f"a {grid.grid_type} grid" if grid is not None else "not in the grid registry"
        click.echo(f"SKIP: {stem} ({reason}) is not a vector destination")
    if not destinations:
        raise click.UsageError(
            f"No vector destination grids (*.fgb) found in {grids_path} other than {src_grid_name!r}"
        )
    return destinations


@cli.group()
def crosswalk():
    """Generate crosswalk files between grids."""


@crosswalk.command()
@click.argument("src_grid")
@click.option(
    "--to",
    "to_grids",
    multiple=True,
    help="Destination vector grid to crosswalk to; repeatable. Defaults to every other vector grid with a .fgb file in --grids-path.",
)
@click.option(
    "--grids-path",
    type=click.Path(path_type=Path),
    default=lambda: config.GRIDS_PATH,
    show_default=True,
    help="Directory of grid .fgb files.",
)
@click.option(
    "--crosswalks-path",
    type=click.Path(path_type=Path),
    default=lambda: config.CROSSWALKS_PATH,
    show_default=True,
    help="Directory to write crosswalk files to.",
)
@click.option(
    "--batch-size",
    type=int,
    default=65536,
    show_default=True,
    help="Arrow batch size used when streaming the source and destination grids.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite an existing crosswalk file instead of skipping it.",
)
@click.pass_context
def vector(ctx, src_grid, to_grids, grids_path, crosswalks_path, batch_size, overwrite):
    """Build vector-to-vector crosswalk(s) from SRC_GRID to one or more destination grids."""

    _require_grid(src_grid, grid_type="vector")
    grids_path = Path(grids_path)
    crosswalks_path = Path(crosswalks_path)
    destinations = _resolve_destinations(to_grids, grids_path, src_grid)

    click.echo(f"Indexing {src_grid} centroids...")
    start = time.perf_counter()
    tree, src_ids = read_centroid_tree(grids_path / f"{src_grid}.fgb", batch_size=batch_size)
    click.echo(f"DONE: indexed {len(src_ids)} centroids in {time.perf_counter() - start:.1f}s")

    failures = []
    for dst_grid in destinations:
        out_path = crosswalks_path / f"{src_grid}__{dst_grid}.parquet"
        if out_path.exists() and not overwrite:
            click.echo(f"SKIP: {out_path} already exists")
            continue

        start = time.perf_counter()
        try:
            out_path, row_count = write_vector_crosswalk(
                src_grid,
                dst_grid,
                grids_path,
                crosswalks_path,
                batch_size=batch_size,
                tree=(tree, src_ids),
            )
        except Exception as exc:  # noqa: BLE001 - one bad destination must not abort the rest
            click.echo(f"FAIL: {src_grid} -> {dst_grid}: {exc}", err=True)
            failures.append(dst_grid)
            continue

        click.echo(f"DONE: {row_count} rows -> {out_path} in {time.perf_counter() - start:.1f}s")

    if failures:
        click.echo(f"FAILED: {len(failures)} of {len(destinations)} destination(s): {', '.join(failures)}", err=True)
        ctx.exit(1)


@crosswalk.command()
@click.argument("raster_grid")
@click.option(
    "--to",
    "to_grids",
    multiple=True,
    help="Destination vector grid to crosswalk to; repeatable. Defaults to every other vector grid with a .fgb file in --grids-path.",
)
@click.option(
    "--grids-path",
    type=click.Path(path_type=Path),
    default=lambda: config.GRIDS_PATH,
    show_default=True,
    help="Directory of grid .fgb files.",
)
@click.option(
    "--crosswalks-path",
    type=click.Path(path_type=Path),
    default=lambda: config.CROSSWALKS_PATH,
    show_default=True,
    help="Directory to write crosswalk files to.",
)
@click.option(
    "--block-size",
    type=int,
    default=512,
    show_default=True,
    help="Window size (pixels) used when rasterizing destination polygons.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite an existing crosswalk file instead of skipping it.",
)
@click.pass_context
def raster(ctx, raster_grid, to_grids, grids_path, crosswalks_path, block_size, overwrite):
    """Build raster-to-vector crosswalk(s) from RASTER_GRID to one or more destination grids."""

    _require_grid(raster_grid, grid_type="raster")
    grids_path = Path(grids_path)
    crosswalks_path = Path(crosswalks_path)
    destinations = _resolve_destinations(to_grids, grids_path, raster_grid)

    failures = []
    for dst_grid in destinations:
        out_path = crosswalks_path / f"{raster_grid}__{dst_grid}.tif"
        if out_path.exists() and not overwrite:
            click.echo(f"SKIP: {out_path} already exists")
            continue

        start = time.perf_counter()
        try:
            out_path, window_count = write_raster_crosswalk(
                raster_grid, dst_grid, grids_path, crosswalks_path, block_size=block_size
            )
        except Exception as exc:  # noqa: BLE001 - one bad destination must not abort the rest
            click.echo(f"FAIL: {raster_grid} -> {dst_grid}: {exc}", err=True)
            failures.append(dst_grid)
            continue

        click.echo(f"DONE: {window_count} windows -> {out_path} in {time.perf_counter() - start:.1f}s")

    if failures:
        click.echo(f"FAILED: {len(failures)} of {len(destinations)} destination(s): {', '.join(failures)}", err=True)
        ctx.exit(1)
