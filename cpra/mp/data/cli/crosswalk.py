"""`crosswalk` command: generate crosswalk files between grids.

Pure-CLI wrapper around `cpra.mp.data.crosswalks`, itself a pure-Python replacement for
the removed `scripts/generate_crosswalks.py` (vector) and
`scripts/generate_raster_crosswalks.py` (raster). This module owns argument validation,
source/destination discovery, progress reporting, and per-crosswalk failure handling; the
actual work happens in `crosswalks.write_vector_crosswalk` / `write_raster_crosswalk`.

One command covers both source kinds: each `--from` grid dispatches on its `grid_type` in
`structure/grids.csv`, so a mixed run needs no flag to tell the two apart, and the options
that only apply to one kind (`--batch-size`, `--block-size`) are ignored for the other.

Destinations are always vector grids. `write_vector_crosswalk` assigns source centroids to
destination polygons and `write_raster_crosswalk` rasterizes those same polygons into zone
IDs -- neither has anything to do with a destination that isn't polygons.

Usage:

    python -m cpra.mp.data.cli crosswalk
    python -m cpra.mp.data.cli crosswalk --from hydro_compartment_v001
    python -m cpra.mp.data.cli crosswalk --from morph_pixel_v001 --to parish_v2020
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


def _has_geometry(name, grids_path, role):
    """True if `name` has a `.fgb` in `grids_path`; warns and returns False if it doesn't.

    A grid can be registered in `structure/grids.csv` without its geometry having been
    exported yet, and the registry lists more grids than any one checkout of the data tends
    to carry. That's a reason to leave that grid out of the run, not to abort a run
    covering ten others, so it warns rather than raising.
    """

    path = Path(grids_path) / f"{name}.fgb"
    if not path.exists():
        click.echo(f"WARNING: skipping {role} grid {name!r}: no geometry file at {path}", err=True)
        return False
    return True


def _resolve_sources(from_grids, grids_path):
    """Source grids, defaulting to every grid in `structure/grids.csv`."""

    if from_grids:
        names = list(dict.fromkeys(from_grids))
        for name in names:
            _require_grid(name)
    else:
        names = sorted(GRIDS)

    sources = []
    for name in names:
        # A raster source is fully described by grids.csv (crs, bounds, width, height) and
        # reads no geometry file of its own -- only the destination's -- so a missing `.fgb`
        # disqualifies a vector source but never a raster one.
        if GRIDS[name].grid_type == "vector" and not _has_geometry(name, grids_path, "source"):
            continue
        sources.append(name)

    if not sources:
        raise click.UsageError(f"No usable source grids (vector grids need a .fgb in {grids_path})")
    return sources


def _resolve_destinations(to_grids, grids_path):
    """Destination grids, defaulting to every vector grid in `structure/grids.csv`.

    An explicit `--to` naming a raster grid is an error rather than a skip: the caller asked
    for that destination by name, so silently dropping it would be a lie. The default set is
    vector by construction and needs no such check.
    """

    if to_grids:
        names = list(dict.fromkeys(to_grids))
        for name in names:
            _require_grid(name, grid_type="vector")
    else:
        names = [name for name in sorted(GRIDS) if GRIDS[name].grid_type == "vector"]

    destinations = [name for name in names if _has_geometry(name, grids_path, "destination")]
    if not destinations:
        raise click.UsageError(f"No usable destination grids (need a .fgb in {grids_path})")
    return destinations


def _out_path(crosswalks_path, src_grid, dst_grid):
    """Where a crosswalk lands: Parquet for a vector source, a zone-ID COG for a raster one."""

    suffix = "parquet" if GRIDS[src_grid].grid_type == "vector" else "tif"
    return crosswalks_path / f"{src_grid}__{dst_grid}.{suffix}"


@cli.command()
@click.option(
    "--from",
    "from_grids",
    multiple=True,
    help="Source grid to crosswalk from; repeatable. Defaults to every grid in grids.csv.",
)
@click.option(
    "--to",
    "to_grids",
    multiple=True,
    help="Destination vector grid to crosswalk to; repeatable. Defaults to every vector grid in grids.csv.",
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
    help="Arrow batch size used when streaming grids. Vector sources only; ignored for raster sources.",
)
@click.option(
    "--block-size",
    type=int,
    default=512,
    show_default=True,
    help="Window size (pixels) used when rasterizing destinations. Raster sources only; ignored for vector sources.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite an existing crosswalk file instead of skipping it.",
)
@click.pass_context
def crosswalk(ctx, from_grids, to_grids, grids_path, crosswalks_path, batch_size, block_size, overwrite):
    """Build crosswalk file(s) from one or more source grids to one or more destination grids."""

    grids_path = Path(grids_path)
    crosswalks_path = Path(crosswalks_path)
    sources = _resolve_sources(from_grids, grids_path)
    destinations = _resolve_destinations(to_grids, grids_path)

    # No grid needs a crosswalk to itself, so a run whose only pair is a self-pair has
    # nothing to do -- worth saying so rather than exiting 0 having written nothing.
    pairs = [(src, dst) for src in sources for dst in destinations if src != dst]
    if not pairs:
        raise click.UsageError("No crosswalks to build: every destination grid is also its own source")
    click.echo(f"Building {len(pairs)} crosswalk(s): {len(sources)} source(s) -> {len(destinations)} destination(s)")

    failures = []
    written = 0
    skipped = 0
    for src_grid in sources:
        grid = GRIDS[src_grid]

        # Resolve the whole source's work up front: a centroid index costs hundreds of MB on
        # the larger grids, so a source whose outputs all already exist must not pay for one.
        pending = []
        for dst_grid in destinations:
            if dst_grid == src_grid:
                continue
            out_path = _out_path(crosswalks_path, src_grid, dst_grid)
            if out_path.exists() and not overwrite:
                click.echo(f"SKIP: {out_path} already exists")
                skipped += 1
                continue
            pending.append(dst_grid)
        if not pending:
            continue

        # Rebound before the build, so the previous source's index is released first rather
        # than held alongside this one.
        tree = None
        if grid.grid_type == "vector":
            click.echo(f"Indexing {src_grid} centroids...")
            start = time.perf_counter()
            try:
                tree = read_centroid_tree(grids_path / f"{src_grid}.fgb", batch_size=batch_size)
            except Exception as exc:  # noqa: BLE001 - one unreadable source must not abort the rest
                click.echo(f"FAIL: {src_grid}: {exc}", err=True)
                failures.extend(f"{src_grid} -> {dst_grid}" for dst_grid in pending)
                continue
            click.echo(f"DONE: indexed {len(tree[1])} centroids in {time.perf_counter() - start:.1f}s")

        for dst_grid in pending:
            start = time.perf_counter()
            try:
                if grid.grid_type == "vector":
                    out_path, count = write_vector_crosswalk(
                        src_grid, dst_grid, grids_path, crosswalks_path, batch_size=batch_size, tree=tree
                    )
                    unit = "rows"
                else:
                    out_path, count = write_raster_crosswalk(
                        src_grid, dst_grid, grids_path, crosswalks_path, block_size=block_size
                    )
                    unit = "windows"
            except Exception as exc:  # noqa: BLE001 - one bad pair must not abort the rest
                click.echo(f"FAIL: {src_grid} -> {dst_grid}: {exc}", err=True)
                failures.append(f"{src_grid} -> {dst_grid}")
                continue

            written += 1
            click.echo(f"DONE: {count} {unit} -> {out_path} in {time.perf_counter() - start:.1f}s")

    click.echo(f"SUMMARY: {written} written, {skipped} skipped, {len(failures)} failed")
    if failures:
        click.echo(f"FAILED: {len(failures)} of {len(pairs)} crosswalk(s): {', '.join(failures)}", err=True)
        ctx.exit(1)
