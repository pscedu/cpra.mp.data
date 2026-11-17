"""`tile-cache` commands: inspect, prune, or empty the API's on-disk tile cache.

The cache invalidates itself on a new release or on any catalogued write to an input
dataset (see `cpra.mp.data.tile_cache`), so `clear` is for what its keys cannot see --
data copied in by hand, a changed grid or crosswalk -- or for reclaiming the space
outright. Safe to run while the API serves:
a tile deleted mid-request is just a miss.

``--path`` defaults to the configured ``TILE_CACHE_PATH`` (``cpra.mp.data.config``).

Usage:

\b
    python -m cpra.mp.data.cli tile-cache stats
    python -m cpra.mp.data.cli tile-cache sweep --max-gb 5
    python -m cpra.mp.data.cli tile-cache clear
"""

import click

from cpra.mp.data import config
from cpra.mp.data.cli import cli
from cpra.mp.data.tile_cache import TileCache

_path_option = click.option(
    "--path",
    type=click.Path(file_okay=False),
    default=None,
    help="Tile cache directory (default: the configured TILE_CACHE_PATH).",
)


def _cache(path, max_gb=0.0) -> TileCache:
    return TileCache(path or config.TILE_CACHE_PATH, int(max_gb * 1024**3))


def _describe(stats) -> str:
    return f"{stats.files} tiles, {stats.bytes / 1024**3:.2f} GiB"


@cli.group("tile-cache")
def tile_cache():
    """Inspect, prune, or empty the API's tile cache."""


@tile_cache.command()
@_path_option
def stats(path):
    """Report how many tiles the cache holds and their size on disk."""
    cache = _cache(path)
    click.echo(f"{cache.root}: {_describe(cache.stats())}")


@tile_cache.command()
@_path_option
@click.option(
    "--max-gb",
    type=float,
    default=None,
    help="Size to prune to (default: the configured TILE_CACHE_MAX_GB).",
)
def sweep(path, max_gb):
    """Delete least-recently-used tiles until the cache is under its size limit."""
    max_gb = config.TILE_CACHE_MAX_GB if max_gb is None else max_gb
    if max_gb <= 0:
        raise click.UsageError("No size limit: pass --max-gb or set TILE_CACHE_MAX_GB.")
    cache = _cache(path, max_gb)
    click.echo(f"{cache.root}: {_describe(cache.sweep())} left")


@tile_cache.command()
@_path_option
@click.confirmation_option(prompt="Delete every cached tile?")
def clear(path):
    """Delete every cached tile."""
    cache = _cache(path)
    click.echo(f"{cache.root}: removed {_describe(cache.clear())}")
