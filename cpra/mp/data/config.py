import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TRUE_VALUES = ("1", "t", "true", "y", "yes")


def get_config(name, default_value):
    """Get a config variable from the environment."""
    return os.environ.get(f"CPRA_MP_DATA_{name}", default_value)


BASE_PATH = Path(get_config("BASE_PATH", "/ocean/projects/bcs200002p/shared/data"))
CROSSWALKS_PATH = Path(get_config("CROSSWALKS_PATH", "/ocean/projects/bcs200002p/shared/crosswalks"))
CATALOG_PATH = Path(get_config("CATALOG_PATH", str(BASE_PATH.parent / "catalog")))
GRIDS_PATH = Path(get_config("GRIDS_PATH", "/ocean/projects/bcs200002p/shared/grids"))
TILES_PATH = Path(get_config("TILES_PATH", "/ocean/projects/bcs200002p/shared/tiles"))
DEBUG = get_config("DEBUG", "False").lower() in TRUE_VALUES
MPD_URI = get_config("MPD_URI", "")
STREAM_THRESHOLD = int(get_config("STREAM_THRESHOLD", "25"))


def get_base_path(base_path=None):
    """Resolve a base path, falling back to the configured default.

    Read at call time rather than bound as a default argument value so that
    overriding BASE_PATH (in tests, or between calls) takes effect.
    """
    return Path(base_path) if base_path is not None else BASE_PATH


def get_catalog_path(catalog_path=None):
    """Resolve a catalog path, falling back to the configured default.

    Read at call time rather than bound as a default argument value so that
    overriding CATALOG_PATH (in tests, or between calls) takes effect.
    """
    return Path(catalog_path) if catalog_path is not None else CATALOG_PATH


CATALOG_PATH_UNSET = object()


def get_write_catalog_path(catalog_path=CATALOG_PATH_UNSET):
    """Resolve a catalog path for writers, for which ``None`` means "don't write".

    Writers (``write_parquet``, ``write_cog``, ``compact_parquet``, ``compact_cog``)
    accept ``catalog_path=None`` to skip the catalog side-table entirely, so they can't
    also use ``None`` to mean "use the default" the way readers do -- hence the separate
    ``CATALOG_PATH_UNSET`` sentinel for "not passed".

    Resolved at call time rather than bound as a default argument value: binding
    ``CATALOG_PATH`` into a signature captures it at import, so a caller that redirects
    only ``base_path`` (as every test is expected to -- see CLAUDE.md) would still write
    catalog rows to the configured production path.
    """
    if catalog_path is CATALOG_PATH_UNSET:
        return CATALOG_PATH
    return Path(catalog_path) if catalog_path is not None else None


# GDAL thread count for raster reads/writes (see api.utils.raster_gdal_env). Tunable
# because the API runs as a single process and already spreads files across a thread
# pool, so the two widths multiply under concurrent requests.
RASTER_NUM_THREADS = get_config("RASTER_NUM_THREADS", "ALL_CPUS")

# CreateCopy streaming chunk, and the block cache that bounds it -- GDAL clamps the swath
# to half of GDAL_CACHEMAX, so the two must be raised together (see
# api.utils.raster_copy_gdal_env). The cache is process-global, so this also bounds cache
# growth across concurrent requests instead of leaving it at GDAL's default 5% of RAM.
RASTER_SWATH_SIZE_MB = int(get_config("RASTER_SWATH_SIZE_MB", "512"))
RASTER_CACHE_MAX_MB = int(get_config("RASTER_CACHE_MAX_MB", "2048"))

# Kill switch for the VRT+CreateCopy download path, falling back to windowed reads.
RASTER_USE_VRT_COPY = get_config("RASTER_USE_VRT_COPY", "True").lower() in TRUE_VALUES

# Above this, a windowed read's per-window NumPy buffer is big enough that the VRT copy is
# the lighter path; below it the windowed read wins (see Dataset._windowed_read_is_lighter).
RASTER_WINDOW_BUFFER_LIMIT_MB = int(get_config("RASTER_WINDOW_BUFFER_LIMIT_MB", "200"))

# Target size, in MB, for one summarize-loop batch's read buffer (n_bands x block_height x
# batch_width x itemsize -- see api.utils.raster_blocks_per_batch and
# Dataset._apply_transform_params_raster). This is a separate budget from
# RASTER_CACHE_MAX_MB (GDAL's own block cache): capping only GDAL's cache still leaves the
# batch's own NumPy buffer free to grow with a block row's full width regardless of how many
# bands or blocks are requested, which is exactly the defect this budget fixes. Defaulted from
# RASTER_CACHE_MAX_MB (an eighth of it) so the two budgets stay in the same ballpark without
# either one dominating; override directly for a different balance.
RASTER_BATCH_BUFFER_MB = int(get_config("RASTER_BATCH_BUFFER_MB", str(max(RASTER_CACHE_MAX_MB // 8, 1))))

# Explicit override for the number of raster blocks unioned into one summarize-loop batch,
# bypassing the RASTER_BATCH_BUFFER_MB / band-count derivation in
# api.utils.raster_blocks_per_batch entirely. Unset (None) by default.
_raster_blocks_per_batch = get_config("RASTER_BLOCKS_PER_BATCH", None)
RASTER_BLOCKS_PER_BATCH = int(_raster_blocks_per_batch) if _raster_blocks_per_batch else None

# Map tile and style URLs
GLYPHS_BASE_URL = get_config("GLYPHS_BASE_URL", "https://api.mpdp.coastal.la.gov/glyphs")
SPRITES_BASE_URL = get_config("SPRITES_BASE_URL", "https://api.mpdp.coastal.la.gov/sprites")
STATIC_TILES_BASE_URL = get_config("STATIC_TILES_BASE_URL", "https://api.mpdp.coastal.la.gov/tiles")
