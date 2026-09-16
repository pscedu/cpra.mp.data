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
# api.sources.scan_rasters, the Polars IO-plugin implementation of the raster zonal
# summarize path -- RasterGridAnalysis._apply_transform_params_impl is its caller). This is a
# separate budget from RASTER_CACHE_MAX_MB (GDAL's own block cache): capping only GDAL's
# cache still leaves the batch's own NumPy buffer free to grow with a block row's full
# width regardless of how many bands or blocks are requested, which is exactly the defect
# this budget fixes.
#
# 64 MB is a measured choice, not a ballpark: the streaming group_by's in-flight buffering
# scales with (read batch size) x (batches spanned), so the read batch is the memory lever
# for the whole summarize path. At the full 52-band selection, 64 MB derives one 512-block
# per batch (52 x 512 x 512 x 4 B ~= 55 MB) -- the configuration measured at 1140 MB peak
# for `sum` (vs ~3 GB at the previous 256 MB default) and 564 MB for `max`, with no measured
# slowdown relative to the wider default. Narrower selections still derive proportionally
# more blocks per batch, so every batch stays ~64 MB rather than inheriting a size tuned for
# the full band count. The earlier default coupled this to RASTER_CACHE_MAX_MB // 8; the two
# budgets have no real relationship, and the coupling made this one drift with a GDAL-cache
# tuning decision.
RASTER_BATCH_BUFFER_MB = int(get_config("RASTER_BATCH_BUFFER_MB", "64"))

# Explicit override for the number of raster blocks unioned into one summarize-loop batch,
# bypassing the RASTER_BATCH_BUFFER_MB / band-count derivation in
# api.utils.raster_blocks_per_batch entirely. Unset (None) by default.
_raster_blocks_per_batch = get_config("RASTER_BLOCKS_PER_BATCH", None)
RASTER_BLOCKS_PER_BATCH = int(_raster_blocks_per_batch) if _raster_blocks_per_batch else None

# Ceiling on the number of distinct values `count_by_value` may enumerate (per grid, not
# per zone -- see api.expressions.count_by_value_frame). count_by_value's output row count
# is bounded by (number of summarize zones) x (number of distinct values in the scoped
# selection); the zone count is already bounded by the crosswalk file, but the
# distinct-value count is unbounded unless something stops a caller from pointing
# count_by_value at a continuous column (e.g. elevation) and getting one row per raw
# value. 1000 is comfortably above any real land-type/categorical legend in this dataset
# (the largest is a few dozen categories) while still catching a continuous column early.
COUNT_BY_VALUE_MAX_DISTINCT = int(get_config("COUNT_BY_VALUE_MAX_DISTINCT", "1000"))

# Ceiling on the number of rows an analysis pipeline's vector (zone-side) input may have
# before a `project`/spatial `join` step is executed (api.pipeline_exec) -- the same
# "probe before collecting" shape as the retired `BROADCAST_MAX_ROWS` constant this
# replaces: `lf.select(pl.len()).collect(engine="streaming").item()` is checked against
# this BEFORE the vector frame is materialized into the per-step dense lookup arrays
# `sources._dense_lookup_array` builds, since that array is sized to the vector
# frame's own row count (one entry per zone id) and is held in memory for the whole
# scan. 200,000 is comfortably above any real vector grid in this dataset (the largest,
# clara_grid_cell_v001, has 126,174 zones) while still catching a caller who points a
# pipeline's `project`/`join` step at something unbounded.
ANALYSIS_JOIN_MAX_ROWS = int(get_config("ANALYSIS_JOIN_MAX_ROWS", "200000"))

# Row-offset chunk size for the tabular "fold" (api.fold.fold_group_by), which bounds a
# high-cardinality `group_by` (e.g. `grid_cell`, 126,174 distinct values in
# clara_grid_cell_v001) to at most this many input rows per `group_by`/`agg`/`collect`
# call, instead of one `group_by` seeing an entire unfiltered dataset. Measured on the
# real 1.04B-row damage file (`group_by("grid_cell").sum("value")`, streaming engine,
# unfolded baseline ~9.7 GB / ~11.3s):
#
#   step        peak RSS   wall time
#   25,000,000  ~1.0 GB    ~7.7-9.3s
#   10,000,000  ~664 MB    ~12.7s
#
# 25M is the default: it is the largest step measured (least time overhead) while still
# cutting peak memory ~9-10x, comfortably inside a 15 GB VM. A caller that wants a smaller
# footprint at more time cost can lower this; tuning knobs on the underlying scan
# (`low_memory`, `POLARS_STREAMING_CHUNK_SIZE`) do NOT reproduce this effect on their own
# -- restructuring the query (this fold) is the only lever that measured a real change.
# (Run-to-run variance on this machine is real -- peak RSS at 25M was seen anywhere from
# ~988 MB to ~1.1 GB across repeated measurements; treat these as order-of-magnitude,
# not exact, figures.)
TABULAR_FOLD_STEP = int(get_config("TABULAR_FOLD_STEP", "25000000"))

# Map tile and style URLs
GLYPHS_BASE_URL = get_config("GLYPHS_BASE_URL", "https://api.mpdp.coastal.la.gov/glyphs")
SPRITES_BASE_URL = get_config("SPRITES_BASE_URL", "https://api.mpdp.coastal.la.gov/sprites")
STATIC_TILES_BASE_URL = get_config("STATIC_TILES_BASE_URL", "https://api.mpdp.coastal.la.gov/tiles")
