"""Tile-request validation: the two checks a `.png`/`.jpg`/`.tif` or `.pbf` tile route
runs before anything is rendered.

Neither check is op algebra. Both are request LIFECYCLE -- they answer "can this compiled
query answer THIS route at all", from the declared output type, the frame's band count,
and what is on disk under `config.TILES_PATH`. They lived on `BaseAnalysis` only because
that class was also the thing a route held a reference to, and they were already written
against `output_type` rather than against which analysis class produced the frame, so
neither ever needed an override.

`BaseAnalysis` keeps same-named methods as delegations: `api/__init__.py` calls
`validate_tile()`/`validate_vector_tile()` on the compiled object by duck typing, and
`PairedAnalysis` answers the same two names without inheriting them.

`AnalysisError` is imported inside each function, the same local-import idiom
`analysis/utils.py` already uses: it is defined in `analysis/base.py`, which imports this
module at module scope, so a module-scope import back would be a cycle.
"""

import polars as pl

from cpra.mp.data import config
from cpra.mp.data.analysis.types import OutputType


def validate_raster_tile(lf, output_type):
    """Validate that the tile is raster and will have a single band.

    `lf`'s height IS the band count for a raster analysis at every point in the request:
    `lf` is the band metadata, one row per selected band before any group and one row per
    surviving group after one (kept non-degenerate by `raster.BAND_INDEX`). For a vector
    frame it is the ordinary row count, which is what a vector tile's own single-band
    check has always meant -- but a vector OUTPUT is rejected on the line above before the
    count is ever collected, so that reading only applies to a raster grid whose frame has
    already become tabular.
    """
    from cpra.mp.data.analysis.base import AnalysisError

    if output_type == OutputType.vector:
        raise AnalysisError("invalid output type for tile: vector", status_code=400)

    groups = lf.select(pl.len()).collect().item()
    if groups > 1:
        raise AnalysisError(f"tile contains more than one band: {groups} bands found", status_code=400)


def validate_vector_tile(output_type, grid_name):
    """Validate that the tile is a vector grid with an available PMTiles file.

    `grid_name` is the grid whose geometry the tile serves -- the request's
    `summarize_by` destination when it has one, else the dataset's own grid. Read from
    `config.TILES_PATH` at call time, not at import, so a test (or a redeployment) that
    repoints the tiles directory is honored.
    """
    from cpra.mp.data.analysis.base import AnalysisError

    if output_type != OutputType.vector:
        raise AnalysisError("invalid output type for vector tile: raster", status_code=400)

    if not (config.TILES_PATH / f"{grid_name}.pmtiles").exists():
        raise AnalysisError(f"no tile file for grid: {grid_name}", status_code=404)
