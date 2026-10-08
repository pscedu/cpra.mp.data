"""Crosswalk path/read resolution for both directions `execution/project.py` needs:
`read_vector_crosswalk` opens the tabular vector-to-vector artifact directly, and
`resolve_raster_crosswalk_path` locates (but does not open) the raster-to-vector
artifact, since that one is read through `io/raster.py::scan_raster` instead of a plain
Parquet scan -- see that function's own module docstring for why a crosswalk rides along
in the same VRT read as an ordinary band.

`read_vector_crosswalk` opens one directed (source grid, destination grid) crosswalk
artifact as a `pl.LazyFrame` of `(src_id, dst_id, scale_factor)` rows -- one row per
overlapping (source cell, destination cell) pair.

`scale_factor` is an allocation weight, not a probability or a normalized share: it is
the fraction of the SOURCE cell's own area that lies inside the destination cell. A
source cell entirely inside one destination gets a single row at `scale_factor == 1.0`; a
source cell straddling several destinations gets one row per destination it touches, and
those rows' factors sum to at most 1.0 (a pair that only shares a boundary, with zero
overlap area, contributes nothing and is never written -- there is no zero-weight row to
filter out). `execution/project.py` is the only reader today, and is where that weight
gets applied differently per aggregate.

`resolve_raster_crosswalk_path` names a structurally different artifact: a single-band
GeoTIFF of destination zone ids, co-registered with the raster grid, built by
`cpra.mp.data.crosswalks.write_raster_crosswalk` and named `{raster_grid}__{vector_grid}
.tif` (raster grid name FIRST, same directed-pair convention as the vector artifact's
`{src_grid}__{dst_grid}.parquet`). ONE artifact serves both raster-touching `project`
directions -- raster-to-vector groups pixels by it, vector-to-raster joins each pixel's
zone onto it -- so this module resolves it once rather than once per direction.

Crosswalk artifacts are built ahead of time (`python -m cpra.mp.data.cli crosswalk`),
never invented at read time -- this module only opens (or, for the raster one, locates)
what is already on disk.

`io` may import `request` (used here only for `NotFoundError`) but never `execution` --
see `tests/api/test_analysis_layering.py`.
"""

from pathlib import Path

import polars as pl

from cpra.mp.data import config
from cpra.mp.data.analysis.request.errors import NotFoundError

__all__ = ["read_vector_crosswalk", "resolve_raster_crosswalk_path"]


def read_vector_crosswalk(src_grid: str, dst_grid: str) -> pl.LazyFrame:
    """Open the crosswalk from `src_grid` to `dst_grid` -- crosswalks are directed, so
    this is not interchangeable with `read_vector_crosswalk(dst_grid, src_grid)`, which
    names a different (and separately built) artifact.

    Reads `config.CROSSWALKS_PATH` as a module attribute at call time (`config.
    CROSSWALKS_PATH`, not a name imported out of it) so that `tests/conftest.py`'s
    `crosswalk_path` fixture -- which monkeypatches that attribute onto an isolated
    directory -- actually takes effect; a direct `from ... import CROSSWALKS_PATH` would
    capture the production default at import time and never see the patch.

    Raises `request.errors.NotFoundError` if no artifact exists for this exact pair --
    this package's own exception family, which `api/boundary.py` maps to 404.
    """
    path = config.CROSSWALKS_PATH / f"{src_grid}__{dst_grid}.parquet"
    if not path.exists():
        raise NotFoundError(f"no crosswalk from {src_grid!r} to {dst_grid!r} (expected at {path})")
    return pl.scan_parquet(path)


def resolve_raster_crosswalk_path(raster_grid: str, vector_grid: str) -> Path:
    """The path of the raster-to-vector crosswalk artifact between `raster_grid` and
    `vector_grid` -- directed, in that order, same as `read_vector_crosswalk`'s pair is
    not interchangeable with its reverse.

    Resolves only -- does not open or read the file, unlike `read_vector_crosswalk`,
    because this artifact is a GeoTIFF read through `io/raster.py::scan_raster` (a
    windowed, batched raster read, not a single Parquet scan), and that function needs a
    path to include in its VRT, not an already-opened frame. Reading `config.
    CROSSWALKS_PATH` as a module attribute at call time, for the same reason
    `read_vector_crosswalk` does: so `tests/conftest.py`'s `crosswalk_path` fixture, which
    monkeypatches that attribute, actually takes effect.

    Raises `request.errors.NotFoundError` if no artifact exists for this exact pair --
    the raster-crosswalk counterpart of `read_vector_crosswalk`'s own check.
    """
    path = config.CROSSWALKS_PATH / f"{raster_grid}__{vector_grid}.tif"
    if not path.exists():
        raise NotFoundError(f"no raster crosswalk from {raster_grid!r} to {vector_grid!r} (expected at {path})")
    return path
