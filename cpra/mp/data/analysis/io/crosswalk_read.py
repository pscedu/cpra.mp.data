"""The vector crosswalk scan: `read_vector_crosswalk`, this module's only export, opens
one directed (source grid, destination grid) crosswalk artifact as a `pl.LazyFrame` of
`(src_id, dst_id, scale_factor)` rows -- one row per overlapping (source cell,
destination cell) pair.

`scale_factor` is an allocation weight, not a probability or a normalized share: it is
the fraction of the SOURCE cell's own area that lies inside the destination cell. A
source cell entirely inside one destination gets a single row at `scale_factor == 1.0`; a
source cell straddling several destinations gets one row per destination it touches, and
those rows' factors sum to at most 1.0 (a pair that only shares a boundary, with zero
overlap area, contributes nothing and is never written -- there is no zero-weight row to
filter out). `execution/project.py` is the only reader today, and is where that weight
gets applied differently per aggregate.

Crosswalk artifacts are built ahead of time (`python -m cpra.mp.data.cli crosswalk`),
never invented at read time -- this module only opens what is already on disk.

`io` may import `request` (used here only for `NotFoundError`) but never `execution` --
see `tests/api/test_analysis_layering.py`.
"""

import polars as pl

from cpra.mp.data import config
from cpra.mp.data.analysis.request.errors import NotFoundError

__all__ = ["read_vector_crosswalk"]


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
    the vector-to-vector counterpart of the old flat API's `AnalysisError("No crosswalk
    from ... to ...")`, in this module's own exception family instead, since nothing under
    `request/`/`io/`/`execution/` may raise the legacy `analysis.base.AnalysisError`.
    """
    path = config.CROSSWALKS_PATH / f"{src_grid}__{dst_grid}.parquet"
    if not path.exists():
        raise NotFoundError(f"no crosswalk from {src_grid!r} to {dst_grid!r} (expected at {path})")
    return pl.scan_parquet(path)
