"""The vector scan: `scan_vector`, this module's only export, opens one input's Parquet
dataset as a long `pl.LazyFrame` -- dimension columns, `grid_cell`, `value`, one row per
observation (spec's vector row model).

`request.pipeline.InputSelection` names a dataset (`variable`/`time_unit`/`grid`); it does
not filter one -- see that model's own docstring. This scan takes exactly that much and
nothing more: it never narrows to one model run, one percentile, or one time slice, because
that narrowing is always expressed as a `filter` step, applied later in the walk
(`execution/execute.py`'s long-shape `filter` lowering), not baked into what the scan
opens. Every metadata dimension a dataset's structure requires (`model_group_id`,
`scenario_id`, `percentile`/`aggregate_type`, `model`, ...) therefore comes back as an
ordinary column carrying every value actually present on disk, exactly like any other
dimension column -- there is nothing dataset-input-specific about it once the frame is
open.

Reuses `cpra.mp.data.tabular.read_parquet` -- the exact machinery `dataset_path`/
`get_filter_expressions`/`partition_columns`/`required_columns`/`get_input_data_frame`
already give the old flat/dataset API for resolving a dataset's on-disk layout (Hive
partitioning, dataset-dimension literal columns, required-column projection) -- rather
than re-deriving any of that a second time here.

`columns`/`predicate` are the pushdown seam `execution/plan.py`'s `ResolvedInput` carries:
nothing in this vertical slice computes a real projection/predicate decision (that is
later-stage work), but the scan already knows how to use one once a plan supplies it, via
an ordinary `.select`/`.filter` Polars can push into partition pruning and row-group
statistics -- see that module's own docstring for why the decision lives on the plan
rather than being invented here.

`io` may import `request` (used here only for `InputSelection`'s type) but never
`execution` -- see `tests/api/test_analysis_layering.py`.
"""

import polars as pl

from cpra.mp.data.analysis.request.pipeline import InputSelection
from cpra.mp.data.tabular import read_parquet

__all__ = ["scan_vector"]


def scan_vector(
    selection: InputSelection,
    *,
    base_path=None,
    columns: list[str] | None = None,
    predicate: pl.Expr | None = None,
) -> pl.LazyFrame:
    """Open `selection`'s dataset as a long frame.

    `base_path` follows `read_parquet`'s own default (the real `data-long/` tree via
    `cpra.mp.data.config.get_base_path`) when not given -- this module's own tests always
    pass a `tmp_path`-backed sandbox instead (see `tests/conftest.py`'s data guards, and
    the module docstring above for why this scan never writes and reads only what a caller
    explicitly names).

    `columns`/`predicate`, when given, narrow what comes back (projection and predicate
    pushdown respectively) but never change its MEANING -- applying them here, against the
    lazy scan itself, is what lets Polars push a predicate into row-group statistics and a
    projection into which columns are even read off disk, rather than merely which are
    kept in memory afterward.
    """
    lf = read_parquet(
        base_path=base_path,
        variable=selection.variable,
        time_unit=selection.time_unit,
        grid=selection.grid,
    )
    if predicate is not None:
        lf = lf.filter(predicate)
    if columns is not None:
        lf = lf.select(columns)
    return lf
