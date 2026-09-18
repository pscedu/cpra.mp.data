"""The one catalog query a raster input needs answered: which bands does it have.

A vector input is answered entirely by `io/vector.py`'s ordinary Parquet scan -- every
dimension value present on disk comes back as a column, with no separate lookup. A raster
input cannot work that way: its frame is band-wide, so the *columns* Polars will see are
determined before a single pixel is read, by however many (file, band) combinations this
input resolves to on disk. This module answers exactly that question and nothing else --
`resolve_raster_bands` returns the ordered band list a scan needs, never a filtered or
narrowed one.

Like `io/vector.py`'s scan, this takes `InputSelection` -- `variable`/`time_unit`/`grid` --
and nothing more: an input names a dataset, it does not filter one (see that model's own
docstring). Every other dataset dimension a raster requires (`model_group_id`,
`scenario_id`, the calendar dimension, ...) comes back unconstrained, one row per surviving
combination, exactly like `io/vector.py`'s columns carry every value present on disk rather
than one. Narrowing which of those bands survive is a `filter` step's job, applied at
whatever position a pipeline declares it -- not this module's.

Reuses `cpra.mp.data.raster.read_cog`, the exact machinery the old flat/dataset API already
has for resolving a raster dataset's on-disk layout (UUID-named COGs, per-band calendar
values recovered from band descriptions, sorted band order) -- rather than re-deriving any
of that a second time here.

`io` may import `request` (used here only for `InputSelection`'s type) but never
`execution` -- see `tests/api/test_analysis_layering.py`.
"""

import polars as pl

from cpra.mp.data.analysis.request.errors import NotFoundError
from cpra.mp.data.analysis.request.pipeline import InputSelection
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.raster import read_cog

__all__ = ["resolve_raster_bands"]


def resolve_raster_bands(selection: InputSelection, *, base_path=None) -> pl.DataFrame:
    """The ordered band list `selection`'s raster dataset resolves to.

    One row per (file, band) `read_cog` finds on disk, sorted by `read_cog`'s own
    convention (on-path dimensions, then the calendar dimension) -- this row order IS the
    band order a scan built from this frame must honor (`io/raster.py` turns row *i* into
    VRT source *i* into column `b<i>`, positionally and exactly), so this function does not
    re-sort or re-shuffle anything `read_cog` already decided.

    `base_path` follows `read_cog`'s own default (the real `data-long/` tree) when not
    given -- this module's own tests always pass a `tmp_path`-backed sandbox instead (see
    `tests/conftest.py`'s data guards).

    A `variable`/`time_unit`/`grid` combination that does not correspond to a real dataset
    (the variable does not support that grid's geographic unit, or that time unit) fails
    `read_cog`'s own structural validation with `cpra.mp.data.exceptions.ValidationError`;
    that is re-raised here as `NotFoundError`, since a request naming a combination that
    turns out not to exist is exactly what that exception exists for (see
    `request/errors.py`). A combination that IS a real dataset but has no files on disk
    yet is not an error at all -- it is answered with a zero-row frame, the same way an
    absent dimension value is an empty result rather than a failure everywhere else in
    this package: an empty answer to a well-formed question is an answer.
    """
    try:
        return read_cog(
            base_path=base_path,
            variable=selection.variable,
            time_unit=selection.time_unit,
            grid=selection.grid,
        ).collect()
    except ValidationError as err:
        raise NotFoundError(str(err)) from err
