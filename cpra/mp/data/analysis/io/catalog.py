"""The catalog queries an input needs answered before its frame can be given columns:
which bands does it have.

A raster input needs one because its frame is band-wide, so the *columns* Polars will see
are determined before a single pixel is read, by however many (file, band) combinations
this input resolves to on disk. `resolve_raster_bands` answers exactly that -- the ordered
band list a scan needs, never a filtered or narrowed one.

A vector input needs one only when a `("project", ("vector", "raster"))` step is going to
turn its rows INTO bands. Its own scan (`io/vector.py`) brings every dimension value back
as an ordinary column, so nothing about reading it depends on a separate lookup; but the
band-wide frame that projection produces is one column per distinct combination of that
frame's dimension columns, and that width is a plan-time fact for the same reason a
raster's is -- nothing downstream can be sized, and no ceiling can be checked, by waiting
to see how many columns the data happens to produce. `resolve_vector_bands` answers it,
and `band_dimension_columns` is the single rule both it and
`execution.project.project_vector_to_raster` derive that combination from.

Like `io/vector.py`'s scan, both take `InputSelection` -- `variable`/`time_unit`/`grid` --
as the dataset name: an input names a dataset, it does not filter one (see that model's own
docstring). Every other dataset dimension (`model_group_id`, `scenario_id`, the calendar
dimension, ...) comes back unconstrained, one row per surviving combination, exactly like
`io/vector.py`'s columns carry every value present on disk rather than one. Narrowing which
of those bands survive is a `filter` step's job, applied at whatever position a pipeline
declares it -- not this module's. `resolve_vector_bands`' `predicate`/`columns`/`dimensions`
arguments are not an exception to that: they are the pushdown `execution/plan.py` already
decided for the SCAN, handed here so the enumeration counts the same rows the scan will
return rather than a wider set (see that module's docstring).

Reuses `cpra.mp.data.raster.read_cog` and `io/vector.py`'s own `scan_vector` -- the exact
machinery the old flat/dataset API already has for resolving a dataset's on-disk layout
(UUID-named COGs, per-band calendar values recovered from band descriptions, sorted band
order; Hive partitioning and dataset-dimension literal columns on the tabular side) --
rather than re-deriving any of that a second time here.

`io` may import `request` (used here only for `InputSelection`'s type and two column-name
constants) but never `execution` -- see `tests/api/test_analysis_layering.py`.
"""

from collections.abc import Iterable

import polars as pl

from cpra.mp.data.analysis.io.vector import scan_vector
from cpra.mp.data.analysis.request.errors import NotFoundError
from cpra.mp.data.analysis.request.pipeline import (
    DATASET_SELECTION_COLUMNS,
    RESERVED_BAND_INDEX_COLUMN,
    InputSelection,
)
from cpra.mp.data.catalog import CAL_DIMS
from cpra.mp.data.dimensions import METADATA_DIMENSIONS
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.raster import read_cog
from cpra.mp.data.utils import get_input_data_frame, required_columns

__all__ = [
    "band_dimension_columns",
    "dataset_dimension_columns",
    "raster_path_dimension_columns",
    "resolve_raster_bands",
    "resolve_vector_bands",
]


def resolve_raster_bands(selection: InputSelection, *, base_path=None, path_predicate=None) -> pl.DataFrame:
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

    `path_predicate` narrows which FILES are opened (see `read_cog`), and may name only
    `raster_path_dimension_columns(selection)`. It is a cost decision, not an answer: the
    result is still a superset of what the request needs, and `execution/plan.py`'s
    band-identity pass still makes the real narrowing over whatever comes back. Without
    it every file in the dataset is opened just to learn its band descriptions -- for
    `veg_species` that is every species of every model group, on every map tile.

    Because it is only a cost decision, it never takes a dataset that HAS files down to
    zero bands: a predicate matching no file keeps the dataset's first file instead. Zero
    is a different answer from "some bands, none of which survive", and the band-identity
    pass's join-sibling floor depends on the difference -- an operand narrowed to nothing
    by a conjunct declared AFTER its join gets one band back, which it can only do if the
    enumeration had a band to give. One file open is what that costs.
    """
    selection_kwargs = {
        "base_path": base_path,
        "variable": selection.variable,
        "time_unit": selection.time_unit,
        "grid": selection.grid,
    }
    try:
        bands = read_cog(path_predicate=path_predicate, **selection_kwargs).collect()
        if bands.height == 0 and path_predicate is not None:
            # `glob_lazy_frame` sorts its paths, so "first" is deterministic.
            bands = read_cog(path_predicate=pl.int_range(pl.len()) == 0, **selection_kwargs).collect()
        return bands
    except ValidationError as err:
        raise NotFoundError(str(err)) from err


def raster_path_dimension_columns(selection: InputSelection) -> frozenset[str]:
    """The dimension columns of `selection`'s raster dataset that are encoded in its FILE
    PATHS -- the ones a `resolve_raster_bands` `path_predicate` may name.

    `dataset_dimension_columns` minus the calendar dimension (a band description, so known
    only once a file is opened) and the metadata dimensions (dataset tags, likewise).
    Structural, like `dataset_dimension_columns`: nothing is opened to answer it.
    """
    return dataset_dimension_columns(selection) - frozenset(CAL_DIMS) - frozenset(METADATA_DIMENSIONS)


def band_dimension_columns(names: Iterable[str]) -> list[str]:
    """Which of `names` become PIVOTED BAND dimensions when a long frame is projected onto
    a raster grid: everything except `grid_cell`/`value` and `request.pipeline.
    DATASET_SELECTION_COLUMNS` (`variable`, `time_unit`, `grid`).

    Those three name WHICH DATASET a row came from, not an observation dimension a band
    split should honor -- the same reasoning `execution.long_ops.join_long` applies before
    computing its own join key columns. Leaving them in would either pivot on a column that
    never varies (harmless but wasteful) or, worse, leave a stale source-grid value sitting
    in the emitted metadata once the frame addresses a raster grid instead of the vector one
    it came from.

    Order is `names`' own order, which is what makes this usable against a frame's schema
    directly: `execution.project.project_vector_to_raster` derives the pivot key from a
    LazyFrame's column order and this function derives the enumeration's from the scan's,
    and the two agree because it is one rule applied to one column order.
    """
    return [name for name in names if name not in ("grid_cell", "value") and name not in DATASET_SELECTION_COLUMNS]


def dataset_dimension_columns(selection: InputSelection) -> frozenset[str]:
    """The dimension columns `selection`'s dataset carries, STRUCTURALLY -- derived from
    the variable/grid/time-unit structure alone, without opening anything.

    The same set `band_dimension_columns` would return for a frame of this dataset: the
    required dimensions minus `variable`/`time_unit`/`grid`, which name the dataset rather
    than vary within it. `cpra.mp.data.tabular.read_parquet` builds a scan's columns from
    exactly this computation (`required_columns` over `get_input_data_frame`), so the two
    agree by construction rather than by resemblance; `cpra.mp.data.raster.read_cog` does
    the equivalent for the band side.

    Unordered, and a `frozenset` to say so -- this answers "does this dataset HAVE the
    column a predicate names", which is a membership question. The ordered form is
    `band_dimension_columns` against a real schema, and only that one may drive a pivot.

    `execution.plan.compile_plan` is the caller: before it can decide whether a dimension
    conjunct declared somewhere downstream is pushable back to a given origin, it has to
    know whether that origin has the column at all -- a conjunct naming ANOTHER operand's
    dimension would otherwise reach the scan and fail there. It needs that answer for every
    input on every compile, including the many that never resolve to bands, which is why
    this may not read anything.
    """
    return frozenset(
        required_columns(
            get_input_data_frame(
                variable=selection.variable, time_unit=selection.time_unit, grid=selection.grid
            ),
            include_dataset=False,
        )
    )


def resolve_vector_bands(
    selection: InputSelection,
    *,
    base_path=None,
    columns: list[str] | None = None,
    predicate: pl.Expr | None = None,
    dimensions: frozenset[str] | None = None,
) -> pl.DataFrame | None:
    """The bands a `("project", ("vector", "raster"))` step off `selection`'s dataset
    resolves to: one row per distinct combination of the projected frame's dimension
    columns (`band_dimension_columns`), sorted by those columns, which IS the band order
    that projection emits.

    The returned frame is the projection's band metadata, in the same sense
    `resolve_raster_bands`' is a raster input's -- `ResolvedInput.band_metadata` and this
    are the two halves of one question. A selection whose projected frame has no dimension
    column at all (every one of them collapsed by an upstream `group`) has exactly one
    band, and is answered with a one-row frame carrying `request.pipeline.
    RESERVED_BAND_INDEX_COLUMN` alone -- the same column, for the same reason, that
    `execution.frame.add_reserved_band_index` exists for: Polars cannot represent "zero
    columns, N rows", so the height would otherwise be lost.

    `columns`/`predicate` are the pushdown `execution/plan.py` decided for this origin's
    scan, passed straight through to `io.vector.scan_vector`; the enumeration must see the
    same rows the scan will, or it would count bands the projection never produces.
    `dimensions`, when given, is the set of dimension names still standing at the projection
    (an upstream `group` narrows it to its own `by`), intersected with what the scan
    actually carries.

    A `variable`/`time_unit`/`grid` combination that does not correspond to a real dataset
    fails the underlying read's structural validation with `cpra.mp.data.exceptions.
    ValidationError`, re-raised here as `NotFoundError` -- exactly as in
    `resolve_raster_bands`, and for the same reason.

    **Returns `None` for a real dataset with nothing on disk yet.** This is the same
    judgement `resolve_raster_bands` makes when a valid combination has no files (an empty
    answer to a well-formed question is an answer, not a failure), but it cannot be spelled
    the same way: `read_cog` knows a raster dataset's dimension columns structurally, so an
    empty result there is still a frame with a full schema, whereas an absent Parquet
    directory has no schema to read at all -- not even the column names the enumeration
    would be over. `None` says that honestly, and callers treat it as "not enumerated",
    exactly like a projection whose origin could not be traced. Compiling a plan against a
    dataset that has not been written yet stays possible, which is what `pipeline.json`
    (and every test that inspects a plan without staging data) relies on; whatever actually
    reads the dataset is where its absence is the caller's problem.
    """
    try:
        lf = scan_vector(selection, base_path=base_path, columns=columns, predicate=predicate)
        names = band_dimension_columns(lf.collect_schema().names())
    except ValidationError as err:
        raise NotFoundError(str(err)) from err
    except FileNotFoundError:
        return None
    if dimensions is not None:
        names = [name for name in names if name in dimensions]
    if not names:
        return pl.DataFrame({RESERVED_BAND_INDEX_COLUMN: pl.Series([0], dtype=pl.UInt32)})
    return lf.select(names).unique().sort(names).collect()
