from pathlib import Path

import polars as pl

from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.raster import compact_cog, read_cog, vrt, write_cog  # noqa
from cpra.mp.data.tabular import compact_parquet, read_parquet, write_parquet
from cpra.mp.data.utils import read_fortran_array, validate_grid  # noqa

# `read_data`'s selection options -- everything besides a dimension filter that
# `**selection` accepts. Kept as a module-level constant (not computed from `DataOptions`
# directly) since `read_data` deliberately does not build a `DataOptions`/pydantic object
# at all -- see `_apply_read_data_selection`'s docstring for why.
_SELECTION_OPTION_NAMES = frozenset({"grid_cell", "where", "sort", "select"})


def compact_data(**kwargs):
    """Consolidate model data into larger files."""

    grid = validate_grid(**kwargs)

    if grid.grid_type == "vector":
        return compact_parquet(**kwargs)
    return compact_cog(**kwargs)


def read_data(*, variable=None, time_unit=None, grid=None, base_path=None, **selection):
    """Read model data from disk -- the library's raw, un-analyzed access path, and the
    `/v1/datasets` equivalent (`analyze_data` is `/v1/analysis`'s: see that function's
    docstring for the split).

    `variable=`/`time_unit=`/`grid=` name the dataset. `**selection` is everything else,
    and is entirely optional:

    * A **dimension filter** (`scenario_id=7`, or `scenario_id=[7, 8]` for a list) narrows
      which rows/bands are read, exactly as it always has -- forwarded straight through to
      `read_cog`/`read_parquet`.
    * `grid_cell=`, `where=`, `sort=`, `select=` -- the same four fields the HTTP data
      endpoints accept (`DataOptions`, minus `color_map`, which only means something for a
      rendered tile) -- are applied, synchronously, on top of whatever `read_cog`/
      `read_parquet` already returned. See `_apply_read_data_selection` for exactly what
      each one does and its limits for a raster grid.

    Called with no `**selection` at all (or dimension filters only), this returns EXACTLY
    what it always has: a vector grid's long-format `grid_cell`/`value` `LazyFrame`, or a
    raster grid's per-band path/metadata `LazyFrame` -- unfiltered by `grid_cell`/`where`,
    unsorted, unnarrowed. Nothing about that path changed; `grid_cell`/`where`/`sort`/
    `select` only ever apply in addition to it, never instead of it.

    An unrecognized keyword raises `cpra.mp.data.exceptions.ValidationError` naming it (and
    every valid dimension filter/option name) -- the same exception type `validate_grid`
    already raises for a missing/unknown `grid`, so `read_data`'s whole error surface stays
    one exception type rather than mixing in the analysis engine's `AnalysisError`.
    """
    grid_obj = validate_grid(variable=variable, time_unit=time_unit, grid=grid)

    from cpra.mp.data.analysis.parameters import BaseFilterDimensions

    filter_names = set(BaseFilterDimensions.model_fields)

    filter_kwargs = {}
    options = {}
    unknown = []
    for name, value in selection.items():
        if name in _SELECTION_OPTION_NAMES:
            options[name] = value
        elif name in filter_names:
            filter_kwargs[name] = value
        else:
            unknown.append(name)

    if unknown:
        valid = ", ".join(sorted({"variable", "time_unit", "grid"} | filter_names | _SELECTION_OPTION_NAMES))
        raise ValidationError(
            f"read_data() got unexpected keyword argument(s): {', '.join(sorted(unknown))}. "
            f"Valid parameters are: {valid}."
        )

    read_kwargs = dict(variable=variable, time_unit=time_unit, grid=grid, base_path=base_path, **filter_kwargs)
    if grid_obj.grid_type == "raster":
        lf = read_cog(**read_kwargs)
    else:
        lf = read_parquet(**read_kwargs)

    if not options:
        return lf

    try:
        return _apply_read_data_selection(lf, grid_obj, options)
    except Exception as exc:  # AnalysisError or ExpressionError -- see the helper's docstring
        from cpra.mp.data.analysis.base import AnalysisError
        from cpra.mp.data.analysis.expressions import ExpressionError

        if isinstance(exc, (AnalysisError, ExpressionError)):
            raise ValidationError(str(exc)) from exc
        raise


def _apply_read_data_selection(lf, grid, options):
    """Apply `read_data`'s `grid_cell`/`where`/`sort`/`select` options to `lf`, the
    `LazyFrame` `read_cog`/`read_parquet` already returned for `grid`.

    Deliberately does NOT build a `create_analysis`/`BaseAnalysis` object to do this, for
    two concrete reasons rather than as an arbitrary style choice:

    1. **Recursion** -- `analysis.utils.get_lazy_frame` (which `BaseAnalysis.__init__`
       calls to populate its own `self.lf`) calls `cpra.mp.data.read_data` directly. Any
       code path from here into `create_analysis` risks calling back into `read_data`.
    2. **Sync/async mismatch** -- `read_data` is, and must stay, a plain synchronous
       function (this module's whole `read_data`/`write_data`/`compact_data` family is
       sync). `BaseAnalysis.apply_params()` is `async def`, and its vector
       `_apply_grid_cell` `await`s coordinate resolution -- there is no way to route
       through it without either making `read_data` async (breaking its signature) or
       driving an event loop from inside a sync call (fragile, and surprising for a
       caller already inside one, e.g. a notebook cell).

    Instead, this reuses the same underlying, grid-engine-independent building blocks
    `BaseAnalysis` itself is built from: `analysis.expressions.classify_where`/
    `value_predicate_mask` (explicitly documented, in that module's own docstring, as
    written to be reused here), and `analysis.geography.resolve_vector_grid_cell_sync`
    (the same blocking coordinate-resolution path `visualizations/map.py` already uses
    outside an event loop). Applied directly to `lf`, not through `Options`/`DataOptions`.

    `grid_cell` and a `where` VALUE conjunct both require a `value` column to mean
    anything -- present on a vector grid's frame, but never present on a raster grid's
    metadata table (`read_data` never reads raster pixels; see `raster.py::read_cog`'s
    docstring). Both raise for a raster grid, pointing the caller at `analyze_data`
    instead, rather than silently doing nothing or reinterpreting the request.

    `sort`/`select` are validated against `lf`'s own live schema (not the analysis
    engine's `DIMENSION_CHOICES`/`required_columns`-derived whitelists) -- those
    whitelists assume `BaseAnalysis`'s own narrowed-tabular-output shape (vector
    dimension columns plus `grid_cell`/`value`), which is a genuinely different, and for
    a raster grid strictly smaller, set of columns than what `read_data` actually
    returns here (a raster's `path`/`band` columns, in particular, are real and useful to
    `select`/`sort` by here even though `BaseAnalysis` never exposes them to `Options`).
    """
    from cpra.mp.data.analysis.base import AnalysisError
    from cpra.mp.data.analysis.expressions import classify_where, value_predicate_mask
    from cpra.mp.data.analysis.geography import parse_coordinate_pairs, resolve_vector_grid_cell_sync
    from cpra.mp.data.analysis.utils import unpack_values

    schema_names = set(lf.collect_schema().names())
    has_value_column = "value" in schema_names

    grid_cell = options.get("grid_cell")
    where = options.get("where")
    sort = options.get("sort")
    select = options.get("select")

    if grid_cell is not None:
        if grid.grid_type != "vector":
            raise AnalysisError(
                "read_data(): grid_cell samples raster PIXEL values, which read_data() "
                "never touches -- it always returns the raster's band/path metadata "
                "table here, never narrowed by grid_cell -- use analyze_data() instead, "
                "with a pipeline whose input carries `grid_cell` (or that ends in a "
                "`project`/`group` step) to get a frame back, or `output_path=` to write "
                "the raster itself",
                status_code=400,
            )
        values = unpack_values(grid_cell, "grid_cell")
        if all(value.isdigit() for value in values):
            grid_cell_ids = [int(value) for value in values]
        else:
            lngs, lats = parse_coordinate_pairs(values)
            resolved = resolve_vector_grid_cell_sync(grid.name, lngs, lats)
            grid_cell_ids = [geo_id for geo_id in resolved if geo_id is not None]
        lf = lf.filter(pl.col("grid_cell").is_in(grid_cell_ids))

    if where is not None:
        dimension_conjuncts, value_conjuncts = classify_where(where)
        if dimension_conjuncts:
            combined = " and ".join(f"({conjunct})" for conjunct in dimension_conjuncts)
            lf = lf.sql(f"select * from self where {combined}")
        if value_conjuncts:
            if not has_value_column:
                raise AnalysisError(
                    "read_data(): where's value condition(s) need a `value` column, "
                    "which this result doesn't have -- raster grids return band/path "
                    "metadata here, not pixel values -- restrict where to dimension "
                    "conditions only, or use analyze_data() instead",
                    status_code=400,
                )
            combined = " and ".join(f"({conjunct})" for conjunct in value_conjuncts)
            mask = value_predicate_mask(combined, sorted(schema_names))
            lf = lf.filter(mask)

    if sort is not None:
        sort_choices = sorted(schema_names) + [f"-{name}" for name in sorted(schema_names)]
        parts = unpack_values(sort, "sort", sort_choices)
        cols = [part.removeprefix("-") for part in parts]
        desc = [part.startswith("-") for part in parts]
        lf = lf.sort(cols, descending=desc)

    if select is not None:
        select_cols = unpack_values(select, "select", sorted(schema_names))
        lf = lf.select(select_cols)

    return lf


def write_data(data, **kwargs):
    """Write model data to disk."""

    grid = validate_grid(data, **kwargs)

    if grid.grid_type == "raster":
        return write_cog(data, **kwargs)
    else:
        return write_parquet(data, **kwargs)


async def analyze_data(pipeline, *, base_path=None, output_path=None, lazy=False):
    """Compile and run a pipeline analysis and hand back its result directly -- the
    library entry point for a notebook, a script, or any other caller that wants
    `cpra.mp.data.analysis`'s query engine (dimension filters, `where`,
    `group_by`/`group_aggregate`, `summarize_by`/`summarize_aggregate`, `grid_cell`
    sampling, crosswalks, multi-input joins, ...) without going through the HTTP API or
    touching the `BaseAnalysis`/`PairedAnalysis` objects that engine builds along the way.
    `read_data` is the equivalent for the raw, un-analyzed access path (`/v1/datasets`);
    this is `/v1/analysis`'s.

    `pipeline` is the JSON-shaped multi-input pipeline body `analysis.parse_pipeline`/
    `analysis.compile_pipeline` accept: named inputs, an ordered list of
    project/join/filter/group/calculate steps, and an output description -- see
    `analysis.pipeline`'s module docstring for the full grammar. It is the ONLY way to
    describe a request here; there is no flat/keyword calling surface any more (compare
    `cpra.mp.data.analysis.create_analysis`, which still has one, for a caller that wants
    it directly rather than through this wrapper).

    In a notebook, where there's already a running event loop to `await` into::

        df = await analyze_data({
            "inputs": {"eco": {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"}},
            "steps": [
                {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                {"op": "group", "by": [], "aggregate": "mean"},
            ],
            "output": {"format": "csv"},
        })

    From a plain script, wrap the call in `asyncio.run` instead::

        import asyncio
        df = asyncio.run(analyze_data(pipeline))

    **Return contract** -- follows the compiled analysis's own `output_type`, never
    re-derived from the parameters given directly:

    * A **tabular-shaped** result (`output_type` is `OutputType.vector` -- every vector
      grid, plus a raster grid narrowed to a table by a `project`-to-a-coarser-grid or a
      trailing `group` step) returns a polars `DataFrame` (the default) or a `LazyFrame`
      when `lazy=True` (`collect()` it yourself when ready). `output_path=` makes no
      sense here -- there is no file format decision to make once the caller already
      holds the frame -- and raises `AnalysisError` if given.
    * A **raster-shaped** result (`output_type` is `OutputType.raster`: an unnarrowed
      raster-grid analysis, or any compiled `PairedAnalysis`, which is always
      raster-shaped) REQUIRES `output_path=`: the GeoTIFF is written there via
      `write_raster` and the path is returned as a `pathlib.Path`. Omitting
      `output_path=` raises `AnalysisError` explaining that this request produces a
      raster and needs `output_path=` -- add a `project`-to-a-coarser-grid or a trailing
      `group` step instead if a frame was actually wanted. `lazy=True` together with a
      raster-shaped result also raises `AnalysisError`: there is no lazy raster.
    """
    from cpra.mp.data.analysis import AnalysisError, compile_pipeline, parse_pipeline
    from cpra.mp.data.analysis.types import OutputType

    compiled = compile_pipeline(parse_pipeline(pipeline), base_path=base_path)

    await compiled.apply_params()

    if compiled.output_type == OutputType.raster:
        if lazy:
            raise AnalysisError(
                "analyze_data(): lazy=True has no meaning for a raster-shaped result -- "
                "there is no lazy raster; omit lazy= for this request",
                status_code=400,
            )
        if output_path is None:
            raise AnalysisError(
                "analyze_data(): this request produces a raster (its output is an image, "
                "not a table) and needs output_path= to say where to write the GeoTIFF -- "
                "add a project-to-a-coarser-grid or a trailing group step to the pipeline "
                "instead if you wanted a polars frame back",
                status_code=400,
            )
        output_path = Path(output_path)
        await compiled.write_raster(str(output_path))
        return output_path

    if output_path is not None:
        raise AnalysisError(
            "analyze_data(): output_path= was given but this request produces a tabular "
            "result (a polars DataFrame/LazyFrame), not a raster -- drop output_path= and "
            "use the returned frame directly",
            status_code=400,
        )

    if lazy:
        return compiled.materialize("lazy")
    return compiled.materialize()
