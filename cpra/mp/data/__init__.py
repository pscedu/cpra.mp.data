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
    one exception type rather than mixing in the analysis engine's own error family.
    """
    grid_obj = validate_grid(variable=variable, time_unit=time_unit, grid=grid)

    from cpra.mp.data.analysis.request.parameters import BaseFilterDimensions

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
    except Exception as exc:  # RequestError or ExpressionError -- see the helper's docstring
        from cpra.mp.data.analysis.request.errors import RequestError
        from cpra.mp.data.analysis.request.expressions import ExpressionError

        if isinstance(exc, (RequestError, ExpressionError)):
            raise ValidationError(str(exc)) from exc
        raise


def _apply_read_data_selection(lf, grid, options):
    """Apply `read_data`'s `grid_cell`/`where`/`sort`/`select` options to `lf`, the
    `LazyFrame` `read_cog`/`read_parquet` already returned for `grid`.

    Deliberately does NOT build and run a `Pipeline` to do this, for two concrete reasons
    rather than as an arbitrary style choice:

    1. **Recursion** -- the analysis engine's own readers call `cpra.mp.data.read_data`
       directly to populate the frames a pipeline runs over. Any code path from here into
       the engine's entry points risks calling back into `read_data`.
    2. **Sync/async mismatch** -- `read_data` is, and must stay, a plain synchronous
       function (this module's whole `read_data`/`write_data`/`compact_data` family is
       sync). The engine's entry point (`analyze_data`) is `async def`, and coordinate
       resolution for a `grid_cell` filter is awaited -- there is no way to route through
       it without either making `read_data` async (breaking its signature) or driving an
       event loop from inside a sync call (fragile, and surprising for a caller already
       inside one, e.g. a notebook cell).

    Instead, this reuses the same underlying, grid-engine-independent building blocks the
    engine itself is built from: `analysis.request.expressions.classify_where`/
    `value_predicate_mask` (explicitly documented, in that module's own docstring, as
    written to be reused here), and `analysis.io.geography.resolve_vector_grid_cell_sync`
    (the same blocking coordinate-resolution path `visualizations/map.py` already uses
    outside an event loop). Applied directly to `lf`, not through a pipeline.

    `grid_cell` and a `where` VALUE conjunct both require a `value` column to mean
    anything -- present on a vector grid's frame, but never present on a raster grid's
    metadata table (`read_data` never reads raster pixels; see `raster.py::read_cog`'s
    docstring). Both raise for a raster grid, pointing the caller at `analyze_data`
    instead, rather than silently doing nothing or reinterpreting the request.

    `sort`/`select` are validated against `lf`'s own live schema, not against any fixed
    column whitelist -- a whitelist would have to assume the engine's narrowed
    tabular-output shape (vector dimension columns plus `grid_cell`/`value`), which is a
    genuinely different, and for a raster grid strictly smaller, set of columns than what
    `read_data` actually returns here (a raster's `path`/`band` columns, in particular,
    are real and useful to `select`/`sort` by here).
    """
    from cpra.mp.data.analysis.io.geography import parse_coordinate_pairs, resolve_vector_grid_cell_sync
    from cpra.mp.data.analysis.io.raster_io import unpack_values
    from cpra.mp.data.analysis.request.errors import RequestError
    from cpra.mp.data.analysis.request.expressions import classify_where, combine_conjuncts, value_predicate_mask

    schema_names = set(lf.collect_schema().names())
    has_value_column = "value" in schema_names

    grid_cell = options.get("grid_cell")
    where = options.get("where")
    sort = options.get("sort")
    select = options.get("select")

    if grid_cell is not None:
        if grid.grid_type != "vector":
            raise RequestError(
                "read_data(): grid_cell samples raster PIXEL values, which read_data() "
                "never touches -- it always returns the raster's band/path metadata "
                "table here, never narrowed by grid_cell -- use analyze_data() instead, "
                "with a pipeline that filters on `grid_cell` (or that ends in a "
                "`project`/`group` step) to get a frame back, or `output_path=` to write "
                "the raster itself"
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
            lf = lf.sql(f"select * from self where {combine_conjuncts(dimension_conjuncts)}")
        if value_conjuncts:
            if not has_value_column:
                raise RequestError(
                    "read_data(): where's value condition(s) need a `value` column, "
                    "which this result doesn't have -- raster grids return band/path "
                    "metadata here, not pixel values -- restrict where to dimension "
                    "conditions only, or use analyze_data() instead"
                )
            mask = value_predicate_mask(combine_conjuncts(value_conjuncts), sorted(schema_names))
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
    `cpra.mp.data.analysis`'s query engine (filters, `calculate`, `group`, crosswalk
    `project`s, multi-input joins, ...) without going through the HTTP API.
    `read_data` is the equivalent for the raw, un-analyzed access path (`/v1/datasets`);
    this is `/v1/analysis`'s.

    Runs the SAME four stages an HTTP response does, in the same order --
    `request.pipeline.parse_pipeline` (structure), `request.ops.validate_shapes`
    (shape/format agreement), `execution.plan.compile_plan`, then either
    `execution.execute.execute` for a frame or `execution.execute.run_to_path` for a
    GeoTIFF. There is no second, library-only engine here; a request that answers one way
    over HTTP answers the same way from a notebook, including its rejections
    (`request.errors.RequestError` and its subclasses).

    `pipeline` is the JSON-shaped multi-input pipeline body `parse_pipeline` accepts:
    named inputs (a `variable`/`time_unit`/`grid` dataset selection each, and nothing
    else -- every filter is a `filter` STEP), an ordered list of project/join/filter/
    group/calculate steps, and an output description. It is the ONLY way to describe a
    request here; there is no flat/keyword calling surface.

    Still a coroutine, though nothing inside it awaits any more (the new execution path
    is synchronous end to end): `await analyze_data(...)` is the spelling every existing
    notebook and script uses, and changing it would break them for no gain here. In a
    notebook, where there's already a running event loop::

        df = await analyze_data({
            "inputs": {"eco": {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"}},
            "steps": [
                {"op": "filter", "input": "eco", "where": "model_group_id = 500 AND scenario_id = 7"},
                {"op": "group", "by": ["calendar_year"], "aggregate": "mean"},
            ],
            "output": {"format": "csv"},
        })

    From a plain script, wrap the call in `asyncio.run` instead::

        import asyncio
        df = asyncio.run(analyze_data(pipeline))

    **Return contract** -- follows `output.format`, which `validate_shapes` has already
    forced to agree with the pipeline's actual final frame shape, so this is not a second
    guess at that shape:

    * A **tabular** format (`csv`/`ndjson`/`parquet`, i.e. a long final frame) returns a
      polars `DataFrame`, or a `LazyFrame` when `lazy=True` (`collect()` it yourself when
      ready). `output_path=` makes no sense here -- there is no format decision left once
      the caller holds the frame -- and raises `ShapeError` if given.
    * `format="tif"` (a band-wide final frame) REQUIRES `output_path=`: the GeoTIFF is
      written there via `run_to_path` and the path is returned as a `pathlib.Path`.
      Omitting it raises `ShapeError` -- add a `project`-to-a-vector-grid or a trailing
      `group` step instead if a frame was actually wanted. `lazy=True` with `tif` raises
      too: there is no lazy raster. A selection that leaves the result with NO bands
      writes no file at all (`run_to_path` returns `False`), and raises `NotFoundError`
      rather than handing back a path to a file that isn't there -- the same 404 the HTTP
      routes answer that request with.
    * `format="png"`/`"jpg"` are TILE formats and are rejected outright. Both need a
      tile's `x`/`y`/`z` and pixel size, which this signature has nowhere to put --
      `execution.sink.render_tile` is the entry point for those.
    """
    from cpra.mp.data.analysis.execution.execute import execute, run_to_path
    from cpra.mp.data.analysis.execution.plan import compile_plan
    from cpra.mp.data.analysis.execution.sink import apply_output_shaping
    from cpra.mp.data.analysis.request.errors import NotFoundError, ShapeError
    from cpra.mp.data.analysis.request.ops import validate_shapes
    from cpra.mp.data.analysis.request.pipeline import parse_pipeline
    from cpra.mp.data.analysis.request.types import DataFormat, TileFormat

    parsed = parse_pipeline(pipeline)
    validate_shapes(parsed)
    format = parsed.output.format

    if format in {TileFormat.png.value, TileFormat.jpg.value}:
        raise ShapeError(
            f"analyze_data(): output.format={format!r} is a map TILE format, which needs "
            "a tile's x/y/z and pixel size -- analyze_data() has nowhere to put those. "
            "Use format='tif' with output_path= for the whole raster, or call "
            "cpra.mp.data.analysis.execution.sink.render_tile directly for one tile"
        )

    if format == DataFormat.tif.value:
        if lazy:
            raise ShapeError(
                "analyze_data(): lazy=True has no meaning for output.format='tif' -- "
                "there is no lazy raster; omit lazy= for this request"
            )
        if output_path is None:
            raise ShapeError(
                "analyze_data(): output.format='tif' produces a raster (an image, not a "
                "table) and needs output_path= to say where to write the GeoTIFF -- add "
                "a project-to-a-vector-grid or a trailing group step to the pipeline "
                "instead if you wanted a polars frame back"
            )
        output_path = Path(output_path)
        if not run_to_path(parsed, output_path, base_path=base_path):
            raise NotFoundError(
                "analyze_data(): this pipeline's raster result has no bands -- its "
                "selection matched nothing, so there was no band count to declare and "
                f"no GeoTIFF was written to {output_path}"
            )
        return output_path

    if output_path is not None:
        raise ShapeError(
            f"analyze_data(): output_path= was given but output.format={format!r} "
            "produces a tabular result (a polars DataFrame/LazyFrame), not a raster -- "
            "drop output_path= and use the returned frame directly"
        )

    # `apply_output_shaping` rather than a bare `execute`, so `output.sort`/`output.select`
    # mean the same thing here as they do over HTTP -- `run_to_path`'s tabular branch
    # applies exactly this, once, last.
    frame = apply_output_shaping(
        execute(compile_plan(parsed, base_path=base_path), base_path=base_path), parsed.output
    )
    return frame if lazy else frame.collect()
