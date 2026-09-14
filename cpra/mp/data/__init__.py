from pathlib import Path

from cpra.mp.data.raster import compact_cog, read_cog, vrt, write_cog  # noqa
from cpra.mp.data.tabular import compact_parquet, read_parquet, write_parquet
from cpra.mp.data.utils import read_fortran_array, validate_grid  # noqa


def compact_data(**kwargs):
    """Consolidate model data into larger files."""

    grid = validate_grid(**kwargs)

    if grid.grid_type == "vector":
        return compact_parquet(**kwargs)
    return compact_cog(**kwargs)


def read_data(**kwargs):
    """Read model data from disk."""

    grid = validate_grid(**kwargs)

    if grid.grid_type == "raster":
        return read_cog(**kwargs)
    else:
        return read_parquet(**kwargs)


def write_data(data, **kwargs):
    """Write model data to disk."""

    grid = validate_grid(data, **kwargs)

    if grid.grid_type == "raster":
        return write_cog(data, **kwargs)
    else:
        return write_parquet(data, **kwargs)


async def analyze_data(pipeline=None, *, base_path=None, output_path=None, lazy=False, **params):
    """Run an analysis query and hand back its result directly -- the library entry point
    for a notebook, a script, or any other caller that wants `cpra.mp.data.analysis`'s
    query engine (dimension filters, `where`, `group_by`/`group_aggregate`,
    `summarize_by`/`summarize_aggregate`, `grid_cell` sampling, crosswalks, multi-input
    pipelines, ...) without going through the HTTP API or touching the `BaseAnalysis`/
    `PairedAnalysis` objects that engine builds along the way.

    In a notebook, where there's already a running event loop to `await` into::

        df = await analyze_data(
            variable="ssf", time_unit="annual", grid="ecoregion_v001",
            scenario_id=7, group_aggregate="mean",
        )

    From a plain script, wrap the call in `asyncio.run` instead::

        import asyncio
        df = asyncio.run(analyze_data(variable="vct", time_unit="annual",
                                       grid="veg_grid_cell_v001", grid_cell=[1, 2, 3]))

    `read_data` remains the raw, un-analyzed access path: it hands back exactly what's on
    disk (a wide parquet `LazyFrame`, or a raster dataset's own band/path metadata table),
    with no dimension filtering, `where`, aggregation, summarize, or crosswalk applied.
    Reach for `analyze_data` once the request needs any of that; use `read_data` when the
    stored values, untouched, are what's wanted.

    **Two mutually exclusive calling surfaces**, mirroring `cpra.mp.data.analysis`'s own
    two entry points:

    * **Flat** (the common case) -- `variable=`/`time_unit=`/`grid=` name the dataset,
      plus any dimension filter (`scenario_id=7`, or `scenario_id=[7, 8]` for a list) or
      `Options` field (`where=`, `group_by=`, `group_aggregate=`, `summarize_by=`,
      `summarize_aggregate=`, `grid_cell=`, `color_map=`, ...) as further keywords.
      Forwarded to `analysis.create_analysis(**params, base_path=base_path)`; an
      unrecognized keyword raises `AnalysisError` naming it (and every valid name).
    * **Pipeline** -- `analyze_data(pipeline={"inputs": ..., "steps": ..., "output": ...})`,
      the JSON-shaped multi-input pipeline body `analysis.parse_pipeline`/
      `analysis.compile_pipeline` accept (named inputs, an ordered list of
      project/join/filter/group steps, and an output description -- see
      `analysis.pipeline`'s module docstring for the full grammar). Passing `pipeline=`
      together with ANY flat keyword raises `AnalysisError`: the two surfaces describe
      the same kind of request two different ways, not two requests to combine.

    **Return contract** -- follows the compiled analysis's own `output_type`, never
    re-derived from the parameters given directly:

    * A **tabular-shaped** result (`output_type` is `OutputType.vector` -- every vector
      grid, plus a raster grid narrowed to a table by `grid_cell=`/`summarize_by=`, or a
      pipeline compiled the same way) returns a polars `DataFrame` (the default) or a
      `LazyFrame` when `lazy=True` (`collect()` it yourself when ready). `output_path=`
      makes no sense here -- there is no file format decision to make once the caller
      already holds the frame -- and raises `AnalysisError` if given.
    * A **raster-shaped** result (`output_type` is `OutputType.raster`: an unnarrowed
      raster-grid analysis, or any compiled `PairedAnalysis`, which is always
      raster-shaped) REQUIRES `output_path=`: the GeoTIFF is written there via
      `write_raster` and the path is returned as a `pathlib.Path`. Omitting
      `output_path=` raises `AnalysisError` explaining that this request produces a
      raster and needs `output_path=` -- add `summarize_by=`/`grid_cell=` (flat) or a
      `project`-to-a-coarser-grid/trailing `group` step (pipeline) instead if a frame was
      actually wanted. `lazy=True` together with a raster-shaped result also raises
      `AnalysisError`: there is no lazy raster.
    """
    from cpra.mp.data.analysis import (
        AnalysisError,
        compile_pipeline,
        create_analysis,
        parse_pipeline,
    )
    from cpra.mp.data.analysis.types import OutputType

    if pipeline is not None and params:
        raise AnalysisError(
            "analyze_data(): pipeline= and flat keyword parameters (variable=/time_unit=/"
            "grid=/dimension filters/options) are mutually exclusive -- describe the "
            "request one way or the other, not both",
            status_code=400,
        )

    if pipeline is not None:
        compiled = compile_pipeline(parse_pipeline(pipeline), base_path=base_path)
    else:
        compiled = create_analysis(**params, base_path=base_path)

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
                "add summarize_by=/grid_cell= (flat) or a project/group step (pipeline) "
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
