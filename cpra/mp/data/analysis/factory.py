"""`create_analysis`: the single `grid.grid_type` dispatch point for the whole API, and the
supported library entry point for running an analysis without FastAPI or its request-model
objects. A leaf module -- imports `base`/`vector`/`raster`/`parameters`/`grids`, never
`planner`/`paired`/the package `__init__` -- so `planner._dataset_for` can import
`create_analysis` from here without re-entering the `cpra.mp.data.analysis` package while
it's still being initialized (see `analysis/__init__.py`'s module docstring for the
boundary this keeps clean).
"""

from cpra.mp.data.analysis.base import AnalysisError
from cpra.mp.data.analysis.raster import RasterGridAnalysis
from cpra.mp.data.analysis.vector import VectorGridAnalysis


def create_analysis(
    dataset_dimensions=None,
    filter_dimensions=None,
    options=None,
    *,
    variable=None,
    time_unit=None,
    grid=None,
    base_path=None,
    **params,
):
    """Build the `BaseAnalysis` subclass appropriate for the dataset's `grid_type`.

    Two calling conventions:

    * **Plain values** (the public library surface): `variable`/`time_unit`/`grid` name
      the dataset; every other keyword is either a dimension filter (scalar or list,
      e.g. `scenario_id=7` or `scenario_id=[7, 8]`) or an `Options` field under its
      existing name (`where=`, `group_by=`, `group_aggregate=`, `grid_cell=`, ...). An
      unrecognized keyword raises `AnalysisError` naming it and listing every valid
      dimension filter and option name. `analysis.parameters.normalize_analysis_params`
      does the actual normalization into the three objects below.
    * **Validated objects** (the API's fast path): `dataset_dimensions`/
      `filter_dimensions`/`options`, already-validated `DatasetDimensions`/
      `FilterDimensions`/`Options` instances, positionally -- how `api/__init__.py`'s
      route handlers, `analysis.planner._dataset_for`, `visualizations.map.Map`, and
      `tests/api/conftest.py`'s `dataset_factory` fixture already called this before the
      plain-value surface existed, since they hold validated query-model instances
      already and round-tripping them through plain values and back would be pure
      overhead for zero benefit.

    The single `grid.grid_type` dispatch point for the whole API either way.
    """
    if dataset_dimensions is not None:
        if filter_dimensions is None or options is None:
            raise AnalysisError(
                "create_analysis(): dataset_dimensions, filter_dimensions, and options "
                "must all be given together (the validated-object fast path) -- to build "
                "an analysis from plain values instead, pass variable=/time_unit=/grid= "
                "and leave dataset_dimensions unset",
                status_code=400,
            )
        if variable is not None or time_unit is not None or grid is not None or params:
            raise AnalysisError(
                "create_analysis(): cannot mix dataset_dimensions/filter_dimensions/options "
                "with plain-value keywords (variable=/time_unit=/grid=/dimension filters/"
                "options) in the same call",
                status_code=400,
            )
    else:
        from cpra.mp.data.analysis.parameters import normalize_analysis_params

        dataset_dimensions, filter_dimensions, options = normalize_analysis_params(
            variable=variable, time_unit=time_unit, grid=grid, **params
        )

    from cpra.mp.data.grids import GRIDS

    grid_obj = GRIDS[dataset_dimensions.grid]
    if grid_obj.grid_type == "vector":
        analysis_class = VectorGridAnalysis
    elif grid_obj.grid_type == "raster":
        analysis_class = RasterGridAnalysis
    else:
        raise AnalysisError(f"Unsupported grid type for analysis: {grid_obj.grid_type}", status_code=400)

    return analysis_class(dataset_dimensions, filter_dimensions, options, base_path=base_path)
