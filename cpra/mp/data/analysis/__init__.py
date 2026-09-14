"""Analysis layer: a `BaseAnalysis` template-method hierarchy for running one data-API
query against one dataset (the former `api/dataset.py`'s `Dataset` class), plus the
grid-agnostic helpers it's built from (expressions, fold, sources, types, parameters,
utils, geography) and the paired-plan execution path (`planner`/`paired`).

Nothing under this package may import FastAPI -- see `tests/api/test_analysis_boundary.py`,
which walks every module here and asserts it. `analysis.base.AnalysisError` stands in for
`HTTPException`; `api/__init__.py` registers the one place that translates it back into an
HTTP response.

This module is also the supported library entry point for running an analysis without
FastAPI or its request-model objects -- a notebook, a script, or a future `read_data`
extension. `create_analysis` takes plain values (dimension names, filter values, option
names under their existing API names) rather than the `DatasetDimensions`/
`FilterDimensions`/`Options` pydantic models `analysis.parameters` normalizes those values
into internally -- that split is an implementation detail, not something callers need to
know about:

    from cpra.mp.data.analysis import create_analysis

    analysis = create_analysis(
        variable="ssf", time_unit="annual", grid="ecoregion_v001",
        scenario_id=7, group_aggregate="mean",
    )
    await analysis.apply_params()  # the query stages are async
    df = analysis.materialize()  # -> a polars DataFrame (or LazyFrame; see `method=`)

`parse_pipeline`/`compile_pipeline` are the equivalent entry point for the multi-input
pipeline shape (see `analysis.pipeline`'s module docstring for the request representation,
and `analysis.planner`'s for what compiles): `parse_pipeline` validates a JSON-shaped
pipeline body into a `Pipeline`, and `compile_pipeline` compiles it into a `BaseAnalysis`
or `PairedAnalysis`, the same objects `create_analysis` builds directly.
"""

from cpra.mp.data.analysis.base import AnalysisError, BaseAnalysis
from cpra.mp.data.analysis.factory import create_analysis
from cpra.mp.data.analysis.paired import PairedAnalysis
from cpra.mp.data.analysis.pipeline import PipelineError, parse_pipeline
from cpra.mp.data.analysis.planner import compile_pipeline
from cpra.mp.data.analysis.raster import RasterGridAnalysis
from cpra.mp.data.analysis.vector import VectorGridAnalysis

__all__ = [
    "AnalysisError",
    "BaseAnalysis",
    "PairedAnalysis",
    "PipelineError",
    "RasterGridAnalysis",
    "VectorGridAnalysis",
    "compile_pipeline",
    "create_analysis",
    "parse_pipeline",
]
