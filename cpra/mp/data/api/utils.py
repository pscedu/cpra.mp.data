from fastapi import HTTPException

from cpra.mp.data.analysis.types import CompressionAlgorithm, DataFormat, OutputType
from cpra.mp.data.visualizations import VISUALIZATIONS

# The named-analysis registry lands in a concurrent change
# under `cpra/mp/data/analyses/` -- a package this module (and `api/__init__.py`, which
# re-imports these three names from here) does not own and must not create or edit. This
# import is guarded so the rest of the API keeps working -- and the whole app keeps
# importing -- whether or not that package has landed yet: every `/v1/analyses*` route
# simply finds `ANALYSES` empty (a 404 "unknown analysis name" for any name) until it
# does. Only `ModuleNotFoundError` is swallowed here, not `ImportError` in general -- if
# the package exists but this exact name set is wrong (e.g. a rename during
# reconciliation), that must fail loudly rather than silently fall back to this stub.
try:
    from cpra.mp.data.analyses import ANALYSES, DatasetRef, resolve_analysis
    from cpra.mp.data.analyses.base import (
        AnalysisRegistryError,
        AnalysisTypeMismatchError,
        InvalidAnalysisParametersError,
        UnknownAnalysisError,
    )
except ModuleNotFoundError:  # pragma: no cover -- exercised once the registry package lands
    ANALYSES = {}
    DatasetRef = None
    resolve_analysis = None

    class AnalysisRegistryError(ValueError):
        """Stand-in for `cpra.mp.data.analyses.base.AnalysisRegistryError`, used only so
        `api/__init__.py`'s exception handler has something to register against while the
        registry package hasn't landed yet -- never actually raised in that case, since
        `resolve_analysis` itself is `None` and unreachable (every route 404s via
        `get_analysis_class` first)."""

    class UnknownAnalysisError(AnalysisRegistryError):
        pass

    class AnalysisTypeMismatchError(AnalysisRegistryError):
        pass

    class InvalidAnalysisParametersError(AnalysisRegistryError):
        pass

RASTER_COMPRESSION_ALGORITHMS = [
    CompressionAlgorithm.deflate,
    CompressionAlgorithm.lzw,
    CompressionAlgorithm.lzma,
    CompressionAlgorithm.none,
    CompressionAlgorithm.zstd,
]

TABULAR_COMPRESSION_ALGORITHMS = [
    CompressionAlgorithm.gzip,
    CompressionAlgorithm.lz4,
    CompressionAlgorithm.none,
    CompressionAlgorithm.zstd,
]


def validate_compression(compression, format):
    """Validate that the compression algorithm matches the requested data format."""
    # TODO: Also validate compression level.
    if format == DataFormat.tif:
        if compression.compression_algorithm in RASTER_COMPRESSION_ALGORITHMS:
            return True
        allowed_algorithms = RASTER_COMPRESSION_ALGORITHMS
    else:
        if compression.compression_algorithm in TABULAR_COMPRESSION_ALGORITHMS:
            return True
        allowed_algorithms = TABULAR_COMPRESSION_ALGORITHMS

    algs_str = ", ".join([alg.value for alg in allowed_algorithms])
    msg = f"Compression algorithm for format {format.value} must be one of: {algs_str}"
    raise HTTPException(status_code=400, detail=msg)


def validate_format(format, output_type):
    """Validate that the specified data format matches the expected output type."""
    if output_type == OutputType.raster and format != DataFormat.tif:
        msg = "Format for raster output must be tif"
        raise HTTPException(status_code=400, detail=msg)
    elif output_type == OutputType.vector and format == DataFormat.tif:
        formats_str = ", ".join([f.value for f in DataFormat if f != DataFormat.tif])
        msg = f"Format for vector output must be one of: {formats_str}"
        raise HTTPException(status_code=400, detail=msg)


def get_visualization_class(name: str, type: str | None = None):
    """Get a visualization class by name, and optionally, by type."""

    VisualizationClass = VISUALIZATIONS.get(name, None)

    if VisualizationClass is None:
        msg = f"Unknown visualization name: {name}"
        raise HTTPException(status_code=404, detail=msg)

    if type is not None and VisualizationClass.Meta.type != type:
        msg = f"Invalid visualization type: expected {type}, got {VisualizationClass.Meta.type}"
        raise HTTPException(status_code=400, detail=msg)

    return VisualizationClass


def get_analysis_class(name: str, type: str | None = None):
    """Get a named-analysis class by name, and optionally, by type ("dataset" or
    "custom").

    The same 404 (unknown name) / 400 (wrong route kind) distinction
    `get_visualization_class` above already makes, generalized rather than reinvented:
    a `type="dataset"` analysis is rejected on `/v1/analyses/{name}/...`
    and a `type="custom"` analysis is rejected on the dataset-scoped route, both via this
    one function.
    """

    AnalysisClass = ANALYSES.get(name, None)

    if AnalysisClass is None:
        msg = f"Unknown analysis name: {name}"
        raise HTTPException(status_code=404, detail=msg)

    if type is not None and AnalysisClass.Meta.type != type:
        msg = f"Invalid analysis type: expected {type}, got {AnalysisClass.Meta.type}"
        raise HTTPException(status_code=400, detail=msg)

    return AnalysisClass
