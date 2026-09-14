from fastapi import HTTPException

from cpra.mp.data.analysis.types import CompressionAlgorithm, DataFormat, OutputType
from cpra.mp.data.visualizations import VISUALIZATIONS

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
