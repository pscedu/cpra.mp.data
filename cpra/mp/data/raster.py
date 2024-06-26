import polars as pl
import rasterio as rio
from cpra.mp.data.config import BASE_PATH
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import get_grid
from cpra.mp.data.utils import cast_columns, validate_data_frame
from pathlib import Path
from rasterio.transform import from_bounds


def cog_path_expression(df, base_path="", separator="/"):
    """Generate a COG path from a data frame containing dimensions and values."""

    return pl.concat_str(
        [
            pl.concat_str(
                [pl.lit(base_path)]
                + [
                    pl.concat_str(
                        [pl.lit(dim.column), pl.col(dim.column).cast(pl.String)],
                        separator="=",
                    )
                    for dim in DIMENSIONS
                    if dim.column in df.columns
                ],
                separator=separator,
            ),
            pl.lit(".tif"),
        ]
    )


def write_cog(
    data,
    base_path=BASE_PATH,
    validate=True,
    compression="zstd",
    compression_level=6,
    **kwargs,
):
    df = cast_columns(pl.DataFrame(kwargs))
    if validate:
        validate_data_frame(df)

    # Compute the path for the COG.
    df = df.with_columns(cog_path=cog_path_expression(df, base_path))
    cog_path = Path(df.select("cog_path").item())

    # Create the parent directories if needed.
    cog_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a transform based on the raster grid.
    grid = get_grid(kwargs["geographic_unit"], kwargs["plan_version"])
    dst_transform = from_bounds(*grid.bounds, width=grid.width, height=grid.height)

    dst_profile = {
        "compress": compression.upper(),
        "compression_level": compression_level,
        "count": 1,
        "crs": grid.crs,
        "driver": "COG",
        "dtype": data.dtype,
        "height": grid.height,
        "nodata": -9999,
        "transform": dst_transform,
        "width": grid.width,
    }

    with rio.open(cog_path, "w", **dst_profile) as dst:
        dst.write(data.reshape((grid.height, grid.width)), 1)
