import polars as pl
import rasterio as rio
from cpra.mp.data.config import BASE_PATH
from cpra.mp.data.schema import SCHEMA
from cpra.mp.data.utils import cast_columns, validate_data_frame
from pathlib import Path
from rasterio.transform import from_bounds


RASTER_GRIDS = {
    "icm_raster": {
        "bounds": (404500, 3187060, 1024720, 3435010),
        "crs": "epsg:26915",
        "height": 8285,
        "width": 20674,
    }
}


def cog_path_expression(df, base_path="", separator="/"):
    """Generate a COG path from a data frame containing dimensions and values."""

    return pl.concat_str(
        [
            pl.concat_str(
                [pl.lit(base_path)]
                + [
                    pl.concat_str(
                        [pl.lit(level.column), pl.col(level.column).cast(pl.String)],
                        separator="=",
                    )
                    for level in SCHEMA
                    if level.column in df.columns
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
    grid = RASTER_GRIDS[kwargs["geographic_unit"]]
    height = grid["height"]
    width = grid["width"]
    dst_transform = from_bounds(*grid["bounds"], width=width, height=height)

    dst_profile = {
        "compress": compression.upper(),
        "compression_level": compression_level,
        "count": 1,
        "crs": grid["crs"],
        "driver": "COG",
        "dtype": data.dtype,
        "height": height,
        "nodata": -9999,
        "transform": dst_transform,
        "width": width,
    }

    with rio.open(cog_path, "w", **dst_profile) as dst:
        dst.write(data.reshape((height, width)), 1)
