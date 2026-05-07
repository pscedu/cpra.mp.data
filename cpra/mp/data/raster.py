import xml.etree.ElementTree
from pathlib import Path

import polars as pl
import rasterio as rio
from rasterio.io import MemoryFile
from rasterio.shutil import copy
from rasterio.transform import from_bounds

from cpra.mp.data.config import BASE_PATH
from cpra.mp.data.dimensions import DATASET_DIMENSIONS, DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.utils import (
    cast_columns,
    get_filter_expressions,
    get_input_data_frame,
    glob_lazy_frame,
    required_columns,
    validate_data_frame,
)


def cog_path_expression(df, base_path=BASE_PATH, separator="/"):
    """Generate a COG path from a data frame containing dimensions and values."""

    return pl.concat_str(
        [
            pl.concat_str(
                [pl.lit(str(base_path))]
                + [
                    pl.concat_str(
                        [pl.lit(dim.name), pl.col(dim.name).cast(pl.String)],
                        separator="=",
                    )
                    for dim in DIMENSIONS.values()
                    if dim.name in df.columns and dim.type != "metadata"
                ],
                separator=separator,
            ),
            pl.lit(".tif"),
        ]
    )


def vrt(paths):
    """Create a VRT dataset from the given COG paths."""

    # Create individual band VRTs, and combine them.
    # Adapted from: https://gis.stackexchange.com/a/421687
    out_el = None
    for path_idx, path in enumerate(paths, start=1):
        with rio.open(path) as raster, MemoryFile() as memory:
            copy(raster, memory.name, driver="VRT")
            dataset_el = xml.etree.ElementTree.fromstring(memory.read().decode("utf-8"))

            if out_el is None:
                out_el = dataset_el
            else:
                # TODO: Verify that the dimensions etc. of this band match the output raster.
                band_el = dataset_el.find("VRTRasterBand")
                # TODO: Raise an error if there isn't a band?
                if band_el:
                    band_el.set("band", str(path_idx))
                    out_el.append(band_el)

    return xml.etree.ElementTree.tostring(out_el).decode("UTF-8")


def delete_cog(data=None, base_path=BASE_PATH, **kwargs):
    """Delete matching COG paths."""
    input_df = cast_columns(get_input_data_frame(data, **kwargs))

    # Make sure that all dimensions are defined.
    validate_data_frame(input_df)

    # Get the COG paths.
    df = read_cog(input_df, base_path=base_path).select("path").collect()
    for cog_path in df["path"]:
        Path(cog_path).unlink()


def read_cog(data=None, base_path=BASE_PATH, include_file_paths="path", **kwargs):
    """Locate matching COG paths."""

    # Validate the input keyword arguments.
    input_df = get_input_data_frame(data, **kwargs)
    validate_data_frame(input_df, required_cols=DATASET_DIMENSIONS.keys())

    # Construct the glob lazy frame.
    required_cols = required_columns(input_df, include_metadata=False)
    lf = glob_lazy_frame(required_cols=required_cols, extension="tif", include_file_paths=include_file_paths, **kwargs)
    filter_exps = get_filter_expressions(**kwargs)
    if filter_exps:
        lf = lf.filter(*filter_exps)
    return lf


def write_cog(
    data,
    base_path=BASE_PATH,
    validate=True,
    compression="zstd",
    compression_level=6,
    **kwargs,
):
    """Write data from a NumPy array to a COG."""
    df = cast_columns(pl.DataFrame(kwargs))
    if validate:
        validate_data_frame(df)

    # Compute the path for the COG.
    df = df.with_columns(cog_path=cog_path_expression(df, base_path))
    cog_path = Path(df.select("cog_path").item())

    # Create the parent directories if needed.
    cog_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a transform based on the raster grid.
    grid = GRIDS[kwargs["grid"]]
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
        "resampling": "NEAREST",
        "transform": dst_transform,
        "width": grid.width,
    }

    with rio.open(cog_path, "w", **dst_profile) as dst:
        dst.update_tags(**kwargs)
        dst.write(data.reshape((grid.height, grid.width)), 1)
