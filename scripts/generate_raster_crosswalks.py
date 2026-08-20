import sys

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from cpra.mp.data.config import CROSSWALKS_PATH, GRIDS_PATH
from cpra.mp.data.grids import GRIDS

NODATA = -1
ZONE_DTYPES = ["int16", "int32", "int64"]


def get_zone_dtype(dst_ids):
    """Get the narrowest integer dtype that holds every destination ID and the nodata
    fill. Grids identified by census FIPS codes (block groups, tracts) have 12-digit IDs
    that need int64; burning them into a narrower band saturates every pixel at the
    band's maximum value instead of failing."""

    min_id = min(NODATA, min(dst_ids))
    max_id = max(dst_ids)
    for dtype in ZONE_DTYPES:
        dtype_info = np.iinfo(dtype)
        if dtype_info.min <= min_id and max_id <= dtype_info.max:
            return dtype

    raise ValueError(f"Destination IDs from {min_id} to {max_id} do not fit in {ZONE_DTYPES[-1]}")


def generate_raster_crosswalk(raster_grid_name, dst_grid_name):
    """Rasterize a destination vector grid's polygons onto a raster grid's pixel grid,
    producing a zone-ID crosswalk COG used by the data API's raster summarize_by."""

    raster_grid = GRIDS[raster_grid_name]
    gdf = gpd.read_file(GRIDS_PATH / f"{dst_grid_name}.fgb")
    if gdf.crs is not None and gdf.crs.to_string().lower() != raster_grid.crs.lower():
        gdf = gdf.to_crs(raster_grid.crs)

    dst_transform = from_bounds(*raster_grid.bounds, width=raster_grid.width, height=raster_grid.height)
    dst_ids = sorted(gdf["id"].unique().tolist())
    dtype = get_zone_dtype(dst_ids)

    zones = rasterize(
        zip(gdf.geometry, gdf["id"]),
        out_shape=(raster_grid.height, raster_grid.width),
        transform=dst_transform,
        fill=NODATA,
        dtype=dtype,
    )

    # rasterize() burns values through a double and clamps them to the band's range, so
    # check that the largest burned value is a real ID rather than a saturated one.
    max_zone = zones.max()
    if max_zone not in {NODATA, *dst_ids}:
        raise ValueError(f"Zone ID {max_zone} was clamped to the range of {dtype}")

    CROSSWALKS_PATH.mkdir(parents=True, exist_ok=True)
    crosswalk_path = CROSSWALKS_PATH / f"{raster_grid_name}__{dst_grid_name}.tif"

    profile = {
        "compress": "ZSTD",
        "count": 1,
        "crs": raster_grid.crs,
        "driver": "COG",
        "dtype": dtype,
        "height": raster_grid.height,
        "nodata": NODATA,
        "predictor": "YES",
        "resampling": "NEAREST",
        "transform": dst_transform,
        "width": raster_grid.width,
    }

    print(f"Generating raster crosswalk from {raster_grid_name} to {dst_grid_name}")
    with rio.open(crosswalk_path, "w", **profile) as dst:
        dst.write(zones, 1)
        dst.update_tags(dst_ids=",".join(str(dst_id) for dst_id in dst_ids))


def generate_raster_crosswalks(raster_grid_name):
    for fgb_path in GRIDS_PATH.glob("*.fgb"):
        generate_raster_crosswalk(raster_grid_name, fgb_path.name.split(".")[0])


if __name__ == "__main__":
    generate_raster_crosswalks(sys.argv[1])
