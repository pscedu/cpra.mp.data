"""Build vector-to-vector and raster-to-vector crosswalk files.

Pure-Python replacements for `scripts/generate_crosswalks.py` (which shells out to
`ogr2ogr -dialect sqlite` for `st_intersects(st_centroid(src.geometry), dst.geometry)`)
and `scripts/generate_raster_crosswalks.py` (which loads a whole destination grid and a
whole raster array into memory with geopandas). Both replacements exist because the old
scripts' peak memory scales with the largest grid involved, which doesn't fit this
environment's memory budget; everything below is written to scale with a batch or a
window instead.

**Vector crosswalks** (`write_vector_crosswalk`) index the SOURCE grid's point
centroids in an `shapely.STRtree` and stream the DESTINATION grid's polygons through it
in Arrow batches, discarding each batch once it's queried. This is the cheaper way
around: a destination polygon only has to exist in memory for the batch it's read in,
while the source structure has to live for the whole run, so the run-long structure
should be the smaller one (points, not polygons). Measured (see task discussion): indexing
destination polygons cost +275MB for `clara_grid_cell_v001` (112,318 features / 5.2M
coords), while indexing source centroids and streaming all four destinations of
`hydro_compartment_v001` cost +101MB total in 0.8s. `read_centroid_tree` is factored out
so a caller converting one source to many destinations (as `cli` does) builds the tree
once and reuses it.

Row order changes from the SQL's `order by dst.id, src.id` to source-stream order. This
is intentional and safe: `api/crosswalk.py` and `api/dataset.py` both re-sort explicitly
after reading a crosswalk table, so neither depends on the file's own row order.

**Raster crosswalks** (`write_raster_crosswalk`) rasterize a destination grid's polygons
onto a raster grid's pixel grid one window at a time, reading only the polygons that
intersect that window's bounding box (FlatGeobuf's packed R-tree makes this cheap: 693
total feature reads across all 697 windows of `morph_pixel_v001`, versus 44,608 if every
window loaded every parish). The output is written as a tiled GTiff and only turned into
a COG as a final `rasterio.shutil.copy`, because `rio.open(path, "w", driver="COG")` is
CreateCopy-only under the hood: GDAL backs it with an in-RAM MEM dataset that isn't freed
until close, which for `morph_pixel_v001` (8265 x 20674, up to int64) is a >1GB spike a
windowed GTiff write avoids.

A FlatGeobuf-specific GDAL bug (reproduced on GDAL 3.11.4 / pyogrio 0.12.1 and GDAL
3.12.4 / pyogrio 0.13.0) makes the windowed `bbox` reads dangerous without a guard: see
`_iter_features` below.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyogrio
import pyproj
import rasterio as rio
import shapely
from rasterio.features import rasterize
from rasterio.shutil import copy as copy_raster
from rasterio.transform import from_bounds
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform

from cpra.mp.data.grids import GRIDS

NODATA = -1
ZONE_DTYPES = ["int16", "int32", "int64"]

# rasterize() burns zone values through a double (GDAL's rasterizer works in floating
# point regardless of the output band's dtype), and a double only represents integers
# exactly up to 2**53. Past that, an ID would silently round to a neighboring value
# instead of failing loudly, so it's rejected before rasterizing rather than discovered
# later as a wrong zone.
MAX_SAFE_ZONE_ID = 2**53

# GDAL's block cache defaults to 5% of installed physical RAM (uncapped in recent GDAL
# versions), which on a machine with tens of GB of RAM swamps the whole point of writing
# windowed: measured on the real `morph_pixel_v001` -> `parish_v2020` crosswalk, leaving
# the default cache in place let `rasterio.shutil.copy`'s CreateCopy alone add ~530MB of
# RSS, while bounding it here reproduces the ~150MB increment measured on the reference
# 256MB int64 stand-in raster. This has to be a `rasterio.Env`, not a `GDAL_CACHEMAX`
# creation option -- it's a global GDAL config setting, not a file's own metadata.
GDAL_CACHE_BYTES = 64 * 1024 * 1024


def get_zone_dtype(dst_ids):
    """Get the narrowest integer dtype that holds every destination ID and the nodata
    fill. Grids identified by census FIPS codes (block groups, tracts) have 12-digit IDs
    that need int64; burning them into a narrower band saturates every pixel at the
    band's maximum value instead of failing."""

    min_id = min(NODATA, min(dst_ids))
    max_id = max(dst_ids)
    if max(abs(min_id), abs(max_id)) >= MAX_SAFE_ZONE_ID:
        msg = f"Destination IDs from {min_id} to {max_id} exceed what rasterize() burns exactly ({MAX_SAFE_ZONE_ID})"
        raise ValueError(msg)

    for dtype in ZONE_DTYPES:
        dtype_info = np.iinfo(dtype)
        if dtype_info.min <= min_id and max_id <= dtype_info.max:
            return dtype

    raise ValueError(f"Destination IDs from {min_id} to {max_id} do not fit in {ZONE_DTYPES[-1]}")


def _id_arrow_type(dtype_name):
    """Map an OGR/numpy integer dtype name to the Arrow type that holds it without
    truncation. Every grid's `id` field is either 32- or 64-bit (the FIPS-keyed grids are
    64-bit at the source; everything else is 32-bit), so this only ever widens to int64,
    matching the canonical fixture's schema (`tests/factories.py`)."""

    return pa.int64() if dtype_name == "int64" else pa.int32()


def _valid_geometries(ids, wkb):
    """Drop features whose WKB is missing or zero-length.

    This guards a FlatGeobuf-specific bug in GDAL's `GetArrowStream` (reproduced on GDAL
    3.11.4 / pyogrio 0.12.1 and GDAL 3.12.4 / pyogrio 0.13.0, so it isn't fixed by
    upgrading): a `bbox` read that matches zero features doesn't return an empty stream.
    Instead it returns exactly one batch of `batch_size` rows with empty-bytes WKB and
    `id=0`, then stops. The WKB column's `null_count` is 0 for that batch, so a null
    check doesn't catch it -- only a zero-length-bytes check does. GPKG and Shapefile are
    unaffected; every grid here is FlatGeobuf.

    This is silent corruption, not a crash: `id=0` looks like a legitimate zone. Verified
    on the real `morph_pixel_v001` -> `parish_v2020` raster crosswalk that 306 of 697
    windows have zero intersecting features, and 276 of those return the bogus batch --
    unguarded, roughly 40% of the raster would be burned with zone 0 instead of NODATA.
    """

    lengths = np.fromiter((len(v) if v is not None else 0 for v in wkb), dtype=np.int64, count=len(wkb))
    valid = lengths > 0
    return ids[valid], wkb[valid]


def _iter_features(path, batch_size=65536, bbox=None):
    """Stream (ids, wkb) batches of a vector grid's `id` field and geometry, filtering
    out the bogus empty-bbox batch described in `_valid_geometries`. A batch left with no
    valid features after filtering is skipped entirely rather than yielded empty."""

    with pyogrio.open_arrow(
        str(path), columns=["id"], use_pyarrow=True, batch_size=batch_size, bbox=bbox
    ) as (_meta, reader):
        for batch in reader:
            wkb = batch.column("wkb_geometry").to_numpy(zero_copy_only=False)
            ids = batch.column("id").to_numpy(zero_copy_only=False)
            ids, wkb = _valid_geometries(ids, wkb)
            if len(ids) == 0:
                continue
            yield ids, wkb


def read_centroid_tree(fgb_path, batch_size=65536):
    """Build an `STRtree` of a vector grid's centroids plus the parallel array of IDs.

    Reads geometry only long enough to compute each batch's centroids, then discards the
    polygons -- the tree and the ID array are all that live for the rest of the run. This
    is the expensive step in a vector crosswalk (see module docstring), so callers
    converting one source to several destinations should build it once with this
    function and pass the result to `write_vector_crosswalk`'s `tree` argument.
    """

    id_parts = []
    centroid_parts = []
    for ids, wkb in _iter_features(fgb_path, batch_size=batch_size):
        geoms = shapely.from_wkb(wkb)
        centroid_parts.append(shapely.centroid(geoms))
        id_parts.append(ids)

    ids = np.concatenate(id_parts) if id_parts else np.array([], dtype=np.int32)
    centroids = np.concatenate(centroid_parts) if centroid_parts else np.array([], dtype=object)
    return shapely.STRtree(centroids), ids


def write_vector_crosswalk(
    src_grid_name,
    dst_grid_name,
    grids_path,
    crosswalks_path,
    batch_size=65536,
    compression="zstd",
    tree=None,
):
    """Write a `src_grid_name` -> `dst_grid_name` crosswalk table, equivalent to the old
    SQL's `st_intersects(st_centroid(src.geometry), dst.geometry)`.

    `tree`, if given, must be a `(STRtree, ids)` pair as returned by `read_centroid_tree`
    for `src_grid_name` -- passing one in lets several destinations of the same source
    reuse it instead of rebuilding it each time. Written incrementally with
    `pyarrow.parquet.ParquetWriter` so the full join is never materialized at once.

    Returns `(output_path, row_count)`.
    """

    grids_path = Path(grids_path)
    crosswalks_path = Path(crosswalks_path)
    dst_path = grids_path / f"{dst_grid_name}.fgb"

    if tree is None:
        src_path = grids_path / f"{src_grid_name}.fgb"
        tree, src_ids = read_centroid_tree(src_path, batch_size=batch_size)
    else:
        tree, src_ids = tree

    src_id_type = _id_arrow_type(str(src_ids.dtype))

    crosswalks_path.mkdir(parents=True, exist_ok=True)
    out_path = crosswalks_path / f"{src_grid_name}__{dst_grid_name}.parquet"

    row_count = 0
    writer = None
    try:
        with pyogrio.open_arrow(
            str(dst_path), columns=["id"], use_pyarrow=True, batch_size=batch_size
        ) as (meta, reader):
            dst_id_type = _id_arrow_type(meta["dtypes"][0])
            schema = pa.schema(
                [
                    ("src_id", src_id_type),
                    ("dst_id", dst_id_type),
                    ("scale_factor", pa.float64()),
                ]
            )
            writer = pq.ParquetWriter(out_path, schema, compression=compression)

            for batch in reader:
                wkb = batch.column("wkb_geometry").to_numpy(zero_copy_only=False)
                dst_ids = batch.column("id").to_numpy(zero_copy_only=False)
                dst_ids, wkb = _valid_geometries(dst_ids, wkb)
                if len(dst_ids) == 0:
                    continue

                dst_geoms = shapely.from_wkb(wkb)
                # `tree` holds SOURCE centroids and `dst_geoms` is the query side, so
                # `query` returns (dst_index, src_index) -- backwards from how it reads at
                # a glance. Verified empirically; see module docstring's memory rationale
                # for why the tree is built on the source rather than the destination.
                dst_idx, src_idx = tree.query(dst_geoms, predicate="intersects")
                if len(dst_idx) == 0:
                    continue

                out_batch = pa.record_batch(
                    [
                        pa.array(src_ids[src_idx], type=src_id_type),
                        pa.array(dst_ids[dst_idx], type=dst_id_type),
                        pa.array(np.ones(len(dst_idx), dtype=np.float64)),
                    ],
                    schema=schema,
                )
                writer.write_batch(out_batch)
                row_count += len(dst_idx)
    finally:
        if writer is not None:
            writer.close()

    return out_path, row_count


def _block_windows(height, width, block_size):
    """Tile a raster's full extent into `block_size` x `block_size` windows, clipped at
    the bottom and right edges when the dimensions aren't an exact multiple of it."""

    for row_off in range(0, height, block_size):
        win_height = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            win_width = min(block_size, width - col_off)
            yield Window(col_off, row_off, win_width, win_height)


def _reprojected_bbox(bbox, transformer):
    """Transform a (minx, miny, maxx, maxy) bbox through `transformer`, taking the
    min/max of all four corners (not just two) so a reprojection that isn't axis-aligned
    still yields a box that encloses the source rectangle."""

    xs, ys = transformer.transform(
        [bbox[0], bbox[0], bbox[2], bbox[2]],
        [bbox[1], bbox[3], bbox[1], bbox[3]],
    )
    return min(xs), min(ys), max(xs), max(ys)


def write_raster_crosswalk(raster_grid_name, dst_grid_name, grids_path, crosswalks_path, block_size=512):
    """Rasterize a destination vector grid's polygons onto a raster grid's pixel grid,
    producing the zone-ID crosswalk COG the data API's `summarize_by` and
    `api/crosswalk.py` read.

    Processes one `block_size` x `block_size` window at a time: only the destination
    polygons intersecting that window's bounding box are read (cheap, thanks to
    FlatGeobuf's packed R-tree), rasterized into a NODATA-filled window array, and written
    to a tiled GTiff opened once up front. The GTiff is then turned into the final COG
    with `rasterio.shutil.copy` rather than written directly as one, because `driver="COG"`
    is CreateCopy-only in GDAL and would otherwise buffer the whole raster in an in-RAM MEM
    dataset until close (see module docstring for the measured cost). The temporary GTiff
    is created inside `crosswalks_path` via `tempfile` so the final copy is a same-
    filesystem operation, and it's removed in a `finally` regardless of outcome.

    Returns `(output_path, window_count)`.
    """

    grids_path = Path(grids_path)
    crosswalks_path = Path(crosswalks_path)
    dst_path = grids_path / f"{dst_grid_name}.fgb"
    raster_grid = GRIDS[raster_grid_name]

    # Cheap: reads only the `id` field, no geometry, to get every destination ID and
    # decide the zone dtype before any rasterizing starts.
    _meta, dst_table = pyogrio.read_arrow(str(dst_path), columns=["id"], read_geometry=False)
    dst_ids = sorted({int(v) for v in dst_table.column("id").to_numpy(zero_copy_only=False)})
    dst_id_set = set(dst_ids)
    dtype = get_zone_dtype(dst_ids)
    dst_ids_tag = ",".join(str(dst_id) for dst_id in dst_ids)

    transform = from_bounds(*raster_grid.bounds, width=raster_grid.width, height=raster_grid.height)

    # Preserve the old script's CRS guard: reproject the destination polygons into the
    # raster's CRS if they differ. With per-window bbox reads, the window bounds have to
    # go the other way first -- into the destination's CRS -- to select the right
    # features, and the geometries read back have to be reprojected into the raster's CRS
    # before rasterizing. Every grid in `structure/grids.csv` is `epsg:26915` today, so
    # this is a no-op in practice, but it has to stay correct for when that changes.
    dst_info = pyogrio.read_info(str(dst_path))
    dst_crs = dst_info["crs"]
    reproject = dst_crs is not None and dst_crs.lower() != raster_grid.crs.lower()
    to_raster_crs = None
    to_dst_crs = None
    if reproject:
        to_dst_crs = pyproj.Transformer.from_crs(raster_grid.crs, dst_crs, always_xy=True)
        to_raster_crs = pyproj.Transformer.from_crs(dst_crs, raster_grid.crs, always_xy=True)

    crosswalks_path.mkdir(parents=True, exist_ok=True)
    out_path = crosswalks_path / f"{raster_grid_name}__{dst_grid_name}.tif"

    profile = {
        "count": 1,
        "crs": raster_grid.crs,
        "driver": "GTiff",
        "dtype": dtype,
        "height": raster_grid.height,
        "width": raster_grid.width,
        "nodata": NODATA,
        "compress": "ZSTD",
        # The intermediate GTiff needs GDAL's numeric PREDICTOR codes (2 = horizontal
        # differencing, correct for the integer zone dtypes this always writes); the
        # "YES"/"NO" spelling used below is COG-driver-only shorthand.
        "predictor": 2,
        "tiled": True,
        "blockxsize": block_size,
        "blockysize": block_size,
        "transform": transform,
    }

    window_count = 0
    tmp_dir = Path(tempfile.mkdtemp(dir=crosswalks_path))
    tmp_path = tmp_dir / f"{raster_grid_name}__{dst_grid_name}.tmp.tif"
    try:
        # Bound GDAL's block cache for the write and the COG copy below (see
        # `GDAL_CACHE_BYTES`) -- left at its default, it's sized off total system RAM
        # rather than this raster, and swamps the memory this windowed approach saves.
        with rio.Env(GDAL_CACHEMAX=GDAL_CACHE_BYTES):
            with rio.open(tmp_path, "w", **profile) as dst:
                for window in _block_windows(raster_grid.height, raster_grid.width, block_size):
                    window_count += 1
                    bbox = window_bounds(window, transform)
                    read_bbox = _reprojected_bbox(bbox, to_dst_crs) if reproject else bbox

                    shapes = []
                    for ids, wkb in _iter_features(dst_path, bbox=read_bbox):
                        geoms = shapely.from_wkb(wkb)
                        if reproject:
                            geoms = shapely.transform(
                                geoms,
                                lambda coords: np.column_stack(to_raster_crs.transform(coords[:, 0], coords[:, 1])),
                            )
                        shapes.extend(zip(geoms.tolist(), ids.tolist()))

                    zones = np.full((window.height, window.width), NODATA, dtype=dtype)
                    if shapes:
                        rasterize(shapes, out=zones, transform=window_transform(window, transform))

                        # Per-window equivalent of the whole-array saturation check the
                        # old script made: rasterize() burns values through a double and
                        # clamps anything out of the output dtype's range to its max, so
                        # confirm the largest value actually burned in this window is a
                        # real ID (or NODATA) instead of a clamped one.
                        max_zone = int(zones.max())
                        if max_zone != NODATA and max_zone not in dst_id_set:
                            raise ValueError(f"Zone ID {max_zone} was clamped to the range of {dtype}")

                    dst.write(zones, 1, window=window)

                dst.update_tags(dst_ids=dst_ids_tag)

            copy_raster(
                str(tmp_path),
                str(out_path),
                driver="COG",
                compress="ZSTD",
                predictor="YES",
                resampling="NEAREST",
                overviews="YES",
            )

        # `dst_ids` is read by `api/dataset.py` (`crosswalk.tags()["dst_ids"]`), which
        # KeyErrors without it. It is set on the intermediate GTiff above and carried over
        # by CreateCopy -- verified on this build (GDAL bundled with pyogrio 0.12.1 /
        # rasterio 1.5.0) -- but GDAL doesn't document that as guaranteed across drivers
        # or versions, so confirm it rather than trusting it silently.
        #
        # There is deliberately no repair-in-place fallback here: a finished COG can't be
        # reopened "r+" to add the tag, because GDAL refuses a plain update-mode open of a
        # COG-layout GeoTIFF (CPLE_AppDefinedError, directing callers to
        # IGNORE_COG_LAYOUT_BREAK=YES -- which would break the COG layout that makes this
        # file useful). Failing loudly beats writing a crosswalk the API can only 500 on.
        with rio.open(out_path) as written:
            if written.tags().get("dst_ids") != dst_ids_tag:
                raise ValueError(
                    f"CreateCopy dropped the dst_ids tag from {out_path.name}; "
                    "api/dataset.py cannot summarize by this crosswalk without it"
                )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_path, window_count
