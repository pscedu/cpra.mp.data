import tempfile
import uuid
import xml.etree.ElementTree
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import polars as pl
import rasterio as rio
from rasterio.io import MemoryFile
from rasterio.shutil import copy
from rasterio.transform import from_bounds

from cpra.mp.data.config import BASE_PATH
from cpra.mp.data.dimensions import DATASET_DIMENSIONS, DIMENSIONS, METADATA_DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.utils import (
    build_glob_pattern,
    calendar_run_offsets,
    cast_columns,
    dataset_path,
    get_filter_expressions,
    get_input_data_frame,
    glob_lazy_frame,
    required_columns,
    validate_data_frame,
)

_CALENDAR_DIMENSIONS = ("calendar_year", "calendar_day")


def cog_path_expression(df, base_path=BASE_PATH, separator="/"):
    """Generate a COG dataset+partition directory path from a data frame.

    Excludes the calendar dimension (its value lives only in the band description, never
    the path -- see ``write_cog``) and metadata dimensions.  The file itself is named
    with a UUID by the caller, like partitioned Parquet files, not derived from this path.
    """

    return pl.concat_str(
        [pl.lit(str(base_path))]
        + [
            pl.concat_str(
                [pl.lit(dim.name), pl.col(dim.name).cast(pl.String)],
                separator="=",
            )
            for dim in DIMENSIONS.values()
            if dim.name in df.columns and dim.type != "metadata" and dim.name not in _CALENDAR_DIMENSIONS
        ],
        separator=separator,
    )


def vrt(sources):
    """Create a band-separate VRT dataset from the given sources.

    Each source is either a bare path (its band 1 is used) or a ``(path, band)`` pair
    selecting a specific 1-based band -- used by ``compact_cog`` to consolidate
    contiguous runs that span already multi-band COGs.  Output band order matches
    ``sources`` order; each output band inherits its source band's own description.
    """

    # Adapted from: https://gis.stackexchange.com/a/421687
    out_el = None
    for out_idx, source in enumerate(sources, start=1):
        path, band = source if isinstance(source, tuple) else (source, 1)
        with rio.open(path) as raster, MemoryFile() as memory:
            copy(raster, memory.name, driver="VRT")
            dataset_el = xml.etree.ElementTree.fromstring(memory.read().decode("utf-8"))
            band_el = dataset_el.find(f"VRTRasterBand[@band='{band}']")
            band_el.set("band", str(out_idx))

            if out_el is None:
                out_el = dataset_el
                # Keep only the selected band; drop any other bands this source has.
                for extra in list(out_el.findall("VRTRasterBand")):
                    if extra is not band_el:
                        out_el.remove(extra)
            else:
                out_el.append(band_el)

    return xml.etree.ElementTree.tostring(out_el).decode("UTF-8")


def _read_cog_metadata(path):
    """Open a COG and read its band descriptions and dtype, without reading pixel data."""

    with rio.open(path) as ds:
        return ds.descriptions, ds.dtypes[0]


def _parallel_read_cog_metadata(paths):
    """Read ``(descriptions, dtype)`` for several COGs in parallel threads.

    The calendar value a file/band covers can only be determined by opening it (the
    filename is an opaque UUID), so both ``read_cog`` and ``compact_cog`` must open every
    candidate file; doing so in parallel keeps that from being latency-bound.
    """

    with ThreadPoolExecutor() as pool:
        return list(pool.map(_read_cog_metadata, paths))


def _band_calendar_frame(paths, path_column, calendar_col):
    """Expand each COG into one row per band with that band's calendar value.

    Every COG -- single-band (``write_cog``) or multi-band (a contiguous run merged by
    ``compact_cog``) -- stores each band's calendar value in its band description
    (``"<calendar_dim>=<value>"``), which is the sole source read here.
    """

    dtype = DIMENSIONS[calendar_col].dtype
    rows = {path_column: [], "band": [], "_value": []}
    for path, (descriptions, _dtype) in zip(paths, _parallel_read_cog_metadata(paths)):
        for band_idx, desc in enumerate(descriptions, start=1):
            _, _, value = desc.partition("=")
            rows[path_column].append(path)
            rows["band"].append(band_idx)
            rows["_value"].append(value)

    frame = pl.DataFrame(rows, schema={path_column: pl.String, "band": pl.Int64, "_value": pl.String})
    value = pl.col("_value").str.to_date() if dtype == pl.Date else pl.col("_value").cast(dtype)
    return frame.with_columns(value.alias(calendar_col)).drop("_value")


def read_cog(base_path=BASE_PATH, include_file_paths="path", **kwargs):
    """Locate matching COGs and expand them into per-band rows.

    Files are named with a UUID, not by dimension value, so the calendar dimension isn't
    part of the path at all: every file within a matching partition directory is opened
    (in parallel) to read its band descriptions, the sole source of each band's calendar
    value.  ``compact_cog`` merges contiguous calendar runs into larger multi-band COGs
    wherever possible, but a run that can't be extended (a gap on both sides) may remain
    a single-band file.  Every returned row is one band of one file: the on-path
    dimensions, the calendar value, the file ``path``, and the 1-based ``band`` index.
    """

    # Validate the input keyword arguments.
    input_df = get_input_data_frame(**kwargs)
    validate_data_frame(input_df, required_cols=DATASET_DIMENSIONS.keys())

    required_cols = required_columns(input_df, include_metadata=False)
    calendar_col = next((col for col in required_cols if col in _CALENDAR_DIMENSIONS), None)
    path_cols = [col for col in required_cols if col != calendar_col]

    path_column = include_file_paths or "path"
    path_frame = glob_lazy_frame(
        required_cols=path_cols,
        base_path=Path(base_path),
        extension="tif",
        nested=True,
        include_file_paths=path_column,
        **kwargs,
    ).collect()

    if calendar_col is None:
        lf = path_frame.with_columns(band=pl.lit(1, dtype=pl.Int64)).lazy()
    else:
        band_frame = _band_calendar_frame(path_frame[path_column].to_list(), path_column, calendar_col)
        lf = path_frame.join(band_frame, on=path_column, how="inner").lazy()

    # Metadata dimensions (e.g. grid) are in-file columns for tabular data but are not
    # path components in COG files — exclude them from raster filters.
    raster_kwargs = {k: v for k, v in kwargs.items() if k not in METADATA_DIMENSIONS}
    filter_exps = get_filter_expressions(**raster_kwargs)
    if filter_exps:
        lf = lf.filter(*filter_exps)

    sort_cols = [*path_cols, *([calendar_col] if calendar_col else [])]
    if sort_cols:
        lf = lf.sort(*sort_cols)
    if not include_file_paths:
        lf = lf.drop(path_column)
    return lf


def _cog_profile(grid, dtype, count=1, compression="zstd", compression_level=6):
    """Build the rasterio COG write profile for a grid (used by ``write_cog``).

    ``compact_cog`` reuses this for the shared compression/predictor/nodata options on
    its ``CreateCopy`` call.
    """

    return {
        "compress": compression.upper(),
        "level": compression_level,
        "count": count,
        "crs": grid.crs,
        "driver": "COG",
        "dtype": dtype,
        "height": grid.height,
        "nodata": -9999,
        "predictor": "YES",
        "resampling": "NEAREST",
        "transform": from_bounds(*grid.bounds, width=grid.width, height=grid.height),
        "width": grid.width,
    }


def write_cog(
    data,
    base_path=BASE_PATH,
    validate=True,
    compression="zstd",
    compression_level=6,
    **kwargs,
):
    """Write data from a NumPy array to a COG.

    Like partitioned Parquet files, the file itself is named with a UUID rather than by
    dimension value.  The calendar dimension (year or date) is stored only in the band
    description, never as a metadata tag.
    """
    df = cast_columns(pl.DataFrame(kwargs))
    if validate:
        validate_data_frame(df)

    # Compute the dataset+partition directory; the calendar dimension is never a path
    # component (see cog_path_expression), so the file is named with a UUID instead.
    df = df.with_columns(cog_dir=cog_path_expression(df, base_path))
    cog_dir = Path(df.select("cog_dir").item())
    cog_dir.mkdir(parents=True, exist_ok=True)
    cog_path = cog_dir / f"{uuid.uuid4()}.tif"

    calendar_dim = next(dim for dim in _CALENDAR_DIMENSIONS if dim in kwargs)
    description = df.select(
        pl.concat_str([pl.lit(calendar_dim), pl.col(calendar_dim).cast(pl.String)], separator="=")
    ).item()

    grid = GRIDS[kwargs["grid"]]
    profile = _cog_profile(grid, data.dtype, compression=compression, compression_level=compression_level)

    with rio.open(cog_path, "w", **profile) as dst:
        # The calendar value lives only in the band description below, not as a tag.
        dst.update_tags(**{k: v for k, v in kwargs.items() if k not in _CALENDAR_DIMENSIONS})
        dst.set_band_description(1, description)
        dst.write(data.reshape((grid.height, grid.width)), 1)


def _calendar_value(value, calendar_dim):
    """Parse a band description's value (the part after ``"="``) into a typed value."""

    return date.fromisoformat(value) if calendar_dim == "calendar_day" else int(value)


def compact_cog(base_path=BASE_PATH, compression="zstd", compression_level=6, **kwargs):
    """Merge maximal contiguous calendar runs of COGs into larger multi-band COGs.

    Every matching COG within a partition directory -- single-band or already
    multi-band, in any combination -- is opened (in parallel) to read its band
    descriptions, and every band across every file in that directory is pooled into one
    sorted sequence of calendar values.  Wherever that sequence forms a maximal
    contiguous run (consecutive years, decades, months, or days -- see
    ``utils.calendar_run_offsets``, shared with the API catalog's range compression)
    spanning more than one source file, those files are consolidated into a single new
    multi-band COG (named with a UUID, like the sources); a run already living entirely
    in one file is left untouched.  A gap ends a run, so incomplete/sparse time series
    stay physically distinct from complete ones and are safe to compact again later once
    the gap is filled.

    A band-separate VRT is built over each run's ``(path, band)`` sources (``vrt()``) and
    streamed into the output with ``CreateCopy`` (like ``gdalbuildvrt -separate`` +
    ``gdal_translate -of COG``), reusing ``_cog_profile()`` for the shared compression
    settings; band descriptions and dataset tags are inherited automatically through the
    VRT and the copy.  GDAL never materializes the stack and, because the sources already
    carry overviews, reuses them instead of regenerating — markedly faster and lower peak
    memory than copying through a temporary GeoTIFF.
    """

    input_df = cast_columns(pl.DataFrame(kwargs))
    validate_data_frame(input_df, required_cols=DATASET_DIMENSIONS.keys())

    dataset_dir = dataset_path(base_path, **kwargs)
    grid = GRIDS[kwargs["grid"]]

    required_cols = required_columns(input_df, include_dataset=False, include_metadata=False)
    calendar_dim = next((col for col in required_cols if col in _CALENDAR_DIMENSIONS), None)
    if calendar_dim is None:
        return
    path_cols = [col for col in required_cols if col != calendar_dim]
    glob_pattern = build_glob_pattern(required_cols=path_cols, extension="tif", nested=True, **kwargs)
    time_unit = kwargs["time_unit"]

    # Group matched files by directory (each directory holds every file for one
    # combination of non-calendar dimensions).
    groups = {}
    for path in dataset_dir.glob(glob_pattern):
        groups.setdefault(path.parent, []).append(path)

    for directory, tifs in groups.items():
        # Pool every band of every file (regardless of existing band count) into one
        # sorted sequence of (value, path, band) triples.
        triples = []
        dtype = None
        for tif, (descriptions, tif_dtype) in zip(tifs, _parallel_read_cog_metadata(tifs)):
            dtype = tif_dtype
            for band_idx, desc in enumerate(descriptions, start=1):
                _, _, value = desc.partition("=")
                triples.append((_calendar_value(value, calendar_dim), tif, band_idx))

        triples.sort(key=lambda triple: triple[0])
        values = pl.Series([value for value, _tif, _band in triples])
        offsets = calendar_run_offsets(values, calendar_dim, time_unit).to_list()

        runs = {}
        for (_value, tif, band_idx), offset in zip(triples, offsets):
            runs.setdefault(offset, []).append((tif, band_idx))

        for run in runs.values():
            # A run already living entirely in one file (regardless of its band count)
            # is already optimal; nothing to gain from rewriting it.
            unique_paths = {tif for tif, _band in run}
            if len(unique_paths) < 2:
                continue

            out_path = directory / f"{uuid.uuid4()}.tif"

            # The VRT is a throwaway XML sidecar (its SimpleSource entries hold absolute
            # paths, so its own location doesn't matter) -- keep it out of the data
            # directory entirely.
            with tempfile.TemporaryDirectory() as tmp_dir:
                vrt_path = Path(tmp_dir) / f"{out_path.stem}.vrt"
                vrt_path.write_text(vrt(run))
                profile = _cog_profile(
                    grid, dtype, count=len(run), compression=compression, compression_level=compression_level
                )
                profile.update(interleave="PIXEL", num_threads="ALL_CPUS", bigtiff="YES")
                # out_path is a fresh UUID that can't collide with any existing source
                # file, so the copy can be written directly -- no temp-file/rename needed.
                copy(str(vrt_path), str(out_path), **profile)

            for path in unique_paths:
                path.unlink()
