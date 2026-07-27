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

from cpra.mp.data.catalog import CAL_DIMS, Catalog, cal_for, cast_str, parse_path_dims, raster_catalog_rows
from cpra.mp.data.config import BASE_PATH, CATALOG_PATH
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

_CALENDAR_DIMENSIONS = CAL_DIMS  # ("calendar_year", "calendar_day") -- shared with catalog.py

# LEGACY COG SHIM (grep this tag to find every removable piece; delete once no on-disk
# COG directory uses this order -- see the project_multiband_cog memory's "migration
# caveat"). Pre-migration COGs live under this older dataset-directory order instead of
# the current dataset_path()'s (variable, time_unit, grid).
_LEGACY_DATASET_DIMENSION_ORDER = ("variable", "grid", "time_unit")


def _legacy_dataset_path(base_path, **kwargs):
    """LEGACY COG SHIM (see ``_LEGACY_DATASET_DIMENSION_ORDER``; remove together with it).

    Mirrors ``utils.dataset_path()`` but with the pre-migration dimension order, used only
    to *find* legacy files -- every write still goes through the canonical
    ``dataset_path()``.
    """

    path = Path(base_path)
    for dim_name in _LEGACY_DATASET_DIMENSION_ORDER:
        path = path / f"{dim_name}={kwargs[dim_name]!s}"
    return path


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


def _vrt_source_elements(source):
    """Open one ``vrt`` source and return its ``(dataset_el, selected_band_el)`` VRT XML.

    ``source`` is a bare path (band 1) or a ``(path, band)`` pair; ``band_el`` is that
    source's chosen ``VRTRasterBand`` element (still parented to ``dataset_el``).
    """

    path, band = source if isinstance(source, tuple) else (source, 1)
    with rio.open(path) as raster, MemoryFile() as memory:
        copy(raster, memory.name, driver="VRT")
        dataset_el = xml.etree.ElementTree.fromstring(memory.read().decode("utf-8"))
    return dataset_el, dataset_el.find(f"VRTRasterBand[@band='{band}']")


def vrt(sources):
    """Create a band-separate VRT dataset from the given sources.

    Each source is either a bare path (its band 1 is used) or a ``(path, band)`` pair
    selecting a specific 1-based band -- used by ``compact_cog`` to consolidate
    contiguous runs that span already multi-band COGs.  Output band order matches
    ``sources`` order; each output band inherits its source band's own description.

    Each source's XML is extracted in parallel (opening/copying a COG to a VRT is
    I/O-bound and releases the GIL, same as ``_parallel_read_cog_metadata``); the tree is
    then assembled serially so the output is identical regardless of thread scheduling.
    """

    # Adapted from: https://gis.stackexchange.com/a/421687
    with ThreadPoolExecutor() as pool:
        extracted = list(pool.map(_vrt_source_elements, sources))

    out_el = None
    for out_idx, (dataset_el, band_el) in enumerate(extracted, start=1):
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
    """Open a COG and read its band descriptions, dtype, and dataset tags, without
    reading pixel data.

    LEGACY COG SHIM (grep this tag to find every removable piece; delete once no on-disk
    COG is pre-migration -- see the project_multiband_cog memory): the old, single-band
    layout stored its calendar value as a *dataset tag* (set by the old ``write_cog`` via
    a plain ``update_tags(**kwargs)``) instead of a band description. When band 1 has no
    description at all but the dataset carries a calendar-dimension tag, synthesize the
    ``"<dim>=<value>"`` description string so every downstream consumer (``read_cog``,
    ``compact_cog``) treats the file exactly like a native one. The extra ``is_legacy``
    return value only matters to ``compact_cog``, which uses it to force a legacy file
    into the new format even when it has no neighbors to merge with.

    ``tags`` (metadata-typed dims like ``model``, written by ``write_cog`` via
    ``update_tags`` -- see ``write_cog``) is returned alongside so callers that need
    metadata dims for the catalog (``compact_cog``) don't have to open the file again;
    these dims are never COG path components, so tags are their only source.
    """

    with rio.open(path) as ds:
        descriptions = ds.descriptions
        tags = ds.tags()
        is_legacy = False
        if not descriptions[0]:
            legacy_dim = next((dim for dim in _CALENDAR_DIMENSIONS if dim in tags), None)
            if legacy_dim is not None:
                descriptions = (f"{legacy_dim}={tags[legacy_dim]}",)
                is_legacy = True
        return descriptions, ds.dtypes[0], is_legacy, tags


def _parallel_read_cog_metadata(paths):
    """Read ``(descriptions, dtype, is_legacy, tags)`` for several COGs in parallel threads.

    The calendar value a file/band covers can only be determined by opening it (the
    filename is an opaque UUID), so both ``read_cog`` and ``compact_cog`` must open every
    candidate file; doing so in parallel keeps that from being latency-bound.
    """

    with ThreadPoolExecutor() as pool:
        return list(pool.map(_read_cog_metadata, paths))


def _read_cog_band_rows(paths):
    """Yield ``(path, band, name, value, dtype, is_legacy, tags)`` for every band of every COG.

    Reads the given COGs in parallel (``_parallel_read_cog_metadata``) and expands each into
    one record per band, splitting the band description ``"<name>=<value>"`` into its
    calendar-dimension ``name`` and its raw (still-string) ``value``.  The sole place band
    descriptions are parsed -- shared by ``read_cog`` (via ``_band_calendar_frame``),
    ``compact_cog``, and ``catalog.raster_rows``.
    """

    for path, (descriptions, dtype, is_legacy, tags) in zip(paths, _parallel_read_cog_metadata(paths)):
        for band_idx, desc in enumerate(descriptions, start=1):
            name, _, value = desc.partition("=")
            yield path, band_idx, name, value, dtype, is_legacy, tags


def _patch_legacy_vrt(vrt_text, calendar_dim, values):
    """LEGACY COG SHIM (see ``_read_cog_metadata``; remove together with it).

    Two fixes applied purely to the in-memory VRT XML, never touching a source file
    (editing an already-written COG in place is refused by GDAL -- "has Cloud Optimized
    GeoTIFF layout... updating it will generally result in losing part of the
    optimizations" -- so this has to happen before the ``copy()`` that produces the
    output, not after):

    1. ``vrt()``'s root ``<Metadata>`` block is copied from the run's first source; if
       that source is legacy, its stale calendar-dimension tag would otherwise leak onto
       the merged output's dataset tags even though the value now lives correctly in every
       band's description -- violating the invariant that the calendar dimension is never
       a metadata tag (see ``write_cog``). Drop it.
    2. ``vrt()``'s per-band ``<Description>`` only reflects a *native* band description --
       a legacy source never had one (its calendar value was synthesized in Python by
       ``_read_cog_metadata`` and never written back to the file), so its VRTRasterBand
       has no ``<Description>`` at all and the merged output would end up with unreadable
       (empty) descriptions for those bands. Set every band's ``<Description>`` explicitly
       from ``values`` (parallel to the run, same band order the VRT wrote them in) so the
       result is correct regardless of which source bands were native vs. synthesized.
    """

    root = xml.etree.ElementTree.fromstring(vrt_text)

    metadata_el = root.find("Metadata")
    if metadata_el is not None:
        for mdi in metadata_el.findall("MDI"):
            if mdi.get("key") == calendar_dim:
                metadata_el.remove(mdi)

    for band_el, value in zip(root.findall("VRTRasterBand"), values):
        desc_el = band_el.find("Description")
        if desc_el is None:
            desc_el = xml.etree.ElementTree.Element("Description")
            band_el.insert(0, desc_el)
        desc_el.text = f"{calendar_dim}={value}"

    return xml.etree.ElementTree.tostring(root).decode("UTF-8")


def _band_calendar_frame(paths, path_column, calendar_col):
    """Expand each COG into one row per band with that band's calendar value.

    Every COG -- single-band (``write_cog``) or multi-band (a contiguous run merged by
    ``compact_cog``) -- stores each band's calendar value in its band description
    (``"<calendar_dim>=<value>"``), which is the sole source read here.
    """

    dtype = DIMENSIONS[calendar_col].dtype
    rows = {path_column: [], "band": [], "_value": []}
    for path, band_idx, _name, value, _dtype, _is_legacy, _tags in _read_cog_band_rows(paths):
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

    # Metadata dimensions (e.g. model) live in each COG's dataset tags, not the path or a
    # band description, so they can't be used as path/band filters -- exclude them.
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
    catalog_path=CATALOG_PATH,
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

    calendar_dim = cal_for(kwargs["time_unit"])
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

    # Keep the unified catalog side-table in sync with what was just written. Every
    # non-calendar kwarg (including metadata-typed dims like model, which were just
    # written as tags above) belongs on the catalog row -- see raster_catalog_rows.
    if catalog_path:
        dims = {
            dim.name: kwargs[dim.name]
            for dim in DIMENSIONS.values()
            if dim.name in kwargs and dim.name not in _CALENDAR_DIMENSIONS
        }
        rows = raster_catalog_rows(dims, calendar_dim, [kwargs[calendar_dim]])
        Catalog(catalog_path).append(rows)


def _calendar_value(value, calendar_dim):
    """Parse a band description's value (the part after ``"="``) into a typed value."""

    return date.fromisoformat(value) if calendar_dim == "calendar_day" else int(value)


def _write_run(run, out_path, grid, dtype, calendar_dim, values, is_legacy, compression, compression_level):
    """Consolidate one contiguous run into the multi-band COG at ``out_path``.

    A band-separate VRT is built over ``run``'s ``(path, band)`` sources (``vrt()``) and
    streamed into ``out_path`` with ``CreateCopy``, reusing ``_cog_profile()`` for the
    shared compression settings.  ``values`` (the run's per-band calendar values, in run
    order) is only used to repair a legacy source's missing band descriptions.
    """

    # The VRT is a throwaway XML sidecar (its SimpleSource entries hold absolute paths, so
    # its own location doesn't matter) -- keep it out of the data directory entirely.
    with tempfile.TemporaryDirectory() as tmp_dir:
        vrt_path = Path(tmp_dir) / f"{out_path.stem}.vrt"
        vrt_text = vrt(run)
        if is_legacy:  # LEGACY COG SHIM (see _patch_legacy_vrt)
            vrt_text = _patch_legacy_vrt(vrt_text, calendar_dim, values)
        vrt_path.write_text(vrt_text)
        profile = _cog_profile(
            grid, dtype, count=len(run), compression=compression, compression_level=compression_level
        )
        profile.update(interleave="PIXEL", num_threads="ALL_CPUS", bigtiff="YES")
        # out_path is a fresh UUID that can't collide with any existing source file, so the
        # copy can be written directly -- no temp-file/rename needed.  Each CreateCopy
        # already runs with num_threads=ALL_CPUS, so parallelizing runs would only contend.
        copy(str(vrt_path), str(out_path), **profile)


def compact_cog(
    base_path=BASE_PATH,
    catalog_path=CATALOG_PATH,
    output_base_path=None,
    compression="zstd",
    compression_level=6,
    **kwargs,
):
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

    Also migrates pre-multi-band-migration COGs it encounters -- including ones still
    sitting under the older dataset-directory order -- the first time it's swept up here,
    merged with any contiguous neighbors in the same pass rather than as a separate
    rewrite step (grep ``LEGACY COG SHIM`` for every removable piece).

    ``output_base_path``, when given, writes the merged COGs into a *separate* tree
    (canonical dataset-directory order) instead of alongside their sources, and leaves
    the source files in place -- used by the migration to build a long-format tree
    out-of-place without mutating the input ``base_path``. It defaults to ``base_path``
    (the in-place behavior). Out-of-place mode assumes every source needs rewriting
    (true for legacy COGs, which are always rewritten): a run already living optimally
    in a single up-to-date file is skipped and would *not* be copied across.

    Returns ``{"outputs": [Path, ...], "sources_merged": int}`` -- the newly written
    multi-band COGs and the number of source COGs they consumed (both empty/zero when
    every run was already optimal), so callers don't have to re-scan the output tree.
    """

    input_df = cast_columns(pl.DataFrame(kwargs))
    validate_data_frame(input_df, required_cols=DATASET_DIMENSIONS.keys())

    base_path = Path(base_path)
    output_base_path = base_path if output_base_path is None else Path(output_base_path)
    dataset_dir = dataset_path(base_path, **kwargs)
    grid = GRIDS[kwargs["grid"]]

    required_cols = required_columns(input_df, include_dataset=False, include_metadata=False)
    time_unit = kwargs["time_unit"]
    calendar_dim = cal_for(time_unit)
    path_cols = [col for col in required_cols if col != calendar_dim]
    glob_pattern = build_glob_pattern(required_cols=path_cols, extension="tif", nested=True, **kwargs)

    # Group matched files by directory (each directory holds every file for one
    # combination of non-calendar dimensions).
    groups = {}
    for path in dataset_dir.glob(glob_pattern):
        groups.setdefault(path.parent, []).append(path)

    # LEGACY COG SHIM (see _LEGACY_DATASET_DIMENSION_ORDER): also search the
    # pre-migration directory order and remap any matches to their canonical directory
    # (Path.glob on a directory that doesn't exist just yields nothing, so this is a
    # no-op once no legacy directory remains), so legacy files get pooled into the same
    # run-detection/merge pass as everything already in the canonical location.
    legacy_dataset_dir = _legacy_dataset_path(base_path, **kwargs)
    if legacy_dataset_dir != dataset_dir:
        for path in legacy_dataset_dir.glob(glob_pattern):
            canonical_dir = dataset_dir / path.parent.relative_to(legacy_dataset_dir)
            groups.setdefault(canonical_dir, []).append(path)

    outputs = []  # newly written multi-band COGs, across every directory/run
    sources_merged = 0  # source COGs consumed by those writes
    for directory, tifs in groups.items():
        # Pool every band of every file (regardless of existing band count) into one
        # sorted sequence of (value, path, band) triples.
        triples = []
        dtype = None
        legacy_paths = set()  # LEGACY COG SHIM (see _read_cog_metadata); remove with it.
        tif_tags = {}
        for tif, band_idx, _name, value, tif_dtype, is_legacy, tags in _read_cog_band_rows(tifs):
            dtype = tif_dtype
            if is_legacy:
                legacy_paths.add(tif)
            tif_tags[tif] = {k: v for k, v in tags.items() if k in METADATA_DIMENSIONS}
            triples.append((_calendar_value(value, calendar_dim), tif, band_idx))

        triples.sort(key=lambda triple: triple[0])
        values = pl.Series([value for value, _tif, _band in triples])
        offsets = calendar_run_offsets(values, calendar_dim, time_unit).to_list()

        runs = {}
        run_values = {}  # LEGACY COG SHIM (see below); remove alongside it.
        for (value, tif, band_idx), offset in zip(triples, offsets):
            runs.setdefault(offset, []).append((tif, band_idx))
            run_values.setdefault(offset, []).append(value)

        # Self-heal the catalog for this directory regardless of whether a physical
        # merge happens below -- this recomputes every run's rows from what was just
        # read, so directories containing legacy/pre-write_cog-hook files (which never
        # got a catalog row of their own) are backfilled as a side effect of compacting.
        if catalog_path:
            path_dims = parse_path_dims(directory.relative_to(Path(base_path)))
            directory_rows = pl.concat(
                [
                    raster_catalog_rows(
                        {k: cast_str(k, v) for k, v in {**path_dims, **tif_tags[run[0][0]]}.items()},
                        calendar_dim,
                        run_values[offset],
                    )
                    for offset, run in runs.items()
                ],
                how="diagonal_relaxed",
            )
            Catalog(catalog_path).append(directory_rows)

        for offset, run in runs.items():
            unique_paths = {tif for tif, _band in run}
            run_is_legacy = bool(unique_paths & legacy_paths)  # LEGACY COG SHIM
            # A run already living entirely in one up-to-date file is already optimal;
            # nothing to gain from rewriting it. A legacy run is rewritten regardless of
            # file count -- that's its only path into the new format.
            if len(unique_paths) < 2 and not run_is_legacy:
                continue

            # The output directory mirrors `directory`'s canonical dimensions but roots
            # at output_base_path (== base_path for in-place compaction).  LEGACY COG
            # SHIM: `directory` is always pre-existing once every source is already
            # canonical, but a legacy-only group's canonical directory may not exist on
            # disk yet; an out-of-place output directory never exists up front.
            out_directory = output_base_path / directory.relative_to(base_path)
            out_directory.mkdir(parents=True, exist_ok=True)
            out_path = out_directory / f"{uuid.uuid4()}.tif"

            _write_run(
                run, out_path, grid, dtype, calendar_dim, run_values[offset],
                run_is_legacy, compression, compression_level,
            )
            outputs.append(out_path)
            sources_merged += len(unique_paths)

            # Out-of-place output leaves the source tree untouched; only an in-place
            # merge consumes and removes its sources.
            if output_base_path == base_path:
                for path in unique_paths:
                    path.unlink()

    # Coalesce whatever catalog rows now exist for this variable/time_unit/grid
    # partition (the self-healed rows appended above, plus any older, more-fragmented
    # rows from earlier writes) into minimal contiguous-run rows.
    if catalog_path:
        Catalog(catalog_path).compact_partition(kwargs["variable"], kwargs["time_unit"], kwargs["grid"])

    return {"outputs": outputs, "sources_merged": sources_merged}
