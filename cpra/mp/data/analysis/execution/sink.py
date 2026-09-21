"""The sink: turning a finished frame into bytes on disk or bytes in a response, plus the
one shaping step (`output.sort`/`output.select`) every tabular result gets, applied once
and last.

Five destinations, five free functions -- none of them classes, none of them holding a
reference to anything but the arguments they were called with:

- `sink_tabular` -- csv/ndjson/parquet, over a `LongFrame`, always lazily. Delegates the
  actual writing to `io.output.stream_tabular`/`tabular_kwargs`, which already hold the
  CSV float-precision rule; this only builds the `materialize` closure that expects.
- `write_raster` -- `tif`, over a `BandWideFrame`. Returns `False` (not a raise) when the
  frame carries no bands, which is what lets a caller tell "no source data" (a 404) apart
  from every other failure.
- `render_tile` -- `png`/`jpg`, a single map tile, over the `BandWideFrame` `execute`
  returns for a tile-scoped plan. Warps, colors, encodes: `io.raster_io.
  warp_array_to_tile` for the geometry, `io.output.resolve_legend` /
  `legends.Legend.colorize_pixels` for the color. It runs no ops and opens no raster --
  the tile's read window is a compile-time decision (`execution.plan.TileRequest`) and the
  walk that honors it is `execute`'s, the same one every other destination is fed by.
- `empty_tile` -- the image bytes a caller sends when `render_tile` had nothing to render.
  A blank tile of the requested format and size, encoded through the same path a real one
  is, so a client that asked for an image always gets a decodable image back.
- `sink_vector_tile` -- `pbf`, a single map tile, over a `LongFrame`. Injects a `value`
  property into a pre-generated PMTiles boundary layer; a from-scratch rewrite of the old
  engine's tile-injection code (protobuf-level, not spatial), not a port of it.
- `apply_output_shaping` -- `output.sort`/`output.select`, applied once, last, to whatever
  `LongFrame` a tabular or `pbf` sink is about to consume.

`pbf` is deliberately not a member of `request.types.DataFormat`/`TileFormat` -- it is
selected by whichever caller wants a vector tile rather than read off `output.format`, the
same way `png`/`jpg` (raster tiles) are already a separate code path from `tif` (a whole
raster) despite both being requests against a raster frame.

What this module does NOT do: decide `resolve_output_type` (that is structural, from the
frame's own shape, computed once in `io.output`); re-check anything `request.ops.
validate_shapes` already rejects (a long frame under a raster-only format, more than one
value column reaching here); or translate an exception into a transport response (an HTTP
route's job, not written yet -- see `execution.execute`'s own docstring on the same point
for `LoweringNotImplementedError`).
"""

import gzip
from io import BytesIO

import numpy as np
import polars as pl
import rasterio
from mapbox_vector_tile.Mapbox.vector_tile_pb2 import tile as VectorTile
from PIL import Image
from pmtiles.reader import MmapSource, Reader

from cpra.mp.data import config
from cpra.mp.data.analysis.execution.frame import (
    BandWideFrame,
    LongFrame,
    band_column_names,
    has_no_bands,
    row_major_grid_cell,
)
from cpra.mp.data.analysis.execution.plan import DEFAULT_TILE_SIZE, TileRequest
from cpra.mp.data.analysis.io.output import resolve_legend, stream_tabular, tabular_kwargs
from cpra.mp.data.analysis.io.raster import NUMPY_TO_POLARS
from cpra.mp.data.analysis.io.raster_io import warp_array_to_tile
from cpra.mp.data.analysis.request.errors import ShapeError

__all__ = [
    "DEFAULT_NODATA",
    "apply_output_shaping",
    "sink_tabular",
    "write_raster",
    "render_tile",
    "empty_tile",
    "sink_vector_tile",
]

#: The sentinel a raster sink falls back to when neither an explicit `nodata=` argument
#: nor the source file's own declared nodata says what one is. Matches `Variable`'s own
#: default (`variables.py`, `Variable.__init__`) and the long-standing tile-path
#: fallback -- not a new number invented for this module.
DEFAULT_NODATA = -9999

# The inverse of `io.raster.scan_raster`'s own `NUMPY_TO_POLARS` -- built here rather than
# duplicated by hand, so a numpy dtype string this module infers for a GeoTIFF write is
# always one `scan_raster` itself could have produced a column of.
_POLARS_TO_NUMPY = {v: k for k, v in NUMPY_TO_POLARS.items()}


# ----------------------------------------------------------------------------------------
# Output shaping: sort/select, once, last
# ----------------------------------------------------------------------------------------


def apply_output_shaping(lf: LongFrame, output) -> LongFrame:
    """Apply `output.sort`/`output.select` to `lf`, in that order -- the one shaping step
    every `LongFrame` result gets, run once, after every op (`request.pipeline.Output`'s
    own docstring).

    `sort` is a comma-separated list of dimension names, each optionally `-`-prefixed for
    descending (`-calendar_year` sorts that column descending; `calendar_year` sorts it
    ascending). `select` is a comma-separated column subset, applied as a plain
    `.select()` after sorting -- sorting AFTER a `select` that dropped the sort key would
    silently stop working the moment a caller reordered the two, so the order here is
    fixed rather than left to whichever the caller happened to set first.

    Neither list is checked against the frame's actual columns here -- an unknown name
    surfaces as Polars' own `ColumnNotFoundError` when `lf` is collected, the same way an
    unknown column in a `where`/`calculate` expression already does elsewhere in this
    package. `request.pipeline.validate_pipeline` (via `_validate_output_sort_or_select`)
    already checked the SYNTAX of both strings before this ever runs -- no empty
    component, no bare `-`, no `-`-prefixed `select` entry, no name repeated -- so what
    reaches here is syntactically well-formed; existence is deliberately the only half
    left to Polars, since it needs the frame's real column namespace, which is not known
    until compile/run time. The two halves of this split are documented in both places on
    purpose: `request/pipeline.py` cannot check existence, and this function should not
    duplicate a syntax check it has no need to repeat.
    """
    sort = getattr(output, "sort", None)
    select = getattr(output, "select", None)

    if sort:
        parts = [part.strip() for part in sort.split(",") if part.strip()]
        columns = [part[1:] if part.startswith("-") else part for part in parts]
        descending = [part.startswith("-") for part in parts]
        lf = lf.sort(columns, descending=descending)

    if select:
        columns = [part.strip() for part in select.split(",") if part.strip()]
        lf = lf.select(columns)

    return lf


# ----------------------------------------------------------------------------------------
# Tabular: csv / ndjson / parquet
# ----------------------------------------------------------------------------------------


def _materialize_closure(lf: LongFrame):
    """Build the `materialize` callable `io.output.stream_tabular` expects: called with no
    arguments it collects `lf`; called with a method name (plus that method's own args/
    kwargs) it calls that method on `lf` directly (a `sink_*` method, for the streaming
    path) -- see that function's own docstring.

    A failed collect is deliberately NOT translated here: `ColumnNotFoundError`/
    `FileNotFoundError` propagate as themselves, for `api/boundary.py` to map. Which
    status a read failure deserves is a transport-facing concern (the error boundary a
    route installs), not something this closure should decide on its own.
    """

    def materialize(method_name=None, *args, **kwargs):
        if method_name is None:
            return lf.collect()
        return getattr(lf, method_name)(*args, **kwargs)

    return materialize


def sink_tabular(lf: LongFrame, out_path, format, *, float_precision=None, **profile_options) -> None:
    """Write `lf` to `out_path` as `format` (`csv`/`ndjson`/`parquet`/`arrow`), always
    lazily (`io.output.stream_tabular`) -- this sink never materializes the whole frame
    first. `io.output.write_tabular` is the eager, collect-first alternative, reached
    only when a caller has already decided the result is small; nothing estimates a
    compiled `ExecutionPlan`'s result size, so there is only the one (safe-default) path
    here.

    `float_precision` is forwarded to `io.output.tabular_kwargs`, which only ever uses it
    for `format == "csv"` -- see that function's own docstring. `**profile_options` is
    forwarded straight through to the underlying Polars writer -- `compression`/
    `compression_level` for `parquet`, landing here as the plain values `run_to_path`'s
    caller already resolved from a request's own compression choice (see `api.pipeline
    .PipelineTabularSource`'s docstring for why those arrive as plain values rather than
    an `api.models.Compression` object).
    """
    kwargs = tabular_kwargs(format, float_precision)
    kwargs.update(profile_options)
    materialize = _materialize_closure(lf)
    stream_tabular(materialize, out_path, format, **kwargs)


# ----------------------------------------------------------------------------------------
# tif: a whole raster
# ----------------------------------------------------------------------------------------


def _resolve_write_dtype(df: pl.DataFrame, band_names: list[str], dtype: str | None) -> str:
    """The numpy dtype string `write_raster` should declare the file's bands as, when the
    caller did not pass one explicitly.

    A `BandWideFrame`'s band columns are not guaranteed to share one Polars dtype -- a
    `calculate` can rewrite one band's values to a type the others never touch, and
    `execution.band_ops`'s own module docstring notes this directly. A single GeoTIFF, by
    contrast, has exactly one dtype for every band in the file (`rasterio`'s `dtype=`
    profile key is scalar, not per-band). When every surviving band happens to still
    share one Polars dtype, that dtype's numpy equivalent is used, so a write that never
    actually mixed types costs no precision. When they differ, this falls back to
    `float64` -- wide enough to hold anything `NUMPY_TO_POLARS` maps to without loss,
    which is the safe default for a shape this function cannot otherwise resolve on its
    own; a caller that wants a narrower file passes `dtype=` itself.
    """
    if dtype is not None:
        return dtype
    schema = df.schema
    polars_dtypes = {schema[name] for name in band_names}
    if len(polars_dtypes) == 1:
        numpy_dtype = _POLARS_TO_NUMPY.get(next(iter(polars_dtypes)))
        if numpy_dtype is not None:
            return numpy_dtype
    return "float64"


def write_raster(band_wide: BandWideFrame, out_path, *, dtype=None, nodata=DEFAULT_NODATA, **profile_options) -> bool:
    """Write `band_wide` to `out_path` as a GeoTIFF, one band per surviving `b<i>` column.

    Returns `False`, rather than raising, when `band_wide` carries no bands at all (a
    dimension filter narrowed the source to nothing) -- a falsy return rather than an
    exception because "the selection matched nothing" is not a failure, and every caller
    has its own answer for it: `api/pipeline.py`'s `PipelineRasterResponse` turns it into
    a 404, `cpra.mp.data.analyze_data` into a `NotFoundError`. Every other case writes the
    file and returns `True`.

    Row order in `band_wide`'s `LazyFrame` is the READ's emission order, not necessarily
    row-major (see `execution/frame.py`'s band-wide invariants) -- `row_major_grid_cell`
    recovers each row's true pixel position and this sorts by it before reshaping, which
    is what makes the reshape below correct regardless of how the original read was
    batched.

    `dtype`/`nodata` are explicit, caller-supplied parameters rather than derived from a
    `Variable` lookup or a source file's own profile: `band_wide` is the END of a walk
    that may have run a `group` (which drops the `path` column from `metadata_df` --
    `execution.band_ops.group_band_wide` selects only `by`/the reserved index column into
    its output metadata), so there may be no file left to open and ask, and
    `request.pipeline.Output` itself carries neither field. `nodata` defaults to
    `DEFAULT_NODATA`; `dtype`, left `None`, is inferred (`_resolve_write_dtype`).
    Resolving the request-appropriate values from a `Variable`/route parameter is a
    wiring decision for whichever route eventually calls this, mirroring how `api/
    pipeline.py` resolves the request's own compression choice at the route layer rather
    than down here.

    `**profile_options` is forwarded straight into `rasterio.open`'s write profile,
    layered over this function's own defaults (`driver="GTiff"` plus the frame's own
    geometry), so a caller wanting compression, `api.pipeline.COMPRESSION_LEVEL_OPTIONS`-
    style tuning, or a `driver="COG"` override supplies it the same way.
    """
    lf, metadata_df, window = band_wide
    if has_no_bands(metadata_df):
        return False

    band_names = band_column_names(metadata_df)
    ordered = row_major_grid_cell(lf, window).sort("grid_cell").select(band_names)
    df = ordered.collect()

    out_dtype = _resolve_write_dtype(df, band_names, dtype)
    data = np.stack(
        [df.get_column(name).fill_null(nodata).to_numpy().astype(out_dtype, copy=False) for name in band_names]
    ).reshape(len(band_names), window.height, window.width)

    profile = dict(
        driver="GTiff",
        height=window.height,
        width=window.width,
        count=len(band_names),
        dtype=out_dtype,
        crs=window.crs,
        transform=window.transform,
        nodata=nodata,
    )
    profile.update(profile_options)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
    return True


# ----------------------------------------------------------------------------------------
# png / jpg: a single map tile
# ----------------------------------------------------------------------------------------


def _encode_tile_image(color: np.ndarray, format) -> bytes:
    """Encode one `(tile_size, tile_size, 4)` RGBA array as `png`/`jpg` bytes.

    Shared by `render_tile` and `empty_tile` so the blank tile a caller sends for a miss is
    produced by the same encoder as a real one -- a blank tile that a client's decoder
    choked on would defeat the entire point of sending one.

    JPEG carries no alpha channel, so the alpha column is dropped rather than flattened
    against a background: a fully transparent RGBA pixel therefore encodes as whatever RGB
    it was carrying, which for `empty_tile`'s blank array is black. There is no transparent
    JPEG to fall back on, so any blank `jpg` tile is a solid color and this picks the one
    that makes the two formats the same array with and without its fourth channel.
    (`Image.fromarray` infers RGB vs RGBA from the array's own shape; passing `mode=`
    explicitly is deprecated as of Pillow 11.)
    """
    image = Image.fromarray(color[:, :, :3] if format == "jpg" else color)
    with BytesIO() as memfile:
        image.save(memfile, format=("JPEG" if format == "jpg" else "PNG"))
        return memfile.getvalue()


def empty_tile(format, tile_size: int = DEFAULT_TILE_SIZE) -> bytes:
    """A blank `tile_size`-square tile of `format` -- the body a caller pairs with the 404
    it sends when `render_tile` returned `None`.

    The status code is the route's decision, not this module's; what this exists for is that
    a 404 on an image URL should still carry a decodable image. A client that requested
    `.png` and got a JSON error document instead logs a decode failure per missing tile,
    and a map panned past the edge of the data requests a great many of them. The tile is
    transparent (`png`) rather than a rendered nodata tile, because nothing here knows the
    request's legend and a legend's own lowest break is not transparent -- see
    `render_tile`'s note on what a real all-nodata tile actually colors as.

    A caller sending this for a MISSING tile and a caller rendering a real one both go
    through `_encode_tile_image`, so the two cannot drift in format or in size.
    """
    return _encode_tile_image(np.zeros((tile_size, tile_size, 4), dtype="uint8"), format)


def render_tile(band_wide: BandWideFrame, tile: TileRequest, format, *, color_map=None, variable_legend=None, nodata=DEFAULT_NODATA) -> bytes | None:
    """Render one `png`/`jpg` map tile from an already-executed band-wide result.

    `band_wide` is exactly what `execution.execute.execute` returns for a plan compiled
    with this same `tile` (`execution.plan.compile_plan`'s `tile=` argument): its pixels
    are the tile's own window, already narrowed by whatever `filter`/`calculate`/`group`/
    `join`/`project` the request asked for. This function runs no ops of its own. It warps
    the one surviving band onto the tile's pixel grid, colors it, and encodes it -- the
    whole of what "sink" means for an image tile.

    That division is what keeps the straddling-feature widening reachable. Which pixels a
    tile needs off a raster depends on whether a raster-to-vector `project` redistributes
    them onto polygons that cross the tile edge, which is a property of the compiled plan,
    so the window is sized there and the whole op algebra runs against it. A sink that
    opened the raster itself could only ever support the ops that keep a frame band-wide,
    which excludes that `project` exactly.

    Returns `None` -- not a raise -- when the result carries no bands. That covers both
    "the tile's footprint misses this raster entirely" (`plan.ResolvedInput.tile_disjoint`,
    which `execute` turns into the same empty frame) and "a dimension predicate narrowed
    this request to nothing", because they are the same answer to a caller: this tile has
    nothing to show. A route turns that into a 404 whose body is `empty_tile(format)`, so
    the response is still a decodable image of the format the URL asked for.

    The surviving frame must carry EXACTLY one band. `Legend.colorize_pixels` colors one
    2-D array of pixel values, and nothing upstream guarantees a band-wide result narrows
    to one band by the time it reaches a tile -- there is no tile-side analogue of
    `request.ops.validate_shapes`'s single-value-column rule. `colorize_pixels`'s own
    `[0]` index into what it assumes is a length-1 leading axis would silently render
    band 0 of however many survived; this raises instead of guessing which band the
    caller meant.

    `nodata` is the value null pixels are filled with before the warp, and the value the
    warp itself fills the tile's own out-of-source corners with -- a sentinel this function
    is told, not a property of any source file. The scan has already masked each band's
    declared nodata to null (`io.raster.scan_raster`'s `mask_nodata`), so by here there is
    nothing left to read off a file even if this function still opened one.
    `color_map`/`variable_legend` are forwarded to `io.output.resolve_legend` unchanged.

    Note that a nodata pixel is NOT rendered transparent: `Legend.colorize_pixels`
    digitizes against the legend's break points, and a value below the lowest break --
    which `DEFAULT_NODATA` always is -- lands in the first bucket, whose color is an opaque
    one (no `LEGENDS` entry declares a transparent under-color). That is the
    long-standing behavior, reproduced here rather than changed, and it is why
    `empty_tile` builds its blank tile directly instead of colorizing an all-nodata
    array.
    """
    lf, metadata_df, window = band_wide
    if has_no_bands(metadata_df):
        return None

    band_names = band_column_names(metadata_df)
    if len(band_names) != 1:
        raise ShapeError(
            f"render_tile: {len(band_names)} bands reached the tile sink, not 1 -- a "
            "rendered tile needs exactly one band (add a trailing group step to "
            "collapse the rest)"
        )

    # Sorted by row-major position for the same reason `write_raster` sorts: the frame's
    # own row order is the order the read emitted pixels in, which is the window's
    # row-major order only when the read was a single batch. A tile window is, but that is
    # a property of how `scan_raster` happens to batch this window rather than a guarantee
    # this function should rest a reshape on.
    array = (
        row_major_grid_cell(lf, window)
        .sort("grid_cell")
        .select(band_names[0])
        .collect()
        .get_column(band_names[0])
        .fill_null(nodata)
        .to_numpy()
        .reshape(window.height, window.width)
    )
    # `window.transform` already describes these pixels -- `scan_raster` derived it for the
    # window it read -- so this is `warp_array_to_tile`, not `warp_window_to_tile`, which
    # exists to derive that same transform for a caller who has only the source's.
    tile_data = warp_array_to_tile(
        array[np.newaxis, :, :],
        array_transform=window.transform,
        src_crs=window.crs,
        tile_bounds=tile.bounds,
        tile_size=tile.tile_size,
        nodata=nodata,
        resampling=tile.resampling,
    )

    legend = resolve_legend(color_map, variable_legend)
    return _encode_tile_image(legend.colorize_pixels(tile_data), format)


# ----------------------------------------------------------------------------------------
# pbf: a single vector tile
# ----------------------------------------------------------------------------------------


def sink_vector_tile(lf: LongFrame, grid_name: str, tile, *, grid_cell_ids=None, tiles_path=None) -> bytes | None:
    """Render one Mapbox Vector Tile by injecting `lf`'s `value` column into `grid_name`'s
    pre-generated PMTiles boundary layer for `tile` (an `(x, y, z)` tuple).

    Geometry is never decoded here -- only the boundary layer's `keys`/`values`/feature
    `tags` are touched, which is why this is O(features) protobuf work rather than
    O(coordinates) geometry work.

    Returns `None` -- not a raise -- for every "nothing to render" outcome: no PMTiles
    archive for `grid_name`, no tile at these `x`/`y`/`z` in that archive, no boundary
    layer in the parsed tile, no feature surviving `grid_cell_ids` (when given), or no
    row of `lf` landing on a surviving feature with a non-null `value`. A route does not
    need to distinguish which -- they are all "this tile has nothing to show" the same way
    `write_raster`'s `False` is "this dataset has nothing to write" -- and sends a bodiless
    204 for all of them. There is no vector-tile counterpart to `empty_tile`: an MVT client
    treats a 204 as an empty tile natively, so there is nothing to encode.

    `grid_cell_ids`, when given, narrows which of the tile's own feature ids are eligible
    at all, BEFORE `lf` is even queried -- how a `grid_cell` selection (or a projection
    onto a specific destination cell) narrows a tile's answer to a specific id set.
    Resolving which id set (if any) a
    given request implies is a route/request concern this function does not derive on its
    own; `None` (the default) considers every feature id the tile itself carries, which is
    the common case.

    `tiles_path`, left `None`, defaults to `cpra.mp.data.config.TILES_PATH` (the
    configured tile archive directory); tests pass a sandboxed path instead, the same
    `CPRA_MP_DATA_TILES_PATH`-prefixed override `tests/conftest.py` already sets up.
    """
    tiles_path = tiles_path if tiles_path is not None else config.TILES_PATH
    x, y, z = tile

    path = tiles_path / f"{grid_name}.pmtiles"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        compressed = Reader(MmapSource(f)).get(z, x, y)
    if compressed is None:
        return None

    vt = VectorTile()
    vt.ParseFromString(gzip.decompress(compressed))
    boundary = next((layer for layer in vt.layers if layer.name == "boundary"), None)
    if boundary is None or not boundary.features:
        return None

    tile_ids = [feature.id for feature in boundary.features]
    if grid_cell_ids is not None:
        valid = set(grid_cell_ids)
        tile_ids = [feature_id for feature_id in tile_ids if feature_id in valid]
    if not tile_ids:
        return None

    df = lf.filter(pl.col("grid_cell").is_in(tile_ids)).collect()
    if df.is_empty():
        return None

    id_values = {
        cell: value
        for cell, value in zip(df["grid_cell"].to_list(), df["value"].to_list())
        if value is not None
    }
    if not id_values:
        return None

    key_index = len(boundary.keys)
    boundary.keys.append("value")
    for feature in boundary.features:
        value = id_values.get(feature.id)
        if value is not None:
            entry = boundary.values.add()
            entry.double_value = value
            feature.tags.extend([key_index, len(boundary.values) - 1])

    return vt.SerializeToString()
