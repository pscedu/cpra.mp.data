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
  from every other failure. `write_raster_strips` writes the same file from a compiled
  plan, a strip of rows at a time, wherever the plan's ops allow it.
- `render_tile` -- `png`/`jpg`, a single map tile, over the `BandWideFrame` `execute`
  returns for a tile-scoped plan. Warps, colors, encodes: `io.raster_io.
  warp_array_to_tile` for the geometry, `io.output.resolve_legend` /
  `legends.Legend.colorize_pixels` for the color. It runs no ops and opens no raster --
  the tile's read window is a compile-time decision (`execution.plan.TileRequest`) and the
  walk that honors it is `execute`'s, the same one every other destination is fed by.
- `sink_vector_tile` -- `pbf`, a single map tile, over a `LongFrame`. Injects a `value`
  property into a pre-generated PMTiles boundary layer; a from-scratch rewrite of the old
  engine's tile-injection code (protobuf-level, not spatial), not a port of it.
- `apply_output_shaping` -- `output.sort`/`output.select`, applied once, last, to whatever
  `LongFrame` a tabular or `pbf` sink is about to consume.

`pbf` is deliberately not a member of `request.types.DataFormat`/`TileFormat` -- it is
selected by whichever caller wants a vector tile rather than read off `output.format`, the
same way `png`/`jpg` (raster tiles) are already a separate code path from `tif` (a whole
raster) despite both being requests against a raster frame.

What this module does NOT do: decide which shape a request answers in, or re-check
anything `request.ops.validate_shapes` already rejected (a long frame under a raster-only
format, more than one value column reaching here) -- both are settled at compile time, by
typing the frame through the ops themselves. Nor does it translate an exception into a
transport response (the error boundary a route installs -- `api/boundary.py`, named again
at the one place below where the distinction decides what this module does).
"""

import gzip
import math
from io import BytesIO

import numpy as np
import polars as pl
import rasterio
from mapbox_vector_tile.Mapbox.vector_tile_pb2 import tile as VectorTile
from PIL import Image
from pmtiles.reader import MmapSource, Reader
from rasterio.windows import Window

from cpra.mp.data import config
from cpra.mp.data.analysis.execution.execute import execute
from cpra.mp.data.analysis.execution.frame import (
    RESERVED_BAND_INDEX_COLUMN,
    BandWideFrame,
    LongFrame,
    band_column_names,
    has_no_bands,
    row_major_grid_cell,
)
from cpra.mp.data.analysis.execution.plan import ExecutionPlan, TileRequest, with_read_window
from cpra.mp.data.analysis.io.output import resolve_legend, stream_tabular, tabular_kwargs
from cpra.mp.data.analysis.io.raster import NUMPY_TO_POLARS
from cpra.mp.data.analysis.io.raster_io import warp_array_to_tile
from cpra.mp.data.analysis.io.window import ReadWindow
from cpra.mp.data.analysis.request.errors import ShapeError

__all__ = [
    "DEFAULT_NODATA",
    "apply_output_shaping",
    "render_tile",
    "sink_tabular",
    "sink_vector_tile",
    "write_raster",
    "write_raster_strips",
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
        columns = [part.removeprefix("-") for part in parts]
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


def sink_tabular(lf: LongFrame, out_path, data_format, *, float_precision=None, **profile_options) -> None:
    """Write `lf` to `out_path` as `data_format` (`csv`/`ndjson`/`parquet` -- `DataFormat`'s
    tabular members, which are exactly the keys `io.output.SINK_METHODS` carries), always
    lazily (`io.output.stream_tabular`) -- this sink never materializes the whole frame
    first. `io.output.write_tabular` is the eager, collect-first alternative, reached
    only when a caller has already decided the result is small; nothing estimates a
    compiled `ExecutionPlan`'s result size, so there is only the one (safe-default) path
    here.

    `float_precision` is forwarded to `io.output.tabular_kwargs`, which only ever uses it
    for `csv` -- see that function's own docstring. `**profile_options` is
    forwarded straight through to the underlying Polars writer -- `compression`/
    `compression_level` for `parquet`, landing here as the plain values `run_to_path`'s
    caller already resolved from a request's own compression choice (see `api.pipeline
    .PipelineTabularSource`'s docstring for why those arrive as plain values rather than
    an `api.models.Compression` object).
    """
    kwargs = tabular_kwargs(data_format, float_precision)
    kwargs.update(profile_options)
    materialize = _materialize_closure(lf)
    stream_tabular(materialize, out_path, data_format, **kwargs)


# ----------------------------------------------------------------------------------------
# tif: a whole raster
# ----------------------------------------------------------------------------------------


def _resolve_write_dtype(schema: pl.Schema, band_names: list[str], dtype: str | None) -> str:
    """The numpy dtype string `write_raster` should declare the file's bands as, when the
    caller did not pass one explicitly.

    A `BandWideFrame`'s band columns are not guaranteed to share one Polars dtype -- a
    `case` arm can rewrite one band's values to a type the others never touch, and
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
    polars_dtypes = {schema[name] for name in band_names}
    if len(polars_dtypes) == 1:
        numpy_dtype = _POLARS_TO_NUMPY.get(next(iter(polars_dtypes)))
        if numpy_dtype is not None:
            return numpy_dtype
    return "float64"


# `metadata_df` columns that say where a band was READ FROM rather than what it IS: the two
# columns `io.raster.scan_raster` requires, plus the reserved index `execution.frame.
# released_band_metadata` leaves behind when a `group` collapsed every dimension away. None
# of the three identifies a band to a person opening the file, and the reserved one is not
# even allowed out of the walk as a dimension, so a band description never mentions them.
_NON_IDENTIFYING_METADATA_COLUMNS = frozenset({"path", "band", RESERVED_BAND_INDEX_COLUMN})


def _band_descriptions(metadata_df: pl.DataFrame) -> list[str]:
    """One band label per `metadata_df` row, in band order: the row's dimension values
    spelled `"<dimension>=<value>"` and joined with `", "`.

    `"<dimension>=<value>"` is not a format invented here -- it is the one
    `cpra.mp.data.raster` already writes (`write_cog`) and already parses back out
    (`_read_cog_band_rows`, which `read_cog` uses to recover a band's calendar value,
    since a UUID-named COG's path cannot carry it). A single-dimension band therefore
    comes out byte-identical to what that path would have written, which is what keeps a
    downloaded one-calendar-dimension GeoTIFF readable by the same machinery; a band that
    survived with several dimensions spells all of them, which that parser was never
    promised and does not claim to round-trip.

    A null is omitted rather than rendered: `band_ops.join_band_wide` unions its operands'
    metadata columns, so a band from one operand carries nulls for a dimension only the
    other operand had, and "this band has no such dimension" is what that means. A row
    with nothing left to say yields `""`, which `write_raster` skips -- an empty
    description is indistinguishable from an absent one once GDAL has written it, so there
    is no value in setting it.
    """
    columns = [name for name in metadata_df.columns if name not in _NON_IDENTIFYING_METADATA_COLUMNS]
    if not columns:
        return [""] * metadata_df.height
    return [
        ", ".join(f"{name}={value}" for name, value in row.items() if value is not None)
        for row in metadata_df.select(columns).iter_rows(named=True)
    ]


def write_raster(band_wide: BandWideFrame, out_path, *, dtype=None, nodata=DEFAULT_NODATA, **profile_options) -> bool:
    """Write `band_wide` to `out_path` as a GeoTIFF, one band per surviving `b<i>` column.

    Returns `False`, rather than raising, when `band_wide` carries no bands at all (a
    dimension filter narrowed the source to nothing) -- a falsy return rather than an
    exception because "the selection matched nothing" is not a failure, and every caller
    has its own answer for it: `api/pipeline.py`'s `PipelineRasterResponse` turns it into
    a 404, `cpra.mp.data.analyze_data` into a `NotFoundError`. Every other case writes the
    file and returns `True`.

    Each band is labeled with its own `metadata_df` row (`_band_descriptions`), so band
    order alone is not the only record of which band is which -- the file says so itself.
    That matters most for the download path: band order is decided by the walk, and a
    client holding only the bytes has nothing else to read a band's calendar value off.

    The frame is written back in its own row order, one read batch at a time
    (`_write_batches`), rather than sorted into raster order first.

    **Memory.** This used to label every row with its `grid_cell`, sort the whole frame
    by it, and collect it with the default in-memory engine, which materializes every
    intermediate column of the walk at full raster height at once: a full-extent
    `lnd_conf` download (five Int32 bands in, one Float64 band out, 97M pixels) peaked at
    10.6 GiB, and `lnd_conf_chng` (ten in) at up to 16.8 GiB, for a 0.4-0.75 GB result.
    The streaming engine evaluates the walk a morsel at a time, so only the result itself
    is held -- 2.2 and 1.5 GiB peak for the same two. Rows come out of
    `collect(engine="streaming")` in the frame's order, which is what the batch-by-batch
    write relies on. `collect_batches`, which would also bound the result, is NOT safe to
    substitute: on Polars 1.41.2 it returned rows out of order for `lnd_conf_chng` (a
    `calculate` after a band-wide `join`), in different places from run to run, while
    `collect(engine="streaming")` of the same frame matched the in-memory engine exactly.
    What is left scales with the result, so a many-band download still needs more than a
    pod has: `write_raster_strips` bounds it by computing the plan a strip at a time, and
    is what `execute.run_to_path` uses wherever the plan allows.

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
    df = lf.select(band_names).collect(engine="streaming")
    out_dtype = _resolve_write_dtype(df.schema, band_names, dtype)

    with rasterio.open(
        out_path, "w", **_raster_profile(window, len(band_names), out_dtype, nodata, profile_options)
    ) as dst:
        _write_batches(dst, df, band_names, window, (window.col_off, window.row_off), out_dtype, nodata)
        _describe_bands(dst, metadata_df)
    return True


def write_raster_strips(
    plan: ExecutionPlan, out_path, *, base_path=None, dtype=None, nodata=DEFAULT_NODATA, **profile_options
) -> bool:
    """`write_raster` for a compiled `plan` rather than an executed frame: the same GeoTIFF,
    computed and written one full-width strip of rows at a time, so the peak is bounded by
    the strip rather than by the result.

    `write_raster` has to hold the whole result -- every band, every pixel -- because the
    walk is collected once. That is fine for one band (0.4-0.75 GB for `lnd_conf`), and
    not for a plain multi-band download: 55 bands of the 16640 x 5824 grid are ~21 GB.
    Here the plan is executed once per strip instead, with every raster origin's read
    narrowed to it (`plan.with_read_window`), which is exactly how a tile already runs the
    walk over less than the raster. That is only the same answer as the full-extent run
    for a plan whose every op is pixel-local; `plan.windowed_read_blocker` says whether
    this one is, and the caller (`execute.run_to_path`) falls back to `write_raster`
    when it is not.

    Strips are full-width because the file is: a GeoTIFF without `tiled=True` is laid out
    in full-width strips of rows (GDAL picks how many per strip; one, for this grid's
    width and the default compression), so a full-width strip write completes every file
    strip it touches, and each is compressed and written exactly once. A narrower write
    would leave partly filled strips in GDAL's block cache until their neighbours arrived,
    and one evicted early is written twice -- the first copy left behind as dead space in
    the file. `_strip_height` sizes the strip to `config.RASTER_WRITE_STRIP_MB` and keeps
    it a whole number of the output's block rows, and of the source's too when the budget
    allows (a source block straddling two strips is decompressed twice).

    The first `execute` is over the full extent and reads no pixels -- it only opens each
    file for its metadata -- and settles what every strip must agree with: the band list
    (checked again per strip, so a band set that somehow depended on the data raises
    rather than writing mislabeled bands), the dtype, and the window strips are cut from.
    Each strip's own result must come back covering exactly that strip at full
    resolution, which is the check that the window actually reached every read.

    Returns `False` for a result with no bands, as `write_raster` does.
    """
    lf, metadata_df, window = execute(plan, base_path=base_path)
    if has_no_bands(metadata_df):
        return False
    if window.decimation != (1, 1):
        raise ValueError(f"write_raster_strips: the full-extent read is decimated {window.decimation}")

    band_names = band_column_names(metadata_df)
    schema = lf.select(band_names).collect_schema()
    out_dtype = _resolve_write_dtype(schema, band_names, dtype)
    origin = (window.col_off, window.row_off)

    with rasterio.open(
        out_path, "w", **_raster_profile(window, len(band_names), out_dtype, nodata, profile_options)
    ) as dst:
        strip_height = _strip_height(window, len(band_names), dst.block_shapes[0][0])
        for row_off in range(window.row_off, window.row_off + window.height, strip_height):
            strip = Window(
                window.col_off, row_off, window.width, min(strip_height, window.row_off + window.height - row_off)
            )
            strip_lf, strip_metadata, strip_window = execute(with_read_window(plan, strip), base_path=base_path)
            extent = (strip_window.col_off, strip_window.row_off, strip_window.width, strip_window.height)
            if extent != (strip.col_off, strip.row_off, strip.width, strip.height) or strip_window.decimation != (1, 1):
                raise ValueError(
                    f"write_raster_strips: strip {strip} came back as {extent} at decimation "
                    f"{strip_window.decimation} -- the window did not reach every read"
                )
            if not strip_metadata.equals(metadata_df):
                raise ValueError(
                    f"write_raster_strips: strip {strip} produced different bands from the full-extent plan"
                )
            df = strip_lf.select(band_names).collect(engine="streaming")
            _write_batches(dst, df, band_names, strip_window, origin, out_dtype, nodata)
            del df
        _describe_bands(dst, metadata_df)
    return True


#: Bytes per pixel per band `_strip_height` budgets for: a collected strip is held as
#: Polars columns of up to 8 bytes (`Float64`/`Int64`), whatever the file's own dtype.
_STRIP_BYTES_PER_BAND = 8


def _strip_height(window: ReadWindow, n_bands: int, output_block_height: int) -> int:
    """Rows per strip for `write_raster_strips`: as many as `config.RASTER_WRITE_STRIP_MB`
    holds at `n_bands` bands across `window`'s width, rounded down to a whole number of
    output block rows -- and of source block rows, whenever the budget covers at least
    one.

    The source block height is read off the full-extent read's `batch_layout`: a full
    read is batched a block row at a time (`io.raster.scan_raster`), so its first batch is
    one block tall. Strips start at the window's top, so whole source block rows keep
    every strip boundary on a block boundary. A band count too large for even one source
    block row in the budget gets strips shorter than a block, and some source blocks are
    decompressed more than once; the budget is what keeps the pod alive, so it wins.

    At least one output block row, whatever the budget, since a strip cannot be smaller.
    """
    budget_rows = config.RASTER_WRITE_STRIP_MB * 1024 * 1024 // (window.width * n_bands * _STRIP_BYTES_PER_BAND)
    source_block_height = window.batch_layout[0][3] if window.batch_layout else 1
    unit = math.lcm(source_block_height, output_block_height)
    if budget_rows < unit:
        unit = output_block_height
    return max(unit, budget_rows // unit * unit)


def _raster_profile(window: ReadWindow, count: int, dtype: str, nodata, profile_options: dict) -> dict:
    """The write profile both raster sinks open their file with: a GeoTIFF over `window`'s
    pixels, `**profile_options` layered on top (see `write_raster`)."""
    # `read_shape`, not `height`/`width`: those are the window's SOURCE extent, which is the
    # array's shape only when the read was undecimated. `window.transform` is already scaled
    # to match, so the written file is georeferenced correctly either way.
    read_height, read_width = window.read_shape
    profile = {
        "driver": "GTiff",
        "height": read_height,
        "width": read_width,
        "count": count,
        "dtype": dtype,
        "crs": window.crs,
        "transform": window.transform,
        "nodata": nodata,
    }
    profile.update(profile_options)
    return profile


def _write_batches(dst, df: pl.DataFrame, band_names, window: ReadWindow, origin, out_dtype, nodata) -> None:
    """Write `df` -- `window`'s read, collected in the frame's own row order -- into `dst`
    one read batch at a time, each into its own window of the file. `origin` is the
    `(col_off, row_off)` the file's top-left pixel has in `window`'s source pixels.

    Row order is the READ's emission order, not necessarily row-major (see
    `execution/frame.py`'s band-wide invariants): batch after batch, each one row-major over
    its own sub-rectangle, as `window.batch_layout` records -- which is why this writes
    batch by batch rather than sorting the frame into raster order first.
    """
    # An empty layout is one batch covering the whole read, in array units -- the same
    # fallback `row_major_grid_cell` takes, and the only layout a decimated read has.
    read_height, read_width = window.read_shape
    layout = window.batch_layout or ((window.col_off, window.row_off, read_width, read_height),)
    if df.height != sum(width * height for _c, _r, width, height in layout):
        raise ValueError(
            f"write_raster: the frame has {df.height} rows, but its window's batch layout covers "
            f"{sum(width * height for _c, _r, width, height in layout)} pixels"
        )
    origin_col, origin_row = origin
    offset = 0
    for col, row, width, height in layout:
        batch = df.slice(offset, width * height)
        offset += width * height
        data = np.stack(
            [batch.get_column(name).fill_null(nodata).to_numpy().astype(out_dtype, copy=False) for name in band_names]
        ).reshape(len(band_names), height, width)
        dst.write(data, window=Window(col - origin_col, row - origin_row, width, height))


def _describe_bands(dst, metadata_df: pl.DataFrame) -> None:
    """Label each band of `dst` with its own `metadata_df` row (`_band_descriptions`)."""
    for index, description in enumerate(_band_descriptions(metadata_df), start=1):
        if description:
            dst.set_band_description(index, description)


# ----------------------------------------------------------------------------------------
# png / jpg: a single map tile
# ----------------------------------------------------------------------------------------


def _encode_tile_image(color: np.ndarray, image_format) -> bytes:
    """Encode one `(tile_size, tile_size, 4)` RGBA array as `png`/`jpg` bytes.

    JPEG carries no alpha channel, so the alpha column is dropped rather than flattened
    against a background: a fully transparent RGBA pixel encodes as whatever RGB it was
    carrying. Which is a real difference wherever the legend declares a transparent band --
    a `No Data` band being the usual one (see `render_tile`) -- and is the cost of asking
    for `jpg` on a map whose nodata is meant to disappear, not something this can repair
    without inventing a background color the caller never named.
    (`Image.fromarray` infers RGB vs RGBA from the array's own shape; passing `mode=`
    explicitly is deprecated as of Pillow 11.)
    """
    image = Image.fromarray(color[:, :, :3] if image_format == "jpg" else color)
    with BytesIO() as memfile:
        image.save(memfile, format="JPEG" if image_format == "jpg" else "PNG")
        return memfile.getvalue()


def render_tile(
    band_wide: BandWideFrame,
    tile: TileRequest,
    image_format,
    *,
    color_map=None,
    variable_legend=None,
    nodata=DEFAULT_NODATA,
) -> bytes | None:
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
    nothing to show. A route turns that into a bodiless 204, exactly as `sink_vector_tile`'s
    `None` already was.

    This used to be a 404 carrying a blank encoded image (`empty_tile`, now deleted), on the
    argument that a 404 on an image URL should still decode as an image. The argument was
    about the BODY and never about the status code, and it is answered better by not sending
    a 404 at all: a 204 has no body to decode, MapLibre reads a zero-length tile response as
    a transparent texture natively, and a `jpg` miss stops being an opaque black square (the
    blank RGBA array lost its alpha channel on the way through `_encode_tile_image`).

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

    Whether a nodata pixel renders transparent is the LEGEND's answer, not this function's.
    `Legend.colorize_pixels` digitizes against the break points, and a value below the
    lowest break -- which `DEFAULT_NODATA` always is -- lands in the first bucket, so a
    legend whose first entry is a transparent `No Data` band (`lnd_type`, `hsi`,
    `lnd_conf`, ...) drops nodata out of the picture and one without such a band paints it
    in whatever that first color happens to be. Nothing here inspects or special-cases the
    sentinel. It is also why a MISS is a 204 rather than a rendered all-nodata tile: the
    two are different pictures, and only the caller's own legend would say what the second
    one should look like.
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
        .reshape(*window.read_shape)
    )
    # `window.transform` already describes these pixels -- `scan_raster` derived it for the
    # window it read, and scaled it to match if that read was decimated -- so this is
    # `warp_array_to_tile`, not `warp_window_to_tile`, which exists to derive that same
    # transform for a caller who has only the source's. That is also why a decimated tile
    # needs nothing here: the array is smaller and its transform is coarser by exactly the
    # same ratio, so the warp onto the tile grid is the identical call either way.
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
    return _encode_tile_image(legend.colorize_pixels(tile_data), image_format)


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
    204 for all of them. An MVT client treats a 204 as an empty tile natively, so there is
    nothing to encode; `render_tile`'s `None` now answers the same way, which is what closed
    the one place the two tile routes disagreed about a miss.

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
        cell: value for cell, value in zip(df["grid_cell"].to_list(), df["value"].to_list()) if value is not None
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
