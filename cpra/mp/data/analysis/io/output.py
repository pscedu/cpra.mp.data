"""Output rendering for a compiled analysis: how a finished frame is handed to a caller,
and what the response is declared to be.

None of this is op algebra. Deciding that a CSV rounds to a variable's `float_precision`,
that `sink_csv` is the streaming spelling of `write_csv`, that a `color_map=` string parses
into a `Legend`, or that a `grid_cell` selection turns a raster dataset's response into a
tabular one -- all of it is request LIFECYCLE, driven by the request's own options and the
catalog, and none of it depends on which ops ran or in what order.

Every function here is a plain free function taking exactly what it reads, so a caller
holds no compiled object to reach these through.

`resolve_legend`'s rejections are `request.errors.RequestError`s, mapped to 400 by
`api/boundary.py` like every other member of that family; nothing here raises an
`HTTPException`, since `cpra.mp.data.analysis` may not import FastAPI.
"""

import re

import polars as pl

from cpra.mp.data.analysis.io.raster_io import timing
from cpra.mp.data.analysis.request.errors import RequestError
from cpra.mp.data.analysis.request.types import DataFormat
from cpra.mp.data.legends import LEGENDS, Legend

# Polars' two spellings of each tabular writer: the lazy sink (streams, never materializes
# the whole frame) and the eager write (collects first). `api/tabular.py` picks between
# them per request via `collect_in_memory`.
#
# Keyed by `DataFormat`'s tabular members and no others. Both tables also carried an
# `arrow` -> `sink_ipc`/`write_ipc` entry, left over from a format vocabulary the request
# layer no longer has: `DataFormat` is `{csv, ndjson, parquet, tif}`, every route binds its
# `{format}` path segment straight to that enum, and `Output.format` is `DataFormat |
# TileFormat` -- so no request could ever name `arrow` and reach a writer for it. An entry
# no key can select is not a supported format, only one that reads like one.
SINK_METHODS = {
    "csv": "sink_csv",
    "ndjson": "sink_ndjson",
    "parquet": "sink_parquet",
}
WRITE_METHODS = {
    "csv": "write_csv",
    "ndjson": "write_ndjson",
    "parquet": "write_parquet",
}

#: The color spellings `Legend.colorize_pixels` can actually decode: an optional `#`, then
#: either six hex digits (`rrggbb`, which it pads to opaque) or eight (`rrggbbaa`).
#:
#: Checked here because `colorize_pixels` does not check at all -- it slices the string
#: into byte pairs and calls `int(pair, 16)`, so anything else raises a bare `ValueError`
#: deep inside a tile render, which surfaces as a 500 for what is plainly a malformed
#: request. Validating the custom map where it is PARSED makes it a `RequestError` -> 400
#: like the three rejections already in `resolve_legend`, and it only applies to the
#: custom branch: a named `LEGENDS` entry and a variable's own declared legend are this
#: package's own data, not caller input.
_CUSTOM_COLOR_RE = re.compile(r"#?(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\Z")


def tabular_kwargs(data_format, float_precision):
    """Round CSV output to the variable's `float_precision`, when it sets one.

    CSV only: it's the one writer polars gives a `float_precision`, and the only format
    here that renders floats as text at all -- the binary formats hand over the stored
    value and leave rounding to whatever reads them.

    This deliberately applies to every float column in the response, not just `value`.
    Nothing else in a tabular response is a float today (dimensions are integers,
    strings, or dates), and a bare `.round()` on `value` would be wrong anyway: it
    would change the number in the binary formats too, where the request asked for the
    stored value.
    """
    if data_format == DataFormat.csv and float_precision is not None:
        return {"float_precision": float_precision}
    return {}


@timing
def stream_tabular(materialize, out_path, data_format, *args, **kwargs):
    """Sink the frame to `out_path` without materializing it.

    `materialize` is the caller's own materialize callable, passed in rather than an
    `lf` so that the caller keeps ownership of how a failed collect is reported (see
    `execution.sink`'s own materialize closure, which lets `ColumnNotFoundError`/
    `FileNotFoundError` through to `api/boundary.py` untranslated).

    `*args`/`**kwargs` pass straight through to the underlying `sink_*` method -- this is
    how a caller's `compression`/`compression_level` (parquet) reach the Polars writer.
    Row-group sizing is not among them: nothing upstream resolves or forwards a row-group
    parameter today, so `sink_parquet` runs on its own default.
    """
    materialize(SINK_METHODS[data_format], out_path, *args, **kwargs)


@timing
def write_tabular(materialize, out_path, data_format, *args, **kwargs):
    """Collect the frame and write it to `out_path`. See `stream_tabular` for `materialize`
    and for what `*args`/`**kwargs` (and row-group sizing's absence from them) means here.
    """
    df = materialize()
    getattr(df, WRITE_METHODS[data_format])(out_path, *args, **kwargs)


def resolve_legend(color_map, variable_legend) -> Legend:
    """Resolve a request's `color_map` option to a `Legend`.

    A `color_map` containing `:` is a CUSTOM map, spelled
    `continuous:<value>,<color>,...` or `discrete:<color>,<value>,...` -- the two
    orderings differ because a continuous map is a list of (stop value, color) pairs while
    a discrete one is a list of (color, upper bound) pairs, whose last bound is open-ended
    and therefore appended as `None` here. Both halves of a custom map are validated: the
    colors against `_CUSTOM_COLOR_RE` and the values as floats, since a custom map is the
    only legend built from caller input and `Legend` itself checks neither.

    Anything else is a lookup: a named `LEGENDS` entry, else the variable's own declared
    legend (`variable_legend`, `None` when it declares none), else `LEGENDS["default"]`.
    """
    color_map = color_map or ""

    if ":" in color_map:
        map_type, color_str = color_map.split(":", 1)

        if map_type not in ["continuous", "discrete"]:
            raise RequestError(f"invalid color map type: {map_type}")

        color_strs = color_str.split(",")
        if map_type == "continuous":
            colors = color_strs[1::2]
            value_strs = color_strs[::2]
        else:
            colors = color_strs[::2]
            value_strs = color_strs[1::2]

        for color in colors:
            if not _CUSTOM_COLOR_RE.match(color):
                raise RequestError(f"invalid color map color: {color}")

        try:
            values = [float(val) for val in value_strs]
        except ValueError:
            raise RequestError("invalid color map values")

        # Add a null upper bound value to discrete color maps.
        if map_type == "discrete":
            values += [None]

        if len(values) != len(colors):
            raise RequestError("mismatched colors and values in color map")

        return Legend("custom", pl.DataFrame({"color": colors, "upper_bound": values}), type=map_type)

    return LEGENDS.get(color_map) or variable_legend or LEGENDS["default"]
