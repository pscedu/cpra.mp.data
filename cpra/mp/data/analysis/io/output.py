"""Output rendering for a compiled analysis: how a finished frame is handed to a caller,
and what the response is declared to be.

None of this is op algebra. Deciding that a CSV rounds to a variable's `float_precision`,
that `sink_csv` is the streaming spelling of `write_csv`, that a `color_map=` string parses
into a `Legend`, or that a `grid_cell` selection turns a raster dataset's response into a
tabular one -- all of it is request LIFECYCLE, driven by `Options` and the catalog, and
none of it depends on which ops ran or in what order. It lived on `BaseAnalysis` only
because that class was also the thing an API route held a reference to.

Every function here is a plain free function taking exactly what it reads. `BaseAnalysis`
keeps same-named members, but purely as delegations: `api/tabular.py` and `api/__init__.py`
reach for `tabular_kwargs`/`stream_tabular`/`write_tabular`/`output_type`/`legend` on the
compiled object by duck typing (`PairedAnalysis` answers the same surface without
inheriting it), so the METHODS have to stay even though the LOGIC does not.

`AnalysisError` is imported inside the one function that raises it, the same local-import
idiom `io/raster_io.py` already uses: it is defined in `analysis/base.py`, which imports
this module at module scope, so a module-scope import back would be a cycle.
"""

import polars as pl

from cpra.mp.data.analysis.io.raster_io import timing
from cpra.mp.data.analysis.types import DataFormat, OutputType
from cpra.mp.data.legends import LEGENDS, Legend

# Polars' two spellings of each tabular writer: the lazy sink (streams, never materializes
# the whole frame) and the eager write (collects first). `api/tabular.py` picks between
# them per request via `collect_in_memory`.
SINK_METHODS = {
    "arrow": "sink_ipc",
    "csv": "sink_csv",
    "ndjson": "sink_ndjson",
    "parquet": "sink_parquet",
}
WRITE_METHODS = {
    "arrow": "write_ipc",
    "csv": "write_csv",
    "ndjson": "write_ndjson",
    "parquet": "write_parquet",
}


def tabular_kwargs(format, float_precision):
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
    if format == DataFormat.csv and float_precision is not None:
        return {"float_precision": float_precision}
    return {}


@timing
def stream_tabular(materialize, out_path, format, *args, **kwargs):
    """Sink the frame to `out_path` without materializing it.

    `materialize` is the caller's own materialize callable (`BaseAnalysis.materialize`),
    passed in rather than an `lf` because that method is also where the
    `ColumnNotFoundError`/`FileNotFoundError` -> `AnalysisError` mapping lives.
    """
    # TODO: Set parquet compression and row group parameters.
    materialize(SINK_METHODS[format], out_path, *args, **kwargs)


@timing
def write_tabular(materialize, out_path, format, *args, **kwargs):
    """Collect the frame and write it to `out_path`. See `stream_tabular` for `materialize`."""
    # TODO: Set parquet compression and row group parameters.
    df = materialize()
    getattr(df, WRITE_METHODS[format])(out_path, *args, **kwargs)


def resolve_legend(color_map, variable_legend) -> Legend:
    """Resolve a request's `color_map` option to a `Legend`.

    A `color_map` containing `:` is a CUSTOM map, spelled
    `continuous:<value>,<color>,...` or `discrete:<color>,<value>,...` -- the two
    orderings differ because a continuous map is a list of (stop value, color) pairs while
    a discrete one is a list of (color, upper bound) pairs, whose last bound is open-ended
    and therefore appended as `None` here.

    Anything else is a lookup: a named `LEGENDS` entry, else the variable's own declared
    legend (`variable_legend`, `None` when it declares none), else `LEGENDS["default"]`.
    """
    from cpra.mp.data.analysis.base import AnalysisError

    color_map = color_map or ""

    if ":" in color_map:
        map_type, color_str = color_map.split(":", 1)

        if map_type not in ["continuous", "discrete"]:
            raise AnalysisError(f"invalid color map type: {map_type}", status_code=400)

        color_strs = color_str.split(",")
        if map_type == "continuous":
            colors = color_strs[1::2]
            value_strs = color_strs[::2]
        else:
            colors = color_strs[::2]
            value_strs = color_strs[1::2]

        # TODO: Check that the color strings are valid.

        try:
            values = [float(val) for val in value_strs]
        except ValueError:
            raise AnalysisError("invalid color map values", status_code=400)

        # Add a null upper bound value to discrete color maps.
        if map_type == "discrete":
            values += [None]

        if len(values) != len(colors):
            raise AnalysisError("mismatched colors and values in color map", status_code=400)

        return Legend("custom", pl.DataFrame(dict(color=colors, upper_bound=values)), type=map_type)

    return LEGENDS.get(color_map) or variable_legend or LEGENDS["default"]


def resolve_output_type(grid_type, *, grid_cell, summarize_by) -> OutputType:
    """The output type for a request, given its dataset's grid type and the two options
    that can narrow a raster to a table.

    Not simply `grid_type`: a raster grid still answers as `vector` once `grid_cell`
    (pointwise sampling) or `summarize_by` (zonal summarize) narrows it to a tabular
    result, which is why `tiles.validate_raster_tile`/`tiles.validate_vector_tile` check
    this rather than which analysis class produced the frame.
    """
    if grid_type == "vector" or grid_cell or summarize_by:
        return OutputType.vector
    return OutputType.raster
