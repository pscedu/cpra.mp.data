"""Catalog and frame-metadata lookups for a compiled analysis.

Two senses of "metadata" meet here, and both are request LIFECYCLE rather than op algebra:

* **Catalog** lookups -- resolving a dataset's `Variable` out of `cpra.mp.data.variables`,
  and the pre-read size estimate `api/__init__.py` uses to decide whether a tabular
  response is buffered or streamed. Neither reads the frame; both answer from the declared
  dimensions, the grid, and the structure CSVs.
* **Frame-shape** inspection -- `is_raster_metadata_table`, which asks whether a raster
  analysis's frame is still the per-band metadata table rather than a tabular result
  already derived from pixels.

None of it depends on which ops ran or in what order, and none of it needs an instance:
each function below takes exactly what it reads. `BaseAnalysis` keeps `variable` and
`estimate_data_size` as delegations, because `api/__init__.py`, `raster.py` and
`PairedAnalysis` reach for those two by duck typing; `_is_raster_metadata_table` had no
such caller and is gone from the class entirely.
"""

import re

from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.utils import get_input_data_frame, required_columns
from cpra.mp.data.variables import VARIABLES, Variable


def variable_for(dataset_dimensions) -> Variable:
    """The catalog `Variable` a dataset's declared dimensions select."""
    return VARIABLES[dataset_dimensions.variable]


def is_raster_metadata_table(lf, grid_type):
    """Whether `lf` still holds the raster band-metadata table (dimension columns and a
    `path` column), as opposed to a tabular result already derived from raster pixels (via
    grid_cell sampling or summarize_by).

    Unlike `output.resolve_output_type`, which reflects the dataset's final response shape
    and is stable for the whole request, this reflects `lf`'s shape at the point it is
    called, since summarize_by's raster-to-tabular conversion doesn't happen until the
    transform stage.
    """
    return grid_type == "raster" and "path" in lf.collect_schema().names()


def estimate_data_size(dataset_dimensions, filter_dimensions, grid, group_by_cols, *,
                       has_group_by, grid_cell_ids):
    """Estimate the size, in bytes, of data. This does not include dimension columns.

    Returns `None` whenever any factor is unknowable up front (a dimension with no
    enumerable value set, a grid with neither an id count nor a height/width, a variable
    whose dtype name carries no single bit width) -- the caller treats that as "cannot
    estimate", not as zero.

    `group_by_cols` and `has_group_by` come from the caller rather than from `Options`
    directly: resolving `group_by` into column names is validated op-side
    (`BaseAnalysis._get_group_by_columns`, which rejects `group_by` without
    `group_aggregate`), and this function must not re-run that validation as a side effect
    of an estimate.
    """
    required_cols = required_columns(
        get_input_data_frame(**dict(dataset_dimensions)), include_dataset=False, include_metadata=False
    )

    # Estimate rows.
    rows = 1
    for col in required_cols:
        # If there is a group by clause, only the group by columns contribute
        # to the number of output rows.
        if has_group_by and col not in group_by_cols:
            continue

        dimension = DIMENSIONS[col]

        value_count = None
        values = getattr(filter_dimensions, col)
        if values is not None:
            value_count = len(values)
        # Compare against None explicitly: dimension.values is a DataFrame, and
        # Polars raises on bool() rather than reporting emptiness.
        elif dimension.values is not None:
            value_count = len(dimension.values)
        elif col == "calendar_year":
            if dataset_dimensions.time_unit == "decadal":
                value_count = 6
            else:
                value_count = 52
        elif col == "calendar_day":
            if dataset_dimensions.time_unit == "monthly":
                value_count = 52 * 12
            else:
                value_count = 52 * 365

        if value_count is None:
            return None

        rows *= value_count

    # Estimate columns.
    columns = None
    if grid_cell_ids is not None:
        columns = len(grid_cell_ids)
    else:
        columns = getattr(grid, "id_count", None)

        if columns is None:
            height = getattr(grid, "height", None)
            width = getattr(grid, "width", None)
            if height and width:
                columns = height * width

    if columns is None:
        return None

    # Calculate bytes per value.
    variable = variable_for(dataset_dimensions)
    dtype_digits = re.findall(r"\d+", str(variable.dtype))
    if len(dtype_digits) != 1:
        return None

    return rows * columns * int(dtype_digits[0]) / 8
