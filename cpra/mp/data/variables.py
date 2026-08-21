import polars as pl

from cpra.mp.data.legends import LEGENDS, Legend
from cpra.mp.data.structure import read_structure_csv


class Variable:
    def __init__(
        self,
        name,
        label,
        dtype,
        legend,
        geographic_units=None,
        time_units=None,
        required_dimensions=None,
        model=None,
        subroutine=None,
        unit=None,
        display_unit=None,
        display_scale_factor=None,
        display_number_format=None,
        float_precision=None,
        difference_legend_limit=None,
        max_z_error=None,
    ):
        self.name = name
        self.label = label
        self.dtype = getattr(pl, dtype)
        self.legend = legend if isinstance(legend, Legend) else (LEGENDS[legend] if legend else None)
        self.geographic_units = geographic_units
        self.time_units = time_units
        self.required_dimensions = required_dimensions
        self.model = model
        self.subroutine = subroutine
        # `unit` describes the values as *stored*; the display_* attributes describe how they
        # are shown, which differs whenever a variable is stored in a finer unit than it is
        # read in (elev: stored as Int32 millimeters, shown as meters). Nothing outside the
        # presentation layer should consult them -- see cpra.mp.data.visualizations.display,
        # which is the only place the conversion is applied.
        self.unit = unit
        self.display_unit = display_unit
        self.display_scale_factor = display_scale_factor
        self.display_number_format = display_number_format
        # Carried through from the structure spreadsheet; no consumer yet.
        self.float_precision = float_precision
        # In storage units, like a legend's break points -- see visualizations.map, which
        # colorizes raster pixels with it directly and only scales the legend it hands the
        # client for display.
        self.difference_legend_limit = difference_legend_limit
        # When set, raster writes for this variable default to LERC_ZSTD compression at this
        # error tolerance instead of ZSTD+predictor -- see cpra.mp.data.raster._cog_profile.
        # No variable sets it since elev moved to Int32 millimeters + ZSTD/predictor, so the
        # structure spreadsheet has no column for it; the write path still honors it.
        self.max_z_error = max_z_error


VARIABLES = {
    row["name"]: Variable(**row)
    for row in read_structure_csv("variables")
    .with_columns(pl.col("geographic_units", "time_units", "required_dimensions").str.split(", "))
    .iter_rows(named=True)
}
