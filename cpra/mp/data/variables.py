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
        nodata=None,
        difference_legend_limit=None,
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
        # read in. No variable is stored that way on this branch -- `display_scale_factor` is
        # unset throughout, so every conversion below is an identity -- but the presentation
        # layer reads them through one place either way, so setting a scale factor in the
        # structure spreadsheet is all it takes. See cpra.mp.data.visualizations.display.
        self.unit = unit
        self.display_unit = display_unit
        self.display_scale_factor = display_scale_factor
        self.display_number_format = display_number_format
        # Decimal places to write when a dataset is served as CSV, for variables whose
        # models report more precision than they actually resolve -- see
        # cpra.mp.data.api.sources.LazyFrameSource.tabular_kwargs. Unlike the display_*
        # attributes this is not a presentation hint the client may ignore: it changes the
        # bytes in the download, so the values a user gets are the ones the model means.
        self.float_precision = float_precision
        # The value this variable's rasters use for "no data", where it differs from the
        # -9999 the raster writer defaults to. Set from the structure spreadsheet; null for
        # every variable but `elev`, whose values are stored in millimetres.
        self.nodata = nodata
        self.difference_legend_limit = difference_legend_limit


VARIABLES = {
    row["name"]: Variable(**row)
    for row in read_structure_csv("variables")
    .with_columns(pl.col("geographic_units", "time_units", "required_dimensions").str.split(", "))
    .iter_rows(named=True)
}
