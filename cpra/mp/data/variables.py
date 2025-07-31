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
    ):
        self.name = name
        self.label = label
        self.dtype = getattr(pl, dtype)
        self.legend = (
            legend
            if isinstance(legend, Legend)
            else (LEGENDS[legend] if legend else None)
        )
        self.geographic_units = geographic_units
        self.time_units = time_units
        self.required_dimensions = required_dimensions
        self.model = model
        self.subroutine = subroutine
        self.unit = unit


VARIABLES = {
    row["name"]: Variable(**row)
    for row in read_structure_csv("variables")
    .with_columns(
        pl.col("geographic_units", "time_units", "required_dimensions").str.split(", ")
    )
    .iter_rows(named=True)
}
