import polars as pl
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.structure import read_structure_csv
from cpra.mp.data.variables import VARIABLES


class Dimension:
    def __init__(
        self,
        name,
        label,
        type,
        dtype,
        required=True,
        values=None,
        min_value=None,
        max_value=None,
        null=False,
        example_value=None,
        scope=None,
    ):
        self.name = name
        self.label = label
        self.type = type
        self.dtype = getattr(pl, dtype)
        self.values = values
        self.min_value = self.cast(min_value)
        self.max_value = self.cast(max_value)
        self.null = null
        self.required = required
        self.example_value = self.cast(example_value)
        self.scope = scope

    @property
    def partition(self):
        """Partition expression."""
        if self.type == "nonpartition":
            return False
        if self.type == "partition":
            return True
        return (pl.col("time_unit") == "daily").any()

    @property
    def validation_expressions(self):
        exprs = {}
        if not self.null:
            exprs[f"{self.name}_is_not_null"] = pl.col(self.name).is_not_null()

        if self.values is not None:
            exprs[f"{self.name}_values"] = pl.col(self.name).is_in(self.values["value"])

        if self.min_value is not None:
            exprs[f"{self.name}_minimum_value"] = pl.col(self.name) >= self.min_value

        if self.max_value is not None:
            exprs[f"{self.name}_maximum_value"] = pl.col(self.name) <= self.max_value

        return exprs

    def cast(self, value):
        """Cast a value to this dimension's data type."""
        return (
            None if value is None else pl.select(pl.lit(value).cast(self.dtype)).item()
        )

    def validate_dtype(self, series):
        """Validate that a Polars series has the expected data type."""

        if series.dtype != self.dtype:
            msg = f"Series {series.name} has dtype {series.dtype!s}, expected {self.dtype!s}"
            raise ValidationError(msg)


def make_dimensions():
    dim_values_df = read_structure_csv("dimension_values")
    dimensions = []

    for row in (
        read_structure_csv("dimensions")
        .with_columns(pl.col("required").cast(pl.Boolean))
        .iter_rows(named=True)
    ):
        values = dim_values_df.filter(dimension=row["name"]).with_columns(
            pl.col("value").cast(getattr(pl, row["dtype"]))
        )
        if row["required"] is None:
            row["required"] = (
                pl.col("variable")
                .is_in(
                    [
                        v.name
                        for v in VARIABLES
                        if row["name"] in (v.required_dimensions or [])
                    ]
                )
                .any()
            )
        dimensions.append(Dimension(values=(values if len(values) else None), **row))

    return dimensions


DIMENSIONS = {d.name: d for d in make_dimensions()}
DATASET_DIMENSIONS = {d.name: d for d in DIMENSIONS.values() if d.type == "dataset"}

# Override the default required expressions for time dimensions.
DIMENSIONS["variable"].values = pl.DataFrame(
    {"value": list(VARIABLES.keys()), "label": [var.name for var in VARIABLES.values()]}
).with_columns(pl.lit("variable").alias("dimension"))
DIMENSIONS["calendar_year"].required = (
    pl.col("time_unit").is_in(["annual", "decadal"]).any()
)
DIMENSIONS["calendar_day"].required = (pl.col("time_unit") == "daily").any()
