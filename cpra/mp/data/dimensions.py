import polars as pl

from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.grids import GRIDS
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
        if self.type in ("metadata", "nonpartition"):
            return False
        if self.type == "dataset":
            return True
        return (pl.col("time_unit") == "daily").any()

    @property
    def validation_expressions(self):
        exprs = {}
        if not self.null:
            exprs[f"{self.name}_is_not_null"] = pl.col(self.name).is_not_null()

        if self.values is not None:
            exprs[f"{self.name}_values"] = pl.col(self.name).is_in(
                self.values["value"].cast(self.dtype).to_list()
            )

        if self.min_value is not None:
            exprs[f"{self.name}_minimum_value"] = pl.col(self.name) >= self.min_value

        if self.max_value is not None:
            exprs[f"{self.name}_maximum_value"] = pl.col(self.name) <= self.max_value

        return exprs

    def cast(self, value):
        """Cast a value to this dimension's data type."""
        return None if value is None else pl.select(pl.lit(value).cast(self.dtype)).item()

    def validate_dtype(self, series):
        """Validate that a Polars series has the expected data type."""

        if series.dtype != self.dtype:
            msg = f"Series {series.name} has dtype {series.dtype!s}, expected {self.dtype!s}"
            raise ValidationError(msg)


def make_dimensions():
    dim_values_df = read_structure_csv("dimension_values")
    dimensions = []

    for row in read_structure_csv("dimensions").with_columns(pl.col("required").cast(pl.Boolean)).iter_rows(named=True):
        # Float32 dimensions are cast to Float64 so that canonical values like
        # "0.1" survive the round-trip intact for display. The storage dtype
        # (Float32) is preserved on self.dtype and used when building filter
        # and validation expressions.
        display_dtype = pl.Float64 if row["dtype"].startswith("Float") else getattr(pl, row["dtype"])
        values = dim_values_df.filter(dimension=row["name"]).with_columns(
            pl.col("value").cast(display_dtype)
        )
        if row["required"] is False:
            row["required"] = (
                pl.col("variable")
                .is_in([v.name for v in VARIABLES.values() if row["name"] in (v.required_dimensions or [])])
                .any()
            )
        dimensions.append(Dimension(values=(values if len(values) else None), **row))

    return dimensions


def values_df_from_dict(dimension, values_dict, attr="label"):
    """Generate a dimension values DataFrame from a dictionary of objects."""
    return (
        pl.DataFrame(
            {
                "value": list(values_dict.keys()),
                "label": [getattr(obj, attr) for obj in values_dict.values()],
            }
        )
        .with_columns(pl.lit(dimension).alias("dimension"))
        .select("dimension", "value", "label")
    )


DIMENSIONS = {d.name: d for d in make_dimensions()}
DATASET_DIMENSIONS = {d.name: d for d in DIMENSIONS.values() if d.type == "dataset"}
METADATA_DIMENSIONS = {d.name: d for d in DIMENSIONS.values() if d.type == "metadata"}

# Override the default required expressions for time dimensions.
DIMENSIONS["variable"].values = values_df_from_dict("variable", VARIABLES)
DIMENSIONS["grid"].values = values_df_from_dict("grid", GRIDS)
DIMENSIONS["calendar_year"].required = pl.col("time_unit").is_in(["annual", "decadal"]).any()
DIMENSIONS["calendar_day"].required = pl.col("time_unit").is_in(["daily", "monthly"]).any()
