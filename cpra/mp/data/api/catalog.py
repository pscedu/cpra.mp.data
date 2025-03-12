import re
from math import floor

import polars as pl
from cpra.mp.data.config import BASE_PATH
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.utils import get_input_data_frame, required_columns
from fastapi import HTTPException

TIF_EXTENSION = re.compile(r"\.tif$")
RANGE_FUNCTIONS = {"calendar_day": pl.date_range, "calendar_year": pl.arange}
RANGE_KWARGS = {
    "daily": {"interval": "1d"},
    "monthly": {"interval": "1mo"},
    "annual": {"step": 1},
    "decadal": {"step": 10},
}


class CatalogQuery:
    """Data catalog query."""

    def __init__(self, dataset_dimensions, filter_dimensions):
        self.dataset_dimensions = dataset_dimensions
        self.filter_dimensions = filter_dimensions
        self._choices = {}
        self._results = {}
        self._exact = True

    def _update(self, dimension_name=None, dimension_values=None, items=None):
        # Ignore items that don't follow our naming scheme.
        if items is not None:
            items[:] = [item for item in items if "=" in item]
            if dimension_name is None:
                dimension_name = items[0].split("=")[0]

        dimension = DIMENSIONS.get(dimension_name)
        dtype = dimension.dtype.to_python()

        if items is not None:
            dimension_values = sorted([dtype(d.split("=")[1]) for d in items])

        # Update choices with the dimension values.
        if dimension_name in self._choices:
            # Calculate choices that are common to this branch and previous branches.
            common_choices = [
                val for val in self._choices[dimension_name] if val in dimension_values
            ]
            # If there are common choices, assign them as the choices for
            # this dimension. Otherwise, return an error.
            if common_choices:
                self._choices[dimension_name] = common_choices
            else:
                msg = f"No common values for dimension {dimension_name}"
                raise HTTPException(status_code=404, detail=msg)
        else:
            self._choices[dimension_name] = dimension_values

        selected_values = []
        prop_value = getattr(
            self.dataset_dimensions
            if dimension.type == "dataset"
            else self.filter_dimensions,
            dimension_name,
        )
        if prop_value:
            if not isinstance(prop_value, (list, tuple)):
                prop_value = (prop_value,)
            selected_values = sorted(
                [val for val in prop_value if val in self._choices[dimension_name]]
            )
            if len(selected_values) < len(prop_value):
                self._exact = False

        if not selected_values:
            # TODO: Add more sophisticated logic for selecting a default value
            # from the available choices.
            selected_values = [self._choices[dimension_name][0]]
            self._exact = False

        self._results[dimension_name] = selected_values

        # Updates items to prevent walking unnecessary directories.
        if items is not None:
            items[:] = [f"{dimension_name}={str(val)}" for val in selected_values]

    def _compress(self):
        """Compress date/time values into continuous [start, end] intervals."""
        for dimension_name, values in self._choices.items():
            if dimension_name in ("calendar_day", "calendar_year"):
                if not isinstance(values, pl.Series):
                    values = pl.Series(values)
                if dimension_name in ("calendar_day", "calendar_year"):
                    range_function = RANGE_FUNCTIONS[dimension_name]
                    range_kwargs = RANGE_KWARGS[self.dataset_dimensions.time_unit]
                    # Account for decadal data that don't start on an even decade.
                    # TODO: Handle monthly data that don't start on the first of the month.
                    range_start = (
                        floor(values.min() / 10) * 10
                        if self._choices.get("time_unit", None) == ["decadal"]
                        else values.min()
                    )
                    range_end = (
                        values.max()
                        if dimension_name == "calendar_day"
                        else values.max() + 1
                    )
                    groups = (
                        pl.DataFrame(
                            {
                                "value": values,
                                "seq": range_function(
                                    range_start,
                                    range_end,
                                    eager=True,
                                    **range_kwargs,
                                )[: len(values)],
                            }
                        )
                        .group_by(offset=(pl.col("value") - pl.col("seq")))
                        .agg(start=pl.col("value").min(), end=pl.col("value").max())
                        .sort(pl.col("start"))
                    )
                    if dimension_name == "calendar_day":
                        groups = groups.with_columns(
                            pl.col("start").cast(pl.String),
                            pl.col("end").cast(pl.String),
                        )
                        self._results[dimension_name] = [
                            str(v) for v in self._results[dimension_name]
                        ]

                    self._choices[dimension_name] = [
                        row["start"]
                        if row["start"] == row["end"]
                        else [row["start"], row["end"]]
                        for row in groups.iter_rows(named=True)
                    ]
                elif isinstance(values, pl.Series):
                    self._choices[dimension_name] = values.to_list()

    def execute(self):
        """Execute a query for data."""
        input_df = get_input_data_frame(
            **dict(self.dataset_dimensions), **dict(self.filter_dimensions)
        )
        required_cols = required_columns(input_df, include_metadata=False)

        parquet_paths = []
        for root, dirs, files in BASE_PATH.walk():
            if dirs:
                self._update(items=dirs)
            elif files:
                if files[0].endswith(".tif"):
                    self._update(items=[TIF_EXTENSION.sub("", file) for file in files])
                if files[0].endswith(".parquet"):
                    parquet_paths.extend([root / file for file in files])

        if parquet_paths:
            filter_cols = [col for col in required_cols if col not in self._results]
            df = pl.scan_parquet(parquet_paths).select(filter_cols).unique().collect()
            prev_col = None
            for col in filter_cols:
                # Filter the lazy frame by the previous column.
                if prev_col:
                    df = df.filter(pl.col(prev_col).is_in(self._results[prev_col]))

                self._update(
                    dimension_name=col,
                    dimension_values=df.select(col).unique().sort(col)[col].to_list(),
                )

                prev_col = col

        self._compress()
        return {
            "choices": self._choices,
            "results": self._results,
            "exact": self._exact,
        }
