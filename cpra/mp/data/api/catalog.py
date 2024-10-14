import re
from cpra.mp.data.config import BASE_PATH
from cpra.mp.data.dimensions import DIMENSIONS
from fastapi import HTTPException

TIF_EXTENSION = re.compile(r"\.tif$")


class Catalog:
    """Data catalog query."""

    def __init__(self, dimensions):
        self.dimensions = dimensions
        self._choices = {}
        self._results = {}

    def _update(self, items):
        # Ignore items that don't follow our naming scheme.
        items[:] = [item for item in items if "=" in item]

        dimension_name = items[0].split("=")[0]
        dimension = DIMENSIONS.get(dimension_name)
        dtype = dimension.dtype.to_python()
        dimension_values = [dtype(d.split("=")[1]) for d in items]

        # Update choices with the dimension values.
        if dimension_name in self._choices:
            # Geographic unit and time unit are special cases, because the values don't
            # have to match. For these dimensions, we use OR instead of AND.
            if dimension_name in ("geographic_unit", "time_unit"):
                self._choices[dimension_name].extend(
                    [
                        val
                        for val in dimension_values
                        if val not in self._choices[dimension_name]
                    ]
                )
            else:
                # Calculate choices that are common to this branch and previous branches.
                common_choices = [
                    val
                    for val in self._choices[dimension_name]
                    if val in dimension_values
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

        # TODO: Add more sophisticated logic for selecting a default value
        # from the available choices.
        selected_values = [self._choices[dimension_name][0]]

        prop_value = getattr(self.dimensions, dimension_name)
        if prop_value:
            selected_values = sorted([dtype(val) for val in prop_value])
            missing_values = [
                val
                for val in selected_values
                if val not in self._choices[dimension_name]
            ]
            if missing_values:
                missing_str = ", ".join([str(val) for val in missing_values])
                msg = f"Dimension {dimension_name} is missing requested values: {missing_str}"
                raise HTTPException(status_code=404, detail=msg)

        self._results[dimension_name] = selected_values
        items[:] = [f"{dimension_name}={str(val)}" for val in selected_values]

    def execute(self):
        """Execute a query for data."""
        datatsets = []
        for root, dirs, files in BASE_PATH.walk():
            if dirs:
                self._update(dirs)
            elif files:
                if files[0].endswith(".tif"):
                    self._update([TIF_EXTENSION.sub("", file) for file in files])
                if files[0].endswith(".parquet"):
                    pass

        return {
            "choices": self._choices,
            "results": self._results,
        }
