"""Tests for the storage-unit to display-unit boundary.

elev is stored as Int32 millimeters but read as meters, and it is the only variable that
differs -- which is exactly what makes this worth pinning: every other variable has to come
through the same helpers unchanged, or an untested identity case silently rescales the whole
dataset the day a second variable gets a `display_scale_factor`.
"""

import pytest

from cpra.mp.data.variables import VARIABLES
from cpra.mp.data.visualizations.display import (
    display_number_format,
    display_unit,
    display_value_expr,
    to_display,
)

# Stored as Int32 millimeters, shown as meters -- the only variable that is scaled.
SCALED = VARIABLES["elev"]
# Stored in the unit it is shown in, like every other variable.
UNSCALED = VARIABLES["inun"]


class TestDisplayUnit:
    def test_uses_the_display_unit_when_it_differs_from_storage(self):
        assert SCALED.unit == "NAVD88 mm"
        assert display_unit(SCALED) == "NAVD88 m"

    def test_falls_back_to_the_storage_unit(self):
        assert UNSCALED.display_unit is None
        assert display_unit(UNSCALED) == UNSCALED.unit


class TestDisplayNumberFormat:
    def test_returns_the_variables_format(self):
        assert display_number_format(SCALED) == ".3f"

    def test_an_unset_format_is_empty_rather_than_none(self):
        """Altair serializes a `format=None` into the spec as an explicit null, which
        Vega-Lite rejects outright -- an empty string is how it spells "unformatted"."""
        variable = VARIABLES["vct"]
        assert variable.display_number_format is None
        assert display_number_format(variable) == ""


class TestToDisplay:
    def test_divides_by_the_scale_factor(self):
        assert to_display(SCALED, 1500) == 1.5

    def test_leaves_an_unscaled_variables_value_alone(self):
        assert to_display(UNSCALED, 1.5) == 1.5

    def test_passes_none_through(self):
        """Break points and legend limits are optional; a missing one stays missing rather
        than becoming a 0.0."""
        assert to_display(SCALED, None) is None

    @pytest.mark.parametrize("value", [0, -1500])
    def test_scales_zero_and_negatives(self, value):
        """Elevations below the datum are ordinary values here, and 0 must not be
        mistaken for "no value" by the None check."""
        assert to_display(SCALED, value) == value / 1000


class TestDisplayValueExpr:
    def test_builds_a_vega_division(self):
        assert display_value_expr(SCALED) == "datum.value / 1000"

    def test_scales_the_given_field(self):
        """Joined-in sources land in a field named after their variable, not `value`."""
        assert display_value_expr(SCALED, "datum['elev']") == "datum['elev'] / 1000"

    def test_an_unscaled_variable_returns_the_field_untouched(self):
        """Returning the field verbatim is what lets callers detect "no conversion
        needed" and skip emitting a transform at all."""
        assert display_value_expr(UNSCALED) == "datum.value"

    def test_no_variable_returns_the_field_untouched(self):
        """A custom visualization can name a data source after something that isn't a
        variable, and it must not be assumed to need scaling."""
        assert display_value_expr(None) == "datum.value"
