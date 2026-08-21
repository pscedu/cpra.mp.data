"""Tests for the storage-unit to display-unit boundary.

No variable in the structure spreadsheet sets `display_scale_factor` on this branch, so
every helper here is currently an identity -- which is exactly what the first class below
pins, since a regression that started scaling values would otherwise be silent. The
scaling itself is exercised against a variable built for the purpose, so the machinery is
known to work the day a real one is stored in a finer unit than it is read in.
"""

import pytest

from cpra.mp.data.variables import VARIABLES, Variable
from cpra.mp.data.visualizations.display import (
    display_number_format,
    display_unit,
    display_value_expr,
    to_display,
)


@pytest.fixture
def scaled():
    """A variable stored in millimeters but shown in meters."""

    return Variable(
        name="test_scaled",
        label="Test Scaled",
        dtype="Int32",
        legend=None,
        unit="NAVD88 mm",
        display_unit="NAVD88 m",
        display_scale_factor=1000,
        display_number_format=".3f",
    )


class TestEveryRealVariableIsUnscaled:
    """The invariant that makes this branch's storage units and display units the same."""

    def test_no_variable_sets_a_scale_factor(self):
        assert [v.name for v in VARIABLES.values() if v.display_scale_factor] == []

    def test_no_variable_sets_a_display_unit(self):
        assert [v.name for v in VARIABLES.values() if v.display_unit] == []

    @pytest.mark.parametrize("name", ["elev", "inun", "vct"])
    def test_values_pass_through_untouched(self, name):
        """elev in particular: it is stored and shown in meters here, unlike on the
        long-format branch where it is stored as Int32 millimeters."""
        variable = VARIABLES[name]
        assert to_display(variable, 1.5) == 1.5
        assert display_value_expr(variable) == "datum.value"
        assert display_unit(variable) == variable.unit


class TestDisplayUnit:
    def test_uses_the_display_unit_when_it_differs_from_storage(self, scaled):
        assert display_unit(scaled) == "NAVD88 m"

    def test_falls_back_to_the_storage_unit(self):
        variable = VARIABLES["inun"]
        assert variable.display_unit is None
        assert display_unit(variable) == variable.unit


class TestDisplayNumberFormat:
    def test_returns_the_variables_format(self):
        assert display_number_format(VARIABLES["elev"]) == ".2f"

    def test_an_unset_format_is_empty_rather_than_none(self):
        """Altair serializes a `format=None` into the spec as an explicit null, which
        Vega-Lite rejects outright -- an empty string is how it spells "unformatted"."""
        variable = VARIABLES["vct"]
        assert variable.display_number_format is None
        assert display_number_format(variable) == ""


class TestToDisplay:
    def test_divides_by_the_scale_factor(self, scaled):
        assert to_display(scaled, 1500) == 1.5

    def test_passes_none_through(self, scaled):
        """Break points and legend limits are optional; a missing one stays missing rather
        than becoming a 0.0."""
        assert to_display(scaled, None) is None

    @pytest.mark.parametrize("value", [0, -1500])
    def test_scales_zero_and_negatives(self, scaled, value):
        """Values below the datum are ordinary here, and 0 must not be mistaken for "no
        value" by the None check."""
        assert to_display(scaled, value) == value / 1000


class TestDisplayValueExpr:
    def test_builds_a_vega_division(self, scaled):
        assert display_value_expr(scaled) == "datum.value / 1000"

    def test_scales_the_given_field(self, scaled):
        """Joined-in sources land in a field named after their variable, not `value`."""
        assert display_value_expr(scaled, "datum['elev']") == "datum['elev'] / 1000"

    def test_an_unscaled_variable_returns_the_field_untouched(self):
        """Returning the field verbatim is what lets callers detect "no conversion
        needed" and skip emitting a transform at all."""
        assert display_value_expr(VARIABLES["inun"]) == "datum.value"

    def test_no_variable_returns_the_field_untouched(self):
        """A custom visualization can name a data source after something that isn't a
        variable, and it must not be assumed to need scaling."""
        assert display_value_expr(None) == "datum.value"
