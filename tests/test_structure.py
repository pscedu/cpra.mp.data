"""Consistency checks on the structure CSVs.

The five CSVs under ``cpra/mp/data/structure/`` are the de-facto schema for the whole
library: dimension validation, partition layout, path globbing, and the API's response
models are all derived from them at import time. A typo in a legend name or a geographic
unit produces no error until some unrelated read fails, so these tests check the
cross-references directly.
"""

import re

import polars as pl
import pytest

from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.legends import LEGENDS
from cpra.mp.data.structure import read_structure_csv
from cpra.mp.data.variables import VARIABLES

# validate_data_frame strips a trailing version suffix before matching a grid name
# against a variable's geographic_units (see utils.validate_data_frame).
GRID_VERSION_SUFFIX = re.compile(r"_v[0-9.]+")

TIME_UNITS = {"daily", "monthly", "annual", "decadal"}


@pytest.mark.parametrize("name", ["dimensions", "dimension_values", "grids", "legends", "variables"])
def test_structure_csv_is_readable(name):
    assert len(read_structure_csv(name)) > 0


class TestVariables:
    def test_legends_resolve(self):
        missing = {
            var.name: row["legend"]
            for var, row in zip(VARIABLES.values(), read_structure_csv("variables").iter_rows(named=True))
            if row["legend"] and row["legend"] not in LEGENDS
        }
        assert missing == {}

    def test_geographic_units_match_a_grid(self):
        """Every geographic unit must match some grid once the _vNNN suffix is stripped."""
        grid_units = {GRID_VERSION_SUFFIX.sub("", name) for name in GRIDS}
        offenders = {
            var.name: sorted(set(var.geographic_units) - grid_units)
            for var in VARIABLES.values()
            if var.geographic_units and set(var.geographic_units) - grid_units
        }
        assert offenders == {}

    def test_time_units_are_known(self):
        offenders = {
            var.name: sorted(set(var.time_units) - TIME_UNITS)
            for var in VARIABLES.values()
            if var.time_units and set(var.time_units) - TIME_UNITS
        }
        assert offenders == {}

    def test_required_dimensions_exist(self):
        offenders = {
            var.name: sorted(set(var.required_dimensions) - set(DIMENSIONS))
            for var in VARIABLES.values()
            if var.required_dimensions and set(var.required_dimensions) - set(DIMENSIONS)
        }
        assert offenders == {}

    def test_dtypes_are_polars_dtypes(self):
        for var in VARIABLES.values():
            assert isinstance(var.dtype, pl.datatypes.classes.DataTypeClass), var.name

    def test_float_precision_is_set_only_for_salinity(self):
        """Salinity is reported to a tenth of a ppt; nothing else is rounded on the way out."""
        assert {var.name: var.float_precision for var in VARIABLES.values() if var.float_precision is not None} == {
            "sal": 1
        }

    def test_float_precision_is_only_set_for_float_variables(self):
        """Rounding decimal places out of an integer variable would be meaningless -- polars
        applies `float_precision` to float columns only, so it would also be a silent no-op."""
        offenders = [
            var.name
            for var in VARIABLES.values()
            if var.float_precision is not None and not str(var.dtype).startswith("Float")
        ]
        assert offenders == []


class TestDimensionsCsv:
    def test_dtypes_are_polars_dtypes(self):
        for dim in DIMENSIONS.values():
            assert isinstance(dim.dtype, pl.datatypes.classes.DataTypeClass), dim.name

    def test_types_are_known(self):
        known = {"dataset", "partition", "nonpartition", "metadata", "tag"}
        assert {dim.type for dim in DIMENSIONS.values()} <= known

    def test_dimension_values_reference_real_dimensions(self):
        # variable and grid are populated from VARIABLES/GRIDS rather than the CSV.
        csv_dimensions = set(read_structure_csv("dimension_values")["dimension"].unique())
        assert csv_dimensions <= set(DIMENSIONS)

    @pytest.mark.parametrize(
        "dim_name", [name for name, dim in DIMENSIONS.items() if dim.example_value is not None]
    )
    def test_example_values_satisfy_their_own_constraints(self, dim_name):
        """A dimension's documented example must pass that dimension's own validation."""
        dim = DIMENSIONS[dim_name]
        df = pl.DataFrame({dim.name: pl.Series([dim.example_value], dtype=dim.dtype)})
        results = df.select(**dim.validation_expressions)
        assert all(series.all() for series in results), f"{dim.name}={dim.example_value}"


# The colors are the vegetation model's own map symbology, and `SpeciesStackedBar` reads
# both them and their row order straight out of this CSV -- so an edit here is an edit to
# that chart, and these are the invariants it relies on.
COLORED_DIMENSIONS = sorted(
    read_structure_csv("dimension_values").drop_nulls("color")["dimension"].unique()
)


class TestDimensionValueColors:
    def test_only_the_species_are_colored(self):
        """Not a rule, just the current fact -- kept here so that coloring a second
        dimension is a deliberate act that comes past these tests."""
        assert COLORED_DIMENSIONS == ["veg_species_code"]

    @pytest.mark.parametrize("dim_name", COLORED_DIMENSIONS)
    def test_every_value_is_colored_or_none_is(self, dim_name):
        """A partly-colored dimension is the bad middle: `SpeciesStackedBar` looks a color up
        per row, and a null one is a segment Vega draws in its own default rather than an
        error anybody sees."""
        assert DIMENSIONS[dim_name].values["color"].null_count() == 0

    @pytest.mark.parametrize("dim_name", COLORED_DIMENSIONS)
    def test_colors_are_hex(self, dim_name):
        pattern = re.compile(r"^#[0-9a-f]{6}([0-9a-f]{2})?$")
        for color in DIMENSIONS[dim_name].values["color"]:
            assert pattern.match(color), f"{dim_name}: {color}"

    @pytest.mark.parametrize("dim_name", COLORED_DIMENSIONS)
    def test_colors_are_unique(self, dim_name):
        """`SpeciesStackedBar` hands Vega the colors in the data and lets it pair them with a
        domain it infers from the rows that survived filtering: two values sharing a color
        collapse into one entry of that range and shift every color after it."""
        colors = DIMENSIONS[dim_name].values["color"]
        assert colors.n_unique() == len(colors)

    @pytest.mark.parametrize("dim_name", COLORED_DIMENSIONS)
    def test_values_are_unique(self, dim_name):
        """The chart joins on the value, so a repeat leaves which color wins to the lookup."""
        values = DIMENSIONS[dim_name].values["value"]
        assert values.n_unique() == len(values)

    @pytest.mark.parametrize("dim_name", COLORED_DIMENSIONS)
    def test_every_value_has_a_label(self, dim_name):
        """The label is the legend entry and the tooltip row, so a null one is a blank
        line in both."""
        assert DIMENSIONS[dim_name].values["label"].null_count() == 0


class TestGridsCsv:
    def test_grid_types_are_known(self):
        assert {grid.grid_type for grid in GRIDS.values()} == {"vector", "raster"}

    def test_raster_grids_have_complete_geometry(self):
        for grid in GRIDS.values():
            if grid.grid_type != "raster":
                continue
            assert len(grid.bounds) == 4, grid.name
            assert grid.height > 0 and grid.width > 0, grid.name
            left, bottom, right, top = grid.bounds
            assert left < right and bottom < top, grid.name

    def test_vector_grid_id_bounds_are_consistent(self):
        for grid in GRIDS.values():
            if grid.grid_type != "vector" or grid.id_min is None:
                continue
            assert grid.id_min <= grid.id_max, grid.name
            if grid.id_count is not None:
                assert grid.id_count == grid.id_max - grid.id_min + 1, grid.name


# Legend.colorize_pixels drops null upper bounds when building `breaks` but keeps every
# row when building `colors`, so np.digitize's 0..len(breaks) output only indexes the
# colors correctly when there is exactly one null, at the end.
class TestLegendsCsv:
    @pytest.mark.parametrize("legend_name", sorted(LEGENDS))
    def test_break_points_are_ascending_with_one_open_top(self, legend_name):
        bounds = LEGENDS[legend_name].break_points["upper_bound"].to_list()
        finite = [b for b in bounds if b is not None]
        assert finite == sorted(finite)
        # The final class is open-ended, so exactly one null upper bound is expected.
        assert bounds.count(None) == 1
        assert bounds[-1] is None

    @pytest.mark.parametrize("legend_name", sorted(LEGENDS))
    def test_color_count_matches_digitize_range(self, legend_name):
        """colorize_pixels indexes colors by np.digitize, which yields 0..len(breaks)."""
        legend = LEGENDS[legend_name]
        bounds = legend.break_points["upper_bound"].to_list()
        n_breaks = len([b for b in bounds if b is not None])
        assert len(legend.break_points["color"]) == n_breaks + 1

    def test_colors_are_hex(self):
        pattern = re.compile(r"^#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$")
        for legend in LEGENDS.values():
            for color in legend.break_points["color"]:
                assert pattern.match(color), f"{legend.name}: {color}"


class TestLegendSort:
    @pytest.mark.parametrize("legend_name", sorted(LEGENDS))
    def test_sort_is_all_or_nothing(self, legend_name):
        """`legends.csv`'s `sort` orders the map key; a partly-filled one would push the
        unsorted rows to the end of it."""
        sort = LEGENDS[legend_name].break_points["sort"]
        assert sort.null_count() in (0, len(sort))
