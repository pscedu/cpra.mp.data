"""Tests for the Dimension class and the derived DIMENSIONS registries."""

import datetime

import polars as pl
import pytest

from cpra.mp.data.dimensions import (
    DATASET_DIMENSIONS,
    DIMENSIONS,
    METADATA_DIMENSIONS,
    Dimension,
    values_df_from_dict,
)
from cpra.mp.data.exceptions import ValidationError


def make_dimension(**overrides):
    kwargs = dict(name="test_dim", label="Test", type="partition", dtype="Int32")
    kwargs.update(overrides)
    return Dimension(**kwargs)


class TestCast:
    @pytest.mark.parametrize(
        ("dtype", "value", "expected"),
        [
            ("Int32", "5", 5),
            ("Int8", 3, 3),
            ("Float32", "0.5", 0.5),
            ("String", 7, "7"),
            ("Date", "2030-01-01", datetime.date(2030, 1, 1)),
        ],
    )
    def test_cast_converts_to_dtype(self, dtype, value, expected):
        assert make_dimension(dtype=dtype).cast(value) == expected

    def test_cast_none_is_none(self):
        assert make_dimension().cast(None) is None

    def test_min_max_and_example_are_cast_on_construction(self):
        dim = make_dimension(min_value="1", max_value="9", example_value="4")
        assert (dim.min_value, dim.max_value, dim.example_value) == (1, 9, 4)


class TestValidateDtype:
    def test_matching_dtype_passes(self):
        make_dimension(dtype="Int32").validate_dtype(pl.Series("test_dim", [1], dtype=pl.Int32))

    def test_mismatched_dtype_raises(self):
        dim = make_dimension(dtype="Int32")
        with pytest.raises(ValidationError, match="expected Int32"):
            dim.validate_dtype(pl.Series("test_dim", [1], dtype=pl.Int64))


class TestValidationExpressions:
    def _evaluate(self, dim, values):
        df = pl.DataFrame({dim.name: pl.Series(values, dtype=dim.dtype)})
        return df.select(**dim.validation_expressions)

    def test_not_null_expression_present_by_default(self):
        dim = make_dimension()
        assert "test_dim_is_not_null" in dim.validation_expressions

    def test_not_null_expression_absent_when_nullable(self):
        dim = make_dimension(null=True)
        assert "test_dim_is_not_null" not in dim.validation_expressions

    def test_null_fails_validation(self):
        results = self._evaluate(make_dimension(), [None])
        assert not results["test_dim_is_not_null"].all()

    def test_min_and_max_bounds(self):
        dim = make_dimension(min_value=2, max_value=4)
        results = self._evaluate(dim, [1, 2, 4, 5])
        assert results["test_dim_minimum_value"].to_list() == [False, True, True, True]
        assert results["test_dim_maximum_value"].to_list() == [True, True, True, False]

    def test_bounds_absent_when_unset(self):
        exprs = make_dimension().validation_expressions
        assert "test_dim_minimum_value" not in exprs
        assert "test_dim_maximum_value" not in exprs

    def test_values_membership(self):
        values = pl.DataFrame({"value": [10, 20], "label": ["a", "b"]})
        dim = make_dimension(values=values)
        results = self._evaluate(dim, [10, 20, 30])
        assert results["test_dim_values"].to_list() == [True, True, False]


class TestPartitionProperty:
    def test_dataset_dimensions_always_partition(self):
        assert make_dimension(type="dataset").partition is True

    @pytest.mark.parametrize("dim_type", ["metadata", "nonpartition"])
    def test_metadata_and_nonpartition_never_partition(self, dim_type):
        assert make_dimension(type=dim_type).partition is False

    def test_partition_dimensions_depend_on_time_unit(self):
        """Partition-type dimensions become directories only for daily data."""
        expr = make_dimension(type="partition").partition
        daily = pl.DataFrame({"time_unit": ["daily"]}).select(result=expr)
        annual = pl.DataFrame({"time_unit": ["annual"]}).select(result=expr)
        assert daily["result"].item() is True
        assert annual["result"].item() is False


class TestRegistries:
    def test_dataset_dimensions_are_the_path_prefix(self):
        assert list(DATASET_DIMENSIONS) == ["variable", "grid", "time_unit"]

    def test_metadata_dimensions(self):
        assert list(METADATA_DIMENSIONS) == ["model"]

    def test_registries_are_subsets_of_dimensions(self):
        assert set(DATASET_DIMENSIONS) <= set(DIMENSIONS)
        assert set(METADATA_DIMENSIONS) <= set(DIMENSIONS)

    def test_variable_and_grid_values_come_from_their_registries(self):
        from cpra.mp.data.grids import GRIDS
        from cpra.mp.data.variables import VARIABLES

        assert set(DIMENSIONS["variable"].values["value"]) == set(VARIABLES)
        assert set(DIMENSIONS["grid"].values["value"]) == set(GRIDS)

    def test_time_dimension_requirements_are_overridden(self):
        """calendar_year is required for annual/decadal, calendar_day for daily/monthly."""
        year_expr = DIMENSIONS["calendar_year"].required
        day_expr = DIMENSIONS["calendar_day"].required

        annual = pl.DataFrame({"time_unit": ["annual"]})
        daily = pl.DataFrame({"time_unit": ["daily"]})

        assert annual.select(r=year_expr)["r"].item() is True
        assert annual.select(r=day_expr)["r"].item() is False
        assert daily.select(r=year_expr)["r"].item() is False
        assert daily.select(r=day_expr)["r"].item() is True

    def test_variable_driven_requirement(self):
        """percentile is required for variables that list it in required_dimensions."""
        expr = DIMENSIONS["percentile"].required
        assert pl.DataFrame({"variable": ["ssf"]}).select(r=expr)["r"].item() is True
        assert pl.DataFrame({"variable": ["sal"]}).select(r=expr)["r"].item() is False


class TestFloatDisplayDtype:
    """Float32 dimensions store values as Float64 for display.

    See the comment at dimensions.make_dimensions: the storage dtype stays Float32 for
    filtering and validation, but canonical values like 0.1 are kept as Float64 so they
    survive the round-trip to JSON intact instead of becoming 0.10000000149011612.
    """

    def test_storage_dtype_is_float32(self):
        assert DIMENSIONS["aep"].dtype == pl.Float32

    def test_display_values_are_float64(self):
        assert DIMENSIONS["aep"].values["value"].dtype == pl.Float64

    def test_canonical_value_survives_round_trip(self):
        values = DIMENSIONS["aep"].values["value"].to_list()
        assert 0.1 in values

    def test_float32_cast_of_canonical_value_is_lossy(self):
        """The reason the split exists: casting 0.1 to Float32 and back is not 0.1."""
        lossy = pl.Series([0.1], dtype=pl.Float32).cast(pl.Float64).item()
        assert lossy != 0.1


class TestValuesDfFromDict:
    def test_builds_dimension_value_frame(self):
        class Obj:
            def __init__(self, label):
                self.label = label

        df = values_df_from_dict("thing", {"a": Obj("A"), "b": Obj("B")})
        assert df.columns == ["dimension", "value", "label", "color"]
        assert df["dimension"].to_list() == ["thing", "thing"]
        assert df["value"].to_list() == ["a", "b"]
        assert df["label"].to_list() == ["A", "B"]

    def test_colors_are_null_but_present(self):
        """Values built from a dict have no colors, but they have to carry the column: the
        API selects it off every `Dimension.values` frame without asking which built it."""

        class Obj:
            label = "A"

        df = values_df_from_dict("thing", {"a": Obj()})
        assert df["color"].to_list() == [None]
        assert df["color"].dtype == pl.String

    def test_respects_attr_argument(self):
        class Obj:
            name = "n"
            label = "L"

        df = values_df_from_dict("thing", {"a": Obj()}, attr="name")
        assert df["label"].to_list() == ["n"]
