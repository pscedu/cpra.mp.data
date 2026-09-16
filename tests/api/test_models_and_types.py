"""Tests for the Pydantic query models and the dimension-derived enum types."""

from datetime import date
from enum import Enum

import pytest

from cpra.mp.data.analysis.types import (
    AggregateFunction,
    CompressionAlgorithm,
    DataFormat,
    OutputType,
    TileFormat,
    VectorGrid,
    dimension_type,
)
from cpra.mp.data.api.models import (
    Compression,
    DataQuery,
    DatasetDimensions,
    FilterDimensions,
    Options,
    TileQuery,
)
from cpra.mp.data.dimensions import DATASET_DIMENSIONS, DIMENSIONS
from cpra.mp.data.grids import GRIDS


class TestDimensionType:
    def test_integer_dimensions_map_to_int(self):
        assert dimension_type("model_group_id") is int

    def test_float_dimensions_map_to_float(self):
        assert dimension_type("aep") is float

    def test_date_dimensions_map_to_date(self):
        assert dimension_type("calendar_day") is date

    def test_string_dimensions_with_values_become_enums(self):
        result = dimension_type("time_unit")
        assert issubclass(result, Enum)
        assert {member.value for member in result} == {"daily", "monthly", "annual", "decadal"}

    def test_string_dimensions_without_values_stay_strings(self):
        """region has no entries in dimension_values.csv, so there is no enum to build."""
        assert dimension_type("region") is str

    def test_string_dimensions_with_csv_values_become_enums(self):
        assert issubclass(dimension_type("model"), Enum)

    def test_enum_names_are_camel_cased(self):
        assert dimension_type("time_unit").__name__ == "TimeUnit"


class TestEnums:
    def test_vector_grid_lists_only_vector_grids(self):
        expected = {name for name, grid in GRIDS.items() if grid.grid_type == "vector"}
        assert {member.value for member in VectorGrid} == expected

    def test_aggregate_functions(self):
        # `count` is included as a valid `summarize_aggregate`/`group_by`-aggregate value
        # on both grid types (see the `AggregateFunction` docstring in types.py) even
        # though it is explicitly rejected as a raster `group_aggregate` -- that
        # rejection is enforced by `Dataset`, not by the enum's membership. `count_by_value`
        # is similarly a member here but rejected by `Dataset` as a `group_aggregate` on
        # either grid type -- see the comment on it in types.py.
        assert {m.value for m in AggregateFunction} == {
            "count", "count_by_value", "max", "mean", "min", "sum"
        }

    def test_output_types(self):
        assert {m.value for m in OutputType} == {"raster", "vector"}

    def test_tile_formats(self):
        assert {m.value for m in TileFormat} == {"jpg", "png"}

    def test_tif_is_a_data_format(self):
        assert DataFormat.tif in DataFormat


class TestDatasetDimensions:
    def test_fields_match_the_dataset_dimensions(self):
        assert set(DatasetDimensions.model_fields) == set(DATASET_DIMENSIONS)

    def test_enum_values_are_unwrapped_to_strings(self):
        """use_enum_values keeps downstream code working with plain strings."""
        dims = DatasetDimensions(variable="sal", grid="ecoregion_v001", time_unit="annual")
        assert dims.time_unit == "annual"
        assert isinstance(dims.time_unit, str)

    def test_dict_conversion_yields_plain_values(self):
        dims = DatasetDimensions(variable="sal", grid="ecoregion_v001", time_unit="annual")
        assert dict(dims) == {"variable": "sal", "grid": "ecoregion_v001", "time_unit": "annual"}


class TestFilterDimensions:
    def test_excludes_dataset_and_metadata_dimensions(self):
        fields = set(FilterDimensions.model_fields)
        assert not fields & set(DATASET_DIMENSIONS)
        assert "model" not in fields

    def test_includes_partition_and_nonpartition_dimensions(self):
        fields = set(FilterDimensions.model_fields)
        assert {"model_group_id", "scenario_id", "calendar_year", "calendar_day"} <= fields

    def test_fields_accept_lists(self):
        filters = FilterDimensions(model_group_id=[500, 520])
        assert filters.model_group_id == [500, 520]

    def test_unset_fields_default_to_none(self):
        assert FilterDimensions().model_group_id is None

    def test_every_filterable_dimension_is_present(self):
        expected = {d.name for d in DIMENSIONS.values() if d.type not in ("dataset", "metadata")}
        assert set(FilterDimensions.model_fields) == expected


class TestOptions:
    def test_defaults(self):
        options = Options()
        assert options.grid_cell is None
        assert options.group_aggregate is None
        assert options.calculate is None

    def test_aggregate_enum_is_unwrapped(self):
        assert Options(group_aggregate="sum").group_aggregate == "sum"

    def test_summarize_by_accepts_a_vector_grid(self):
        assert Options(summarize_by="hydro_compartment_v001").summarize_by == "hydro_compartment_v001"

    def test_summarize_by_rejects_a_raster_grid(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Options(summarize_by="morph_pixel_v001")


class TestQuerySplitting:
    """TileQuery and DataQuery arrive flat from the query string and are split apart."""

    def test_tile_query_splits_filters_from_options(self):
        # `group_by`/`group_aggregate`/`calculate`/... are `Options`-only fields, not
        # `TileQuery`'s (`TileQuery` inherits `DataOptions`, not `Options` --
        # see `TileQuery`'s docstring) -- so only a `DataOptions` field (`where`) is
        # exercised here; `query.options` still reports `Options` defaults for the
        # fields `TileQuery` itself doesn't carry.
        query = TileQuery(model_group_id=[500], where="calendar_year > 2035", grid_cell="1,2")
        assert query.filter_dimensions.model_group_id == [500]
        assert query.options.where == "calendar_year > 2035"
        assert query.options.grid_cell == "1,2"
        assert query.options.group_aggregate is None

    def test_filter_dimensions_drops_option_fields(self):
        query = TileQuery(model_group_id=[500], where="calendar_year > 2035")
        assert not hasattr(query.filter_dimensions, "where")

    def test_options_drops_filter_fields(self):
        query = TileQuery(model_group_id=[500], where="calendar_year > 2035")
        assert not hasattr(query.options, "model_group_id")

    def test_data_query_exposes_compression(self):
        query = DataQuery(compression_algorithm="gzip", compression_level=3)
        assert query.compression.compression_algorithm == CompressionAlgorithm.gzip
        assert query.compression.compression_level == 3

    def test_data_query_compression_defaults(self):
        assert DataQuery().compression.compression_algorithm == CompressionAlgorithm.zstd
        assert DataQuery().compression.compression_level == 6

    def test_compression_defaults(self):
        compression = Compression()
        assert compression.compression_algorithm == CompressionAlgorithm.zstd
        assert compression.compression_level == 6
