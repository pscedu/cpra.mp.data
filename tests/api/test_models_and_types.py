"""Tests for the Pydantic query models and the dimension-derived enum types."""

from datetime import date
from enum import Enum

from cpra.mp.data.analysis.request.types import (
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
    DataOptions,
    DataQuery,
    DatasetDimensions,
    FilterDimensions,
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
        # Membership here is the enum's whole job: it is the closed vocabulary a
        # `project`/`group` step may NAME, not a statement that every member is legal in
        # every step. `count_by_value` is the case that makes the distinction visible --
        # a member of this enum, accepted by `project`, and rejected by `group`, because
        # it collapses to one row per (group, distinct value) and only `project` has
        # somewhere to put the extra rows. That rejection lives in
        # `request/aggregates.py::validate_group_aggregate`, not in this set.
        assert {m.value for m in AggregateFunction} == {"count", "count_by_value", "max", "mean", "min", "sum"}

    def test_output_types(self):
        assert {m.value for m in OutputType} == {"raster", "vector"}

    def test_tile_formats(self):
        assert {m.value for m in TileFormat} == {"jpg", "pbf", "png"}

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


class TestDataOptions:
    """`DataOptions` is the WHOLE options vocabulary a flat request has. The five fields
    below are it -- every parameter that used to describe how a request TRANSFORMED its
    data (`group_by`/`group_aggregate`/`summarize_*`/`calculate`) is a pipeline step now,
    and the `Options` superset that carried them has been deleted rather than kept as a
    shape nothing reads. `TestRetiredParams` in `test_endpoints_data.py` pins the 400s
    those names get; this pins that nothing here quietly grew one back.
    """

    def test_defaults(self):
        options = DataOptions()
        assert options.grid_cell is None
        assert options.where is None

    def test_the_field_set_is_exactly_the_five_data_access_options(self):
        assert set(DataOptions.model_fields) == {"grid_cell", "where", "sort", "select", "color_map"}


class TestDataQuery:
    """`DataQuery` is one flat model now, not a pair of halves waiting to be split.

    The `filter_dimensions`/`options`/`compression` splitter properties are gone with
    `api.data_query.build_data_query`, their only caller, and `TileQuery` is gone with the
    split tile routes that named it -- `partition_query(allow_compression=False)` rejects a
    compression parameter on a tile outright rather than merely declining to bind it. What
    the model is still for is binding on `VisualizationQuery` and being SUBTRACTED to name
    the render-option category, so what these pin is the field set, not a round trip.
    """

    def test_it_is_the_union_of_its_three_bases(self):
        fields = set(DataQuery.model_fields)
        assert set(DataOptions.model_fields) <= fields
        assert set(FilterDimensions.model_fields) <= fields
        assert set(Compression.model_fields) <= fields

    def test_a_filter_and_an_option_bind_side_by_side(self):
        query = DataQuery(model_group_id=[500], where="calendar_year > 2035", grid_cell="1,2")
        assert query.model_group_id == [500]
        assert query.where == "calendar_year > 2035"
        assert query.grid_cell == "1,2"

    def test_compression_binds_flat(self):
        query = DataQuery(compression_algorithm="gzip", compression_level=3)
        assert query.compression_algorithm == CompressionAlgorithm.gzip.value
        assert query.compression_level == 3

    def test_compression_defaults_flat(self):
        assert DataQuery().compression_algorithm == CompressionAlgorithm.zstd.value
        assert DataQuery().compression_level == 6

    def test_compression_defaults(self):
        compression = Compression()
        assert compression.compression_algorithm == CompressionAlgorithm.zstd
        assert compression.compression_level == 6
