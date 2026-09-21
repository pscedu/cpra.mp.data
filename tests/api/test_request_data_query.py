"""Tests for `cpra.mp.data.analysis.request.data_query.build_data_query_pipeline` -- the
plain-data-query -> `Pipeline` builder the flat `/v1/datasets/...` routes will delegate to.

Written and exercised in isolation, before any route is wired to it: every assertion here
is against the returned `Pipeline`'s shape/`where` string directly, never through the
ASGI app, so a mistake in the builder itself is never masked by (or blamed on) routing.
No file is ever opened -- `DatasetDimensions`/`FilterDimensions`/`DataOptions` are plain
pydantic models and `build_data_query_pipeline` does no I/O -- so these tests need no
`base_path`/`tmp_path` fixture at all, unlike most of this package's other tests.

Real variables/grids/dimensions from `cpra/mp/data/structure/` throughout, following
`test_request_pipeline.py`'s and `test_request_ops.py`'s own convention.
"""

from datetime import date

import pytest

from cpra.mp.data.analysis.parameters import DataOptions, DatasetDimensions, FilterDimensions
from cpra.mp.data.analysis.request.data_query import build_data_query_pipeline
from cpra.mp.data.analysis.request.errors import RequestError
from cpra.mp.data.analysis.request.expressions import PredicateKind, classify_where_kinds
from cpra.mp.data.analysis.request.ops import validate_shapes
from cpra.mp.data.analysis.request.pipeline import FilterStep, Pipeline
from cpra.mp.data.grids import GRIDS

VECTOR_GRID = "ecoregion_v001"
RASTER_GRID = "veg_grid_cell_v001"

assert GRIDS[VECTOR_GRID].grid_type == "vector"
assert GRIDS[RASTER_GRID].grid_type == "raster"


def _dataset_dimensions(**overrides):
    defaults = {"variable": "ssf", "time_unit": "annual", "grid": VECTOR_GRID}
    defaults.update(overrides)
    return DatasetDimensions(**defaults)


class TestBareTriple:
    """The representative case: a dataset triple, a couple of dimension filters, a
    `grid_cell`, a caller `where`, and every `Output` field -- one call, everything
    checked at once.
    """

    def test_builds_one_input_one_filter_step(self):
        dataset_dimensions = _dataset_dimensions()
        filter_dimensions = FilterDimensions(scenario_id=[7, 8], calendar_day=[date(2030, 1, 1)])
        options = DataOptions(
            grid_cell="1,2",
            where="value > 3 OR value < 0",
            sort="calendar_year",
            select="calendar_year,value",
            color_map="viridis",
        )

        pipeline = build_data_query_pipeline(dataset_dimensions, filter_dimensions, options, "csv")

        assert isinstance(pipeline, Pipeline)
        assert set(pipeline.inputs) == {"data"}
        assert pipeline.inputs["data"].variable == "ssf"
        assert pipeline.inputs["data"].time_unit == "annual"
        assert pipeline.inputs["data"].grid == VECTOR_GRID

        assert len(pipeline.steps) == 1
        (step,) = pipeline.steps
        assert isinstance(step, FilterStep)
        # The FIRST step must name its input explicitly -- see `validate_pipeline`'s
        # chaining rule -- not rely on implicit chaining, since it has nothing to chain
        # from.
        assert step.input == "data"
        assert step.where == (
            "calendar_day = DATE '2030-01-01' AND scenario_id IN (7, 8) AND "
            "grid_cell IN (1, 2) AND (value > 3 OR value < 0)"
        )

        assert pipeline.output.format == "csv"
        assert pipeline.output.sort == "calendar_year"
        assert pipeline.output.select == "calendar_year,value"
        assert pipeline.output.color_map == "viridis"

    def test_result_passes_shape_validation_too(self):
        """The builder only runs `validate_pipeline`'s structural checks; confirm the
        result also survives `ops.validate_shapes` (the shape-dependent half), since a
        route will run that immediately afterward and this builder exists specifically to
        feed it.
        """
        dataset_dimensions = _dataset_dimensions()
        filter_dimensions = FilterDimensions(scenario_id=[7])
        options = DataOptions()
        pipeline = build_data_query_pipeline(dataset_dimensions, filter_dimensions, options, "csv")
        validate_shapes(pipeline)  # must not raise


class TestNoFilters:
    def test_zero_filters_means_zero_steps(self):
        """A plain triple with no dimension filters, no `grid_cell`, no `where` -- the
        zero-step pipeline `validate_pipeline` allows for exactly-one-input requests.
        """
        pipeline = build_data_query_pipeline(_dataset_dimensions(), FilterDimensions(), DataOptions(), "csv")
        assert pipeline.steps == []


class TestDimensionQuoting:
    def test_int_dimension_is_unquoted(self):
        pipeline = build_data_query_pipeline(
            _dataset_dimensions(), FilterDimensions(scenario_id=[7]), DataOptions(), "csv"
        )
        assert pipeline.steps[0].where == "scenario_id = 7"

    def test_float_dimension_is_unquoted(self):
        pipeline = build_data_query_pipeline(_dataset_dimensions(), FilterDimensions(aep=[2.0]), DataOptions(), "csv")
        assert pipeline.steps[0].where == "aep = 2.0"

    def test_string_dimension_is_quoted(self):
        pipeline = build_data_query_pipeline(
            _dataset_dimensions(),
            FilterDimensions(infrastructure_category=["public_schools"]),
            DataOptions(),
            "csv",
        )
        assert pipeline.steps[0].where == "infrastructure_category = 'public_schools'"

    def test_string_dimension_embedded_quote_is_escaped(self):
        """`vegetation_coverage_type` (unlike `infrastructure_category`) has no enumerated
        value set (`dim.values is None`), so `dimension_type` leaves it a plain `str`
        rather than building a closed `Enum` -- the one free-text dimension available to
        exercise quote-escaping against arbitrary caller input."""
        pipeline = build_data_query_pipeline(
            _dataset_dimensions(),
            FilterDimensions(vegetation_coverage_type=["o'brien"]),
            DataOptions(),
            "csv",
        )
        assert pipeline.steps[0].where == "vegetation_coverage_type = 'o''brien'"

    def test_date_dimension_uses_sql_date_literal(self):
        pipeline = build_data_query_pipeline(
            _dataset_dimensions(),
            FilterDimensions(calendar_day=[date(2030, 6, 15)]),
            DataOptions(),
            "csv",
        )
        assert pipeline.steps[0].where == "calendar_day = DATE '2030-06-15'"

    def test_multiple_values_use_in(self):
        pipeline = build_data_query_pipeline(
            _dataset_dimensions(), FilterDimensions(scenario_id=[7, 8, 9]), DataOptions(), "csv"
        )
        assert pipeline.steps[0].where == "scenario_id IN (7, 8, 9)"

    def test_multiple_dimensions_sort_alphabetically_by_name(self):
        """Composed order does not depend on which order the caller set the fields in --
        `_dimension_conjuncts` iterates `sorted(...)`, so `aep` (a) precedes `scenario_id`
        (s) regardless of construction order.
        """
        filter_dimensions = FilterDimensions(scenario_id=[7], aep=[2.0])
        pipeline = build_data_query_pipeline(_dataset_dimensions(), filter_dimensions, DataOptions(), "csv")
        assert pipeline.steps[0].where == "aep = 2.0 AND scenario_id = 7"


class TestCallerWhere:
    def test_where_alone_is_parenthesized(self):
        pipeline = build_data_query_pipeline(
            _dataset_dimensions(), FilterDimensions(), DataOptions(where="value > 3 OR value < 0"), "csv"
        )
        assert pipeline.steps[0].where == "(value > 3 OR value < 0)"

    def test_where_composes_with_dimension_filters_by_and(self):
        """The load-bearing case this parenthesization exists for: without it, `scenario_id
        = 7 AND value > 3 OR value < 0` would test `value < 0` unconditionally (`OR` binds
        looser than `AND`); parenthesizing keeps the caller's `OR` scoped to what they
        wrote.
        """
        pipeline = build_data_query_pipeline(
            _dataset_dimensions(),
            FilterDimensions(scenario_id=[7]),
            DataOptions(where="value > 3 OR value < 0"),
            "csv",
        )
        assert pipeline.steps[0].where == "scenario_id = 7 AND (value > 3 OR value < 0)"

    def test_a_multi_conjunct_where_is_parenthesized_per_conjunct(self):
        """The other half of the rule, and the one a whole-string wrap would break: the
        caller's own top-level `AND` seam survives, because `expressions.
        classify_where_kinds` sorts predicates one top-level conjunct at a time.
        """
        pipeline = build_data_query_pipeline(
            _dataset_dimensions(),
            FilterDimensions(),
            DataOptions(where="calendar_year > 2035 AND value > 2040.16"),
            "csv",
        )
        assert pipeline.steps[0].where == "(calendar_year > 2035) AND (value > 2040.16)"

    def test_a_mixed_kind_where_stays_classifiable(self):
        """What that seam is FOR, asserted against the classifier itself rather than
        against the string: wrapping the whole `where` in one pair of parens would hand
        `classify_where_kinds` a single conjunct naming both a dimension and `value`,
        which it rejects as a mixed-kind predicate -- even though the caller wrote
        exactly the two separate conjuncts the rule asks for.
        """
        pipeline = build_data_query_pipeline(
            _dataset_dimensions(),
            FilterDimensions(scenario_id=[7]),
            DataOptions(where="calendar_year > 2035 AND value > 2040.16"),
            "csv",
        )
        kinds = classify_where_kinds(pipeline.steps[0].where)
        assert kinds[PredicateKind.value] == ["(value > 2040.16)"]
        assert kinds[PredicateKind.dimension] == ["scenario_id = 7", "(calendar_year > 2035)"]

    def test_an_or_inside_a_conjunct_still_keeps_its_parentheses(self):
        """Splitting never reaches inside an `OR`: `split_top_level_and` only finds a seam
        at paren depth 0, so the second conjunct here stays whole and stays wrapped.
        """
        pipeline = build_data_query_pipeline(
            _dataset_dimensions(),
            FilterDimensions(),
            DataOptions(where="calendar_year > 2035 AND (value > 3 OR value < 0)"),
            "csv",
        )
        assert pipeline.steps[0].where == "(calendar_year > 2035) AND ((value > 3 OR value < 0))"


class TestGridCell:
    def test_single_digit_id(self):
        pipeline = build_data_query_pipeline(_dataset_dimensions(), FilterDimensions(), DataOptions(grid_cell="1"), "csv")
        assert pipeline.steps[0].where == "grid_cell = 1"

    def test_multiple_digit_ids(self):
        pipeline = build_data_query_pipeline(
            _dataset_dimensions(), FilterDimensions(), DataOptions(grid_cell="1,2,3"), "csv"
        )
        assert pipeline.steps[0].where == "grid_cell IN (1, 2, 3)"

    def test_non_digit_grid_cell_without_override_is_rejected(self):
        """A coordinate-valued `grid_cell` (or any raster `grid_cell`, which is always
        coordinates) needs `io/geography.py` to resolve -- `request/` may not import
        `io/` -- so without a `grid_cell_where` override this is a caller error, not
        something this function can parse itself.
        """
        with pytest.raises(RequestError, match="grid_cell"):
            build_data_query_pipeline(
                _dataset_dimensions(), FilterDimensions(), DataOptions(grid_cell="-90.1,29.9"), "csv"
            )

    def test_negative_number_is_rejected_as_non_digit(self):
        """`str.isdigit()` rejects a leading `-` -- mirrors `analysis.vector.
        apply_grid_cell_selection`'s own digit-vs-coordinate rule exactly, including this
        edge case (a negative literal ID is not representable through this parameter,
        matching the old engine's behavior)."""
        with pytest.raises(RequestError, match="grid_cell"):
            build_data_query_pipeline(_dataset_dimensions(), FilterDimensions(), DataOptions(grid_cell="-1"), "csv")

    def test_grid_cell_where_override_bypasses_digit_parsing(self):
        """The mechanism a route uses after resolving a coordinate `grid_cell` itself
        (via `io.geography.resolve_vector_grid_cell`, async, which `request/` cannot call)
        -- the override is used verbatim and `options.grid_cell`'s own content is never
        parsed at all.
        """
        pipeline = build_data_query_pipeline(
            _dataset_dimensions(),
            FilterDimensions(),
            DataOptions(grid_cell="-90.1,29.9"),
            "csv",
            grid_cell_where="grid_cell IN (5, 6)",
        )
        assert pipeline.steps[0].where == "grid_cell IN (5, 6)"

    def test_grid_cell_where_override_composes_with_dimension_filters(self):
        pipeline = build_data_query_pipeline(
            _dataset_dimensions(),
            FilterDimensions(scenario_id=[7]),
            DataOptions(),
            "csv",
            grid_cell_where="grid_cell != grid_cell",
        )
        assert pipeline.steps[0].where == "scenario_id = 7 AND grid_cell != grid_cell"


class TestOutputFormat:
    def test_raster_format_is_accepted_and_passed_through(self):
        """The builder itself does not reject a raster-format request for a vector grid
        (or vice versa) -- that is `ops.validate_shapes`'s job, run by the caller
        afterward, not this function's.
        """
        dataset_dimensions = DatasetDimensions(variable="vct", time_unit="annual", grid=RASTER_GRID)
        pipeline = build_data_query_pipeline(dataset_dimensions, FilterDimensions(), DataOptions(), "tif")
        assert pipeline.output.format == "tif"
        validate_shapes(pipeline)  # must not raise: raster input, raster (tif) output

    def test_tile_format_is_accepted(self):
        dataset_dimensions = DatasetDimensions(variable="vct", time_unit="annual", grid=RASTER_GRID)
        pipeline = build_data_query_pipeline(dataset_dimensions, FilterDimensions(), DataOptions(), "png")
        assert pipeline.output.format == "png"
