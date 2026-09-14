"""Tests for the Dataset query pipeline on vector (Parquet) grids.

The five stages run in the order documented in docs/data-analysis-api.md:
filter -> calculate -> group and aggregate -> transform and extract -> sort.
"""

import polars as pl
import pytest
from rasterio.crs import CRS
from rasterio.warp import transform
from shapely.geometry import box

from cpra.mp.data.analysis import AnalysisError
from cpra.mp.data.analysis import vector as dataset_module
from cpra.mp.data.analysis.types import DataFormat, OutputType
from cpra.mp.data.tabular import write_parquet
from tests.factories import (
    ANNUAL_KWARGS,
    DAILY_KWARGS,
    annual_frame,
    write_grid_geometry,
    write_vector_crosswalk,
)

DIMENSIONS = {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}
# sal, the one variable whose structure row sets a `float_precision`.
SAL_DIMENSIONS = {k: DAILY_KWARGS[k] for k in ("variable", "grid", "time_unit")}
# The dimension columns every wide annual frame carries, so tests can isolate the
# per-cell columns (unpivoted into grid_cell/value on write) beside them.
DIMENSION_NAMES = set(annual_frame(years=(2030,)).select(pl.col("^[a-z].*$")).columns)


def _lng_lat(x, y):
    """Project an ecoregion_v001 (EPSG:26915) point to WGS84 for a grid_cell= query."""
    lng, lat = [c[0] for c in transform(CRS.from_epsg(26915), CRS.from_epsg(4326), [x], [y])]
    return f"{lng!s},{lat!s}"


@pytest.fixture
def annual_data(base_path):
    """Two years of ssf on ecoregion_v001; cell (year, geo) holds year + geo/100."""
    write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
    return base_path


@pytest.fixture
def two_scenario_data(base_path):
    write_parquet(annual_frame(years=(2030,)), base_path=base_path)
    write_parquet(annual_frame(years=(2030,), scenario_id=8), base_path=base_path)
    return base_path


class TestOutputType:
    def test_vector_grid_is_vector_output(self, dataset_factory, annual_data):
        assert dataset_factory(DIMENSIONS).output_type == OutputType.vector

    def test_variable_and_legend_resolve(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS)
        assert dataset.variable.name == "ssf"
        assert dataset.legend.name == "ssf"

    def test_color_map_option_overrides_the_variable_legend(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, color_map="accr")
        assert dataset.legend.name == "accr"

    def test_unknown_color_map_falls_back_to_default(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, color_map="not_a_legend")
        assert dataset.legend.name == "ssf"


class TestTabularKwargs:
    """The writer arguments a Dataset contributes, on top of the request's own.

    These read the variable's structure row rather than any data, so no dataset is written
    for them -- the queries are never collected.
    """

    def test_a_variable_with_a_float_precision_rounds_its_csv(self, dataset_factory):
        dataset = dataset_factory(SAL_DIMENSIONS)
        assert dataset.variable.float_precision == 1
        assert dataset.tabular_kwargs(DataFormat.csv) == {"float_precision": 1}

    @pytest.mark.parametrize("format", [f for f in DataFormat if f != DataFormat.csv])
    def test_only_csv_is_rounded(self, dataset_factory, format):
        """The binary formats have no `float_precision` to give -- they carry the stored
        value, and whatever reads them decides how to show it."""
        assert dataset_factory(SAL_DIMENSIONS).tabular_kwargs(format) == {}

    def test_a_variable_without_a_float_precision_adds_nothing(self, dataset_factory):
        dataset = dataset_factory(DIMENSIONS)
        assert dataset.variable.float_precision is None
        assert dataset.tabular_kwargs(DataFormat.csv) == {}


class TestCustomColorMap:
    def test_discrete_color_map_is_parsed(self, dataset_factory, annual_data):
        """Discrete maps are color,bound,color,...,color -- the final color is open-ended,
        so there is one fewer bound than colors."""
        dataset = dataset_factory(DIMENSIONS, color_map="discrete:#ff0000,1,#00ff00")
        legend = dataset.legend
        assert legend.type == "discrete"
        assert legend.break_points["color"].to_list() == ["#ff0000", "#00ff00"]
        assert legend.break_points["upper_bound"].to_list() == [1.0, None]

    def test_discrete_with_a_bound_per_color_is_rejected(self, dataset_factory, annual_data):
        """The implicit open-ended bound means an equal count is one value too many."""
        with pytest.raises(AnalysisError) as excinfo:
            dataset_factory(DIMENSIONS, color_map="discrete:#ff0000,1,#00ff00,2").legend
        assert "mismatched colors and values" in excinfo.value.detail

    def test_discrete_with_three_classes(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, color_map="discrete:#ff0000,1,#00ff00,2,#0000ff")
        assert dataset.legend.break_points["upper_bound"].to_list() == [1.0, 2.0, None]

    def test_continuous_color_map_is_parsed(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, color_map="continuous:0,#ff0000,10,#0000ff")
        legend = dataset.legend
        assert legend.type == "continuous"
        assert legend.break_points["color"].to_list() == ["#ff0000", "#0000ff"]
        assert legend.break_points["upper_bound"].to_list() == [0.0, 10.0]

    def test_invalid_map_type_raises_400(self, dataset_factory, annual_data):
        with pytest.raises(AnalysisError) as excinfo:
            dataset_factory(DIMENSIONS, color_map="sideways:#ff0000,1").legend
        assert excinfo.value.status_code == 400
        assert "invalid color map type" in excinfo.value.detail

    def test_non_numeric_values_raise_400(self, dataset_factory, annual_data):
        with pytest.raises(AnalysisError) as excinfo:
            dataset_factory(DIMENSIONS, color_map="continuous:low,#ff0000").legend
        assert "invalid color map values" in excinfo.value.detail

    def test_mismatched_colors_and_values_raise_400(self, dataset_factory, annual_data):
        with pytest.raises(AnalysisError) as excinfo:
            dataset_factory(DIMENSIONS, color_map="continuous:0,#ff0000,10").legend
        assert "mismatched colors and values" in excinfo.value.detail


class TestFilterStage:
    async def test_where_clause(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, where="calendar_year > 2035")
        await dataset.apply_params()
        assert dataset.materialize()["calendar_year"].unique().to_list() == [2040]

    async def test_grid_cell_selects_rows(self, dataset_factory, annual_data):
        """A grid_cell filter on a vector grid selects rows, not columns, now that each
        cell is a row rather than a column."""
        dataset = dataset_factory(DIMENSIONS, grid_cell="1,2")
        await dataset.apply_params()
        df = dataset.materialize()
        assert sorted(df["grid_cell"].unique().to_list()) == [1, 2]

    async def test_grid_cell_keeps_dimension_columns(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, grid_cell="1")
        await dataset.apply_params()
        assert "calendar_year" in dataset.materialize().columns

    async def test_grid_cell_records_the_ids(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, grid_cell="1,2")
        await dataset.apply_params()
        assert dataset.grid_cell_ids == ["1", "2"]

    async def test_dimension_filters_are_applied(self, dataset_factory, two_scenario_data):
        dataset = dataset_factory(DIMENSIONS, filters={"scenario_id": [8]})
        await dataset.apply_params()
        assert dataset.materialize()["scenario_id"].unique().to_list() == [8]


class TestValueWhereStage:
    """A `where` VALUE conjunct (a predicate over VALUES, using the placeholder column
    `value`, unlike a `where` DIMENSION conjunct -- see `expressions.classify_where`).
    Cell (year, geo) holds `year + geo/100` (see `annual_frame`), so a threshold between
    the two years' ranges isolates exactly one year's rows.
    """

    async def test_non_matching_values_are_dropped_for_a_plain_request(
        self, dataset_factory, annual_data
    ):
        """PLAIN (non-aggregating: no group_aggregate/summarize_aggregate) is the one
        case where a `where` value conjunct behaves differently from the retired
        `value_where`, which always masked -- see `Dataset._apply_filter_params`'s
        docstring: there is no aggregate downstream here for a masked null to fall into
        unnoticed, so a failing row is dropped outright instead."""
        dataset = dataset_factory(DIMENSIONS, where="value > 2035")
        await dataset.apply_params()
        df = dataset.materialize()
        assert df.height == 32
        assert df["calendar_year"].unique().to_list() == [2040]
        assert df["value"].null_count() == 0

    async def test_masked_values_are_excluded_from_a_group_by_count(
        self, dataset_factory, annual_data
    ):
        """AGGREGATING (group_aggregate set): null-skipping `count()` on a masked column
        is exactly "how many values satisfied the predicate" -- the conditional-count
        capability, unchanged from the retired `value_where`'s own behavior."""
        dataset = dataset_factory(DIMENSIONS, where="value > 2035", group_aggregate="count")
        await dataset.apply_params()
        df = dataset.materialize()
        assert df.height == 32
        assert df["value"].to_list() == [1] * 32

    async def test_masked_values_are_excluded_from_a_group_by_sum(
        self, dataset_factory, annual_data
    ):
        """A non-count aggregate over a masked column is a conditional aggregate for
        free ("sum of values where value > x"), not just conditional counting."""
        source = annual_frame(years=(2030, 2040))
        dataset = dataset_factory(DIMENSIONS, where="value > 2035", group_aggregate="sum")
        await dataset.apply_params()
        df = dataset.materialize().filter(pl.col("grid_cell") == 1)
        assert df["value"][0] == pytest.approx(source["1"][1], rel=1e-5)

    async def test_mixed_dimension_and_value_conjuncts_aggregating(
        self, dataset_factory, annual_data
    ):
        """A dimension condition and a value condition inside one `where` are two
        separate top-level AND-ed conjuncts, not one combined predicate the way the
        retired `value_where` allowed (see `expressions.classify_where`'s "mixed
        conjunct" rejection) -- each applies with ITS OWN semantics. The dimension
        conjunct is a real row filter, dropping calendar_year=2030 BEFORE the group_by
        even sees it (unlike the old value_where, which only masked); the value
        conjunct still just masks a non-matching survivor, composing with
        group_aggregate the same as a value-only `where` does.
        """
        dataset = dataset_factory(
            DIMENSIONS, where="calendar_year > 2035 AND value > 0", group_aggregate="count"
        )
        await dataset.apply_params()
        df = dataset.materialize()
        # Every remaining (2040-only) row already satisfies value > 0, so nothing is
        # masked -- this pins the ROUTING (dimension filters, value masks), not masking
        # itself, which the tests above already cover.
        assert df.height == 32
        assert df["value"].to_list() == [1] * 32

    async def test_mixed_dimension_and_value_conjuncts_plain(self, dataset_factory, annual_data):
        """The PLAIN counterpart of the aggregating case above: the dimension conjunct
        still drops calendar_year=2030 as a real filter, and the value conjunct drops
        (not masks) any surviving row that fails it, per the PLAIN/drop contract."""
        dataset = dataset_factory(DIMENSIONS, where="calendar_year > 2035 AND value > 2040.16")
        await dataset.apply_params()
        df = dataset.materialize()
        assert df["calendar_year"].unique().to_list() == [2040]
        assert df["value"].null_count() == 0
        assert (df["value"] > 2040.16).all()
        assert df.height < 32  # some geo cells' 2040 value is <= 2040.16 and were dropped

    async def test_mixed_conjunct_is_rejected(self, dataset_factory, annual_data):
        """A single AND-term referencing both a dimension column and `value` (as
        opposed to two separate AND-ed conjuncts, see the tests above) has no
        ambiguity-free classification and is rejected outright -- substituted here for
        the retired value_where's own "unknown column" rejection, since a bare
        dimension-only conjunct is no longer eagerly namespace-checked at all (it goes
        to the same unrestricted `lf.sql()` path any other `where` dimension conjunct
        does); this still pins the same observable contract the old test did: a 400
        naming the offending token, raised before any data is read.
        """
        dataset = dataset_factory(DIMENSIONS, where="value > not_a_column")
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "not_a_column" in excinfo.value.detail

    async def test_aggregate_inside_the_predicate_is_rejected(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, where="value = min(value)")
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "min" in excinfo.value.detail

    async def test_disallowed_function_is_rejected(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, where="upper(value) = 'X'")
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400

    async def test_value_conjunct_subquery_is_rejected_at_plan_time(self, dataset_factory, annual_data):
        """The value-conjunct half of the closed pushdown-defeating hole (see
        api/expressions.py's module docstring and `classify_where`): `value in (select
        ... from self)` fails to parse at `classify_where` time (no `self` relation is
        registered there) before `apply_params` gets anywhere near `.collect()`/
        `materialize()`. No spy is needed the way the raster equivalent uses one
        (`scan_rasters` is a late-bound call the transform stage makes); here,
        `get_lazy_frame` already runs unconditionally in `Dataset.__init__`, before any
        `where` classification -- the guarantee this test actually pins is that
        rejection happens with no `.collect()`/`materialize()` ever reached, which not
        calling `dataset.materialize()` below already establishes.
        """
        dataset = dataset_factory(DIMENSIONS, where="value in (select max(value) from self)")
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400

    async def test_dimension_conjunct_subquery_is_rejected_at_plan_time(self, dataset_factory, annual_data):
        """The dimension-conjunct half of the same closed hole: a subquery inside a
        DIMENSION conjunct is refused just as eagerly as one inside a value conjunct is
        (see `test_value_conjunct_subquery_is_rejected_at_plan_time` for why no spy is
        used here either)."""
        dataset = dataset_factory(
            DIMENSIONS, where="calendar_year in (select max(calendar_year) from self)"
        )
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400


class TestGridCellCoordinates:
    """grid_cell= on a vector grid may be grid cell IDs (as above) or lat/lng pairs,
    resolved to grid cell IDs via point-in-polygon lookup against config.GRIDS_PATH."""

    @pytest.fixture
    def ecoregion_geometry(self, geometry_path):
        """Two adjacent 1km squares in ecoregion_v001's CRS, at IDs 1 and 2."""
        return write_grid_geometry(
            geometry_path,
            "ecoregion_v001",
            [
                (1, box(500000, 3300000, 501000, 3301000)),
                (2, box(501000, 3300000, 502000, 3301000)),
            ],
        )

    async def test_coordinate_resolves_to_grid_cell(
        self, dataset_factory, annual_data, ecoregion_geometry
    ):
        label = _lng_lat(500500, 3300500)  # inside square 1
        dataset = dataset_factory(DIMENSIONS, grid_cell=label)
        await dataset.apply_params()
        assert dataset.grid_cell_ids == ["1"]

    async def test_multiple_coordinates_resolve_in_order(
        self, dataset_factory, annual_data, ecoregion_geometry
    ):
        label1 = _lng_lat(500500, 3300500)  # square 1
        label2 = _lng_lat(501500, 3300500)  # square 2
        dataset = dataset_factory(DIMENSIONS, grid_cell=f"{label1},{label2}")
        await dataset.apply_params()
        assert dataset.grid_cell_ids == ["1", "2"]

    async def test_unmatched_coordinate_is_ignored(
        self, dataset_factory, annual_data, ecoregion_geometry
    ):
        """A point outside every polygon is dropped rather than erroring."""
        inside = _lng_lat(500500, 3300500)
        outside = _lng_lat(0, 0)
        dataset = dataset_factory(DIMENSIONS, grid_cell=f"{inside},{outside}")
        await dataset.apply_params()
        assert dataset.grid_cell_ids == ["1"]

    async def test_grid_cell_ids_still_work(
        self, dataset_factory, annual_data, ecoregion_geometry
    ):
        """IDs (digit-only tokens) take the original, non-geometry path."""
        dataset = dataset_factory(DIMENSIONS, grid_cell="1,2")
        await dataset.apply_params()
        assert dataset.grid_cell_ids == ["1", "2"]

    async def test_coordinates_in_one_cell_select_it_once(
        self, dataset_factory, annual_data, ecoregion_geometry
    ):
        """A vector grid_cell is its ID, and each ID is one row, so the cell can only
        appear once however many coordinates landed in it."""
        first = _lng_lat(500400, 3300400)
        second = _lng_lat(500600, 3300600)
        dataset = dataset_factory(DIMENSIONS, grid_cell=f"{first},{second}")
        await dataset.apply_params()
        assert dataset.grid_cell_ids == ["1"]
        df = dataset.materialize()
        assert df["grid_cell"].unique().to_list() == [1]

    async def test_grid_cell_column_reports_the_resolved_cell(
        self, dataset_factory, annual_data, ecoregion_geometry
    ):
        """The response reports cell IDs whichever way the caller selected them — the
        coordinate is how it was asked for, not what the values describe."""
        label = _lng_lat(500500, 3300500)
        dataset = dataset_factory(DIMENSIONS, grid_cell=label)
        await dataset.apply_params()
        assert dataset.materialize()["grid_cell"].unique().to_list() == [1]

    async def test_unpivoting_reports_the_cell_id(
        self, dataset_factory, annual_data, ecoregion_geometry
    ):
        """The shape every visualization requests, so the ID has to reach the grid_cell
        column rather than the coordinate that selected it. Data is already long on disk,
        so unpivot=True is a no-op here, but it's kept for backward API compatibility."""
        label = _lng_lat(500500, 3300500)
        dataset = dataset_factory(DIMENSIONS, grid_cell=label, unpivot=True)
        await dataset.apply_params()
        assert dataset.materialize()["grid_cell"].unique().to_list() == [1]

    async def test_a_select_naming_only_dimensions_still_works(
        self, dataset_factory, annual_data, ecoregion_geometry
    ):
        dataset = dataset_factory(
            DIMENSIONS, grid_cell=_lng_lat(500500, 3300500), select="calendar_year"
        )
        await dataset.apply_params()
        assert dataset.materialize().columns == ["calendar_year"]

    async def test_no_geometry_file_raises_400(self, dataset_factory, annual_data, geometry_path):
        dataset = dataset_factory(DIMENSIONS, grid_cell=_lng_lat(500500, 3300500))
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "No geometry file" in excinfo.value.detail


class TestCalculateStage:
    async def test_negate_flips_matching_rows(self, dataset_factory, two_scenario_data):
        dataset = dataset_factory(DIMENSIONS, negate="scenario_id = 8")
        await dataset.apply_params()
        df = dataset.materialize().filter(pl.col("grid_cell") == 1).sort("scenario_id")
        assert df["value"][0] > 0
        assert df["value"][1] < 0

    async def test_multiply10(self, dataset_factory, annual_data):
        base = annual_frame(years=(2030,))["1"][0]
        dataset = dataset_factory(DIMENSIONS, multiply10="calendar_year = 2030")
        await dataset.apply_params()
        df = dataset.materialize().filter(pl.col("grid_cell") == 1).sort("calendar_year")
        assert df["value"][0] == pytest.approx(base * 10, rel=1e-5)

    async def test_multiply100(self, dataset_factory, annual_data):
        base = annual_frame(years=(2030,))["1"][0]
        dataset = dataset_factory(DIMENSIONS, multiply100="calendar_year = 2030")
        await dataset.apply_params()
        df = dataset.materialize().filter(pl.col("grid_cell") == 1).sort("calendar_year")
        assert df["value"][0] == pytest.approx(base * 100, rel=1e-5)

    async def test_non_matching_rows_are_untouched(self, dataset_factory, annual_data):
        base = annual_frame(years=(2040,))["1"][0]
        dataset = dataset_factory(DIMENSIONS, multiply10="calendar_year = 2030")
        await dataset.apply_params()
        df = dataset.materialize().filter(pl.col("grid_cell") == 1).sort("calendar_year")
        assert df["value"][1] == pytest.approx(base, rel=1e-5)

    async def test_no_calculate_params_is_a_no_op(self, dataset_factory, annual_data):
        base = annual_frame(years=(2030,))["1"][0]
        dataset = dataset_factory(DIMENSIONS)
        await dataset.apply_params()
        df = dataset.materialize().filter(pl.col("grid_cell") == 1).sort("calendar_year")
        assert df["value"][0] == pytest.approx(base, rel=1e-5)

    async def test_first_matching_expression_wins(self, dataset_factory, annual_data):
        """The when/then chain is evaluated in negate, multiply10, multiply100 order."""
        base = annual_frame(years=(2030,))["1"][0]
        dataset = dataset_factory(
            DIMENSIONS, negate="calendar_year = 2030", multiply10="calendar_year = 2030"
        )
        await dataset.apply_params()
        df = dataset.materialize().filter(pl.col("grid_cell") == 1).sort("calendar_year")
        assert df["value"][0] == pytest.approx(-base, rel=1e-5)


class TestGroupStage:
    async def test_group_by_without_aggregate_raises_400(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, group_by="calendar_year")
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "group_aggregate must be specified" in excinfo.value.detail

    async def test_aggregate_without_group_by_collapses_everything(
        self, dataset_factory, annual_data
    ):
        """grid_cell always stays part of the group key in long format -- each row is one
        grid cell -- so "everything" collapsing means every non-grid_cell dimension, not the
        whole dataset into a single row: each of ecoregion_v001's 32 cells gets one row."""
        dataset = dataset_factory(DIMENSIONS, group_aggregate="sum")
        await dataset.apply_params()
        df = dataset.materialize()
        assert df.height == 32
        expected = sum(annual_frame(years=(2030, 2040))["1"].to_list())
        cell_1 = df.filter(pl.col("grid_cell") == 1)
        assert cell_1["value"][0] == pytest.approx(expected, rel=1e-5)

    async def test_group_by_a_dimension(self, dataset_factory, two_scenario_data):
        """One row per (scenario_id, grid_cell), since grid_cell always stays in the group
        key in long format."""
        dataset = dataset_factory(DIMENSIONS, group_by="scenario_id", group_aggregate="max")
        await dataset.apply_params()
        df = dataset.materialize()
        assert df.height == 2 * 32
        assert sorted(df["scenario_id"].unique().to_list()) == [7, 8]

    async def test_ungrouped_dimensions_are_dropped(self, dataset_factory, two_scenario_data):
        """A dimension collapsed away by group_by (calendar_year here, not part of the
        `group_by="scenario_id"` key) is absent from the result entirely -- it used to
        come back as an all-null, Null-dtype placeholder column, which was a meaningless
        artifact every caller of `analyze_data`/the tabular API saw for every such
        dimension (see `VectorGridAnalysis._collapse_group`'s comment)."""
        dataset = dataset_factory(DIMENSIONS, group_by="scenario_id", group_aggregate="max")
        await dataset.apply_params()
        df = dataset.materialize()
        assert "calendar_year" not in df.columns

    @pytest.mark.parametrize("aggregate", ["min", "max", "sum", "mean"])
    async def test_each_aggregate_function(self, dataset_factory, annual_data, aggregate):
        dataset = dataset_factory(DIMENSIONS, group_aggregate=aggregate)
        await dataset.apply_params()
        values = annual_frame(years=(2030, 2040))["1"].to_list()
        expected = {
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
            "mean": sum(values) / len(values),
        }[aggregate]
        df = dataset.materialize().filter(pl.col("grid_cell") == 1)
        assert df["value"][0] == pytest.approx(expected, rel=1e-5)

    async def test_invalid_group_by_dimension_raises_400(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, group_by="nope", group_aggregate="sum")
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400


class TestSummarizeStage:
    @pytest.fixture
    def crosswalk(self, crosswalk_path):
        """Map ecoregions 1-4 onto two destination compartments, halving each source."""
        return write_vector_crosswalk(
            crosswalk_path,
            "ecoregion_v001",
            "hydro_compartment_v001",
            [(1, 100, 0.5), (2, 100, 0.5), (3, 200, 0.5), (4, 200, 0.5)],
        )

    async def test_summarize_by_without_aggregate_raises_400(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, summarize_by="hydro_compartment_v001")
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "must be used together" in excinfo.value.detail

    async def test_aggregate_without_summarize_by_raises_400(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, summarize_aggregate="sum")
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400

    async def test_missing_crosswalk_raises_400(
        self, dataset_factory, annual_data, crosswalk_path
    ):
        """Regression: the missing-crosswalk branch was guarded by `except
        FileNotFoundError` around a lazy scan_parquet, which never raises there, so the
        request failed later with an opaque Polars error instead of a 400."""
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="parish_v2020", summarize_aggregate="sum"
        )
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "No crosswalk from ecoregion_v001 to parish_v2020" in excinfo.value.detail

    async def test_summarize_produces_destination_grid_cells(
        self, dataset_factory, annual_data, crosswalk
    ):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="sum"
        )
        await dataset.apply_params()
        df = dataset.materialize()
        assert sorted(df["grid_cell"].unique().to_list()) == [100, 200]

    async def test_scale_factors_are_applied(self, dataset_factory, annual_data, crosswalk):
        source = annual_frame(years=(2030, 2040))
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="sum"
        )
        await dataset.apply_params()
        df = dataset.materialize().filter(pl.col("grid_cell") == 100).sort("calendar_year")
        expected = (source["1"][0] + source["2"][0]) * 0.5
        assert df["value"][0] == pytest.approx(expected, rel=1e-5)

    async def test_geographic_unit_column_is_added(self, dataset_factory, annual_data, crosswalk):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="sum"
        )
        await dataset.apply_params()
        assert dataset.materialize()["geographic_unit"].unique().to_list() == [
            "hydro_compartment_v001"
        ]

    async def test_summarize_grid_cell_narrows_destinations(
        self, dataset_factory, annual_data, crosswalk
    ):
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="sum",
            summarize_grid_cell="200",
        )
        await dataset.apply_params()
        df = dataset.materialize()
        assert df["grid_cell"].unique().to_list() == [200]

    async def test_grid_cell_filter_narrows_sources(self, dataset_factory, annual_data, crosswalk):
        """A source-side grid_cell filter removes its contribution from the summary.

        Regression: `grid_cell_ids` stay strings because they were originally column
        names, and were fed straight into `is_in` against the crosswalk's integer `src_id`.
        """
        source = annual_frame(years=(2030,))
        dataset = dataset_factory(
            DIMENSIONS,
            grid_cell="1,3",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="sum",
        )
        await dataset.apply_params()
        df = dataset.materialize().filter(pl.col("grid_cell") == 100).sort("calendar_year")
        assert df["value"][0] == pytest.approx(source["1"][0] * 0.5, rel=1e-5)

    async def test_grid_cell_filter_drops_excluded_sources_entirely(
        self, dataset_factory, annual_data, crosswalk
    ):
        """Source cells 2 and 4 are excluded, so each destination has one contributor."""
        source = annual_frame(years=(2030,))
        dataset = dataset_factory(
            DIMENSIONS,
            grid_cell="1,3",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="sum",
        )
        await dataset.apply_params()
        df = dataset.materialize().filter(pl.col("grid_cell") == 200).sort("calendar_year")
        assert df["value"][0] == pytest.approx(source["3"][0] * 0.5, rel=1e-5)

    async def test_summarize_grid_cell_ids_are_cast_to_int(self, dataset_factory, annual_data):
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="sum",
            summarize_grid_cell="100, 200",
        )
        dataset._validate_summarize_params()
        assert dataset.summarize_grid_cell_ids == [100, 200]

    async def test_summarize_switches_output_to_vector(self, dataset_factory, annual_data):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="max"
        )
        assert dataset.output_type == OutputType.vector


class TestFoldRouting:
    """Dataset-level check that the two vector group_by sites actually call the fold
    (api/fold.py) exactly when `api.fold.should_fold_group_by`'s rule says they should.
    `test_fold.py::TestShouldFoldGroupBy` pins the rule itself as a pure function; this
    pins that `_apply_group_params` and `_apply_transform_params_vector` feed it the
    arguments the rule needs and honor its answer.
    """

    @pytest.fixture
    def spy_fold(self, monkeypatch):
        """Wraps the real `fold_group_by` (rather than replacing it with a mock) so
        these tests also double as correctness checks on this package's fixture-scale
        data, not just call-count checks."""
        calls = []
        real_fold_group_by = dataset_module.fold_group_by

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real_fold_group_by(*args, **kwargs)

        monkeypatch.setattr(dataset_module, "fold_group_by", spy)
        return calls

    @pytest.fixture
    def crosswalk(self, crosswalk_path):
        return write_vector_crosswalk(
            crosswalk_path,
            "ecoregion_v001",
            "hydro_compartment_v001",
            [(1, 100, 0.5), (2, 100, 0.5), (3, 200, 0.5), (4, 200, 0.5)],
        )

    async def test_unfiltered_group_aggregate_routes_through_the_fold(
        self, dataset_factory, annual_data, spy_fold
    ):
        """Cliff-shaped: no group_by, no selective filter -- the group key is
        `grid_cell` alone, exactly what `should_fold_group_by` folds."""
        dataset = dataset_factory(DIMENSIONS, group_aggregate="sum")
        await dataset.apply_params()
        dataset.materialize()
        assert len(spy_fold) == 1

    async def test_a_grid_cell_filter_skips_the_fold(self, dataset_factory, annual_data, spy_fold):
        dataset = dataset_factory(DIMENSIONS, group_aggregate="sum", grid_cell="1")
        await dataset.apply_params()
        dataset.materialize()
        assert spy_fold == []

    async def test_a_dimension_filter_skips_the_fold(self, dataset_factory, annual_data, spy_fold):
        dataset = dataset_factory(
            DIMENSIONS, group_aggregate="sum", filters={"calendar_year": [2030]}
        )
        await dataset.apply_params()
        dataset.materialize()
        assert spy_fold == []

    async def test_a_where_clause_skips_the_fold(self, dataset_factory, annual_data, spy_fold):
        dataset = dataset_factory(
            DIMENSIONS, group_aggregate="sum", where="calendar_year > 2035"
        )
        await dataset.apply_params()
        dataset.materialize()
        assert spy_fold == []

    async def test_crosswalk_reaggregation_routes_through_the_fold_when_unfiltered(
        self, dataset_factory, annual_data, crosswalk, spy_fold
    ):
        """No `group_by`/`group_aggregate` here, so `_apply_group_params` returns before
        ever considering a fold -- only `_apply_transform_params_vector`'s crosswalk
        reaggregation site is reachable, and it fires because the request is unfiltered."""
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="sum"
        )
        await dataset.apply_params()
        dataset.materialize()
        assert len(spy_fold) == 1

    async def test_summarize_grid_cell_skips_the_crosswalk_fold(
        self, dataset_factory, annual_data, crosswalk, spy_fold
    ):
        """summarize_grid_cell narrows the crosswalk's destinations -- exactly the
        selective-filter signal that should skip its fold."""
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="sum",
            summarize_grid_cell="100",
        )
        await dataset.apply_params()
        dataset.materialize()
        assert spy_fold == []

    async def test_group_aggregate_and_summarize_together_fold_both_sites(
        self, dataset_factory, annual_data, crosswalk, spy_fold
    ):
        """With both `group_aggregate` and `summarize_by` set and nothing filtering the
        request, both fold-eligible sites are reachable and both fire: the group_by
        collapse in `_apply_group_params` (always keyed on grid_cell in long format,
        which runs first) and the crosswalk reaggregation in
        `_apply_transform_params_vector` (some crosswalks land on a finer destination
        than the source grid; see that method's docstring), operating on the
        already-collapsed result of the first."""
        dataset = dataset_factory(
            DIMENSIONS,
            group_aggregate="sum",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="sum",
        )
        await dataset.apply_params()
        dataset.materialize()
        assert len(spy_fold) == 2


class TestCountByValue:
    """count_by_value: one row per (dimension group, destination zone, distinct value),
    with the count of source grid cells sharing that value -- a categorical count, a
    genuinely different output SHAPE from every other summarize_aggregate (which collapse
    a zone to one row). See `expressions.count_by_value_frame` for the cardinality bound
    and `Dataset._apply_transform_params_vector`'s count_by_value branch for the
    vector-side wiring: no `scale_factor` multiply (unlike every other summarize_aggregate
    here), since rescaling a categorical code before counting it would change which raw
    values collapse into "the same" category.
    """

    @pytest.fixture
    def crosswalk(self, crosswalk_path):
        """Map ecoregions 1-4 onto two destination compartments -- the same split
        `TestSummarizeStage.crosswalk` uses, redefined locally since this class's fixture
        data (categorical, not `annual_frame`'s default continuous value_fn) differs."""
        return write_vector_crosswalk(
            crosswalk_path,
            "ecoregion_v001",
            "hydro_compartment_v001",
            [(1, 100, 0.5), (2, 100, 0.5), (3, 200, 0.5), (4, 200, 0.5)],
        )

    @pytest.fixture
    def categorical_annual_data(self, base_path):
        """Each ecoregion's value is a small integer category (geo_id % 3), not
        `annual_frame`'s default continuous `year + geo_id/100` -- a bounded categorical
        column (e.g. a land-type code) is exactly what count_by_value is for."""
        write_parquet(
            annual_frame(years=(2030,), value_fn=lambda _year, geo_id: float(geo_id % 3)),
            base_path=base_path,
        )
        return base_path

    def _expected_counts(self):
        """zone -> {value: count}, derived from the same (ecoregion -> compartment)
        mapping `crosswalk` writes and the same value_fn `categorical_annual_data` uses,
        rather than hand-computed, so it can't drift from what the fixtures actually
        contain."""
        mapping = {1: 100, 2: 100, 3: 200, 4: 200}
        counts = {100: {}, 200: {}}
        for geo_id, zone in mapping.items():
            value = float(geo_id % 3)
            counts[zone][value] = counts[zone].get(value, 0) + 1
        return counts

    async def test_counts_source_cells_per_zone_per_value(
        self, dataset_factory, categorical_annual_data, crosswalk
    ):
        expected = self._expected_counts()
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="count_by_value"
        )
        await dataset.apply_params()
        df = dataset.materialize()

        assert set(df.columns) >= {"grid_cell", "value", "count", "geographic_unit"}
        for zone_id, value_counts in expected.items():
            zone_df = df.filter(pl.col("grid_cell") == zone_id)
            actual = dict(zip(zone_df["value"].to_list(), zone_df["count"].to_list()))
            assert actual == value_counts

    async def test_composes_with_where_value_conjunct_mask_then_count(
        self, dataset_factory, categorical_annual_data, crosswalk
    ):
        """A `where` value conjunct masks a non-matching source cell's value to null in
        `_apply_filter_params` -- upstream of this stage -- so it is excluded here the
        same way `count_by_value_frame` excludes any other null: not a category, and not
        counted under any value."""
        expected = self._expected_counts()
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="count_by_value",
            where="value != 1",
        )
        await dataset.apply_params()
        df = dataset.materialize()
        for zone_id, value_counts in expected.items():
            zone_df = df.filter(pl.col("grid_cell") == zone_id)
            actual = dict(zip(zone_df["value"].to_list(), zone_df["count"].to_list()))
            expected_masked = {v: c for v, c in value_counts.items() if v != 1}
            assert actual == expected_masked

    async def test_cardinality_ceiling_is_enforced(
        self, dataset_factory, annual_data, crosswalk, monkeypatch
    ):
        """`annual_data`'s default value_fn (`year + geo_id/100`) makes every source cell
        its own distinct value -- exactly what the ceiling exists to catch (see
        config.COUNT_BY_VALUE_MAX_DISTINCT and count_by_value_frame's docstring)."""
        monkeypatch.setattr(dataset_module, "COUNT_BY_VALUE_MAX_DISTINCT", 1)
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="count_by_value"
        )
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "distinct" in excinfo.value.detail

    async def test_group_aggregate_count_by_value_is_rejected(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, group_aggregate="count_by_value")
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "count_by_value" in excinfo.value.detail

    async def test_calculate_params_are_rejected_with_count_by_value(
        self, dataset_factory, categorical_annual_data, crosswalk
    ):
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="count_by_value",
            negate="1 = 1",
        )
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "count_by_value" in excinfo.value.detail


class TestSortAndSelectStage:
    async def test_sort_ascending(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, sort="calendar_year")
        await dataset.apply_params()
        # Long format repeats each year once per grid cell, so compare the order years
        # first appear in rather than the raw (now 64-row) list.
        years = dataset.materialize()["calendar_year"].unique(maintain_order=True).to_list()
        assert years == [2030, 2040]

    async def test_sort_descending(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, sort="-calendar_year")
        await dataset.apply_params()
        years = dataset.materialize()["calendar_year"].unique(maintain_order=True).to_list()
        assert years == [2040, 2030]

    async def test_multiple_sort_keys(self, dataset_factory, two_scenario_data):
        dataset = dataset_factory(DIMENSIONS, sort="-scenario_id,calendar_year")
        await dataset.apply_params()
        assert dataset.materialize()["scenario_id"].to_list()[0] == 8

    async def test_invalid_sort_key_raises_400(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, sort="nonsense")
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400

    async def test_select_restricts_columns(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, select="calendar_year,scenario_id")
        await dataset.apply_params()
        assert dataset.materialize().columns == ["calendar_year", "scenario_id"]

    async def test_select_rejects_unknown_columns(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, select="calendar_year,nope")
        with pytest.raises(AnalysisError) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400

    async def test_select_allows_grid_cell_and_value(self, dataset_factory, annual_data):
        """grid_cell/value are always present in long format, but are still valid select
        targets (unpivot=True is kept for backward API compatibility, and is a no-op)."""
        dataset = dataset_factory(DIMENSIONS, unpivot=True, select="grid_cell,value")
        await dataset.apply_params()
        assert dataset.materialize().columns == ["grid_cell", "value"]


class TestUnpivot:
    async def test_produces_grid_cell_and_value_columns(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, unpivot=True)
        await dataset.apply_params()
        df = dataset.materialize()
        assert "grid_cell" in df.columns
        assert "value" in df.columns

    async def test_row_count_is_years_times_geographies(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, unpivot=True)
        await dataset.apply_params()
        assert dataset.materialize().height == 2 * 32

    async def test_values_are_preserved(self, dataset_factory, annual_data):
        source = annual_frame(years=(2030, 2040))
        dataset = dataset_factory(DIMENSIONS, unpivot=True)
        await dataset.apply_params()
        df = dataset.materialize().filter(
            (pl.col("calendar_year") == 2030) & (pl.col("grid_cell") == 1)
        )
        assert df["value"][0] == pytest.approx(source["1"][0], rel=1e-5)


class TestEstimateDataSize:
    """estimate_data_size decides whether a response is buffered or streamed.

    It is called for every vector `/data.{format}` request, so an exception here surfaces
    as a 500 rather than a slow response.
    """

    ALL_FILTERED = {
        "model_group_id": [500],
        "scenario_id": [7],
        "percentile": [50],
        "calendar_year": [2030],
    }

    def test_unknown_dimension_cardinality_returns_none(self, dataset_factory, annual_data):
        """model_group_id has no CSV values and no filter, so rows can't be estimated."""
        assert dataset_factory(DIMENSIONS).estimate_data_size() is None

    def test_grid_id_count_is_used_for_the_column_count(self, dataset_factory, annual_data):
        """ssf is Float32 on a 32-cell grid: 1 row x 32 columns x 4 bytes."""
        dataset = dataset_factory(DIMENSIONS, filters=self.ALL_FILTERED)
        assert dataset.estimate_data_size() == pytest.approx(32 * 32 / 8)

    def test_filters_multiply_the_row_estimate(self, dataset_factory, annual_data):
        one = dataset_factory(DIMENSIONS, filters=self.ALL_FILTERED).estimate_data_size()
        two = dataset_factory(
            DIMENSIONS, filters={**self.ALL_FILTERED, "calendar_year": [2030, 2040]}
        ).estimate_data_size()
        assert two == 2 * one

    def test_grid_cell_ids_override_the_grid_column_count(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, filters=self.ALL_FILTERED)
        dataset.grid_cell_ids = ["1", "2"]
        assert dataset.estimate_data_size() == pytest.approx(2 * 32 / 8)

    def test_group_by_shrinks_the_estimate(self, dataset_factory, annual_data):
        """Only the group_by columns contribute rows, so scenario_id's two values collapse."""
        filters = {**self.ALL_FILTERED, "scenario_id": [7, 8], "calendar_year": [2030, 2040]}
        grouped = dataset_factory(
            DIMENSIONS, filters=filters, group_by="calendar_year", group_aggregate="sum"
        ).estimate_data_size()
        ungrouped = dataset_factory(DIMENSIONS, filters=filters).estimate_data_size()
        assert ungrouped == 4 * 32 * 32 / 8
        assert grouped == 2 * 32 * 32 / 8

    def test_unfiltered_values_backed_dimension_uses_its_value_count(
        self, dataset_factory, annual_data
    ):
        """percentile is unfiltered, so all of its documented values contribute rows.

        Regression: `elif dimension.values:` called bool() on a Polars DataFrame, which
        raises rather than reporting emptiness. It only surfaced once every preceding
        dimension was filtered -- an earlier dimension with no values returns None first
        and short-circuits the loop -- so it reached production as a 500 on /data.csv.
        """
        from cpra.mp.data.dimensions import DIMENSIONS as ALL_DIMENSIONS

        percentile_count = len(ALL_DIMENSIONS["percentile"].values)
        dataset = dataset_factory(
            DIMENSIONS,
            filters={"model_group_id": [500], "scenario_id": [7], "calendar_year": [2030]},
        )
        assert dataset.estimate_data_size() == pytest.approx(percentile_count * 32 * 32 / 8)
