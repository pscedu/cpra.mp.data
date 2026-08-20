"""Tests for the Dataset query pipeline on vector (Parquet) grids.

The five stages run in the order documented in docs/data-analysis-api.md:
filter -> calculate -> group and aggregate -> transform and extract -> sort.
"""

import polars as pl
import pytest
from fastapi import HTTPException
from rasterio.crs import CRS
from rasterio.warp import transform
from shapely.geometry import box

from cpra.mp.data.api.types import OutputType
from cpra.mp.data.tabular import write_parquet

from tests.factories import ANNUAL_KWARGS, annual_frame, write_grid_geometry, write_vector_crosswalk

DIMENSIONS = {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}
# The dimension columns every wide annual frame carries, so tests can isolate the
# geography columns beside them.
DIMENSION_NAMES = set(annual_frame(years=(2030,)).select(pl.col("^[a-z].*$")).columns)


def _lng_lat(x, y):
    """Project an ecoregion_v001 (EPSG:26915) point to WGS84 for a geography= query."""
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
        with pytest.raises(HTTPException) as excinfo:
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
        with pytest.raises(HTTPException) as excinfo:
            dataset_factory(DIMENSIONS, color_map="sideways:#ff0000,1").legend
        assert excinfo.value.status_code == 400
        assert "invalid color map type" in excinfo.value.detail

    def test_non_numeric_values_raise_400(self, dataset_factory, annual_data):
        with pytest.raises(HTTPException) as excinfo:
            dataset_factory(DIMENSIONS, color_map="continuous:low,#ff0000").legend
        assert "invalid color map values" in excinfo.value.detail

    def test_mismatched_colors_and_values_raise_400(self, dataset_factory, annual_data):
        with pytest.raises(HTTPException) as excinfo:
            dataset_factory(DIMENSIONS, color_map="continuous:0,#ff0000,10").legend
        assert "mismatched colors and values" in excinfo.value.detail


class TestFilterStage:
    async def test_where_clause(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, where="calendar_year > 2035")
        await dataset.apply_params()
        assert dataset.materialize()["calendar_year"].to_list() == [2040]

    async def test_geography_selects_columns(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, geography="1,2")
        await dataset.apply_params()
        columns = dataset.materialize().columns
        assert [c for c in columns if c[0].isdigit()] == ["1", "2"]

    async def test_geography_keeps_dimension_columns(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, geography="1")
        await dataset.apply_params()
        assert "calendar_year" in dataset.materialize().columns

    async def test_geography_records_the_ids(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, geography="1,2")
        await dataset.apply_params()
        assert dataset.geography_ids == ["1", "2"]

    async def test_unknown_geography_becomes_400(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, geography="9999")
        await dataset.apply_params()
        with pytest.raises(HTTPException) as excinfo:
            dataset.materialize()
        assert excinfo.value.status_code == 400

    async def test_dimension_filters_are_applied(self, dataset_factory, two_scenario_data):
        dataset = dataset_factory(DIMENSIONS, filters={"scenario_id": [8]})
        await dataset.apply_params()
        assert dataset.materialize()["scenario_id"].unique().to_list() == [8]


class TestGeographyCoordinates:
    """geography= on a vector grid may be grid cell IDs (as above) or lat/lng pairs,
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
        dataset = dataset_factory(DIMENSIONS, geography=label)
        await dataset.apply_params()
        assert dataset.geography_ids == ["1"]

    async def test_multiple_coordinates_resolve_in_order(
        self, dataset_factory, annual_data, ecoregion_geometry
    ):
        label1 = _lng_lat(500500, 3300500)  # square 1
        label2 = _lng_lat(501500, 3300500)  # square 2
        dataset = dataset_factory(DIMENSIONS, geography=f"{label1},{label2}")
        await dataset.apply_params()
        assert dataset.geography_ids == ["1", "2"]

    async def test_unmatched_coordinate_is_ignored(
        self, dataset_factory, annual_data, ecoregion_geometry
    ):
        """A point outside every polygon is dropped rather than erroring."""
        inside = _lng_lat(500500, 3300500)
        outside = _lng_lat(0, 0)
        dataset = dataset_factory(DIMENSIONS, geography=f"{inside},{outside}")
        await dataset.apply_params()
        assert dataset.geography_ids == ["1"]

    async def test_grid_cell_ids_still_work(
        self, dataset_factory, annual_data, ecoregion_geometry
    ):
        """IDs (digit-only tokens) take the original, non-geometry path."""
        dataset = dataset_factory(DIMENSIONS, geography="1,2")
        await dataset.apply_params()
        assert dataset.geography_ids == ["1", "2"]

    async def test_coordinates_in_one_cell_select_it_once(
        self, dataset_factory, annual_data, ecoregion_geometry
    ):
        """A vector geography is its ID, and IDs double as column names, so the cell can
        only appear once however many coordinates landed in it."""
        first = _lng_lat(500400, 3300400)
        second = _lng_lat(500600, 3300600)
        dataset = dataset_factory(DIMENSIONS, geography=f"{first},{second}")
        await dataset.apply_params()
        assert dataset.geography_ids == ["1"]
        df = dataset.materialize()
        assert [c for c in df.columns if c not in DIMENSION_NAMES] == ["1"]

    async def test_columns_are_named_for_the_resolved_cell(
        self, dataset_factory, annual_data, ecoregion_geometry
    ):
        """The response reports cell IDs whichever way the caller selected them — the
        coordinate is how it was asked for, not what the values describe."""
        label = _lng_lat(500500, 3300500)
        dataset = dataset_factory(DIMENSIONS, geography=label)
        await dataset.apply_params()
        columns = dataset.materialize().columns
        assert "1" in columns
        assert label not in columns

    async def test_unpivoting_reports_the_cell_id(
        self, dataset_factory, annual_data, ecoregion_geometry
    ):
        """The shape every visualization requests, so the ID has to reach the geography
        column rather than the coordinate that selected it."""
        label = _lng_lat(500500, 3300500)
        dataset = dataset_factory(DIMENSIONS, geography=label, unpivot=True)
        await dataset.apply_params()
        assert dataset.materialize()["geography"].unique().to_list() == ["1"]

    async def test_a_select_that_drops_the_geography_columns_still_works(
        self, dataset_factory, annual_data, ecoregion_geometry
    ):
        """`select` can only name dimensions, so without unpivot it drops every geography
        column."""
        dataset = dataset_factory(
            DIMENSIONS, geography=_lng_lat(500500, 3300500), select="calendar_year"
        )
        await dataset.apply_params()
        assert dataset.materialize().columns == ["calendar_year"]

    async def test_no_geometry_file_raises_400(self, dataset_factory, annual_data, geometry_path):
        dataset = dataset_factory(DIMENSIONS, geography=_lng_lat(500500, 3300500))
        with pytest.raises(HTTPException) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "No geometry file" in excinfo.value.detail


class TestCalculateStage:
    async def test_negate_flips_matching_rows(self, dataset_factory, two_scenario_data):
        dataset = dataset_factory(DIMENSIONS, negate="scenario_id = 8")
        await dataset.apply_params()
        df = dataset.materialize().sort("scenario_id")
        assert df["1"][0] > 0
        assert df["1"][1] < 0

    async def test_multiply10(self, dataset_factory, annual_data):
        base = annual_frame(years=(2030,))["1"][0]
        dataset = dataset_factory(DIMENSIONS, multiply10="calendar_year = 2030")
        await dataset.apply_params()
        df = dataset.materialize().sort("calendar_year")
        assert df["1"][0] == pytest.approx(base * 10, rel=1e-5)

    async def test_multiply100(self, dataset_factory, annual_data):
        base = annual_frame(years=(2030,))["1"][0]
        dataset = dataset_factory(DIMENSIONS, multiply100="calendar_year = 2030")
        await dataset.apply_params()
        df = dataset.materialize().sort("calendar_year")
        assert df["1"][0] == pytest.approx(base * 100, rel=1e-5)

    async def test_non_matching_rows_are_untouched(self, dataset_factory, annual_data):
        base = annual_frame(years=(2040,))["1"][0]
        dataset = dataset_factory(DIMENSIONS, multiply10="calendar_year = 2030")
        await dataset.apply_params()
        df = dataset.materialize().sort("calendar_year")
        assert df["1"][1] == pytest.approx(base, rel=1e-5)

    async def test_no_calculate_params_is_a_no_op(self, dataset_factory, annual_data):
        base = annual_frame(years=(2030,))["1"][0]
        dataset = dataset_factory(DIMENSIONS)
        await dataset.apply_params()
        assert dataset.materialize().sort("calendar_year")["1"][0] == pytest.approx(base, rel=1e-5)

    async def test_first_matching_expression_wins(self, dataset_factory, annual_data):
        """The when/then chain is evaluated in negate, multiply10, multiply100 order."""
        base = annual_frame(years=(2030,))["1"][0]
        dataset = dataset_factory(
            DIMENSIONS, negate="calendar_year = 2030", multiply10="calendar_year = 2030"
        )
        await dataset.apply_params()
        df = dataset.materialize().sort("calendar_year")
        assert df["1"][0] == pytest.approx(-base, rel=1e-5)


class TestGroupStage:
    async def test_group_by_without_aggregate_raises_400(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, group_by="calendar_year")
        with pytest.raises(HTTPException) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "group_aggregate must be specified" in excinfo.value.detail

    async def test_aggregate_without_group_by_collapses_everything(
        self, dataset_factory, annual_data
    ):
        dataset = dataset_factory(DIMENSIONS, group_aggregate="sum")
        await dataset.apply_params()
        df = dataset.materialize()
        assert df.height == 1
        expected = sum(annual_frame(years=(2030, 2040))["1"].to_list())
        assert df["1"][0] == pytest.approx(expected, rel=1e-5)

    async def test_group_by_a_dimension(self, dataset_factory, two_scenario_data):
        dataset = dataset_factory(DIMENSIONS, group_by="scenario_id", group_aggregate="max")
        await dataset.apply_params()
        assert dataset.materialize().height == 2

    async def test_ungrouped_dimensions_become_null(self, dataset_factory, two_scenario_data):
        dataset = dataset_factory(DIMENSIONS, group_by="scenario_id", group_aggregate="max")
        await dataset.apply_params()
        df = dataset.materialize()
        assert df["calendar_year"].null_count() == df.height

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
        assert dataset.materialize()["1"][0] == pytest.approx(expected, rel=1e-5)

    async def test_invalid_group_by_dimension_raises_400(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, group_by="nope", group_aggregate="sum")
        with pytest.raises(HTTPException) as excinfo:
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
        with pytest.raises(HTTPException) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "must be used together" in excinfo.value.detail

    async def test_aggregate_without_summarize_by_raises_400(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, summarize_aggregate="sum")
        with pytest.raises(HTTPException) as excinfo:
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
        with pytest.raises(HTTPException) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400
        assert "No crosswalk from ecoregion_v001 to parish_v2020" in excinfo.value.detail

    async def test_summarize_produces_destination_columns(
        self, dataset_factory, annual_data, crosswalk
    ):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="sum"
        )
        await dataset.apply_params()
        df = dataset.materialize()
        assert [c for c in df.columns if c[0].isdigit()] == ["100", "200"]

    async def test_scale_factors_are_applied(self, dataset_factory, annual_data, crosswalk):
        source = annual_frame(years=(2030, 2040))
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="sum"
        )
        await dataset.apply_params()
        df = dataset.materialize().sort("calendar_year")
        expected = (source["1"][0] + source["2"][0]) * 0.5
        assert df["100"][0] == pytest.approx(expected, rel=1e-5)

    async def test_geographic_unit_column_is_added(self, dataset_factory, annual_data, crosswalk):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="sum"
        )
        await dataset.apply_params()
        assert dataset.materialize()["geographic_unit"].unique().to_list() == [
            "hydro_compartment_v001"
        ]

    async def test_summarize_geography_narrows_destinations(
        self, dataset_factory, annual_data, crosswalk
    ):
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="sum",
            summarize_geography="200",
        )
        await dataset.apply_params()
        df = dataset.materialize()
        assert [c for c in df.columns if c[0].isdigit()] == ["200"]

    async def test_geography_filter_narrows_sources(self, dataset_factory, annual_data, crosswalk):
        """A source-side geography filter removes its contribution from the summary.

        Regression: `geography_ids` stay strings because they double as column names, and
        were fed straight into `is_in` against the crosswalk's integer `src_id`.
        """
        source = annual_frame(years=(2030,))
        dataset = dataset_factory(
            DIMENSIONS,
            geography="1,3",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="sum",
        )
        await dataset.apply_params()
        df = dataset.materialize().sort("calendar_year")
        assert df["100"][0] == pytest.approx(source["1"][0] * 0.5, rel=1e-5)

    async def test_geography_filter_drops_excluded_sources_entirely(
        self, dataset_factory, annual_data, crosswalk
    ):
        """Geographies 2 and 4 are excluded, so each destination has one contributor."""
        source = annual_frame(years=(2030,))
        dataset = dataset_factory(
            DIMENSIONS,
            geography="1,3",
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="sum",
        )
        await dataset.apply_params()
        df = dataset.materialize().sort("calendar_year")
        assert df["200"][0] == pytest.approx(source["3"][0] * 0.5, rel=1e-5)

    async def test_summarize_geography_ids_are_cast_to_int(self, dataset_factory, annual_data):
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="sum",
            summarize_geography="100, 200",
        )
        dataset._validate_summarize_params()
        assert dataset.summarize_geography_ids == [100, 200]

    async def test_summarize_switches_output_to_vector(self, dataset_factory, annual_data):
        dataset = dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="max"
        )
        assert dataset.output_type == OutputType.vector


class TestSortAndSelectStage:
    async def test_sort_ascending(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, sort="calendar_year")
        await dataset.apply_params()
        assert dataset.materialize()["calendar_year"].to_list() == [2030, 2040]

    async def test_sort_descending(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, sort="-calendar_year")
        await dataset.apply_params()
        assert dataset.materialize()["calendar_year"].to_list() == [2040, 2030]

    async def test_multiple_sort_keys(self, dataset_factory, two_scenario_data):
        dataset = dataset_factory(DIMENSIONS, sort="-scenario_id,calendar_year")
        await dataset.apply_params()
        assert dataset.materialize()["scenario_id"].to_list()[0] == 8

    async def test_invalid_sort_key_raises_400(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, sort="nonsense")
        with pytest.raises(HTTPException) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400

    async def test_select_restricts_columns(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, select="calendar_year,scenario_id")
        await dataset.apply_params()
        assert dataset.materialize().columns == ["calendar_year", "scenario_id"]

    async def test_select_rejects_unknown_columns(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, select="calendar_year,nope")
        with pytest.raises(HTTPException) as excinfo:
            await dataset.apply_params()
        assert excinfo.value.status_code == 400

    async def test_select_allows_geography_and_value(self, dataset_factory, annual_data):
        """These only exist after unpivot, but are valid select targets."""
        dataset = dataset_factory(DIMENSIONS, unpivot=True, select="geography,value")
        await dataset.apply_params()
        assert dataset.materialize().columns == ["geography", "value"]


class TestUnpivot:
    async def test_produces_geography_and_value_columns(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, unpivot=True)
        await dataset.apply_params()
        df = dataset.materialize()
        assert "geography" in df.columns
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
            (pl.col("calendar_year") == 2030) & (pl.col("geography") == "1")
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

    def test_geography_ids_override_the_grid_column_count(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, filters=self.ALL_FILTERED)
        dataset.geography_ids = ["1", "2"]
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
