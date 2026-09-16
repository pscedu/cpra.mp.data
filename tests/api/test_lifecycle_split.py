"""Tests for the separation of request LIFECYCLE from op ALGEBRA in the analysis layer.

`BaseAnalysis` used to interleave four concerns that have nothing to do with the op
algebra it also carries: output rendering, tile validation, catalog/metadata lookups, and
options handling. Each has been lifted into plain free functions in its own module
(`analysis/output.py`, `analysis/tiles.py`, `analysis/metadata.py`, `analysis/stages.py`),
with the class keeping only the members other packages reach for by duck typing, or that
the fixed five-stage flow names as its own vocabulary -- as delegations, not as a second
implementation.

Two things this file exists to prove for each extraction, mirroring
`tests/api/test_paired_port.py`'s convention:

* The extracted code actually EXECUTES from its new home (a monkeypatch spy on the free
  function, observed through the surface a route would use), and the old implementation is
  GONE rather than merely bypassed -- pinned by asserting the names that implementation
  needed are no longer bound in `analysis/base.py` at all.
* The behavior is unchanged, checked against real output (a written file's bytes, a real
  `Legend`'s break points) rather than against a shape or a row count.
"""

import polars as pl
import pytest

from cpra.mp.data.analysis import base as base_module
from cpra.mp.data.analysis import metadata as metadata_module
from cpra.mp.data.analysis import output as output_module
from cpra.mp.data.analysis import stages as stages_module
from cpra.mp.data.analysis import tiles as tiles_module
from cpra.mp.data.analysis.expressions import ExpressionError
from cpra.mp.data.analysis.parameters import CalculateEntry
from cpra.mp.data.analysis.types import DataFormat, OutputType
from cpra.mp.data.tabular import write_parquet
from tests.factories import ANNUAL_KWARGS, RASTER_KWARGS, annual_frame, write_vector_crosswalk

DIMENSIONS = {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}
RASTER_DIMENSIONS = {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}
# Every dimension pinned to one value, so a size estimate is fully determined -- the same
# set `test_dataset_vector.py::TestEstimateDataSize` uses.
ALL_FILTERED = {
    "model_group_id": [500],
    "scenario_id": [7],
    "percentile": [50],
    "calendar_year": [2030],
}


@pytest.fixture
def annual_data(base_path):
    """Two years of ssf on ecoregion_v001 -- the same shape `test_dataset_vector.py` uses."""
    write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
    return base_path


@pytest.fixture
def crosswalk(crosswalk_path):
    """Map ecoregions 1-4 onto two destination compartments, halving each source."""
    return write_vector_crosswalk(
        crosswalk_path,
        "ecoregion_v001",
        "hydro_compartment_v001",
        [(1, 100, 0.5), (2, 100, 0.5), (3, 200, 0.5), (4, 200, 0.5)],
    )


class TestOutputRenderingRunsFromItsNewHome:
    """Gate 1 for the output-rendering extraction."""

    def test_legend_runs_output_resolve_legend(self, dataset_factory, annual_data, monkeypatch):
        seen = []
        sentinel = object()

        def spy(color_map, variable_legend):
            seen.append((color_map, variable_legend.name if variable_legend else None))
            return sentinel

        monkeypatch.setattr(output_module, "resolve_legend", spy)
        dataset = dataset_factory(DIMENSIONS, color_map="discrete:#ff0000,1")

        assert dataset.legend is sentinel, "the property must return what the free function returned"
        assert seen == [("discrete:#ff0000,1", "ssf")]

    def test_output_type_runs_output_resolve_output_type(
        self, dataset_factory, annual_data, monkeypatch
    ):
        seen = []

        def spy(grid_type, *, grid_cell, summarize_by):
            seen.append((grid_type, grid_cell, summarize_by))
            return OutputType.raster

        monkeypatch.setattr(output_module, "resolve_output_type", spy)
        dataset = dataset_factory(DIMENSIONS, summarize_by="hydro_compartment_v001")

        # A vector grid can only ever answer `vector` -- getting `raster` back proves the
        # answer came from the patched function and nowhere else.
        assert dataset.output_type == OutputType.raster
        assert seen == [("vector", None, "hydro_compartment_v001")]

    def test_tabular_kwargs_runs_output_tabular_kwargs(
        self, dataset_factory, annual_data, monkeypatch
    ):
        seen = []

        def spy(format, float_precision):
            seen.append((format, float_precision))
            return {"marker": True}

        monkeypatch.setattr(output_module, "tabular_kwargs", spy)
        assert dataset_factory(DIMENSIONS).tabular_kwargs(DataFormat.csv) == {"marker": True}
        assert seen == [(DataFormat.csv, None)]

    def test_stream_and_write_tabular_run_from_the_output_module(
        self, dataset_factory, annual_data, monkeypatch, tmp_path
    ):
        seen = []

        def spy(materialize, out_path, format, *args, **kwargs):
            seen.append((format, str(out_path), materialize().height))

        monkeypatch.setattr(output_module, "stream_tabular", spy)
        monkeypatch.setattr(output_module, "write_tabular", spy)

        dataset = dataset_factory(DIMENSIONS)
        rows = dataset.materialize().height
        dataset.stream_tabular(tmp_path / "streamed.csv", DataFormat.csv)
        dataset.write_tabular(tmp_path / "written.csv", DataFormat.csv)

        assert seen == [
            (DataFormat.csv, str(tmp_path / "streamed.csv"), rows),
            (DataFormat.csv, str(tmp_path / "written.csv"), rows),
        ]
        assert not (tmp_path / "streamed.csv").exists(), "the spy replaced the real writer"

    def test_no_rendering_implementation_survives_in_base(self):
        """Complementary to the spies above: the rendering logic is GONE from `base.py`,
        not merely shadowed by a delegation that happens to be called first.

        Asserted through the module namespace rather than by reading source: every name
        the old implementation needed (the legend registry, the writer-method tables' one
        imported dependency, the CSV-precision format enum, the `@timing` decorator on the
        two writers) had to be imported into `base.py` to be used there. None of them is
        bound any more, so no copy of that code can still be running from this module.
        """
        assert not hasattr(base_module, "LEGENDS")
        assert not hasattr(base_module, "DataFormat")
        assert not hasattr(base_module, "timing")
        assert not hasattr(base_module, "SINK_METHODS")
        assert not hasattr(base_module, "WRITE_METHODS")


class TestTileValidationRunsFromItsNewHome:
    """Gate 1 for the tile-validation extraction."""

    def test_validate_tile_runs_tiles_validate_raster_tile(
        self, dataset_factory, annual_data, monkeypatch
    ):
        seen = []

        def spy(lf, output_type):
            seen.append((lf.collect_schema().names(), output_type))

        monkeypatch.setattr(tiles_module, "validate_raster_tile", spy)
        dataset = dataset_factory(DIMENSIONS)

        # A vector grid would normally 400 here ("invalid output type for tile: vector");
        # returning cleanly proves the check ran only inside the patched function.
        dataset.validate_tile()
        (names, output_type), = seen
        assert output_type == OutputType.vector
        assert "grid_cell" in names, "the frame itself is handed over, not a row count"

    def test_validate_vector_tile_runs_tiles_validate_vector_tile(
        self, dataset_factory, annual_data, monkeypatch
    ):
        seen = []
        monkeypatch.setattr(
            tiles_module, "validate_vector_tile", lambda output_type, grid_name: seen.append(
                (output_type, grid_name)
            )
        )

        # No pmtiles file exists for this grid, so the real function would 404.
        dataset_factory(DIMENSIONS).validate_vector_tile()
        assert seen == [(OutputType.vector, "ecoregion_v001")]

    def test_summarize_by_picks_the_destination_grids_tile_file(
        self, dataset_factory, annual_data, monkeypatch
    ):
        """The grid whose geometry a `.pbf` serves is the summarize DESTINATION when the
        request has one -- resolved by the caller now, so pin that it still is.
        """
        seen = []
        monkeypatch.setattr(
            tiles_module, "validate_vector_tile", lambda output_type, grid_name: seen.append(grid_name)
        )
        dataset_factory(
            DIMENSIONS, summarize_by="hydro_compartment_v001", summarize_aggregate="mean"
        ).validate_vector_tile()
        assert seen == ["hydro_compartment_v001"]

    def test_no_tile_validation_implementation_survives_in_base(self):
        """Complementary to the spies above: the checks are GONE from `base.py`. The
        PMTiles lookup needed `cpra.mp.data.config` imported there to reach `TILES_PATH`,
        and nothing else in that module ever used it; it is no longer bound.
        """
        assert not hasattr(base_module, "config")


class TestTileValidationBehaviorIsUnchanged:
    """Gate 2 for the tile-validation extraction: the real checks, against real state."""

    def test_a_vector_output_is_rejected_for_a_raster_tile_before_any_collect(self):
        """The output-type check short-circuits: an `lf` that would raise on `.collect()`
        proves the band count is never computed for a vector output.
        """
        exploding = pl.LazyFrame({"value": [1]}).select(pl.col("nope"))
        with pytest.raises(base_module.AnalysisError) as excinfo:
            tiles_module.validate_raster_tile(exploding, OutputType.vector)
        assert excinfo.value.detail == "invalid output type for tile: vector"
        assert excinfo.value.status_code == 400

    def test_one_band_passes_and_two_bands_report_the_count(self):
        tiles_module.validate_raster_tile(pl.LazyFrame({"band": [0]}), OutputType.raster)
        with pytest.raises(base_module.AnalysisError) as excinfo:
            tiles_module.validate_raster_tile(pl.LazyFrame({"band": [0, 1]}), OutputType.raster)
        assert excinfo.value.detail == "tile contains more than one band: 2 bands found"
        assert excinfo.value.status_code == 400

    def test_a_raster_output_is_rejected_for_a_vector_tile(self, tiles_path):
        with pytest.raises(base_module.AnalysisError) as excinfo:
            tiles_module.validate_vector_tile(OutputType.raster, "ecoregion_v001")
        assert excinfo.value.detail == "invalid output type for vector tile: raster"
        assert excinfo.value.status_code == 400

    def test_a_missing_pmtiles_file_is_a_404_and_a_present_one_passes(self, tiles_path):
        with pytest.raises(base_module.AnalysisError) as excinfo:
            tiles_module.validate_vector_tile(OutputType.vector, "ecoregion_v001")
        assert excinfo.value.detail == "no tile file for grid: ecoregion_v001"
        assert excinfo.value.status_code == 404

        (tiles_path / "ecoregion_v001.pmtiles").touch()
        tiles_module.validate_vector_tile(OutputType.vector, "ecoregion_v001")


class TestCatalogLookupsRunFromTheirNewHome:
    """Gate 1 for the catalog/metadata extraction."""

    def test_variable_runs_metadata_variable_for(self, dataset_factory, annual_data, monkeypatch):
        seen = []
        sentinel = object()

        def spy(dataset_dimensions):
            seen.append(dataset_dimensions.variable)
            return sentinel

        monkeypatch.setattr(metadata_module, "variable_for", spy)
        assert dataset_factory(DIMENSIONS).variable is sentinel
        assert seen == ["ssf"]

    def test_estimate_data_size_runs_metadata_estimate_data_size(
        self, dataset_factory, annual_data, monkeypatch
    ):
        seen = []

        def spy(dataset_dimensions, filter_dimensions, grid, group_by_cols, *, has_group_by, grid_cell_ids):
            seen.append((dataset_dimensions.variable, grid.name, group_by_cols, has_group_by, grid_cell_ids))
            return 12345

        monkeypatch.setattr(metadata_module, "estimate_data_size", spy)
        dataset = dataset_factory(DIMENSIONS, group_by="calendar_year", group_aggregate="mean")

        assert dataset.estimate_data_size() == 12345
        assert seen == [("ssf", "ecoregion_v001", ["calendar_year"], True, None)]

    def test_the_metadata_table_check_is_gone_from_the_class(self, dataset_factory, annual_data):
        """`_is_raster_metadata_table` had no caller outside `base.py` itself, so unlike
        `variable`/`estimate_data_size` it did not have to stay as a delegation. A rename
        that left the method importable would let a stale caller keep working against a
        second copy of the rule.
        """
        assert not hasattr(base_module.BaseAnalysis, "_is_raster_metadata_table")
        assert not hasattr(dataset_factory(DIMENSIONS), "_is_raster_metadata_table")

    def test_no_catalog_lookup_implementation_survives_in_base(self):
        """Complementary to the spies above: the names those lookups needed in `base.py`
        -- the variable registry and the `re` module the dtype bit-width parse used -- are
        no longer bound there.
        """
        assert not hasattr(base_module, "VARIABLES")
        assert not hasattr(base_module, "re")


class TestCatalogLookupBehaviorIsUnchanged:
    """Gate 2 for the catalog/metadata extraction: real catalog answers, real frames."""

    def test_variable_for_resolves_the_real_catalog_entry(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS)
        resolved = metadata_module.variable_for(dataset.dataset_dimensions)
        assert resolved.name == "ssf"
        assert resolved is dataset.variable

    def test_a_raster_frame_is_metadata_until_it_has_no_path_column(self):
        band_metadata = pl.LazyFrame({"path": ["a.tif"], "band": [1], "calendar_year": [2030]})
        sampled = pl.LazyFrame({"grid_cell": [1], "value": [1.0]})

        assert metadata_module.is_raster_metadata_table(band_metadata, "raster") is True
        assert metadata_module.is_raster_metadata_table(sampled, "raster") is False
        # A vector frame is never band metadata, whatever columns it happens to carry.
        assert metadata_module.is_raster_metadata_table(band_metadata, "vector") is False

    def test_the_free_function_and_the_class_agree_on_a_real_estimate(
        self, dataset_factory, annual_data
    ):
        """ssf is Float32 on a 32-cell grid and every dimension is filtered to one value:
        1 row x 32 columns x 4 bytes, computed the same way from either entry point.
        """
        dataset = dataset_factory(DIMENSIONS, filters=ALL_FILTERED)
        direct = metadata_module.estimate_data_size(
            dataset.dataset_dimensions,
            dataset.filter_dimensions,
            dataset.grid,
            [],
            has_group_by=False,
            grid_cell_ids=None,
        )
        assert direct == pytest.approx(32 * 32 / 8)
        assert dataset.estimate_data_size() == direct

    def test_grid_cell_ids_override_the_grid_column_count(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS, filters=ALL_FILTERED)
        assert metadata_module.estimate_data_size(
            dataset.dataset_dimensions,
            dataset.filter_dimensions,
            dataset.grid,
            [],
            has_group_by=False,
            grid_cell_ids=["1", "2"],
        ) == pytest.approx(2 * 32 / 8)

    def test_an_unknowable_dimension_returns_none_not_zero(self, dataset_factory, annual_data):
        """model_group_id has no CSV values and no filter here, so rows cannot be
        estimated at all -- which the caller must be able to tell apart from an estimate
        of zero bytes.
        """
        dataset = dataset_factory(DIMENSIONS)
        assert (
            metadata_module.estimate_data_size(
                dataset.dataset_dimensions,
                dataset.filter_dimensions,
                dataset.grid,
                [],
                has_group_by=False,
                grid_cell_ids=None,
            )
            is None
        )


class TestOutputRenderingBehaviorIsUnchanged:
    """Gate 2 for the output-rendering extraction: real output, not shapes."""

    def test_written_bytes_match_between_the_class_and_the_free_function(
        self, dataset_factory, annual_data, tmp_path
    ):
        dataset = dataset_factory(DIMENSIONS)
        through_class = tmp_path / "class.csv"
        through_function = tmp_path / "function.csv"

        dataset.write_tabular(through_class, DataFormat.csv)
        output_module.write_tabular(dataset.materialize, through_function, DataFormat.csv)

        assert through_class.read_bytes() == through_function.read_bytes()
        assert pl.read_csv(through_class).height == dataset.materialize().height

    def test_streamed_bytes_match_the_written_bytes(self, dataset_factory, annual_data, tmp_path):
        dataset = dataset_factory(DIMENSIONS)
        streamed = tmp_path / "streamed.csv"
        written = tmp_path / "written.csv"

        output_module.stream_tabular(dataset.materialize, streamed, DataFormat.csv)
        output_module.write_tabular(dataset.materialize, written, DataFormat.csv)

        assert streamed.read_bytes() == written.read_bytes()

    def test_a_custom_discrete_legend_keeps_its_open_ended_upper_bound(self):
        """A discrete map's final color is open-ended, so it carries one fewer declared
        bound than colors and the parser appends the `None` itself.
        """
        legend = output_module.resolve_legend("discrete:#ff0000,1,#00ff00,2,#0000ff", None)
        assert legend.name == "custom"
        assert legend.type == "discrete"
        assert legend.break_points["upper_bound"].to_list() == [1.0, 2.0, None]
        assert legend.break_points["color"].to_list() == ["#ff0000", "#00ff00", "#0000ff"]

    def test_a_custom_continuous_legend_reads_value_then_color(self):
        legend = output_module.resolve_legend("continuous:0,#ff0000,10,#0000ff", None)
        assert legend.type == "continuous"
        assert legend.break_points["upper_bound"].to_list() == [0.0, 10.0]
        assert legend.break_points["color"].to_list() == ["#ff0000", "#0000ff"]

    @pytest.mark.parametrize(
        "color_map, message",
        [
            ("sideways:#ff0000,1", "invalid color map type: sideways"),
            ("continuous:low,#ff0000", "invalid color map values"),
            ("continuous:0,#ff0000,10", "mismatched colors and values in color map"),
        ],
    )
    def test_the_rejections_keep_their_exact_messages_and_status(self, color_map, message):
        with pytest.raises(base_module.AnalysisError) as excinfo:
            output_module.resolve_legend(color_map, None)
        assert excinfo.value.detail == message
        assert excinfo.value.status_code == 400

    def test_a_raster_grid_answers_vector_once_grid_cell_narrows_it(self):
        assert (
            output_module.resolve_output_type("raster", grid_cell=None, summarize_by=None)
            == OutputType.raster
        )
        assert (
            output_module.resolve_output_type("raster", grid_cell="1,2", summarize_by=None)
            == OutputType.vector
        )
        assert (
            output_module.resolve_output_type("raster", grid_cell=None, summarize_by="parish_v2020")
            == OutputType.vector
        )
        assert (
            output_module.resolve_output_type("vector", grid_cell=None, summarize_by=None)
            == OutputType.vector
        )

    def test_csv_precision_comes_from_the_variable_only_for_csv(self):
        assert output_module.tabular_kwargs(DataFormat.csv, 1) == {"float_precision": 1}
        assert output_module.tabular_kwargs(DataFormat.csv, None) == {}
        assert output_module.tabular_kwargs(DataFormat.parquet, 1) == {}


class TestOptionsHandlingRunsFromItsNewHome:
    """Gate 1 for the options-handling extraction.

    Each spy returns a marker no real implementation could produce, so the assertion that
    the marker reached `self` also proves nothing in `base.py` recomputed the answer.
    """

    async def test_filter_stage_runs_stages_apply_filter_options(
        self, dataset_factory, annual_data, monkeypatch
    ):
        seen = []
        marker = pl.LazyFrame({"marker": [1]})

        def spy(lf, where, *, grid_type, is_aggregating):
            seen.append((where, grid_type, is_aggregating))
            return marker, "value > 1", True

        monkeypatch.setattr(stages_module, "apply_filter_options", spy)
        dataset = dataset_factory(
            DIMENSIONS, where="calendar_year = 2030", group_by="calendar_year", group_aggregate="mean"
        )
        await dataset._apply_filter_params()

        assert seen == [("calendar_year = 2030", "vector", True)]
        assert dataset.lf is marker
        assert dataset._value_where == "value > 1"
        assert dataset._where_has_dimension_conjunct is True

    async def test_calculate_stage_runs_both_stages_functions_in_order(
        self, dataset_factory, annual_data, monkeypatch
    ):
        seen = []
        plans = ["plan-marker"]
        marker = pl.LazyFrame({"marker": [1]})

        def resolve_spy(calculate):
            seen.append(("resolve", [entry.expression for entry in calculate]))
            return plans

        def apply_spy(lf, calculate_plans, grid_type):
            seen.append(("apply", calculate_plans is plans, grid_type))
            return marker

        monkeypatch.setattr(stages_module, "resolve_calculate_plans", resolve_spy)
        monkeypatch.setattr(stages_module, "apply_calculate_options", apply_spy)
        dataset = dataset_factory(DIMENSIONS, calculate=[{"expression": "value * 2"}])
        dataset._apply_calculate_params()

        # Resolve strictly first: the plans are stashed before the rewrite, which is what
        # lets a raster's pixel-read paths replay them even though the frame is untouched.
        assert seen == [("resolve", ["value * 2"]), ("apply", True, "vector")]
        assert dataset._calculate_plans is plans
        assert dataset.lf is marker

    def test_sort_stage_runs_stages_apply_sort_options(
        self, dataset_factory, annual_data, monkeypatch
    ):
        seen = []
        marker = pl.LazyFrame({"marker": [1]})

        def spy(lf, sort, select, dataset_dimensions):
            seen.append((sort, select, dataset_dimensions.variable))
            return marker

        monkeypatch.setattr(stages_module, "apply_sort_options", spy)
        dataset = dataset_factory(DIMENSIONS, sort="-calendar_year", select="calendar_year,value")
        dataset._apply_sort_params()

        assert seen == [("-calendar_year", "calendar_year,value", "ssf")]
        assert dataset.lf is marker

    def test_group_stage_runs_stages_reject_count_by_value_group(
        self, dataset_factory, annual_data, monkeypatch
    ):
        seen = []
        collapsed = []
        monkeypatch.setattr(
            stages_module, "reject_count_by_value_group", lambda group_aggregate: seen.append(group_aggregate)
        )
        dataset = dataset_factory(DIMENSIONS, group_by="calendar_year", group_aggregate="count_by_value")
        monkeypatch.setattr(type(dataset), "_collapse_group", lambda self, cols: collapsed.append(cols))

        # Neutering the rejection lets the stage run on to the collapse, which the real
        # rejection would never allow -- so reaching it proves the check happens only
        # inside the patched function.
        dataset._apply_group_params()
        assert seen == ["count_by_value"]
        assert collapsed == [["calendar_year"]]

    async def test_summarize_validation_runs_stages_validate_summarize_options(
        self, dataset_factory, annual_data, monkeypatch
    ):
        seen = []

        def spy(summarize_by, summarize_aggregate, summarize_grid_cell):
            seen.append((summarize_by, summarize_aggregate, summarize_grid_cell))
            return False, [77]

        monkeypatch.setattr(stages_module, "validate_summarize_options", spy)
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="sum",
            summarize_grid_cell="100, 200",
        )
        await dataset._apply_transform_params()

        assert seen == [("hydro_compartment_v001", "sum", "100, 200")]
        # `False` means "no summarize requested", so the transform returned without a
        # crosswalk -- none exists here, and the real function would have said `True`.
        assert dataset.summarize_grid_cell_ids == [77], "the caller stores the parsed IDs"

    async def test_the_count_by_value_calculate_rejection_runs_from_stages(
        self, dataset_factory, annual_data, crosswalk, monkeypatch
    ):
        seen = []

        class Marker(Exception):
            pass

        def spy(calculate):
            seen.append([entry.expression for entry in calculate])
            raise Marker

        monkeypatch.setattr(stages_module, "reject_calculate_with_count_by_value", spy)
        dataset = dataset_factory(
            DIMENSIONS,
            summarize_by="hydro_compartment_v001",
            summarize_aggregate="count_by_value",
            calculate=[{"expression": "value * -1"}],
        )
        with pytest.raises(Marker):
            await dataset.apply_params()
        assert seen == [["value * -1"]]

    def test_no_options_handling_implementation_survives_in_base(self):
        """Complementary to the spies above: the option handling is GONE from `base.py`.

        The three members that had no caller outside the class are unbound entirely; the
        four stage methods that remain are proved not to hold a second copy by the names
        their old bodies needed -- the sort choice list, the calculate compiler, and the
        two `select`-validation helpers -- no longer being bound in that module.
        """
        assert not hasattr(base_module.BaseAnalysis, "_resolve_calculate_entries")
        assert not hasattr(base_module.BaseAnalysis, "_validate_summarize_params")
        assert not hasattr(base_module.BaseAnalysis, "_reject_calculate_with_count_by_value")
        assert not hasattr(base_module, "SORT_CHOICES")
        assert not hasattr(base_module, "compile_calculate_entries")
        assert not hasattr(base_module, "required_columns")
        assert not hasattr(base_module, "get_input_data_frame")


class TestOptionsHandlingBehaviorIsUnchanged:
    """Gate 2 for the options-handling extraction: real frames, real values, real
    messages -- and gate 3, since the two `count_by_value` rejections and the summarize
    pairing rule are checks that moved rather than disappeared.
    """

    @pytest.fixture
    def long_frame(self):
        """Two cells, two years, one value each -- long format, as the filter stage sees
        a vector frame."""
        return pl.LazyFrame(
            {
                "calendar_year": [2030, 2030, 2040, 2040],
                "grid_cell": [1, 2, 1, 2],
                "value": [1.0, 9.0, 3.0, 11.0],
            }
        )

    def test_an_aggregating_request_masks_and_a_plain_one_drops(self, long_frame):
        """The same predicate, the same frame: masking keeps all four rows with nulls
        where it failed, dropping keeps only the rows that passed.
        """
        masked, value_where, _ = stages_module.apply_filter_options(
            long_frame, "value > 5", grid_type="vector", is_aggregating=True
        )
        dropped, _, _ = stages_module.apply_filter_options(
            long_frame, "value > 5", grid_type="vector", is_aggregating=False
        )

        assert value_where == "(value > 5)"
        assert masked.collect()["value"].to_list() == [None, 9.0, None, 11.0]
        assert dropped.collect()["value"].to_list() == [9.0, 11.0]

    def test_a_rasters_value_conjunct_is_returned_unapplied(self, long_frame):
        """A raster frame is band metadata at this point, so the value half waits for
        pixels -- it comes back as a string with the frame untouched.
        """
        lf, value_where, has_dimension_conjunct = stages_module.apply_filter_options(
            long_frame, "value > 5", grid_type="raster", is_aggregating=True
        )
        assert value_where == "(value > 5)"
        assert has_dimension_conjunct is False
        assert lf.collect()["value"].to_list() == [1.0, 9.0, 3.0, 11.0]

    def test_a_dimension_conjunct_filters_rows_and_is_reported(self, long_frame):
        lf, value_where, has_dimension_conjunct = stages_module.apply_filter_options(
            long_frame, "calendar_year = 2040", grid_type="vector", is_aggregating=False
        )
        assert (value_where, has_dimension_conjunct) == (None, True)
        assert lf.collect()["value"].to_list() == [3.0, 11.0]

    def test_no_where_is_a_no_op(self, long_frame):
        lf, value_where, has_dimension_conjunct = stages_module.apply_filter_options(
            long_frame, None, grid_type="vector", is_aggregating=False
        )
        assert (lf is long_frame, value_where, has_dimension_conjunct) == (True, None, False)

    def test_calculate_entries_are_labelled_and_applied_in_order(self, long_frame):
        """Entry N+1 sees entry N's output, so `+1` then `*10` is 10x+10, not 10x+1."""
        entries = [
            CalculateEntry(expression="value + 1"),
            CalculateEntry(expression="value * 10"),
        ]
        assert [label for label, _, _ in stages_module.resolve_calculate_entries(entries)] == [
            "calculate.1",
            "calculate.2",
        ]
        plans = stages_module.resolve_calculate_plans(entries)
        applied = stages_module.apply_calculate_options(long_frame, plans, "vector")
        assert applied.collect()["value"].to_list() == [20.0, 100.0, 40.0, 120.0]

    def test_a_raster_metadata_frame_is_left_for_the_pixel_paths(self):
        """The plans are still compiled (a bad expression must fail before any file is
        read), but the band-metadata frame itself is untouched -- it has no `value`.
        """
        band_metadata = pl.LazyFrame({"path": ["a.tif"], "band": [1], "calendar_year": [2030]})
        plans = stages_module.resolve_calculate_plans([CalculateEntry(expression="value * 2")])
        assert plans
        assert stages_module.apply_calculate_options(band_metadata, plans, "raster") is band_metadata

    def test_a_bad_calculate_expression_is_rejected_at_resolve_time(self):
        with pytest.raises(ExpressionError):
            stages_module.resolve_calculate_plans([CalculateEntry(expression="os.system('x')")])

    def test_count_by_value_is_rejected_as_a_group_aggregate_only(self):
        stages_module.reject_count_by_value_group("mean")
        stages_module.reject_count_by_value_group(None)
        with pytest.raises(base_module.AnalysisError) as excinfo:
            stages_module.reject_count_by_value_group("count_by_value")
        assert excinfo.value.detail == (
            "group_aggregate 'count_by_value' is not supported -- "
            "use summarize_by and summarize_aggregate=count_by_value instead"
        )
        assert excinfo.value.status_code == 400

    def test_calculate_is_rejected_alongside_count_by_value(self):
        stages_module.reject_calculate_with_count_by_value(None)
        stages_module.reject_calculate_with_count_by_value([])
        with pytest.raises(base_module.AnalysisError) as excinfo:
            stages_module.reject_calculate_with_count_by_value([CalculateEntry(expression="value * 2")])
        assert excinfo.value.detail == "calculate parameters cannot be combined with count_by_value"
        assert excinfo.value.status_code == 400

    def test_summarize_options_must_be_given_together(self):
        assert stages_module.validate_summarize_options(None, None, None) == (False, None)
        assert stages_module.validate_summarize_options("parish_v2020", "sum", None) == (True, None)
        for by, aggregate in [("parish_v2020", None), (None, "sum")]:
            with pytest.raises(base_module.AnalysisError) as excinfo:
                stages_module.validate_summarize_options(by, aggregate, None)
            assert excinfo.value.detail == (
                "summarize_by and summarize_aggregate parameters must be used together"
            )
            assert excinfo.value.status_code == 400

    def test_summarize_grid_cell_is_only_parsed_when_a_summarize_is_happening(self):
        """It names cells of the DESTINATION grid, so without one there is nothing for it
        to mean -- and parsing it anyway would leak into the vector tile path, which reads
        the parsed IDs whenever they are not None.
        """
        assert stages_module.validate_summarize_options("parish_v2020", "sum", "100, 200") == (
            True,
            [100, 200],
        )
        assert stages_module.validate_summarize_options(None, None, "100, 200") == (False, None)

    def test_sort_accepts_a_descending_prefix_and_select_narrows_columns(
        self, dataset_factory, annual_data
    ):
        dataset = dataset_factory(DIMENSIONS)
        sorted_lf = stages_module.apply_sort_options(
            dataset.lf, "-calendar_year", "calendar_year,value", dataset.dataset_dimensions
        )
        df = sorted_lf.collect()
        assert df.columns == ["calendar_year", "value"]
        assert df["calendar_year"].unique(maintain_order=True).to_list() == [2040, 2030]

    def test_neither_option_given_leaves_the_frame_alone(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS)
        assert stages_module.apply_sort_options(dataset.lf, None, None, dataset.dataset_dimensions) is dataset.lf

    def test_an_unknown_sort_column_is_rejected(self, dataset_factory, annual_data):
        dataset = dataset_factory(DIMENSIONS)
        with pytest.raises(base_module.AnalysisError) as excinfo:
            stages_module.apply_sort_options(dataset.lf, "not_a_dimension", None, dataset.dataset_dimensions)
        assert excinfo.value.status_code == 400
        assert "sort" in excinfo.value.detail
