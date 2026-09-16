"""Tests for the analysis-pipeline compiler/executor (api/pipeline_exec.py).

Three layers, mirroring the module's own structure:

* **Compile-time checks** (`TestCompileChecks`) -- the plain-function rules this module
  owns (dimension-name validity, `project` cardinality, `join` alignment, `filter.where`
  namespace, the row ceiling, the output single-value-column count). Exercised directly,
  no `BaseAnalysis`/file IO involved where avoidable.
* **`PairedAnalysis._prepare`/`_collapse_expr`** (`TestPairedAnalysisPrepare`) -- the per-band
  `of`-kind resolution and the per-pixel collapse expression, tested against small fake
  "dataset" stand-ins (only `.materialize()` and `.dataset_dimensions.grid` are needed)
  rather than real files, since none of this logic touches a crosswalk or a raster file.
* **End-to-end compiles and ASGI routes** (`TestCompileSingleInputChain`,
  `TestCompilePairedNotebookShape`, `TestNotebookQueryEndToEnd`,
  `TestSingleInputEndToEnd`, `TestProjectOnlyReaggregationEndToEnd`,
  `TestRejections`) -- the three MUST-WORK shapes from the spec, built on real `BaseAnalysis`
  over `tmp_path` fixtures, following `tests/factories.py`'s "genuine variables/grids"
  convention.

**On variable/grid choices below.** The paired (notebook) shape's fixtures
(`notebook_query_data`) use the EXACT variable/grid/dimension names from the spec's own
exit-criterion example (`ffibs_coverage_ratio`/`veg_grid_cell_v001` and
`sal`/`hydro_compartment_v001`) -- `tests/factories.py` has no ready-made builder for
either (its `annual_frame`/`write_raster_dataset` helpers are hardcoded to `ssf`/`vct`), so
this module writes them directly with `write_parquet`/`write_cog`, mirroring those helpers'
own shape. The single-input and project-only-reaggregation shapes below, which have no
"verbatim" requirement in the spec, reuse `tests/factories.py`'s `ssf`/`ecoregion_v001` and
`vct`/`veg_grid_cell_v001` fixtures directly, per that module's own established convention
-- the compiler cares about grid TYPES and SHAPES, not real-world variable semantics.

**A note on raster `filter` steps.** `RasterGridAnalysis._apply_transform_params_impl` rejects a
`where` value-conjunct on a raster grid outright unless paired with `summarize_by`/
`summarize_aggregate` ("a plain, non-summarized raster selection has no ... per-pixel
frame the API exposes ... at all") -- a pre-existing constraint of the machinery this
module lowers single-input chains onto, not something `pipeline_exec.py` introduces. That
means a single-input pipeline with a `filter` step over a RASTER grid that does not also
reaggregate onto a coarser grid will compile (there's nothing wrong with the pipeline
itself) but 400 at `apply_params()` -- see `TestSingleInputEndToEnd`'s vector-only choice
for the filter+group shape, and the raster case's group-only (no filter) shape below.
"""

import copy
import io
import types

import mercantile
import numpy as np
import polars as pl
import pytest
import rasterio as rio
from rasterio.warp import transform_bounds

from cpra.mp.data import config
from cpra.mp.data.analysis import paired as paired_module
from cpra.mp.data.analysis.base import AnalysisError, BaseAnalysis
from cpra.mp.data.analysis.paired import (
    PairedAnalysis,
    _validate_paired_group_aggregate,
)
from cpra.mp.data.analysis.pipeline import (
    GroupStep,
    PipelineError,
    parse_pipeline,
    pipeline_to_query,
)
from cpra.mp.data.analysis import raster as raster_module
from cpra.mp.data.analysis.raster import RasterGridAnalysis
from cpra.mp.data.analysis.vector import VectorGridAnalysis
from cpra.mp.data.analysis.planner import (
    _check_input_dimensions,
    _check_join_row_ceiling,
    _extract_post_project_grid_cell_filter,
    _project_direction,
    _validate_filter_where,
    _validate_join_alignment,
    _validate_output_columns,
    _validate_project_cardinality,
    compile_pipeline,
)
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.raster import write_cog
from cpra.mp.data.tabular import write_parquet
from tests.factories import (
    ANNUAL_KWARGS,
    RASTER_KWARGS,
    annual_frame,
    raster_array,
    write_raster_crosswalk,
    write_raster_dataset,
    write_vector_crosswalk,
)

# The analysis-pipeline design's own exit-criterion example, byte-exact (see
# tests/api/test_pipeline.py's NOTEBOOK_QUERY -- reproduced here rather than imported so
# this module doesn't reach into another test module's private constant).
NOTEBOOK_QUERY = {
    "inputs": {
        "marsh": {
            "variable": "ffibs_coverage_ratio",
            "time_unit": "annual",
            "grid": "veg_grid_cell_v001",
            "model_group_id": 500,
            "scenario_id": 7,
            "ffibs_type": "pl_fm",
        },
        "salinity": {
            "variable": "sal",
            "time_unit": "annual",
            "grid": "hydro_compartment_v001",
            "model_group_id": 500,
            "scenario_id": 7,
            "aggregate_type": "2_wk_max",
        },
    },
    "steps": [
        {"op": "project", "input": "salinity", "to": "veg_grid_cell_v001", "as": "sal"},
        {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
        {"op": "filter", "where": "marsh > 0.5 AND sal > 5.5"},
        {"op": "group", "by": [], "aggregate": "min", "of": "calendar_year"},
    ],
    "output": {"format": "tif"},
}


def _write_ffibs_raster(base_path, values_by_year):
    """`ffibs_coverage_ratio`/`veg_grid_cell_v001`, uniformly filled per year -- there is
    no `tests/factories.py` builder for this real variable, so it's written directly with
    `write_cog`, mirroring `write_raster_dataset`'s own shape.
    """
    grid = GRIDS["veg_grid_cell_v001"]
    for year, value in values_by_year.items():
        data = np.full((grid.height, grid.width), value, dtype="float32")
        write_cog(
            data,
            base_path=base_path,
            calendar_year=year,
            variable="ffibs_coverage_ratio",
            grid="veg_grid_cell_v001",
            time_unit="annual",
            model_group_id=500,
            scenario_id=7,
            model="icm_v2023.0.0",
            ffibs_type="pl_fm",
        )


def _write_ffibs_raster_arrays(base_path, arrays_by_year):
    """Like `_write_ffibs_raster`, but each year gets an arbitrary per-pixel array
    rather than a uniform fill -- needed to exercise `_write_paired_notebook_raster`'s
    streaming accumulator (`_accumulate_paired_window`) across pixels that differ in how
    many bands validate the `where` clause (all of them, some of them, or none), which a
    uniform raster can never produce.
    """
    for year, data in arrays_by_year.items():
        write_cog(
            data.astype("float32"),
            base_path=base_path,
            calendar_year=year,
            variable="ffibs_coverage_ratio",
            grid="veg_grid_cell_v001",
            time_unit="annual",
            model_group_id=500,
            scenario_id=7,
            model="icm_v2023.0.0",
            ffibs_type="pl_fm",
        )


def _write_sal_annual(base_path, values_by_year, geo_id=1, **overrides):
    """`sal`/`hydro_compartment_v001`, annual, one geo column -- no `tests/factories.py`
    builder covers annual `sal` (only `DAILY_KWARGS`/`daily_frame` do, with
    `aggregate_type="mean"`), so this mirrors `annual_frame`'s wide-then-unpivot shape
    directly. `validate=False`: `hydro_compartment_v001` has 1,657 real cells, and only
    one is needed here (`write_vector_crosswalk`/`write_raster_crosswalk`'s counterpart
    for the pairing side is the crosswalk raster itself, not this frame).
    """
    years = sorted(values_by_year)
    kwargs = {
        "model_group_id": 500,
        "scenario_id": 7,
        "aggregate_type": "2_wk_max",
        "model": "icm_v2023.0.0",
        **overrides,
    }
    df = pl.DataFrame(
        {
            "variable": pl.Series(["sal"] * len(years), dtype=pl.String),
            "grid": pl.Series(["hydro_compartment_v001"] * len(years), dtype=pl.String),
            "time_unit": pl.Series(["annual"] * len(years), dtype=pl.String),
            "model_group_id": pl.Series([kwargs["model_group_id"]] * len(years), dtype=pl.Int32),
            "scenario_id": pl.Series([kwargs["scenario_id"]] * len(years), dtype=pl.Int32),
            "aggregate_type": pl.Series([kwargs["aggregate_type"]] * len(years), dtype=pl.String),
            "calendar_year": pl.Series(years, dtype=pl.Int32),
            "model": pl.Series([kwargs["model"]] * len(years), dtype=pl.String),
            str(geo_id): pl.Series([values_by_year[y] for y in years], dtype=pl.Float32),
        }
    )
    write_parquet(df, base_path=base_path, validate=False)


@pytest.fixture
def notebook_query_data(base_path, crosswalk_path, monkeypatch):
    """Backing data for the byte-exact `NOTEBOOK_QUERY`: `marsh` (fine, raster) is 0.3 in
    2030 and 0.7 in 2040 (crosses the query's own `marsh > 0.5`); `salinity` (coarse,
    vector, one zone) is 4.0 in 2030 and 6.0 in 2040 (crosses `sal > 5.5`) -- so only 2040
    satisfies BOTH conjuncts, at every pixel (the crosswalk below maps every
    `veg_grid_cell_v001` pixel to the same single `hydro_compartment_v001` zone), making
    `{"op": "group", "aggregate": "min", "of": "calendar_year"}`'s expected output a
    UNIFORM 2040 raster -- an easy, exact assertion.
    """
    _write_ffibs_raster(base_path, {2030: 0.3, 2040: 0.7})
    _write_sal_annual(base_path, {2030: 4.0, 2040: 6.0}, geo_id=1)
    monkeypatch.setattr(config, "BASE_PATH", base_path)

    grid = GRIDS["veg_grid_cell_v001"]
    zones = np.ones((grid.height, grid.width), dtype="int32")
    write_raster_crosswalk(crosswalk_path, "veg_grid_cell_v001", "hydro_compartment_v001", zones)
    return base_path


@pytest.fixture
def accumulator_edge_case_data(base_path, crosswalk_path, monkeypatch):
    """Backing data for `TestPairedAccumulatorEdgeCases`: `sal` is a uniform 6.0 in both
    2030 and 2040 (always satisfies the query's `sal > 5.5`), while `marsh` varies by ROW
    band so `marsh > 0.5` differs pixel-by-pixel in how many of the two bands satisfy it
    -- a uniform raster (as `notebook_query_data` uses) can never validate the streaming
    accumulator's per-pixel `any_valid`/valid-count bookkeeping, only its arithmetic.
    Three equal row bands:

    * rows [0, h/3): `marsh` is 0.3 in both years -- NEVER satisfies `marsh > 0.5`, so
      the pixel has zero valid contributing bands in every aggregate (`any_valid` stays
      False) and must come out as `nodata_value` (-1 for the `of=calendar_year` dtype).
    * rows [h/3, 2h/3): `marsh` is 0.7 in 2030, 0.2 in 2040 -- satisfies the where-clause
      in ONLY 2030, so exactly one band contributes.
    * rows [2h/3, h): `marsh` is 0.6 in both years -- satisfies the where-clause in BOTH
      bands.
    """
    grid = GRIDS["veg_grid_cell_v001"]
    third = grid.height // 3
    marsh_2030 = np.empty((grid.height, grid.width), dtype="float32")
    marsh_2040 = np.empty((grid.height, grid.width), dtype="float32")
    marsh_2030[:third, :] = 0.3
    marsh_2040[:third, :] = 0.3
    marsh_2030[third : 2 * third, :] = 0.7
    marsh_2040[third : 2 * third, :] = 0.2
    marsh_2030[2 * third :, :] = 0.6
    marsh_2040[2 * third :, :] = 0.6

    _write_ffibs_raster_arrays(base_path, {2030: marsh_2030, 2040: marsh_2040})
    _write_sal_annual(base_path, {2030: 6.0, 2040: 6.0}, geo_id=1)
    monkeypatch.setattr(config, "BASE_PATH", base_path)

    zones = np.ones((grid.height, grid.width), dtype="int32")
    write_raster_crosswalk(crosswalk_path, "veg_grid_cell_v001", "hydro_compartment_v001", zones)
    return base_path, third


@pytest.fixture
def count_edge_case_data(base_path, crosswalk_path, monkeypatch):
    """Backing data for `TestPairedAnalysisCountConsistency`: five equal row bands, one per
    zero-vs-nodata case `count`'s design calls out. `sal` (zone 1 only) is a uniform 6.0
    in both years (always satisfies the query's `sal > 5.5`), so every zone-1 band's
    count outcome below is driven purely by `marsh`/nodata/pairing, exactly as
    `accumulator_edge_case_data` isolates `marsh` alone for the other aggregates.

    * rows [0, h/5): zone 1 (a real zone the coarse `sal` table has data for); `marsh` is
      0.3 in both years, never crossing `marsh > 0.5` -- ALL BANDS FAIL the `where`
      clause, in-domain -> a real count of 0.
    * rows [h/5, 2h/5): zone 1; `marsh` is `write_cog`'s own fixed fine-raster nodata
      sentinel (-9999) in both years -- FINE NODATA in every band, in-domain -> 0.
    * rows [2h/5, 3h/5): zone 2 -- a real (nonzero) crosswalk zone id, but one the coarse
      `sal` table carries no column for in EITHER year (only zone 1's "1" column is ever
      written) -- a PAIRING MISS in every band regardless of `marsh`'s own value,
      in-domain -> 0.
    * rows [3h/5, 4h/5): zone 0 -- the crosswalk raster's own nodata value
      (`write_raster_crosswalk` fixes it at 0) -- OUTSIDE the pairing crosswalk's domain
      entirely -> the nodata sentinel (-1), never a real 0, regardless of `marsh`/`sal`.
    * rows [4h/5, h): zone 1; `marsh` crosses 0.5 in 2030 only -> a MIXED pixel, exactly
      one contributing band -> a count of 1.
    """
    grid = GRIDS["veg_grid_cell_v001"]
    fifth = grid.height // 5
    zones = np.empty((grid.height, grid.width), dtype="int32")
    zones[:fifth, :] = 1  # all-bands-fail
    zones[fifth : 2 * fifth, :] = 1  # fine-nodata-all-bands
    zones[2 * fifth : 3 * fifth, :] = 2  # pairing-miss-all-bands (no coarse data for zone 2)
    zones[3 * fifth : 4 * fifth, :] = 0  # outside-zone (crosswalk nodata)
    zones[4 * fifth :, :] = 1  # mixed

    marsh_2030 = np.empty((grid.height, grid.width), dtype="float32")
    marsh_2040 = np.empty((grid.height, grid.width), dtype="float32")
    marsh_2030[:fifth, :] = 0.3
    marsh_2040[:fifth, :] = 0.3
    marsh_2030[fifth : 2 * fifth, :] = -9999.0
    marsh_2040[fifth : 2 * fifth, :] = -9999.0
    marsh_2030[2 * fifth : 3 * fifth, :] = 0.6
    marsh_2040[2 * fifth : 3 * fifth, :] = 0.6
    marsh_2030[3 * fifth : 4 * fifth, :] = 0.6
    marsh_2040[3 * fifth : 4 * fifth, :] = 0.6
    marsh_2030[4 * fifth :, :] = 0.7
    marsh_2040[4 * fifth :, :] = 0.2

    _write_ffibs_raster_arrays(base_path, {2030: marsh_2030, 2040: marsh_2040})
    # Only zone 1 ever gets a coarse `sal` column -- zone 2 is a real crosswalk id with
    # no matching data in either year, the "pairing-miss" case above.
    _write_sal_annual(base_path, {2030: 6.0, 2040: 6.0}, geo_id=1)
    monkeypatch.setattr(config, "BASE_PATH", base_path)

    write_raster_crosswalk(crosswalk_path, "veg_grid_cell_v001", "hydro_compartment_v001", zones)
    return base_path, fifth


def _tile_for_grid(base_path, grid_glob="*.tif", zoom=8):
    with rio.open(next(base_path.rglob(grid_glob))) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return mercantile.tile((west + east) / 2, (south + north) / 2, zoom)


# --------------------------------------------------------------------------------------
# Compile-time checks (pure functions, no BaseAnalysis/file IO)
# --------------------------------------------------------------------------------------


class TestCompileChecks:
    def _selection(self, **filters):
        pipeline = parse_pipeline(
            {
                "inputs": {"x": {"variable": "vct", "time_unit": "annual", "grid": "veg_grid_cell_v001", **filters}},
                "steps": [{"op": "project", "input": "x", "to": "veg_grid_cell_v001"}],
                "output": {"format": "tif"},
            }
        )
        return pipeline.inputs["x"]

    def test_check_input_dimensions_accepts_known_dimension(self):
        _check_input_dimensions("x", self._selection(model_group_id=500))

    def test_check_input_dimensions_rejects_unknown_dimension(self):
        selection = self._selection(not_a_real_dimension=1)
        with pytest.raises(PipelineError, match="unknown dimension"):
            _check_input_dimensions("x", selection)

    def test_check_input_dimensions_rejects_dataset_invalid_dimension(self):
        """`aggregate_type` is a real, GLOBALLY filterable dimension
        (`FilterDimensions.model_fields` has it, so the "unknown dimension" check above
        doesn't catch it) but is only required for a variable that lists it in
        `VARIABLES[...].required_dimensions` -- `vct` (this class's `_selection` helper's
        variable) has none, so filtering on it here is exactly the bug this check closes:
        before the fix, this input compiled clean and only failed once the pipeline
        actually scanned `vct`'s data, as a bare `ColumnNotFoundError` naming no input,
        dimension, or valid alternative (reproduced by hand against pre-fix code: a
        `vct`/`veg_grid_cell_v001` single-input pipeline filtering `aggregate_type`
        compiles to a `BaseAnalysis` and only raises
        `polars.exceptions.ColumnNotFoundError: unable to find column "aggregate_type"`
        once `BaseAnalysis.apply_params()` runs).
        """
        selection = self._selection(aggregate_type="2_wk_max")
        with pytest.raises(PipelineError, match="not one this dataset"):
            _check_input_dimensions("x", selection)

    def test_project_direction_raster_to_raster_rejected(self):
        with pytest.raises(PipelineError, match="raster-to-raster"):
            _project_direction("veg_grid_cell_v001", "morph_pixel_v001", None)

    def test_project_direction_raster_to_vector_is_many_to_one(self):
        assert _project_direction("veg_grid_cell_v001", "hydro_compartment_v001", "mean") == "many_to_one"

    def test_project_direction_vector_to_raster_is_one_to_many(self):
        assert _project_direction("ecoregion_v001", "veg_grid_cell_v001", None) == "one_to_many"

    def test_project_direction_vector_to_vector_depends_on_aggregate(self):
        assert _project_direction("ecoregion_v001", "hydro_compartment_v001", "mean") == "many_to_one"
        assert _project_direction("ecoregion_v001", "hydro_compartment_v001", None) == "one_to_many"

    def test_validate_project_cardinality_many_to_one_requires_aggregate(self):
        with pytest.raises(PipelineError, match="requires an `aggregate`"):
            _validate_project_cardinality(1, "veg_grid_cell_v001", "hydro_compartment_v001", None)

    def test_validate_project_cardinality_one_to_many_rejects_aggregate(self):
        with pytest.raises(PipelineError, match="must not specify `aggregate`"):
            _validate_project_cardinality(1, "ecoregion_v001", "veg_grid_cell_v001", "mean")

    def test_validate_project_cardinality_accepts_valid_many_to_one(self):
        direction = _validate_project_cardinality(1, "veg_grid_cell_v001", "hydro_compartment_v001", "mean")
        assert direction == "many_to_one"

    def test_validate_join_alignment_rejects_misaligned_shared_dimension(self):
        marsh = self._selection(scenario_id=7)
        salinity = self._selection(scenario_id=8)
        with pytest.raises(PipelineError, match="does not align"):
            _validate_join_alignment(2, ["marsh", "salinity"], [marsh, salinity])

    def test_validate_join_alignment_rejects_unpinned_partial_dimension(self):
        marsh = self._selection(ffibs_type=["pl_fm", "pl_bf"])
        salinity = self._selection()
        with pytest.raises(PipelineError, match="must be pinned to a single"):
            _validate_join_alignment(2, ["marsh", "salinity"], [marsh, salinity])

    def test_validate_join_alignment_accepts_aligned_and_pinned(self):
        marsh = self._selection(scenario_id=7, ffibs_type="pl_fm")
        salinity = self._selection(scenario_id=7)
        _validate_join_alignment(2, ["marsh", "salinity"], [marsh, salinity])

    def test_validate_filter_where_accepts_known_column(self):
        _validate_filter_where(3, "marsh > 0.5", {"marsh", "sal"})

    def test_validate_filter_where_accepts_unknown_dimension_column(self):
        # G8: a conjunct that does not reference the `value` placeholder is a DIMENSION
        # conjunct under `classify_where`'s split (see that function's docstring), and a
        # dimension conjunct is no longer namespace-restricted here -- exactly like the
        # flat `where=` parameter (`Dataset._apply_filter_params`), which has no
        # compile-time namespace check for a dimension conjunct either; an unrecognized
        # name fails at EXECUTION (`lf.sql()`), not at pipeline-compile time. This
        # replaces a prior version of this test (`test_validate_filter_where_rejects_
        # unknown_column`) that asserted the opposite -- that assertion was itself G8's
        # bug: it only held because every conjunct, dimension or not, was wrongly run
        # through the value-expression whitelist's namespace check.
        _validate_filter_where(3, "not_a_column > 0.5", {"marsh", "sal"})

    def test_validate_filter_where_rejects_dimension_subquery(self):
        # Still rejected -- `_check_no_dimension_subquery_or_aggregate` is exactly the
        # safety net a dimension conjunct keeps under the new split.
        with pytest.raises(PipelineError, match="subquery"):
            _validate_filter_where(3, "not_a_column in (select 1)", {"marsh", "sal"})

    def test_validate_filter_where_rejects_dimension_aggregate(self):
        with pytest.raises(PipelineError, match="aggregate function"):
            _validate_filter_where(3, "not_a_column = max(not_a_column)", {"marsh", "sal"})

    def test_validate_filter_where_accepts_dimension_not(self):
        # The actual portal shape G8 exists to unblock: `not (a = x and b = y)` is a pure
        # dimension conjunct (references no `value`), and `not` is not in
        # `ALLOWED_FUNCTIONS` -- it would have been rejected by the old (single-path)
        # implementation even though it is exactly what the unrestricted `lf.sql()` path
        # the flat `where=` surface already accepts.
        _validate_filter_where(3, "not (marsh = 1 and sal = 2)", {"marsh", "sal"})

    def test_validate_filter_where_rejects_value_conjunct_outside_whitelist(self):
        # A conjunct that DOES reference `value` is still routed through the restricted
        # value-expression whitelist -- G8 does not weaken this half at all.
        with pytest.raises(PipelineError, match="not in the allowed whitelist"):
            _validate_filter_where(3, "not (value > 1)", {"value"})

    def test_validate_filter_where_rejects_mixed_conjunct(self):
        # `classify_where` itself rejects a single conjunct that references both a
        # dimension column and `value` -- surfaced here as a `PipelineError` naming the
        # conjunct, not silently misclassified as one kind or the other.
        with pytest.raises(PipelineError, match="references both"):
            _validate_filter_where(3, "value > marsh", {"marsh", "value"})

    def test_check_join_row_ceiling_accepts_under_limit(self):
        lf = pl.LazyFrame({"grid_cell": range(10)})
        _check_join_row_ceiling(2, "eco", lf)

    def test_check_join_row_ceiling_rejects_over_limit(self, monkeypatch):
        monkeypatch.setattr(config, "ANALYSIS_JOIN_MAX_ROWS", 5)
        lf = pl.LazyFrame({"grid_cell": range(10)})
        with pytest.raises(PipelineError, match="row ceiling"):
            _check_join_row_ceiling(2, "eco", lf)

    def test_validate_output_columns_rejects_join_without_group_for_ndjson(self):
        # `pipeline.py`'s own `_validate_output_rule` only catches a `tif`/tile output
        # whose LAST step is the join; ndjson isn't single-value-column at all, so a plain
        # `Pipeline(...)` build accepts an ndjson output ending in a join -- exercising
        # `_track_output_columns` itself requires reaching past that step with something
        # that DOES narrow to one column afterward, then flipping the output format.
        pipeline = parse_pipeline(
            {
                "inputs": {
                    "marsh": {"variable": "ffibs_coverage_ratio", "time_unit": "annual", "grid": "veg_grid_cell_v001"},
                    "salinity": {"variable": "sal", "time_unit": "annual", "grid": "hydro_compartment_v001"},
                },
                "steps": [
                    {"op": "project", "input": "salinity", "to": "veg_grid_cell_v001", "as": "sal"},
                    {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
                    {"op": "filter", "where": "marsh > 0.5"},
                ],
                "output": {"format": "ndjson"},
            }
        )
        pipeline.output.format = "tif"
        with pytest.raises(PipelineError, match="produces 2"):
            _validate_output_columns(pipeline)

    def test_validate_output_columns_accepts_join_then_group(self):
        pipeline = parse_pipeline(NOTEBOOK_QUERY)
        _validate_output_columns(pipeline)

    def test_validate_paired_group_aggregate_accepts_every_streaming_aggregate(self):
        # Duck-typed, not a real `GroupStep` -- this check only ever reads `.aggregate`,
        # and (per its own docstring) is unreachable through `parse_pipeline` today since
        # `_STREAMING_AGGREGATES` is exactly `fold.DECOMPOSABLE_AGGREGATES`; exercised
        # directly here the way `TestCompileChecks`'s other members are.
        for aggregate in ("min", "max", "sum", "mean", "count"):
            _validate_paired_group_aggregate(types.SimpleNamespace(aggregate=aggregate))

    def test_validate_paired_group_aggregate_rejects_unsupported_aggregate(self):
        # This is the defense-in-depth check itself: converts what used to be a bare
        # `KeyError` out of `horizontal_aggregate_expr` (the failure mode `count` hit
        # before this change) into a plan-time `PipelineError` for ANY aggregate this
        # shape doesn't implement, present-day `count_by_value` included.
        with pytest.raises(PipelineError, match="not yet supported in analysis pipelines"):
            _validate_paired_group_aggregate(types.SimpleNamespace(aggregate="count_by_value"))

    def test_extract_post_project_grid_cell_filter_returns_none_when_absent(self):
        # An ordinary value/dimension filter that never mentions `grid_cell` at all --
        # not this function's business, fold into `Options.where` as before.
        assert _extract_post_project_grid_cell_filter(2, "calendar_year = 2030") is None

    def test_extract_post_project_grid_cell_filter_accepts_eq(self):
        assert _extract_post_project_grid_cell_filter(2, "grid_cell = 5") == "5"

    def test_extract_post_project_grid_cell_filter_accepts_in(self):
        assert _extract_post_project_grid_cell_filter(2, "grid_cell IN (5, 6, 7)") == "5,6,7"

    def test_extract_post_project_grid_cell_filter_rejects_mixed_column(self):
        with pytest.raises(PipelineError, match="mixes `grid_cell`"):
            _extract_post_project_grid_cell_filter(2, "grid_cell = 5 AND calendar_year = 2030")

    def test_extract_post_project_grid_cell_filter_rejects_non_literal_comparison(self):
        with pytest.raises(PipelineError, match="literal ID selection"):
            _extract_post_project_grid_cell_filter(2, "grid_cell > 100")


# --------------------------------------------------------------------------------------
# PairedAnalysis._prepare / _collapse_expr, against fake datasets (no crosswalk/raster IO)
# --------------------------------------------------------------------------------------


class _FakeDataset:
    def __init__(self, df, grid):
        self._df = df
        self.dataset_dimensions = types.SimpleNamespace(grid=grid)

    def materialize(self):
        return self._df


class TestPairedAnalysisPrepare:
    FINE_DF = pl.DataFrame(
        {
            "path": ["a.tif", "b.tif"],
            "band": [1, 1],
            "calendar_year": [2030, 2040],
            "model": ["icm_v2023.0.0", "icm_v2023.0.0"],
        }
    )
    COARSE_DF = pl.DataFrame({"calendar_year": [2030, 2040], "grid_cell": [1, 1], "value": [2030.01, 2040.01]})

    def _plan(self, of, where=None, aggregate="min"):
        return PairedAnalysis(
            fine_dataset=_FakeDataset(self.FINE_DF, "veg_grid_cell_v001"),
            coarse_dataset=_FakeDataset(self.COARSE_DF, "hydro_compartment_v001"),
            fine_name="marsh",
            coarse_alias="sal",
            where=where,
            group_step=GroupStep(by=[], aggregate=aggregate, of=of),
            color_map=None,
        )

    def test_numeric_dimension_of_resolves_to_dimension_kind(self):
        plan = self._plan("calendar_year")
        plan._prepare()
        assert plan.of_kind == "dimension"
        assert plan.of_values == [2030, 2040]
        assert plan.out_dtype == "int32"
        assert plan.nodata_value == -1

    def test_non_numeric_dimension_of_is_rejected(self):
        """Regression test for the dtype-guard fix: `of` naming a real but non-numeric
        dimension column (here `model`, a String) used to fall through to the
        `int32`/`-1` branch unconditionally regardless of dtype -- now it raises instead
        of silently mis-typing the output.
        """
        plan = self._plan("model")
        with pytest.raises(PipelineError, match="only a numeric dimension"):
            plan._prepare()

    def test_unknown_of_is_rejected(self):
        plan = self._plan("not_a_real_column")
        with pytest.raises(PipelineError, match="is not a column"):
            plan._prepare()

    def test_of_named_after_fine_input(self):
        plan = self._plan("marsh")
        plan._prepare()
        assert plan.of_kind == "named"
        assert plan.of_name == "marsh"
        assert plan.out_dtype == "float32"

    def test_of_named_after_coarse_alias(self):
        plan = self._plan("sal")
        plan._prepare()
        assert plan.of_kind == "named"
        assert plan.of_name == "sal"

    def test_collapse_expr_with_where_picks_first_satisfying_band(self):
        plan = self._plan("calendar_year", where="marsh > 35 AND sal > 2035")
        plan._prepare()
        # Row 1: only band 1 (marsh=40, sal=2040.01) satisfies -> 2040.
        # Row 2: both bands satisfy -> min(2030, 2040) = 2030.
        df = pl.DataFrame({"b0": [30, 40], "b1": [40, 40], "p0": [2030.01, 2036.0], "p1": [2040.01, 2036.0]})
        result = df.select(plan._collapse_expr())["value"].to_list()
        assert result == [2040, 2030]

    def test_collapse_expr_nodata_when_nothing_satisfies(self):
        plan = self._plan("calendar_year", where="marsh > 1000")
        plan._prepare()
        df = pl.DataFrame({"b0": [30], "b1": [40], "p0": [2030.01], "p1": [2040.01]})
        result = df.select(plan._collapse_expr())["value"].to_list()
        assert result == [-1]  # int32 nodata sentinel

    def test_count_rejects_non_default_of(self):
        # `of`'s pydantic default is "value" -- `pipeline.RESERVED_NAMES` forbids any
        # real input/step from being named that, so `of != "value"` unambiguously means
        # the caller set it explicitly. `count` has no aggregand for `of` to name.
        plan = self._plan("calendar_year", aggregate="count")
        with pytest.raises(PipelineError, match="has no aggregand"):
            plan._prepare()

    def test_count_with_default_of_resolves_to_count_kind(self):
        plan = self._plan("value", aggregate="count")
        plan._prepare()
        assert plan.of_kind == "count"
        assert plan.out_dtype == "int32"
        assert plan.nodata_value == -1

    def test_collapse_expr_count_with_where_tallies_satisfying_bands(self):
        plan = self._plan("value", where="marsh > 35 AND sal > 2035", aggregate="count")
        plan._prepare()
        # Row 1: band 0 (marsh=30) fails, band 1 (marsh=40, sal=2040.01) satisfies -> 1.
        # Row 2: both bands (marsh=40, sal=2036.0) satisfy -> 2.
        df = pl.DataFrame({"b0": [30, 40], "b1": [40, 40], "p0": [2030.01, 2036.0], "p1": [2040.01, 2036.0]})
        result = df.select(plan._collapse_expr())["value"].to_list()
        assert result == [1, 2]

    def test_collapse_expr_count_with_no_where_tallies_non_null_bands(self):
        # With no `where` clause at all, a band "contributes" to `count` whenever it has
        # real fine AND paired data -- `_band_valid_expr`'s no-`where` fallback.
        plan = self._plan("value", aggregate="count")
        plan._prepare()
        df = pl.DataFrame({"b0": [1.0, None], "b1": [None, None], "p0": [10.0, 10.0], "p1": [20.0, None]})
        result = df.select(plan._collapse_expr())["value"].to_list()
        # Row 1: band 0 valid (b0, p0 both present); band 1 invalid (b1 null) -> 1.
        # Row 2: band 0 invalid (b0 null); band 1 invalid (b1 and p1 both null) -> 0.
        assert result == [1, 0]


# --------------------------------------------------------------------------------------
# Shape 1: single-input chains
# --------------------------------------------------------------------------------------


class TestCompileSingleInputChain:
    # A test used to live here exercising `_compile_single_input_chain`'s own "first step
    # must be `project`" rejection, by constructing a `Pipeline` directly (bypassing
    # `parse_pipeline`/`validate_pipeline`) with a step-1 `group`. Removed along with the
    # code path it exercised: see `_compile_single_input_chain`'s own comment, where that
    # check used to raise, for why it's genuinely unreachable through any
    # `parse_pipeline`-validated pipeline rather than merely shadowed by call order.

    def test_rejects_identity_project_with_aggregate(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [{"op": "project", "input": "eco", "to": "ecoregion_v001", "aggregate": "mean"}],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(PipelineError, match="no aggregate is possible"):
            compile_pipeline(pipeline, base_path=base_path)

    def test_rejects_broadcast_for_single_input(self, base_path):
        """A raster->vector `project` is always `_project_direction`'s "many_to_one"
        branch (so it hits the "requires an `aggregate`" rule instead, tested above via
        `test_validate_project_cardinality_many_to_one_requires_aggregate`) -- the
        broadcast (one-to-many) rejection this single-input chain owns is only reachable
        the other way around, vector->raster, which is unconditionally "one_to_many"
        regardless of whether `aggregate` was given.
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [{"op": "project", "input": "eco", "to": "veg_grid_cell_v001"}],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(PipelineError, match="broadcasts onto a finer grid"):
            compile_pipeline(pipeline, base_path=base_path)

    async def test_filter_after_group_compiles_and_executes_for_vector_input(self, base_path):
        """VECTOR ORDERING RELAXATION: a `filter` after
        a `group` step used to be rejected unconditionally -- it is now accepted for a
        VECTOR input (`ecoregion_v001` here), since `VectorGridAnalysis.apply_steps`
        applies a post-`GroupOp` `FilterOp` against the already-collapsed rows, in
        declared order, with no shared `Options` bucket to collide over (see
        `planner._compile_single_input_chain`'s `not is_vector_input` guard on this
        rejection for the full argument). Compiling is not the point here -- correctness
        is -- so this test hand-computes the hand-verifiable hand-computed answer, not
        just the absence of an exception.

        DECLARED (and, since the interpreter runs ops in declared order, ACTUAL): a
        single year means `group(by=[], aggregate=max)` collapses each grid_cell's ONE
        row onto itself unchanged (1.0, 2.0, 3.0, 4.0 for geo_ids 1-4) -- then
        `filter(eco > 2)` keeps only grid_cell 3 (3.0) and grid_cell 4 (4.0).
        """
        write_parquet(
            annual_frame(years=(2030,), geo_ids=(1, 2, 3, 4), value_fn=lambda _year, geo_id: float(geo_id)),
            base_path=base_path,
            validate=False,
        )
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                    {"op": "group", "by": [], "aggregate": "max"},
                    {"op": "filter", "where": "eco > 2"},
                ],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert isinstance(compiled, BaseAnalysis)
        await compiled.apply_params()
        df = compiled.materialize()
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))
        assert actual == {3: 3.0, 4: 4.0}

    def test_accepts_filter_after_group_for_raster_input(self, base_path):
        """The raster side of the vector ordering relaxation above, now lifted too: the
        raster interpreter records a post-group dimension filter at its own position and
        narrows the running band metadata with the same primitive the pixel-read replay
        uses, so nothing indexes a band by a position captured before the group. Only the
        compile is asserted here (the shape no longer raises) -- the answer it produces is
        pinned by
        `TestRasterInterpreterDeclaredOrder::test_dimension_filter_after_group_selects_from_the_grouped_bands`.
        """
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda y: y - 2000)
        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "group", "by": ["calendar_year"], "aggregate": "max"},
                    {"op": "filter", "where": "calendar_year = 2030"},
                ],
                "output": {"format": "tif"},
            }
        )
        assert isinstance(compile_pipeline(pipeline, base_path=base_path), BaseAnalysis)

    def test_rejects_filter_after_group_for_a_grid_cell_raster_input(self, base_path):
        """A `grid_cell`-bearing raster selection is the one shape that still executes
        `BaseAnalysis.apply_params`'s fixed five-stage flow (`_apply_grid_cell` reshapes
        `self.lf` to a long frame before any op could run, so the planner withholds
        `_pipeline_ops` -- see
        `TestRasterInterpreterLoadBearing::test_raster_input_with_grid_cell_stays_on_the_old_engine`).
        That flow has a single Filter stage running strictly before its Group stage, so a
        filter declared after a group would silently narrow the PRE-group bands -- the
        rejection stays armed for it alone. Pure compile-time rejection (like
        `TestPostProjectGridCellFilterRejections` above, which follows the same pattern) --
        it fires inside `_compile_single_input_chain`'s step loop, before `_dataset_for`
        ever needs a crosswalk or a raster file to exist, so none is written here.
        """
        pipeline = parse_pipeline(
            {
                "inputs": {
                    "vct": {
                        **{k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")},
                        "grid_cell": "-91.0,29.5",
                    }
                },
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "group", "by": [], "aggregate": "max"},
                    {"op": "filter", "where": "vct > 2035"},
                ],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(PipelineError, match="comes after a group step"):
            compile_pipeline(pipeline, base_path=base_path)

    def test_rejects_group_of_other_than_value(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                    {"op": "group", "by": [], "aggregate": "max", "of": "calendar_year"},
                ],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(PipelineError, match="only aggregate its own running value"):
            compile_pipeline(pipeline, base_path=base_path)

    async def test_two_group_steps_compile_and_execute_for_vector_input(self, base_path):
        """VECTOR ORDERING RELAXATION: the "more than
        one `group` step" limit is lifted for a VECTOR input -- `Options.group_by`/
        `group_aggregate` are scalar fields, so a SECOND `group` step had nowhere to
        lower onto but to collide with the first; the vector interpreter has no such
        bucket, applying each `GroupOp` in declared order against whatever the previous
        op left behind (see `planner._compile_single_input_chain`'s `not is_vector_input`
        guard on this rejection for the full argument).

        DECLARED (and, since the interpreter runs ops in declared order, ACTUAL): two
        years (2030, 2040) mean the FIRST `group(by=[], aggregate=max)` collapses each
        grid_cell's two rows into one -- grid_cell 1: max(10.0, 11.0) = 11.0; grid_cell
        2: max(20.0, 21.0) = 21.0. The SECOND `group(by=[], aggregate=min)` then runs
        against that already-one-row-per-grid_cell result, where min of a single value
        is that same value -- so it is a no-op here, but still proves the second group
        step is actually applied (not silently dropped) against the first group's
        OUTPUT, not the original per-year rows.
        """
        write_parquet(
            annual_frame(
                years=(2030, 2040),
                geo_ids=(1, 2),
                value_fn=lambda year, geo_id: float(geo_id) * 10 + (0.0 if year == 2030 else 1.0),
            ),
            base_path=base_path,
            validate=False,
        )
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                    {"op": "group", "by": [], "aggregate": "max", "as": "g1"},
                    {"op": "group", "by": [], "aggregate": "min"},
                ],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert isinstance(compiled, BaseAnalysis)
        await compiled.apply_params()
        df = compiled.materialize()
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))
        assert actual == {1: 11.0, 2: 21.0}

    def test_accepts_two_group_steps_for_raster_input(self, base_path):
        """The raster side of the vector ordering relaxation above, now lifted too: each
        group is APPLIED to the running band metadata and recorded as its own entry, so a
        second group collapses the first one's output instead of replacing it. The answer
        is pinned by
        `TestRasterInterpreterDeclaredOrder::test_a_second_group_collapses_the_first_groups_output`.
        """
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda y: y - 2000)
        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "group", "by": ["calendar_year"], "aggregate": "max", "as": "g1"},
                    {"op": "group", "by": [], "aggregate": "min"},
                ],
                "output": {"format": "tif"},
            }
        )
        assert isinstance(compile_pipeline(pipeline, base_path=base_path), BaseAnalysis)

    def test_rejects_two_group_steps_for_a_grid_cell_raster_input(self, base_path):
        """Still rejected for the one raster shape that stays on the fixed-stage engine
        (see `test_rejects_filter_after_group_for_a_grid_cell_raster_input` above): that
        engine has a single `Options.group_by`/`group_aggregate` slot and calls
        `_collapse_group` exactly once, so a second group would silently replace the first.
        Pure compile-time rejection, like `TestPostProjectGridCellFilterRejections` -- no
        data file needed.
        """
        pipeline = parse_pipeline(
            {
                "inputs": {
                    "vct": {
                        **{k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")},
                        "grid_cell": "-91.0,29.5",
                    }
                },
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "group", "by": [], "aggregate": "max", "as": "g1"},
                    {"op": "group", "by": [], "aggregate": "min"},
                ],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(PipelineError, match="more than one `group` step"):
            compile_pipeline(pipeline, base_path=base_path)

    def test_accepts_identity_project_returns_a_dataset(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [{"op": "project", "input": "eco", "to": "ecoregion_v001"}],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert isinstance(compiled, BaseAnalysis)


class TestCalculateStepCompile:
    """`calculate` steps in a single-input chain compile onto `Options.calculate` -- the
    same generic entries the flat `calculate.N.*` parameters populate -- in step order,
    with the running value's in-scope name (the input/step name, or the `value`
    placeholder) rewritten to `value` exactly like `filter.where`."""

    def _pipeline(self, steps):
        return parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [{"op": "project", "input": "eco", "to": "ecoregion_v001"}, *steps],
                "output": {"format": "csv"},
            }
        )

    def test_compiles_onto_generic_calculate_entries(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        pipeline = self._pipeline(
            [
                {"op": "calculate", "expression": "eco * 0.6", "where": "calendar_year = 2030 AND eco > 1"},
                {"op": "calculate", "expression": "value * 0.5"},
            ]
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert [dict(e) for e in compiled.options.calculate] == [
            {"expression": "value * 0.6", "where": "calendar_year = 2030 AND value > 1"},
            {"expression": "value * 0.5", "where": None},
        ]

    async def test_calculate_after_group_compiles_and_executes_for_vector_input(self, base_path):
        """VECTOR ORDERING RELAXATION: a `calculate`
        after a `group` step used to be rejected unconditionally -- now accepted for a
        VECTOR input, since `VectorGridAnalysis.apply_steps` applies a post-`GroupOp`
        `CalculateOp` against the already-collapsed rows, in declared order (see
        `planner._compile_single_input_chain`'s `not is_vector_input` guard on this
        rejection). Compiling is not the point -- correctness is.

        DECLARED (and ACTUAL): a single year means `group(by=[], aggregate=max)`
        collapses each grid_cell's one row onto itself unchanged (1.0, 2.0, 3.0, 4.0 for
        geo_ids 1-4) -- then `calculate(eco * 0.6)` rewrites each: 0.6, 1.2, 1.8, 2.4.
        """
        write_parquet(
            annual_frame(years=(2030,), geo_ids=(1, 2, 3, 4), value_fn=lambda _year, geo_id: float(geo_id)),
            base_path=base_path,
            validate=False,
        )
        pipeline = self._pipeline(
            [
                {"op": "group", "by": [], "aggregate": "max"},
                {"op": "calculate", "expression": "eco * 0.6"},
            ]
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert isinstance(compiled, BaseAnalysis)
        await compiled.apply_params()
        df = compiled.materialize()
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))
        assert actual == pytest.approx({1: 0.6, 2: 1.2, 3: 1.8, 4: 2.4})

    def test_accepts_calculate_after_group_for_raster_input(self, base_path):
        """The raster side of the vector ordering relaxation above, now lifted too: the
        pending calculate entries are flushed into the recorded op list at the position
        they were declared at, so nothing applies them ahead of the collapse any more. The
        answer is pinned by
        `TestRasterInterpreterDeclaredOrder::test_calculate_after_group_squares_the_mean_not_the_other_way_round`.
        """
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda y: y - 2000)
        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "group", "by": [], "aggregate": "max"},
                    {"op": "calculate", "expression": "vct * 0.6"},
                ],
                "output": {"format": "tif"},
            }
        )
        assert isinstance(compile_pipeline(pipeline, base_path=base_path), BaseAnalysis)

    def test_rejects_calculate_after_group_for_a_grid_cell_raster_input(self, base_path):
        """Still rejected for the one raster shape that stays on the fixed-stage engine:
        that engine's Calculate stage (2) runs unconditionally before its Group stage (3),
        so this would compute the aggregate of the expression instead of the expression of
        the aggregate. Pure compile-time rejection -- no data file needed.
        """
        pipeline = parse_pipeline(
            {
                "inputs": {
                    "vct": {
                        **{k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")},
                        "grid_cell": "-91.0,29.5",
                    }
                },
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "group", "by": [], "aggregate": "max"},
                    {"op": "calculate", "expression": "vct * 0.6"},
                ],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(PipelineError, match="comes after a group step"):
            compile_pipeline(pipeline, base_path=base_path)

    async def test_value_filter_after_calculate_compiles_and_executes_for_vector_input(self, base_path):
        """VECTOR ORDERING RELAXATION: a value-testing
        `filter` after a `calculate` step used to be rejected unconditionally -- now
        accepted for a VECTOR input, since `VectorGridAnalysis.apply_steps` applies a
        post-`CalculateOp` `FilterOp` against the already-recalculated `value` column, in
        declared order (see `planner._compile_single_input_chain`'s `not is_vector_input`
        guard on this rejection). Compiling is not the point -- correctness is.

        DECLARED (and ACTUAL): `calculate(eco * 0.6)` rewrites geo_ids 1-4's values
        (1.0, 2.0, 3.0, 4.0) to 0.6, 1.2, 1.8, 2.4 -- then `filter(eco > 1)` tests the
        POST-calculate value (not the pre-calculate one), keeping geo_ids 2, 3, 4 (1.2,
        1.8, 2.4) and dropping geo_id 1 (0.6, which does not exceed 1).
        """
        write_parquet(
            annual_frame(years=(2030,), geo_ids=(1, 2, 3, 4), value_fn=lambda _year, geo_id: float(geo_id)),
            base_path=base_path,
            validate=False,
        )
        pipeline = self._pipeline(
            [
                {"op": "calculate", "expression": "eco * 0.6"},
                {"op": "filter", "where": "eco > 1"},
            ]
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert isinstance(compiled, BaseAnalysis)
        await compiled.apply_params()
        df = compiled.materialize()
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))
        assert actual == pytest.approx({2: 1.2, 3: 1.8, 4: 2.4})

    def test_accepts_value_filter_after_calculate_for_raster_input(self, base_path):
        """The raster side of the vector ordering relaxation above, now lifted too: the
        pending calculate entries are flushed into the recorded op list before the filter's
        own masking entry is appended, so the mask tests the REWRITTEN pixel value. The
        answer is pinned by
        `TestRasterInterpreterDeclaredOrder::test_value_filter_after_calculate_masks_the_rewritten_value`
        (which pairs the same two steps with the summarizing `project` a raster value
        filter needs -- `_apply_transform_params_impl` rejects a standalone one, as
        `TestRasterInterpreterDifferential::
        test_standalone_value_filter_without_summarize_still_rejected_by_both_engines`
        pins, and that rejection is a runtime concern unaffected by this compile).
        """
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda y: y - 2000)
        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "calculate", "expression": "vct * 0.6"},
                    {"op": "filter", "where": "vct > 100"},
                ],
                "output": {"format": "csv"},
            }
        )
        assert isinstance(compile_pipeline(pipeline, base_path=base_path), BaseAnalysis)

    def test_rejects_value_filter_after_calculate_for_a_grid_cell_raster_input(self, base_path):
        """Still rejected for the one raster shape that stays on the fixed-stage engine:
        that engine applies a single `Options.where` in the Filter stage (1), before the
        Calculate stage (2), so the mask would test the PRE-calculate value. Pure
        compile-time rejection -- no data file needed.
        """
        pipeline = parse_pipeline(
            {
                "inputs": {
                    "vct": {
                        **{k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")},
                        "grid_cell": "-91.0,29.5",
                    }
                },
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "calculate", "expression": "vct * 0.6"},
                    {"op": "filter", "where": "vct > 100"},
                ],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(PipelineError, match="pre-calculate"):
            compile_pipeline(pipeline, base_path=base_path)

    def test_allows_dimension_filter_after_calculate(self, base_path):
        """A dimension condition commutes with calculate (calculate never touches
        dimensions), so it stays legal in either order."""
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        pipeline = self._pipeline(
            [
                {"op": "calculate", "expression": "eco * 0.6"},
                {"op": "filter", "where": "calendar_year = 2030"},
            ]
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert isinstance(compiled, BaseAnalysis)

    def test_rejects_invalid_expression_naming_the_step(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        pipeline = self._pipeline([{"op": "calculate", "expression": "scenario_id * 2"}])
        with pytest.raises(PipelineError, match=r"step 2 \(calculate\)"):
            compile_pipeline(pipeline, base_path=base_path)


class TestInputGridCellCompile:
    """`grid_cell` on an `InputSelection` is a reserved Filter-group field (not a
    dimension filter -- see `TestInputSelectionGridCell` in test_pipeline.py), forwarded
    unchanged onto `Options.grid_cell` by `planner.py::_dataset_for`."""

    def test_lowers_onto_options_grid_cell(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        pipeline = parse_pipeline(
            {
                "inputs": {
                    "eco": {
                        **{k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")},
                        "grid_cell": "5,6",
                    }
                },
                "steps": [{"op": "project", "input": "eco", "to": "ecoregion_v001"}],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert compiled.options.grid_cell == "5,6"

    def test_omitted_grid_cell_leaves_options_grid_cell_unset(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [{"op": "project", "input": "eco", "to": "ecoregion_v001"}],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert compiled.options.grid_cell is None


class TestOutputSortSelectCompile:
    """`output.sort`/`output.select` lower onto exactly `Options.sort`/`.select` -- and
    must do so for BOTH single-input-chain sub-shapes: a chain with a `filter`/`group`
    tail, and the degenerate project-only-reaggregation shape (no `steps[1:]` at all),
    since the lowering sits unconditionally at the tail of
    `_compile_single_input_chain` rather than inside the per-step loop.
    """

    def test_lowers_for_a_chain_with_filter_and_group(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                    {"op": "filter", "where": "eco > 0"},
                    {"op": "group", "by": [], "aggregate": "max"},
                ],
                "output": {"format": "csv", "sort": "-value", "select": "grid_cell,value"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert compiled.options.sort == "-value"
        assert compiled.options.select == "grid_cell,value"

    def test_lowers_for_project_only_reaggregation(self, base_path, crosswalk_path):
        grid = GRIDS[RASTER_KWARGS["grid"]]
        zones = np.zeros((grid.height, grid.width), dtype="int32")
        zones[: grid.height // 2, :] = 1
        zones[grid.height // 2 :, :] = 2
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _y: raster_array())

        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"}
                ],
                "output": {"format": "csv", "sort": "-value"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert compiled.options.sort == "-value"

    def test_omitted_sort_select_leave_options_unset(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [{"op": "project", "input": "eco", "to": "ecoregion_v001"}],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert compiled.options.sort is None
        assert compiled.options.select is None


class TestOutputColorMapCompile:
    """`output.color_map` (G9) lowers onto exactly `Options.color_map` for a single-input
    chain -- `_compile_single_input_chain` used to never set it at all, so `color_map` had
    no effect on any single-input pipeline (only the paired shape, `_compile_paired_
    notebook_shape`, forwarded it, onto `PairedAnalysis.color_map` directly). Mirrors
    `TestOutputSortSelectCompile`'s two sub-shapes -- the lowering sits unconditionally at
    the tail of `_compile_single_input_chain`, alongside `sort`/`select`, so both the
    ordinary chain and the degenerate project-only-reaggregation shape must pick it up.
    """

    def test_lowers_for_a_chain_with_filter_and_group(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                    {"op": "filter", "where": "eco > 0"},
                    {"op": "group", "by": [], "aggregate": "max"},
                ],
                "output": {"format": "csv", "color_map": "ffibs"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert compiled.options.color_map == "ffibs"

    def test_lowers_for_project_only_reaggregation(self, base_path, crosswalk_path):
        grid = GRIDS[RASTER_KWARGS["grid"]]
        zones = np.zeros((grid.height, grid.width), dtype="int32")
        zones[: grid.height // 2, :] = 1
        zones[grid.height // 2 :, :] = 2
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _y: raster_array())

        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"}
                ],
                "output": {"format": "csv", "color_map": "ffibs"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert compiled.options.color_map == "ffibs"

    def test_omitted_color_map_leaves_options_unset(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [{"op": "project", "input": "eco", "to": "ecoregion_v001"}],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert compiled.options.color_map is None


class TestProjectAggregateCountByValueCompile:
    """`count_by_value` as a `project` aggregate -- a ZONAL aggregate, not subject to
    `fold.DECOMPOSABLE_AGGREGATES` (see `pipeline.py::_validate_step_rules`'s `ProjectStep`
    branch) -- lowers onto exactly the same `Options.summarize_aggregate` a `mean`/`sum`
    project aggregate would.
    """

    def test_lowers_onto_summarize_aggregate(self, base_path):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {
                        "op": "project",
                        "input": "eco",
                        "to": "hydro_compartment_v001",
                        "aggregate": "count_by_value",
                    }
                ],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert compiled.options.summarize_by == "hydro_compartment_v001"
        assert compiled.options.summarize_aggregate == "count_by_value"


class TestPostProjectOrderingGuard:
    """SUMMARIZING-PROJECT ORDER GUARD (see `planner._compile_single_input_chain`'s
    comment of that name for the full argument): `_compile_single_input_chain` lowers a
    pipeline's whole `project -> ...` chain onto `Options`, whose stage order is FIXED
    regardless of declared step order: filter(1) -> calculate(2) -> group(3) ->
    transform(4) -> sort(5) (`BaseAnalysis.apply_params`, base.py:606-612). A `project`
    step's `summarize_by`/`summarize_aggregate` lower onto the TRANSFORM stage (4th), so
    any `filter`/`calculate`/`group` DECLARED after a many-to-one (summarizing) `project`
    actually RUNS before it -- silently, with no error, producing a plausible-looking
    wrong answer. This class pins the compile-time rejections that close
    that hole, using a worked example byte-for-byte: ecoregions 1-4 valued
    1.0/2.0/3.0/4.0, crosswalked (1,2)->hydro_compartment 100 and (3,4)->200 at scale
    1.0, `aggregate="sum"`.

    This guard must not change behavior for an IDENTITY project
    (`to` equal to the input's own grid) -- it sets no `summarize_by` at all, so there is
    no stage-4 work for anything to run "before". Every registered analysis
    (`difference`/`concatenate`/`exit_criterion`) projects to its own grid, which is
    exactly why the full existing suite (none of it touches a cross-grid project
    followed by a filter/calculate/group) must stay green with these rejections in
    place -- see `test_identity_project_is_unaffected` below for the explicit version
    of that claim.
    """

    def _write_fixture(self, base_path, crosswalk_path):
        write_parquet(
            annual_frame(years=(2030,), geo_ids=(1, 2, 3, 4), value_fn=lambda _year, geo_id: float(geo_id)),
            base_path=base_path,
            validate=False,
        )
        write_vector_crosswalk(
            crosswalk_path,
            "ecoregion_v001",
            "hydro_compartment_v001",
            [(1, 100, 1.0), (2, 100, 1.0), (3, 200, 1.0), (4, 200, 1.0)],
        )

    def _pipeline(self, steps_after_project):
        return parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "eco", "to": "hydro_compartment_v001", "aggregate": "sum"},
                    *steps_after_project,
                ],
                "output": {"format": "csv"},
            }
        )

    def test_rejects_value_filter_after_summarizing_project(self, base_path, crosswalk_path):
        # DECLARED (sum, then keep value > 2): {100: 3.0, 200: 7.0} (1.0+2.0=3.0 kept
        # since 3.0>2; 3.0+4.0=7.0 kept since 7.0>2).
        # ACTUAL, absent this rejection: the
        # flat machinery runs its whole Filter stage before the project's Transform-stage
        # crosswalk join, so the filter drops ecoregion 1 and 2's SOURCE rows (1.0 and
        # 2.0 both fail `> 2`) before the join ever sees them -- compartment 100 then
        # sums an EMPTY set to 0.0: {100: 0.0, 200: 7.0}. No error raised; a
        # plausible-looking wrong answer. This is exactly why it must be rejected at
        # compile time rather than left as an unimplemented feature.
        self._write_fixture(base_path, crosswalk_path)
        pipeline = self._pipeline([{"op": "filter", "where": "eco > 2"}])
        with pytest.raises(PipelineError, match="tests the running value after a many-to-one"):
            compile_pipeline(pipeline, base_path=base_path)

    def test_rejects_calculate_after_summarizing_project(self, base_path, crosswalk_path):
        # DECLARED (sum, then square): {100: 9.0, 200: 49.0} (3.0**2, 7.0**2).
        # ACTUAL, absent this rejection:
        # `calculate` lowers onto the Calculate stage (2nd), before the project's
        # Transform-stage (4th) aggregation, so each SOURCE cell is squared before
        # summing: {100: 1.0+4.0=5.0, 200: 9.0+16.0=25.0}. Silently wrong, no error.
        self._write_fixture(base_path, crosswalk_path)
        pipeline = self._pipeline([{"op": "calculate", "expression": "eco * eco"}])
        with pytest.raises(PipelineError, match="comes after a many-to-one"):
            compile_pipeline(pipeline, base_path=base_path)

    def test_rejects_group_after_summarizing_project(self, base_path, crosswalk_path):
        # This test doesn't work out numbers the way the filter/calculate tests above
        # do, but the mechanism pinning this rejection is identical: `group` lowers onto
        # `Options.group_by`/`group_aggregate`, applied in the Group stage (3rd),
        # still strictly before the project's Transform-stage (4th) aggregation
        # regardless of declared order, so it would collapse pre-project source rows
        # rather than the declared post-project zones.
        self._write_fixture(base_path, crosswalk_path)
        pipeline = self._pipeline([{"op": "group", "by": [], "aggregate": "max"}])
        with pytest.raises(PipelineError, match="comes after a many-to-one"):
            compile_pipeline(pipeline, base_path=base_path)

    def test_leading_filter_without_explicit_input_still_rejected(self, base_path, crosswalk_path):
        """Gaining an optional `input` field did not make it implicit: a `filter` at
        step 1 that omits `input` still fails exactly the pre-existing chaining-rule
        message (`pipeline.py`'s `validate_pipeline`) -- a leading filter must still
        name its input explicitly, exactly the way a leading `project` already must.
        This is the schema-level (`parse_pipeline`) half of what makes a leading filter
        possible at all; `test_leading_filter_before_project_compiles_and_executes_
        declared_order` below is the planner-level (`compile_pipeline`) half.
        """
        self._write_fixture(base_path, crosswalk_path)
        with pytest.raises(PipelineError, match="has no input to chain from"):
            parse_pipeline(
                {
                    "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                    "steps": [
                        {"op": "filter", "where": "eco > 1"},
                        {"op": "project", "to": "hydro_compartment_v001", "aggregate": "sum"},
                    ],
                    "output": {"format": "csv"},
                }
            )

    async def test_leading_filter_before_project_compiles_and_executes_declared_order(
        self, base_path, crosswalk_path
    ):
        """The remediation this whole class's guard exists to unblock: `filter` naming
        its `input` explicitly as step 1, BEFORE the many-to-one `project` -- the
        spelling `test_rejects_value_filter_after_summarizing_project`'s rejection
        message above points authors at ("move a value condition before the project
        step"). Declared and executed order now agree by construction: the filter
        lowers onto `Options.where`, applied in the Filter stage (stage 1), strictly
        before the project's Transform-stage aggregation (stage 4) -- there is no
        reordering left to verify against, unlike the post-project case above.

        DECLARED (and, since the filter genuinely runs first, ACTUAL): keep eco >= 2
        (drops ecoregion 1's value 1.0 as a SOURCE row), then sum per compartment --
        compartment 100 keeps only ecoregion 2 (value 2.0) -> 2.0; compartment 200 keeps
        ecoregions 3 and 4 (3.0 + 4.0) -> 7.0.
        """
        self._write_fixture(base_path, crosswalk_path)
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "filter", "input": "eco", "where": "eco >= 2"},
                    {"op": "project", "to": "hydro_compartment_v001", "aggregate": "sum"},
                ],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert isinstance(compiled, BaseAnalysis)
        assert compiled.options.where == "(value >= 2)"
        assert compiled.options.summarize_by == "hydro_compartment_v001"
        await compiled.apply_params()
        df = compiled.materialize()
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))
        assert actual == {100: 2.0, 200: 7.0}

    def test_leading_filter_does_not_exempt_a_later_value_filter_from_the_guard(
        self, base_path, crosswalk_path
    ):
        """The guard must survive intact: adding the leading-filter acceptance must not
        accidentally widen it to also accept a value filter declared AFTER the project.
        A `filter` BEFORE the project is accepted (classified as a leading filter, never
        reaching the `is_summarizing_project` check at all), while a second `filter`
        placed AFTER the same project is still rejected exactly as
        `test_rejects_value_filter_after_summarizing_project` pins above.
        """
        self._write_fixture(base_path, crosswalk_path)
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "filter", "input": "eco", "where": "eco > 0"},
                    {"op": "project", "to": "hydro_compartment_v001", "aggregate": "sum"},
                    {"op": "filter", "where": "eco > 2"},
                ],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(PipelineError, match="tests the running value after a many-to-one"):
            compile_pipeline(pipeline, base_path=base_path)

    def test_accepts_dimension_only_filter_after_summarizing_project(self, base_path, crosswalk_path):
        """A DIMENSION condition commutes with a many-to-one `project` -- a projection
        rewrites `grid_cell` and `value` (it re-keys rows onto a coarser zone and
        re-aggregates their values), never a dimension column -- the identical argument
        the pre-existing calculate-then-filter rule already relies on (planner.py's
        comment near the `after_calculate` check). Stays accepted, folded into
        `Options.where` exactly as before.
        """
        self._write_fixture(base_path, crosswalk_path)
        pipeline = self._pipeline([{"op": "filter", "where": "calendar_year = 2030"}])
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert isinstance(compiled, BaseAnalysis)
        assert compiled.options.where == "(calendar_year = 2030)"

    def test_grid_cell_filter_after_summarizing_project_is_unaffected(self, base_path, crosswalk_path):
        """The pre-existing `summarize_grid_cell` special-case
        (`_extract_post_project_grid_cell_filter`) is untouched by this guard -- a
        `grid_cell` filter after a summarizing `project` is still routed onto
        `Options.summarize_grid_cell`, the one post-project filter shape the compiler
        already honors correctly rather than rejecting.
        """
        self._write_fixture(base_path, crosswalk_path)
        pipeline = self._pipeline([{"op": "filter", "where": "grid_cell = 100"}])
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert compiled.options.summarize_grid_cell == "100"

    async def test_second_post_project_grid_cell_filter_compiles_and_executes_for_vector_input(
        self, base_path, crosswalk_path
    ):
        """VECTOR ORDERING RELAXATION: a SECOND
        post-project `grid_cell` filter used to be rejected unconditionally ("only one is
        supported") -- now accepted for a VECTOR input. `Options.summarize_grid_cell` is
        a single scalar bucket, but it stopped being what actually filters the
        materialized rows once the vector interpreter landed: `apply_steps` applies each
        post-`ProjectOp` `FilterOp` -- including a `grid_cell` one -- directly against
        the already-projected `grid_cell` column, in declared order, same as any other
        filter (see `planner._compile_single_input_chain`'s `not is_vector_input` guard
        on this rejection, and `TestVectorInterpreterDifferential
        .test_summarizing_project_then_grid_cell_filter`, which already pins the single-
        filter case frame-for-frame against the legacy engine).

        DECLARED (and, since the interpreter runs ops in declared order, ACTUAL):
        `project(sum)` gives {100: 3.0 (1.0+2.0), 200: 7.0 (3.0+4.0)} -- the FIRST filter
        `grid_cell IN (100, 200)` keeps both zones unchanged -- the SECOND filter
        `grid_cell = 100` then narrows further, to just zone 100.
        """
        self._write_fixture(base_path, crosswalk_path)
        pipeline = self._pipeline(
            [
                {"op": "filter", "where": "grid_cell IN (100, 200)"},
                {"op": "filter", "where": "grid_cell = 100"},
            ]
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert isinstance(compiled, BaseAnalysis)
        await compiled.apply_params()
        df = compiled.materialize()
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))
        assert actual == {100: 3.0}

    async def test_identity_project_is_unaffected(self, base_path):
        """The whole point of the guard, made explicit: an IDENTITY project (`to` ==
        the input's own grid) sets no `summarize_by` at all (see the `if`/`else` at the
        top of `_compile_single_input_chain`), so there is no stage-4 work for a later
        step to silently run "before" -- every registered analysis
        (`difference`/`concatenate`/`exit_criterion`) projects to its own grid, so this
        is the one shape that MUST compile and execute identically to before this
        change. Exercised end to end (compiled, `apply_params()`, `materialize()`) with
        a filter, a calculate, another (dimension) filter, and a trailing group all
        present, to prove none of the three new rejections above ever fire for it.
        """
        write_parquet(
            annual_frame(years=(2030,), geo_ids=(1, 2, 3, 4), value_fn=lambda _year, geo_id: float(geo_id)),
            base_path=base_path,
            validate=False,
        )
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                    {"op": "filter", "where": "eco > 0"},
                    {"op": "calculate", "expression": "eco * 2"},
                    {"op": "filter", "where": "calendar_year = 2030"},
                    {"op": "group", "by": [], "aggregate": "sum"},
                ],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert isinstance(compiled, BaseAnalysis)
        await compiled.apply_params()
        df = compiled.materialize()
        # Identity project (no `summarize_by`): 4 ecoregion rows survive untouched by
        # this class's rejections, each doubled by `calculate` (2.0, 4.0, 6.0, 8.0);
        # `group by=[]` collapses over `calendar_year` (a single value here), so the
        # per-grid_cell values are unchanged by it.
        assert sorted(zip(df["grid_cell"].to_list(), df["value"].to_list())) == [
            (1, 2.0),
            (2, 4.0),
            (3, 6.0),
            (4, 8.0),
        ]

    async def test_mixed_dimension_and_value_conjuncts_in_one_filter_step_apply_correctly(self, base_path):
        """Pre-existing bug, fixed alongside the ordering-guard work above even though
        it is unrelated to ordering: the compiler used to wrap each FILTER STEP's whole (possibly
        multi-conjunct) `where` in a single paren before joining multiple steps with
        `AND` (the old `" AND ".join(f"({w})" for w in rewritten_filters)`). A single
        filter step mixing a dimension conjunct and a value conjunct in one `where` --
        e.g. `"model_group_id = 500 AND eco > 2"` -- used to combine to
        `Options.where = "(model_group_id = 500 AND eco > 2)"`. At EXECUTION time,
        `BaseAnalysis._apply_filter_params` re-runs `classify_where` on that combined
        string (base.py:329), and the outer paren keeps the inner `AND` at paren-depth 1
        the whole way through, so `split_top_level_and` returns ONE conjunct spanning
        both columns -- which `classify_where` then rejects outright ("references both a
        dimension column and `value`"), a false rejection of a perfectly legal filter at
        RUNTIME, on a pipeline this module's own `compile_pipeline` had already accepted.
        Splitting into atomic conjuncts at COMPILE time (this fix) prevents that: each
        atomic conjunct is now wrapped individually.
        """
        write_parquet(
            annual_frame(years=(2030,), geo_ids=(1, 2, 3, 4), value_fn=lambda _year, geo_id: float(geo_id)),
            base_path=base_path,
            validate=False,
        )
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                    {"op": "filter", "where": "model_group_id = 500 AND eco > 2"},
                ],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        # Each atomic conjunct wrapped on its own -- NOT
        # "(model_group_id = 500 AND value > 2)", the old shape that hid the inner `AND`.
        assert compiled.options.where == "(model_group_id = 500) AND (value > 2)"
        await compiled.apply_params()
        df = compiled.materialize()
        assert sorted(df["grid_cell"].to_list()) == [3, 4]


class TestLeadingCalculateInputExtension:
    """`CalculateStep` gains an optional `input` field, extending
    `FilterStep.input`'s existing pattern: letting a
    `calculate` step lead a pipeline exactly the way a leading `filter` already can.
    Unlike the vector-only ordering relaxations above (`TestPostProjectOrderingGuard`'s
    grid_cell/group/calculate/filter classes), this extension is NOT vector-only:
    `Options.calculate` is a list that already accumulates leading and post-project
    entries side by side without collision, and the Calculate stage (stage 2) runs
    unconditionally before the Transform stage (stage 4) a `project`'s aggregation lowers
    onto, for BOTH engines -- see `CalculateStep`'s docstring in pipeline.py. `GroupStep`
    deliberately did NOT gain the same field (see `GroupStep`'s docstring for why --
    `Options.group_by`/`group_aggregate` are scalar, not list, buckets); this class also
    pins that a `group` step still cannot lead a pipeline, unaffected by this change.
    """

    def _write_fixture(self, base_path, crosswalk_path):
        write_parquet(
            annual_frame(years=(2030,), geo_ids=(1, 2, 3, 4), value_fn=lambda _year, geo_id: float(geo_id)),
            base_path=base_path,
            validate=False,
        )
        write_vector_crosswalk(
            crosswalk_path,
            "ecoregion_v001",
            "hydro_compartment_v001",
            [(1, 100, 1.0), (2, 100, 1.0), (3, 200, 1.0), (4, 200, 1.0)],
        )

    def test_leading_calculate_without_explicit_input_still_rejected(self, base_path, crosswalk_path):
        """Gaining an optional `input` field did not make it implicit: a `calculate` at
        step 1 that omits `input` still fails the pre-existing chaining-rule message
        (`pipeline.py`'s `validate_pipeline`) -- exactly `test_leading_filter_without_
        explicit_input_still_rejected`'s claim (`TestPostProjectOrderingGuard`),
        mirrored for `calculate`.
        """
        self._write_fixture(base_path, crosswalk_path)
        with pytest.raises(PipelineError, match="has no input to chain from"):
            parse_pipeline(
                {
                    "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                    "steps": [
                        {"op": "calculate", "expression": "eco * 2"},
                        {"op": "project", "to": "hydro_compartment_v001", "aggregate": "sum"},
                    ],
                    "output": {"format": "csv"},
                }
            )

    def test_leading_group_step_still_rejected(self, base_path, crosswalk_path):
        """`GroupStep` was deliberately excluded from this extension (see this class's
        docstring and `GroupStep`'s own docstring) -- a `group` step at position 1 still
        fails the chaining rule exactly as it did before `CalculateStep` gained its
        optional `input` field, since `GroupStep` still has no `input`
        field to satisfy it with.
        """
        self._write_fixture(base_path, crosswalk_path)
        with pytest.raises(PipelineError, match="has no input to chain from"):
            parse_pipeline(
                {
                    "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                    "steps": [
                        {"op": "group", "by": [], "aggregate": "sum"},
                        {"op": "project", "to": "hydro_compartment_v001", "aggregate": "sum"},
                    ],
                    "output": {"format": "csv"},
                }
            )

    async def test_leading_calculate_before_summarizing_project_compiles_and_executes_declared_order(
        self, base_path, crosswalk_path
    ):
        """The vector-side proof this class exists for, worked against the exact numbers
        `TestPostProjectOrderingGuard.test_rejects_calculate_after_summarizing_
        project` uses for the REJECTED (post-project) shape -- that test's own comment
        records what the silently-wrong answer for `calculate` declared
        after a summarizing project would have been, absent that rejection: squaring 1.0/2.0/3.0/4.0 as SOURCE
        rows before summing, {100: 5.0, 200: 25.0} (1+4, 9+16). A LEADING `calculate`
        (this test) declares exactly that same execution order on purpose -- square each
        ecoregion's value first, then sum per compartment -- so the numbers match, but
        here they are the DECLARED and CORRECT answer rather than a silent bug, because
        the calculate step is honestly declared before the project rather than being
        reordered underneath a later declaration.
        """
        self._write_fixture(base_path, crosswalk_path)
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "calculate", "input": "eco", "expression": "eco * eco"},
                    {"op": "project", "to": "hydro_compartment_v001", "aggregate": "sum"},
                ],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert isinstance(compiled, BaseAnalysis)
        assert [dict(e) for e in compiled.options.calculate] == [{"expression": "value * value", "where": None}]
        assert compiled.options.summarize_by == "hydro_compartment_v001"
        await compiled.apply_params()
        df = compiled.materialize()
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))
        assert actual == pytest.approx({100: 5.0, 200: 25.0})

    async def test_leading_calculate_before_summarizing_project_compiles_and_executes_for_raster_input(
        self, base_path, client, monkeypatch
    ):
        """The raster side of the same claim: a leading `calculate` lowers onto
        `Options.calculate` (Calculate stage 2) exactly as a post-project one does --
        this is the point of `CalculateStep`'s docstring's claim that the extension is
        NOT vector-only, unlike the vector-only ordering relaxations above. Mirrors
        `TestCalculateStepEndToEnd.test_weighted_band_sum_on_the_data_route`'s weighted
        band-sum idiom byte-for-byte, except both `calculate` steps are declared BEFORE
        `project` here instead of after -- same fixture (year 2030 filled 30, year 2040
        filled 40, i.e. `fill_fn=lambda y: y - 2000`), same weights (0.5 for 2030, 0.25
        for 2040), same expected pixel value (30*0.5 + 40*0.25 = 15 + 10 = 25).
        """
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "calculate", "input": "vct", "expression": "vct * 0.5", "where": "calendar_year = 2030"},
                {"op": "calculate", "expression": "value * 0.25", "where": "calendar_year = 2040"},
                {"op": "project", "to": "veg_grid_cell_v001"},
                {"op": "group", "by": [], "aggregate": "sum"},
            ],
            "output": {"format": "tif"},
        }
        response = client.post("/v1/analysis.tif", json=pipeline)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            assert src.count == 1
            assert src.read(1)[0, 0] == 25

    async def test_leading_filter_after_leading_calculate_preserves_declared_order_for_vector_input(
        self, base_path, crosswalk_path
    ):
        """Correctness for the interleaving fix (planner.py's unified `leading_steps`
        pass, tracking a running `leading_after_calculate` bool): a leading `filter`
        declared AFTER a leading `calculate` must inherit `after_calculate=True` --
        exactly like a POST-project filter declared after a POST-project calculate
        (`TestCalculateStepCompile.test_value_filter_after_calculate_compiles_and_
        executes_for_vector_input`) -- so it tests the POST-calculate value, not the
        pre-calculate one. A first draft of this fix used two SEPARATE lists for leading
        filters/calculates, which would have silently lost this ordering information;
        this test is what would have caught that regression.

        DECLARED (and, since both leading steps run in declared order, ACTUAL):
        `calculate(eco * 0.6)` rewrites geo_ids 1-4's values (1.0, 2.0, 3.0, 4.0) to 0.6,
        1.2, 1.8, 2.4 -- then `filter(eco > 1)` tests the POST-calculate value, keeping
        geo_ids 2, 3, 4 (1.2, 1.8, 2.4) and dropping geo_id 1 (0.6, which does not exceed
        1). Identical numbers to the post-project analog above, on purpose.
        """
        write_parquet(
            annual_frame(years=(2030,), geo_ids=(1, 2, 3, 4), value_fn=lambda _year, geo_id: float(geo_id)),
            base_path=base_path,
            validate=False,
        )
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "calculate", "input": "eco", "expression": "eco * 0.6"},
                    {"op": "filter", "where": "eco > 1"},
                    {"op": "project", "to": "ecoregion_v001"},
                ],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert isinstance(compiled, BaseAnalysis)
        await compiled.apply_params()
        df = compiled.materialize()
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))
        assert actual == pytest.approx({2: 1.2, 3: 1.8, 4: 2.4})

    def test_leading_filter_after_leading_calculate_still_rejected_for_a_grid_cell_raster_input(
        self, base_path
    ):
        """`leading_after_calculate` (the flag this class exists to pin) is still consulted:
        a `grid_cell`-bearing raster selection stays on the fixed-stage engine, whose
        `Options.where` is applied in the Filter stage (1) strictly before the Calculate
        stage (2) regardless of whether the calculate/filter pair was declared leading or
        post-project -- so the leading shape is rejected for exactly the inputs the
        post-project shape is (`TestCalculateStepCompile::
        test_rejects_value_filter_after_calculate_for_a_grid_cell_raster_input`). Pure
        compile-time rejection -- no data file needed.
        """
        pipeline = parse_pipeline(
            {
                "inputs": {
                    "vct": {
                        **{k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")},
                        "grid_cell": "-91.0,29.5",
                    }
                },
                "steps": [
                    {"op": "calculate", "input": "vct", "expression": "vct * 0.6"},
                    {"op": "filter", "where": "vct > 100"},
                    {"op": "project", "to": "veg_grid_cell_v001"},
                ],
                "output": {"format": "csv"},
            }
        )
        with pytest.raises(PipelineError, match="pre-calculate"):
            compile_pipeline(pipeline, base_path=base_path)


class TestVectorInterpreterLoadBearing:
    """The ops interpreter
    (`vector.VectorGridAnalysis.apply_steps`) is wired into `_compile_single_input_chain`
    as a SECOND engine, gated on the INPUT grid's type (`GRIDS[selection.grid].grid_type
    == "vector"`, planner.py's "OPS-INTERPRETER FAST PATH" comment) -- `dataset._pipeline_ops`
    is attached to `ops.compile_ops(pipeline)` for that shape, and `VectorGridAnalysis
    .apply_params` dispatches to `apply_steps` instead of `BaseAnalysis`'s fixed five-stage
    flow whenever that attribute is set. A wiring mistake here would be invisible to every
    OTHER test in this module: they all assert on `materialize()`'s output, which stays
    correct even if `_pipeline_ops` were attached but silently ignored, as long as the old
    engine still runs underneath. This class exists solely to rule that out -- confirming
    the attribute really is set for a vector pipeline, really is `None` for the raster-input
    shape that must stay on the old engine, and `apply_steps` really is the method invoked.
    """

    async def test_interpreter_actually_runs_for_a_vector_pipeline(self, base_path, monkeypatch):
        write_parquet(
            annual_frame(years=(2030,), geo_ids=(1, 2, 3, 4), value_fn=lambda _year, geo_id: float(geo_id)),
            base_path=base_path,
            validate=False,
        )
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [{"op": "project", "input": "eco", "to": "ecoregion_v001"}],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        # The fast path's own precondition: an ops list was actually attached, not left
        # `None` (which would silently fall through to the old engine regardless of the
        # spy below -- this assertion is what makes the spy's count below meaningful).
        assert compiled._pipeline_ops is not None

        calls = []
        original_apply_steps = VectorGridAnalysis.apply_steps

        async def _spy(self, ordered_ops):
            calls.append(ordered_ops)
            return await original_apply_steps(self, ordered_ops)

        monkeypatch.setattr(VectorGridAnalysis, "apply_steps", _spy)
        await compiled.apply_params()
        # Proves `apply_steps` is load-bearing, not dead code that happens to pass because
        # the legacy flow still runs underneath: it was invoked exactly once, with exactly
        # the ops list the planner attached.
        assert calls == [compiled._pipeline_ops]
        df = compiled.materialize()
        assert sorted(df["grid_cell"].to_list()) == [1, 2, 3, 4]

    async def test_raster_input_summarizing_onto_a_vector_grid_now_takes_the_raster_interpreter(
        self, base_path, crosswalk_path
    ):
        """The fast
        path's gate admits a raster input too (`RasterGridAnalysis.apply_steps`, gated on
        `is_raster_input and selection.grid_cell is None` in `_compile_single_input_chain`).
        See `TestRasterInterpreterLoadBearing` below for
        the raster counterpart of this whole class (the load-bearing spy, the one raster
        shape that STILL stays on the old engine, and the differential tests against the
        legacy `Options`-lowering engine).
        """
        grid = GRIDS[RASTER_KWARGS["grid"]]
        zones = np.zeros((grid.height, grid.width), dtype="int32")
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _y: raster_array())
        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"},
                ],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert compiled._pipeline_ops is not None


class TestVectorInterpreterDifferential:
    """The ops interpreter
    (`vector.VectorGridAnalysis.apply_steps`) must compute EXACTLY what the legacy
    `Options`-lowering engine (`BaseAnalysis.apply_params`'s fixed five-stage flow)
    computes, for every declared-order shape `planner._compile_single_input_chain`
    accepts -- the fast path added to planner.py is only supposed to change WHICH ENGINE
    runs a pipeline, never WHAT a pipeline computes.

    Each test compiles the SAME pipeline via TWO independent `compile_pipeline` calls
    against the same fixture data: one left exactly as the planner produces it (takes the
    interpreter, since its input grid is vector), and one with `_pipeline_ops` forced back
    to `None` immediately after compiling -- which reroutes `VectorGridAnalysis
    .apply_params` onto `super().apply_params()`, i.e. the untouched legacy flow, per its
    own `if self._pipeline_ops is not None: ... else: await super().apply_params()`
    dispatch. Both are then materialized and compared with `_assert_frames_equal
    _ignoring_order` -- sorted by every column, so row order (which the two engines are
    not obliged to agree on) doesn't cause a false mismatch.

    Pipelines below are deliberately drawn from shapes already pinned END-TO-END elsewhere
    in this module (`TestPostProjectOrderingGuard`, `TestProjectAggregateCountByValue
    EndToEnd`) -- reusing a shape whose CORRECT answer is independently known lets each
    test check two things at once: the interpreter's answer is right, AND it agrees with
    the legacy engine.
    """

    @staticmethod
    def _assert_frames_equal_ignoring_order(a, b):
        assert sorted(a.columns) == sorted(b.columns)
        cols = sorted(a.columns)
        assert a.select(cols).sort(cols).equals(b.select(cols).sort(cols))

    async def _run_both_engines(self, pipeline, base_path):
        interpreter_compiled = compile_pipeline(pipeline, base_path=base_path)
        assert interpreter_compiled._pipeline_ops is not None
        legacy_compiled = compile_pipeline(pipeline, base_path=base_path)
        legacy_compiled._pipeline_ops = None

        await interpreter_compiled.apply_params()
        await legacy_compiled.apply_params()
        return interpreter_compiled.materialize(), legacy_compiled.materialize()

    async def test_leading_filter_then_summarizing_project(self, base_path, crosswalk_path):
        # Same fixture/pipeline as
        # TestPostProjectOrderingGuard.test_leading_filter_before_project_compiles_and_executes_declared_order
        # (known-correct answer: {100: 2.0, 200: 7.0}).
        write_parquet(
            annual_frame(years=(2030,), geo_ids=(1, 2, 3, 4), value_fn=lambda _year, geo_id: float(geo_id)),
            base_path=base_path,
            validate=False,
        )
        write_vector_crosswalk(
            crosswalk_path,
            "ecoregion_v001",
            "hydro_compartment_v001",
            [(1, 100, 1.0), (2, 100, 1.0), (3, 200, 1.0), (4, 200, 1.0)],
        )
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "filter", "input": "eco", "where": "eco >= 2"},
                    {"op": "project", "to": "hydro_compartment_v001", "aggregate": "sum"},
                ],
                "output": {"format": "csv"},
            }
        )
        interp_df, legacy_df = await self._run_both_engines(pipeline, base_path)
        self._assert_frames_equal_ignoring_order(interp_df, legacy_df)
        actual = dict(zip(interp_df["grid_cell"].to_list(), interp_df["value"].to_list()))
        assert actual == {100: 2.0, 200: 7.0}

    async def test_identity_project_filter_calculate_filter_group(self, base_path):
        # Same fixture/pipeline as
        # TestPostProjectOrderingGuard.test_identity_project_is_unaffected
        # (known-correct answer: [(1,2.0),(2,4.0),(3,6.0),(4,8.0)]) -- exercises all of
        # filter/calculate/filter/group in one chain, over an identity project.
        write_parquet(
            annual_frame(years=(2030,), geo_ids=(1, 2, 3, 4), value_fn=lambda _year, geo_id: float(geo_id)),
            base_path=base_path,
            validate=False,
        )
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                    {"op": "filter", "where": "eco > 0"},
                    {"op": "calculate", "expression": "eco * 2"},
                    {"op": "filter", "where": "calendar_year = 2030"},
                    {"op": "group", "by": [], "aggregate": "sum"},
                ],
                "output": {"format": "csv"},
            }
        )
        interp_df, legacy_df = await self._run_both_engines(pipeline, base_path)
        self._assert_frames_equal_ignoring_order(interp_df, legacy_df)
        assert sorted(zip(interp_df["grid_cell"].to_list(), interp_df["value"].to_list())) == [
            (1, 2.0),
            (2, 4.0),
            (3, 6.0),
            (4, 8.0),
        ]

    async def test_mixed_dimension_and_value_conjunct_in_one_filter_step(self, base_path):
        # Same fixture/pipeline as
        # TestPostProjectOrderingGuard
        # .test_mixed_dimension_and_value_conjuncts_in_one_filter_step_apply_correctly
        # (known-correct answer: grid_cell in [3, 4]).
        write_parquet(
            annual_frame(years=(2030,), geo_ids=(1, 2, 3, 4), value_fn=lambda _year, geo_id: float(geo_id)),
            base_path=base_path,
            validate=False,
        )
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                    {"op": "filter", "where": "model_group_id = 500 AND eco > 2"},
                ],
                "output": {"format": "csv"},
            }
        )
        interp_df, legacy_df = await self._run_both_engines(pipeline, base_path)
        self._assert_frames_equal_ignoring_order(interp_df, legacy_df)
        assert sorted(interp_df["grid_cell"].to_list()) == [3, 4]

    async def test_summarizing_project_then_grid_cell_filter(self, base_path, crosswalk_path):
        """The post-project `grid_cell` filter shape
        (`_extract_post_project_grid_cell_filter`/`Options.summarize_grid_cell`) lowers
        onto a genuinely different MECHANISM in each engine: the legacy engine narrows the
        crosswalk join itself (`Options.summarize_grid_cell`, applied inside the Transform
        stage), while the interpreter runs an ordinary post-`ProjectOp` `FilterOp` on the
        already-projected `grid_cell` column, dropped in plain declared order like any
        other filter (`apply_steps`'s `FilterOp` branch). This is the one shape in this
        class where the two engines' internals genuinely diverge rather than merely their
        code path, which is exactly why it is worth checking frame-for-frame.
        """
        write_parquet(
            annual_frame(years=(2030,), geo_ids=(1, 2, 3, 4), value_fn=lambda _year, geo_id: float(geo_id)),
            base_path=base_path,
            validate=False,
        )
        write_vector_crosswalk(
            crosswalk_path,
            "ecoregion_v001",
            "hydro_compartment_v001",
            [(1, 100, 1.0), (2, 100, 1.0), (3, 200, 1.0), (4, 200, 1.0)],
        )
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "eco", "to": "hydro_compartment_v001", "aggregate": "sum"},
                    {"op": "filter", "where": "grid_cell = 100"},
                ],
                "output": {"format": "csv"},
            }
        )
        interp_df, legacy_df = await self._run_both_engines(pipeline, base_path)
        self._assert_frames_equal_ignoring_order(interp_df, legacy_df)
        assert interp_df["grid_cell"].to_list() == [100]
        assert interp_df["value"].to_list() == [3.0]

    async def test_summarizing_project_with_count_by_value_aggregate(self, base_path, crosswalk_path):
        # Modeled on
        # TestProjectAggregateCountByValueEndToEnd.test_categorical_count_per_zone --
        # geo_id 1,3 -> value 1.0, geo_id 2,4 -> value 0.0, all crosswalked to one of two
        # zones, so each zone's per-value counts are hand-verifiable.
        write_parquet(
            annual_frame(years=(2030,), geo_ids=(1, 2, 3, 4), value_fn=lambda _year, geo_id: float(geo_id % 2)),
            base_path=base_path,
            validate=False,
        )
        write_vector_crosswalk(
            crosswalk_path,
            "ecoregion_v001",
            "hydro_compartment_v001",
            [(1, 100, 1.0), (2, 100, 1.0), (3, 200, 1.0), (4, 200, 1.0)],
        )
        pipeline = parse_pipeline(
            {
                "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {
                        "op": "project",
                        "input": "eco",
                        "to": "hydro_compartment_v001",
                        "aggregate": "count_by_value",
                    },
                ],
                "output": {"format": "csv"},
            }
        )
        interp_df, legacy_df = await self._run_both_engines(pipeline, base_path)
        self._assert_frames_equal_ignoring_order(interp_df, legacy_df)
        actual = {
            (int(row["grid_cell"]), float(row["value"])): int(row["count"])
            for row in interp_df.iter_rows(named=True)
        }
        assert actual == {(100, 1.0): 1, (100, 0.0): 1, (200, 1.0): 1, (200, 0.0): 1}


class TestRasterInterpreterLoadBearing:
    """The raster counterpart
    of `TestVectorInterpreterLoadBearing` -- proves `RasterGridAnalysis.apply_steps`
    (the band-wide interpreter) is really invoked for the raster shapes the fast path now
    admits, rather than dead code that happens to pass because the old fixed-stage engine
    still runs underneath, and that the one raster shape still excluded from the fast path
    (`InputSelection.grid_cell` set -- see `apply_steps`' own docstring for why) really
    does still get `_pipeline_ops is None`.
    """

    async def test_interpreter_actually_runs_for_a_raster_pipeline(self, base_path, monkeypatch):
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "group", "by": [], "aggregate": "max"},
                ],
                "output": {"format": "tif"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        # The fast path's own precondition, mirroring the vector test above.
        assert compiled._pipeline_ops is not None

        calls = []
        original_apply_steps = RasterGridAnalysis.apply_steps

        async def _spy(self, ordered_ops):
            calls.append(ordered_ops)
            return await original_apply_steps(self, ordered_ops)

        monkeypatch.setattr(RasterGridAnalysis, "apply_steps", _spy)
        await compiled.apply_params()
        assert calls == [compiled._pipeline_ops]
        # Both raw bands (2030, 2040) collapsed into a single group (`group by=[]`) --
        # proof `_collapse_group` really ran from inside `apply_steps`, not skipped.
        assert compiled._band_ops == [("group", [], "max")]
        assert compiled._band_frame().height == 2
        assert compiled.materialize().height == 1

    async def test_raster_input_with_grid_cell_stays_on_the_old_engine(self, base_path):
        """`RasterGridAnalysis._apply_grid_cell` reshapes `self.lf` straight to a long
        (grid_cell/value) frame via `.unpivot(...)` before any pipeline op could run --
        a shape change `apply_steps`' PRE-project mode (which assumes `self.lf` starts as
        the band-metadata table) does not accommodate, so this one shape must keep
        `_pipeline_ops is None` even though its input grid is raster.
        """
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _y: raster_array())
        pipeline = parse_pipeline(
            {
                "inputs": {
                    "vct": {
                        **{k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")},
                        "grid_cell": "-91.0,29.5",
                    }
                },
                "steps": [{"op": "project", "input": "vct", "to": "veg_grid_cell_v001"}],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert compiled._pipeline_ops is None


class TestRasterInterpreterDifferential:
    """The raster counterpart
    of `TestVectorInterpreterDifferential` -- the ops interpreter
    (`RasterGridAnalysis.apply_steps`) must compute EXACTLY what the legacy
    `Options`-lowering engine (`BaseAnalysis.apply_params`'s fixed five-stage flow)
    computes, for every declared-order raster shape `planner._compile_single_input_chain`
    now admits to the fast path.

    Same two-compile pattern as the vector class: one `compile_pipeline` call left as the
    planner produces it (takes the interpreter), one with `_pipeline_ops` forced back to
    `None` (reroutes `RasterGridAnalysis.apply_params` onto `super().apply_params()`, the
    untouched legacy flow).
    """

    async def _run_both_engines(self, pipeline, base_path):
        interpreter_compiled = compile_pipeline(pipeline, base_path=base_path)
        assert interpreter_compiled._pipeline_ops is not None
        legacy_compiled = compile_pipeline(pipeline, base_path=base_path)
        legacy_compiled._pipeline_ops = None

        await interpreter_compiled.apply_params()
        await legacy_compiled.apply_params()
        return interpreter_compiled, legacy_compiled

    @staticmethod
    def _assert_frames_equal_ignoring_order(a, b):
        assert sorted(a.columns) == sorted(b.columns)
        cols = sorted(a.columns)
        assert a.select(cols).sort(cols).equals(b.select(cols).sort(cols))

    async def test_tif_output_identity_project_group_only(self, base_path, tmp_path):
        # Same fixture/pipeline as
        # TestSingleInputEndToEnd.test_raster_identity_project_group_only
        # (known-correct answer: every pixel is 40, the max of the two years' fills).
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "group", "by": [], "aggregate": "max"},
                ],
                "output": {"format": "tif"},
            }
        )
        interp, legacy = await self._run_both_engines(pipeline, base_path)

        interp_path = tmp_path / "interp.tif"
        legacy_path = tmp_path / "legacy.tif"
        await interp.write_raster(interp_path)
        await legacy.write_raster(legacy_path)

        with rio.open(interp_path) as src:
            interp_data = src.read()
        with rio.open(legacy_path) as src:
            legacy_data = src.read()
        assert np.array_equal(interp_data, legacy_data)
        assert (interp_data == 40).all()

    async def test_dimension_filter_before_group_narrows_the_scan(self, base_path, tmp_path):
        """The pushdown prefix: a dimension filter declared before any `group` is applied
        straight to the band-metadata frame `_band_frame` hands the pixel reader, so the
        dropped year's file is never opened at all -- and the answer still matches the
        legacy engine's own `Options.where` dimension conjunct, band for band.
        """
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "filter", "where": "calendar_year = 2040"},
                    {"op": "group", "by": [], "aggregate": "max"},
                ],
                "output": {"format": "tif"},
            }
        )
        interp, legacy = await self._run_both_engines(pipeline, base_path)
        # One band reaches the reader, not two -- the filter really narrowed the scan.
        assert interp._band_frame().height == 1

        interp_path = tmp_path / "interp.tif"
        legacy_path = tmp_path / "legacy.tif"
        await interp.write_raster(interp_path)
        await legacy.write_raster(legacy_path)
        with rio.open(interp_path) as src:
            interp_data = src.read()
        with rio.open(legacy_path) as src:
            legacy_data = src.read()
        assert np.array_equal(interp_data, legacy_data)
        assert (interp_data == 40).all()

    async def test_tabular_output_summarize_to_vector_mean(self, base_path, crosswalk_path):
        # Same fixture/pipeline as
        # TestProjectOnlyReaggregationEndToEnd.test_raster_to_vector_mean.
        grid = GRIDS[RASTER_KWARGS["grid"]]
        zones = np.zeros((grid.height, grid.width), dtype="int32")
        zones[: grid.height // 2, :] = 1
        zones[grid.height // 2 :, :] = 2
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _y: raster_array())

        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"},
                ],
                "output": {"format": "csv"},
            }
        )
        interp, legacy = await self._run_both_engines(pipeline, base_path)
        interp_df, legacy_df = interp.materialize(), legacy.materialize()
        self._assert_frames_equal_ignoring_order(interp_df, legacy_df)

        categories = raster_array()
        actual = dict(zip(interp_df["grid_cell"].to_list(), interp_df["value"].to_list()))
        for zone_id in (1, 2):
            expected_mean = categories[zones == zone_id].astype("float64").mean()
            assert actual[zone_id] == pytest.approx(expected_mean, rel=1e-4)

    async def test_value_filter_before_summarizing_project_masks_not_drops(self, base_path, crosswalk_path):
        """Exercises the pushdown gate (`_write_raster_windowed`'s
        `apply_value_filter_band_wide` call): a pre-project value `FilterOp` stashes onto
        `self._value_where` (this method's PRE-project branch) rather than dropping rows,
        and `_summarize_zone_stats_plugin` masks each band's raw pixel value against it
        before the `mean` collapse -- exactly like the legacy engine's own `where=`
        value-conjunct handling for a summarizing raster request.
        """
        grid = GRIDS[RASTER_KWARGS["grid"]]
        zones = np.ones((grid.height, grid.width), dtype="int32")
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _y: raster_array())

        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "filter", "input": "vct", "where": "vct > 100"},
                    {"op": "project", "to": "hydro_compartment_v001", "aggregate": "mean"},
                ],
                "output": {"format": "csv"},
            }
        )
        interp, legacy = await self._run_both_engines(pipeline, base_path)
        interp_df, legacy_df = interp.materialize(), legacy.materialize()
        self._assert_frames_equal_ignoring_order(interp_df, legacy_df)

        categories = raster_array()
        expected_mean = categories[categories > 100].astype("float64").mean()
        actual = dict(zip(interp_df["grid_cell"].to_list(), interp_df["value"].to_list()))
        assert actual[1] == pytest.approx(expected_mean, rel=1e-4)

    async def test_post_project_grid_cell_filter_matches_the_legacy_crosswalk_narrowing(
        self, base_path, crosswalk_path
    ):
        # Same fixture/pipeline as
        # TestPostProjectGridCellFilterEndToEnd.test_grid_cell_eq_lowers_onto_summarize_grid_cell,
        # but exercised as a differential: the legacy engine narrows the crosswalk's
        # `zone_ids` via `Options.summarize_grid_cell` (`_extract_post_project_grid_cell_
        # filter`), while the interpreter runs an ordinary post-`ProjectOp` `FilterOp` on
        # the already-projected `grid_cell` column instead (see `apply_steps`' own
        # docstring for why those are numerically equivalent) -- so this is the one shape
        # where the two engines' internal MECHANISM genuinely diverges, worth checking
        # frame-for-frame rather than just at the ASGI-route level.
        grid = GRIDS[RASTER_KWARGS["grid"]]
        zones = np.zeros((grid.height, grid.width), dtype="int32")
        zones[: grid.height // 2, :] = 1
        zones[grid.height // 2 :, :] = 2
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _y: raster_array())

        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"},
                    {"op": "filter", "where": "grid_cell = 1"},
                ],
                "output": {"format": "csv"},
            }
        )
        interp, legacy = await self._run_both_engines(pipeline, base_path)
        interp_df, legacy_df = interp.materialize(), legacy.materialize()
        self._assert_frames_equal_ignoring_order(interp_df, legacy_df)

        assert interp_df["grid_cell"].to_list() == [1]
        categories = raster_array()
        expected_mean = categories[zones == 1].astype("float64").mean()
        assert interp_df["value"].to_list() == [pytest.approx(expected_mean, rel=1e-4)]

    async def test_standalone_value_filter_without_summarize_still_rejected_by_both_engines(
        self, base_path
    ):
        """The trickiest ordering edge case this step had to get right: `project`
        (identity) declared BEFORE a value `FilterOp`, with no `summarize_by`/
        `summarize_aggregate` anywhere. The legacy engine's stage 4
        (`_apply_transform_params_impl`) always runs LAST, regardless of declared order,
        and is what raises this rejection -- so `apply_steps` must reproduce that exact
        "runs last" positioning for an identity project (see its own docstring), rather
        than resolving the project immediately at its declared (earlier) position and
        never re-checking once the later filter stashes `self._value_where`. Same
        fixture/pipeline shape as
        `TestRejections.test_raster_filter_without_summarize_is_400`, exercised against
        both engines directly rather than through the ASGI route.
        """
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "filter", "where": "vct > 35"},
                    {"op": "group", "by": [], "aggregate": "min"},
                ],
                "output": {"format": "tif"},
            }
        )
        interpreter_compiled = compile_pipeline(pipeline, base_path=base_path)
        assert interpreter_compiled._pipeline_ops is not None
        legacy_compiled = compile_pipeline(pipeline, base_path=base_path)
        legacy_compiled._pipeline_ops = None

        with pytest.raises(AnalysisError, match="summarize_by"):
            await interpreter_compiled.apply_params()
        with pytest.raises(AnalysisError, match="summarize_by"):
            await legacy_compiled.apply_params()


class TestRasterInterpreterDeclaredOrder:
    """The four raster shapes `planner._compile_single_input_chain` used to reject
    outright, one test each, every answer hand-computed.

    A differential against the legacy engine (`TestRasterInterpreterDifferential`, above)
    is the wrong instrument here: these are precisely the shapes the legacy engine gets
    WRONG -- it would agree with the interpreter only if the interpreter reproduced the
    bug. So each test states the declared meaning, states the number the old fixed stage
    order would have produced instead, and asserts the declared one.
    """

    @staticmethod
    async def _write_tif(pipeline, base_path, out_path):
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert compiled._pipeline_ops is not None
        await compiled.apply_params()
        await compiled.write_raster(out_path)
        with rio.open(out_path) as src:
            return src.read()

    async def test_calculate_after_group_squares_the_mean_not_the_other_way_round(
        self, base_path, tmp_path
    ):
        """Bands [2, 4]. DECLARED: `group(mean)` then `calculate(vct * vct)` -- mean(x)**2
        = 3**2 = 9. The fixed calculate-before-collapse order this rejection existed for
        computes mean(x**2) = (4 + 16) / 2 = 10 instead: a silent wrong answer, which is
        why 9 vs 10 is the whole assertion.
        """
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: 2 if y == 2030 else 4)
        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "group", "by": [], "aggregate": "mean"},
                    {"op": "calculate", "expression": "vct * vct"},
                ],
                "output": {"format": "tif"},
            }
        )
        data = await self._write_tif(pipeline, base_path, tmp_path / "out.tif")

        assert data.shape[0] == 1
        assert (data == 9).all(), "mean(x)**2, not mean(x**2) (which would be 10)"

    async def test_dimension_filter_after_group_selects_from_the_grouped_bands(
        self, base_path, tmp_path
    ):
        """Bands 2030 -> 30, 2040 -> 40. DECLARED: `group(by=calendar_year, max)` -- which
        keeps both bands, one per year -- then `filter(calendar_year = 2030)`, selecting
        the FIRST of the two grouped bands. The rejection existed because the group's
        member positions used to be recorded before the filter renumbered `b0..bN`; the
        surviving band being 30 (not 40, and not an IndexError) is what proves the filter
        resolves against the post-group metadata.
        """
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "group", "by": ["calendar_year"], "aggregate": "max"},
                    {"op": "filter", "where": "calendar_year = 2030"},
                ],
                "output": {"format": "tif"},
            }
        )
        data = await self._write_tif(pipeline, base_path, tmp_path / "out.tif")

        assert data.shape[0] == 1, "one band survives the filter, out of the group's two"
        assert (data == 30).all()

    async def test_a_second_group_collapses_the_first_groups_output(self, base_path, tmp_path):
        """Bands [2, 4, 6]. DECLARED: `group(by=calendar_year, mean)` -- one band per year,
        so still [2, 4, 6] -- then `group(by=[], sum)` over the FIRST group's three output
        bands... which is 12 either way, so that shape would not discriminate. Instead the
        first group is `by=[]` too: mean(2, 4, 6) = 4 collapsed into ONE band, and the
        second group sums that single band to 4. A second group that re-derived itself
        against the original, uncollapsed band set (the structural discard this rejection
        existed for) would return sum(2, 4, 6) = 12.

        It also pins the per-op aggregate: `Options.group_aggregate` is a single scalar the
        planner overwrites last-write-wins, so both groups would use `sum` if `apply_steps`
        did not pass each `GroupOp`'s own aggregate -- and sum(2, 4, 6) is again 12.
        """
        write_raster_dataset(base_path, years=(2030, 2040, 2050), fill_fn={2030: 2, 2040: 4, 2050: 6}.get)
        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "group", "by": [], "aggregate": "mean", "as": "g1"},
                    {"op": "group", "by": [], "aggregate": "sum"},
                ],
                "output": {"format": "tif"},
            }
        )
        data = await self._write_tif(pipeline, base_path, tmp_path / "out.tif")

        assert data.shape[0] == 1
        assert (data == 4).all(), "sum(mean(2, 4, 6)), not sum(2, 4, 6) = 12"

    async def test_filter_after_group_naming_a_collapsed_dimension_is_a_400(self, base_path):
        """The flip side of admitting a post-group filter: `group(by=[])` collapses EVERY
        dimension away, so a filter declared after it that names one is asking about a
        column the grouped bands no longer have. That has to surface as a 400 naming the
        `where`, not as a raw Polars column-not-found escaping `apply_steps`.
        """
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "group", "by": [], "aggregate": "max"},
                    {"op": "filter", "where": "calendar_year = 2030"},
                ],
                "output": {"format": "tif"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)

        with pytest.raises(AnalysisError, match="calendar_year") as excinfo:
            await compiled.apply_params()
        assert excinfo.value.status_code == 400

    async def test_value_filter_after_calculate_masks_the_rewritten_value(
        self, base_path, crosswalk_path
    ):
        """DECLARED: `calculate(vct * 0.6)` then `filter(vct > 60)` -- the predicate tests
        the POST-calculate value, so the surviving pixels are those whose RAW value exceeds
        100, and the zone mean is taken over `raw * 0.6` for exactly those. The fixed
        Filter-before-Calculate stage order this rejection existed for would instead test
        the raw value against 60, keeping every pixel above 60 -- a strictly larger, and
        differently-valued, population.

        A raster value filter has to be paired with a summarizing `project` to run at all
        (`_apply_transform_params_impl` rejects a standalone one), and a value filter
        declared AFTER a summarizing project is still rejected by the summarizing-project
        order guard -- so calculate/filter/project, in that order, is the one executable
        arrangement of this shape.
        """
        grid = GRIDS[RASTER_KWARGS["grid"]]
        zones = np.ones((grid.height, grid.width), dtype="int32")
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _y: raster_array())

        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "calculate", "input": "vct", "expression": "vct * 0.6"},
                    {"op": "filter", "where": "vct > 60"},
                    {"op": "project", "to": "hydro_compartment_v001", "aggregate": "mean"},
                ],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert compiled._pipeline_ops is not None
        await compiled.apply_params()
        df = compiled.materialize()

        rewritten = raster_array().astype("float64") * 0.6
        expected = rewritten[rewritten > 60].mean()
        pre_calculate_reading = rewritten[raster_array() > 60].mean()
        assert expected != pytest.approx(pre_calculate_reading), "fixture must discriminate"
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))
        assert actual[1] == pytest.approx(expected, rel=1e-6)


class TestRasterBandOpsReplayIsLoadBearing:
    """Gate the newly-admitted shapes on the machinery actually being the thing that runs
    them, not on the answers alone: a correct answer could in principle come from a
    surviving fallback, and a deleted code path is only really deleted if nothing still
    reaches for it.
    """

    async def test_apply_band_ops_receives_the_declared_op_list_and_produces_the_output(
        self, base_path, tmp_path, monkeypatch
    ):
        """The spy: `raster.apply_band_ops` is called with the ops IN DECLARED ORDER, each
        entry carrying the caller's own group key/plans/predicate rather than any resolved
        band position, and the frame it returns is what gets written.
        """
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: 2 if y == 2030 else 4)
        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "group", "by": ["calendar_year"], "aggregate": "max"},
                    {"op": "filter", "where": "calendar_year = 2040"},
                    {"op": "calculate", "expression": "vct * vct"},
                ],
                "output": {"format": "tif"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        await compiled.apply_params()

        seen = []
        original = raster_module.apply_band_ops

        def _spy(lf, metadata_df, band_ops, **kwargs):
            seen.append(band_ops)
            return original(lf, metadata_df, band_ops, **kwargs)

        monkeypatch.setattr(raster_module, "apply_band_ops", _spy)
        out_path = tmp_path / "out.tif"
        await compiled.write_raster(out_path)

        assert seen, "the write path never reached apply_band_ops"
        kinds = [entry[0] for entry in seen[0]]
        assert kinds == ["group", "filter", "calculate"], "declared order, not a fixed stage order"
        assert seen[0][0][:2] == ("group", ["calendar_year"])
        assert seen[0][1] == ("filter", "(calendar_year = 2040)")
        with rio.open(out_path) as src:
            data = src.read()
        assert data.shape[0] == 1
        assert (data == 16).all()  # max over the 2040 band alone, squared

    def test_the_stored_row_index_machinery_is_gone(self):
        """The complementary half of the spy: the attributes the old, index-carrying path
        was built out of no longer exist anywhere, so a shape that "works" cannot be
        working through a surviving fallback.
        """
        from cpra.mp.data.analysis import sources as sources_module

        assert not hasattr(RasterGridAnalysis, "_get_group_row_idxs")
        assert not hasattr(RasterGridAnalysis, "_band_applicability")
        assert not hasattr(RasterGridAnalysis, "_current_metadata_df")
        assert not hasattr(RasterGridAnalysis, "_band_ops_is_simple")
        assert not hasattr(sources_module, "band_collapse_exprs")

    async def test_no_request_leaves_a_stored_band_group_attribute_behind(self, base_path):
        """`_band_groups` -- the stored row-index structure every one of these rejections
        traced back to -- is not set by any code path, on a shape that would have populated
        it."""
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                    {"op": "group", "by": ["calendar_year"], "aggregate": "max"},
                ],
                "output": {"format": "tif"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        await compiled.apply_params()

        assert not hasattr(compiled, "_band_groups")
        assert compiled._band_ops == [("group", ["calendar_year"], "max")]


# --------------------------------------------------------------------------------------
# Shape 2: the paired (notebook) shape
# --------------------------------------------------------------------------------------


class TestCompilePairedNotebookShape:
    def test_rejects_non_raster_output_format(self, notebook_query_data):
        bad = copy.deepcopy(NOTEBOOK_QUERY)
        bad["output"]["format"] = "csv"
        with pytest.raises(PipelineError, match="only tif/png/jpg"):
            compile_pipeline(parse_pipeline(bad), base_path=notebook_query_data)

    def test_rejects_join_with_more_than_two_names(self, notebook_query_data):
        bad = copy.deepcopy(NOTEBOOK_QUERY)
        bad["steps"][1]["inputs"] = ["marsh", "sal", "salinity"]
        with pytest.raises(PipelineError, match="other than two"):
            compile_pipeline(parse_pipeline(bad), base_path=notebook_query_data)

    def test_rejects_group_by_non_empty(self, notebook_query_data):
        bad = copy.deepcopy(NOTEBOOK_QUERY)
        bad["steps"][3]["by"] = ["grid_cell"]
        with pytest.raises(PipelineError, match="only by=\\[\\]"):
            compile_pipeline(parse_pipeline(bad), base_path=notebook_query_data)

    def test_rejects_missing_trailing_group(self, notebook_query_data):
        """`_compile_paired_notebook_shape` used to own a defensive "missing trailing
        `group`" rejection of its own, but it was genuinely unreachable: a paired
        shape's `output.format` must be tif/png/jpg to even reach that function
        (`_RASTER_OUTPUT_FORMATS`), which is exactly `_SINGLE_VALUE_COLUMN_FORMATS` --
        so `_validate_output_columns` (run unconditionally, first, in
        `compile_pipeline`) always caught a join with no subsequent group before that
        function's own check would ever run (see
        `test_validate_output_columns_rejects_join_without_group_for_ndjson` for the
        same underlying check exercised via a different output format). Removed as dead
        code (see `_compile_paired_notebook_shape`'s own comment, where that check used
        to raise); this test now goes through the real `compile_pipeline` entry point
        and asserts on the message that actually fires for this mistake.
        """
        bad = copy.deepcopy(NOTEBOOK_QUERY)
        bad["steps"] = bad["steps"][:3]
        with pytest.raises(PipelineError, match="add a `group` step"):
            compile_pipeline(parse_pipeline(bad), base_path=notebook_query_data)

    def test_rejects_two_group_steps(self, notebook_query_data):
        bad = copy.deepcopy(NOTEBOOK_QUERY)
        bad["steps"].append({"op": "group", "by": [], "aggregate": "max"})
        with pytest.raises(PipelineError, match="more than one `group`"):
            compile_pipeline(parse_pipeline(bad), base_path=notebook_query_data)

    def test_rejects_join_dimension_misalignment(self, notebook_query_data):
        bad = copy.deepcopy(NOTEBOOK_QUERY)
        bad["inputs"]["salinity"]["scenario_id"] = 8
        with pytest.raises(PipelineError, match="does not align"):
            compile_pipeline(parse_pipeline(bad), base_path=notebook_query_data)

    def test_rejects_fine_input_not_on_a_raster_grid(self, notebook_query_data):
        bad = copy.deepcopy(NOTEBOOK_QUERY)
        # Swap which input the join treats as "fine": point the join at the coarse
        # input's own name plus the projected name (the other, real raster input is left
        # dangling, but that surfaces as a different error) -- simplest way to make the
        # "fine" side resolve to a non-raster grid is to project the RASTER input onto
        # the vector grid instead, and join it with the (still-vector) coarse input.
        bad["steps"][0] = {"op": "project", "input": "marsh", "to": "hydro_compartment_v001", "aggregate": "mean"}
        bad["steps"][1] = {"op": "join", "type": "spatial", "inputs": ["salinity", "marsh"], "as": "paired"}
        with pytest.raises(PipelineError, match="raster grid"):
            compile_pipeline(parse_pipeline(bad), base_path=notebook_query_data)

    def test_accepts_and_compiles_to_a_paired_plan(self, notebook_query_data):
        compiled = compile_pipeline(parse_pipeline(NOTEBOOK_QUERY), base_path=notebook_query_data)
        assert isinstance(compiled, PairedAnalysis)

    def test_rejects_count_aggregate_with_non_default_of(self, notebook_query_data):
        # End-to-end version of `TestPairedAnalysisPrepare.test_count_rejects_non_default_of`
        # -- `compile_pipeline` runs `PairedAnalysis._prepare()` as its very last step, so
        # this is a real plan-time (not merely unit-level) 400 for the mistake.
        bad = copy.deepcopy(NOTEBOOK_QUERY)
        bad["steps"][3] = {"op": "group", "by": [], "aggregate": "count", "of": "calendar_year"}
        with pytest.raises(PipelineError, match="has no aggregand"):
            compile_pipeline(parse_pipeline(bad), base_path=notebook_query_data)

    def test_accepts_count_aggregate_with_default_of(self, notebook_query_data):
        query = copy.deepcopy(NOTEBOOK_QUERY)
        query["steps"][3] = {"op": "group", "by": [], "aggregate": "count"}
        compiled = compile_pipeline(parse_pipeline(query), base_path=notebook_query_data)
        assert isinstance(compiled, PairedAnalysis)
        assert compiled.of_kind == "count"

    def test_rejects_grid_cell_on_the_fine_input(self, notebook_query_data):
        # `PairedAnalysis`'s `fine_dataset`/`coarse_dataset` never call `apply_params()`
        # (see `_dataset_for`'s own docstring) -- only `_apply_grid_cell()` (a stage hook
        # of `apply_params()`) reads `Options.grid_cell`, so blindly forwarding it here
        # would silently no-op rather than filter anything.
        bad = copy.deepcopy(NOTEBOOK_QUERY)
        bad["inputs"]["marsh"]["grid_cell"] = "1"
        with pytest.raises(PipelineError, match="grid_cell"):
            compile_pipeline(parse_pipeline(bad), base_path=notebook_query_data)

    def test_rejects_grid_cell_on_the_coarse_input(self, notebook_query_data):
        bad = copy.deepcopy(NOTEBOOK_QUERY)
        bad["inputs"]["salinity"]["grid_cell"] = "1"
        with pytest.raises(PipelineError, match="grid_cell"):
            compile_pipeline(parse_pipeline(bad), base_path=notebook_query_data)

    def test_rejects_output_sort(self, notebook_query_data):
        # `_apply_sort_params` is likewise only ever invoked from `apply_params()`, so
        # `output.sort` would silently no-op on this shape the same way `grid_cell` would.
        bad = copy.deepcopy(NOTEBOOK_QUERY)
        bad["output"]["sort"] = "-value"
        with pytest.raises(PipelineError, match="sort"):
            compile_pipeline(parse_pipeline(bad), base_path=notebook_query_data)

    def test_rejects_output_select(self, notebook_query_data):
        bad = copy.deepcopy(NOTEBOOK_QUERY)
        bad["output"]["select"] = "grid_cell"
        with pytest.raises(PipelineError, match="select"):
            compile_pipeline(parse_pipeline(bad), base_path=notebook_query_data)


# --------------------------------------------------------------------------------------
# End-to-end ASGI routes -- the three MUST-WORK shapes
# --------------------------------------------------------------------------------------


class TestNotebookQueryEndToEnd:
    """The exit-criterion notebook query, byte-exact, through the real ASGI routes."""

    def test_post(self, client, notebook_query_data):
        response = client.post("/v1/analysis.tif", json=NOTEBOOK_QUERY)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
        assert (data == 2040).all()

    def test_get_dotted_query(self, client, notebook_query_data):
        query = pipeline_to_query(parse_pipeline(NOTEBOOK_QUERY))
        response = client.get("/v1/analysis.tif", params=query)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
        assert (data == 2040).all()

    def test_get_png_tile(self, client, notebook_query_data):
        tile_query = copy.deepcopy(NOTEBOOK_QUERY)
        tile_query["output"] = {"format": "png"}
        query = pipeline_to_query(parse_pipeline(tile_query))
        tile = _tile_for_grid(notebook_query_data)
        response = client.get(f"/v1/analysis/{tile.z}/{tile.x}/{tile.y}.png", params=query)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_pbf_tile_rejects_the_paired_shape_gracefully(self, client, notebook_query_data):
        """`PairedAnalysis.validate_vector_tile()` -- the paired/notebook shape's output is
        always raster (see that method's docstring), so a `.pbf` request against it must
        400 through the SAME `compiled.validate_vector_tile()` call
        `get_analysis_vector_tile` makes for a single-input pipeline
        (`TestAnalysisVectorTileRoute.test_rejects_a_raster_output_grid` in
        `test_vector_tiles.py`), not an unhandled `AttributeError`/500 -- which is exactly
        what happened before `PairedAnalysis` implemented this method at all (only a
        route-layer `hasattr` shim, since removed, papered over it).
        """
        query = pipeline_to_query(parse_pipeline(NOTEBOOK_QUERY))
        tile = _tile_for_grid(notebook_query_data)
        response = client.get(f"/v1/analysis/{tile.z}/{tile.x}/{tile.y}.pbf", params=query)
        assert response.status_code == 400, response.text
        assert "invalid output type for vector tile" in response.json()["detail"]

    def test_raster_tile_pipeline_error_is_a_400_not_a_500(
        self, client, notebook_query_data, monkeypatch
    ):
        """The sibling of the `.pbf` case above, for the RASTER tile route.

        `PairedAnalysis.validate_tile()` raises `PipelineError` (not `AnalysisError`) when
        a prepared paired plan has no band sources, and the routes catch `PipelineError`
        only around `pipeline_from_query`/`compile_pipeline` -- not around the
        `validate_tile()` call that follows. So that raise escaped as a 500 for what is a
        client error. An `@app.exception_handler(PipelineError)` backstop now gives it the
        same 400 the explicit `except` clauses produce.

        `validate_tile`'s own no-band-sources branch needs a paired plan that prepares to
        zero bands, which this fixture can't produce, so the raise is monkeypatched -- the
        handler mapping is what's under test here, not the branch's own condition.
        """
        monkeypatch.setattr(
            PairedAnalysis,
            "validate_tile",
            lambda self: (_ for _ in ()).throw(PipelineError("no data available")),
        )
        tile_query = copy.deepcopy(NOTEBOOK_QUERY)
        tile_query["output"] = {"format": "png"}
        query = pipeline_to_query(parse_pipeline(tile_query))
        tile = _tile_for_grid(notebook_query_data)
        response = client.get(f"/v1/analysis/{tile.z}/{tile.x}/{tile.y}.png", params=query)
        assert response.status_code == 400, response.text
        assert "no data available" in response.json()["detail"]


class TestPairedAccumulatorEdgeCases:
    """`_write_paired_notebook_raster`'s streaming per-band accumulator
    (`_accumulate_paired_window`) covers `sum`/`mean`/`min`/`max` -- `min` is already
    exercised byte-exact by `TestNotebookQueryEndToEnd` (via a uniform raster), but a
    uniform raster can't distinguish "zero bands validate" from "all bands validate" or
    exercise `mean`'s divide-by-VALID-count (not divide-by-total-bands) semantics. These
    use `accumulator_edge_case_data`'s three row bands -- see that fixture's docstring --
    to pin those cases against hand-computed values.
    """

    def _pipeline(self, aggregate):
        query = copy.deepcopy(NOTEBOOK_QUERY)
        query["steps"][-1]["aggregate"] = aggregate
        return query

    def test_sum_aggregate_by_valid_band_count(self, client, accumulator_edge_case_data):
        _base_path, third = accumulator_edge_case_data
        response = client.post("/v1/analysis.tif", json=self._pipeline("sum"))
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
            nodata = src.nodata
        assert nodata == -1
        # Never-valid band: no contributor at all -> nodata, not 0 (a naive sum-with-
        # null-as-zero over an ALL-null row would wrongly emit 0 here).
        assert (data[:third, :] == -1).all()
        # Valid only in 2030 -> sum is that single contributor, not padded with a zero
        # for 2040's non-contributing band.
        assert (data[third : 2 * third, :] == 2030).all()
        # Valid in both bands -> sum of both.
        assert (data[2 * third :, :] == 2030 + 2040).all()

    def test_mean_aggregate_divides_by_valid_count_not_band_count(self, client, accumulator_edge_case_data):
        _base_path, third = accumulator_edge_case_data
        response = client.post("/v1/analysis.tif", json=self._pipeline("mean"))
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
            nodata = src.nodata
        assert nodata == -1
        assert (data[:third, :] == -1).all()
        # One valid contributor -> mean IS that contributor (2030), not (2030 + 0) / 2 --
        # the bug a "divide by total band count" implementation would produce.
        assert (data[third : 2 * third, :] == 2030).all()
        # Two valid contributors -> their arithmetic mean.
        assert (data[2 * third :, :] == (2030 + 2040) // 2).all()

    def test_max_aggregate_all_masked_row_is_nodata(self, client, accumulator_edge_case_data):
        _base_path, third = accumulator_edge_case_data
        response = client.post("/v1/analysis.tif", json=self._pipeline("max"))
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
            nodata = src.nodata
        assert nodata == -1
        assert (data[:third, :] == -1).all()
        assert (data[third : 2 * third, :] == 2030).all()
        assert (data[2 * third :, :] == 2040).all()


class TestPairedAnalysisCountConsistency:
    """`count` on a paired pipeline: the zero-vs-nodata fixture scenarios (`count_edge_
    case_data`'s five row bands), the accumulator/wide-fallback consistency proof, and
    an ASGI tile test. `count` is in `_STREAMING_AGGREGATES` now, so a real request takes
    the streaming accumulator (`_accumulate_paired_window`), but `_collapse_paired_
    window_wide` -- also what `_paired_notebook_tile` always uses, and what any FUTURE
    non-streaming aggregate would fall back to -- must produce IDENTICAL output; the
    second test below proves that directly by forcing the wide path via the same
    `_STREAMING_AGGREGATES`-monkeypatch trick `/tmp/pipetest/scale_probe.py` uses to
    measure it at scale, applied here to prove the two paths AGREE rather than just to
    measure one of them.
    """

    def _pipeline(self):
        query = copy.deepcopy(NOTEBOOK_QUERY)
        query["steps"][-1] = {"op": "group", "by": [], "aggregate": "count"}
        return query

    def test_zero_vs_nodata_and_mixed_pixel(self, client, count_edge_case_data):
        _base_path, fifth = count_edge_case_data
        response = client.post("/v1/analysis.tif", json=self._pipeline())
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
            nodata = src.nodata
        assert nodata == -1
        assert data.dtype == np.int32
        assert (data[:fifth, :] == 0).all()  # all-bands-fail (in-domain) -> real 0
        assert (data[fifth : 2 * fifth, :] == 0).all()  # fine-nodata-all-bands -> real 0
        assert (data[2 * fifth : 3 * fifth, :] == 0).all()  # pairing-miss-all-bands -> real 0
        assert (data[3 * fifth : 4 * fifth, :] == -1).all()  # outside-zone -> nodata
        assert (data[4 * fifth :, :] == 1).all()  # mixed pixel -> exact count

    def test_accumulator_matches_wide_fallback(self, client, count_edge_case_data, monkeypatch):
        # The streaming path (default -- `count` is in `_STREAMING_AGGREGATES`).
        streaming_response = client.post("/v1/analysis.tif", json=self._pipeline())
        assert streaming_response.status_code == 200, streaming_response.text
        with rio.open(io.BytesIO(streaming_response.content)) as src:
            streaming_data = src.read(1)

        # Force every aggregate, `count` included, onto `_collapse_paired_window_wide`.
        monkeypatch.setattr(paired_module, "_STREAMING_AGGREGATES", frozenset())
        wide_response = client.post("/v1/analysis.tif", json=self._pipeline())
        assert wide_response.status_code == 200, wide_response.text
        with rio.open(io.BytesIO(wide_response.content)) as src:
            wide_data = src.read(1)

        assert np.array_equal(streaming_data, wide_data)

    def test_get_png_tile(self, client, count_edge_case_data):
        base_path, _fifth = count_edge_case_data
        tile_query = self._pipeline()
        tile_query["output"] = {"format": "png"}
        query = pipeline_to_query(parse_pipeline(tile_query))
        tile = _tile_for_grid(base_path)
        response = client.get(f"/v1/analysis/{tile.z}/{tile.x}/{tile.y}.png", params=query)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


class TestSingleInputEndToEnd:
    """Single-input pipelines equivalent to today's flat requests: an explicit (identity)
    leading `project`, since `filter`/`group` can never open a pipeline, plus optional
    `filter`/`group` and a tabular/tif output.
    """

    VECTOR_PIPELINE = {
        "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
        "steps": [
            {"op": "project", "input": "eco", "to": "ecoregion_v001"},
            {"op": "filter", "where": "eco > 2035"},
            {"op": "group", "by": [], "aggregate": "max"},
        ],
        "output": {"format": "csv"},
    }

    def test_vector_identity_project_filter_group(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        response = client.post("/v1/analysis.csv", json=self.VECTOR_PIPELINE)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert df.height == 32  # one row per ecoregion -- 2030 fails `eco > 2035` everywhere.
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))
        for geo_id in range(1, 33):
            assert actual[geo_id] == pytest.approx(2040 + geo_id / 100, abs=1e-3)

    def test_raster_identity_project_group_only(self, client, base_path, monkeypatch):
        """A raster counterpart, without a `filter` step -- see the module docstring's
        note on why a raster `filter` needs a summarize target, which this shape avoids
        by using only `group` (a pure band-collapse `Options` already supports without
        one).
        """
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                {"op": "group", "by": [], "aggregate": "max"},
            ],
            "output": {"format": "tif"},
        }
        response = client.post("/v1/analysis.tif", json=pipeline)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
        assert (data == 40).all()

    def test_get_dotted_query_matches_post(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        query = pipeline_to_query(parse_pipeline(self.VECTOR_PIPELINE))
        post_response = client.post("/v1/analysis.csv", json=self.VECTOR_PIPELINE)
        get_response = client.get("/v1/analysis.csv", params=query)
        assert get_response.status_code == 200, get_response.text
        # Row order isn't guaranteed stable across two independent query-plan executions
        # (no explicit sort in this pipeline), so compare parsed/sorted frames rather than
        # raw bytes.
        post_df = pl.read_csv(io.BytesIO(post_response.content)).sort("grid_cell")
        get_df = pl.read_csv(io.BytesIO(get_response.content)).sort("grid_cell")
        assert get_df.equals(post_df)


class TestFilterWhereDimensionNotEndToEnd:
    """G8: a pipeline `filter` step's `where` must accept exactly what the flat `where=`
    parameter accepts for a dimension-only conjunct -- including a `not (...)` group, which
    the portal actually emits (e.g. `not (model_group_id=X and <dim>=Y)`). Before this fix,
    every top-level conjunct (dimension or value) was forced through
    `expressions.parse_value_expr`'s restricted whitelist, which has no `not` in
    `ALLOWED_FUNCTIONS` and so rejected this shape outright, even though it never
    references `value` at all.

    The real assertion here isn't just "the pipeline compiles" (that's
    `TestCompileChecks.test_validate_filter_where_accepts_dimension_not`) -- it's that the
    pipeline surface and the flat `where=` surface, given the logically identical
    predicate, return the SAME ROWS.
    """

    def test_not_filter_matches_flat_where_equivalent(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                {"op": "filter", "where": "not (calendar_year = 2030 and grid_cell = 5)"},
            ],
            "output": {"format": "csv"},
        }
        pipeline_response = client.post("/v1/analysis.csv", json=pipeline)
        assert pipeline_response.status_code == 200, pipeline_response.text
        pipeline_df = pl.read_csv(io.BytesIO(pipeline_response.content)).sort(
            ["calendar_year", "grid_cell"]
        )

        # One row -- (calendar_year=2030, grid_cell=5) -- is excluded out of 64 total
        # (32 grid cells x 2 years); every other combination is kept.
        assert pipeline_df.height == 63
        assert not (
            (pipeline_df["calendar_year"] == 2030) & (pipeline_df["grid_cell"] == 5)
        ).any()

        flat_response = client.get(
            "/v1/datasets/ssf/annual/ecoregion_v001/data.csv",
            params={"where": "not (calendar_year = 2030 and grid_cell = 5)"},
        )
        assert flat_response.status_code == 200, flat_response.text
        flat_df = pl.read_csv(io.BytesIO(flat_response.content)).sort(["calendar_year", "grid_cell"])

        assert flat_df.select(["calendar_year", "grid_cell", "value"]).equals(
            pipeline_df.select(["calendar_year", "grid_cell", "value"])
        )


class TestInputGridCellEndToEnd:
    """`input.<name>.grid_cell` (G1), through both request encodings, filters the input
    the same way a flat `grid_cell=` query parameter would."""

    PIPELINE = {
        "inputs": {
            "eco": {
                **{k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")},
                "grid_cell": "5,6",
            }
        },
        "steps": [{"op": "project", "input": "eco", "to": "ecoregion_v001"}],
        "output": {"format": "csv"},
    }

    def test_grid_cell_filters_the_input(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        response = client.post("/v1/analysis.csv", json=self.PIPELINE)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert sorted(df["grid_cell"].to_list()) == [5, 6]

    def test_get_dotted_query_matches_post(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        query = pipeline_to_query(parse_pipeline(self.PIPELINE))
        assert query["input.eco.grid_cell"] == "5,6"
        post_df = pl.read_csv(
            io.BytesIO(client.post("/v1/analysis.csv", json=self.PIPELINE).content)
        ).sort("grid_cell")
        get_response = client.get("/v1/analysis.csv", params=query)
        assert get_response.status_code == 200, get_response.text
        get_df = pl.read_csv(io.BytesIO(get_response.content)).sort("grid_cell")
        assert get_df.equals(post_df)


class TestOutputSortSelectEndToEnd:
    """`output.sort`/`output.select` (G2) actually sort/narrow the response, not just
    lower onto the right `Options` attributes (see `TestOutputSortSelectCompile`)."""

    def test_sort_and_select_applied(self, client, base_path, monkeypatch):
        # `sort`/`select`'s flat semantics (parameters.py/base.py) accept DIMENSION names
        # (`SORT_CHOICES` is built from `DIMENSIONS`, which has no `grid_cell`/"value"
        # entry) -- so this sorts by `calendar_year`, descending, matching the existing
        # flat-endpoint convention (`test_endpoints_data.py`'s `sort=-calendar_year`).
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [{"op": "project", "input": "eco", "to": "ecoregion_v001"}],
            "output": {"format": "csv", "sort": "-calendar_year", "select": "calendar_year,value"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert list(df.columns) == ["calendar_year", "value"]
        assert df["calendar_year"].to_list() == sorted(df["calendar_year"].to_list(), reverse=True)


class TestOutputColorMapEndToEnd:
    """`output.color_map` actually changes the rendered raster tile's pixels for a
    SINGLE-INPUT pipeline. Previously this could only be demonstrated working through the
    paired shape (`TestCustomAnalysisColorMap` in test_analyses_endpoints.py, against the
    `exit-criterion` analysis), because `_compile_single_input_chain` silently dropped
    `output.color_map` until this fix, even though the route layer accepted and forwarded
    it correctly. A dataset-scoped single-input pipeline rendered as a raster PNG tile
    (`/v1/analysis/{z}/{x}/{y}.png`) is exactly the combination that was blocked before
    this fix (the portal's difference map on a raster grid is the motivating consumer),
    so this proves the real consumer shape works, not just that
    `compiled.options.color_map` gets set (see
    `TestOutputColorMapCompile` for that).
    """

    def test_color_map_changes_rendered_pixels(self, client, base_path, monkeypatch):
        # A uniform fill of 30 falls in a distinct bucket under both legends used below:
        # `default`'s "0 to 49" bucket vs. `ffibs`'s last (open-ended, > 17) "Saline
        # Marsh" bucket -- so the two color maps are guaranteed to paint different colors
        # for the same pixels, not just POSSIBLY differ depending on how the fixture data
        # happens to fall relative to two arbitrary legends' bucket edges.
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 30)
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        tile = _tile_for_grid(base_path)

        def _pipeline(color_map):
            return {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [{"op": "project", "input": "vct", "to": "veg_grid_cell_v001"}],
                "output": {"format": "png", "color_map": color_map},
            }

        default_query = pipeline_to_query(parse_pipeline(_pipeline("default")))
        ffibs_query = pipeline_to_query(parse_pipeline(_pipeline("ffibs")))

        default_response = client.get(f"/v1/analysis/{tile.z}/{tile.x}/{tile.y}.png", params=default_query)
        ffibs_response = client.get(f"/v1/analysis/{tile.z}/{tile.x}/{tile.y}.png", params=ffibs_query)

        assert default_response.status_code == 200, default_response.text
        assert ffibs_response.status_code == 200, ffibs_response.text
        assert default_response.content != ffibs_response.content


class TestProjectAggregateCountByValueEndToEnd:
    """`count_by_value` as a `project` aggregate (G3) -- the documented "Categorical
    Count per Zone" query (docs/data-analysis-api.md) expressed as a pipeline. Locally
    constructed categorical fixture, modeled on `test_endpoints_data.py`'s
    `categorical_vector_data` (off limits to import from directly per this task's file
    ownership split)."""

    def test_categorical_count_per_zone(self, client, base_path, crosswalk_path, monkeypatch):
        # geo_id 1..16 -> zone 100, geo_id 17..32 -> zone 200; value = geo_id % 3, so
        # each zone's per-value counts are hand-computable below.
        write_parquet(
            annual_frame(years=(2030,), value_fn=lambda _year, geo_id: float(geo_id % 3)),
            base_path=base_path,
        )
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        write_vector_crosswalk(
            crosswalk_path,
            "ecoregion_v001",
            "hydro_compartment_v001",
            [(geo_id, 100 if geo_id <= 16 else 200, 1.0) for geo_id in range(1, 33)],
        )

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {
                    "op": "project",
                    "input": "eco",
                    "to": "hydro_compartment_v001",
                    "aggregate": "count_by_value",
                }
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))

        expected = {}
        for geo_id in range(1, 33):
            zone = 100 if geo_id <= 16 else 200
            value = float(geo_id % 3)
            expected[(zone, value)] = expected.get((zone, value), 0) + 1
        actual = {
            (int(row["grid_cell"]), float(row["value"])): int(row["count"])
            for row in df.iter_rows(named=True)
        }
        assert actual == expected

    def test_rejected_as_a_group_aggregate(self, client, base_path, monkeypatch):
        """The other half of G3: `count_by_value` stays rejected as a `group.aggregate`
        (a running per-row fold across an offset-batched read has no meaning for it) --
        end-to-end confirmation that `pipeline.py::_validate_step_rules` still enforces
        this for `group` even though `project` now permits it.
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                {"op": "group", "by": [], "aggregate": "count_by_value"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 400
        assert "offset-batched fold" in response.json()["detail"]


class TestPostProjectGridCellFilterEndToEnd:
    """A `filter` naming `grid_cell` after a many-to-one `project` (G4) must be lowered
    onto `summarize_grid_cell` -- i.e. filter the SUMMARIZED zones, not the source
    grid's own cells (which would silently drop everything, since the source grid's
    cell IDs and the summarized zone IDs live in different ID spaces here)."""

    def _setup(self, base_path, crosswalk_path, monkeypatch):
        grid = GRIDS[RASTER_KWARGS["grid"]]
        zones = np.zeros((grid.height, grid.width), dtype="int32")
        zones[: grid.height // 2, :] = 1
        zones[grid.height // 2 :, :] = 2
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _y: raster_array())
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        return zones

    def test_grid_cell_eq_lowers_onto_summarize_grid_cell(self, client, base_path, crosswalk_path, monkeypatch):
        zones = self._setup(base_path, crosswalk_path, monkeypatch)
        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"},
                {"op": "filter", "where": "grid_cell = 1"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["grid_cell"].to_list() == [1]
        categories = raster_array()
        expected_mean = categories[zones == 1].astype("float64").mean()
        assert df["value"][0] == pytest.approx(expected_mean, rel=1e-4)

    def test_grid_cell_in_lowers_onto_summarize_grid_cell(self, client, base_path, crosswalk_path, monkeypatch):
        self._setup(base_path, crosswalk_path, monkeypatch)
        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"},
                {"op": "filter", "where": "grid_cell IN (1, 2)"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert sorted(df["grid_cell"].to_list()) == [1, 2]

    def test_identity_project_grid_cell_filter_is_unaffected(self, client, base_path, monkeypatch):
        """Non-regression: an identity project (no `summarize_by`) still folds a
        `grid_cell` filter into `Options.where` as an ordinary source-grid filter -- G4's
        recognition only special-cases a many-to-one (summarizing) `project`."""
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                {"op": "filter", "where": "grid_cell = 5"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["grid_cell"].to_list() == [5]


class TestPostProjectGridCellFilterRejections:
    """The other half of G4, which must ship together with the success path: any
    post-project `grid_cell` filter that ISN'T a clean literal ID selection is rejected
    outright rather than silently mis-executed against the wrong grid."""

    def _pipeline(self, where):
        return parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"},
                    {"op": "filter", "where": where},
                ],
                "output": {"format": "csv"},
            }
        )

    def test_rejects_mixed_grid_cell_and_other_column(self, base_path):
        pipeline = self._pipeline("grid_cell = 1 AND calendar_year = 2030")
        with pytest.raises(PipelineError, match="mixes `grid_cell`"):
            compile_pipeline(pipeline, base_path=base_path)

    def test_rejects_non_extractable_predicate(self, base_path):
        pipeline = self._pipeline("grid_cell > 100")
        with pytest.raises(PipelineError, match="literal ID selection"):
            compile_pipeline(pipeline, base_path=base_path)

    async def test_second_post_project_grid_cell_filter_compiles_and_executes_for_raster_input(
        self, base_path, crosswalk_path
    ):
        """This replaces a prior test that pinned the
        `PipelineError` this rejection used to raise unconditionally for a raster input.
        That rejection is now relaxed for raster too, on the same grounds it was already
        relaxed for vector (`planner._compile_single_input_chain`'s comment on this
        rejection): a `ProjectOp` that summarizes onto a coarser grid genuinely
        materializes `self.lf` before any post-project `FilterOp` runs
        (`RasterGridAnalysis.apply_steps`'s POST-project branch), so a SECOND `grid_cell`
        filter is just an ordinary additional row-drop, ANDed with the first, exactly like
        vector's `test_second_post_project_grid_cell_filter_compiles_and_executes_for_
        vector_input` above -- no `Options.summarize_grid_cell` bucket to collide over at
        execution time even though it is still the (single, last-write-wins) bucket used
        to narrow the initial crosswalk scan.

        This is a HAND-COMPUTED correctness check, not a legacy-engine differential: the
        OLD fixed-stage engine cannot compute the right answer for this shape at all --
        `_extract_post_project_grid_cell_filter` absorbs BOTH `grid_cell` filters into the
        same single `Options.summarize_grid_cell` bucket (last-write-wins), so the legacy
        engine would narrow the crosswalk scan to only the SECOND filter's zones ({2, 3})
        and never intersect it with the first ({1, 2}) at all -- it would wrongly keep
        zone 3. Only the interpreter's real sequential `FilterOp` application computes the
        declared intersection correctly, so comparing against the legacy engine here would
        be comparing against a wrong answer, not a matching one.
        """
        grid = GRIDS[RASTER_KWARGS["grid"]]
        third = grid.height // 3
        zones = np.empty((grid.height, grid.width), dtype="int32")
        zones[:third, :] = 1
        zones[third : 2 * third, :] = 2
        zones[2 * third :, :] = 3
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _y: raster_array())

        pipeline = parse_pipeline(
            {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [
                    {"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"},
                    {"op": "filter", "where": "grid_cell IN (1, 2)"},
                    {"op": "filter", "where": "grid_cell IN (2, 3)"},
                ],
                "output": {"format": "csv"},
            }
        )
        compiled = compile_pipeline(pipeline, base_path=base_path)
        assert isinstance(compiled, BaseAnalysis)
        await compiled.apply_params()
        df = compiled.materialize()
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))

        categories = raster_array()
        expected_mean = categories[zones == 2].astype("float64").mean()
        assert actual == pytest.approx({2: expected_mean}, rel=1e-4)


class TestCalculateStepEndToEnd:
    """The `calculate` op through all three request encodings (POST JSON, dotted GET,
    tile route), plus the paired-shape rejection and the flat-request equivalence."""

    WEIGHTED_PIPELINE = {
        "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
        "steps": [
            {"op": "project", "input": "eco", "to": "ecoregion_v001"},
            {"op": "calculate", "expression": "eco * 0.5", "where": "calendar_year = 2030"},
            {"op": "calculate", "expression": "eco * 0.25", "where": "calendar_year = 2040"},
            {"op": "group", "by": [], "aggregate": "sum"},
        ],
        "output": {"format": "csv"},
    }

    def test_post_json(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        response = client.post("/v1/analysis.csv", json=self.WEIGHTED_PIPELINE)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))
        for geo_id in (1, 32):
            expected = (2030 + geo_id / 100) * 0.5 + (2040 + geo_id / 100) * 0.25
            assert actual[geo_id] == pytest.approx(expected, rel=1e-4)

    def test_get_dotted_query_matches_post(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        query = pipeline_to_query(parse_pipeline(self.WEIGHTED_PIPELINE))
        post_df = pl.read_csv(
            io.BytesIO(client.post("/v1/analysis.csv", json=self.WEIGHTED_PIPELINE).content)
        ).sort("grid_cell")
        get_response = client.get("/v1/analysis.csv", params=query)
        assert get_response.status_code == 200, get_response.text
        get_df = pl.read_csv(io.BytesIO(get_response.content)).sort("grid_cell")
        assert get_df.equals(post_df)

    def test_tile_route(self, client, base_path, monkeypatch):
        """A raster single-input chain with calculate steps, rendered on the analysis
        tile route: bands weighted 0.5/0.25 and collapsed to the single band a tile
        requires."""
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                {"op": "calculate", "expression": "vct * 0.5", "where": "calendar_year = 2030"},
                {"op": "calculate", "expression": "vct * 0.25", "where": "calendar_year = 2040"},
                {"op": "group", "by": [], "aggregate": "sum"},
            ],
            "output": {"format": "png"},
        }
        query = pipeline_to_query(parse_pipeline(pipeline))

        with rio.open(next(base_path.rglob("*.tif"))) as src:
            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        tile = mercantile.tile((west + east) / 2, (south + north) / 2, 8)

        response = client.get(f"/v1/analysis/{tile.z}/{tile.x}/{tile.y}.png", params=query)
        assert response.status_code == 200, response.text
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_weighted_sum_with_a_grid_cell_filter(self, client, base_path, monkeypatch):
        """The confidence-weighting idiom (moved here from `test_endpoints_data.py`'s
        retired `TestCalculateParams`, since `calculate.N.expression`/`calculate.N.where`
        have no flat spelling anymore), combined with a `grid_cell` filter on the
        pipeline's own input -- `InputSelection.grid_cell` is a reserved field, exactly
        like `Options.grid_cell`'s flat spelling."""
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {
                "ssf": {
                    "variable": "ssf",
                    "time_unit": "annual",
                    "grid": "ecoregion_v001",
                    "grid_cell": "1",
                }
            },
            "steps": [
                {"op": "project", "input": "ssf", "to": "ecoregion_v001"},
                {"op": "calculate", "expression": "value * 0.5", "where": "calendar_year = 2030"},
                {"op": "calculate", "expression": "value * 0.25", "where": "calendar_year = 2040"},
                {"op": "group", "by": [], "aggregate": "sum"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert df.filter(pl.col("grid_cell") == 1)["value"][0] == pytest.approx(
            2030.01 * 0.5 + 2040.01 * 0.25, rel=1e-5
        )

    def test_non_matching_rows_pass_through_with_no_group_step(self, client, base_path, monkeypatch):
        """Rows/bands not matching a `calculate` step's `where` pass through unchanged --
        `calculate` never drops data, unlike `filter` (moved here from
        `test_endpoints_data.py`'s retired `TestCalculateParams`)."""
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {
                "ssf": {
                    "variable": "ssf",
                    "time_unit": "annual",
                    "grid": "ecoregion_v001",
                    "grid_cell": "1",
                }
            },
            "steps": [
                {"op": "project", "input": "ssf", "to": "ecoregion_v001"},
                {"op": "calculate", "expression": "value * 0", "where": "calendar_year = 2030"},
            ],
            "output": {"format": "csv", "sort": "calendar_year"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["value"].to_list() == pytest.approx([0.0, 2040.01], rel=1e-5)

    def test_weighted_band_sum_on_the_data_route(self, client, base_path, monkeypatch):
        """The same weighted band-sum idiom rendered through the plain (non-tile)
        `.tif` output, complementing `test_tile_route` above (moved here from
        `test_endpoints_data.py`'s retired `TestCalculateParams`)."""
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                {"op": "calculate", "expression": "value * 0.5", "where": "calendar_year = 2030"},
                {"op": "calculate", "expression": "value * 0.25", "where": "calendar_year = 2040"},
                {"op": "group", "by": [], "aggregate": "sum"},
            ],
            "output": {"format": "tif"},
        }
        response = client.post("/v1/analysis.tif", json=pipeline)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            assert src.count == 1
            assert src.read(1)[0, 0] == 25

    def test_flat_calculate_params_are_retired_on_the_data_route(self, client, base_path, monkeypatch):
        """The flat `calculate.N.expression`/`calculate.N.where` query parameters this
        class used to compare its pipeline steps against are retired from the plain data
        route: the `calculate` pipeline step above is now the only way to
        express this, and a request still spelling it the old way gets a loud 400 naming
        the replacement rather than being silently ignored."""
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        flat_response = client.get(
            "/v1/datasets/ssf/annual/ecoregion_v001/data.csv",
            params={
                "calculate.1.expression": "value * 0.5",
                "calculate.1.where": "calendar_year = 2030",
                "calculate.2.expression": "value * 0.25",
                "calculate.2.where": "calendar_year = 2040",
                "group_aggregate": "sum",
            },
        )
        assert flat_response.status_code == 400
        detail = flat_response.json()["detail"]
        assert "calculate" in detail
        assert "group_aggregate" in detail

    def test_paired_shape_rejects_calculate(self, client, notebook_query_data):
        pipeline = copy.deepcopy(NOTEBOOK_QUERY)
        pipeline["steps"].insert(2, {"op": "calculate", "expression": "sal * 0.5"})
        response = client.post("/v1/analysis.tif", json=pipeline)
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "not yet supported" in detail
        assert "paired" in detail


class TestProjectOnlyReaggregationEndToEnd:
    """One input + `project to:<coarser grid> aggregate:mean` + tabular output -- no
    `filter`/`group` needed at all.
    """

    def test_raster_to_vector_mean(self, client, base_path, crosswalk_path, monkeypatch):
        grid = GRIDS[RASTER_KWARGS["grid"]]
        zones = np.zeros((grid.height, grid.width), dtype="int32")
        zones[: grid.height // 2, :] = 1
        zones[grid.height // 2 :, :] = 2
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _y: raster_array())
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [{"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"}],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))

        categories = raster_array()
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))
        for zone_id in (1, 2):
            expected_mean = categories[zones == zone_id].astype("float64").mean()
            assert actual[zone_id] == pytest.approx(expected_mean, rel=1e-4)


class TestRejections:
    def test_post_format_mismatch_is_400(self, client, notebook_query_data):
        response = client.post("/v1/analysis.csv", json=NOTEBOOK_QUERY)
        assert response.status_code == 400
        assert "does not match" in response.json()["detail"]

    def test_get_dotted_format_mismatch_is_400(self, client, notebook_query_data):
        query = pipeline_to_query(parse_pipeline(NOTEBOOK_QUERY))
        response = client.get("/v1/analysis.csv", params=query)
        assert response.status_code == 400
        assert "does not match" in response.json()["detail"]

    def test_three_inputs_is_400(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {
                "a": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")},
                "b": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")},
                "c": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")},
            },
            "steps": [
                {"op": "project", "input": "a", "to": "ecoregion_v001", "as": "a2"},
                {"op": "join", "type": "spatial", "inputs": ["a2", "b"], "as": "ab"},
                {"op": "join", "type": "spatial", "inputs": ["ab", "c"], "as": "abc"},
                {"op": "group", "by": [], "aggregate": "mean"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 400
        assert "not yet supported" in response.json()["detail"]

    def test_tile_route_rejects_vector_output(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [{"op": "project", "input": "eco", "to": "ecoregion_v001"}],
            "output": {"format": "png"},
        }
        query = pipeline_to_query(parse_pipeline(pipeline))
        response = client.get("/v1/analysis/3/1/2.png", params=query)
        assert response.status_code == 400
        assert "invalid output type" in response.json()["detail"]

    def test_raster_filter_without_summarize_is_400(self, client, base_path, monkeypatch):
        """The pre-existing `BaseAnalysis` constraint this module's docstring calls out:
        a `filter` over a raw (non-reaggregated) raster grid has no per-pixel frame to
        filter -- compiles fine (nothing wrong with the pipeline shape itself) but 400s
        once `BaseAnalysis.apply_params()` actually runs.
        """
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "project", "input": "vct", "to": "veg_grid_cell_v001"},
                {"op": "filter", "where": "vct > 35"},
                {"op": "group", "by": [], "aggregate": "min"},
            ],
            "output": {"format": "tif"},
        }
        response = client.post("/v1/analysis.tif", json=pipeline)
        assert response.status_code == 400
        assert "summarize_by" in response.json()["detail"]
