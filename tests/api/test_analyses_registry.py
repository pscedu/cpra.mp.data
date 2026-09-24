"""The registry-drift test, plus an execution check per registered analysis.

Every entry in `cpra.mp.data.analyses.ANALYSES` must resolve (via `resolve_analysis`) to a
real `Pipeline` that `request.ops.validate_shapes` and `execution.plan.compile_plan`
accept against a representative, on-disk dataset -- enforcing the constraint that nothing
is reachable only through a named analysis, only through the exact same request/execution
path any hand-authored pipeline goes through. Everything here drives
`validate_shapes`/`compile_plan`/`execute`/`run_to_path` directly, exactly as
`tests/api/test_execute.py` does for hand-authored pipelines.

**Representative datasets.** `difference`/`concatenate` are `type="dataset"` and take
whatever `DatasetRef` a route resolves for them; `tests/factories.py`'s `ssf`/
`ecoregion_v001` (`ANNUAL_KWARGS`) is used here as that representative dataset, per that
module's own documented convention ("the compiler cares about grid TYPES and SHAPES, not
real-world variable semantics") -- `concatenate`'s real `Meta.variables = ["lnd_type"]`
lives on `morph_pixel_v001` (8265x20674 pixels), far too large to write as a test fixture;
the `Meta.variables` declaration itself is unaffected by this substitution, only which
dataset proves the pipeline-construction MACHINERY works.

`salinity-veg-investigation` is `type="custom"` with hardcoded inputs
(`ffibs_coverage_ratio`/`veg_grid_cell_v001`, `sal`/`hydro_compartment_v001`), written by
`tests/factories.py`'s `write_ffibs_coverage_dataset`/`write_sal_annual_dataset` and
assembled by the `salinity_veg_data` fixture in `tests/api/conftest.py` -- shared with
`tests/api/test_analyses_endpoints.py`, which needs the same data to drive the same
analysis through its HTTP routes.

`land-type-confidence` is also `type="dataset"`, but its own `Meta.variables = ["lnd_type"]`
lives only on a raster grid and its `pipeline()` sinks to `tif`, so a vector stand-in would
not survive `validate_shapes`; it runs on `tests/factories.py`'s `vct`/`veg_grid_cell_v001`
raster stand-in (`RASTER_KWARGS`) instead, through `run_to_path` rather than `_run`.

`lnd-type-plus` is `type="custom"` too, with THREE hardcoded inputs across two grids
(`elev` and `lnd_type` on `morph_pixel_v002`, `stg` on `hydro_compartment_v002`) -- and
those are the one real, un-substituted grid pair in this module, because the analysis names
them itself. At its real size `morph_pixel_v002` is ~97 million pixels per band, so the
`morph_pixel_v002_shrunk` fixture in `tests/api/conftest.py` shrinks the grid OBJECT to 2x4
instead of substituting a stand-in dataset; `tests/factories.py`'s
`write_lnd_type_plus_datasets` then writes all three inputs and the crosswalk between the
two grids. It does not store depths: the analysis COMPUTES depth as
`stage - elevation/1000`, so the factory solves for the elevation that yields each depth a
test asks for.

**The band-wide entries get their own execution classes.** The long-shaped registered
analyses run through `_run` below; `salinity-veg-investigation` is band-wide end to end and
is the only entry that exercises a vector-to-raster `project` feeding a band-wide `join`
together with a `filter` naming that join's operand value columns, so
`TestSalinityVegInvestigationExecution` runs it separately and pins the two halves apart.
`lnd-type-plus` is band-wide end to end as well, and is the only entry that collapses a
band-wide join with a `group` -- whose horizontal reduce IS both its subtraction (`sum`
over a negated operand) and its "max of the two values" step -- so
`TestLndTypePlusExecution` pins that against a raster small enough to assert pixel by
pixel.
"""

import pytest
import rasterio as rio

from cpra.mp.data import config
from cpra.mp.data.analyses import ANALYSES, DatasetRef, resolve_analysis
from cpra.mp.data.analyses.land_type_confidence import SCENARIO_WEIGHTS
from cpra.mp.data.analysis.execution.execute import execute, run_to_path
from cpra.mp.data.analysis.execution.plan import compile_plan
from cpra.mp.data.analysis.request.ops import validate_shapes
from tests.factories import (
    ANNUAL_KWARGS,
    LND_TYPE_PLUS_SELECTION,
    RASTER_KWARGS,
    write_annual_dataset,
    write_ffibs_coverage_dataset,
    write_lnd_type_plus_datasets,
    write_raster_dataset,
)

SSF_DATASET = DatasetRef(
    variable=ANNUAL_KWARGS["variable"], time_unit=ANNUAL_KWARGS["time_unit"], grid=ANNUAL_KWARGS["grid"]
)
COMMON_ANNUAL_FILTERS = {
    "model_group_id": ANNUAL_KWARGS["model_group_id"],
    "scenario_id": ANNUAL_KWARGS["scenario_id"],
    "percentile": ANNUAL_KWARGS["percentile"],
}
# `land-type-confidence`'s real dataset is `lnd_type`, whose only grid in `variables.csv`
# is `morph_pixel` -- a RASTER grid, and far too large to write as a fixture. The
# representative dataset here is therefore `vct`/`veg_grid_cell_v001` (`RASTER_KWARGS`),
# the same raster stand-in `tests/api/test_analyses_endpoints.py` uses, rather than the
# vector `ssf` the other dataset-scoped entries run on: the analysis's own `output.format`
# is `tif`, so only a raster selection gets past `validate_shapes`.
RASTER_DATASET = DatasetRef(
    variable=RASTER_KWARGS["variable"], time_unit=RASTER_KWARGS["time_unit"], grid=RASTER_KWARGS["grid"]
)
# It supplies `scenario_id` itself and rejects a caller-selected one, so its filters are
# `vct`'s other required dimensions plus the single year the ensemble is written for.
ENSEMBLE_FILTERS = {"model_group_id": RASTER_KWARGS["model_group_id"], "calendar_year": 2050}


def _write_ensemble(base_path, code_by_scenario, years=(2050,)):
    """Write one COG per `scenario_id`, each uniformly filled with a `lnd_type` code. The
    stand-in variable's own semantics never come into it -- the analysis only ever
    compares `value` against `1`-`5`.
    """
    for scenario_id, code in code_by_scenario.items():
        write_raster_dataset(
            base_path, years=years, scenario_id=scenario_id, fill_fn=lambda year, code=code: code
        )


# `lnd-type-plus`'s parameters are its inputs' shared selection plus the one year the
# fixtures write. All three inputs are named by the analysis itself, so none of this is a
# filter -- and `aggregate_type`, which only `stg` carries, is pinned by the analysis too.
LND_TYPE_PLUS_PARAMS = {**LND_TYPE_PLUS_SELECTION, "calendar_year": 2050}


def _run(resolved, base_path):
    """Take a `ResolvedAnalysis` all the way through the real request/execution path --
    `validate_shapes` -> `compile_plan` -> `execute` -> `.collect()` -- and return the
    resulting `DataFrame`. Every entry run through this helper is long-shaped (a vector
    grid, no `project`), so `execute`'s result is always a `LongFrame`.
    """
    validate_shapes(resolved.pipeline)
    plan = compile_plan(resolved.pipeline, base_path=base_path)
    return execute(plan, base_path=base_path).collect()


# ------------------------------------------------------------------------------------------
# Every registered analysis resolves, type-checks and compiles
# ------------------------------------------------------------------------------------------


class TestRegistryDrift:
    """Every `ANALYSES` entry, expanded via `resolve_analysis` and run through
    `validate_shapes` + `compile_plan` against a representative on-disk dataset -- catches
    an analysis added to the registry whose `pipeline()` builds something the request or
    execution layer doesn't actually accept (nothing may be reachable only through a named
    analysis).
    """

    def test_every_registered_analysis_name_is_covered_by_this_test(self):
        # If another analysis lands and nobody updates the tests below, this is the
        # tripwire: it fails loudly instead of the drift test silently covering only some
        # of the entries.
        assert set(ANALYSES) == {
            "salinity-veg-investigation",
            "difference",
            "concatenate",
            "land-type-confidence",
            "lnd-type-plus",
        }

    def test_difference_compiles(self, base_path):
        write_annual_dataset(base_path, years=(2030, 2040))
        resolved = resolve_analysis(
            "difference",
            dataset=SSF_DATASET,
            filters={**COMMON_ANNUAL_FILTERS, "calendar_year": [2030, 2040]},
            params={"dimension": "calendar_year", "order": "asc"},
        )
        validate_shapes(resolved.pipeline)
        assert compile_plan(resolved.pipeline, base_path=base_path) is not None

    def test_concatenate_two_image_compiles(self, base_path):
        write_annual_dataset(base_path, years=(2030, 2040))
        resolved = resolve_analysis(
            "concatenate",
            dataset=SSF_DATASET,
            filters={**COMMON_ANNUAL_FILTERS, "calendar_year": [2030, 2040]},
            params={"dimension": "calendar_year"},
        )
        validate_shapes(resolved.pipeline)
        assert compile_plan(resolved.pipeline, base_path=base_path) is not None

    def test_concatenate_three_image_compiles(self, base_path):
        write_annual_dataset(base_path, years=(2030, 2040), model_group_id=500)
        write_annual_dataset(base_path, years=(2030, 2040), model_group_id=520)
        resolved = resolve_analysis(
            "concatenate",
            dataset=SSF_DATASET,
            filters={
                "scenario_id": ANNUAL_KWARGS["scenario_id"],
                "percentile": ANNUAL_KWARGS["percentile"],
                "calendar_year": [2030, 2040],
                "model_group_id": [500, 520],
            },
            params={"dimension": "calendar_year"},
        )
        validate_shapes(resolved.pipeline)
        assert compile_plan(resolved.pipeline, base_path=base_path) is not None

    def test_concatenate_compiles_on_a_raster_grid_too(self, base_path):
        """`concatenate`'s own `Meta.variables` (`lnd_type`) is raster-only, so the raster
        selection is the one it really runs under -- and its `output.format` is derived
        from the grid kind, which `validate_shapes` then checks against the final frame.
        The two-image case above proves the vector half of the same derivation.
        """
        write_raster_dataset(base_path, years=(2030, 2040))
        resolved = resolve_analysis(
            "concatenate",
            dataset=RASTER_DATASET,
            filters={
                "model_group_id": RASTER_KWARGS["model_group_id"],
                "scenario_id": RASTER_KWARGS["scenario_id"],
                "calendar_year": [2030, 2040],
            },
            params={"dimension": "calendar_year"},
        )
        assert resolved.pipeline.output.format == "tif"
        validate_shapes(resolved.pipeline)
        assert compile_plan(resolved.pipeline, base_path=base_path) is not None

    def test_land_type_confidence_compiles(self, base_path):
        _write_ensemble(base_path, dict.fromkeys(SCENARIO_WEIGHTS, 1))
        resolved = resolve_analysis(
            "land-type-confidence",
            dataset=RASTER_DATASET,
            filters=ENSEMBLE_FILTERS,
            params={},
        )
        # Its own `output.format` is `tif`, unoverridden here -- which is exactly the form
        # `pipeline.json` publishes, so this also pins that a caller resubmitting that
        # pipeline verbatim gets a valid one rather than a tabular-sink rejection.
        assert resolved.pipeline.output.format == "tif"
        validate_shapes(resolved.pipeline)
        assert compile_plan(resolved.pipeline, base_path=base_path) is not None

    def test_lnd_type_plus_compiles(self, morph_pixel_v002_shrunk, base_path, crosswalk_path):
        write_lnd_type_plus_datasets(
            base_path, crosswalk_path, depths=[0.2] * 8, land_types=[1] * 8
        )
        resolved = resolve_analysis(
            "lnd-type-plus", dataset=None, filters={}, params=LND_TYPE_PLUS_PARAMS
        )
        # Band-wide end to end (the one `project` lands ON the raster grid), so `tif` is
        # the only sink `validate_shapes` accepts -- and it is what `pipeline.json`
        # publishes.
        assert resolved.pipeline.output.format == "tif"
        validate_shapes(resolved.pipeline)
        assert compile_plan(resolved.pipeline, base_path=base_path) is not None

    def test_salinity_veg_investigation_compiles(self, salinity_veg_data, base_path):
        resolved = resolve_analysis(
            "salinity-veg-investigation",
            dataset=None,
            filters={},
            params={"model_group_id": 500, "scenario_id": 7},
        )
        validate_shapes(resolved.pipeline)
        assert compile_plan(resolved.pipeline, base_path=base_path) is not None


# ------------------------------------------------------------------------------------------
# Successful compilation + real execution
# ------------------------------------------------------------------------------------------


class TestDifferenceExecution:
    def test_asc_order_produces_a_negative_later_value(self, base_path):
        write_annual_dataset(
            base_path, years=(2030, 2040), value_fn=lambda year, geo_id: 100.0 + geo_id
        )
        resolved = resolve_analysis(
            "difference",
            dataset=SSF_DATASET,
            filters={**COMMON_ANNUAL_FILTERS, "calendar_year": [2030, 2040]},
            params={"dimension": "calendar_year", "order": "asc"},
        )
        df = _run(resolved, base_path)
        # order="asc" negates 2040 (the later value) before summing across
        # calendar_year, so the grouped total is (2030's value) - (2040's value); both
        # years share the same value_fn, so the result is 0 for every geography.
        assert set(df["value"].to_list()) == {0.0}

    def test_desc_order_negates_the_earlier_value_instead(self, base_path):
        write_annual_dataset(
            base_path, years=(2030, 2040), value_fn=lambda year, geo_id: float(year)
        )
        resolved = resolve_analysis(
            "difference",
            dataset=SSF_DATASET,
            filters={**COMMON_ANNUAL_FILTERS, "calendar_year": [2030, 2040]},
            params={"dimension": "calendar_year", "order": "desc"},
        )
        df = _run(resolved, base_path)
        # order="desc" negates 2030 (the earlier value): -2030 + 2040 == 10.
        assert set(df["value"].to_list()) == {10.0}

    def test_an_unselected_year_is_excluded_by_the_leading_filter_step(self, base_path):
        """The dimension filters now reach the pipeline as a leading `filter` STEP, not as
        part of the input selection -- so a third year present on disk but absent from
        `filters` must not contribute to the sum.
        """
        write_annual_dataset(
            base_path, years=(2030, 2040, 2050), value_fn=lambda year, geo_id: float(year)
        )
        resolved = resolve_analysis(
            "difference",
            dataset=SSF_DATASET,
            filters={**COMMON_ANNUAL_FILTERS, "calendar_year": [2030, 2040]},
            params={"dimension": "calendar_year", "order": "asc"},
        )
        df = _run(resolved, base_path)
        # 2050 excluded; 2030 - 2040 == -10. Had it leaked in, the sum would be 2040.
        assert set(df["value"].to_list()) == {-10.0}

    def test_a_grid_cell_filter_narrows_the_result(self, base_path):
        write_annual_dataset(
            base_path, years=(2030, 2040), value_fn=lambda year, geo_id: float(year)
        )
        resolved = resolve_analysis(
            "difference",
            dataset=SSF_DATASET,
            filters={**COMMON_ANNUAL_FILTERS, "calendar_year": [2030, 2040], "grid_cell": "3,5"},
            params={"dimension": "calendar_year", "order": "asc"},
        )
        df = _run(resolved, base_path)
        assert sorted(df["grid_cell"].to_list()) == [3, 5]


class TestConcatenateExecution:
    def test_two_image_mode_multiplies_t0_by_ten(self, base_path):
        write_annual_dataset(base_path, years=(2030, 2040), value_fn=lambda year, geo_id: 1.0)
        resolved = resolve_analysis(
            "concatenate",
            dataset=SSF_DATASET,
            filters={**COMMON_ANNUAL_FILTERS, "calendar_year": [2030, 2040]},
            params={"dimension": "calendar_year"},
        )
        df = _run(resolved, base_path)
        # t0 (2030) is multiplied by 10 (-> 10), t1 (2040) is untouched (-> 1); summed
        # per geography across the two years -> 11.
        assert set(df["value"].to_list()) == {11.0}

    def test_three_image_mode_encodes_all_three_corners(self, base_path):
        write_annual_dataset(
            base_path, years=(2030, 2040), model_group_id=500, value_fn=lambda year, geo_id: 1.0
        )
        write_annual_dataset(
            base_path, years=(2030, 2040), model_group_id=520, value_fn=lambda year, geo_id: 1.0
        )
        resolved = resolve_analysis(
            "concatenate",
            dataset=SSF_DATASET,
            filters={
                "scenario_id": ANNUAL_KWARGS["scenario_id"],
                "percentile": ANNUAL_KWARGS["percentile"],
                "calendar_year": [2030, 2040],
                "model_group_id": [500, 520],
            },
            params={"dimension": "calendar_year"},
        )
        df = _run(resolved, base_path)
        # Surviving corners after the `NOT (mg=520 AND year=2030)` conjunct:
        # (500, 2030) -> *100 -> 100; (500, 2040) -> *10 -> 10; (520, 2040) -> untouched -> 1.
        # Grouped (summed) over model_group_id/calendar_year per geography: 100 + 10 + 1 = 111.
        assert set(df["value"].to_list()) == {111.0}

    def test_the_three_image_corner_conjunct_really_drops_that_corner(self, base_path):
        """The redundant corner is dropped by a `NOT (... AND ...)` conjunct on the
        leading `filter` step. Give (520, 2030) a value no other corner has, so its
        survival would be visible in the sum rather than hidden by uniform values.
        """
        write_annual_dataset(
            base_path, years=(2030, 2040), model_group_id=500, value_fn=lambda year, geo_id: 1.0
        )
        write_annual_dataset(
            base_path,
            years=(2030, 2040),
            model_group_id=520,
            value_fn=lambda year, geo_id: 1000.0 if year == 2030 else 1.0,
        )
        resolved = resolve_analysis(
            "concatenate",
            dataset=SSF_DATASET,
            filters={
                "scenario_id": ANNUAL_KWARGS["scenario_id"],
                "percentile": ANNUAL_KWARGS["percentile"],
                "calendar_year": [2030, 2040],
                "model_group_id": [500, 520],
            },
            params={"dimension": "calendar_year"},
        )
        df = _run(resolved, base_path)
        # Still 111: the 1000.0 corner is (520, 2030), which the NOT conjunct drops. Had
        # it survived it would have been multiplied by 100 and added 100000 to the sum.
        assert set(df["value"].to_list()) == {111.0}


class TestLandTypeConfidenceExecution:
    """The weighted sum, over an ensemble whose members are written one uniform `lnd_type`
    code each -- so every pixel's result is the total weight of the members this test made
    land, and a member leaking in (or dropping out) moves that total by its own weight
    rather than being averaged away.

    Unlike the other `type="dataset"` entries above, this one runs on a RASTER grid and
    ends band-wide, so it goes through `run_to_path` and rasterio rather than `_run`.
    """

    def _run_confidence(self, base_path, tmp_path, dataset=RASTER_DATASET):
        """Resolve the analysis over what's already on disk and run it to a GeoTIFF."""
        resolved = resolve_analysis(
            "land-type-confidence", dataset=dataset, filters=ENSEMBLE_FILTERS, params={}
        )
        validate_shapes(resolved.pipeline)
        out = tmp_path / "confidence.tif"
        run_to_path(resolved.pipeline, out, base_path=base_path)
        return out

    def _assert_confidence(self, base_path, tmp_path, value_by_scenario, expected):
        """Run the analysis over an ensemble built from `value_by_scenario` and assert
        every pixel's confidence is `expected`. Compared with a tolerance rather than
        exactly: the members are accumulated in Float32, so the five weighted terms land a
        bit or two either side of the exact answer.
        """
        _write_ensemble(base_path, value_by_scenario)
        with rio.open(self._run_confidence(base_path, tmp_path)) as src:
            # One band: the group keeps `model_group_id` and `calendar_year`, and the
            # ensemble is written for one of each.
            assert src.count == 1
            band = src.read(1, masked=True)
            assert band.count(), "the analysis returned no valid pixels at all"
            assert band.compressed() == pytest.approx(expected)

    def test_every_member_land_sums_to_one(self, base_path, tmp_path):
        # 1 is a land code, so every member votes land and the weights total 1.0 --
        # the property that makes the result readable as a fraction at all.
        self._assert_confidence(base_path, tmp_path, dict.fromkeys(SCENARIO_WEIGHTS, 1), 1.0)

    def test_every_member_water_sums_to_zero(self, base_path, tmp_path):
        # 2 is a water code: mapped to 0.0, weighted, and summed -- still 0.0.
        self._assert_confidence(base_path, tmp_path, dict.fromkeys(SCENARIO_WEIGHTS, 2), 0.0)

    def test_a_split_ensemble_sums_to_the_land_members_weight(self, base_path, tmp_path):
        # 34 and 36 say land (codes 1 and 3), the rest water (codes 2 and 5).
        self._assert_confidence(
            base_path,
            tmp_path,
            {34: 1, 35: 2, 36: 3, 37: 5, 38: 2},
            SCENARIO_WEIGHTS[34] + SCENARIO_WEIGHTS[36],
        )

    def test_every_land_code_maps_the_same_way(self, base_path, tmp_path):
        # Codes 1, 3 and 4 are all land, so an ensemble split across them is no different
        # from an all-1 ensemble -- the mapping is by class, not by code.
        self._assert_confidence(base_path, tmp_path, {34: 1, 35: 3, 36: 4, 37: 1, 38: 3}, 1.0)

    def test_an_unmapped_code_contributes_nothing(self, base_path, tmp_path):
        """A code outside 1-5 is dropped by the value-domain filter rather than reaching
        the sum as itself. Member 36 carries the largest weight, so its absence is the
        most visible: had the 9 survived unmapped it would have added 9 * 0.33.
        """
        self._assert_confidence(
            base_path,
            tmp_path,
            {34: 1, 35: 1, 36: 9, 37: 1, 38: 1},
            1.0 - SCENARIO_WEIGHTS[36],
        )

    def test_a_scenario_outside_the_ensemble_is_excluded(self, base_path, tmp_path):
        """`scenario_id` is collapsed by the trailing group, so a sixth scenario on disk
        would be summed straight into the result if the leading filter let it through --
        and it carries no weight `calculate`, so it would arrive as its raw code.
        """
        _write_ensemble(base_path, {7: 1})
        self._assert_confidence(base_path, tmp_path, dict.fromkeys(SCENARIO_WEIGHTS, 1), 1.0)

    def test_an_unselected_year_is_excluded(self, base_path, tmp_path):
        """The group keeps `calendar_year`, so a second year would surface as its own band
        rather than corrupting 2050's -- but only the filtered year should be read at all.
        """
        _write_ensemble(base_path, dict.fromkeys(SCENARIO_WEIGHTS, 2), years=(2040,))
        _write_ensemble(base_path, dict.fromkeys(SCENARIO_WEIGHTS, 1), years=(2050,))
        with rio.open(self._run_confidence(base_path, tmp_path)) as src:
            assert src.count == 1
            assert "calendar_year=2050" in src.descriptions[0]
            assert src.read(1, masked=True).compressed() == pytest.approx(1.0)


class TestSalinityVegInvestigationExecution:
    """`salinity-veg-investigation` is the only registered analysis whose pipeline is
    band-wide end to end, and the only one that exercises the two pieces of machinery a
    paired raster shape needs: a vector-to-raster `project` feeding a band-wide `join`
    (both operands read off ONE `scan_raster`, so their `batch_layout`s agree), and a
    `filter` whose `where` names the join's OPERAND value columns rather than `value`.
    Each is pinned here against the smallest pipeline that isolates it.
    """

    def test_the_analysis_itself_runs_to_a_uniform_2040_raster(self, salinity_veg_data, tmp_path):
        """The whole analysis, end to end, over `salinity_veg_data` -- marsh crosses 0.5
        only in 2040, salinity crosses 5.5 only in 2040, so the `min of=calendar_year`
        answer is 2040 at every pixel. The vector operand reaches the join through a
        vector-to-raster `project`, which is the half of the shape the sibling test below
        deliberately leaves out.
        """
        resolved = resolve_analysis(
            "salinity-veg-investigation",
            dataset=None,
            filters={},
            params={"model_group_id": 500, "scenario_id": 7},
        )
        out = tmp_path / "out.tif"
        run_to_path(resolved.pipeline, out, base_path=salinity_veg_data)

        with rio.open(out) as src:
            assert (src.read(1) == 2040).all()

    def test_a_filter_on_a_band_wide_joins_operand_names_pairs_the_operands(self, base_path, monkeypatch):
        """The other half, isolated: two RASTER operands (no `project` at all), so the
        only thing under test is the trailing `filter where="marsh > ... AND sal > ..."`.
        Both conjuncts name operand value columns, so the filter has to pair each
        operand's band with its counterpart on the dimensions they share and mask the
        pair, rather than reading `marsh`/`sal` as dimension names on the band metadata.
        2030 is (0.3, 4.0) and fails both conjuncts; 2040 is (0.7, 6.0) and passes both,
        so the surviving `min of=calendar_year` is 2040.
        """
        from cpra.mp.data.analysis.request.pipeline import parse_pipeline
        from tests.factories import write_raster_dataset

        write_ffibs_coverage_dataset(base_path, {2030: 0.3, 2040: 0.7})
        write_raster_dataset(
            base_path, years=(2030, 2040), fill_fn=lambda year: 4.0 if year == 2030 else 6.0
        )
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = parse_pipeline(
            {
                "inputs": {
                    "coverage": {
                        "variable": "ffibs_coverage_ratio",
                        "time_unit": "annual",
                        "grid": "veg_grid_cell_v001",
                    },
                    "veg_type": {
                        "variable": "vct",
                        "time_unit": "annual",
                        "grid": "veg_grid_cell_v001",
                    },
                },
                "steps": [
                    {"op": "filter", "input": "coverage", "where": "ffibs_type = 'pl_fm'", "as": "marsh"},
                    {"op": "filter", "input": "veg_type", "where": "model_group_id = 500", "as": "sal"},
                    {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
                    {"op": "filter", "where": "marsh > 0.5 AND sal > 5.5"},
                    {"op": "group", "by": [], "aggregate": "min", "of": "calendar_year"},
                ],
                "output": {"format": "tif"},
            }
        )
        validate_shapes(pipeline)
        plan = compile_plan(pipeline, base_path=base_path)
        df = execute(plan, base_path=base_path)[0].collect()

        assert df.columns == ["b0"]
        assert df["b0"].unique().to_list() == [2040]


class TestLndTypePlusExecution:
    """`lnd-type-plus` is the only registered analysis that collapses a band-wide join with
    a `group` rather than a `filter`, and it does so twice: once with `sum` over a negated
    operand, which is how `stage - elevation/1000` is expressed at all, and once with `max`,
    which is the "max of the two values" step. It also collapses two joins the other way,
    with a paired `filter` -- the elevation presence guard and the water restriction, each
    masking one operand by a predicate over the pair. All four are what these tests pin.
    Every case is a single 2x4 raster whose eight pixels are independent, so one run covers
    the whole reclassification table plus its edges.

    Depths are never stored -- `write_lnd_type_plus_datasets` writes the elevation that
    produces each one against a uniform projected stage, so a test that says `0.46` is
    asserting on a computed value, not a read one.
    """

    def _run_plus(self, base_path, tmp_path, params=None):
        resolved = resolve_analysis(
            "lnd-type-plus", dataset=None, filters={}, params=params or LND_TYPE_PLUS_PARAMS
        )
        validate_shapes(resolved.pipeline)
        out = tmp_path / "lnd_type_plus.tif"
        run_to_path(resolved.pipeline, out, base_path=base_path)
        return out

    def test_every_reclassification_case_at_once(
        self, morph_pixel_v002_shrunk, base_path, crosswalk_path, tmp_path
    ):
        """One pixel per rule. Every pixel is open water (`2`), which is the only land type
        a depth class can reach at all -- `test_only_open_water_gets_a_shallow_class` is
        where that restriction is pinned, and here it is held constant so the table is the
        only variable:

        | depth   | expected | why                                                    |
        |---------|----------|---------------------------------------------------------|
        | 0.20    | 6        | shallow class outranks the water it sits on            |
        | 0.70    | 7        | mid class likewise                                     |
        | 1.50    | 2        | deep class is `0`, so the water shows through          |
        | 0.46    | 6        | the first bound belongs to the class BELOW it          |
        | 0.47    | 7        | just past it, so the mid class                         |
        | 1.00    | 7        | the second bound likewise belongs below                |
        | 1.01    | 2        | just past it -- deep, so the water shows through       |
        | 0.00    | 2        | zero is NOT inundation: the leading `value > 0` filter |
        |         |          | masks it and the land type stands alone                |

        Every interval is open at its lower bound and closed at its upper one, so each
        bound is pinned from both sides. The bound rows are only meaningful at the
        factory's default stage of zero, which is what makes `stage - elevation/1000`
        reproduce a decimal literal exactly -- see `LND_TYPE_PLUS_STAGE`. A test that moves
        the stage has to stay off the bounds.
        """
        write_lnd_type_plus_datasets(
            base_path,
            crosswalk_path,
            depths=[0.20, 0.70, 1.50, 0.46, 0.47, 1.00, 1.01, 0.00],
            land_types=[2] * 8,
        )
        with rio.open(self._run_plus(base_path, tmp_path)) as src:
            assert src.count == 1
            assert src.read(1).ravel().tolist() == [6, 7, 2, 6, 7, 7, 2, 2]

    def test_only_open_water_gets_a_shallow_class(
        self, morph_pixel_v002_shrunk, base_path, crosswalk_path, tmp_path
    ):
        """The water restriction, across every land type at ONE shallow depth. `max` alone
        would put a `6` over all eight pixels -- `6` outranks every `lnd_type` code -- so
        anything but `[…, 6, …]` at the two water pixels and the land type everywhere else
        means the mask ahead of the final join stopped doing its job.

        Pixel 5's land type is nodata, which is not water either: there is nothing to call
        water, so the depth is masked too and the pixel has no contributor left at all.
        """
        write_lnd_type_plus_datasets(
            base_path,
            crosswalk_path,
            depths=[0.20] * 8,
            land_types=[1, 2, 3, 4, 5, None, 2, 1],
        )
        with rio.open(self._run_plus(base_path, tmp_path)) as src:
            band = src.read(1, masked=True)
            assert band.mask.ravel().tolist() == [False] * 5 + [True] + [False] * 2
            assert band.ravel().tolist()[:5] == [1, 6, 3, 4, 5]
            assert band.ravel().tolist()[6:] == [6, 1]

    def test_the_projected_stage_is_what_makes_a_pixel_wet(
        self, morph_pixel_v002_shrunk, base_path, crosswalk_path, tmp_path
    ):
        """Depth is COMPUTED, so the `project` that carries stage from the compartment grid
        onto the pixel grid has to actually contribute. Here every elevation is ABOVE the
        datum (+0.10 m, written as 100 mm), so the only way any pixel reads as inundated is
        if the 0.30 m stage was added to it: dropped, each depth would be -0.10 m, masked by
        the `value > 0` filter, and every pixel would read as its land type -- `2` here, not
        `6`.
        """
        write_lnd_type_plus_datasets(
            base_path,
            crosswalk_path,
            depths=[0.20] * 8,
            land_types=[2] * 8,
            stage=0.30,
        )
        with rio.open(self._run_plus(base_path, tmp_path)) as src:
            assert set(src.read(1).ravel().tolist()) == {6}

    def test_a_pixel_with_stage_but_no_elevation_is_masked_rather_than_read_as_bare_stage(
        self, morph_pixel_v002_shrunk, base_path, crosswalk_path, tmp_path
    ):
        """The presence guard. The band-wide `sum` SKIPS nulls rather than propagating them,
        so without the paired `IS NOT NULL` filter ahead of it a pixel with stage and no
        elevation would sum to the BARE STAGE -- 0.30 m here, a perfectly plausible shallow
        depth -- and paint class 6 across whatever the elevation surface does not cover.
        Guarded, pixel 0 contributes nothing and its land type stands alone.
        """
        write_lnd_type_plus_datasets(
            base_path,
            crosswalk_path,
            depths=[None, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20],
            land_types=[2] * 8,
            stage=0.30,
        )
        with rio.open(self._run_plus(base_path, tmp_path)) as src:
            assert src.read(1).ravel().tolist() == [2] + [6] * 7

    def test_a_pixel_that_is_not_inundated_falls_back_to_its_land_type(
        self, morph_pixel_v002_shrunk, base_path, tmp_path, crosswalk_path
    ):
        """Where the depth half has nothing to say it is MASKED rather than defaulted -- a
        masked operand contributes nothing to the max, so the land type stands alone.

        Pixels 0-3 are DRY: the elevation stands above the water level (a negative depth) or
        exactly at it (zero), which the `value > 0` filter masks, and each pixel's own land
        type comes through -- including pixel 2, which is open water that simply is not
        inundated this year and must read as `2` rather than as any depth class. Pixels 4-7
        are real depths over water, so the depth class stands.
        """
        write_lnd_type_plus_datasets(
            base_path,
            crosswalk_path,
            depths=[-0.50, -2.00, -0.01, 0.00, 0.20, 0.70, 1.50, 0.47],
            land_types=[4, 1, 2, 3, 2, 2, 2, 2],
        )
        with rio.open(self._run_plus(base_path, tmp_path)) as src:
            assert src.read(1).ravel().tolist() == [4, 1, 2, 3, 6, 7, 2, 7]

    def test_a_pixel_outside_both_domains_stays_nodata(
        self, morph_pixel_v002_shrunk, base_path, crosswalk_path, tmp_path
    ):
        """The cases with no contributor at all: masking rather than defaulting is what keeps
        them distinguishable from a real class. Pixel 0 has neither an elevation nor a land
        type. Pixel 7 has a real depth but no land type -- which is not water, so the water
        restriction masks the depth too and nothing is left; it is the one place the
        restriction turns a pixel that used to carry a class into nodata.
        """
        write_lnd_type_plus_datasets(
            base_path,
            crosswalk_path,
            depths=[None, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 1.50],
            land_types=[None, 2, 2, 2, 2, 2, 2, None],
        )
        with rio.open(self._run_plus(base_path, tmp_path)) as src:
            band = src.read(1, masked=True)
            assert band.mask.ravel().tolist() == [True] + [False] * 6 + [True]
            assert band.ravel().tolist()[1:7] == [6] * 6

    def test_an_unselected_year_is_excluded(
        self, morph_pixel_v002_shrunk, base_path, crosswalk_path, tmp_path
    ):
        """`calendar_year` is a parameter rather than a filter here, but it still lands on
        the same leading `filter` steps -- and the trailing `group` KEEPS the dimension, so a
        leaked year would surface as a second band rather than corrupting 2050's.
        """
        write_lnd_type_plus_datasets(
            base_path, crosswalk_path, depths=[0.20] * 8, land_types=[2] * 8, year=2040
        )
        write_lnd_type_plus_datasets(
            base_path, crosswalk_path, depths=[0.70] * 8, land_types=[2] * 8, year=2050
        )
        with rio.open(self._run_plus(base_path, tmp_path)) as src:
            assert src.count == 1
            assert "calendar_year=2050" in src.descriptions[0]
            assert set(src.read(1).ravel().tolist()) == {7}

    def test_omitting_calendar_year_keeps_every_year_as_its_own_band(
        self, morph_pixel_v002_shrunk, base_path, crosswalk_path, tmp_path
    ):
        """`calendar_year` is the one optional parameter. Omitted, the leading filters pin
        only the model group and scenario, and the `group.by` still carries the year -- so
        each year is reduced against its OWN counterpart band rather than across years.
        """
        write_lnd_type_plus_datasets(
            base_path, crosswalk_path, depths=[0.20] * 8, land_types=[2] * 8, year=2040
        )
        write_lnd_type_plus_datasets(
            base_path, crosswalk_path, depths=[1.50] * 8, land_types=[4] * 8, year=2050
        )
        params = {k: v for k, v in LND_TYPE_PLUS_PARAMS.items() if k != "calendar_year"}
        with rio.open(self._run_plus(base_path, tmp_path, params=params)) as src:
            assert src.count == 2
            by_year = {
                description: src.read(index).ravel().tolist()
                for index, description in enumerate(src.descriptions, start=1)
            }
            # A grouped band is described by its whole `group.by` tuple, not the calendar
            # dimension alone -- the year is the last conjunct of the three.
            assert {d: set(v) for d, v in by_year.items()} == {
                "model_group_id=705, scenario_id=7, calendar_year=2040": {6},
                "model_group_id=705, scenario_id=7, calendar_year=2050": {4},
            }

    def test_another_scenario_on_disk_does_not_reach_the_join(
        self, morph_pixel_v002_shrunk, base_path, crosswalk_path, tmp_path
    ):
        """`scenario_id` is pinned by the leading filters and kept by the `group`, so a
        second scenario is neither reduced into 7's band nor added as a band of its own. It
        is written with values that would be unmistakable either way.
        """
        write_lnd_type_plus_datasets(
            base_path, crosswalk_path, depths=[0.20] * 8, land_types=[4] * 8, scenario_id=8
        )
        write_lnd_type_plus_datasets(
            base_path, crosswalk_path, depths=[1.50] * 8, land_types=[2] * 8
        )
        with rio.open(self._run_plus(base_path, tmp_path)) as src:
            assert src.count == 1
            assert set(src.read(1).ravel().tolist()) == {2}
