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

`exit-criterion` is `type="custom"` with hardcoded inputs (`ffibs_coverage_ratio`/
`veg_grid_cell_v001`, `sal`/`hydro_compartment_v001`), written by
`tests/factories.py`'s `write_ffibs_coverage_dataset`/`write_sal_annual_dataset` and
assembled by the `exit_criterion_data` fixture in `tests/api/conftest.py` -- shared with
`tests/api/test_analyses_endpoints.py`, which needs the same data to drive the same
analysis through its HTTP routes.

**`exit-criterion` has no execution test here, on purpose.** Its pipeline resolves,
type-checks and compiles -- all three are covered below -- but running it needs two
lowerings the execution engine does not have yet: a vector-to-raster `project` whose
output feeds a band-wide `join` (the two operands come from separate `scan_raster` calls
with different band counts, so their read windows disagree on `batch_layout` and
`join_band_wide`'s pixel-grid precondition rejects them), and a `filter` whose `where`
names a band-wide join's OPERAND value columns (`marsh`/`sal`), which
`expressions.classify_predicate_kind` reads as dimension names and looks for on the
band metadata table. `TestExitCriterionUnsupportedExecution` pins both failures so the
gap is recorded as a known, reproducible shortfall rather than an untested claim -- and
so whichever lowering lands first has a test that changes when it does.
"""

import polars as pl
import pytest

from cpra.mp.data import config
from cpra.mp.data.analyses import ANALYSES, DatasetRef, resolve_analysis
from cpra.mp.data.analysis.execution.execute import execute, run_to_path
from cpra.mp.data.analysis.execution.plan import compile_plan
from cpra.mp.data.analysis.request.ops import validate_shapes
from tests.factories import ANNUAL_KWARGS, write_annual_dataset, write_ffibs_coverage_dataset

SSF_DATASET = DatasetRef(
    variable=ANNUAL_KWARGS["variable"], time_unit=ANNUAL_KWARGS["time_unit"], grid=ANNUAL_KWARGS["grid"]
)
COMMON_ANNUAL_FILTERS = {
    "model_group_id": ANNUAL_KWARGS["model_group_id"],
    "scenario_id": ANNUAL_KWARGS["scenario_id"],
    "percentile": ANNUAL_KWARGS["percentile"],
}


def _run(resolved, base_path):
    """Take a `ResolvedAnalysis` all the way through the real request/execution path --
    `validate_shapes` -> `compile_plan` -> `execute` -> `.collect()` -- and return the
    resulting `DataFrame`. Every `type="dataset"` entry here is long-shaped (a vector
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
        # If a fourth analysis lands and nobody updates the tests below, this is the
        # tripwire: it fails loudly instead of the drift test silently covering only 3
        # of 4 entries.
        assert set(ANALYSES) == {"exit-criterion", "difference", "concatenate"}

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

    def test_exit_criterion_compiles(self, exit_criterion_data, base_path):
        resolved = resolve_analysis(
            "exit-criterion",
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


class TestExitCriterionUnsupportedExecution:
    """`exit-criterion` resolves, type-checks and compiles (see `TestRegistryDrift`), but
    cannot yet RUN -- see this module's docstring. Both missing lowerings are pinned here,
    each against the smallest pipeline that isolates it, so the gap is a recorded,
    reproducible fact and so the test changes when either lowering lands.
    """

    def test_the_analysis_itself_fails_at_the_band_wide_join(self, exit_criterion_data, tmp_path):
        """A vector-to-raster `project` reads only the crosswalk (one band) in its own
        `scan_raster` call, while the sibling raster operand is read whole (two bands
        here). A read's `batch_layout` depends on its band count, so the two frames
        disagree about how pixels are chunked and cannot be aligned row-for-row.
        """
        resolved = resolve_analysis(
            "exit-criterion",
            dataset=None,
            filters={},
            params={"model_group_id": 500, "scenario_id": 7},
        )
        with pytest.raises(AssertionError, match="do not share a pixel grid"):
            run_to_path(resolved.pipeline, tmp_path / "out.tif", base_path=exit_criterion_data)

    def test_a_filter_on_a_band_wide_joins_operand_names_is_read_as_a_dimension(self, base_path, monkeypatch):
        """The second gap, isolated from the first: two RASTER operands DO share one scan
        (so `batch_layout` matches), but the trailing `filter where="marsh > ... AND
        sal > ..."` still fails -- neither conjunct names `value` or `grid_cell`, so
        `classify_predicate_kind` sorts both as dimension predicates and they are
        evaluated against the band metadata table, which has no such column.
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
        with pytest.raises(pl.exceptions.ColumnNotFoundError, match='unable to find column "marsh"'):
            execute(plan, base_path=base_path)[0].collect()
