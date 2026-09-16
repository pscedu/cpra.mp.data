"""The registry-drift test, plus an exit-criterion end-to-end functional check.

Every entry in `cpra.mp.data.analyses.ANALYSES` must resolve (via `resolve_analysis`) to a
real `Pipeline` that `compile_pipeline` accepts against a representative, on-disk dataset --
enforcing the constraint that nothing is reachable only through a named analysis, only through
the exact same compiler any hand-authored pipeline goes through. This module is deliberately
independent of the concurrently-developed `cpra/mp/data/api/` route layer (a different
in-flight package this task does not own and should not couple to): everything here drives
`compile_pipeline`/`BaseAnalysis.apply_params`/`materialize`/`PairedAnalysis.apply_params`/
`write_raster` directly, exactly as `tests/api/test_pipeline_exec.py`'s own
`TestCompileChecks`/`TestCompileSingleInputChain`/`TestCompilePairedNotebookShape` sections
do for hand-authored pipelines.

**Representative datasets.** `difference`/`concatenate` are `type="dataset"` and take
whatever `DatasetRef` a route resolves for them; `tests/factories.py`'s `ssf`/
`ecoregion_v001` (`ANNUAL_KWARGS`) is used here as that representative dataset, per that
module's own documented convention ("the compiler cares about grid TYPES and SHAPES, not
real-world variable semantics" -- see `test_pipeline_exec.py`'s module docstring) --
`concatenate`'s real `Meta.variables = ["lnd_type"]` lives on `morph_pixel_v001`
(8265x20674 pixels), far too large to write as a test fixture; the `Meta.variables`
declaration itself is unaffected by this substitution, only which dataset proves the
pipeline-construction MACHINERY compiles.

`exit-criterion` is `type="custom"` with hardcoded inputs (`ffibs_coverage_ratio`/
`veg_grid_cell_v001`, `sal`/`hydro_compartment_v001`) -- `_write_ffibs_raster`/
`_write_sal_annual` below are local, reproduced copies of `test_pipeline_exec.py`'s
identically-named fixture writers (that module's own docstring already establishes this
"reproduced here rather than imported" convention for the same reason: this module must not
import from or modify another test file it doesn't own).
"""

import numpy as np
import polars as pl
import pytest

from cpra.mp.data import config
from cpra.mp.data.analyses import ANALYSES, DatasetRef, resolve_analysis
from cpra.mp.data.analysis.paired import PairedAnalysis
from cpra.mp.data.analysis.planner import compile_pipeline
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.raster import write_cog
from cpra.mp.data.tabular import write_parquet
from tests.factories import ANNUAL_KWARGS, write_annual_dataset, write_raster_crosswalk

SSF_DATASET = DatasetRef(
    variable=ANNUAL_KWARGS["variable"], time_unit=ANNUAL_KWARGS["time_unit"], grid=ANNUAL_KWARGS["grid"]
)
COMMON_ANNUAL_FILTERS = {
    "model_group_id": ANNUAL_KWARGS["model_group_id"],
    "scenario_id": ANNUAL_KWARGS["scenario_id"],
    "percentile": ANNUAL_KWARGS["percentile"],
}


def _write_ffibs_raster(base_path, values_by_year, model_group_id=500, scenario_id=7):
    """`ffibs_coverage_ratio`/`veg_grid_cell_v001`, uniformly filled per year -- a local
    copy of `tests/api/test_pipeline_exec.py`'s identically-named fixture writer (see this
    module's own docstring for why it is reproduced, not imported).
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
            model_group_id=model_group_id,
            scenario_id=scenario_id,
            model="icm_v2023.0.0",
            ffibs_type="pl_fm",
        )


def _write_sal_annual(base_path, values_by_year, geo_id=1, model_group_id=500, scenario_id=7):
    """`sal`/`hydro_compartment_v001`, annual, one geo column -- a local copy of
    `tests/api/test_pipeline_exec.py`'s identically-named fixture writer.
    """
    years = sorted(values_by_year)
    df = pl.DataFrame(
        {
            "variable": pl.Series(["sal"] * len(years), dtype=pl.String),
            "grid": pl.Series(["hydro_compartment_v001"] * len(years), dtype=pl.String),
            "time_unit": pl.Series(["annual"] * len(years), dtype=pl.String),
            "model_group_id": pl.Series([model_group_id] * len(years), dtype=pl.Int32),
            "scenario_id": pl.Series([scenario_id] * len(years), dtype=pl.Int32),
            "aggregate_type": pl.Series(["2_wk_max"] * len(years), dtype=pl.String),
            "calendar_year": pl.Series(years, dtype=pl.Int32),
            "model": pl.Series(["icm_v2023.0.0"] * len(years), dtype=pl.String),
            str(geo_id): pl.Series([values_by_year[y] for y in years], dtype=pl.Float32),
        }
    )
    write_parquet(df, base_path=base_path, validate=False)


@pytest.fixture
def exit_criterion_data(base_path, crosswalk_path, monkeypatch):
    """Marsh (raster) 0.3->2030, 0.7->2040 (crosses `marsh > 0.5` only in 2040);
    salinity (vector, one zone) 4.0->2030, 6.0->2040 (crosses `sal > 5.5` only in
    2040) -- every `veg_grid_cell_v001` pixel maps to the same single
    `hydro_compartment_v001` zone, so `{"aggregate": "min", "of": "calendar_year"}`'s
    expected output is a uniform 2040 raster. Identical to
    `test_pipeline_exec.py`'s `notebook_query_data`, reproduced here for the same "don't
    import from another test module" reason as the writer functions above.
    """
    _write_ffibs_raster(base_path, {2030: 0.3, 2040: 0.7})
    _write_sal_annual(base_path, {2030: 4.0, 2040: 6.0}, geo_id=1)
    monkeypatch.setattr(config, "BASE_PATH", base_path)

    grid = GRIDS["veg_grid_cell_v001"]
    zones = np.ones((grid.height, grid.width), dtype="int32")
    write_raster_crosswalk(crosswalk_path, "veg_grid_cell_v001", "hydro_compartment_v001", zones)
    return base_path


# ------------------------------------------------------------------------------------------
# Every registered analysis compiles
# ------------------------------------------------------------------------------------------


class TestRegistryDrift:
    """Every `ANALYSES` entry, expanded via `resolve_analysis` and run through
    `compile_pipeline` against a representative on-disk dataset -- catches an analysis
    added to the registry whose `pipeline()` builds something `compile_pipeline` doesn't
    actually accept (nothing may be reachable only through a named analysis).
    """

    def test_every_registered_analysis_name_is_covered_by_this_test(self):
        # If a fourth analysis lands and nobody updates the parametrization below, this
        # is the tripwire: it fails loudly instead of the drift test silently covering
        # only 3 of 4 entries.
        assert set(ANALYSES) == {"exit-criterion", "difference", "concatenate"}

    def test_difference_compiles(self, base_path):
        write_annual_dataset(base_path, years=(2030, 2040))
        resolved = resolve_analysis(
            "difference",
            dataset=SSF_DATASET,
            filters={**COMMON_ANNUAL_FILTERS, "calendar_year": [2030, 2040]},
            params={"dimension": "calendar_year", "order": "asc"},
        )
        compiled = compile_pipeline(resolved.pipeline, base_path=base_path)
        assert compiled is not None

    def test_concatenate_two_image_compiles(self, base_path):
        write_annual_dataset(base_path, years=(2030, 2040))
        resolved = resolve_analysis(
            "concatenate",
            dataset=SSF_DATASET,
            filters={**COMMON_ANNUAL_FILTERS, "calendar_year": [2030, 2040]},
            params={"dimension": "calendar_year"},
        )
        compiled = compile_pipeline(resolved.pipeline, base_path=base_path)
        assert compiled is not None

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
        compiled = compile_pipeline(resolved.pipeline, base_path=base_path)
        assert compiled is not None

    def test_exit_criterion_compiles(self, exit_criterion_data, base_path):
        resolved = resolve_analysis(
            "exit-criterion",
            dataset=None,
            filters={},
            params={"model_group_id": 500, "scenario_id": 7},
        )
        compiled = compile_pipeline(resolved.pipeline, base_path=base_path)
        assert isinstance(compiled, PairedAnalysis)


# ------------------------------------------------------------------------------------------
# Successful compilation + real execution, one per analysis
# ------------------------------------------------------------------------------------------


class TestDifferenceExecution:
    async def test_asc_order_produces_a_negative_later_value(self, base_path):
        write_annual_dataset(
            base_path, years=(2030, 2040), value_fn=lambda year, geo_id: 100.0 + geo_id
        )
        resolved = resolve_analysis(
            "difference",
            dataset=SSF_DATASET,
            filters={**COMMON_ANNUAL_FILTERS, "calendar_year": [2030, 2040]},
            params={"dimension": "calendar_year", "order": "asc"},
        )
        compiled = compile_pipeline(resolved.pipeline, base_path=base_path)
        await compiled.apply_params()
        df = compiled.materialize()
        # order="asc" negates 2040 (the later value) before summing across
        # calendar_year, so the grouped total is (2030's value) - (2040's value); both
        # years share the same value_fn, so the result is 0 for every geography.
        assert set(df["value"].to_list()) == {0.0}

    async def test_desc_order_negates_the_earlier_value_instead(self, base_path):
        write_annual_dataset(
            base_path, years=(2030, 2040), value_fn=lambda year, geo_id: float(year)
        )
        resolved = resolve_analysis(
            "difference",
            dataset=SSF_DATASET,
            filters={**COMMON_ANNUAL_FILTERS, "calendar_year": [2030, 2040]},
            params={"dimension": "calendar_year", "order": "desc"},
        )
        compiled = compile_pipeline(resolved.pipeline, base_path=base_path)
        await compiled.apply_params()
        df = compiled.materialize()
        # order="desc" negates 2030 (the earlier value): -2030 + 2040 == 10.
        assert set(df["value"].to_list()) == {10.0}


class TestConcatenateExecution:
    async def test_two_image_mode_multiplies_t0_by_ten(self, base_path):
        write_annual_dataset(base_path, years=(2030, 2040), value_fn=lambda year, geo_id: 1.0)
        resolved = resolve_analysis(
            "concatenate",
            dataset=SSF_DATASET,
            filters={**COMMON_ANNUAL_FILTERS, "calendar_year": [2030, 2040]},
            params={"dimension": "calendar_year"},
        )
        compiled = compile_pipeline(resolved.pipeline, base_path=base_path)
        await compiled.apply_params()
        df = compiled.materialize()
        # t0 (2030) is multiplied by 10 (-> 10), t1 (2040) is untouched (-> 1); summed
        # per geography across the two years -> 11.
        assert set(df["value"].to_list()) == {11.0}

    async def test_three_image_mode_encodes_all_three_corners(self, base_path):
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
        compiled = compile_pipeline(resolved.pipeline, base_path=base_path)
        await compiled.apply_params()
        df = compiled.materialize()
        # Surviving corners after the `not (mg=520 AND year=2030)` filter:
        # (500, 2030) -> *100 -> 100; (500, 2040) -> *10 -> 10; (520, 2040) -> untouched -> 1.
        # Grouped (summed) over model_group_id/calendar_year per geography: 100 + 10 + 1 = 111.
        assert set(df["value"].to_list()) == {111.0}


class TestExitCriterionExecution:
    """The exit-criterion notebook query, byte-exact in shape (see
    `exit_criterion.py`'s own docstring pointing at `docs/data-analysis-api.md`'s "The
    Exit-Criterion Query"), but reached here via `resolve_analysis("exit-criterion", ...)`
    instead of hand-authored JSON -- proves the named analysis is behaviorally identical
    to the documented example, not just structurally similar.
    """

    async def test_produces_a_uniform_2040_raster(self, exit_criterion_data, tmp_path):
        resolved = resolve_analysis(
            "exit-criterion",
            dataset=None,
            filters={},
            params={"model_group_id": 500, "scenario_id": 7},
        )
        compiled = compile_pipeline(resolved.pipeline, base_path=exit_criterion_data)
        assert isinstance(compiled, PairedAnalysis)
        await compiled.apply_params()

        out_path = tmp_path / "exit_criterion.tif"
        await compiled.write_raster(str(out_path))

        import rasterio as rio

        with rio.open(out_path) as src:
            data = src.read(1)
        assert (data == 2040).all()

    async def test_custom_thresholds_change_the_result(self, exit_criterion_data, tmp_path):
        # A salinity threshold above 6.0 is never crossed in either year -> no calendar
        # year ever satisfies the filter, so `min(calendar_year)` is nodata everywhere.
        resolved = resolve_analysis(
            "exit-criterion",
            dataset=None,
            filters={},
            params={"model_group_id": 500, "scenario_id": 7, "salinity_threshold": 100.0},
        )
        compiled = compile_pipeline(resolved.pipeline, base_path=exit_criterion_data)
        await compiled.apply_params()

        out_path = tmp_path / "exit_criterion_unreachable.tif"
        await compiled.write_raster(str(out_path))

        import rasterio as rio

        with rio.open(out_path) as src:
            data = src.read(1)
            nodata = src.nodata
        assert (data == nodata).all()
