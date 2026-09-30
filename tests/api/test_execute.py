"""Tests for `cpra.mp.data.analysis.execution.execute` -- the walk.

Five concerns, in five test classes:

- `TestLoweringTableAgreement`: the bidirectional coverage check the module docstring
  promises -- every `(op, shape)` combination `request.ops`'s lowering tables (or, for
  `calculate`/`group`, `Shape` itself) declare legal is covered by EXACTLY one of
  `execute.py`'s registered executors -- no more, no fewer.
- `TestLoweringNotImplemented`: the exception hierarchy `LoweringNotImplementedError`
  itself must hold, plus the one dispatch key that still raises it on purpose -- see that
  class's own docstring for why a raster-to-vector `project` reached through an earlier
  spatial filter is not the same combination `("raster", "vector")` names.
- `TestEndToEnd`: the resume contract itself -- a JSON pipeline over a real fixture
  dataset, taken through `parse_pipeline` -> `validate_shapes` -> `compile_plan` ->
  `execute` -> `.collect()`, including a declared-order test proving that reordering two
  steps changes the answer, and a `project` step carried all the way through a real
  crosswalk file.
- `TestBandWideEndToEnd`: the same resume contract for a raster input -- opening a
  band-wide frame, running `filter`/`calculate`/`group`/`join`/`project` against it, and
  the mid-walk transition to long shape a spatial filter causes.
- `TestTileScopedReads`: the same walk again, over a plan compiled for a map tile --
  every raster origin read at its own window instead of whole, including the shared
  window a band-wide join group's single scan uses and the widening an origin feeding a
  raster-to-vector `project` needs to see whole features across the tile's edge.
"""

import dataclasses

import mercantile
import numpy as np
import polars as pl
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from cpra.mp.data.analysis.execution import project
from cpra.mp.data.analysis.execution.execute import (
    LoweringNotImplementedError,
    _LOWERINGS,
    _parse_point_sample_ids,
    _point_sample_overrides,
    _point_sample_reads,
    _shared_point_sample_window,
    _shared_tile_window,
    execute,
)
from cpra.mp.data.analysis.execution.frame import (
    RESERVED_BAND_INDEX_COLUMN,
    band_column_names,
    has_no_bands,
)
from cpra.mp.data.analysis.execution.plan import TileRequest, compile_plan
from cpra.mp.data.analysis.io.raster_io import raster_grid_geometry, tile_source_window
from cpra.mp.data.analysis.request.ops import (
    FILTER_LOWERING_TABLE,
    FrameType,
    JOIN_LOWERING_TABLE,
    PROJECT_LOWERING_TABLE,
    Shape,
    UNPIVOT_LOWERING_TABLE,
    validate_shapes,
)
from cpra.mp.data.analysis.request.pipeline import parse_pipeline
from cpra.mp.data.grids import GRIDS
from tests.factories import (
    ANNUAL_KWARGS,
    RASTER_KWARGS,
    raster_array,
    write_annual_dataset,
    write_grid_geometry,
    write_omar_dataset,
    write_raster_crosswalk,
    write_raster_dataset,
    write_vector_crosswalk,
)

VECTOR_GRID = "hydro_compartment_v001"
RASTER_GRID = "veg_grid_cell_v001"
RASTER_GRID_2 = "morph_pixel_v001"

assert GRIDS[VECTOR_GRID].grid_type == "vector"
assert GRIDS[RASTER_GRID].grid_type == "raster"
assert GRIDS[RASTER_GRID_2].grid_type == "raster"


def vector_input(grid=VECTOR_GRID, variable="sal"):
    return {"variable": variable, "time_unit": "annual", "grid": grid}


def raster_input(grid=RASTER_GRID, variable="ffibs_coverage_ratio"):
    return {"variable": variable, "time_unit": "annual", "grid": grid}


def frame_types(grid, **shapes):
    """Hand-build the `plan.frame_types` map for a test that constructs an
    `ExecutionPlan` directly. `grid` is the grid every named frame is addressed against,
    which is all these tests need -- none of them dispatches a `project`, the only op whose
    lowering the grid participates in selecting.
    """
    return {
        name: FrameType(
            shape=shape, grid=grid, has_count_by_value=False, value_columns=frozenset({"value"})
        )
        for name, shape in shapes.items()
    }


def _pipeline(data):
    pipeline = parse_pipeline(data)
    validate_shapes(pipeline)
    return pipeline


class TestLoweringTableAgreement:
    """`request.ops.py` types shape purely as data (the three tables, plus `Shape` itself
    for the two ops with no table); `execute.py` must agree with that data, not drift from
    it by way of a hand-written case list. Each check below enumerates the SAME source of
    truth `request/ops.py` itself consults and asserts `execute.py`'s registries partition
    it exactly.
    """

    def _registered(self, kind):
        """The dispatch keys `execute._LOWERINGS` actually registers for `kind` -- read
        off the one registry the walk itself dispatches through, so a lowering that exists
        but was never wired into that registry counts as missing here exactly as it would
        at run time.
        """
        return {key for (registered_kind, key) in _LOWERINGS if registered_kind == kind}

    def test_filter_shapes_are_fully_covered(self):
        domain = {shape for (_kind, shape) in FILTER_LOWERING_TABLE}
        assert self._registered("filter") == domain

    def test_join_shapes_are_fully_covered(self):
        assert self._registered("join") == set(JOIN_LOWERING_TABLE)

    def test_calculate_has_no_lowering_of_its_own(self):
        # `compile_ops` lowers a `calculate` to an arm-less `CaseOp`, so nothing downstream
        # of the request layer ever dispatches on the kind -- an entry registered for it
        # would be dead code that looked like coverage.
        assert self._registered("calculate") == set()

    def test_case_shapes_are_fully_covered(self):
        # `case` has no lowering TABLE: `_type_case` types it unconditionally
        # shape-preserving (only a spatial arm could change shape, and that is rejected
        # outright), so there is nothing to consult and the declared domain is `Shape`
        # itself. This is the entry a `calculate` reaches too.
        assert self._registered("case") == set(Shape)

    def test_group_shapes_are_fully_covered(self):
        assert self._registered("group") == set(Shape)

    def test_project_directions_are_fully_covered(self):
        # `project` is keyed by direction, not shape -- it is the op the incoming shape
        # cannot select a lowering for, so its domain is `PROJECT_LOWERING_TABLE`'s pairs
        # rather than `Shape`'s members.
        assert self._registered("project") == set(PROJECT_LOWERING_TABLE)

    def test_unpivot_shapes_are_fully_covered(self):
        # The one table with a shape MISSING from it on purpose: an `unpivot` is long-only
        # (`_type_unpivot` rejects a band-wide input outright), and the absent `band_wide`
        # key is what this check reads that from. Written as the same set comparison as
        # every other op above rather than as an assertion about which shape is absent, so
        # that adding a band-wide lowering to one side without the other still fails.
        assert self._registered("unpivot") == set(UNPIVOT_LOWERING_TABLE)

    def test_every_registered_lowering_is_named_by_a_table(self):
        # The other direction, across all seven ops at once: no entry may sit in
        # `_LOWERINGS` under an op kind the request layer does not type, or under a key
        # that op's own declared domain does not contain.
        domains = {
            "filter": {shape for (_kind, shape) in FILTER_LOWERING_TABLE},
            "join": set(JOIN_LOWERING_TABLE),
            "calculate": set(Shape),
            "case": set(Shape),
            "group": set(Shape),
            "project": set(PROJECT_LOWERING_TABLE),
            "unpivot": set(UNPIVOT_LOWERING_TABLE),
        }
        for kind, key in _LOWERINGS:
            assert kind in domains, f"{kind!r} is not an op the request layer types"
            assert key in domains[kind], f"{kind!r} has no declared lowering for {key!r}"


class TestLoweringNotImplemented:
    """`LoweringNotImplementedError`'s own contract, plus the one case that raises it on
    purpose even though every `(op, key)` combination the lowering tables declare legal has
    a registered lowering.

    `project_direction` keys a `project` off the incoming frame's DECLARED grid
    (`frame_types[op.input].grid`), not its current actual shape -- so a raster-origin
    frame an earlier spatial filter already turned long still dispatches to the
    `("raster", "vector")` entry, which is real and expects a band-wide frame to zonal-
    aggregate. It cannot do that to a frame that is already one row per sampled pixel --
    that data has already been through a different reduction (point sampling, not zonal
    aggregation), and reusing the raster-to-vector lowering on it would silently answer a
    different question than the one it exists to answer -- so the registered lowering
    itself raises `LoweringNotImplementedError` for exactly that narrower case. The
    alternative -- refusing the whole `("raster", "vector")` direction, which is the finest
    distinction the dispatch key can draw -- would reject the ordinary case too, and that
    one works fine (see `TestBandWideEndToEnd`'s own raster-to-vector `project` test).
    """

    def test_raster_to_vector_project_after_a_spatial_filter_raises(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [
                    {"op": "filter", "input": "a", "where": "grid_cell == 0", "as": "sampled"},
                    {"op": "project", "input": "sampled", "to": VECTOR_GRID, "aggregate": "sum"},
                ],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline)
        assert plan.frame_types["sampled"].shape is Shape.long
        with pytest.raises(LoweringNotImplementedError):
            execute(plan)

    def test_not_yet_implemented_is_not_a_request_error(self):
        # Deliberately NOT a member of `request.errors.RequestError`'s family -- the
        # pipeline itself is well-formed (see `LoweringNotImplementedError`'s own
        # docstring); this pins that a caller catching `RequestError` does NOT also catch
        # this.
        from cpra.mp.data.analysis.request.errors import RequestError

        assert not issubclass(LoweringNotImplementedError, RequestError)
        assert issubclass(LoweringNotImplementedError, NotImplementedError)


class TestEndToEnd:
    """The resume contract: a JSON pipeline over a fixture Parquet dataset, through
    `parse_pipeline` -> `validate_shapes` -> `compile_plan` -> `execute` -> `.collect()`,
    ending in a real tabular result -- and declared-order semantics asserted concretely.

    Fixture: `write_annual_dataset` with its defaults -- `ssf` on `ecoregion_v001`,
    years (2030, 2040), `ECOREGION_IDS = range(1, 33)` (32 cells), `value_fn = year +
    geo_id / 100`. Hand-computed ranges used below:
    - year=2030: value in [2030.01, 2030.32]
    - year=2040: value in [2040.01, 2040.32]
    """

    @pytest.fixture
    def annual_dataset(self, base_path):
        write_annual_dataset(base_path, years=(2030, 2040))
        return base_path

    def _run(self, base_path, steps):
        pipeline = _pipeline(
            {
                "inputs": {"a": {"variable": ANNUAL_KWARGS["variable"], "time_unit": "annual", "grid": ANNUAL_KWARGS["grid"]}},
                "steps": steps,
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline)
        return execute(plan, base_path=base_path).collect()

    def test_filter_then_calculate_then_group_produces_a_tabular_result(self, annual_dataset):
        df = self._run(
            annual_dataset,
            [
                {"op": "filter", "input": "a", "where": "value > 2035", "as": "f"},
                {"op": "calculate", "expression": "value - 2000", "as": "c"},
            ],
        )
        # Only year=2040 rows exceed 2035 (year=2030's max is 2030.32) -- all 32 cells
        # survive the filter; `- 2000` then lands every value in [40.01, 40.32].
        assert df.height == 32
        assert df["value"].min() == pytest.approx(40.01, abs=1e-3)
        assert df["value"].max() == pytest.approx(40.32, abs=1e-3)

    def test_declared_order_changes_the_answer(self, annual_dataset):
        # Same two steps, same predicate and expression, reversed order.
        filter_then_calculate = self._run(
            annual_dataset,
            [
                {"op": "filter", "input": "a", "where": "value > 2035", "as": "f"},
                {"op": "calculate", "expression": "value - 2000", "as": "c"},
            ],
        )
        calculate_then_filter = self._run(
            annual_dataset,
            [
                {"op": "calculate", "input": "a", "expression": "value - 2000", "as": "c"},
                {"op": "filter", "where": "value > 2035", "as": "f"},
            ],
        )
        # filter-then-calculate: keeps year=2040 (all values > 2035 pre-subtraction), then
        # subtracts 2000 -> 32 rows, values ~40.xx.
        assert filter_then_calculate.height == 32
        # calculate-then-filter: subtracts 2000 FIRST, so every value is now in
        # [30.01, 40.32] -- none exceed 2035 -- so the filter step drops every row.
        assert calculate_then_filter.height == 0
        # Same declared ops, same input data, different order -> different answer: this is
        # exactly what "a pipeline's declared order IS its semantics" means concretely.
        assert filter_then_calculate.height != calculate_then_filter.height

    def test_group_collapses_to_one_row_per_grid_cell(self, annual_dataset):
        df = self._run(
            annual_dataset,
            # A `group` naming its input is a whole pipeline on its own -- read the
            # dataset, collapse it -- with no filter in front purely to chain from.
            [{"op": "group", "input": "a", "by": [], "aggregate": "sum"}],
        )
        # One row per grid_cell (32 cells), summed over the two years:
        # (2030 + g/100) + (2040 + g/100) = 4070 + 2*g/100, for g in 1..32.
        assert df.height == 32
        expected = sorted(4070 + 2 * g / 100 for g in range(1, 33))
        assert sorted(df["value"].to_list()) == pytest.approx(expected, abs=1e-3)

    def test_case_reclassifies_through_the_walk(self, annual_dataset):
        # The two years straddle the one bound, so both branches are exercised in one
        # result: 2040's 32 cells match the arm, 2030's 32 fall through to the `else`. A
        # `case` never drops a row, so the height is the full 64 either way -- which is
        # what separates it from the `filter` this replaces in `lnd_conf`.
        df = self._run(
            annual_dataset,
            [
                {
                    "op": "case",
                    "input": "a",
                    "cases": [{"when": "value > 2035", "then": "1"}],
                    "else": "0",
                }
            ],
        )
        assert df.height == 64
        assert sorted(df["value"].value_counts().rows()) == [(0.0, 32), (1.0, 32)]

    def test_execute_dispatches_join_through_the_walk(self, monkeypatch):
        # `join_long`'s own correctness (key computation, renaming) is already covered by
        # `tests/api/test_long_ops.py::TestJoinLong`; what THIS test pins is that `execute`
        # itself routes a `JoinOp` to it, with both operands already scanned, rather than
        # exercising the join lowering a second time. The `ExecutionPlan` is built by hand
        # rather than through `parse_pipeline`/`validate_shapes`/`compile_plan`, because a
        # bare join's own output legitimately carries two value columns (`a` and `b` --
        # `_type_join` sets `value_columns` to `frozenset(op.inputs)`) and
        # `validate_shapes`'s "exactly one value column at the sink" rule (already tested
        # in `test_request_ops.py`) would reject ending a pipeline there; this test only
        # needs `execute`'s dispatch, not a full sink-legal pipeline.
        #
        # `scan_vector` is stubbed so the frames are hand-written and the expected values
        # hand-computed; what a real scan's literal `variable`/`time_unit`/`grid` columns
        # do to the join key is the separate concern of
        # `test_cross_variable_join_over_two_real_datasets` below.
        import cpra.mp.data.analysis.execution.execute as execute_module
        from cpra.mp.data.analysis.execution.plan import ExecutionPlan, ResolvedInput
        from cpra.mp.data.analysis.request.ops import JoinOp

        frame_a = pl.LazyFrame({"calendar_year": [2030, 2030], "grid_cell": [1, 2], "value": [5.0, 6.0]})
        frame_b = pl.LazyFrame({"calendar_year": [2030, 2030], "grid_cell": [1, 2], "value": [100.0, 200.0]})

        def fake_scan_vector(selection, *, base_path=None, columns=None, predicate=None):
            return {"ssf": frame_a, "sal": frame_b}[selection.variable]

        monkeypatch.setattr(execute_module, "scan_vector", fake_scan_vector)

        sel_a = parse_pipeline(
            {"inputs": {"a": {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"}}, "output": {"format": "csv"}}
        ).inputs["a"]
        sel_b = parse_pipeline(
            {"inputs": {"b": {"variable": "sal", "time_unit": "annual", "grid": "ecoregion_v001"}}, "output": {"format": "csv"}}
        ).inputs["b"]
        plan = ExecutionPlan(
            inputs={"a": ResolvedInput(selection=sel_a), "b": ResolvedInput(selection=sel_b)},
            ops=[JoinOp(step_no=1, type="attribute", inputs=["a", "b"], output="j")],
            frame_types=frame_types("ecoregion_v001", a=Shape.long, b=Shape.long, j=Shape.long),
        )

        df = execute(plan).collect().sort("grid_cell")
        assert df["a"].to_list() == [5.0, 6.0]
        assert df["b"].to_list() == [100.0, 200.0]

    def test_cross_variable_join_over_two_real_datasets(self, base_path):
        # The end-to-end counterpart to `test_long_ops.py`'s hand-built join frames: two
        # REAL fixture datasets, two different variables on one grid, read through the
        # real `io.vector.scan_vector` -- which materializes `variable`/`time_unit`/`grid`
        # as literal columns, disagreeing on `variable` in every row. This is the shape
        # that returns an empty frame if those columns reach the join key, and it is the
        # most ordinary join there is, so it is worth spending real files on.
        #
        # The plan is hand-built for the same reason the dispatch test above hand-builds
        # one: a join's own output carries two value columns (`a`, `b`), which
        # `validate_shapes`'s single-value-column sink rule rejects as a pipeline's final
        # frame. Stopping at the join is what lets this test assert on those two columns
        # directly; `test_join_then_calculate_runs_as_a_whole_json_pipeline` below covers
        # the sink-legal form, where a `calculate` collapses them back to one `value`.
        from cpra.mp.data.analysis.execution.plan import ExecutionPlan, ResolvedInput
        from cpra.mp.data.analysis.request.ops import JoinOp
        from cpra.mp.data.analysis.request.pipeline import InputSelection

        write_annual_dataset(base_path, years=(2030, 2040))
        write_omar_dataset(base_path, years=(2030, 2040))

        selections = {
            "a": InputSelection(variable="ssf", time_unit="annual", grid="ecoregion_v001"),
            "b": InputSelection(variable="omar", time_unit="annual", grid="ecoregion_v001"),
        }
        plan = ExecutionPlan(
            inputs={name: ResolvedInput(selection=s) for name, s in selections.items()},
            ops=[JoinOp(step_no=1, type="spatial", inputs=["a", "b"], output="j")],
            frame_types=frame_types("ecoregion_v001", a=Shape.long, b=Shape.long, j=Shape.long),
        )

        df = execute(plan, base_path=base_path).collect()
        # 32 cells x 2 years, matched one-to-one -- not the zero rows an inner join on the
        # dataset-selection columns would give.
        assert df.height == 64
        assert not {"variable", "time_unit", "grid"} & set(df.columns)
        # Both fixtures use the same `value_fn` (year + geo_id/100), so every matched pair
        # holds the same number twice: proof the rows lined up on (year, cell) rather than
        # merging across cells.
        assert df["a"].to_list() == pytest.approx(df["b"].to_list(), abs=1e-3)
        # `ffibs_type` is a dimension only `omar` carries; it rides along rather than
        # being dropped or forced into the key.
        assert set(df["ffibs_type"].to_list()) == {"pl_bf"}

    def test_join_then_calculate_runs_as_a_whole_json_pipeline(self, base_path):
        # The sink-legal form of the test above, and the first pipeline in this suite that
        # goes all the way through `parse_pipeline` -> `validate_shapes` -> `compile_plan`
        # -> `execute` with more than one input: the `calculate` collapses the join's two
        # operand-named value columns back to one `value`, which is the mechanism the
        # single-value-column sink rule relies on.
        write_annual_dataset(base_path, years=(2030, 2040))
        write_omar_dataset(base_path, years=(2030, 2040), value_fn=lambda year, geo_id: geo_id)

        pipeline = _pipeline(
            {
                "inputs": {
                    "a": {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"},
                    "b": {"variable": "omar", "time_unit": "annual", "grid": "ecoregion_v001"},
                },
                "steps": [
                    {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                    {"op": "calculate", "expression": "a - b"},
                ],
                "output": {"format": "csv"},
            }
        )
        df = execute(compile_plan(pipeline), base_path=base_path).collect()

        assert df.height == 64
        # One value column named `value`, and the operand columns gone -- exactly what
        # `_type_calculate` types the output as carrying.
        assert "value" in df.columns
        assert not {"a", "b"} & set(df.columns)
        # ssf = year + geo_id/100, omar = geo_id, so every row's answer is distinct per
        # (year, cell) -- a mispairing across cells or years would not reproduce this.
        expected = sorted(
            year + geo_id / 100 - geo_id for year in (2030, 2040) for geo_id in range(1, 33)
        )
        assert sorted(df["value"].to_list()) == pytest.approx(expected, abs=1e-3)

    def test_project_runs_as_a_whole_json_pipeline(self, annual_dataset, crosswalk_path):
        # The resume contract's end-to-end walk: a real fixture dataset, a real crosswalk
        # file on disk, taken through the whole `parse_pipeline` -> `validate_shapes` ->
        # `compile_plan` -> `execute` chain, ending in `project`'s own long-shaped
        # lowering (`execution/project.py`) rather than a hand-built frame the way
        # `tests/api/test_project.py` exercises it.
        #
        # Only cells 1 and 2 are given a crosswalk row (weights 0.5/0.5 into one
        # destination) -- the other 30 fixture cells have no row at all, so the crosswalk
        # join drops them, and the resulting frame has exactly one destination cell per
        # year. `ssf`'s fixture value is `year + geo_id / 100`, so:
        #   sum = 0.5*(year + 0.01) + 0.5*(year + 0.02) = year + 0.015
        write_vector_crosswalk(
            crosswalk_path, "ecoregion_v001", "parish_v2020", [(1, 900, 0.5), (2, 900, 0.5)]
        )
        pipeline = _pipeline(
            {
                "inputs": {
                    "a": {"variable": ANNUAL_KWARGS["variable"], "time_unit": "annual", "grid": ANNUAL_KWARGS["grid"]}
                },
                "steps": [{"op": "project", "input": "a", "to": "parish_v2020", "aggregate": "sum"}],
                "output": {"format": "csv"},
            }
        )
        df = execute(compile_plan(pipeline), base_path=annual_dataset).collect().sort("calendar_year")

        assert df["grid_cell"].to_list() == [900, 900]
        assert df["calendar_year"].to_list() == [2030, 2040]
        assert df["value"].to_list() == pytest.approx([2030.015, 2040.015])
        # The `grid` column a vector scan materializes names the DESTINATION grid here,
        # not `a`'s own -- those `grid_cell` values are parish ids. This is the one place
        # the whole chain runs, so it is the one place the column reaching a writer can be
        # checked against the cell ids sitting next to it.
        assert set(df["grid"].to_list()) == {"parish_v2020"}


class TestBandWideEndToEnd:
    """The resume contract for a raster input: opening a band-wide frame through
    `execute()`, running `filter`/`calculate`/`group`/`join`/`project` against it, and the
    mid-walk transition to long shape a spatial filter causes -- each covered end to end
    rather than only through the lowering's own unit tests
    (`test_band_ops.py`/`test_project_raster.py`/`test_raster_scan.py`).

    Fixture: `write_raster_dataset` with `RASTER_KWARGS`'s defaults -- `vct` on
    `RASTER_GRID`, one band per `calendar_year`, ordered ascending (`cpra.mp.data.
    raster.read_cog`'s own sort order, `path_cols` then the calendar column) -- so a
    two-year dataset always resolves `b0` to the earlier year and `b1` to the later one.
    """

    def test_filter_calculate_group_runs_through_execute(self, base_path):
        # b0 (2030) fills 30, b1 (2040) fills 40. `value > 35` masks b0 to null (30 is not
        # > 35) and keeps b1; `value * 2` then only touches the surviving band (b0 stays
        # null -- `calculate` never revives a masked pixel, it only rewrites one that is
        # still there); `group by=[] aggregate="sum"` reduces the two bands to one, and
        # `sum_horizontal` skips the null contributor, so the answer is 80.0 (0 + 40*2)
        # everywhere, not (30*2 + 40*2) and not null. A version that ran `calculate` before
        # `filter`, or that summed before doubling, would both land on a different number.
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda year: 30 if year == 2030 else 40)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [
                    {"op": "filter", "input": "a", "where": "value > 35", "as": "f"},
                    {"op": "calculate", "expression": "value * 2", "as": "c"},
                    {"op": "group", "by": [], "aggregate": "sum"},
                ],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        final = plan.ops[-1].output
        assert plan.frame_types[final].shape is Shape.band_wide

        lf, metadata_df, window = execute(plan, base_path=base_path)
        df = lf.collect()

        assert df.columns == ["b0"]
        stats = df.select(
            pl.col("b0").min().alias("min"),
            pl.col("b0").max().alias("max"),
            pl.col("b0").null_count().alias("nulls"),
        ).row(0, named=True)
        assert stats["min"] == pytest.approx(80)
        assert stats["max"] == pytest.approx(80)
        assert stats["nulls"] == 0
        # `group_band_wide(by=[])` collapses every dimension column out of `metadata_df`,
        # leaving the reserved band-index column as the ONLY thing carrying "one band
        # survived" -- so what `execute()`'s own cleanup must not do here is drop it, which
        # would take the height with it (Polars cannot hold zero columns and one row) and
        # report an empty result for a raster that plainly has 383,980 pixels of data in
        # it. The band count, not the column's absence, is what this asserts: the earlier
        # spelling (`RESERVED_BAND_INDEX_COLUMN not in metadata_df.columns`) was satisfied
        # by a metadata frame collapsed to shape (0, 0), which is exactly the bug.
        assert not has_no_bands(metadata_df)
        assert metadata_df.height == 1
        assert band_column_names(metadata_df) == ["b0"]

    def test_case_reclassifies_band_wide_through_the_walk(self, base_path):
        # b0 (2030) fills 30, b1 (2040) fills 40, and the two arms straddle 35 -- so the
        # two bands take different branches and neither can be right by accident. The
        # `group sum` afterwards is what proves the frame stayed BAND-WIDE across the
        # `case`: 1 + 2 = 3 only if both bands survived it as bands.
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda year: 30 if year == 2030 else 40)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [
                    {
                        "op": "case",
                        "input": "a",
                        "cases": [{"when": "value > 35", "then": "2"}, {"when": "value > 0", "then": "1"}],
                        "as": "classed",
                    },
                    {"op": "group", "by": [], "aggregate": "sum"},
                ],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        assert plan.frame_types["classed"].shape is Shape.band_wide

        lf, metadata_df, _window = execute(plan, base_path=base_path)
        df = lf.collect()

        assert df.columns == ["b0"]
        assert band_column_names(metadata_df) == ["b0"]
        stats = df.select(
            pl.col("b0").min().alias("min"), pl.col("b0").max().alias("max")
        ).row(0, named=True)
        assert stats["min"] == pytest.approx(3)
        assert stats["max"] == pytest.approx(3)

    def test_spatial_filter_turns_the_frame_long_mid_walk(self, base_path):
        # Same two-band fixture as above. `grid_cell == 0` is a SPATIAL conjunct, so
        # `band_ops.filter_band_wide` dispatches to `sample_grid_cells_band_wide`, which
        # unpivots every pixel of every band into one long row per (pixel, band) and then
        # applies the whole `where` -- turning the frame long, with exactly one row per
        # band at pixel 0 (2030: 30.0, 2040: 40.0). The `calculate` after it must then run
        # `calculate_long`, not `calculate_band_wide` -- asserted below via `frame_types`,
        # not merely inferred from the answer being right, since a walk that accidentally
        # kept dispatching to the band-wide lowering on an already-long frame would fail
        # with a schema error, not silently produce a plausible-looking wrong number.
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda year: 30 if year == 2030 else 40)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [
                    {"op": "filter", "input": "a", "where": "grid_cell == 0", "as": "sampled"},
                    {"op": "calculate", "expression": "value * 2", "as": "doubled"},
                ],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        assert plan.frame_types["sampled"].shape is Shape.long
        assert plan.frame_types["doubled"].shape is Shape.long

        df = execute(plan, base_path=base_path).collect()

        assert df["grid_cell"].to_list() == [0, 0]
        assert sorted(df["calendar_year"].to_list()) == [2030, 2040]
        assert sorted(df["value"].to_list()) == pytest.approx([60.0, 80.0])

    def test_point_sample_reads_a_narrower_window_and_the_value_it_names(self, base_path, monkeypatch):
        # This is the BOUNDING-BOX fallback, not the point reader: the pipeline puts a
        # dimension `filter` between the origin and the spatial one, which
        # `_point_sample_reads` rejects (its gate requires the spatial filter to be the
        # origin's only reader), so the origin is opened band-wide over a narrowed window
        # exactly as it always was. The extra step is what makes this test still test the
        # narrowing -- without it the gate fires and `scan_raster` is never called at all.
        # `TestRasterPointReader` covers the gated path.
        #
        # `RASTER_GRID` is 365 rows x 1052 columns, so a pixel away from the origin --
        # (row=200, col=800) here -- has an ABSOLUTE row-major id (200*1052+800 = 211200).
        # Narrowing the read window to that single pixel must not change what id comes
        # back: `sample_grid_cells_band_wide` emits `grid_cell` as this ABSOLUTE id
        # (`row_major_grid_cell`'s `grid_width=` mode), never a position relative to
        # whatever window the read actually used, so the literal a caller filtered on
        # (211200) is exactly the id the one surviving row reports -- even though the read
        # itself only covered a 1x1 window whose only pixel would be relative id 0. Two
        # independent, provable failure shapes this test can still catch: (a) the window
        # narrows correctly but the emitted id is window-relative instead of absolute --
        # the assertion on `grid_cell` below would see `0`, not `211200`; (b) the window
        # itself is derived wrong (e.g. `divmod` against the wrong axis) -- the read then
        # covers some OTHER pixel, which this fixture deliberately fills with a different
        # value (1) than the target pixel (77), so a wrong window silently returns the
        # wrong number instead of raising. Asserting the narrowed window, the absolute
        # `grid_cell`, and the value together catches any of these; none could hide behind
        # the others.
        row, col = 200, 800
        grid = GRIDS[RASTER_GRID]
        assert grid.width != grid.height  # test premise: a row/col swap must be detectable
        cell_id = row * grid.width + col
        target_value = 77

        def data_fn(year):
            data = raster_array(fill=1, grid_name=RASTER_GRID, dtype="int32")
            data[row, col] = target_value
            return data

        write_raster_dataset(base_path, years=(2030,), data_fn=data_fn)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [
                    {"op": "filter", "input": "a", "where": "calendar_year = 2030", "as": "d"},
                    {"op": "filter", "where": f"grid_cell = {cell_id}"},
                ],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        assert plan.inputs["a"].point_sample_step is not None

        import cpra.mp.data.analysis.execution.execute as execute_module
        from cpra.mp.data.analysis.io.raster import scan_raster as real_scan_raster

        windows = []

        def capturing_scan_raster(*args, **kwargs):
            windows.append(kwargs.get("window"))
            return real_scan_raster(*args, **kwargs)

        monkeypatch.setattr(execute_module, "scan_raster", capturing_scan_raster)

        df = execute(plan, base_path=base_path).collect()

        assert len(windows) == 1
        window = windows[0]
        assert (window.col_off, window.row_off, window.width, window.height) == (col, row, 1, 1)

        # `grid_cell` in the result is the ABSOLUTE id the caller's `where` named, not a
        # position relative to the narrowed window the read actually used -- narrowing the
        # window is a pure read-size optimization, invisible in what an id in the output
        # means.
        assert df["grid_cell"].to_list() == [cell_id]
        assert df["value"].to_list() == pytest.approx([float(target_value)])

    def test_point_sample_ids_round_trip_away_from_the_raster_origin(self, base_path, monkeypatch):
        # A two-pixel round trip that the origin-adjacent fixtures elsewhere in this class
        # (`grid_cell == 0`, at the raster's own origin) cannot exercise: at the origin,
        # window-relative and absolute ids happen to coincide by coincidence (row_off=
        # col_off=0), so a test filtering there would pass whether or not the emitted id was
        # absolute. `row=200, col=800` and its immediate row-neighbor `col=801` sit far
        # from the origin, so window-relative ids (`0, 1`) and absolute ids (`211200,
        # 211201`) are provably different values here -- only absolute ids make this round
        # trip correctly, and a window-relative id would return `[0, 1]` instead. This
        # pipeline is gate-eligible, so it round-trips through the POINT reader (see
        # `TestRasterPointReader`); the bounding-box path makes the same round trip in the
        # test above.
        row, col = 200, 800
        grid = GRIDS[RASTER_GRID]
        cell_id_a = row * grid.width + col
        cell_id_b = row * grid.width + col + 1

        def data_fn(year):
            data = raster_array(fill=1, grid_name=RASTER_GRID, dtype="int32")
            data[row, col] = 11
            data[row, col + 1] = 22
            return data

        write_raster_dataset(base_path, years=(2030,), data_fn=data_fn)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [
                    {"op": "filter", "input": "a", "where": f"grid_cell IN ({cell_id_a}, {cell_id_b})"}
                ],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        assert plan.inputs["a"].point_sample_step is not None

        df = execute(plan, base_path=base_path).collect()

        rows = {row["grid_cell"]: row["value"] for row in df.to_dicts()}
        assert set(rows) == {cell_id_a, cell_id_b}
        assert rows[cell_id_a] == pytest.approx(11.0)
        assert rows[cell_id_b] == pytest.approx(22.0)

    def test_band_wide_join_reads_operands_from_one_shared_scan(self, base_path, monkeypatch):
        # Two DIFFERENT raster variables on `RASTER_GRID` -- `vct` fills 10 everywhere,
        # `ffibs_score` fills 5.0 everywhere -- joined spatially into one band-wide frame,
        # then a trailing `group by=[]` to collapse back to the single-value-column sink
        # rule (`_type_join` types a join's output as carrying `value_columns={*op.inputs}`,
        # which every output rejects; `_type_group` collapses it to `{"value"}`). `sum`
        # rather than some other aggregate because 10 + 5 = 15 is only reachable if BOTH
        # operands' bands made it into the frame with their own values intact.
        # `plan.band_wide_join_scans` should record this join's two operands as one
        # shared-scan group; `scan_raster` is monkeypatched to a call-counting wrapper
        # around the real function so this is checked at the IO level -- if the walk
        # opened each operand with its own `scan_raster` call instead of the one shared
        # scan `_open_band_wide_join_group` performs, this would still likely produce the
        # right VALUES (both files cover the identical window) but would call `scan_raster`
        # twice, which inspecting only the answer could never catch.
        write_raster_dataset(
            base_path,
            years=(2030,),
            variable="vct",
            data_fn=lambda year: raster_array(fill=10, grid_name=RASTER_GRID, dtype="int32"),
        )
        write_raster_dataset(
            base_path,
            years=(2030,),
            variable="ffibs_score",
            data_fn=lambda year: raster_array(fill=5, grid_name=RASTER_GRID, dtype="float32"),
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable="vct"), "b": raster_input(variable="ffibs_score")},
                "steps": [
                    {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                    {"op": "group", "by": [], "aggregate": "sum"},
                ],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        assert plan.frame_types["j"].shape is Shape.band_wide
        assert plan.band_wide_join_scans == {"j": ("a", "b")}

        import cpra.mp.data.analysis.execution.execute as execute_module
        from cpra.mp.data.analysis.io.raster import scan_raster as real_scan_raster

        calls = []

        def counting_scan_raster(*args, **kwargs):
            calls.append((args, kwargs))
            return real_scan_raster(*args, **kwargs)

        monkeypatch.setattr(execute_module, "scan_raster", counting_scan_raster)

        lf, metadata_df, window = execute(plan, base_path=base_path)
        df = lf.collect()

        assert len(calls) == 1
        assert df.columns == ["b0"]
        assert df["b0"].unique().to_list() == pytest.approx([15.0])

    def test_a_pixel_named_after_the_join_narrows_the_shared_scan_to_it(self, base_path, monkeypatch):
        # `analyses/sediment_dynamics.py`'s shape, at two operands instead of five: join
        # the rasters band-for-band, name ONE pixel, fold the operand columns into rows.
        # The join is what used to defeat the narrowing -- `_Validity` tracks a single
        # origin, so the joined frame carried none and neither operand recorded a
        # `point_sample_step`, leaving both rasters to be read whole and sampled
        # afterwards. `compile_plan` now records the sample for the GROUP, and
        # `_shared_point_sample_window` reduces the two windows to the one this scan reads.
        #
        # Asserted on the window rather than only on the values, for the reason the
        # single-origin test above gives at length: both files cover the same extent, so a
        # full-grid read returns the identical numbers and inspecting the answer alone
        # could never catch the regression. `(row=200, col=800)` again, so an absolute id
        # (211200) and a window-relative one (0) are provably different, and a `divmod`
        # against the wrong axis lands on a pixel this fixture fills differently.
        row, col = 200, 800
        grid = GRIDS[RASTER_GRID]
        cell_id = row * grid.width + col

        def _fill(background, target, dtype):
            def data_fn(year):
                data = raster_array(fill=background, grid_name=RASTER_GRID, dtype=dtype)
                data[row, col] = target
                return data

            return data_fn

        write_raster_dataset(
            base_path, years=(2030,), variable="vct", data_fn=_fill(1, 77, "int32")
        )
        write_raster_dataset(
            base_path, years=(2030,), variable="ffibs_score", data_fn=_fill(2, 88, "float32")
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable="vct"), "b": raster_input(variable="ffibs_score")},
                "steps": [
                    {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                    {"op": "filter", "input": "j", "where": f"grid_cell = {cell_id}", "as": "s"},
                    {"op": "unpivot", "input": "s"},
                ],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        assert plan.band_wide_join_scans == {"j": ("a", "b")}
        assert [plan.inputs[name].point_sample_step for name in ("a", "b")] == [2, 2]

        import cpra.mp.data.analysis.execution.execute as execute_module
        from cpra.mp.data.analysis.io.raster import scan_raster as real_scan_raster

        windows = []

        def capturing_scan_raster(*args, **kwargs):
            windows.append(kwargs.get("window"))
            return real_scan_raster(*args, **kwargs)

        monkeypatch.setattr(execute_module, "scan_raster", capturing_scan_raster)

        df = execute(plan, base_path=base_path).collect()

        # One scan for the group, over one pixel -- not two scans, and not the whole grid.
        assert len(windows) == 1
        window = windows[0]
        assert (window.col_off, window.row_off, window.width, window.height) == (col, row, 1, 1)

        # Both operands' own values, under the folded labels the operand names became, at
        # the ABSOLUTE id the caller named -- the narrowing is invisible in all three.
        assert df["grid_cell"].unique().to_list() == [cell_id]
        assert {row["variable"]: row["value"] for row in df.to_dicts()} == pytest.approx(
            {"a": 77.0, "b": 88.0}
        )

    def test_one_shared_scan_still_aligns_operands_narrowed_by_different_amounts(
        self, base_path, monkeypatch
    ):
        # The shared scan above, with the symmetry that made it obviously safe taken away.
        # `_merge_shared_origin_groups` fuses a join's origins into one `scan_raster` call
        # because chunking follows band count and `join_band_wide` aligns positionally --
        # both of which were easy to believe when every origin contributed its whole band
        # list. The dimension pushdown narrows origins independently, so here `a` is down
        # to one band (a filter on its own side) while `b` keeps both of its: one scan over
        # three bands, and each origin's slice out of it a different width.
        #
        # `sum` is what checks that each slice landed on its OWN bands rather than on the
        # first N of the combined scan: 20 (`a`'s 2040) + 5 (`b`'s 2030) + 7 (`b`'s 2040)
        # is 32, and no other assignment of those three bands to those two origins reaches
        # it.
        write_raster_dataset(
            base_path,
            years=(2030, 2040),
            variable="vct",
            data_fn=lambda year: raster_array(
                fill=10 if year == 2030 else 20, grid_name=RASTER_GRID, dtype="int32"
            ),
        )
        write_raster_dataset(
            base_path,
            years=(2030, 2040),
            variable="ffibs_score",
            data_fn=lambda year: raster_array(
                fill=5 if year == 2030 else 7, grid_name=RASTER_GRID, dtype="float32"
            ),
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable="vct"), "b": raster_input(variable="ffibs_score")},
                "steps": [
                    {"op": "filter", "input": "a", "where": "calendar_year = 2040", "as": "fa"},
                    {"op": "join", "type": "spatial", "inputs": ["fa", "b"], "as": "j"},
                    {"op": "group", "by": [], "aggregate": "sum"},
                ],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        assert plan.inputs["a"].band_metadata["calendar_year"].to_list() == [2040]
        assert plan.inputs["b"].band_metadata["calendar_year"].to_list() == [2030, 2040]
        assert plan.band_wide_join_scans == {"j": ("a", "b")}

        import cpra.mp.data.analysis.execution.execute as execute_module
        from cpra.mp.data.analysis.io.raster import scan_raster as real_scan_raster

        calls = []

        def counting_scan_raster(*args, **kwargs):
            calls.append((args, kwargs))
            return real_scan_raster(*args, **kwargs)

        monkeypatch.setattr(execute_module, "scan_raster", counting_scan_raster)

        df = execute(plan, base_path=base_path)[0].collect()

        assert len(calls) == 1
        assert df.columns == ["b0"]
        assert df["b0"].unique().to_list() == pytest.approx([32.0])

    def test_a_post_join_conjunct_that_empties_one_operand_still_runs(self, base_path):
        # `plan.py`'s join-sibling floor, end to end. `a` has only 2030, `b` has 2030 and
        # 2040, and the conjunct -- declared AFTER the join, so it reads paired metadata --
        # asks for 2040. `b`'s 2040 row pairs with no `a` row, so the narrowing leaves `a`
        # with zero bands beside a one-band `b`: correct as an answer and unreadable as a
        # plan, since `join_band_wide` needs both operands to hand it the same pixel row
        # count. Without the floor this raises `partial zero-band mismatch` on a pipeline
        # that is perfectly legal, and that used to work when nothing was pushed at all.
        #
        # 7.0 -- `b`'s 2040 fill alone -- is the answer with no pushdown whatsoever, which
        # is the point: the band `a` keeps is read and then dropped again by the `filter`
        # step re-executing over the joined frame, so it costs one band of I/O and changes
        # nothing.
        write_raster_dataset(
            base_path,
            years=(2030,),
            variable="vct",
            data_fn=lambda year: raster_array(fill=10, grid_name=RASTER_GRID, dtype="int32"),
        )
        write_raster_dataset(
            base_path,
            years=(2030, 2040),
            variable="ffibs_score",
            data_fn=lambda year: raster_array(
                fill=5 if year == 2030 else 7, grid_name=RASTER_GRID, dtype="float32"
            ),
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable="vct"), "b": raster_input(variable="ffibs_score")},
                "steps": [
                    {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                    {"op": "filter", "input": "j", "where": "calendar_year = 2040", "as": "d"},
                    {"op": "group", "by": [], "aggregate": "sum"},
                ],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        assert plan.inputs["a"].band_metadata["calendar_year"].to_list() == [2030]
        assert plan.inputs["b"].band_metadata["calendar_year"].to_list() == [2040]

        df = execute(plan, base_path=base_path)[0].collect()

        assert df.columns == ["b0"]
        assert df["b0"].unique().to_list() == pytest.approx([7.0])

    def test_five_operand_join_then_point_filter_then_unpivot(self, base_path):
        # `sediment_dynamics`'s own shape, end to end and at its real operand count: five
        # rasters on one grid, joined band-wide, sampled at a single `grid_cell`, then
        # folded so the five operand-named value columns become five rows of one `value`.
        # Three things only this combination checks:
        #
        # * The `unpivot` is what makes the pipeline WRITABLE. A five-operand join's output
        #   carries five value columns, which every sink rejects; a `calculate` could
        #   collapse them only by doing arithmetic across five different measurements,
        #   which is not what a five-series chart wants. This is the op that was missing.
        # * The fold is per-observation, not per-column: one cell x two years x five
        #   variables is ten rows, and each carries its own year and cell. A fold that
        #   dropped the index columns would still give ten rows of the right values.
        # * Each variable's own fill is distinct and hand-checked below, so a fold that
        #   mislabeled the stack -- `variable` rotated by one against `value` -- fails
        #   here even though the row count and the two column contents would each be right
        #   on their own.
        grid = GRIDS[RASTER_GRID]
        cell_id = 200 * grid.width + 800
        # Five variables `variables.csv` actually places on this grid, rather than
        # `sediment_dynamics`'s own five: those live on `morph_pixel`, whose 8265x20674
        # pixels are not something to write five COGs of in a unit test. What this test is
        # about is the operand COUNT and the fold, neither of which depends on which
        # variables they are -- and the three that require a dimension of their own are
        # kept in on purpose, since those dimension columns ride through the fold as index
        # columns and are exactly what a fold that dropped its index would lose.
        operands = {
            "vct": ({}, "int32"),
            "ffibs_score": ({}, "float32"),
            "veg_area": ({}, "int64"),
            "ffibs_coverage_ratio": ({"ffibs_type": "pl_fm"}, "float32"),
            "hsi": ({"species_code": "allig"}, "float32"),
        }
        fills = dict(zip(operands, (1, 2, 3, 4, 5)))
        for variable, (dimensions, dtype) in operands.items():
            write_raster_dataset(
                base_path,
                years=(2030, 2040),
                variable=variable,
                data_fn=lambda year, fill=fills[variable], dtype=dtype: raster_array(
                    fill=fill * 10 + (year - 2030) // 10, grid_name=RASTER_GRID, dtype=dtype
                ),
                **dimensions,
            )
        pipeline = _pipeline(
            {
                "inputs": {name: raster_input(variable=name) for name in fills},
                "steps": [
                    {"op": "join", "type": "spatial", "inputs": list(fills), "as": "j"},
                    {"op": "filter", "input": "j", "where": f"grid_cell = {cell_id}", "as": "pt"},
                    {"op": "unpivot", "input": "pt", "as": "stacked"},
                ],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        assert plan.frame_types["j"].shape is Shape.band_wide
        assert plan.frame_types["pt"].shape is Shape.long
        assert plan.frame_types["stacked"].value_columns == frozenset({"value"})
        assert plan.frame_types["stacked"].folded_columns == frozenset(fills)

        df = execute(plan, base_path=base_path).collect()

        # Five times the two-year, one-cell result the join produced.
        assert df.height == 10
        assert set(df["variable"].to_list()) == set(fills)
        assert df["grid_cell"].unique().to_list() == [cell_id]
        assert not set(fills) & set(df.columns)
        assert {"ffibs_type", "species_code"} <= set(df.columns)
        assert dict(zip(zip(df["variable"], df["calendar_year"]), df["value"])) == {
            (variable, year): pytest.approx(fill * 10 + (year - 2030) // 10)
            for variable, fill in fills.items()
            for year in (2030, 2040)
        }

    def test_raster_to_vector_project_runs_through_execute(self, base_path, crosswalk_path):
        # A constant-fill (10) raster and a crosswalk with ONE zone covering the whole
        # grid: mean-of-a-constant is invariant to how many pixels contributed, which
        # sidesteps hand-counting `veg_grid_cell_v001`'s ~384k pixels while still proving
        # `project_raster_to_vector` is reachable from `execute()`'s own
        # `("raster", "vector")` registry entry -- with a real crosswalk file and a real
        # catalog-resolved band list, not the hand-built `metadata_df` `test_project_
        # raster.py` uses for its own, narrower unit tests.
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda year: 10)
        grid = GRIDS[RASTER_GRID]
        zones = np.full((grid.height, grid.width), 900, dtype="int32")
        write_raster_crosswalk(crosswalk_path, RASTER_GRID, "ecoregion_v001", zones)

        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [{"op": "project", "input": "a", "to": "ecoregion_v001", "aggregate": "mean"}],
                "output": {"format": "csv"},
            }
        )
        df = execute(compile_plan(pipeline, base_path=base_path), base_path=base_path).collect()

        assert df["grid_cell"].to_list() == [900]
        assert df["value"].to_list() == pytest.approx([10.0])

    def test_a_join_group_whose_operands_all_resolve_to_zero_bands_opens_nothing(
        self, base_path, monkeypatch
    ):
        # The all-zero counterpart to the partial-mismatch case: when EVERY operand of a
        # shared-scan group is narrowed to zero bands by pushdown there is nothing to read
        # and nothing to mismatch, so the group is built from the grid's own geometry with
        # no scan at all. Covered here because the branch fabricates a result rather than
        # raising -- an untested branch that invents an answer looks, from the outside,
        # exactly like one that works. The `calculate` between the filters and the join is
        # deliberate: it walks the zero-band frame through one more lowering, which is the
        # shape that used to crash on metadata with no dimension columns left to read.
        write_raster_dataset(base_path, years=(2030,), variable="vct")
        write_raster_dataset(base_path, years=(2030,), variable="ffibs_score")
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable="vct"), "b": raster_input(variable="ffibs_score")},
                "steps": [
                    {"op": "filter", "input": "a", "where": "calendar_year == 2099", "as": "fa"},
                    {"op": "calculate", "input": "fa", "expression": "value * 2", "as": "ca"},
                    {"op": "filter", "input": "b", "where": "calendar_year == 2099", "as": "fb"},
                    {"op": "join", "type": "spatial", "inputs": ["ca", "fb"], "as": "j"},
                    {"op": "group", "by": [], "aggregate": "sum"},
                ],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        assert plan.band_wide_join_scans == {"j": ("a", "b")}
        assert plan.inputs["a"].band_metadata.height == 0
        assert plan.inputs["b"].band_metadata.height == 0

        import cpra.mp.data.analysis.execution.execute as execute_module

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("scan_raster must not be called when every operand is zero-band")

        monkeypatch.setattr(execute_module, "scan_raster", _fail_if_called)

        lf, metadata_df, window = execute(plan, base_path=base_path)

        assert metadata_df.is_empty()
        assert window.source == ()
        assert window.width == GRIDS[RASTER_GRID].width

    def test_a_project_off_a_shared_scan_join_reads_the_crosswalk_from_that_same_scan(
        self, base_path, crosswalk_path
    ):
        # The one pipeline shape where the crosswalk attachment and the shared-scan
        # prelude have to cooperate: two raster origins joined band-wide, then projected
        # onto a vector grid. Both origins are reachable from the `project`, so BOTH carry
        # a `straddling_vector_grid`, and the crosswalk has to ride on the one combined
        # read rather than on either operand's own -- which is why the shared scan gives
        # each origin its own internal zone column and re-aliases it back to the canonical
        # name, instead of attaching one crosswalk the way a single origin's read does.
        # Without a test at this shape the whole per-origin naming path runs in production
        # having never run here: a single-origin `project` never reaches it, and the join
        # test above never projects.
        #
        # `vct` fills 10, `ffibs_score` fills 5, so the joined frame's two bands are 10 and
        # 5; one zone covers the grid, so a zonal mean of each constant band is that band,
        # and the two bands come back as two long rows distinguished by their operand
        # column. A crosswalk lost or mispaired in the combined read would surface as a
        # missing column or an empty result, not as a plausible wrong number.
        write_raster_dataset(
            base_path,
            years=(2030,),
            variable="vct",
            data_fn=lambda year: raster_array(fill=10, grid_name=RASTER_GRID, dtype="int32"),
        )
        write_raster_dataset(
            base_path,
            years=(2030,),
            variable="ffibs_score",
            data_fn=lambda year: raster_array(fill=5, grid_name=RASTER_GRID, dtype="float32"),
        )
        grid = GRIDS[RASTER_GRID]
        zones = np.full((grid.height, grid.width), 900, dtype="int32")
        write_raster_crosswalk(crosswalk_path, RASTER_GRID, "ecoregion_v001", zones)

        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable="vct"), "b": raster_input(variable="ffibs_score")},
                "steps": [
                    {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                    {"op": "project", "to": "ecoregion_v001", "aggregate": "mean"},
                ],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        assert plan.band_wide_join_scans == {"j": ("a", "b")}
        assert plan.inputs["a"].straddling_vector_grid == "ecoregion_v001"
        assert plan.inputs["b"].straddling_vector_grid == "ecoregion_v001"

        df = execute(plan, base_path=base_path).collect()

        assert set(df["grid_cell"].to_list()) == {900}
        assert sorted(df["value"].to_list()) == pytest.approx([5.0, 10.0])
        # Neither the canonical zone column nor either origin's internal, per-origin one
        # may survive into the long result -- the whole point of re-aliasing them inside
        # the shared scan is that they stay internal to it.
        assert not [name for name in df.columns if "zone" in name]

    def test_a_project_after_a_by_nothing_group_does_not_leak_the_band_index_column(
        self, base_path, crosswalk_path
    ):
        # `group by=[]` collapses both bands to one and leaves the reserved band-index
        # column standing on the metadata (that lowering's own docstring says so). A
        # raster-to-vector `project` then turns every remaining metadata column into a
        # DIMENSION column on its long output, so the index would ride out of the walk
        # looking exactly like a dimension the caller asked for -- and `execute`'s own
        # end-of-walk cleanup cannot catch it, because that cleanup only runs for a
        # band-wide result, and by this point the index is an ordinary column of a long
        # one. The value is checked alongside it so this stays a test of the whole chain
        # rather than only of the column list: 30 and 40 summed horizontally is 70, and a
        # `project` that re-read raw pixels instead of the grouped frame would report the
        # two original bands separately rather than one row of 70.
        write_raster_dataset(
            base_path, years=(2030, 2040), fill_fn=lambda year: 30 if year == 2030 else 40
        )
        grid = GRIDS[RASTER_GRID]
        zones = np.full((grid.height, grid.width), 900, dtype="int32")
        write_raster_crosswalk(crosswalk_path, RASTER_GRID, "ecoregion_v001", zones)

        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [
                    {"op": "calculate", "input": "a", "expression": "value", "as": "c"},
                    {"op": "group", "by": [], "aggregate": "sum"},
                    {"op": "project", "to": "ecoregion_v001", "aggregate": "mean"},
                ],
                "output": {"format": "csv"},
            }
        )
        df = execute(compile_plan(pipeline, base_path=base_path), base_path=base_path).collect()

        assert RESERVED_BAND_INDEX_COLUMN not in df.columns
        assert df.height == 1
        assert df["grid_cell"].to_list() == [900]
        assert df["value"].to_list() == pytest.approx([70.0])

    def test_vector_to_raster_project_runs_through_execute(self, base_path, crosswalk_path):
        # The reverse direction, over the same crosswalk artifact (one raster<->vector
        # grid pair serves both directions -- see `test_project_raster.py`'s own
        # docstring): every pixel is tagged zone 1, so every pixel's broadcast value is
        # unambiguously ecoregion cell 1's `ssf` value at year 2030 (`year + geo_id / 100`
        # = 2030.01). Runs through `execute()`'s `("vector", "raster")` entry, which is
        # `project_vector_to_raster` registered directly (its signature already matches
        # the uniform registry shape -- see `execute.py`'s own comments for why only the
        # raster-to-vector direction needs an adapter).
        write_annual_dataset(base_path, years=(2030,))
        grid = GRIDS[RASTER_GRID]
        zones = np.full((grid.height, grid.width), 1, dtype="int32")
        write_raster_crosswalk(crosswalk_path, RASTER_GRID, "ecoregion_v001", zones)

        pipeline = _pipeline(
            {
                "inputs": {
                    "a": {"variable": ANNUAL_KWARGS["variable"], "time_unit": "annual", "grid": ANNUAL_KWARGS["grid"]}
                },
                "steps": [{"op": "project", "input": "a", "to": RASTER_GRID}],
                "output": {"format": "tif"},
            }
        )
        lf, metadata_df, window = execute(compile_plan(pipeline, base_path=base_path), base_path=base_path)
        df = lf.collect()

        assert metadata_df["calendar_year"].to_list() == [2030]
        assert df["b0"].unique().to_list() == pytest.approx([2030.01])
        # `project_vector_to_raster`'s zero-pivot-dimension case does not apply here (there
        # IS a pivot dimension, `calendar_year`), but the cleanup at the end of `execute()`
        # must be unconditional regardless -- this pins that it does not misfire and drop
        # a real dimension column by mistake.
        assert RESERVED_BAND_INDEX_COLUMN not in metadata_df.columns

    def test_zero_band_selection_builds_an_empty_frame_without_opening_a_file(self, base_path, monkeypatch):
        # A dimension filter (`calendar_year == 2099`) that matches no band on-disk narrows
        # `ResolvedInput.band_metadata` to zero rows entirely at COMPILE time (pushdown --
        # see `execution/plan.py`'s module docstring), before `execute()` ever runs.
        # `scan_raster` is monkeypatched to fail loudly if called at all -- a stronger check
        # than merely getting the right (empty) answer, since a version that fell through to
        # a real scan over zero metadata would raise `ValueError` from deep inside
        # `io.raster.scan_raster` instead of ever reaching this assertion, which would look
        # like a passing test failing for an unrelated reason rather than pinning the actual
        # behaviour under test.
        write_raster_dataset(base_path, years=(2030,))
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [{"op": "filter", "input": "a", "where": "calendar_year == 2099", "as": "f"}],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        assert plan.inputs["a"].band_metadata.height == 0
        assert plan.frame_types["f"].shape is Shape.band_wide

        import cpra.mp.data.analysis.execution.execute as execute_module

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("scan_raster must not be called for a zero-band selection")

        monkeypatch.setattr(execute_module, "scan_raster", _fail_if_called)

        lf, metadata_df, window = execute(plan, base_path=base_path)

        assert lf.collect().is_empty()
        assert metadata_df.is_empty()
        # The window is taken from the raster GRID's own geometry, not derived from any
        # file (none was opened) -- `source=()` names none, and the extent/transform match
        # a full, undecimated read of the same grid computed independently here rather than
        # by calling the production helper that built it, so a bug in that helper would not
        # be masked by comparing its output to itself.
        grid = GRIDS[RASTER_GRID]
        assert window.source == ()
        assert window.width == grid.width
        assert window.height == grid.height
        assert window.transform == from_bounds(*grid.bounds, width=grid.width, height=grid.height)
        assert window.crs == CRS.from_user_input(grid.crs)
        assert window.batch_layout == ()

    def test_project_reads_upstream_calculate_work_not_a_raw_rescan(self, base_path, crosswalk_path):
        # Regression test: `project_raster_to_vector` used to perform its OWN independent
        # `scan_raster` call, silently discarding whatever an upstream `calculate` had
        # already done to `frames[op.input]`'s `lf` -- it read only the metadata half of
        # that frame and re-scanned raw pixels from disk for the data half. A single
        # constant-fill (10) raster, doubled by `calculate` before the `project`, with a
        # crosswalk whose one zone covers the whole grid (mean-of-a-constant is invariant
        # to pixel count, so this doesn't need to hand-count the grid): the answer is 20.0
        # if the doubling was honored, 10.0 -- the raw on-disk value -- if `project`
        # re-read the file itself instead of the already-open, already-processed frame.
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda year: 10)
        grid = GRIDS[RASTER_GRID]
        zones = np.full((grid.height, grid.width), 900, dtype="int32")
        write_raster_crosswalk(crosswalk_path, RASTER_GRID, "ecoregion_v001", zones)

        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [
                    {"op": "calculate", "input": "a", "expression": "value * 2", "as": "c"},
                    {"op": "project", "input": "c", "to": "ecoregion_v001", "aggregate": "mean"},
                ],
                "output": {"format": "csv"},
            }
        )
        df = execute(compile_plan(pipeline, base_path=base_path), base_path=base_path).collect()

        assert df["grid_cell"].to_list() == [900]
        assert df["value"].to_list() == pytest.approx([20.0])

    def test_project_reads_upstream_filter_masking_not_a_raw_rescan(self, base_path, crosswalk_path):
        # Same regression as above, for `filter` instead of `calculate`. `value > 35`
        # masks the 2030 band (fill 30, not > 35) to null and leaves 2040 (fill 40)
        # unmasked. A `project` reading the SAME already-filtered frame sees an all-null
        # 2030 contribution (mean of nothing -> null) and 40.0 for 2040; a `project` that
        # re-scanned raw pixels on its own would see the filter's masking never happened
        # at all and report 30.0 for 2030 instead of null.
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda year: 30 if year == 2030 else 40)
        grid = GRIDS[RASTER_GRID]
        zones = np.full((grid.height, grid.width), 900, dtype="int32")
        write_raster_crosswalk(crosswalk_path, RASTER_GRID, "ecoregion_v001", zones)

        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [
                    {"op": "filter", "input": "a", "where": "value > 35", "as": "f"},
                    {"op": "project", "input": "f", "to": "ecoregion_v001", "aggregate": "mean"},
                ],
                "output": {"format": "csv"},
            }
        )
        df = (
            execute(compile_plan(pipeline, base_path=base_path), base_path=base_path)
            .collect()
            .sort("calendar_year")
        )

        assert df["calendar_year"].to_list() == [2030, 2040]
        values = df["value"].to_list()
        assert values[0] is None
        assert values[1] == pytest.approx(40.0)

    def test_zero_band_project_returns_empty_without_opening_a_file(self, base_path, crosswalk_path, monkeypatch):
        # Corner case: a dimension filter (`calendar_year == 2099`) narrows a raster
        # origin to zero bands at compile time, upstream of a raster-to-vector `project`.
        # `_open_band_wide_input`'s zero-band branch never calls `scan_raster` (pinned by
        # monkeypatching it to fail loudly), and `project.project_raster_to_vector`'s own
        # `has_no_bands` guard returns an empty frame rather than grouping by a
        # `ZONE_COLUMN` that was never attached (`extra_columns=()` for this case -- see
        # `_open_band_wide_input`'s own docstring).
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda year: 10)
        grid = GRIDS[RASTER_GRID]
        zones = np.full((grid.height, grid.width), 900, dtype="int32")
        write_raster_crosswalk(crosswalk_path, RASTER_GRID, "ecoregion_v001", zones)

        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [
                    {"op": "filter", "input": "a", "where": "calendar_year == 2099", "as": "f"},
                    {"op": "project", "input": "f", "to": "ecoregion_v001", "aggregate": "mean"},
                ],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        assert plan.inputs["a"].band_metadata.height == 0

        import cpra.mp.data.analysis.execution.execute as execute_module

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("scan_raster must not be called for a zero-band selection")

        monkeypatch.setattr(execute_module, "scan_raster", _fail_if_called)

        result = execute(plan, base_path=base_path)

        assert result.collect().is_empty()

    def test_spatial_filter_downstream_of_a_fanned_out_origin_does_not_leak_the_crosswalk_column(
        self, base_path, crosswalk_path
    ):
        # Corner case: "a" is read by two steps -- a `project` to a vector grid (which
        # makes `_open_band_wide_input` attach a crosswalk column under `project.
        # ZONE_COLUMN` to "a"'s own open frame) and, separately, a spatial `filter` that
        # turns "a" long. The spatial filter's own `extra_columns` handling
        # (`_run_filter_band_wide`) must strip that crosswalk column before the frame goes
        # long -- it names no destination cell of its own once the frame stops being the
        # raster origin, and a later `group`/`join` could silently key on it as if it were
        # a real dimension otherwise.
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda year: 10)
        grid = GRIDS[RASTER_GRID]
        zones = np.full((grid.height, grid.width), 900, dtype="int32")
        write_raster_crosswalk(crosswalk_path, RASTER_GRID, "ecoregion_v001", zones)

        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [
                    {"op": "project", "input": "a", "to": "ecoregion_v001", "aggregate": "mean", "as": "p"},
                    {"op": "filter", "input": "a", "where": "grid_cell == 0", "as": "sampled"},
                ],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        assert plan.inputs["a"].straddling_vector_grid == "ecoregion_v001"
        assert plan.frame_types["sampled"].shape is Shape.long

        df = execute(plan, base_path=base_path).collect()

        assert project.ZONE_COLUMN not in df.columns
        assert df["grid_cell"].to_list() == [0]
        assert df["value"].to_list() == pytest.approx([10.0])

    def test_final_defensive_strip_removes_the_crosswalk_column_from_a_band_wide_result(
        self, base_path, crosswalk_path
    ):
        # A different fan-out shape than the spatial-filter case above: "a" feeds a
        # `project` on one branch (attaching `project.ZONE_COLUMN` to "a"'s own open
        # frame) and an ordinary `calculate` on another, which stays BAND-WIDE and becomes
        # the pipeline's actual last declared step -- so the mid-walk strip inside
        # `_run_filter_band_wide` never runs at all (no filter is involved), and only
        # `execute()`'s own final, defensive strip (see the module docstring's closing
        # paragraph) is what keeps the crosswalk column from leaking into the returned
        # frame's `lf`.
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda year: 10)
        grid = GRIDS[RASTER_GRID]
        zones = np.full((grid.height, grid.width), 900, dtype="int32")
        write_raster_crosswalk(crosswalk_path, RASTER_GRID, "ecoregion_v001", zones)

        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [
                    {"op": "project", "input": "a", "to": "ecoregion_v001", "aggregate": "mean", "as": "p"},
                    {"op": "calculate", "input": "a", "expression": "value * 2", "as": "c"},
                ],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        assert plan.inputs["a"].straddling_vector_grid == "ecoregion_v001"
        assert plan.frame_types["c"].shape is Shape.band_wide

        lf, metadata_df, window = execute(plan, base_path=base_path)
        df = lf.collect()

        assert project.ZONE_COLUMN not in df.columns
        assert df.columns == ["b0"]
        assert df["b0"].unique().to_list() == pytest.approx([20.0])

    def test_a_partial_zero_band_join_operand_raises(self, base_path):
        # `plan.band_wide_join_scans` groups both operands of a band-wide `join` for one
        # shared scan whenever every origin in the group is raster -- including when a
        # dimension filter has already narrowed ONE operand to zero bands via pushdown
        # (`origin_sets` is provenance, not pushdown validity -- see `plan.py`'s own
        # module docstring). `_open_band_wide_join_group` cannot silently slice a
        # zero-band operand out of the shared scan (that slice would collapse to height 0
        # while its sibling keeps the scan's real row count -- a shape mismatch
        # `join_band_wide`'s own precondition would also catch, but only after producing a
        # confusing error with no indication which operand was actually empty), so it
        # raises instead.
        #
        # `NotFoundError`, not a bare `ValueError`: this is reachable from an ordinary,
        # well-formed request whose DERIVED selection happens to have no data -- `lnd_chng`
        # deriving a baseline `calendar_year` the dataset does not carry is the case that
        # found it, as a 500 in a deployed portal. `api.boundary` maps it to a 404 with the
        # message below, which is the answer a caller can act on. The message names each
        # empty operand's DATASET as well as its step name, since a registered analysis
        # writes its own pipeline and a caller has never seen the name `'a'`.
        write_raster_dataset(
            base_path,
            years=(2030,),
            variable="vct",
            data_fn=lambda year: raster_array(fill=10, grid_name=RASTER_GRID, dtype="int32"),
        )
        write_raster_dataset(
            base_path,
            years=(2030,),
            variable="ffibs_score",
            data_fn=lambda year: raster_array(fill=5, grid_name=RASTER_GRID, dtype="float32"),
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable="vct"), "b": raster_input(variable="ffibs_score")},
                "steps": [
                    {"op": "filter", "input": "a", "where": "calendar_year == 2099", "as": "fa"},
                    {"op": "join", "type": "spatial", "inputs": ["fa", "b"], "as": "j"},
                    {"op": "group", "by": [], "aggregate": "sum"},
                ],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        assert plan.inputs["a"].band_metadata.height == 0
        assert plan.band_wide_join_scans == {"j": ("a", "b")}

        from cpra.mp.data.analysis.request.errors import NotFoundError

        with pytest.raises(NotFoundError, match=r"no data for 'a' \(variable=vct,") as excinfo:
            execute(plan, base_path=base_path)
        assert "'b' did while it did not" in str(excinfo.value)


def _tile_over_the_raster_grid(zoom=12):
    """A tile covering the middle of `RASTER_GRID`'s extent. At z=12 that is a 27x27-pixel
    window out of the grid's 1052x365 -- small enough that reading it instead of the whole
    raster is unmistakable in any count below."""
    grid = GRIDS[RASTER_GRID]
    west, south, east, north = transform_bounds(grid.crs, "EPSG:4326", *grid.bounds)
    return mercantile.tile((west + east) / 2, (south + north) / 2, zoom)


class TestRasterPointReader:
    """`_point_sample_reads` -- the gate that swaps `io.raster_points.sample_raster_points`
    in for the band-wide read, and the conditions under which it declines to.

    What the reader itself produces is pinned in `test_raster_points.py`, including the
    frame-for-frame equivalence with the band-wide reader that makes the swap legitimate.
    What this class pins is the SWAP: that a gated origin never opens a band-wide read at
    all, that a rejected one still does, and that both answer the same question the same
    way.
    """

    @staticmethod
    def _counting_scan_raster(monkeypatch):
        """Count `scan_raster` calls the walk makes, without replacing what it does."""
        import cpra.mp.data.analysis.execution.execute as execute_module
        from cpra.mp.data.analysis.io.raster import scan_raster as real_scan_raster

        calls = []

        def counting(*args, **kwargs):
            calls.append(kwargs.get("window"))
            return real_scan_raster(*args, **kwargs)

        monkeypatch.setattr(execute_module, "scan_raster", counting)
        return calls

    @staticmethod
    def _point_pipeline(cells):
        return _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [
                    {
                        "op": "filter",
                        "input": "a",
                        "where": f"grid_cell IN ({', '.join(str(cell) for cell in cells)})",
                    }
                ],
                "output": {"format": "csv"},
            }
        )

    @staticmethod
    def _plant(pixels):
        def data_fn(year):
            data = raster_array(fill=1, grid_name=RASTER_GRID, dtype="int32")
            for (row, col), value in pixels.items():
                data[row, col] = value + year
            return data

        return data_fn

    def test_a_gated_origin_never_opens_a_band_wide_read(self, base_path, monkeypatch):
        # The point of the whole change: for this pipeline the walk must not call
        # `scan_raster` even once. `grid_cell` ids far from the origin (row 200, col 800 and
        # its neighbour) so that a transposed decode would sample the fill value (1 + year)
        # instead of the planted one and show up in `value`.
        grid = GRIDS[RASTER_GRID]
        pixels = {(200, 800): 11, (300, 900): 22}
        cells = [row * grid.width + col for row, col in pixels]
        write_raster_dataset(base_path, years=(2030,), data_fn=self._plant(pixels))
        plan = compile_plan(self._point_pipeline(cells), base_path=base_path)
        assert set(_point_sample_reads(plan)) == {"a"}
        # And nothing was left behind for the fallback to act on.
        assert _point_sample_overrides(plan, skip=_point_sample_reads(plan)) == {}

        calls = self._counting_scan_raster(monkeypatch)
        df = execute(plan, base_path=base_path).collect()

        assert calls == []
        assert dict(zip(df["grid_cell"], df["value"])) == {
            cells[0]: pytest.approx(11 + 2030),
            cells[1]: pytest.approx(22 + 2030),
        }

    def test_the_gated_and_ungated_paths_give_the_same_answer(self, base_path, monkeypatch):
        # The substitution, checked end to end rather than at the reader boundary: the same
        # compiled plan run with the gate forced off (every origin band-wide, the bounding
        # box this walk always used) and with it on must produce the same frame, columns and
        # dtypes included. Two bands, so the comparison spans the band-identity join too.
        grid = GRIDS[RASTER_GRID]
        pixels = {(200, 800): 11, (300, 900): 22, (7, 3): 33}
        cells = [row * grid.width + col for row, col in pixels]
        write_raster_dataset(base_path, years=(2030, 2040), data_fn=self._plant(pixels))
        pipeline = self._point_pipeline(cells)

        import cpra.mp.data.analysis.execution.execute as execute_module

        with monkeypatch.context() as ungated:
            ungated.setattr(execute_module, "_point_sample_reads", lambda plan: {})
            plan = compile_plan(pipeline, base_path=base_path)
            assert set(_point_sample_overrides(plan)) == {"a"}  # the fallback did narrow
            fallback = execute(plan, base_path=base_path).collect()

        plan = compile_plan(pipeline, base_path=base_path)
        gated = execute(plan, base_path=base_path).collect()

        sort_by = ["grid_cell", "calendar_year"]
        assert gated.columns == fallback.columns
        assert gated.schema == fallback.schema
        assert gated.sort(sort_by).equals(fallback.sort(sort_by))
        assert fallback.height == len(cells) * 2  # not a comparison of two empty frames

    def test_a_step_between_the_origin_and_the_filter_is_not_gated(self, base_path):
        # The condition the gate cannot do without. The long lowering applies `where` to
        # whatever frame it is handed; the point reader's frame holds only the named pixels.
        # Those agree only when nothing ran in between -- a `group` here would have reduced
        # every pixel of the raster on the band-wide path and three pixels on the point one,
        # which are different numbers, not the same number computed two ways.
        grid = GRIDS[RASTER_GRID]
        cell = 200 * grid.width + 800
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda year: 5)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [
                    {"op": "calculate", "input": "a", "expression": "value * 2", "as": "c"},
                    {"op": "filter", "where": f"grid_cell = {cell}"},
                ],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)

        assert plan.inputs["a"].point_sample_step is not None
        assert _point_sample_reads(plan) == {}
        # Rejected by the gate, but still narrowed by the fallback -- the two are separate
        # decisions, and this is the case that proves it.
        assert set(_point_sample_overrides(plan)) == {"a"}

        df = execute(plan, base_path=base_path).collect()
        assert df["value"].to_list() == pytest.approx([10.0])

    def test_a_predicate_naming_no_literal_ids_is_not_gated(self, base_path):
        # `grid_cell > N` is still a SPATIAL conjunct -- it turns the frame long exactly the
        # same way -- but it does not say which pixels before a file is opened, so neither
        # the reader nor the fallback window has anything to work from and the whole raster
        # is read.
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda year: 5)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [{"op": "filter", "input": "a", "where": "grid_cell > 383978"}],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)

        assert plan.inputs["a"].point_sample_step is not None
        assert _point_sample_reads(plan) == {}
        assert _point_sample_overrides(plan) == {}

        df = execute(plan, base_path=base_path).collect()
        assert sorted(df["grid_cell"].to_list()) == [383979]

    def test_an_id_that_names_no_pixel_of_the_grid_is_not_gated(self, base_path):
        # A `grid_cell` id past the end of the grid parses into a literal id perfectly well
        # -- what rejects it is the range check `_derive_point_sample_window` runs, the same
        # one the fallback runs, which is why the gate asks for both and not just the parse.
        # It matters that this falls back rather than being gated: the point reader decodes
        # ids against the grid too, and an id it cannot place is a `ValueError` there, where
        # the band-wide path simply matches nothing and returns an empty frame. Asking a
        # request layer to turn a pixel nobody has into a 500 would be a regression.
        grid = GRIDS[RASTER_GRID]
        off_grid = grid.height * grid.width + 5
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda year: 5)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [{"op": "filter", "input": "a", "where": f"grid_cell = {off_grid}"}],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)

        assert _parse_point_sample_ids(plan.ops[0].where) == [off_grid]  # the parse did succeed
        assert _point_sample_reads(plan) == {}
        assert _point_sample_overrides(plan) == {}

        assert execute(plan, base_path=base_path).collect().height == 0

    def test_an_origin_read_by_a_second_step_is_not_gated(self, base_path):
        # A fan-out: `a` feeds both a band-wide `group` and the spatial `filter`. Only one
        # read of `a` happens (one `ResolvedInput`, one entry in `frames`), so opening it
        # long for the filter's sake would hand the `group` a frame it cannot run against at
        # all. What forecloses that is upstream of the gate: `compile_plan` invalidates any
        # origin more than one op reads, so it never records a `point_sample_step` for `a`
        # in the first place, and the gate's first condition rejects on that. Asserted on
        # the plan rather than through the walk because the walk would fail with a shape
        # error rather than a wrong answer -- a failure that proves nothing about the gate.
        grid = GRIDS[RASTER_GRID]
        cell = 200 * grid.width + 800
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda year: 5)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [
                    {"op": "group", "input": "a", "by": [], "aggregate": "sum", "as": "q"},
                    {"op": "filter", "input": "a", "where": f"grid_cell = {cell}", "as": "p"},
                ],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)

        assert plan.inputs["a"].point_sample_step is None
        assert _point_sample_reads(plan) == {}
        # The same origin and the same filter, minus the second reader, IS gated -- so what
        # the assertions above caught is the fan-out and not some unrelated thing this
        # pipeline shape got wrong.
        solo = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [{"op": "filter", "input": "a", "where": f"grid_cell = {cell}", "as": "p"}],
                "output": {"format": "csv"},
            }
        )
        assert _point_sample_reads(compile_plan(solo, base_path=base_path)) == {"a": [cell]}

    def test_a_tile_scoped_origin_is_not_gated(self, base_path):
        # A tile's own window already governs what gets read, and a point reader has no
        # window to reconcile with it. Same skip the fallback makes, for the same reason.
        grid = GRIDS[RASTER_GRID]
        cell = 200 * grid.width + 800
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda year: 5)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [{"op": "filter", "input": "a", "where": f"grid_cell = {cell}"}],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path, tile=TileRequest(*_tile_over_the_raster_grid()))

        assert plan.inputs["a"].read_window is not None
        assert _point_sample_reads(plan) == {}


class TestTileScopedReads:
    """A plan compiled for a map tile (`compile_plan(..., tile=...)`) reads each raster
    origin at that tile's own window instead of whole, and the walk over it is the same walk
    as any other -- no tile-specific op path, which is what lets a tile request run the ops
    a band-wide sink could not (`project` onto a vector grid, below).

    Which window a tile needs is `execution.plan`'s decision and is pinned there
    (`test_execution_plan.py::TestTileReadWindow`); what this class pins is that the decision
    reaches the reader and moves the actual pixels.
    """

    def test_a_tile_scoped_read_returns_only_that_window(self, base_path):
        # A unique value in every pixel (`raster_array`'s `arange`), so this can assert the
        # read landed on the RIGHT 27x27 block and not merely on a 27x27-shaped one -- a
        # window whose offset was dropped somewhere between the compiler and `scan_raster`
        # would still be the right size. The expectation is read straight off the file with
        # rasterio, which shares no code with the scan under test.
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _year: raster_array())
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [],
                "output": {"format": "tif"},
            }
        )
        tile = TileRequest(*_tile_over_the_raster_grid())
        plan = compile_plan(pipeline, base_path=base_path, tile=tile)
        compiled = plan.inputs["a"].read_window

        lf, _metadata_df, window = execute(plan, base_path=base_path)
        df = lf.collect()

        assert (window.col_off, window.row_off) == (int(compiled.col_off), int(compiled.row_off))
        assert (window.width, window.height) == (int(compiled.width), int(compiled.height))
        assert df.height == window.width * window.height

        path = next(base_path.rglob("*.tif"))
        with rasterio.open(path) as src:
            expected = src.read(1, window=compiled)
        np.testing.assert_array_equal(
            df["b0"].to_numpy().reshape(window.height, window.width), expected
        )

    def test_a_tile_that_misses_the_grid_opens_nothing(self, base_path, monkeypatch):
        # `tile_disjoint` is settled at compile time, so the reader is never reached at all
        # -- pinned by making `scan_raster` fail loudly rather than by checking only that
        # the answer came back empty. The window still describes the whole grid, because a
        # frame has to carry SOME geometry and the grid's is the only one available when
        # nothing was opened (the same window the zero-band case builds).
        write_raster_dataset(base_path, years=(2030,))
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path, tile=TileRequest(1, 1, 1))
        assert plan.inputs["a"].tile_disjoint is True

        import cpra.mp.data.analysis.execution.execute as execute_module

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("scan_raster must not be called for a tile-disjoint origin")

        monkeypatch.setattr(execute_module, "scan_raster", _fail_if_called)

        lf, metadata_df, window = execute(plan, base_path=base_path)

        assert lf.collect().is_empty()
        assert has_no_bands(metadata_df)
        assert (window.width, window.height) == (GRIDS[RASTER_GRID].width, GRIDS[RASTER_GRID].height)

    def test_a_band_wide_join_reads_both_operands_from_one_tile_window(self, base_path, monkeypatch):
        # The shared-scan prelude under a tile: two origins on one grid, one scan, one
        # window. Both operands want the same window here (neither feeds a project), so what
        # this pins is that the tile scoping survives the join path at all -- the union logic
        # that handles operands wanting DIFFERENT windows is pinned directly below.
        write_raster_dataset(
            base_path,
            years=(2030,),
            variable="vct",
            data_fn=lambda _year: raster_array(fill=10, grid_name=RASTER_GRID, dtype="int32"),
        )
        write_raster_dataset(
            base_path,
            years=(2030,),
            variable="ffibs_score",
            data_fn=lambda _year: raster_array(fill=5, grid_name=RASTER_GRID, dtype="float32"),
        )
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable="vct"), "b": raster_input(variable="ffibs_score")},
                "steps": [
                    {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "j"},
                    {"op": "group", "by": [], "aggregate": "sum"},
                ],
                "output": {"format": "tif"},
            }
        )
        tile = TileRequest(*_tile_over_the_raster_grid())
        plan = compile_plan(pipeline, base_path=base_path, tile=tile)
        compiled = plan.inputs["a"].read_window

        import cpra.mp.data.analysis.execution.execute as execute_module
        from cpra.mp.data.analysis.io.raster import scan_raster as real_scan_raster

        calls = []

        def counting_scan_raster(*args, **kwargs):
            calls.append(kwargs.get("window"))
            return real_scan_raster(*args, **kwargs)

        monkeypatch.setattr(execute_module, "scan_raster", counting_scan_raster)

        lf, _metadata_df, window = execute(plan, base_path=base_path)
        df = lf.collect()

        assert len(calls) == 1
        assert (window.width, window.height) == (int(compiled.width), int(compiled.height))
        assert df.height == window.width * window.height
        # 10 + 5: the trailing `group by=[] aggregate="sum"` collapses the join's two
        # operand-named value columns to one, and only reaches 15 if both operands' bands
        # came off the shared scan with their own values.
        assert df["b0"].unique().to_list() == pytest.approx([15.0])

    def test_a_join_groups_shared_window_is_the_union_of_its_operands(self, base_path):
        # `_shared_tile_window` directly, since a pipeline that makes two operands of one
        # join want different windows takes a fan-out, a project, a geometry file and a
        # crosswalk to set up -- all of which this would then be testing through rather
        # than testing. Only an origin feeding a raster-to-vector `project` is widened, so
        # the mixed case is real; the union is what the one shared scan has to read, since a
        # window too small for either operand silently drops that operand's pixels.
        write_raster_dataset(base_path, years=(2030,))
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [],
                "output": {"format": "tif"},
            }
        )
        resolved = compile_plan(pipeline, base_path=base_path).inputs["a"]
        inputs = {
            "a": dataclasses.replace(resolved, read_window=Window(10, 20, 30, 40)),
            "b": dataclasses.replace(resolved, read_window=Window(5, 25, 30, 40)),
        }

        shared = _shared_tile_window(("a", "b"), inputs)

        # West edge from "b" (5), north edge from "a" (20), east edge from "a" (10+30=40),
        # south edge from "b" (25+40=65) -- every side taken from whichever operand reaches
        # furthest, which is the definition of "covers both".
        assert (int(shared.col_off), int(shared.row_off)) == (5, 20)
        assert (int(shared.col_off) + int(shared.width)) == 40
        assert (int(shared.row_off) + int(shared.height)) == 65

    def test_a_join_group_mixing_a_disjoint_operand_with_an_overlapping_one_raises(self, base_path):
        # Disjointness is a property of the tile and the GRID, and a band-wide join's
        # operands all sit on one grid -- so operands disagreeing about it means the group
        # was built wrong upstream, not that one of them happens to be empty. Raising names
        # that, rather than reading the union of "a window" and "no window at all".
        write_raster_dataset(base_path, years=(2030,))
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [],
                "output": {"format": "tif"},
            }
        )
        resolved = compile_plan(pipeline, base_path=base_path).inputs["a"]
        inputs = {
            "a": dataclasses.replace(resolved, read_window=Window(10, 20, 30, 40)),
            "b": dataclasses.replace(resolved, read_window=None, tile_disjoint=True),
        }

        with pytest.raises(ValueError, match="mixes tile-disjoint and overlapping"):
            _shared_tile_window(("a", "b"), inputs)

    def test_a_join_groups_point_sample_window_refuses_to_union(self):
        # `_shared_point_sample_window`'s two refusals, directly -- neither is reachable
        # from a pipeline (`compile_plan` records a group's `point_sample_step`s together,
        # off one filter, and a band-wide join's operands share a grid, so
        # `_derive_point_sample_window` runs the same `where` against the same grid for
        # each of them), and both are the reason the reduction is not a union.
        #
        # A tile window may be unioned: a wider window is more pixels, never different
        # ones. A point sample's window IS the answer, so two origins disagreeing about
        # which pixel was asked for have to fall back to the full read rather than be
        # averaged into a window neither of them named -- and an origin with no window at
        # all (tile-scoped, disjoint, or zero-band, which `_point_sample_overrides` skips)
        # takes its whole sibling group with it, because the group is read as one.
        window = Window(800, 200, 1, 1)
        assert _shared_point_sample_window(("a", "b"), {"a": window, "b": window}) is window
        assert _shared_point_sample_window(("a", "b"), {"a": window}) is None
        assert (
            _shared_point_sample_window(("a", "b"), {"a": window, "b": Window(801, 200, 1, 1)})
            is None
        )

    def test_a_tile_scoped_project_reads_whole_features_across_the_tile_edge(
        self, base_path, crosswalk_path, geometry_path
    ):
        # The case that only exists because the window is a compile-time decision: a tile
        # whose pixels feed a raster-to-vector `project`, where the zone they project onto
        # extends past the tile's own edge. Half of that zone's pixels lie outside the
        # tile's window; a read clipped to the tile would drop them and report half the sum,
        # with nothing anywhere in the chain to indicate it had.
        #
        # The zone is a 10x10 block straddling the west edge of the tile's own window -- 5
        # columns inside it, 5 outside -- planted BOTH in the crosswalk (which decides which
        # pixels belong to zone 900) and in the vector grid's geometry file (which is what
        # the widening reads to find out how far past the tile the zone reaches). Every
        # pixel holds 1, so `sum` counts pixels: 100 when the whole block was read, 50 when
        # only the part inside the tile was.
        from shapely.geometry import box

        grid = GRIDS[RASTER_GRID]
        transform = from_bounds(*grid.bounds, width=grid.width, height=grid.height)
        tile = TileRequest(*_tile_over_the_raster_grid())
        plain = tile_source_window(raster_grid_geometry(RASTER_GRID), tile.bounds)
        col_off, row_off = int(plain.col_off) - 5, int(plain.row_off) + 2
        inside_the_tile = 5 * 10

        zones = np.zeros((grid.height, grid.width), dtype="int32")
        zones[row_off : row_off + 10, col_off : col_off + 10] = 900
        write_raster_crosswalk(crosswalk_path, RASTER_GRID, "ecoregion_v001", zones)
        write_raster_dataset(
            base_path,
            years=(2030,),
            data_fn=lambda _year: raster_array(fill=1, grid_name=RASTER_GRID, dtype="int32"),
        )
        west, north = transform * (col_off, row_off)
        east, south = transform * (col_off + 10, row_off + 10)
        write_grid_geometry(geometry_path, "ecoregion_v001", [(900, box(west, south, east, north))])

        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input(variable=RASTER_KWARGS["variable"])},
                "steps": [{"op": "project", "input": "a", "to": "ecoregion_v001", "aggregate": "sum"}],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path, tile=tile)
        widened = plan.inputs["a"].read_window

        assert plan.inputs["a"].straddling_vector_grid == "ecoregion_v001"
        assert int(widened.col_off) <= col_off  # the block's west edge is inside the read
        assert int(plain.col_off) > col_off  # ...and was outside the unwidened one

        df = execute(plan, base_path=base_path).collect()

        assert df["grid_cell"].to_list() == [900]
        assert df["value"].to_list() == pytest.approx([100.0])

        # The same walk over the same files with the widening undone -- the control that
        # makes the 100 above mean something. It answers 50, a plausible-looking number no
        # caller could tell apart from a correct one, which is the whole reason the widening
        # is a compile-time decision rather than something the sink could fix up later.
        unwidened = dataclasses.replace(
            plan, inputs={"a": dataclasses.replace(plan.inputs["a"], read_window=plain)}
        )
        df_unwidened = execute(unwidened, base_path=base_path).collect()

        assert df_unwidened["value"].to_list() == pytest.approx([float(inside_the_tile)])
