"""Tests for `cpra.mp.data.analysis.execution.execute` -- the walk.

Three concerns, in three test classes:

- `TestLoweringTableAgreement`: the bidirectional coverage check the module docstring
  promises -- every `(op, shape)` combination `request.ops`'s lowering tables (or, for
  `calculate`/`group`, `Shape` itself) declare legal is covered by EXACTLY one of
  `execute.py`'s registered executors or its `NOT_YET_IMPLEMENTED` set, never both, never
  neither.
- `TestLoweringNotImplemented`: the walk's failure path for a combination
  `NOT_YET_IMPLEMENTED` names -- band-wide anything, and `project` at every shape.
- `TestEndToEnd`: the resume contract itself -- a JSON pipeline over a real fixture
  dataset, taken through `parse_pipeline` -> `validate_shapes` -> `compile_plan` ->
  `execute` -> `.collect()`, including a declared-order test proving that reordering two
  steps changes the answer.
"""

import polars as pl
import pytest

from cpra.mp.data.analysis.execution.execute import (
    NOT_YET_IMPLEMENTED,
    LoweringNotImplementedError,
    _LOWERINGS,
    execute,
)
from cpra.mp.data.analysis.execution.plan import compile_plan
from cpra.mp.data.analysis.request.ops import (
    FILTER_LOWERING_TABLE,
    FrameType,
    JOIN_LOWERING_TABLE,
    PROJECT_LOWERING_TABLE,
    Shape,
    validate_shapes,
)
from cpra.mp.data.analysis.request.pipeline import parse_pipeline
from cpra.mp.data.grids import GRIDS
from tests.factories import ANNUAL_KWARGS, write_annual_dataset, write_omar_dataset

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

    def test_filter_shapes_are_fully_partitioned(self):
        domain = {shape for (_kind, shape) in FILTER_LOWERING_TABLE}
        registered = self._registered("filter")
        not_yet = NOT_YET_IMPLEMENTED["filter"]
        assert registered | not_yet == domain
        assert registered.isdisjoint(not_yet)

    def test_join_shapes_are_fully_partitioned(self):
        domain = set(JOIN_LOWERING_TABLE)
        registered = self._registered("join")
        not_yet = NOT_YET_IMPLEMENTED["join"]
        assert registered | not_yet == domain
        assert registered.isdisjoint(not_yet)

    def test_calculate_shapes_are_fully_partitioned(self):
        # `calculate` has no lowering table of its own (`_type_calculate` types it
        # unconditionally shape-preserving) -- the declared domain is `Shape` itself.
        domain = set(Shape)
        registered = self._registered("calculate")
        not_yet = NOT_YET_IMPLEMENTED["calculate"]
        assert registered | not_yet == domain
        assert registered.isdisjoint(not_yet)

    def test_group_shapes_are_fully_partitioned(self):
        domain = set(Shape)
        registered = self._registered("group")
        not_yet = NOT_YET_IMPLEMENTED["group"]
        assert registered | not_yet == domain
        assert registered.isdisjoint(not_yet)

    def test_project_directions_are_fully_partitioned(self):
        # `project` is keyed by direction, not shape -- it is the op the incoming shape
        # cannot select a lowering for. No direction is registered in this vertical slice,
        # so every `PROJECT_LOWERING_TABLE` pair must be named not-yet-implemented, and
        # nothing beyond the table may appear there.
        domain = set(PROJECT_LOWERING_TABLE)
        registered = self._registered("project")
        not_yet = NOT_YET_IMPLEMENTED["project"]
        assert registered | not_yet == domain
        assert registered.isdisjoint(not_yet)

    def test_every_registered_lowering_is_named_by_a_table(self):
        # The other direction, across all five ops at once: no entry may sit in
        # `_LOWERINGS` under an op kind the request layer does not type, or under a key
        # that op's own declared domain does not contain.
        domains = {
            "filter": {shape for (_kind, shape) in FILTER_LOWERING_TABLE},
            "join": set(JOIN_LOWERING_TABLE),
            "calculate": set(Shape),
            "group": set(Shape),
            "project": set(PROJECT_LOWERING_TABLE),
        }
        assert set(NOT_YET_IMPLEMENTED) == set(domains)
        for kind, key in _LOWERINGS:
            assert kind in domains, f"{kind!r} is not an op the request layer types"
            assert key in domains[kind], f"{kind!r} has no declared lowering for {key!r}"

    def test_named_not_yet_implemented_set_is_exactly_band_wide_plus_project(self):
        # A concrete pin on the "deliberate, named" set the brief asks for -- this test
        # fails the moment a later stage adds a real band-wide or project lowering,
        # forcing that stage to shrink this dict rather than leaving it stale.
        assert NOT_YET_IMPLEMENTED["filter"] == frozenset({Shape.band_wide})
        assert NOT_YET_IMPLEMENTED["calculate"] == frozenset({Shape.band_wide})
        assert NOT_YET_IMPLEMENTED["group"] == frozenset({Shape.band_wide})
        assert NOT_YET_IMPLEMENTED["join"] == frozenset({Shape.band_wide})
        assert NOT_YET_IMPLEMENTED["project"] == frozenset(
            {("vector", "vector"), ("raster", "vector"), ("vector", "raster")}
        )


class TestLoweringNotImplemented:
    """The walk's failure path: an `(op, shape)` combination `request/ops.py`'s shape
    typing already accepts (so the pipeline validates fine) but for which no executor
    exists yet. None of these need real raster data on disk -- shape typing runs from grid
    KIND alone (`GRIDS[name].grid_type`), and the walk raises before ever needing to open
    the band-wide input's frame (see `execute`'s own docstring: a band-wide input is never
    scanned up front).
    """

    def test_band_wide_filter_raises(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "filter", "input": "a", "where": "value > 1"}],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline)
        with pytest.raises(LoweringNotImplementedError):
            execute(plan)

    def test_band_wide_calculate_raises(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "calculate", "input": "a", "expression": "value * 2"}],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline)
        with pytest.raises(LoweringNotImplementedError):
            execute(plan)

    def test_band_wide_group_raises(self):
        # `group` never has an explicit `input` field (it always chains from a preceding
        # step -- `GroupStep`'s own docstring: "can never be a pipeline's first step"),
        # and every OTHER op is equally unimplemented for band-wide, so there is no legal
        # pipeline JSON that reaches a bare band-wide `group` first. Building the
        # `ExecutionPlan` directly isolates exactly the one thing this test checks --
        # `group`'s own not-yet-implemented gate -- from that chaining requirement.
        from cpra.mp.data.analysis.execution.plan import ExecutionPlan, ResolvedInput
        from cpra.mp.data.analysis.request.ops import GroupOp
        from cpra.mp.data.analysis.request.pipeline import InputSelection

        selection = InputSelection(**raster_input())
        plan = ExecutionPlan(
            inputs={"a": ResolvedInput(selection=selection)},
            ops=[GroupOp(step_no=1, input="a", by=[], aggregate="sum", output="g")],
            frame_types=frame_types(RASTER_GRID, a=Shape.band_wide, g=Shape.band_wide),
        )
        with pytest.raises(LoweringNotImplementedError):
            execute(plan)

    def test_band_wide_join_raises(self):
        from cpra.mp.data.analysis.execution.plan import ExecutionPlan, ResolvedInput
        from cpra.mp.data.analysis.request.ops import JoinOp
        from cpra.mp.data.analysis.request.pipeline import InputSelection

        selection_a = InputSelection(**raster_input())
        selection_b = InputSelection(**raster_input(grid=RASTER_GRID, variable="veg_area"))
        plan = ExecutionPlan(
            inputs={"a": ResolvedInput(selection=selection_a), "b": ResolvedInput(selection=selection_b)},
            ops=[JoinOp(step_no=1, type="attribute", inputs=["a", "b"], output="j")],
            frame_types=frame_types(RASTER_GRID, a=Shape.band_wide, b=Shape.band_wide, j=Shape.band_wide),
        )
        with pytest.raises(LoweringNotImplementedError):
            execute(plan)

    def test_project_raises_regardless_of_direction(self):
        pipeline = _pipeline(
            {
                "inputs": {"a": vector_input()},
                "steps": [{"op": "project", "input": "a", "to": VECTOR_GRID, "aggregate": "sum"}],
                "output": {"format": "csv"},
            }
        )
        plan = compile_plan(pipeline)
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
            [
                # `group` can never be a pipeline's first step (`GroupStep`'s own
                # docstring -- it has no field to be explicit with); this pass-through
                # filter (every fixture value is well above -1) exists only to give it
                # something to chain from.
                {"op": "filter", "input": "a", "where": "value > -1", "as": "f"},
                {"op": "group", "by": [], "aggregate": "sum"},
            ],
        )
        # One row per grid_cell (32 cells), summed over the two years:
        # (2030 + g/100) + (2040 + g/100) = 4070 + 2*g/100, for g in 1..32.
        assert df.height == 32
        expected = sorted(4070 + 2 * g / 100 for g in range(1, 33))
        assert sorted(df["value"].to_list()) == pytest.approx(expected, abs=1e-3)

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
