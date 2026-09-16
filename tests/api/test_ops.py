"""Tests for the op IR (`cpra.mp.data.analysis.ops`). Every test here builds a `Pipeline`
via `parse_pipeline` (real variables/grids from `cpra/mp/data/structure/`, per
`tests/factories.py`'s established convention -- no fake dimensions) and calls
`ops.compile_ops` directly, never `planner.compile_pipeline`: this module's whole
point is that it needs no catalog/crosswalk/file IO at all, so none of these tests touch
`base_path`, `tmp_path`, or any dataset on disk.

The deliverable these tests pin is that the DECLARED order survives compilation into
`list[Op]` -- previously nothing verified this for the flat `Options`-lowering path
(`planner._compile_single_input_chain`), which silently inverts a `project`'s position
relative to what follows it. `TestDeclaredOrderSurvives` below is the direct test of that
claim, including for exactly the shapes `planner._compile_single_input_chain`'s
SUMMARIZING-PROJECT ORDER GUARD now rejects at compile time (a `calculate`/value-`filter`
declared after a many-to-one `project`) -- reachable here only because `ops.compile_ops`
is a separate function that never calls that guard (see `ops.py`'s own module docstring).
This is not a workaround: it is the point of the op IR, demonstrated directly, and it does
not touch or weaken the guard itself (still exercised, unchanged, by
`tests/api/test_pipeline_exec.py::TestPostProjectOrderingGuard`).
"""

import pytest

from cpra.mp.data.analysis import ops
from cpra.mp.data.analysis.ops import (
    CalculateOp,
    FilterOp,
    GroupOp,
    JoinOp,
    LOWERING_TABLE,
    ProjectOp,
    Shape,
)
from cpra.mp.data.analysis.pipeline import PipelineError, parse_pipeline


def _single_input_pipeline(steps, variable="ssf", grid="ecoregion_v001"):
    return parse_pipeline(
        {
            "inputs": {"eco": {"variable": variable, "time_unit": "annual", "grid": grid}},
            "steps": steps,
            "output": {"format": "csv"},
        }
    )


# The design's own exit-criterion example, verbatim (matches
# `tests/api/test_pipeline.py::NOTEBOOK_QUERY`) -- the paired shape: two inputs, a leading
# `project` (broadcast), a `join`, a post-join `filter`, and a trailing `group`.
PAIRED_PIPELINE = {
    "inputs": {
        "marsh": {
            "variable": "ffibs_coverage_ratio",
            "time_unit": "annual",
            "grid": "veg_grid_cell_v001",
        },
        "salinity": {
            "variable": "sal",
            "time_unit": "annual",
            "grid": "hydro_compartment_v001",
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


class TestOpShapeAndFields:
    """Each op type carries exactly the fields its `Step` counterpart declares, plus
    `step_no`/`output` (see `ops.py`'s module docstring on why every `Op` needs both).
    """

    def test_project_op_fields(self):
        pipeline = _single_input_pipeline(
            [{"op": "project", "input": "eco", "to": "hydro_compartment_v001", "aggregate": "sum"}]
        )
        (op,) = ops.compile_ops(pipeline)
        assert isinstance(op, ProjectOp)
        assert op.step_no == 1
        assert op.input == "eco"
        assert op.to == "hydro_compartment_v001"
        assert op.aggregate == "sum"
        # No `as` given -- the synthesized output name is the step number, a bare digit
        # string that can never collide with a real `as`/input name (see
        # `ops._default_output_name`'s docstring: every legal name must start with a
        # lowercase letter or underscore).
        assert op.output == "1"

    def test_project_op_with_as_uses_it_as_output(self):
        pipeline = _single_input_pipeline(
            [{"op": "project", "input": "eco", "to": "hydro_compartment_v001", "aggregate": "sum", "as": "z"}]
        )
        (op,) = ops.compile_ops(pipeline)
        assert op.output == "z"

    def test_filter_op_fields(self):
        pipeline = _single_input_pipeline([{"op": "filter", "input": "eco", "where": "eco > 1"}])
        (op,) = ops.compile_ops(pipeline)
        assert isinstance(op, FilterOp)
        assert op.input == "eco"
        assert op.where == "eco > 1"

    def test_calculate_op_fields(self):
        pipeline = _single_input_pipeline(
            [
                {"op": "filter", "input": "eco", "where": "eco > 0"},
                {"op": "calculate", "expression": "eco * 2", "where": "eco > 1"},
            ]
        )
        _filter_op, calc_op = ops.compile_ops(pipeline)
        assert isinstance(calc_op, CalculateOp)
        assert calc_op.expression == "eco * 2"
        assert calc_op.where == "eco > 1"
        # Implicit chain: `calculate` has no `input` field at all (see
        # `CalculateStep`'s docstring), so it must resolve to the filter's own output.
        assert calc_op.input == "1"

    def test_group_op_fields(self):
        # `group` has no `input` field at all (see `GroupStep`'s docstring, and contrast
        # `FilterStep`'s own optional `input`) -- this pipeline is rejected by
        # `validate_pipeline`'s chaining rule since `group` cannot be step 1 either way;
        # assert that, then exercise a real chain below.
        with pytest.raises(PipelineError, match="has no input to chain from"):
            parse_pipeline(
                {
                    "inputs": {"eco": {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"}},
                    "steps": [{"op": "group", "by": ["calendar_year"], "aggregate": "mean"}],
                    "output": {"format": "csv"},
                }
            )
        pipeline = _single_input_pipeline(
            [
                {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                {"op": "group", "by": ["calendar_year"], "aggregate": "mean"},
            ]
        )
        _project_op, group_op = ops.compile_ops(pipeline)
        assert isinstance(group_op, GroupOp)
        assert group_op.by == ["calendar_year"]
        assert group_op.aggregate == "mean"
        assert group_op.of == "value"
        assert group_op.input == "1"

    def test_join_op_fields_and_no_implicit_chain(self):
        pipeline = parse_pipeline(PAIRED_PIPELINE)
        project_op, join_op = ops.compile_ops(pipeline)[:2]
        assert isinstance(join_op, JoinOp)
        assert join_op.type == "spatial"
        # `join.inputs` is always explicit (`JoinStep` has no implicit-chain default at
        # all -- see `pipeline._step_explicit_inputs`), and its second entry must be the
        # `project` step's own `as` ("sal"), not a synthesized name -- a `join` can never
        # silently chain from "whatever the previous step happened to be".
        assert join_op.inputs == ["marsh", project_op.output] == ["marsh", "sal"]
        assert join_op.output == "paired"


class TestDeclaredOrderSurvives:
    """Assert the declared step order survives compilation into `list[Op]`, for pipelines
    the flat `Options` lowering gets right AND for ones it silently inverts or currently
    rejects at compile time for exactly that reason
    (`planner._compile_single_input_chain`'s SUMMARIZING-PROJECT ORDER GUARD).
    `ops.compile_ops` has no opinion on any of that -- it only ever preserves
    `pipeline.steps`' own order, because it does not lower onto anything with a fixed
    stage order to invert (see `ops.py`'s module docstring).
    """

    def test_project_filter_calculate_group_preserves_declared_order(self):
        """The ordinary, uncontroversial case, included as a baseline: none of these
        steps trigger the order guard's rejection (an IDENTITY project sets no
        `summarize_by`), so `planner.compile_pipeline` would also execute this in
        declared order -- this test is here for the op-type-sequence assertion the later
        tests all reuse, not because the order question is contested here.
        """
        pipeline = _single_input_pipeline(
            [
                {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                {"op": "filter", "where": "eco > 0"},
                {"op": "calculate", "expression": "eco * 2"},
                {"op": "group", "by": [], "aggregate": "sum"},
            ]
        )
        compiled = ops.compile_ops(pipeline)
        assert [type(op) for op in compiled] == [ProjectOp, FilterOp, CalculateOp, GroupOp]
        assert [op.step_no for op in compiled] == [1, 2, 3, 4]

    def test_project_then_value_filter_preserves_declared_order_despite_compile_time_rejection(self):
        """This is exactly the shape that silently produces the wrong answer under the
        flat `Options` lowering, and is rejected at
        compile time by `planner._compile_single_input_chain`'s SUMMARIZING-PROJECT ORDER
        GUARD (message: "tests the running value after a many-to-one ... project") -- see
        `tests/api/test_pipeline_exec.py::TestPostProjectOrderingGuard::
        test_rejects_value_filter_after_summarizing_project` for that rejection, still in
        place and unweakened. `ops.compile_ops` is a different function that never
        calls that guard, so it compiles this pipeline and preserves PROJECT-THEN-FILTER
        as declared -- the `filter` op's `input` is the `project` op's own `output`, not
        the original `eco` input, proving the filter is wired to run AFTER the
        (many-to-one, `aggregate="sum"`) project, not before it the way the flat
        `Options` lowering actually executes it today.
        """
        pipeline = _single_input_pipeline(
            [
                {"op": "project", "input": "eco", "to": "hydro_compartment_v001", "aggregate": "sum"},
                {"op": "filter", "where": "eco > 2"},
            ]
        )
        project_op, filter_op = ops.compile_ops(pipeline)
        assert isinstance(project_op, ProjectOp)
        assert isinstance(filter_op, FilterOp)
        assert filter_op.input == project_op.output
        assert filter_op.input != "eco"

    def test_project_then_calculate_preserves_declared_order_despite_compile_time_rejection(self):
        """Same shape as the filter case above, for `calculate` -- silently wrong under
        the flat lowering (squared-then-summed instead of the declared summed-then-
        squared) and rejected at compile time by `planner._compile_single_input_chain`
        ("comes after a many-to-one ... project"). `ops.compile_ops` preserves
        PROJECT-THEN-CALCULATE as declared.
        """
        pipeline = _single_input_pipeline(
            [
                {"op": "project", "input": "eco", "to": "hydro_compartment_v001", "aggregate": "sum"},
                {"op": "calculate", "expression": "eco * eco"},
            ]
        )
        project_op, calculate_op = ops.compile_ops(pipeline)
        assert isinstance(project_op, ProjectOp)
        assert isinstance(calculate_op, CalculateOp)
        assert calculate_op.input == project_op.output

    def test_project_then_group_preserves_declared_order_despite_compile_time_rejection(self):
        """Same shape again, for `group` -- also rejected by the SUMMARIZING-PROJECT
        ORDER GUARD ("comes after a many-to-one ... project").
        """
        pipeline = _single_input_pipeline(
            [
                {"op": "project", "input": "eco", "to": "hydro_compartment_v001", "aggregate": "sum"},
                {"op": "group", "by": [], "aggregate": "max"},
            ]
        )
        project_op, group_op = ops.compile_ops(pipeline)
        assert isinstance(project_op, ProjectOp)
        assert isinstance(group_op, GroupOp)
        assert group_op.input == project_op.output

    def test_leading_filter_before_project_preserves_declared_order(self):
        """The other half of what makes the order guard safe to enforce: a value `filter`
        may lead a pipeline, naming its `input` explicitly, exactly like `project.input`
        (`FilterStep.input`; see that class's own docstring for why it exists). Without
        this, rejecting a post-project filter would leave no way to express a filter that
        needs to run before the project at all. `ops.compile_ops` preserves
        FILTER-THEN-PROJECT as declared -- the `project` op's `input` is the filter op's
        own `output`, not the original `eco` input, matching what
        `planner._compile_single_input_chain` also does for this shape today (a leading
        filter lowers onto `Options.where`, emitted first, strictly before the project's
        own summarize aggregation -- see
        `test_pipeline_exec.py::TestPostProjectOrderingGuard::
        test_leading_filter_before_project_compiles_and_executes_declared_order`) -- this
        is the one shape where the IR and the flat lowering already agree, included here
        so the leading-filter shape has its own direct IR-level test rather than only an
        implicit one.
        """
        pipeline = _single_input_pipeline(
            [
                {"op": "filter", "input": "eco", "where": "eco >= 2"},
                {"op": "project", "to": "hydro_compartment_v001", "aggregate": "sum"},
            ]
        )
        filter_op, project_op = ops.compile_ops(pipeline)
        assert isinstance(filter_op, FilterOp)
        assert isinstance(project_op, ProjectOp)
        assert filter_op.input == "eco"
        assert project_op.input == filter_op.output
        assert project_op.input != "eco"

    def test_paired_shape_preserves_project_join_filter_group_order(self):
        """The paired shape (two inputs + `join`): `project` (broadcast), `join`,
        `filter`, `group`, in that order -- matches
        `tests/api/test_pipeline.py::NOTEBOOK_QUERY` and
        `planner._compile_paired_notebook_shape`'s own required opening
        (`steps[0]` must be `project`, `steps[1]` must be `join`), reproduced here as an
        IR-level order assertion rather than a compiled-`PairedAnalysis` one.
        """
        pipeline = parse_pipeline(PAIRED_PIPELINE)
        compiled = ops.compile_ops(pipeline)
        assert [type(op) for op in compiled] == [ProjectOp, JoinOp, FilterOp, GroupOp]
        assert [op.step_no for op in compiled] == [1, 2, 3, 4]
        project_op, join_op, filter_op, group_op = compiled
        assert join_op.inputs == ["marsh", project_op.output]
        assert filter_op.input == join_op.output
        assert group_op.input == filter_op.output

    def test_reordering_filter_and_calculate_changes_the_ir_order_too(self):
        """A stronger order check than "the op types appear somewhere": swap the
        declared order of two adjacent steps and confirm the IR's own op sequence swaps
        with it -- not just their count or types, but which one comes first, and (via
        `input`) which one actually feeds the other.
        """
        filter_first = _single_input_pipeline(
            [
                {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                {"op": "filter", "where": "eco > 0"},
                {"op": "calculate", "expression": "eco * 2"},
            ]
        )
        calc_first = _single_input_pipeline(
            [
                {"op": "project", "input": "eco", "to": "ecoregion_v001"},
                {"op": "calculate", "expression": "eco * 2"},
                {"op": "filter", "where": "eco > 0"},
            ]
        )
        _p1, f1, c1 = ops.compile_ops(filter_first)
        assert isinstance(f1, FilterOp) and isinstance(c1, CalculateOp)
        assert c1.input == f1.output

        _p2, c2, f2 = ops.compile_ops(calc_first)
        assert isinstance(c2, CalculateOp) and isinstance(f2, FilterOp)
        assert f2.input == c2.output


class TestLoweringTable:
    """`LOWERING_TABLE` is data, not code -- these tests check its SHAPE (every op has an
    entry for both `Shape`s, `filter` has both a value and a dimension entry) rather than
    any executed behavior, since this module implements no lowering itself, only the
    table describing it (see `ops.py`'s module docstring).
    """

    def test_every_non_filter_op_has_both_shapes(self):
        for op_name in ("project", "join", "calculate", "group"):
            assert (op_name, Shape.long) in LOWERING_TABLE
            assert (op_name, Shape.band_wide) in LOWERING_TABLE

    def test_filter_has_both_predicate_kinds_and_both_shapes(self):
        for predicate_kind in ("filter (value)", "filter (dimension)"):
            assert (predicate_kind, Shape.long) in LOWERING_TABLE
            assert (predicate_kind, Shape.band_wide) in LOWERING_TABLE

    def test_band_wide_value_filter_is_a_mask_not_a_row_filter(self):
        """Pins the value-vs-dimension filter distinction directly: a value filter on
        band-wide shape must NOT be described the same way as a long-shape row filter,
        since dropping the row would drop that pixel from every band.
        """
        assert LOWERING_TABLE[("filter (value)", Shape.band_wide)] != LOWERING_TABLE[
            ("filter (value)", Shape.long)
        ]
        assert "mask" in LOWERING_TABLE[("filter (value)", Shape.band_wide)]

    def test_band_wide_dimension_filter_is_a_column_selection(self):
        assert "column selection" in LOWERING_TABLE[("filter (dimension)", Shape.band_wide)]

    def test_table_has_no_stray_entries_beyond_the_five_ops(self):
        op_names = {key[0].split(" ")[0] for key in LOWERING_TABLE}
        assert op_names == {"filter", "calculate", "group", "project", "join"}


class TestCompilePipelineRejectsAnUnchainedNonFirstStep:
    """`ops.compile_ops` assumes its input already passed `validate_pipeline`'s
    chaining rule; the guard inside it (`input_name is None`) is unreachable through
    `parse_pipeline` and is exercised here only by constructing a `Pipeline` directly
    (bypassing `parse_pipeline`), matching the module's own docstring on why that guard
    exists at all.
    """

    def test_direct_construction_bypassing_chaining_rule_raises(self):
        from cpra.mp.data.analysis.pipeline import CalculateStep, InputSelection, Output, Pipeline

        pipeline = Pipeline(
            inputs={"eco": InputSelection(variable="ssf", time_unit="annual", grid="ecoregion_v001")},
            steps=[CalculateStep(expression="value * 2")],
            output=Output(format="csv"),
        )
        with pytest.raises(PipelineError, match="has no input to chain from"):
            ops.compile_ops(pipeline)
