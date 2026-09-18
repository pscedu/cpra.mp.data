"""Tests for `cpra.mp.data.analysis.io.vector.scan_vector` -- the vector scan this
package's execution layer opens a long dataset through.

Real structure -- `ssf` on `ecoregion_v001` -- via `tests/factories.py`'s
`write_annual_dataset`, matching `test_request_ops.py`'s own convention of using genuine
grids/variables rather than fabricated ones. Every fixture writes through `base_path`
(`tmp_path / "data"`, never a real data directory -- see `tests/conftest.py`).
"""

import polars as pl
import pytest

from cpra.mp.data.analysis.io.vector import scan_vector
from cpra.mp.data.analysis.request.pipeline import InputSelection
from tests.factories import write_annual_dataset

SELECTION = InputSelection(variable="ssf", time_unit="annual", grid="ecoregion_v001")


@pytest.fixture
def annual_dataset(base_path):
    write_annual_dataset(base_path, years=(2030, 2040))
    return base_path


class TestScanShape:
    def test_returns_one_row_per_observation(self, annual_dataset):
        # annual_frame's default geo_ids is ECOREGION_IDS (32 cells); two years * 32
        # cells is 64 observations -- one row each, the long-shape contract.
        df = scan_vector(SELECTION, base_path=annual_dataset).collect()
        assert df.height == 64

    def test_columns_are_dimensions_plus_grid_cell_plus_value(self, annual_dataset):
        df = scan_vector(SELECTION, base_path=annual_dataset).collect()
        # Every dimension `ssf`/`annual`/`ecoregion_v001` requires (dataset dims plus the
        # metadata `write_annual_dataset` gave it), plus the two long-shape columns.
        assert set(df.columns) == {
            "variable",
            "time_unit",
            "grid",
            "model_group_id",
            "scenario_id",
            "percentile",
            "calendar_year",
            "model",
            "grid_cell",
            "value",
        }

    def test_values_match_the_fixture_formula(self, annual_dataset):
        # annual_frame's default value_fn is `year + geo_id / 100` -- hand-computed, not
        # captured from a run.
        df = scan_vector(SELECTION, base_path=annual_dataset).collect()
        row = df.filter(pl.col("calendar_year") == 2030, pl.col("grid_cell") == 5)
        assert row["value"].item() == pytest.approx(2030 + 5 / 100)

    def test_dataset_dimensions_come_back_as_the_selection_named(self, annual_dataset):
        df = scan_vector(SELECTION, base_path=annual_dataset).collect()
        assert set(df["variable"].unique().to_list()) == {"ssf"}
        assert set(df["time_unit"].unique().to_list()) == {"annual"}
        assert set(df["grid"].unique().to_list()) == {"ecoregion_v001"}


class TestScanDoesNotFilterToOneRun:
    """`InputSelection` names a dataset; it does not filter one down to a single model
    run (see that model's own docstring, and `scan_vector`'s module docstring, which this
    test exercises directly rather than trusting the docstring's claim). Writing two
    different `scenario_id`s to the SAME `variable`/`time_unit`/`grid` and reading with
    only the three `InputSelection` fields must return rows for both -- narrowing to one
    scenario is a `filter` step's job, applied later in the walk, never the scan's.
    """

    def test_scanning_returns_every_scenario_present_on_disk(self, base_path):
        write_annual_dataset(base_path, years=(2030, 2040), scenario_id=7)
        write_annual_dataset(base_path, years=(2030, 2040), scenario_id=8)

        df = scan_vector(SELECTION, base_path=base_path).collect()

        assert df.height == 128  # 64 rows per scenario, both scenarios present
        assert set(df["scenario_id"].unique().to_list()) == {7, 8}


class TestPushdownSeam:
    """`columns`/`predicate` are the seam `execution/plan.py`'s `ResolvedInput` fills with
    its pushdown decision. Which decision that is belongs to the compiler and is pinned
    there; what these check is the other half -- that the scan applies whatever it is
    handed, exercised directly rather than through a compiled plan.
    """

    def test_predicate_narrows_rows(self, annual_dataset):
        df = scan_vector(
            SELECTION, base_path=annual_dataset, predicate=pl.col("calendar_year") == 2030
        ).collect()
        assert df.height == 32
        assert set(df["calendar_year"].unique().to_list()) == {2030}

    def test_columns_narrows_the_schema(self, annual_dataset):
        df = scan_vector(SELECTION, base_path=annual_dataset, columns=["grid_cell", "value"]).collect()
        assert list(df.columns) == ["grid_cell", "value"]

    def test_columns_and_predicate_compose(self, annual_dataset):
        df = scan_vector(
            SELECTION,
            base_path=annual_dataset,
            columns=["grid_cell", "value"],
            predicate=pl.col("calendar_year") == 2030,
        ).collect()
        assert list(df.columns) == ["grid_cell", "value"]
        assert df.height == 32
