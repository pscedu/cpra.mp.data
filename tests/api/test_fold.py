"""Tests for the tabular offset-batched fold (api/fold.py).

`fold_group_by` and `should_fold_group_by` take and return plain values and raise
`FoldError` (an ordinary `ValueError`), never `HTTPException` -- these tests exercise the
library layer directly, with no `Dataset`/FastAPI involved, matching `api/expressions.py`'s
"no FastAPI below the translation layer" design constraint (see `test_expressions.py` for
the same pattern applied to that module).
"""

import json
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from cpra.mp.data.analysis.execution.fold import (
    FoldError,
    fold_group_by,
    should_fold_group_by,
)
from cpra.mp.data.analysis.request.aggregates import DECOMPOSABLE_AGGREGATES

AGGREGATES = sorted(DECOMPOSABLE_AGGREGATES)


@pytest.fixture
def scattered_nulls_frame():
    """Four groups, one ("c") entirely null-valued, values otherwise real -- exercises
    every aggregate's null-skipping behavior in the same fixture, including the
    all-null-group edge case Polars itself treats specially (`sum` -> 0.0, `min`/`max`
    -> null, `count` -> 0)."""
    return pl.DataFrame(
        {
            "key": ["a", "a", "a", "b", "b", "c", "c", "c", "c", "d"],
            "value": [1.0, 2.0, None, 5.0, 5.0, None, None, None, None, 10.0],
        }
    )


class TestFoldGroupByMatchesPlainGroupBy:
    """Numeric equality against a single, unfolded `group_by` -- the fold is only useful
    if a caller can never tell, from the result, that it ran in more than one piece."""

    @pytest.mark.parametrize("aggregate", AGGREGATES)
    @pytest.mark.parametrize("step", [1, 3, 100])
    def test_matches_plain_group_by(self, scattered_nulls_frame, aggregate, step):
        """`step=1` forces ten single-row chunks (the maximum possible number of merges
        for this fixture); `step=3` forces uneven chunk boundaries that don't line up
        with group boundaries; `step=100` exceeds the fixture's ten rows, so the whole
        frame lands in one chunk -- the single-chunk degenerate case, which must be exact
        because it IS the plain `group_by`, reached through one extra `.slice(0, 100)`
        that takes everything."""
        expected = (
            scattered_nulls_frame.group_by("key")
            .agg(getattr(pl.col("value"), aggregate)().alias("value"))
            .sort("key")
        )
        actual = fold_group_by(
            lambda: scattered_nulls_frame.lazy(), ["key"], aggregate, step=step
        ).sort("key")
        assert actual.to_dict(as_series=False) == expected.to_dict(as_series=False)

    @pytest.mark.parametrize("aggregate", AGGREGATES)
    def test_all_null_group_matches_plain_group_by(self, scattered_nulls_frame, aggregate):
        """Pinned separately from the parametrized sweep above so a future change to the
        fixture can't accidentally drop the all-null group ("c") without a test noticing."""
        expected_row = (
            scattered_nulls_frame.group_by("key")
            .agg(getattr(pl.col("value"), aggregate)().alias("value"))
            .filter(pl.col("key") == "c")
        )
        actual_row = fold_group_by(
            lambda: scattered_nulls_frame.lazy(), ["key"], aggregate, step=1
        ).filter(pl.col("key") == "c")
        assert actual_row.to_dict(as_series=False) == expected_row.to_dict(as_series=False)


class TestFoldGroupByEmptyChunk:
    def test_a_filter_that_empties_a_middle_chunk_is_not_a_special_case(self):
        """`chunk_transform` (applied after the slice, per the module docstring -- the
        crosswalk reaggregation site's join/scale uses this same hook) filters out an
        entire contiguous row range that lines up exactly with one chunk's slice window,
        so that chunk's partial `group_by` collects zero rows before ever reaching a
        merge. `pl.concat` against that empty, schema-matching partial must be a no-op,
        not a special case needing its own branch.
        """
        df = pl.DataFrame(
            {
                "key": ["x", "y"] * 6,
                "value": [float(i) for i in range(12)],
                # Rows 4-7 (the whole second step=4 chunk) are dropped by chunk_transform.
                "keep": [True] * 4 + [False] * 4 + [True] * 4,
            }
        )
        expected = (
            df.filter(pl.col("keep"))
            .group_by("key")
            .agg(pl.col("value").sum().alias("value"))
            .sort("key")
        )
        actual = fold_group_by(
            lambda: df.lazy(),
            ["key"],
            "sum",
            step=4,
            chunk_transform=lambda chunk: chunk.filter(pl.col("keep")),
        ).sort("key")
        assert actual.to_dict(as_series=False) == expected.to_dict(as_series=False)

    def test_an_empty_source_still_returns_the_correct_empty_schema(self):
        df = pl.DataFrame({"key": [], "value": []}, schema={"key": pl.String, "value": pl.Float64})
        actual = fold_group_by(lambda: df.lazy(), ["key"], "sum")
        assert actual.height == 0
        assert set(actual.columns) == {"key", "value"}


class TestMeanOfMeansTrap:
    def test_mean_is_the_true_weighted_mean_not_a_mean_of_chunk_means(self):
        """Nine rows of 10.0 land in the first `step=9` chunk, one row of 100.0 lands
        alone in the second -- a mean of per-chunk means would average 10.0 and 100.0
        into 55.0, silently wrong the moment two chunks contribute unequal counts for the
        same key. The true mean, carried here as (sum, count) and divided exactly once,
        is (9*10 + 100) / 10 = 19.0.
        """
        df = pl.DataFrame({"key": ["p"] * 10, "value": [10.0] * 9 + [100.0]})
        result = fold_group_by(lambda: df.lazy(), ["key"], "mean", step=9)
        assert result["value"].item() == pytest.approx(19.0)
        assert result["value"].item() != pytest.approx(55.0)


class TestValidateDecomposableRejection:
    def test_non_decomposable_aggregate_is_rejected(self):
        with pytest.raises(FoldError, match="median"):
            fold_group_by(lambda: pl.DataFrame({"key": ["a"], "value": [1.0]}).lazy(), ["key"], "median")

    def test_rejection_happens_before_lf_source_is_ever_called(self):
        """Plan-time rejection, matching `api/expressions.py`'s "reject before parsing"
        discipline -- `lf_source` here raises if called at all, so this test fails loudly
        (an `AssertionError`, not a `FoldError`) if `validate_decomposable` ever moves
        after the first `lf_source()` call.
        """

        def boom():
            raise AssertionError("lf_source must not be called for a non-decomposable aggregate")

        with pytest.raises(FoldError, match="n_unique"):
            fold_group_by(boom, ["key"], "n_unique")

    def test_non_positive_step_is_rejected(self):
        with pytest.raises(FoldError, match="step"):
            fold_group_by(lambda: pl.DataFrame({"key": ["a"], "value": [1.0]}).lazy(), ["key"], "sum", step=0)


class TestShouldFoldGroupBy:
    """Pure-function routing tests -- no Dataset, no Polars schema inspection; see the
    function's own docstring for the exact rule this pins."""

    def test_high_cardinality_key_without_a_filter_folds(self):
        assert should_fold_group_by(["grid_cell"], has_selective_filter=False) is True

    def test_high_cardinality_key_with_a_selective_filter_does_not_fold(self):
        assert should_fold_group_by(["grid_cell"], has_selective_filter=True) is False

    def test_low_cardinality_key_without_a_filter_does_not_fold(self):
        assert should_fold_group_by(["calendar_year"], has_selective_filter=False) is False

    def test_low_cardinality_key_with_a_filter_does_not_fold(self):
        assert should_fold_group_by(["calendar_year"], has_selective_filter=True) is False

    def test_a_non_string_key_is_harmless_alongside_grid_cell(self):
        """`stages.resolve_group_by_columns` no longer produces a non-string key --
        the group_aggregate-without-group_by case is an empty list now, not a `[True]`
        literal sentinel (see its docstring for why: grouping by `True` leaked a
        meaningless "literal" column into every such response) -- but `group_by_keys`
        is a plain `Sequence[Any]` here, so a non-string key from some other future
        caller must still not break the rule: it can never match a high-cardinality
        column name, but its presence in the key list must not stop `grid_cell` (also
        in the list) from tripping it."""
        assert should_fold_group_by([True, "grid_cell"], has_selective_filter=False) is True

    def test_a_custom_high_cardinality_set_is_honored(self):
        assert (
            should_fold_group_by(
                ["zone"], has_selective_filter=False, high_cardinality_columns=frozenset({"zone"})
            )
            is True
        )


# Runs in a subprocess so each measurement gets its own `ru_maxrss` high-water mark
# (a single process's peak only grows, so folded-then-unfolded in one process would
# report the unfolded run's peak for both).
_MEMORY_WORKER = """
import json
import resource
import sys

import polars as pl

from cpra.mp.data.analysis.execution.fold import fold_group_by

mode, path, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
if mode == "folded":
    result = fold_group_by(lambda: pl.scan_parquet(path), ["key"], "sum", step=step)
else:
    result = pl.scan_parquet(path).group_by("key").agg(pl.col("value").sum()).collect(engine="streaming")
result = result.sort("key")
print(json.dumps({
    "peak_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "keys": result["key"].to_list(),
    "values": result["value"].to_list(),
}))
"""


class TestMemoryRegression:
    def test_folded_peak_rss_is_lower_than_unfolded_on_a_high_cardinality_group_by(self, tmp_path):
        """Reproduces, at a size a test suite can afford, the same mechanism measured at
        production scale in `api/fold.py`'s module docstring and `config.TABULAR_FOLD_STEP`
        (~9.7 GB unfolded vs ~1.0 GB folded on the real 1.04B-row damage file,
        `group_by("grid_cell").sum("value")`):
        peak memory in a `group_by` scales with the group key's cardinality, not with the
        size of its hash table. 6M rows over ~126,174 distinct keys (the real
        `clara_grid_cell_v001` cardinality) reliably shows folded peak RSS well under
        unfolded's on this machine (measured directly while writing this test); the
        assertion below only requires a comfortable, noise-tolerant fraction of that gap.

        `preexec_fn=lambda: None` forces Python's classic `fork()`+`exec()` path for each
        subprocess rather than the `posix_spawn`/`vfork`-based fast path `subprocess.run`
        otherwise prefers -- without it, `ru_maxrss` for a child spawned from an already
        memory-heavy pytest process (e.g. after `test_dataset_raster.py`'s rasterio/GDAL
        tests have run first) was observed to read back the PARENT's own high-water mark
        instead of the child's, making the folded and unfolded measurements numerically
        identical and this test spuriously fail depending on run order. `preexec_fn`
        being non-`None` is documented as incompatible with `subprocess`'s fast-path
        optimizations, which is exactly the point here.
        """
        n_rows = 6_000_000
        n_keys = 126_174
        rng = np.random.default_rng(0)
        path = tmp_path / "fold_mem_fixture.parquet"
        pl.DataFrame(
            {
                "key": rng.integers(0, n_keys, size=n_rows).astype("int64"),
                "value": rng.random(n_rows),
            }
        ).write_parquet(path)

        def run(mode, step):
            proc = subprocess.run(
                [sys.executable, "-c", _MEMORY_WORKER, mode, str(path), str(step)],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
                preexec_fn=lambda: None,
            )
            return json.loads(proc.stdout)

        unfolded = run("plain", 0)
        folded = run("folded", 50_000)

        assert folded["peak_kb"] < unfolded["peak_kb"] * 0.9
        assert folded["keys"] == unfolded["keys"]
        # Floating-point summation order differs between one big sum and many partial
        # sums merged back together -- the *answer* must still agree to a tight tolerance,
        # not bit-for-bit (see the mean-of-means trap test for the failure this is NOT).
        diffs = [abs(a - b) for a, b in zip(folded["values"], unfolded["values"])]
        assert max(diffs) < 1e-6
