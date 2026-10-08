"""Tests for the aggregate vocabulary's shared cardinality rule
(`cpra.mp.data.analysis.request.aggregates.validate_count_by_value_cardinality`).

The vocabulary's other rules are exercised at the sites that read them -- `test_fold.py`
for `DECOMPOSABLE_AGGREGATES`/`validate_decomposable`, `test_project_raster.py` for
`validate_project_aggregate`. This one has three readers rather than one (`project`'s
vector-to-vector and zonal `count_by_value` lowerings, and `fold_count_by_value`), so its
message and its boundary are pinned once, here, rather than three times over.
"""

import pytest

from cpra.mp.data.analysis.request.aggregates import validate_count_by_value_cardinality
from cpra.mp.data.analysis.request.errors import AggregateError


class TestValidateCountByValueCardinality:
    def test_a_count_at_the_limit_is_accepted(self):
        """The comparison is strictly greater-than: a column with exactly `max_distinct`
        categories is bounded categorical data, which is what the aggregate is for."""
        assert validate_count_by_value_cardinality(1000, "value", 1000) is None

    def test_a_count_over_the_limit_is_rejected(self):
        with pytest.raises(AggregateError) as excinfo:
            validate_count_by_value_cardinality(1001, "value", 1000)
        assert str(excinfo.value) == (
            "count_by_value would produce approximately 1001 distinct values for "
            "'value', over the limit of 1000 -- count_by_value is meant for bounded "
            "categorical data (e.g. a land-type code), not continuous data"
        )

    def test_the_message_names_the_column_that_was_probed(self):
        """Two of the three copies this rule replaced hardcoded `'value'` in their
        message and the third interpolated the probed column. Both were correct where
        they stood, since the two hardcoding copies also probe a literal `value` column
        -- but only one of the two spellings can survive being shared, and it has to be
        the interpolated one, since `fold_count_by_value` is parameterized by the column
        it probes.
        """
        with pytest.raises(AggregateError, match=r"distinct values for 'ws_value',"):
            validate_count_by_value_cardinality(2, "ws_value", 1)
