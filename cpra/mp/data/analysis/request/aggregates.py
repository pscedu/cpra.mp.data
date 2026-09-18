"""The aggregate vocabulary's decomposability fact and legality rules.

`DECOMPOSABLE_AGGREGATES` lives here, in `request/`, rather than alongside the fold
algorithm that consumes it, because whether an aggregate decomposes into per-chunk
partials is a property of the REQUEST vocabulary (which aggregate name a caller may ask
for and expect a correct offset-batched answer), not of how the fold happens to be
executed -- the same reason `execution/fold.py` may depend on this module but not the
other way around.

The functions below are this module's OWN check-and-raise wrappers over that fact, for
callers inside `request/` (`request/pipeline.py`'s structural validation,
`request/ops.py`'s shape typing) -- they cannot reuse `execution/fold.py`'s
`validate_decomposable`, which raises `execution.fold.FoldError`, because `request/` may
not import from `execution/` (see `tests/api/test_analysis_layering.py`). Both wrappers
read the identical `DECOMPOSABLE_AGGREGATES` set, so there is exactly one place the LIST
of which aggregates fold can drift; only the trivial "not in the set: raise" wrapper is
duplicated, once per layer, each raising that layer's own exception type.

SCOPE NOTE: this module does not yet carry the `mean` (sum, count) accumulator pair or
`count_by_value`'s partial-fold accumulator -- both are execution machinery, not a
vocabulary fact, and so belong with `execution/fold.py`'s own aggregate lowering rather
than here.
"""

from cpra.mp.data.analysis.request.errors import AggregateError

# `sum`, `min`, `max`, and `count` are directly associative: a partial computed the same
# way over any subset of rows merges into the whole with the identical aggregate applied
# again to the partials (`count`'s merge is the one exception spelled out in
# `fold_group_by` -- summing partial counts, not counting them). `mean` is decomposable
# only when carried as a (sum, count) pair and divided once at the very end; a mean of
# per-chunk means silently gives the wrong answer whenever chunks contribute unequal
# counts (a short final chunk, or a filter that empties some chunks unevenly), which is
# also the reason `fold_group_by` never exposes a "mean of means" code path at all.
#
# `count_by_value` is deliberately absent: it changes the output SHAPE (one row per
# (group, distinct value), not one row per group) rather than folding a running scalar,
# so "decomposable" does not even apply to it the way it applies to the other five. It is
# still a legal `project` aggregate -- see `validate_project_aggregate` below, which
# checks for it explicitly rather than by set membership here.
DECOMPOSABLE_AGGREGATES = frozenset({"sum", "min", "max", "count", "mean"})


def validate_decomposable(aggregate: str) -> None:
    """Raise `AggregateError` unless `aggregate` folds cleanly across an offset-batched
    read. The request-layer counterpart of `execution.fold.validate_decomposable` --
    same fact, same rule, different exception type, because `request/` cannot import
    `execution/fold.py`'s (see the module docstring).
    """
    if aggregate not in DECOMPOSABLE_AGGREGATES:
        raise AggregateError(
            f"aggregate {aggregate!r} cannot be computed by an offset-batched fold -- "
            f"only {sorted(DECOMPOSABLE_AGGREGATES)} decompose into per-chunk partials"
        )


def validate_group_aggregate(aggregate: str) -> None:
    """`group.aggregate` must both (a) not be `count_by_value` and (b) actually fold.

    `count_by_value` is rejected outright, before the decomposability check, because the
    reason it cannot be a `group.aggregate` is not that it fails to fold -- it is that it
    produces one row per (group, distinct value) rather than one row per group, which
    would break `group`'s own one-row-per-group contract regardless of foldability.
    `count_by_value` is available on `project` instead, where per-target-cell categorical
    counts are exactly the point (see `validate_project_aggregate`).
    """
    if aggregate == "count_by_value":
        raise AggregateError(
            "count_by_value cannot be a group.aggregate -- it produces one row per "
            "(group, distinct value) rather than one row per group, which would break "
            "group's one-row-per-group contract; it is available on project instead"
        )
    validate_decomposable(aggregate)


def validate_project_aggregate(aggregate: str | None, *, direction_takes_aggregate: bool) -> None:
    """`project.aggregate` legality, which depends on the project's own direction:

    - a vector-to-raster project (`direction_takes_aggregate=False`) broadcasts one
      source value to every target pixel it covers; it never summarizes, so an aggregate
      supplied here is rejected outright, and `None` is required.
    - every other legal direction (vector-to-vector, raster-to-vector) is many-to-one and
      therefore REQUIRES an aggregate -- there is no default, since a target cell may
      draw from any number of source cells. `count_by_value` is accepted here even though
      it is not in `DECOMPOSABLE_AGGREGATES`: it is a ZONAL aggregate, meaningful for a
      many-to-one reaggregation the same way `sum`/`mean`/etc. are, even though it cannot
      be folded across a chunk boundary the way those can (contrast
      `validate_group_aggregate`, which rejects it unconditionally).
    """
    if not direction_takes_aggregate:
        if aggregate is not None:
            raise AggregateError(
                "a vector-to-raster project takes no aggregate -- it broadcasts one "
                "value per target pixel, it does not summarize"
            )
        return
    if aggregate is None:
        raise AggregateError(
            "project requires an aggregate here -- there is no default, since a target "
            "cell may draw from any number of source cells"
        )
    if aggregate == "count_by_value":
        return
    validate_decomposable(aggregate)


def validate_group_by(by: list[str]) -> None:
    """`value` is not a dimension, so it is never a legal member of `group.by` -- a
    group's `by` key must be built from dimension columns (and, after a `count_by_value`,
    the categorical `value_category` dimension it introduces), never the measured
    quantity itself.
    """
    if "value" in by:
        raise AggregateError(
            "'value' is not a dimension and cannot be a member of group.by -- group by "
            "value_category after a count_by_value if categorical grouping is what's "
            "wanted"
        )
