"""The one declarative fact `execution/fold.py` needs about aggregate decomposability.

`DECOMPOSABLE_AGGREGATES` lives here, in `request/`, rather than alongside the fold
algorithm that consumes it, because whether an aggregate decomposes into per-chunk
partials is a property of the REQUEST vocabulary (which aggregate names a caller may ask
for and expect a correct offset-batched answer), not of how the fold happens to be
executed -- the same reason `execution/fold.py` may depend on this module but not the
other way around.
"""

# `sum`, `min`, `max`, and `count` are directly associative: a partial computed the same
# way over any subset of rows merges into the whole with the identical aggregate applied
# again to the partials (`count`'s merge is the one exception spelled out in
# `fold_group_by` -- summing partial counts, not counting them). `mean` is decomposable
# only when carried as a (sum, count) pair and divided once at the very end; a mean of
# per-chunk means silently gives the wrong answer whenever chunks contribute unequal
# counts (a short final chunk, or a filter that empties some chunks unevenly), which is
# also the reason `fold_group_by` never exposes a "mean of means" code path at all.
DECOMPOSABLE_AGGREGATES = frozenset({"sum", "min", "max", "count", "mean"})
