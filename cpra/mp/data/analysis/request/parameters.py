"""The pydantic models a caller's flat parameters arrive as, built structurally off the
live `cpra.mp.data.dimensions` table rather than typed out by hand: `DatasetDimensions`
(which dataset), `FilterDimensions` (which slice of it), and `DataOptions` (what to do with
the result). `api/models.py` subclasses these for the HTTP query surface, and
`request/data_query.py` turns the triple into a `request.pipeline.Pipeline`.

`DataOptions` is the WHOLE options vocabulary a flat request has. It used to have a
superset, `Options`, carrying the fields that described how a request executed
(`group_by`/`group_aggregate`/`summarize_*`/`calculate`); every one of those is now a
pipeline step instead, `api.__init__.reject_retired_params` 400s the flat spellings with
the step to use, and the model itself is gone rather than kept as a shape nothing reads.

These are a sibling of `request/types.py`, not a layer above it: that module owns the
closed enum vocabularies (the grid/variable/time-unit types, `AggregateFunction`) and this
one owns the field sets those vocabularies type.
"""

from pydantic import BaseModel, ConfigDict, create_model

from cpra.mp.data.analysis.request.types import dimension_type
from cpra.mp.data.dimensions import DATASET_DIMENSIONS, DIMENSIONS

BaseDatasetDimensions = create_model(
    "BaseDatasetDimensions",
    __config__=ConfigDict(protected_namespaces=()),
    **{dim.name: (dimension_type(dim.name), None) for dim in DATASET_DIMENSIONS.values()},
)

BaseFilterDimensions = create_model(
    "BaseFilterDimensions",
    __config__=ConfigDict(protected_namespaces=()),
    **{
        dim.name: (
            list[dimension_type(dim.name)] | None,
            None,
        )
        for dim in DIMENSIONS.values()
        if dim.type not in ("dataset", "metadata")
    },
)


class DatasetDimensions(BaseDatasetDimensions):
    model_config = ConfigDict(use_enum_values=True)


class FilterDimensions(BaseFilterDimensions):
    model_config = ConfigDict(use_enum_values=True)


class DataOptions(BaseModel):
    """Plain data-access options: every field the HTTP data/tile endpoints
    (`api.models.TileQuery`/`DataQuery`) accept, and the whole of what a flat request may
    say about what to do with its result. Anything that TRANSFORMS the data is a pipeline
    step now -- see this module's docstring.
    """

    model_config = ConfigDict(use_enum_values=True)

    grid_cell: str | None = None
    # Accepts both DIMENSION conditions (against the raster metadata table's path/band/
    # dimensions, or a vector frame's dimension columns -- historically the only kind
    # `where` accepted) and VALUE conditions (against actual pixel/cell VALUES, written
    # against the placeholder column `value` -- historically the separate, now-retired
    # `value_where` parameter), AND-ed together in one string, e.g.
    # `calendar_year > 2040 AND value > 3`. `request.expressions.classify_where` splits
    # `where` on top-level `AND` and sorts each conjunct by which kind it is: a dimension
    # conjunct is handed to `lf.sql()` largely unrestricted, the same as always; a value
    # conjunct is parsed through `request.expressions.parse_value_expr`'s restricted
    # whitelist instead, since it runs on the hot per-row path an aggregate or a subquery
    # inside it can force off of streaming pushdown entirely -- see that module's own
    # docstring. A single AND-term may not reference both kinds (rejected with an
    # error explaining the rule); write dimension and value conditions as separate AND-ed
    # conjuncts instead.
    #
    # What a failing value conjunct does depends on the frame's SHAPE, not on anything
    # else in the request: BAND-WIDE, the pixel is masked to null in that band alone
    # (`execution.band_ops.mask_value_band_wide`), because dropping its row would drop it
    # from every other band too, including the ones the predicate holds for. LONG, the row
    # is dropped (`execution.long_ops.filter_long`), since a long row carries exactly one
    # measurement and there is no sibling to protect. Masking is also what composes with a
    # downstream aggregate's null-skipping -- including `count`, which is what makes
    # "count pixels where value > x" work.
    where: str | None = None
    sort: str | None = None
    select: str | None = None
    color_map: str | None = None
