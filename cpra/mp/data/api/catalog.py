"""`GET /v1/catalog.json`'s body: what a decoded request can still be steered by, and what
it would produce.

The route takes the same parameter set as the four data routes and resolves the same
`Pipeline` through the same decoder, so "the catalog for this request" and "the data for
this request" can no longer describe different things -- which they could, and did, when the
catalog was keyed on a `{variable}/{time_unit}/{grid}` path triple that an analysis fixing
its own inputs had no way to state. `lnd_type_plus` had no catalog route at all; the portal
covered for it by inventing a plausible triple (`CANONICAL_DERIVED_DATASETS`) and keeping it
in agreement with what the analysis actually does, by hand.

Two halves, both derived from the one decode:

- `choices`/`results`/`exact` -- `catalog.CatalogQuery` over the pipeline's INPUTS, strictly
  intersected, minus whatever the analysis collapsed. This is the half about what the caller
  may still choose.
- `output` -- the output descriptor plus the analysis's presentation overrides. This is the
  half about what they would get, and it is the reason the route can serve an analysis at
  all: the grid a `lnd_type_plus` result is addressed against is a fact about the resolved
  pipeline (`morph_pixel_v002`) and about none of its three inputs.
"""

from cpra.mp.data.analyses.base import ResolvedAnalysis, normalized_filter_values
from cpra.mp.data.analysis.request.descriptor import OutputDescriptor
from cpra.mp.data.analysis.request.unified import SELECTION_OPTIONS
from cpra.mp.data.api.models import CatalogOutput
from cpra.mp.data.api.unified import UnifiedRequest
from cpra.mp.data.catalog import CatalogQuery


def catalog_filters(filters: dict) -> dict[str, list]:
    """The decoder's raw selection, in the form `CatalogQuery` compares against.

    Two differences from what `partition.filters` holds. `grid_cell` and `where` are dropped:
    they narrow ROWS, and the side-table has one row per dimension combination x calendar run
    with no grid cells and no values in it, so neither can change which dimension values
    exist. And each remaining value list is cast to its dimension's own dtype -- a query
    string's values are always strings (see `pipeline.pipeline_from_query`), while the
    catalog's columns are typed, so `scenario_id=7` has to become the integer 7 before
    `_update` can find it among the choices.
    """
    return {
        name: normalized_filter_values(name, values)
        for name, values in filters.items()
        if name not in SELECTION_OPTIONS
    }


def output_body(descriptor: OutputDescriptor, analysis: ResolvedAnalysis | None) -> CatalogOutput:
    """The output descriptor, with the resolved analysis's `Meta` overlaid.

    The precedence is `visualizations.map`'s, written once: an analysis that declares a
    `legend`/`label`/`result_kind` has said what its result IS, and that wins over anything
    derivable from the pipeline's inputs -- `lnd_type_plus` outputs land-type codes, not the
    elevation its first input reads. A request with no analysis falls back to its one
    presentation variable, and to `"value"` for the kind, which is what a plain dataset read
    always was.

    Takes the two pieces rather than a `UnifiedRequest` because `/v1/analyses.json` has no
    request to take them from: it resolves each analysis with nothing selected purely to ask
    what it produces (`api/__init__.py::analyses_catalog`). `catalog_output` below is the
    request-shaped caller, and the two must not answer differently -- the portal builds its
    variable picker from the catalog's copy and then draws the map from this route's.
    """
    variable = descriptor.variable

    return CatalogOutput(
        grid=descriptor.grid,
        grid_type=descriptor.grid_type,
        shape=descriptor.shape,
        time_unit=descriptor.time_unit,
        time_column=descriptor.time_column,
        value_columns=list(descriptor.value_columns),
        variables=[None if var is None else var.name for var in descriptor.variables],
        folded_into=descriptor.folded_into,
        label=(analysis.meta.label if analysis is not None else (variable.label if variable is not None else None)),
        legend=(
            (analysis.legend if analysis is not None else None)
            or (variable.legend.name if variable is not None and variable.legend else None)
        ),
        result_kind=analysis.result_kind if analysis is not None else "value",
        collapsed_dimensions=sorted(analysis.collapsed_dimensions if analysis is not None else ()),
    )


def catalog_output(decoded: UnifiedRequest) -> CatalogOutput:
    """`output_body` for a decoded request -- the descriptor the decoder already computed
    once (`UnifiedRequest.descriptor`) and the analysis it resolved, if any."""
    return output_body(decoded.descriptor, decoded.analysis)


def catalog_body(decoded: UnifiedRequest) -> dict:
    """The whole response body. Raises `ValueError` from `CatalogQuery` when the request
    names a dataset nothing has been written for, or when the inputs share no value for some
    dimension -- the route translates both into a 404.
    """
    output = catalog_output(decoded)
    query = CatalogQuery(
        decoded.pipeline.inputs.values(),
        catalog_filters(decoded.partition.filters),
        omit=output.collapsed_dimensions,
    )
    return {**query.execute(), "output": output.model_dump(mode="json")}
