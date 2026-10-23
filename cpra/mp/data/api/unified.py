"""The unified decoder's orchestrator: one request, one `Pipeline`, for every consolidated
route.

`analysis.request.unified` owns the pure half -- the five-category partition of the query
string and the pipeline-assembly primitives. This module owns the three things that half
structurally cannot do, and runs the six steps in order.

**Why the split is here and not one module.** `request/` may import neither `io/` nor
`analyses/`:

- resolving a coordinate-valued `grid_cell` reads real grid geometry, which is `io/`. That
  is the same reason `request.data_query.build_data_query_pipeline` takes an
  already-resolved `grid_cell_where` and `api.data_query.build_data_query` is what resolves
  it -- this module is the same split one route-generation later;
- `analyses/` (the named-analysis registry) imports `request/` for `Pipeline` and
  `append_steps`, so `request/` cannot import it back. `resolve_analysis_chain` and
  `selection_filter_where` therefore have to be called from above both, which is here.

The plan's §7 put both in `request/unified.py`; it names only `io/` as the reason for a
split. Extending it one notch to `analyses/` follows the precedent the plan itself cites
and is what the layering test (`tests/api/test_analysis_layering.py`) enforces.

**Where the caller's selection lands, and why it is not always a trailing step.** The
selection -- dimension filters plus `grid_cell`/`where` -- is composed into one `where`
string by `analyses.base.selection_filter_where`, once, for every route. Where that string
goes depends on whether an analysis was named:

- **No analysis** (the `download` sentinel, or a caller's own dotted pipeline): there is no
  pipeline to place it inside, so it is one trailing `filter` step
  (`request.unified.append_selection`). Trailing costs nothing at the read --
  `execution.plan`'s two dimension passes push a dimension conjunct declared anywhere down
  to every origin.
- **An analysis was named**: the selection is handed to `resolve_analysis_chain` as
  `filters`, and the analysis places it. This is not a concession; it is the only correct
  placement, and `diff` is the proof. `diff`'s whole subject is two selected values of one
  dimension: it filters to them, negates one, and sums across the dimension -- which
  removes that dimension from the frame. The same conjunct appended at the tail would name
  a column that no longer exists. `lnd_type_plus` likewise places its selection between its
  `join` and its `case`, because the `case` collapses the pairing the dimension filter
  needs standing. Which step a selection belongs on is a fact about the pipeline's shape,
  and the registry is what knows the shape.

**`grid_cell` needs the output grid, so a coordinate selection resolves in two passes.**
What a `grid_cell` value MEANS -- polygon ids or `"lng,lat"` pairs -- depends on the grid
the result is addressed against, and for an analysis that grid is a fact about the resolved
pipeline (`lnd_type_plus` projects onto `morph_pixel_v002`, which none of its inputs is
on). So the pipeline is resolved once with a STAND-IN spatial conjunct
(`PROBE_GRID_CELL_WHERE`) purely to read `output_descriptor(...).grid` off it, `grid_cell`
is resolved against that, and the pipeline is resolved again with the answer. The stand-in
rather than nothing at all because a spatial conjunct decides a frame's SHAPE and not just
its rows -- see that constant. The extra pass opens no file --
`annotate_frame_types` works off declared ops alone -- and is skipped entirely when the
request sends no `grid_cell`, which is the common case.
"""

import dataclasses
from collections.abc import Iterable

from cpra.mp.data.analyses import DatasetRef, ResolvedAnalysis, resolve_analysis_chain
from cpra.mp.data.analyses.base import dataset_output_format, selection_filter_where
from cpra.mp.data.analysis.request.descriptor import OutputDescriptor, output_descriptor
from cpra.mp.data.analysis.request.errors import RequestError
from cpra.mp.data.analysis.request.ops import validate_shapes
from cpra.mp.data.analysis.request.parameters import Compression
from cpra.mp.data.analysis.request.pipeline import Pipeline
from cpra.mp.data.analysis.request.unified import (
    DATASET_SUGAR,
    QueryPartition,
    append_selection,
    build_pipeline,
    download_pipeline,
    partition_query,
)
from cpra.mp.data.api.data_query import resolve_grid_cell, resolve_grid_cell_sync


@dataclasses.dataclass(frozen=True)
class UnifiedRequest:
    """One decoded request: the `Pipeline` to run, plus the four things a route needs that
    the `Pipeline` itself does not record.

    `analysis` is the `ResolvedAnalysis` when one was named and `None` under the `download`
    sentinel -- carried because its `meta`/`legend`/`result_kind`/`collapsed_dimensions`
    are presentation facts that OVERRIDE what `descriptor` derives, and only the registry
    has them.

    `relabel` is `io.geography.resolve_raster_point_labels`'s absolute-id -> coordinate-
    string map for a raster point query and `None` for everything else, exactly as on
    `api.data_query.ResolvedDataQuery`: the pipeline's `where` names integer ids because
    that is what the executor filters on, and mapping them back to the caller's own
    coordinate strings is a property of the RESULT.

    `compression` is a transport concern that never rides inside the `Pipeline`; a route
    still calls `api.utils.validate_compression` against its own format, which draws a
    different allowed-algorithm line per format and so cannot be decided here.

    `descriptor` is computed once and handed on rather than recomputed per consumer -- the
    catalog route, the visualization render context and `float_precision` all want it, and
    it is already paid for by the `grid_cell` pass above.
    """

    pipeline: Pipeline
    partition: QueryPartition
    analysis: ResolvedAnalysis | None
    descriptor: OutputDescriptor
    compression: Compression
    relabel: dict[int, list[str]] | None

    @property
    def float_precision(self) -> int | None:
        """The result's CSV rounding, from its one presentation variable -- `None` when the
        pipeline carries several value columns or none resolved, which is the same "no
        opinion" `api.pipeline.PipelineTabularSource` already takes for a general pipeline.
        """
        variable = self.descriptor.variable
        return None if variable is None else variable.float_precision


def _dataset_ref(partition: QueryPartition) -> DatasetRef | None:
    """The `variable`/`time_unit`/`grid` triple as a `DatasetRef`, or `None` when the
    caller named no dataset at all.

    A PARTIAL triple is a 400 here rather than further down. `resolve_analysis_chain` reads
    `dataset is None` as "this must be a `type='custom'` analysis" and would answer a
    caller who merely misspelled `time_unit` with a type-mismatch message about a
    distinction they never asked about.
    """
    if not partition.dataset:
        return None
    missing = sorted(DATASET_SUGAR - set(partition.dataset))
    if missing:
        raise RequestError(
            f"incomplete dataset selection: missing {missing} -- a dataset is named by all "
            f"of {sorted(DATASET_SUGAR)} together"
        )
    return DatasetRef(**partition.dataset)


def _assemble(
    partition: QueryPartition,
    format: str | None,
    filters: dict,
    grid_cell_where: str | None = None,
) -> tuple[Pipeline, ResolvedAnalysis | None]:
    """Steps 2-5 of the decoder, as one function so the `grid_cell` probe pass and the real
    pass cannot drift: resolve the analysis (or the sentinel), merge the caller's dotted
    parameters and output options over it, and append the selection where no analysis took
    it.

    `filters` is passed in rather than read off `partition` because the probe pass runs
    with `grid_cell` REMOVED: an unresolved coordinate value reaching
    `request.data_query.grid_cell_conjunct` is a hard rejection (it parses digit ids only,
    by design -- see its docstring), and the probe exists precisely to learn which grid to
    resolve those coordinates against. The probe passes `PROBE_GRID_CELL_WHERE` in its
    place, so what it drops is only WHICH pixel was asked for, never the fact that one was.
    """
    dataset = _dataset_ref(partition)

    if partition.is_download:
        # `download` plus a dataset is that dataset as this request's one input; `download`
        # plus dotted `input.*` parameters is the caller's own pipeline and needs no base
        # at all. Neither means there is nothing to read -- so the empty `Pipeline` the
        # sentinel names is rejected here, before `Pipeline`'s own `min_length=1` would
        # report it as a field error about a model the caller never mentioned.
        if dataset is None and not partition.pipeline_params:
            raise RequestError(
                "no dataset selected -- name one with variable=/time_unit=/grid=, give an "
                "analysis= (see /v1/analyses.json), or write the inputs directly as "
                "input.<name>.variable= etc."
            )
        # `format or dataset_output_format(dataset)`: with no URL extension to take a
        # format from (`/v1/analyses/{analysis}.json`), a plain download's sink is the one
        # its own grid implies -- `tif` for a raster grid, `csv` for a vector one -- which
        # is the same rule every dataset-scoped analysis applies to its own output. A flat
        # `csv` there is a pipeline that cannot run: a raster dataset's final frame is
        # band-wide, and `validate_shapes` rejects it against a tabular sink.
        base = (
            None
            if dataset is None
            else download_pipeline(partition.dataset, format or dataset_output_format(dataset))
        )
        analysis = None
    else:
        analysis = resolve_analysis_chain(
            list(partition.analysis),
            dataset=dataset,
            filters=filters,
            params=partition.params,
            grid_cell_where=grid_cell_where,
        )
        base = analysis.pipeline

    pipeline = build_pipeline(
        base, partition.pipeline_params, partition.output, format=format
    )

    if analysis is None:
        pipeline = append_selection(
            pipeline, selection_filter_where(filters, grid_cell_where=grid_cell_where)
        )

    return pipeline, analysis


def _finish(
    partition: QueryPartition,
    pipeline: Pipeline,
    analysis: ResolvedAnalysis | None,
    relabel: dict[int, list[str]] | None,
    validate_sink: bool = True,
) -> UnifiedRequest:
    """Step 6 and the packaging: shape-check the final pipeline, then describe it.

    `validate_shapes` runs before `output_descriptor` deliberately. Both walk the same
    annotated frames, but only `validate_shapes` runs the sink rules -- so a `grid_cell`
    selection asked for as a `tif` is reported as the targeted rejection it has
    (`request/ops.py`'s spatial-lowering message) rather than surfacing later as a
    descriptor whose shape a renderer cannot use.

    `validate_sink=False` is the visualization route, and skips ONLY those sink rules: a
    spec is not a response and writes nothing, so it has no sink for them to be about. A map
    of a raster dataset is the case that forces it -- its final frame is band-wide and its
    tiles are `png`, but its `format` at decode time can only be a guess, since which of
    `png`/`pbf`/`csv` a visualization ends up fetching depends on the grid the descriptor
    has not been computed yet to report. Every PER-OP rule still runs: `output_descriptor`
    reaches the same annotated frames through `output_frame_type`, which is the same walk
    minus the two sink checks.
    """
    if validate_sink:
        validate_shapes(pipeline)
    return UnifiedRequest(
        pipeline=pipeline,
        partition=partition,
        analysis=analysis,
        descriptor=output_descriptor(pipeline),
        compression=Compression(**partition.compression),
        relabel=relabel,
    )


def _probe_filters(partition: QueryPartition) -> dict:
    """`partition.filters` with `grid_cell` removed -- what the probe pass selects on. See
    `_assemble`."""
    return {key: value for key, value in partition.filters.items() if key != "grid_cell"}


# The spatial conjunct the probe pass stands in for the caller's with. The probe cannot
# write the real one -- resolving `grid_cell=lng,lat` to IDs is what it exists to make
# possible -- but it must not therefore assemble a pipeline with NO spatial conjunct in it,
# because a spatial conjunct is not only a narrowing: it is what lowers a band-wide frame to
# long (`request.ops.FILTER_LOWERING_TABLE`), and so what decides whether the steps after it
# type at all. `sediment_dynamics` is the case that proves it -- its `unpivot` is long-only,
# so a probe that dropped the pixel rejected the request with "there is nothing to stack
# them into" for a caller who had named one. `0` is chosen because it selects nothing on any
# grid (IDs are 1-based) and nothing is read here anyway: the probe's pipeline is assembled
# and described, never executed, and all that is read back off it is `descriptor.grid`,
# which no filter can change.
PROBE_GRID_CELL_WHERE = "grid_cell IN (0)"


async def decode_request(
    items: Iterable[tuple[str, str]],
    *,
    format: str | None,
    render_fields: Iterable[str] = (),
    allow_compression: bool = True,
    validate_sink: bool = True,
) -> UnifiedRequest:
    """Decode one request into a runnable `Pipeline`. `items` is the raw query string's
    `(key, value)` pairs with repeats preserved -- Starlette's `QueryParams.multi_items()`,
    since a repeated dimension key is how a multi-value selection is spelled.

    `format` is what the URL's own extension resolved to and always wins over anything the
    analysis or the caller declared -- or `None` on the one route whose URL has no
    extension (`/v1/analyses/{analysis}.json`), where what the analysis declared stands
    instead (see `request.unified.build_pipeline`). `render_fields` are the visualization
    class's own render options, bound only on that route; `allow_compression=False` is the
    tile routes; `validate_sink=False` is the visualization route (see `_finish`).
    """
    partition = partition_query(
        items, render_fields=render_fields, allow_compression=allow_compression
    )
    if not partition.filters.get("grid_cell"):
        return _finish(
            partition, *_assemble(partition, format, partition.filters), None, validate_sink
        )

    probe, _ = _assemble(
        partition, format, _probe_filters(partition), PROBE_GRID_CELL_WHERE
    )
    grid_cell_where, relabel = await resolve_grid_cell(
        output_descriptor(probe).grid, partition.filters["grid_cell"]
    )
    return _finish(
        partition,
        *_assemble(partition, format, partition.filters, grid_cell_where),
        relabel,
        validate_sink,
    )


def decode_request_sync(
    items: Iterable[tuple[str, str]],
    *,
    format: str | None,
    render_fields: Iterable[str] = (),
    allow_compression: bool = True,
    validate_sink: bool = True,
) -> UnifiedRequest:
    """Blocking form of `decode_request`, for a caller outside the event loop.

    A visualization builds its spec from synchronous cached properties and still has to
    decode the request that produced it; the same trade `api.data_query.
    resolve_grid_cell_sync` and `io.geography.resolve_vector_grid_cell_sync` already exist
    for, one and two layers down. Everything but the vector point lookup is shared, so the
    two forms differ in exactly the one `await`.
    """
    partition = partition_query(
        items, render_fields=render_fields, allow_compression=allow_compression
    )
    if not partition.filters.get("grid_cell"):
        return _finish(
            partition, *_assemble(partition, format, partition.filters), None, validate_sink
        )

    probe, _ = _assemble(
        partition, format, _probe_filters(partition), PROBE_GRID_CELL_WHERE
    )
    grid_cell_where, relabel = resolve_grid_cell_sync(
        output_descriptor(probe).grid, partition.filters["grid_cell"]
    )
    return _finish(
        partition,
        *_assemble(partition, format, partition.filters, grid_cell_where),
        relabel,
        validate_sink,
    )
