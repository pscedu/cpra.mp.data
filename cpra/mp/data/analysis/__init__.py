"""Analysis layer: the query engine behind both the HTTP API and `cpra.mp.data.
analyze_data`, organized as four one-way layers, each of which may import from the ones
above it and never the other way round:

* `request/` -- what was asked for. Parses and validates a JSON-shaped pipeline body into
  a `Pipeline` (`request.pipeline`), types its frames and checks the shape rules
  (`request.ops`), parses `where`/`calculate` expressions (`request.expressions`), and
  owns the error family every rejection here belongs to (`request.errors`). Opens no
  file and knows no crosswalk.
* `io/` -- how bytes are read. Rasters (`io.raster`/`io.raster_io`), vectors
  (`io.vector`), crosswalks (`io.crosswalk_read`), the dataset catalog (`io.catalog`),
  grid geometry and coordinate resolution (`io.geography`), and tabular output
  (`io.output`).
* `execution/` -- how a `Pipeline` runs. Compiles it to an `ExecutionPlan`
  (`execution.plan`), walks that plan over long and band-wide frames
  (`execution.execute`, `execution.long_ops`, `execution.band_ops`,
  `execution.project`), and writes the result (`execution.sink`).
* the HTTP binding, which does NOT live here -- see `cpra.mp.data.api`.

Nothing under this package may import FastAPI -- see `tests/api/test_analysis_boundary.py`,
which walks every module here and asserts it, and `tests/api/test_analysis_layering.py`,
which asserts the layer order above. `request.errors.RequestError` and its subclasses
stand in for `HTTPException`; `api.boundary` is the one place that translates them back
into HTTP status codes.

The two supported entry points, neither of which needs FastAPI or a request model:

    from cpra.mp.data import analyze_data          # a pipeline, run to a frame or a GeoTIFF
    from cpra.mp.data import read_data             # one dataset, read without analysis

Below those, `request.pipeline.parse_pipeline` validates a JSON-shaped body into a
`Pipeline`, `request.ops.validate_shapes` checks it against its inputs' grid kinds, and
`execution.plan.compile_plan` / `execution.execute.execute` run it -- the same four stages
`analyze_data` and every HTTP response go through, in that order. Import them from their
own modules; this package's `__init__` deliberately re-exports nothing, so there is no
second name for anything here.
"""
