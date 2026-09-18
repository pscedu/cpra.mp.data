"""The op algebra: turning a validated request into a plan and running it over frames.

No request parsing and no route knowledge lives here -- see `request` and `api/`.

`run_to_path` is the one export at this level: compile a `Pipeline`, run it, and sink the
result to a path in one call, for the two output kinds (tabular, `tif`) that need nothing
beyond a pipeline and a destination. Every other public name in this package (`compile_plan`,
`execute`, and every function in `execution.sink`, including the tile/vector-tile sinks a
per-tile caller needs directly) is reached by importing that module itself, the same way a
caller already reaches `execution.plan`/`execution.execute`/`execution.frame` today --
`run_to_path` is a composition convenience, not a signal that this package now re-exports
everything beneath it.
"""

from cpra.mp.data.analysis.execution.execute import run_to_path

__all__ = ["run_to_path"]
