from fastapi import HTTPException
from polars.exceptions import ColumnNotFoundError

from cpra.mp.data.analysis.io import output


class LazyFrameSource:
    def __init__(self, lf):
        self.lf = lf

    def tabular_kwargs(self, format):
        """Writer arguments this source needs for `format`, beyond the ones the request asks
        for.

        A plain source needs none; `Dataset` overrides this to round CSV output to its
        variable's `float_precision`. Catalog sources (crosswalks, the dataset list) cover
        many variables at once, so there is no single precision to apply to them.
        """
        return {}

    def materialize(self, method="collect", *args, **kwargs):
        """Materialize the LazyFrame with error handling."""
        try:
            return getattr(self.lf, method)(*args, **kwargs)
        except ColumnNotFoundError as err:
            raise HTTPException(status_code=400, detail=str(err))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Dataset does not exist")

    def stream_tabular(self, out_path, format, *args, **kwargs):
        """Sink the frame to `out_path` without materializing it.

        The writing is `io.output.stream_tabular`, which owns the format -> `sink_*` method
        table and the `@timing` wrapper. What this class contributes is `materialize`:
        `io.output`'s two writers take the materialize callable as a parameter exactly so
        the caller keeps ownership of how a failed collect is reported, and this one
        reports it as an `HTTPException` -- which `io.output` could not raise itself, since
        nothing under `analysis/` may import FastAPI.
        """
        output.stream_tabular(self.materialize, out_path, format, *args, **kwargs)

    def write_tabular(self, out_path, format, *args, **kwargs):
        """Collect the frame and write it to `out_path`. See `stream_tabular` for what
        `io.output` owns here and what this class adds.
        """
        output.write_tabular(self.materialize, out_path, format, *args, **kwargs)
