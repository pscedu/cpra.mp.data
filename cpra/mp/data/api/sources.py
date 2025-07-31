from fastapi import HTTPException
from polars.exceptions import ColumnNotFoundError

from cpra.mp.data.api.utils import timing


class LazyFrameSource:
    def __init__(self, lf):
        self.lf = lf

    def materialize(self, method="collect", *args, **kwargs):
        """Materialize the LazyFrame with error handling."""
        try:
            return getattr(self.lf, method)(*args, **kwargs)
        except ColumnNotFoundError as err:
            raise HTTPException(status_code=400, detail=str(err))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Dataset does not exist")

    @timing
    def stream_tabular(self, out_path, format, *args, **kwargs):
        sink_method = {
            "arrow": "sink_ipc",
            "csv": "sink_csv",
            "ndjson": "sink_ndjson",
            "parquet": "sink_parquet",
        }[format]
        # TODO: Set parquet compression and row group parameters.
        self.materialize(sink_method, out_path, *args, **kwargs)
