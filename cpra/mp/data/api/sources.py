from fastapi import HTTPException
from polars.exceptions import ColumnNotFoundError

from cpra.mp.data.api.utils import timing


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

    @timing
    def write_tabular(self, out_path, format, *args, **kwargs):
        write_method = {
            "arrow": "write_ipc",
            "csv": "write_csv",
            "ndjson": "write_ndjson",
            "parquet": "write_parquet",
        }[format]
        # TODO: Set parquet compression and row group parameters.
        df = self.materialize()
        getattr(df, write_method)(out_path, *args, **kwargs)
