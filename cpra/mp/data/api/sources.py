import duckdb
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from fastapi import HTTPException
from polars.exceptions import ColumnNotFoundError

from cpra.mp.data.api.utils import timing


class LazyFrameSource:
    """Polars-backed source — used by the raster tile pipeline."""

    def __init__(self, lf):
        self.lf = lf

    def materialize(self, method="collect", *args, **kwargs):
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
        self.materialize(sink_method, out_path, *args, **kwargs)

    @timing
    def write_tabular(self, out_path, format, *args, **kwargs):
        write_method = {
            "arrow": "write_ipc",
            "csv": "write_csv",
            "ndjson": "write_ndjson",
            "parquet": "write_parquet",
        }[format]
        df = self.materialize()
        getattr(df, write_method)(out_path, *args, **kwargs)


class DuckDBSource:
    """DuckDB-backed source — used by the tabular data pipeline."""

    def __init__(self, conn: duckdb.DuckDBPyConnection, sql: str):
        self.conn = conn
        self.sql = sql

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def materialize(self):
        try:
            return self.conn.execute(self.sql).pl()
        except duckdb.OutOfMemoryException:
            raise HTTPException(status_code=507, detail="Query exceeded memory limit")
        except (duckdb.CatalogException, duckdb.ParserException, duckdb.BinderException) as err:
            raise HTTPException(status_code=400, detail=str(err))
        except duckdb.IOException as err:
            raise HTTPException(status_code=404, detail=str(err))

    @timing
    def stream_tabular(self, out_path, format, compression="zstd", compression_level=6, **kwargs):
        try:
            rel = self.conn.sql(self.sql)
            if format == "csv":
                rel.write_csv(out_path)
            elif format == "parquet":
                rel.write_parquet(
                    out_path,
                    compression=compression if compression != "none" else "uncompressed",
                    compression_level=compression_level if compression != "none" else None,
                )
            elif format == "ndjson":
                self.conn.execute(
                    f"COPY ({self.sql}) TO '{out_path}' (FORMAT JSON, ARRAY false)"
                )
            elif format == "arrow":
                table: pa.Table = rel.arrow()
                with pa_ipc.new_file(out_path, table.schema) as writer:
                    writer.write_table(table)
        except duckdb.OutOfMemoryException:
            raise HTTPException(status_code=507, detail="Query exceeded memory limit")
        except (duckdb.CatalogException, duckdb.ParserException, duckdb.BinderException) as err:
            raise HTTPException(status_code=400, detail=str(err))
        finally:
            self.close()
