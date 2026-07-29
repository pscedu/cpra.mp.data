import duckdb
import polars as pl

from cpra.mp.data.config import DUCKDB_MAX_CONCURRENT, DUCKDB_MAX_THREADS, DUCKDB_MEMORY_LIMIT_MB

# Capacity limiter — caps simultaneous DuckDB queries across all requests.
# Created lazily inside an async context on first use.
_limiter = None


def get_limiter():
    global _limiter
    if _limiter is None:
        from anyio import CapacityLimiter
        _limiter = CapacityLimiter(DUCKDB_MAX_CONCURRENT)
    return _limiter


def polars_to_duckdb_type(polars_dtype) -> str:
    mapping = {
        pl.Int8: "TINYINT",
        pl.Int16: "SMALLINT",
        pl.Int32: "INTEGER",
        pl.Int64: "BIGINT",
        pl.UInt8: "UTINYINT",
        pl.UInt16: "USMALLINT",
        pl.UInt32: "UINTEGER",
        pl.UInt64: "UBIGINT",
        pl.Float32: "FLOAT",
        pl.Float64: "DOUBLE",
        pl.String: "VARCHAR",
        pl.Utf8: "VARCHAR",
        pl.Date: "DATE",
        pl.Boolean: "BOOLEAN",
    }
    return mapping.get(polars_dtype, "VARCHAR")


def open_connection(
    data_view_sql: str,
    crosswalk_views: dict[str, str] | None = None,
) -> duckdb.DuckDBPyConnection:
    """
    Create a per-request DuckDB connection with resource limits and pre-registered views.

    data_view_sql: a SELECT expression (typically read_parquet(...)) that becomes
                   the 'data' view accessible in user SQL.
    crosswalk_views: optional dict of {view_name: select_sql} for crosswalk tables.
    """
    conn = duckdb.connect()
    conn.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT_MB}MB'")
    conn.execute(f"SET threads={DUCKDB_MAX_THREADS}")
    conn.execute("SET enable_progress_bar=false")

    try:
        conn.install_extension("spatial")
        conn.load_extension("spatial")
    except Exception:
        pass  # spatial extension unavailable — raster point sampling falls back to rasterio

    conn.execute(f"CREATE VIEW data AS {data_view_sql}")

    for name, view_sql in (crosswalk_views or {}).items():
        conn.execute(f"CREATE VIEW {name} AS {view_sql}")

    return conn
