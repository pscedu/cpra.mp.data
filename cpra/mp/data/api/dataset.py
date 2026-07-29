"""
Dataset classes for the CPRA Master Plan data API.

TabularDataset  — DuckDB-backed; used by the data-download endpoint.
Dataset         — Polars + rasterio; used by raster/vector tile endpoints.
"""

import itertools
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path

import duckdb
import mercantile
import numpy as np
import polars as pl
import polars.selectors as cs
import rasterio as rio
from fastapi import HTTPException
from PIL import Image
from rasterio import windows
from rasterio.coords import BoundingBox, disjoint_bounds
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform

from cpra.mp.data.api.duckdb import open_connection, polars_to_duckdb_type
from cpra.mp.data.api.models import DatasetDimensions, FilterDimensions, Options
from cpra.mp.data.api.sources import DuckDBSource, LazyFrameSource
from cpra.mp.data.api.types import AggregateFunction, OutputType
from cpra.mp.data.api.utils import (
    asynctiming,
    get_lazy_frame,
    parallel_read_rasters,
    parallel_sample_rasters,
    unpack_values,
)
from cpra.mp.data.config import BASE_PATH, DUCKDB_MEMORY_LIMIT_MB
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.legends import LEGENDS, Legend
from cpra.mp.data.utils import (
    dataset_path,
    get_input_data_frame,
    partition_columns,
    required_columns,
)
from cpra.mp.data.variables import VARIABLES, Variable

DIMENSION_CHOICES = list(DIMENSIONS.keys())
SORT_CHOICES = DIMENSION_CHOICES + [f"-{dim_name}" for dim_name in DIMENSION_CHOICES]
DIMENSION_COLUMNS = cs.matches(r"^[a-z]") - cs.by_name("grid_cell", "value")
VALUE_COLUMNS = cs.by_name("value")


# ---------------------------------------------------------------------------
# TabularDataset — DuckDB-backed data-download pipeline
# ---------------------------------------------------------------------------

class TabularDataset:
    """DuckDB-backed dataset for tabular data downloads and raster point extraction."""

    def __init__(self, dataset_dimensions: DatasetDimensions, options: Options):
        self.dataset_dimensions = dataset_dimensions
        self.options = options
        self.grid = GRIDS[dataset_dimensions.grid]

        self.source: DuckDBSource | None = None
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._spatial_loaded = False

    def _error(self, msg, status_code=400):
        raise HTTPException(status_code=status_code, detail=msg)

    @property
    def output_type(self) -> OutputType:
        if self.grid.grid_type == "vector" or self.options.grid_cell:
            return OutputType.vector
        return OutputType.raster

    @property
    def variable(self) -> Variable:
        return VARIABLES[self.dataset_dimensions.variable]

    @property
    def legend(self) -> Legend:
        color_map = self.options.color_map or ""
        if ":" in color_map:
            map_type, color_str = color_map.split(":", 1)
            if map_type not in ["continuous", "discrete"]:
                self._error(f"Invalid color map type: {map_type}")
            color_strs = color_str.split(",")
            if map_type == "continuous":
                colors = color_strs[1::2]
                value_strs = color_strs[::2]
            else:
                colors = color_strs[::2]
                value_strs = color_strs[1::2]
            try:
                values = [float(val) for val in value_strs]
            except ValueError:
                self._error("Invalid color map values")
            if map_type == "discrete":
                values += [None]
            if len(values) != len(colors):
                self._error("Mismatched colors and values in color map")
            return Legend("custom", pl.DataFrame(dict(color=colors, upper_bound=values)), type=map_type)
        return LEGENDS.get(self.options.color_map) or self.variable.legend or LEGENDS["default"]

    # ------------------------------------------------------------------
    # SQL construction helpers
    # ------------------------------------------------------------------

    def _build_data_view_sql(self) -> str:
        """Build the SQL expression that becomes the 'data' view in DuckDB."""
        dir_path = dataset_path(BASE_PATH, **dict(self.dataset_dimensions))

        # Determine Hive partition column types for explicit schema hints.
        input_df = get_input_data_frame(**dict(self.dataset_dimensions))
        part_col_names = partition_columns(input_df, include_dataset=False)
        hive_types = {
            dim.name: polars_to_duckdb_type(dim.dtype)
            for dim in DIMENSIONS.values()
            if dim.name in part_col_names
        }
        hive_types_fragment = ""
        if hive_types:
            pairs = ", ".join(f"'{k}': {v}" for k, v in hive_types.items())
            hive_types_fragment = f"hive_types={{{pairs}}}, "

        # Dataset dimensions (variable, time_unit) are baked into dir_path
        # and won't be visible as Hive columns, so add them as literals.
        lit_cols = ", ".join(
            f"'{val}'::VARCHAR AS {name}"
            for name, val in dict(self.dataset_dimensions).items()
        )

        return (
            f"SELECT {lit_cols}, p.* "
            f"FROM read_parquet("
            f"  '{dir_path}/**/*.parquet', "
            f"  hive_partitioning=true, "
            f"  {hive_types_fragment}"
            f"  union_by_name=true"
            f") p"
        )

    def _build_crosswalk_views(self) -> dict[str, str]:
        """Return {view_name: sql} for any crosswalk tables needed by summarize_by."""
        if not self.options.summarize_by:
            return {}
        crosswalk_name = f"{self.grid.name}__{self.options.summarize_by}.parquet"
        crosswalk_path = Path(__file__).parent.parent / "crosswalk" / crosswalk_name
        if not crosswalk_path.exists():
            self._error(f"No crosswalk from {self.grid.name} to {self.options.summarize_by}")
        return {"xw": f"SELECT * FROM read_parquet('{crosswalk_path}')"}

    def _wrap_with_summarize(self, inner_sql: str) -> str:
        """Wrap inner_sql with a crosswalk join for geographic upsampling."""
        agg_fn = str(self.options.summarize_aggregate or AggregateFunction.mean).upper()
        if agg_fn == "MEAN":
            agg_fn = "AVG"

        dst_filter = ""
        if self.options.summarize_grid_cell:
            ids = [s.strip() for s in self.options.summarize_grid_cell.split(",")]
            id_list = ", ".join(ids)
            dst_filter = f"AND xw.dst_id IN ({id_list})"

        return f"""
            SELECT
                subq.* EXCLUDE (grid_cell, value),
                xw.dst_id AS grid_cell,
                {agg_fn}(subq.value * xw.scale_factor) AS value,
                '{self.options.summarize_by}' AS geographic_unit
            FROM ({inner_sql}) subq
            JOIN xw ON subq.grid_cell = xw.src_id
            WHERE TRUE {dst_filter}
            GROUP BY ALL
        """

    def _build_user_sql(self) -> str:
        """Return the effective SQL query, wrapping the user's sql with any server-side logic."""
        base = self.options.sql or "SELECT * FROM data"

        # For vector grids with grid_cell, add an IN filter around the user's query.
        if self.options.grid_cell and self.grid.grid_type == "vector":
            ids = [s.strip() for s in self.options.grid_cell.split(",")]
            try:
                int_ids = [int(i) for i in ids]
            except ValueError:
                self._error("grid_cell must contain integer IDs for vector grids")
            id_list = ", ".join(str(i) for i in int_ids)
            base = f"SELECT * FROM ({base}) _gc_filter WHERE grid_cell IN ({id_list})"

        if self.options.summarize_by:
            base = self._wrap_with_summarize(base)

        return base

    # ------------------------------------------------------------------
    # Raster point extraction
    # ------------------------------------------------------------------

    def _parse_raster_coords(self) -> tuple[list[float], list[float]]:
        """Parse and validate lng,lat pairs from options.grid_cell."""
        try:
            coords = [float(s.strip()) for s in self.options.grid_cell.split(",")]
        except ValueError:
            self._error("grid_cell contains non-numeric coordinates")
        if len(coords) % 2:
            self._error("grid_cell must contain an even number of coordinates")
        lngs, lats = [], []
        for i in range(0, len(coords), 2):
            lng, lat = coords[i], coords[i + 1]
            if abs(lng) > 180 or abs(lat) > 90:
                self._error(f"Invalid coordinate pair: {lng}, {lat}")
            lngs.append(lng)
            lats.append(lat)
        return lngs, lats

    def _sample_rasters_duckdb(
        self,
        paths: list[str],
        paths_df: pl.DataFrame,
        lngs: list[float],
        lats: list[float],
        dst_crs: str,
    ) -> pl.DataFrame | None:
        """
        Sample COG values at WGS84 coordinates using the DuckDB spatial extension.

        Returns a DataFrame with one row per (path, coordinate) and columns from
        paths_df plus 'coord_idx' and 'value', or None if the extension does not
        support the required raster functions.
        """
        if not self._spatial_loaded:
            return None

        # ST_Value only samples band 1, but a compacted multi-band COG packs several
        # calendar values into one file — fall back to the band-aware rasterio path
        # rather than silently returning band 1's value for every row.
        if "band" in paths_df.columns and (paths_df["band"] != 1).any():
            return None

        conn = duckdb.connect()
        conn.load_extension("spatial")

        conn.register("_paths", paths_df)
        coord_records = pl.DataFrame({
            "coord_idx": list(range(len(lngs))),
            "lng": lngs,
            "lat": lats,
        })
        conn.register("_coords", coord_records)

        try:
            # raster_scan() returns a relation with a 'rast' column per file.
            # ST_Value samples the band-1 value at the projected point.
            result = conn.execute(f"""
                SELECT
                    p.path,
                    c.coord_idx,
                    c.lng,
                    c.lat,
                    ST_Value(
                        (SELECT rast FROM raster_scan(p.path) LIMIT 1),
                        ST_Transform(
                            ST_Point(c.lng, c.lat),
                            'EPSG:4326',
                            '{dst_crs}'
                        )
                    ) AS value
                FROM _paths p
                CROSS JOIN _coords c
            """).pl()
        except Exception:
            return None
        finally:
            conn.close()

        return result

    async def _sample_rasters_rasterio(
        self,
        paths_df: pl.DataFrame,
        lngs: list[float],
        lats: list[float],
    ) -> list[np.ndarray]:
        """Sample COG values using the existing parallel rasterio approach."""
        with ExitStack() as stack:
            datasets = [stack.enter_context(rio.open(p)) for p in paths_df["path"]]
            src_crs = CRS.from_epsg(4326)
            projected_pairs = list(zip(*transform(src_crs, datasets[0].crs, lngs, lats)))
            bands = paths_df["band"].to_list() if "band" in paths_df.columns else None
            return await parallel_sample_rasters(datasets, projected_pairs, bands=bands)

    def _raster_samples_to_relation(
        self,
        paths_df: pl.DataFrame,
        samples: list[np.ndarray],
        lngs: list[float],
        lats: list[float],
    ) -> pl.LazyFrame:
        """Convert rasterio sample arrays into a wide-format LazyFrame."""
        columns = [np.concatenate(col) for col in zip(*samples)]
        column_labels = [f"{x!s},{y!s}" for x, y in zip(lngs, lats)]
        return (
            paths_df
            .with_columns(**dict(zip(column_labels, columns)))
            .select(cs.exclude("path", "band"))
            .lazy()
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def apply_params(self):
        """Resolve all options and prepare self.source for streaming."""
        if self.options.grid_cell and self.grid.grid_type == "raster":
            await self._setup_raster_point_pipeline()
        else:
            self._setup_tabular_pipeline()

    def _setup_tabular_pipeline(self):
        data_view_sql = self._build_data_view_sql()
        crosswalk_views = self._build_crosswalk_views()
        self._conn = open_connection(data_view_sql, crosswalk_views)

        # Check whether the spatial extension actually loaded.
        try:
            loaded = self._conn.execute(
                "SELECT loaded FROM duckdb_extensions() WHERE extension_name='spatial'"
            ).fetchone()
            self._spatial_loaded = bool(loaded and loaded[0])
        except Exception:
            self._spatial_loaded = False

        user_sql = self._build_user_sql()
        self.source = DuckDBSource(self._conn, user_sql)

    async def _setup_raster_point_pipeline(self):
        """Point extraction from COG files — tries DuckDB spatial, falls back to rasterio."""
        from cpra.mp.data.raster import read_cog

        lngs, lats = self._parse_raster_coords()
        paths_lf = read_cog(**dict(self.dataset_dimensions))
        paths_df = paths_lf.collect()

        if paths_df.is_empty():
            self._error("No raster data found for the specified parameters", 404)

        # Open a connection just for the spatial extension check.
        self._conn = open_connection("SELECT 1 AS _dummy", {})
        try:
            loaded = self._conn.execute(
                "SELECT loaded FROM duckdb_extensions() WHERE extension_name='spatial'"
            ).fetchone()
            self._spatial_loaded = bool(loaded and loaded[0])
        except Exception:
            self._spatial_loaded = False

        dst_crs = self.grid.crs

        duckdb_result = self._sample_rasters_duckdb(
            paths_df["path"].to_list(), paths_df, lngs, lats, dst_crs
        )

        if duckdb_result is not None:
            # DuckDB returned long-format (path, coord_idx, lng, lat, value).
            # Pivot to the same wide format the rasterio path produces so that
            # downstream output is consistent.
            coord_labels = [f"{lng},{lat}" for lng, lat in zip(lngs, lats)]
            dim_cols = [c for c in paths_df.columns if c not in ("path", "band")]
            wide = (
                duckdb_result
                .join(paths_df.select(dim_cols + ["path"]), on="path")
                .pivot(on="coord_idx", index=dim_cols, values="value")
                .rename({str(i): coord_labels[i] for i in range(len(coord_labels))})
            )
            lf = wide.lazy()
        else:
            samples = await self._sample_rasters_rasterio(paths_df, lngs, lats)
            lf = self._raster_samples_to_relation(paths_df, samples, lngs, lats)

        # Materialise into DuckDB so the user's sql (if any) can be applied.
        materialized = lf.collect()
        self._conn.close()
        self._conn = duckdb.connect()
        self._conn.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT_MB}MB'")
        self._conn.register("data", materialized)

        user_sql = self.options.sql or "SELECT * FROM data"
        self.source = DuckDBSource(self._conn, user_sql)


# ---------------------------------------------------------------------------
# Dataset — Polars + rasterio tile pipeline (unchanged interface)
# ---------------------------------------------------------------------------

class Dataset(LazyFrameSource):
    def __init__(
        self,
        dataset_dimensions: DatasetDimensions,
        filter_dimensions: FilterDimensions,
        options,
    ):
        self.dataset_dimensions = dataset_dimensions
        self.filter_dimensions = filter_dimensions
        self.options = options
        self.dataset_bounds = None
        self.grid_cell_ids = None
        self.summarize_grid_cell_ids = None
        self.raster_groups = None

        self.lf = get_lazy_frame(self.dataset_dimensions, self.filter_dimensions)
        self.grid = GRIDS[self.dataset_dimensions.grid]

    def _error(self, msg, status_code=400):
        raise HTTPException(status_code=status_code, detail=msg)

    @property
    def legend(self) -> Legend:
        color_map = getattr(self.options, "color_map", None) or ""
        if ":" in color_map:
            map_type, color_str = color_map.split(":", 1)
            if map_type not in ["continuous", "discrete"]:
                self._error(f"Invalid color map type: {map_type}")
            color_strs = color_str.split(",")
            if map_type == "continuous":
                colors = color_strs[1::2]
                value_strs = color_strs[::2]
            else:
                colors = color_strs[::2]
                value_strs = color_strs[1::2]
            try:
                values = [float(val) for val in value_strs]
            except ValueError:
                self._error("Invalid color map values")
            if map_type == "discrete":
                values += [None]
            if len(values) != len(colors):
                self._error("Mismatched colors and values in color map")
            return Legend("custom", pl.DataFrame(dict(color=colors, upper_bound=values)), type=map_type)
        return LEGENDS.get(color_map) or self.variable.legend or LEGENDS["default"]

    @property
    def output_type(self) -> OutputType:
        grid_cell = getattr(self.options, "grid_cell", None)
        if self.grid.grid_type == "vector" or grid_cell:
            return OutputType.vector
        return OutputType.raster

    @property
    def variable(self) -> Variable:
        return VARIABLES[self.dataset_dimensions.variable]

    async def _apply_filter_params(self):
        where = getattr(self.options, "where", None)
        if where:
            self.lf = self.lf.sql(f"select * from self where {where}")

        grid_cell = getattr(self.options, "grid_cell", None)
        if grid_cell:
            if self.grid.grid_type == "vector":
                await self._apply_grid_cell_vector(grid_cell)
            else:
                await self._apply_grid_cell_raster(grid_cell)

    async def _apply_grid_cell_vector(self, grid_cell):
        self.grid_cell_ids = unpack_values(grid_cell, "grid_cell")
        self.lf = self.lf.filter(pl.col("grid_cell").is_in([int(g) for g in self.grid_cell_ids]))

    async def _apply_grid_cell_raster(self, grid_cell):
        try:
            coords = [float(coord) for coord in unpack_values(grid_cell, "grid_cell")]
        except ValueError:
            raise HTTPException(status_code=400, detail="grid_cell contains non-numeric coordinates")
        coords_len = len(coords)
        if coords_len % 2:
            raise HTTPException(status_code=400, detail="grid_cell must contain an even number of coordinates")
        lngs, lats = [], []
        for start_idx in range(0, coords_len, 2):
            lng, lat = coords[start_idx: (start_idx + 2)]
            if abs(lng) > 180 or abs(lat) > 90:
                raise HTTPException(status_code=400, detail=f"Invalid coordinate pair: {lng}, {lat}")
            lngs.append(lng)
            lats.append(lat)
        df = self.materialize()
        with ExitStack() as stack:
            datasets = [stack.enter_context(rio.open(path)) for path in df["path"]]
            src_crs = CRS.from_epsg(4326)
            dst_crs = datasets[0].crs
            projected_pairs = list(zip(*transform(src_crs, dst_crs, lngs, lats)))
            bands = df["band"].to_list() if "band" in df.columns else None
            samples = await parallel_sample_rasters(datasets, projected_pairs, bands=bands)

        # Add the resulting values to the LazyFrame, and remove the internal path/band columns.
        columns = [np.concatenate(col) for col in zip(*samples)]
        column_labels = [f"{x!s},{y!s}" for x, y in zip(lngs, lats)]
        self.lf = df.with_columns(**dict(zip(column_labels, columns))).select(cs.exclude("path", "band")).lazy()

    def _apply_calculate_params(self):
        if self.output_type == OutputType.raster:
            self.lf = self.lf.with_columns(multiplier=pl.lit(1))
            value_cols = pl.col("multiplier")
        else:
            value_cols = VALUE_COLUMNS

        negate = getattr(self.options, "negate", None)
        multiply10 = getattr(self.options, "multiply10", None)
        multiply100 = getattr(self.options, "multiply100", None)

        if negate is None and multiply10 is None and multiply100 is None:
            return

        sql_expressions = [negate, multiply10, multiply100]
        factors = [-1, 10, 100]
        when_expression = pl
        for sql_expression, factor in zip(sql_expressions, factors):
            if sql_expression is None:
                continue
            when_expression = when_expression.when(pl.sql_expr(sql_expression)).then(value_cols * factor)
        when_expression = when_expression.otherwise(value_cols)
        self.lf = self.lf.with_columns(when_expression)

    def _apply_sort_params(self):
        sort = getattr(self.options, "sort", None)
        select = getattr(self.options, "select", None)
        if sort is not None:
            parts = unpack_values(sort, "sort", SORT_CHOICES)
            cols = [part.replace("-", "") for part in parts]
            desc = [part.startswith("-") for part in parts]
            self.lf = self.lf.sort(cols, descending=desc)
        if select is not None:
            required_cols = required_columns(
                get_input_data_frame(**dict(self.dataset_dimensions)),
                include_grid_cell=True,
                include_value=True,
            )
            select_cols = unpack_values(select, "select", required_cols)
            self.lf = self.lf.select(select_cols)

    def _get_group_by_columns(self):
        group_by = getattr(self.options, "group_by", None)
        group_aggregate = getattr(self.options, "group_aggregate", None)
        if group_by is not None and group_aggregate is None:
            raise HTTPException(status_code=400, detail="group_aggregate must be specified when group_by is used")
        if group_by is None:
            return [True]
        return unpack_values(group_by, "group_by", DIMENSION_CHOICES)

    def _apply_group_params(self):
        group_by = getattr(self.options, "group_by", None)
        group_aggregate = getattr(self.options, "group_aggregate", None)
        if group_by is None and group_aggregate is None:
            return
        group_by_cols = self._get_group_by_columns()
        if self.output_type == OutputType.vector:
            required_cols = required_columns(get_input_data_frame(**dict(self.dataset_dimensions)))
            null_cols = [col for col in required_cols if col not in group_by_cols]

            # grid_cell must always be part of the group key in long format since each
            # row represents a single grid_cell.
            self.lf = self.lf.group_by(*group_by_cols, "grid_cell").agg(
                getattr(pl.col("value"), group_aggregate)()
            )
            if null_cols:
                self.lf = self.lf.with_columns(*[pl.lit(None).alias(col) for col in null_cols])
        else:
            self.raster_groups = self.lf.group_by(*group_by_cols).agg(row_idxs=pl.col("path").agg_groups())

    def _apply_transform_params_vector(self):
        summarize_by = getattr(self.options, "summarize_by", None)
        summarize_aggregate = getattr(self.options, "summarize_aggregate", None)
        if summarize_by is None and summarize_aggregate is None:
            return
        if summarize_by is None or summarize_aggregate is None:
            raise HTTPException(status_code=400, detail="summarize_by and summarize_aggregate must be used together")
        summarize_grid_cell = getattr(self.options, "summarize_grid_cell", None)
        if summarize_grid_cell:
            self.summarize_grid_cell_ids = [
                int(geo_id) for geo_id in unpack_values(summarize_grid_cell, "summarize_grid_cell")
            ]
        crosswalk_name = f"{self.grid.name}__{summarize_by}.parquet"
        crosswalk_path = Path(__file__).parent.parent / "crosswalk" / crosswalk_name
        try:
            crosswalk_lf = pl.scan_parquet(str(crosswalk_path))
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=f"No crosswalk from {self.grid.name} to {summarize_by}")
        if self.grid_cell_ids:
            crosswalk_lf = crosswalk_lf.filter(pl.col("src_id").is_in([int(g) for g in self.grid_cell_ids]))
        if self.summarize_grid_cell_ids:
            crosswalk_lf = crosswalk_lf.filter(pl.col("dst_id").is_in(self.summarize_grid_cell_ids))
        dim_cols = cs.expand_selector(self.lf, DIMENSION_COLUMNS)
        self.lf = (
            self.lf.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner")
            .with_columns(value=pl.col("value") * pl.col("scale_factor"))
            .group_by(*dim_cols, pl.col("dst_id").alias("grid_cell"))
            .agg(getattr(pl.col("value"), summarize_aggregate)())
            .with_columns(geographic_unit=pl.lit(summarize_by))
        )

    def _apply_transform_params(self):
        if self.output_type == OutputType.vector:
            self._apply_transform_params_vector()

    async def apply_params(self):
        await self._apply_filter_params()
        self._apply_calculate_params()
        self._apply_group_params()
        self._apply_transform_params()
        self._apply_sort_params()

    def validate_tile(self):
        if self.output_type == OutputType.vector:
            raise HTTPException(status_code=400, detail="invalid output type for tile: vector")
        group_lf = self.raster_groups if self.raster_groups is not None else self.lf
        groups = group_lf.select(pl.len()).collect().item()
        if groups > 1:
            raise HTTPException(status_code=400, detail=f"tile contains more than one band: {groups} bands found")

    def validate_vector_tile(self):
        from cpra.mp.data.api.vector import TILES_PATH
        if self.output_type != OutputType.vector:
            raise HTTPException(status_code=400, detail="invalid output type for vector tile: raster")
        grid_name = getattr(self.options, "summarize_by", None) or self.grid.name
        if not (TILES_PATH / f"{grid_name}.pmtiles").exists():
            raise HTTPException(status_code=404, detail=f"no tile file for grid: {grid_name}")

    @asynctiming
    async def get_vector_tile(self, x: int, y: int, z: int) -> bytes | None:
        """Get an MVT tile with dataset values injected into pre-generated geometry."""
        from cpra.mp.data.api.vector import (
            _data_to_id_value,
            inject_and_serialize,
            parse_tile,
            read_pmtile,
        )

        grid_name = getattr(self.options, "summarize_by", None) or self.grid.name
        tile = mercantile.Tile(x, y, z)
        compressed = read_pmtile(grid_name, tile)
        if compressed is None:
            return None
        vt, raw_ids = parse_tile(compressed)
        if not raw_ids:
            return None
        tile_id_strs = [str(fid) for fid in raw_ids]
        if self.summarize_grid_cell_ids is not None:
            valid = {str(gid) for gid in self.summarize_grid_cell_ids}
            val_cols = [c for c in tile_id_strs if c in valid]
        elif self.grid_cell_ids is not None:
            valid = {str(gid) for gid in self.grid_cell_ids}
            val_cols = [c for c in tile_id_strs if c in valid]
        else:
            val_cols = tile_id_strs
        if not val_cols:
            return None
        tile_ints = [int(c) for c in val_cols]
        df = self.lf.filter(pl.col("grid_cell").is_in(tile_ints)).collect()
        if df.is_empty():
            return None
        id_values = _data_to_id_value(df)
        if not id_values:
            return None
        return inject_and_serialize(vt, id_values)

    async def get_raster_array(self, df, rasters, window, out_shape=None):
        """Read a raster array and apply the operations specified
        in the index DataFrame."""
        # Each index row selects one band of its (possibly multi-band) file.
        bands = df["band"].to_list() if "band" in df.columns else None
        data = np.vstack(await parallel_read_rasters(rasters, window, out_shape, bands=bands))

        # Apply multipliers from calculate parameters.
        if not (df["multiplier"] == 1).all():
            data = data * df["multiplier"].to_numpy()[..., np.newaxis, np.newaxis]
        if self.raster_groups is not None:
            aggregate_fn = getattr(np, getattr(self.options, "group_aggregate", "mean"))
            data = np.vstack([
                aggregate_fn(data[row_idxs, :, :], axis=0, keepdims=True)
                for row_idxs in self.raster_groups.collect()["row_idxs"]
            ])
        return data

    @asynctiming
    async def get_tile(self, x, y, z, format, tile_size=256, **kwargs):
        tile_bounds = BoundingBox(*mercantile.xy_bounds(x, y, z))
        if self.dataset_bounds and disjoint_bounds(self.dataset_bounds, tile_bounds):
            return None
        df = self.materialize()
        with ExitStack() as stack:
            inputs = [stack.enter_context(rio.open(path)) for path in df["path"]]
            vrts = [
                WarpedVRT(
                    dataset, crs="EPSG:3857", nodata=-9999, add_alpha=False,
                    src_nodata=-9999, dtype=dataset.dtypes[0], resampling=Resampling.nearest,
                )
                for dataset in inputs
            ]
            if not self.dataset_bounds:
                self.dataset_bounds = vrts[0].bounds
                if disjoint_bounds(self.dataset_bounds, tile_bounds):
                    return None
            read_bounds = BoundingBox(
                max(tile_bounds.left, self.dataset_bounds.left),
                max(tile_bounds.bottom, self.dataset_bounds.bottom),
                min(tile_bounds.right, self.dataset_bounds.right),
                min(tile_bounds.top, self.dataset_bounds.top),
            )
            scale = tile_size / abs(tile_bounds.right - tile_bounds.left)
            out_shape = (
                1,
                round(scale * abs(read_bounds.top - read_bounds.bottom)),
                round(scale * abs(read_bounds.right - read_bounds.left)),
            )
            window = vrts[0].window(*read_bounds)
            data = await self.get_raster_array(df, vrts, window, out_shape)
            color = self.legend.colorize_pixels(data)
            image = Image.fromarray(
                color[:, :, :3] if format == "jpg" else color,
                mode=("RGB" if format == "jpg" else "RGBA"),
            )
            if image.size != (tile_size, tile_size):
                left_offset = round(abs(read_bounds.left - tile_bounds.left) * scale)
                top_offset = round(abs(tile_bounds.top - read_bounds.top) * scale)
                padded_image = Image.new(image.mode, (tile_size, tile_size))
                padded_image.paste(image, (left_offset, top_offset), mask=image)
                image = padded_image
            with BytesIO() as memfile:
                image.save(memfile, format=("JPEG" if format == "jpg" else "PNG"))
                return memfile.getvalue()

    @asynctiming
    async def write_raster(self, out_path, **kwargs):
        df = self.materialize()
        if not len(df):
            return False
        with ExitStack() as stack:
            inputs = [stack.enter_context(rio.open(path)) for path in df["path"]]

            # Each selected (path, band) row — or each aggregated group — becomes one
            # output band, so the source's own band count must be overridden.
            if self.raster_groups is not None:
                out_count = self.raster_groups.select(pl.len()).collect().item()
            else:
                out_count = len(df)
            profile = inputs[0].profile.copy()
            profile.update(count=out_count, **kwargs)
            output = stack.enter_context(rio.open(out_path, "w", **profile))

            # Label output bands with their calendar value when rows map 1:1 to bands.
            if self.raster_groups is None:
                calendar_col = next((c for c in ("calendar_year", "calendar_day") if c in df.columns), None)
                if calendar_col is not None:
                    for band_idx, value in enumerate(df[calendar_col].to_list(), start=1):
                        output.set_band_description(band_idx, f"{calendar_col}={value}")

            # Group raster blocks by row, and read an entire row of blocks at once.
            groups = itertools.groupby(inputs[0].block_windows(1), key=lambda bw: bw[0][0])
            for _group_row, group in groups:
                window = windows.union(*[w for ji, w in group])
                data = await self.get_raster_array(df, inputs, window)
                output.write(data, window=window)
        return True
