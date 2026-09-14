"""`VectorGridAnalysis`: the vector half of the former `api/dataset.py`'s `Dataset` class,
plus the pure MVT (Mapbox Vector Tile) helper functions formerly in `api/vector.py` (moved
here because `get_vector_tile` below is their only caller besides tests, and
`cpra.mp.data.analysis` may not import FastAPI -- see `analysis/base.py`'s docstring).
`api/vector.py` keeps only `VectorTileResponse`, which calls `get_vector_tile` on whatever
`BaseAnalysis` subclass it's given rather than these helpers directly.

See `analysis/base.py`'s module docstring for the staged `apply_params` flow and the hook
contract every method here fills in; see `analysis/raster.py` for the raster half.
"""

import gzip

import mercantile
import polars as pl
import polars.selectors as cs
from mapbox_vector_tile.Mapbox.vector_tile_pb2 import tile as VectorTile
from pmtiles.reader import MmapSource, Reader

from cpra.mp.data import config
from cpra.mp.data.analysis.base import BaseAnalysis
from cpra.mp.data.analysis.expressions import ExpressionError, count_by_value_frame
from cpra.mp.data.analysis.fold import FoldError, fold_group_by, should_fold_group_by
from cpra.mp.data.analysis.geography import (
    parse_coordinate_pairs,
    resolve_vector_grid_cell,
)
from cpra.mp.data.analysis.utils import asynctiming, unpack_values
from cpra.mp.data.config import COUNT_BY_VALUE_MAX_DISTINCT

DIMENSION_COLUMNS = cs.matches(r"^[a-z]") - cs.by_name("grid_cell", "value")


def read_pmtile(grid_name: str, tile: mercantile.Tile) -> bytes | None:
    """Return the raw (still gzip-compressed) MVT bytes for a tile, or None."""
    path = config.TILES_PATH / f"{grid_name}.pmtiles"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = Reader(MmapSource(f)).get(tile.z, tile.x, tile.y)
    return data  # may be None if tile not in archive


def _data_to_id_value(df: pl.DataFrame) -> dict[int, float]:
    """Extract {grid_cell_id: value} from a long-format DataFrame."""
    if df.is_empty() or "grid_cell" not in df.columns:
        return {}
    return {
        cell: val
        for cell, val in zip(df["grid_cell"].to_list(), df["value"].to_list())
        if val is not None
    }


def parse_tile(compressed: bytes) -> tuple[VectorTile, list[int]]:
    """Decompress and parse a gzip-compressed MVT tile once.

    Returns the parsed VectorTile object and the feature IDs from the boundary
    layer. Callers should pass the VectorTile to inject_and_serialize rather than
    re-parsing from bytes.
    """
    vt = VectorTile()
    vt.ParseFromString(gzip.decompress(compressed))
    ids = []
    for layer in vt.layers:
        if layer.name == "boundary":
            ids = [f.id for f in layer.features]
            break
    return vt, ids


def inject_and_serialize(vt: VectorTile, id_values: dict[int, float]) -> bytes:
    """Inject a 'value' property into each boundary feature and serialize.

    Geometry bytes are never decoded — only keys/values/tags are modified — so
    this is O(features) string/int work rather than O(coordinates) float work.
    """
    for layer in vt.layers:
        if layer.name != "boundary":
            continue
        key_idx = len(layer.keys)
        layer.keys.append("value")
        for feature in layer.features:
            v = id_values.get(feature.id)
            if v is not None:
                val = layer.values.add()
                val.double_value = v
                feature.tags.extend([key_idx, len(layer.values) - 1])

    return vt.SerializeToString()


class VectorGridAnalysis(BaseAnalysis):
    async def _apply_grid_cell(self):
        """Apply the grid_cell filter to a vector-grid dataset.

        Accepts either grid cell IDs or lat/lng coordinate pairs, matching the raster
        convention. A grid_cell value is treated as IDs only if every token is a plain
        (unsigned, non-decimal) digit string; otherwise the whole list is parsed as
        coordinate pairs and resolved to grid cell IDs via point-in-polygon lookup
        against the grid's geometry. Coordinates that don't fall inside any polygon are
        dropped rather than erroring.

        Either way the result is reported under grid cell IDs, never the coordinates that
        selected them: a vector grid_cell *is* its ID. Several coordinates landing in one
        cell therefore select it once.
        """
        values = unpack_values(self.options.grid_cell, "grid_cell")

        if all(value.isdigit() for value in values):
            # TODO: Validate that the given IDs fall within the grid min and max.
            grid_cell_ids = values
        else:
            lngs, lats = parse_coordinate_pairs(values)
            resolved = await resolve_vector_grid_cell(self.grid.name, lngs, lats)
            grid_cell_ids = [str(geo_id) for geo_id in resolved if geo_id is not None]

        self.grid_cell_ids = list(dict.fromkeys(grid_cell_ids))
        self.lf = self.lf.filter(pl.col("grid_cell").is_in([int(g) for g in self.grid_cell_ids]))

    def _collapse_group(self, group_by_cols):
        """Collapse `self.lf` per `group_by_cols`, a real vector dimension-collapse
        across rows per grid_cell (via `group_by(...).agg(...)`, or
        `analysis.fold.fold_group_by` when cliff-shaped -- see `should_fold_group_by`).

        A dimension column left out of `group_by_cols` is simply absent from the result
        -- `group_by(...).agg(...)` (and `fold_group_by`, which delegates to the same
        Polars primitive per chunk) already drops any column that is neither a group key
        nor an aggregated expression, so no extra step is needed to make that happen.
        This used to be undone right after: every such column was added straight back as
        an all-null, Null-dtype placeholder, so a caller saw a schema-stable but
        meaningless column for every dimension the request didn't group by (e.g.
        `model_group_id`, `scenario_id`, `calendar_day` all appearing, entirely null, in
        a plain `group_aggregate='count'` response). Dropping instead of null-filling is
        the honest shape for "this dimension no longer has a value" and matches
        `VectorGridAnalysis`'s own crosswalk-reaggregation branch below (`summarize_by`)
        and `PairedAnalysis`/the pipeline shape, neither of which ever reintroduces a
        collapsed dimension as null -- see the README's `analyze_data` example, whose
        documented output columns (`grid_cell, calendar_year, value`) already assumed
        the collapsed dimensions were simply gone. A `select` naming a since-collapsed
        dimension now surfaces as a clean `ColumnNotFoundError` -> 400 out of
        `materialize`, rather than silently returning an all-null column.
        """
        # grid_cell must always be part of the group key in long format since each
        # row represents a single grid_cell -- this is exactly the high-cardinality
        # collapse the fold exists for (see analysis.fold's module docstring: 126,174
        # distinct grid_cell values, ~9.8 GB unfolded). `should_fold_group_by` sees
        # only the OUTPUT key names, and `group_by_cols` here is always plain dimension
        # name strings (never the `[True]` literal sentinel -- see
        # `BaseAnalysis._get_group_by_columns`), so no filtering is needed before adding
        # grid_cell to build the fold-routing key.
        key_names = [*group_by_cols, "grid_cell"]
        if should_fold_group_by(
            key_names, has_selective_filter=self._has_selective_dimension_filter()
        ):
            source_lf = self.lf
            try:
                collapsed = fold_group_by(
                    lambda: source_lf, [*group_by_cols, "grid_cell"], self.options.group_aggregate
                )
            except FoldError as exc:
                self._error(str(exc))
            self.lf = collapsed.lazy()
        else:
            self.lf = self.lf.group_by(*group_by_cols, "grid_cell").agg(
                getattr(pl.col("value"), self.options.group_aggregate)()
            )

    async def _apply_transform_params_impl(self):
        """Apply transform and extract parameters to the LazyFrame.

        Declared `async` (with no internal `await`) purely to match the one hook name
        `BaseAnalysis._apply_transform_params` calls on both subclasses -- see that
        module's docstring for why the vector/raster pair collapses to a single hook.
        """

        if not self._validate_summarize_params():
            return

        # Generate column expressions.
        crosswalk_name = f"{self.grid.name}__{self.options.summarize_by}.parquet"
        crosswalk_path = config.CROSSWALKS_PATH / crosswalk_name

        # scan_parquet is lazy and won't raise for a missing file until collection,
        # so check up front to report the missing crosswalk as a 400.
        if not crosswalk_path.exists():
            msg = f"No crosswalk from {self.grid.name} to {self.options.summarize_by}"
            self._error(msg)

        crosswalk_lf = pl.scan_parquet(str(crosswalk_path))

        if self.grid_cell_ids:
            crosswalk_lf = crosswalk_lf.filter(pl.col("src_id").is_in([int(g) for g in self.grid_cell_ids]))

        if self.summarize_grid_cell_ids:
            crosswalk_lf = crosswalk_lf.filter(pl.col("dst_id").is_in(self.summarize_grid_cell_ids))

        dim_cols = cs.expand_selector(self.lf, DIMENSION_COLUMNS)

        if self.options.summarize_aggregate == "count_by_value":
            # No `scale_factor` multiply here (see _reject_calculate_with_count_by_value's
            # docstring for the same reasoning applied to the crosswalk's own scale
            # factor: rescaling a categorical code before counting it would change which
            # raw values collapse into "the same" category). Group by the destination
            # zone (aliased to `grid_cell`, matching every other summarize output) plus
            # the value itself -- `count_by_value_frame` bounds this group key's
            # cardinality; see its docstring for the math.
            self._reject_calculate_with_count_by_value()
            joined = self.lf.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner").with_columns(
                pl.col("dst_id").alias("grid_cell")
            )
            try:
                self.lf = count_by_value_frame(
                    joined, [*dim_cols, "grid_cell"], "value", COUNT_BY_VALUE_MAX_DISTINCT
                ).with_columns(geographic_unit=pl.lit(self.options.summarize_by))
            except ExpressionError as exc:
                self._error(str(exc))
            return

        # Join the data on grid_cell == src_id, scale the values, then aggregate
        # per destination grid_cell. The join replaces the wide-format horizontal
        # aggregation that was used when each source grid_cell was a column.
        agg_fn = self.options.summarize_aggregate

        def _scale_and_join(chunk):
            return chunk.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner").with_columns(
                value=pl.col("value") * pl.col("scale_factor")
            )

        # The destination zone is aliased to `grid_cell` here too, so the same
        # high-cardinality signal `_collapse_group` uses applies -- and it has to:
        # some crosswalks land on a FINER destination than the source grid (e.g.
        # hydro_compartment_v001 -> clara_grid_cell_v001, 109,854 distinct destination
        # cells), so this reaggregation is not uniformly safe just because most
        # `summarize_by` targets (ecoregion, parish, hydro_compartment, ...) are small.
        # `summarize_grid_cell` narrows destinations the same way `grid_cell` narrows
        # sources, so it's added to the same selective-filter signal here.
        key_names = [*dim_cols, "grid_cell"]
        has_selective_filter = (
            self._has_selective_dimension_filter() or self.options.summarize_grid_cell is not None
        )
        if should_fold_group_by(key_names, has_selective_filter=has_selective_filter):
            source_lf = self.lf
            try:
                collapsed = fold_group_by(
                    lambda: source_lf,
                    [*dim_cols, pl.col("dst_id").alias("grid_cell")],
                    agg_fn,
                    chunk_transform=_scale_and_join,
                )
            except FoldError as exc:
                self._error(str(exc))
            self.lf = collapsed.lazy().with_columns(geographic_unit=pl.lit(self.options.summarize_by))
        else:
            joined = self.lf.join(crosswalk_lf, left_on="grid_cell", right_on="src_id", how="inner").with_columns(
                value=pl.col("value") * pl.col("scale_factor")
            )
            self.lf = (
                joined.group_by(*dim_cols, pl.col("dst_id").alias("grid_cell"))
                .agg(getattr(pl.col("value"), agg_fn)())
                .with_columns(geographic_unit=pl.lit(self.options.summarize_by))
            )

    @asynctiming
    async def get_vector_tile(self, x: int, y: int, z: int) -> bytes | None:
        """Get an MVT tile with dataset values injected into pre-generated geometry."""
        grid_name = self.options.summarize_by or self.grid.name
        tile = mercantile.Tile(x, y, z)

        compressed = read_pmtile(grid_name, tile)
        if compressed is None:
            return None

        # Parse once: get the VectorTile object and boundary feature IDs together.
        # The parsed object is reused for injection, avoiding a second decompress+parse.
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
