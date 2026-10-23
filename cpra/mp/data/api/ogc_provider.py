import pyogrio
from pygeoapi.provider.base import (
    BaseProvider,
    ProviderInvalidQueryError,
    ProviderItemNotFoundError,
)
from pyproj import Transformer
from shapely.geometry import mapping

from cpra.mp.data.analysis.execution.execute import execute
from cpra.mp.data.analysis.execution.plan import compile_plan
from cpra.mp.data.analysis.request.errors import RequestError
from cpra.mp.data.api.unified import decode_request_sync
from cpra.mp.data import config
from cpra.mp.data.grids import GRIDS

# Query params that belong to the OGC API itself rather than to the data request. They
# are dropped before the rest of the query string is decoded as a portal data request.
_OGC_PARAMS = {
    "f", "lang", "limit", "offset", "startindex", "bbox", "bbox-crs", "crs", "datetime",
    "resulttype", "skipgeometry", "properties", "sortby", "filter", "filter-lang",
    "filter-crs", "q",
}


def _invalid(message):
    """A 400 whose message reaches the client (pygeoapi hides it by default)."""
    return ProviderInvalidQueryError(message, user_msg=message)


def strip_ogc_params(items):
    """`(key, value)` query pairs minus the OGC API's own parameters."""
    return [(k, v) for k, v in items if k.lower() not in _OGC_PARAMS]


def data_request_items(args):
    """The data request carried in a Flask request's query string."""
    return strip_ogc_params(args.items(multi=True))


class MasterPlanDataProvider(BaseProvider):
    """OGC API - Features provider for CPRA Master Plan vector grid data.

    Geometry comes from the grid's FlatGeoBuf file, read with pyogrio. Each feature's
    `value` is the result of the data request carried in the query string -- the same
    parameters the portal puts in its URL -- run through the analysis engine, so a link
    exported from the portal loads the same view in a GIS.
    """

    def get_fields(self):
        return {"id": {"type": "integer"}, "label": {"type": "string"}}

    def query(self, offset=0, limit=10, resulttype="results", bbox=[], skip_geometry=False,
              **kwargs):
        path, values = self._prepare()
        storage_bbox = _to_storage_bbox(bbox, path)
        matched = _count(path, storage_bbox)

        if resulttype == "hits":
            return {"type": "FeatureCollection", "numberMatched": matched, "features": []}

        gdf = pyogrio.read_dataframe(
            path,
            bbox=storage_bbox,
            skip_features=offset,
            max_features=limit,
            read_geometry=not skip_geometry,
        )
        return {
            "type": "FeatureCollection",
            "numberMatched": matched,
            "features": _to_features(gdf, values, skip_geometry),
        }

    def get(self, identifier, **kwargs):
        path, values = self._prepare()
        try:
            where = f"id = {int(identifier)}"
        except ValueError:
            raise ProviderItemNotFoundError(f"Feature not found: {identifier}")
        gdf = pyogrio.read_dataframe(path, where=where)
        if gdf.empty:
            raise ProviderItemNotFoundError(f"Feature not found: {identifier}")
        return _to_features(gdf, values, False)[0]

    def _prepare(self):
        """Run the request's data query.

        Returns (fgb path, {grid_cell id: value}).
        """
        from flask import request

        try:
            decoded = decode_request_sync(data_request_items(request.args), format="parquet")
            grid = decoded.descriptor.grid
            if GRIDS[grid].grid_type != "vector":
                raise _invalid(
                    f"Grid '{grid}' is a raster grid and is not available "
                    "through the OGC Features API."
                )
            df = execute(compile_plan(decoded.pipeline)).collect()
        except RequestError as exc:
            raise _invalid(str(exc))

        if df["grid_cell"].n_unique() != len(df):
            raise _invalid(
                "The selection returns more than one value per grid cell; "
                "narrow it so each dimension has a single value."
            )
        values = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))
        return str(config.GRIDS_PATH / f"{grid}.fgb"), values


def _to_storage_bbox(bbox, path):
    """Transform a CRS84 bbox from the request into the FlatGeoBuf's CRS."""
    if not bbox:
        return None
    storage_crs = pyogrio.read_info(path)["crs"]
    return Transformer.from_crs("OGC:CRS84", storage_crs, always_xy=True).transform_bounds(
        *bbox[:4]
    )


def _count(path, storage_bbox):
    if storage_bbox is None:
        return pyogrio.read_info(path)["features"]
    return len(
        pyogrio.read_dataframe(path, bbox=storage_bbox, columns=["id"], read_geometry=False)
    )


def _to_features(gdf, values, skip_geometry):
    """Convert a GeoDataFrame in the storage CRS to GeoJSON features in CRS84 carrying
    each cell's `value`."""
    if not skip_geometry:
        gdf = gdf.to_crs("OGC:CRS84")
    geometries = [None] * len(gdf) if skip_geometry else [mapping(g) for g in gdf.geometry]
    rows = gdf.drop(columns="geometry", errors="ignore").to_dict("records")
    return [
        {
            "type": "Feature",
            "id": int(row["id"]),
            "properties": {**row, "value": values.get(int(row["id"]))},
            "geometry": geometry,
        }
        for row, geometry in zip(rows, geometries)
    ]
