import pyogrio
from pygeoapi.provider.base import (
    BaseProvider,
    ProviderInvalidQueryError,
    ProviderItemNotFoundError,
)
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


# The FlatGeoBuf grids are stored, and always served, in this CRS; `ogc_config.yaml`'s
# `storage_crs` must match.
STORAGE_CRS_URI = "http://www.opengis.net/def/crs/EPSG/0/26915"

EXPORT_HINT = (
    "This collection serves one view of the data at a time, so each request must carry "
    "that view as query parameters (variable, time_unit, grid and the other dimensions). "
    "Use Export in the Master Plan Data Portal (https://mpdp.coastal.la.gov) to get a "
    "link or QGIS layer file for the view you are looking at."
)


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
        # pygeoapi has already moved the bbox into the collection's storage CRS.
        storage_bbox = tuple(bbox[:4]) if bbox else None
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
            # GeoJSON assumes CRS84 unless told otherwise; OGR and QGIS read this member.
            "crs": {"type": "name", "properties": {"name": STORAGE_CRS_URI}},
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

        crs = request.args.get("crs")
        if crs is not None and crs != STORAGE_CRS_URI:
            raise _invalid(
                f"CRS {crs!r} is not supported. This collection is only served in "
                f"{STORAGE_CRS_URI}."
            )
        items = data_request_items(request.args)
        if not items:
            raise _invalid(EXPORT_HINT)

        try:
            decoded = decode_request_sync(items, format="parquet")
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


def _count(path, storage_bbox):
    if storage_bbox is None:
        return pyogrio.read_info(path)["features"]
    return len(
        pyogrio.read_dataframe(path, bbox=storage_bbox, columns=["id"], read_geometry=False)
    )


def _to_features(gdf, values, skip_geometry):
    """Convert a GeoDataFrame to GeoJSON features carrying each cell's `value`. Geometry
    stays in the storage CRS; pygeoapi's `crs_transform_spec` is deliberately ignored."""
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
