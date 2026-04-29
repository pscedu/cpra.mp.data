import asyncio

from pygeoapi.provider.base import ProviderQueryError
from pygeoapi.provider.ogr import OGRProvider
from pydantic import ValidationError

from cpra.mp.data.api.dataset import Dataset
from cpra.mp.data.api.models import DatasetDimensions, FilterDimensions, Options
from cpra.mp.data.config import GRIDS_PATH
from cpra.mp.data.dimensions import DATASET_DIMENSIONS, DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.variables import VARIABLES

_DATASET_DIM_NAMES = set(DATASET_DIMENSIONS.keys())
_FILTER_DIM_NAMES = {
    name for name, dim in DIMENSIONS.items()
    if dim.type not in ("dataset", "metadata")
}
_OPTION_FIELDS = set(Options.model_fields.keys())


def parse_ogc_params(args):
    """Parse flask.request.args into (DatasetDimensions, FilterDimensions, Options).

    Raises ProviderQueryError on Pydantic validation failure.
    """
    try:
        dataset_dims = DatasetDimensions(
            **{k: args.get(k) for k in _DATASET_DIM_NAMES if args.get(k) is not None}
        )
    except ValidationError as exc:
        raise ProviderQueryError(str(exc))

    missing = [k for k in _DATASET_DIM_NAMES if getattr(dataset_dims, k, None) is None]
    if missing:
        raise ProviderQueryError(
            f"Missing required query parameters: {', '.join(sorted(missing))}"
        )

    try:
        filter_kwargs = {}
        for k in _FILTER_DIM_NAMES:
            values = args.getlist(k)
            if values:
                filter_kwargs[k] = values
        filter_dims = FilterDimensions(**filter_kwargs)
    except ValidationError as exc:
        raise ProviderQueryError(str(exc))

    try:
        option_kwargs = {k: args.get(k) for k in _OPTION_FIELDS if args.get(k) is not None}
        if "unpivot" in option_kwargs:
            option_kwargs["unpivot"] = option_kwargs["unpivot"].lower() in ("1", "true", "yes")
        options = Options(**option_kwargs)
    except ValidationError as exc:
        raise ProviderQueryError(str(exc))

    return dataset_dims, filter_dims, options


def validate_cardinality(filter_dims, required_dimensions, group_by):
    """Raise ProviderQueryError if any required dimension outside group_by lacks exactly one value."""
    violations = []
    for dim_name in (required_dimensions or []):
        if dim_name in group_by:
            continue
        values = getattr(filter_dims, dim_name, None)
        if not values or len(values) != 1:
            violations.append(dim_name)
    if violations:
        raise ProviderQueryError(
            f"Each required dimension must have exactly one value. "
            f"Missing or ambiguous: {', '.join(violations)}"
        )


def _join_values(features, df, group_by_dims):
    """Attach a 'value' property to each GeoJSON feature from a polars DataFrame.

    df columns: dimension columns (alpha start) + geography ID columns (digit start).
    group_by_dims: list of dimension column names used as group keys.
    """
    value_cols = [c for c in df.columns if c[0].isdigit() or c[0] == "-"]

    if not group_by_dims:
        row = df.row(0, named=True) if len(df) > 0 else {}
        lookup = {col: row.get(col) for col in value_cols}
        for feature in features:
            fid = str(feature["id"])
            feature["properties"]["value"] = lookup.get(fid)
    else:
        lookup = {col: {} for col in value_cols}
        for row in df.iter_rows(named=True):
            group_key = "_".join(str(row[d]) for d in group_by_dims)
            for col in value_cols:
                lookup[col][group_key] = row[col]
        for feature in features:
            fid = str(feature["id"])
            feature["properties"]["value"] = lookup.get(fid)

    return features


class MasterPlanDataProvider(OGRProvider):
    """OGC API - Features provider for CPRA Master Plan vector grid data.

    Reads geometry from FlatGeoBuf files and joins attribute data from Parquet
    via the Dataset class. The grid, variable, time_unit, and all dimension
    filters are supplied as extra query string parameters on each request.
    """

    def query(self, offset=0, limit=10, resulttype="results", bbox=[], datetime_=None,
              properties=[], sortby=[], select_properties=[], skip_geometry=False,
              q=None, filterq=None, **kwargs):
        from flask import request
        args = request.args

        dataset_dims, filter_dims, options = parse_ogc_params(args)
        self._validate_vector_grid(dataset_dims.grid)

        variable = VARIABLES.get(dataset_dims.variable)
        if variable is None:
            raise ProviderQueryError(f"Unknown variable: {dataset_dims.variable}")

        group_by_dims = (
            [d.strip() for d in options.group_by.split(",")]
            if options.group_by else []
        )
        validate_cardinality(filter_dims, variable.required_dimensions, set(group_by_dims))

        self.data_def['source'] = str(GRIDS_PATH / f"{dataset_dims.grid}.fgb")
        self.layer_name = dataset_dims.grid

        dataset = Dataset(dataset_dims, filter_dims, options)
        asyncio.run(dataset.apply_params())
        df = dataset.lf.collect()

        feature_collection = super().query(
            offset=offset, limit=limit, resulttype=resulttype,
            bbox=bbox, datetime_=datetime_, properties=properties,
            sortby=sortby, select_properties=select_properties,
            skip_geometry=skip_geometry, q=q, filterq=filterq,
            **kwargs,
        )

        if resulttype == "hits":
            return feature_collection

        feature_collection["features"] = _join_values(
            feature_collection["features"], df, group_by_dims
        )
        return feature_collection

    def get(self, identifier, **kwargs):
        from flask import request
        args = request.args

        dataset_dims, filter_dims, options = parse_ogc_params(args)
        self._validate_vector_grid(dataset_dims.grid)

        variable = VARIABLES.get(dataset_dims.variable)
        if variable is None:
            raise ProviderQueryError(f"Unknown variable: {dataset_dims.variable}")

        group_by_dims = (
            [d.strip() for d in options.group_by.split(",")]
            if options.group_by else []
        )
        validate_cardinality(filter_dims, variable.required_dimensions, set(group_by_dims))

        self.data_def['source'] = str(GRIDS_PATH / f"{dataset_dims.grid}.fgb")
        self.layer_name = dataset_dims.grid

        dataset = Dataset(dataset_dims, filter_dims, options)
        asyncio.run(dataset.apply_params())
        df = dataset.lf.collect()

        feature = super().get(identifier, **kwargs)
        if feature:
            _join_values([feature], df, group_by_dims)
        return feature

    def _validate_vector_grid(self, grid_name):
        grid = GRIDS.get(grid_name)
        if grid is None:
            raise ProviderQueryError(f"Unknown grid: {grid_name}")
        if grid.grid_type != "vector":
            raise ProviderQueryError(
                f"Grid '{grid_name}' is a raster grid and is not available "
                "through the OGC Features API."
            )
