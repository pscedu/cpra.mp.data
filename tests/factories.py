"""Builders for the smallest datasets that are still valid under the real structure CSVs.

Everything here uses genuine variables, grids, and dimensions from
``cpra/mp/data/structure/`` — no fake dimensions and no monkeypatched ``DIMENSIONS``, so
the fixtures exercise the same validation rules production data does.

Two shapes carry most of the suite:

* **annual / vector** — ``ssf`` on ``ecoregion_v001`` (32 geography columns, ``percentile``
  required). Small enough that per-column assertions stay readable, unlike
  ``hydro_compartment_v001``'s 1,657 columns.
* **daily / vector** — ``sal`` on ``hydro_compartment_v001`` (``aggregate_type`` required).
  The only shape that exercises partition-type dimensions, because ``Dimension.partition``
  is ``(pl.col("time_unit") == "daily").any()`` for them.

The raster shape is ``vct`` on ``veg_grid_cell_v001`` (365 x 1052 = ~384k pixels), the
smallest real raster grid.
"""

import gzip

import numpy as np
import polars as pl

from cpra.mp.data.grids import GRIDS
from cpra.mp.data.raster import write_cog
from cpra.mp.data.tabular import write_parquet

# Annual vector dataset: ssf on ecoregion_v001, which requires `percentile`.
ANNUAL_KWARGS = dict(
    variable="ssf",
    grid="ecoregion_v001",
    time_unit="annual",
    model_group_id=500,
    scenario_id=7,
    percentile=50,
    model="icm_v2023.0.0",
)

# Daily vector dataset: sal on hydro_compartment_v001, which requires `aggregate_type`.
DAILY_KWARGS = dict(
    variable="sal",
    grid="hydro_compartment_v001",
    time_unit="daily",
    model_group_id=500,
    scenario_id=7,
    aggregate_type="mean",
    model="icm_v2023.0.0",
)

# Annual raster dataset: vct on veg_grid_cell_v001, which requires no extra dimensions.
RASTER_KWARGS = dict(
    variable="vct",
    grid="veg_grid_cell_v001",
    time_unit="annual",
    model_group_id=500,
    scenario_id=7,
    model="icm_v2023.0.0",
)

ECOREGION_IDS = tuple(range(1, 33))


def annual_frame(years=(2030, 2040), geo_ids=ECOREGION_IDS, value_fn=None, **overrides):
    """Wide annual DataFrame for ssf / ecoregion_v001, valid for write_parquet.

    ``value_fn(year, geo_id)`` supplies cell values; it defaults to ``year + geo_id / 100``
    so every cell is distinguishable in assertions.
    """
    if value_fn is None:

        def value_fn(year, geo_id):
            return year + geo_id / 100

    kwargs = {**ANNUAL_KWARGS, **overrides}
    years = list(years)

    columns = {
        "variable": pl.Series([kwargs["variable"]] * len(years), dtype=pl.String),
        "grid": pl.Series([kwargs["grid"]] * len(years), dtype=pl.String),
        "time_unit": pl.Series([kwargs["time_unit"]] * len(years), dtype=pl.String),
        "model_group_id": pl.Series([kwargs["model_group_id"]] * len(years), dtype=pl.Int32),
        "scenario_id": pl.Series([kwargs["scenario_id"]] * len(years), dtype=pl.Int32),
        "percentile": pl.Series([kwargs["percentile"]] * len(years), dtype=pl.Int8),
        "calendar_year": pl.Series(years, dtype=pl.Int32),
        "model": pl.Series([kwargs["model"]] * len(years), dtype=pl.String),
    }
    for geo_id in geo_ids:
        columns[str(geo_id)] = pl.Series(
            [value_fn(year, geo_id) for year in years], dtype=pl.Float32
        )

    return pl.DataFrame(columns)


def daily_frame(days=None, geo_ids=(1, 2, 3), **overrides):
    """Wide daily DataFrame for sal / hydro_compartment_v001.

    Intended for ``write_parquet(..., validate=False)``: it deliberately carries only a
    few geography columns rather than the grid's full 1,657, so it is useful for
    partition-layout tests but not for column-count validation.
    """
    import datetime

    if days is None:
        days = [datetime.date(2030, 1, 1), datetime.date(2030, 1, 2)]

    kwargs = {**DAILY_KWARGS, **overrides}
    days = list(days)

    columns = {
        "variable": pl.Series([kwargs["variable"]] * len(days), dtype=pl.String),
        "grid": pl.Series([kwargs["grid"]] * len(days), dtype=pl.String),
        "time_unit": pl.Series([kwargs["time_unit"]] * len(days), dtype=pl.String),
        "model_group_id": pl.Series([kwargs["model_group_id"]] * len(days), dtype=pl.Int32),
        "scenario_id": pl.Series([kwargs["scenario_id"]] * len(days), dtype=pl.Int32),
        "aggregate_type": pl.Series([kwargs["aggregate_type"]] * len(days), dtype=pl.String),
        "calendar_day": pl.Series(days, dtype=pl.Date),
        "model": pl.Series([kwargs["model"]] * len(days), dtype=pl.String),
    }
    for geo_id in geo_ids:
        columns[str(geo_id)] = pl.Series([float(geo_id)] * len(days), dtype=pl.Float32)

    return pl.DataFrame(columns)


def write_annual_dataset(base_path, years=(2030, 2040), **overrides):
    """Write an annual ssf dataset and return the kwargs needed to read it back."""
    df = annual_frame(years=years, **overrides)
    write_parquet(df, base_path=base_path)
    return {**ANNUAL_KWARGS, **overrides}


def write_omar_dataset(base_path, years=(2030, 2040), ffibs_type="pl_bf", value_fn=None, **overrides):
    """Write an annual omar dataset — the SECOND variable on ``ecoregion_v001``, so a test
    can put two different variables on one grid without leaving the real structure CSVs.

    Same cells and years as :func:`write_annual_dataset`, so the two line up row for row
    on ``(calendar_year, grid_cell)``; ``value_fn`` defaults to the same one too, so a
    test that wants the two variables to hold DIFFERENT numbers passes its own. Two
    differences from ``ssf`` come from ``variables.csv`` and are deliberately not smoothed
    over: ``omar`` is ``Float64`` (hence the cast), and it requires ``ffibs_type`` on top
    of ``percentile``, so its rows carry one dimension column ``ssf``'s do not.
    """
    df = annual_frame(years=years, value_fn=value_fn, variable="omar", **overrides).with_columns(
        pl.lit(ffibs_type).alias("ffibs_type"),
        *[pl.col(str(geo_id)).cast(pl.Float64) for geo_id in ECOREGION_IDS],
    )
    write_parquet(df, base_path=base_path)
    return {**ANNUAL_KWARGS, "variable": "omar", "ffibs_type": ffibs_type, **overrides}


def raster_array(fill=None, grid_name=RASTER_KWARGS["grid"], dtype="int32"):
    """A pixel array matching the shape of a raster grid."""
    grid = GRIDS[grid_name]
    if fill is None:
        return np.arange(grid.height * grid.width, dtype=dtype).reshape(grid.height, grid.width)
    return np.full((grid.height, grid.width), fill, dtype=dtype)


def write_raster_dataset(base_path, years=(2030, 2040), fill_fn=None, data_fn=None, **overrides):
    """Write one COG per year and return the kwargs needed to read them back.

    ``data_fn(year)``, when given, returns the full pixel array for that year directly
    (e.g. to plant nodata pixels at specific locations), taking precedence over
    ``fill_fn``'s uniform fill.
    """
    kwargs = {**RASTER_KWARGS, **overrides}
    for year in years:
        if data_fn is not None:
            data = data_fn(year)
        else:
            data = raster_array(fill=(fill_fn(year) if fill_fn else year))
        write_cog(data, base_path=base_path, calendar_year=year, **kwargs)
    return kwargs


def write_ffibs_coverage_dataset(base_path, values_by_year, **overrides):
    """``ffibs_coverage_ratio`` / ``veg_grid_cell_v001``, one uniformly-filled COG per
    ``values_by_year`` entry.

    The raster half of the marsh-coverage-plus-salinity pair: the two variables the
    ``exit-criterion`` named analysis hardcodes, and the smallest real dataset that puts a
    raster grid and a vector grid on the same pair of years. ``ffibs_type`` is required by
    ``variables.csv`` and defaults to the ``pl_fm`` value that analysis selects.
    """
    kwargs = {**RASTER_KWARGS, "variable": "ffibs_coverage_ratio", "ffibs_type": "pl_fm", **overrides}
    for year, value in values_by_year.items():
        data = raster_array(fill=value, grid_name=kwargs["grid"], dtype="float32")
        write_cog(data, base_path=base_path, calendar_year=year, **kwargs)
    return kwargs


def write_sal_annual_dataset(base_path, values_by_year, geo_ids=(1,), **overrides):
    """``sal`` / ``hydro_compartment_v001``, ANNUAL rather than :func:`daily_frame`'s
    daily, with one value per year per cell.

    The vector half of the pair above. ``hydro_compartment_v001`` has 1,657 columns, so
    this writes only ``geo_ids`` and passes ``validate=False`` for the same reason
    :func:`daily_frame` does.
    """
    kwargs = {**DAILY_KWARGS, "time_unit": "annual", "aggregate_type": "2_wk_max", **overrides}
    years = sorted(values_by_year)
    columns = {
        "variable": pl.Series([kwargs["variable"]] * len(years), dtype=pl.String),
        "grid": pl.Series([kwargs["grid"]] * len(years), dtype=pl.String),
        "time_unit": pl.Series([kwargs["time_unit"]] * len(years), dtype=pl.String),
        "model_group_id": pl.Series([kwargs["model_group_id"]] * len(years), dtype=pl.Int32),
        "scenario_id": pl.Series([kwargs["scenario_id"]] * len(years), dtype=pl.Int32),
        "aggregate_type": pl.Series([kwargs["aggregate_type"]] * len(years), dtype=pl.String),
        "calendar_year": pl.Series(years, dtype=pl.Int32),
        "model": pl.Series([kwargs["model"]] * len(years), dtype=pl.String),
    }
    for geo_id in geo_ids:
        columns[str(geo_id)] = pl.Series([values_by_year[y] for y in years], dtype=pl.Float32)

    write_parquet(pl.DataFrame(columns), base_path=base_path, validate=False)
    return kwargs


def write_vector_crosswalk(crosswalk_path, src_grid, dst_grid, mapping):
    """Write a src->dst crosswalk Parquet.

    ``mapping`` is an iterable of ``(src_id, dst_id, scale_factor)`` triples, matching
    the schema ``analysis.io.crosswalk_read.scan_crosswalk`` returns and
    ``analysis.execution.project`` joins against.
    """
    path = crosswalk_path / f"{src_grid}__{dst_grid}.parquet"
    pl.DataFrame(
        {
            "src_id": pl.Series([m[0] for m in mapping], dtype=pl.Int32),
            "dst_id": pl.Series([m[1] for m in mapping], dtype=pl.Int32),
            "scale_factor": pl.Series([m[2] for m in mapping], dtype=pl.Float64),
        }
    ).write_parquet(path)
    return path


def write_raster_crosswalk(crosswalk_path, src_grid, dst_grid, zones):
    """Write a zone-ID crosswalk COG with a `dst_ids` tag.

    Mirrors what ``python -m cpra.mp.data.cli crosswalk`` emits for a raster source: a
    single-band raster of destination zone IDs plus a comma-separated ``dst_ids`` tag
    listing the zones present.
    """
    import rasterio as rio
    from rasterio.transform import from_bounds

    grid = GRIDS[src_grid]
    path = crosswalk_path / f"{src_grid}__{dst_grid}.tif"
    dst_ids = sorted({int(z) for z in np.unique(zones) if z > 0})

    profile = {
        "count": 1,
        "crs": grid.crs,
        "driver": "GTiff",
        "dtype": "int32",
        "height": grid.height,
        "nodata": 0,
        "tiled": True,
        "transform": from_bounds(*grid.bounds, width=grid.width, height=grid.height),
        "width": grid.width,
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(zones.astype("int32"), 1)
        dst.update_tags(dst_ids=",".join(str(i) for i in dst_ids))


def write_grid_geometry(geometry_path, grid_name, features):
    """Write a FlatGeoBuf geometry file for a vector grid.

    ``features`` is an iterable of ``(id, geometry)`` pairs of shapely geometries.
    Mirrors the real files in ``config.GRIDS_PATH``: an ``id`` field plus geometry, in
    the grid's own CRS.
    """
    import geopandas as gpd

    grid = GRIDS[grid_name]
    path = geometry_path / f"{grid_name}.fgb"
    ids, geometries = zip(*features)
    gpd.GeoDataFrame({"id": list(ids)}, geometry=list(geometries), crs=grid.crs).to_file(
        path, driver="FlatGeobuf"
    )
    return path


def write_pmtiles(tiles_path, grid_name, tile, feature_ids):
    """Write a single-tile PMTiles archive with a `boundary` layer.

    Each feature gets a trivial point geometry; only the feature IDs matter, since
    ``inject_and_serialize`` never decodes geometry.
    """
    from mapbox_vector_tile.Mapbox.vector_tile_pb2 import tile as VectorTile
    from pmtiles.tile import Compression, TileType
    from pmtiles.writer import Writer

    vt = VectorTile()
    layer = vt.layers.add()
    layer.name = "boundary"
    layer.version = 2
    layer.extent = 4096
    for feature_id in feature_ids:
        feature = layer.features.add()
        feature.id = feature_id
        feature.type = 1  # POINT
        # MoveTo(1), then a single zigzag-encoded (x, y) pair.
        feature.geometry.extend([9, 10, 10])

    path = tiles_path / f"{grid_name}.pmtiles"
    with open(path, "wb") as f:
        writer = Writer(f)
        writer.write_tile(
            _zxy_to_tileid(tile.z, tile.x, tile.y), gzip.compress(vt.SerializeToString())
        )
        writer.finalize(
            {
                "tile_type": TileType.MVT,
                "tile_compression": Compression.GZIP,
                "min_zoom": tile.z,
                "max_zoom": tile.z,
                "min_lon_e7": -1800000000,
                "min_lat_e7": -850000000,
                "max_lon_e7": 1800000000,
                "max_lat_e7": 850000000,
                "center_zoom": tile.z,
                "center_lon_e7": 0,
                "center_lat_e7": 0,
            },
            {},
        )
    return path


def _zxy_to_tileid(z, x, y):
    from pmtiles.tile import zxy_to_tileid

    return zxy_to_tileid(z, x, y)
