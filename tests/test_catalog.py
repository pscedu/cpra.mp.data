"""Tests for the incremental catalog-sync hooks in write_parquet/compact_parquet
(tabular.py) and write_cog/compact_cog (raster.py), plus the underlying shared
logic in cpra.mp.data.catalog that scripts/build_catalog.py also uses.

All data is written under tmp_path-based base_path/catalog_path fixtures (see
conftest.py) -- never the real data/ directory (see repo CLAUDE.md).
"""

import numpy as np
import polars as pl
import rasterio

from cpra.mp.data.catalog import raster_catalog_from_paths, unify_dtypes
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.raster import _cog_profile, compact_cog, write_cog
from cpra.mp.data.tabular import compact_parquet, write_parquet

TABULAR_KWARGS = dict(
    variable="sal",
    grid="hydro_compartment_v001",
    time_unit="annual",
    model_group_id=500,
    scenario_id=7,
    aggregate_type="mean",
    model="icm_v2023.0.0",
)

RASTER_KWARGS = dict(
    variable="hsi",
    grid="veg_grid_cell_v001",
    time_unit="annual",
    model_group_id=500,
    scenario_id=7,
    species_code="allig",
    model="icm_v2023.0.0",
)


def _tabular_data(years, grid_cells=(1, 2)):
    rows = [(gc, yr) for gc in grid_cells for yr in years]
    return pl.DataFrame(
        {
            "grid_cell": [r[0] for r in rows],
            "calendar_year": [r[1] for r in rows],
            "value": [1.0] * len(rows),
        },
        schema={"grid_cell": pl.Int32, "calendar_year": pl.Int32, "value": pl.Float32},
    )


def _raster_array():
    grid = GRIDS[RASTER_KWARGS["grid"]]
    return np.zeros((grid.height, grid.width), dtype="float32")


def _read_catalog(catalog_path):
    return pl.read_parquet(catalog_path, hive_partitioning=True)


def test_write_parquet_appends_catalog_row(base_path, catalog_path):
    write_parquet(_tabular_data([2020, 2021, 2022]), base_path=base_path, catalog_path=catalog_path, **TABULAR_KWARGS)

    catalog = _read_catalog(catalog_path)
    assert catalog.height == 1
    row = catalog.row(0, named=True)
    assert row["data_type"] == "tabular"
    assert (row["calendar_year_min"], row["calendar_year_max"], row["calendar_year_count"]) == (2020, 2022, 3)


def test_write_parquet_gap_then_compact_preserves_gap(base_path, catalog_path):
    write_parquet(_tabular_data([2020, 2021, 2022]), base_path=base_path, catalog_path=catalog_path, **TABULAR_KWARGS)
    write_parquet(_tabular_data([2025, 2026, 2027]), base_path=base_path, catalog_path=catalog_path, **TABULAR_KWARGS)

    assert _read_catalog(catalog_path).height == 2

    compact_parquet(base_path=base_path, catalog_path=catalog_path, **TABULAR_KWARGS)

    catalog = _read_catalog(catalog_path)
    assert catalog.height == 2
    ranges = sorted(zip(catalog["calendar_year_min"].to_list(), catalog["calendar_year_max"].to_list()))
    assert ranges == [(2020, 2022), (2025, 2027)]


def test_write_parquet_fill_gap_then_compact_merges(base_path, catalog_path):
    write_parquet(_tabular_data([2020, 2021, 2022]), base_path=base_path, catalog_path=catalog_path, **TABULAR_KWARGS)
    write_parquet(_tabular_data([2025, 2026, 2027]), base_path=base_path, catalog_path=catalog_path, **TABULAR_KWARGS)
    write_parquet(_tabular_data([2023, 2024]), base_path=base_path, catalog_path=catalog_path, **TABULAR_KWARGS)

    compact_parquet(base_path=base_path, catalog_path=catalog_path, **TABULAR_KWARGS)

    catalog = _read_catalog(catalog_path)
    assert catalog.height == 1
    row = catalog.row(0, named=True)
    assert (row["calendar_year_min"], row["calendar_year_max"], row["calendar_year_count"]) == (2020, 2027, 8)


def test_write_cog_appends_catalog_row_with_metadata_dim(base_path, catalog_path):
    write_cog(_raster_array(), base_path=base_path, catalog_path=catalog_path, calendar_year=2020, **RASTER_KWARGS)

    catalog = _read_catalog(catalog_path)
    assert catalog.height == 1
    row = catalog.row(0, named=True)
    assert row["data_type"] == "raster"
    assert row["model"] == "icm_v2023.0.0"
    assert (row["calendar_year_min"], row["calendar_year_max"], row["calendar_year_count"]) == (2020, 2020, 1)


def test_compact_cog_merges_files_and_coalesces_catalog(base_path, catalog_path):
    for year in (2020, 2021, 2022):
        write_cog(_raster_array(), base_path=base_path, catalog_path=catalog_path, calendar_year=year, **RASTER_KWARGS)

    compact_cog(base_path=base_path, catalog_path=catalog_path, **RASTER_KWARGS)

    tifs = list(base_path.rglob("*.tif"))
    assert len(tifs) == 1
    with rasterio.open(tifs[0]) as ds:
        assert ds.count == 3

    catalog = _read_catalog(catalog_path)
    assert catalog.height == 1
    row = catalog.row(0, named=True)
    assert (row["calendar_year_min"], row["calendar_year_max"], row["calendar_year_count"]) == (2020, 2022, 3)
    assert row["model"] == "icm_v2023.0.0"


def test_compact_cog_backfills_legacy_files_into_catalog(base_path, catalog_path):
    grid = GRIDS[RASTER_KWARGS["grid"]]
    directory = (
        base_path
        / f"variable={RASTER_KWARGS['variable']}"
        / f"time_unit={RASTER_KWARGS['time_unit']}"
        / f"grid={RASTER_KWARGS['grid']}"
        / f"model_group_id={RASTER_KWARGS['model_group_id']}"
        / f"scenario_id={RASTER_KWARGS['scenario_id']}"
        / f"species_code={RASTER_KWARGS['species_code']}"
    )
    directory.mkdir(parents=True)

    # A pre-migration/legacy COG: calendar value stored as a dataset tag (not a band
    # description) -- write_cog never produced files like this, but old on-disk data
    # does, and it never went through write_cog's catalog hook, so no catalog row for
    # it exists yet.
    profile = _cog_profile(grid, "float32")
    with rasterio.open(directory / "legacy.tif", "w", **profile) as dst:
        dst.update_tags(calendar_year=2019, **{k: v for k, v in RASTER_KWARGS.items() if k != "grid"})
        dst.write(np.zeros((grid.height, grid.width), dtype="float32"), 1)

    assert not catalog_path.exists()

    compact_cog(base_path=base_path, catalog_path=catalog_path, **RASTER_KWARGS)

    catalog = _read_catalog(catalog_path)
    assert catalog.height == 1
    row = catalog.row(0, named=True)
    assert row["model"] == "icm_v2023.0.0"
    assert (row["calendar_year_min"], row["calendar_year_max"], row["calendar_year_count"]) == (2019, 2019, 1)


def test_catalog_path_none_disables_catalog_writes(base_path, tmp_path):
    write_parquet(_tabular_data([2020]), base_path=base_path, catalog_path=None, **TABULAR_KWARGS)
    write_cog(_raster_array(), base_path=base_path, catalog_path=None, calendar_year=2020, **RASTER_KWARGS)
    compact_parquet(base_path=base_path, catalog_path=None, **TABULAR_KWARGS)
    compact_cog(base_path=base_path, catalog_path=None, **RASTER_KWARGS)

    assert {p.name for p in tmp_path.iterdir()} == {"data"}


def test_raster_catalog_from_paths_includes_metadata_dim(base_path):
    write_cog(_raster_array(), base_path=base_path, catalog_path=None, calendar_year=2020, **RASTER_KWARGS)
    tif_paths = list(base_path.rglob("*.tif"))
    assert len(tif_paths) == 1

    # Exercises the same full-rescan path scripts/build_catalog.py's `build`
    # command uses, independent of the incremental write_cog hook above.
    catalog = unify_dtypes(raster_catalog_from_paths(base_path, tif_paths))
    assert catalog.height == 1
    row = catalog.row(0, named=True)
    assert row["model"] == "icm_v2023.0.0"
