"""Tests for the incremental catalog-sync hooks in write_parquet/compact_parquet
(tabular.py) and write_cog/compact_cog (raster.py), plus the underlying shared
logic in cpra.mp.data.catalog that the `cpra.mp.data.cli catalog` CLI also uses.

All data is written under tmp_path-based base_path/catalog_path fixtures (see
conftest.py) -- never the real data/ directory (see repo CLAUDE.md).
"""

import datetime

import numpy as np
import polars as pl
import pytest
import rasterio

from cpra.mp.data.api.models import DatasetDimensions, FilterDimensions
from cpra.mp.data.catalog import CatalogQuery, contiguous_runs, raster_catalog_from_paths, unify_dtypes
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

    result = compact_cog(base_path=base_path, catalog_path=catalog_path, **RASTER_KWARGS)

    tifs = list(base_path.rglob("*.tif"))
    assert len(tifs) == 1
    with rasterio.open(tifs[0]) as ds:
        assert ds.count == 3

    # compact_cog reports what it wrote so callers needn't re-scan the output tree.
    assert result["outputs"] == tifs
    assert result["sources_merged"] == 3

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

    # Exercises the same full-rescan path the `cpra.mp.data.cli catalog build`
    # command uses, independent of the incremental write_cog hook above.
    catalog = unify_dtypes(raster_catalog_from_paths(base_path, tif_paths))
    assert catalog.height == 1
    row = catalog.row(0, named=True)
    assert row["model"] == "icm_v2023.0.0"


def test_contiguous_runs_treats_consecutive_months_as_one_run():
    """`period_of`'s monthly branch (unlike the daily/annual branches, exercised above
    via write_parquet/write_cog for annual and via TestDailyCompression-style daily data
    elsewhere in the suite) has no monthly variable in the real dimension set to drive it
    through write_parquet, so it's exercised directly on `contiguous_runs`: three
    consecutive months are one run, and a one-month gap starts a new one."""
    days = pl.DataFrame(
        {
            "grid_cell": [1, 1, 1, 1],
            "calendar_day": pl.Series(
                ["2030-01-15", "2030-02-15", "2030-03-15", "2030-05-15"], dtype=pl.String
            ).str.to_date(),
        }
    )

    runs = contiguous_runs(days, ["grid_cell"], "calendar_day", "monthly")

    ranges = sorted(zip(runs["calendar_day_min"].to_list(), runs["calendar_day_max"].to_list()))
    assert ranges == [
        (datetime.date(2030, 1, 15), datetime.date(2030, 3, 15)),
        (datetime.date(2030, 5, 15), datetime.date(2030, 5, 15)),
    ]


class TestCatalogQueryPreloadedCatalog:
    """`CatalogQuery(..., catalog=<frame>)` skips the disk read entirely -- used by the
    CLI/tests per the class docstring. These craft the preloaded frame directly rather
    than through write_parquet/write_cog, which also makes it easy to construct a
    required-but-entirely-null column (see the second test) that a real write would
    never produce.
    """

    def _catalog_frame(self, percentile=50):
        return pl.DataFrame(
            {
                "variable": ["ssf"],
                "time_unit": ["annual"],
                "grid": ["ecoregion_v001"],
                "model_group_id": pl.Series([500], dtype=pl.Int32),
                "scenario_id": pl.Series([7], dtype=pl.Int32),
                "percentile": pl.Series([percentile], dtype=pl.Int8),
                "model": ["icm_v2023.0.0"],
                "calendar_year_min": pl.Series([2020], dtype=pl.Int32),
                "calendar_year_max": pl.Series([2020], dtype=pl.Int32),
                "calendar_year_count": pl.Series([1], dtype=pl.Int64),
                "calendar_day_min": pl.Series([None], dtype=pl.Date),
                "calendar_day_max": pl.Series([None], dtype=pl.Date),
                "calendar_day_count": pl.Series([None], dtype=pl.Int64),
                "data_type": ["tabular"],
            }
        )

    def test_preloaded_catalog_is_used_without_touching_catalog_path(self, tmp_path):
        """`catalog_path` points at a directory that doesn't even exist -- proof the
        preloaded frame short-circuits `_load_catalog`'s disk read rather than merely
        taking priority over it."""
        dd = DatasetDimensions(variable="ssf", grid="ecoregion_v001", time_unit="annual")
        query = CatalogQuery(dd, FilterDimensions(), catalog=self._catalog_frame(), catalog_path=tmp_path / "missing")

        result = query.execute()

        assert result["results"]["scenario_id"] == [7]
        assert result["results"]["percentile"] == [50]

    def test_required_dimension_with_no_values_is_excluded_from_choices(self, tmp_path):
        """`percentile` is required for `ssf`, but if every row's value for it is null
        (e.g. a legacy/partial catalog row), CatalogQuery skips it rather than reporting
        an empty choice list or crashing."""
        dd = DatasetDimensions(variable="ssf", grid="ecoregion_v001", time_unit="annual")
        query = CatalogQuery(
            dd, FilterDimensions(), catalog=self._catalog_frame(percentile=None), catalog_path=tmp_path / "missing"
        )

        result = query.execute()

        assert "percentile" not in result["choices"]
        assert "percentile" not in result["results"]
        assert result["results"]["scenario_id"] == [7]


class TestCatalogQueryRepeatedExecute:
    """`CatalogQuery` keeps `_choices`/`_results` as instance state rather than resetting
    it in `execute()`, so calling `execute()` a second time on the same instance re-visits
    `_update` for every dimension name and merges into the choices already recorded from
    the first call -- the only way `_update`'s "already seen this dimension" branch (and
    the `ValueError` it can raise) is reachable, since a single `execute()` call visits
    each dimension name exactly once (see tests/api/test_catalog_query.py's
    TestChoiceIntersection docstring, which notes that branch is otherwise dead)."""

    def _write(self, base_path, catalog_path, scenario_id):
        df = pl.DataFrame(
            {"grid_cell": [1], "calendar_year": [2020], "value": [1.0]},
            schema={"grid_cell": pl.Int32, "calendar_year": pl.Int32, "value": pl.Float32},
        )
        write_parquet(
            df,
            base_path=base_path,
            catalog_path=catalog_path,
            variable="sal",
            grid="hydro_compartment_v001",
            time_unit="annual",
            model_group_id=500,
            scenario_id=scenario_id,
            aggregate_type="mean",
            model="icm_v2023.0.0",
        )

    def test_second_call_merges_into_the_first_when_values_overlap(self, base_path, catalog_path):
        self._write(base_path, catalog_path, scenario_id=7)
        dd = DatasetDimensions(variable="sal", grid="hydro_compartment_v001", time_unit="annual")
        query = CatalogQuery(dd, FilterDimensions(), catalog_path=catalog_path)

        first = query.execute()
        second = query.execute()

        assert first["choices"]["scenario_id"] == [7]
        assert second["choices"]["scenario_id"] == [7]

    def test_second_call_raises_when_the_catalog_changed_with_no_overlap(self, base_path, catalog_path):
        self._write(base_path, catalog_path, scenario_id=7)
        dd = DatasetDimensions(variable="sal", grid="hydro_compartment_v001", time_unit="annual")
        query = CatalogQuery(dd, FilterDimensions(), catalog_path=catalog_path)
        query.execute()

        # Rewrite the on-disk catalog so scenario_id no longer overlaps what the first
        # execute() call already recorded as `_choices["scenario_id"]`.
        for f in catalog_path.rglob("*.parquet"):
            pl.read_parquet(f).with_columns(pl.lit(99).cast(pl.Int32).alias("scenario_id")).write_parquet(f)

        with pytest.raises(ValueError, match="No common values for dimension scenario_id"):
            query.execute()
