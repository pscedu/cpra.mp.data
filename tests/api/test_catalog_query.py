"""Tests for CatalogQuery, which resolves a request's available dimension choices and
selected results from the materialized catalog side-table (see cpra/mp/data/catalog.py and
tests/test_catalog.py, which tests how write_parquet/write_cog/the CLI backfill keep that
table in sync) rather than walking the data tree the way main's CatalogQuery did.

The query returns three things: `choices` (everything available), `results` (what was
selected, defaulting when the request was unsatisfiable), and `exact` (whether the request
was honoured in full).

CatalogQuery raises ValueError, not an HTTP-framework exception, so cpra.mp.data.catalog has
no web-framework dependency -- see tests/api/test_catalog_endpoints.py for the API route's
translation of that into a 404.
"""

import datetime

import numpy as np
import polars as pl
import pytest

from cpra.mp.data.api.models import DatasetDimensions, FilterDimensions
from cpra.mp.data.catalog import CatalogQuery
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.raster import write_cog
from cpra.mp.data.tabular import write_parquet


def query(catalog_path, dimensions, **filters):
    return CatalogQuery(
        DatasetDimensions(**dimensions), FilterDimensions(**filters), catalog_path=catalog_path
    ).execute()


# vct/veg_grid_cell_v001 needs no extra required dimensions beyond the always-required
# model_group_id/scenario_id/model, so it's the simplest raster fixture.
RASTER_KWARGS = dict(
    variable="vct",
    grid="veg_grid_cell_v001",
    time_unit="annual",
    model="icm_v2023.0.0",
)

# ssf/ecoregion_v001 requires `percentile`, and ecoregion_v001 is small (32 grid cells).
ANNUAL_KWARGS = dict(
    variable="ssf",
    grid="ecoregion_v001",
    time_unit="annual",
    model_group_id=500,
    scenario_id=7,
    model="icm_v2023.0.0",
)

# damage/clara_grid_cell_v001 is the decadal CLARA variable, and the one with a Float32
# `aep` dimension.
DAMAGE_KWARGS = dict(
    variable="damage",
    grid="clara_grid_cell_v001",
    time_unit="decadal",
    model_group_id=500,
    scenario_id=7,
    model="clara_v2023.0.0",
    pumping_id=0.5,
    fragility_scenario=0,
    population_variant=0,
    clara_variant=0,
    asset_type=3,
    percentile=50,
)
DAMAGE_DIMENSIONS = {k: DAMAGE_KWARGS[k] for k in ("variable", "grid", "time_unit")}


def _raster_array(fill=1):
    grid = GRIDS[RASTER_KWARGS["grid"]]
    return np.full((grid.height, grid.width), fill, dtype="float32")


def _annual_frame(years, grid_cells=(1, 2), percentile=50):
    rows = [(gc, yr) for gc in grid_cells for yr in years]
    return pl.DataFrame(
        {
            "grid_cell": [r[0] for r in rows],
            "calendar_year": [r[1] for r in rows],
            "percentile": [percentile] * len(rows),
            "value": [1.0] * len(rows),
        },
        schema={
            "grid_cell": pl.Int32,
            "calendar_year": pl.Int32,
            "percentile": pl.Int8,
            "value": pl.Float32,
        },
    )


def _daily_frame(days, grid_cells=(1, 2, 3)):
    rows = [(gc, d) for gc in grid_cells for d in days]
    return pl.DataFrame(
        {
            "grid_cell": [r[0] for r in rows],
            "calendar_day": [r[1] for r in rows],
            "value": [1.0] * len(rows),
        },
        schema={"grid_cell": pl.Int32, "calendar_day": pl.Date, "value": pl.Float32},
    )


def _write_damage(base_path, catalog_path, years=(2030,), aeps=(2.0,)):
    """Write a minimal decadal CLARA dataset: one write_parquet (and catalog append) per
    (year, aep) pair, so a gap in the years is a gap between separate catalog rows."""
    for year in years:
        for aep in aeps:
            df = pl.DataFrame(
                {"grid_cell": [1], "calendar_year": [year], "value": [1.0]},
                schema={"grid_cell": pl.Int32, "calendar_year": pl.Int32, "value": pl.Float64},
            ).with_columns(pl.lit(aep).alias("aep"))
            write_parquet(
                df,
                base_path=base_path,
                catalog_path=catalog_path,
                validate=False,
                sort=False,
                **DAMAGE_KWARGS,
            )


@pytest.fixture
def raster_tree(base_path, catalog_path):
    """COGs for two scenarios across three consecutive years."""
    for scenario in (7, 8):
        for year in (2030, 2031, 2032):
            write_cog(
                _raster_array(),
                base_path=base_path,
                catalog_path=catalog_path,
                calendar_year=year,
                model_group_id=500,
                scenario_id=scenario,
                **RASTER_KWARGS,
            )
    return catalog_path


@pytest.fixture
def raster_dimensions():
    return {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}


@pytest.fixture
def annual_tree(base_path, catalog_path):
    write_parquet(
        _annual_frame(years=(2030, 2031, 2032)),
        base_path=base_path,
        catalog_path=catalog_path,
        **ANNUAL_KWARGS,
    )
    return catalog_path


@pytest.fixture
def annual_dimensions():
    return {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}


class TestCatalogChoices:
    def test_reports_available_choices(self, raster_tree, raster_dimensions):
        result = query(raster_tree, raster_dimensions)
        assert result["choices"]["scenario_id"] == [7, 8]
        assert result["choices"]["model_group_id"] == [500]

    def test_choices_are_typed_by_dimension_dtype(self, raster_tree, raster_dimensions):
        result = query(raster_tree, raster_dimensions)
        assert all(isinstance(v, int) for v in result["choices"]["scenario_id"])

    def test_selecting_an_available_value_is_exact(self, raster_tree, raster_dimensions):
        """`exact` requires every dimension the catalog tracks to be pinned -- including
        metadata dimensions like `model`, which the old file-walking CatalogQuery could
        never see (COG tags, not path components) and so never factored into `exact` at
        all. See test_metadata_dimensions_are_tracked_as_choices below."""
        result = query(
            raster_tree,
            raster_dimensions,
            model_group_id=[500],
            scenario_id=[8],
            calendar_year=[2031],
            model=["icm_v2023.0.0"],
        )
        assert result["results"]["scenario_id"] == [8]
        assert result["results"]["calendar_year"] == [2031]
        assert result["exact"] is True

    def test_metadata_dimensions_are_tracked_as_choices(self, raster_tree, raster_dimensions):
        """The materialized catalog carries metadata dimensions (e.g. `model`, sourced from
        COG/file tags) as ordinary columns, so CatalogQuery reports choices/results for them
        the same as any path-derived dimension. main's directory walk had no way to do this
        at all, since a metadata dimension is never a path component (see
        cpra/mp/data/catalog.py's module docstring) -- new coverage, not a port."""
        result = query(raster_tree, raster_dimensions)
        assert result["choices"]["model"] == ["icm_v2023.0.0"]
        assert result["results"]["model"] == ["icm_v2023.0.0"]

    def test_leaving_a_dimension_unspecified_is_inexact(self, raster_tree, raster_dimensions):
        """model_group_id is unspecified, so it falls back to a default."""
        result = query(
            raster_tree,
            raster_dimensions,
            scenario_id=[8],
            calendar_year=[2031],
            model=["icm_v2023.0.0"],
        )
        assert result["results"]["scenario_id"] == [8]
        assert result["exact"] is False

    def test_defaults_to_the_first_choice_when_nothing_is_requested(self, raster_tree, raster_dimensions):
        result = query(raster_tree, raster_dimensions)
        assert result["results"]["scenario_id"] == [7]
        assert result["exact"] is False

    def test_unavailable_value_falls_back_and_is_inexact(self, raster_tree, raster_dimensions):
        result = query(raster_tree, raster_dimensions, scenario_id=[99])
        assert result["results"]["scenario_id"] == [7]
        assert result["exact"] is False

    def test_partially_available_selection_is_inexact(self, raster_tree, raster_dimensions):
        result = query(raster_tree, raster_dimensions, scenario_id=[7, 99])
        assert result["results"]["scenario_id"] == [7]
        assert result["exact"] is False

    def test_multiple_values_are_all_selected(self, raster_tree, raster_dimensions):
        result = query(raster_tree, raster_dimensions, scenario_id=[7, 8])
        assert result["results"]["scenario_id"] == [7, 8]

    def test_dataset_dimensions_reflect_the_request(self, raster_tree, raster_dimensions):
        result = query(raster_tree, raster_dimensions)
        assert result["results"]["variable"] == ["vct"]
        assert result["results"]["grid"] == ["veg_grid_cell_v001"]


class TestChoiceIntersection:
    """Each dimension's choices are computed from the catalog rows left after filtering by
    every dimension resolved before it, since `execute()` narrows its working frame as it
    resolves each dimension in turn -- the materialized-catalog analogue of main's walk only
    descending into the branches its earlier selections/defaults picked.

    Two of main's tests are deliberately NOT ported (see the port report for detail): with
    the table-backed query, explicitly selecting *multiple* values of an earlier dimension
    no longer intersects the next dimension's choices down to the values common to every
    selected branch -- it unions them, because `_update` is only ever called once per
    dimension name per query (there is no repeated visit to reconcile). That also makes the
    "no common values" ValueError `_update` can raise unreachable in the current code.
    """

    def _write(self, base_path, catalog_path, pairs):
        for model_group, scenario in pairs:
            write_cog(
                _raster_array(),
                base_path=base_path,
                catalog_path=catalog_path,
                calendar_year=2030,
                model_group_id=model_group,
                scenario_id=scenario,
                **RASTER_KWARGS,
            )

    def test_unselected_dimension_defaults_and_narrows_the_next(self, base_path, catalog_path, raster_dimensions):
        """With no model_group_id requested, it defaults to the first choice (500), which
        narrows scenario_id's choices to just that group's own two scenarios."""
        self._write(base_path, catalog_path, [(500, 7), (500, 8), (520, 7)])
        result = query(catalog_path, raster_dimensions)
        assert result["choices"]["model_group_id"] == [500, 520]
        assert result["choices"]["scenario_id"] == [7, 8]


class TestCompressYears:
    def test_contiguous_years_collapse_to_a_range(self, raster_tree, raster_dimensions):
        result = query(raster_tree, raster_dimensions)
        assert result["choices"]["calendar_year"] == [[2030, 2032]]

    def test_gaps_produce_separate_ranges(self, base_path, catalog_path, raster_dimensions):
        for year in (2030, 2031, 2040, 2041):
            write_cog(
                _raster_array(),
                base_path=base_path,
                catalog_path=catalog_path,
                calendar_year=year,
                model_group_id=500,
                scenario_id=7,
                **RASTER_KWARGS,
            )
        result = query(catalog_path, raster_dimensions)
        assert result["choices"]["calendar_year"] == [[2030, 2031], [2040, 2041]]

    def test_a_lone_year_is_reported_as_a_scalar(self, base_path, catalog_path, raster_dimensions):
        write_cog(
            _raster_array(),
            base_path=base_path,
            catalog_path=catalog_path,
            calendar_year=2030,
            model_group_id=500,
            scenario_id=7,
            **RASTER_KWARGS,
        )
        result = query(catalog_path, raster_dimensions)
        assert result["choices"]["calendar_year"] == [2030]

    def test_decadal_ranges_align_to_the_decade(self, base_path, catalog_path):
        """A decadal series starting mid-decade is still stepped by 10 from floor(min/10)*10.

        Uses damage/clara_grid_cell_v001 because every decadal variable is a CLARA (vector)
        variable -- no raster grid carries decadal data.
        """
        _write_damage(base_path, catalog_path, years=(2035, 2045, 2055))
        result = query(catalog_path, DAMAGE_DIMENSIONS)
        assert result["choices"]["calendar_year"] == [[2035, 2055]]

    def test_decadal_gaps_still_split(self, base_path, catalog_path):
        _write_damage(base_path, catalog_path, years=(2035, 2045, 2075))
        result = query(catalog_path, DAMAGE_DIMENSIONS)
        assert result["choices"]["calendar_year"] == [[2035, 2045], 2075]


class TestTabularCatalogChoices:
    """Unlike a raster grid (whose non-calendar dimensions are mostly path components, see
    RASTER_KWARGS), a Parquet dataset's dimensions almost all live inside the file (see
    cpra/mp/data/dimensions.py's `Dimension.partition`: a partition-typed dimension is only
    ever an actual Hive directory when time_unit == "daily"). The old file-walking
    CatalogQuery had to merge two different sources -- directory names and file columns --
    to answer a query; the materialized catalog flattens both into the same row columns, so
    there's nothing left to distinguish at query time.
    """

    def test_reads_choices_from_the_catalog(self, annual_tree, annual_dimensions):
        result = query(annual_tree, annual_dimensions)
        assert result["choices"]["calendar_year"] == [[2030, 2032]]

    def test_non_directory_dimensions_are_still_choices(self, annual_tree, annual_dimensions):
        """model_group_id and percentile are never Hive directories for annual data, only
        columns inside the file -- yet the catalog reports them as choices all the same."""
        result = query(annual_tree, annual_dimensions)
        assert result["choices"]["model_group_id"] == [500]
        assert result["choices"]["percentile"] == [50]

    def test_selection_narrows_results(self, annual_tree, annual_dimensions):
        """Only calendar_day is stringified by _compress; calendar_year stays an int."""
        result = query(annual_tree, annual_dimensions, calendar_year=[2031])
        assert result["results"]["calendar_year"] == [2031]

    def test_multiple_files_are_scanned_together(self, base_path, catalog_path, annual_dimensions):
        """Each write_parquet call appends its own catalog file; the query unions every
        file under the requested variable/time_unit/grid partition."""
        write_parquet(
            _annual_frame(years=(2030,)), base_path=base_path, catalog_path=catalog_path, **ANNUAL_KWARGS
        )
        write_parquet(
            _annual_frame(years=(2040,)),
            base_path=base_path,
            catalog_path=catalog_path,
            **{**ANNUAL_KWARGS, "scenario_id": 8},
        )
        result = query(catalog_path, annual_dimensions)
        assert result["choices"]["scenario_id"] == [7, 8]


class TestDailyCompression:
    def test_contiguous_days_collapse_to_a_string_range(self, base_path, catalog_path):
        days = [datetime.date(2030, 1, 1) + datetime.timedelta(days=i) for i in range(5)]
        write_parquet(
            _daily_frame(days),
            base_path=base_path,
            catalog_path=catalog_path,
            validate=False,
            sort=False,
            variable="sal",
            grid="hydro_compartment_v001",
            time_unit="daily",
            model_group_id=500,
            scenario_id=7,
            aggregate_type="mean",
            model="icm_v2023.0.0",
        )

        result = query(
            catalog_path, {"variable": "sal", "grid": "hydro_compartment_v001", "time_unit": "daily"}
        )
        assert result["choices"]["calendar_day"] == [["2030-01-01", "2030-01-05"]]

    def test_selected_days_are_stringified(self, base_path, catalog_path):
        days = [datetime.date(2030, 1, 1), datetime.date(2030, 1, 2)]
        write_parquet(
            _daily_frame(days),
            base_path=base_path,
            catalog_path=catalog_path,
            validate=False,
            sort=False,
            variable="sal",
            grid="hydro_compartment_v001",
            time_unit="daily",
            model_group_id=500,
            scenario_id=7,
            aggregate_type="mean",
            model="icm_v2023.0.0",
        )

        result = query(
            catalog_path, {"variable": "sal", "grid": "hydro_compartment_v001", "time_unit": "daily"}
        )
        assert all(isinstance(v, str) for v in result["results"]["calendar_day"])


class TestFloatNormalisation:
    """Float32 storage means a canonical 0.1 reads back as 0.10000000149011612.

    CatalogQuery maps values back onto the dimension's canonical list so that choices,
    results, and /v1/dimensions.json all agree, and so that a client can select using
    either form.
    """

    @pytest.fixture
    def aep_tree(self, base_path, catalog_path):
        _write_damage(base_path, catalog_path, aeps=(0.1, 2.0))
        return catalog_path

    def test_choices_use_canonical_values(self, aep_tree):
        result = query(aep_tree, DAMAGE_DIMENSIONS)
        assert 0.1 in result["choices"]["aep"]

    def test_canonical_input_selects_the_stored_value(self, aep_tree):
        result = query(aep_tree, DAMAGE_DIMENSIONS, aep=[0.1])
        assert result["results"]["aep"] == [0.1]

    def test_float32_repr_input_resolves_to_the_same_value(self, aep_tree):
        """A client echoing back the Float32 repr must not be treated as a miss."""
        result = query(aep_tree, DAMAGE_DIMENSIONS, aep=[0.10000000149011612])
        assert result["results"]["aep"] == [0.1]

    def test_unknown_float_still_falls_back(self, aep_tree):
        result = query(aep_tree, DAMAGE_DIMENSIONS, aep=[99.0])
        assert result["results"]["aep"] == [0.1]
        assert result["exact"] is False
