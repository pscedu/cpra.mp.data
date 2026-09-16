"""End-to-end tests for the data and tile endpoints through the ASGI app.

Unlike the Dataset unit tests, these go through real routing and query-string parsing, so
they cover the parameter binding that FastAPI does before a Dataset is ever built.
"""

import io
import json
from urllib.parse import parse_qs, urlparse

import mercantile
import numpy as np
import polars as pl
import pytest
import rasterio as rio
from rasterio.crs import CRS
from rasterio.warp import transform
from shapely.geometry import box

from cpra.mp.data import config
from cpra.mp.data.analysis import fold as fold_module
from cpra.mp.data.tabular import write_parquet
from tests.factories import (
    RASTER_KWARGS,
    annual_frame,
    daily_frame,
    raster_array,
    write_grid_geometry,
    write_pmtiles,
    write_raster_crosswalk,
    write_raster_dataset,
    write_vector_crosswalk,
)

VECTOR_URL = "/v1/datasets/ssf/annual/ecoregion_v001"
RASTER_URL = "/v1/datasets/vct/annual/veg_grid_cell_v001"
SALINITY_URL = "/v1/datasets/sal/daily/hydro_compartment_v001"
# Float32-representable, and far enough from a .x5 tie that the rounding is unambiguous.
SALINITY_VALUES = [12.34375, 9.96875]


def _lng_lat(x, y):
    """Project an ecoregion_v001 (EPSG:26915) point to WGS84, as a `grid_cell=` value."""
    lng, lat = [c[0] for c in transform(CRS.from_epsg(26915), CRS.from_epsg(4326), [x], [y])]
    return f"{lng!s},{lat!s}"


@pytest.fixture
def vector_data(base_path, monkeypatch):
    # The catalog side-table isolation (config.CATALOG_PATH) is handled centrally by the
    # autouse `_isolated_catalog_path` fixture in tests/conftest.py -- otherwise catalog
    # rows written by one test (e.g. the ssf/ecoregion_v001/annual years this fixture
    # always uses) would leak into another test's /catalog.json assertions.
    write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
    monkeypatch.setattr(config, "BASE_PATH", base_path)
    return base_path


@pytest.fixture
def ecoregion_geometry(geometry_path):
    """A single 1km square at ecoregion_v001 cell 1, for coordinate lookups."""
    return write_grid_geometry(
        geometry_path, "ecoregion_v001", [(1, box(500000, 3300000, 501000, 3301000))]
    )


@pytest.fixture
def salinity_data(base_path, monkeypatch):
    """Two days of sal on hydro_compartment_v001, with values that rounding changes.

    ``validate=False`` because ``daily_frame`` carries three grid cell columns rather than
    the grid's full 1,657.
    """
    df = daily_frame().with_columns(
        pl.Series("1", SALINITY_VALUES, dtype=pl.Float32),
        pl.Series("2", [-0.449, 1.24], dtype=pl.Float32),
        pl.Series("3", [0.0, 0.04], dtype=pl.Float32),
    )
    write_parquet(df, base_path=base_path, validate=False, sort=False)
    monkeypatch.setattr(config, "BASE_PATH", base_path)
    return base_path


@pytest.fixture
def two_model_group_salinity_data(base_path, monkeypatch):
    """Two model groups of daily sal, the shape a `difference` over `model_group_id` needs.

    ``validate=False`` for the same reason as ``salinity_data``: three grid cell columns
    rather than the grid's full 1,657.
    """
    for model_group_id in (500, 521):
        write_parquet(
            daily_frame(model_group_id=model_group_id),
            base_path=base_path,
            validate=False,
            sort=False,
        )
    monkeypatch.setattr(config, "BASE_PATH", base_path)
    return base_path


@pytest.fixture
def raster_data(base_path, monkeypatch):
    write_raster_dataset(base_path, years=(2030,), fill_fn=lambda y: y - 2000)
    monkeypatch.setattr(config, "BASE_PATH", base_path)
    return base_path


@pytest.fixture
def categorical_vector_data(base_path, crosswalk_path, monkeypatch):
    """ssf/ecoregion_v001 with a bounded categorical value (geo_id % 3) instead of
    `annual_frame`'s default continuous `year + geo_id/100`, plus a crosswalk onto
    hydro_compartment_v001 -- the shape count_by_value needs end to end (`summarize_by` +
    `summarize_aggregate=count_by_value`), through routing and query-string parsing.
    """
    write_parquet(
        annual_frame(years=(2030,), value_fn=lambda _year, geo_id: float(geo_id % 3)),
        base_path=base_path,
    )
    monkeypatch.setattr(config, "BASE_PATH", base_path)
    write_vector_crosswalk(
        crosswalk_path,
        "ecoregion_v001",
        "hydro_compartment_v001",
        [(geo_id, 100 if geo_id <= 16 else 200, 1.0) for geo_id in range(1, 33)],
    )
    return base_path


@pytest.fixture
def categorical_raster_with_zones(base_path, crosswalk_path, monkeypatch):
    """A raster whose pixels cycle through a handful of categories, split into two zones
    -- the count_by_value counterpart to `gradient_raster_with_zones` below."""
    from cpra.mp.data.grids import GRIDS

    write_raster_dataset(base_path, years=(2030,), data_fn=lambda _year: raster_array() % 5)
    monkeypatch.setattr(config, "BASE_PATH", base_path)

    grid = GRIDS[RASTER_KWARGS["grid"]]
    zones = np.zeros((grid.height, grid.width), dtype="int32")
    zones[: grid.height // 2, :] = 1
    zones[grid.height // 2 :, :] = 2
    write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
    return base_path, grid, zones


@pytest.fixture
def gradient_raster_with_zones(base_path, crosswalk_path, monkeypatch):
    """A raster whose pixels are a gradient (see `raster_array`'s no-fill case), split
    into two zones by row, for a `summarize_by`+`where`-value-conjunct conditional-count
    request -- the exit-criterion query, end to end through the ASGI app rather than a
    unit-level `Dataset`.
    """
    from cpra.mp.data.grids import GRIDS

    write_raster_dataset(base_path, years=(2030,), data_fn=lambda _year: raster_array())
    monkeypatch.setattr(config, "BASE_PATH", base_path)

    grid = GRIDS[RASTER_KWARGS["grid"]]
    zones = np.zeros((grid.height, grid.width), dtype="int32")
    zones[: grid.height // 2, :] = 1
    zones[grid.height // 2 :, :] = 2
    write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
    return base_path, grid, zones


class TestFloatPrecision:
    """sal's structure row sets `float_precision`, so its CSV is rounded on the way out.

    Checked as text rather than through `pl.read_csv`: the point is the bytes the client
    receives, and a reader would parse "12.3" and "12.3456" into indistinguishable floats.
    """

    # Data is served in long format (grid_cell/value columns) unconditionally now --
    # `unpivot` used to force this shape and is retired; no query-string equivalent is
    # needed since there's no other shape left to ask for.
    QUERY = "?select=calendar_day,grid_cell,value"

    def test_csv_values_are_rounded_to_one_decimal(self, client, salinity_data):
        response = client.get(f"{SALINITY_URL}/data.csv{self.QUERY}")
        assert response.status_code == 200
        rows = response.text.strip().splitlines()
        assert rows[0] == "calendar_day,grid_cell,value"
        assert rows[1:4] == [
            "2030-01-01,1,12.3",
            "2030-01-01,2,-0.4",
            "2030-01-01,3,0.0",
        ]

    def test_parquet_keeps_the_stored_value(self, client, salinity_data):
        response = client.get(f"{SALINITY_URL}/data.parquet{self.QUERY}")
        assert response.status_code == 200
        df = pl.read_parquet(io.BytesIO(response.content))
        # An integer, not "1": unpivoting a vector grid reports `grid_cell` as the cell ID.
        assert df.filter(pl.col("grid_cell") == 1)["value"].to_list() == SALINITY_VALUES

    def test_an_unrounded_variable_is_unaffected(self, client, vector_data):
        """ssf sets no precision, so its floats are written at full width."""
        response = client.get(f"{VECTOR_URL}/data.csv?grid_cell=1&select=value")
        assert response.status_code == 200
        assert response.text.strip().splitlines()[1:] == ["2030.01", "2040.01"]


class TestVectorData:
    def test_csv(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv")
        assert response.status_code == 200
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["calendar_year"].unique().sort().to_list() == [2030, 2040]

    def test_parquet(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.parquet")
        assert response.status_code == 200
        # Long format: one row per (grid cell x calendar year), so 32 cells x 2 years.
        assert pl.read_parquet(io.BytesIO(response.content)).height == 64

    def test_ndjson(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.ndjson")
        assert response.status_code == 200
        assert pl.read_ndjson(io.BytesIO(response.content)).height == 64

    def test_dimension_filter_from_the_query_string(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv?calendar_year=2030")
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["calendar_year"].unique().to_list() == [2030]
        assert df.height == 32

    def test_repeated_filter_values(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv?calendar_year=2030&calendar_year=2040")
        df = pl.read_csv(io.BytesIO(response.content))
        assert df.height == 64
        assert set(df["calendar_year"]) == {2030, 2040}

    def test_grid_cell_selects_rows(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv?grid_cell=1,2")
        df = pl.read_csv(io.BytesIO(response.content))
        assert sorted(df["grid_cell"].unique().to_list()) == [1, 2]

    def test_grid_cell_accepts_a_coordinate_pair(self, client, vector_data, ecoregion_geometry):
        """grid_cell= on a vector grid also accepts lat/lng, resolved via the grid's
        FlatGeoBuf geometry and reported back under the grid cell ID it resolved to."""
        label = _lng_lat(500500, 3300500)

        response = client.get(f"{VECTOR_URL}/data.csv?grid_cell={label}")
        assert response.status_code == 200
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["grid_cell"].unique().to_list() == [1]
        assert label not in df.columns

    def test_data_is_reported_in_long_format(self, client, vector_data):
        """Data is served long (grid_cell/value columns) unconditionally now -- `unpivot`
        used to request this shape and is retired."""
        response = client.get(f"{VECTOR_URL}/data.csv?grid_cell=1")
        df = pl.read_csv(io.BytesIO(response.content))
        assert {"grid_cell", "value"} <= set(df.columns)

    def test_reports_cell_ids_for_a_coordinate_selection(
        self, client, vector_data, ecoregion_geometry
    ):
        """The shape every visualization requests: long format, and still keyed by ID, so
        the caller can join it against anything else the vector grid keys that way."""
        label = _lng_lat(500500, 3300500)

        response = client.get(f"{VECTOR_URL}/data.csv?grid_cell={label}")
        assert response.status_code == 200
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["grid_cell"].unique().to_list() == [1]

    def test_coordinates_sharing_a_cell_report_it_once(
        self, client, vector_data, ecoregion_geometry
    ):
        """Two coordinates in one cell describe one series — there's a single value, so
        reporting it twice under one ID would only be ambiguous."""
        first, second = _lng_lat(500400, 3300400), _lng_lat(500600, 3300600)

        response = client.get(
            f"{VECTOR_URL}/data.csv?grid_cell={first},{second}&calendar_year=2030"
        )
        assert response.status_code == 200
        df = pl.read_csv(io.BytesIO(response.content))
        assert dict(zip(df["grid_cell"], df["value"])) == {1: pytest.approx(2030.01)}

    def test_select(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv?select=calendar_year")
        assert pl.read_csv(io.BytesIO(response.content)).columns == ["calendar_year"]

    def test_sort_descending(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv?sort=-calendar_year")
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["calendar_year"].unique(maintain_order=True).to_list() == [2040, 2030]

    def test_where_clause(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv?where=calendar_year%20%3E%202035")
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["calendar_year"].unique().to_list() == [2040]
        assert df.height == 32

    def test_tif_is_rejected_for_a_vector_grid(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.tif")
        assert response.status_code == 400
        assert "must be one of" in response.json()["detail"]

    def test_unknown_variable_is_rejected(self, client, vector_data):
        assert client.get("/v1/datasets/nope/ecoregion_v001/annual/data.csv").status_code == 422

    def test_unknown_grid_is_rejected(self, client, vector_data):
        assert client.get("/v1/datasets/ssf/nope/annual/data.csv").status_code == 422

    def test_missing_dataset_is_404(self, client, base_path, monkeypatch):
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        response = client.get(f"{VECTOR_URL}/data.csv")
        assert response.status_code == 404

    def test_invalid_sort_is_400(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv?sort=nonsense")
        assert response.status_code == 400

    def test_group_by_is_retired_on_the_data_route(self, client, vector_data):
        """`group_by` is retired from the flat data route -- rejected outright
        rather than validated (it used to 400 here for lacking `group_aggregate`; now it
        400s regardless, before that check is ever reached)."""
        response = client.get(f"{VECTOR_URL}/data.csv?group_by=calendar_year")
        assert response.status_code == 400
        assert "group_by" in response.json()["detail"]

    def test_conditional_count_is_one_request(self, client, vector_data):
        """"Count cells where value > x" -- one `where` value conjunct +
        `group_aggregate=count` request, no separate fetch-then-filter-then-count round
        trip on the client side. Cell (year, geo) holds year + geo/100 (see
        `annual_frame`), so a threshold between the two written years isolates exactly
        one year's worth of matches.

        `group_aggregate` (like `group_by`) is retired from the flat data route
        -- this now goes through the ad-hoc `/v1/analysis.csv` pipeline route's
        `filter` + `group` steps instead, the pipeline equivalent of the same request.
        The leading identity `project` (`to` the input's own grid) exists purely to give
        `filter` something to chain from -- a pipeline's first step must name its input
        explicitly, which only `project`/`join` can do.
        """
        pipeline = {
            "inputs": {"ssf": {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"}},
            "steps": [
                {"op": "project", "input": "ssf", "to": "ecoregion_v001"},
                {"op": "filter", "where": "value > 2035"},
                {"op": "group", "by": [], "aggregate": "count"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert df.height == 32
        assert df["value"].to_list() == [1] * 32

    def test_cliff_shaped_group_aggregate_goes_through_the_fold(self, client, base_path, monkeypatch):
        """A request shaped exactly like the memory cliff this package's `api.fold`
        module fixes -- no `group_by`, no selective filter, `group_aggregate` set, so
        the group key is `grid_cell` alone -- routed end to end through the ASGI app via
        the `/v1/analysis.csv` pipeline route's `group` step (`group_aggregate` has no
        flat spelling anymore). `TABULAR_FOLD_STEP` is monkeypatched down to 40
        so this fixture-scale dataset (50 years x 32 cells = 1,600 rows) is still forced
        through ~40 real fold chunks and merges, rather than degenerating to the
        single-chunk case every other fixture-scale test in this suite exercises.
        """
        monkeypatch.setattr(fold_module, "TABULAR_FOLD_STEP", 40)
        years = range(2000, 2050)
        write_parquet(annual_frame(years=years), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"ssf": {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"}},
            "steps": [
                {"op": "project", "input": "ssf", "to": "ecoregion_v001"},
                {"op": "group", "by": [], "aggregate": "sum"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content)).sort("grid_cell")

        assert df.height == 32
        year_sum = sum(years)
        expected = [year_sum + len(years) * geo_id / 100 for geo_id in range(1, 33)]
        assert df["value"].to_list() == pytest.approx(expected, rel=1e-5)

    def test_count_by_value_per_zone_is_one_request(self, client, categorical_vector_data):
        """"How many cells carry each distinct value per zone" -- one `project`-to-a-
        coarser-grid step with `aggregate=count_by_value`, asserting the long output
        shape (grid_cell/value/count/geographic_unit), through the `/v1/analysis.csv`
        pipeline route (`summarize_by`/`summarize_aggregate` have no flat spelling
        anymore -- `project`+`aggregate` is their pipeline equivalent).
        """
        pipeline = {
            "inputs": {"ssf": {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"}},
            "steps": [
                {"op": "project", "input": "ssf", "to": "hydro_compartment_v001", "aggregate": "count_by_value"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert {"grid_cell", "value", "count", "geographic_unit"} <= set(df.columns)

        expected = {100: {}, 200: {}}
        for geo_id in range(1, 33):
            zone = 100 if geo_id <= 16 else 200
            value = float(geo_id % 3)
            expected[zone][value] = expected[zone].get(value, 0) + 1

        for zone_id, value_counts in expected.items():
            zone_df = df.filter(pl.col("grid_cell") == zone_id)
            actual = dict(zip(zone_df["value"].to_list(), zone_df["count"].to_list()))
            assert actual == value_counts

    def test_mixed_dimension_and_value_where_aggregating(self, client, vector_data):
        """A dimension conjunct and a value conjunct, through the `/v1/analysis.csv`
        pipeline route's `filter` + `group` steps (`group_aggregate` has no flat
        spelling anymore). `calendar_year > 2035` (a dimension conjunct) is a
        real row filter, isolating the 2040 rows; `value > 0` (a value conjunct) is
        vacuously true for every remaining cell (see `annual_frame`'s
        `year + geo_id / 100`), so `aggregate=count` should still see all 32 cells.

        The two conjuncts are deliberately spelled as two separate `filter` steps
        rather than one `"calendar_year > 2035 AND value > 0"` string: a single-input
        pipeline compiles every `filter` step's `where` onto one combined
        `Options.where` by wrapping each step's clause in its own parens and AND-ing
        them together (`planner._compile_single_input_chain`), which is exactly right
        when each step's own clause is a single conjunct, but corrupts a step whose own
        `where` is *itself* multiple AND-ed conjuncts (the parens hide the inner AND
        from the executor's own later re-split of `Options.where` into dimension/value
        conjuncts, so a perfectly valid one-dimension-one-value `where` gets rejected
        as if it were one mixed conjunct). That's a pre-existing pipeline-compiler
        limitation, not something introduced by this test, and out of scope for this
        change -- splitting across two `filter` steps here avoids it entirely without giving up
        anything this test is meant to demonstrate.
        """
        pipeline = {
            "inputs": {"ssf": {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"}},
            "steps": [
                {"op": "project", "input": "ssf", "to": "ecoregion_v001"},
                {"op": "filter", "where": "calendar_year > 2035"},
                {"op": "filter", "where": "value > 0"},
                {"op": "group", "by": [], "aggregate": "count"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert df.height == 32
        assert df["value"].to_list() == [1] * 32

    def test_mixed_dimension_and_value_where_plain(self, client, vector_data):
        """The same mixed `where` as above, but with no aggregate: the dimension
        conjunct still restricts to 2040's rows, and the value conjunct now drops (not
        masks) any row that fails it, since a plain tabular request has no aggregate
        downstream for a masked null to fall into unnoticed.
        """
        response = client.get(
            f"{VECTOR_URL}/data.csv",
            params={"where": "calendar_year > 2035 AND value > 2040.16"},
        )
        assert response.status_code == 200
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["calendar_year"].unique().to_list() == [2040]
        assert df["value"].null_count() == 0
        assert (df["value"] > 2040.16).all()
        assert df.height < 32

    def test_retired_value_where_param_is_rejected(self, client, vector_data):
        """`value_where` was retired in favor of `where` (see
        `api.reject_retired_value_where`); a request that still sends it gets a loud 400
        naming the replacement, rather than having the parameter silently ignored the way
        FastAPI/pydantic treat any other unrecognized query parameter."""
        response = client.get(f"{VECTOR_URL}/data.csv", params={"value_where": "value > 2035"})
        assert response.status_code == 400
        assert "where" in response.json()["detail"]


class TestRasterData:
    def test_tif(self, client, raster_data):
        response = client.get(f"{RASTER_URL}/data.tif")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/tiff; application=geotiff"
        with rio.open(io.BytesIO(response.content)) as src:
            assert src.read(1)[0, 0] == 30

    def test_csv_is_rejected_for_a_raster_grid(self, client, raster_data):
        response = client.get(f"{RASTER_URL}/data.csv")
        assert response.status_code == 400
        assert "must be tif" in response.json()["detail"]

    def test_geography_makes_csv_valid(self, client, raster_data):
        """Sampling at coordinates converts raster output to tabular."""
        with rio.open(next(raster_data.rglob("*.tif"))) as src:
            from rasterio.warp import transform

            x, y = src.xy(src.height // 2, src.width // 2)
            lng, lat = [c[0] for c in transform(src.crs, "EPSG:4326", [x], [y])]

        response = client.get(f"{RASTER_URL}/data.csv?grid_cell={lng},{lat}")
        assert response.status_code == 200
        df = pl.read_csv(io.BytesIO(response.content))
        # A raster grid_cell selection is keyed by the "lng,lat" label itself, not an ID.
        assert df["grid_cell"].to_list() == [f"{lng},{lat}"]
        assert df["value"].to_list() == [30]

    def test_tabular_compression_is_rejected_for_tif(self, client, raster_data):
        response = client.get(f"{RASTER_URL}/data.tif?compression_algorithm=gzip")
        assert response.status_code == 400

    def test_missing_dataset_is_404(self, client, base_path, monkeypatch):
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        assert client.get(f"{RASTER_URL}/data.tif").status_code == 404

    def test_conditional_count_per_zone_is_one_request(self, client, gradient_raster_with_zones):
        """"Count pixels per compartment where value > X" expressed as one
        `/v1/analysis.csv` pipeline request: a LEADING `filter` step
        (naming its `input` explicitly, since it is step 1 -- see `FilterStep.input`'s
        docstring) followed by a `project`-to-a-coarser-grid step with `aggregate=count`.
        `summarize_by`/`summarize_aggregate` have no flat spelling, so
        this goes through the pipeline route whose `project`+`aggregate` step compiles onto
        those `Options` fields rather than the plain data route.

        The filter is now declared exactly where it executes: both steps lower onto
        `Options` (`where`, and `summarize_by`/`summarize_aggregate` respectively), and
        the flat engine always applies the Filter stage (stage 1 of
        `BaseAnalysis.apply_params`) before the Transform stage's zonal aggregation
        (stage 4) -- so declaring the filter first is what makes "pixel-level threshold,
        not a post-count one" the TRUTH of the declared order, not a fact the reader has
        to already know about the underlying engine to get the right answer.

        This pipeline used to be spelled the other way around --
        `project(aggregate=count) -> filter(value > X)`, filter declared AFTER project --
        and relied on the engine silently reordering the filter before the project's
        aggregation to produce the right answer despite the declared order promising the
        opposite. `planner._compile_single_input_chain`'s SUMMARIZING-PROJECT ORDER GUARD
        (its `is_summarizing_project` checks) now rejects that old spelling outright
        (a value-testing `filter` declared after a many-to-one `project` is a verified
        wrong answer, not just an unimplemented feature -- see the guard's own
        `PipelineError` for the numeric counterexample); the leading-filter spelling
        below is its replacement.
        """
        _base_path, _grid, zones = gradient_raster_with_zones
        gradient = raster_array()
        threshold = 100_000

        pipeline = {
            "inputs": {"vct": {"variable": "vct", "time_unit": "annual", "grid": "veg_grid_cell_v001"}},
            "steps": [
                {"op": "filter", "input": "vct", "where": f"vct > {threshold}"},
                {"op": "project", "to": "hydro_compartment_v001", "aggregate": "count"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        for zone_id in (1, 2):
            expected = int((gradient[zones == zone_id] > threshold).sum())
            actual = df.filter(pl.col("grid_cell") == zone_id)["value"][0]
            assert actual == expected

    def test_count_by_value_per_zone_is_one_request(self, client, categorical_raster_with_zones):
        """"How many pixels of each land-type value per hydro compartment" -- one
        `project`-to-a-coarser-grid step with `aggregate=count_by_value`, asserting the
        long output shape (grid_cell/value/count/geographic_unit), through the
        `/v1/analysis.csv` pipeline route (`summarize_by`/`summarize_aggregate` are
        retired flat-query parameters with no pipeline-step equivalent by that name --
        this test exercises their `project`/`aggregate` replacement instead).
        """
        _base_path, _grid, zones = categorical_raster_with_zones
        categories = raster_array() % 5

        pipeline = {
            "inputs": {"vct": {"variable": "vct", "time_unit": "annual", "grid": "veg_grid_cell_v001"}},
            "steps": [
                {"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "count_by_value"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert {"grid_cell", "value", "count", "geographic_unit"} <= set(df.columns)

        for zone_id in (1, 2):
            values, counts = np.unique(categories[zones == zone_id], return_counts=True)
            expected = dict(zip(values.tolist(), counts.tolist()))
            zone_df = df.filter(pl.col("grid_cell") == zone_id)
            actual = dict(zip(zone_df["value"].to_list(), zone_df["count"].to_list()))
            assert actual == expected


class TestRasterTiles:
    def _tile(self, raster_data, zoom=8):
        with rio.open(next(raster_data.rglob("*.tif"))) as src:
            from rasterio.warp import transform_bounds

            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return mercantile.tile((west + east) / 2, (south + north) / 2, zoom)

    def test_png(self, client, raster_data):
        tile = self._tile(raster_data)
        response = client.get(f"{RASTER_URL}/data/{tile.z}/{tile.x}/{tile.y}.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_jpg(self, client, raster_data):
        tile = self._tile(raster_data)
        response = client.get(f"{RASTER_URL}/data/{tile.z}/{tile.x}/{tile.y}.jpg")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"

    def test_tile_outside_the_dataset_is_404(self, client, raster_data):
        assert client.get(f"{RASTER_URL}/data/2/0/0.png").status_code == 404

    def test_unknown_tile_format_is_422(self, client, raster_data):
        tile = self._tile(raster_data)
        response = client.get(f"{RASTER_URL}/data/{tile.z}/{tile.x}/{tile.y}.gif")
        assert response.status_code == 422

    def test_multiband_tile_is_400(self, client, base_path, monkeypatch):
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        response = client.get(f"{RASTER_URL}/data/8/0/0.png")
        assert response.status_code == 400
        assert "more than one band" in response.json()["detail"]

    def test_color_map_is_accepted(self, client, raster_data):
        tile = self._tile(raster_data)
        response = client.get(
            f"{RASTER_URL}/data/{tile.z}/{tile.x}/{tile.y}.png?color_map=lnd_type"
        )
        assert response.status_code == 200


@pytest.fixture
def two_year_raster_data(base_path, monkeypatch):
    """Two COGs (2030 filled with 30, 2040 with 40), for band-weighting requests."""
    write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
    monkeypatch.setattr(config, "BASE_PATH", base_path)
    return base_path


class TestRetiredParams:
    """Every flat analysis parameter this class used to name (the indexed
    `calculate.N.expression`/`calculate.N.where` query parameters, plus `negate`/
    `multiply10`/`multiply100`/`group_aggregate`/`summarize_by`/`summarize_aggregate`/
    `summarize_grid_cell`/`unpivot`) is retired from the HTTP surface -- see
    `api.reject_retired_params`/`api._RETIRED_PARAMS`. What used to be end-to-end
    functional coverage of the indexed `calculate.N.*` encoding and the legacy
    negate/multiply sugar (weighted sums, band selection, tile rendering, the
    mixing-rejection rule) moved to `tests/api/test_pipeline_exec.py`'s
    `TestCalculateStepEndToEnd`, which exercises the same functionality through the
    `calculate`/`group` pipeline steps that are its only surviving spelling. What
    remains here is one rejection test per retired name -- proving each one 400s with a
    detail message naming its pipeline replacement, rather than being silently ignored
    the way FastAPI/pydantic treat any other unrecognized query parameter. (`group_by`
    and `value_where` have their own rejection tests in `TestVectorData` above.)
    """

    def test_negate_is_retired(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv", params={"negate": "calendar_year = 2030"})
        assert response.status_code == 400
        assert "negate" in response.json()["detail"]

    def test_multiply10_is_retired(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv", params={"multiply10": "calendar_year = 2030"})
        assert response.status_code == 400
        assert "multiply10" in response.json()["detail"]

    def test_multiply100_is_retired(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv", params={"multiply100": "calendar_year = 2030"})
        assert response.status_code == 400
        assert "multiply100" in response.json()["detail"]

    def test_group_aggregate_is_retired(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv", params={"group_aggregate": "sum"})
        assert response.status_code == 400
        assert "group_aggregate" in response.json()["detail"]

    def test_summarize_by_is_retired(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv", params={"summarize_by": "hydro_compartment_v001"})
        assert response.status_code == 400
        assert "summarize_by" in response.json()["detail"]

    def test_summarize_aggregate_is_retired(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv", params={"summarize_aggregate": "sum"})
        assert response.status_code == 400
        assert "summarize_aggregate" in response.json()["detail"]

    def test_summarize_grid_cell_is_retired(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv", params={"summarize_grid_cell": "1"})
        assert response.status_code == 400
        assert "summarize_grid_cell" in response.json()["detail"]

    def test_unpivot_is_retired(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv", params={"unpivot": "true"})
        assert response.status_code == 400
        assert "unpivot" in response.json()["detail"]

    def test_bare_calculate_parameter_is_retired(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/data.csv", params={"calculate": "value * 2"})
        assert response.status_code == 400
        assert "calculate" in response.json()["detail"]

    def test_indexed_calculate_parameters_are_retired(self, client, vector_data):
        response = client.get(
            f"{VECTOR_URL}/data.csv",
            params={
                "calculate.1.expression": "value * 0.5",
                "calculate.1.where": "calendar_year = 2030",
            },
        )
        assert response.status_code == 400
        assert "calculate" in response.json()["detail"]

    def test_retired_params_are_rejected_on_the_tile_route_too(self, client, raster_data):
        """`reject_retired_params` is wired onto the tile/visualization routes as well as
        the flat data route (see `api.get_model_tile`/`get_vector_tile`); spot-checked
        here on `.tif` rather than repeated per retired name."""
        response = client.get(f"{RASTER_URL}/data/8/0/0.png", params={"group_aggregate": "sum"})
        assert response.status_code == 400
        assert "group_aggregate" in response.json()["detail"]


class TestVectorTiles:
    def test_serves_a_tile(self, client, vector_data, tiles_path):
        tile = mercantile.Tile(x=1, y=2, z=3)
        write_pmtiles(tiles_path, "ecoregion_v001", tile, [1, 2])

        response = client.get(
            f"{VECTOR_URL}/tiles/{tile.z}/{tile.x}/{tile.y}.pbf?calendar_year=2030"
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.mapbox-vector-tile"

    def test_missing_pmtiles_file_is_404(self, client, vector_data, tiles_path):
        response = client.get(f"{VECTOR_URL}/tiles/3/1/2.pbf")
        assert response.status_code == 404
        assert "no tile file for grid" in response.json()["detail"]

    def test_raster_grid_is_400(self, client, raster_data, tiles_path):
        response = client.get(f"{RASTER_URL}/tiles/3/1/2.pbf")
        assert response.status_code == 400
        assert "invalid output type for vector tile" in response.json()["detail"]


class TestCatalogEndpoint:
    def test_returns_choices_results_and_exact(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/catalog.json")
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"choices", "results", "exact"}

    def test_reports_available_years(self, client, vector_data):
        """2030 and 2040 are ten apart, so an annual series reports them separately."""
        body = client.get(f"{VECTOR_URL}/catalog.json").json()
        assert body["choices"]["calendar_year"] == [2030, 2040]

    def test_contiguous_years_are_compressed_into_a_range(
        self, client, base_path, monkeypatch
    ):
        write_parquet(annual_frame(years=(2030, 2031, 2032)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        body = client.get(f"{VECTOR_URL}/catalog.json").json()
        assert body["choices"]["calendar_year"] == [[2030, 2032]]

    def test_selection_is_reflected_in_results(self, client, vector_data):
        body = client.get(f"{VECTOR_URL}/catalog.json?calendar_year=2040").json()
        assert body["results"]["calendar_year"] == [2040]

    def test_unavailable_selection_is_inexact(self, client, vector_data):
        body = client.get(f"{VECTOR_URL}/catalog.json?calendar_year=1999").json()
        assert body["exact"] is False


class TestVisualizationEndpoints:
    def test_dataset_visualization_returns_a_vega_spec(self, client, vector_data):
        response = client.get(
            f"{VECTOR_URL}/visualization/time-series-line.json?grid_cell=1&calendar_year=2030"
        )
        assert response.status_code == 200
        assert "$schema" in response.json()

    def test_dataset_visualization_accepts_coordinates_on_a_vector_grid(
        self, client, vector_data, ecoregion_geometry
    ):
        """The spec has to describe the data.csv it points at, which reports the resolved
        cell ID however the caller selected it — a number, whatever the request looked
        like. The coordinate survives only as a label."""
        label = _lng_lat(500500, 3300500)
        response = client.get(
            f"{VECTOR_URL}/visualization/time-series-line.json?grid_cell={label}&calendar_year=2030"
        )
        assert response.status_code == 200
        spec = response.json()
        assert label.replace(",", ", ") in json.dumps(spec)
        assert spec["data"]["format"]["parse"] == {
            "value": "number",
            "grid_cell": "number",
            "calendar_year": "number",
        }

    def test_time_series_line_labels_a_coordinate_with_its_grid_cell(
        self, client, vector_data, ecoregion_geometry
    ):
        """The legend and tooltips report the cell the coordinate resolved to, since that's
        what the values describe, while the data stays keyed by the coordinate itself. Only
        the legend repeats the coordinate — a tooltip row has no width to spare."""
        label = _lng_lat(500500, 3300500)
        response = client.get(
            f"{VECTOR_URL}/visualization/time-series-line.json?grid_cell={label}&calendar_year=2030"
        )
        assert response.status_code == 200
        spec = response.json()

        display = f"1 ({label.replace(',', ', ')})"
        line = spec["layer"][0]
        assert line["encoding"]["color"]["field"] == "grid_cell_label"
        assert line["encoding"]["color"]["scale"]["domain"] == [display]
        # The lookup that puts that label on each row is keyed by the API's own value.
        lookup = line["transform"][-1]
        assert lookup["lookup"] == "grid_cell"
        assert spec["datasets"][lookup["from"]["data"]["name"]] == [
            {"grid_cell_label_idx": 0, "grid_cell": 1, "grid_cell_label": display}
        ]
        tooltips = spec["layer"][2]["encoding"]["tooltip"]
        assert [t["title"] for t in tooltips[1:]] == ["1: "]

    @pytest.mark.parametrize("name", ["map", "stacked-bar", "time-series-line"])
    def test_every_dataset_visualization_accepts_coordinates(
        self, client, vector_data, ecoregion_geometry, name
    ):
        label = _lng_lat(500500, 3300500)
        response = client.get(
            f"{VECTOR_URL}/visualization/{name}.json?grid_cell={label}&calendar_year=2030"
        )
        assert response.status_code == 200

    def test_dataset_visualization_parses_grid_cell_ids_as_numbers(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/visualization/time-series-line.json?grid_cell=1")
        assert response.json()["data"]["format"]["parse"]["grid_cell"] == "number"

    def test_unknown_visualization_is_404(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/visualization/nope.json")
        assert response.status_code == 404

    def test_custom_visualization_rejected_on_the_dataset_route(self, client, vector_data):
        response = client.get(f"{VECTOR_URL}/visualization/sediment-dynamics.json?grid_cell=1")
        assert response.status_code == 400

    def test_dataset_visualization_rejected_on_the_custom_route(self, client, vector_data):
        response = client.get("/v1/visualizations/time-series-line.json?grid_cell=1")
        assert response.status_code == 400


class TestMapAnalysisTileRouting:
    """`Map._get_tile_url_template`/`_get_vector_tile_url_template` route to the named
    analysis's own tile endpoints (`/analyses/{name}/data/{z}/{x}/{y}.png` and
    `/analyses/{name}/tiles/{z}/{x}/{y}.pbf`) once `analysis=` is selected, forwarding only
    an allowlisted set of query parameters -- these go through the real ASGI app and the
    real `ANALYSES` registry (unlike `test_analyses_endpoints.py`'s fake-registry route
    tests), inspecting the `map.json` style spec's `sources.data.tiles[0]` URL, the same
    thing a MapLibre client would actually fetch.
    """

    def test_raster_map_routes_to_the_analysis_tile_endpoint(self, client, two_year_raster_data):
        response = client.get(
            f"{RASTER_URL}/visualization/map.json"
            "?analysis=difference&dimension=calendar_year&order=asc"
            "&calendar_year=2030&calendar_year=2040"
        )
        assert response.status_code == 200
        tile_url = response.json()["sources"]["data"]["tiles"][0]
        assert "/analyses/difference/data/" in tile_url
        assert "dimension=calendar_year" in tile_url
        assert "order=asc" in tile_url
        assert "calendar_year=2030" in tile_url
        assert "calendar_year=2040" in tile_url

    def test_raster_map_without_an_analysis_still_uses_the_plain_tile_endpoint(
        self, client, two_year_raster_data
    ):
        response = client.get(f"{RASTER_URL}/visualization/map.json?calendar_year=2030")
        assert response.status_code == 200
        tile_url = response.json()["sources"]["data"]["tiles"][0]
        assert "/analyses/" not in tile_url
        assert "/data/" in tile_url

    def test_vector_map_routes_to_the_analysis_vector_tile_endpoint(self, client, vector_data):
        response = client.get(
            f"{VECTOR_URL}/visualization/map.json"
            "?analysis=difference&dimension=calendar_year&order=asc"
            "&calendar_year=2030&calendar_year=2040"
        )
        assert response.status_code == 200
        tile_url = response.json()["sources"]["data"]["tiles"][0]
        assert "/analyses/difference/tiles/" in tile_url
        assert "dimension=calendar_year" in tile_url
        assert "order=asc" in tile_url

    def test_vector_map_over_a_date_dimension(self, client, salinity_data):
        """Regression, reported from the live portal: a difference map over `calendar_day`
        returned 400 ("dimension filter 'calendar_day' must be a scalar or a list of
        scalars, got date in it") for every daily dataset. The `calendar_year` cases above
        all bind to `int`, so nothing here exercised a dimension whose declared dtype makes
        FastAPI hand `resolve_analysis` a real `datetime.date`.
        """
        response = client.get(
            f"{SALINITY_URL}/visualization/map.json"
            "?analysis=difference&dimension=calendar_day&order=asc"
            "&calendar_day=2030-01-01&calendar_day=2030-01-02"
            "&model_group_id=500&scenario_id=7&aggregate_type=mean"
        )
        assert response.status_code == 200
        tile_url = response.json()["sources"]["data"]["tiles"][0]
        assert "/analyses/difference/tiles/" in tile_url
        assert "dimension=calendar_day" in tile_url
        assert "calendar_day=2030-01-01" in tile_url
        assert "calendar_day=2030-01-02" in tile_url

    def test_vector_map_without_an_analysis_still_uses_the_plain_vector_tile_endpoint(
        self, client, vector_data
    ):
        response = client.get(f"{VECTOR_URL}/visualization/map.json?calendar_year=2030")
        assert response.status_code == 200
        tile_url = response.json()["sources"]["data"]["tiles"][0]
        assert "/analyses/" not in tile_url
        assert "/tiles/" in tile_url

    def test_time_series_line_data_csv_url_also_forwards_the_analysis_parameters(
        self, client, vector_data
    ):
        """`get_data_uri` (base.py) has the exact same forwarding requirement as the tile
        URL templates above -- covered separately here since it's a different call site
        (the chart's own `data.csv` source, not a tile), reached through a visualization
        that actually has one (`Map`'s vector branch only builds a `data.csv` URL as a
        PMTiles-mode fallback with no `request`, so `time-series-line` is used here
        instead)."""
        response = client.get(
            f"{VECTOR_URL}/visualization/time-series-line.json"
            "?analysis=difference&dimension=calendar_year&order=asc"
            "&calendar_year=2030&calendar_year=2040&grid_cell=1"
        )
        assert response.status_code == 200
        data_url = response.json()["data"]["url"]
        assert "/analyses/difference/data.csv" in data_url
        assert "dimension=calendar_year" in data_url
        assert "order=asc" in data_url


class TestChartDataUrlIsFetchable:
    """A chart's emitted `data.url` must not just LOOK right -- it must come back 200.

    `TestMapAnalysisTileRouting` above asserts on the shape of the emitted URL (which
    parameters it forwards); these follow the URL through and actually fetch it. That
    distinction is the whole point: the difference time-series chart emitted a URL that
    passed every shape assertion and still 400'd, because `get_data_uri` added the
    analysis's own collapsed dimension to `select` -- a column the analysis's output frame
    no longer has ("unable to find column \\"model_group_id\\"").
    """

    def _data_url(self, client, query):
        response = client.get(f"{SALINITY_URL}/visualization/time-series-line.json?{query}")
        assert response.status_code == 200
        return response.json()["data"]["url"]

    def test_difference_over_model_group_id_is_fetchable(self, client, two_model_group_salinity_data):
        data_url = self._data_url(
            client,
            "analysis=difference&dimension=model_group_id&order=asc"
            "&model_group_id=500&model_group_id=521"
            "&scenario_id=7&aggregate_type=mean&grid_cell=1",
        )
        assert "/analyses/difference/data.csv" in data_url
        assert client.get(data_url).status_code == 200

    def test_the_collapsed_dimension_is_not_selected_back(
        self, client, two_model_group_salinity_data
    ):
        """It still has to be SENT -- the two values are what the difference is taken
        between -- so this pins the asymmetry, not just the absence."""
        data_url = self._data_url(
            client,
            "analysis=difference&dimension=model_group_id&order=asc"
            "&model_group_id=500&model_group_id=521"
            "&scenario_id=7&aggregate_type=mean&grid_cell=1",
        )
        query = parse_qs(urlparse(data_url).query)
        assert query["model_group_id"] == ["500", "521"]
        assert "model_group_id" not in query["select"][0].split(",")

    def test_a_collapsed_time_column_is_also_dropped(self, client, two_model_group_salinity_data):
        """The time column reaches `select` by a DIFFERENT path than the multi-value
        dimensions -- it goes in unconditionally -- so a `difference` over `calendar_day`
        hit the identical "unable to find column" failure even once the multi-value path was
        guarded. Guarding the finished list instead of each append is what covers both.

        (A time-series chart differencing across time has no time axis left to plot; that
        degeneracy is the portal's to prevent. This only ensures the request itself is
        well-formed rather than a 400.)
        """
        data_url = self._data_url(
            client,
            "analysis=difference&dimension=calendar_day&order=asc"
            "&calendar_day=2030-01-01&calendar_day=2030-01-02"
            "&model_group_id=500&model_group_id=521"
            "&scenario_id=7&aggregate_type=mean&grid_cell=1",
        )
        query = parse_qs(urlparse(data_url).query)
        assert "calendar_day" not in query["select"][0].split(",")
        assert query["calendar_day"] == ["2030-01-01", "2030-01-02"]
        assert client.get(data_url).status_code == 200

    def test_a_multi_valued_dimension_the_analysis_keeps_is_still_selected_back(
        self, client, two_model_group_salinity_data
    ):
        """The guard must be narrow: an uncollapsed dimension given two values is a real
        compare channel and still belongs in `select`. Here the difference is taken over
        `calendar_day`, so `model_group_id` survives the analysis and must come back."""
        data_url = self._data_url(
            client,
            "analysis=difference&dimension=calendar_day&order=asc"
            "&calendar_day=2030-01-01&calendar_day=2030-01-02"
            "&model_group_id=500&model_group_id=521"
            "&scenario_id=7&aggregate_type=mean&grid_cell=1",
        )
        query = parse_qs(urlparse(data_url).query)
        assert "model_group_id" in query["select"][0].split(",")
        assert client.get(data_url).status_code == 200

    def test_without_an_analysis_a_multi_valued_dimension_is_still_selected_back(
        self, client, two_model_group_salinity_data
    ):
        data_url = self._data_url(
            client,
            "model_group_id=500&model_group_id=521&scenario_id=7&aggregate_type=mean&grid_cell=1",
        )
        query = parse_qs(urlparse(data_url).query)
        assert "model_group_id" in query["select"][0].split(",")
        assert client.get(data_url).status_code == 200


class TestCors:
    """The portal is served from a different origin than the API, so CORS must permit it.

    Whether the allow-origin header comes back as `*` or as the echoed origin is a
    Starlette implementation detail that has changed across releases (the condition moved
    from `allow_all_origins and has_cookie` to `allow_all_origins and allow_credentials`),
    so these assert only that the requesting origin is permitted.
    """

    ORIGIN = "https://example.com"

    def test_request_origin_is_permitted(self, client, vector_data):
        response = client.get("/v1/variables.json", headers={"Origin": self.ORIGIN})
        assert response.headers["access-control-allow-origin"] in ("*", self.ORIGIN)

    def test_credentials_are_allowed(self, client, vector_data):
        response = client.get("/v1/variables.json", headers={"Origin": self.ORIGIN})
        assert response.headers["access-control-allow-credentials"] == "true"

    def test_preflight_advertises_methods_and_headers(self, client, vector_data):
        response = client.options(
            f"{VECTOR_URL}/data.csv",
            headers={
                "Origin": self.ORIGIN,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-Custom",
            },
        )
        assert response.status_code == 200
        assert "GET" in response.headers["access-control-allow-methods"]
        assert "X-Custom" in response.headers["access-control-allow-headers"]
