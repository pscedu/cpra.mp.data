"""End-to-end tests for hand-written pipelines over the data routes
(`cpra/mp/data/api/pipeline.py`), through the real ASGI app rather than against the engine's
internals.

Every test here posts (or GETs) a JSON-shaped pipeline at one of the two paths the
consolidation left -- `/v1/data.<format>` (POST with a JSON body, GET with the dotted query
codec) and `/v1/data/{z}/{x}/{y}.<format>` (`png` and `pbf`) -- and asserts on the HTTP
response. A pipeline written by hand is just one of the things those paths accept now;
`analysis=` and the dataset triple are the others, and a request may carry all three. The layers underneath
(`request.pipeline`'s parse/validate, `execution.plan`'s compile, `execution.execute`'s
walk, `execution.sink`'s write) each have their own focused tests; what this module pins
is that the routes wire them together, that the request encodings agree with each other,
and that a rejection reaches the caller as a 4xx with a message naming the problem.

**On variable/grid choices below.** The paired shape's fixtures (`notebook_query_data` and
its two siblings) use the variable/grid/dimension names from the
`sal_veg_investigation` named analysis (`ffibs_coverage_ratio`/`veg_grid_cell_v001`
paired against `sal`/`hydro_compartment_v001`). `tests/factories.py` has no ready-made
builder for either (its `annual_frame`/`write_raster_dataset` helpers are hardcoded to
`ssf`/`vct`), so this module writes them directly with `write_parquet`/`write_cog`,
mirroring those helpers' own shape. The single-input shapes, which have no such
requirement, reuse
`tests/factories.py`'s `ssf`/`ecoregion_v001` and `vct`/`veg_grid_cell_v001` fixtures
directly, per that module's own convention -- the engine cares about grid TYPES and
SHAPES, not real-world variable semantics.
"""

import copy
import io
import json
from typing import ClassVar

import mercantile
import numpy as np
import polars as pl
import pytest
import rasterio as rio
from rasterio.warp import transform_bounds

from cpra.mp.data import config
from cpra.mp.data.analysis.request.pipeline import parse_pipeline, pipeline_to_query
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.raster import write_cog
from cpra.mp.data.tabular import write_parquet
from tests.factories import (
    ANNUAL_KWARGS,
    RASTER_KWARGS,
    annual_frame,
    raster_array,
    write_raster_crosswalk,
    write_raster_dataset,
    write_vector_crosswalk,
)

# The nodata sentinel these routes write into a GeoTIFF. Spelled as a literal rather than
# imported from `execution.sink.DEFAULT_NODATA`, because what these tests pin is the value
# a client reads back out of the file, not the constant that happens to produce it: the
# `.tif` route passes no `nodata=`, so the sink's default is what lands on disk, and the
# same number is used regardless of the band dtype.
NODATA = -9999

# The `sal_veg_investigation` analysis's own shape, spelled as a raw pipeline body:
# two inputs on different grids, one `project`ed onto the other's grid, joined, tested
# against both value columns by a single-armed `case`, and collapsed to the earliest
# calendar year that satisfies the test.
#
# An input here names a dataset and nothing else -- `request.pipeline.InputSelection` is
# `extra="forbid"` with only `variable`/`time_unit`/`grid`, by design (see its own
# docstring: "An input names a dataset; it does not filter one ... every filter ... is
# expressed as a `filter` step instead"). The fixtures backing this query write exactly one
# value per dimension, so there is no dimension `filter` step to add here.
NOTEBOOK_QUERY = {
    "inputs": {
        "marsh": {
            "variable": "ffibs_coverage_ratio",
            "time_unit": "annual",
            "grid": "veg_grid_cell_v001",
        },
        "salinity": {
            "variable": "sal",
            "time_unit": "annual",
            "grid": "hydro_compartment_v001",
        },
    },
    "steps": [
        {"op": "project", "input": "salinity", "to": "veg_grid_cell_v001", "as": "sal"},
        {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
        {"op": "case", "cases": [{"when": "marsh > 0.5 AND sal > 5.5", "then": "marsh"}]},
        {"op": "group", "by": [], "aggregate": "min", "of": "calendar_year"},
    ],
    "output": {"format": "tif"},
}


def _write_ffibs_raster(base_path, values_by_year):
    """`ffibs_coverage_ratio`/`veg_grid_cell_v001`, uniformly filled per year -- there is
    no `tests/factories.py` builder for this real variable, so it's written directly with
    `write_cog`, mirroring `write_raster_dataset`'s own shape.
    """
    grid = GRIDS["veg_grid_cell_v001"]
    for year, value in values_by_year.items():
        data = np.full((grid.height, grid.width), value, dtype="float32")
        write_cog(
            data,
            base_path=base_path,
            calendar_year=year,
            variable="ffibs_coverage_ratio",
            grid="veg_grid_cell_v001",
            time_unit="annual",
            model_group_id=500,
            scenario_id=7,
            model="icm_v2023.0.0",
            ffibs_type="pl_fm",
        )


def _write_ffibs_raster_arrays(base_path, arrays_by_year):
    """Like `_write_ffibs_raster`, but each year gets an arbitrary per-pixel array
    rather than a uniform fill -- needed to exercise the paired per-pixel band collapse
    across pixels that differ in how many bands validate the `where` clause (all of them,
    some of them, or none), which a uniform raster can never produce.
    """
    for year, data in arrays_by_year.items():
        write_cog(
            data.astype("float32"),
            base_path=base_path,
            calendar_year=year,
            variable="ffibs_coverage_ratio",
            grid="veg_grid_cell_v001",
            time_unit="annual",
            model_group_id=500,
            scenario_id=7,
            model="icm_v2023.0.0",
            ffibs_type="pl_fm",
        )


def _write_sal_annual(base_path, values_by_year, geo_id=1, **overrides):
    """`sal`/`hydro_compartment_v001`, annual, one geo column -- no `tests/factories.py`
    builder covers annual `sal` (only `DAILY_KWARGS`/`daily_frame` do, with
    `aggregate_type="mean"`), so this mirrors `annual_frame`'s wide-then-unpivot shape
    directly. `validate=False`: `hydro_compartment_v001` has 1,657 real cells, and only
    one is needed here (`write_vector_crosswalk`/`write_raster_crosswalk`'s counterpart
    for the pairing side is the crosswalk raster itself, not this frame).
    """
    years = sorted(values_by_year)
    kwargs = {
        "model_group_id": 500,
        "scenario_id": 7,
        "aggregate_type": "2_wk_max",
        "model": "icm_v2023.0.0",
        **overrides,
    }
    df = pl.DataFrame(
        {
            "variable": pl.Series(["sal"] * len(years), dtype=pl.String),
            "grid": pl.Series(["hydro_compartment_v001"] * len(years), dtype=pl.String),
            "time_unit": pl.Series(["annual"] * len(years), dtype=pl.String),
            "model_group_id": pl.Series([kwargs["model_group_id"]] * len(years), dtype=pl.Int32),
            "scenario_id": pl.Series([kwargs["scenario_id"]] * len(years), dtype=pl.Int32),
            "aggregate_type": pl.Series([kwargs["aggregate_type"]] * len(years), dtype=pl.String),
            "calendar_year": pl.Series(years, dtype=pl.Int32),
            "model": pl.Series([kwargs["model"]] * len(years), dtype=pl.String),
            str(geo_id): pl.Series([values_by_year[y] for y in years], dtype=pl.Float32),
        }
    )
    write_parquet(df, base_path=base_path, validate=False)


@pytest.fixture
def notebook_query_data(base_path, crosswalk_path, monkeypatch):
    """Backing data for `NOTEBOOK_QUERY`: `marsh` (the raster-grid input) is
    0.3 in 2030 and 0.7 in 2040 (crosses the query's own `marsh > 0.5`); `salinity` (the
    vector-grid input, one zone) is 4.0 in 2030 and 6.0 in 2040 (crosses `sal > 5.5`) -- so
    only 2040 satisfies BOTH conjuncts, at every pixel (the crosswalk below maps every
    `veg_grid_cell_v001` pixel to the same single `hydro_compartment_v001` zone), making
    `{"op": "group", "aggregate": "min", "of": "calendar_year"}`'s expected output a
    UNIFORM 2040 raster -- an easy, exact assertion.
    """
    _write_ffibs_raster(base_path, {2030: 0.3, 2040: 0.7})
    _write_sal_annual(base_path, {2030: 4.0, 2040: 6.0}, geo_id=1)
    monkeypatch.setattr(config, "BASE_PATH", base_path)

    grid = GRIDS["veg_grid_cell_v001"]
    zones = np.ones((grid.height, grid.width), dtype="int32")
    write_raster_crosswalk(crosswalk_path, "veg_grid_cell_v001", "hydro_compartment_v001", zones)
    return base_path


@pytest.fixture
def accumulator_edge_case_data(base_path, crosswalk_path, monkeypatch):
    """Backing data for `TestPairedAccumulatorEdgeCases`: `sal` is a uniform 6.0 in both
    2030 and 2040 (always satisfies the query's `sal > 5.5`), while `marsh` varies by ROW
    band so `marsh > 0.5` differs pixel-by-pixel in how many of the two bands satisfy it
    -- a uniform raster (as `notebook_query_data` uses) can never validate the streaming
    accumulator's per-pixel `any_valid`/valid-count bookkeeping, only its arithmetic.
    Three equal row bands:

    * rows [0, h/3): `marsh` is 0.3 in both years -- NEVER satisfies `marsh > 0.5`, so
      the pixel has zero valid contributing bands in every aggregate (`any_valid` stays
      False) and must come out as the raster sink's nodata sentinel
      (`execution.sink.DEFAULT_NODATA`, -9999).
    * rows [h/3, 2h/3): `marsh` is 0.7 in 2030, 0.2 in 2040 -- satisfies the where-clause
      in ONLY 2030, so exactly one band contributes.
    * rows [2h/3, h): `marsh` is 0.6 in both years -- satisfies the where-clause in BOTH
      bands.
    """
    grid = GRIDS["veg_grid_cell_v001"]
    third = grid.height // 3
    marsh_2030 = np.empty((grid.height, grid.width), dtype="float32")
    marsh_2040 = np.empty((grid.height, grid.width), dtype="float32")
    marsh_2030[:third, :] = 0.3
    marsh_2040[:third, :] = 0.3
    marsh_2030[third : 2 * third, :] = 0.7
    marsh_2040[third : 2 * third, :] = 0.2
    marsh_2030[2 * third :, :] = 0.6
    marsh_2040[2 * third :, :] = 0.6

    _write_ffibs_raster_arrays(base_path, {2030: marsh_2030, 2040: marsh_2040})
    _write_sal_annual(base_path, {2030: 6.0, 2040: 6.0}, geo_id=1)
    monkeypatch.setattr(config, "BASE_PATH", base_path)

    zones = np.ones((grid.height, grid.width), dtype="int32")
    write_raster_crosswalk(crosswalk_path, "veg_grid_cell_v001", "hydro_compartment_v001", zones)
    return base_path, third


@pytest.fixture
def count_edge_case_data(base_path, crosswalk_path, monkeypatch):
    """Backing data for `TestPairedCountConsistency`: five equal row bands, one per
    zero-vs-nodata case `count` has to distinguish. `sal` (zone 1 only) is a uniform 6.0
    in both years (always satisfies the query's `sal > 5.5`), so every zone-1 band's
    count outcome below is driven purely by `marsh`/nodata/pairing, exactly as
    `accumulator_edge_case_data` isolates `marsh` alone for the other aggregates.

    * rows [0, h/5): zone 1 (a real zone the vector `sal` table has data for); `marsh` is
      0.3 in both years, never crossing `marsh > 0.5` -- ALL BANDS FAIL the `where`
      clause, in-domain -> a real count of 0.
    * rows [h/5, 2h/5): zone 1; `marsh` is `write_cog`'s own fixed raster nodata
      sentinel (-9999) in both years -- RASTER NODATA in every band, in-domain -> 0.
    * rows [2h/5, 3h/5): zone 2 -- a real (nonzero) crosswalk zone id, but one the vector
      `sal` table carries no column for in EITHER year (only zone 1's "1" column is ever
      written) -- a PAIRING MISS in every band regardless of `marsh`'s own value,
      in-domain -> 0.
    * rows [3h/5, 4h/5): zone 0 -- the crosswalk raster's own nodata value
      (`write_raster_crosswalk` fixes it at 0) -- OUTSIDE the pairing crosswalk's domain
      entirely -> the nodata sentinel (`execution.sink.DEFAULT_NODATA`, -9999), never a
      real 0, regardless of `marsh`/`sal`.
    * rows [4h/5, h): zone 1; `marsh` crosses 0.5 in 2030 only -> a MIXED pixel, exactly
      one contributing band -> a count of 1.
    """
    grid = GRIDS["veg_grid_cell_v001"]
    fifth = grid.height // 5
    zones = np.empty((grid.height, grid.width), dtype="int32")
    zones[:fifth, :] = 1  # all-bands-fail
    zones[fifth : 2 * fifth, :] = 1  # raster-nodata-all-bands
    zones[2 * fifth : 3 * fifth, :] = 2  # pairing-miss-all-bands (no vector data for zone 2)
    zones[3 * fifth : 4 * fifth, :] = 0  # outside-zone (crosswalk nodata)
    zones[4 * fifth :, :] = 1  # mixed

    marsh_2030 = np.empty((grid.height, grid.width), dtype="float32")
    marsh_2040 = np.empty((grid.height, grid.width), dtype="float32")
    marsh_2030[:fifth, :] = 0.3
    marsh_2040[:fifth, :] = 0.3
    marsh_2030[fifth : 2 * fifth, :] = -9999.0
    marsh_2040[fifth : 2 * fifth, :] = -9999.0
    marsh_2030[2 * fifth : 3 * fifth, :] = 0.6
    marsh_2040[2 * fifth : 3 * fifth, :] = 0.6
    marsh_2030[3 * fifth : 4 * fifth, :] = 0.6
    marsh_2040[3 * fifth : 4 * fifth, :] = 0.6
    marsh_2030[4 * fifth :, :] = 0.7
    marsh_2040[4 * fifth :, :] = 0.2

    _write_ffibs_raster_arrays(base_path, {2030: marsh_2030, 2040: marsh_2040})
    # Only zone 1 ever gets a vector `sal` column -- zone 2 is a real crosswalk id with
    # no matching data in either year, the "pairing-miss" case above.
    _write_sal_annual(base_path, {2030: 6.0, 2040: 6.0}, geo_id=1)
    monkeypatch.setattr(config, "BASE_PATH", base_path)

    write_raster_crosswalk(crosswalk_path, "veg_grid_cell_v001", "hydro_compartment_v001", zones)
    return base_path, fifth


def _tile_for_grid(base_path, grid_glob="*.tif", zoom=8):
    with rio.open(next(base_path.rglob(grid_glob))) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return mercantile.tile((west + east) / 2, (south + north) / 2, zoom)


# --------------------------------------------------------------------------------------
# The paired (two-input, band-wide join) shape
# --------------------------------------------------------------------------------------


class TestNotebookQueryEndToEnd:
    """`NOTEBOOK_QUERY`, byte-exact, through the real ASGI routes.

    The paired shape these exercise is the one that puts a raw raster input (`marsh`, on
    `veg_grid_cell_v001`) and a vector input broadcast onto that same grid (`sal`, via
    `execution.project`'s crosswalk) through a single band-wide `join`. Getting the two
    operands onto the same pixel grid is not free: `execution.frame.assert_same_pixel_grid`
    compares `ReadWindow.batch_layout`, and band count drives chunking, so a `project` that
    opened its crosswalk in its own one-band read would land its pixels somewhere other
    than where the four-band `marsh` read put them. Both operands therefore come off the
    ONE `scan_raster(..., crosswalks=...)` the join group plans -- see
    `execution.plan.ExecutionPlan.band_wide_join_zone_crosswalks`. These tests are the
    end-to-end proof of that, which is why they assert pixel values and not just a 200.
    """

    def test_post(self, client, notebook_query_data):
        response = client.post("/v1/data.tif", json=NOTEBOOK_QUERY)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
        assert (data == 2040).all()

    def test_get_dotted_query(self, client, notebook_query_data):
        query = pipeline_to_query(parse_pipeline(NOTEBOOK_QUERY))
        response = client.get("/v1/data.tif", params=query)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
        assert (data == 2040).all()

    def test_get_png_tile(self, client, notebook_query_data):
        tile_query = copy.deepcopy(NOTEBOOK_QUERY)
        tile_query["output"] = {"format": "png"}
        query = pipeline_to_query(parse_pipeline(tile_query))
        tile = _tile_for_grid(notebook_query_data)
        response = client.get(f"/v1/data/{tile.z}/{tile.x}/{tile.y}.png", params=query)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_pbf_tile_rejects_the_paired_shape_gracefully(self, client, notebook_query_data):
        """This shape's output is always raster (band-wide on `veg_grid_cell_v001`), so a
        `.pbf` request against it must 400 -- and, since a `.pbf` request never reaches
        `execute()` at all (the route overwrites `output.format` with the URL's `pbf` and
        `validate_shapes`'s `pbf` rule rejects from the declared ops alone, before any data
        is read or any plan is compiled), this one test in the class is NOT affected by the
        join/batch_layout gap the rest of the class documents.

        Note what the query says and what it does not: `NOTEBOOK_QUERY`'s own
        `output.format` is `tif`, and it is the URL that makes this a vector-tile request.
        """
        query = pipeline_to_query(parse_pipeline(NOTEBOOK_QUERY))
        tile = _tile_for_grid(notebook_query_data)
        response = client.get(f"/v1/data/{tile.z}/{tile.x}/{tile.y}.pbf", params=query)
        assert response.status_code == 400, response.text
        assert "is not a long frame on a vector grid" in response.json()["detail"]

    def test_raster_tile_pipeline_error_is_a_400_not_a_500(self, client, notebook_query_data, monkeypatch):
        """The sibling of the `.pbf` case above, for the RASTER tile route: a `RequestError`
        raised while rendering a tile must reach the caller as 400, not an unhandled 500.

        `api.pipeline.respond_to_analysis_tile` is a straight `compile_plan -> execute ->
        sink.render_tile` sequence run inside `anyio.to_thread.run_sync`, so the seam
        forced to fail here is `api.pipeline.execute` (imported by name into that module,
        hence patchable there), standing in for any of the real subclasses `execute()` can
        genuinely raise. The boundary mapping for `RequestError` itself is exercised
        end-to-end in `tests/api/test_analysis_boundary_module.py`; this test is only about
        the tile route's `anyio.to_thread.run_sync` propagation wiring.
        """
        from cpra.mp.data.analysis.request.errors import RequestError

        def _boom(*_args, **_kwargs):
            raise RequestError("no data available")

        monkeypatch.setattr("cpra.mp.data.api.pipeline.execute", _boom)
        tile_query = copy.deepcopy(NOTEBOOK_QUERY)
        tile_query["output"] = {"format": "png"}
        query = pipeline_to_query(parse_pipeline(tile_query))
        tile = _tile_for_grid(notebook_query_data)
        response = client.get(f"/v1/data/{tile.z}/{tile.x}/{tile.y}.png", params=query)
        assert response.status_code == 400, response.text
        assert "no data available" in response.json()["detail"]


class TestPairedAccumulatorEdgeCases:
    """The paired per-pixel band collapse over `sum`/`mean`/`min`/`max` -- `min` is already
    exercised byte-exact by `TestNotebookQueryEndToEnd` (via a uniform raster), but a
    uniform raster can't distinguish "zero bands validate" from "all bands validate" or
    exercise `mean`'s divide-by-VALID-count (not divide-by-total-bands) semantics. These
    use `accumulator_edge_case_data`'s three row bands -- see that fixture's docstring --
    to pin those cases against hand-computed values.

    What `sum` and `mean` are really pinning is that the collapse counts CONTRIBUTORS, not
    bands: a band that failed the paired value filter contributes nothing, and must not be
    silently read as a zero (which would drag `mean` down and inflate `sum`). A pixel with
    no contributor at all is nodata rather than 0 -- that distinction belongs to `count`
    alone, and `TestPairedCountConsistency` owns it.
    """

    def _pipeline(self, aggregate):
        query = copy.deepcopy(NOTEBOOK_QUERY)
        query["steps"][-1]["aggregate"] = aggregate
        return query

    def test_sum_aggregate_by_valid_band_count(self, client, accumulator_edge_case_data):
        _base_path, third = accumulator_edge_case_data
        response = client.post("/v1/data.tif", json=self._pipeline("sum"))
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
            nodata = src.nodata
        assert nodata == NODATA
        # Never-valid band: no contributor at all -> nodata, not 0 (a naive sum-with-
        # null-as-zero over an ALL-null row would wrongly emit 0 here).
        assert (data[:third, :] == NODATA).all()
        # Valid only in 2030 -> sum is that single contributor, not padded with a zero
        # for 2040's non-contributing band.
        assert (data[third : 2 * third, :] == 2030).all()
        # Valid in both bands -> sum of both.
        assert (data[2 * third :, :] == 2030 + 2040).all()

    def test_mean_aggregate_divides_by_valid_count_not_band_count(self, client, accumulator_edge_case_data):
        _base_path, third = accumulator_edge_case_data
        response = client.post("/v1/data.tif", json=self._pipeline("mean"))
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
            nodata = src.nodata
        assert nodata == NODATA
        assert (data[:third, :] == NODATA).all()
        # One valid contributor -> mean IS that contributor (2030), not (2030 + 0) / 2 --
        # the bug a "divide by total band count" implementation would produce.
        assert (data[third : 2 * third, :] == 2030).all()
        # Two valid contributors -> their arithmetic mean.
        assert (data[2 * third :, :] == (2030 + 2040) // 2).all()

    def test_max_aggregate_all_masked_row_is_nodata(self, client, accumulator_edge_case_data):
        _base_path, third = accumulator_edge_case_data
        response = client.post("/v1/data.tif", json=self._pipeline("max"))
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
            nodata = src.nodata
        assert nodata == NODATA
        assert (data[:third, :] == NODATA).all()
        assert (data[third : 2 * third, :] == 2030).all()
        assert (data[2 * third :, :] == 2040).all()


class TestPairedCountConsistency:
    """`count` on a paired pipeline: the zero-vs-nodata fixture scenarios (`count_edge_
    case_data`'s five row bands) and an ASGI tile test.

    `count` is the only aggregate for which "no band contributed" and "this pixel was
    never in the crosswalk's domain" are different answers -- 0 and nodata respectively.
    The other four reduce both to nodata, so nothing else in this module can tell whether
    `execution.frame.DOMAIN_COLUMN` is being carried and honored. These tests can.
    """

    def _pipeline(self):
        query = copy.deepcopy(NOTEBOOK_QUERY)
        query["steps"][-1] = {"op": "group", "by": [], "aggregate": "count"}
        return query

    def test_zero_vs_nodata_and_mixed_pixel(self, client, count_edge_case_data):
        _base_path, fifth = count_edge_case_data
        response = client.post("/v1/data.tif", json=self._pipeline())
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
            nodata = src.nodata
        assert nodata == NODATA
        assert data.dtype == np.int32
        assert (data[:fifth, :] == 0).all()  # all-bands-fail (in-domain) -> real 0
        assert (data[fifth : 2 * fifth, :] == 0).all()  # raster-nodata-all-bands -> real 0
        assert (data[2 * fifth : 3 * fifth, :] == 0).all()  # pairing-miss-all-bands -> real 0
        assert (data[3 * fifth : 4 * fifth, :] == NODATA).all()  # outside-zone -> nodata
        assert (data[4 * fifth :, :] == 1).all()  # mixed pixel -> exact count

    def test_get_png_tile(self, client, count_edge_case_data):
        base_path, _fifth = count_edge_case_data
        tile_query = self._pipeline()
        tile_query["output"] = {"format": "png"}
        query = pipeline_to_query(parse_pipeline(tile_query))
        tile = _tile_for_grid(base_path)
        response = client.get(f"/v1/data/{tile.z}/{tile.x}/{tile.y}.png", params=query)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------------------
# Single-input pipelines
# --------------------------------------------------------------------------------------


class TestSingleInputEndToEnd:
    """Single-input pipelines equivalent to a flat dataset request.

    `filter`/`group` cannot open a pipeline on their own -- an opening step must name its
    `input` explicitly, which `FilterStep`/`CalculateStep.input` exist for. A same-grid
    `project` is NOT the way to satisfy that rule: `ProjectStep` is "Move a dataset across
    grids via a crosswalk", and `request.ops`'s lowering rules enforce that literally --
    a same-grid vector project needs an explicit `aggregate` (there is no default, since
    even an identity crosswalk is, in general, "a target cell may draw from any number of
    source cells"), and a raster-to-raster project is rejected outright (no
    `PROJECT_LOWERING_TABLE` entry -- "there is no crosswalk between two raster lattices").

    A raw (unprojected) input's value column is always literally `"value"`, never the
    input's own dict key, so predicates and expressions below reference `"value"`.
    """

    VECTOR_PIPELINE: ClassVar[dict] = {
        "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
        "steps": [
            {"op": "filter", "input": "eco", "where": "value > 2035"},
            {"op": "group", "by": [], "aggregate": "max"},
        ],
        "output": {"format": "csv"},
    }

    def test_vector_filter_group(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        response = client.post("/v1/data.csv", json=self.VECTOR_PIPELINE)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert df.height == 32  # one row per ecoregion -- 2030 fails `value > 2035` everywhere.
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))
        for geo_id in range(1, 33):
            assert actual[geo_id] == pytest.approx(2040 + geo_id / 100, abs=1e-3)

    def test_raster_filter_group(self, client, base_path, monkeypatch):
        """A raster counterpart, opened with `filter(input=vct, where="calendar_year > 0")`
        -- true for every real year, so it keeps every band and only serves to name the
        opening input.
        """
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "filter", "input": "vct", "where": "calendar_year > 0"},
                {"op": "group", "by": [], "aggregate": "max"},
            ],
            "output": {"format": "tif"},
        }
        response = client.post("/v1/data.tif", json=pipeline)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
        assert (data == 40).all()

    def test_get_dotted_query_matches_post(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        query = pipeline_to_query(parse_pipeline(self.VECTOR_PIPELINE))
        post_response = client.post("/v1/data.csv", json=self.VECTOR_PIPELINE)
        get_response = client.get("/v1/data.csv", params=query)
        assert get_response.status_code == 200, get_response.text
        # Row order isn't guaranteed stable across two independent query-plan executions
        # (no explicit sort in this pipeline), so compare parsed/sorted frames rather than
        # raw bytes.
        post_df = pl.read_csv(io.BytesIO(post_response.content)).sort("grid_cell")
        get_df = pl.read_csv(io.BytesIO(get_response.content)).sort("grid_cell")
        assert get_df.equals(post_df)


class TestFilterWhereDimensionNot:
    """A pipeline `filter` step's `where` must accept exactly what the flat `where=`
    parameter accepts for a dimension-only conjunct -- including a `not (...)` group, which
    the portal actually emits (e.g. `not (model_group_id=X and <dim>=Y)`).

    The assertion here isn't just "the pipeline runs" -- it's that the pipeline surface and
    the flat `where=` surface, given the logically identical predicate, return the SAME
    ROWS.
    """

    def test_not_filter_matches_flat_where_equivalent(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        # The predicate references only dimension/spatial columns (`calendar_year`/
        # `grid_cell`), never the value column.
        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "filter", "input": "eco", "where": "not (calendar_year = 2030 and grid_cell = 5)"},
            ],
            "output": {"format": "csv"},
        }
        pipeline_response = client.post("/v1/data.csv", json=pipeline)
        assert pipeline_response.status_code == 200, pipeline_response.text
        pipeline_df = pl.read_csv(io.BytesIO(pipeline_response.content)).sort(["calendar_year", "grid_cell"])

        # One row -- (calendar_year=2030, grid_cell=5) -- is excluded out of 64 total
        # (32 grid cells x 2 years); every other combination is kept.
        assert pipeline_df.height == 63
        assert not ((pipeline_df["calendar_year"] == 2030) & (pipeline_df["grid_cell"] == 5)).any()

        flat_response = client.get(
            "/v1/data.csv",
            params={
                "variable": "ssf",
                "time_unit": "annual",
                "grid": "ecoregion_v001",
                "where": "not (calendar_year = 2030 and grid_cell = 5)",
            },
        )
        assert flat_response.status_code == 200, flat_response.text
        flat_df = pl.read_csv(io.BytesIO(flat_response.content)).sort(["calendar_year", "grid_cell"])

        assert flat_df.select(["calendar_year", "grid_cell", "value"]).equals(
            pipeline_df.select(["calendar_year", "grid_cell", "value"])
        )


class TestInputGridCellEndToEnd:
    """A `grid_cell` filter, through both request encodings, restricting the input the same
    way a flat `grid_cell=` query parameter would.

    Spelled with `OR` rather than `IN (5, 6)` only to keep this test independent of
    `TestPostProjectGridCellFilterEndToEnd.test_grid_cell_in_lowers_onto_summarize_grid_cell`,
    which already covers the `IN` spelling.
    """

    PIPELINE: ClassVar[dict] = {
        "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
        "steps": [{"op": "filter", "input": "eco", "where": "grid_cell = 5 OR grid_cell = 6"}],
        "output": {"format": "csv"},
    }

    def test_grid_cell_filters_the_input(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        response = client.post("/v1/data.csv", json=self.PIPELINE)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert sorted(df["grid_cell"].to_list()) == [5, 6]

    def test_get_dotted_query_matches_post(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        query = pipeline_to_query(parse_pipeline(self.PIPELINE))
        assert query["step.1.where"] == "grid_cell = 5 OR grid_cell = 6"
        post_df = pl.read_csv(io.BytesIO(client.post("/v1/data.csv", json=self.PIPELINE).content)).sort("grid_cell")
        get_response = client.get("/v1/data.csv", params=query)
        assert get_response.status_code == 200, get_response.text
        get_df = pl.read_csv(io.BytesIO(get_response.content)).sort("grid_cell")
        assert get_df.equals(post_df)


class TestOutputSortSelect:
    """`output.sort`/`output.select` actually sort and narrow the response body."""

    def test_sort_and_select_applied(self, client, base_path, monkeypatch):
        # `sort` accepts DIMENSION names, matching the flat endpoints' own convention
        # (`test_endpoints_data.py`'s `sort=-calendar_year`).
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        # A zero-step, one-input pipeline is legal (`validate_pipeline`'s rule: a zero-step
        # pipeline must have exactly one input), so no opening step is needed here.
        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [],
            "output": {"format": "csv", "sort": "-calendar_year", "select": "calendar_year,value"},
        }
        response = client.post("/v1/data.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert list(df.columns) == ["calendar_year", "value"]
        assert df["calendar_year"].to_list() == sorted(df["calendar_year"].to_list(), reverse=True)


class TestOutputColorMap:
    """`output.color_map` actually changes a rendered raster tile's pixels for a
    SINGLE-INPUT pipeline -- a dataset-scoped pipeline rendered as a raster PNG tile
    (`/v1/data/{z}/{x}/{y}.png`) is the portal's difference-map shape on a raster grid.
    """

    def test_color_map_changes_rendered_pixels(self, client, base_path, monkeypatch):
        # A uniform fill of 30 falls in a distinct bucket under both legends used below:
        # `default`'s "0 to 49" bucket vs. `ffibs`'s last (open-ended, > 17) "Saline
        # Marsh" bucket -- so the two color maps are guaranteed to paint different colors
        # for the same pixels, not just POSSIBLY differ depending on how the fixture data
        # happens to fall relative to two arbitrary legends' bucket edges.
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 30)
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        tile = _tile_for_grid(base_path)

        def _pipeline(color_map):
            return {
                "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
                "steps": [],
                "output": {"format": "png", "color_map": color_map},
            }

        default_query = pipeline_to_query(parse_pipeline(_pipeline("default")))
        ffibs_query = pipeline_to_query(parse_pipeline(_pipeline("ffibs")))

        default_response = client.get(f"/v1/data/{tile.z}/{tile.x}/{tile.y}.png", params=default_query)
        ffibs_response = client.get(f"/v1/data/{tile.z}/{tile.x}/{tile.y}.png", params=ffibs_query)

        assert default_response.status_code == 200, default_response.text
        assert ffibs_response.status_code == 200, ffibs_response.text
        assert default_response.content != ffibs_response.content


# --------------------------------------------------------------------------------------
# project / grid_cell
# --------------------------------------------------------------------------------------


class TestProjectAggregateCountByValue:
    """`count_by_value` as a `project` aggregate -- the documented "Categorical Count per
    Zone" query (docs/data-analysis-api.md) expressed as a pipeline. Locally constructed
    categorical fixture, modeled on `test_endpoints_data.py`'s `categorical_vector_data`.
    """

    def test_categorical_count_per_zone(self, client, base_path, crosswalk_path, monkeypatch):
        # geo_id 1..16 -> zone 100, geo_id 17..32 -> zone 200; value = geo_id % 3, so
        # each zone's per-value counts are hand-computable below.
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

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {
                    "op": "project",
                    "input": "eco",
                    "to": "hydro_compartment_v001",
                    "aggregate": "count_by_value",
                }
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/data.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))

        expected = {}
        for geo_id in range(1, 33):
            zone = 100 if geo_id <= 16 else 200
            value = float(geo_id % 3)
            expected[(zone, value)] = expected.get((zone, value), 0) + 1
        # `execution.project`'s `_zonal_count_by_value` names its output columns
        # `grid_cell` / `value_category` (the distinct SOURCE value) / `value` (the count).
        actual = {
            (int(row["grid_cell"]), float(row["value_category"])): int(row["value"]) for row in df.iter_rows(named=True)
        }
        assert actual == expected

    def test_rejected_as_a_group_aggregate(self, client, base_path, monkeypatch):
        """`count_by_value` stays rejected as a `group.aggregate` -- it produces one row
        per (group, distinct value), which breaks `group`'s one-row-per-group contract --
        even though `project` permits it.
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        # `calendar_year > 0` keeps every row (the fixture has one year); the `filter` is
        # here only to name the opening input ahead of the `group` under test.
        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "filter", "input": "eco", "where": "calendar_year > 0"},
                {"op": "group", "by": [], "aggregate": "count_by_value"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/data.csv", json=pipeline)
        assert response.status_code == 400, response.text
        assert "cannot be a group.aggregate" in response.json()["detail"]


class TestPostProjectGridCellFilterEndToEnd:
    """A `filter` naming `grid_cell` after a many-to-one `project` filters the SUMMARIZED
    zones, not the source grid's own cells (which would silently drop everything, since the
    source grid's cell IDs and the summarized zone IDs live in different ID spaces here).
    """

    def _setup(self, base_path, crosswalk_path, monkeypatch):
        grid = GRIDS[RASTER_KWARGS["grid"]]
        zones = np.zeros((grid.height, grid.width), dtype="int32")
        zones[: grid.height // 2, :] = 1
        zones[grid.height // 2 :, :] = 2
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _y: raster_array())
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        return zones

    def test_grid_cell_eq_lowers_onto_summarize_grid_cell(self, client, base_path, crosswalk_path, monkeypatch):
        zones = self._setup(base_path, crosswalk_path, monkeypatch)
        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"},
                {"op": "filter", "where": "grid_cell = 1"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/data.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["grid_cell"].to_list() == [1]
        categories = raster_array()
        expected_mean = categories[zones == 1].astype("float64").mean()
        assert df["value"][0] == pytest.approx(expected_mean, rel=1e-4)

    def test_grid_cell_in_lowers_onto_summarize_grid_cell(self, client, base_path, crosswalk_path, monkeypatch):
        self._setup(base_path, crosswalk_path, monkeypatch)
        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"},
                {"op": "filter", "where": "grid_cell IN (1, 2)"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/data.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert sorted(df["grid_cell"].to_list()) == [1, 2]

    def test_two_post_project_grid_cell_filters_intersect(self, client, base_path, crosswalk_path, monkeypatch):
        """Two sequential post-project `grid_cell` filters AND together, as declared --
        `IN (1, 2)` then `IN (2, 3)` keeps zone 2 alone. A hand-computed correctness check:
        an implementation that absorbed both filters into a single last-write-wins zone
        bucket would narrow to {2, 3} and wrongly keep zone 3.
        """
        grid = GRIDS[RASTER_KWARGS["grid"]]
        third = grid.height // 3
        zones = np.empty((grid.height, grid.width), dtype="int32")
        zones[:third, :] = 1
        zones[third : 2 * third, :] = 2
        zones[2 * third :, :] = 3
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _y: raster_array())
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"},
                {"op": "filter", "where": "grid_cell IN (1, 2)"},
                {"op": "filter", "where": "grid_cell IN (2, 3)"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/data.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))

        categories = raster_array()
        expected_mean = categories[zones == 2].astype("float64").mean()
        assert actual == pytest.approx({2: expected_mean}, rel=1e-4)

    def test_grid_cell_filter_without_a_project_is_unaffected(self, client, base_path, monkeypatch):
        """Non-regression: with no preceding many-to-one (summarizing) `project`, a
        `grid_cell` filter still filters the source grid's own cell IDs directly.
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "filter", "input": "eco", "where": "grid_cell = 5"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/data.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["grid_cell"].to_list() == [5]


class TestProjectOnlyReaggregation:
    """One input + `project to:<coarser grid> aggregate:mean` + tabular output -- no
    `filter`/`group` needed at all.
    """

    def test_raster_to_vector_mean(self, client, base_path, crosswalk_path, monkeypatch):
        grid = GRIDS[RASTER_KWARGS["grid"]]
        zones = np.zeros((grid.height, grid.width), dtype="int32")
        zones[: grid.height // 2, :] = 1
        zones[grid.height // 2 :, :] = 2
        write_raster_crosswalk(crosswalk_path, RASTER_KWARGS["grid"], "hydro_compartment_v001", zones)
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _y: raster_array())
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [{"op": "project", "input": "vct", "to": "hydro_compartment_v001", "aggregate": "mean"}],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/data.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))

        categories = raster_array()
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))
        for zone_id in (1, 2):
            expected_mean = categories[zones == zone_id].astype("float64").mean()
            assert actual[zone_id] == pytest.approx(expected_mean, rel=1e-4)


# --------------------------------------------------------------------------------------
# calculate
# --------------------------------------------------------------------------------------


class TestWeightedCaseEndToEnd:
    """The per-dimension weighting idiom -- one `case` arm per year -- through all three
    request encodings (POST JSON, dotted GET, tile route), plus the flat-request
    equivalence. This is the shape `analyses/lnd_conf.py` uses, and the one a
    chain of `calculate`s used to spell before `calculate` became unconditional.

    A raw input's value column is always literally `"value"`, never its own dict key, so
    every expression below references `"value"`.
    """

    WEIGHTED_PIPELINE: ClassVar[dict] = {
        "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
        "steps": [
            {
                "op": "case",
                "input": "eco",
                "cases": [
                    {"when": "calendar_year = 2030", "then": "value * 0.5"},
                    {"when": "calendar_year = 2040", "then": "value * 0.25"},
                ],
            },
            {"op": "group", "by": [], "aggregate": "sum"},
        ],
        "output": {"format": "csv"},
    }

    def test_post_json(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        response = client.post("/v1/data.csv", json=self.WEIGHTED_PIPELINE)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        actual = dict(zip(df["grid_cell"].to_list(), df["value"].to_list()))
        for geo_id in (1, 32):
            expected = (2030 + geo_id / 100) * 0.5 + (2040 + geo_id / 100) * 0.25
            assert actual[geo_id] == pytest.approx(expected, rel=1e-4)

    def test_get_dotted_query_matches_post(self, client, base_path, monkeypatch):
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        query = pipeline_to_query(parse_pipeline(self.WEIGHTED_PIPELINE))
        post_df = pl.read_csv(io.BytesIO(client.post("/v1/data.csv", json=self.WEIGHTED_PIPELINE).content)).sort(
            "grid_cell"
        )
        get_response = client.get("/v1/data.csv", params=query)
        assert get_response.status_code == 200, get_response.text
        get_df = pl.read_csv(io.BytesIO(get_response.content)).sort("grid_cell")
        assert get_df.equals(post_df)

    def test_tile_route(self, client, base_path, monkeypatch):
        """A raster single-input chain with a weighting `case`, rendered on the analysis
        tile route: each arm selects the BAND its year names, and the two weighted bands
        collapse to the single band a tile requires."""
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {
                    "op": "case",
                    "input": "vct",
                    "cases": [
                        {"when": "calendar_year = 2030", "then": "value * 0.5"},
                        {"when": "calendar_year = 2040", "then": "value * 0.25"},
                    ],
                },
                {"op": "group", "by": [], "aggregate": "sum"},
            ],
            "output": {"format": "png"},
        }
        query = pipeline_to_query(parse_pipeline(pipeline))

        with rio.open(next(base_path.rglob("*.tif"))) as src:
            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        tile = mercantile.tile((west + east) / 2, (south + north) / 2, 8)

        response = client.get(f"/v1/data/{tile.z}/{tile.x}/{tile.y}.png", params=query)
        assert response.status_code == 200, response.text
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_weighted_sum_with_a_grid_cell_filter(self, client, base_path, monkeypatch):
        """The confidence-weighting idiom combined with a `grid_cell` filter. The filter is
        an ordinary `filter` step (there is no flat spelling for a value rewrite anymore,
        and an input names a dataset rather than filtering one), which doubles as this
        pipeline's opening step.
        """
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"ssf": {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"}},
            "steps": [
                {"op": "filter", "input": "ssf", "where": "grid_cell = 1"},
                {
                    "op": "case",
                    "cases": [
                        {"when": "calendar_year = 2030", "then": "value * 0.5"},
                        {"when": "calendar_year = 2040", "then": "value * 0.25"},
                    ],
                },
                {"op": "group", "by": [], "aggregate": "sum"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/data.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert df.filter(pl.col("grid_cell") == 1)["value"][0] == pytest.approx(
            2030.01 * 0.5 + 2040.01 * 0.25, rel=1e-5
        )

    def test_an_else_of_value_passes_the_unnamed_rows_through(self, client, base_path, monkeypatch):
        """The pass-through a `calculate`'s `where` used to leave implicit, now written:
        rows no arm names take `else: "value"` and keep what they arrived with. The step
        still drops nothing -- that is `filter`'s job -- so both years come back.
        """
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"ssf": {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"}},
            "steps": [
                {"op": "filter", "input": "ssf", "where": "grid_cell = 1"},
                {
                    "op": "case",
                    "cases": [{"when": "calendar_year = 2030", "then": "value * 0"}],
                    "else": "value",
                },
            ],
            "output": {"format": "csv", "sort": "calendar_year"},
        }
        response = client.post("/v1/data.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["value"].to_list() == pytest.approx([0.0, 2040.01], rel=1e-5)

    def test_without_an_else_the_unnamed_rows_go_null(self, client, base_path, monkeypatch):
        """The other half of the same choice, and the reason the pass-through has to be
        written: omitting `else` is not "leave them alone", it is "they have no answer".
        """
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"ssf": {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"}},
            "steps": [
                {"op": "filter", "input": "ssf", "where": "grid_cell = 1"},
                {"op": "case", "cases": [{"when": "calendar_year = 2030", "then": "value * 0"}]},
            ],
            "output": {"format": "csv", "sort": "calendar_year"},
        }
        response = client.post("/v1/data.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        assert df["value"].to_list() == [0.0, None]

    def test_weighted_band_sum_on_the_data_route(self, client, base_path, monkeypatch):
        """The same weighted band-sum idiom rendered through the plain (non-tile) `.tif`
        output, complementing `test_tile_route` above."""
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {
                    "op": "case",
                    "input": "vct",
                    "cases": [
                        {"when": "calendar_year = 2030", "then": "value * 0.5"},
                        {"when": "calendar_year = 2040", "then": "value * 0.25"},
                    ],
                },
                {"op": "group", "by": [], "aggregate": "sum"},
            ],
            "output": {"format": "tif"},
        }
        response = client.post("/v1/data.tif", json=pipeline)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            assert src.count == 1
            assert src.read(1)[0, 0] == 25

    def test_flat_calculate_params_are_retired_on_the_data_route(self, client, base_path, monkeypatch):
        """The flat `calculate.N.expression`/`calculate.N.where` query parameters are
        retired from the plain data route: the `calculate` pipeline step is now the only
        way to express this, and a request still spelling it the old way gets a loud 400
        naming the replacement rather than being silently ignored."""
        write_parquet(annual_frame(years=(2030, 2040)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        flat_response = client.get(
            "/v1/data.csv",
            params={
                "variable": "ssf",
                "time_unit": "annual",
                "grid": "ecoregion_v001",
                "calculate.1.expression": "value * 0.5",
                "calculate.1.where": "calendar_year = 2030",
                "calculate.2.expression": "value * 0.25",
                "calculate.2.where": "calendar_year = 2040",
                "group_aggregate": "sum",
            },
        )
        assert flat_response.status_code == 400
        detail = flat_response.json()["detail"]
        assert "calculate" in detail
        assert "group_aggregate" in detail

    def test_a_calculate_collapses_a_paired_frame(self, client, notebook_query_data):
        """A `calculate` over a joined frame names the OPERANDS, not `value` -- it is an
        arm-less `case`, and a `case` over a paired frame is the reduce that collapses the
        pairing. Inserted ahead of the `case` step, it makes that step's own arms
        unsatisfiable (`marsh`/`sal` are gone), so what this asserts is the collapse and
        the 200, not the numbers.
        """
        pipeline = copy.deepcopy(NOTEBOOK_QUERY)
        pipeline["steps"][2] = {"op": "calculate", "expression": "marsh - sal * 0.5"}
        response = client.post("/v1/data.tif", json=pipeline)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            assert src.count == 1


# --------------------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------------------


class TestRejections:
    def test_post_url_format_overwrites_the_bodys_own(self, client, notebook_query_data):
        """The URL's extension WINS over the body's `output.format`; it is not cross-checked
        against it (which is what this route used to do, 400-ing with "does not match").

        `NOTEBOOK_QUERY` declares `tif` and ends band-wide, so asking for it as `.csv` is
        still a 400 -- but a 400 from the SINK-SHAPE rule, which is the observable
        difference: the message names `output.format='csv'`, the URL's value, proving the
        overwrite happened before `validate_shapes` ran rather than a mismatch being caught
        before it.
        """
        response = client.post("/v1/data.csv", json=NOTEBOOK_QUERY)
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "output.format='csv'" in detail
        assert "needs a tabular sink" in detail

    def test_get_dotted_format_overwrites_the_querys_own(self, client, notebook_query_data):
        """The same rule on the dotted GET spelling -- `output.format=tif` in the query
        string loses to the `.csv` in the URL.
        """
        query = pipeline_to_query(parse_pipeline(NOTEBOOK_QUERY))
        assert query["output.format"] == "tif"
        response = client.get("/v1/data.csv", params=query)
        assert response.status_code == 400
        assert "output.format='csv'" in response.json()["detail"]

    def test_url_format_overwrite_is_not_only_a_rejection(self, client, base_path, monkeypatch):
        """The positive half of the rule the two tests above only see reject: a pipeline
        that declares `csv` and is fetched as `.ndjson` is SERVED, as ndjson.

        This is the case the old cross-check got wrong in a way that mattered. The portal
        holds one pipeline per map and offers it both as tiles and as a download, so
        "re-request this exact pipeline in another format" is an ordinary client move, and
        under the cross-check it forced the client to rewrite `output.format` to match
        every extension it fetched -- making `pipeline_to_query()` no longer one stable
        identity for the thing being looked at.
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [],
            "output": {"format": "csv"},
        }
        response = client.get("/v1/data.ndjson", params=pipeline_to_query(parse_pipeline(pipeline)))
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("application/x-ndjson")
        assert json.loads(response.content.splitlines()[0])["calendar_year"] == 2030

    def test_three_inputs_is_400(self, client, base_path, monkeypatch):
        """A chained 3-input join (two spatial joins folding a, b, c together) is
        structurally well-formed, so it gets past `validate_pipeline`; what rejects it is
        the shape walk, because the second join's `ab` operand carries two value columns
        and a join has only one name per operand to give them. Left unchecked this reaches
        execution and dies on `group`'s default `of="value"`, which the first join already
        renamed away -- a 400 either way (boundary.py maps `ColumnNotFoundError` too), but
        one whose message says nothing about what the caller actually asked for.
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {
                "a": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")},
                "b": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")},
                "c": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")},
            },
            "steps": [
                {"op": "join", "type": "spatial", "inputs": ["a", "b"], "as": "ab"},
                {"op": "join", "type": "spatial", "inputs": ["ab", "c"], "as": "abc"},
                {"op": "group", "by": [], "aggregate": "mean"},
            ],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/data.csv", json=pipeline)
        assert response.status_code == 400
        assert "not yet supported" in response.json()["detail"]

    def test_tile_route_rejects_vector_output(self, client, base_path, monkeypatch):
        """`eco` is a vector input left un-projected (a bare zero-step, one-input
        pipeline). Its final frame is long/tabular, never a raster surface, so the `png`
        tile route must reject it.
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [],
            "output": {"format": "png"},
        }
        query = pipeline_to_query(parse_pipeline(pipeline))
        response = client.get("/v1/data/3/1/2.png", params=query)
        assert response.status_code == 400
        assert "raster surface" in response.json()["detail"]

    def test_raw_raster_filter_needs_no_summarizing_project(self, client, base_path, monkeypatch):
        """A `filter` over a raw (non-reaggregated) raster grid works directly, without a
        prior summarizing `project` -- `execution/frame.py` supports band-wide filters, so
        there is no per-pixel-frame constraint to trip over.

        Data: years (2030, 2040) filled with `year - 2000`, i.e. bands of 30 and 40.
        `filter(value > 35)` keeps only the 2040 band (value 40); the 2030 band (30)
        is dropped entirely, so `group(min)` collapses a single surviving band and the
        output is uniformly 40 -- if the filter or the group silently changed behavior
        (e.g. filtering nothing, or including the 30 band), this would compute a
        different constant and the assertion below would catch it.
        """
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "filter", "input": "vct", "where": "value > 35"},
                {"op": "group", "by": [], "aggregate": "min"},
            ],
            "output": {"format": "tif"},
        }
        response = client.post("/v1/data.tif", json=pipeline)
        assert response.status_code == 200, response.text
        with rio.MemoryFile(response.content) as memfile, memfile.open() as dataset:
            band = dataset.read(1)
        assert (band == 40).all()
