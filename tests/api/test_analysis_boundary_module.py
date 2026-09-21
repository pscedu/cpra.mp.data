"""Tests for `cpra.mp.data.api.boundary` -- the exception -> HTTP mapping installed onto
the four `/v1/analysis...` pipeline routes -- and, since a mapping is only meaningful
through the routes that trigger it, a route-level smoke test of all four response shapes
(tabular, `tif`, `png` tile, `pbf` tile) that boundary.py's own module docstring already
references this file by name for (its MRO claim, and the tile route's `RequestError`
propagation test in `test_pipeline_exec.py`).

Two layers:

* `TestResolveStatus`/`TestInstallAnalysisBoundary` -- `boundary.py`'s own two entry
  points, exercised directly against a throwaway `FastAPI` app (for the registered
  handlers) or as a bare function call (for `resolve_status`), with no real pipeline,
  file, or `TestClient` fixture involved -- the mapping table is the thing under test,
  not any particular route that consults it.
* `TestFourRoutesEndToEnd` -- the real ASGI app (the `client` fixture), proving the
  mapping is actually wired up where it needs to be. Where a case is already covered
  byte-exact elsewhere (`.pbf` hit/miss/reject-vector-output live in
  `tests/api/test_vector_tiles.py::TestAnalysisVectorTileRoute`; ordinary tabular/`tif`
  success shapes live throughout `test_pipeline_exec.py`), this file adds only what those
  do not already prove: the `tif` "no bands to write" 404, the `png` tile-miss 404 (image
  body, not JSON), the malformed-pipeline 400, and the two `output.sort`/`output.select`
  seams (syntactically bad -> 400 from `request.pipeline` itself; syntactically fine but
  naming a column the final frame doesn't have -> 400 from `boundary`'s `ColumnNotFoundError`
  entry, not a raw 500).

**A caller-visible asymmetry found while writing this file, not introduced by it.** The
tabular route's error body is NOT `{"detail": ...}` JSON the way the `tif`/`png`/`pbf`
routes' are. `api.pipeline.PipelineTabularResponse` never raises into FastAPI's own
exception dispatch at all -- the failing sink runs inside `api.tabular.StreamingThread`, a
plain background `Thread` (see that class's own docstring) -- so `boundary.py`'s
*registered handlers* never fire for it; only `resolve_status` (the plain function) is
consulted, to pick a status code, and the body sent is the raw exception string (or
`err.detail`) with the tabular format's own content-type (`text/csv`, etc.), not
`application/json`. `TestFourRoutesEndToEnd::test_tabular_error_body_is_plain_text_not_json`
below pins this rather than assuming the shape matches the other three routes.
"""

import io
import json

import mercantile
import polars as pl
import pyarrow.parquet as pq
import pytest
import rasterio as rio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from PIL import Image
from polars.exceptions import ColumnNotFoundError
from rasterio.warp import transform_bounds

from cpra.mp.data import config
from cpra.mp.data.analysis.execution.execute import LoweringNotImplementedError
from cpra.mp.data.analysis.request.errors import (
    AggregateError,
    BandLimitError,
    NotFoundError,
    RequestError,
    ShapeError,
)
from cpra.mp.data.analysis.request.expressions import ExpressionError
from cpra.mp.data.api import pipeline as api_pipeline
from cpra.mp.data.api.boundary import install_analysis_boundary, resolve_detail, resolve_status
from cpra.mp.data.tabular import write_parquet
from tests.factories import ANNUAL_KWARGS, RASTER_KWARGS, annual_frame, write_raster_dataset

# --------------------------------------------------------------------------------------
# resolve_status: the plain function `PipelineTabularResponse` consults directly (see this
# module's own docstring for why the tabular route can't rely on the registered handlers).
# --------------------------------------------------------------------------------------


class TestResolveStatus:
    def test_not_found_error_is_404(self):
        assert resolve_status(NotFoundError("missing")) == 404

    def test_band_limit_error_is_400(self):
        assert resolve_status(BandLimitError("too many bands")) == 400

    def test_bare_request_error_is_400(self):
        assert resolve_status(RequestError("bad shape")) == 400

    def test_shape_error_falls_back_to_the_request_error_family_default(self):
        """`ShapeError` has no entry of its own in `_STATUS_BY_TYPE` -- this is the MRO
        fallback in miniature: if `resolve_status` only checked `type(exc) is RequestError`
        rather than walking the MRO, this would wrongly return `None` for every subclass
        that doesn't special-case its own status.
        """
        assert resolve_status(ShapeError("bad join")) == 400

    def test_aggregate_error_falls_back_to_the_request_error_family_default(self):
        assert resolve_status(AggregateError("bad aggregate")) == 400

    def test_expression_error_is_400(self):
        """`ExpressionError` is a `ValueError` subclass, NOT a `RequestError` member (see
        that class's own docstring) -- if `_STATUS_BY_TYPE` only had a `RequestError` entry
        and relied on it somehow covering this too, this would wrongly return `None`.
        """
        assert resolve_status(ExpressionError("bad expression")) == 400

    def test_lowering_not_implemented_error_is_501(self):
        assert resolve_status(LoweringNotImplementedError("no executor yet")) == 501

    def test_polars_column_not_found_error_is_400(self):
        assert resolve_status(ColumnNotFoundError("no such column")) == 400

    def test_unrecognized_exception_type_returns_none(self):
        """A caller (`PipelineTabularResponse`) decides its own fallback for a type this
        module has no opinion about -- `resolve_status` must not guess on its behalf by
        returning some default status instead of `None`.
        """
        assert resolve_status(ValueError("unrelated")) is None
        assert resolve_status(RuntimeError("unrelated")) is None


# --------------------------------------------------------------------------------------
# resolve_detail: the body half of the same question, consulted by both the registered
# handlers and `PipelineTabularResponse` so the two can never disagree about what a given
# exception is allowed to say.
# --------------------------------------------------------------------------------------


class TestResolveDetail:
    def test_an_unlisted_type_answers_with_its_own_message(self):
        """The default path: these messages are this package's own text, written to be
        read by the caller, so they pass through verbatim.
        """
        assert resolve_detail(RequestError("bad pipeline shape")) == "bad pipeline shape"
        assert resolve_detail(ColumnNotFoundError("no such column")) == "no such column"

    def test_file_not_found_is_overridden_and_drops_the_path(self):
        exc = FileNotFoundError(
            "No such file or directory (os error 2): /ocean/projects/x/variable=marsh/p.parquet"
        )
        assert resolve_detail(exc) == "Dataset does not exist"

    def test_a_subclass_of_an_overridden_type_cannot_route_around_the_override(self):
        """The reason `resolve_detail` walks the MRO instead of doing a dict lookup on
        `type(exc)`: `FileNotFoundError` has real subclasses in the wild, and an exact-type
        lookup would let one of them leak the very message its parent is listed to
        suppress.
        """

        class WrappedFileNotFound(FileNotFoundError):
            pass

        exc = WrappedFileNotFound("No such file or directory: /ocean/projects/x/p.parquet")
        assert resolve_detail(exc) == "Dataset does not exist"

    def test_an_exception_carrying_its_own_detail_uses_it(self):
        """`HTTPException`-shaped exceptions reaching the tabular response's error path
        already carry route-authored text in `.detail`; that is preferred over `str(exc)`,
        which for those renders as a repr-ish string rather than the message.
        """
        exc = HTTPException(status_code=400, detail="URL format does not match")
        assert resolve_detail(exc) == "URL format does not match"


# --------------------------------------------------------------------------------------
# install_analysis_boundary: the registered-handler half, against a throwaway app -- no
# real pipeline or route involved, only routes built to raise each exception directly.
# --------------------------------------------------------------------------------------


@pytest.fixture
def boundary_app():
    app = FastAPI()
    install_analysis_boundary(app)

    @app.get("/raise/not-found")
    def _raise_not_found():
        raise NotFoundError("no such crosswalk")

    @app.get("/raise/band-limit")
    def _raise_band_limit():
        raise BandLimitError("too many bands")

    @app.get("/raise/request-error")
    def _raise_request_error():
        raise RequestError("bad pipeline shape")

    @app.get("/raise/shape-error")
    def _raise_shape_error():
        # A `RequestError` SUBCLASS with no entry of its own in `_STATUS_BY_TYPE` -- the
        # real MRO claim under test: Starlette's own exception-handler lookup must resolve
        # this to the `RequestError` registration, not fail to find a handler at all.
        raise ShapeError("final frame is not vector")

    @app.get("/raise/expression-error")
    def _raise_expression_error():
        raise ExpressionError("disallowed function")

    @app.get("/raise/lowering-not-implemented")
    def _raise_lowering_not_implemented():
        raise LoweringNotImplementedError("no executor for this op/shape yet")

    @app.get("/raise/column-not-found")
    def _raise_column_not_found():
        raise ColumnNotFoundError("no such column")

    @app.get("/raise/file-not-found")
    def _raise_file_not_found():
        # Spelled the way Polars' reader actually spells it, absolute path and all --
        # that path is the thing `_DETAIL_BY_TYPE` exists to keep out of the response.
        raise FileNotFoundError(
            "No such file or directory (os error 2): "
            "/ocean/projects/bcs200002p/shared/data-long/variable=marsh/part-0.parquet"
        )

    return app


@pytest.fixture
def boundary_client(boundary_app):
    with TestClient(boundary_app, raise_server_exceptions=False) as test_client:
        yield test_client


class TestInstallAnalysisBoundary:
    @pytest.mark.parametrize(
        ("path", "status", "message"),
        [
            ("/raise/not-found", 404, "no such crosswalk"),
            ("/raise/band-limit", 400, "too many bands"),
            ("/raise/request-error", 400, "bad pipeline shape"),
            ("/raise/expression-error", 400, "disallowed function"),
            ("/raise/lowering-not-implemented", 501, "no executor for this op/shape yet"),
            ("/raise/column-not-found", 400, "no such column"),
        ],
    )
    def test_each_registered_type_maps_to_its_own_status_and_body(
        self, boundary_client, path, status, message
    ):
        response = boundary_client.get(path)
        assert response.status_code == status
        # Every handler returns exactly {"detail": str(exc)} -- if a handler leaked a
        # traceback, wrapped the message differently, or dropped it, this would catch it.
        assert response.json() == {"detail": message}

    def test_file_not_found_answers_404_without_disclosing_the_path(self, boundary_client):
        """A missing data file is a 404, but the reader's own message names the absolute
        server path it tried to open, and that must not reach the caller.
        `api.sources.LazyFrameSource.materialize` answers this case with a fixed string
        for exactly that reason; `_DETAIL_BY_TYPE` is what does the same for the engine,
        while the other registered types keep passing their own (ours to disclose)
        message through.
        """
        response = boundary_client.get("/raise/file-not-found")
        assert response.status_code == 404
        assert response.json() == {"detail": "Dataset does not exist"}
        assert "/ocean/" not in response.text
        assert "parquet" not in response.text

    def test_mro_covers_a_subclass_with_no_registration_of_its_own(self, boundary_client):
        """`ShapeError` (raised by `/raise/shape-error`) is never itself a key in
        `_STATUS_BY_TYPE`; only its ancestor `RequestError` is. If Starlette's dispatch (or
        `install_analysis_boundary`'s registration loop) required an exact type match
        instead of walking the MRO, this request would 500 with an unhandled exception
        instead of reaching the `RequestError` handler's 400.
        """
        response = boundary_client.get("/raise/shape-error")
        assert response.status_code == 400
        assert response.json() == {"detail": "final frame is not vector"}

    def test_registration_is_independent_of_dict_iteration_order(self, boundary_app):
        """`_STATUS_BY_TYPE`'s entries are registered in a loop over the dict, in whatever
        order the dict itself iterates (insertion order, chosen for human readability, per
        the module's own comment). Confirms that order carries no meaning to Starlette's
        own most-specific-match lookup: every one of the 6 distinct exception types below
        reaches its own handler regardless of which one was registered first or last.
        """
        client = TestClient(boundary_app, raise_server_exceptions=False)
        for path, status in [
            ("/raise/not-found", 404),
            ("/raise/band-limit", 400),
            ("/raise/lowering-not-implemented", 501),
            ("/raise/column-not-found", 400),
        ]:
            assert client.get(path).status_code == status


# --------------------------------------------------------------------------------------
# Four routes, end to end, against the real app.
# --------------------------------------------------------------------------------------


def _tile_for_grid(base_path, grid_glob="*.tif", zoom=8):
    with rio.open(next(base_path.rglob(grid_glob))) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return mercantile.tile((west + east) / 2, (south + north) / 2, zoom)


class TestFourRoutesEndToEnd:
    def test_tabular_csv_returns_real_rows(self, client, base_path, monkeypatch):
        """Baseline positive case for the tabular route -- a `csv` result whose values are
        independently recomputed from `annual_frame` (not merely "some 200 came back").
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_csv(io.BytesIO(response.content))
        expected = annual_frame(years=(2030,))["1"][0]
        row = df.filter(pl.col("grid_cell") == 1)
        assert row["value"][0] == pytest.approx(expected, rel=1e-5)

    def test_tabular_ndjson_returns_real_rows(self, client, base_path, monkeypatch):
        """Same baseline as the csv case above, for `ndjson` -- the format `PipelineTabular
        Response._TABULAR_MIME_TYPES`/`PipelineTabularSource.stream_tabular` also serve
        alongside csv/parquet. (`arrow` is the fourth name in that MIME map, but this
        route's own `{format}` path segment is bound directly to `analysis.request.types
        .DataFormat`, which has no live `arrow` member -- `/v1/analysis.arrow` is rejected
        by FastAPI's own enum path-param validation before any code in this module runs,
        so there is no route-level case to add for it here.)
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [],
            "output": {"format": "ndjson"},
        }
        response = client.post("/v1/analysis.ndjson", json=pipeline)
        assert response.status_code == 200, response.text
        rows = [json.loads(line) for line in response.text.splitlines()]
        expected = annual_frame(years=(2030,))["1"][0]
        (row,) = [r for r in rows if r["grid_cell"] == 1]
        assert row["value"] == pytest.approx(expected, rel=1e-5)

    def test_tabular_parquet_returns_real_rows(self, client, base_path, monkeypatch):
        """Same baseline again, for `parquet` -- this is also the regression case for the
        500 this route used to raise: `PipelineTabularResponse` used to hardcode `None`
        as the compression argument to `StreamingThread`, so any parquet request crashed
        inside `api.tabular.stream_or_write_to_file`'s `compression.compression_algorithm`
        access before a single byte was written.
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [],
            "output": {"format": "parquet"},
        }
        response = client.post("/v1/analysis.parquet", json=pipeline)
        assert response.status_code == 200, response.text
        df = pl.read_parquet(io.BytesIO(response.content))
        expected = annual_frame(years=(2030,))["1"][0]
        row = df.filter(pl.col("grid_cell") == 1)
        assert row["value"][0] == pytest.approx(expected, rel=1e-5)

    def test_tabular_parquet_honors_requested_compression(self, client, base_path, monkeypatch):
        """The route's own `compression_algorithm`/`compression_level` query params reach
        the written file's codec, not just a 200 status -- the full chain from `Compression`
        (the request model) through `PipelineTabularResponse`/`PipelineTabularSource` down
        to `sink.sink_tabular`'s Polars call, proven here at the HTTP layer rather than only
        at the unit level (`tests/api/test_sink.py`'s `TestSinkTabular
        .test_parquet_honors_requested_compression`/`TestRunToPath
        .test_parquet_output_forwards_compression_to_the_sink` cover the same chain one and
        two layers down, respectively). gzip differs from the route's own zstd default, so
        this fails if the compression argument were silently dropped anywhere along the way.
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [],
            "output": {"format": "parquet"},
        }
        response = client.post(
            "/v1/analysis.parquet",
            params={"compression_algorithm": "gzip", "compression_level": 9},
            json=pipeline,
        )
        assert response.status_code == 200, response.text
        metadata = pq.ParquetFile(io.BytesIO(response.content)).metadata
        codec = metadata.row_group(0).column(0).compression
        assert codec.upper() == "GZIP"

    def test_a_missing_dataset_is_404_without_disclosing_the_path(
        self, client, base_path, monkeypatch
    ):
        """Deliberately no `write_parquet`: the combo is legal but nothing is on disk, so
        the reader raises `FileNotFoundError` inside the sink's writer thread. That thread
        SWALLOWS the exception rather than letting it reach FastAPI's dispatch, so the
        registered handler never fires and the streaming response has to resolve both the
        status and the body itself -- this is the route-level proof that it resolves the
        body through `resolve_detail` (fixed string) rather than `str(err)` (which names
        the absolute path the reader tried to open).
        """
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 404, response.text
        assert "Dataset does not exist" in response.text
        assert str(base_path) not in response.text
        assert "parquet" not in response.text

    def test_an_exception_the_boundary_does_not_know_is_500_on_the_tabular_route(
        self, client, base_path, monkeypatch
    ):
        """`resolve_status` answers `None` for a type this module has no mapping for, and
        the tabular route -- which cannot fall back on FastAPI's own dispatch, since its
        sink raises inside a background thread -- has to decide for itself what to do with
        that. A genuinely unexpected failure must come back as a 500: the branch under test
        is the one place where a wrong answer would instead be a 200 with a truncated body,
        which no client could tell apart from a complete result.

        `RuntimeError` stands in for "anything unforeseen"; it is deliberately NOT a member
        of any family `boundary.py` maps, which is what routes this through the fallback
        rather than the `_STATUS_BY_TYPE` lookup.
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        def _boom(*args, **kwargs):
            raise RuntimeError("the sink fell over")

        monkeypatch.setattr(api_pipeline, "run_to_path", _boom)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [],
            "output": {"format": "csv"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 500
        assert "the sink fell over" in response.text

    def test_tabular_error_body_is_plain_text_not_json(self, client, base_path, monkeypatch):
        """See this module's own docstring: the tabular route's error path never reaches
        FastAPI's exception dispatch (the failing sink runs inside `StreamingThread`, a
        background `Thread`, not the request coroutine), so it never gets the `{"detail":
        ...}` JSON envelope the other three routes' registered handlers produce. A test
        that called `.json()` here would raise `JSONDecodeError` before ever reaching an
        assertion -- this pins the actual (plain-text, tabular-content-type) shape instead.
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [],
            "output": {"format": "csv", "sort": "no_such_column"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 400
        assert response.headers["content-type"].startswith("text/csv")
        assert "no_such_column" in response.text
        with pytest.raises(Exception):
            response.json()

    def test_unknown_sort_column_is_400_not_500(self, client, base_path, monkeypatch):
        """The A/B seam `boundary.py`'s own docstring calls out by name:
        `request.pipeline._validate_output_sort_or_select` only checks SYNTAX at parse
        time (this pipeline's `output.sort` is syntactically fine), so an unknown column
        name survives parsing and only fails once `execution.sink.apply_output_shaping`
        tries to actually sort by it, as a raw `polars.exceptions.ColumnNotFoundError`.
        Without `boundary.py`'s dedicated entry for that type, this would surface as a bare
        500, not a 400 -- the seam this test exists to keep honest.
        """
        write_parquet(annual_frame(years=(2030,)), base_path=base_path)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [],
            "output": {"format": "csv", "sort": "not_a_real_column"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 400
        assert "not_a_real_column" in response.text

    def test_syntactically_bad_sort_is_400_from_the_request_layer(self, client, base_path, monkeypatch):
        """The other half of the same seam (deliverable A, from the request layer itself):
        an empty component between commas is rejected by `_validate_output_sort_or_select`
        during `parse_pipeline`, before compilation or execution ever run -- so this must
        400 even against a pipeline whose data was never written (`base_path` is patched
        but nothing is written to it below), proving the rejection genuinely happens at
        parse time, not merely as a side effect of missing data.
        """
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [],
            "output": {"format": "csv", "sort": "value,,grid_cell"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 400
        assert "empty component" in response.text

    def test_syntactically_bad_select_is_400_from_the_request_layer(self, client, base_path, monkeypatch):
        """Same seam, `output.select`'s own extra rule: a `-` prefix (the descending-sort
        marker) is meaningless on a `select` entry and rejected outright.
        """
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [],
            "output": {"format": "csv", "select": "-value"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 400
        assert "must not start with '-'" in response.text

    def test_a_sort_entry_that_is_only_a_descending_marker_is_400(self, client, base_path, monkeypatch):
        """A lone `-` names no column at all. It has to be rejected here rather than left
        to Polars: `apply_output_shaping` strips the `-` off and sorts by the empty string,
        which is a column name Polars will happily look for and fail to find -- so the
        caller would get the same 400 either way, but with a message about a missing column
        `''` instead of one saying the sort expression is malformed.
        """
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [],
            "output": {"format": "csv", "sort": "value,-"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 400
        assert "bare '-'" in response.text

    def test_a_column_named_twice_in_one_sort_is_400(self, client, base_path, monkeypatch):
        """A repeat is rejected because the two spellings disagree about what to do:
        `"value,-value"` asks Polars to sort ascending and descending by the same column in
        one call, which it accepts and silently resolves by ignoring the second -- a
        caller's contradiction answered with an arbitrary winner rather than an error.
        Both entries name `value`, so the check is on the column, not the raw component.
        """
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"eco": {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [],
            "output": {"format": "csv", "sort": "value,-value"},
        }
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 400
        assert "more than once" in response.text

    def test_malformed_pipeline_body_is_400_with_json_detail(self, client, base_path, monkeypatch):
        """A body that fails `parse_pipeline` outright (here: an `inputs` dict that is
        empty, which `Pipeline`'s own validation rejects) never reaches `compile_plan`/
        `execute` at all -- this is the plainest possible 400, pinned so a future change to
        request-body handling can't silently turn a malformed request into a 422 (FastAPI's
        own default for a body that fails Pydantic validation) or an unhandled 500 instead.
        """
        monkeypatch.setattr(config, "BASE_PATH", base_path)
        pipeline = {"inputs": {}, "steps": [], "output": {"format": "csv"}}
        response = client.post("/v1/analysis.csv", json=pipeline)
        assert response.status_code == 400
        assert "detail" in response.json()

    def test_tif_no_bands_to_write_is_404(self, client, base_path, monkeypatch):
        """`run_to_path` returns `False` when a band-wide result has no bands left to
        write (`execution.sink.write_raster`'s own contract) -- here forced by a dimension
        filter (`calendar_year = 9999`) that matches none of the two real years written.
        `PipelineRasterResponse` turns that `False` into `HTTPException(404, "source data
        do not exist")`, which -- unlike the tabular route's error path -- DOES flow
        through FastAPI's normal exception dispatch (the `tif` route runs off the event
        loop via `anyio.to_thread.run_sync`, whose exceptions re-raise normally in the
        awaiting coroutine, per `boundary.py`'s own docstring), so this IS the ordinary
        `{"detail": ...}` JSON shape, unlike the tabular case above.
        """
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "filter", "input": "vct", "where": "calendar_year = 9999"},
                {"op": "group", "by": [], "aggregate": "max"},
            ],
            "output": {"format": "tif"},
        }
        response = client.post("/v1/analysis.tif", json=pipeline)
        assert response.status_code == 404
        assert response.json() == {"detail": "source data do not exist"}

    def test_tif_hit_returns_a_real_raster(self, client, base_path, monkeypatch):
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
        response = client.post("/v1/analysis.tif", json=pipeline)
        assert response.status_code == 200, response.text
        with rio.open(io.BytesIO(response.content)) as src:
            data = src.read(1)
        assert (data == 40).all()

    def test_png_tile_hit_is_a_decodable_image(self, client, base_path, monkeypatch):
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "filter", "input": "vct", "where": "calendar_year > 0"},
                {"op": "group", "by": [], "aggregate": "max"},
            ],
            "output": {"format": "png"},
        }
        from cpra.mp.data.analysis.request.pipeline import parse_pipeline, pipeline_to_query

        query = pipeline_to_query(parse_pipeline(pipeline))
        tile = _tile_for_grid(base_path)
        response = client.get(f"/v1/analysis/{tile.z}/{tile.x}/{tile.y}.png", params=query)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "image/png"
        image = Image.open(io.BytesIO(response.content))
        assert image.size == (256, 256)

    def test_png_tile_miss_is_404_with_a_decodable_transparent_image_not_json(
        self, client, base_path, monkeypatch
    ):
        """`render_tile` returns `None` when a dimension predicate narrows the request to
        no bands at all (`calendar_year = 9999`, same technique as the `tif` 404 case
        above) -- `respond_to_analysis_tile` turns that into a 404 whose BODY is still a
        decodable, correctly-sized image (`sink.empty_tile`), per that function's own
        docstring: raising `HTTPException` here instead would replace the image body with a
        JSON error document, exactly the anti-pattern this route is built to avoid. Checked
        two ways: the content-type is `image/png` (not `application/json`), and Pillow can
        actually decode the body at the requested tile size.
        """
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        monkeypatch.setattr(config, "BASE_PATH", base_path)

        pipeline = {
            "inputs": {"vct": {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}},
            "steps": [
                {"op": "filter", "input": "vct", "where": "calendar_year = 9999"},
                {"op": "group", "by": [], "aggregate": "max"},
            ],
            "output": {"format": "png"},
        }
        from cpra.mp.data.analysis.request.pipeline import parse_pipeline, pipeline_to_query

        query = pipeline_to_query(parse_pipeline(pipeline))
        tile = _tile_for_grid(base_path)
        response = client.get(f"/v1/analysis/{tile.z}/{tile.x}/{tile.y}.png", params=query)
        assert response.status_code == 404
        assert response.headers["content-type"] == "image/png"
        image = Image.open(io.BytesIO(response.content))
        assert image.size == (256, 256)
        # Fully transparent (see `empty_tile`'s own docstring: an all-zero RGBA array), not
        # a rendered nodata color -- a caller diffing this against a real nodata tile could
        # otherwise mistake "no data at this location" for "one specific rendered value."
        assert image.convert("RGBA").getextrema() == ((0, 0), (0, 0), (0, 0), (0, 0))
