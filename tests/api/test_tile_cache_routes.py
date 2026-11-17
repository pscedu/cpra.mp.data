"""The tile routes' use of `cpra.mp.data.tile_cache`: hits skip the render, a write to an
input dataset or a changed request misses, ETags revalidate, and concurrent requests for
one tile share its render."""

import copy
import threading
import time

import anyio
import numpy as np
import pytest

from cpra.mp.data import config
from cpra.mp.data.analysis.request.pipeline import parse_pipeline, pipeline_to_query
from cpra.mp.data.api import pipeline as api_pipeline
from cpra.mp.data.grids import GRIDS
from tests.api.test_healthz import _receive_until
from tests.api.test_pipeline_routes import (
    NOTEBOOK_QUERY,
    _tile_for_grid,
    _write_ffibs_raster,
    _write_sal_annual,
)
from tests.factories import write_raster_crosswalk


@pytest.fixture
def notebook_query_data(base_path, crosswalk_path, monkeypatch):
    """`test_pipeline_routes.notebook_query_data`'s data: a two-input pipeline whose tile
    reads a raster and a vector dataset through a crosswalk."""
    _write_ffibs_raster(base_path, {2030: 0.3, 2040: 0.7})
    _write_sal_annual(base_path, {2030: 4.0, 2040: 6.0}, geo_id=1)
    monkeypatch.setattr(config, "BASE_PATH", base_path)
    grid = GRIDS["veg_grid_cell_v001"]
    zones = np.ones((grid.height, grid.width), dtype="int32")
    write_raster_crosswalk(crosswalk_path, "veg_grid_cell_v001", "hydro_compartment_v001", zones)
    return base_path


@pytest.fixture
def tile_cache_dir(tmp_path, monkeypatch):
    path = tmp_path / "tile-cache"
    monkeypatch.setattr(config, "TILE_CACHE_PATH", path)
    monkeypatch.setattr(config, "TILE_CACHE_MAX_GB", 1.0)
    return path


@pytest.fixture
def render_count(monkeypatch):
    calls = []
    real_execute = api_pipeline.execute

    def counting_execute(*args, **kwargs):
        calls.append(1)
        return real_execute(*args, **kwargs)

    monkeypatch.setattr(api_pipeline, "execute", counting_execute)
    return calls


def _tile_url(base_path, format="png"):
    query = copy.deepcopy(NOTEBOOK_QUERY)
    query["output"] = {"format": format}
    tile = _tile_for_grid(base_path)
    return f"/v1/data/{tile.z}/{tile.x}/{tile.y}.{format}", pipeline_to_query(parse_pipeline(query))


def test_second_request_is_served_from_the_cache(client, notebook_query_data, tile_cache_dir, render_count):
    url, params = _tile_url(notebook_query_data)
    first = client.get(url, params=params)
    second = client.get(url, params=params)

    assert first.status_code == 200, first.text
    assert second.status_code == 200
    assert second.content == first.content
    assert second.headers["content-type"] == "image/png"
    assert second.headers["etag"] == first.headers["etag"]
    assert len(render_count) == 1
    assert any(tile_cache_dir.rglob("*.png"))


def test_a_hit_does_not_compile_a_plan(client, notebook_query_data, tile_cache_dir, monkeypatch):
    """The key comes from the request and the catalog, so a hit skips `compile_plan`,
    which is most of what keying on the data files themselves would cost."""
    url, params = _tile_url(notebook_query_data)
    client.get(url, params=params)
    compiles = []
    real_compile = api_pipeline.compile_plan
    monkeypatch.setattr(api_pipeline, "compile_plan", lambda *a, **k: compiles.append(1) or real_compile(*a, **k))

    assert client.get(url, params=params).status_code == 200
    assert compiles == []


def test_a_write_to_an_input_dataset_misses(client, notebook_query_data, tile_cache_dir, render_count):
    url, params = _tile_url(notebook_query_data)
    first = client.get(url, params=params)
    # Writing another year's COG appends to the dataset's catalog partition.
    _write_ffibs_raster(notebook_query_data, {2050: 0.9})
    second = client.get(url, params=params)

    assert second.status_code == 200
    assert second.headers["etag"] != first.headers["etag"]
    assert len(render_count) == 2


def test_a_different_request_misses(client, notebook_query_data, tile_cache_dir, render_count):
    url, params = _tile_url(notebook_query_data)
    client.get(url, params=params)
    query = copy.deepcopy(NOTEBOOK_QUERY)
    query["steps"][-1]["aggregate"] = "max"
    query["output"] = {"format": "png"}
    client.get(url, params=pipeline_to_query(parse_pipeline(query)))
    assert len(render_count) == 2


def test_matching_etag_is_a_304_without_a_render(client, notebook_query_data, tile_cache_dir, render_count):
    url, params = _tile_url(notebook_query_data)
    etag = client.get(url, params=params).headers["etag"]

    response = client.get(url, params=params, headers={"If-None-Match": etag})

    assert response.status_code == 304
    assert response.headers["etag"] == etag
    assert response.content == b""
    assert len(render_count) == 1


def test_cache_off_still_sends_an_etag_and_renders_every_time(client, notebook_query_data, render_count, tmp_path):
    assert config.TILE_CACHE_MAX_GB <= 0
    url, params = _tile_url(notebook_query_data)
    first = client.get(url, params=params)
    second = client.get(url, params=params)

    assert first.status_code == second.status_code == 200
    assert first.headers["etag"] == second.headers["etag"]
    assert first.headers["cache-control"] == "no-cache"
    assert len(render_count) == 2
    assert client.get(url, params=params, headers={"If-None-Match": first.headers["etag"]}).status_code == 304


def test_empty_tile_is_cached(client, notebook_query_data, tile_cache_dir, render_count):
    """A tile far from the data renders to a 204, and the second ask doesn't render it again."""
    _url, params = _tile_url(notebook_query_data)
    url = "/v1/data/8/0/0.png"
    assert client.get(url, params=params).status_code == 204
    assert client.get(url, params=params).status_code == 204
    assert len(render_count) == 1


def test_concurrent_requests_for_one_tile_share_its_render():
    renders = []
    release = threading.Event()

    def render():
        renders.append(1)
        release.wait(5)
        return b"tile"

    async def scenario():
        results = []

        async def tile():
            results.append(await api_pipeline._render_tile_once("k", render, None))

        async with anyio.create_task_group() as tg:
            for _ in range(3):
                tg.start_soon(tile)
            await anyio.sleep(0.05)
            release.set()
        return results

    assert anyio.run(scenario) == [b"tile"] * 3
    assert len(renders) == 1


def test_waiters_render_it_themselves_when_the_first_client_leaves_first(monkeypatch):
    """The first request for a tile queued behind a full limiter and then hung up: nothing
    was rendered, so the request waiting on it must not come back empty."""
    monkeypatch.setattr(config, "TILE_RENDER_CONCURRENCY", 1)
    release = threading.Event()

    def blocker():
        release.wait(5)
        return b"other"

    async def scenario():
        gone = anyio.Event()
        results = {}

        async def ask(name, key, render, receive):
            results[name] = await api_pipeline._render_tile_once(key, render, receive)

        async with anyio.create_task_group() as tg:
            tg.start_soon(ask, "blocker", "other", blocker, None)
            await anyio.sleep(0.05)
            tg.start_soon(ask, "leaver", "k", lambda: b"tile", _receive_until(gone))
            await anyio.sleep(0.05)
            tg.start_soon(ask, "waiter", "k", lambda: b"tile", None)
            await anyio.sleep(0.05)
            gone.set()
            await anyio.sleep(0.05)
            release.set()
        return results

    started = time.monotonic()
    results = anyio.run(scenario)
    assert time.monotonic() - started < 5
    assert results == {"blocker": b"other", "leaver": None, "waiter": b"tile"}


def test_waiters_get_the_render_error():
    def render():
        time.sleep(0.05)
        raise FileNotFoundError("gone")

    async def scenario():
        errors = []

        async def tile():
            try:
                await api_pipeline._render_tile_once("k", render, None)
            except FileNotFoundError as exc:
                errors.append(exc)

        async with anyio.create_task_group() as tg:
            for _ in range(2):
                tg.start_soon(tile)
        return errors

    assert len(anyio.run(scenario)) == 2
