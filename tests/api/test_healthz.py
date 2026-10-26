"""The /healthz liveness endpoint."""

import threading
import time

import anyio
import anyio.to_thread
import pytest
from httpx import ASGITransport, AsyncClient

from cpra.mp.data.api import app


def test_healthz_ok(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz_not_in_schema(client):
    assert "/healthz" not in client.get("/openapi.json").json()["paths"]


def test_healthz_blocks_when_thread_pool_is_exhausted():
    """The point of the endpoint: it must stop answering once every worker thread is stuck
    (as on a hung Lustre read), where an async route like /openapi.json would not. The
    default limiter is per event loop, so shrinking it here doesn't leak into other tests."""
    release = threading.Event()

    async def exhaust_and_probe():
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 2
        async with anyio.create_task_group() as tg:
            for _ in range(limiter.total_tokens):
                tg.start_soon(anyio.to_thread.run_sync, release.wait)
            await anyio.sleep(0.1)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                assert (await ac.get("/openapi.json")).status_code == 200
                with pytest.raises(TimeoutError):
                    with anyio.fail_after(0.5):
                        await ac.get("/healthz")
            release.set()

    anyio.run(exhaust_and_probe)


def test_healthz_503_when_tile_renders_stall(client, monkeypatch):
    """Tile renders queue on their own limiter, so a hung tile read cannot exhaust the
    default pool above; a saturated tile limiter that has made no progress is the signal
    instead."""
    from cpra.mp.data import config
    from cpra.mp.data.api import pipeline

    monkeypatch.setattr(pipeline, "_tiles_in_flight", config.TILE_RENDER_CONCURRENCY)
    monkeypatch.setattr(pipeline, "_last_tile_progress", time.monotonic() - pipeline.TILE_RENDER_STALL_SECONDS - 1)
    assert client.get("/healthz").status_code == 503

    # Saturated but still progressing is load, not a stall.
    monkeypatch.setattr(pipeline, "_last_tile_progress", time.monotonic())
    assert client.get("/healthz").status_code == 200

    # Stale progress with a free slot is just an idle server.
    monkeypatch.setattr(pipeline, "_tiles_in_flight", config.TILE_RENDER_CONCURRENCY - 1)
    monkeypatch.setattr(pipeline, "_last_tile_progress", 0.0)
    assert client.get("/healthz").status_code == 200


def test_tile_renders_are_capped(monkeypatch):
    """A burst of tile requests renders TILE_RENDER_CONCURRENCY at a time, not all at once
    -- the cap that keeps a map view's tile burst from OOM-killing the pod."""
    from cpra.mp.data import config
    from cpra.mp.data.api import pipeline

    monkeypatch.setattr(config, "TILE_RENDER_CONCURRENCY", 2)
    lock = threading.Lock()
    running = peak = 0

    def render():
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1
        return b"tile"

    async def burst():
        async with anyio.create_task_group() as tg:
            for _ in range(8):
                tg.start_soon(pipeline._run_tile_render, render)

    anyio.run(burst)
    assert peak == 2
    assert pipeline._tiles_in_flight == 0
