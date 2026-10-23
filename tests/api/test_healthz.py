"""The /healthz liveness endpoint."""

import threading

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
