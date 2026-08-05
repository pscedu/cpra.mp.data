"""Shared fixtures and production-data isolation for the test suite.

The environment assignments below MUST run before the first ``cpra`` import.
``cpra/mp/data/config.py`` calls ``load_dotenv()`` at import time, and ``.env`` is a
symlink to the developer's real secrets file, which points ``CPRA_MP_DATA_BASE_PATH`` at
tens of gigabytes of production model output. ``load_dotenv`` does not override variables
that are already set, so assigning them here wins.

This is a backstop, not the primary mechanism: individual tests still pass
``base_path=tmp_path`` explicitly. The sandbox exists so that a test which *forgets* to
inject a path writes into an empty temporary directory rather than into ``data/``.
"""

import os
import tempfile
from pathlib import Path

_SANDBOX = Path(tempfile.mkdtemp(prefix="cpra-test-sandbox-"))

os.environ["CPRA_MP_DATA_BASE_PATH"] = str(_SANDBOX / "data")
os.environ["CPRA_MP_DATA_TILES_PATH"] = str(_SANDBOX / "tiles")
os.environ["CPRA_MP_DATA_CROSSWALK_PATH"] = str(_SANDBOX / "crosswalk")
os.environ["CPRA_MP_DATA_MPD_URI"] = "sqlite://"
os.environ["CPRA_MP_DATA_DEBUG"] = "False"

import pytest  # noqa: E402

from cpra.mp.data import config as cpra_config  # noqa: E402
from cpra.mp.data.api.utils import _read_data_cached  # noqa: E402


def pytest_configure(config):
    """Fail loudly if the sandbox did not take effect."""
    if not str(cpra_config.BASE_PATH).startswith(str(_SANDBOX)):
        raise RuntimeError(
            f"Test sandbox not applied: BASE_PATH is {cpra_config.BASE_PATH}. "
            "Refusing to run against a real data directory."
        )


@pytest.fixture(autouse=True)
def clear_read_cache():
    """Reset the API's TTL cache around every test.

    ``api.utils._read_data_cached`` is wrapped in a ``TTLCache`` keyed only on the
    dimension kwargs. Without this, two tests using identical dimensions but different
    ``tmp_path`` directories would silently share the first test's LazyFrame.
    """
    _read_data_cached.cache_clear()
    yield
    _read_data_cached.cache_clear()


@pytest.fixture
def base_path(tmp_path):
    """An isolated, empty data directory."""
    path = tmp_path / "data"
    path.mkdir()
    return path


@pytest.fixture
def crosswalk_path(tmp_path, monkeypatch):
    """An isolated crosswalk directory, patched over the configured one."""
    path = tmp_path / "crosswalk"
    path.mkdir()
    monkeypatch.setattr(cpra_config, "CROSSWALK_PATH", path)
    return path


@pytest.fixture
def tiles_path(tmp_path, monkeypatch):
    """An isolated PMTiles directory, patched over the configured one."""
    path = tmp_path / "tiles"
    path.mkdir()
    monkeypatch.setattr(cpra_config, "TILES_PATH", path)
    return path
