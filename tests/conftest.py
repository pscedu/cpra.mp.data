import os

import pytest

# cpra.mp.data.api builds a SQLAlchemy engine at import time, so importing anything from
# it (e.g. Dataset) needs a parseable URL. Set before any cpra import binds config.MPD_URI.
os.environ.setdefault("CPRA_MP_DATA_MPD_URI", "sqlite://")


@pytest.fixture
def base_path(tmp_path):
    path = tmp_path / "data"
    path.mkdir()
    return path


@pytest.fixture
def catalog_path(tmp_path):
    return tmp_path / "catalog"
