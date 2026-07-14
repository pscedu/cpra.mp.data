import pytest


@pytest.fixture
def base_path(tmp_path):
    path = tmp_path / "data"
    path.mkdir()
    return path


@pytest.fixture
def catalog_path(tmp_path):
    return tmp_path / "catalog"
