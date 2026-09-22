"""Fixtures for the API tests.

The MPD database is stubbed with in-memory SQLite via a dependency override, so no
PostgreSQL server (and no psycopg2) is needed. Only /v1/dimensions.json touches it.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from cpra.mp.data import config
from cpra.mp.data.api import app, get_mpd_session
from cpra.mp.data.api.models import MPDDimensionValue
from cpra.mp.data.grids import GRIDS
from tests.factories import (
    ANNUAL_KWARGS,
    RASTER_KWARGS,
    write_ffibs_coverage_dataset,
    write_raster_crosswalk,
    write_sal_annual_dataset,
)

# Dimension values the real MPD serves but the structure CSVs do not. Described as plain
# tuples rather than model instances: a SQLModel instance becomes bound to the session
# that persisted it, so reusing one across tests silently fails to re-insert.
MPD_ROWS = [
    ("model_group_id", 500, "Future Without Action"),
    ("model_group_id", 520, "Implementation Period 1"),
    ("scenario_id", 7, "Lower Scenario"),
]


@pytest.fixture
def mpd_engine():
    """An in-memory MPD stub, disposed on teardown.

    StaticPool keeps a single sqlite3 connection open for the engine's lifetime, so an
    engine that is merely dropped gets collected with that connection still open. That
    raises a ResourceWarning attributed to whatever code happened to trigger the GC,
    which is both noisy and misleading.
    """
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        for dimension, value_int, label in MPD_ROWS:
            session.add(
                MPDDimensionValue(dimension=dimension, value_int=value_int, label=label)
            )
        session.commit()
    yield engine
    engine.dispose()


@pytest.fixture
def client(mpd_engine):
    def override():
        with Session(mpd_engine) as session:
            yield session

    app.dependency_overrides[get_mpd_session] = override
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def annual_dimensions():
    return {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}


@pytest.fixture
def raster_dimensions():
    return {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}


@pytest.fixture
def salinity_veg_data(base_path, crosswalk_path, monkeypatch):
    """The two hardcoded inputs of the `salinity-veg-investigation` named analysis, on disk.

    Marsh (raster) 0.3->2030, 0.7->2040 (crosses `marsh > 0.5` only in 2040); salinity
    (vector, one zone) 4.0->2030, 6.0->2040 (crosses `sal > 5.5` only in 2040) -- and
    every `veg_grid_cell_v001` pixel maps to the same single `hydro_compartment_v001`
    zone, so the analysis's `{"aggregate": "min", "of": "calendar_year"}` yields a uniform
    2040 raster.

    Lives here rather than in either module because two sibling modules need the same
    data from opposite ends: `test_analyses_registry.py` drives the analysis through
    `resolve_analysis`/`run_to_path` directly, `test_analyses_endpoints.py` through the
    custom-analysis HTTP routes.
    """
    write_ffibs_coverage_dataset(base_path, {2030: 0.3, 2040: 0.7})
    write_sal_annual_dataset(base_path, {2030: 4.0, 2040: 6.0})
    monkeypatch.setattr(config, "BASE_PATH", base_path)

    grid = GRIDS["veg_grid_cell_v001"]
    zones = np.ones((grid.height, grid.width), dtype="int32")
    write_raster_crosswalk(crosswalk_path, "veg_grid_cell_v001", "hydro_compartment_v001", zones)
    return base_path
