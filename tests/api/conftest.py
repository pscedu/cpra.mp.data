"""Fixtures for the API tests.

The MPD database is stubbed with in-memory SQLite via a dependency override, so no
PostgreSQL server (and no psycopg2) is needed. Only /v1/dimensions.json touches it.
"""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from cpra.mp.data.api import app, get_mpd_session
from cpra.mp.data.api.models import (
    DatasetDimensions,
    FilterDimensions,
    MPDDimensionValue,
    Options,
)
from tests.factories import ANNUAL_KWARGS, RASTER_KWARGS

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
def dataset_factory(base_path):
    """Build a `BaseAnalysis` subclass against an isolated base_path.

    Mirrors how the route handlers construct one (via `create_analysis`), but with the
    base path injected so the query never reaches the configured data directory.
    """
    from cpra.mp.data.analysis import create_analysis

    def build(dimensions, filters=None, base=None, **options):
        return create_analysis(
            DatasetDimensions(**dimensions),
            FilterDimensions(**(filters or {})),
            Options(**options),
            base_path=(base if base is not None else base_path),
        )

    return build


@pytest.fixture
def annual_dimensions():
    return {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}


@pytest.fixture
def raster_dimensions():
    return {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}
