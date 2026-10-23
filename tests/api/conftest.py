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
    LND_TYPE_PLUS_SHAPE,
    RASTER_KWARGS,
    write_ffibs_coverage_dataset,
    write_lnd_type_plus_datasets,
    write_raster_crosswalk,
    write_sal_annual_dataset,
    write_sediment_dynamics_datasets,
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
    """The two hardcoded inputs of the `sal_veg_investigation` named analysis, on disk.

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


@pytest.fixture
def morph_pixel_v002_shrunk(monkeypatch):
    """Shrink the real `morph_pixel_v002` grid to `LND_TYPE_PLUS_SHAPE` for one test --
    see the comment above that constant in `tests/factories.py`. `monkeypatch` restores
    both attributes afterwards, so no other test sees the smaller grid.
    """
    grid = GRIDS["morph_pixel_v002"]
    monkeypatch.setattr(grid, "height", LND_TYPE_PLUS_SHAPE[0])
    monkeypatch.setattr(grid, "width", LND_TYPE_PLUS_SHAPE[1])
    return grid


@pytest.fixture
def lnd_type_plus_data(morph_pixel_v002_shrunk, base_path, crosswalk_path, monkeypatch):
    """The three hardcoded inputs of the `lnd_type_plus` named analysis, on disk -- one pixel
    per reclassification rule over open water, plus one that shows the water restriction:
    0.20 m -> 6, 0.70 m -> 7, 1.50 m -> 2 (the deep class is `0`, so the water shows
    through), and 0.20 m over land type 1 -> 1 (a depth class may only overwrite water).
    The remaining four pixels repeat the shallow case, so a uniform result would be visibly
    wrong.

    The depths are what the analysis COMPUTES, not what is stored: `write_lnd_type_plus_datasets`
    writes the elevation that yields each one against a uniform stage, plus the one-zone stage
    series and the crosswalk that carries it onto the pixel grid.

    Lives here rather than in either module for the same reason `salinity_veg_data` does:
    `test_analyses_registry.py` drives the analysis through `resolve_analysis`/`run_to_path`
    directly, `test_analyses_endpoints.py` through the custom-analysis HTTP routes.
    """
    write_lnd_type_plus_datasets(
        base_path,
        crosswalk_path,
        depths=[0.20, 0.70, 1.50, 0.20, 0.20, 0.20, 0.20, 0.20],
        land_types=[2, 2, 2, 1, 2, 2, 2, 2],
    )
    monkeypatch.setattr(config, "BASE_PATH", base_path)
    return base_path


# The years `sediment_dynamics_data` writes, exported so a test can assert the series it
# gets back rather than restating the fixture's own default.
SEDIMENT_DYNAMICS_DATA_YEARS = (2040, 2050)


@pytest.fixture
def sediment_dynamics_data(morph_pixel_v002_shrunk, base_path, monkeypatch):
    """The five inputs of the `sediment_dynamics` analysis on disk, one COG per (variable,
    year) on the shrunk `morph_pixel_v002` -- what the chart of the same name draws.

    The sibling of `lnd_type_plus_data` above, and here for the same reason: the analysis
    is driven through `resolve_analysis` by `test_analyses_registry.py` and through the
    HTTP routes (both the spec route and the data URL that spec emits) by
    `test_endpoints_data.py`, so the one writer is shared rather than repeated.
    """
    selection = write_sediment_dynamics_datasets(base_path, years=SEDIMENT_DYNAMICS_DATA_YEARS)
    monkeypatch.setattr(config, "BASE_PATH", base_path)
    return selection
