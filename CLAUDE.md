# Running Python in this repo

## Never use `.venv` — it points outside the sandbox

`.venv` is tied to a Python interpreter that lives outside the sandbox, so it cannot be
used from here. Don't try to repair it, don't `uv sync` it, and don't run anything
through `.venv/bin/python`.

Instead, build your own environment under `.verify-scratch/` with `uv`. That directory is
gitignored, so environments there are safe to create, rebuild, or delete freely.

The best option is to sync the real dependency set there, which keeps you on the same
versions as CI (see "Running the tests" below):

```sh
UV_PROJECT_ENVIRONMENT=.verify-scratch/locked uv sync --python 3.13 --extra api --group dev
```

`UV_PROJECT_ENVIRONMENT` is what makes `uv sync` safe here — it redirects the sync away
from `.venv` to the given path, so the broken `.venv` is never touched.

For a quick one-off check that doesn't need the whole project, an ad-hoc venv is fine:

```sh
uv python install 3.13
uv venv --python 3.13 .verify-scratch/<name>
uv pip install --python .verify-scratch/<name>/bin/python <packages>
PYTHONPATH=. .verify-scratch/<name>/bin/python -c "..."
```

Pin the Python version explicitly — don't let `uv venv` pick for you. Some compiled
dependencies (rasterio/GDAL, notably) may not have prebuilt wheels for every Python
version available on this machine even though `pyproject.toml`'s `requires-python`
constraint would technically allow it; 3.13 is confirmed to work for raster work.

`PYTHONPATH=.` is required for the ad-hoc form — the package isn't installed into a bare
`uv venv`, so imports resolve against the repo checkout directly. A `uv sync`-built
environment installs the project and doesn't need it.

## uv-managed interpreters go dangling between sessions

`uv`-installed CPython builds live under `~/.local/share/uv/python/` outside the repo and
get garbage-collected independently of any venv that references them. A venv directory
and its installed packages can still be on disk while its `bin/python` symlink points at
nothing. **Never trust an existing scratch venv without checking first:**

```sh
readlink -f .verify-scratch/<name>/bin/python   # empty output = dangling
```

If it's dangling, rebuild it — don't try to work around a dangling interpreter.

## Running the tests

Sync from the lockfile, redirecting `uv sync` away from `.venv` with
`UV_PROJECT_ENVIRONMENT`:

```sh
UV_PROJECT_ENVIRONMENT=.verify-scratch/locked uv sync --python 3.13 --extra api --group dev
.verify-scratch/locked/bin/python -m pytest
```

**Prefer this over `uv pip install <packages>`.** An unconstrained install resolves the
newest release of every dependency, which can differ from what `uv.lock` pins and what CI
therefore runs. That is not hypothetical: an ad-hoc scratch env once resolved Starlette
1.4.0 while the lockfile pinned 0.48.0, and the two versions return different
`Access-Control-Allow-Origin` headers, so a CORS test passed locally and failed for
everyone else.

`PYTHONPATH=.` is not needed here — `uv sync` installs the project itself. The test suite
stubs the MPD database with in-memory SQLite and never needs PostgreSQL, so no PostgreSQL
server or `psycopg2` setup is required even though the `api` extra declares it.

## Testing code that touches `data/`

Never point a write path (`write_parquet`, `write_cog`, `compact_parquet`, migration
scripts, etc.) at the real `data/` directory to test a change. `data/` is tens of
gigabytes of production model output — there's no cheap way to reconstruct it if a bug in
the code under test corrupts or deletes something.

The test suite guards this two ways, and new tests should keep both intact:

1. `tests/conftest.py` sets `CPRA_MP_DATA_BASE_PATH`, `CPRA_MP_DATA_TILES_PATH`,
   `CPRA_MP_DATA_CROSSWALKS_PATH`, and `CPRA_MP_DATA_GRIDS_PATH` into a throwaway sandbox
   **before the first `cpra` import**. `cpra/mp/data/config.py` calls `load_dotenv()`, and
   `.env` is a symlink to real secrets pointing at production data; `load_dotenv` does not
   override variables already set in the environment, so assigning them first wins.
2. Individual tests still pass `base_path=tmp_path` explicitly. The env sandbox is a
   backstop for code paths that don't accept an injected path, not the primary mechanism.

For ad-hoc checks outside the test suite, copy the relevant subtree to `/tmp` first
(preserving relative structure, e.g. `cp -r --parents "data/variable=elev/..."
/tmp/some_test_dir/`) and point `base_path` at the copy.
