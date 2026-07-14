# Running Python in this repo

## uv-managed interpreters go dangling between sessions

`uv`-installed CPython builds live under `~/.local/share/uv/python/` (or
`~/.local/bin/python3.x`) outside the repo, and get garbage-collected independently of
any venv that references them. A venv directory and its installed packages can still be
on disk while its `bin/python` symlink points at nothing. **Never trust an existing venv
without checking first:**

```sh
readlink -f .venv/bin/python   # empty output / "No such file or directory" = dangling
```

If it's dangling, rebuild it (see below) before running anything through it — don't try
to work around a dangling interpreter, just recreate the venv.

## Don't `uv sync` the project `.venv` for exploration

`uv sync` will silently reconcile `.venv` to match `pyproject.toml`'s lockfile, which can
mean **deleting and reinstalling packages that were manually added** (e.g. a package
installed ad hoc with `uv pip install` for a one-off check). It can also fall back to a
different interpreter than you expect (observed: falling back to system Python 3.14 when
`.venv`'s pinned 3.13 interpreter had gone dangling) and can fail partway through with a
permission error, leaving `.venv` in a half-wiped state.

If you just need to run something once or check an import, **build an isolated scratch
venv instead of touching `.venv`** (see below). Only run `uv sync` against `.venv` when
you actually intend to bring the real project environment in line with `pyproject.toml`.

## Building a scratch venv

Pin the Python version explicitly — don't let `uv venv`/`uv sync` pick for you. Some
compiled dependencies (rasterio/GDAL, notably) may not have prebuilt wheels for every
Python version available on this machine even though `pyproject.toml`'s
`requires-python` constraint would technically allow it; 3.13 is confirmed to work for
raster work as of this writing.

```sh
uv python install 3.13
uv venv --python 3.13 .verify-scratch/<name>
uv pip install --python .verify-scratch/<name>/bin/python <packages>
PYTHONPATH=. .verify-scratch/<name>/bin/python -c "..."
```

`.verify-scratch/` is the existing convention for these throwaway environments — it's
gitignored, so venvs there are safe to leave, rebuild, or delete freely. Prefer reusing
it over creating new scratch locations elsewhere in the repo.

`PYTHONPATH=.` is required — the package isn't installed into scratch venvs, so imports
resolve against the repo checkout directly.

## Testing code that touches `data/`

Never point a write path (`write_cog`, `compact_cog`, migration scripts, etc.) at the
real `data/` directory to test a change. Copy the relevant subtree to an isolated `/tmp`
location first (preserving the relative path structure, e.g. `cp -r --parents
"data/variable=elev/..." /tmp/some_test_dir/`) and point `base_path` at the copy. `data/`
is tens of gigabytes of production model output — there's no cheap way to reconstruct it
if a bug in the code under test corrupts or deletes something.
