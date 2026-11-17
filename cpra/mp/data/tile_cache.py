"""On-disk cache of rendered map tiles, shared by every API replica that mounts it.

One file per tile, named by its key (`tile_key`), under `<root>/v1/<aa>/<bb>/`. A tile that
rendered to nothing (the routes' bodiless 204) is cached too, as a zero-byte file: an
empty tile costs the same plan compile and read as a full one.

**Invalidation is by key, never by deletion.** A key hashes the package version, the
request itself, and the path, size and mtime of the catalog files of every dataset the
request reads (see `api.pipeline._catalog_files`, and `file_fingerprints` here). A new
release, or any `write_cog`/`write_parquet`/backup restore into an input dataset, produces
a different key, so a stale tile is simply never asked for again and ages out. (`data
compact` may leave a single-file catalog partition untouched; that is harmless, since
compaction preserves values.) What the key does NOT see -- grids, crosswalks, data files
edited, replaced or deleted by hand -- is what
`python -m cpra.mp.data.cli tile-cache clear` is for.

**Eviction is least-recently-used by mtime.** A hit re-touches its file (at most once per
`TOUCH_INTERVAL_SECONDS`, so a popular tile is not a metadata write per request), and a
sweep deletes the oldest files once the cache passes `max_bytes`, down to
`SWEEP_TARGET_FRACTION` of it. mtime rather than atime because `relatime` mounts make
atime unreliable. Sizes are counted in allocated blocks, so the zero-byte empties count.

Writes go to a temp file renamed into place, so a reader -- in this process or another
replica -- never sees half a tile. Every filesystem error is swallowed into a miss: the
cache is an optimization and must never fail a tile request.
"""

import hashlib
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from cpra.mp.data import config

logger = logging.getLogger(__name__)

# Bump to orphan every existing entry if the on-disk layout or key recipe changes.
LAYOUT_VERSION = "v1"
TOUCH_INTERVAL_SECONDS = 3600
SWEEP_TARGET_FRACTION = 0.9
# A temp file this old was left by a crashed write, not one in progress.
STALE_TEMP_SECONDS = 3600
_TEMP_SUFFIX = ".tmp"
# Every name this module ever writes: an entry (`<key>.<format>`) or a temp file
# (`.<key>.<random>.tmp`). Deletion is confined to names that match, and to `v<n>` layout
# directories, so a TILE_CACHE_PATH pointed at the wrong directory deletes nothing else.
_ENTRY_NAME = re.compile(r"\.?[0-9a-f]{64}\.[A-Za-z0-9_.]+")
_LAYOUT_DIR = re.compile(r"v\d+")


def _package_version() -> str:
    try:
        return version("cpra.mp.data")
    except PackageNotFoundError:
        return "unknown"


PACKAGE_VERSION = _package_version()


def file_fingerprints(paths) -> list[tuple[str, int, int]]:
    """`(path, size, mtime_ns)` for each of `paths`, sorted. A path that has vanished is
    recorded with `-1`s rather than raising: the render will make its own decision about
    a missing file, and the key still differs from the one taken while it existed.
    """
    fingerprints = []
    for path in {str(p) for p in paths}:
        try:
            st = os.stat(path)
            fingerprints.append((path, st.st_size, st.st_mtime_ns))
        except OSError:
            fingerprints.append((path, -1, -1))
    return sorted(fingerprints)


def tile_key(request: str, format: str, z: int, x: int, y: int, sources) -> str:
    """The cache key for one tile: `request` is a canonical serialization of everything
    about the request that shapes the tile (the pipeline, the base path), `sources` the
    `file_fingerprints` of everything it reads."""
    digest = hashlib.sha256()
    for part in (LAYOUT_VERSION, PACKAGE_VERSION, request, format, f"{z}/{x}/{y}", repr(sources)):
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass
class CacheStats:
    files: int
    bytes: int


def _disk_bytes(st: os.stat_result) -> int:
    return st.st_blocks * 512


def _walk(root: Path):
    """Yield `(path, stat)` for every cache-written file under `root`, tolerating files
    removed mid-walk (another replica's sweep)."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if not _ENTRY_NAME.fullmatch(name):
                continue
            path = os.path.join(dirpath, name)
            try:
                yield path, os.stat(path)
            except OSError:
                continue


class TileCache:
    def __init__(self, root, max_bytes: int):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self._entries = self.root / LAYOUT_VERSION
        # Bytes believed to be in the cache: seeded by the first sweep, then advanced by
        # this process's own writes. Other replicas' writes are only seen at the next
        # sweep, which is fine -- the limit is a target, not a quota.
        self._estimated_bytes: int | None = None
        self._lock = threading.Lock()
        self._sweeping = False

    def _path(self, key: str, format: str) -> Path:
        return self._entries / key[:2] / key[2:4] / f"{key}.{format}"

    def get(self, key: str, format: str) -> bytes | None:
        """The cached body (`b""` for a cached empty tile), or `None` on a miss."""
        path = self._path(key, format)
        try:
            body = path.read_bytes()
            if time.time() - path.stat().st_mtime > TOUCH_INTERVAL_SECONDS:
                os.utime(path)
        except FileNotFoundError:
            return None
        except OSError:
            logger.warning("tile cache read failed: %s", path, exc_info=True)
            return None
        return body

    def put(self, key: str, format: str, body: bytes | None) -> None:
        """Store `body` (`None` for an empty tile) under `key`."""
        path = self._path(key, format)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{key}.", suffix=_TEMP_SUFFIX)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(body or b"")
                os.replace(tmp, path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
            written = _disk_bytes(path.stat())
        except OSError:
            logger.warning("tile cache write failed: %s", path, exc_info=True)
            return
        with self._lock:
            if self._estimated_bytes is not None:
                self._estimated_bytes += written
            due = self._estimated_bytes is None or self._estimated_bytes > self.max_bytes
            if due and not self._sweeping:
                self._sweeping = True
                threading.Thread(target=self._sweep_in_background, name="tile-cache-sweep", daemon=True).start()

    def _sweep_in_background(self) -> None:
        try:
            self.sweep()
        except Exception:
            logger.warning("tile cache sweep failed", exc_info=True)
        finally:
            with self._lock:
                self._sweeping = False

    def sweep(self, max_bytes: int | None = None) -> CacheStats:
        """Delete crashed writes' temp files and, if the cache is over `max_bytes`
        (default: this cache's own limit), its least-recently-used tiles down to
        `SWEEP_TARGET_FRACTION` of it. Returns what is left."""
        max_bytes = self.max_bytes if max_bytes is None else max_bytes
        now = time.time()
        entries = []
        total = 0
        for path, st in _walk(self._entries):
            if path.endswith(_TEMP_SUFFIX):
                if now - st.st_mtime > STALE_TEMP_SECONDS:
                    Path(path).unlink(missing_ok=True)
                continue
            entries.append((st.st_mtime, path, _disk_bytes(st)))
            total += entries[-1][2]
        files = len(entries)
        if total > max_bytes:
            target = max_bytes * SWEEP_TARGET_FRACTION
            entries.sort()
            for _mtime, path, size in entries:
                if total <= target:
                    break
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    continue
                total -= size
                files -= 1
            logger.info("tile cache swept to %d bytes in %d files", total, files)
        with self._lock:
            self._estimated_bytes = total
        return CacheStats(files=files, bytes=total)

    def stats(self) -> CacheStats:
        files = total = 0
        for path, st in _walk(self._entries):
            if not path.endswith(_TEMP_SUFFIX):
                files += 1
                total += _disk_bytes(st)
        return CacheStats(files=files, bytes=total)

    def clear(self) -> CacheStats:
        """Delete every entry, including any left by older layout versions. Returns what
        was removed."""
        removed = CacheStats(files=0, bytes=0)
        layouts = (
            [p for p in self.root.iterdir() if p.is_dir() and _LAYOUT_DIR.fullmatch(p.name)]
            if self.root.is_dir()
            else []
        )
        for layout in layouts:
            for path, st in _walk(layout):
                try:
                    os.unlink(path)
                except OSError:
                    continue
                removed.files += 1
                removed.bytes += _disk_bytes(st)
            # Only now-empty directories go; `rmdir` refuses anything else.
            for dirpath, _dirnames, _filenames in os.walk(layout, topdown=False):
                try:
                    os.rmdir(dirpath)
                except OSError:
                    pass
        with self._lock:
            self._estimated_bytes = 0
        return removed


_cache: TileCache | None = None
_cache_config: tuple | None = None


def get_tile_cache() -> TileCache | None:
    """The configured cache, or `None` when caching is off (`TILE_CACHE_MAX_GB` <= 0).
    Rebuilt if the config changes, so tests can monkeypatch it."""
    global _cache, _cache_config
    current = (config.TILE_CACHE_PATH, config.TILE_CACHE_MAX_GB)
    if current != _cache_config:
        _cache_config = current
        _cache = TileCache(current[0], int(current[1] * 1024**3)) if current[1] > 0 else None
    return _cache
