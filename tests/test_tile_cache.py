"""`cpra.mp.data.tile_cache`: the on-disk store, its eviction, and its key."""

import os
import time

from cpra.mp.data.tile_cache import TOUCH_INTERVAL_SECONDS, TileCache, file_fingerprints, tile_key


def _key(n: int) -> str:
    return f"{n:064x}"


def _age(cache, key, format, seconds):
    path = cache._path(key, format)
    then = time.time() - seconds
    os.utime(path, (then, then))
    return path


def test_round_trip_and_miss(tmp_path):
    cache = TileCache(tmp_path, 10**9)
    assert cache.get(_key(1), "png") is None
    cache.put(_key(1), "png", b"tile")
    assert cache.get(_key(1), "png") == b"tile"
    assert cache.get(_key(1), "jpg") is None


def test_empty_tile_is_cached_as_empty_bytes(tmp_path):
    """A 204 tile is a hit too, distinguishable from a miss."""
    cache = TileCache(tmp_path, 10**9)
    cache.put(_key(1), "png", None)
    assert cache.get(_key(1), "png") == b""


def test_hit_refreshes_a_stale_mtime(tmp_path):
    cache = TileCache(tmp_path, 10**9)
    cache.put(_key(1), "png", b"tile")
    path = _age(cache, _key(1), "png", TOUCH_INTERVAL_SECONDS + 60)
    cache.get(_key(1), "png")
    assert time.time() - path.stat().st_mtime < 60


def test_sweep_evicts_least_recently_used(tmp_path):
    cache = TileCache(tmp_path, 10**9)
    for n in range(4):
        cache.put(_key(n), "png", b"x" * 10_000)
    # Oldest first: 0, 1, 2, 3 -- but 0 was just read, so it is the newest.
    for n, age in enumerate([400, 300, 200, 100]):
        _age(cache, _key(n), "png", age)
    cache.get(_key(0), "png")
    os.utime(cache._path(_key(0), "png"))
    per_tile = cache.stats().bytes // 4

    stats = cache.sweep(max_bytes=per_tile * 3 - 1)

    assert stats.files == 2
    assert [cache.get(_key(n), "png") is not None for n in range(4)] == [True, False, False, True]


def test_sweep_removes_abandoned_temp_files(tmp_path):
    cache = TileCache(tmp_path, 10**9)
    cache.put(_key(1), "png", b"tile")
    temp = cache._path(_key(1), "png").parent / f".{_key(2)}.abc123.tmp"
    temp.write_bytes(b"half")
    fresh = temp.with_name(f".{_key(3)}.def456.tmp")
    fresh.write_bytes(b"in progress")
    then = time.time() - 2 * 3600
    os.utime(temp, (then, then))

    assert cache.sweep().files == 1
    assert not temp.exists()
    assert fresh.exists()


def test_put_past_the_limit_sweeps(tmp_path):
    cache = TileCache(tmp_path, 1)
    cache.put(_key(1), "png", b"x" * 10_000)
    deadline = time.time() + 5
    while cache.stats().files and time.time() < deadline:
        time.sleep(0.01)
    assert cache.stats().files == 0


def test_clear_deletes_only_what_the_cache_wrote(tmp_path):
    """A TILE_CACHE_PATH pointed at the wrong directory must not take it with it."""
    cache = TileCache(tmp_path, 10**9)
    cache.put(_key(1), "png", b"tile")
    (tmp_path / "v1" / "notes.txt").write_text("keep")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / f"{_key(2)}.tif").write_bytes(b"keep")

    removed = cache.clear()

    assert removed.files == 1
    assert (tmp_path / "v1" / "notes.txt").exists()
    assert (tmp_path / "data" / f"{_key(2)}.tif").exists()
    assert cache.get(_key(1), "png") is None


def test_key_follows_source_files(tmp_path):
    source = tmp_path / "a.tif"
    source.write_bytes(b"one")
    before = tile_key("req", "png", 8, 1, 2, file_fingerprints([source]))
    assert tile_key("req", "png", 8, 1, 2, file_fingerprints([source])) == before

    source.write_bytes(b"changed")
    assert tile_key("req", "png", 8, 1, 2, file_fingerprints([source])) != before
    assert tile_key("req", "png", 8, 1, 3, file_fingerprints([source])) != tile_key(
        "req", "png", 8, 1, 2, file_fingerprints([source])
    )
    assert tile_key("other", "png", 8, 1, 2, file_fingerprints([source])) != tile_key(
        "req", "png", 8, 1, 2, file_fingerprints([source])
    )


def test_missing_source_is_fingerprinted_not_raised(tmp_path):
    assert file_fingerprints([tmp_path / "gone.tif"]) == [(str(tmp_path / "gone.tif"), -1, -1)]
