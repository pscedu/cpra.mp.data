"""Stage compacted files where no reader can see them until they are complete.

Compaction (``tabular.compact_parquet``, ``raster.compact_cog``,
``catalog.Catalog.compact_partition``) replaces many files with one. Writing the new file
straight into its directory would let a reader find it half-written, and a compaction
killed part-way (an OOM kill, a node restart) would leave a truncated file behind under a
name every reader matches.

A hidden name in the same directory isn't enough either: ``tabular.read_parquet`` scans
the whole dataset directory, and Polars takes every file under it -- hidden or not, of any
extension -- and fails on one that isn't Parquet. So new files are written into a hidden
``.compact-*`` directory directly under the data (or catalog) root. Every reader there
globs ``variable=*/...``, which never matches it, and it is on the same filesystem as the
data, so ``publish`` can move a finished file into place with an atomic rename. Only then
are the sources removed; between the two a reader can see both, so a read in that moment
may count the merged rows twice.

The staging directory is removed when compaction finishes or fails. One left by a killed
process is removed by ``remove_stale`` at the start of the next compaction, which assumes
no other compaction is running against the same root at the same time.
"""

import contextlib
import shutil
import tempfile
from pathlib import Path

STAGING_PREFIX = ".compact-"


def remove_stale(root) -> list[Path]:
    """Remove staging directories an interrupted compaction left under ``root``."""
    removed = []
    for path in Path(root).glob(f"{STAGING_PREFIX}*"):
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(path)
    return removed


@contextlib.contextmanager
def area(root):
    """A fresh hidden directory under ``root`` for files that aren't finished yet, removed
    (with anything still in it) on exit."""
    with tempfile.TemporaryDirectory(dir=root, prefix=STAGING_PREFIX) as directory:
        yield Path(directory)


def publish(staged: Path, final: Path) -> Path:
    """Move a finished file from the staging directory to ``final``."""
    staged.rename(final)
    return final
