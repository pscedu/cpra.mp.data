"""Merge already-sorted Parquet files into one sorted file in bounded memory.

``tabular.compact_parquet`` uses this instead of reading a partition into memory and
sorting it. Every input was written sorted by the same key (``utils.long_sort_expressions``:
the outer dims, then ``grid_cell``, then the calendar column), so the output order can be
built from the inputs' own order instead of re-sorting:

1. Each file is split into *pieces*, one per run (the leading ``run_cols`` of the key,
   e.g. model_group_id/scenario_id), found from row-group statistics where they can be.
   A run whose rows all sit in one piece -- every run of a compacted file, unless more of
   it has been written since -- is copied through in key order.
2. A run's pieces whose key ranges don't overlap are copied through too. Overlapping
   pieces are *interleaved* when they share one sequence of prefix groups
   (every key column except the calendar one), each file holding a fixed number of rows
   per group -- one run written a year (or a few years) at a time. Output positions are
   index arithmetic: per group, the first file's block, then the next file's, and so on.
   That includes years added to a run already in a compacted file. The layout is checked
   chunk by chunk, before anything from a chunk is written.
3. Anything else is a *windowed merge*: each piece contributes one small batch, and only
   the rows that can't be preceded by any unread row are emitted. That covers e.g. the
   same year written twice. If the interleave's layout check fails part-way, the merge
   takes over from that point; the rows already written stay valid.

At most ``max_open_files`` pieces are read at once. A larger group of overlapping pieces is
merged in rounds through temporary files, so peak memory is set by ``max_open_files``,
``batch_size`` and ``row_group_size``, never by the number of rows. The default of 55 is
one file per year of a full run, so a run written a year at a time merges in one round.
Measured through ``compact_parquet`` with ``ARROW_DEFAULT_MEMORY_POOL=jemalloc`` (about
180 MB of each figure is the package imports): 55 one-year files with three columns peak at
about 580 MB (460 MB with 16 open); 13 Float64 columns, about 730 MB (500 MB with 16 open).
Each open file costs roughly 3.5-6 MB, depending on the number of columns.

If an input isn't actually sorted, no rows are lost, but the output isn't fully sorted
either -- except that a file whose runs are out of order raises ``ValidationError``.
"""

import bisect
import contextlib
import tempfile
import uuid
from itertools import accumulate
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from cpra.mp.data.exceptions import ValidationError

# Default number of input files read at once. Each costs roughly one decompressed page per
# column (about 7 MB with our 500k-row row groups); see the module docstring.
MAX_OPEN_FILES = 55

# Rows pulled from an input file at a time. Small, because pyarrow's reader memory grows
# with it and the output order never needs more than a block per group from each file.
BATCH_SIZE = 4096


class _LayoutError(Exception):
    """The inputs don't fit the interleave's layout; fall back to the windowed merge."""


class _Piece(NamedTuple):
    """Rows ``start`` up to ``stop`` of the file at ``path``."""

    path: Path
    start: int
    stop: int

    @classmethod
    def whole(cls, path):
        return cls(Path(path), 0, pq.read_metadata(path).num_rows)


def _row_group_offsets(metadata):
    """The first row of each row group, plus the file's row count."""
    return [0, *accumulate(metadata.row_group(i).num_rows for i in range(metadata.num_row_groups))]


def _iter_rows(file, start, stop, batch_size, columns):
    """Yield rows ``start`` up to ``stop`` of an open ParquetFile, decoding only the row
    groups that hold them."""
    offsets = _row_group_offsets(file.metadata)
    first = bisect.bisect_right(offsets, start) - 1
    last = bisect.bisect_left(offsets, stop)
    skip, remaining = start - offsets[first], stop - start
    for batch in file.iter_batches(batch_size=batch_size, row_groups=list(range(first, last)), columns=columns):
        if skip >= batch.num_rows:
            skip -= batch.num_rows
            continue
        batch = batch.slice(skip, remaining)
        skip = 0
        remaining -= batch.num_rows
        if batch.num_rows:
            yield batch
        if not remaining:
            return


class _Source:
    """One input piece read forward in small batches, with a peekable buffer."""

    def __init__(self, piece, schema, batch_size):
        self.path = piece.path
        self.file = pq.ParquetFile(piece.path, pre_buffer=False)
        self.rows = piece.stop - piece.start
        self._schema = schema
        self._batches = _iter_rows(self.file, piece.start, piece.stop, batch_size, schema.names)
        self._buffer = []
        self._buffered = 0

    def close(self):
        self.file.close()

    def _pull(self):
        for batch in self._batches:
            if batch.num_rows:
                table = pa.Table.from_batches([batch])
                if not table.schema.equals(self._schema, check_metadata=False):
                    table = table.cast(self._schema)
                self._buffer.append(table)
                self._buffered += table.num_rows
                return True
        return False

    def peek(self, n=None):
        """Up to ``n`` buffered rows (all of the current buffer when ``n`` is None, pulling
        one batch if it's empty), without consuming them."""
        if n is None:
            if not self._buffered:
                self._pull()
        else:
            while self._buffered < n and self._pull():
                pass
        if not self._buffered:
            return None
        table = pa.concat_tables(self._buffer)
        return table if n is None else table.slice(0, n)

    def drop(self, n):
        table = pa.concat_tables(self._buffer).slice(n)
        self._buffer = [table] if table.num_rows else []
        self._buffered = table.num_rows

    def drain(self):
        """Yield every remaining row, buffered first."""
        yield from self._buffer
        self._buffer, self._buffered = [], 0
        while self._pull():
            yield self._buffer.pop()
            self._buffered = 0


class _Writer:
    """Write a stream of tables as row groups of exactly ``row_group_size`` rows."""

    def __init__(self, path, schema, row_group_size, write_options):
        self._writer = pq.ParquetWriter(path, schema, **write_options)
        self._row_group_size = row_group_size
        self._pending = []
        self._buffered = 0
        self.rows = 0

    def write(self, table):
        self._pending.append(table)
        self._buffered += table.num_rows
        while self._buffered >= self._row_group_size:
            pending = pa.concat_tables(self._pending)
            self._writer.write_table(pending.slice(0, self._row_group_size), row_group_size=self._row_group_size)
            rest = pending.slice(self._row_group_size)
            self._pending, self._buffered = [rest], rest.num_rows
            self.rows += self._row_group_size

    def close(self):
        if self._buffered:
            self._writer.write_table(pa.concat_tables(self._pending), row_group_size=self._row_group_size)
            self.rows += self._buffered
        self._pending, self._buffered = [], 0
        self._writer.close()


def common_schema(paths):
    """The schema every input is read as: the first file's, which every other file must
    match by column name (types are cast to it)."""
    schemas = [pq.read_schema(path) for path in paths]
    first = schemas[0].remove_metadata()
    for path, schema in zip(paths[1:], schemas[1:]):
        if set(schema.names) != set(first.names):
            msg = (
                f"cannot compact {path}: its columns {sorted(schema.names)} differ from "
                f"{paths[0]}'s {sorted(first.names)}"
            )
            raise ValidationError(msg)
    return first


def _key_frame(table, key):
    return pl.from_arrow(table.select(key))


def _bounds(piece, key):
    """The first and last key of a non-empty piece, without reading the whole piece.

    Small batches throughout: reading the last row group whole would, across many files,
    set the peak before any merging starts."""
    with pq.ParquetFile(piece.path, pre_buffer=False) as file:
        (first,) = _iter_rows(file, piece.start, piece.start + 1, BATCH_SIZE, key)
        (last,) = _iter_rows(file, piece.stop - 1, piece.stop, BATCH_SIZE, key)
    return pa.Table.from_batches([first]).select(key), pa.Table.from_batches([last]).select(key)


def _run_pieces(path, run_cols):
    """``(run, piece)`` for each run (value of ``run_cols``) in a file, in file order.

    A row group whose statistics show one run isn't decoded, so a compacted file costs
    little more than its metadata. Raises ValidationError if the runs aren't in order."""
    path = Path(path)
    with pq.ParquetFile(path, pre_buffer=False) as file:
        metadata = file.metadata
        if not run_cols:
            return [((), _Piece(path, 0, metadata.num_rows))]
        columns = [file.schema_arrow.get_field_index(col) for col in run_cols]
        starts = []  # (run, first row), one per change of run
        offsets = _row_group_offsets(metadata)
        for i in range(metadata.num_row_groups):
            stats = [metadata.row_group(i).column(col).statistics for col in columns]
            if all(stat is not None and stat.has_min_max and stat.min == stat.max for stat in stats):
                starts.append((tuple(stat.min for stat in stats), offsets[i]))
                continue
            position = offsets[i]
            for batch in file.iter_batches(batch_size=65_536, row_groups=[i], columns=run_cols):
                frame = pl.from_arrow(batch).with_row_index("__i", offset=position)
                changed = pl.any_horizontal([pl.col(col).ne_missing(pl.col(col).shift()) for col in run_cols])
                first_rows = frame.filter(changed | (pl.col("__i") == position))
                starts.extend((row[1:], row[0]) for row in first_rows.select("__i", *run_cols).iter_rows())
                position += batch.num_rows

    pieces = []
    for run, start in starts:
        if pieces and pieces[-1][0] == run:
            continue
        if pieces and not pieces[-1][0] < run:
            msg = f"cannot merge {path}: it isn't sorted by {', '.join(run_cols)} ({pieces[-1][0]} before {run})"
            raise ValidationError(msg)
        pieces.append((run, start))
    stops = [start for _, start in pieces[1:]] + [metadata.num_rows]
    return [(run, _Piece(path, start, stop)) for (run, start), stop in zip(pieces, stops)]


def _dense_rank(frame, key):
    """Rank rows of ``frame`` by ``key`` in the order Polars sorts them (equal keys share a
    rank), which is the order the files were written in."""
    ordered = frame.with_row_index("__i").sort(key, maintain_order=True)
    changed = pl.any_horizontal([pl.col(col).ne_missing(pl.col(col).shift()) for col in key])
    return ordered.with_columns(changed.cum_sum().alias("__rank")).sort("__i")["__rank"].to_list()


def _clusters(pieces, key, schema):
    """Group non-empty pieces into clusters of overlapping key ranges, each in key order.

    Pieces of one cluster have to be merged; consecutive clusters can just be concatenated.
    """
    if len(pieces) == 1:
        return [pieces]
    key_schema = pa.schema([schema.field(col) for col in key])
    bounds = [_bounds(piece, key) for piece in pieces]
    firsts = pl.concat([_key_frame(first.cast(key_schema), key) for first, _ in bounds])
    lasts = pl.concat([_key_frame(last.cast(key_schema), key) for _, last in bounds])
    ranks = _dense_rank(pl.concat([firsts, lasts]), key)
    first_rank, last_rank = ranks[: len(pieces)], ranks[len(pieces) :]

    order = sorted(range(len(pieces)), key=lambda i: (first_rank[i], last_rank[i]))
    clusters, reach = [], None
    for i in order:
        # A piece starting before everything so far has ended overlaps it. Equal keys may
        # sit either way round, so a tie at the boundary doesn't count.
        if clusters and first_rank[i] < reach:
            clusters[-1].append(pieces[i])
            reach = max(reach, last_rank[i])
        else:
            clusters.append([pieces[i]])
            reach = last_rank[i]
    return clusters


def _block_length(source, prefix):
    """Rows in the source's first prefix group (one per calendar value it holds)."""
    n = BATCH_SIZE
    while True:
        frame = _key_frame(source.peek(n), prefix)
        changed = pl.any_horizontal([pl.col(col).ne_missing(pl.lit(frame[col][0])) for col in prefix])
        hits = frame.select(changed.arg_true()).to_series()
        if hits.len():
            return hits[0]
        if frame.height < n:
            return frame.height
        n *= 2


def _interleave(sources, prefix, cal, row_group_size):
    """Yield the interleaved output of ``sources`` (in calendar order), group chunk by
    group chunk. Raises _LayoutError before yielding any chunk that doesn't fit, with every
    source still positioned at the start of that chunk."""
    blocks = [_block_length(source, prefix) for source in sources]
    groups = {source.rows // block for source, block in zip(sources, blocks)}
    if len(groups) != 1 or any(source.rows % block for source, block in zip(sources, blocks)):
        raise _LayoutError
    remaining = groups.pop()
    chunk_groups = max(1, row_group_size // sum(blocks))
    previous = None  # prefix of the last group already written

    while remaining:
        g = min(chunk_groups, remaining)
        tables, starts = [], None
        for source, block in zip(sources, blocks):
            # One row past the chunk shows whether the last block really ends there.
            table = source.peek(g * block + 1)
            frame = _key_frame(table, [*prefix, cal])
            if frame.height < g * block:
                raise _LayoutError
            block_starts = frame.select(prefix).gather(np.arange(g) * block)
            # Every row of a block has its block's prefix...
            expanded = block_starts.gather(np.repeat(np.arange(g), block))
            if not frame.head(g * block).select(prefix).equals(expanded):
                raise _LayoutError
            # ...and the row after the chunk, if any, starts a different group.
            if frame.height > g * block and frame.select(prefix).row(g * block) == block_starts.row(g - 1):
                raise _LayoutError
            if starts is None:
                starts = block_starts
                # Consecutive groups differ, including across chunks.
                with_previous = pl.concat([previous, starts]) if previous is not None else starts
                distinct = with_previous.select(
                    pl.any_horizontal([pl.col(col).ne_missing(pl.col(col).shift()) for col in prefix])
                ).to_series()
                if not distinct.slice(1).all():
                    raise _LayoutError
            elif not block_starts.equals(starts):  # every file holds the same groups
                raise _LayoutError
            tables.append((table.slice(0, g * block), frame[cal]))

        # Calendar ranges ascend from file to file within every group.
        for (_, cal_a), (_, cal_b), block_a, block_b in zip(tables, tables[1:], blocks, blocks[1:]):
            last_a = cal_a.gather(np.arange(g) * block_a + block_a - 1)
            first_b = cal_b.gather(np.arange(g) * block_b)
            if not (last_a < first_b).all():
                raise _LayoutError

        offsets = np.cumsum([0] + [g * block for block in blocks[:-1]])
        index = np.hstack(
            [
                offset + np.arange(g)[:, None] * block + np.arange(block)[None, :]
                for offset, block in zip(offsets, blocks)
            ]
        ).ravel()
        chunk = pa.concat_tables([table for table, _ in tables]).take(index)
        del tables
        for source, block in zip(sources, blocks):
            source.drop(g * block)
        previous = starts.tail(1)
        remaining -= g
        yield chunk


def _window_merge(sources, key):
    """Yield the k-way merge of sorted ``sources`` from wherever each currently stands."""
    while True:
        live = []
        for source in sources:
            table = source.peek()
            if table is not None:
                live.append((source, table))
        if not live:
            return
        if len(live) == 1:
            yield from live[0][0].drain()
            return

        frames = [
            _key_frame(table, key).with_columns(pl.lit(i).alias("__s"), pl.int_range(pl.len()).alias("__r"))
            for i, (_, table) in enumerate(live)
        ]
        # Every unread row of a source sorts at or after its last buffered row, so all rows
        # up to the smallest such last row are final.
        lasts = pl.concat([frame.tail(1) for frame in frames]).sort(key, maintain_order=True)
        stop_source, stop_row = lasts["__s"][0], lasts["__r"][0]
        window = pl.concat(frames).sort(key, maintain_order=True)
        cut = window.select(((pl.col("__s") == stop_source) & (pl.col("__r") == stop_row)).arg_true()).item()
        emitted = window.head(cut + 1)

        offsets = np.cumsum([0] + [table.num_rows for _, table in live[:-1]])
        index = offsets[emitted["__s"].to_numpy()] + emitted["__r"].to_numpy()
        yield pa.concat_tables([table for _, table in live]).take(index)

        counts = dict(emitted.group_by("__s").len().iter_rows())
        for i, (source, _) in enumerate(live):
            if counts.get(i):
                source.drop(counts[i])


def _merge(pieces, key, schema, batch_size, row_group_size, stats):
    """Yield the merged output of overlapping ``pieces`` (at most max_open_files of them)."""
    with contextlib.ExitStack() as stack:
        sources = []
        for piece in pieces:
            source = _Source(piece, schema, batch_size)
            stack.callback(source.close)
            sources.append(source)
        try:
            yield from _interleave(sources, key[:-1], key[-1], row_group_size)
            stats["interleaved"] += 1
            return
        except _LayoutError:
            stats["merged"] += 1
        yield from _window_merge(sources, key)


def _merge_cluster(pieces, key, schema, options, stats, temp_root):
    """Yield the merged output of one cluster, in rounds when it has too many pieces."""
    max_open = options["max_open_files"]
    while len(pieces) > max_open:
        stats["rounds"] += 1
        # Consecutive pieces (in key order) stay together, so a run's years remain in
        # order and the next round can still interleave.
        next_round = []
        for start in range(0, len(pieces), max_open):
            group = pieces[start : start + max_open]
            if len(group) == 1:
                next_round.append(group[0])
                continue
            temp = temp_root() / f"{uuid.uuid4()}.parquet"
            writer = _Writer(temp, schema, options["row_group_size"], options["write_options"])
            try:
                for chunk in _merge(group, key, schema, options["batch_size"], options["row_group_size"], stats):
                    writer.write(chunk)
            finally:
                writer.close()
            next_round.append(_Piece(temp, 0, writer.rows))
        # Intermediate files of the previous round are no longer needed.
        for piece in pieces:
            if piece.path.parent == temp_root() and piece not in next_round:
                piece.path.unlink()
        pieces = next_round
    yield from _merge(pieces, key, schema, options["batch_size"], options["row_group_size"], stats)


def merge_sorted_files(
    paths,
    out_path,
    key,
    *,
    run_cols=(),
    sort=True,
    max_open_files=MAX_OPEN_FILES,
    batch_size=BATCH_SIZE,
    row_group_size,
    write_options,
    temp_dir,
):
    """Write ``paths`` (each sorted by ``key``) to ``out_path`` as one file sorted by ``key``.

    ``key`` ends with the calendar column, and starts with ``run_cols``: the columns
    identifying a run, whose rows are only ever merged with the same run's rows in other
    files. With ``sort=False`` the files are just concatenated in the order given. ``write_options`` go to ``pq.ParquetWriter``.
    Temporary files for multi-round merges go in a directory created under ``temp_dir``
    only when needed.

    Raises ``ValidationError`` -- after removing anything it wrote -- if the inputs' columns
    differ or the output's row count doesn't match the inputs'. Returns a dict of what it
    did: rows written and how many clusters were copied, interleaved or merged.
    """
    if max_open_files < 2:
        msg = f"max_open_files must be at least 2, got {max_open_files}"
        raise ValueError(msg)
    paths = [Path(path) for path in paths]
    schema = common_schema(paths)
    missing = [col for col in key if col not in schema.names]
    if missing:
        msg = f"cannot merge {paths[0].parent}: sort key columns missing from the files: {missing}"
        raise ValidationError(msg)
    run_cols = list(run_cols)
    if key[: len(run_cols)] != run_cols:
        msg = f"run_cols {run_cols} must lead the sort key {key}"
        raise ValueError(msg)
    counts = [pq.read_metadata(path).num_rows for path in paths]
    expected = sum(counts)
    nonempty = [path for path, count in zip(paths, counts) if count]

    options = {
        "max_open_files": max_open_files,
        "batch_size": batch_size,
        "row_group_size": row_group_size,
        "write_options": write_options,
    }
    stats = {"rows": 0, "copied": 0, "interleaved": 0, "merged": 0, "rounds": 0}

    with contextlib.ExitStack() as stack:
        temp = []

        def temp_root():
            if not temp:
                temp.append(Path(stack.enter_context(tempfile.TemporaryDirectory(dir=temp_dir, prefix=".compact-"))))
            return temp[0]

        writer = _Writer(out_path, schema, row_group_size, write_options)
        try:
            if sort:
                runs = {}
                for path in nonempty:
                    for run, piece in _run_pieces(path, run_cols):
                        runs.setdefault(run, []).append(piece)
                clusters = [cluster for run in sorted(runs) for cluster in _clusters(runs[run], key, schema)]
            else:
                clusters = [[_Piece.whole(path)] for path in nonempty]
            for cluster in clusters:
                if len(cluster) == 1:
                    with contextlib.closing(_Source(cluster[0], schema, batch_size)) as source:
                        for table in source.drain():
                            writer.write(table)
                    stats["copied"] += 1
                else:
                    for chunk in _merge_cluster(cluster, key, schema, options, stats, temp_root):
                        writer.write(chunk)
        except BaseException:
            writer.close()
            Path(out_path).unlink(missing_ok=True)
            raise
        writer.close()

    stats["rows"] = writer.rows
    if writer.rows != expected:
        Path(out_path).unlink(missing_ok=True)
        msg = f"merging {len(paths)} file(s) wrote {writer.rows:,} rows, expected {expected:,}; nothing replaced"
        raise ValidationError(msg)
    return stats
