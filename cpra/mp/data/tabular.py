import itertools
import queue
import tempfile
import threading
import uuid
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from cpra.mp.data import staging
from cpra.mp.data.catalog import CAL_DIMS, EXCLUDE, Catalog, tabular_catalog_rows
from cpra.mp.data.config import (
    CATALOG_PATH_UNSET,
    get_base_path,
    get_write_catalog_path,
)
from cpra.mp.data.dimensions import DATASET_DIMENSIONS, DIMENSIONS
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.merge import MAX_OPEN_FILES, common_schema, merge_sorted_files
from cpra.mp.data.utils import (
    build_glob_pattern,
    dataset_path,
    extract_dimension_value,
    get_filter_expressions,
    get_input_data_frame,
    long_sort_expressions,
    partition_columns,
    required_columns,
    validate_data_frame,
    validate_grid,
)
from cpra.mp.data.variables import VARIABLES

_DEFAULT_ROW_GROUP_SIZE = 500_000

# Sentinel distinguishing "temp_dir not passed" (stream via sink_batches) from
# "temp_dir passed, possibly None" (use the bounded temp-file bridge).
_TEMP_DIR_UNSET = object()

# Columns sorted ahead of / alongside the calendar tail that are monotonic
# within each group, so DELTA_BINARY_PACKED encodes them far more compactly than
# the default dictionary/RLE.  grid_cell leads the sort; the calendar columns
# are consecutive within each grid cell.
_DELTA_COLUMNS = ("grid_cell", "calendar_day", "calendar_year")

# The dims identifying one model run, which lead the sort key.
_RUN_COLUMNS = ("model_group_id", "scenario_id")


def _row_group_size(grid, time_unit):
    """Return the default row group size.

    Data is sorted grid_cell-first, so each row group covers a contiguous range
    of grid cells and predicate pushdown can skip row groups for
    grid_cell-filtered queries.  A fixed, moderate size keeps that skipping
    fine-grained without producing excessively many groups.
    """
    return _DEFAULT_ROW_GROUP_SIZE


def read_parquet(base_path=None, cache=True, include_file_paths=None, **kwargs):
    """Read tabular data into a data frame from a Parquet file."""

    base_path = get_base_path(base_path)

    # Validate the input keyword arguments.
    input_df = get_input_data_frame(**kwargs)
    validate_data_frame(input_df, required_cols=DATASET_DIMENSIONS.keys())

    # Calculate the dataset directory path and expressions.
    dataset_dir = dataset_path(base_path, **kwargs)
    dataset_exprs = [pl.lit(kwargs[dim.name]).cast(dim.dtype).alias(dim.name) for dim in DATASET_DIMENSIONS.values()]
    filter_args = get_filter_expressions(**kwargs)

    # Default to an un-partitioned dataset.
    hive_partitioning = False
    hive_schema = None

    # Provide Hive partitioning schema since this is known in advance.
    partition_cols = partition_columns(input_df, include_dataset=False)
    if partition_cols:
        hive_partitioning = True
        hive_schema = {dim.name: dim.dtype for dim in DIMENSIONS.values() if dim.name in partition_cols}

    lf = pl.scan_parquet(
        dataset_dir,
        cache=cache,
        hive_partitioning=hive_partitioning,
        hive_schema=hive_schema,
        include_file_paths=include_file_paths,
    )

    # Filter based on keyword arguments.
    if filter_args:
        lf = lf.filter(*filter_args)

    # Sort columns and add dataset values.
    required_cols = required_columns(input_df)
    lf = lf.select(
        *dataset_exprs,
        *[pl.col(col) for col in required_cols if col not in DATASET_DIMENSIONS],
        pl.col("grid_cell"),
        pl.col("value"),
        *([pl.col(include_file_paths)] if include_file_paths else []),
    )

    return lf


def write_parquet(
    data,
    base_path=None,
    catalog_path=CATALOG_PATH_UNSET,
    validate=True,
    sort=True,
    compression="zstd",
    compression_level=6,
    row_group_size=None,
    temp_dir=_TEMP_DIR_UNSET,
    **kwargs,
):
    """Write data from a data frame to a Parquet file.

    Accepts a Polars DataFrame, LazyFrame, or any object that can be passed to
    pl.DataFrame().  DataFrames are wrapped with .lazy() before writing.  A
    single metadata row is collected upfront to drive grid/sort/partition
    decisions.

    The sorted data is streamed through PyArrow so the grid_cell and calendar
    key columns can be DELTA_BINARY_PACKED encoded (Polars' ``sink_parquet``
    cannot set per-column encodings).  The bridge between Polars and PyArrow
    depends on ``temp_dir``:

    * not passed (default): the data is streamed directly via
      ``LazyFrame.sink_batches`` with no intermediate file.  This is faster but
      peak memory grows with dataset size, so it suits the production system
      (ample RAM, slow storage).
    * passed (a directory path, or ``None`` for the system default): the data is
      streamed to a temporary Parquet file with Polars and read back as an
      ordered batch stream.  Peak memory is bounded and independent of dataset
      size, at the cost of an extra write+read pass — used by the migration to
      stay within a memory limit.  Pass a path on the same fast/large filesystem
      as the output for very large datasets.
    """

    base_path = get_base_path(base_path)
    catalog_path = get_write_catalog_path(catalog_path)
    lf = data if isinstance(data, pl.LazyFrame) else pl.DataFrame(data).lazy()

    # Add keyword arguments as literal columns.
    lf = lf.with_columns(**{k: pl.lit(v) for k, v in kwargs.items()})

    # If this is wide-format data (grid_cell IDs as column names), unpivot to long.
    _cols = lf.collect_schema().names()
    numeric_cols = [col for col in _cols if col[:1].isdigit()]
    if numeric_cols:
        alpha_cols = [col for col in _cols if not col[:1].isdigit()]
        lf = lf.unpivot(index=alpha_cols, variable_name="grid_cell", value_name="value").with_columns(
            pl.col("grid_cell").cast(pl.Int32)
        )

    # Collect one metadata row to drive decisions that need concrete values.
    meta = lf.head(1).collect()

    grid = validate_grid(meta, **kwargs)
    time_unit = extract_dimension_value("time_unit", meta, **kwargs)
    required_cols = required_columns(meta)

    # Validate and sort the data.
    if validate or sort:
        if validate:
            validate_data_frame(lf, required_cols=required_cols)

            # Reject columns that are neither a required dimension nor grid_cell/value.
            # Without this an unrecognized column -- a typo, most likely -- is silently
            # dropped by the sort's select() below when sort=True, and silently written
            # into the output file when sort=False. Behaviour that depends on `sort` is
            # not a useful contract, and the wide-format writer rejected these outright.
            known_cols = {*required_cols, "grid_cell", "value"}
            invalid_cols = [col for col in lf.collect_schema().names() if col not in known_cols]
            if invalid_cols:
                msg = f"Invalid columns: {', '.join(sorted(invalid_cols))}"
                raise ValidationError(msg)

            # Validate grid_cell ID range.
            if grid.id_min is not None or grid.id_max is not None:
                gc_stats = lf.select(
                    pl.col("grid_cell").min().alias("min"),
                    pl.col("grid_cell").max().alias("max"),
                ).collect()
                gc_min, gc_max = gc_stats["min"][0], gc_stats["max"][0]
                if grid.id_min is not None and gc_min < grid.id_min:
                    msg = f"Grid cell ID below minimum for grid {grid.name}: {gc_min}"
                    raise ValidationError(msg)
                if grid.id_max is not None and gc_max > grid.id_max:
                    msg = f"Grid cell ID above maximum for grid {grid.name}: {gc_max}"
                    raise ValidationError(msg)

            # Partial writes are not allowed: every grid cell ID must be present, and
            # each must appear the same number of times (once per implied time step) --
            # otherwise some grid cells would be missing time steps that others have.
            if grid.id_count is not None:
                counts = lf.group_by("grid_cell").agg(pl.len().alias("n")).collect()
                if counts.height != grid.id_count:
                    msg = (
                        f"Grid cell IDs incomplete for grid {grid.name}: "
                        f"expected {grid.id_count} unique IDs, found {counts.height}"
                    )
                    raise ValidationError(msg)
                if counts["n"].n_unique() != 1:
                    msg = f"Grid cell IDs have inconsistent row counts for grid {grid.name}"
                    raise ValidationError(msg)

            # Validate value column dtype.
            variable = VARIABLES[meta["variable"][0]]
            value_dtype = lf.collect_schema()["value"]
            if value_dtype != variable.dtype:
                msg = f"Column 'value' has dtype {value_dtype!s}, expected {variable.dtype!s}"
                raise ValidationError(msg)

        if sort:
            # The select below needs every required column, so check for them eagerly
            # even when validate=False. Otherwise the failure surfaces as a Polars
            # ColumnNotFoundError raised inside a sink producer thread, once the lazy
            # plan is materialized and far from the caller that omitted the column.
            missing_cols = [col for col in required_cols if col not in lf.collect_schema().names()]
            if missing_cols:
                msg = f"Required columns are missing: {', '.join(missing_cols)}"
                raise ValidationError(msg)

            outer_cols = [col for col in required_cols if not col.startswith("calendar_") and col not in kwargs]
            sort_exprs = long_sort_expressions(outer_cols, time_unit)
            lf = lf.select(*required_cols, "grid_cell", "value").sort(*sort_exprs)

    # Compute row group size from grid and time_unit if not provided explicitly.
    if row_group_size is None:
        row_group_size = _row_group_size(grid, time_unit)

    # Derive partition columns from the metadata row.
    partition_cols = partition_columns(meta)

    common = {
        "base_path": Path(base_path),
        "partition_cols": partition_cols,
        "compression": compression,
        "compression_level": compression_level,
        "row_group_size": row_group_size,
    }
    if temp_dir is _TEMP_DIR_UNSET:
        _write_via_sink_batches(lf, **common)
    else:
        _write_via_temp_file(lf, temp_dir=temp_dir, **common)

    # Keep the unified catalog side-table in sync with what was just written.
    if catalog_path:
        group_cols = [col for col in required_cols if col not in EXCLUDE and col not in CAL_DIMS]
        rows = tabular_catalog_rows(lf, group_cols, time_unit)
        Catalog(catalog_path).append(rows)


def _write_options(file_cols, compression, compression_level):
    """Parquet writer options for a file holding ``file_cols``: shared by write_parquet and
    compact_parquet so a compacted file is encoded like a freshly written one."""
    delta_cols = [col for col in _DELTA_COLUMNS if col in file_cols]
    return {
        "compression": compression,
        "compression_level": compression_level,
        "use_dictionary": [col for col in file_cols if col not in delta_cols and col != "value"],
        "column_encoding": {col: "DELTA_BINARY_PACKED" for col in delta_cols},
    }


def _write_dataset(reader, schema, base_path, partition_cols, compression, compression_level, row_group_size):
    """Write an ordered batch stream to a Hive-partitioned, delta-encoded dataset.

    Applies DELTA_BINARY_PACKED to grid_cell and the calendar key columns (which
    Polars cannot set), dictionary encoding to the small dimension columns, and
    leaves the high-cardinality value column as PLAIN.  ``preserve_order`` keeps
    the reader's global sort order in the on-disk layout.
    """
    file_cols = [name for name in schema.names if name not in partition_cols]
    write_options = ds.ParquetFileFormat().make_write_options(
        **_write_options(file_cols, compression, compression_level)
    )
    partitioning = (
        ds.partitioning(pa.schema([schema.field(col) for col in partition_cols]), flavor="hive")
        if partition_cols
        else None
    )
    ds.write_dataset(
        reader,
        base_dir=base_path,
        format="parquet",
        partitioning=partitioning,
        file_options=write_options,
        basename_template=f"{uuid.uuid4()}-{{i}}.parquet",
        existing_data_behavior="overwrite_or_ignore",
        min_rows_per_group=row_group_size,
        max_rows_per_group=row_group_size,
        preserve_order=True,
    )


def _write_via_temp_file(lf, temp_dir, row_group_size, **kwargs):
    """Bounded-memory bridge: stream the sorted data to a temp Parquet file, then
    read it back as an ordered batch stream.  Peak memory is independent of size."""
    with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdir:
        tmp_sorted = Path(tmpdir) / "sorted.parquet"
        lf.sink_parquet(tmp_sorted, row_group_size=row_group_size)
        parquet_file = pq.ParquetFile(tmp_sorted)
        schema = parquet_file.schema_arrow
        reader = pa.RecordBatchReader.from_batches(schema, parquet_file.iter_batches(batch_size=row_group_size))
        _write_dataset(reader, schema, row_group_size=row_group_size, **kwargs)


def _write_via_sink_batches(lf, row_group_size, **kwargs):
    """Direct bridge: stream batches straight from Polars to PyArrow via
    sink_batches with no intermediate file.  Faster, but peak memory grows with
    dataset size, so it is reserved for the memory-rich production system."""
    batch_queue = queue.Queue(maxsize=4)
    done = object()
    stop = threading.Event()

    def _produce():
        def _emit(df):
            if stop.is_set():  # consumer went away (e.g. write error); stop early
                return True
            batch_queue.put(df.to_arrow())
            return None

        try:
            lf.sink_batches(_emit, chunk_size=row_group_size, maintain_order=True, engine="streaming")
        finally:
            batch_queue.put(done)

    producer = threading.Thread(target=_produce, daemon=True)
    producer.start()
    try:
        first = batch_queue.get()

        def _batches(first_table):
            table = first_table
            while table is not done:
                yield from table.to_batches()
                table = batch_queue.get()

        # Empty result: fall back to the declared schema so the reader is valid.
        schema = lf.limit(0).collect().to_arrow().schema if first is done else first.schema
        reader = pa.RecordBatchReader.from_batches(schema, _batches(first))
        _write_dataset(reader, schema, row_group_size=row_group_size, **kwargs)
    finally:
        # Unblock the producer if it is waiting on a full queue, then wait for it.
        stop.set()
        while producer.is_alive():
            try:
                batch_queue.get(timeout=0.1)
            except queue.Empty:
                pass
        producer.join()


def compact_parquet(
    base_path=None,
    catalog_path=CATALOG_PATH_UNSET,
    sort=True,
    compression="zstd",
    compression_level=6,
    row_group_size=None,
    max_open_files=MAX_OPEN_FILES,
    temp_dir=None,
    **kwargs,
):
    """Rewrite Parquet datasets with one file per partition directory.

    Every partition directory holding more than one Parquet file is rewritten as a single
    file, and the originals are then removed. A directory that already holds one file is
    left untouched.

    The files are streamed, never loaded whole: each is already sorted (as
    ``write_parquet`` writes it), so ``merge.merge_sorted_files`` copies, interleaves or
    merges them into the same order a full sort would give, reading at most
    ``max_open_files`` at a time. Peak memory depends on that and ``row_group_size``, not
    on the amount of data; see ``cpra.mp.data.merge``. A cluster of more than
    ``max_open_files`` overlapping files is merged in rounds through temporary files in a
    hidden directory under ``temp_dir`` (default: the staging directory below). With
    ``sort=False`` the files are concatenated as they are.

    The merged file is written in a hidden staging directory under ``base_path`` (see
    ``cpra.mp.data.staging``), and only moved into place, and the originals removed, once
    its row count matches theirs. Staging left by a compaction that was killed part-way is
    removed first, from ``base_path`` and ``temp_dir``.

    Beyond ``variable``/``time_unit``/``grid``, kwargs may only name dims that are
    partition directories for this dataset (``model_group_id``/``scenario_id``/... on
    daily data, nothing on annual/decadal); anything else raises ``ValidationError``
    rather than silently deleting the rows it would filter out.

    Returns ``{"outputs": [Path, ...], "sources_merged": int}`` -- the newly written
    Parquet files and the number of source files they consumed (both empty/zero when
    every partition was already a single file), so callers don't have to re-scan the
    output tree. Mirrors ``raster.compact_cog``.
    """

    base_path = get_base_path(base_path)
    catalog_path = get_write_catalog_path(catalog_path)

    # Validate the input keyword arguments.
    input_df = pl.DataFrame(kwargs)
    validate_data_frame(input_df, required_cols=DATASET_DIMENSIONS)

    # Calculate the dataset directory path.
    dataset_dir = dataset_path(base_path, **kwargs)

    # Construct the glob pattern from the dims actually Hive-partitioned on disk
    # (partition_columns(), not required_columns() -- partition-typed dims like
    # model_group_id/scenario_id/aggregate_type are only real directories when
    # time_unit == "daily", see Dimension.partition; files are UUID-named, never
    # named after a dimension, so the match must be nested with a wildcard).
    partition_cols = partition_columns(input_df, include_dataset=False)

    # Each multi-file directory is read back filtered by kwargs, rewritten, and then
    # every original file in it is deleted. A filter on a dim that is a column rather
    # than a directory (model_group_id on annual data, calendar_year, model, ...) would
    # drop the other rows from the rewrite, and the deletion would then destroy them.
    unsafe = sorted(k for k in kwargs if k not in DATASET_DIMENSIONS and k not in partition_cols)
    if unsafe:
        msg = (
            f"compact_parquet can only narrow by partition directories "
            f"({', '.join(partition_cols) or 'none for this time_unit'}); got {', '.join(unsafe)}. "
            f"Compaction rewrites whole directories, so filtering by a column would delete "
            f"the rows it excludes."
        )
        raise ValidationError(msg)

    glob_pattern = (
        build_glob_pattern(required_cols=partition_cols, extension="parquet", nested=True, **kwargs)
        if partition_cols
        else "*.parquet"
    )
    paths = list(dataset_dir.glob(glob_pattern))

    # Clear out what an interrupted compaction left behind. The sources are only removed
    # once the merged file is in place, so nothing in staging is the only copy of anything.
    staging.remove_stale(base_path)
    if temp_dir is not None:
        staging.remove_stale(temp_dir)

    # Bail out before building the frame below: with no paths its columns would have a
    # Null dtype, which str.extract rejects.
    if not paths:
        return {"outputs": [], "sources_merged": 0}

    # Group by directory, and extract dimension values for directories with
    # more than one file.
    partitions_df = (
        pl.DataFrame(
            {
                "directory": [str(p.parent) for p in paths],
                "file": [str(p.name) for p in paths],
            },
            schema={"directory": pl.String, "file": pl.String},
        )
        .group_by(pl.col("directory"))
        .agg(pl.col("file"))
        .filter(pl.col("file").list.len() > 1)
        .with_columns(
            [
                pl.col("directory").str.extract(f"{dim.name}=([^/]+)(/|$)", 1).cast(dim.dtype).alias(dim.name)
                for dim in DIMENSIONS.values()
                if dim.name in partition_cols
            ]
        )
    )

    # The files' sort key: write_parquet's, minus the dims stored as directories rather
    # than columns (dataset dims, and partition dims on daily data).
    time_unit = kwargs["time_unit"]
    required_cols = required_columns(input_df)
    outer_cols = [col for col in required_cols if not col.startswith("calendar_")]
    key = [
        name
        for name in (expr.meta.output_name() for expr in long_sort_expressions(outer_cols, time_unit))
        if name not in DATASET_DIMENSIONS and name not in partition_cols
    ]
    # A model run's rows are contiguous in every file and never interleave with another
    # run's, so the merge handles each run on its own. On daily data the run is the
    # directory, so the whole key is within one run.
    run_cols = list(itertools.takewhile(lambda col: col in _RUN_COLUMNS, key))
    group_cols = [col for col in required_cols if col not in EXCLUDE and col not in CAL_DIMS]
    dataset_exprs = [pl.lit(kwargs[dim.name]).cast(dim.dtype).alias(dim.name) for dim in DATASET_DIMENSIONS.values()]

    # Merge each directory's files into one.
    outputs = []
    sources_merged = 0
    with staging.area(base_path) as staging_dir:
        for partition in partitions_df.iter_rows(named=True):
            directory = Path(partition["directory"])
            sources = [directory / file for file in sorted(partition["file"])]

            # write_parquet's naming scheme, so the result looks like any other written file.
            final = directory / f"{uuid.uuid4()}-0.parquet"
            staged = staging_dir / final.name
            merge_sorted_files(
                sources,
                staged,
                key,
                run_cols=run_cols,
                sort=sort,
                max_open_files=max_open_files,
                row_group_size=row_group_size or _DEFAULT_ROW_GROUP_SIZE,
                write_options=_write_options(common_schema(sources).names, compression, compression_level),
                temp_dir=temp_dir if temp_dir is not None else staging_dir,
            )
            outputs.append(staging.publish(staged, final))

            # merge_sorted_files has checked the row counts, so the sources can go.
            for source in sources:
                source.unlink()
                sources_merged += 1

            # Catalog rows for the merged file -- the sources' rows plus any that were never
            # cataloged -- computed by a streaming scan.
            if catalog_path:
                partition_exprs = [
                    pl.lit(partition[dim.name]).cast(dim.dtype).alias(dim.name)
                    for dim in DIMENSIONS.values()
                    if dim.name in partition_cols
                ]
                lf = pl.scan_parquet(final, hive_partitioning=False).with_columns(*dataset_exprs, *partition_exprs)
                Catalog(catalog_path).append(tabular_catalog_rows(lf, group_cols, time_unit))

    # Coalesce whatever catalog rows now exist for this variable/time_unit/grid
    # partition (the fresh rows from the write_parquet calls above, plus any older,
    # more-fragmented rows from earlier writes) into minimal contiguous-run rows.
    if catalog_path:
        Catalog(catalog_path).compact_partition(kwargs["variable"], kwargs["time_unit"], kwargs["grid"])

    return {"outputs": outputs, "sources_merged": sources_merged}
