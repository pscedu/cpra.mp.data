"""Tests for the public read_data / write_data / compact_data dispatch."""

import numpy as np
import polars as pl
import pytest

from cpra.mp.data import compact_data, read_data, write_data
from cpra.mp.data.exceptions import ValidationError
from tests.factories import (
    ANNUAL_KWARGS,
    RASTER_KWARGS,
    annual_frame,
    raster_array,
    write_raster_dataset,
)

ANNUAL_DATASET = {k: ANNUAL_KWARGS[k] for k in ("variable", "grid", "time_unit")}
RASTER_DATASET = {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}


class TestWriteData:
    def test_vector_grid_writes_parquet(self, base_path):
        write_data(annual_frame(), base_path=base_path, **ANNUAL_DATASET)
        assert list(base_path.rglob("*.parquet"))
        assert not list(base_path.rglob("*.tif"))

    def test_raster_grid_writes_a_cog(self, base_path):
        write_data(raster_array(fill=1), base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        assert list(base_path.rglob("*.tif"))
        assert not list(base_path.rglob("*.parquet"))

    def test_grid_is_read_from_the_data_frame(self, base_path):
        """write_data can infer the grid from a column instead of a keyword argument."""
        write_data(annual_frame(), base_path=base_path)
        assert list(base_path.rglob("*.parquet"))

    def test_missing_grid_raises(self, base_path):
        with pytest.raises(ValidationError, match="Grid is required"):
            write_data(pl.DataFrame({"variable": ["ssf"]}), base_path=base_path)

    def test_unknown_grid_raises(self, base_path):
        with pytest.raises(ValidationError, match="not a valid grid"):
            write_data(np.zeros(4), base_path=base_path, grid="nope_v001")


class TestReadData:
    def test_vector_grid_returns_a_lazy_frame(self, base_path):
        write_data(annual_frame(), base_path=base_path, **ANNUAL_DATASET)
        lf = read_data(base_path=base_path, **ANNUAL_DATASET)
        assert isinstance(lf, pl.LazyFrame)
        # Long format repeats each calendar_year once per grid_cell, so compare the
        # distinct years present rather than the raw (repeated) column values.
        assert sorted(lf.collect()["calendar_year"].unique().to_list()) == [2030, 2040]

    def test_raster_grid_returns_a_path_table(self, base_path):
        """For raster grids, read_data yields COG metadata rather than pixels."""
        write_raster_dataset(base_path, years=(2030, 2040))
        df = read_data(base_path=base_path, **RASTER_DATASET).collect()
        assert "path" in df.columns
        assert df["calendar_year"].to_list() == [2030, 2040]

    def test_missing_grid_raises(self, base_path):
        with pytest.raises(ValidationError, match="Grid is required"):
            read_data(base_path=base_path, variable="ssf")

    def test_unknown_grid_raises(self, base_path):
        with pytest.raises(ValidationError, match="not a valid grid"):
            read_data(base_path=base_path, variable="ssf", grid="nope_v001", time_unit="annual")


class TestReadDataSelection:
    """`read_data`'s `grid_cell`/`where`/`sort`/`select` selection parameters (Phase 5).

    The shape decision these tests pin down: with no selection arguments (or dimension
    filters only), `read_data` returns EXACTLY what it always has (see `TestReadData`
    above) -- a vector grid's long-format `grid_cell`/`value` `LazyFrame`, unchanged
    either way, since storage/read is already long-format (there is no wide-to-long shape
    change to reconcile here). `grid_cell`/`where`/`sort`/`select` apply IN ADDITION to
    that same long-format frame -- they narrow/reorder/reshape it, but never change its
    fundamental column vocabulary (`grid_cell`/`value`/dimension columns). For a raster
    grid, `read_data` always returns the band/path metadata table (never pixels): `where`'s
    dimension conjuncts and `sort`/`select` apply to that table's own columns (including
    `path`), but `grid_cell` and a `where` value conjunct are rejected outright, since
    both require actual pixel data `read_data` never touches -- `analyze_data` is where
    those belong.
    """

    def test_where_dimension_conjunct_filters_rows(self, base_path):
        write_data(annual_frame(years=(2030, 2040)), base_path=base_path, **ANNUAL_DATASET)
        lf = read_data(base_path=base_path, where="calendar_year > 2035", **ANNUAL_DATASET)
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert df["calendar_year"].unique().to_list() == [2040]
        assert df.height == 32

    def test_where_value_conjunct_filters_rows(self, base_path):
        write_data(annual_frame(years=(2030,)), base_path=base_path, **ANNUAL_DATASET)
        df = read_data(base_path=base_path, where="value > 2030.02", **ANNUAL_DATASET).collect()
        # value = calendar_year + grid_cell / 100, so only grid_cell 3.. survive 2030.02.
        assert df["grid_cell"].min() > 2

    def test_where_value_conjunct_rejected_for_raster(self, base_path):
        write_raster_dataset(base_path, years=(2030,))
        with pytest.raises(ValidationError, match="value"):
            read_data(base_path=base_path, where="value > 0", **RASTER_DATASET)

    def test_grid_cell_by_id_filters_rows(self, base_path):
        write_data(annual_frame(years=(2030,)), base_path=base_path, **ANNUAL_DATASET)
        df = read_data(base_path=base_path, grid_cell="1,2", **ANNUAL_DATASET).collect()
        assert sorted(df["grid_cell"].unique().to_list()) == [1, 2]

    def test_grid_cell_rejected_for_raster(self, base_path):
        write_raster_dataset(base_path, years=(2030,))
        with pytest.raises(ValidationError, match="analyze_data"):
            read_data(base_path=base_path, grid_cell="1,2", **RASTER_DATASET)

    def test_sort_descending(self, base_path):
        write_data(annual_frame(years=(2030, 2040)), base_path=base_path, **ANNUAL_DATASET)
        df = read_data(base_path=base_path, sort="-calendar_year", **ANNUAL_DATASET).collect()
        assert df["calendar_year"].unique(maintain_order=True).to_list() == [2040, 2030]

    def test_invalid_sort_raises(self, base_path):
        write_data(annual_frame(years=(2030,)), base_path=base_path, **ANNUAL_DATASET)
        with pytest.raises(ValidationError, match="sort"):
            read_data(base_path=base_path, sort="nonsense", **ANNUAL_DATASET)

    def test_select_narrows_columns(self, base_path):
        write_data(annual_frame(years=(2030,)), base_path=base_path, **ANNUAL_DATASET)
        df = read_data(base_path=base_path, select="calendar_year", **ANNUAL_DATASET).collect()
        assert df.columns == ["calendar_year"]

    def test_select_includes_raster_metadata_columns(self, base_path):
        """Unlike the analysis engine's own `select` (restricted to dimension columns plus
        `grid_cell`/`value`), `read_data`'s raster return is the band/path metadata table
        itself, so `path`/`band` are real, selectable columns here."""
        write_raster_dataset(base_path, years=(2030,))
        df = read_data(base_path=base_path, select="path,calendar_year", **RASTER_DATASET).collect()
        assert df.columns == ["path", "calendar_year"]

    def test_combined_selection_parameters(self, base_path):
        write_data(annual_frame(years=(2030, 2040)), base_path=base_path, **ANNUAL_DATASET)
        df = read_data(
            base_path=base_path,
            grid_cell="1,2",
            where="calendar_year > 2035",
            sort="-grid_cell",
            select="grid_cell,value",
            **ANNUAL_DATASET,
        ).collect()
        assert df.columns == ["grid_cell", "value"]
        assert df["grid_cell"].to_list() == [2, 1]

    def test_unknown_selection_keyword_raises(self, base_path):
        with pytest.raises(ValidationError, match="bogus"):
            read_data(base_path=base_path, bogus="x", **ANNUAL_DATASET)

    def test_no_selection_args_is_unchanged(self, base_path):
        """The exact case `TestReadData.test_vector_grid_returns_a_lazy_frame` already
        pins down -- restated here to make the shape decision explicit: calling with a
        dimension filter but none of `grid_cell`/`where`/`sort`/`select` returns the exact
        same object `read_cog`/`read_parquet` would, with no extra processing applied."""
        write_data(annual_frame(years=(2030,), scenario_id=7), base_path=base_path, **ANNUAL_DATASET)
        direct = read_data(base_path=base_path, scenario_id=7, **ANNUAL_DATASET).collect()
        assert "grid_cell" in direct.columns and "value" in direct.columns


class TestCompactData:
    def test_raster_grid_dispatches_to_compact_cog(self, base_path):
        """compact_data's raster branch delegates to compact_cog's summary dict.

        2030 and 2040 aren't a contiguous calendar run, so compact_cog has nothing to
        merge here -- the point of this test is the dispatch (raster grids reach
        compact_cog, not compact_parquet), not the merge logic itself.
        """
        write_raster_dataset(base_path, years=(2030, 2040))
        result = compact_data(base_path=base_path, **RASTER_DATASET)
        assert result == {"outputs": [], "sources_merged": 0}
        assert len(list(base_path.rglob("*.tif"))) == 2

    def test_unknown_grid_raises(self, base_path):
        with pytest.raises(ValidationError, match="not a valid grid"):
            compact_data(base_path=base_path, grid="nope_v001")

    def test_vector_grid_dispatches_to_compact_parquet(self, base_path):
        write_data(annual_frame(years=(2030,)), base_path=base_path, **ANNUAL_DATASET)
        write_data(annual_frame(years=(2040,)), base_path=base_path, **ANNUAL_DATASET)

        result = compact_data(base_path=base_path, **ANNUAL_DATASET)

        assert result["sources_merged"] == 2
        assert len(list(base_path.rglob("*.parquet"))) == 1

    def test_compact_data_returns_the_summary(self, base_path):
        """The summary is passed through from compact_parquet unchanged."""
        write_data(annual_frame(years=(2030,)), base_path=base_path, **ANNUAL_DATASET)
        write_data(annual_frame(years=(2040,)), base_path=base_path, **ANNUAL_DATASET)

        result = compact_data(base_path=base_path, **ANNUAL_DATASET)

        assert set(result) == {"outputs", "sources_merged"}
        assert result["outputs"] == list(base_path.rglob("*.parquet"))


class TestRoundTrip:
    def test_vector_values_survive(self, base_path):
        original = annual_frame(years=(2030, 2040, 2050))
        write_data(original, base_path=base_path, **ANNUAL_DATASET)
        result = read_data(base_path=base_path, **ANNUAL_DATASET).collect()

        # Long format stores one grid_cell/value pair per row instead of one column per
        # cell, so pull each cell's time series back out by filtering on grid_cell.
        for col in ("1", "16", "32"):
            values = (
                result.filter(pl.col("grid_cell") == int(col)).sort("calendar_year")["value"].to_list()
            )
            assert values == pytest.approx(original[col].to_list())

    def test_raster_values_survive(self, base_path):
        import rasterio as rio

        data = raster_array()
        write_data(data, base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        df = read_data(base_path=base_path, **RASTER_DATASET).collect()

        with rio.open(df["path"][0]) as src:
            assert np.array_equal(src.read(1), data)
