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
