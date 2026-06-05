import datetime

import polars as pl
import pytest

from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.tabular import read_parquet, write_parquet
from cpra.mp.data.utils import validate_grids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _daily_df(
    geo_ids=(1, 2, 3),
    days=(datetime.date(2030, 1, 1), datetime.date(2030, 1, 2)),
    grid="hydro_compartment_v001",
    variable="sal",
    model_group_id=500,
    scenario_id=7,
):
    """Minimal long-format daily DataFrame without a calendar_year column."""
    rows = [
        {
            "variable": variable,
            "grid": grid,
            "time_unit": "daily",
            "model_group_id": model_group_id,
            "scenario_id": scenario_id,
            "aggregate_type": "2_wk_max",
            "model": "icm_v2023.0.0",
            "calendar_day": day,
            "geography_id": geo_id,
            "value": float(geo_id),
        }
        for geo_id in geo_ids
        for day in days
    ]
    return pl.DataFrame(rows).with_columns(
        pl.col("model_group_id").cast(pl.Int32),
        pl.col("scenario_id").cast(pl.Int32),
        pl.col("geography_id").cast(pl.Int32),
        pl.col("value").cast(pl.Float32),
    )


# ---------------------------------------------------------------------------
# validate_grids
# ---------------------------------------------------------------------------

class TestValidateGrids:
    def _df(self, grid, geo_ids):
        return pl.DataFrame({
            "grid": [grid] * len(geo_ids),
            "geography_id": pl.Series(geo_ids, dtype=pl.Int32),
        })

    def test_valid_single_grid(self):
        df = self._df("hydro_compartment_v001", [1, 100, 1657])
        result = validate_grids(df)
        assert "hydro_compartment_v001" in result

    def test_valid_no_bounds_grid(self):
        # block_group_v2020 has id_min=None, id_max=None — any ID is accepted
        df = self._df("block_group_v2020", [99999, 200000])
        result = validate_grids(df)
        assert "block_group_v2020" in result

    def test_valid_multi_grid(self):
        df = pl.concat([
            self._df("hydro_compartment_v001", [1, 2]),
            self._df("ecoregion_v001", [1, 32]),
        ])
        result = validate_grids(df)
        assert set(result) == {"hydro_compartment_v001", "ecoregion_v001"}

    def test_invalid_grid_name(self):
        df = self._df("not_a_real_grid_v001", [1])
        with pytest.raises(ValidationError, match="not a valid grid"):
            validate_grids(df)

    def test_geography_id_below_min(self):
        # hydro_compartment_v001 id_min=1
        df = self._df("hydro_compartment_v001", [0])
        with pytest.raises(ValidationError, match="below minimum"):
            validate_grids(df)

    def test_geography_id_above_max(self):
        # hydro_compartment_v001 id_max=1657
        df = self._df("hydro_compartment_v001", [1658])
        with pytest.raises(ValidationError, match="above maximum"):
            validate_grids(df)


# ---------------------------------------------------------------------------
# write_parquet
# ---------------------------------------------------------------------------

class TestWriteParquet:
    def test_long_format_accepted(self, tmp_path):
        """Long-format DataFrame is written without error; no pivot is performed."""
        df = _daily_df()
        write_parquet(df, base_path=tmp_path, validate=False)
        assert any(tmp_path.rglob("*.parquet"))

    def test_calendar_year_derived_from_calendar_day(self, tmp_path):
        """calendar_year is auto-derived from calendar_day when absent."""
        df = _daily_df(days=[datetime.date(2030, 6, 15)])
        assert "calendar_year" not in df.columns
        write_parquet(df, base_path=tmp_path, validate=False)
        year_dirs = [p for p in tmp_path.rglob("*") if p.is_dir() and "calendar_year=2030" in p.name]
        assert year_dirs, "Expected a calendar_year=2030 partition directory"

    def test_multi_year_creates_separate_directories(self, tmp_path):
        """Rows spanning two calendar years land in separate calendar_year= directories."""
        df = _daily_df(days=[datetime.date(2030, 12, 31), datetime.date(2031, 1, 1)])
        write_parquet(df, base_path=tmp_path, validate=False)
        year_dirs = {p.name for p in tmp_path.rglob("*") if p.is_dir() and "calendar_year=" in p.name}
        assert year_dirs == {"calendar_year=2030", "calendar_year=2031"}

    def test_directory_structure(self, tmp_path):
        """grid= must not appear as a path segment; calendar_year= must."""
        df = _daily_df()
        write_parquet(df, base_path=tmp_path, validate=False)
        all_paths = [str(p) for p in tmp_path.rglob("*.parquet")]
        assert all_paths, "No parquet files written"
        for path in all_paths:
            assert "grid=" not in path, f"Unexpected grid= in path: {path}"
            assert "calendar_year=" in path, f"Missing calendar_year= in path: {path}"

    def test_multi_grid_write(self, tmp_path):
        """DataFrames containing two different grids are written without error."""
        df = pl.concat([
            _daily_df(geo_ids=[1, 2], grid="hydro_compartment_v001"),
            _daily_df(geo_ids=[1, 2], grid="hydro_compartment_v002"),
        ])
        write_parquet(df, base_path=tmp_path, validate=False)
        assert any(tmp_path.rglob("*.parquet"))

    def test_missing_geography_id_raises(self, tmp_path):
        df = _daily_df().drop("geography_id")
        with pytest.raises(ValidationError, match="geography_id"):
            write_parquet(df, base_path=tmp_path, validate=True, sort=False)

    def test_missing_value_raises(self, tmp_path):
        df = _daily_df().drop("value")
        with pytest.raises(ValidationError, match="value"):
            write_parquet(df, base_path=tmp_path, validate=True, sort=False)

    def test_sort_order(self, tmp_path):
        """Rows are stored in (geography_id, calendar_day) order within a year file."""
        df = _daily_df(
            geo_ids=[3, 1, 2],
            days=[datetime.date(2030, 1, 2), datetime.date(2030, 1, 1)],
        )
        write_parquet(df, base_path=tmp_path, validate=False)
        files = list(tmp_path.rglob("*.parquet"))
        # All rows are from 2030 so there is exactly one year partition file.
        assert len(files) == 1
        result = pl.read_parquet(files[0])
        assert result["geography_id"].to_list() == [1, 1, 2, 2, 3, 3]
        assert result["calendar_day"].to_list() == [
            datetime.date(2030, 1, 1), datetime.date(2030, 1, 2),
            datetime.date(2030, 1, 1), datetime.date(2030, 1, 2),
            datetime.date(2030, 1, 1), datetime.date(2030, 1, 2),
        ]


# ---------------------------------------------------------------------------
# read_parquet
# ---------------------------------------------------------------------------

class TestReadParquet:
    def test_calendar_day_prunes_by_calendar_year(self, tmp_path):
        """A calendar_day filter adds a calendar_year predicate for partition pruning."""
        df_2030 = _daily_df(days=[datetime.date(2030, 6, 1)])
        df_2031 = _daily_df(days=[datetime.date(2031, 6, 1)])
        write_parquet(pl.concat([df_2030, df_2031]), base_path=tmp_path, validate=False)

        result = read_parquet(
            base_path=tmp_path,
            variable="sal",
            time_unit="daily",
            calendar_day=datetime.date(2030, 6, 1),
        ).collect()

        assert result.height > 0
        assert (result["calendar_year"] == 2030).all()
        assert not (result["calendar_year"] == 2031).any()
