from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from cpra.mp.data.tabular import delete_parquet, write_parquet
from cpra.mp.data.utils import cast_columns, get_input_data_frame


class TestDeleteParquet:
    @pytest.fixture
    def data(self, tmp_path):
        """Write test parquet files."""

        values = np.random.uniform(0, 5.0, 1657).astype(np.float32)
        values_df = pl.DataFrame({str(i): [v] for i, v in enumerate(values, start=1)})
        df = cast_columns(
            get_input_data_frame(
                values_df,
                variable=["sal"],
                grid=["hydro_compartment_v001"],
                time_unit=["annual"],
                model_group_id=[100, 101, 102, 103, 104, 105],
                scenario_id=[1, 2, 3, 4],
                aggregate_type=["mean"],
                calendar_year=[2000, 2001, 2002, 2003, 2004, 2005],
                model=["icm_v2023.0.0"],
            )
        )

        a_df = df.filter(pl.col("model_group_id").le(102))
        b_df = df.filter(pl.col("model_group_id").gt(102))

        write_parquet(data=a_df, base_path=tmp_path)
        a_path = next(tmp_path.rglob("*.parquet"))

        write_parquet(data=b_df, base_path=tmp_path)
        b_path = next(path for path in tmp_path.rglob("*.parquet") if path != a_path)

        return SimpleNamespace(base_path=tmp_path, a_df=a_df, a_path=a_path, b_df=b_df, b_path=b_path)

    def test_data_written(self, data):
        assert data.a_path.exists()
        assert data.b_path.exists()

    def test_delete_all_data_removes_file(self, data):
        delete_parquet(data.a_df, base_path=data.base_path)
        assert not data.a_path.exists()
        assert data.b_path.exists()

    def test_delete_some_preserves_remaining(self, data):
        delete_df = data.a_df.filter(model_group_id=101)
        delete_parquet(delete_df, base_path=data.base_path)
        assert not data.a_path.exists()
        assert data.b_path.exists()

        paths = list(data.base_path.rglob("*.parquet"))
        assert len(paths) == 2
        c_path = next(path for path in paths if path != data.b_path)
