"""Narrow characterization tests for `cpra.mp.data.analysis.io.output`.

Every test here calls the free functions directly with a plain `pl.LazyFrame` and a
hand-rolled `materialize` callable (the same one-argument-method-name contract
`execution.sink`'s own closure implements) -- no compiled plan, no catalog files on
disk. `output.py`'s own module docstring is explicit that
none of this depends on which ops ran or in what order, so a plain frame is enough to
pin its behavior.
"""

import polars as pl
import pytest

from cpra.mp.data.analysis.io.output import (
    SINK_METHODS,
    WRITE_METHODS,
    resolve_legend,
    resolve_output_type,
    stream_tabular,
    tabular_kwargs,
    write_tabular,
)
from cpra.mp.data.analysis.request.errors import RequestError
from cpra.mp.data.analysis.request.types import DataFormat, OutputType
from cpra.mp.data.api.boundary import resolve_status
from cpra.mp.data.legends import LEGENDS, Legend


def _materialize(lf):
    """A minimal stand-in for `execution.sink`'s materialize closure: call a method on `lf` by name,
    defaulting to `collect`, with no error translation -- `output.py`'s functions don't
    depend on the translation, only on the call shape.
    """

    def materialize(method="collect", *args, **kwargs):
        return getattr(lf, method)(*args, **kwargs)

    return materialize


class TestTabularKwargs:
    def test_csv_rounds_to_the_variables_float_precision(self):
        assert tabular_kwargs(DataFormat.csv, 1) == {"float_precision": 1}

    def test_csv_with_no_declared_precision_gets_no_kwargs(self):
        assert tabular_kwargs(DataFormat.csv, None) == {}

    def test_binary_formats_never_get_a_precision_kwarg_even_if_one_is_declared(self):
        assert tabular_kwargs(DataFormat.parquet, 1) == {}
        assert tabular_kwargs(DataFormat.ndjson, 1) == {}


class TestSinkAndWriteMethodTables:
    def test_every_format_has_both_a_sink_and_a_write_spelling(self):
        assert SINK_METHODS.keys() == WRITE_METHODS.keys()

    def test_the_method_names_are_the_real_polars_lazyframe_and_dataframe_methods(self):
        lf = pl.LazyFrame({"value": [1]})
        df = lf.collect()
        for sink_method in SINK_METHODS.values():
            assert hasattr(lf, sink_method)
        for write_method in WRITE_METHODS.values():
            assert hasattr(df, write_method)


class TestStreamAndWriteTabular:
    def test_streamed_and_written_csv_bytes_match(self, tmp_path):
        lf = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        streamed = tmp_path / "streamed.csv"
        written = tmp_path / "written.csv"

        stream_tabular(_materialize(lf), streamed, DataFormat.csv)
        write_tabular(_materialize(lf), written, DataFormat.csv)

        assert streamed.read_bytes() == written.read_bytes()
        assert pl.read_csv(streamed).to_dicts() == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]

    def test_write_tabular_collects_before_writing(self, tmp_path):
        """write_tabular calls `materialize()` with no arguments (defaulting to
        `collect`) and then a write method on the resulting DataFrame -- distinct from
        stream_tabular, which hands the sink method straight to `materialize`."""
        seen = []
        lf = pl.LazyFrame({"a": [1]})

        def materialize(*args, **kwargs):
            seen.append((args, kwargs))
            return getattr(lf, args[0] if args else "collect")(*args[1:], **kwargs)

        out_path = tmp_path / "out.csv"
        write_tabular(materialize, out_path, DataFormat.csv)

        assert seen == [((), {})]
        assert out_path.exists()

    def test_stream_tabular_passes_the_sink_method_name_straight_through(self, tmp_path):
        seen = []
        lf = pl.LazyFrame({"a": [1]})

        def materialize(method, *args, **kwargs):
            seen.append(method)
            return getattr(lf, method)(*args, **kwargs)

        stream_tabular(materialize, tmp_path / "out.parquet", DataFormat.parquet)

        assert seen == ["sink_parquet"]

    def test_extra_args_and_kwargs_pass_through_to_the_writer(self, tmp_path):
        lf = pl.LazyFrame({"a": [1, 2]})
        out_path = tmp_path / "out.csv"

        write_tabular(_materialize(lf), out_path, DataFormat.csv, separator="|")

        assert out_path.read_text().splitlines()[0] == "a"
        assert "|" not in out_path.read_text()  # single column, separator has nothing to separate


class TestResolveLegend:
    def test_a_custom_discrete_map_appends_an_open_ended_upper_bound(self):
        legend = resolve_legend("discrete:#ff0000,1,#00ff00,2,#0000ff", None)
        assert legend.name == "custom"
        assert legend.type == "discrete"
        assert legend.break_points["color"].to_list() == ["#ff0000", "#00ff00", "#0000ff"]
        assert legend.break_points["upper_bound"].to_list() == [1.0, 2.0, None]

    def test_a_custom_continuous_map_reads_value_then_color_with_no_open_end(self):
        legend = resolve_legend("continuous:0,#ff0000,10,#0000ff", None)
        assert legend.type == "continuous"
        assert legend.break_points["upper_bound"].to_list() == [0.0, 10.0]
        assert legend.break_points["color"].to_list() == ["#ff0000", "#0000ff"]

    def test_an_unknown_map_type_is_rejected(self):
        with pytest.raises(RequestError) as excinfo:
            resolve_legend("sideways:#ff0000,1", None)
        assert str(excinfo.value) == "invalid color map type: sideways"
        assert resolve_status(excinfo.value) == 400

    def test_a_non_numeric_value_is_rejected(self):
        with pytest.raises(RequestError) as excinfo:
            resolve_legend("continuous:low,#ff0000", None)
        assert str(excinfo.value) == "invalid color map values"

    def test_mismatched_colors_and_values_are_rejected(self):
        with pytest.raises(RequestError) as excinfo:
            resolve_legend("continuous:0,#ff0000,10", None)
        assert str(excinfo.value) == "mismatched colors and values in color map"

    def test_a_named_legend_is_looked_up_by_name(self):
        assert resolve_legend("default", None) is LEGENDS["default"]

    def test_no_color_map_falls_back_to_the_variables_own_legend(self):
        variable_legend = Legend("mine", pl.DataFrame({"color": ["#fff"], "upper_bound": [None]}))
        assert resolve_legend(None, variable_legend) is variable_legend
        assert resolve_legend("", variable_legend) is variable_legend

    def test_no_color_map_and_no_variable_legend_falls_back_to_default(self):
        assert resolve_legend(None, None) is LEGENDS["default"]
        assert resolve_legend("not-a-real-legend-name", None) is LEGENDS["default"]


class TestResolveOutputType:
    def test_a_vector_grid_is_always_vector(self):
        assert resolve_output_type("vector", grid_cell=None, summarize_by=None) == OutputType.vector

    def test_a_raster_grid_with_no_narrowing_option_stays_raster(self):
        assert resolve_output_type("raster", grid_cell=None, summarize_by=None) == OutputType.raster

    def test_a_raster_grid_narrowed_by_grid_cell_becomes_vector(self):
        assert resolve_output_type("raster", grid_cell="1,2", summarize_by=None) == OutputType.vector

    def test_a_raster_grid_narrowed_by_summarize_by_becomes_vector(self):
        assert (
            resolve_output_type("raster", grid_cell=None, summarize_by="parish_v2020") == OutputType.vector
        )
