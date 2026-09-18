"""Tests for `cpra.mp.data.analysis.execution.sink` -- the sink.

One class per destination, plus one for the `execution.run_to_path` composition helper:

- `TestApplyOutputShaping`: `output.sort`/`output.select`, in isolation from any op walk.
- `TestSinkTabular`: csv/ndjson/parquet, both the streaming and the collect-first path.
- `TestWriteRaster`: `tif`, including the `False`-not-raised no-bands case.
- `TestRenderTile`: `png`/`jpg`, including a pre-colorization-array assertion (the array
  `legends.Legend.colorize_pixels` is actually handed, captured via a stub legend rather
  than decoded back out of PNG bytes) and both ways a tile can come back empty.
- `TestEmptyTile`: the blank-image body a route sends alongside that empty answer, checked
  the way a client sees it -- decoded back out of the bytes.
- `TestSinkVectorTile`: `pbf`, parsed back at the protobuf level to check exactly which
  boundary features gained a `value` tag.
- `TestRunToPath`: the composition helper, for its two supported output kinds.

Fixture note: raster tests use `RASTER_KWARGS` (`vct` on `veg_grid_cell_v001`, EPSG:26915,
1052x365 at 480m pixels, bounds ``(404710, 3199480, 909670, 3374680)``) -- the same fixture
`tests/api/test_execute.py::TestBandWideEndToEnd` uses, so numbers already derived and
pinned there (the `30/40 -> filter>35 -> *2 -> sum` chain landing on a constant 80) are
reused here rather than re-derived, where a test needs that exact chain.

Degenerate-metadata note, since two tests below turn on it: a `group` with `by=[]` leaves
`metadata_df` holding nothing but the reserved band-index column, and Polars cannot
represent a frame of zero columns and one row (``pl.DataFrame({"x": [0]}).drop("x")``
returns shape ``(0, 0)``, discarding the row count along with the last column -- the same
"zero columns loses the row count" collapse `execution.frame.empty_band_wide`'s docstring
warns about). So the band count only survives the walk because
`execution.frame.released_band_metadata` declines to drop that column when it is the last
one standing. `write_raster` reads exactly that count, via `execution.frame.has_no_bands`,
to tell "no source data" (`False`, no file written) from "wrote a raster" --
`test_group_by_nothing_collapses_to_one_band` covers the `by=[]` spelling and
`test_group_by_a_real_column_collapses_to_one_band` the ordinary one. `render_tile` reads
the same count for its own "nothing to show" answer, so its `by=[]` case
(`test_filter_calculate_group_collapses_to_one_band`) puts the same released metadata to
work at the other destination: one surviving band there means a rendered tile, and zero
means `None` (a 404 carrying `empty_tile`'s blank image).
"""

import gzip
from io import BytesIO

import mercantile
import numpy as np
import polars as pl
import pytest
import rasterio
from mapbox_vector_tile.Mapbox.vector_tile_pb2 import tile as VectorTile
from PIL import Image
from rasterio.enums import Resampling

from cpra.mp.data.analysis.execution import plan as plan_module
from cpra.mp.data.analysis.execution import run_to_path
from cpra.mp.data.analysis.execution import sink as sink_module
from cpra.mp.data.analysis.execution.execute import execute
from cpra.mp.data.analysis.execution.frame import empty_band_wide
from cpra.mp.data.analysis.execution.plan import TileRequest, compile_plan
from cpra.mp.data.analysis.execution.sink import (
    DEFAULT_NODATA,
    _resolve_write_dtype,
    apply_output_shaping,
    empty_tile,
    render_tile,
    sink_tabular,
    sink_vector_tile,
    write_raster,
)
from cpra.mp.data.analysis.io.raster_io import tile_source_window, warp_window_to_tile
from cpra.mp.data.analysis.request.ops import validate_shapes
from cpra.mp.data.analysis.request.pipeline import Output, parse_pipeline
from cpra.mp.data.grids import GRIDS
from tests.factories import RASTER_KWARGS, raster_array, write_pmtiles, write_raster_dataset

RASTER_GRID = RASTER_KWARGS["grid"]

# The Web Mercator tile that fully contains `veg_grid_cell_v001`'s bounds -- the smallest
# single tile that does (see the module-level derivation note in `TestRenderTile`).
CONTAINING_TILE = mercantile.Tile(x=0, y=0, z=1)
# The opposite quadrant of the z=1 split -- eastern/southern hemisphere, nowhere near the
# Gulf Coast dataset this file's raster fixtures use -- for the disjoint-bounds case.
DISJOINT_TILE = mercantile.Tile(x=1, y=1, z=1)
# A tile covering a strict sub-rectangle of `veg_grid_cell_v001` -- window
# ``col_off=512, row_off=0, 297x226`` against the grid's 1052x365, so both a narrower read
# and a nonzero offset, which a chain that quietly read the whole raster could not produce.
PARTIAL_TILE = mercantile.Tile(x=63, y=105, z=8)


def raster_input(variable=RASTER_KWARGS["variable"], grid=RASTER_GRID):
    return {"variable": variable, "time_unit": "annual", "grid": grid}


def _pipeline(data):
    pipeline = parse_pipeline(data)
    validate_shapes(pipeline)
    return pipeline


class TestApplyOutputShaping:
    """`sort`/`select`, applied to a plain `LazyFrame` -- no op walk involved, so every
    expected value here is arithmetic-free: just which rows land in which order, and
    which columns survive.
    """

    def _lf(self):
        return pl.DataFrame(
            {"grid_cell": [1, 1, 2], "value": [5, 1, 9], "extra": ["a", "b", "c"]}
        ).lazy()

    def test_sort_ascending_by_default(self):
        out = apply_output_shaping(self._lf(), Output(format="csv", sort="value")).collect()
        assert out["value"].to_list() == [1, 5, 9]

    def test_leading_dash_sorts_that_column_descending(self):
        # Same rows as above; `-value` reverses only the ordering, not which rows exist.
        out = apply_output_shaping(self._lf(), Output(format="csv", sort="-value")).collect()
        assert out["value"].to_list() == [9, 5, 1]

    def test_multi_key_sort_mixes_ascending_and_descending(self):
        # grid_cell ascending (1s before 2), value descending within each group
        # (5 before 1): the three rows (1,5) (1,1) (2,9) land in exactly that order.
        out = apply_output_shaping(self._lf(), Output(format="csv", sort="grid_cell,-value")).collect()
        assert list(zip(out["grid_cell"].to_list(), out["value"].to_list())) == [(1, 5), (1, 1), (2, 9)]

    def test_select_narrows_columns_without_reordering_rows(self):
        out = apply_output_shaping(self._lf(), Output(format="csv", select="value,grid_cell")).collect()
        assert out.columns == ["value", "grid_cell"]
        assert out["value"].to_list() == [5, 1, 9]

    def test_sort_runs_before_select_even_when_the_sort_key_is_dropped(self):
        # `select` keeps only `extra`, dropping `value` -- the column `sort` orders by.
        # If `select` ran first, `value` would already be gone and the sort would fail
        # (or silently no-op); running `sort` first, then `select`, is what makes this
        # combination possible at all. Expected order (value desc: 9, 5, 1) is read off
        # `extra`'s corresponding letters, since `value` itself is no longer in the output.
        out = apply_output_shaping(
            self._lf(), Output(format="csv", sort="-value", select="extra")
        ).collect()
        assert out.columns == ["extra"]
        assert out["extra"].to_list() == ["c", "a", "b"]

    def test_neither_option_set_is_a_no_op(self):
        out = apply_output_shaping(self._lf(), Output(format="csv")).collect()
        assert out["grid_cell"].to_list() == [1, 1, 2]
        assert out.columns == ["grid_cell", "value", "extra"]


class TestSinkTabular:
    """Each format's own writer is Polars' own (`write_csv`/`write_ndjson`/
    `write_parquet`, or their `sink_*` streaming counterparts) -- this only checks that
    `sink_tabular` wires `materialize`, `collect_in_memory`, and `float_precision`
    correctly, via output any of those writers would produce for the given input, pinned
    by direct computation against the same writer (not captured from `sink_tabular`
    itself) in each test.
    """

    def _lf(self):
        return pl.DataFrame({"grid_cell": [1, 2], "value": [1.23456, 2.5]}).lazy()

    def test_csv_collect_in_memory_rounds_to_float_precision(self, tmp_path):
        # `float_precision=2`: 1.23456 -> "1.23", 2.5 -> "2.50" -- Polars' own CSV writer
        # behavior, confirmed independently against `DataFrame.write_csv` before pinning
        # here, not merely captured from `sink_tabular`'s own output.
        out_path = tmp_path / "out.csv"
        sink_tabular(self._lf(), out_path, "csv", collect_in_memory=True, float_precision=2)
        assert out_path.read_text() == "grid_cell,value\n1,1.23\n2,2.50\n"

    def test_csv_streaming_matches_the_collect_path(self, tmp_path):
        # Same frame, same float_precision, `collect_in_memory=False` this time --
        # streamed through `sink_csv` rather than collected first. Byte-identical output
        # is the whole point of the split: which one runs must not change the answer.
        out_path = tmp_path / "out.csv"
        sink_tabular(self._lf(), out_path, "csv", collect_in_memory=False, float_precision=2)
        assert out_path.read_text() == "grid_cell,value\n1,1.23\n2,2.50\n"

    def test_ndjson_ignores_float_precision(self, tmp_path):
        # tabular_kwargs only ever produces a float_precision kwarg for csv (io.output's
        # own rule) -- ndjson's writer takes no such argument, so passing one through
        # unconditionally would raise; this proves that branch is skipped for ndjson.
        out_path = tmp_path / "out.ndjson"
        sink_tabular(self._lf(), out_path, "ndjson", collect_in_memory=True, float_precision=2)
        assert out_path.read_text() == '{"grid_cell":1,"value":1.23456}\n{"grid_cell":2,"value":2.5}\n'

    def test_parquet_streaming_round_trips_the_exact_values(self, tmp_path):
        out_path = tmp_path / "out.parquet"
        sink_tabular(self._lf(), out_path, "parquet", collect_in_memory=False)
        back = pl.read_parquet(out_path)
        assert back["grid_cell"].to_list() == [1, 2]
        assert back["value"].to_list() == [1.23456, 2.5]


class TestWriteRaster:
    """`RASTER_KWARGS`'s grid (`veg_grid_cell_v001`) is 1052x365 pixels; every test below
    that writes a real file checks `count`/`height`/`width`/`crs`/`nodata` against that
    grid's own known geometry, and every pixel's value against arithmetic on the fixture's
    own uniform fill (every cell of a `write_raster_dataset` year is the SAME value, so a
    band's min/max/mean collapsing to one number is a correctness check, not an
    approximation).
    """

    def test_writes_one_band_per_surviving_column_with_no_group(self, base_path):
        # calculate-only, no group: two input years survive as two output bands,
        # 30 -> 60 and 40 -> 80 (both times an int32 column doubled by a constant, which
        # Polars keeps as int32 -- confirmed directly, not assumed).
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: 30 if y == 2030 else 40)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "calculate", "input": "a", "expression": "value * 2"}],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        result = execute(plan, base_path=base_path)

        out_path = base_path.parent / "written.tif"
        assert write_raster(result, out_path) is True

        grid = GRIDS[RASTER_GRID]
        with rasterio.open(out_path) as src:
            assert src.count == 2
            assert (src.height, src.width) == (grid.height, grid.width)
            assert src.crs == grid.crs
            assert src.nodata == DEFAULT_NODATA
            assert src.dtypes == ("int32", "int32")
            band1, band2 = src.read()
            assert (band1 == 60).all()
            assert (band2 == 80).all()

    def test_group_by_a_real_column_collapses_to_one_band(self, base_path):
        # `variable` is constant ("vct") across both years, so grouping by it merges both
        # bands into one group -- the collapse-to-one-band case reached via a REAL
        # surviving metadata column. The `by=[]` spelling of the same collapse, where no
        # non-reserved column survives at all, is the test below this one.
        # sum_horizontal(60, 80) = 140 at every pixel.
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: 30 if y == 2030 else 40)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [
                    {"op": "calculate", "input": "a", "expression": "value * 2", "as": "c"},
                    {"op": "group", "by": ["variable"], "aggregate": "sum"},
                ],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        result = execute(plan, base_path=base_path)
        _lf, metadata_df, _window = result
        assert metadata_df.height == 1  # the real column survived -- no collapse bug here

        out_path = base_path.parent / "grouped.tif"
        assert write_raster(result, out_path) is True
        with rasterio.open(out_path) as src:
            assert src.count == 1
            assert (src.read(1) == 140).all()

    def test_group_by_nothing_collapses_to_one_band(self, base_path):
        # `by=[]` drops every dimension column out of `metadata_df`, so the reserved
        # band-index column is the only thing left carrying "one band survived" -- and
        # since Polars cannot hold a frame of zero columns and one row, dropping it on the
        # way out of the walk would take the band count with it and leave `write_raster`
        # reading a perfectly good single-band raster as no data at all (`False`, no file,
        # a 404 once a route is bound). `execution.frame.released_band_metadata` is what
        # keeps it standing in exactly that case; this pins the consequence at the layer
        # where it bites. Same chain `tests/api/test_execute.py::TestBandWideEndToEnd`
        # derives: 30/40 -> keep > 35 (2030 goes null) -> *2 -> sum = 80 at every pixel.
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: 30 if y == 2030 else 40)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [
                    {"op": "filter", "input": "a", "where": "value > 35", "as": "f"},
                    {"op": "calculate", "expression": "value * 2", "as": "c"},
                    {"op": "group", "by": [], "aggregate": "sum"},
                ],
                "output": {"format": "tif"},
            }
        )
        plan = compile_plan(pipeline, base_path=base_path)
        result = execute(plan, base_path=base_path)

        out_path = base_path.parent / "grouped-by-nothing.tif"
        assert write_raster(result, out_path) is True

        grid = GRIDS[RASTER_GRID]
        with rasterio.open(out_path) as src:
            assert src.count == 1
            assert (src.height, src.width) == (grid.height, grid.width)
            assert (src.read(1) == 80).all()

    def test_no_bands_returns_false_and_writes_nothing(self, base_path):
        # The canonical empty band-wide result (execution.frame.empty_band_wide) --
        # constructed directly rather than via a dimension filter that happens to narrow
        # everything away, since that is exactly what this function documents itself as
        # the result of. `write_raster` must return `False`, not raise, and must not
        # create `out_path` at all -- a caller distinguishing "no data" (404) from
        # "wrote a file" needs both halves of that.
        write_raster_dataset(base_path, years=(2030,))
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
        plan = compile_plan(pipeline, base_path=base_path)
        _lf, _metadata_df, window = execute(plan, base_path=base_path)
        empty = empty_band_wide(window)

        out_path = base_path.parent / "empty.tif"
        assert write_raster(empty, out_path) is False
        assert not out_path.exists()

    def test_explicit_dtype_and_nodata_override_the_defaults(self, base_path):
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 7)
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
        plan = compile_plan(pipeline, base_path=base_path)
        result = execute(plan, base_path=base_path)

        out_path = base_path.parent / "typed.tif"
        assert write_raster(result, out_path, dtype="int16", nodata=-1) is True
        with rasterio.open(out_path) as src:
            assert src.dtypes == ("int16",)
            assert src.nodata == -1
            assert (src.read(1) == 7).all()


class TestResolveWriteDtype:
    """`_resolve_write_dtype` in isolation: the small pure function `write_raster` defers
    to for its default-dtype inference, cheaper to pin directly than only indirectly
    through a full raster write.
    """

    def test_explicit_dtype_wins_outright(self):
        df = pl.DataFrame({"b0": pl.Series([1], dtype=pl.Int8)})
        assert _resolve_write_dtype(df, ["b0"], "float64") == "float64"

    def test_shared_dtype_is_inferred(self):
        df = pl.DataFrame({"b0": pl.Series([1], dtype=pl.Float32), "b1": pl.Series([2], dtype=pl.Float32)})
        assert _resolve_write_dtype(df, ["b0", "b1"], None) == "float32"

    def test_mixed_dtypes_fall_back_to_float64(self):
        df = pl.DataFrame({"b0": pl.Series([1], dtype=pl.Float32), "b1": pl.Series([2], dtype=pl.Int32)})
        assert _resolve_write_dtype(df, ["b0", "b1"], None) == "float64"


# The tile fully containing `veg_grid_cell_v001`'s bounds. `veg_grid_cell_v001` spans
# roughly -94.0 to -88.7 degrees longitude, 28.9 to 30.5 north -- wide enough that no
# single tile at z=2 or finer contains it whole (confirmed by exhaustive check over
# z=1..7: z=1's four-way split is the finest level at which one tile still does), so a
# geometry-independent single-tile render test has to use z=1 or accept a multi-tile
# dataset. Using CONTAINING_TILE keeps the read window (and therefore the independently
# recomputed expected array below) equal to the WHOLE source raster, clipped to nothing --
# `tile_source_window`'s buffer/clip arithmetic reduces to a no-op, which is what makes
# hand-deriving the expected pre-warp array tractable without re-deriving that arithmetic.
class TestRenderTile:
    """Every array assertion below is against the array `render_tile` hands to
    `Legend.colorize_pixels`, BEFORE any color is applied -- captured by monkeypatching
    `sink.resolve_legend` to return a stub whose `colorize_pixels` records its argument
    and returns a fixed, trivial RGBA array (so `Image.fromarray`/PNG encoding downstream
    still succeeds, but the colors themselves are never asserted on -- legend color
    selection is `legends.py`'s own contract, out of scope here).

    Each test drives the whole compile -> execute -> sink chain, because that is what a
    tile render IS here: `compile_plan(..., tile=TileRequest(...))` decides which pixels of
    each raster the tile needs, `execute` reads exactly those and runs the ops over them,
    and `render_tile` warps/colors/encodes whatever comes out. Nothing below hands
    `render_tile` a hand-built frame -- a tile-scoped window that `compile_plan` and
    `execute` disagreed about would be invisible to a test that skipped either.

    Expected pre-colorization arrays are computed independently, in each test, by calling
    the lower-level primitives that chain rests on (`io.raster_io.tile_source_window`/
    `warp_window_to_tile`) directly against a hand-derived pre-warp array -- never by
    calling `render_tile` a second time. Because every fixture raster is spatially UNIFORM
    (one fill value everywhere), the pre-warp array a `filter`/`calculate`/`group` chain
    produces is itself a constant, computable by hand from the chain's own arithmetic
    (documented per-test) without needing to replicate pixel-by-pixel masking -- so the
    only thing genuinely delegated to `warp_window_to_tile` here is the reprojection
    geometry itself, which is `io.raster_io`'s own already-tested code, not this module's.
    """

    def _stub_legend(self, monkeypatch, captured):
        class _StubLegend:
            def colorize_pixels(self, pixel_values):
                captured.append(pixel_values)
                return np.zeros((*pixel_values.shape[1:], 4), dtype="uint8")

        monkeypatch.setattr(sink_module, "resolve_legend", lambda *a, **k: _StubLegend())

    def _render(self, base_path, pipeline, *, tile=CONTAINING_TILE, tile_size=16, format="png"):
        """Compile for `tile`, execute, render -- the whole chain, as a route would run it.

        `mercantile.Tile` is an `(x, y, z)` namedtuple, so it splats straight into
        `TileRequest`'s first three fields.
        """
        plan = compile_plan(pipeline, base_path=base_path, tile=TileRequest(*tile, tile_size=tile_size))
        return plan, render_tile(execute(plan, base_path=base_path), plan.tile, format)

    def _expected_tile_data(self, base_path, fill_value, dtype, tile_size, *, tile=CONTAINING_TILE):
        """Independently warp a constant-valued array of `fill_value` (the hand-derived
        result of whatever op chain a test ran) through the same window/tile geometry the
        compile-execute-render chain resolves for `tile` against this fixture's first
        raster file.
        """
        path = next((base_path).rglob("*.tif"))
        with rasterio.open(path) as src:
            tile_bounds = mercantile.xy_bounds(*tile)
            window = tile_source_window(src, tile_bounds, resampling=Resampling.nearest)
            data = np.full((1, int(window.height), int(window.width)), fill_value, dtype=dtype)
            return warp_window_to_tile(
                data,
                window=window,
                src_transform=src.transform,
                src_crs=src.crs,
                tile_bounds=tile_bounds,
                tile_size=tile_size,
                nodata=DEFAULT_NODATA,
                resampling=Resampling.nearest,
            )

    def test_calculate_only_matches_independently_warped_expectation(self, base_path, monkeypatch):
        # Single year, fill=50; `value * 2` -> a constant 100 everywhere real data exists.
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 50)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "calculate", "input": "a", "expression": "value * 2"}],
                "output": {"format": "tif"},
            }
        )
        captured = []
        self._stub_legend(monkeypatch, captured)
        tile_size = 16
        _plan, result = self._render(base_path, pipeline, tile_size=tile_size)

        assert result is not None
        assert len(captured) == 1
        expected = self._expected_tile_data(base_path, 100, "int32", tile_size)
        np.testing.assert_array_equal(captured[0], expected)
        # The encode path itself ran too: valid PNG bytes of the requested tile size.
        image = Image.open(BytesIO(result))
        assert image.size == (tile_size, tile_size)

    def test_a_partial_tile_reads_only_its_own_window(self, base_path, monkeypatch):
        # The case the uniform-fill tests above cannot show, since every pixel of those
        # rasters is interchangeable: PARTIAL_TILE covers a strict sub-rectangle of the
        # raster, so the compiled window's OFFSET has to survive the whole chain -- through
        # the scan, through the frame's row order, into the transform the sink warps with --
        # for the right pixels to land in the right place. A unique value per pixel
        # (`raster_array`'s `arange`, shifted off zero so no pixel collides with a nodata
        # sentinel) is what makes a lost or wrong offset visible at all.
        #
        # The expectation is built by reading that same window straight off the file with
        # rasterio and warping it, which shares no code with the chain under test past
        # `warp_window_to_tile` itself.
        grid = GRIDS[RASTER_GRID]
        write_raster_dataset(base_path, years=(2030,), data_fn=lambda _y: raster_array() + 1)
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
        captured = []
        self._stub_legend(monkeypatch, captured)
        tile_size = 16
        plan, result = self._render(base_path, pipeline, tile=PARTIAL_TILE, tile_size=tile_size)

        window = plan.inputs["a"].read_window
        assert int(window.width) < grid.width
        assert int(window.height) < grid.height
        assert int(window.col_off) > 0
        assert result is not None

        path = next(base_path.rglob("*.tif"))
        with rasterio.open(path) as src:
            expected = warp_window_to_tile(
                src.read(1, window=window)[np.newaxis, :, :],
                window=window,
                src_transform=src.transform,
                src_crs=src.crs,
                tile_bounds=mercantile.xy_bounds(*PARTIAL_TILE),
                tile_size=tile_size,
                nodata=DEFAULT_NODATA,
                resampling=Resampling.nearest,
            )
        np.testing.assert_array_equal(captured[0], expected)
        # ...and the values really do vary across the tile, so the comparison above has
        # something to disagree about.
        assert len(np.unique(captured[0])) > 1

    def test_filter_calculate_group_collapses_to_one_band(self, base_path, monkeypatch):
        # Same 30/40-fill, filter>35, *2, group(by=[], sum) chain as
        # tests/api/test_execute.py::TestBandWideEndToEnd -- pinned there as landing on a
        # constant 80 everywhere real data exists (30 masked null by `value > 35`, 40
        # doubled to 80, `sum_horizontal` skipping the null contributor). Reused rather
        # than re-derived. `render_tile` hands no metadata back to an outside caller (see
        # this file's module docstring), so the reserved-column release that a `by=[]`
        # group makes load-bearing on the `write_raster` path does not apply here.
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: 30 if y == 2030 else 40)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [
                    {"op": "filter", "input": "a", "where": "value > 35", "as": "f"},
                    {"op": "calculate", "expression": "value * 2", "as": "c"},
                    {"op": "group", "by": [], "aggregate": "sum"},
                ],
                "output": {"format": "tif"},
            }
        )
        captured = []
        self._stub_legend(monkeypatch, captured)
        tile_size = 16
        _plan, result = self._render(base_path, pipeline, tile_size=tile_size)

        assert result is not None
        expected = self._expected_tile_data(base_path, 80, "int32", tile_size)
        np.testing.assert_array_equal(captured[0], expected)

    def test_a_tile_that_misses_the_raster_returns_none(self, base_path):
        # DISJOINT_TILE's footprint (the opposite z=1 quadrant) shares no pixel with this
        # raster, which `compile_plan` settles before any read: `tile_disjoint` on the
        # resolved input, an empty band-wide result out of `execute`, and `None` -- a 404
        # -- out of the sink.
        #
        # The old engine's tile path split this in two, rendering an all-nodata tile when a
        # footprint overlapped the dataset's reprojected bounds but still resolved to no
        # source pixels, and 404ing only when the bounds themselves were disjoint. Both are
        # a miss here. The two differ only across a sliver the width of the disagreement
        # between two bbox approximations of the same reprojected (curved) quad, and an
        # empty tile and a missing one render identically on a map.
        write_raster_dataset(base_path, years=(2030,))
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
        plan, result = self._render(base_path, pipeline, tile=DISJOINT_TILE)

        assert plan.inputs["a"].tile_disjoint is True
        assert plan.inputs["a"].read_window is None
        assert result is None

    def test_a_dimension_predicate_that_narrows_to_no_bands_returns_none(self, base_path):
        # The other half of the `None` contract, and the half that has nothing to do with
        # geography: CONTAINING_TILE does overlap, but no band survives `calendar_year ==
        # 1999` against a 2030-only dataset, so there is no pixel to color. Same answer to
        # the caller as the disjoint case above -- this tile has nothing to show.
        write_raster_dataset(base_path, years=(2030,))
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "filter", "input": "a", "where": "calendar_year == 1999"}],
                "output": {"format": "tif"},
            }
        )
        _plan, result = self._render(base_path, pipeline)

        assert result is None

    def test_more_than_one_surviving_band_raises(self, base_path):
        # Two years, no group: two bands reach the end of the (empty) op chain, which
        # `Legend.colorize_pixels` cannot color (it assumes exactly one).
        write_raster_dataset(base_path, years=(2030, 2040))
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})

        with pytest.raises(ValueError, match="exactly one band"):
            self._render(base_path, pipeline)

    def test_jpg_drops_the_alpha_channel(self, base_path, monkeypatch):
        # The one branch that separates the two image formats: JPEG carries no alpha, so
        # the RGBA array the legend returns is narrowed to RGB before encoding. Decoded
        # back out of the bytes rather than asserted on the array, since the narrowing is
        # only observable in what gets encoded.
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 50)
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
        self._stub_legend(monkeypatch, [])
        _plan, result = self._render(base_path, pipeline, tile_size=8, format="jpg")

        image = Image.open(BytesIO(result))
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (8, 8)

    def test_a_rendered_tile_and_an_empty_one_share_an_encoder(self, base_path, monkeypatch):
        # A blank tile a client cannot decode would defeat the whole point of sending one,
        # so `empty_tile` goes through this function's own encoder rather than a second
        # spelling of it. The stub legend here returns an all-zero RGBA array -- which is
        # exactly the array `empty_tile` builds -- so a real render and a blank one must
        # come out byte for byte identical.
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 50)
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
        self._stub_legend(monkeypatch, [])
        _plan, rendered = self._render(base_path, pipeline, tile_size=16)

        assert rendered == empty_tile("png", tile_size=16)


class TestEmptyTile:
    """The body a route pairs with the 404 it sends when `render_tile` returned `None`.

    The point of it is entirely on the client side -- an image URL that answers with a JSON
    error document makes every map pan past the edge of the data log a decode failure -- so
    every assertion here is what a client's decoder would see, read back out of the encoded
    bytes rather than off the array that produced them.
    """

    def test_a_png_is_a_fully_transparent_tile_of_the_requested_size(self):
        result = empty_tile("png", tile_size=32)

        image = Image.open(BytesIO(result))
        assert image.format == "PNG"
        assert image.mode == "RGBA"
        assert image.size == (32, 32)
        assert np.asarray(image)[:, :, 3].max() == 0  # every pixel fully transparent

    def test_a_jpg_is_a_solid_tile_of_the_requested_size(self):
        # JPEG has no alpha channel, so there is no transparent answer to give -- a blank
        # `jpg` tile is necessarily a solid color, and is black because dropping the alpha
        # column off the same all-zero RGBA array is what produces it. Asserted as "one
        # color everywhere" rather than exact bytes: JPEG is lossy, and a uniform image is
        # the strongest thing its encoder actually promises to round-trip.
        result = empty_tile("jpg", tile_size=32)

        image = Image.open(BytesIO(result))
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (32, 32)
        assert np.asarray(image).max() == 0

    def test_it_defaults_to_the_same_tile_size_a_tile_request_does(self):
        # Both sides of a miss have to agree on size without being told: a route sending
        # this for a tile it could not render has a `TileRequest` whose `tile_size` the
        # client never supplied either, so the two defaults must be the one constant.
        assert Image.open(BytesIO(empty_tile("png"))).size == (
            TileRequest(0, 0, 0).tile_size,
            TileRequest(0, 0, 0).tile_size,
        )


class TestSinkVectorTile:
    """`write_pmtiles` (tests/factories.py) builds a single-tile archive with a `boundary`
    layer carrying only feature ids (trivial point geometry, no keys/values) -- exactly
    the shape `sink_vector_tile` expects to inject into.
    """

    TILE = mercantile.Tile(x=1, y=2, z=3)

    def _parse(self, result):
        vt = VectorTile()
        vt.ParseFromString(result)
        return next(layer for layer in vt.layers if layer.name == "boundary")

    def test_injects_value_only_for_features_with_a_matching_nonnull_row(self, tiles_path):
        write_pmtiles(tiles_path, "ecoregion_v001", self.TILE, [1, 2, 3])
        lf = pl.DataFrame(
            {"grid_cell": [1, 2, 99], "value": [10.5, None, 5.0]},
            schema={"grid_cell": pl.Int64, "value": pl.Float64},
        ).lazy()

        result = sink_vector_tile(lf, "ecoregion_v001", self.TILE)

        assert result is not None
        layer = self._parse(result)
        assert list(layer.keys) == ["value"]
        by_id = {f.id: f for f in layer.features}
        # Feature 1: matched, non-null -- gets the (key_index=0, value_index) tag pair.
        assert list(by_id[1].tags) == [0, 0]
        assert layer.values[0].double_value == 10.5
        # Feature 2: matched but null value -- no tag at all.
        assert list(by_id[2].tags) == []
        # Feature 3: present in the tile but grid_cell 99 never matched it -- no tag.
        assert list(by_id[3].tags) == []

    def test_result_is_raw_protobuf_not_gzipped(self, tiles_path):
        # Matches the old engine's inject_and_serialize contract: the injected tile is
        # handed back as plain protobuf bytes, not re-gzipped -- a caller's own transport
        # layer (or none) decides compression, the same way the tile read off disk was
        # decompressed once on the way in and never recompressed.
        write_pmtiles(tiles_path, "ecoregion_v001", self.TILE, [1])
        lf = pl.DataFrame({"grid_cell": [1], "value": [1.0]}).lazy()
        result = sink_vector_tile(lf, "ecoregion_v001", self.TILE)
        with pytest.raises(gzip.BadGzipFile):
            gzip.decompress(result)
        # But it does parse as a plain VectorTile message.
        self._parse(result)

    def test_missing_archive_returns_none(self, tiles_path):
        lf = pl.DataFrame({"grid_cell": [1], "value": [1.0]}).lazy()
        assert sink_vector_tile(lf, "no_such_grid", self.TILE) is None

    def test_tile_absent_from_archive_returns_none(self, tiles_path):
        write_pmtiles(tiles_path, "ecoregion_v001", self.TILE, [1])
        lf = pl.DataFrame({"grid_cell": [1], "value": [1.0]}).lazy()
        assert sink_vector_tile(lf, "ecoregion_v001", mercantile.Tile(x=9, y=9, z=9)) is None

    def test_boundary_layer_with_no_features_returns_none(self, tiles_path):
        # The tile exists in the archive and decompresses fine, but its own `boundary`
        # layer carries zero features (an empty `feature_ids` list at write time) -- a
        # different "nothing to render" cause than a missing archive/tile, but still
        # mapped to the same `None`.
        write_pmtiles(tiles_path, "ecoregion_v001", self.TILE, [])
        lf = pl.DataFrame({"grid_cell": [1], "value": [1.0]}).lazy()
        assert sink_vector_tile(lf, "ecoregion_v001", self.TILE) is None

    def test_matched_rows_all_null_returns_none(self, tiles_path):
        # `grid_cell` 1 and 2 both match a real feature in the tile, but every matching
        # row's `value` is null -- `id_values` ends up empty even though `df` itself is
        # not, a case distinct from `test_no_row_survives_at_all_returns_none` below
        # (where no row matches at all, so `df` itself is empty).
        write_pmtiles(tiles_path, "ecoregion_v001", self.TILE, [1, 2])
        lf = pl.DataFrame(
            {"grid_cell": [1, 2], "value": [None, None]}, schema={"grid_cell": pl.Int64, "value": pl.Float64}
        ).lazy()
        assert sink_vector_tile(lf, "ecoregion_v001", self.TILE) is None

    def test_grid_cell_ids_narrows_which_features_are_eligible(self, tiles_path):
        write_pmtiles(tiles_path, "ecoregion_v001", self.TILE, [1, 2, 3])
        lf = pl.DataFrame({"grid_cell": [1, 2], "value": [10.0, 20.0]}).lazy()

        # Only feature 2 is in scope; feature 1 has a matching, non-null row but is
        # excluded before `lf` is even queried.
        result = sink_vector_tile(lf, "ecoregion_v001", self.TILE, grid_cell_ids=[2, 3])

        assert result is not None
        layer = self._parse(result)
        by_id = {f.id: f for f in layer.features}
        assert list(by_id[1].tags) == []
        assert list(by_id[2].tags) == [0, 0]

    def test_grid_cell_ids_excluding_every_feature_returns_none(self, tiles_path):
        write_pmtiles(tiles_path, "ecoregion_v001", self.TILE, [1, 2, 3])
        lf = pl.DataFrame({"grid_cell": [1], "value": [10.0]}).lazy()
        assert sink_vector_tile(lf, "ecoregion_v001", self.TILE, grid_cell_ids=[999]) is None

    def test_no_row_survives_at_all_returns_none(self, tiles_path):
        write_pmtiles(tiles_path, "ecoregion_v001", self.TILE, [1, 2, 3])
        lf = pl.DataFrame({"grid_cell": [42], "value": [1.0]}).lazy()
        assert sink_vector_tile(lf, "ecoregion_v001", self.TILE) is None

    def test_default_tiles_path_reads_the_configured_directory(self, tiles_path):
        # No explicit tiles_path= -- relies on the `tiles_path` fixture's own
        # monkeypatch of `cpra.mp.data.config.TILES_PATH`, never the real production
        # default (tests/conftest.py's sandboxing, confirmed by this test using it).
        write_pmtiles(tiles_path, "ecoregion_v001", self.TILE, [1])
        lf = pl.DataFrame({"grid_cell": [1], "value": [1.0]}).lazy()
        assert sink_vector_tile(lf, "ecoregion_v001", self.TILE) is not None

    def test_explicit_tiles_path_overrides_the_configured_one(self, tmp_path):
        sandbox = tmp_path / "elsewhere"
        sandbox.mkdir()
        write_pmtiles(sandbox, "ecoregion_v001", self.TILE, [1])
        lf = pl.DataFrame({"grid_cell": [1], "value": [1.0]}).lazy()
        assert sink_vector_tile(lf, "ecoregion_v001", self.TILE, tiles_path=sandbox) is not None


class TestRunToPath:
    """The compile -> execute -> sink composition helper, for its two supported output
    kinds. Not re-testing `compile_plan`/`execute`'s own correctness (covered by
    `test_execution_plan.py`/`test_execute.py`) -- only that this function wires the
    three stages together and reaches the right sink.
    """

    def test_tabular_output_applies_shaping_and_writes_the_file(self, base_path):
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: 30 if y == 2030 else 40)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [
                    {"op": "filter", "input": "a", "where": "grid_cell < 2", "as": "f"},
                ],
                "output": {"format": "csv", "sort": "-calendar_year", "select": "calendar_year,value"},
            }
        )
        out_path = base_path.parent / "out.csv"
        assert run_to_path(pipeline, out_path, base_path=base_path, collect_in_memory=True) is True

        # grid_cell < 2 keeps pixels 0 and 1 of each band; both years both pixels share
        # one fill value each (30 for 2030, 40 for 2040), so the CSV is 2 rows per year,
        # sorted by calendar_year descending -- 2040's rows (value 40) before 2030's
        # (value 30), and `select` drops every column but the two named.
        text = out_path.read_text()
        assert text.splitlines()[0] == "calendar_year,value"
        rows = [line.split(",") for line in text.splitlines()[1:]]
        assert [r[0] for r in rows] == ["2040", "2040", "2030", "2030"]
        assert {r[1] for r in rows[:2]} == {"40"}
        assert {r[1] for r in rows[2:]} == {"30"}

    def test_tif_output_writes_a_raster_and_returns_true(self, base_path):
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 12)
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "calculate", "input": "a", "expression": "value + 1"}],
                "output": {"format": "tif"},
            }
        )
        out_path = base_path.parent / "out.tif"
        assert run_to_path(pipeline, out_path, base_path=base_path) is True
        with rasterio.open(out_path) as src:
            assert src.nodata == DEFAULT_NODATA  # left at write_raster's own default
            assert (src.read(1) == 13).all()

    def test_tif_output_forwards_an_explicit_falsy_nodata(self, base_path):
        # nodata=0 is a legitimate declared value, not "unset" -- must reach the file
        # exactly, not be silently replaced by DEFAULT_NODATA via a bare `nodata or ...`.
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 5)
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
        out_path = base_path.parent / "out_zero_nodata.tif"
        assert run_to_path(pipeline, out_path, base_path=base_path, nodata=0) is True
        with rasterio.open(out_path) as src:
            assert src.nodata == 0

    def test_tif_output_forwards_an_explicit_dtype(self, base_path):
        write_raster_dataset(base_path, years=(2030,), fill_fn=lambda _y: 5)
        pipeline = _pipeline({"inputs": {"a": raster_input()}, "steps": [], "output": {"format": "tif"}})
        out_path = base_path.parent / "out_int16.tif"
        assert run_to_path(pipeline, out_path, base_path=base_path, dtype="int16") is True
        with rasterio.open(out_path) as src:
            assert src.dtypes == ("int16",)

    def test_tif_output_with_no_bands_returns_false(self, base_path):
        # A `value` predicate (however extreme) only masks pixel values -- it never
        # removes a band from `metadata_df`, so it does not exercise the no-bands path
        # (confirmed the hard way: an earlier draft of this test used `value > 999999`
        # and got back `True`, one band, every pixel null). What DOES narrow the band
        # list to nothing at compile time is a DIMENSION predicate pushed into
        # `io.catalog.resolve_raster_bands` (execution/plan.py's pushdown) that no band
        # satisfies -- the dataset only has a 2030 band, so `calendar_year == 9999`
        # leaves zero rows in `metadata_df`, the genuine `empty_band_wide` case
        # `write_raster` reports as `False`.
        write_raster_dataset(base_path, years=(2030,))
        pipeline = _pipeline(
            {
                "inputs": {"a": raster_input()},
                "steps": [{"op": "filter", "input": "a", "where": "calendar_year == 9999"}],
                "output": {"format": "tif"},
            }
        )
        out_path = base_path.parent / "should_not_exist.tif"
        assert run_to_path(pipeline, out_path, base_path=base_path) is False
        assert not out_path.exists()
