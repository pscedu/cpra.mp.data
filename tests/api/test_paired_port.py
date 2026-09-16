"""Tests for the paired shape's two output paths being built from the general
`pivot_vector_to_band_wide` + `scan_rasters` pair (see `cpra/mp/data/analysis/paired.py`'s
module docstring for the motivation and scope).

Two things this file exists to prove, neither covered elsewhere:

* Both paths -- the raster WRITE and the map TILE -- actually EXECUTE those operations, and
  the bespoke paired source and hand-rolled tile loop they replaced are gone rather than
  merely bypassed. Not merely that their output happens to look right, which a bug that
  silently fell back to some other code path could also produce
  (`TestOutputPathsUseTheGeneralOps`).
* The ported paths' actual pixel output is IDENTICAL to the pre-port hand-rolled collapse
  across every supported aggregate, including `count`'s zero-vs-nodata split and its extra
  out-of-domain (pairing-miss) case (`TestPortMatchesReferenceImplementation`,
  `TestTilePortMatchesTheWarpFirstImplementation`).

THE PRE-PORT IMPLEMENTATIONS LIVE IN THIS FILE, not in `paired.py`. They were kept in
production for a while as "pinned reference implementations", which is a fair description of
what they are FOR but not of where they belong: nothing shipped calls them, so leaving them
next to the code they are meant to check makes them look like a live alternative path and
invites the two to drift under one edit. Moved verbatim (`_reference_collapse_paired_window_
wide`, `_reference_accumulate_paired_window`, plus the two per-band evaluators they call and
the dense lookup arrays they read), they are exactly what a differential oracle should be:
a second implementation the code under test cannot accidentally share a bug with.

`tests/api/test_pipeline_exec.py` already pins exact hand-computed numeric expectations
for these aggregates through the real ASGI routes (`TestPairedAccumulatorEdgeCases`,
`TestPairedAnalysisCountConsistency`) -- those values are not re-derived here, and this
file does not import anything from that one (mirroring that module's own "reproduced
here rather than imported" convention for NOTEBOOK_QUERY, so the two files stay
independently editable). This file instead proves a different, more direct thing for a
pure refactor: the new code produces exactly what the old code already produced, on the
exact same input.
"""

from contextlib import ExitStack
from io import BytesIO

import mercantile
import numpy as np
import polars as pl
import pytest
import rasterio as rio
from PIL import Image
from rasterio.coords import BoundingBox
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from cpra.mp.data import config
from cpra.mp.data.analysis import paired as paired_module
from cpra.mp.data.analysis.paired import _count_zone_gate, _legend_for
from cpra.mp.data.analysis.pipeline import parse_pipeline
from cpra.mp.data.analysis.planner import compile_pipeline
from cpra.mp.data.analysis.sources import (
    _batched_windows,
    _crosswalk_lookup_values,
    _dense_lookup_array,
)
from cpra.mp.data.analysis.utils import raster_blocks_per_batch, raster_gdal_env
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.raster import write_cog
from cpra.mp.data.tabular import write_parquet
from tests.factories import write_raster_crosswalk

# Real variable/grid names, matching test_pipeline_exec.py's NOTEBOOK_QUERY fixtures
# byte-for-byte (this is the one combination `tests/factories.py` has no ready-made
# builder for and the spec's own exit-criterion example uses) -- reproduced here rather
# than imported, see the module docstring.
RASTER_INPUT_KWARGS = dict(
    variable="ffibs_coverage_ratio",
    grid="veg_grid_cell_v001",
    time_unit="annual",
    model_group_id=500,
    scenario_id=7,
    ffibs_type="pl_fm",
    model="icm_v2023.0.0",
)

VECTOR_INPUT_KWARGS = dict(
    variable="sal",
    grid="hydro_compartment_v001",
    time_unit="annual",
    model_group_id=500,
    scenario_id=7,
    aggregate_type="2_wk_max",
    model="icm_v2023.0.0",
)


def _pipeline_query(aggregate):
    """The notebook-shape query (project/join/filter/group), varying only the trailing
    `group` step's aggregate. `count` has no aggregand (`PairedAnalysis._prepare`
    rejects a non-default `of` alongside it), so `of` is omitted entirely for it, not
    just set to something harmless.
    """
    group_step = {"op": "group", "by": [], "aggregate": aggregate}
    if aggregate != "count":
        group_step["of"] = "calendar_year"
    return {
        "inputs": {
            "marsh": {k: RASTER_INPUT_KWARGS[k] for k in ("variable", "grid", "time_unit", "model_group_id", "scenario_id", "ffibs_type")},
            "salinity": {
                k: VECTOR_INPUT_KWARGS[k]
                for k in ("variable", "grid", "time_unit", "model_group_id", "scenario_id", "aggregate_type")
            },
        },
        "steps": [
            {"op": "project", "input": "salinity", "to": "veg_grid_cell_v001", "as": "sal"},
            {"op": "join", "type": "spatial", "inputs": ["marsh", "sal"], "as": "paired"},
            {"op": "filter", "where": "marsh > 0.5 AND sal > 5.5"},
            group_step,
        ],
        "output": {"format": "tif"},
    }


@pytest.fixture
def paired_port_data(base_path, crosswalk_path, monkeypatch):
    """Real raster-grid/vector-grid datasets with five equal row bands, each
    isolating one thing that matters to at least one aggregate -- `sal` (zone 1) is a
    uniform 6.0 in both years (always satisfies the query's `sal > 5.5`), so every
    band's outcome below is driven purely by `marsh`/raster-nodata/pairing, mirroring
    `test_pipeline_exec.py`'s `accumulator_edge_case_data`/`count_edge_case_data`
    fixtures (reproduced here, not imported -- see the module docstring):

    * rows [0, h/5): `marsh` is 0.3 in both years -- never satisfies `marsh > 0.5`, in
      the pairing crosswalk's domain, zero contributing bands.
    * rows [h/5, 2h/5): `marsh` is `write_cog`'s own raster-nodata sentinel (-9999) in
      both years -- in-domain, zero contributing bands via raster nodata rather than a
      failed `where`.
    * rows [2h/5, 3h/5): `marsh` satisfies `where` in 2030 only (0.7 vs 0.2) -- exactly
      one contributing band.
    * rows [3h/5, 4h/5): `marsh` satisfies `where` in both years (0.6, 0.6) -- two
      contributing bands.
    * rows [4h/5, h): would satisfy `where` in both years (0.6, 0.6) exactly like the
      band above, but the crosswalk's own nodata sentinel covers this band -- OUTSIDE
      the pairing crosswalk's domain entirely, proving the exclusion comes from the
      crosswalk, not from `marsh` itself.
    """
    grid = GRIDS["veg_grid_cell_v001"]
    fifth = grid.height // 5
    marsh_2030 = np.empty((grid.height, grid.width), dtype="float32")
    marsh_2040 = np.empty((grid.height, grid.width), dtype="float32")

    marsh_2030[:fifth, :] = 0.3
    marsh_2040[:fifth, :] = 0.3
    marsh_2030[fifth : 2 * fifth, :] = -9999.0
    marsh_2040[fifth : 2 * fifth, :] = -9999.0
    marsh_2030[2 * fifth : 3 * fifth, :] = 0.7
    marsh_2040[2 * fifth : 3 * fifth, :] = 0.2
    marsh_2030[3 * fifth : 4 * fifth, :] = 0.6
    marsh_2040[3 * fifth : 4 * fifth, :] = 0.6
    marsh_2030[4 * fifth :, :] = 0.6
    marsh_2040[4 * fifth :, :] = 0.6

    for year, data in ((2030, marsh_2030), (2040, marsh_2040)):
        write_cog(data.astype("float32"), base_path=base_path, calendar_year=year, **RASTER_INPUT_KWARGS)

    years = (2030, 2040)
    df = pl.DataFrame(
        {
            "variable": pl.Series([VECTOR_INPUT_KWARGS["variable"]] * len(years), dtype=pl.String),
            "grid": pl.Series([VECTOR_INPUT_KWARGS["grid"]] * len(years), dtype=pl.String),
            "time_unit": pl.Series([VECTOR_INPUT_KWARGS["time_unit"]] * len(years), dtype=pl.String),
            "model_group_id": pl.Series([VECTOR_INPUT_KWARGS["model_group_id"]] * len(years), dtype=pl.Int32),
            "scenario_id": pl.Series([VECTOR_INPUT_KWARGS["scenario_id"]] * len(years), dtype=pl.Int32),
            "aggregate_type": pl.Series([VECTOR_INPUT_KWARGS["aggregate_type"]] * len(years), dtype=pl.String),
            "calendar_year": pl.Series(list(years), dtype=pl.Int32),
            "model": pl.Series([VECTOR_INPUT_KWARGS["model"]] * len(years), dtype=pl.String),
            "1": pl.Series([6.0, 6.0], dtype=pl.Float32),
        }
    )
    write_parquet(df, base_path=base_path, validate=False)
    monkeypatch.setattr(config, "BASE_PATH", base_path)

    zones = np.ones((grid.height, grid.width), dtype="int32")
    zones[4 * fifth :, :] = 0  # write_raster_crosswalk's own nodata sentinel
    write_raster_crosswalk(crosswalk_path, "veg_grid_cell_v001", "hydro_compartment_v001", zones)
    return base_path, fifth


def _reference_lookup_arrays(plan):
    """The dense per-calendar-step lookup arrays `PairedAnalysis._prepare` used to build and
    hang on the plan as `plan.lookup_arrays`, rebuilt here from the long vector frame the
    ported paths pivot instead.

    Keyed by `_vector_long`'s own `_step` vocabulary, which is what `plan._vector_band_steps`
    indexes bands in -- so the old `lookup_arrays.get(calendar_values[i], ...get(None))`
    double-lookup (a `None` key standing in for "no shared calendar dimension") collapses to
    a plain `arrays[plan._vector_band_steps[i]]`: with no calendar dimension both sides are
    the single sentinel step `0`.
    """
    arrays = {}
    for (step,), group in plan._vector_long.group_by("_step"):
        ids = group["grid_cell"].to_numpy()
        values = group["value"].to_numpy().astype("float64")
        arrays[step] = _dense_lookup_array(ids, values)
    return arrays


def _reference_band_valid_mask(plan, i, raster_flat, paired_flat):
    """Band `i`'s boolean valid mask, as a flat NumPy array -- the numpy-accumulator
    counterpart to `PairedAnalysis._band_valid_expr` (mirrors `_reference_band_masked_value`'s
    own relationship to `_band_value_expr`: evaluated over a throwaway two-column frame
    instead of the full wide one, so `_reference_accumulate_paired_window` never holds more
    than one band's arrays alive at once). See `_reference_band_masked_value`'s docstring for
    why `fill_nan(None)` is needed before evaluating either expression over `raster_flat`/
    `paired_flat`.
    """
    raster_col, paired_col = plan.raster_cols[i], plan.paired_cols[i]
    df = pl.DataFrame({raster_col: raster_flat, paired_col: paired_flat})
    df = df.with_columns([pl.col(c).fill_nan(None) for c in (raster_col, paired_col)])
    result = df.select(plan._band_valid_expr(i).alias("v"))
    return result["v"].to_numpy()


def _reference_band_masked_value(plan, i, raster_flat, paired_flat):
    """Band `i`'s masked collapse value ALONE, as a flat float64 NumPy array (`NaN` standing
    in for Polars null -- the same substitution `fill_nan(None)` undoes on the way in and
    `to_numpy()` reintroduces on the way out). This is exactly what band `i` contributes to
    `PairedAnalysis._collapse_expr`'s wide `horizontal_aggregate_expr` call, evaluated over a
    throwaway two-column, single-band frame instead of the full `2 * len(band_sources)`-column
    one, so `_reference_accumulate_paired_window` never holds more than one band's worth of
    source arrays alive at once. `raster_flat`/`paired_flat` carry `NaN` for
    nodata/pairing-miss on entry; converting those to true Polars nulls with `fill_nan(None)`
    before evaluating the expression matters because Polars does NOT treat float `NaN` as null
    on its own (`NaN > x` is `True`, `NaN.is_null()` is `False`) -- skipping this step would
    silently change which pixels the where-clause and the nodata-mask treat as missing.

    Calls the plan's OWN `_band_value_expr`, deliberately: what this oracle reimplements is
    the per-window DRIVE (which pixels, in what order, folded how), not the where-clause and
    value-selection semantics, and re-deriving those in NumPy would make the differential
    fail for reasons that have nothing to do with the port.
    """
    raster_col, paired_col = plan.raster_cols[i], plan.paired_cols[i]
    df = pl.DataFrame({raster_col: raster_flat, paired_col: paired_flat})
    df = df.with_columns([pl.col(c).fill_nan(None) for c in (raster_col, paired_col)])
    result = df.select(plan._band_value_expr(i).alias("v"))
    return result["v"].to_numpy().astype("float64")


def _reference_band_columns(plan, lookup_arrays, raster_srcs, window, pairing_ids, pairing_valid, i):
    """Band `i`'s `(raster_flat, paired_flat)` pair for one window, `NaN` for raster-nodata
    and for a pairing miss -- the read both reference collapses share."""
    path, band = plan.band_sources[i]
    src = raster_srcs[path]
    raster_flat = src.read(band, window=window).ravel().astype("float64")
    if src.nodata is not None:
        raster_flat = np.where(raster_flat == src.nodata, np.nan, raster_flat)
    lookup = lookup_arrays[plan._vector_band_steps[i]]
    return raster_flat, _crosswalk_lookup_values(lookup, pairing_ids, pairing_valid)


def _reference_collapse_paired_window_wide(
    plan, lookup_arrays, raster_srcs, window, pairing_ids, pairing_valid, shape, collapse_expr
):
    """The pre-port per-window collapse, moved here from `paired.py` (see the module
    docstring): every raster band and its paired lookup read into one
    `2 * len(band_sources)`-column frame, then `plan._collapse_expr()` evaluated once across
    all of it.
    """
    cols = {}
    for i in range(len(plan.band_sources)):
        raster_flat, paired_flat = _reference_band_columns(
            plan, lookup_arrays, raster_srcs, window, pairing_ids, pairing_valid, i
        )
        cols[plan.raster_cols[i]] = raster_flat
        cols[plan.paired_cols[i]] = paired_flat

    df = pl.DataFrame(cols)
    df = df.with_columns([pl.col(c).fill_nan(None) for c in plan.raster_cols + plan.paired_cols])
    result = df.select(collapse_expr)
    values = _count_zone_gate(plan, result["value"].to_numpy(), pairing_valid)
    return values.reshape(shape)


def _reference_accumulate_paired_window(plan, lookup_arrays, raster_srcs, window, pairing_ids, pairing_valid, shape):
    """Per-band streaming counterpart to `_reference_collapse_paired_window_wide`, also moved
    here verbatim from `paired.py`: folds one band's masked value into a running accumulator,
    so at most one band's raster/paired arrays are ever alive at once instead of all
    `len(band_sources)` of them plus the wide Polars frame built from them.

    Bit-identical to `horizontal_aggregate_expr`'s own null-skipping reduction (see
    sources.py): `min`/`max` seed their running accumulator at +-inf so a pixel no band ever
    validates for never updates it; `sum` seeds at 0 and only adds a band's value where that
    band is valid (Polars' own `sum_horizontal` treats a null band the same way -- as a zero
    contribution, not as excluding the row); `mean` accumulates a running sum and a running
    per-pixel valid-band COUNT and divides at the end, which is what makes it differ from
    `sum` (dividing by contributor count, not by `len(band_sources)`). `any_valid` -- OR'd
    across every band -- is this function's counterpart to `horizontal_aggregate_expr`'s
    `no_contributor_at` (`all_horizontal` over the per-band null masks): wherever no band ever
    validated, the pixel gets the plan's own `nodata_value` instead of whatever the (otherwise
    untouched, +-inf/0) accumulator seed happens to hold.

    `count` is handled separately from the `any_valid`-gated aggregates above: it folds
    `_reference_band_valid_mask` into a plain running integer tally rather than a
    `min`/`max`/`sum`-style accumulator, and is gated by `_count_zone_gate`'s `pairing_valid`
    check, NOT by `any_valid` -- see that function's own docstring for why those two gates
    disagree on purpose (a real 0 tally is `count`'s correct answer for an in-domain pixel;
    the `any_valid`-gated aggregates have no equivalent "real 0" to protect).
    """
    aggregate = plan.group_step.aggregate
    flat_shape = pairing_ids.shape

    if aggregate == "count":
        tally = np.zeros(flat_shape, dtype="int64")
        for i in range(len(plan.band_sources)):
            raster_flat, paired_flat = _reference_band_columns(
                plan, lookup_arrays, raster_srcs, window, pairing_ids, pairing_valid, i
            )
            tally += _reference_band_valid_mask(plan, i, raster_flat, paired_flat).astype("int64")

        return _count_zone_gate(plan, tally, pairing_valid).reshape(shape)

    if aggregate == "min":
        acc = np.full(flat_shape, np.inf, dtype="float64")
    elif aggregate == "max":
        acc = np.full(flat_shape, -np.inf, dtype="float64")
    else:  # sum / mean
        acc = np.zeros(flat_shape, dtype="float64")
    count = np.zeros(flat_shape, dtype="int64") if aggregate == "mean" else None
    any_valid = np.zeros(flat_shape, dtype=bool)

    for i in range(len(plan.band_sources)):
        raster_flat, paired_flat = _reference_band_columns(
            plan, lookup_arrays, raster_srcs, window, pairing_ids, pairing_valid, i
        )
        masked = _reference_band_masked_value(plan, i, raster_flat, paired_flat)
        valid = ~np.isnan(masked)
        any_valid |= valid

        if aggregate == "min":
            acc = np.minimum(acc, np.where(valid, masked, np.inf))
        elif aggregate == "max":
            acc = np.maximum(acc, np.where(valid, masked, -np.inf))
        else:
            acc = acc + np.where(valid, masked, 0.0)
            if aggregate == "mean":
                count += valid

    if aggregate == "mean":
        safe_count = np.where(count == 0, 1, count)
        acc = acc / safe_count

    result = np.where(any_valid, acc, plan.nodata_value)
    return result.reshape(shape)


def _pairing_domain(pairing_src, window):
    """`(pairing_ids, pairing_valid, shape)` for one window of the pairing crosswalk."""
    pairing_raw = pairing_src.read(1, window=window)
    pairing_ids = pairing_raw.ravel().astype("int64")
    if pairing_src.nodata is not None:
        pairing_valid = pairing_ids != int(pairing_src.nodata)
    else:
        pairing_valid = np.ones(pairing_ids.shape, dtype=bool)
    return pairing_ids, pairing_valid, pairing_raw.shape


def _reference_write_paired_raster(plan, out_path, force_wide=False):
    """A verbatim copy of the body `_write_paired_notebook_raster` had before it was
    rebuilt onto the shared ops: the hand-rolled rasterio block-window loop dispatching
    directly onto the two reference collapses above, reading `raster_srcs`/`pairing_src`
    itself rather than through a Polars source. This is the "old" side of this file's
    write-path differential -- see the module docstring.

    `force_wide` is the parameter that replaced `paired_module._STREAMING_AGGREGATES`. That
    was a module-level frozenset in production which a test monkeypatched to route the old
    write path onto the wide collapse; with the dispatch it fed gone from production and both
    collapses living here, it is an ordinary argument to the only thing that ever branched
    on it.
    """
    with ExitStack() as stack, raster_gdal_env():
        ref_path, _ref_band = plan.band_sources[0]
        ref_src = stack.enter_context(rio.open(ref_path))
        profile = ref_src.profile.copy()
        block_height, block_width = ref_src.block_shapes[0]
        itemsize = np.dtype(plan.out_dtype).itemsize
        n_units = max(len(plan.band_sources) * 2, 1)
        blocks_per_batch = raster_blocks_per_batch(n_units, block_height, block_width, itemsize)
        window_list = list(_batched_windows(ref_src, blocks_per_batch))

        raster_srcs = {path: stack.enter_context(rio.open(path)) for path in {p for p, _ in plan.band_sources}}
        pairing_src = stack.enter_context(rio.open(str(plan.pairing_crosswalk_path)))

        profile.update(count=1, dtype=plan.out_dtype, nodata=plan.nodata_value, compress="deflate")
        dst = stack.enter_context(rio.open(str(out_path), "w", **profile))

        lookup_arrays = _reference_lookup_arrays(plan)
        collapse_expr = plan._collapse_expr() if force_wide else None

        for window in window_list:
            pairing_ids, pairing_valid, shape = _pairing_domain(pairing_src, window)
            if force_wide:
                values = _reference_collapse_paired_window_wide(
                    plan, lookup_arrays, raster_srcs, window, pairing_ids, pairing_valid, shape, collapse_expr
                )
            else:
                values = _reference_accumulate_paired_window(
                    plan, lookup_arrays, raster_srcs, window, pairing_ids, pairing_valid, shape
                )
            dst.write(values.astype(plan.out_dtype), 1, window=window)


def _grid_tile(base_path, zoom=10):
    """`(x, y, z)` for a tile over the middle of the raster grid's own extent."""
    grid = GRIDS["veg_grid_cell_v001"]
    west, south, east, north = transform_bounds(grid.crs, "EPSG:4326", *grid.bounds)
    tile = mercantile.tile((west + east) / 2, (south + north) / 2, zoom)
    return tile.x, tile.y, tile.z


def _spy(monkeypatch, attr):
    """Record every call to `paired_module.<attr>` while still running the real thing."""
    calls = []
    real = getattr(paired_module, attr)

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(paired_module, attr, spy)
    return calls


def _capture_returns(monkeypatch, attr):
    """Like `_spy`, but records what `paired_module.<attr>` RETURNED.

    Reaching the tile path's collapsed pixel array through the function that warps it is the
    only way to compare numbers: `get_tile` hands that array straight to a legend and returns
    a PNG, and a legend maps whole ranges onto one color.
    """
    results = []
    real = getattr(paired_module, attr)

    def capture(*args, **kwargs):
        result = real(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(paired_module, attr, capture)
    return results


class TestOutputPathsUseTheGeneralOps:
    """Proves BOTH output paths actually drive the two general operations they are documented
    to be built from -- `pivot_vector_to_band_wide` and `scan_rasters` -- not merely that
    their output happens to be correct, which a bug that silently kept some other code path
    alive alongside them could also produce.
    """

    async def test_write_raster_calls_pivot_and_scan_rasters(self, paired_port_data, monkeypatch, tmp_path):
        base_path, _fifth = paired_port_data
        pipeline = parse_pipeline(_pipeline_query("sum"))
        plan = compile_pipeline(pipeline, base_path=base_path)

        pivot_calls = _spy(monkeypatch, "pivot_vector_to_band_wide")
        scan_calls = _spy(monkeypatch, "scan_rasters")

        out_path = tmp_path / "ported.tif"
        result = await plan.write_raster(str(out_path))
        assert result is True
        assert len(pivot_calls) == 1, "the write path must pivot the vector frame exactly once per write"
        assert len(scan_calls) == 1, "the write path must call scan_rasters exactly once per write"

        # The scan is the GENERAL one, carrying the pairing crosswalk as an ordinary
        # `GridCrosswalk` with the pivoted table attached -- not a paired-specific source.
        (crosswalk,) = scan_calls[0][1]["crosswalks"]
        assert crosswalk.name == paired_module.PAIRED_CROSSWALK_NAME
        assert crosswalk.values is not None, "the pivoted table must reach the scan"
        assert crosswalk.values.columns == [paired_module.PAIRED_CROSSWALK_NAME, *plan.paired_cols]

        grid = GRIDS["veg_grid_cell_v001"]
        with rio.open(out_path) as src:
            data = src.read(1)
        assert data.shape == (grid.height, grid.width)

    def test_no_bespoke_paired_source_survives(self):
        """Complementary to the spy above: the old paired-specific source is GONE, not
        merely bypassed. A module-level rename that left the old function importable would
        let a stale caller keep working and silently diverge.
        """
        from cpra.mp.data.analysis import sources as sources_module

        assert not hasattr(sources_module, "scan_paired")
        assert not hasattr(sources_module, "PairedCrosswalk")
        assert not hasattr(paired_module, "scan_paired")

    def test_no_hand_rolled_paired_collapse_survives(self):
        """The same complement for the TILE port. Every piece of the pre-port paired
        machinery is gone from production -- the two window collapses and their per-band
        evaluators (this file owns them now, as differential oracles), the dense lookup
        arrays only the old tile loop read, and the streaming/wide dispatch knob that chose
        between the collapses.

        `WarpedVRT` specifically: `paired.py` no longer imports it at all, so nothing there
        can reproject a source -- not a raster band and, more importantly, not the pairing
        crosswalk, whose pixels are zone IDs.
        """
        assert not hasattr(paired_module, "_accumulate_paired_window")
        assert not hasattr(paired_module, "_collapse_paired_window_wide")
        assert not hasattr(paired_module, "_STREAMING_AGGREGATES")
        assert not hasattr(paired_module, "WarpedVRT")
        assert not hasattr(paired_module.PairedAnalysis, "_band_masked_value")
        assert not hasattr(paired_module.PairedAnalysis, "_band_valid_mask")

    async def test_prepare_no_longer_builds_dense_lookup_arrays(self, paired_port_data):
        """`PairedAnalysis._prepare` used to build TWO forms of the same vector selection --
        the long frame the write path pivots, and a dict of dense per-step lookup arrays only
        the old tile loop read. One consumer, one form: the second is gone, and the first is
        what both paths now run through `_paired_scan`.
        """
        base_path, _fifth = paired_port_data
        plan = compile_pipeline(parse_pipeline(_pipeline_query("sum")), base_path=base_path)
        plan._prepare()

        assert not hasattr(plan, "lookup_arrays")
        assert not hasattr(plan, "calendar_values")
        assert plan._vector_long.columns == ["grid_cell", "_step", "value"]
        assert len(plan._vector_band_steps) == len(plan.band_sources)

    async def test_count_reads_the_pairing_crosswalk_once_per_window(
        self, paired_port_data, monkeypatch, tmp_path
    ):
        """`count` needs the pairing crosswalk TWICE conceptually -- once for each band's
        paired value, once for `_count_zone_gate`'s in-domain test -- and the pre-unification
        shape paid for both, opening the crosswalk a second time in the scan's separate
        attach-only zone mode. Under one `GridCrosswalk` the zone column and the value
        columns come out of the SAME read, so the file is opened exactly as many times as a
        non-`count` write opens it.
        """
        base_path, _fifth = paired_port_data

        opens = []
        real_open = rio.open

        def counting_open(path, *args, **kwargs):
            opens.append(str(path))
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(rio, "open", counting_open)

        counts = {}
        for aggregate in ("sum", "count"):
            plan = compile_pipeline(parse_pipeline(_pipeline_query(aggregate)), base_path=base_path)
            opens.clear()
            assert await plan.write_raster(str(tmp_path / f"{aggregate}.tif")) is True
            counts[aggregate] = sum(1 for p in opens if p.endswith("veg_grid_cell_v001__hydro_compartment_v001.tif"))

        assert counts["sum"] > 0, "the pairing crosswalk must actually be read"
        assert counts["count"] == counts["sum"], (
            f"count opened the pairing crosswalk {counts['count']} times vs "
            f"{counts['sum']} for sum -- the zone gate must reuse the crosswalk read the "
            "value columns already did, not add one of its own"
        )

    async def test_get_tile_calls_the_same_pivot_and_scan(self, paired_port_data, monkeypatch):
        """The tile path runs the SAME two ops the write path does -- it used to run neither,
        because it warped every source onto the tile grid first and then hand-rolled its own
        collapse. It restricts the scan to one window rather than using a different scan.
        """
        base_path, _fifth = paired_port_data
        plan = compile_pipeline(parse_pipeline(_pipeline_query("sum")), base_path=base_path)

        pivot_calls = _spy(monkeypatch, "pivot_vector_to_band_wide")
        scan_calls = _spy(monkeypatch, "scan_rasters")
        window_calls = _spy(monkeypatch, "tile_source_window")
        warp_calls = _spy(monkeypatch, "warp_window_to_tile")

        image = await plan.get_tile(*_grid_tile(base_path), "png", tile_size=256)
        assert image is not None
        assert len(pivot_calls) == 1, "the tile path must pivot the vector frame exactly once"
        assert len(scan_calls) == 1, "the tile path must call scan_rasters exactly once"
        assert len(window_calls) == 1
        assert len(warp_calls) == 1

        # The scan is the general one, carrying the pairing crosswalk as an ordinary
        # `GridCrosswalk` -- identical to the write path's -- restricted to one window.
        (crosswalk,) = scan_calls[0][1]["crosswalks"]
        assert crosswalk.name == paired_module.PAIRED_CROSSWALK_NAME
        assert crosswalk.values is not None
        (window,) = scan_calls[0][1]["read_windows"]

        # A window, not the whole raster: the point of pushing tile bounds into the read.
        grid = GRIDS["veg_grid_cell_v001"]
        assert window.width * window.height < grid.width * grid.height

    async def test_a_tile_that_misses_the_pixels_renders_nodata(self, paired_port_data):
        """z=0 covers the whole world, so it overlaps the (rectangular, reprojected) dataset
        bounds; the source window it reduces to may still miss every pixel. That is a
        rendered tile of nothing, not a missing tile -- and not a crash.
        """
        base_path, _fifth = paired_port_data
        plan = compile_pipeline(parse_pipeline(_pipeline_query("sum")), base_path=base_path)
        image = await plan.get_tile(0, 0, 0, "png", tile_size=256)
        assert image is not None
        assert image[:8] == b"\x89PNG\r\n\x1a\n"


class TestPortMatchesReferenceImplementation:
    """Differential test: the ported write path's actual COG output must be
    byte-identical to the pre-port hand-rolled collapse's output, on the same input,
    for every supported aggregate and for both of the old implementation's two
    branches (`_reference_accumulate_paired_window` streaming and
    `_reference_collapse_paired_window_wide` wide-frame fallback).
    """

    @pytest.mark.parametrize("aggregate", ["min", "max", "sum", "mean", "count"])
    @pytest.mark.parametrize("force_wide_reference", [False, True], ids=["streaming", "wide"])
    async def test_port_matches_old_implementation(
        self, paired_port_data, tmp_path, aggregate, force_wide_reference
    ):
        base_path, _fifth = paired_port_data
        pipeline = parse_pipeline(_pipeline_query(aggregate))
        plan = compile_pipeline(pipeline, base_path=base_path)

        ported_path = tmp_path / f"ported_{aggregate}_{force_wide_reference}.tif"
        reference_path = tmp_path / f"reference_{aggregate}_{force_wide_reference}.tif"

        result = await plan.write_raster(str(ported_path))
        assert result is True
        _reference_write_paired_raster(plan, reference_path, force_wide=force_wide_reference)

        with rio.open(ported_path) as src:
            ported_data = src.read(1)
            ported_nodata = src.nodata
            ported_dtype = src.dtypes[0]
        with rio.open(reference_path) as src:
            reference_data = src.read(1)
            reference_nodata = src.nodata
            reference_dtype = src.dtypes[0]

        assert ported_dtype == reference_dtype
        assert ported_nodata == reference_nodata
        assert np.array_equal(ported_data, reference_data), (
            f"ported vs reference mismatch for aggregate={aggregate!r} "
            f"force_wide_reference={force_wide_reference!r}"
        )


def _reference_paired_tile(plan, x, y, z, tile_size=256):
    """The pre-port tile geometry, verbatim from `paired.py`: one `WarpedVRT` per raster band
    AND one over the pairing crosswalk, all onto the tile's own pixel grid, then the wide
    collapse over those tile pixels.

    Returns the collapsed value array -- not a PNG -- so the differential below compares
    numbers rather than colors (a legend maps whole ranges onto one color, which is exactly
    the resolution at which a mis-collapsed pixel disappears).
    """
    tile_bounds = BoundingBox(*mercantile.xy_bounds(x, y, z))
    lookup_arrays = _reference_lookup_arrays(plan)

    with ExitStack() as stack:
        raster_srcs = [stack.enter_context(rio.open(path)) for path, _ in plan.band_sources]
        pairing_src = stack.enter_context(rio.open(str(plan.pairing_crosswalk_path)))
        tile_transform = from_bounds(*tile_bounds, tile_size, tile_size)

        cols = {}
        for i, ((_path, band), src) in enumerate(zip(plan.band_sources, raster_srcs)):
            raster_nodata = src.nodata if src.nodata is not None else -9999
            vrt = stack.enter_context(
                WarpedVRT(
                    src,
                    crs="EPSG:3857",
                    transform=tile_transform,
                    width=tile_size,
                    height=tile_size,
                    nodata=raster_nodata,
                    add_alpha=False,
                )
            )
            data = vrt.read(band).ravel().astype("float64")
            cols[plan.raster_cols[i]] = np.where(data == raster_nodata, np.nan, data)

        pairing_nodata = pairing_src.nodata if pairing_src.nodata is not None else -1
        pairing_vrt = stack.enter_context(
            WarpedVRT(
                pairing_src,
                crs="EPSG:3857",
                transform=tile_transform,
                width=tile_size,
                height=tile_size,
                nodata=pairing_nodata,
                add_alpha=False,
            )
        )
        pairing_ids = pairing_vrt.read(1).ravel().astype("int64")
        pairing_valid = pairing_ids != int(pairing_nodata)

        for i in range(len(plan.band_sources)):
            cols[plan.paired_cols[i]] = _crosswalk_lookup_values(
                lookup_arrays[plan._vector_band_steps[i]], pairing_ids, pairing_valid
            )

    df = pl.DataFrame(cols)
    df = df.with_columns([pl.col(c).fill_nan(None) for c in plan.raster_cols + plan.paired_cols])
    result = df.select(plan._collapse_expr())
    values = _count_zone_gate(plan, result["value"].to_numpy(), pairing_valid)
    return values.astype(plan.out_dtype, copy=False).reshape((1, tile_size, tile_size))


class TestTilePortMatchesTheWarpFirstImplementation:
    """Differential on real pixel arrays for the TILE path: collapse-then-warp against the
    pre-port warp-then-collapse, for every supported aggregate.

    The two orders agree only because nothing here resamples anything: `warp_window_to_tile`
    uses `Resampling.nearest`, so each tile pixel takes one whole source pixel, and the
    collapse is per-pixel across bands. `tests/api/test_tile_warp.py` measures that geometry
    directly -- including the control that proves the measurement can fail; this file's job is
    the paired COLLAPSE being unchanged by the reordering, over a fixture whose five row
    bands each isolate one thing the collapse has to get right (a failed `where`, raster
    nodata, a partial year, a full year, and a pairing miss).
    """

    @pytest.mark.parametrize("aggregate", ["min", "max", "sum", "mean", "count"])
    async def test_tile_pixels_match(self, paired_port_data, monkeypatch, aggregate):
        base_path, _fifth = paired_port_data
        plan = compile_pipeline(parse_pipeline(_pipeline_query(aggregate)), base_path=base_path)

        warped = _capture_returns(monkeypatch, "warp_window_to_tile")
        x, y, z = _grid_tile(base_path)
        assert await plan.get_tile(x, y, z, "png", tile_size=256) is not None
        (ported,) = warped

        reference = _reference_paired_tile(plan, x, y, z, tile_size=256)

        assert ported.shape == reference.shape == (1, 256, 256)
        real = reference != plan.nodata_value
        assert real.sum() > 1000, "the reference tile is empty -- nothing was compared"
        assert np.array_equal(ported, reference), (
            f"aggregate={aggregate}: collapse-then-warp differs from warp-then-collapse at "
            f"{int((ported != reference).sum())} of {ported.size} pixels"
        )

    async def test_the_rendered_png_matches(self, paired_port_data):
        """End to end on the bytes a client receives, for the aggregate the notebook query
        actually uses."""
        base_path, _fifth = paired_port_data
        plan = compile_pipeline(parse_pipeline(_pipeline_query("max")), base_path=base_path)
        x, y, z = _grid_tile(base_path)

        body = await plan.get_tile(x, y, z, "png", tile_size=256)
        with Image.open(BytesIO(body)) as image:
            ported_pixels = np.array(image)

        reference = _reference_paired_tile(plan, x, y, z, tile_size=256)
        reference_pixels = _legend_for(plan.color_map).colorize_pixels(reference)

        assert np.array_equal(ported_pixels, reference_pixels)
