"""`api.unified` -- the unified decoder's orchestrator: the three things the pure half
(`analysis.request.unified`, tested in `test_request_unified.py`) structurally cannot do.

Resolving `analysis=` needs the registry, which imports `request/`; resolving a coordinate
`grid_cell` needs `io/`, which `request/` may not import. Both are called from here, and
the assertions below are about the resolved `Pipeline` rather than a response body --
a mispartitioned `model_group_id` is a 200 with the wrong data and invisible at the
response level.

The suite runs under `asyncio_mode = "auto"`, so `async def test_...` needs no marker.
"""

import pytest
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.transform import xy as raster_xy
from rasterio.warp import transform
from shapely.geometry import box

from cpra.mp.data.analyses.base import InvalidAnalysisParametersError
from cpra.mp.data.analysis.request.data_query import build_data_query_pipeline
from cpra.mp.data.analysis.request.errors import RequestError
from cpra.mp.data.analysis.request.parameters import (
    DataOptions,
    DatasetDimensions,
    FilterDimensions,
)
from cpra.mp.data.analysis.request.pipeline import pipeline_to_query
from cpra.mp.data.api.unified import decode_request, decode_request_sync
from cpra.mp.data.grids import GRIDS
from tests.factories import write_grid_geometry

VECTOR = {"variable": "ssf", "time_unit": "annual", "grid": "ecoregion_v001"}
RASTER = {"variable": "vct", "time_unit": "annual", "grid": "veg_grid_cell_v001"}


def _lng_lat(x, y):
    """Project an ecoregion_v001 (EPSG:26915) point to WGS84, as a `grid_cell=` value --
    the same helper `test_endpoints_data.py` uses. Written out rather than hardcoded
    because the fixture's box is in projected metres and a hand-picked lng/lat silently
    lands outside it, which reads as a passing "matched nothing" test."""
    lng, lat = [c[0] for c in transform(CRS.from_epsg(26915), CRS.from_epsg(4326), [x], [y])]
    return f"{lng!s},{lat!s}"


INSIDE = _lng_lat(500500, 3300500)  # the centre of `ecoregion_geometry`'s one cell
OUTSIDE = _lng_lat(400000, 3300500)  # 100km west of it, still a valid EPSG:26915 point


def _items(*pairs, **kwargs):
    """A query string as `(key, value)` pairs. Positional pairs come first so a repeated
    dimension key can be written alongside the scalar ones."""
    return [*pairs, *kwargs.items()]


@pytest.fixture
def ecoregion_geometry(geometry_path):
    """A single 1km square at ecoregion_v001 cell 1, for coordinate lookups -- the same
    fixture `test_endpoints_data.py` uses, redeclared rather than shared because it is two
    lines and importing a fixture across test modules is not something pytest does."""
    return write_grid_geometry(geometry_path, "ecoregion_v001", [(1, box(500000, 3300000, 501000, 3301000))])


class TestDownloadSentinel:
    async def test_the_flat_sugar_is_the_flat_route_s_own_pipeline(self):
        """The equivalence the whole consolidation rests on: the sugar and the route it
        replaces resolve to the SAME pipeline, asserted through `pipeline_to_query`, which
        is canonical (natural-sorted keys, defaults omitted)."""
        decoded = await decode_request(
            _items(("calendar_year", "2030"), ("calendar_year", "2040"), **VECTOR), format="csv"
        )
        old = build_data_query_pipeline(
            DatasetDimensions(**VECTOR),
            FilterDimensions(calendar_year=[2030, 2040]),
            DataOptions(),
            "csv",
        )
        assert pipeline_to_query(decoded.pipeline) == pipeline_to_query(old)

    async def test_a_selection_becomes_one_trailing_filter(self):
        decoded = await decode_request(_items(calendar_year="2030", where="value > 3", **VECTOR), format="csv")
        assert len(decoded.pipeline.steps) == 1
        assert decoded.pipeline.steps[0].where == "calendar_year = 2030 AND (value > 3)"
        assert decoded.pipeline.steps[0].input == "data"

    async def test_no_selection_leaves_a_stepless_pipeline(self):
        decoded = await decode_request(_items(**VECTOR), format="csv")
        assert decoded.pipeline.steps == []
        assert decoded.analysis is None

    async def test_nothing_selected_at_all_is_rejected_before_a_pipeline_is_built(self):
        """The sentinel's empty pipeline is never constructed -- `Pipeline.inputs` is
        `min_length=1`, and letting pydantic report it would answer a caller who selected
        nothing with a field error about a model they never mentioned."""
        with pytest.raises(RequestError, match="no dataset selected"):
            await decode_request([], format="csv")

    async def test_a_dotted_pipeline_needs_no_dataset_sugar(self):
        decoded = await decode_request(
            _items(
                **{
                    "input.data.variable": "ssf",
                    "input.data.time_unit": "annual",
                    "input.data.grid": "ecoregion_v001",
                }
            ),
            format="csv",
        )
        assert list(decoded.pipeline.inputs) == ["data"]

    async def test_a_partial_triple_names_what_is_missing(self):
        with pytest.raises(RequestError, match=r"missing \['grid'\]"):
            await decode_request(_items(variable="ssf", time_unit="annual"), format="csv")

    async def test_the_url_format_wins_over_a_dotted_one(self):
        decoded = await decode_request(_items(**VECTOR, **{"output.format": "tif"}), format="csv")
        assert decoded.pipeline.output.format == "csv"

    async def test_compression_is_decoded_but_left_for_the_route_to_validate(self):
        decoded = await decode_request(
            _items(compression_algorithm="gzip", compression_level="9", **VECTOR), format="csv"
        )
        assert decoded.compression.compression_algorithm == "gzip"
        assert decoded.compression.compression_level == 9


class TestNamedAnalysis:
    async def test_the_selection_goes_to_the_analysis_not_the_tail(self):
        """`diff`'s whole subject is two selected values of one dimension: it filters to
        them, negates one, and sums ACROSS the dimension -- which removes it from the
        frame. The same conjunct appended at the tail would name a column that no longer
        exists, which is why the selection is handed to the registry rather than grafted on
        from outside."""
        decoded = await decode_request(
            _items(
                ("calendar_year", "2030"),
                ("calendar_year", "2040"),
                analysis="diff",
                dimension="calendar_year",
                order="asc",
                **VECTOR,
            ),
            format="csv",
        )
        ops = [step.op for step in decoded.pipeline.steps]
        assert ops == ["filter", "case", "group"]
        assert decoded.pipeline.steps[0].where == "calendar_year IN (2030, 2040)"
        assert decoded.analysis is not None
        assert decoded.analysis.meta.name == "diff"

    async def test_the_analysis_s_own_parameters_are_the_leftovers(self):
        """No fixed field set at this layer -- which names are legal varies per analysis,
        and `resolve_analysis_chain` is what validates them against the stage that
        declared them."""
        with pytest.raises(InvalidAnalysisParametersError, match="nonsense"):
            await decode_request(
                _items(analysis="diff", dimension="calendar_year", order="asc", nonsense="1", **VECTOR),
                format="csv",
            )

    async def test_a_chain_resolves_stage_by_stage(self):
        decoded = await decode_request(
            _items(
                ("calendar_year", "2030"),
                ("calendar_year", "2040"),
                analysis="lnd_conf,diff",
                dimension="calendar_year",
                order="asc",
                variable="lnd_type",
                time_unit="annual",
                grid="ecoregion_v001",
            ),
            format="csv",
        )
        assert decoded.analysis.meta.name == "lnd_conf"
        # `lnd_conf`'s own steps, then `diff`'s two appended after them.
        assert [step.op for step in decoded.pipeline.steps][-2:] == ["case", "group"]

    async def test_a_fixed_input_analysis_takes_its_dimensions_as_filters(self):
        """The positive form of what used to be pinned here as a known mispartition.

        Both fixed-input analyses declared `model_group_id`/`scenario_id` as required
        `Parameters`, which are dimension NAMES -- and the partition routes by name, so they
        landed in `filters` and no query string could reach `params` at all: the analysis
        took its own defaults and answered for the wrong model group. B9 deleted the
        declarations, `analyses._register` forbids reintroducing one, and the values now
        arrive where the partition was always going to put them.
        """
        decoded = await decode_request(
            _items(analysis="lnd_type_plus", model_group_id="500", scenario_id="7"),
            format="tif",
        )
        assert decoded.pipeline.steps[-2].where == "model_group_id = 500 AND scenario_id = 7"
        assert decoded.partition.params == {}

    async def test_a_fixed_input_analysis_is_resolved_with_no_dataset(self):
        """What the decoder itself is responsible for: an analysis that names its own inputs
        gets `dataset=None`, so it reaches its own `reject_dataset` rather than being
        rejected here for naming no `variable`/`time_unit`/`grid`. `tif` because that is the
        sink its own pipeline declares -- a `csv` here is the shape rejection, not this."""
        decoded = await decode_request(_items(analysis="sal_veg_investigation"), format="tif")
        assert decoded.partition.dataset == {}
        assert set(decoded.pipeline.inputs) == {"coverage", "salinity"}

    async def test_an_unknown_analysis_is_reported_by_name(self):
        from cpra.mp.data.analyses.base import UnknownAnalysisError

        with pytest.raises(UnknownAnalysisError, match="nope"):
            await decode_request(_items(analysis="nope", **VECTOR), format="csv")

    async def test_the_url_format_overwrites_the_analysis_s_declared_one(self):
        decoded = await decode_request(
            _items(
                ("calendar_year", "2030"),
                ("calendar_year", "2040"),
                analysis="diff",
                dimension="calendar_year",
                order="asc",
                **VECTOR,
            ),
            format="pbf",
        )
        assert decoded.pipeline.output.format == "pbf"


class TestGridCell:
    async def test_digit_ids_need_no_geometry(self):
        """The literal-id spelling is resolved by `grid_cell_conjunct`, one definition
        shared with the flat routes -- no file is opened and no grid is consulted."""
        decoded = await decode_request(_items(grid_cell="1,2", **VECTOR), format="csv")
        assert decoded.pipeline.steps[0].where == "grid_cell IN (1, 2)"
        assert decoded.relabel is None

    async def test_coordinates_resolve_against_the_output_grid(self, ecoregion_geometry):
        """Two passes: the first exists only to read the output grid off the resolved
        pipeline, because for an analysis that grid is a fact about the pipeline and not
        about any input (`lnd_type_plus` projects onto a grid none of its inputs is on)."""
        decoded = await decode_request(_items(grid_cell=INSIDE, **VECTOR), format="csv")
        assert decoded.pipeline.steps[0].where == "grid_cell = 1"

    async def test_a_point_outside_every_polygon_matches_nothing(self, ecoregion_geometry):
        """Dropped, not rejected -- `resolve_vector_grid_cell`'s long-standing convention,
        which answers an empty result rather than a 400. Paired with the case above so the
        two spellings are told apart: a point that merely fails to project into the box
        would pass this assertion for the wrong reason."""
        decoded = await decode_request(_items(grid_cell=OUTSIDE, **VECTOR), format="csv")
        assert decoded.pipeline.steps[0].where == "grid_cell != grid_cell"

    async def test_a_raster_coordinate_carries_the_relabel_map_back(self):
        """The one case that produces a `relabel`: a raster `grid_cell` coordinate resolves
        to a pixel ID, and the response labels that column with the coordinate the caller
        actually asked for rather than the ID. Computed from `rasterio.transform` rather
        than hardcoded, so the assertion proves the round trip instead of restating
        `resolve_raster_pixel`'s arithmetic."""
        row, col = 10, 20
        grid = GRIDS[RASTER["grid"]]
        x, y = raster_xy(from_bounds(*grid.bounds, width=grid.width, height=grid.height), row, col)
        coordinate = _lng_lat(x, y)
        expected_id = row * grid.width + col

        decoded = await decode_request(_items(grid_cell=coordinate, **RASTER), format="csv")
        assert decoded.pipeline.steps[0].where == f"grid_cell = {expected_id}"
        assert decoded.relabel == {expected_id: [coordinate]}

    async def test_a_raster_point_selection_cannot_be_written_as_a_surface(self):
        """Already what the engine does -- the spatial filter lowers the final frame to
        long and `validate_shapes` rejects it against a raster format. The decoder runs
        `validate_shapes` so the rejection arrives here rather than at execution."""
        with pytest.raises(RequestError, match="grid_cell"):
            await decode_request(_items(grid_cell="-92.09,29.81", **RASTER), format="tif")

    def test_the_sync_form_agrees_with_the_async_one(self, ecoregion_geometry):
        """A visualization builds its spec from synchronous cached properties, so the two
        forms exist for the same reason `io.geography.resolve_vector_grid_cell_sync` does
        -- and must not drift."""
        decoded = decode_request_sync(_items(grid_cell=INSIDE, **VECTOR), format="csv")
        assert decoded.pipeline.steps[0].where == "grid_cell = 1"

    def test_the_sync_form_skips_the_probe_when_nothing_is_spatial(self):
        """The common case: no `grid_cell` means one pass, not two -- asserted through the
        result rather than a call count, since a second pass would be invisible here except
        as latency."""
        decoded = decode_request_sync(_items(calendar_year="2030", **VECTOR), format="csv")
        assert decoded.pipeline.steps[0].where == "calendar_year = 2030"


class TestDescriptor:
    async def test_the_descriptor_is_computed_once_and_carried(self):
        decoded = await decode_request(_items(**VECTOR), format="csv")
        assert decoded.descriptor.grid == "ecoregion_v001"
        assert decoded.descriptor.grid_type == "vector"
        assert decoded.descriptor.time_unit == "annual"

    async def test_float_precision_comes_from_the_one_presentation_variable(self):
        decoded = await decode_request(_items(**VECTOR), format="csv")
        assert decoded.float_precision == decoded.descriptor.variable.float_precision
