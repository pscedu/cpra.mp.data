"""Tests for how visualizations label and parse the geography they were asked for.

A visualization never reads data itself — it emits a spec pointing at a `data.csv` URL and
has to describe that response correctly in advance: which `geography` values to expect, and
whether they arrive as numbers or strings. Both answers change when a vector grid is
queried by coordinate rather than by grid cell ID, so they are pinned here. What the reader
is *shown* is a third answer, since a coordinate names a point but the data describes the
cell that contains it.
"""

import pytest
from rasterio.crs import CRS
from rasterio.warp import transform
from shapely.geometry import box

from cpra.mp.data.visualizations.base import (
    DatasetVisualization,
    geography_tokens,
    is_geography_ids,
)

from tests.factories import write_grid_geometry

ANNUAL_VECTOR = dict(variable="ssf", grid="ecoregion_v001", time_unit="annual")
ANNUAL_RASTER = dict(variable="vct", grid="veg_grid_cell_v001", time_unit="annual")

# Two coordinates inside cell 7, one inside cell 8, and one in neither.
INSIDE = (-91.5, 29.5)
ALSO_INSIDE = (-91.49, 29.49)
IN_CELL_8 = (-91.3, 29.5)
OUTSIDE = (-92.5, 30.5)


@pytest.fixture
def ecoregion_geometry(geometry_path):
    """10km squares around `INSIDE` and `IN_CELL_8`, as ecoregion_v001 cells 7 and 8.

    Each is big enough to hold its own coordinates — `ALSO_INSIDE` lands in cell 7 too, so
    the shared-cell case is reachable — and far enough from `OUTSIDE` that it isn't.
    """

    def square(lng, lat):
        x, y = [c[0] for c in transform(CRS.from_epsg(4326), CRS.from_epsg(26915), [lng], [lat])]
        return box(x - 5000, y - 5000, x + 5000, y + 5000)

    return write_grid_geometry(
        geometry_path,
        "ecoregion_v001",
        [(7, square(*INSIDE)), (8, square(*IN_CELL_8))],
    )


def _geography(*coords):
    """Flatten coordinate pairs into a `geography` query value."""
    return ",".join([str(c) for pair in coords for c in pair])


class TestGeographyTokens:
    def test_splits_a_comma_separated_string(self):
        assert geography_tokens("1,2,3") == ["1", "2", "3"]

    def test_strips_surrounding_whitespace(self):
        """unpack_values strips too, so the API's own ID/coordinate check sees the same
        tokens this one does."""
        assert geography_tokens(" 1 , 2 ") == ["1", "2"]

    def test_accepts_a_sequence(self):
        """Notebook callers pass a list rather than a query-string fragment."""
        assert geography_tokens([1, 2]) == ["1", "2"]


class TestIsGeographyIds:
    def test_digit_tokens_are_ids(self):
        assert is_geography_ids("1,2") is True

    def test_a_decimal_point_makes_it_coordinates(self):
        assert is_geography_ids("29.5,91.0") is False

    def test_a_sign_makes_it_coordinates(self):
        """A negative longitude has no decimal point but is still not an ID."""
        assert is_geography_ids("-91,29") is False

    def test_an_absent_geography_is_ids(self):
        """No filter means the whole grid, which is keyed by ID."""
        assert is_geography_ids(None) is True


class TestGeographies:
    """`geographies` has to equal what the API puts in its own `geography` column, since
    scale domains and lookup keys are matched against it."""

    def test_vector_ids_stay_integers(self):
        viz = DatasetVisualization(**ANNUAL_VECTOR, geography="1,2")
        assert viz.geographies == [1, 2]

    def test_vector_coordinates_resolve_to_cell_ids(self, ecoregion_geometry):
        """The data endpoint reports a vector geography by ID however it was selected, so a
        coordinate has to be resolved here to know what to match against."""
        viz = DatasetVisualization(**ANNUAL_VECTOR, geography=_geography(INSIDE, IN_CELL_8))
        assert viz.geographies == [7, 8]

    def test_coordinates_outside_the_grid_are_dropped(self, ecoregion_geometry):
        """The API omits them from the data, so keeping them here would leave a legend entry
        and a tooltip slot with nothing behind them."""
        viz = DatasetVisualization(**ANNUAL_VECTOR, geography=_geography(INSIDE, OUTSIDE))
        assert viz.geographies == [7]

    def test_coordinates_in_one_cell_collapse(self, ecoregion_geometry):
        """The response carries one series for the cell, so a second entry would be a key
        the data never matches."""
        viz = DatasetVisualization(**ANNUAL_VECTOR, geography=_geography(INSIDE, ALSO_INSIDE))
        assert viz.geographies == [7]

    def test_raster_coordinates_become_pair_labels(self):
        """A raster grid has no IDs, so its geography stays the coordinate it was given."""
        viz = DatasetVisualization(**ANNUAL_RASTER, geography="-91.5,29.5")
        assert viz.geographies == ["-91.5,29.5"]

    def test_raster_coordinates_are_normalized_through_float(self):
        """The API labels with str(float(token)), so `91` comes back as `91.0`."""
        viz = DatasetVisualization(**ANNUAL_RASTER, geography="91,29")
        assert viz.geographies == ["91.0,29.0"]


class TestGeographyLabels:
    """What the reader sees where there's room for it — the legend. A coordinate identifies
    the point asked about, but the values belong to the whole grid cell containing it, so
    both are shown."""

    def test_vector_ids_are_shown_bare(self):
        viz = DatasetVisualization(**ANNUAL_VECTOR, geography="1,2")
        assert viz.geography_labels == ["1", "2"]

    def test_vector_coordinates_are_shown_with_the_resolved_cell(self, ecoregion_geometry):
        viz = DatasetVisualization(**ANNUAL_VECTOR, geography=_geography(INSIDE))
        assert viz.geography_labels == ["7 (-91.5, 29.5)"]

    def test_a_shared_cell_is_labeled_with_the_first_coordinate(self, ecoregion_geometry):
        """The two coordinates are one series in the data, so they're one legend entry, and
        the first of them is what names it."""
        viz = DatasetVisualization(**ANNUAL_VECTOR, geography=_geography(INSIDE, ALSO_INSIDE))
        assert viz.geography_labels == ["7 (-91.5, 29.5)"]

    def test_raster_coordinates_are_shown_parenthesized(self):
        """A raster pixel has no ID to report, so the coordinate stands alone."""
        viz = DatasetVisualization(**ANNUAL_RASTER, geography="-91.5,29.5")
        assert viz.geography_labels == ["(-91.5, 29.5)"]

    def test_labels_stay_aligned_with_geographies(self, ecoregion_geometry):
        """The chart zips them together by position, so dropping and collapsing have to
        happen to all three lists at once."""
        viz = DatasetVisualization(
            **ANNUAL_VECTOR, geography=_geography(INSIDE, OUTSIDE, ALSO_INSIDE, IN_CELL_8)
        )
        assert (
            len(viz.geography_labels) == len(viz.geography_short_labels) == len(viz.geographies) == 2
        )


class TestGeographyShortLabels:
    """What goes where a full label would crowd out the numbers — a tooltip row."""

    def test_vector_ids_are_unchanged(self):
        viz = DatasetVisualization(**ANNUAL_VECTOR, geography="1,2")
        assert viz.geography_short_labels == ["1", "2"]

    def test_a_resolved_cell_drops_the_coordinate(self, ecoregion_geometry):
        """The ID is what the values describe; the legend still carries the coordinate that
        selected it."""
        viz = DatasetVisualization(**ANNUAL_VECTOR, geography=_geography(INSIDE))
        assert viz.geography_short_labels == ["7"]

    def test_a_shared_cell_appears_once(self, ecoregion_geometry):
        """One series in the data, so one tooltip row."""
        viz = DatasetVisualization(**ANNUAL_VECTOR, geography=_geography(INSIDE, ALSO_INSIDE))
        assert viz.geography_short_labels == ["7"]

    def test_raster_coordinates_are_kept(self):
        """There's no ID to fall back on, so dropping the coordinate would leave nothing."""
        viz = DatasetVisualization(**ANNUAL_RASTER, geography="-91.5,29.5")
        assert viz.geography_short_labels == ["(-91.5, 29.5)"]


class TestDataSourceParse:
    """Vega-Lite needs the CSV field types up front; a raster grid's coordinate label is not
    a number, but a vector grid's cell ID always is."""

    @pytest.fixture
    def viz(self):
        return DatasetVisualization(**ANNUAL_VECTOR)

    def _parse(self, viz, **kwargs):
        source = viz.get_data_source(**ANNUAL_VECTOR, **kwargs)
        return source.to_dict()["format"]["parse"]

    def test_vector_ids_are_parsed_as_numbers(self, viz):
        assert self._parse(viz, geography="1,2")["geography"] == "number"

    def test_vector_coordinates_are_parsed_as_numbers_too(self, viz):
        """They come back as the cell IDs they resolved to."""
        assert self._parse(viz, geography="-91.5,29.5")["geography"] == "number"

    def test_an_unfiltered_vector_grid_is_parsed_as_numbers(self, viz):
        assert self._parse(viz)["geography"] == "number"

    def test_raster_geography_is_left_as_strings(self):
        viz = DatasetVisualization(**ANNUAL_RASTER)
        source = viz.get_data_source(**ANNUAL_RASTER, geography="-91.5,29.5")
        assert "geography" not in source.to_dict()["format"]["parse"]
