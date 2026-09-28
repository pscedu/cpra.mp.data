"""Tests for the stacked bar charts.

`StackedBar` and `SpeciesStackedBar` are one chart with two answers to a single question --
what the series are -- so this file is organized around that seam rather than around the two
classes. `TestSharedSkeleton` runs over both, on the parts that shouldn't differ.
`TestCompareDimensions` through `TestBrushedTimeAxis` are the parent's own answer: series
come from whichever dimensions the reader selected more than one value of.
`TestSpeciesStackedBar*` are the subclass's: the series are every vegetation species, and
the ones worth naming are decided in the browser.

None of these read data. A visualization emits a spec pointing at a `data.csv` URL, so what
is testable here is what the spec promises about a response nothing has fetched yet -- and,
where that promise is one Vega-Lite or Vega will interpret rather than one the chart applies
itself, the tests say which interpretation is being relied on.
"""

import datetime as dt
import json
import re
from urllib.parse import parse_qs, urlparse

import pytest

from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.visualizations.species_stacked_bar import (
    KEY_DIGITS,
    SPECIES,
    TOOLTIP_FIELD,
    TOOLTIP_REFERENCE,
    SpeciesStackedBar,
)
from cpra.mp.data.visualizations.stacked_bar import COMPARE_CHANNELS, StackedBar

# A vector grid queried by ID needs no geometry on disk, and a raster grid never has any, so
# every chart here builds from the structure CSVs alone.
ANNUAL = dict(variable="ssf", grid="ecoregion_v001", time_unit="annual", grid_cell="1,2")
DAILY = dict(variable="sal", grid="hydro_compartment_v001", time_unit="daily", grid_cell="1")
VEG_SPECIES = dict(
    variable="veg_species", grid="veg_grid_cell_v001", time_unit="annual", grid_cell="-91.5,29.5"
)


def stacked_bar(**params):
    return StackedBar(**{**ANNUAL, **params})


def species_bar(**params):
    return SpeciesStackedBar(**{**VEG_SPECIES, **params})


def layers(spec, mark):
    """The layers drawn with `mark`, in the order they're drawn."""
    return [layer for layer in spec["layer"] if layer["mark"]["type"] == mark]


def transform(layer, key):
    """The first transform of a given kind on `layer`."""
    return next(step for step in layer["transform"] if key in step)


def calculated(layer, name):
    """The expression a layer calculates into the field `name`."""
    return next(step["calculate"] for step in layer["transform"] if step.get("as") == name)


class _Row:
    """A stand-in for a Vega `datum`, for `evaluate`."""

    def __init__(self, fields):
        self.__dict__.update(fields)


def evaluate(expression, **fields):
    """Evaluate a generated Vega expression as Python, against a row of `fields`.

    Only used on `order_idx`, which is parentheses, `*` and `+` -- those mean the same thing
    in both languages, so this really does compute what Vega will compute, rather than
    restating the formula the chart used to build it.
    """
    return eval(expression.replace("datum.", "row."), {"row": _Row(fields)})  # noqa: S307


class TestSharedSkeleton:
    """Both charts are the same chart with different series, so everything that isn't the
    series should be indistinguishable between them. `SpeciesStackedBar` overrides only
    `StackedBar`'s series seams to keep that true; these run over both charts so that
    re-forking the skeleton -- easy to do by overriding `chart` again, and something no other
    test would notice -- breaks something.
    """

    @pytest.fixture(params=["stacked-bar", "species-stacked-bar"])
    def viz(self, request):
        return species_bar if request.param == "species-stacked-bar" else stacked_bar

    @pytest.fixture
    def spec(self, viz):
        return viz().chart().to_dict()

    def test_the_bars_touch(self, spec):
        """Zero band padding is what makes a stacked bar read as a continuous area."""
        scale = layers(spec, "bar")[0]["encoding"]["x"]["scale"]
        assert (scale["paddingInner"], scale["paddingOuter"]) == (0, 0)

    def test_the_value_axis_is_stacked_from_zero(self, spec):
        assert layers(spec, "bar")[0]["encoding"]["y"]["stack"] == "zero"

    def test_the_bars_are_ordered_explicitly(self, spec):
        """Without an order channel Vega-Lite picks its own stacking order, which is stable
        but arbitrary -- and so not the order either chart's legend is listed in."""
        assert "order" in layers(spec, "bar")[0]["encoding"]

    def test_a_hover_rule_reports_the_whole_stack(self, spec):
        """The rule sits on a pivot of the bars' own data, so it can name every series at the
        pointer's time period rather than just the segment under the pointer."""
        rule = layers(spec, "rule")[0]
        assert "pivot" in transform(rule, "pivot")
        assert "tooltip" in rule["encoding"]

    def test_the_rule_is_invisible_until_hovered(self, spec):
        opacity = layers(spec, "rule")[0]["encoding"]["opacity"]
        assert [condition["param"] for condition in opacity["condition"]] == ["hover"]
        assert opacity["value"] == 0

    def test_the_hover_selection_is_nearest_and_keyed_on_time(self, spec):
        """A one-pixel rule is unhittable, so the selection is by proximity -- which is what
        makes Vega-Lite lay a voronoi over the plot. See `TOOLTIP_REFERENCE`."""
        param = next(param for param in spec["params"] if param["name"] == "hover")
        assert param["select"]["nearest"] is True
        assert param["select"]["fields"] == ["calendar_year"]

    def test_a_difference_chart_marks_zero(self, viz):
        """Vega-Lite stacks negative segments below zero, so without this the baseline a
        difference is read against is wherever the axis happens to start."""
        spec = (
            viz(analysis="diff", dimension="model_group_id", order="asc", model_group_id=[500, 520])
            .chart()
            .to_dict()
        )
        assert {"datum": 0} in [rule["encoding"].get("y") for rule in layers(spec, "rule")]

    def test_a_plain_chart_does_not(self, spec):
        assert {"datum": 0} not in [rule["encoding"].get("y") for rule in layers(spec, "rule")]

    def test_the_zero_rule_is_drawn_under_the_bars(self, viz):
        """It's a reference line, not an annotation on top of the data."""
        spec = (
            viz(analysis="diff", dimension="model_group_id", order="asc", model_group_id=[500, 520])
            .chart()
            .to_dict()
        )
        marks = [layer["mark"]["type"] for layer in spec["layer"]]
        assert marks.index("rule") < marks.index("bar")

    def test_the_container_gets_the_size(self, viz):
        """`size_to_container` reinterprets width/height as the whole container's, and lets
        autosize give the plot area whatever is left after real chrome is measured."""
        spec = viz(width=900, height=480).chart().to_dict()
        assert spec["autosize"] == {"contains": "padding", "type": "fit"}
        assert (spec["width"], spec["height"]) == (900, 480)

    def test_the_time_axis_is_titled(self, spec):
        assert layers(spec, "bar")[0]["encoding"]["x"]["title"] == "Year"

    def test_the_value_axis_is_titled_with_the_variable(self, spec, viz):
        assert layers(spec, "bar")[0]["encoding"]["y"]["title"] == viz().variable_label


class TestCompareDimensions:
    """Which of the reader's selections become series. Only a dimension they picked *several*
    values of is a comparison; one value is a filter, and the API has already applied it."""

    def test_several_values_are_compared(self):
        assert list(stacked_bar(model_group_id=[500, 520]).compare_dimensions) == ["model_group_id"]

    def test_a_single_value_is_not(self):
        assert stacked_bar(model_group_id=[500]).compare_dimensions == {}

    def test_a_bare_value_is_not(self):
        """The API accepts a scalar as well as a list, and a scalar can't be a comparison."""
        assert stacked_bar(model_group_id=500).compare_dimensions == {}

    def test_the_collapsed_dimension_is_excluded(self):
        """Its two values *are* the difference being drawn, so drawing them as two series
        would draw the difference and both of its operands at once. `compare_dimensions`
        excludes `self.resolved_analysis.collapsed_dimensions`; `negate_dimension`, the
        property that used to compute this without a resolved analysis, was deleted along
        with the legacy `negate` parameter it read -- `test_the_negate_parameter_
        may_name_a_value_too` and `test_no_negate_parameter_leaves_no_negate_dimension`,
        which pinned that property directly, were deleted rather than converted, since
        there is nothing left on `StackedBar` for them to call."""
        viz = stacked_bar(
            model_group_id=[500, 520], analysis="diff", dimension="model_group_id", order="asc"
        )
        assert viz.compare_dimensions == {}

    def test_a_collapsed_dimension_does_not_exclude_the_others(self):
        viz = stacked_bar(
            model_group_id=[500, 520],
            scenario_id=[7, 8],
            analysis="diff",
            dimension="model_group_id",
            order="asc",
        )
        assert list(viz.compare_dimensions) == ["scenario_id"]

    def test_only_as_many_as_there_are_channels(self):
        """Vega-Lite would stack by the combination of every discrete channel present, so a
        third dimension has nowhere to go that stays readable -- it's dropped, not merged."""
        viz = stacked_bar(
            model_group_id=[500, 520], scenario_id=[7, 8], aggregate_type=["mean", "max"]
        )
        assert len(viz.compare_dimensions) == len(COMPARE_CHANNELS)

    def test_the_dimension_registry_decides_which_channel(self):
        """Not the order the reader's parameters arrived in: the comparison is built by
        walking `DIMENSIONS`, so which dimension gets Color is fixed by the registry and the
        same two selections can't come back on opposite channels."""
        order = list(DIMENSIONS)
        forwards = stacked_bar(model_group_id=[500, 520], scenario_id=[7, 8])
        backwards = stacked_bar(scenario_id=[7, 8], model_group_id=[500, 520])
        assert list(forwards.compare_dimensions) == list(backwards.compare_dimensions)
        assert list(forwards.compare_dimensions) == sorted(
            forwards.compare_dimensions, key=order.index
        )


class TestSeriesChannels:
    """How one series is told from another, once `compare_dimensions` has decided what they
    are."""

    def test_nothing_compared_leaves_the_bars_one_color(self):
        spec = stacked_bar().chart().to_dict()
        assert set(layers(spec, "bar")[0]["encoding"]) == {"x", "y", "order"}

    def test_the_first_dimension_takes_color(self):
        spec = stacked_bar(model_group_id=[500, 520]).chart().to_dict()
        color = layers(spec, "bar")[0]["encoding"]["color"]
        assert color["field"] == "model_group_id"
        assert color["type"] == "nominal"

    def test_the_second_takes_opacity(self):
        spec = stacked_bar(model_group_id=[500, 520], scenario_id=[7, 8]).chart().to_dict()
        encoding = layers(spec, "bar")[0]["encoding"]
        assert encoding["color"]["field"] == "model_group_id"
        assert encoding["opacity"]["field"] == "scenario_id"

    def test_legends_are_titled_from_the_dimension(self):
        """The reader chose these by their portal labels, so that's what names the legend."""
        spec = stacked_bar(model_group_id=[500, 520]).chart().to_dict()
        title = layers(spec, "bar")[0]["encoding"]["color"]["title"]
        assert title == DIMENSIONS["model_group_id"].label


class TestSeriesTransforms:
    """The declared order of a dimension's values is the order its segments stack in, which
    is what `order_idx` carries. Getting it from the data would give alphabetical or
    first-seen order instead."""

    @pytest.fixture
    def two_dimensions(self):
        return stacked_bar(model_group_id=[500, 520], scenario_id=[7, 8])

    @pytest.fixture
    def bars(self, two_dimensions):
        return layers(two_dimensions.chart().to_dict(), "bar")[0]

    def test_an_index_per_compared_dimension(self, bars):
        assert calculated(bars, "model_group_id_idx")
        assert calculated(bars, "scenario_id_idx")

    def test_values_are_matched_loosely(self, bars):
        """`transform_lookup` would be the obvious way to attach an index, but Vega's coerces
        the key with String() and so misses 0 against "0.0". Loose `==` doesn't, and `===`
        would put the bug straight back."""
        expression = calculated(bars, "model_group_id_idx")
        assert "datum.model_group_id == 500 ? 0" in expression
        assert "===" not in expression

    def test_an_unselected_value_indexes_to_null(self, bars):
        """The API filtered them out, so a row that still has one is a row this chart cannot
        place -- better absent from the stack than stacked somewhere arbitrary."""
        assert calculated(bars, "model_group_id_idx").endswith(" : null")

    def test_string_values_are_quoted(self):
        """An unquoted string would be read as a field reference, matching nothing."""
        bars = layers(stacked_bar(aggregate_type=["mean", "max"]).chart().to_dict(), "bar")[0]
        assert 'datum.aggregate_type == "mean" ? 0' in calculated(bars, "aggregate_type_idx")

    def test_the_combined_index_enumerates_the_stack(self, bars, two_dimensions):
        """Two dimensions have to collapse into one stacking order, and the combination is
        positional: a collision would silently merge two series into one stack position, and
        a gap would order them by something other than what was declared."""
        expression = calculated(bars, "order_idx")
        produced = [
            evaluate(expression, model_group_id_idx=first, scenario_id_idx=second)
            for first, second in two_dimensions.compare_choice_indices
        ]
        assert produced == list(range(len(two_dimensions.compare_choices)))

    def test_one_dimension_orders_by_its_own_index(self):
        viz = stacked_bar(model_group_id=[500, 520])
        bars = layers(viz.chart().to_dict(), "bar")[0]
        produced = [
            evaluate(calculated(bars, "order_idx"), model_group_id_idx=index)
            for (index,) in viz.compare_choice_indices
        ]
        assert produced == [0, 1]

    def test_nothing_compared_is_still_ordered(self):
        """One series, so every segment shares a position rather than having none."""
        bars = layers(stacked_bar().chart().to_dict(), "bar")[0]
        assert calculated(bars, "order_idx") == "0"


class TestTooltip:
    """`StackedBar` knows its series up front, so its tooltip is a list of field definitions
    -- one per combination, named for the pivot column that combination produced. The two
    namings are generated separately and have to agree; a mismatch shows up as a tooltip of
    "NaN"s rather than as an error."""

    @pytest.fixture
    def rule(self):
        spec = stacked_bar(model_group_id=[500, 520], scenario_id=[7, 8]).chart().to_dict()
        return layers(spec, "rule")[0]

    def test_the_time_period_leads(self, rule):
        assert rule["encoding"]["tooltip"][0]["field"] == "calendar_year"

    def test_a_row_per_combination(self, rule):
        assert len(rule["encoding"]["tooltip"]) == 1 + 4

    def test_rows_are_titled_with_the_selected_values(self, rule):
        titles = [row.get("title") for row in rule["encoding"]["tooltip"][1:]]
        assert titles == ["500, 7", "500, 8", "520, 7", "520, 8"]

    def test_the_fields_are_the_pivot_s_own_column_names(self, rule):
        """The pivot names each column by joining the compared dimensions' indices with "_",
        in the order it concatenated them; the tooltip has to ask for exactly those."""
        key_expression = calculated(rule, "tooltip")
        pivoted_dimensions = re.findall(r"datum\.(\w+)_idx", key_expression)
        viz = stacked_bar(model_group_id=[500, 520], scenario_id=[7, 8])
        expected = [
            "_".join(str(index) for index in indices) for indices in viz.compare_choice_indices
        ]
        assert [row["field"] for row in rule["encoding"]["tooltip"][1:]] == expected
        assert pivoted_dimensions == list(viz.compare_dimensions)

    def test_values_are_formatted_for_the_variable(self, rule):
        assert {row.get("format") for row in rule["encoding"]["tooltip"][1:]} == {".1f"}

    def test_nothing_compared_leaves_one_row_named_for_the_variable(self):
        viz = stacked_bar()
        rule = layers(viz.chart().to_dict(), "rule")[0]
        row = rule["encoding"]["tooltip"][1]
        assert (row["field"], row["title"]) == ("value", viz.variable_label)
        assert calculated(rule, "tooltip") == "'value'"


class TestBrushedTimeAxis:
    """Data keyed on `calendar_day` -- daily *and* monthly, despite the name `is_daily` --
    gets a continuous axis, a brush, and a summary strip below to brush on, because a bar per
    day is far more bars than a plot is wide."""

    @pytest.fixture
    def spec(self):
        return StackedBar(**DAILY).chart().to_dict()

    def test_monthly_data_is_brushed_too(self):
        """`is_daily` reads as a time-unit check but is really "keyed on calendar_day", and
        monthly data is."""
        assert StackedBar(**{**DAILY, "time_unit": "monthly"}).is_daily is True
        assert StackedBar(**ANNUAL).is_daily is False

    def test_the_bars_are_keyed_on_a_continuous_date(self, spec):
        """`date` is calculated by the base chart; the raw time column can't be brushed."""
        bars = layers(spec["vconcat"][0], "bar")[0]
        assert bars["encoding"]["x"]["field"] == "date"
        assert bars["encoding"]["x"]["type"] == "temporal"

    def test_a_continuous_axis_gets_no_band_padding(self, spec):
        """There are no bands to pad."""
        assert "scale" not in layers(spec["vconcat"][0], "bar")[0]["encoding"]["x"]

    def test_the_summary_strip_is_below_the_chart(self, spec):
        assert len(spec["vconcat"]) == 2
        assert spec["vconcat"][1]["height"] == 60

    def test_the_strip_is_a_monthly_mean(self, spec):
        """A daily strip would be as unreadable as the chart it exists to navigate."""
        summary = spec["vconcat"][1]
        assert summary["mark"]["type"] == "bar"
        assert summary["encoding"]["x"]["timeUnit"] == "yearmonth"
        assert summary["encoding"]["y"]["aggregate"] == "mean"

    def test_the_brush_lives_on_the_strip(self, spec):
        """Brushing the chart it filters would fight with the hover selection."""
        assert spec["vconcat"][0]["transform"] == [{"filter": {"param": "date_range"}}]

    def test_the_brush_defaults_to_a_year(self, spec):
        """A whole run brushed at once is the unreadable chart the brush exists to avoid, so
        the default is a window, not the extent. Altair renders a `date` as the parts Vega
        needs to build one in the browser's own timezone."""
        assert self._brush(spec) == [
            {"year": 2025, "month": 1, "date": 1},
            {"year": 2026, "month": 1, "date": 1},
        ]

    def test_given_dates_set_the_initial_brush(self):
        """The UI reads the brush back out into the URL, so it has to be able to put one back
        in -- see `base-chart.tsx`'s `date_range` signal listener."""
        spec = (
            StackedBar(**DAILY, start_date=dt.date(2025, 3, 1), end_date=dt.date(2025, 9, 1))
            .chart()
            .to_dict()
        )
        assert self._brush(spec) == [
            {"year": 2025, "month": 3, "date": 1},
            {"year": 2025, "month": 9, "date": 1},
        ]

    def _brush(self, spec):
        brush = next(param for param in spec["params"] if param["name"] == "date_range")
        return brush["value"]["x"]

    def test_only_the_width_is_fitted(self, spec):
        """Vega-Lite's autosize won't fit the height of a vconcat spec, so the height is the
        one measurement this chart still has to do by hand."""
        assert spec["autosize"] == {"contains": "padding", "type": "fit-x"}

    def test_the_height_allows_for_the_strip(self):
        assert StackedBar(**DAILY, height=400).chart_height == 400 - 222

    def test_a_container_too_short_for_the_allowance_still_gets_a_plot(self):
        assert StackedBar(**DAILY, height=100).chart_height == 50


class TestSpeciesStackedBarDataSource:
    """The whole chart is one dimension wide, so the response has to carry that dimension
    and must not have been filtered down to a single value of it."""

    def _params(self, **kwargs):
        uri = species_bar(**kwargs).data_source.to_dict()["url"]
        return parse_qs(urlparse(uri).query)

    def test_a_species_filter_is_dropped(self):
        """The reader's species selection belongs to the other charts; this one is the stack."""
        assert "veg_species_code" not in self._params(veg_species_code=["spal", "sppa"])

    def test_the_species_column_is_selected(self):
        """`get_data_uri` only selects a dimension it was given several values to filter by,
        so dropping the filter would drop the column the chart keys on with it."""
        assert self._params()["select"] == ["calendar_year,veg_species_code,grid_cell,value"]

    def test_other_filters_are_kept(self):
        assert self._params(model_group_id=[500])["model_group_id"] == ["500"]

    def test_a_dimension_the_analysis_collapsed_is_dropped_from_the_explicit_select(self):
        """This chart is the one place that hands `get_data_uri` a ready-made `select`, which
        is exactly what let it keep asking for a column a named analysis had collapsed away:
        a `diff` over `calendar_year` groups that column out of its output, so naming
        it here 400s the read ("unable to find column"). The two years still have to be SENT
        -- they are what the difference is taken between -- so only the `select` entry goes.
        """
        params = self._params(
            analysis="diff",
            dimension="calendar_year",
            order="asc",
            calendar_year=[2030, 2040],
        )
        assert params["select"] == ["veg_species_code,grid_cell,value"]
        assert params["calendar_year"] == ["2030", "2040"]


class TestSpeciesStackedBarSeries:
    """The `veg_species_code` dimension decides both the color of a species and where it sits
    in the stack, and the spec reaches the second one through the first: the sort prefix on
    `species_key` is what keeps a data-driven color range paired with its domain."""

    def test_the_dimension_s_order_is_the_series_order(self):
        """`species_order` is a value's position in the dimension and nothing else, so the
        rows come back in that order reversed -- last-listed first."""
        dimension_order = DIMENSIONS["veg_species_code"].values["value"].to_list()
        assert SPECIES["veg_species_code"].to_list() == dimension_order[::-1]

    def test_the_sort_prefix_orders_keys_the_same_way(self):
        """Vega-Lite sorts the color range by the domain field ascending and honors no other
        order, so ascending `species_key` has to *be* descending `species_order`."""
        keys = SPECIES["species_key"].to_list()
        assert keys == sorted(keys)

    def test_the_prefix_is_wide_enough_to_sort_numerically(self):
        """A prefix that overflows `KEY_DIGITS` sorts "10" before "9" and hands two species
        each other's colors."""
        prefixes = [key[:KEY_DIGITS] for key in SPECIES["species_key"]]
        assert prefixes == sorted(prefixes)
        assert [int(prefix) for prefix in prefixes] == list(range(len(SPECIES)))

    def test_every_species_has_a_color_and_a_label(self):
        assert SPECIES["species_color"].null_count() == 0
        assert SPECIES["label"].null_count() == 0

    def test_colors_are_distinct(self):
        """Vega pairs the range with the domain by taking the distinct colors in the data;
        two species sharing one would leave every species after them a color out of step."""
        assert SPECIES["species_color"].n_unique() == len(SPECIES)

    def test_the_stack_reads_down_in_the_legend_s_order(self):
        """The two orders are read from opposite ends -- Vega stacks the *lowest*
        `species_order` at the baseline and works up, while the legend lists the *lowest*
        `species_key` at the top -- so they agree on screen only when they disagree as sorts.
        Negating either one reverses the bars against the legend while leaving everything else
        in the spec perfectly consistent, which is why this is pinned."""
        legend_downward = SPECIES.sort("species_key")["label"].to_list()
        stack_upward = SPECIES.sort("species_order")["label"].to_list()
        assert stack_upward == legend_downward[::-1]

    def test_the_last_species_listed_is_on_top(self):
        """Which end "descending" puts where -- the dimension's last value tops the stack, as
        it does in the model's own symbology."""
        top = SPECIES.sort("species_order", descending=True)["veg_species_code"][0]
        assert top == DIMENSIONS["veg_species_code"].values["value"][-1]

    def test_the_colors_are_the_dimension_s_own(self):
        """The whole point of sourcing these from the dimension: the color a species is drawn
        in here is the one it carries everywhere else, so it is read rather than restated."""
        dimension = DIMENSIONS["veg_species_code"].values
        expected = dict(zip(dimension["value"], dimension["color"]))
        assert dict(zip(SPECIES["veg_species_code"], SPECIES["species_color"])) == expected


class TestSpeciesStackedBarChart:
    @pytest.fixture
    def spec(self):
        return species_bar().chart().to_dict()

    @pytest.fixture
    def bars(self, spec):
        return layers(spec, "bar")[0]

    def test_the_color_scale_names_no_domain(self, bars):
        """The domain Vega-Lite infers from the filtered data is the whole mechanism for
        keeping the legend down to the species on screen; naming all 47 here puts them back."""
        assert "domain" not in bars["encoding"]["color"]["scale"]

    def test_the_colors_ride_along_in_the_data(self, bars):
        """A range listed here would be paired against that inferred domain positionally,
        which nothing knows the contents of up front."""
        assert bars["encoding"]["color"]["scale"]["range"] == {"field": "species_color"}

    def test_the_lookup_attaches_the_color_and_the_order(self, bars):
        """Both have to be on the row: the color because the range reads it out of the data,
        the order because the stack sorts on it."""
        lookup = transform(bars, "lookup")
        assert lookup["lookup"] == "veg_species_code"
        assert lookup["from"]["fields"] == ["species_key", "species_color", "species_order"]

    def test_the_legend_hides_the_sort_prefix(self, bars):
        label_expr = bars["encoding"]["color"]["legend"]["labelExpr"]
        assert label_expr == f"slice(datum.label, {KEY_DIGITS + 1})"
        assert SPECIES["species_key"][0][KEY_DIGITS + 1 :] == SPECIES["label"][0]

    def test_species_flat_at_zero_are_filtered_out(self, bars):
        """Both the legend and the tooltip follow the data, so this filter is the one place
        "greater than zero in at least one year" is expressed."""
        assert {"filter": "datum.species_extent > 0"} in bars["transform"]
        assert {
            "joinaggregate": [{"op": "max", "field": "species_magnitude", "as": "species_extent"}],
            "groupby": ["veg_species_code"],
        } in bars["transform"]

    def test_the_filter_reads_magnitude_not_value(self, bars):
        """A difference chart's declining species is still a species that was drawn, and
        filtering the signed value would drop its bars, not just its legend entry."""
        assert {"calculate": "abs(datum.value)", "as": "species_magnitude"} in bars["transform"]

    def test_the_y_axis_names_the_quantity(self, bars):
        """`veg_species`'s label names the variable, not what is stacked up the axis."""
        assert bars["encoding"]["y"]["title"] == "Vegetation Type: Species Coverage"

    def test_a_difference_axis_keeps_the_suffix_last(self):
        """The word belongs to the variable, so it goes in front of the suffix -- appending it
        to `variable_label` would read "... Species Difference Coverage"."""
        viz = species_bar(
            model_group_id=[500, 520], analysis="diff", dimension="model_group_id", order="asc"
        )
        assert viz.variable_label == "Vegetation Type: Species Coverage Difference"


class TestSpeciesStackedBarOverridesTheSeries:
    """The parent's answer to "what are the series" has to be gone, not merely added to."""

    @pytest.fixture
    def spec(self):
        """Built with a second dimension multi-selected, which is what `StackedBar` would
        turn into series."""
        return species_bar(model_group_id=[500, 520]).chart().to_dict()

    def test_no_other_dimension_becomes_a_series(self, spec):
        """`StackedBar` would put this one on Opacity. Here Color is spoken for by the
        species, and a stack of species is the whole of what the chart says -- so a reader
        comparing scenarios uses the general stacked bar for that."""
        assert set(layers(spec, "bar")[0]["encoding"]) == {"x", "y", "order", "color"}

    def test_the_parent_s_stacking_index_is_not_calculated(self, spec):
        """`species_order` replaces it rather than joining it; two order fields would leave
        which one Vega sorts on down to the encoding."""
        bars = layers(spec, "bar")[0]
        assert not any(step.get("as") == "order_idx" for step in bars["transform"])
        assert bars["encoding"]["order"]["field"] == "species_order"

    def test_the_pivot_is_keyed_on_the_species_itself(self, spec):
        """The parent calculates a pivot key out of its compared dimensions' indices; here the
        species code already is that key."""
        rule = layers(spec, "rule")[0]
        assert transform(rule, "pivot")["pivot"] == "veg_species_code"

    def test_it_is_offered_for_veg_species_alone(self):
        """`Meta.variables` is what the UI's visualization list filters on (see
        `visualization-select.tsx`), and its absence on the parent is what keeps that chart
        general."""
        assert SpeciesStackedBar.Meta.variables == ["veg_species"]
        assert not hasattr(StackedBar.Meta, "variables")


class TestSpeciesStackedBarTooltip:
    """The tooltip's rows are the species the browser kept, so they are one expression rather
    than a list of field definitions -- see `SpeciesStackedBar.tooltip_expr`. That buys the
    narrowing at the cost of Vega-Lite's help: field references in a *field definition* get
    rewritten when Vega-Lite copies the tooltip onto the transparent overlay that `nearest`
    puts over the plot, and an expression does not, so where the expression looks for its row
    is this chart's own problem. The tests below are mostly about that."""

    @pytest.fixture
    def expr(self):
        return species_bar().tooltip_expr

    @pytest.fixture
    def rule(self):
        return layers(species_bar().chart().to_dict(), "rule")[0]

    def test_the_time_column_leads(self, expr):
        assert expr.startswith('{"Year": datum["calendar_year"]')

    def test_a_row_per_species_in_stack_order(self, expr):
        quoted_keys = re.findall(r'"((?:[^"\\]|\\.)*)": (?:datum|isValid)', expr)
        keys = [json.loads(f'"{key}"') for key in quoted_keys]
        assert keys == ["Year"] + SPECIES["label"].to_list()

    def test_an_absent_species_falls_through_unformatted(self, expr):
        """`format` of a column the pivot never created renders "NaN"; the bare field is
        `undefined`, which vega-tooltip drops instead."""
        assert 'isValid(datum["spal"]) ? format(datum["spal"], "") : datum["spal"]' in expr

    def test_a_label_with_an_apostrophe_stays_quoted(self, expr):
        """A single-quoted key would end the string mid-label and leave the expression
        unparseable -- `scam6`'s label is the one that reaches that."""
        label = SPECIES.filter(veg_species_code="scam6")["label"][0]
        assert "'" in label
        assert f"{json.dumps(label)}: isValid" in expr

    def test_the_object_is_built_in_a_transform(self, rule, expr):
        assert {"calculate": expr, "as": TOOLTIP_FIELD} in rule["transform"]

    def test_the_encoding_only_dereferences_it(self, rule):
        assert rule["encoding"]["tooltip"] == {"value": {"expr": TOOLTIP_REFERENCE}}

    def test_the_reference_finds_the_row_from_either_mark(self, rule):
        """`nearest` lays a voronoi over the plot, and that mark -- the one the pointer
        actually hits -- carries the rule's scenegraph *item*, with the row a level down under
        `datum`. Reading the item as though it were the row doesn't lose one entry, it
        resolves all of them to `undefined` at once and the tooltip comes up empty."""
        reference = rule["encoding"]["tooltip"]["value"]["expr"]
        assert reference == f"(datum.datum || datum).{TOOLTIP_FIELD}"

    def test_the_object_is_calculated_after_the_pivot(self, rule):
        """`transform_pivot` drops every field outside its groupby, so a tooltip calculated
        ahead of it would not reach the mark."""
        steps = [(next(iter(step)), step.get("as")) for step in rule["transform"]]
        pivot = next(i for i, (kind, _) in enumerate(steps) if kind == "pivot")
        calculate = next(i for i, (_, name) in enumerate(steps) if name == TOOLTIP_FIELD)
        assert pivot < calculate

    def test_the_field_name_cannot_collide_with_a_species(self):
        """The pivot names a column after each species code, in the row the tooltip object is
        calculated into -- a collision would overwrite one silently."""
        assert TOOLTIP_FIELD not in SPECIES["veg_species_code"].to_list()
