"""The `veg_species` stacked bar: one series per vegetation species, all species at once.

`StackedBar` colors its segments by whichever dimensions the *reader* selected more than
one value of, which is the wrong shape for species coverage. A cell's species coverages are
the parts of a whole and are only worth reading together, so this chart drops the
`veg_species_code` filter entirely and stacks every species the model wrote, coloring and
ordering them from the `veg_species_code` dimension's own values so a species carries the
same color here as it does in the model's own map symbology.

That is the *only* difference, so it is expressed as the only difference: what the series
are is `StackedBar`'s four series seams, this chart overrides exactly those, and the rest of
the skeleton -- axes, hover rule, zero reference, container sizing, and the daily branch
`veg_species` never reaches -- is inherited unchanged.

Most of a cell's ~47 species are flat at zero for the whole run, though, and naming those
in the legend and the hover tooltip would bury the handful that are actually drawn. The
species that *are* drawn is a fact about the data, which this chart never reads -- it emits
a spec pointing at a `data.csv` URL, the way every `DatasetVisualization` does. So the
narrowing happens in the browser instead, and both the legend and the tooltip are built to
follow it rather than to be enumerated up front:

* the legend's entries come from the color scale's domain, which Vega-Lite computes from
  the filtered data, and the colors come with them out of the data (`alt.FieldRange`)
  rather than from a domain listed here that would put all 47 species back in the legend;
* the tooltip is one Vega expression rather than a list of field definitions, because a
  species that was filtered out has no pivoted column and so no row to name -- see
  `tooltip_expr` and `TOOLTIP_REFERENCE`.
"""

import json
from functools import cached_property
from typing import final

import altair as alt
import polars as pl

from cpra.mp.data.analysis.request.errors import RequestError
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.visualizations.display import display_number_format, display_unit
from cpra.mp.data.visualizations.stacked_bar import StackedBar

SPECIES_DIMENSION = "veg_species_code"

# The dimension carries a color per species and lists them in the model's own symbology
# order, which is the order this chart stacks them in -- see `Dimension` for both.
SPECIES_VALUES = DIMENSIONS[SPECIES_DIMENSION].values

# Digits in a `species_key` sort prefix -- see `_species_series` for what the prefix is
# for, and `series_channels` for the `labelExpr` that hides it again.
KEY_DIGITS = len(str(len(SPECIES_VALUES) - 1))

# The field `hover_transforms` calculates the tooltip object into, and the expression that reads it
# back out. The rule mark's own datum is the pivoted row, but `nearest` lays a transparent
# voronoi overlay over the plot -- that overlay is what the pointer actually hits -- and the
# overlay's datum is the rule's scenegraph *item*, with the row one level down under
# `datum`. Vega-Lite rewrites field references when it copies a tooltip onto the overlay,
# but only for field definitions; an expression it copies verbatim, so the expression has to
# find the row itself. Hence the indirection: the object is built once in a transform, where
# `datum` is unambiguously the row, and only this short reference has to work in both
# places. Reading the row wrong here empties the tooltip completely rather than degrading
# it, since every key would then be looked up on the wrong object at once.
TOOLTIP_FIELD = "species_tooltip"
TOOLTIP_REFERENCE = f"(datum.datum || datum).{TOOLTIP_FIELD}"


def _species_series():
    """One row per species this chart can draw, outermost first.

    Carries the dimension's own color and label for each species, plus the two fields the
    spec sorts on. The dimension's values are already in the model symbology's order, so
    "which species is outermost" is a row's position in that list and nothing else has to
    say it. Both sort fields put the last-listed species at the top -- of the legend, and of
    the stack -- so that the two read down in the same order, but they get there from
    opposite directions:

    `species_order` is that position as-is. Vega stacks the lowest value of an `Order`
    channel at the baseline and works up, so ascending position leaves the last-listed
    species at the top of the stack. (Negating it is the mistake this comment exists to
    prevent: it stacks correctly *bottom*-first, which is the legend's order upside down.)

    `species_key` is what the color encoding actually uses, and is the label behind a
    zero-padded numeric prefix that counts the same positions back down. The prefix is there
    because Vega-Lite pairs a data-driven color range with the scale's domain by sorting
    *both* by the domain's own field (ascending), and honors no other sort for the range: a
    `sort` on the encoding reorders the domain alone and would hand each species another
    species' color. Encoding the order into the key instead makes plain ascending order the
    order that's wanted, and the legend's `labelExpr` slices the prefix back off.
    """

    position = pl.int_range(pl.len())
    return (
        SPECIES_VALUES.select(
            pl.col("value").alias(SPECIES_DIMENSION),
            pl.col("label"),
            species_color=pl.col("color"),
            species_order=position,
            species_key=pl.format(
                "{} {}",
                (pl.len() - 1 - position).cast(pl.String).str.pad_start(KEY_DIGITS, "0"),
                pl.col("label"),
            ),
        )
        .sort("species_order", descending=True)
    )


SPECIES = _species_series()


class SpeciesStackedBar(StackedBar):
    """Stacked bar chart of every vegetation species' coverage in one grid cell."""

    @final
    class Meta:
        name = "species-stacked-bar"
        label = "Species Stacked Bar"
        type = "dataset"
        max_grid_cells = 1
        # `veg_species` is the only variable `veg_species_code` describes, and this chart
        # is built entirely around that dimension -- the catalog offers it for no other.
        variables = ["veg_species"]

    def validate(self):
        """This chart is a breakdown of ONE variable's species, so it needs that variable.

        `Meta.variables` already says which one, and a client reads it to decide whether to
        offer the chart at all -- but nothing enforced it server-side, and the route is
        public. A request naming an analysis that fixes its own inputs (`lnd_type_plus`)
        has no presentation variable at all, and every property here dereferenced that
        `None`: `AttributeError` -> 500, for what is an ordinary bad request.
        """
        super().validate()
        if self.variable is None or self.variable.name not in self.Meta.variables:
            named = "no single variable" if self.variable is None else repr(self.variable.name)
            raise RequestError(
                f"visualization {self.Meta.name!r} breaks one variable down by "
                f"{SPECIES_DIMENSION}, so it can only draw {self.Meta.variables[0]!r} -- this "
                f"request produces {named}."
            )

    @cached_property
    def variable_label(self):
        """The y-axis title: the variable's own label, named as the coverage it is.

        `veg_species`'s label is the variable ("Vegetation Type: Species"), not the quantity,
        and on an axis of stacked coverages the quantity is what wants saying. Built here
        rather than appended to `super().variable_label`, because that ends in the difference
        suffix and the unit, and the word belongs to the variable in front of both -- a
        difference chart's axis reads "... Species Coverage Difference".
        """

        label = f"{self.variable.label} Coverage{self.difference_suffix}"
        unit = display_unit(self.variable)
        return f"{label} ({unit})" if unit else label

    @cached_property
    def data_source(self):
        """The Altair data source: every species, whatever the reader selected.

        Drops any `veg_species_code` filter, since the point of this chart is the whole
        stack. That alone would lose the column the chart keys on, though: `get_data_uri`
        adds a dimension to `select` only when it was given more than one value to filter
        by, so with no filter at all the response would carry no `veg_species_code` at all.
        """

        params = {name: value for name, value in self.source_params.items() if name != SPECIES_DIMENSION}
        params["select"] = ",".join([self.time_column, SPECIES_DIMENSION, "grid_cell", "value"])
        return self.get_data_source(self.variable.name, self.grid.name, self.time_unit, **params)

    @cached_property
    def tooltip_expr(self):
        """A Vega expression building the hover rule's whole-stack tooltip table.

        `StackedBar` lists its tooltip's rows as Altair `Tooltip` field definitions, one per
        compared dimension value, which it can do because it knows those values up front.
        Here the rows are only known in the browser: `transform_pivot` gives a column to
        each species that survived the `species_extent` filter and to no other, so every
        row in this table is a field that may or may not exist.

        A `Tooltip` field definition for a missing column is no help -- it compiles to
        `format(datum[...], ...)`, which renders "NaN" rather than dropping the row -- so
        the tooltip object is built as a single expression instead. vega-tooltip skips a key
        whose value is `undefined`, and `undefined` is exactly what `datum[<code>]` is for a
        species the pivot left out, which collapses these ~47 keys down to the species on
        screen. `isValid` guards the `format` call so that a value that *is* there is still
        formatted; formatting `undefined` is what would produce the "NaN" row.

        This is calculated into a field rather than encoded as the tooltip directly, so that
        `datum` here is always the pivoted row -- see `TOOLTIP_REFERENCE` for the mark that
        would otherwise reach it with a `datum` one level off.
        """

        number_format = display_number_format(self.variable)
        entries = [f"{json.dumps(self.time_dimension.label)}: datum[{json.dumps(self.time_column)}]"]
        for row in SPECIES.iter_rows(named=True):
            field = f"datum[{json.dumps(row[SPECIES_DIMENSION])}]"
            entries.append(
                f"{json.dumps(row['label'])}: isValid({field})"
                f" ? format({field}, {json.dumps(number_format)}) : {field}"
            )
        return "{" + ", ".join(entries) + "}"

    def series_transforms(self, base):
        """The species are the series, so the parent's compare-dimension indices give way to
        the color map lookup and the filter that drops the species flat at zero.

        Nothing puts a second dimension on Opacity the way `StackedBar` would: `species_key`
        has Color, and a stack of species is already the whole of what this chart says.
        """

        return (
            base.transform_lookup(
                SPECIES_DIMENSION,
                from_=alt.LookupData(
                    SPECIES.select(SPECIES_DIMENSION, "species_key", "species_color", "species_order"),
                    SPECIES_DIMENSION,
                    ["species_key", "species_color", "species_order"],
                ),
            )
            # The magnitude rather than the value, so that a difference chart keeps a
            # species that only ever declined: "greater than zero in at least one year"
            # means "not flat at zero", and on a difference chart a species that isn't can
            # show up as a negative value just as readily as a positive one. Filtering the
            # value itself would drop those bars from the chart, not just from the legend.
            .transform_calculate(species_magnitude="abs(datum.value)")
            .transform_joinaggregate(species_extent="max(species_magnitude)", groupby=[SPECIES_DIMENSION])
            .transform_filter("datum.species_extent > 0")
        )

    @cached_property
    def series_channels(self):
        """Color, from the data rather than from a scale listed here."""

        return [
            alt.Color("species_key:N")
            # No `domain`: the domain Vega-Lite infers from the filtered data is what keeps
            # the legend down to the species being drawn, and listing all 47 here would put
            # every one of them back. The colors ride along in the data instead, as the
            # `species_color` the lookup attached to each row.
            .scale(range=alt.FieldRange(field="species_color")).legend(
                title=DIMENSIONS[SPECIES_DIMENSION].label,
                labelExpr=f"slice(datum.label, {KEY_DIGITS + 1})",
                labelLimit=400,
            )
        ]

    @cached_property
    def order_channel(self):
        return alt.Order("species_order:Q")

    def hover_transforms(self, base):
        """One column per drawn species, plus the tooltip object built over them.

        The parent pivots on a key it calculated from the compared dimensions' indices; here
        the species code is already that key. `groupby` needs no second field either --
        `veg_species` is annual-only, so `time_field` and `time_column` are the same column.
        """

        return base.transform_pivot(
            SPECIES_DIMENSION, value="value", groupby=[self.time_column]
        ).transform_calculate(**{TOOLTIP_FIELD: self.tooltip_expr})

    @cached_property
    def tooltip_encoding(self):
        return alt.value({"expr": TOOLTIP_REFERENCE})
