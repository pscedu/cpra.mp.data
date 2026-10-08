"""`sediment_dynamics` -- the five morphology surfaces that describe how one pixel's
ground is built and drowned, read at that pixel and stacked into a single tabular series:
topobathymetric elevation (`elev`), inundation depth (`inun`), land type (`lnd_type`), and
the two accretion components (`test_mnrl_accr`, `test_org_accr`). One row per (year,
variable), with the variable name in a `variable` column and the measurement in `value`.

It backs the chart of the same name (`visualizations/sediment_dynamics.py`), which is a
single-pixel time series -- `max_grid_cells = 1` -- and is the reason the analysis exists
as an analysis at all rather than as five requests the chart joins client-side. That is
what the chart used to do: five `get_data_source` calls, four `transform_lookup`s to glue
them back together on `calendar_year`, and a `transform_fold` to stack the result, with the
client-side join keyed on a `grid_cell` column each source had to carry for the purpose.
One request returns the stacked frame directly.

**The shape is join -> spatial filter -> unpivot, and the order is forced.** The five
inputs are rasters on one grid, so a spatial `join` pairs them band-for-band on the
dimensions they share and leaves five operand-named value columns on a band-wide frame. An
`unpivot` is long-only (`request.ops._type_unpivot`), and the one thing that turns a
band-wide frame long is a spatial or value conjunct (`request.ops.FILTER_LOWERING_TABLE`)
-- so the `grid_cell` selection is not merely this chart's narrowing, it is the step that
makes the fold expressible. Hence the explicit requirement in `pipeline()` below: without
a pixel there is nothing to unpivot, and the caller should hear that rather than
`_type_unpivot`'s own (accurate, but unhelpful here) "cannot unpivot a band-wide frame".

**Why an `unpivot` rather than a `case`.** Every other multi-input analysis here
(`lnd_type_plus`, `sal_veg_investigation`) reduces its operands to ONE number -- a
reclassification, a threshold year -- because `ops.validate_shapes` allows exactly one
value column at a sink. This one has nothing to reduce: the five surfaces are five
independent measurements in three different units, and a chart that drew any function of
them would be drawing something none of the five is. So they are stacked instead of
collapsed, which keeps all five and still hands the sink the single `value` column every
consumer of a result names literally. See `pipeline.UnpivotStep` for the same argument from
the other side.

**The operands are named after their variables on purpose.** A `join`'s output value
columns take the OPERAND names (`ops.validate_shapes`), and an `unpivot` folds those names
into the label column -- so naming each operand after the variable it reads is what makes
the `variable` column hold real variable names. `request.descriptor.output_descriptor`
reads them back that way and reports the five in `output.value_columns`, paired with their
`Variable`s in `output.variables`, which is how the chart keys per-variable units, number
formats and `display_scale_factor` off one source. Rename an operand and that lookup
silently stops resolving.

**Grid and time unit come from the caller, the variables do not.** This is the one entry
that takes a dataset and then reads five of them: `variable` names which of the five the
caller was looking at when they asked for the chart (`Meta.variables` is the list, and the
portal offers the chart for exactly those datasets), while `grid` and `time_unit` say WHICH
morphology grid and step to read all five on. That is `morph_pixel_v001` or `_v002` in
practice, and which grids actually carry all five is answered by `/v1/catalog.json`'s
input-intersected choices rather than by a constant here.

**`Meta.legend` is `None`, and has to be.** A legend names one value domain, and this
result carries five in three units (NAVD88 metres, metres, centimetres, plus a categorical
land type). `api.catalog.output_body` falls through to the sole presentation variable when
an analysis declares none -- and for a folded result there is no sole variable either
(`descriptor.OutputDescriptor.variable` is `None` when `folded_into` is set), so `legend`
comes back null and the per-variable lookup described above is the only correct way to
label these rows. `label` and `result_kind` still come from this `Meta`, which is why they
are declared.

**The five rasters are read through a one-pixel window, and that is not free by default.**
A spatial filter sitting after a `join` used to record no `point_sample_step` for its
origins -- `execution.plan`'s `incoming is None` gate retires the joined frame's origin
state before the per-op branches run, and `_Validity` tracks a single origin -- so each of
the five was read WHOLE and sampled afterwards. On a `morph_pixel` grid that is the whole
8265x20674 surface, five times, per selected band. That gate now records the sample for
every raster origin of the joined frame instead (`band_wide_join_origin_groups` names
them), and `execution.execute._shared_point_sample_window` reduces the five windows to the
one the shared scan reads. The shape this analysis declares is therefore load-bearing for
its cost as well as its typing: the `filter` has to sit DIRECTLY on the `join` for that
gate to see it, and the pixel has to be one of that filter's conjuncts rather than a step
of its own. Moving it, or interposing a `calculate` between the two, is correct and reads
the whole grid.
"""

from cpra.mp.data.analyses.base import (
    Analysis,
    AnalysisMeta,
    InvalidAnalysisParametersError,
    selection_filter_where,
)
from cpra.mp.data.analysis.request.pipeline import parse_pipeline

# The five surfaces, in the order the chart declares them (`visualizations/
# sediment_dynamics.py`'s own `Meta.variables`). Order is presentational only: an
# `unpivot` stacks the folded columns in sorted name order
# (`execution.long_ops.unpivot_long`), so the rows come back the same however this is
# written -- but the two lists are the same five names and should stay visibly so.
#
# `test_mnrl_accr`/`test_org_accr` rather than `mnrl_accr`/`org_accr`: those are the pair
# the chart has always drawn, and `variables.csv` carries both. Swapping them is a data
# question, not a pipeline one, and belongs in one edit to both `Meta.variables` lists.
VARIABLE_NAMES = ("elev", "inun", "lnd_type", "test_mnrl_accr", "test_org_accr")

# The label column the fold writes each row's variable name into -- `UnpivotStep.into`'s
# own default, stated explicitly because the chart reads `datum.variable` by this name and
# the two have to agree. It is a DIMENSION name, which an `as`/input name may not be and an
# `into` deliberately may be (see `pipeline._validate_label_column_name`): these labels ARE
# variable names, so they belong in the column that means that everywhere else.
LABEL_COLUMN = "variable"


class SedimentDynamics(Analysis):
    Meta = AnalysisMeta(
        name="sediment_dynamics",
        label="Sediment Dynamics",
        # The five datasets this analysis is offered for -- the same list the chart
        # declares. Unlike `lnd_type_plus`/`sal_veg_investigation`, this is a real list
        # rather than `None`: the analysis DOES read the caller's dataset (for its grid and
        # time unit), so "which datasets is it meaningful for" has an answer.
        variables=list(VARIABLE_NAMES),
        # None -- see "`Meta.legend` is `None`, and has to be" in the module docstring.
        legend=None,
        result_kind="value",
        # Neither a variable-dropdown entry nor something to chain onto: the chart is how
        # this is reached, and a `kind="dimension"` stage appended to a folded frame would
        # be grouping across five incommensurable measurements at once.
        kind="internal",
    )

    class Parameters(Analysis.Parameters):
        """No parameters of its own. The pixel is a `grid_cell` filter and the dimension
        selection is ordinary filters; neither is an analysis parameter (see
        `analyses._register`, which forbids a dimension-named one outright).
        """

    def pipeline(self):
        dataset = self.require_dataset()
        # Composed once and stated once, on the joined frame -- the same discipline
        # `lnd_type_plus` documents at length. The dimension conjuncts in it reach every
        # origin's read through `execution.plan`'s dimension passes wherever they are
        # declared; the spatial one is what lowers the frame so the `unpivot` below can run.
        selection_where = selection_filter_where(self.filters, grid_cell_where=self.grid_cell_where)
        if self.grid_cell_where is None and not self.filters.get("grid_cell"):
            # `grid_cell=lng,lat` and not "a cell id or a coordinate pair": the five
            # variables are morphology surfaces, so `dataset.grid` is always a RASTER grid,
            # and for one of those `api.data_query._resolve_without_geometry` sends every
            # `grid_cell` token straight to `io.geography.resolve_raster_point_labels`,
            # which parses WGS84 pairs and nothing else. The digit-id form is a VECTOR
            # grid's spelling (and a POST body's, which states its own `where`); offering
            # it here would name a value this route rejects with a coordinate-parse error.
            raise InvalidAnalysisParametersError(
                f"analysis {self.Meta.name!r} reads five variables at ONE pixel -- name it "
                "with grid_cell=lng,lat. Without a pixel the five surfaces stay band-wide "
                "and there is nothing to stack them into."
            )

        data = {
            # One input per variable, all on the caller's grid and time unit. The NAMES are
            # the variable names because the join's operands are, and the fold turns those
            # operand names into the `variable` column's values -- see "The operands are
            # named after their variables on purpose" in the module docstring.
            "inputs": {
                name: {
                    "variable": name,
                    "time_unit": dataset.time_unit,
                    "grid": dataset.grid,
                }
                for name in VARIABLE_NAMES
            },
            "steps": [
                # All five at once. `JoinStep.inputs` is `min_length=2` with no maximum, and
                # `band_ops._pair_band_groups` keys on the dimensions the operands share and
                # wants one band per operand per group for any number of them -- these five
                # share `model_group_id`/`scenario_id`/`calendar_year` and declare no
                # variable-specific dimension of their own, so every year pairs five-wide.
                {
                    "op": "join",
                    "type": "spatial",
                    "inputs": list(VARIABLE_NAMES),
                    "as": "paired",
                },
                # The pixel and the caller's dimension selection, in one step. It sits
                # BEFORE the fold because a spatial filter over a band-wide frame is a point
                # sample of every operand at once (`band_ops.sample_grid_cells_band_wide`),
                # which is both the narrowing and the lowering; after the fold there would
                # be five times as many rows to drop and the frame would already be long.
                {"op": "filter", "where": selection_where, "as": "sampled"},
                # Five operand-named value columns in, one `value` plus one `variable` out.
                {"op": "unpivot", "into": LABEL_COLUMN, "as": "stacked"},
            ],
            # Always tabular: the fold is long-only, so the final frame is long whatever
            # grid the caller named, and `validate_shapes` would reject a raster sink over
            # it. The `data.{format}` route overwrites this from its own URL segment
            # (`csv`/`ndjson`/`parquet` all work); `pipeline.json` publishes it as-is, and
            # what it publishes has to still validate if a caller resubmits it verbatim.
            "output": {"format": "csv"},
        }
        return parse_pipeline(data)
