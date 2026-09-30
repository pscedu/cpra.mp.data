from functools import cached_property
from urllib.parse import urlencode

import altair as alt
import polars as pl

from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.utils import required_columns
from cpra.mp.data.visualizations.colors import selection_color
from cpra.mp.data.visualizations.display import display_unit, display_value_expr


def grid_cell_tokens(grid_cell):
    """Split a `grid_cell` parameter into stripped string tokens.

    Accepts either the comma-separated string the API binds from the query string or the
    plain sequence a notebook is more likely to pass.
    """
    values = grid_cell.split(",") if isinstance(grid_cell, str) else grid_cell
    return [str(value).strip() for value in values]


def is_grid_cell_ids(grid_cell):
    """Are these grid_cell values grid cell IDs rather than lat/lng coordinate pairs?

    Mirrors the rule in `Dataset._apply_grid_cell_vector`: values are IDs only when every
    token is a plain unsigned digit string, so anything carrying a decimal point or a sign
    is a coordinate. Only meaningful for a vector grid, since a raster grid_cell is always
    coordinates. An absent grid_cell selects the whole grid, which is keyed by ID.
    """
    return grid_cell is None or all(token.isdigit() for token in grid_cell_tokens(grid_cell))


def analysis_chain(implied: str, chain: str | None) -> str:
    """`chain` with `implied` at its head, stated once.

    A visualization that declares `Meta.analysis` IS the presentation of that analysis
    (`SedimentDynamics`), so the analysis has to lead: it is what produces the frame, and
    only a `kind="dimension"` analysis may follow one. A caller's own chain is kept, after
    it -- `analysis=diff` on this chart means the five surfaces' differences, not a
    different chart -- and a caller who named the implied analysis themselves gets it once
    rather than twice.
    """
    names = [name for name in (chain or "").split(",") if name and name != implied]
    return ",".join([implied, *names])


class DatasetVisualization:
    """Visualization for a single dataset."""

    def __init__(self, request=None, decoded=None, height=400, width=600, **kwargs):
        self.request = request
        self._decoded = decoded
        self.height = height
        self.width = width
        self.params = kwargs

    @classmethod
    def request_items(cls, items):
        """The query pairs the ROUTE should decode for this visualization: the caller's own,
        with `Meta.analysis` forced to the head of the chain where the class declares one.

        A chart that is the presentation of a named analysis (`SedimentDynamics`) should be
        reachable by naming the CHART -- a client picking "Sediment Dynamics" from a chart
        menu has said everything it needs to, and requiring it to also know which analysis
        backs it is a coupling with no upside and a silent failure mode (the spec would be
        built over a plain single-variable read and every layer would filter it to nothing).
        So the route asks the class rather than the query string.

        Stated here rather than in `__init__` because the route decodes BEFORE it
        constructs. `__init__` has the same rule to apply to `self.params`, which is what
        puts `analysis=` on the data URLs the spec emits, and both go through
        `analysis_chain` so there is one statement of what "implied" means.
        """
        implied = getattr(getattr(cls, "Meta", None), "analysis", None)
        if implied is None:
            return list(items)
        chain = ",".join(str(value) for key, value in items if key == "analysis")
        return [
            *((key, value) for key, value in items if key != "analysis"),
            ("analysis", analysis_chain(implied, chain)),
        ]

    @property
    def names_dataset(self):
        """Whether a data/tile URL this spec emits should carry the `variable`/`time_unit`/
        `grid` triple.

        It should exactly when the request this was built from carried one. An analysis that
        fixes its own inputs (`lnd_type_plus`) takes no dataset and `resolve_analysis_chain`
        rejects one outright, so re-emitting a triple would turn a working spec's own data
        URL into a 400 -- while a plain filtered read has nothing else to say what to read.
        `partition.dataset` is that question already answered, by the same partition the
        pipeline was built from, so this cannot disagree with it.
        """
        return bool(self.decoded.partition.dataset)

    @property
    def dataset_params(self):
        """That triple as URL pairs -- `[]` when `names_dataset` is false.

        The caller's OWN triple, read back off the partition, and not the output one
        (`self.variable`/`self.grid`/`self.time_unit`). A data URL is this same request
        asked again, so what it has to carry is what to READ; the output triple is what
        comes back, and the two differ for any analysis that projects or folds. Emitting the
        output one was harmless while they agreed and stopped being harmless with
        `sediment_dynamics`, whose output has no single variable at all -- the URL came out
        carrying a literal `variable=None`, which is a 400 on the route it points at.

        `query_items` states the same rule for the notebook decode, off `self.params` --
        there because it is what the partition is built FROM and reading it back would be a
        cycle. Same triple either way: the partition's `dataset` is those three parameters
        sorted into their category, unchanged.
        """
        dataset = self.decoded.partition.dataset
        return [(name, dataset[name]) for name in ("variable", "time_unit", "grid") if name in dataset]

    def forward_query(self, params=None, *, compression=True):
        """This visualization's own request re-emitted as URL `(key, value)` pairs, for a
        data or tile URL the spec points at. `params` defaults to `self.params`.

        Three ways this is not a plain `urlencode(self.params)`:

        - **Render options are dropped.** `height`/`width`/`show_water`/... are category 5 of
          THIS route's partition and of no other's; `/v1/data.csv` has no visualization class
          to declare them against, so under `analysis=download` they are leftovers and a 400,
          and under a named analysis they reach its `extra="forbid"` `Parameters` and are a
          400 from further in. `api.models.RENDER_OPTIONS` derives the set from
          `VisualizationQuery` structurally, so a field added there cannot be missed here.
        - **The named analysis's own parameters are added back.** `dimension=`/`order=`/...
          have no fixed field set, so `VisualizationQuery` cannot declare them and FastAPI
          never binds them into `self.params` for a request-driven visualization;
          `partition.params` is precisely the set the decoder could attribute to no other
          category, which is precisely them. A notebook caller passes them as ordinary
          keyword arguments, so they are already in `self.params` and the merge is a no-op.
        - **`grid_cell` is comma-joined, not repeated.** It is one scalar parameter carrying
          a comma-separated list, unlike a dimension, which repeats.
        - **The dataset triple is dropped.** A notebook caller passes `variable`/`time_unit`/
          `grid` as ordinary keyword arguments, so they land in `self.params` with everything
          else, but whether an emitted URL should carry them is a question only
          `names_dataset` answers -- an analysis that fixes its own inputs 400s on a triple.
          Dropping them here leaves `get_data_uri` and `map._tile_url_template` as the one
          place that re-emits them, rather than having them arrive twice by two routes.

        `compression=False` is the tile routes: `partition_query(allow_compression=False)`
        rejects a compression parameter outright rather than ignoring it, and `Compression`'s
        fields have non-`None` defaults, so a request-driven visualization always has them in
        `self.params` and would always send them.

        Reads `self._decoded` -- the decode the ROUTE handed in -- rather than `self.decoded`,
        which is not the same thing for a notebook caller: `DatasetVisualization.decoded`
        builds its decode out of `query_items`, which calls this, so going through the
        property would be a cycle. There is nothing to recover in that case anyway; that is
        the branch where the analysis's parameters are already in `self.params`.
        """
        from cpra.mp.data.analysis.request.unified import COMPRESSION_OPTIONS, DATASET_SUGAR
        from cpra.mp.data.api.models import RENDER_OPTIONS

        params = self.params if params is None else params
        analysis_params = self._decoded.partition.params if self._decoded is not None else {}
        dropped = set(RENDER_OPTIONS) | DATASET_SUGAR
        if not compression:
            dropped |= COMPRESSION_OPTIONS

        pairs = []
        for key, value in {**params, **analysis_params}.items():
            if value is None or key in dropped:
                continue
            if isinstance(value, (list, tuple)):
                if key == "grid_cell":
                    pairs.append((key, ",".join(str(item) for item in value)))
                else:
                    pairs.extend((key, item) for item in value)
            else:
                pairs.append((key, value))
        return pairs

    def chart(self):
        """Build the visualization using Altair."""

        raise NotImplementedError

    def get_base_chart(
        self,
        source: alt.UrlData,
        time_column: str,
        date_column: str | None = "date",
    ) -> alt.Chart:
        """Generate a base chart from the given data source and options.

        One source, because every visualization now has one: `SedimentDynamics` was the only
        caller that passed several (five single-variable reads it joined here with a
        `transform_lookup` per source, keyed on the time column) and it reads the stacked
        output of the `sediment_dynamics` analysis instead. What went with that loop: the
        per-source `display_scale_factor` applied to the field a lookup landed in -- which
        was written twice, so a joined source was divided by its factor twice, and
        `sediment_dynamics`'s hover tooltip showed `elev` in metres/1000.
        """
        # The API serves storage units; every chart downstream reads display units, so the
        # conversion happens here, once, on the way in -- see visualizations.display. A
        # result that carries no single variable (an analysis that folds several) has no one
        # factor to apply and is left alone; `SedimentDynamics` scales per row instead.
        value_expr = display_value_expr(self.variable)

        calculate_exprs = {}
        if value_expr != "datum.value":
            calculate_exprs["value"] = value_expr

        if date_column:
            if time_column == "calendar_year":
                calculate_exprs[date_column] = "toDate(toString(datum.calendar_year))"
            else:
                calculate_exprs[date_column] = "datum.calendar_day"

        chart = alt.Chart(source)
        if calculate_exprs:
            chart = chart.transform_calculate(**calculate_exprs)
        return chart

    def get_data_uri(self, variable, grid, time_unit, **kwargs):
        """The `/v1/data.csv` URL for one of this spec's data sources.

        One route for every case now. What used to pick between three routes -- the flat
        dataset route, the dataset-scoped analysis route and the custom-analysis route -- is
        two ordinary query parameters on this one: `analysis=` (forwarded from `kwargs` like
        any other, rather than being lifted into a path segment), and the
        `variable`/`time_unit`/`grid` triple, emitted by `dataset_params` only when this
        request actually names a dataset.

        `kwargs` is the caller's parameters for this source -- `self.params`, less the
        dataset triple this takes as three arguments of its own. `forward_query` is what
        turns them into pairs; see it for why they are not simply urlencoded.

        The three arguments are the OUTPUT triple, and describe the frame that comes back:
        which columns it requires, which time column to select and parse. What the URL
        SENDS is `dataset_params` -- see it for why the two are not the same triple.
        """

        required_cols = (
            required_columns(pl.DataFrame({"variable": variable, "time_unit": time_unit, "grid": grid}))
            if variable is not None
            else []
        )
        select = [self.get_time_column(time_unit), "grid_cell", "value"]

        # A caller-supplied `select` (`SpeciesStackedBar.data_source`) replaces the computed
        # one outright, but still has to be filtered below, so it is taken out of `kwargs`
        # here and re-emitted as the single `select` parameter at the end rather than being
        # passed straight through by `forward_query`.
        caller_select = kwargs.pop("select", None)

        params = [*self.dataset_params, *self.forward_query(kwargs)]

        # A dimension the caller selected several values of stays a column of the result, so
        # the chart needs it named in `select` to tell the series apart.
        for param_name, param_values in kwargs.items():
            if isinstance(param_values, (list, tuple)) and param_name in required_cols and len(param_values) > 1:
                select.append(param_name)

        # A FOLDED result tells its rows apart the same way: `sediment_dynamics` returns one
        # row per (year, variable) with the variable name in a label column, so without that
        # column the five surfaces arrive as five unlabelled values per year. It is named by
        # the descriptor rather than assumed (`UnpivotStep.into` is the analysis's choice),
        # and is `None` for every unfolded result, which is all of them today but one.
        folded_into = self.decoded.descriptor.folded_into if self.decoded is not None else None
        if folded_into is not None:
            select.append(folded_into)

        if caller_select is not None:
            select = caller_select.split(",") if isinstance(caller_select, str) else list(caller_select)

        # A dimension the analysis COLLAPSES is gone from its output frame -- `diff`
        # over `model_group_id` returns one value per grid cell per day, not one per model
        # group -- so naming it in `select` asks for a column that no longer exists, and the
        # analysis routes fail the whole read ("unable to find column"). The two values are
        # exactly what a `diff` needs as FILTERS and still have to be SENT, so this
        # drops only the `select` entry, never the parameter itself.
        #
        # Filtering one finished list rather than guarding each place a column is added is
        # deliberate: a collapsed dimension reaches `select` three ways, and the multi-value
        # branch above is only one of them. The time column goes in unconditionally at the
        # top (so a `diff` over `calendar_day` hit the identical failure), and
        # `caller_select` bypasses both (so `SpeciesStackedBar` did too, over `calendar_year`).
        # `TimeSeriesLine`/`StackedBar` already exclude collapsed dimensions from their own
        # compare-channel choices, for the same underlying reason -- a collapsed dimension
        # has nothing left to compare across -- so this keeps the request in step with the
        # chart that consumes it. With no analysis selected, nothing is collapsed.
        resolved = self.decoded.analysis if self.decoded is not None else None
        if resolved is not None:
            select = [column for column in select if column not in resolved.collapsed_dimensions]

        # Set select param to limit columns returned.
        params.append(("select", ",".join(select)))

        if self.request:
            dataset_url = self.request.url_for("get_data", format="csv")
        else:
            dataset_url = "https://api.mpdp.coastal.la.gov/v1/data.csv"

        return f"{dataset_url}?{urlencode(params)}"

    def get_data_source(self, variable, grid, time_unit, name=None, parse=None, **kwargs):
        """Get an Altair data source for the given dataset and query parameters."""
        data_uri = self.get_data_uri(
            variable,
            grid,
            time_unit,
            **{k: v for k, v in kwargs.items() if v is not None},
        )

        if parse is None:
            parse = {"value": "number"}
            # A vector grid reports grid_cell as numeric grid cell IDs however the caller
            # asked for it; a raster grid reports the "lng,lat" string it was given.
            if GRIDS[grid].grid_type == "vector":
                parse["grid_cell"] = "number"
            time_column = self.get_time_column(time_unit)
            parse[time_column] = "date:%Y-%m-%d" if time_column == "calendar_day" else "number"

        # `name or variable or "value"`: a source is named for the variable it carries. An
        # output with no single variable (`lnd_type_plus`, `sediment_dynamics`) has nothing
        # to name it after, so it takes the name of the field it fills.
        return alt.UrlData(
            data_uri, format=alt.CsvDataFormat(type="csv", parse=parse), name=(name or variable or "value")
        )

    def get_time_column(self, time_unit):
        """Get the name of the time column."""
        return "calendar_year" if time_unit in ("decadal", "annual") else "calendar_day"

    def get_time_tooltip(self, time_column: str):
        """Create a tooltip for the given time column."""
        return alt.Tooltip(
            time_column,
            type="quantitative" if time_column == "calendar_year" else "temporal",
            title=DIMENSIONS[time_column].label,
        )

    def size_to_container(self, chart, is_concat: bool = False):
        """Size a finished chart to exactly fill `self.width` x `self.height` (the frontend's
        container), whatever room the axes/title/legend end up taking.

        In Vega-Lite, a mark's `width`/`height` describe the plotting area only — the
        rendered SVG is that area plus the surrounding chrome, so a naive `.properties(width=,
        height=)` on the container size leaves the chrome to overflow it. `autosize`'s `fit`
        mode reinterprets `width`/`height` as the *total* size instead and shrinks the
        plotting area to make room, which is exactly what's wanted here since it measures real
        text metrics rather than guessing at legend width the way this code used to.

        `fit` only works for single or layered views, though — Vega-Lite rejects it on a
        vconcat spec (warns and silently falls back to `pad`, which overflows again exactly
        like the bug this is fixing). `fit-x` is the documented workaround: it fits the width
        only, which vconcat does support, and is enough on its own because concat's overflow
        here only ever came from the legend pushing the width out — the height side of a
        concat spec still has to be sized by hand (see the `- 222` arithmetic in
        `TimeSeriesLine`/`StackedBar` for the daily/vconcat branch). Don't "simplify" this back
        to `fit` for a concat chart; it will silently stop being honored.

        A top-level vconcat spec has no `width`/`height` property at all in the Vega-Lite
        schema — sizing there comes entirely from each concatenated child's own `width`/
        `height` (set where the child is built, e.g. `chart_width`/`chart_height` in
        `TimeSeriesLine`/`StackedBar`). So for a concat chart this method only sets
        `autosize`; `fit-x` then stretches each child's plotting area so the child's own
        declared width is honored as a total rather than a plot-area-only size, the same
        way `fit` does for a single/layered chart's top-level width.

        Apply this to the final, top-level chart object only (e.g. after a `&` concat) — a
        nested/intermediate layer's `properties(width=, height=)` is what actually gets fit,
        and putting autosize somewhere other than the top level has no effect.
        """
        autosize = alt.AutoSizeParams(type="fit-x" if is_concat else "fit", contains="padding")
        if is_concat:
            return chart.properties(autosize=autosize)
        return chart.properties(width=self.width, height=self.height, autosize=autosize)

    @property
    def spec(self):
        """Get the Vega-Lite specification."""
        chart = self.chart()
        return chart.to_dict()

    @cached_property
    def descriptor(self):
        """What this request's pipeline PRODUCES -- the render context every property below
        is derived from.

        This was three constructor arguments, which is what let the route and a notebook give
        different answers to the same question. `grid` and `time_unit` are facts about the
        output frame, not about what the caller typed: an analysis may project onto a grid
        none of its inputs is on (`lnd_type_plus` outputs `morph_pixel_v002`) or fold several
        variables into one categorical result. The route had already been deriving them from
        `output_descriptor`; a notebook was still passing its own input triple straight
        through. Reading them off `decoded` here means there is one derivation, and the
        triple goes back to being what it is in a URL -- an ordinary request parameter, in
        `self.params` with the rest of them.
        """
        return self.decoded.descriptor

    @cached_property
    def variable(self):
        """The one variable the result is about, or `None` when it carries none: the analysis
        that produced it folded several (`lnd_type_plus` reads three), so there is no single
        presentation variable to take a label, a unit or a legend from.

        `None` also means this request named no dataset -- an analysis that fixes its own
        inputs is the only way to get here -- which is what `names_dataset` keys off.
        """
        return self.descriptor.variable

    @cached_property
    def grid(self):
        """The grid the result is ON, which is not always a grid its inputs are on."""
        return GRIDS[self.descriptor.grid]

    @cached_property
    def time_unit(self):
        """The time unit the result steps by, or `None` when it has no time column left."""
        return self.descriptor.time_unit

    @cached_property
    def decoded(self):
        """This visualization's request, decoded -- the route's own decode when there was a
        route, and one performed here when there wasn't.

        The fallback is the notebook case: `Map(variable="elev", grid=..., time_unit=...,
        calendar_year=[2050], analysis="diff", dimension="model_group_id")` has to reach the
        same `Pipeline` the equivalent URL would, and re-deriving it here from
        `self.params` -- through the very same `partition_query`/`resolve_analysis_chain`/
        `build_pipeline` sequence -- is the only way to be sure it does. `decode_request_sync`
        exists for exactly this: a spec is built from synchronous cached properties, with no
        event loop to await the async form on.

        This is also what replaced `map.py`'s `_data_query_pipeline`, which lowered the
        no-`analysis=` case onto a `Pipeline` by hand through `build_data_query_pipeline`.
        There is no second path to keep in step now: analysis or not, notebook or route, one
        decoder answers.
        """
        if self._decoded is not None:
            return self._decoded

        # Imported lazily: `cpra.mp.data.api`'s `__init__.py` imports `VISUALIZATIONS` from
        # this package at module level, so importing it back from here at module level would
        # be a circular import -- see `map.py`'s `feature_colors`, which does the same for
        # its own engine imports.
        from cpra.mp.data.api.models import RENDER_OPTIONS
        from cpra.mp.data.api.unified import decode_request_sync

        # `validate_sink=False`, for the same reason `api.__init__.get_visualization` passes
        # it on the route half: `format="csv"` here names the format the SPEC's data URLs
        # will ask for, not a sink this decode writes to. A visualization over a raster
        # dataset has a band-wide final frame until one of its data sources adds a
        # `grid_cell`, so validating the tabular sink against this pipeline would reject
        # every raster visualization before its first chart was built. The two sink rules
        # still bind where they mean something -- on the `/v1/data.{format}` request the
        # emitted URL turns into.
        return decode_request_sync(
            self.query_items(),
            format="csv",
            render_fields=RENDER_OPTIONS,
            validate_sink=False,
        )

    def query_items(self):
        """This visualization's constructor arguments as the raw `(key, value)` query pairs
        `decode_request_sync` takes -- the notebook half of `decoded` above.

        The dataset triple is re-attached straight from `self.params`, and NOT from
        `self.variable`/`self.grid`/`self.time_unit`: those are derived from `descriptor`,
        which is derived from `decoded`, which is built out of this -- reading them here
        would be a cycle. It is also the wrong triple to send. These are the datasets to
        READ; the properties describe what comes back, and for an analysis that projects,
        the two differ. A caller that named none simply has none of the three in `params`,
        which is exactly the request `resolve_analysis_chain` wants for an analysis that
        fixes its own inputs.

        `forward_query` drops the triple for this reason -- so it is emitted here, once,
        rather than arriving again among the ordinary parameters.

        Values are stringified, since a query string is what the decoder parses -- a notebook
        writing `calendar_year=[2050, 2060]` and a URL writing
        `calendar_year=2050&calendar_year=2060` have to arrive identically, and `forward_query`
        already renders exactly that (minus the render options, which the decoder would only
        put back in a category this class never reads).
        """
        dataset = [
            (name, str(self.params[name]))
            for name in ("variable", "time_unit", "grid")
            if self.params.get(name) is not None
        ]
        return [*dataset, *((key, str(value)) for key, value in self.forward_query())]

    @cached_property
    def source_params(self):
        """`self.params` without the dataset triple: the parameters a data source's URL
        carries, as against the dataset it reads.

        A notebook caller passes `variable`/`time_unit`/`grid` as ordinary keyword arguments
        -- there is nowhere else for them to go now that they are not constructor arguments
        -- so they land in `params` with everything else. `get_data_source` takes them as
        three arguments of its own and would get two values for each.
        """
        from cpra.mp.data.analysis.request.unified import DATASET_SUGAR

        return {key: value for key, value in self.params.items() if key not in DATASET_SUGAR}

    @cached_property
    def data_source(self):
        """Get the Altair data source.

        The triple handed down is the OUTPUT one -- what the chart has to parse, index and
        label by. Whether a URL should carry one at all is `get_data_uri`'s question, and
        `names_dataset` is how it answers.
        """
        variable = self.variable.name if self.variable is not None else None
        return self.get_data_source(variable, self.grid.name, self.time_unit, **self.source_params)

    @cached_property
    def _analysis_name(self):
        """The `analysis=` request parameter, or `None` when not given."""
        return self.params.get("analysis")

    @cached_property
    def resolved_analysis(self):
        """The `ResolvedAnalysis` behind this visualization -- its `Pipeline` plus the
        presentation metadata that overrides what the pipeline alone would say
        (`.legend`/`.result_kind`/`.collapsed_dimensions`/`.title_suffix`/`.meta`) -- or
        `None` when no `analysis=` was selected, which every hook in this module (and in
        `map.py`/`time_series_line.py`/`stacked_bar.py`) reads as "plain dataset behavior".

        Read straight off `decoded`, which is the decode the route already performed (or,
        for a notebook caller, the one `decoded` performs for itself). Nothing is resolved
        here any more: the four properties this replaced -- `_analysis_is_custom`,
        `_analysis_filters`, `_analysis_params` and this one's own body -- existed to
        reproduce, on this side, the filters/params split and the dataset-or-not routing the
        route layer was doing on its own side. Both sides now run one decoder, so there is
        nothing left to keep in agreement.

        In particular there is no `type="custom"` fork left. Whether an analysis takes the
        caller's dataset is a fact its own `pipeline()` states (`require_dataset`/
        `reject_dataset`), a caller who names a dataset it rejects gets
        `AnalysisDatasetMismatchError` -> 400, and this class simply does not name one when
        `self.variable is None` (see `query_items`).
        """
        return self.decoded.analysis if self.decoded is not None else None

    @cached_property
    def difference_suffix(self):
        """Suffix for the chart title and y-axis if this is a difference chart."""
        resolved = self.resolved_analysis
        if resolved is not None:
            return resolved.title_suffix
        return ""

    @cached_property
    def grid_cells(self):
        """Get the selected grid cells as the API keys them in its own output.

        A vector grid takes either grid cell IDs or lat/lng coordinate pairs; a raster
        grid only takes coordinates. These are what a `grid_cell` value in the response
        will actually equal, so they're what a scale domain or a lookup key has to match —
        see `grid_cell_labels` for what to show the reader instead.
        """
        return [grid_cell for grid_cell, _, _, _ in self._grid_cells]

    @cached_property
    def grid_cell_labels(self):
        """Get a display label per selected grid cell, in the same order as `grid_cells`.

        Identifies a grid cell in full, for somewhere with room for it — a legend. A
        coordinate is shown parenthesized, and on a vector grid it's prefixed with the ID of
        the cell it landed in, since the cell — not the point — is what the values describe:
        `24`, `24 (-90.49, 30.01)`, or `(-90.49, 30.01)` on a raster grid.
        """
        return [label for _, label, _, _ in self._grid_cells]

    @cached_property
    def grid_cell_short_labels(self):
        """Get a compact display label per selected grid cell, in the same order.

        For somewhere a full label would crowd out the numbers being read, like a tooltip
        with one row per grid cell. Where a grid cell ID is available it stands alone, since
        it's what the values describe and the coordinate that selected it only repeats what
        the legend already says. A raster pixel has no ID, so the coordinate is all there is
        to show. Two coordinates inside one cell therefore share a short label; the legend is
        what tells them apart.
        """
        return [short_label for _, _, short_label, _ in self._grid_cells]

    @cached_property
    def grid_cell_colors(self):
        """Get the color the map draws each selected grid cell with, in the same order.

        A chart is read beside the map, so a grid cell has to carry the same color in both.
        The map's color comes from the grid cell's place in the *selection*, which is not its
        place in this list: a coordinate that fell outside the grid, or inside a cell an
        earlier coordinate already took, is left out here and takes its color out with it.
        Pairing the whole palette with this list positionally would instead shift every
        series after such a coordinate onto the color of the pin before it.
        """
        return [color for _, _, _, color in self._grid_cells]

    @cached_property
    def _grid_cells(self):
        """One `(API grid_cell key, label, short label, color)` tuple per selected grid cell.

        A raster grid is keyed by the coordinate itself, reported as a `"lng,lat"` string
        built from the parsed floats — so round-trip the tokens through `float` to match
        it (`-91` in the request becomes `-91.0` in the response). A vector grid is always
        keyed by grid cell ID, whichever way the caller asked for it, so a coordinate has
        to be resolved to a cell the same way the data endpoint does. That also means a
        coordinate outside the grid drops out, as it does there, and coordinates sharing a
        cell collapse into the one series the response actually carries.

        The color is the map's, and so is taken from the grid cell's place in the selection
        rather than from its place in this list, which dropping and collapsing shorten.
        """
        grid_cell = self.params["grid_cell"]

        if self.grid.grid_type == "vector" and is_grid_cell_ids(grid_cell):
            ids = [int(token) for token in grid_cell_tokens(grid_cell)]
            return [
                (grid_cell_id, str(grid_cell_id), str(grid_cell_id), selection_color(i))
                for i, grid_cell_id in enumerate(ids)
            ]

        coords = [float(token) for token in grid_cell_tokens(grid_cell)]
        pairs = list(zip(coords[::2], coords[1::2]))

        if self.grid.grid_type != "vector":
            return [
                (f"{lng!s},{lat!s}", f"({lng}, {lat})", f"({lng}, {lat})", selection_color(i))
                for i, (lng, lat) in enumerate(pairs)
            ]

        # Imported lazily: this pulls in the geospatial stack, and a notebook that only
        # builds specs shouldn't need it. Same reason map.py imports the analysis engine
        # where it's used rather than at module scope.
        from cpra.mp.data.analysis.io.geography import resolve_vector_grid_cell_sync

        ids = resolve_vector_grid_cell_sync(self.grid.name, [p[0] for p in pairs], [p[1] for p in pairs])

        grid_cells = {}
        for i, ((lng, lat), grid_cell_id) in enumerate(zip(pairs, ids)):
            # First coordinate wins a shared cell: the response has one series for it, so a
            # second entry would be a legend key the data never matches. Its color is the
            # first coordinate's too, which is how the map draws the cell's outline.
            if grid_cell_id is not None and grid_cell_id not in grid_cells:
                grid_cells[grid_cell_id] = (
                    grid_cell_id,
                    f"{grid_cell_id} ({lng}, {lat})",
                    str(grid_cell_id),
                    selection_color(i),
                )
        return list(grid_cells.values())

    @cached_property
    def time_column(self):
        """Get the name of the time column."""
        return self.get_time_column(self.time_unit)

    @cached_property
    def time_dimension(self):
        """Get the temporal dimension."""
        return DIMENSIONS[self.time_column]

    @cached_property
    def result_label(self):
        """What this visualization's values are called, for a title or an axis.

        The variable's own label wherever there is one variable. Where there isn't -- an
        analysis that folds several into one result, so `output_descriptor(...).variable` is
        `None` -- the analysis's own `Meta.label` is the only thing that names the result,
        and is exactly the override `descriptor.py` declines to fold in on the grounds that
        `analyses/` imports `request/` and not the other way round. The last fallback is for
        neither: a multi-input pipeline written out by hand, which nothing names.
        """
        if self.variable is not None:
            return self.variable.label
        resolved = self.resolved_analysis
        return resolved.meta.label if resolved is not None else "Value"

    @cached_property
    def title(self):
        """Get the chart title."""
        title = self.result_label
        aggregate_type = self.params.get("aggregate_type", None)
        if aggregate_type and len(aggregate_type) == 1:
            aggregate_label = DIMENSIONS["aggregate_type"].values.filter(value=aggregate_type[0])["label"][0]
            title = f"{aggregate_label} {title}"
        # `time_unit` is `None` when the pipeline's inputs disagree on one, which no analysis
        # in the registry does today but which `OutputDescriptor` allows for; the unit is
        # simply left off the title rather than guessed at.
        prefix = f"{self.time_unit.title()} " if self.time_unit else ""
        return f"{prefix}{title}{self.difference_suffix}"

    @cached_property
    def variable_label(self):
        """Get the variable label and unit.

        The unit shown is the display unit, matching the scaling `get_base_chart` applies
        to the values this labels. A result with no single variable has no unit to show, and
        `display_unit` answers `None` for it.
        """
        label = f"{self.result_label}{self.difference_suffix}"
        unit = display_unit(self.variable)
        if unit:
            label += f" ({unit})"
        return label
