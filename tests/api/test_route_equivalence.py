"""The route-consolidation gate: every one of the 17 deleted routes, and the one-route
spelling that replaced it, resolve to the same `Pipeline`.

`pipeline_to_query` (`analysis/request/pipeline.py:994`) is a canonical encoding --
natural-sorted keys, defaults and `None`s omitted -- so two pipelines that are equivalent
after normalization serialize to the identical parameter set. Comparing THAT, rather than a
response body, is the whole point of this file: the failure this consolidation could
plausibly cause is a MISPARTITION, a `model_group_id` read as a named analysis's own
parameter instead of as a dimension filter. That is a 200 with the wrong rows in it, and no
status-code or even row-count assertion would see it. Here it is a diff in one string.

The 17 routes, and where each is covered:

| deleted route                                             | covered by            |
| --------------------------------------------------------- | -------------------- |
| `/v1/datasets/{v}/{t}/{g}/data.{format}`                    | `plain-*`            |
| `/v1/datasets/{v}/{t}/{g}/data/{z}/{x}/{y}.{format}`        | `plain-raster`       |
| `/v1/datasets/{v}/{t}/{g}/tiles/{z}/{x}/{y}.pbf`            | `plain-vector`       |
| `/v1/datasets/{v}/{t}/{g}/visualization/{name}.json`        | `plain-vector`       |
| `/v1/datasets/{v}/{t}/{g}/analyses/{a}/data.{format}`       | `dataset-analysis-*` |
| `/v1/datasets/{v}/{t}/{g}/analyses/{a}/data/{z}/{x}/{y}.{format}` | `dataset-analysis-raster` |
| `/v1/datasets/{v}/{t}/{g}/analyses/{a}/tiles/{z}/{x}/{y}.pbf`     | `dataset-analysis-vector` |
| `/v1/datasets/{v}/{t}/{g}/analyses/{a}/pipeline.json`       | `dataset-analysis-vector` |
| `/v1/analyses/{a}/data.{format}`                            | `fixed-input-*`      |
| `/v1/analyses/{a}/data/{z}/{x}/{y}.{format}`                | `fixed-input-raster` |
| `/v1/analyses/{a}/tiles/{z}/{x}/{y}.pbf`                    | `fixed-input-vector` |
| `/v1/analyses/{a}/pipeline.json`                            | `fixed-input-raster` |
| `POST /v1/analysis.{format}`                                | `dotted-vector`      |
| `GET /v1/analysis.{format}`                                 | `dotted-*`           |
| `GET /v1/analysis/{z}/{x}/{y}.{format}`                     | `dotted-raster`      |
| `GET /v1/visualizations/{name}.json`                        | see below            |
| `GET /v1/datasets/{v}/{t}/{g}/catalog.json`                 | `plain-vector`       |

One of the 17 has no pipeline to compare, and is covered elsewhere:

- `/v1/visualizations/{name}.json` survives as the one spec route rather than being
  replaced by it. What it used to serve alone -- a visualization that builds its own
  sources and decodes nothing -- still has no pipeline;
  `test_endpoints_data.py::TestVisualizationEndpoints` covers the merge at the route level.

`GET /v1/analysis/{z}/{x}/{y}.pbf` and `GET /v1/analysis/{z}/{x}/{y}.{format}` were two
declarations of one route, split only because Starlette dispatches on first path match and
`{format}` would otherwise have swallowed `.pbf` (see `get_data_tile`'s docstring); they
count as one here.

Asserted at the DECODER rather than through the ASGI app, and deliberately: these need no
fixture data on disk, `decode_request` is what every one of the four routes actually calls,
and the arguments each route passes it are few enough to reproduce exactly (see `_decode`).
The route-level behaviour is covered by `test_endpoints_data.py` and
`test_analyses_endpoints.py`.
"""

from dataclasses import dataclass
from urllib.parse import parse_qsl

import pytest

from cpra.mp.data.analysis.request.pipeline import parse_pipeline, pipeline_to_query
from cpra.mp.data.api.models import RENDER_OPTIONS
from cpra.mp.data.api.unified import decode_request_sync


@dataclass(frozen=True)
class Spelling:
    """One deleted route and the request that replaces it.

    `route` is the deleted route's path template, `new` the path it is spelled against now
    (the family's query string rides on all of them identically -- that IS the consolidation)
    and `format` the `output.format` the resolved pipeline should carry.
    """

    route: str
    new: str
    format: str


@dataclass(frozen=True)
class Family:
    """One request, spelled every way the old route table made you spell it.

    `pipeline` is the canonical `pipeline_to_query` encoding WITHOUT `output.format`, which
    is the one key a spelling is allowed to change -- the URL's extension overwrites it on
    every route that has one, so that a saved pipeline can be refetched as `.csv` and as
    `.png` without becoming a different request.
    """

    id: str
    query: str
    spellings: tuple[Spelling, ...]
    pipeline: dict[str, str]


VECTOR = "variable=ssf&time_unit=annual&grid=ecoregion_v001"
RASTER = "variable=vct&time_unit=annual&grid=veg_grid_cell_v001"
DIFFERENCE = "analysis=diff&dimension=calendar_year&order=asc&calendar_year=2030&calendar_year=2040"

PLAIN_VECTOR = {
    "input.data.grid": "ecoregion_v001",
    "input.data.time_unit": "annual",
    "input.data.variable": "ssf",
    "output.sort": "grid_cell",
    "step.1.input": "data",
    "step.1.op": "filter",
    "step.1.where": "calendar_year = 2030",
}

PLAIN_RASTER = {
    "input.data.grid": "veg_grid_cell_v001",
    "input.data.time_unit": "annual",
    "input.data.variable": "vct",
    "step.1.input": "data",
    "step.1.op": "filter",
    "step.1.where": "calendar_year = 2030",
}

# `diff`'s own three steps, with the caller's two-year selection lowered into the leading
# filter rather than appended: `Difference` states its own selection, and `group.by` is the
# dimensions it did NOT collapse, which differ per grid (a raster dataset has no
# `percentile`).
def _difference(grid: str, variable: str, by: str) -> dict[str, str]:
    return {
        "input.x.grid": grid,
        "input.x.time_unit": "annual",
        "input.x.variable": variable,
        "step.1.input": "x",
        "step.1.op": "filter",
        "step.1.where": "calendar_year IN (2030, 2040)",
        "step.2.cases.1.then": "value * -1",
        "step.2.cases.1.when": "calendar_year = 2040",
        "step.2.else": "value",
        "step.2.op": "case",
        "step.3.aggregate": "sum",
        "step.3.by": by,
        "step.3.op": "group",
    }


# `lnd_type_plus` fixes all three of its inputs. Every value the caller sent is a dimension
# name, and every one of them is in `step.4.where` -- the trailing selection filter -- rather
# than anywhere in the analysis's own steps. That is the mispartition this file exists for.
LND_TYPE_PLUS = {
    "input.elevation.grid": "morph_pixel_v002",
    "input.elevation.time_unit": "annual",
    "input.elevation.variable": "elev",
    "input.land_type.grid": "morph_pixel_v002",
    "input.land_type.time_unit": "annual",
    "input.land_type.variable": "lnd_type",
    "input.stage.grid": "hydro_compartment_v002",
    "input.stage.time_unit": "annual",
    "input.stage.variable": "stg",
    "step.1.as": "selected_stage",
    "step.1.input": "stage",
    "step.1.op": "filter",
    "step.1.where": "aggregate_type = 'mean'",
    "step.2.as": "stage_m",
    "step.2.input": "selected_stage",
    "step.2.op": "project",
    "step.2.to": "morph_pixel_v002",
    "step.3.as": "paired",
    "step.3.inputs": "stage_m,elevation,land_type",
    "step.3.op": "join",
    "step.3.type": "spatial",
    "step.4.as": "selected",
    "step.4.op": "filter",
    "step.4.where": "calendar_year = 2050 AND model_group_id = 705 AND scenario_id = 7",
    "step.5.cases.1.then": "NULL",
    "step.5.cases.1.when": "stage_m IS NULL",
    "step.5.cases.2.then": "2",
    "step.5.cases.2.when": "land_type = 2 AND stage_m - elevation / 1000 > 1.0",
    "step.5.cases.3.then": "7",
    "step.5.cases.3.when": "land_type = 2 AND stage_m - elevation / 1000 > 0.46",
    "step.5.cases.4.then": "6",
    "step.5.cases.4.when": "land_type = 2",
    "step.5.else": "land_type",
    "step.5.op": "case",
}

# `sal_veg_investigation`, likewise fixed-input, plus the appended `project` that makes it a
# legal `.pbf`. Every registered fixed-input analysis is raster-output today, so the deleted
# `/v1/analyses/{a}/tiles/{z}/{x}/{y}.pbf` could not in fact have served one -- the appended
# step is what a caller would have to write now, and it is a combination none of the 17 could
# express, which is the other half of what consolidation bought.
SAL_VEG_PROJECTED = {
    "input.coverage.grid": "veg_grid_cell_v001",
    "input.coverage.time_unit": "annual",
    "input.coverage.variable": "ffibs_coverage_ratio",
    "input.salinity.grid": "hydro_compartment_v001",
    "input.salinity.time_unit": "annual",
    "input.salinity.variable": "sal",
    "step.1.as": "marsh",
    "step.1.input": "coverage",
    "step.1.op": "filter",
    "step.1.where": "ffibs_type = 'pl_fm'",
    "step.2.as": "selected_salinity",
    "step.2.input": "salinity",
    "step.2.op": "filter",
    "step.2.where": "aggregate_type = '2_wk_max'",
    "step.3.as": "sal",
    "step.3.input": "selected_salinity",
    "step.3.op": "project",
    "step.3.to": "veg_grid_cell_v001",
    "step.4.as": "paired",
    "step.4.inputs": "marsh,sal",
    "step.4.op": "join",
    "step.4.type": "spatial",
    "step.5.op": "filter",
    "step.5.where": "model_group_id = 500 AND scenario_id = 7",
    "step.6.cases.1.then": "marsh",
    "step.6.cases.1.when": "marsh > 0.5 AND sal > 5.5",
    "step.6.op": "case",
    "step.7.aggregate": "min",
    "step.7.by": "",
    "step.7.of": "calendar_year",
    "step.7.op": "group",
    "step.8.aggregate": "mean",
    "step.8.op": "project",
    "step.8.to": "ecoregion_v001",
}

DOTTED_PIPELINE = (
    "step.1.op=filter&step.1.input=data&step.1.where=calendar_year+%3D+2030"
)

FAMILIES = [
    Family(
        id="plain-vector",
        query=f"{VECTOR}&calendar_year=2030&sort=grid_cell",
        spellings=(
            Spelling("/v1/datasets/{v}/{t}/{g}/data.{format}", "/v1/data.csv", "csv"),
            Spelling(
                "/v1/datasets/{v}/{t}/{g}/tiles/{z}/{x}/{y}.pbf", "/v1/data/3/1/2.pbf", "pbf"
            ),
            Spelling(
                "/v1/datasets/{v}/{t}/{g}/visualization/{name}.json",
                "/v1/visualizations/time-series-line.json",
                "csv",
            ),
            Spelling(
                "/v1/datasets/{v}/{t}/{g}/catalog.json", "/v1/catalog.json", "csv"
            ),
        ),
        pipeline=PLAIN_VECTOR,
    ),
    Family(
        id="plain-raster",
        query=f"{RASTER}&calendar_year=2030",
        spellings=(
            Spelling("/v1/datasets/{v}/{t}/{g}/data.{format}", "/v1/data.tif", "tif"),
            Spelling(
                "/v1/datasets/{v}/{t}/{g}/data/{z}/{x}/{y}.{format}", "/v1/data/3/1/2.png", "png"
            ),
        ),
        pipeline=PLAIN_RASTER,
    ),
    Family(
        id="dataset-analysis-vector",
        query=f"{VECTOR}&{DIFFERENCE}",
        spellings=(
            Spelling(
                "/v1/datasets/{v}/{t}/{g}/analyses/{a}/data.{format}", "/v1/data.csv", "csv"
            ),
            Spelling(
                "/v1/datasets/{v}/{t}/{g}/analyses/{a}/tiles/{z}/{x}/{y}.pbf",
                "/v1/data/3/1/2.pbf",
                "pbf",
            ),
            Spelling(
                "/v1/datasets/{v}/{t}/{g}/analyses/{a}/pipeline.json",
                "/v1/analyses/diff.json",
                "csv",
            ),
        ),
        pipeline=_difference("ecoregion_v001", "ssf", "model_group_id,scenario_id,percentile"),
    ),
    Family(
        id="dataset-analysis-raster",
        query=f"{RASTER}&{DIFFERENCE}",
        spellings=(
            Spelling(
                "/v1/datasets/{v}/{t}/{g}/analyses/{a}/data.{format}", "/v1/data.tif", "tif"
            ),
            Spelling(
                "/v1/datasets/{v}/{t}/{g}/analyses/{a}/data/{z}/{x}/{y}.{format}",
                "/v1/data/3/1/2.png",
                "png",
            ),
        ),
        pipeline=_difference("veg_grid_cell_v001", "vct", "model_group_id,scenario_id"),
    ),
    Family(
        id="fixed-input-raster",
        query="analysis=lnd_type_plus&model_group_id=705&scenario_id=7&calendar_year=2050",
        spellings=(
            Spelling("/v1/analyses/{a}/data.{format}", "/v1/data.tif", "tif"),
            Spelling("/v1/analyses/{a}/data/{z}/{x}/{y}.{format}", "/v1/data/3/1/2.png", "png"),
            Spelling("/v1/analyses/{a}/pipeline.json", "/v1/analyses/lnd_type_plus.json", "tif"),
        ),
        pipeline=LND_TYPE_PLUS,
    ),
    Family(
        id="fixed-input-vector",
        query=(
            "analysis=sal_veg_investigation&model_group_id=500&scenario_id=7"
            "&step.1.op=project&step.1.to=ecoregion_v001&step.1.aggregate=mean"
        ),
        spellings=(
            Spelling("/v1/analyses/{a}/data.{format}", "/v1/data.csv", "csv"),
            Spelling("/v1/analyses/{a}/tiles/{z}/{x}/{y}.pbf", "/v1/data/3/1/2.pbf", "pbf"),
        ),
        pipeline=SAL_VEG_PROJECTED,
    ),
    Family(
        id="dotted-vector",
        query=(
            "input.data.variable=ssf&input.data.time_unit=annual"
            f"&input.data.grid=ecoregion_v001&{DOTTED_PIPELINE}"
        ),
        spellings=(
            Spelling("GET /v1/analysis.{format}", "/v1/data.csv", "csv"),
            Spelling("POST /v1/analysis.{format}", "POST /v1/data.csv", "csv"),
        ),
        pipeline={k: v for k, v in PLAIN_VECTOR.items() if k != "output.sort"},
    ),
    Family(
        id="dotted-raster",
        query=(
            "input.data.variable=vct&input.data.time_unit=annual"
            f"&input.data.grid=veg_grid_cell_v001&{DOTTED_PIPELINE}"
        ),
        spellings=(
            Spelling("GET /v1/analysis.{format}", "/v1/data.tif", "tif"),
            Spelling("GET /v1/analysis/{z}/{x}/{y}.{format}", "/v1/data/3/1/2.png", "png"),
        ),
        pipeline=PLAIN_RASTER,
    ),
]

CASES = [(family, spelling) for family in FAMILIES for spelling in family.spellings]

# The one of the 17 with no pipeline to compare. See the module docstring.
UNCOVERED = {"GET /v1/visualizations/{name}.json"}


def _decode(spelling: Spelling, query: str):
    """`decode_request` with exactly the arguments `spelling.new`'s route passes it.

    Five branches for five routes (`api/__init__.py`): the spec route decodes for a render
    context and so skips the sink rules, the tile route rejects a compression parameter
    rather than ignoring it, the pipeline-JSON and catalog routes have no `{format}` segment
    to overwrite `output.format` from (and the catalog route, serving JSON metadata, rejects
    compression as well), and `/v1/data.{format}` is the plain case.
    """
    path, _, extension = spelling.new.rpartition(".")

    if path == "/v1/catalog":
        return decode_request_sync(
            parse_qsl(query), format=None, allow_compression=False, validate_sink=False
        )
    if path.startswith("/v1/visualizations/"):
        return decode_request_sync(
            parse_qsl(query), format="csv", render_fields=RENDER_OPTIONS, validate_sink=False
        )
    if path.startswith("/v1/analyses/"):
        return decode_request_sync(parse_qsl(query), format=None)
    if path.startswith("/v1/data/"):
        return decode_request_sync(parse_qsl(query), format=extension, allow_compression=False)
    return decode_request_sync(parse_qsl(query), format=extension)


@pytest.mark.parametrize(
    ("family", "spelling"), CASES, ids=[f"{f.id}:{s.route}" for f, s in CASES]
)
def test_the_new_spelling_resolves_to_what_the_old_route_did(family, spelling):
    assert pipeline_to_query(_decode(spelling, family.query).pipeline) == {
        **family.pipeline,
        "output.format": spelling.format,
    }


@pytest.mark.parametrize("family", FAMILIES, ids=[f.id for f in FAMILIES])
def test_one_request_is_one_pipeline_whatever_it_is_fetched_as(family):
    """The consolidation's actual claim, stated without reference to the goldens above: the
    same query string means the same thing on all four routes, and `output.format` is the
    only thing an extension is allowed to change. This is what lets a client save one
    pipeline and refetch it as `.csv`, as `.png` and as a spec.
    """
    encoded = {}
    for spelling in family.spellings:
        query = pipeline_to_query(_decode(spelling, family.query).pipeline)
        encoded[spelling.route] = {k: v for k, v in query.items() if k != "output.format"}

    assert len(set(map(str, encoded.values()))) == 1, encoded


def test_every_deleted_route_is_accounted_for():
    """A coverage assertion, not a behaviour one: the point of this file is that NO old
    route was left un-reasoned-about, so a route missing from the table above should fail
    here rather than silently reduce what is checked.
    """
    covered = {spelling.route for _, spelling in CASES} | UNCOVERED
    assert len(covered) == 17


@pytest.mark.parametrize("family", FAMILIES, ids=[f.id for f in FAMILIES])
def test_the_encoding_is_canonical_not_merely_reversible(family):
    """Re-decoding a pipeline's own encoding has to land on the same pipeline, or the
    equality above would be comparing two spellings of one thing rather than one thing.
    This is also the POST route's equivalence, stated once: `parse_pipeline` of a body is
    the same object the dotted query form decodes to.
    """
    pipeline = _decode(family.spellings[0], family.query).pipeline
    assert pipeline_to_query(parse_pipeline(pipeline.model_dump(mode="json"))) == (
        pipeline_to_query(pipeline)
    )


@pytest.mark.parametrize(
    ("family_id", "dimensions"),
    [
        ("dataset-analysis-vector", ["calendar_year"]),
        ("fixed-input-raster", ["calendar_year", "model_group_id", "scenario_id"]),
        ("fixed-input-vector", ["model_group_id", "scenario_id"]),
    ],
)
def test_a_dimension_value_is_always_a_filter_never_an_analysis_parameter(family_id, dimensions):
    """The failure mode the goldens are here to catch, asserted structurally so that a
    golden blessed in error still fails: every dimension value the caller sent has to appear
    in some `filter` step's `where`. If one had been read as the analysis's own parameter it
    would have changed the analysis's steps instead and left no conjunct behind -- a 200 with
    the wrong rows, invisible at the response level.
    """
    family = next(f for f in FAMILIES if f.id == family_id)
    query = pipeline_to_query(_decode(family.spellings[0], family.query).pipeline)
    wheres = " ".join(value for key, value in query.items() if key.endswith(".where"))

    for dimension in dimensions:
        assert dimension in wheres, query
