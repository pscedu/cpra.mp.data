import altair as alt

hc_topo_url = (
    "https://cpra-mpdp.bridges2.psc.edu/raw_data/grids/hydro_compartment_v001.json"
)
data_url = "https://api.mpdp.coastal.la.gov/v1/datasets/sal/hydro_compartment_v001/annual/data.csv?scenario_id=7&model_group_id=500&aggregate_type=mean&calendar_year=2030&melt=true"


topo = alt.topo_feature(hc_topo_url, "hydro_compartment_v001")
source = alt.UrlData(data_url, format=alt.CsvDataFormat(type="csv"), name="salinity")

viz = (
    alt.Chart(topo, title="Salinity: S07 G500 2030")
    .mark_geoshape()
    .encode(
        color=alt.Y("value:Q").title("Salinity (PPM)"),
    )
    .transform_lookup(lookup="id", from_=alt.LookupData(source, "geography", ["value"]))
    .project(type="mercator")
    .properties(width="container", height=800)
)

viz.save("altair_map.html")
