from cpra.mp.data.colors import COLORS


class Legend:
    def __init__(self, name, colors, labels, breaks=None):
        self.name = name
        self.colors = colors if isinstance(colors, list) else COLORS[colors]
        self.labels = labels
        self.breaks = breaks


LEGENDS = {}


def add_legend(**kwargs):
    """Create a new legend instance, and add it to the index."""

    LEGENDS[kwargs["name"]] = Legend(**kwargs)


add_legend(
    name="accretion",
    colors="seq5",
    labels=["0 to < 2", "2 to < 4", "4 to < 6", "6 to < 8", "8 to 10"],
    breaks=[2, 4, 6, 8],
)
add_legend(
    name="default",
    colors="spec7",
    labels=[
        "0 to 49",
        "50 to 99",
        "100 to 249",
        "250 to 499",
        "500 to 999",
        "1,000 to 4,999",
        "5,000 or more",
    ],
    breaks=[50, 100, 250, 500, 1000, 5000],
)
add_legend(
    name="eadd",
    colors=[
        "#DDC5E0",
        "#CCAAD1",
        "#B282BA",
        "#A064A9",
        "#5D4795",
        "#423575",
        "#221F53",
    ],
    labels=[
        "$0 to < $1M",
        "$1M to < $10M",
        "$10M to < $50M",
        "$50M to < $100M",
        "$100M to < $500M",
        "$500M to < $1B",
        "$1B or more",
    ],
    breaks=[1000000, 10000000, 50000000, 100000000, 500000000, 1000000000],
)
add_legend(
    name="easd",
    colors=[
        "#FEDB9E",
        "#FBBC76",
        "#F7903B",
        "#F36F41",
        "#DC4840",
        "#BA233E",
        "#701231",
    ],
    labels=[
        "0 to 49",
        "50 to 99",
        "100 to 249",
        "250 to 499",
        "500 to 999",
        "1,000 to 4,999",
        "5,000 or more",
    ],
    breaks=[50, 100, 250, 500, 1000, 5000],
)
add_legend(
    name="elevation",
    colors="spec8",
    labels=[
        "Less than -2.0",
        "-2.0  to < -1.0",
        "-1.0 to < -0.5",
        "-0.5 to < 0.0",
        "0.0 to < 0.5",
        "0.5 to < 1.0",
        "1.0 to < 2.0",
        "2.0 or greater",
    ],
    breaks=[-2, -1, -0.5, 0, 0.5, 1, 2],
)
add_legend(
    name="exposure_asset_count",
    colors="spec8",
    labels=[
        "0 to 9",
        "10 to 49",
        "50 to 99",
        "100 to 249",
        "250 to 499",
        "500 to 999",
        "1,000 to 9,999",
        "10,000 or more",
    ],
    breaks=[10, 50, 100, 250, 500, 1000, 10000],
)
add_legend(
    name="exposure_exposed_structures",
    colors=["#00000000"] + COLORS["spec6"],
    labels=[
        "No structures",
        "0 structures exposed",
        "1 structure exposed",
        "2 structures exposed",
        "3 to 5 structures exposed",
        "6 to 9 structures exposed",
        "10 or more structures exposed",
    ],
    breaks=[None, 1, 2, 3, 6, 10],
)
add_legend(
    name="ffibs",
    colors=["#5E823A", "#DAE86D", "#92CBE8", "#735753", "#0679D2"],
    labels=[
        "Fresh Forest",
        "Fresh Marsh",
        "Intermediate Marsh",
        "Brackish Marsh",
        "Saline Marsh",
    ],
)
add_legend(
    name="ffibs_score",
    colors=["#00000000", "#5E823A", "#DAE86D", "#92CBE8", "#735753", "#0679D2"],
    labels=[
        "No Data",
        "Fresh Forest",
        "Fresh Marsh",
        "Intermediate Marsh",
        "Brackish Marsh",
        "Saline Marsh",
    ],
    breaks=[0, 0.2, 2, 5, 17],
)
add_legend(
    name="flood_depth",
    colors=[
        "#728D97",
        "#73B7EA",
        "#99D9A7",
        "#F8D88B",
        "#E9B023",
        "#E56524",
        "#B73865",
    ],
    labels=[
        "1 to < 4 feet",
        "4 to < 7 feet",
        "7 to < 10 feet",
        "10 to < 13 feet",
        "13 to < 16 feet",
        "16 to < 21 feet",
        "21+ feet",
    ],
)
add_legend(
    name="habitat_suitability_unit",
    colors="spec10",
    labels=[
        "0",
        "1 to 49",
        "50 to 100",
        "100 to 499",
        "500 to 999",
        "1,000 to 1,499",
        "1,500 to 1,999",
        "2,000 to 2,499",
        "2,500 to 2,999",
        "3,000 or more",
    ],
    breaks=[1, 50, 100, 500, 1000, 1500, 2000, 2500, 3000],
)
add_legend(
    name="inundation_depth",
    colors="spec8",
    labels=[
        "-2.00 to < -1.00",
        "-1.00 to < -0.50",
        "-0.50 to < -0.25",
        "-0.25 to < 0.00",
        "0.00 to < 0.25",
        "0.25 to < 0.50",
        "0.50 to < 1.00",
        "1.00 or more",
    ],
    breaks=[-1, -0.5, -0.25, 0, 0.25, 0.5, 1],
)
add_legend(
    name="land_change",
    colors=["#4591ac", "#00f0f0", "#99c104", "#265118"],
    labels=["Land Loss", "Induced Loss", "Land Maintained", "Land Gain"],
)
add_legend(
    name="land_type",
    colors=["#00000000", "#008000", "#6495ed", "#8b4513", "#999999", "#afeeee"],
    labels=[
        "No Data",
        "Vegetated Wetland",
        "Open Water",
        "Bare/Unvegetated Wetland",
        "Upland/Developed Areas",
        "Flotant Marsh",
    ],
    breaks=[1, 2, 3, 4, 5],
)
add_legend(
    name="mineral_deposition",
    colors="seq4",
    labels=["0 to < 1", "1 to < 10", "10 to < 100", "100 to 1,000"],
    breaks=[1, 10, 100],
)
add_legend(
    name="organic_accumulation",
    colors="seq5",
    labels=[
        "0.05 to < 0.07",
        "0.07 to < 0.09",
        "0.09 to < 0.11",
        "0.11 to < 0.13",
        "0.13 to 0.15",
    ],
    breaks=[0.07, 0.09, 0.11, 0.13],
)
add_legend(
    name="population",
    colors="spec11",
    labels=[
        "0 to 9,999",
        "10,000 to 19,999",
        "20,000 to 29,999",
        "30,000 to 39,999",
        "40,000 to 49,999",
        "50,000 to 74,999",
        "75,000 to 99,999",
        "100,000 to 199,999",
        "200,000 to 499,999",
        "500,000 to 999,999",
        "1,000,000 or more",
    ],
    breaks=[10000, 20000, 30000, 40000, 50000, 75000, 100000, 200000, 500000, 1000000],
)
add_legend(
    name="population_change",
    colors="spec9",
    labels=[
        "0 to 9",
        "10 to 29",
        "30 to 49",
        "50 to 99",
        "100 to 249",
        "250 to 499",
        "500 to 999",
        "1,000 to 4,999",
        "5,000 or more",
    ],
    breaks=[10, 30, 50, 100, 250, 500, 1000, 5000],
)
add_legend(
    name="salinity",
    colors="spec8",
    labels=[
        "0 to < 0.1",
        "0.1 to < 0.5",
        "0.5 to < 1.0",
        "1.0 to < 2.0",
        "2.0 to < 5.0",
        "5.0 to < 10.0",
        "10.0 to < 20.0",
        "20.0 to 35.0",
    ],
    breaks=[0.1, 0.5, 1, 2, 5, 10, 20],
)
add_legend(
    name="subsidence_deep",
    colors=["#00000000"] + COLORS["seq5"],
    labels=["No Data", "0 to < 4", "4 to < 8", "8 to < 12", "12 to < 16", "16 to 20"],
    breaks=[0, 4, 8, 12, 16],
)
add_legend(
    name="subsidence_shallow",
    colors=["#00000000"] + COLORS["seq4"],
    labels=["No Data", "0 to < 2", "2 to < 4", "4 to < 6", "6 to < 8"],
    breaks=[0, 2, 4, 6],
)
add_legend(
    name="surge_water_surface_elevations",
    colors="spec9",
    labels=[
        "0.09 or less",
        "0.10 to 0.24",
        "0.25 to 0.49",
        "0.50 to 0.99",
        "1.00 to 1.49",
        "1.50 to 1.99",
        "2.00 to 2.99",
        "3.00 to 4.99",
        "5.00 or more",
    ],
    breaks=[0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5],
)
add_legend(
    name="temperature",
    colors="spec9",
    labels=[
        "Less than 19",
        "19 to < 20",
        "20 to < 21",
        "21 to < 22",
        "22 to < 23",
        "23 to < 24",
        "24 to < 25",
        "25 to < 26",
        "27 or more",
    ],
    breaks=[19, 20, 21, 22, 23, 24, 25, 26],
)
add_legend(
    name="total_suspended_solids",
    colors="spec8",
    labels=[
        "0 to < 10",
        "10 to < 50",
        "50 to < 100",
        "100 to < 200",
        "200 to < 300",
        "300 to < 400",
        "400 to < 500",
        "500 or more",
    ],
    breaks=[10, 50, 100, 200, 300, 400, 500],
)
add_legend(
    name="vct",
    colors=[
        "#a6cecf",
        "#1f78b4",
        "#b2df8a",
        "#5ba02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#6a3d9a",
        "#ffff99",
        "#005a00",
        "#ff80ff",
        "#7f5f1e",
        "#8cf0f0",
    ],
    labels=[
        "Maidencane",
        "Three-Square",
        "Rosseau Cane",
        "Paspalum",
        "Wiregrass",
        "Bulltongue",
        "Needlerush",
        "Bulrush",
        "Brackish Mix",
        "Oystergrass",
        "Saltgrass",
        "Bottomland Hardwood Forest + Swamp Forest",
        "Barrier Island Vegetation",
        "Unvegetated",
        "Flotant",
    ],
)
add_legend(
    name="vegetation_coverage_area",
    colors="spec11",
    labels=[
        "0",
        "1 to 2,499,999",
        "2,500,000 to 4,999,999",
        "5,000,000 to 9,999,999",
        "10,000,000 to 19,999,999",
        "20,000,000 to 49,999,999",
        "50,000,000 to 99,9999,999",
        "100,000,000 to 199,999,999",
        "200,000,000 to 399,999,999",
        "300,000,000 to 499,999,999",
        "500,000,000 or more",
    ],
    breaks=[
        1,
        2500000,
        5000000,
        10000000,
        20000000,
        50000000,
        100000000,
        200000000,
        300000000,
        500000000,
    ],
)
add_legend(
    name="water_level",
    colors="spec10",
    labels=[
        "0 to < 0.2",
        "0.2 to < 0.4",
        "0.4 to < 0.6",
        "0.6 to < 0.8",
        "0.8 to < 1.0",
        "1.0 to < 1.5",
        "1.5 to < 2.0",
        "2.0 to < 3.0",
        "3.0 to < 5.0",
        "5.0 or more",
    ],
    breaks=[0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 5],
)
add_legend(
    name="water_level_variability",
    colors="spec6",
    labels=[
        "0.09 or less",
        "0.10 to 0.19",
        "0.20 to 0.29",
        "0.30 to 0.39",
        "0.40 to 0.49",
        "0.50  or more",
    ],
    breaks=[0.1, 0.2, 0.3, 0.4, 0.5],
)
add_legend(
    name="wave_heights",
    colors="spec5",
    labels=[
        "0.99 or less",
        "1.00 to 1.99",
        "2.00 to 4.99",
        "5.00 to 9.99",
        "10.0 or more",
    ],
    breaks=[1, 2, 5, 10],
)
