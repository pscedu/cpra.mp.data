import polars as pl
from cpra.mp.data.legends import Legend, LEGENDS


class Variable:
    def __init__(
        self,
        name,
        label,
        dtype,
        legend,
        unit=None,
    ):
        self.name = name
        self.label = label
        self.dtype = dtype
        self.legend = legend if isinstance(legend, Legend) else LEGENDS[legend]
        self.unit = unit


VARIABLES = {}


def add_variable(**kwargs):
    """Create a new variable instance, and add it to the index."""

    VARIABLES[kwargs["name"]] = Variable(**kwargs)


add_variable(
    name="annual_exceedance_damages_dollars",
    label="Annual Exceedance Damages, Dollars",
    dtype=pl.Float64,
    legend="eadd",
    unit="$",
)
add_variable(
    name="deep_subsidence",
    label="Deep Subsidence",
    dtype=pl.Float32,
    legend="subsidence_deep",
    unit="mm/yr",
)
add_variable(
    name="elevation",
    label="Elevation",
    dtype=pl.Float32,
    legend="elevation",
    unit="NAVD88 m",
)
add_variable(
    name="expected_annual_damages_dollars",
    label="Expected Annual Damages, Dollars",
    dtype=pl.Float64,
    legend="eadd",
    unit="$",
)
add_variable(
    name="expected_annual_structural_damage",
    label="Expected Annual Structural Damage",
    dtype=pl.Float64,
    legend="eadd",
    unit="structures",
)
add_variable(
    name="exposure_asset_count",
    label="Exposure: Asset Count",
    dtype=pl.Int32,
    legend="exposure_asset_count",
    unit="structures",
)
add_variable(
    name="exposure_critical_infrastructure_count",
    label="Exposure: Critical Infrastructure Count",
    dtype=pl.Int32,
    legend="exposure_exposed_structures",
    unit="structures",
)
add_variable(
    name="exposure_historic_properties_count",
    label="Exposure: Historic Properties Count",
    dtype=pl.Int32,
    legend="exposure_exposed_structures",
    unit="structures",
)
add_variable(
    name="flood_depth",
    label="Flood Depth",
    dtype=pl.Float32,
    legend="flood_depth",
    unit="ft",
)
add_variable(
    name="inundation_depth",
    label="Inundation Depth",
    dtype=pl.Float64,
    legend="inundation_depth",
    unit="m",
)
add_variable(
    name="land_change", label="Land Change", dtype=pl.Int32, legend="land_change"
)
add_variable(name="land_type", label="Land Type", dtype=pl.Int32, legend="land_type")
add_variable(
    name="mineral_accretion",
    label="Mineral Accretion",
    dtype=pl.Float64,
    legend="accretion",
    unit="cm",
)
add_variable(
    name="mineral_deposition",
    label="Mineral Deposition",
    dtype=pl.Float64,
    legend="mineral_deposition",
    unit="g/cm^2-yr",
)
add_variable(
    name="nonstructural_expected_annual_damage_dollars",
    label="Nonstructural Expected Annual Damage, Dollars",
    dtype=pl.Float64,
    legend="eadd",
    unit="$",
)
add_variable(
    name="nonstructural_expected_annual_structural_damage",
    label="Nonstructural Expected Annual Structural Damage",
    dtype=pl.Float64,
    legend="easd",
    unit="structure equivalents",
)
add_variable(
    name="organic_accretion",
    label="Organic Accretion",
    dtype=pl.Float64,
    legend="accretion",
    unit="cm",
)
add_variable(
    name="organic_accumulation",
    label="Organic Accumulation",
    dtype=pl.Float64,
    legend="organic_accumulation",
    unit="g/cm^2-yr",
)
add_variable(
    name="population",
    label="Population",
    dtype=pl.Float64,
    legend="population",
    unit="persons",
)
add_variable(
    name="population_change",
    label="Population Change",
    dtype=pl.Float64,
    legend="population_change",
    unit="persons",
)
add_variable(
    name="salinity", label="Salinity", dtype=pl.Float32, legend="salinity", unit="ppt"
)
add_variable(
    name="shallow_subsidence",
    label="Shallow Subsidence",
    dtype=pl.Float32,
    legend="subsidence_shallow",
    unit="mm/yr",
)
add_variable(
    name="surge_water_surface_elevations_by_exceedance",
    label="Surge Water Surface Elevations by Exceedance",
    dtype=pl.Float32,
    legend="surge_water_surface_elevations",
    unit="NAVD88 ft",
)
add_variable(
    name="surge_water_surface_elevations_by_storm",
    label="Surge Water Surface Elevations by Storm",
    dtype=pl.Float32,
    legend="surge_water_surface_elevations",
    unit="NAVD88 ft",
)
add_variable(
    name="temperature",
    label="Temperature",
    dtype=pl.Float64,
    legend="temperature",
    unit="°C",
)
add_variable(
    name="total_suspended_solids",
    label="Total Suspended Solids",
    dtype=pl.Float64,
    legend="total_suspended_solids",
    unit="mg/L",
)
add_variable(
    name="vegetation_coverage_area",
    label="Vegetation Coverage Area",
    dtype=pl.Int64,
    legend="vegetation_coverage_area",
    unit="square m",
)
add_variable(
    name="vegetation_type_ffibs",
    label="Vegetation Type: FFIBS",
    dtype=pl.Int32,
    legend="ffibs",
)
add_variable(
    name="vegetation_type_ffibs_extraction_point",
    label="Vegetation Type: FFIBS (Extraction Point)",
    dtype=pl.Int32,
    legend="ffibs_score",
)
add_variable(
    name="vegetation_type_vct",
    label="Vegetation Type: VCT",
    dtype=pl.Int32,
    legend="vct",
)
add_variable(
    name="water_level",
    label="Water Level",
    dtype=pl.Float32,
    legend="water_level",
    unit="NAVD88 m",
)
add_variable(
    name="water_level_variability",
    label="Water Level Variability",
    dtype=pl.Float32,
    legend="water_level_variability",
    unit="m",
)
add_variable(
    name="wave_heights_by_exceedance",
    label="Wave Heights by Exceedance",
    dtype=pl.Float32,
    legend="wave_heights",
    unit="ft",
)
add_variable(
    name="wave_heights_by_storm",
    label="Wave Heights by Storm",
    dtype=pl.Float64,
    legend="wave_heights",
    unit="ft",
)
