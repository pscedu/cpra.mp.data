import polars as pl

from cpra.mp.data.exceptions import SchemaValidationError
from cpra.mp.data.variables import VARIABLES


class SchemaLevel:
    def __init__(
        self,
        column,
        name,
        dtype,
        partition=(pl.col("time_unit") == "daily").any(),
        required=True,
        values=None,
        labels=None,
        min_value=None,
        max_value=None,
        null=False,
        example_value=None,
    ):
        self.column = column
        self.name = name
        self.dtype = dtype
        self.partition = partition
        self.values = values
        self.labels = labels
        self.min_value = min_value
        self.max_value = max_value
        self.null = null
        self.required = required
        self.example_value = example_value

    def validate_series(self, series):
        """Validate that a Polars series conforms to the specification for this level."""

        if series.dtype != self.dtype:
            raise SchemaValidationError(
                f"Series {series.name} has dtype {series.dtype!s}, expected {self.dtype!s}"
            )

        if (not self.null) and series.is_null().any():
            raise SchemaValidationError(f"Series {series.name} contains null values")

        # Allowed values have to be made into a series so they can be cast to the correct
        # dtype for comparison.
        if (
            self.values
            and not series.is_in(pl.Series(self.values).cast(self.dtype)).all()
        ):
            raise SchemaValidationError(f"Series {series.name} has invalid values")

        if self.min_value is not None:
            min_value = series.min()
            if min_value < self.min_value:
                raise SchemaValidationError(
                    f"Series {series.name} has an invalid minimum value: {min_value}"
                )

        if self.max_value is not None:
            max_value = series.max()
            if max_value > self.max_value:
                raise SchemaValidationError(
                    f"Series {series.name} has an invalid maximum value: {max_value}"
                )


SCHEMA = [
    SchemaLevel(
        column="variable",
        name="Variable",
        dtype=pl.String,
        partition=True,
        values=list(VARIABLES.keys()),
        labels=[variable.label for variable in VARIABLES.values()],
        example_value="salinity_mean",
    ),
    SchemaLevel(
        column="geographic_unit",
        name="Geography",
        dtype=pl.String,
        partition=True,
        values=[
            "block_group",
            "clara",
            "ecoregion",
            "extraction_point",
            "hydro_compartment",
            "icm_raster",
            "parish",
        ],
        labels=[
            "Census Block Group",
            "Clara Community",
            "Ecoregion",
            "Extraction Point",
            "Hydro Compartment",
            "ICM Raster",
            "Parish",
        ],
        example_value="hydro_compartment",
    ),
    SchemaLevel(
        column="time_unit",
        name="Time Unit",
        dtype=pl.String,
        partition=True,
        values=["annual", "daily", "decadal"],
        labels=["Annual", "Daily", "Decadal"],
        example_value="daily",
    ),
    SchemaLevel(
        column="plan_version",
        name="Plan Version",
        dtype=pl.String,
        partition=True,
        example_value="2023.0.0",
    ),
    SchemaLevel(
        column="model_group",
        name="Plan Implementation Scenario",
        dtype=pl.Int32,
        min_value=100,
        max_value=999,
        example_value=500,
    ),
    SchemaLevel(
        column="scenario",
        name="Environmental Scenario",
        dtype=pl.Int32,
        min_value=1,
        max_value=100,
        example_value=7,
    ),
    SchemaLevel(
        column="aep",
        name="Annual Exceedance Probability",
        dtype=pl.Float32,
        required=(
            pl.col("variable")
            .is_in(
                [
                    "flood_depth",
                    "annual_exceedance_damages_dollars",
                    "surge_water_surface_elevations_by_exceedance",
                    "wave_heights_by_exceedance",
                    "exposure_asset_count",
                    "exposure_critical_infrastructure_count",
                    "exposure_historic_properties_count",
                ]
            )
            .any()
        ),
        values=[20, 10, 2, 1, 0.2, 0.1],
        labels=["20.0", "10.0", "2.0", "1.0", "0.2", "0.1"],
        example_value=0.2,
    ),
    SchemaLevel(
        column="storm_id",
        name="Synthetic Storm ID",
        dtype=pl.Int32,
        required=(
            pl.col("variable")
            .is_in(["surge_water_surface_elevations_by_storm", "wave_heights_by_storm"])
            .any()
        ),
        min_value=1,
        max_value=1000,
        example_value=182,
    ),
    SchemaLevel(
        column="pumping_id",
        name="Pumping Scenario",
        dtype=pl.Float32,
        required=(
            pl.col("variable")
            .is_in(
                [
                    "annual_exceedance_damages_dollars",
                    "expected_annual_damages_dollars",
                    "expected_annual_structural_damage",
                    "exposure_asset_count",
                    "exposure_critical_infrastructure_count",
                    "exposure_historic_properties_count",
                    "nonstructural_expected_annual_damage_dollars",
                    "nonstructural_expected_annual_structural_damage",
                ]
            )
            .any()
        ),
        values=[0, 0.5, 1],
        labels=["No Pumping", "50% Pumping Capacity", "Full Pumping Capacity"],
        example_value=0.5,
    ),
    SchemaLevel(
        column="fragility_scenario",
        name="Fragility Scenario",
        dtype=pl.Int8,
        required=(
            pl.col("variable")
            .is_in(
                [
                    "annual_exceedance_damages_dollars",
                    "expected_annual_damages_dollars",
                    "expected_annual_structural_damage",
                    "exposure_asset_count",
                    "exposure_critical_infrastructure_count",
                    "exposure_historic_properties_count",
                    "nonstructural_expected_annual_damage_dollars",
                    "nonstructural_expected_annual_structural_damage",
                ]
            )
            .any()
        ),
        values=[0, 1],
        labels=["No Fragility", "IPET Fragility"],
        example_value=0,
    ),
    SchemaLevel(
        column="asset_type",
        name="Asset Type",
        dtype=pl.Int8,
        required=(
            pl.col("variable")
            .is_in(
                [
                    "annual_exceedance_damages_dollars",
                    "expected_annual_damages_dollars",
                    "expected_annual_structural_damage",
                    "exposure_asset_count",
                    "nonstructural_expected_annual_damage_dollars",
                    "nonstructural_expected_annual_structural_damage",
                ]
            )
            .any()
        ),
        values=[6, 1, 2, 3, 4, 5],
        labels=[
            "All Asset Types",
            "Small Residential (single-family, manufactured homes, or duplex)",
            "Other Multi-family Residential",
            "Commercial, Industrial, or Agricultural",
            "Other Structural (public, education, or religion)",
            "Non-structural (crops, vehicles, or roads)",
        ],
        example_value=3,
    ),
    SchemaLevel(
        column="category",
        name="Category",
        dtype=pl.String,
        required=(pl.col("variable") == "exposure_critical_infrastructure_count").any(),
        values=[
            "daycares",
            "drink_water_sources",
            "drinking_water_treatment_plants",
            "electric_substations",
            "emergency_medical_service_stations",
            "fire_stations",
            "gas_stations",
            "hospitals",
            "local_emergency_operations_centers",
            "local_law_enforcement",
            "nursing_homes",
            "nursing_residential_care_facilities",
            "power_plants",
            "public_schools",
            "veterans_health_administration_medical_facilities",
        ],
        labels=[
            "Daycares",
            "Drink Water Sources",
            "Drinking Water Treatment Plants",
            "Electric Substations",
            "Emergency Medical Service Stations",
            "Fire Stations",
            "Gas Stations",
            "Hospitals",
            "Local Emergency Operations Centers",
            "Local Law Enforcement",
            "Nursing Homes",
            "Nursing Residential Care Facilities",
            "Power Plants",
            "Public Schools",
            "Veterans Health Administration Medical Facilities",
        ],
        example_value="public_schools",
    ),
    SchemaLevel(
        column="exposure_type",
        name="Exposure Type",
        dtype=pl.String,
        required=(
            pl.col("variable")
            .is_in(
                [
                    "exposure_asset_count",
                    "exposure_critical_infrastructure_count",
                    "exposure_historic_properties_count",
                ]
            )
            .any()
        ),
        values=["exposure", "moderate_exposure", "severe_exposure"],
        labels=["Exposure", "Moderate Exposure", "Severe Exposure"],
        example_value="moderate_exposure",
    ),
    SchemaLevel(
        column="ns_project_id",
        name="Nonstructural Project",
        dtype=pl.Int8,
        required=(
            pl.col("variable")
            .is_in(
                [
                    "nonstructural_expected_annual_damage_dollars",
                    "nonstructural_expected_annual_structural_damage",
                ]
            )
            .any()
        ),
        values=[1, 3],
        example_value=3,
    ),
    SchemaLevel(
        column="participation_rate",
        name="Participation Rate",
        dtype=pl.Float32,
        required=(
            pl.col("variable")
            .is_in(
                [
                    "nonstructural_expected_annual_damage_dollars",
                    "nonstructural_expected_annual_structural_damage",
                ]
            )
            .any()
        ),
        values=[0.5, 0.75, 1],
        labels=["50% Participation", "75% Participation", "Full Participation"],
        example_value=0.75,
    ),
    SchemaLevel(
        column="migration",
        name="Migration",
        dtype=pl.String,
        required=(
            pl.col("variable")
            .is_in(
                [
                    "population",
                    "population_change",
                ]
            )
            .any()
        ),
        values=["base", "mig"],
        labels=["BASE", "MIG"],
        example_value="base",
    ),
    SchemaLevel(
        column="ssp",
        name="Shared Socioeconomic Pathway",
        dtype=pl.Int8,
        required=(
            pl.col("variable")
            .is_in(
                [
                    "population",
                    "population_change",
                ]
            )
            .any()
        ),
        values=[1, 2, 3, 4, 5],
        example_value=4,
    ),
    SchemaLevel(
        column="vegetation_code",
        name="Vegetation Type",
        dtype=pl.String,
        required=(pl.col("variable") == "vegetation_coverage_area").any(),
        values=["brm", "for", "frm", "inm", "sam", "upl", "wat"],
        labels=[
            "Brackish Marsh",
            "Fresh Forest",
            "Fresh Marsh",
            "Intermediate Marsh",
            "Saline Marsh",
            "Upland",
            "Water",
        ],
        example_value="brm",
    ),
    SchemaLevel(
        column="percentile",
        name="Percentile",
        dtype=pl.Int8,
        partition=False,
        required=(
            pl.col("variable")
            .is_in(
                [
                    "expected_annual_damages_dollars",
                    "expected_annual_structural_damage",
                    "annual_exceedance_damages_dollars",
                    "nonstructural_expected_annual_damage_dollars",
                    "nonstructural_expected_annual_structural_damage",
                    "surge_water_surface_elevations_by_exceedance",
                    "wave_heights_by_exceedance",
                    "population_change",
                ]
            )
            .any()
        ),
        min_value=0,
        max_value=100,
        example_value=50,
    ),
    SchemaLevel(
        column="calendar_year",
        name="Year",
        dtype=pl.Int32,
        partition=False,
        required=(
            pl.col("time_unit")
            .is_in(
                [
                    "annual",
                    "decadal",
                ]
            )
            .any()
        ),
        min_value=2000,
        max_value=2100,
        example_value=2030,
    ),
    SchemaLevel(
        column="calendar_date",
        name="Date",
        dtype=pl.Date,
        partition=False,
        required=(pl.col("time_unit") == "daily").any(),
        example_value="2030-01-01",
    ),
]
