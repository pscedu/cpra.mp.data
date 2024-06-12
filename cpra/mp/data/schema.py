import polars as pl

from cpra.mp.data.exceptions import SchemaValidationError


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
        values=[
            "annual_exceedance_damages_dollars",
            "expected_annual_damages_dollars",
            "expected_annual_structural_damage",
            "exposure_asset_count",
            "exposure_critical_infrastructure_count",
            "exposure_historic_properties_count",
            "flood_depth",
            "land_change",
            "nonstructural_expected_annual_damage_dollars",
            "nonstructural_expected_annual_structural_damage",
            "population_change",
            "population",
            "salinity",
            "surge_water_surface_elevations_by_exceedance",
            "surge_water_surface_elevations_by_storm",
            "temperature",
            "total_suspended_solids",
            "vegetation_coverage_area",
            "vegetation_type_ffibs",
            "vegetation_type_vct",
            "water_level_variability",
            "water_level",
            "wave_heights_by_exceedance",
            "wave_heights_by_storm",
        ],
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
            "parish",
        ],
        labels=[
            "Census Block Group",
            "Clara Community",
            "Ecoregion",
            "Extraction Point",
            "Hydro Compartment",
            "Parish",
        ],
    ),
    SchemaLevel(
        column="time_unit",
        name="Time Unit",
        dtype=pl.String,
        partition=True,
        values=["annual", "daily", "decadal"],
        labels=["Annual", "Daily", "Decadal"],
    ),
    SchemaLevel(
        column="plan_version",
        name="Plan Version",
        dtype=pl.String,
        partition=True,
    ),
    SchemaLevel(
        column="model_group",
        name="Plan Implementation Scenario",
        dtype=pl.String,
        min_value=100,
        max_value=999
    ),
    SchemaLevel(
        column="scenario",
        name="Environmental Scenario",
        dtype=pl.String,
        min_value=1,
        max_value=100
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
    ),
    SchemaLevel(
        column="category",
        name="Category",
        dtype=pl.String,
        required=(pl.col("variable") == "exposure_critical_infrastructure_count").any(),
        values=[
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
        values=["Exposure", "Moderate Exposure", "Severe Exposure"],
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
    ),
    SchemaLevel(
        column="percentile",
        name="Percentile",
        dtype=pl.Int8,
        partition=False,
        required=True,
    ),
    SchemaLevel(
        column="calendar_year",
        name="Year",
        dtype=pl.Date,
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
    ),
    SchemaLevel(
        column="calendar_date",
        name="Date",
        dtype=pl.Date,
        partition=False,
        required=(pl.col("time_unit") == "daily").any(),
    ),
]
