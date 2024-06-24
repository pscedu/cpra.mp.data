# Data Structure

## Dimensions

A hierarchy of dimensions is used to store and catalog Parquet and COG files. The tables below summarize the columns that define these dimensions, as well as the data types and conditions under which the dimensions are required.

### Dataset Dimensions

Together, the dataset dimensions define a single Parquet dataset. They are also used to create the directory structure for COGs (see File Paths below).

<!--BEGIN TABLE dataset_dimensions-->
| Column | Description | Data Type | Required Condition | Example Value |
| --- | --- | --- | --- | --- |
| `variable` | Variable | `String` | `True` | salinity_mean |
| `geographic_unit` | Geography | `String` | `True` | hydro_compartment |
| `time_unit` | Time Unit | `String` | `True` | daily |
| `plan_version` | Plan Version | `String` | `True` | 2023.0.0 |
<!--END TABLE-->

### Partition Dimensions

Partition dimensions are used to partition daily data stored in Parquet files, as well as all COGs.

<!--BEGIN TABLE partition_dimensions-->
| Column | Description | Data Type | Required Condition | Example Value |
| --- | --- | --- | --- | --- |
| `model_group` | Plan Implementation Scenario | `Int32` | `True` | 500 |
| `scenario` | Environmental Scenario | `Int32` | `True` | 7 |
| `aep` | Annual Exceedance Probability | `Float32` | `variable` in ("annual_exceedance_damages_dollars", "exposure_asset_count", "exposure_critical_infrastructure_count", "exposure_historic_properties_count", "flood_depth", "surge_water_surface_elevations_by_exceedance", "wave_heights_by_exceedance") | 0.2 |
| `storm_id` | Synthetic Storm ID | `Int32` | `variable` in ("surge_water_surface_elevations_by_storm", "wave_heights_by_storm") | 182 |
| `pumping_id` | Pumping Scenario | `Float32` | `variable` in ("annual_exceedance_damages_dollars", "expected_annual_damages_dollars", "expected_annual_structural_damage", "exposure_asset_count", "exposure_critical_infrastructure_count", "exposure_historic_properties_count", "nonstructural_expected_annual_damage_dollars", "nonstructural_expected_annual_structural_damage") | 0.5 |
| `fragility_scenario` | Fragility Scenario | `Int8` | `variable` in ("annual_exceedance_damages_dollars", "expected_annual_damages_dollars", "expected_annual_structural_damage", "exposure_asset_count", "exposure_critical_infrastructure_count", "exposure_historic_properties_count", "nonstructural_expected_annual_damage_dollars", "nonstructural_expected_annual_structural_damage") |  |
| `asset_type` | Asset Type | `Int8` | `variable` in ("annual_exceedance_damages_dollars", "expected_annual_damages_dollars", "expected_annual_structural_damage", "exposure_asset_count", "nonstructural_expected_annual_damage_dollars", "nonstructural_expected_annual_structural_damage") | 3 |
| `category` | Category | `String` | `variable` = "exposure_critical_infrastructure_count" | public_schools |
| `exposure_type` | Exposure Type | `String` | `variable` in ("exposure_asset_count", "exposure_critical_infrastructure_count", "exposure_historic_properties_count") | moderate_exposure |
| `ns_project_id` | Nonstructural Project | `Int8` | `variable` in ("nonstructural_expected_annual_damage_dollars", "nonstructural_expected_annual_structural_damage") | 3 |
| `participation_rate` | Participation Rate | `Float32` | `variable` in ("nonstructural_expected_annual_damage_dollars", "nonstructural_expected_annual_structural_damage") | 0.75 |
| `migration` | Migration | `String` | `variable` in ("population", "population_change") | base |
| `ssp` | Shared Socioeconomic Pathway | `Int8` | `variable` in ("population", "population_change") | 4 |
| `vegetation_code` | Vegetation Type | `String` | `variable` = "vegetation_coverage_area" | brm |
<!--END TABLE-->

### Non-Partition Dimensions

Non-partition dimensions are present in Parquet data files but are not used to create partition directories.

<!--BEGIN TABLE nonpartition_dimensions-->
| Column | Description | Data Type | Required Condition | Example Value |
| --- | --- | --- | --- | --- |
| `percentile` | Percentile | `Int8` | `variable` in ("annual_exceedance_damages_dollars", "expected_annual_damages_dollars", "expected_annual_structural_damage", "nonstructural_expected_annual_damage_dollars", "nonstructural_expected_annual_structural_damage", "population_change", "surge_water_surface_elevations_by_exceedance", "wave_heights_by_exceedance") | 50 |
| `calendar_year` | Year | `Int32` | `time_unit` in ("annual", "decadal") | 2030 |
| `calendar_date` | Date | `Date` | `time_unit` = "daily" | 2030-01-01 |
<!--END TABLE-->

## File Paths

Parquet and COG files are saved in nested directories corresponding to the dataset and partition dimensions:

```
.../master_plan=2029/variable=salinity/geographic_unit=hydro_compartment/aggregate=mean/time_unit=daily/model_group=500/scenario=7/4f3b1a8418a74c6dacb41378fd69ec26-0.parquet
```

The partitioned Parquet dataset lives within the `aggregate=mean` directory. The directories above and including this level are just filesystem directories, while the directories below it (`model_group=500` and `scenario=7`) are partition directories that can be used by the Parquet reader to understand the structure of the full daily mean hydro compartment salinity dataset.

While they do not use partitioning, COGs are saved in the same directory structure:

```
.../master_plan=2029/variable=flood_depth/time_unit=decadal/model_group=500/scenario=7/aep=10.0/calendar_year=2030.tiff
```

The nested directories create the data structure needed for the catalog. By traversing the catalog tree (and in some cases, summarizing specific columns from the Parquet files selected through the traversal), the catalog API can respond to queries about available data and valid choices at each level of the catalog hierarchy.