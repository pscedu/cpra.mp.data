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

## Variables

<!--BEGIN TABLE variables-->
| Variable | Description | Data Type | Unit | Legend |
| --- | --- | --- | --- | --- |
| `annual_exceedance_damages_dollars` | Annual Exceedance Damages, Dollars | `Int32` | $ | `eadd` |
| `deep_subsidence` | Deep Subsidence | `Int32` | mm/yr | `subsidence_deep` |
| `elevation` | Elevation | `Int32` | NAVD88 m | `elevation` |
| `expected_annual_damages_dollars` | Expected Annual Damages, Dollars | `Int32` | $ | `eadd` |
| `expected_annual_structural_damage` | Expected Annual Structural Damage | `Int32` | structures | `eadd` |
| `exposure_asset_count` | Exposure: Asset Count | `Int32` | structures | `exposure_asset_count` |
| `exposure_critical_infrastructure_count` | Exposure: Critical Infrastructure Count | `Int32` | structures | `exposure_exposed_structures` |
| `exposure_historic_properties_count` | Exposure: Historic Properties Count | `Int32` | structures | `exposure_exposed_structures` |
| `flood_depth` | Flood Depth | `Int32` | ft | `flood_depth` |
| `inundation_depth` | Inundation Depth | `Int32` | m | `inundation_depth` |
| `land_change` | Land Change | `Int32` |  | `land_change` |
| `land_type` | Land Type | `Int32` |  | `land_type` |
| `mineral_accretion` | Mineral Accretion | `Int32` | cm | `accretion` |
| `mineral_deposition` | Mineral Deposition | `Int32` | g/cm^2-yr | `mineral_deposition` |
| `nonstructural_expected_annual_damage_dollars` | Nonstructural Expected Annual Damage, Dollars | `Int32` | $ | `eadd` |
| `nonstructural_expected_annual_structural_damage` | Nonstructural Expected Annual Structural Damage | `Int32` | structure equivalents | `easd` |
| `organic_accretion` | Organic Accretion | `Int32` | cm | `accretion` |
| `organic_accumulation` | Organic Accumulation | `Int32` | g/cm^2-yr | `organic_accumulation` |
| `population` | Population | `Int32` | persons | `population` |
| `population_change` | Population Change | `Int32` | persons | `population_change` |
| `salinity` | Salinity | `Int32` | ppt | `salinity` |
| `shallow_subsidence` | Shallow Subsidence | `Int32` | mm/yr | `subsidence_shallow` |
| `surge_water_surface_elevations_by_exceedance` | Surge Water Surface Elevations by Exceedance | `Int32` | NAVD88 ft | `surge_water_surface_elevations` |
| `surge_water_surface_elevations_by_storm` | Surge Water Surface Elevations by Storm | `Int32` | NAVD88 ft | `surge_water_surface_elevations` |
| `temperature` | Temperature | `Int32` | °C | `temperature` |
| `total_suspended_solids` | Total Suspended Solids | `Int32` | mg/L | `total_suspended_solids` |
| `vegetation_coverage_area` | Vegetation Coverage Area | `Int32` | square m | `vegetation_coverage_area` |
| `vegetation_type_ffibs` | Vegetation Type: FFIBS | `Int32` |  | `ffibs` |
| `vegetation_type_ffibs_extraction_point` | Vegetation Type: FFIBS (Extraction Point) | `Int32` |  | `ffibs_score` |
| `vegetation_type_vct` | Vegetation Type: VCT | `Int32` |  | `vct` |
| `water_level` | Water Level | `Int32` | NAVD88 m | `water_level` |
| `water_level_variability` | Water Level Variability | `Int32` | m | `water_level_variability` |
| `wave_heights_by_exceedance` | Wave Heights by Exceedance | `Int32` | ft | `wave_heights` |
| `wave_heights_by_storm` | Wave Heights by Storm | `Int32` | ft | `wave_heights` |
<!--END TABLE-->
