# Data Structure

## Dimensions

A hierarchy of dimensions is used to store and catalog Parquet and COG files. The tables below summarize the columns that define these dimensions, as well as the data types and conditions under which the dimensions are required.

### Dataset Dimensions

Together, the dataset dimensions define a single Parquet dataset. They are also used to create the directory structure for COGs (see File Paths below).

<!--BEGIN SECTION dataset_dimensions-->
| Column | Description | Data Type | Example Value |
| --- | --- | --- | --- |
| `plan_version` | Plan Version | `String` | 2023.0.0 |
| `variable` | Variable | `String` | salinity_mean |
| `geographic_unit` | Geography | `String` | hydro_compartment |
| `time_unit` | Time Unit | `String` | daily |

#### Plan Version

**Column:** `plan_version`  
**Data Type:** `String`  
**Required Condition:** `True`  

#### Variable

**Column:** `variable`  
**Data Type:** `String`  
**Required Condition:** `True`  
| Value | Label |
| --- | --- |
| annual_exceedance_damages_dollars | Annual Exceedance Damages, Dollars |
| deep_subsidence | Deep Subsidence |
| elevation | Elevation |
| expected_annual_damages_dollars | Expected Annual Damages, Dollars |
| expected_annual_structural_damage | Expected Annual Structural Damage |
| exposure_asset_count | Exposure: Asset Count |
| exposure_critical_infrastructure_count | Exposure: Critical Infrastructure Count |
| exposure_historic_properties_count | Exposure: Historic Properties Count |
| flood_depth | Flood Depth |
| inundation_depth | Inundation Depth |
| land_change | Land Change |
| land_type | Land Type |
| mineral_accretion | Mineral Accretion |
| mineral_deposition | Mineral Deposition |
| nonstructural_expected_annual_damage_dollars | Nonstructural Expected Annual Damage, Dollars |
| nonstructural_expected_annual_structural_damage | Nonstructural Expected Annual Structural Damage |
| organic_accretion | Organic Accretion |
| organic_accumulation | Organic Accumulation |
| population | Population |
| population_change | Population Change |
| salinity | Salinity |
| shallow_subsidence | Shallow Subsidence |
| surge_water_surface_elevations_by_exceedance | Surge Water Surface Elevations by Exceedance |
| surge_water_surface_elevations_by_storm | Surge Water Surface Elevations by Storm |
| temperature | Temperature |
| total_suspended_solids | Total Suspended Solids |
| vegetation_coverage_area | Vegetation Coverage Area |
| vegetation_type_ffibs | Vegetation Type: FFIBS |
| vegetation_type_ffibs_extraction_point | Vegetation Type: FFIBS (Extraction Point) |
| vegetation_type_vct | Vegetation Type: VCT |
| water_level | Water Level |
| water_level_variability | Water Level Variability |
| wave_heights_by_exceedance | Wave Heights by Exceedance |
| wave_heights_by_storm | Wave Heights by Storm |

#### Geography

**Column:** `geographic_unit`  
**Data Type:** `String`  
**Required Condition:** `True`  
| Value | Label |
| --- | --- |
| block_group | Census Block Group |
| clara | Clara Community |
| ecoregion | Ecoregion |
| extraction_point | Extraction Point |
| hydro_compartment | Hydro Compartment |
| icm_raster | ICM Raster |
| parish | Parish |

#### Time Unit

**Column:** `time_unit`  
**Data Type:** `String`  
**Required Condition:** `True`  
| Value | Label |
| --- | --- |
| annual | Annual |
| daily | Daily |
| decadal | Decadal |

<!--END SECTION-->

### Partition Dimensions

Partition dimensions are used to partition daily data stored in Parquet files, as well as all COGs.

<!--BEGIN SECTION partition_dimensions-->
| Column | Description | Data Type | Example Value |
| --- | --- | --- | --- |
| `model_group_id` | Plan Implementation Scenario | `Int32` | 500 |
| `scenario_id` | Environmental Scenario | `Int32` | 7 |
| `aep_ratio` | Annual Exceedance Probability | `Float32` | 0.02 |
| `storm_id` | Synthetic Storm ID | `Int32` | 182 |
| `pumping_id` | Pumping Scenario | `Float32` | 0.5 |
| `fragility_scenario` | Fragility Scenario | `Int8` |  |
| `asset_type` | Asset Type | `Int8` | 3 |
| `category` | Category | `String` | public_schools |
| `exposure_type` | Exposure Type | `String` | moderate_exposure |
| `ns_project_id` | Nonstructural Project | `Int8` | 3 |
| `participation_rate` | Participation Rate | `Float32` | 0.75 |
| `migration` | Migration | `String` | base |
| `ssp` | Shared Socioeconomic Pathway | `Int8` | 4 |
| `vegetation_code` | Vegetation Type | `String` | brm |

#### Plan Implementation Scenario

**Column:** `model_group_id`  
**Data Type:** `Int32`  
**Required Condition:** `True`  
**Minimum Value:** 100  
**Maximum Value:** 999  

#### Environmental Scenario

**Column:** `scenario_id`  
**Data Type:** `Int32`  
**Required Condition:** `True`  
**Minimum Value:** 1  
**Maximum Value:** 100  

#### Annual Exceedance Probability

**Column:** `aep_ratio`  
**Data Type:** `Float32`  
**Required Condition:** `variable` in ("annual_exceedance_damages_dollars", "exposure_asset_count", "exposure_critical_infrastructure_count", "exposure_historic_properties_count", "flood_depth", "surge_water_surface_elevations_by_exceedance", "wave_heights_by_exceedance")  
| Value | Label |
| --- | --- |
| 0.2 | 20.0 |
| 0.1 | 10.0 |
| 0.02 | 2.0 |
| 0.01 | 1.0 |
| 0.002 | 0.2 |
| 0.001 | 0.1 |

#### Synthetic Storm ID

**Column:** `storm_id`  
**Data Type:** `Int32`  
**Required Condition:** `variable` in ("surge_water_surface_elevations_by_storm", "wave_heights_by_storm")  
**Minimum Value:** 1  
**Maximum Value:** 1000  

#### Pumping Scenario

**Column:** `pumping_id`  
**Data Type:** `Float32`  
**Required Condition:** `variable` in ("annual_exceedance_damages_dollars", "expected_annual_damages_dollars", "expected_annual_structural_damage", "exposure_asset_count", "exposure_critical_infrastructure_count", "exposure_historic_properties_count", "nonstructural_expected_annual_damage_dollars", "nonstructural_expected_annual_structural_damage")  
| Value | Label |
| --- | --- |
| 0 | No Pumping |
| 0.5 | 50% Pumping Capacity |
| 1 | Full Pumping Capacity |

#### Fragility Scenario

**Column:** `fragility_scenario`  
**Data Type:** `Int8`  
**Required Condition:** `variable` in ("annual_exceedance_damages_dollars", "expected_annual_damages_dollars", "expected_annual_structural_damage", "exposure_asset_count", "exposure_critical_infrastructure_count", "exposure_historic_properties_count", "nonstructural_expected_annual_damage_dollars", "nonstructural_expected_annual_structural_damage")  
| Value | Label |
| --- | --- |
| 0 | No Fragility |
| 1 | IPET Fragility |

#### Asset Type

**Column:** `asset_type`  
**Data Type:** `Int8`  
**Required Condition:** `variable` in ("annual_exceedance_damages_dollars", "expected_annual_damages_dollars", "expected_annual_structural_damage", "exposure_asset_count", "nonstructural_expected_annual_damage_dollars", "nonstructural_expected_annual_structural_damage")  
| Value | Label |
| --- | --- |
| 1 | Small Residential (single-family, manufactured homes, or duplex) |
| 2 | Other Multi-family Residential |
| 3 | Commercial, Industrial, or Agricultural |
| 4 | Other Structural (public, education, or religion) |
| 5 | Non-structural (crops, vehicles, or roads) |

#### Category

**Column:** `category`  
**Data Type:** `String`  
**Required Condition:** `variable` = "exposure_critical_infrastructure_count"  
| Value | Label |
| --- | --- |
| daycares | Daycares |
| drink_water_sources | Drink Water Sources |
| drinking_water_treatment_plants | Drinking Water Treatment Plants |
| electric_substations | Electric Substations |
| emergency_medical_service_stations | Emergency Medical Service Stations |
| fire_stations | Fire Stations |
| gas_stations | Gas Stations |
| hospitals | Hospitals |
| local_emergency_operations_centers | Local Emergency Operations Centers |
| local_law_enforcement | Local Law Enforcement |
| nursing_homes | Nursing Homes |
| nursing_residential_care_facilities | Nursing Residential Care Facilities |
| power_plants | Power Plants |
| public_schools | Public Schools |
| veterans_health_administration_medical_facilities | Veterans Health Administration Medical Facilities |

#### Exposure Type

**Column:** `exposure_type`  
**Data Type:** `String`  
**Required Condition:** `variable` in ("exposure_asset_count", "exposure_critical_infrastructure_count", "exposure_historic_properties_count")  
| Value | Label |
| --- | --- |
| exposure | Exposure |
| moderate_exposure | Moderate Exposure |
| severe_exposure | Severe Exposure |

#### Nonstructural Project

**Column:** `ns_project_id`  
**Data Type:** `Int8`  
**Required Condition:** `variable` in ("nonstructural_expected_annual_damage_dollars", "nonstructural_expected_annual_structural_damage")  
| Value | Label |
| --- | --- |
| 1 | 1 |
| 3 | 3 |

#### Participation Rate

**Column:** `participation_rate`  
**Data Type:** `Float32`  
**Required Condition:** `variable` in ("nonstructural_expected_annual_damage_dollars", "nonstructural_expected_annual_structural_damage")  
| Value | Label |
| --- | --- |
| 0.5 | 50% Participation |
| 0.75 | 75% Participation |
| 1 | Full Participation |

#### Migration

**Column:** `migration`  
**Data Type:** `String`  
**Required Condition:** `variable` in ("population", "population_change")  
| Value | Label |
| --- | --- |
| base | BASE |
| mig | MIG |

#### Shared Socioeconomic Pathway

**Column:** `ssp`  
**Data Type:** `Int8`  
**Required Condition:** `variable` in ("population", "population_change")  
| Value | Label |
| --- | --- |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 5 |

#### Vegetation Type

**Column:** `vegetation_code`  
**Data Type:** `String`  
**Required Condition:** `variable` = "vegetation_coverage_area"  
| Value | Label |
| --- | --- |
| brm | Brackish Marsh |
| for | Fresh Forest |
| frm | Fresh Marsh |
| inm | Intermediate Marsh |
| sam | Saline Marsh |
| upl | Upland |
| wat | Water |

<!--END SECTION-->

### Non-Partition Dimensions

Non-partition dimensions are present in Parquet data files but are not used to create partition directories.

<!--BEGIN SECTION nonpartition_dimensions-->
| Column | Description | Data Type | Example Value |
| --- | --- | --- | --- |
| `percentile` | Percentile | `Int8` | 50 |
| `calendar_year` | Year | `Int32` | 2030 |
| `calendar_day` | Date | `Date` | 2030-01-01 |

#### Percentile

**Column:** `percentile`  
**Data Type:** `Int8`  
**Required Condition:** `variable` in ("annual_exceedance_damages_dollars", "expected_annual_damages_dollars", "expected_annual_structural_damage", "nonstructural_expected_annual_damage_dollars", "nonstructural_expected_annual_structural_damage", "population_change", "surge_water_surface_elevations_by_exceedance", "wave_heights_by_exceedance")  
**Maximum Value:** 100  

#### Year

**Column:** `calendar_year`  
**Data Type:** `Int32`  
**Required Condition:** `time_unit` in ("annual", "decadal")  
**Minimum Value:** 2000  
**Maximum Value:** 2100  

#### Date

**Column:** `calendar_day`  
**Data Type:** `Date`  
**Required Condition:** `time_unit` = "daily"  

<!--END SECTION-->

## File Paths

Parquet and COG files are saved in nested directories corresponding to the dataset and partition dimensions:

```
.../variable=salinity_mean/geographic_unit=hydro_compartment/time_unit=daily/plan_version=2023.0.0/model_group_id=500/scenario_id=7/4f3b1a8418a74c6dacb41378fd69ec26-0.parquet
```

The partitioned Parquet dataset lives within the `aggregate=mean` directory. The directories above and including this level are just filesystem directories, while the directories below it (`model_group=500` and `scenario=7`) are partition directories that can be used by the Parquet reader to understand the structure of the full daily mean hydro compartment salinity dataset.

While they do not use partitioning, COGs are saved in the same directory structure:

```
.../variable=flood_depth/geographic_unit=icm_raster/time_unit=decadal/plan_version=2023.0.0/model_group_id=500/scenario_id=7/aep_ratio=0.1/calendar_year=2030.tiff
```

The nested directories create the data structure needed for the catalog. By traversing the catalog tree (and in some cases, summarizing specific columns from the Parquet files selected through the traversal), the catalog API can respond to queries about available data and valid choices at each level of the catalog hierarchy.

## Variables

<!--BEGIN SECTION variables-->
| Variable | Description | Data Type | Unit | Legend |
| --- | --- | --- | --- | --- |
| `annual_exceedance_damages_dollars` | Annual Exceedance Damages, Dollars | `Float64` | $ | `eadd` |
| `deep_subsidence` | Deep Subsidence | `Float32` | mm/yr | `subsidence_deep` |
| `elevation` | Elevation | `Float32` | NAVD88 m | `elevation` |
| `expected_annual_damages_dollars` | Expected Annual Damages, Dollars | `Float64` | $ | `eadd` |
| `expected_annual_structural_damage` | Expected Annual Structural Damage | `Float64` | structures | `eadd` |
| `exposure_asset_count` | Exposure: Asset Count | `Int32` | structures | `exposure_asset_count` |
| `exposure_critical_infrastructure_count` | Exposure: Critical Infrastructure Count | `Int32` | structures | `exposure_exposed_structures` |
| `exposure_historic_properties_count` | Exposure: Historic Properties Count | `Int32` | structures | `exposure_exposed_structures` |
| `flood_depth` | Flood Depth | `Float32` | ft | `flood_depth` |
| `inundation_depth` | Inundation Depth | `Float64` | m | `inundation_depth` |
| `land_change` | Land Change | `Int32` |  | `land_change` |
| `land_type` | Land Type | `Int32` |  | `land_type` |
| `mineral_accretion` | Mineral Accretion | `Float64` | cm | `accretion` |
| `mineral_deposition` | Mineral Deposition | `Float64` | g/cm^2-yr | `mineral_deposition` |
| `nonstructural_expected_annual_damage_dollars` | Nonstructural Expected Annual Damage, Dollars | `Float64` | $ | `eadd` |
| `nonstructural_expected_annual_structural_damage` | Nonstructural Expected Annual Structural Damage | `Float64` | structure equivalents | `easd` |
| `organic_accretion` | Organic Accretion | `Float64` | cm | `accretion` |
| `organic_accumulation` | Organic Accumulation | `Float64` | g/cm^2-yr | `organic_accumulation` |
| `population` | Population | `Float64` | persons | `population` |
| `population_change` | Population Change | `Float64` | persons | `population_change` |
| `salinity` | Salinity | `Float32` | ppt | `salinity` |
| `shallow_subsidence` | Shallow Subsidence | `Float32` | mm/yr | `subsidence_shallow` |
| `surge_water_surface_elevations_by_exceedance` | Surge Water Surface Elevations by Exceedance | `Float32` | NAVD88 ft | `surge_water_surface_elevations` |
| `surge_water_surface_elevations_by_storm` | Surge Water Surface Elevations by Storm | `Float32` | NAVD88 ft | `surge_water_surface_elevations` |
| `temperature` | Temperature | `Float64` | °C | `temperature` |
| `total_suspended_solids` | Total Suspended Solids | `Float64` | mg/L | `total_suspended_solids` |
| `vegetation_coverage_area` | Vegetation Coverage Area | `Int64` | square m | `vegetation_coverage_area` |
| `vegetation_type_ffibs` | Vegetation Type: FFIBS | `Int32` |  | `ffibs` |
| `vegetation_type_ffibs_extraction_point` | Vegetation Type: FFIBS (Extraction Point) | `Int32` |  | `ffibs_score` |
| `vegetation_type_vct` | Vegetation Type: VCT | `Int32` |  | `vct` |
| `water_level` | Water Level | `Float32` | NAVD88 m | `water_level` |
| `water_level_variability` | Water Level Variability | `Float32` | m | `water_level_variability` |
| `wave_heights_by_exceedance` | Wave Heights by Exceedance | `Float32` | ft | `wave_heights` |
| `wave_heights_by_storm` | Wave Heights by Storm | `Float64` | ft | `wave_heights` |
<!--END SECTION-->

## Grids

Grids represent the spatial units for master plan data. Each grid is associated with a geographic unit and one or more plan versions.

### Raster Grids

<!--BEGIN SECTION raster_grids-->
| Geographic Unit | Plan Version | Bounds | Height | Width | CRS |
| --- | --- | --- | --- | --- | --- |
| icm_raster | 2023.0.0 | (404500, 3187060, 1024720, 3435010) | 8285 | 20674 | EPSG:26915 |
<!--END SECTION-->

### Vector Grids

<!--BEGIN SECTION vector_grids-->
| Geographic Unit | Plan Version | CRS |
| --- | --- | --- |
| block_group | 2023.0.0 | EPSG:26915 |
| ecoregion | 2023.0.0 | EPSG:26915 |
| extraction_point | 2023.0.0 | EPSG:26915 |
| hydro_compartment | 2023.0.0 | EPSG:26915 |
| parish | 2023.0.0 | EPSG:26915 |
<!--END SECTION-->
