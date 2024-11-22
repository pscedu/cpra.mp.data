# Data Structure

## Variables

<!--BEGIN SECTION variables-->
| Label | Name | Model | Subroutine | Data Type | Geographic Units | Time Units | Unit | Legend | Required Dimensions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vegetation Type: VCT | ann | icm | lavegmod | Int32 | veg_grid_cell | annual |  | vct |  |
| Ground Elevation | elev | icm | morph | Float32 | morph_pixel | annual | NAVD88 m | elev |  |
| Vegetation Type: FFIBS | ffibs | icm | lavegmod | Int32 | veg_grid_cell | annual |  | ffibs |  |
| Habitat Suitability Index | hsi | icm | hsi | Float32 | veg_grid_cell | annual |  | hsi |  |
| Inundation Depth | inun | icm | morph | Float32 | morph_pixel | annual | m | inun |  |
| Land Change | lnd_chng | icm | morph | Int32 | morph_pixel | annual |  | lnd_chng_id |  |
| Land Type | lnd_type | icm | morph | Int32 | morph_pixel | annual |  | lnd_type_id |  |
| Mineral Accretion | mnrl_accr | icm | morph | Float64 | morph_pixel | annual | cm | accr |  |
| Mineral Deposition Rate | mnrl_dep | icm | morph | Float64 | morph_pixel | annual | g/cm^2-yr | mnrl_dep |  |
| Organic Matter Accumulation Rate | omar | icm | morph | Float64 | morph_pixel | annual | g/cm^2-yr | omar |  |
| Organic Accretion | org_accr | icm | morph | Float64 | morph_pixel | annual | cm | accr |  |
| Salinity | sal | icm | hydro | Float32 | hydro_compartment | daily, monthly, annual | ppt | sal | aggregate_type |
| Shallow Soil Factors | ssf | icm | morph | Float32 | hydro_compartment | annual | mm/yr | ssf |  |
| Stage (Water Level) | stg | icm | hydro | Float32 | hydro_compartment | daily, monthly, annual | NAVD88 m | stg | aggregate_type |
| Subsidence | sub | icm | morph | Float32 | morph_pixel | annual | mm/yr | sub |  |
| Water Temperature | tmp | icm | hydro | Float64 | hydro_compartment | daily, monthly, annual | degrees C | tmp | aggregate_type |
| Daily Tidal Range | trg | icm | hydro | Float32 | hydro_compartment | daily, annual | m |  | aggregate_type |
| Total Suspended Solids | tss | icm | hydro | Float64 | hydro_compartment | daily, monthly, annual | mg/L | tss | aggregate_type |
| Vegetation Coverage Area | veg_area | icm | lavegmod | Int64 | veg_grid_cell | annual | square m | veg_area | vegetation_code |
| Water Level Variability | wlv | icm | hydro | Float32 | hydro_compartment | annual | m | wlv |  |
<!--END SECTION-->

## Dimensions

A hierarchy of dimensions is used to store and catalog Parquet and COG files. The table below summarizes the columns that define these dimensions, as well as the data types and conditions under which the dimensions are required.

**Dataset dimensions** define a single Parquet dataset. They are also used to create the directory structure for COGs (see File Paths below). **Partition dimensions** are used to partition daily and monthly data stored in Parquet files, as well as all COGs. **Non-partition dimensions** are present in Parquet data files but are not used to create partition directories.

<!--BEGIN SECTION dimensions-->
| Label | Name | Type | Dtype | Required | Min Value | Max Value | Example Value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Variable | variable | dataset | String | True |  |  | sal |
| Grid | grid | dataset | String | True |  |  | hydro_compartment_v1 |
| Time Unit | time_unit | dataset | String | True |  |  | daily |
| Plan Implementation Scenario | model_group_id | parition | Int32 | True | 100 | 999 | 500 |
| Environmental Scenario | scenario_id | parition | Int32 | True | 1 | 100 | 7 |
| Aggregate Type | aggregate_type | partition | String | False |  |  | 2_wk_max |
| Annual Exceedance Probability | aep | parition | Float32 | False |  |  | 2 |
| Synthetic Storm ID | storm_id | parition | Int32 | False | 1 | 1000 | 182 |
| Pumping Scenario | pumping_id | parition | Float32 | False |  |  | 0.5 |
| Fragility Scenario | fragility_scenario | parition | Int8 | False |  |  | 0 |
| Asset Type | asset_type | parition | Int8 | False |  |  | 3 |
| Infrastructure Category | infrastructure_category | parition | String | False |  |  | public_schools |
| Exposure Type | exposure_type | parition | String | False |  |  | moderate_exposure |
| Nonstructural Project | ns_project_id | parition | Int8 | False |  |  | 3 |
| Participation Rate | participation_rate | parition | Float32 | False |  |  | 0.75 |
| Migration | migration | parition | String | False |  |  | base |
| Shared Socioeconomic Pathway | ssp | parition | Int8 | False |  |  | 4 |
| Vegetation Type | vegetation_code | parition | String | False |  |  | brm |
| Percentile | percentile | nonpartition | Int8 | False | 0 | 100 | 50 |
| Year | calendar_year | nonpartition | Int32 | False | 2000 | 2100 | 2030 |
| Date | calendar_day | nonpartition | Date | False |  |  | 2030-01-01 |
| Model | model | metadata | String | True |  |  | icm_v2023.0.0 |


 ### Time Unit Values

| Value | Label |
| --- | --- |
| annual | Annual |
| daily | Daily |
| decadal | Decadal |
| monthly | Monthly |


 ### Aggregate Type Values

| Value | Label |
| --- | --- |
| mean | Mean |
| mean_oct_apr | Mean October to April |
| mean_sep_mar | Mean September to March |
| mean_summer | Mean Summer |
| growing_season | Growing Season |
| max | Max |
| var_summer | Variance Summer |
| std_dev | Standard Deviation |
| 2_wk_max | Two Week Maximum |


 ### Annual Exceedance Probability Values

| Value | Label |
| --- | --- |
| 20 | 20 |
| 10 | 10 |
| 2 | 2 |
| 1 | 1 |
| 0.2 | 0.2 |
| 0.1 | 0.1 |


 ### Pumping Scenario Values

| Value | Label |
| --- | --- |
| 0 | No Pumping |
| 0.5 | 50% Pumping Capacity |
| 1 | Full Pumping Capacity |


 ### Fragility Scenario Values

| Value | Label |
| --- | --- |
| 0 | No Fragility |
| 1 | IPET Fragility |


 ### Asset Type Values

| Value | Label |
| --- | --- |
| 1 | Small Residential (single-family, manufactured homes, or duplex) |
| 2 | Other Multi-family Residential |
| 3 | Commercial, Industrial, or Agricultural |
| 4 | Other Structural (public, education, or religion) |
| 5 | Non-structural (crops, vehicles, or roads) |


 ### Infrastructure Category Values

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


 ### Exposure Type Values

| Value | Label |
| --- | --- |
| exposure | Exposure |
| moderate_exposure | Moderate Exposure |
| severe_exposure | Severe Exposure |


 ### Nonstructural Project Values

| Value | Label |
| --- | --- |
| 1 |  |
| 3 |  |


 ### Participation Rate Values

| Value | Label |
| --- | --- |
| 0.5 | 50% Participation |
| 0.75 | 75% Participation |
| 1 | Full Participation |


 ### Migration Values

| Value | Label |
| --- | --- |
| base | BASE |
| mig | MIG |


 ### Shared Socioeconomic Pathway Values

| Value | Label |
| --- | --- |
| 1 |  |
| 2 |  |
| 3 |  |
| 4 |  |
| 5 |  |


 ### Vegetation Type Values

| Value | Label |
| --- | --- |
| brm | Brackish Marsh |
| for | Fresh Forest |
| frm | Fresh Marsh |
| inm | Intermediate Marsh |
| sam | Saline Marsh |
| upl | Upland |
| wat | Water |


 ### Model Values

| Value | Label |
| --- | --- |
| icm_v2023.0.0 | ICM v2023.0.0 |
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

## Grids

Grids represent the spatial units for master plan data. Each grid is associated with a geographic unit and one or more plan versions.

<!--BEGIN SECTION grids-->
| Name | Type | CRS | Geographies | Bounds | Height | Width |
| --- | --- | --- | --- | --- | --- | --- |
| morph_pixel_v001 | raster | epsg:26915 |  | 404500, 3187060, 1024720, 3435010 | 8285 | 20674 |
| veg_grid_cell_v001 | vector | epsg:26915 |  |  |  |  |
| hydro_compartment_v001 | vector | epsg:26915 | 1787 |  |  |  |
| ecoregion_v001 | vector | epsg:26915 | 32 |  |  |  |
| block_group_v2020 | vector | epsg:26915 |  |  |  |  |
| clara_community_v001 | vector | epsg:26915 |  |  |  |  |
| parish_v2020 | vector | epsg:26915 |  |  |  |  |
<!--END SECTION-->

## Legends

Legends determine the breakpoints and colors used to map a variable.

<!--BEGIN SECTION legends-->
### `ssf` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| No Data | #00000000 | 0.0 |
| 0 to < 2 | #feebe2 | 2.0 |
| 2 to < 4 | #fbb4b9 | 4.0 |
| 4 to < 6 | #f768a1 | 6.0 |
| 6 to < 8 | #ae017e |  |


### `omar` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0.05 to < 0.07 | #feebe2 | 0.07 |
| 0.07 to < 0.09 | #fbb4b9 | 0.09 |
| 0.09 to < 0.11 | #f768a1 | 0.11 |
| 0.11 to < 0.13 | #c51b8a | 0.13 |
| 0.13 to 0.15 | #7a0177 |  |


### `ffibs` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| Fresh Forest | #5E823A |  |
| Fresh Marsh | #DAE86D |  |
| Intermediate Marsh | #92CBE8 |  |
| Brackish Marsh | #735753 |  |
| Saline Marsh | #0679D2 |  |


### `mnrl_dep` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0 to < 1 | #feebe2 | 1.0 |
| 1 to < 10 | #fbb4b9 | 10.0 |
| 10 to < 100 | #f768a1 | 100.0 |
| 100 to 1,000 | #ae017e |  |


### `population` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0 to 9,999 | #9e0142 | 10000.0 |
| 10,000 to 19,999 | #d53e4f | 20000.0 |
| 20,000 to 29,999 | #f46d43 | 30000.0 |
| 30,000 to 39,999 | #fdae61 | 40000.0 |
| 40,000 to 49,999 | #fee08b | 50000.0 |
| 50,000 to 74,999 | #ffffbf | 75000.0 |
| 75,000 to 99,999 | #e6f598 | 100000.0 |
| 100,000 to 199,999 | #abdda4 | 200000.0 |
| 200,000 to 499,999 | #66c2a5 | 500000.0 |
| 500,000 to 999,999 | #3288bd | 1000000.0 |
| 1,000,000 or more | #5e4fa2 |  |


### `default` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0 to 49 | #d53e4f | 50.0 |
| 50 to 99 | #fc8d59 | 100.0 |
| 100 to 249 | #fee08b | 250.0 |
| 250 to 499 | #ffffbf | 500.0 |
| 500 to 999 | #e6f598 | 1000.0 |
| 1,000 to 4,999 | #99d594 | 5000.0 |
| 5,000 or more | #3288bd |  |


### `veg_area` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0 | #9e0142 | 1.0 |
| 1 to 2,499,999 | #d53e4f | 2500000.0 |
| 2,500,000 to 4,999,999 | #f46d43 | 5000000.0 |
| 5,000,000 to 9,999,999 | #fdae61 | 10000000.0 |
| 10,000,000 to 19,999,999 | #fee08b | 20000000.0 |
| 20,000,000 to 49,999,999 | #ffffbf | 50000000.0 |
| 50,000,000 to 99,9999,999 | #e6f598 | 100000000.0 |
| 100,000,000 to 199,999,999 | #abdda4 | 200000000.0 |
| 200,000,000 to 399,999,999 | #66c2a5 | 300000000.0 |
| 300,000,000 to 499,999,999 | #3288bd | 500000000.0 |
| 500,000,000 or more | #5e4fa2 |  |


### `inun` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| -2.00 to < -1.00 | #d53e4f | -1.0 |
| -1.00 to < -0.50 | #f46d43 | -0.5 |
| -0.50 to < -0.25 | #fdae61 | -0.25 |
| -0.25 to < 0.00 | #fee08b | 0.0 |
| 0.00 to < 0.25 | #e6f598 | 0.25 |
| 0.25 to < 0.50 | #abdda4 | 0.5 |
| 0.50 to < 1.00 | #66c2a5 | 1.0 |
| 1.00 or more | #3288bd |  |


### `lnd_chng_id` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| Land Loss | #4591ac |  |
| Induced Loss | #00f0f0 |  |
| Land Maintained | #99c104 |  |
| Land Gain | #265118 |  |


### `sal` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0 to < 0.1 | #d53e4f | 0.1 |
| 0.1 to < 0.5 | #f46d43 | 0.5 |
| 0.5 to < 1.0 | #fdae61 | 1.0 |
| 1.0 to < 2.0 | #fee08b | 2.0 |
| 2.0 to < 5.0 | #e6f598 | 5.0 |
| 5.0 to < 10.0 | #abdda4 | 10.0 |
| 10.0 to < 20.0 | #66c2a5 | 20.0 |
| 20.0 to 35.0 | #3288bd |  |


### `eadd` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| $0 to < $1M | #DDC5E0 | 1000000.0 |
| $1M to < $10M | #CCAAD1 | 10000000.0 |
| $10M to < $50M | #B282BA | 50000000.0 |
| $50M to < $100M | #A064A9 | 100000000.0 |
| $100M to < $500M | #5D4795 | 500000000.0 |
| $500M to < $1B | #423575 | 1000000000.0 |
| $1B or more | #221F53 |  |


### `population_change` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0 to 9 | #d53e4f | 10.0 |
| 10 to 29 | #f46d43 | 30.0 |
| 30 to 49 | #fdae61 | 50.0 |
| 50 to 99 | #fee08b | 100.0 |
| 100 to 249 | #ffffbf | 250.0 |
| 250 to 499 | #e6f598 | 500.0 |
| 500 to 999 | #abdda4 | 1000.0 |
| 1,000 to 4,999 | #66c2a5 | 5000.0 |
| 5,000 or more | #3288bd |  |


### `wave_heights` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0.99 or less | #d7191c | 1.0 |
| 1.00 to 1.99 | #fdae61 | 2.0 |
| 2.00 to 4.99 | #ffffbf | 5.0 |
| 5.00 to 9.99 | #abdda4 | 10.0 |
| 10.0 or more | #2b83ba |  |


### `elev` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| Less than -2.0 | #d53e4f | -2.0 |
| -2.0 to < -1.0 | #f46d43 | -1.0 |
| -1.0 to < -0.5 | #fdae61 | -0.5 |
| -0.5 to < 0.0 | #fee08b | 0.0 |
| 0.0 to < 0.5 | #e6f598 | 0.5 |
| 0.5 to < 1.0 | #abdda4 | 1.0 |
| 1.0 to < 2.0 | #66c2a5 | 2.0 |
| 2.0 or greater | #3288bd |  |


### `exposure_asset_count` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0 to 9 | #d53e4f | 10.0 |
| 10 to 49 | #f46d43 | 50.0 |
| 50 to 99 | #fdae61 | 100.0 |
| 100 to 249 | #fee08b | 250.0 |
| 250 to 499 | #e6f598 | 500.0 |
| 500 to 999 | #abdda4 | 1000.0 |
| 1,000 to 9,999 | #66c2a5 | 10000.0 |
| 10,000 or more | #3288bd |  |


### `exposure_exposed_structures` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| No structures | #00000000 |  |
| 0 structures exposed | #d53e4f | 1.0 |
| 1 structure exposed | #fc8d59 | 2.0 |
| 2 structures exposed | #fee08b | 3.0 |
| 3 to 5 structures exposed | #e6f598 | 6.0 |
| 6 to 9 structures exposed | #99d594 | 10.0 |
| 10 or more structures exposed | #3288bd |  |


### `flood_depth` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 1 to < 4 feet | #728D97 |  |
| 4 to < 7 feet | #73B7EA |  |
| 7 to < 10 feet | #99D9A7 |  |
| 10 to < 13 feet | #F8D88B |  |
| 13 to < 16 feet | #E9B023 |  |
| 16 to < 21 feet | #E56524 |  |
| 21+ feet | #B73865 |  |


### `surge_water_surface_elevations` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0.09 or less | #d53e4f | 0.1 |
| 0.10 to 0.24 | #f46d43 | 0.25 |
| 0.25 to 0.49 | #fdae61 | 0.5 |
| 0.50 to 0.99 | #fee08b | 1.0 |
| 1.00 to 1.49 | #ffffbf | 1.5 |
| 1.50 to 1.99 | #e6f598 | 2.0 |
| 2.00 to 2.99 | #abdda4 | 3.0 |
| 3.00 to 4.99 | #66c2a5 | 5.0 |
| 5.00 or more | #3288bd |  |


### `accr` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0 to < 2 | #feebe2 | 2.0 |
| 2 to < 4 | #fbb4b9 | 4.0 |
| 4 to < 6 | #f768a1 | 6.0 |
| 6 to < 8 | #c51b8a | 8.0 |
| 8 to 10 | #7a0177 |  |


### `stg` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0 to < 0.2 | #9e0142 | 0.2 |
| 0.2 to < 0.4 | #d53e4f | 0.4 |
| 0.4 to < 0.6 | #f46d43 | 0.6 |
| 0.6 to < 0.8 | #fdae61 | 0.8 |
| 0.8 to < 1.0 | #fee08b | 1.0 |
| 1.0 to < 1.5 | #e6f598 | 1.5 |
| 1.5 to < 2.0 | #abdda4 | 2.0 |
| 2.0 to < 3.0 | #66c2a5 | 3.0 |
| 3.0 to < 5.0 | #3288bd | 5.0 |
| 5.0 or more | #5e4fa2 |  |


### `easd` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0 to 49 | #FEDB9E | 50.0 |
| 50 to 99 | #FBBC76 | 100.0 |
| 100 to 249 | #F7903B | 250.0 |
| 250 to 499 | #F36F41 | 500.0 |
| 500 to 999 | #DC4840 | 1000.0 |
| 1,000 to 4,999 | #BA233E | 5000.0 |
| 5,000 or more | #701231 |  |


### `lnd_type_id` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| No Data | #00000000 | 1.0 |
| Vegetated Wetland | #008000 | 2.0 |
| Open Water | #6495ed | 3.0 |
| Bare/Unvegetated Wetland | #8b4513 | 4.0 |
| Upland/Developed Areas | #999999 | 5.0 |
| Flotant Marsh | #afeeee |  |


### `sub` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| No Data | #00000000 | 0.0 |
| 0 to < 4 | #feebe2 | 4.0 |
| 4 to < 8 | #fbb4b9 | 8.0 |
| 8 to < 12 | #f768a1 | 12.0 |
| 12 to < 16 | #c51b8a | 16.0 |
| 16 to 20 | #7a0177 |  |


### `tss` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0 to < 10 | #d53e4f | 10.0 |
| 10 to < 50 | #f46d43 | 50.0 |
| 50 to < 100 | #fdae61 | 100.0 |
| 100 to < 200 | #fee08b | 200.0 |
| 200 to < 300 | #e6f598 | 300.0 |
| 300 to < 400 | #abdda4 | 400.0 |
| 400 to < 500 | #66c2a5 | 500.0 |
| 500 or more | #3288bd |  |


### `tmp` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| Less than 19 | #d53e4f | 19.0 |
| 19 to < 20 | #f46d43 | 20.0 |
| 20 to < 21 | #fdae61 | 21.0 |
| 21 to < 22 | #fee08b | 22.0 |
| 22 to < 23 | #ffffbf | 23.0 |
| 23 to < 24 | #e6f598 | 24.0 |
| 24 to < 25 | #abdda4 | 25.0 |
| 25 to < 26 | #66c2a5 | 26.0 |
| 27 or more | #3288bd |  |


### `hsi` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0 | #9e0142 | 1.0 |
| 1 to 49 | #d53e4f | 50.0 |
| 50 to 100 | #f46d43 | 100.0 |
| 100 to 499 | #fdae61 | 500.0 |
| 500 to 999 | #fee08b | 1000.0 |
| 1,000 to 1,499 | #e6f598 | 1500.0 |
| 1,500 to 1,999 | #abdda4 | 2000.0 |
| 2,000 to 2,499 | #66c2a5 | 2500.0 |
| 2,500 to 2,999 | #3288bd | 3000.0 |
| 3,000 or more | #5e4fa2 |  |


### `vct` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| Maidencane | #a6cecf |  |
| Three-Square | #1f78b4 |  |
| Rosseau Cane | #b2df8a |  |
| Paspalum | #5ba02c |  |
| Wiregrass | #fb9a99 |  |
| Bulltongue | #e31a1c |  |
| Needlerush | #fdbf6f |  |
| Bulrush | #ff7f00 |  |
| Brackish Mix | #cab2d6 |  |
| Oystergrass | #6a3d9a |  |
| Saltgrass | #ffff99 |  |
| Bottomland Hardwood Forest + Swamp Forest | #005a00 |  |
| Barrier Island Vegetation | #ff80ff |  |
| Unvegetated | #7f5f1e |  |
| Flotant | #8cf0f0 |  |


### `ffibs_score` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| No Data | #00000000 | 0.0 |
| Fresh Forest | #5E823A | 0.2 |
| Fresh Marsh | #DAE86D | 2.0 |
| Intermediate Marsh | #92CBE8 | 5.0 |
| Brackish Marsh | #735753 | 17.0 |
| Saline Marsh | #0679D2 |  |


### `wlv` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0.09 or less | #d53e4f | 0.1 |
| 0.10 to 0.19 | #fc8d59 | 0.2 |
| 0.20 to 0.29 | #fee08b | 0.3 |
| 0.30 to 0.39 | #e6f598 | 0.4 |
| 0.40 to 0.49 | #99d594 | 0.5 |
| 0.50 or more | #3288bd |  |
<!--END SECTION-->