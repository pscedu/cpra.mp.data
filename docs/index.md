# Data Structure

## Dimensions

A hierarchy of dimensions is used to store and catalog Parquet and COG files. The table below summarizes the columns that define these dimensions, as well as the data types and conditions under which the dimensions are required.

**Dataset dimensions** define a single Parquet dataset. They are also used to create the directory structure for COGs (see File Paths below). **Partition dimensions** are used to partition daily data stored in Parquet files, as well as all COGs. **Non-partition dimensions** and **metadata dimensions** are present in Parquet data files but are not used to create partition directories.

<!--BEGIN SECTION dimensions-->
| Label | Name | Type | Dtype | Required | Min Value | Max Value | Example Value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Variable | variable | dataset | String | True |  |  | sal |
| Grid | grid | dataset | String | True |  |  | hydro_compartment_v001 |
| Time Unit | time_unit | dataset | String | True |  |  | daily |
| Plan Implementation Scenario | model_group_id | partition | Int32 | True | 0 | 999 | 500 |
| Environmental Scenario | scenario_id | partition | Int32 | True | 0 | 100 | 7 |
| Aggregate Type | aggregate_type | partition | String | False |  |  | 2_wk_max |
| Annual Exceedance Probability | aep | partition | Float32 | False |  |  | 2 |
| Synthetic Storm ID | storm_id | partition | Int32 | False | 1 | 1000 | 182 |
| Pumping Scenario | pumping_id | partition | Float32 | False |  |  | 0.5 |
| Fragility Scenario | fragility_scenario | partition | Int8 | False |  |  | 0 |
| Population Variant | population_variant | partition | Int8 | False |  |  | 0 |
| CLARA Variant | clara_variant | partition | Int8 | False |  |  | 0 |
| Asset Type | asset_type | partition | Int8 | False |  |  | 3 |
| Infrastructure Category | infrastructure_category | partition | String | False |  |  | public_schools |
| Exposure Type | exposure_type | partition | String | False |  |  | moderate_exposure |
| Nonstructural Project | ns_project_id | partition | Int8 | False |  |  | 3 |
| Participation Rate | participation_rate | partition | Float32 | False |  |  | 0.75 |
| Migration | migration | partition | String | False |  |  | base |
| Shared Socioeconomic Pathway | ssp | partition | Int8 | False |  |  | 4 |
| Species | species_code | partition | String | False |  |  |  |
| Vegetation: Coverage Type | vegetation_coverage_type | partition | String | False |  |  |  |
| Percentile | percentile | partition | Int8 | False | 0 | 100 | 50 |
| Vegetation: Species | veg_species_code | nonpartition | String | False |  |  | coes |
| FFIBS: Coverage Type | ffibs_type | nonpartition | String | False |  |  | 0.5 |
| Year | calendar_year | nonpartition | Int32 | False | 2000 | 2100 | 2030 |
| Date | calendar_day | nonpartition | Date | False |  |  | 2030-01-01 |
| Model | model | metadata | String | True |  |  | icm_v2023.0.0 |
| Ecoregion | ecoregion | tag | String | False |  |  | UBA |
| Region | region | tag | String | False |  |  | BA |
<!--END SECTION-->

## Variables

Variables represent a type of model output. Each variable has a single data type and unit. In addition to the dimensions required for all variables (see above), some variables support additional dimensions, as described in the Required Dimensions column below.

<!--BEGIN SECTION variables-->
| Label | Name | Model | Subroutine | Data Type | Geographic Units | Time Units | Unit | Number Format | Legend | Required Dimensions | Difference Legend Limit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Damage by Return Period in Dollars | damage | clara |  | Float64 | clara_grid_cell | decadal | $ |  | damage | aep, pumping_id, fragility_scenario, population_variant, clara_variant, asset_type, percentile | 1 |
| Expected Annual Damages in Dollars | eadd | clara |  | Float64 | clara_grid_cell | decadal | $ |  | eadd | pumping_id, fragility_scenario, population_variant, clara_variant, asset_type, percentile | 1 |
| Nonstructural Expected Annual Damage, Dollars | eadd_ns | clara |  | Float64 | clara_grid_cell | decadal | $ |  | eadd | pumping_id, fragility_scenario, population_variant, clara_variant, asset_type, ns_project_id, participation_rate, percentile | 1 |
| Expected Annual Damages in Dollars, Structural Portion | eadd_str | clara |  | Float64 | clara_grid_cell | decadal | $ |  | eadd | pumping_id, fragility_scenario, population_variant, clara_variant, asset_type, percentile | 1 |
| Nonstructural Expected Annual Structural Damage | eadd_str_ns | clara |  | Float64 | clara_grid_cell | decadal | $ | .2f | easd | pumping_id, fragility_scenario, population_variant, clara_variant, asset_type, ns_project_id, participation_rate, percentile | 1 |
| Expected Annual Number of Structures Damaged | easd | clara |  | Float64 | clara_grid_cell | decadal | structure equivalents | .2f | easd | pumping_id, fragility_scenario, population_variant, clara_variant, asset_type, percentile | 1 |
| Nonstructural Expected Annual Number of Structures Damaged | easd_ns | clara |  | Float64 | clara_grid_cell | decadal | structure equivalents | .2f | easd | pumping_id, fragility_scenario, population_variant, clara_variant, asset_type, ns_project_id, participation_rate, percentile | 1 |
| Exposure: Asset Count | exposure_count_asset | clara |  | Int32 | clara_grid_cell | decadal | structures | d | exposure_count_asset | aep, pumping_id, fragility_scenario, population_variant, clara_variant, asset_type, exposure_type | 1 |
| Flood Depth | flood_depth_ft | clara |  | Float32 | clara_grid_cell | decadal | ft | .1f | flood_depth | aep, pumping_id, fragility_scenario, population_variant, clara_variant, percentile | 1 |
| Habitat Suitability Index | hsi | icm | hsi | Float32 | veg_grid_cell | annual |  | .2f | hsi | species_code | -1 |
| Flow into Marsh | flom | icm | hydro | Float32 | hydro_compartment | daily |  |  |  | aggregate_type | 1 |
| Salinity | sal | icm | hydro | Float32 | hydro_compartment | daily, monthly, annual | ppt | .1f | sal | aggregate_type | -10 |
| Sediment Accumulation in Open Water | sedacc | icm | hydro | Float32 | hydro_compartment | daily, annual | g/m2/yr | d |  | aggregate_type | -1 |
| Sediment Accumulation at Marsh Edge | sedacc_marshedge | icm | hydro | Float32 | hydro_compartment | daily, annual | g/m2/yr | d |  | aggregate_type | -1 |
| Sediment Accumulation at Marsh Interior | sedacc_marshint | icm | hydro | Float32 | hydro_compartment | daily, annual | g/m2/yr | d |  | aggregate_type | -1 |
| Stage (Water Level) | stg | icm | hydro | Float32 | hydro_compartment | daily, monthly, annual | NAVD88 m | .2f | stg | aggregate_type | 1 |
| Hourly Stage (Water Level) | stghr | icm | hydro | Float32 | hydro_compartment | annual | NAVD88 m | .2f |  | aggregate_type | 1 |
| Stage (Water Level) in Marsh | stgm | icm | hydro | Float32 | hydro_compartment | daily, annual | NAVD88 m | .2f |  | aggregate_type | 1 |
| Water Temperature | tmp | icm | hydro | Float64 | hydro_compartment | daily, annual | degrees C | .1f | tmp | aggregate_type | -1 |
| Tidal Range | trg | icm | hydro | Float32 | hydro_compartment | daily, annual | m | .2f |  | aggregate_type | 1 |
| Total Suspended Solids | tss | icm | hydro | Float64 | hydro_compartment | daily, monthly, annual | mg/L | .1f | tss | aggregate_type | -1 |
| Vegetation Type: FFIBS Coverage | ffibs_coverage_ratio | icm | lavegmod | Float32 | veg_grid_cell | annual |  |  |  | ffibs_type | 1 |
| Vegetation Type: FFIBS | ffibs_score | icm | lavegmod | Float32 | veg_grid_cell | annual |  | .1f | ffibs |  | 1 |
| Vegetation Type: VCT | vct | icm | lavegmod | Int32 | veg_grid_cell | annual |  |  | vct |  | 1 |
| Vegetation Type: Habitat | veg_area | icm | lavegmod | Int64 | veg_grid_cell | annual | square m |  | veg_area |  | 1 |
| Vegetation Type: Species | veg_species | icm | lavegmod | Float32 | veg_grid_cell | annual |  |  |  | veg_species_code | 1 |
| Topobathymetric Elevation | elev | icm | morph | Float32 | morph_pixel | annual | NAVD88 m | .2f | elev |  | 1 |
| Inundation Depth | inun | icm | morph | Float32 | morph_pixel | annual | m | .2f | inun |  | 1 |
| Land Change | lnd_chng | icm | morph | Int32 | morph_pixel | annual |  | d | lnd_chng |  | 1 |
| Land Type | lnd_type | icm | morph | Int32 | morph_pixel | annual |  | d | lnd_type |  | 1 |
| Mineral Accretion | mnrl_accr | icm | morph | Float64 | morph_pixel | annual | cm | .2f | accr |  | -1 |
| Mineral Deposition Rate | mnrl_dep | icm | morph | Float64 | morph_pixel | annual | g/cm^2-yr | .2f | mnrl_dep |  | -1 |
| Organic Matter Accumulation Rate | omar | icm | morph | Float64 | ecoregion | annual | g/cm^2-yr | .2f | omar | ffibs_type, percentile | -1 |
| Organic Accretion | org_accr | icm | morph | Float64 | morph_pixel | annual | cm | .2f | accr |  | -1 |
| Shallow Soil Factors | ssf | icm | morph | Float32 | ecoregion | annual | mm/yr | .1f | ssf | percentile | 1 |
| Subsidence | sub | icm | morph | Float32 | morph_pixel | annual | mm/yr | .1f | sub |  | 1 |
| TEST Mineral Accretion | test_mnrl_accr | icm | morph | Float64 | morph_pixel | annual | cm | .2f | accr |  | -1 |
| TEST Organic Accretion | test_org_accr | icm | morph | Float64 | morph_pixel | annual | cm | .2f | accr |  | -1 |
<!--END SECTION-->

## Grids

Grids represent the spatial units for master plan data. Each grid name includes a version suffix.

<!--BEGIN SECTION grids-->
| Label | Name | Type | CRS | Id Min | Id Max | Id Count | Bounds | Height | Width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Morphology Pixel | morph_pixel_v001 | raster | epsg:26915 |  |  |  | 404500, 3187060, 1024720, 3435010 | 8265 | 20674 |
| Vegetation Grid Cell | veg_grid_cell_v001 | raster | epsg:26915 |  |  |  | 404710, 3199480, 909670, 3374680 | 365 | 1052 |
| Hydro Compartment | hydro_compartment_v001 | vector | epsg:26915 | 1 | 1657 | 1657 |  |  |  |
| Ecoregion | ecoregion_v001 | vector | epsg:26915 | 1 | 32 | 32 |  |  |  |
| Block Group | block_group_v2020 | vector | epsg:26915 |  |  |  |  |  |  |
| CLARA Community | clara_community_v001 | vector | epsg:26915 |  |  |  |  |  |  |
| Parish | parish_v2020 | vector | epsg:26915 |  |  |  |  |  |  |
| CLARA Grid Cell | clara_grid_cell_v001 | vector | epsg:26915 | 1 | 126174 | 126174 |  |  |  |
| Morphology Pixel | morph_pixel_v002 | raster | epsg:26915 |  |  |  | 410470, 3199960, 909670, 3374680 | 5824 | 16640 |
| Vegetation Grid Cell | veg_grid_cell_v002 | raster | epsg:26915 |  |  |  | 410470, 3199960, 909670, 3374680 | 364 | 1040 |
| Hydro Compartment | hydro_compartment_v002 | vector | epsg:26915 | 1 | 1878 | 1878 |  |  |  |
<!--END SECTION-->

<!--BEGIN SECTION dimension_values-->


## Time Units

| Value | Label |
| --- | --- |
| annual | Annual |
| daily | Daily |
| decadal | Decadal |
| monthly | Monthly |


## Aggregate Types

| Value | Label |
| --- | --- |
| 2_wk_max | Two Week Maximum |
| growing_season | Growing Season |
| max | Max |
| mean | Mean |
| mean_oct_apr | Mean October to April |
| mean_sep_mar | Mean September to March |
| mean_summer | Mean Summer |
| range | Range |
| std_dev | Standard Deviation |
| var_summer | Variance Summer |


## Annual Exceedance Probabilities

| Value | Label |
| --- | --- |
| 0.1 | 0.1 |
| 0.2 | 0.2 |
| 1 | 1 |
| 2 | 2 |
| 10 | 10 |
| 20 | 20 |


## Pumping Scenarios

| Value | Label |
| --- | --- |
| 0 | No Pumping |
| 0.5 | 50% Pumping Capacity |
| 1 | Full Pumping Capacity |


## Fragility Scenarios

| Value | Label |
| --- | --- |
| 0 | No Fragility |
| 1 | IPET Fragility |


## Population Variants

| Value | Label |
| --- | --- |
| 0 | No population migration considered (MP23 data) |


## CLARA Variants

| Value | Label |
| --- | --- |
| 0 | No additional variant considered (MP23 data) |


## Asset Types

| Value | Label |
| --- | --- |
| 1 | Small Residential (single-family, manufactured homes, or duplex) |
| 2 | Other Multi-family Residential |
| 3 | Commercial, Industrial, or Agricultural |
| 4 | Other Structural (public, education, or religion) |
| 5 | Non-structural (crops, vehicles, or roads) |


## Infrastructure Categories

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


## Exposure Types

| Value | Label |
| --- | --- |
| exposure | Exposure |
| moderate_exposure | Moderate Exposure |
| severe_exposure | Severe Exposure |


## Nonstructural Projects

| Value | Label |
| --- | --- |
| 1 |  |
| 3 |  |


## Participation Rates

| Value | Label |
| --- | --- |
| 0.5 | 50% Participation |
| 0.75 | 75% Participation |
| 1 | Full Participation |


## Migrations

| Value | Label |
| --- | --- |
| base | BASE |
| mig | MIG |


## Shared Socioeconomic Pathways

| Value | Label |
| --- | --- |
| 1 |  |
| 2 |  |
| 3 |  |
| 4 |  |
| 5 |  |


## Species

| Value | Label |
| --- | --- |
| allig | American Alligator |
| blucj | Juvenile Blue Crab |
| bshrl | Large Juvenile Brown Shrimp |
| bshrs | Small Juvenile Brown Shrimp |
| crayf | Crayfish |
| eagle | Bald Eagle |
| gadwa | Gadwall |
| gmena | Adult Gulf Menhaden |
| gmenj | Juvenile Gulf Menhaden |
| lmbas | Largemouth Bass |
| motdk | Mottled Duck |
| oyste | Eastern Oyster |
| sparr | Seaside Sparroow |
| spsta | Adult Spotted Seatrout |
| spstj | Juvenile Spotted Seatrout |
| wshrl | Large Juvenile White Shrimp |
| wshrs | Small Juvenile White Shrimp |


## Percentiles

| Value | Label |
| --- | --- |
| 10 | 10th Percentile |
| 25 | 25th Percentile |
| 50 | 50th Percentile |
| 75 | 75th Percentile |
| 90 | 90th Percentile |


## Vegetation: Species

| Value | Label |
| --- | --- |
| avge | Avicennia germinans (L.) L. (Black Mangrove) |
| bahabi | Baccharis halimifolia L. (Baccharis - Barrier Island) |
| baregrnd_flt | Flotant Bareground |
| baregrnd_new | New Bareground |
| baregrnd_old | Old Bareground |
| coes | Colocasia esculenta (Elephant Ear) |
| clma10 | Cladium mariscus (L.) Pohl (Sawgrass) |
| dead_flt | Dead Flotant |
| disp | Distichlis spicata (L.) Greene (Saltgrass) |
| dispbi | Distichlis spicata (L.) Greene (Saltgrass - Barrier Island) |
| elba2_flt | Eleocharis baldwinii (Torr.) Chapm. (Spikerush - floating) |
| elce | Eleocharis cellulose (Gulf Coast Spikerush) |
| ivfr | Iva frutescens L. (Marsh-elder) |
| juro | Juncus roemerianus Scheele (Needlegrass) |
| moce2 | Morella cerifera (L.) Small (Wax myrtle) |
| notmod | Not Modeled Coverage |
| nyaq2 | Nyssa aquatica L. (Tupelo) |
| paam2 | Panicum amarum Elliott (Beachgrass – Barrier Island) |
| pahe2 | Panicum hemitomon Schult. (Maidencane) |
| pahe2_flt | Panicum hemitomon Schult. (Maidencane - floating) |
| pava | Paspalum vaginatum Sw. (Paspalum) |
| phau7 | Phragmites australis (Cav.) Trin. ex Steud. (Roseau Cane) |
| popu5 | Polygonum punctatum (Dotted Smartweed) |
| qula3 | Quercus laurifolia Michx. (Laurel Oak) |
| qule | Quercus xleana Nutt. (pro sp.) [imbricara x velutina] (Hybrid Oak) |
| quni | Quercus nigra L. (Water Oak) |
| qute | Quercus texana Buckley (Texas Red Oak) |
| quvi | Quercus virginiana Mill. (Live oak) |
| sala | Sagittaria lancifolia L. (Bulltongue) |
| sala2 | Sagittaria latifolia Willd. (Arrowhead) |
| sani | Salix nigra Marsh. (Black Willow) |
| scam6 | Schoenoplectus americanus (Chainmaker's bulrush) |
| scca11 | Schoenoplectus californicus (C.A. Mey.) Palla (Bullwhip) |
| scro5 | Schoenoplectus robustus (Salt-Marsh bulrush) |
| sose | Solidago sempervirens L. (Seaside goldenrod) |
| spal | Spartina alterniflora Loisel. (Oystergrass) |
| spcy | Spartina cynosuroides (Big Cordgrass) |
| sppa | Spartina patens (Aiton) Muhl. (Wiregrass) |
| sppabi | Spartina patens (Aiton) Muhl. (Wiregrass – Barrier Island) |
| spvi3 | Sporobolus virginicus (L.) Kunth (Seashore dropseed) |
| sthe9 | Strophostyles helvola (L.) Elliott (Amberique bean) |
| tadi2 | Taxodium distichum (L.) Rich. (Cypress) |
| tydo | Typha domingensis Pers. (Cattail) |
| ulam | Ulmus Americana L. (American Elm) |
| unpa | Uniola paniculata L. (Seaoats – Barrier Island) |
| water | Water |
| zimi | Zizaniopsis miliacea (Michx.) Döll & Asch. (Giant Cutgrass) |


## FFIBS: Coverage Types

| Value | Label |
| --- | --- |
| active_fresh_omar | Active Fresh Marsh |
| brack_omar | Brackish Marsh |
| fresh_omar | Fresh Marsh |
| inter_omar | Intermediate Marsh |
| pl_bf | Bottomland Hardwood Forest |
| pl_bm | Brackish Marsh |
| pl_fm | Fresh Marsh |
| pl_im | Intermediate Marsh |
| pl_sf | Swamp Forest |
| pl_sm | Saline Marsh |
| saline_omar | Saline Marsh |
| swamp_omar | Swamp Marsh |


## Models

| Value | Label |
| --- | --- |
| clara_v2023.0.0 | CLARA v2023.0.0 |
| icm_v2023.0.0 | ICM v2023.0.0 |
<!--END SECTION-->

## File Paths

Parquet and COG files are saved in nested directories corresponding to the dataset and partition dimensions:

```
.../variable=salinity/grid=hydro_compartment_v001/time_unit=daily/model_group_id=500/scenario_id=7/aggregate_type=mean/4f3b1a8418a74c6dacb41378fd69ec26-0.parquet
```

The partitioned Parquet dataset lives within the `time_unit=daily` directory. The directories above and including this level are just filesystem directories, while the directories below it (`model_group=500` and `scenario=7`) are partition directories that can be used by the Parquet reader to understand the structure of the full daily mean hydro compartment salinity dataset.

While they do not use partitioning, COGs are saved in the same directory structure:

```
.../variable=lnd_chng/grid=morph_pixel_v001/time_unit=decadal/plan_version=2023.0.0/model_group_id=500/scenario_id=7/aep_ratio=0.1/calendar_year=2030.tiff
```

The nested directories create the data structure needed for the catalog. By traversing the catalog tree (and in some cases, summarizing specific columns from the Parquet files selected through the traversal), the catalog API can respond to queries about available data and valid choices at each level of the catalog hierarchy.

## Legends

Legends determine the breakpoints and colors used to map a variable.

<!--BEGIN SECTION legends-->
### `accr` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0 to < 2 | #feebe2 | 2.0 |
| 2 to < 4 | #fbb4b9 | 4.0 |
| 4 to < 6 | #f768a1 | 6.0 |
| 6 to < 8 | #c51b8a | 8.0 |
| 8 to 10 | #7a0177 |  |


### `damage` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| $0 to < $1K | #edf8fb | 1000.0 |
| $1K to < $10K | #b3cde3 | 10000.0 |
| $10K to < $100K | #8c96c6 | 100000.0 |
| $100K to < $1M | #8856a7 | 1000000.0 |
| $1M or more | #810f7c |  |


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


### `eadd` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| $0 to < $100 | #feebe2 | 100.0 |
| $100 to < $1K | #fcc5c0 | 1000.0 |
| $1K to < $10K | #fa9fb5 | 10000.0 |
| $10K to < $100K | #f768a1 | 100000.0 |
| $100K to < $1M | #c51b8a | 1000000.0 |
| $1M or more | #7a0177 |  |


### `easd` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0 to < 0.0001 | #ffffb2 | 0.0001 |
| 0.0001 to < 0.001 | #fed976 | 0.001 |
| 0.001 to < 0.01 | #feb24c | 0.01 |
| 0.01 to < 0.1 | #fd8d3c | 0.1 |
| 0.1 to < 1 | #f03b20 | 1.0 |
| 1 or more | #bd0026 |  |


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


### `exposure_count_asset` Legend

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


### `ffibs` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| No Data | #00000000 | 0.0 |
| Fresh Forest | #5E823A | 0.2 |
| Fresh Marsh | #DAE86D | 2.0 |
| Intermediate Marsh | #92CBE8 | 5.0 |
| Brackish Marsh | #735753 | 17.0 |
| Saline Marsh | #0679D2 |  |


### `flood_depth` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| No Data | #00000000 | 1.0 |
| 1 to < 4 feet | #728d97 | 4.0 |
| 4 to < 7 feet | #73b7ea | 7.0 |
| 7 to < 10 feet | #99d9a7 | 10.0 |
| 10 to < 13 feet | #f8d88b | 13.0 |
| 13 to < 16 feet | #e9b023 | 16.0 |
| 16 to < 21 feet | #e56524 | 21.0 |
| 21+ feet | #b73865 |  |


### `hsi` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0 to < 0.1 | #313695 | 0.1 |
| 0.1 to < 0.2 | #4575b4 | 0.2 |
| 0.2 to < 0.3 | #74add1 | 0.3 |
| 0.3 to < 0.4 | #abd9e9 | 0.4 |
| 0.4 to < 0.5 | #e0f3f8 | 0.5 |
| 0.5 to < 0.6 | #fee090 | 0.6 |
| 0.6 to < 0.7 | #fdae61 | 0.7 |
| 0.7 to < 0.8 | #f46d43 | 0.8 |
| 0.8 to < 0.9 | #d73027 | 0.9 |
| 0.9 to 1 | #a50026 |  |


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


### `lnd_chng` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| No Data | #00000000 | 12.0 |
| Land Loss | #4591ac | 13.0 |
| No Data | #00000000 | 21.0 |
| Land Gain | #265118 | 22.0 |
| No Data | #00000000 | 23.0 |
| Land Gain | #265118 | 31.0 |
| No Data | #00000000 | 32.0 |
| Land Loss | #4591ac | 33.0 |
| No Data | #00000000 | 52.0 |
| Land Loss | #4591ac | 55.0 |
| No Data | #00000000 | 112.0 |
| Induced Loss | #00f0f0 | 113.0 |
| No Data | #00000000 | 121.0 |
| Land Maintained | #99c104 | 122.0 |
| Land Loss | #4591ac | 123.0 |
| Land Maintained | #99c104 | 131.0 |
| No Data | #00000000 | 132.0 |
| Induced Loss | #00f0f0 | 133.0 |
| No Data | #00000000 | 211.0 |
| Land Gain | #265118 | 212.0 |
| Induced Loss | #00f0f0 | 213.0 |
| Land Gain | #265118 | 222.0 |
| No Data | #00000000 | 223.0 |
| Land Gain | #265118 | 232.0 |
| Induced Loss | #00f0f0 | 233.0 |
| Land Gain | #265118 | 311.0 |
| No Data | #00000000 | 312.0 |
| Induced Loss | #00f0f0 | 313.0 |
| No Data | #00000000 | 321.0 |
| Land Maintained | #99c104 | 322.0 |
| Land Loss | #4591ac | 323.0 |
| Land Maintained | #99c104 | 331.0 |
| No Data | #00000000 | 332.0 |
| Induced Loss | #00f0f0 | 333.0 |
| No Data | #00000000 | 512.0 |
| Induced Loss | #00f0f0 | 513.0 |
| No Data | #00000000 | 521.0 |
| Land Maintained | #99c104 | 522.0 |
| Land Loss | #4591ac | 523.0 |
| Land Maintained | #99c104 | 531.0 |
| No Data | #00000000 | 532.0 |
| Induced Loss | #00f0f0 | 533.0 |
| No Data | #00000000 | 552.0 |
| Induced Loss | #00f0f0 | 553.0 |
| No Data | #00000000 |  |


### `lnd_type` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| No Data | #00000000 | 1.0 |
| Vegetated Wetland | #008000 | 2.0 |
| Open Water | #6495ed | 3.0 |
| Bare/Unvegetated Wetland | #8b4513 | 4.0 |
| Upland/Developed Areas | #999999 | 5.0 |
| Flotant Marsh | #afeeee |  |


### `mnrl_dep` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0 to < 1 | #feebe2 | 1.0 |
| 1 to < 10 | #fbb4b9 | 10.0 |
| 10 to < 100 | #f768a1 | 100.0 |
| 100 to 1,000 | #ae017e |  |


### `omar` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0.05 to < 0.07 | #feebe2 | 0.07 |
| 0.07 to < 0.09 | #fbb4b9 | 0.09 |
| 0.09 to < 0.11 | #f768a1 | 0.11 |
| 0.11 to < 0.13 | #c51b8a | 0.13 |
| 0.13 to 0.15 | #7a0177 |  |


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


### `ssf` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| No Data | #00000000 | 0.0 |
| 0 to < 2 | #feebe2 | 2.0 |
| 2 to < 4 | #fbb4b9 | 4.0 |
| 4 to < 6 | #f768a1 | 6.0 |
| 6 to < 8 | #ae017e |  |


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


### `sub` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| No Data | #00000000 | 0.0 |
| 0 to < 4 | #feebe2 | 4.0 |
| 4 to < 8 | #fbb4b9 | 8.0 |
| 8 to < 12 | #f768a1 | 12.0 |
| 12 to < 16 | #c51b8a | 16.0 |
| 16 to 20 | #7a0177 |  |


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


### `vct` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| No Data | #00000000 | 1.0 |
| Maidencane | #a6cecf | 2.0 |
| Three-Square | #1f78b4 | 3.0 |
| Rosseau Cane | #b2df8a | 4.0 |
| Paspalum | #5ba02c | 5.0 |
| Wiregrass | #fb9a99 | 6.0 |
| Bulltongue | #e31a1c | 7.0 |
| Needlerush | #fdbf6f | 8.0 |
| Bulrush | #ff7f00 | 9.0 |
| Brackish Mix | #cab2d6 | 10.0 |
| Oystergrass | #6a3d9a | 11.0 |
| Saltgrass | #ffff99 | 12.0 |
| Bottomland Hardwood Forest + Swamp Forest | #005a00 | 13.0 |
| Barrier Island Vegetation | #ff80ff | 14.0 |
| No Data | #ffffff | 300.0 |
| Unvegetated | #7f5f1e | 400.0 |
| No Data | #00000000 | 500.0 |
| Flotant | #8cf0f0 | 9999.0 |
| No Data | #ffffff |  |


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


### `wave_heights` Legend

| Label | Color | Upper Bound |
| --- | --- | --- |
| 0.99 or less | #d7191c | 1.0 |
| 1.00 to 1.99 | #fdae61 | 2.0 |
| 2.00 to 4.99 | #ffffbf | 5.0 |
| 5.00 to 9.99 | #abdda4 | 10.0 |
| 10.0 or more | #2b83ba |  |


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
