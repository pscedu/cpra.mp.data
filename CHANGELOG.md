# Changelog

## [0.7.0](https://github.com/pscedu/cpra.mp.data/compare/v0.6.0...v0.7.0) (2024-10-15)


### Features

* **api:** add geographic summary for tabular data ([e5cc111](https://github.com/pscedu/cpra.mp.data/commit/e5cc111a4f4454a5ebeb66afa7b025faaa116917))
* **api:** add tree-walking catalog query ([fe6f809](https://github.com/pscedu/cpra.mp.data/commit/fe6f8090c65d1b93f1235f269e6ecca5c753487b))
* **dimensions:** require calendar_day for monthly data ([ac578d7](https://github.com/pscedu/cpra.mp.data/commit/ac578d78f768c6ccf5434fe904f614f7a95a3d20))
* **structure:** update data structure from spreadsheet ([8a0c4e3](https://github.com/pscedu/cpra.mp.data/commit/8a0c4e35ee0796c6e792f6c3bd1729753098aa1d))
* **update_structure:** ignore sheets and columns starting with _ ([610b33d](https://github.com/pscedu/cpra.mp.data/commit/610b33db0a95e87c884f5ab455abc68030aff4f4))

## [0.6.0](https://github.com/pscedu/cpra.mp.data/compare/v0.5.0...v0.6.0) (2024-10-01)


### Features

* **api:** add melting of vector tables ([6400cfb](https://github.com/pscedu/cpra.mp.data/commit/6400cfb1f8126ad3ac73952dc36225100a752ed7))
* **tabular:** add function for compacting Parquet partitions ([73a5c13](https://github.com/pscedu/cpra.mp.data/commit/73a5c13360214f9d53c5d2b6d4b0065e052cee19))


### Bug Fixes

* **api:** remove bad import ([b828eaf](https://github.com/pscedu/cpra.mp.data/commit/b828eafcc5d22fcc10f45172ff9f04805cdc6d0a))
* **utils:** allow dimension extraction from Pandas data frames ([65a64fc](https://github.com/pscedu/cpra.mp.data/commit/65a64fc963871d54e7891cf3568814f422df1a8f))


### Documentation

* **tabular:** remove extra word in compact_parquet docstring ([41f8ff3](https://github.com/pscedu/cpra.mp.data/commit/41f8ff3773b3283bea1a5c620b0a045c58707a51))

## [0.5.0](https://github.com/pscedu/cpra.mp.data/compare/v0.4.0...v0.5.0) (2024-09-17)


### Features

* **api:** add calculate and group params for vector data ([a1a0adc](https://github.com/pscedu/cpra.mp.data/commit/a1a0adc022f4a08084ff2d3049e7f61428ccad20))
* **api:** implement presort and sort params ([2130634](https://github.com/pscedu/cpra.mp.data/commit/213063493af7a43e9a380f236e73902827f1514e))


### Bug Fixes

* **api:** fix reporting of unhandled errors while streaming ([542d2cb](https://github.com/pscedu/cpra.mp.data/commit/542d2cb6d4fef0340445d559a984cb986d74ffbd))

## [0.4.0](https://github.com/pscedu/cpra.mp.data/compare/v0.3.1...v0.4.0) (2024-09-05)


### Features

* **api:** add geography filtering for tabular data ([0542b9b](https://github.com/pscedu/cpra.mp.data/commit/0542b9b8a0e5d8af35b22f1cd97d3e9714f4957f))
* **api:** add raster sampling ([4f8c0de](https://github.com/pscedu/cpra.mp.data/commit/4f8c0ded5494b64c8706fc776541ec0058da4ca3))
* **api:** add support for SQL where clauses ([f6dcac0](https://github.com/pscedu/cpra.mp.data/commit/f6dcac00e526773d730491c1aa0f37da03895b55))


### Performance Improvements

* **api:** read tile images in parallel ([6eccdba](https://github.com/pscedu/cpra.mp.data/commit/6eccdba527cc7e79da360addd1ce7bb4e61c5243))


### Documentation

* **data-analysis-api:** migrate docs from cpra-documentation repository ([524c6d2](https://github.com/pscedu/cpra.mp.data/commit/524c6d2d79ec22a4ec0e5870fb59ea5098b4e740))

## [0.3.1](https://github.com/pscedu/cpra.mp.data/compare/v0.3.0...v0.3.1) (2024-08-30)


### Bug Fixes

* **api:** add scaling and padding of tile images ([6631353](https://github.com/pscedu/cpra.mp.data/commit/6631353ae716fef86dd2830b06a25e9d85fe5611))


### Performance Improvements

* **api:** cache LazyFrames for 10 minutes ([8f8ca0c](https://github.com/pscedu/cpra.mp.data/commit/8f8ca0c7c6c09d0038704230d6db1be80e8a8ac6))

## [0.3.0](https://github.com/pscedu/cpra.mp.data/compare/v0.2.0...v0.3.0) (2024-08-30)


### Features

* **api:** add raster tile proof-of-concept ([89a6c9c](https://github.com/pscedu/cpra.mp.data/commit/89a6c9cfc995ec11294e17be0f435a8b634122e6))
* **api:** return 404 response for out-of-bounds tiles ([635630a](https://github.com/pscedu/cpra.mp.data/commit/635630a9aac4e9bb3b558b56ba599e02b1d79914))


### Bug Fixes

* **api:** update timing helper for async functions ([ae5d7c3](https://github.com/pscedu/cpra.mp.data/commit/ae5d7c3f1200d5cbc5276b4eecc773279cb2fc15))


### Performance Improvements

* **api:** read rasters by row of blocks ([79adcda](https://github.com/pscedu/cpra.mp.data/commit/79adcda4f702e09832ed82eac1c1dafb5684b6ee))

## [0.2.0](https://github.com/pscedu/cpra.mp.data/compare/v0.1.0...v0.2.0) (2024-08-20)


### Features

* **api:** add analysis parameters ([aa7106e](https://github.com/pscedu/cpra.mp.data/commit/aa7106e4a21480618a7747627022afbf406a34b6))
* **api:** add proof of concept raster differencing ([805fe6d](https://github.com/pscedu/cpra.mp.data/commit/805fe6d6a1179e8f99e44c9393670bb15569eb1c))
* **api:** temporarily enable CORS for testing ([8b7b735](https://github.com/pscedu/cpra.mp.data/commit/8b7b73597063c60380b338459e1ea3b29274bccd))


### Bug Fixes

* **api:** temporarily disable Arrow format ([bb9a5f6](https://github.com/pscedu/cpra.mp.data/commit/bb9a5f6691b2be70511922ee26e25de14078e621))
* **raster:** use nearest neighbor resampling for COGs ([c9d48f5](https://github.com/pscedu/cpra.mp.data/commit/c9d48f53726d1cbf48bf68b685d0572ab9fb8ae7))


### Performance Improvements

* **api:** read input rasters in parallel ([97b9ef1](https://github.com/pscedu/cpra.mp.data/commit/97b9ef133269c91d54cd30f32c58036af5ce1728))
* **api:** use the default thread pool for reading rasters ([4af16a9](https://github.com/pscedu/cpra.mp.data/commit/4af16a94b1ddb1c2e6d2625a06d99e712444edcd))

## 0.1.0 (2024-08-07)


### Features

* **api:** add route for reading tabular data ([9c4a556](https://github.com/pscedu/cpra.mp.data/commit/9c4a5565c85c5b0336fb76aa631181bd525d73a3))
* **api:** add title and description ([73e35aa](https://github.com/pscedu/cpra.mp.data/commit/73e35aa314b2693dfe8d19701ceee8a684787cea))
* **cli:** add convert command for MP2023 data ([9b761ba](https://github.com/pscedu/cpra.mp.data/commit/9b761ba30f1e349476a6581b2f9f59746e2648f1))
* **config:** allow setting config with environment variables ([ba31b29](https://github.com/pscedu/cpra.mp.data/commit/ba31b293e2109037e86fd6e32e4aab759c01140d))
* **grids:** add raster and vector grid classes ([2541dcf](https://github.com/pscedu/cpra.mp.data/commit/2541dcf44adf759f264317011d2f0fe29c848e16))
* **grids:** set geography IDs for 2023 hydro compartments ([040e6da](https://github.com/pscedu/cpra.mp.data/commit/040e6da81fd18f7769b6e0f89ea2c669901150b6))
* **raster:** add COG writing function ([c65905e](https://github.com/pscedu/cpra.mp.data/commit/c65905e1d0401d51467b7ff60d2b946be47399a2))
* **raster:** add function to find matching COGs ([aa0c7ae](https://github.com/pscedu/cpra.mp.data/commit/aa0c7ae6e1c42519553fa4212e09d262cc1b1816))
* **raster:** sort the COG data frame ([326fe7b](https://github.com/pscedu/cpra.mp.data/commit/326fe7b8d52741b02466d3ee050a81a5489c1e77))
* **schema:** add schema and initial write implementation ([188415e](https://github.com/pscedu/cpra.mp.data/commit/188415ecf51868fedaca8af05ad6d258a0e2c00a))
* **scripts:** add script to update documentation tables ([b35ae1e](https://github.com/pscedu/cpra.mp.data/commit/b35ae1e0c0e17ebf0e4e71b059177b58f928feb5))
* **tabular:** add function for reading Parquet data ([44086dd](https://github.com/pscedu/cpra.mp.data/commit/44086dd92a42a9cb5d76cd54249dabeb165a6cfb))
* **tabular:** add validation of geometry columns ([8e46144](https://github.com/pscedu/cpra.mp.data/commit/8e46144106d658dadeb581414d273f6756546a2b))


### Bug Fixes

* **api:** disable warning about protected namespaces ([5e319be](https://github.com/pscedu/cpra.mp.data/commit/5e319be73614e9cfab262315713bb11ed40523b7))
* **dimensions:** make plan_version the top-level dimension ([669d299](https://github.com/pscedu/cpra.mp.data/commit/669d299d62cd8c1651be2c3b1a24a6a609ae29c1))
* **raster:** cast path column to a string for empty data frames ([0099dc9](https://github.com/pscedu/cpra.mp.data/commit/0099dc99793fae122b9759ab5ad71afaccfe9a4d))
* **variables:** update variable data types ([b7fbaf7](https://github.com/pscedu/cpra.mp.data/commit/b7fbaf7b2f8387f07cbec9203a96be64e8cefaaa))
* **write:** update default base path ([e01aecb](https://github.com/pscedu/cpra.mp.data/commit/e01aecbe30550d5b7155a9795cdfa65ce4c4ca6b))


### Performance Improvements

* **tabular:** precalculate Hive partition schema for reads ([f751649](https://github.com/pscedu/cpra.mp.data/commit/f751649de7e701cf07810ef66e587f530f8df79e))


### Documentation

* add dimension details ([c9f7e5e](https://github.com/pscedu/cpra.mp.data/commit/c9f7e5ed422c4fc7f3acac3dca47b29028ebc743))
* add note that data reading is not implemented ([c2b3524](https://github.com/pscedu/cpra.mp.data/commit/c2b3524c6fda8e0b18ca8d19d707699fda4fc766))
* add variables summary table ([659e4c1](https://github.com/pscedu/cpra.mp.data/commit/659e4c11e0a42255b46f8da82e9c433da42fe402))
* **raster:** add example for writing a COG ([c10373a](https://github.com/pscedu/cpra.mp.data/commit/c10373a4e6b2cc8058f242837b977c0a18b1ba2d))
* **tabular:** add note about missing data exceptions ([182cfdb](https://github.com/pscedu/cpra.mp.data/commit/182cfdb7caca3100696ce6ac0f761fbad8486eff))
* update file path examples ([18d23ce](https://github.com/pscedu/cpra.mp.data/commit/18d23ce05864e843bc3a31e307779ad7d1ebecb0))

## [0.0.0]
