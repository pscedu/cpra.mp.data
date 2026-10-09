# Changelog

## [0.54.0](https://github.com/pscedu/cpra.mp.data/compare/v0.53.0...v0.54.0) (2026-10-09)


### Features

* **api:** add health check route ([71c7b04](https://github.com/pscedu/cpra.mp.data/commit/71c7b04f1a69d37083cc35b987485224f8f66759))


### Bug Fixes

* pin Polars to version 1.x ([a0912b1](https://github.com/pscedu/cpra.mp.data/commit/a0912b1f63138a3f8a0a5f4ccce28154f55e960a))

## [0.53.0](https://github.com/pscedu/cpra.mp.data/compare/v0.52.2...v0.53.0) (2026-10-09)


### Features

* **structure:** add ecorevion_v002 grid ([2ffe575](https://github.com/pscedu/cpra.mp.data/commit/2ffe57536b36b24de8a4251de5193b1fe44954a8))


### Bug Fixes

* **analyses:** update baseline years for lnd_chng ([a040bc7](https://github.com/pscedu/cpra.mp.data/commit/a040bc761373cec105fc86502baeeacd0e4b270f))
* **analysis:** exclude point sampling from raster band limit ([4a703b7](https://github.com/pscedu/cpra.mp.data/commit/4a703b7ad34879388f4d3b0f88ea76fbdb60e100))


### Documentation

* update data structure docs ([21c5be1](https://github.com/pscedu/cpra.mp.data/commit/21c5be14f8454321595bb8246f6b4f65eebb2d92))

## [0.52.2](https://github.com/pscedu/cpra.mp.data/compare/v0.52.1...v0.52.2) (2026-10-09)


### Performance Improvements

* **raster:** remove unnecessary metadata reads ([2d0eb6f](https://github.com/pscedu/cpra.mp.data/commit/2d0eb6f01b5b77889efb4ea5fc0d001406a595e0))

## [0.52.1](https://github.com/pscedu/cpra.mp.data/compare/v0.52.0...v0.52.1) (2026-10-09)


### Bug Fixes

* make Release Please update uv lock file ([badb795](https://github.com/pscedu/cpra.mp.data/commit/badb795ecd1eef2e9c987902652a83e0db1ff8ad))

## [0.52.0](https://github.com/pscedu/cpra.mp.data/compare/v0.51.0...v0.52.0) (2026-10-09)


### Features

* add analyze_data function ([1a6a771](https://github.com/pscedu/cpra.mp.data/commit/1a6a7717d1749fb089f017fac7051b36acb19c76))
* **analyses:** add derived variable and dimension analyses ([fcee01d](https://github.com/pscedu/cpra.mp.data/commit/fcee01d4c74e6f1d6de654ed959ab2a89c5ba568))
* **analyses:** add land type confidence analysis ([11c72b8](https://github.com/pscedu/cpra.mp.data/commit/11c72b806aa62590dc4b777c7618b43823eb35f8))
* **analyses:** add land type plus named analysis ([e55d655](https://github.com/pscedu/cpra.mp.data/commit/e55d655493e8ea60a0c39fc4e2661ee8d0d4a368))
* **analyses:** improve land confidence legends ([51c79bc](https://github.com/pscedu/cpra.mp.data/commit/51c79bc23f65ba821879845c7c952d1bb95f0fb4))
* **analysis:** add case operation and optimize filter pushdown ([3fcbe2d](https://github.com/pscedu/cpra.mp.data/commit/3fcbe2d6bddccaa4ac27bd5b639c232182981e67))
* **analysis:** add flexible calculate parameters ([fa02cb5](https://github.com/pscedu/cpra.mp.data/commit/fa02cb508ee959735e23f35503e8382b1e2ae98f))
* **analysis:** add raster point reader ([9f9de04](https://github.com/pscedu/cpra.mp.data/commit/9f9de04dfeb8faa8cdf3d457f24596c9c6878c4a))
* **analysis:** connect lowerings to execute ([a4b0a5d](https://github.com/pscedu/cpra.mp.data/commit/a4b0a5d1ea5a66cf540c42aed240d9caf66392c8))
* **analysis:** connect tile decimation ([4956fa2](https://github.com/pscedu/cpra.mp.data/commit/4956fa24b14e9af95ab1e33e8bd1747727230cd7))
* **analysis:** implement band-wide lowerings ([9b961e9](https://github.com/pscedu/cpra.mp.data/commit/9b961e991af0d97a272bca1373ec408b0e63ac31))
* **analysis:** implement data query ([64c79f1](https://github.com/pscedu/cpra.mp.data/commit/64c79f1b58ba340b900dc74cd689c5a3c71e9fa2))
* **analysis:** implement frame logic ([bf6f7f6](https://github.com/pscedu/cpra.mp.data/commit/bf6f7f6ba53cd68709953adf312e37533ff2c0a3))
* **analysis:** implement long lowerings and walk ([6d350b8](https://github.com/pscedu/cpra.mp.data/commit/6d350b8cc7868ccd589f80c383e2c35b65a387f8))
* **analysis:** implement raster scan ([de19618](https://github.com/pscedu/cpra.mp.data/commit/de196185a7814a3debaa2b22042a7140433166b2))
* **analysis:** implement raster to vector projection ([bf8d625](https://github.com/pscedu/cpra.mp.data/commit/bf8d62548a2a0e7b85b2ab6c3349b3b42270d00b))
* **analysis:** implement sinks ([f89bac2](https://github.com/pscedu/cpra.mp.data/commit/f89bac2530ae2d92028f97c408130e79fab550da))
* **analysis:** implement the request layer ([f70099f](https://github.com/pscedu/cpra.mp.data/commit/f70099f516bb92c41638bf22ba9edfc2c467107d))
* **analysis:** implement tiles ([d9bd1aa](https://github.com/pscedu/cpra.mp.data/commit/d9bd1aa5e79ab46f7816ebd6e8df778f17e62866))
* **analysis:** implement vector-to-vector projection ([010de77](https://github.com/pscedu/cpra.mp.data/commit/010de77c6abdf3d7047289707f34d1927513b084))
* **analysis:** initial work on making pipeline step order flexible ([cfa54ba](https://github.com/pscedu/cpra.mp.data/commit/cfa54ba42abb9b209c090b46a948417fb24682c8))
* **analysis:** start connecting API to new analysis module ([b14d837](https://github.com/pscedu/cpra.mp.data/commit/b14d837a1b6400c120b4aedf4c767f84efeb2442))
* **api:** add composable pipelines ([4a0eabb](https://github.com/pscedu/cpra.mp.data/commit/4a0eabbee6af983528ae2b6bcb8efc632579d257))
* **api:** add count-by-value ([6ad61a0](https://github.com/pscedu/cpra.mp.data/commit/6ad61a0c512b59d203581efebc0b8dabe87860e3))
* **api:** add pixel-level expressions ([d4890cb](https://github.com/pscedu/cpra.mp.data/commit/d4890cb45e12a2a61eafd7c0d03b9ce85c6411d2))
* **api:** add where clause parsing and scan broadcast ([e2d822d](https://github.com/pscedu/cpra.mp.data/commit/e2d822d73a57787669b74976c2c3b55614bcda40))
* **api:** enable Polars raster IO plugin by default ([89ceedc](https://github.com/pscedu/cpra.mp.data/commit/89ceedcf16ec6a804e5efdcc7ab932f6f4006ba4))
* **api:** use Polars for raster tile calculations ([a8dd802](https://github.com/pscedu/cpra.mp.data/commit/a8dd8024547bd86a8efbae2cc8c38946040b3c5b))
* **api:** use Polars to do per-window raster calculations ([7ab0069](https://github.com/pscedu/cpra.mp.data/commit/7ab0069dac8656a6adb25da3dc909e0197d4a49f))
* **api:** use the new analysis engine ([a6dec5b](https://github.com/pscedu/cpra.mp.data/commit/a6dec5b3633101f38ac9b221ca1e25aaf39f4f01))
* **scripts:** add --append-missing flag to migration script ([4bb5095](https://github.com/pscedu/cpra.mp.data/commit/4bb5095ef53f15037a7e163c641447116a0412a7))
* **scripts:** add --skip-unchanged option to migration script ([563f39f](https://github.com/pscedu/cpra.mp.data/commit/563f39fd7861672918ad36dc126bbfa99bd24248))
* **scripts:** add opt4 breakpoints ([381021c](https://github.com/pscedu/cpra.mp.data/commit/381021cef5756e3a29985c45bf366d1fe32a65ba))
* **scripts:** add script to calculate land confidence ([714431a](https://github.com/pscedu/cpra.mp.data/commit/714431a90d27e337e93b70fd8043a2320df3ff46))
* **scripts:** add script to generate land type plus PMTiles ([f084cad](https://github.com/pscedu/cpra.mp.data/commit/f084cad9cbce8c1e9e4d5204d57f1aec7e609bc5))
* **scripts:** add script to remove migrated data containing a single ([9108672](https://github.com/pscedu/cpra.mp.data/commit/91086722166527682c52054e0087500631ef4701))
* **scripts:** add script to repair excess rows in migrated data ([3997c83](https://github.com/pscedu/cpra.mp.data/commit/3997c83b30123e52b10f1012118ba595abeb70c6))
* **scripts:** consolidate historic image groupings ([ecafe1c](https://github.com/pscedu/cpra.mp.data/commit/ecafe1cfe021a8f241cc3eaaa25edbc4563fcd49))
* **scripts:** get rid of separate 2004-2006 video image ([f210300](https://github.com/pscedu/cpra.mp.data/commit/f2103008c6a8d28fad88745ffe69d7b3abc7f7c3))
* **scripts:** update confidence change colors ([420100c](https://github.com/pscedu/cpra.mp.data/commit/420100c3de4a363494493d046466ce359928c7c9))
* **scripts:** update historic loss/gain colors ([1f2d8a8](https://github.com/pscedu/cpra.mp.data/commit/1f2d8a84bb6bb9b75a75c319c5909129179e0220))
* **scripts:** update historic loss/gain colors ([88dc94f](https://github.com/pscedu/cpra.mp.data/commit/88dc94fe20b539423379641df2af826ba6cae0ac))
* **scripts:** update video frames to use real confidence data ([ebf63db](https://github.com/pscedu/cpra.mp.data/commit/ebf63db97aed20a9f7c36015ee1b33ae527977d2))
* **scripts:** use bicubic resampling on RGB images ([0cc721b](https://github.com/pscedu/cpra.mp.data/commit/0cc721b87b068caeb284e5fd134d07ec5de10c5a))
* **structure:** add ICM derived variables ([1b690f3](https://github.com/pscedu/cpra.mp.data/commit/1b690f35e5f52e6dd565420fc8fd1dd9179c4a9b))
* **structure:** add land confidence legend ([65959bf](https://github.com/pscedu/cpra.mp.data/commit/65959bf6f9c549efdf917f821e40f931955c550d))
* **structure:** add lnd_type_plus legend ([dd8c8b2](https://github.com/pscedu/cpra.mp.data/commit/dd8c8b28a530722cc3577ddafb9a11573d33426b))
* **structure:** sort legends and update ffibs legend ([25c738a](https://github.com/pscedu/cpra.mp.data/commit/25c738aebf3f9c3ececdb8818d3354bf8fa46eb8))
* **variables:** add NoData value ([e653f88](https://github.com/pscedu/cpra.mp.data/commit/e653f8893df9d2718642500d57388724502f9a4d))
* **visualizations:** update glyph and sprite URLs ([9ea933d](https://github.com/pscedu/cpra.mp.data/commit/9ea933d939064c424d60dd5f48f9cc70e362052b))


### Bug Fixes

* **analyses:** fix output type for difference and concatenate ([1711e99](https://github.com/pscedu/cpra.mp.data/commit/1711e99e144cdbfbf866ca91c7a845e43bb73224))
* **analyses:** make land type plus logic consistent with source notebook ([47830e3](https://github.com/pscedu/cpra.mp.data/commit/47830e3636e0441baeb721f8e29e46719d7d319b))
* **analysis:** fix handling of raster pixel IDs ([57c210f](https://github.com/pscedu/cpra.mp.data/commit/57c210f0a49b84e246528ce2712661d071e026f1))
* **analysis:** fix lnd_chng difference map and download columns ([60bc442](https://github.com/pscedu/cpra.mp.data/commit/60bc4427dc239ab360b8a325bd114df041d69f9b))
* **analysis:** fix post-migration issues and update docs ([592fa11](https://github.com/pscedu/cpra.mp.data/commit/592fa1149772750324eec25977bc69cb510cda21))
* **analysis:** fix raster pixel labeling ([e20b58b](https://github.com/pscedu/cpra.mp.data/commit/e20b58bb419ce05a090007a973378e6e7706d7f5))
* **analysis:** handle missing tiles ([318d417](https://github.com/pscedu/cpra.mp.data/commit/318d4174cd07beb74d0c0533f00526299855ec16))
* **analysis:** make grid_cell an output filter (partial change) ([5cabb8d](https://github.com/pscedu/cpra.mp.data/commit/5cabb8d4a6c80043761c0e2ac22c4fe7f79feef1))
* **analysis:** remove internal analysis columns from output ([1a3e585](https://github.com/pscedu/cpra.mp.data/commit/1a3e5853e68c0f48a6f4f778bb82baef3a5a699f))
* **analysis:** set baseline year for model group 0 ([48955f9](https://github.com/pscedu/cpra.mp.data/commit/48955f9eb1f9bf6575149c45c0f1c7863b6e85e6))
* **api:** restrict sediment dynamics to related variables ([3e6a46a](https://github.com/pscedu/cpra.mp.data/commit/3e6a46a5d4a9e756bece195d1a2251ff9a1c5236))
* **catalog:** handle catalog schema drift ([d006968](https://github.com/pscedu/cpra.mp.data/commit/d0069681871e19720d92ce66cb85e395a018e058))
* **scripts:** fix OOM error in check ([d1809d8](https://github.com/pscedu/cpra.mp.data/commit/d1809d8f6a4bbb93716401be76b721d42c482b2b))
* **scripts:** remove land/water offset and update color ramps ([87a7765](https://github.com/pscedu/cpra.mp.data/commit/87a776535c02c2d636d88625fb9d444b8318d840))
* **scripts:** use 2029 as base year for confidence calculations ([3c1ce50](https://github.com/pscedu/cpra.mp.data/commit/3c1ce502594688ed15f14ee3be00b5fef636c616))
* **structure:** remove hdi and icm_wva variables ([19c3eb1](https://github.com/pscedu/cpra.mp.data/commit/19c3eb13a87d9b0691237349781aa318b878cd62))
* **structure:** remove lnd_conf and lnd_type_plus variables ([866a2d9](https://github.com/pscedu/cpra.mp.data/commit/866a2d954553e2c75b29d96325c52fc510718f39))
* **structure:** remove lnd_conf_chng variable and land change confidence script ([d19cffe](https://github.com/pscedu/cpra.mp.data/commit/d19cffe9571a16d7c07f12384a3d204177773ae2))
* **structure:** set -9999000 as NoData value for millimeter elevation ([a06e9b2](https://github.com/pscedu/cpra.mp.data/commit/a06e9b25652aac449524ef1d8da7cfc148e8c13f))
* **variables:** warn on unknown legends ([6465c53](https://github.com/pscedu/cpra.mp.data/commit/6465c5361a07902a77007fd8c142c6508d20c956))
* **visualizations:** fix time series chart for lnd_chng and ([c20c4c9](https://github.com/pscedu/cpra.mp.data/commit/c20c4c9c0d8559d7787d8886faa698271810b104))


### Performance Improvements

* **analysis:** fix tile memory regression ([18cdb99](https://github.com/pscedu/cpra.mp.data/commit/18cdb99983963d92e1e172b81d59e277dfd7b292))
* **analysis:** group raster sampling by COG tile ([3353094](https://github.com/pscedu/cpra.mp.data/commit/3353094b3643017947ed5fea241b4047cc4ad893))
* **analysis:** implement pushdown ([6be5b48](https://github.com/pscedu/cpra.mp.data/commit/6be5b482af743aafb34f21c97c973e58a66d36ee))
* **analysis:** set GDAL cache size in MB ([7f121fd](https://github.com/pscedu/cpra.mp.data/commit/7f121fd35e740688bedbc9f99b7851ae49478c7d))
* **api:** add fold optimization for tabular data ([26b1da8](https://github.com/pscedu/cpra.mp.data/commit/26b1da8ec1d61c786d5703116910bd72d7d6569e))
* **api:** fix busy-polling of temp file during download ([613143c](https://github.com/pscedu/cpra.mp.data/commit/613143c73977a2de26009cb47a3486a21455b99d))
* **api:** improve performance of raster transform params ([9bbe643](https://github.com/pscedu/cpra.mp.data/commit/9bbe6437d081acfe96cd9c6482183930f63da88a))
* **api:** optimize null guard ([7d2b2c7](https://github.com/pscedu/cpra.mp.data/commit/7d2b2c7aaed87450133e239fe66a31c7d1726c38))
* **raster:** improve VRT build performance ([389dc0c](https://github.com/pscedu/cpra.mp.data/commit/389dc0c6f5afb0a480f663640bc56061c8c0676e))
* **raster:** optimize _parallel_read_cog_metadata ([d8f85fa](https://github.com/pscedu/cpra.mp.data/commit/d8f85facf767997404b9b22e761e54f59ce0bea0))


### Documentation

* **analysis:** clarify group_by behavior ([ba57d20](https://github.com/pscedu/cpra.mp.data/commit/ba57d204f61ba1472b625a9870c93a9661cf3528))
* **analysis:** document analyze_data function ([9816802](https://github.com/pscedu/cpra.mp.data/commit/9816802fb52942fc9c1e320498fecee7b95c4771))
* **analysis:** update stale docstrings ([988e5ec](https://github.com/pscedu/cpra.mp.data/commit/988e5ecc83aadb1b2d55f478291bf5ad89116b71))
* **catalog-api:** update grid cell parameter name ([f6fcd4f](https://github.com/pscedu/cpra.mp.data/commit/f6fcd4f32c9ef3f03099c0b29115211c52b57dad))

## [0.51.0](https://github.com/pscedu/cpra.mp.data/compare/v0.50.0...v0.51.0) (2026-08-25)


### Features

* **structure:** add display format columns ([0ed2fe8](https://github.com/pscedu/cpra.mp.data/commit/0ed2fe858dc1c09127afb81a26c2806101cf12db))
* **structure:** add variable float precision for CSVs ([b911837](https://github.com/pscedu/cpra.mp.data/commit/b9118373390b1b0a3df97bb38930e3f5e283d5c7))
* **visualizations:** add species stacked bar chart ([d9f8bb0](https://github.com/pscedu/cpra.mp.data/commit/d9f8bb040c051247930aa558ecafb8a2aaf20b39))

## [0.50.0](https://github.com/pscedu/cpra.mp.data/compare/v0.49.1...v0.50.0) (2026-08-20)


### Features

* **cli:** generate area-weighted vector-to-vector crosswalks ([a86c1f7](https://github.com/pscedu/cpra.mp.data/commit/a86c1f79cf50bd3e408a3312c3d50dce66694c68))
* **cli:** migrate crosswalk generation scripts to CLI ([55130af](https://github.com/pscedu/cpra.mp.data/commit/55130aff6642f99594d8c287d311c4f07500aeb1))

## [0.49.1](https://github.com/pscedu/cpra.mp.data/compare/v0.49.0...v0.49.1) (2026-08-20)


### Bug Fixes

* **config:** use consistent crosswalk and grid paths ([182bca8](https://github.com/pscedu/cpra.mp.data/commit/182bca87fee98c7c4cdc963c200be416357ff978))

## [0.49.0](https://github.com/pscedu/cpra.mp.data/compare/v0.48.0...v0.49.0) (2026-08-19)


### Features

* **api:** accept coordinate selections for vector grids ([cf9e08e](https://github.com/pscedu/cpra.mp.data/commit/cf9e08e3305879835d5000512406625ea51ac2bd))
* **api:** add coordinate crosswalk to vector grids ([493da16](https://github.com/pscedu/cpra.mp.data/commit/493da160b98e4c44ec493b3523a2902b385cb1ea))
* **api:** add crosswalk route ([0dd0860](https://github.com/pscedu/cpra.mp.data/commit/0dd08605e0f1e73c792091df365defe8c0d1c1de))
* **scripts:** update colors for historical video frames ([7868f14](https://github.com/pscedu/cpra.mp.data/commit/7868f14b14e1ab6fadc6ca77e626861215155411))
* **scripts:** update video frames to be transparent overlays ([588a8bf](https://github.com/pscedu/cpra.mp.data/commit/588a8bf70face8026da7df5515a0b8df3148b82e))
* **visualizations:** improve chart sizing ([ec4a32c](https://github.com/pscedu/cpra.mp.data/commit/ec4a32c5bb8ec9e0c300ded23c581034a6c096a9))


### Bug Fixes

* **api:** report data for vector grids by cell ID ([e17e94e](https://github.com/pscedu/cpra.mp.data/commit/e17e94e20f387b3c037bc479ba6d8f51415c7a3a))

## [0.48.0](https://github.com/pscedu/cpra.mp.data/compare/v0.47.1...v0.48.0) (2026-08-05)


### Features

* **api:** implement transform params for raster grids ([9a99deb](https://github.com/pscedu/cpra.mp.data/commit/9a99debfd2264302a01049652b85159ac498387b))
* **scripts:** add script for generating map video frames ([c5ba74d](https://github.com/pscedu/cpra.mp.data/commit/c5ba74d0acf55a9c761b423089e6b2d9eb777c91))
* **scripts:** add script for generating raster to vector crosswalks ([2d4568a](https://github.com/pscedu/cpra.mp.data/commit/2d4568add9cb19fee9cf4567d2a62390c751a607))
* **scripts:** add script to symbolize and tile land type with water ([a5b5d7f](https://github.com/pscedu/cpra.mp.data/commit/a5b5d7f133a9ad9c8ce1cfe28d1a60be26875056))
* **scripts:** add scripts for test land change confidence ([60372d0](https://github.com/pscedu/cpra.mp.data/commit/60372d02aa377783538b39fc5ee7849251916d5e))
* **visualization:** add stacked bar chart ([c2d2fea](https://github.com/pscedu/cpra.mp.data/commit/c2d2feaa9efc66eb2232e14b5e8a9397a9ef92f4))
* **visualizations:** add portal ID to map project labels ([ed578e6](https://github.com/pscedu/cpra.mp.data/commit/ed578e6fc30f93c62f90d1ceab70484d262efd55))


### Bug Fixes

* **api:** respect compression level in raster streaming responses ([82b8913](https://github.com/pscedu/cpra.mp.data/commit/82b8913bf860ba0c20b60bbd006f1b79f08a9862))
* correct path, compaction, and query defects surfaced by the new ([5c79856](https://github.com/pscedu/cpra.mp.data/commit/5c79856faf2f296955c584d4d53aae2cc805d1ba))
* **scripts:** update overlay and underlay layer names ([1cd4444](https://github.com/pscedu/cpra.mp.data/commit/1cd44449b42a54ca4a42014e49a32c5a14262396))

## [0.47.1](https://github.com/pscedu/cpra.mp.data/compare/v0.47.0...v0.47.1) (2026-07-14)


### Bug Fixes

* **api:** fix pixel shift in raster tile ([3f193fb](https://github.com/pscedu/cpra.mp.data/commit/3f193fbbb46afb744276c376aa4482c8c494a451))

## [0.47.0](https://github.com/pscedu/cpra.mp.data/compare/v0.46.0...v0.47.0) (2026-07-09)


### Features

* **structure:** update coverage legend ([a841339](https://github.com/pscedu/cpra.mp.data/commit/a8413398e7db6d731c9b69fc8e5cbfdd792a7bab))


### Bug Fixes

* **visualizations:** update default date range in time series line ([4eab887](https://github.com/pscedu/cpra.mp.data/commit/4eab88765a7638f442fc3beeed274bd913d8dfc8))

## [0.46.0](https://github.com/pscedu/cpra.mp.data/compare/v0.45.0...v0.46.0) (2026-06-16)


### Features

* **bridges2:** pin base image to a specific commit ([74b1d83](https://github.com/pscedu/cpra.mp.data/commit/74b1d836045a1633104fa1a0484022a414fd339d))
* **map:** allow different/multiple project model groups ([0c1c262](https://github.com/pscedu/cpra.mp.data/commit/0c1c262ddee4d660d330adce637ee18c3f4d06e5))
* **scripts:** add script to colorize land type confidence ([a50a5d7](https://github.com/pscedu/cpra.mp.data/commit/a50a5d7470cade339c73447459eefb7fa993158a))
* **scripts:** add script to convert historic land change ([28e1d19](https://github.com/pscedu/cpra.mp.data/commit/28e1d19124b52e82aecd4a4efa0c1d532f7b4ea4))
* **scripts:** add script to visualize historic land change ([c0c9ce9](https://github.com/pscedu/cpra.mp.data/commit/c0c9ce92e6f2a0ab4e10b0d08c703d83c6d1f889))
* **scripts:** generate test land type confidence data ([03ea6e1](https://github.com/pscedu/cpra.mp.data/commit/03ea6e19b9bfdb6fad7b8bdc73b3f151af15d755))
* **scripts:** use windowed read/write for historic land type ([f7973fc](https://github.com/pscedu/cpra.mp.data/commit/f7973fc27eb53c1a9a54f7b045f81d5e697cebc7))


### Bug Fixes

* **scripts:** use all model groups and scenarios for future data ([7a4c237](https://github.com/pscedu/cpra.mp.data/commit/7a4c23791b757df0a519b16e2424a5748a356464))
* **structure:** allow calendar_year &gt;= 1900 for historical data ([84fc865](https://github.com/pscedu/cpra.mp.data/commit/84fc865fb82c38d811b12f206a3fd1b6e593f2ba))

## [0.45.0](https://github.com/pscedu/cpra.mp.data/compare/v0.44.2...v0.45.0) (2026-06-08)


### Features

* **raster:** use predictor option for writing COGs ([9d06ca6](https://github.com/pscedu/cpra.mp.data/commit/9d06ca6dd68ab07cbd5b426dceec4041c5811802))

## [0.44.2](https://github.com/pscedu/cpra.mp.data/compare/v0.44.1...v0.44.2) (2026-06-04)


### Bug Fixes

* **api:** fix AEP precision issues ([8817d4b](https://github.com/pscedu/cpra.mp.data/commit/8817d4b6d4da31cc978a8fbbe66a63a02182a627))

## [0.44.1](https://github.com/pscedu/cpra.mp.data/compare/v0.44.0...v0.44.1) (2026-06-04)


### Bug Fixes

* add missing pmtiles dependency ([c52cc04](https://github.com/pscedu/cpra.mp.data/commit/c52cc041723101b1019d8b56e10d69f107e6959f))
* **config:** set default tiles path ([e2986dd](https://github.com/pscedu/cpra.mp.data/commit/e2986ddc7cb979cd8e6ed9d8836a5572dd5e98c0))

## [0.44.0](https://github.com/pscedu/cpra.mp.data/compare/v0.43.0...v0.44.0) (2026-06-04)


### Features

* **api:** add MVT generation in the data API ([#67](https://github.com/pscedu/cpra.mp.data/issues/67)) ([ec9762d](https://github.com/pscedu/cpra.mp.data/commit/ec9762d36c6f6932994c26c12ef06df0868231dc))

## [0.43.0](https://github.com/pscedu/cpra.mp.data/compare/v0.42.0...v0.43.0) (2026-06-04)


### Features

* **map:** include number format in legend JSON ([b6c1c6f](https://github.com/pscedu/cpra.mp.data/commit/b6c1c6f5a64aff2abe8e17299a9475325bbebaf9))
* **structure:** set number format and difference legend for CLARA ([ded4ef0](https://github.com/pscedu/cpra.mp.data/commit/ded4ef082e02092b9d0e03961e1293a85493f731))


### Bug Fixes

* **time-series-line:** fix NaN in tooltip for difference charts ([28e4128](https://github.com/pscedu/cpra.mp.data/commit/28e412835627da09a3b9458111a2be1e2688d99b))
* **time-series-line:** fix NaN in tooltip for zero dimension value ([0f8dba5](https://github.com/pscedu/cpra.mp.data/commit/0f8dba58d3f983ddb66bd8be01fc4d49e56a80c3))


### Performance Improvements

* **catalog:** speed up catalog queries ([c1ef43e](https://github.com/pscedu/cpra.mp.data/commit/c1ef43e0510b483eab61fd7004f13511c51413e3))
* **map:** pre-compute colors server-side ([b8ce0ee](https://github.com/pscedu/cpra.mp.data/commit/b8ce0ee5c7a141c5a93a1670ce7c7358af938f6c))

## [0.42.0](https://github.com/pscedu/cpra.mp.data/compare/v0.41.0...v0.42.0) (2026-05-28)


### Features

* **structure:** add icm_v26.0.0 as a model value ([eb19f20](https://github.com/pscedu/cpra.mp.data/commit/eb19f20937682f7b71e2bc2395e0495e672a6098))

## [0.41.0](https://github.com/pscedu/cpra.mp.data/compare/v0.40.1...v0.41.0) (2026-05-28)


### Features

* **image:** add sympy to Bridges-2 image ([b5c8a81](https://github.com/pscedu/cpra.mp.data/commit/b5c8a818087fa54352d0c50dc29d3927d23a80bf))

## [0.40.1](https://github.com/pscedu/cpra.mp.data/compare/v0.40.0...v0.40.1) (2026-05-27)


### Bug Fixes

* **structure:** add no data levels to coverage and hsi legends ([95a1035](https://github.com/pscedu/cpra.mp.data/commit/95a103582cda4287a5447a6a3b45aa28bf92ad70))


### Documentation

* **structure:** update legend docs ([0038844](https://github.com/pscedu/cpra.mp.data/commit/0038844a9b5e5cbcccaa8277a891d55a132fae6c))

## [0.40.0](https://github.com/pscedu/cpra.mp.data/compare/v0.39.0...v0.40.0) (2026-05-27)


### Features

* **structure:** add legends for coverage and trg ([3cbb915](https://github.com/pscedu/cpra.mp.data/commit/3cbb9152c3d8eb389278c93596f5f2fe0448d414))


### Bug Fixes

* **structure:** reverse colors for salinity ([6d333d8](https://github.com/pscedu/cpra.mp.data/commit/6d333d84889bda8da73de33d1bce14732ab4f0bf))

## [0.39.0](https://github.com/pscedu/cpra.mp.data/compare/v0.38.0...v0.39.0) (2026-05-01)


### Features

* **bridges2:** add anywidget and traitlets packages ([dce2897](https://github.com/pscedu/cpra.mp.data/commit/dce2897074cd2f43e5b7424ea588feae5431181a))


### Bug Fixes

* **structure:** remove lnd_chng variable ([45c4445](https://github.com/pscedu/cpra.mp.data/commit/45c4445a40c01d21a9b8c022abdf53241de6da29))

## [0.38.0](https://github.com/pscedu/cpra.mp.data/compare/v0.37.0...v0.38.0) (2026-04-30)


### Features

* add include_file_paths option to read_data ([67cfe33](https://github.com/pscedu/cpra.mp.data/commit/67cfe33e2f587ac4a089d5fe2e27c1650a8efde0))
* **structure:** add MP29 grids ([94928e3](https://github.com/pscedu/cpra.mp.data/commit/94928e325a76596d61848d3ce59a1e87ccdaf386))

## [0.37.0](https://github.com/pscedu/cpra.mp.data/compare/v0.36.4...v0.37.0) (2026-03-26)


### Features

* **structure:** add region and ecoregion dimensions ([c2c0c86](https://github.com/pscedu/cpra.mp.data/commit/c2c0c8690c6d870b88cdd97dd02db2ae56b4f4e8))


### Bug Fixes

* **structure:** add damage legend and rename asset count legend ([8887b1d](https://github.com/pscedu/cpra.mp.data/commit/8887b1d529ff0099493aa58f1ed151a2656b0110))
* **structure:** use grid labels ([7f6d5af](https://github.com/pscedu/cpra.mp.data/commit/7f6d5afd25541ceeb274d9a14baebaf71b5c3832))

## [0.36.4](https://github.com/pscedu/cpra.mp.data/compare/v0.36.3...v0.36.4) (2026-03-18)


### Bug Fixes

* **structure:** update HSI legend ([2e7aa64](https://github.com/pscedu/cpra.mp.data/commit/2e7aa64d279142d1fbe8fb23685e17fba99d638e))

## [0.36.3](https://github.com/pscedu/cpra.mp.data/compare/v0.36.2...v0.36.3) (2026-03-18)


### Bug Fixes

* **structure:** update legends for CLARA and HSI ([acd6cff](https://github.com/pscedu/cpra.mp.data/commit/acd6cff36a5ace6ca688f7459e64ae3cf4802379))


### Documentation

* **structure:** update data structure docs ([99a1c08](https://github.com/pscedu/cpra.mp.data/commit/99a1c08ea8bb92909baa6e8ff055d19f878d4e0d))

## [0.36.2](https://github.com/pscedu/cpra.mp.data/compare/v0.36.1...v0.36.2) (2026-03-02)


### Bug Fixes

* **visualizations:** use initial conditions water for water overlay ([bedf16d](https://github.com/pscedu/cpra.mp.data/commit/bedf16dba688599d240763fd9309d413574b48fb))
* **visualizations:** use unit as map legend subtitle ([6389a88](https://github.com/pscedu/cpra.mp.data/commit/6389a88efb95a80d5331e455315dc9a6bcdde360))

## [0.36.1](https://github.com/pscedu/cpra.mp.data/compare/v0.36.0...v0.36.1) (2026-02-18)


### Bug Fixes

* temporarily pin Polars to a working version ([30216b3](https://github.com/pscedu/cpra.mp.data/commit/30216b3a9c46fe808b38056d09e032a38a78eaba))

## [0.36.0](https://github.com/pscedu/cpra.mp.data/compare/v0.35.0...v0.36.0) (2026-02-12)


### Features

* **visualizations:** add map visualization ([#51](https://github.com/pscedu/cpra.mp.data/issues/51)) ([c2a9c40](https://github.com/pscedu/cpra.mp.data/commit/c2a9c404725c3851f430f43aeaefbe9fd4e32106))

## [0.35.0](https://github.com/pscedu/cpra.mp.data/compare/v0.34.0...v0.35.0) (2026-01-23)


### Features

* **bridges2:** add connectorx dependency ([d6a4353](https://github.com/pscedu/cpra.mp.data/commit/d6a43531dddd5679eec7318b449245df8e2e4914))

## [0.34.0](https://github.com/pscedu/cpra.mp.data/compare/v0.33.0...v0.34.0) (2025-12-03)


### Features

* add geospatial packages to Bridges-2 image ([2c394bf](https://github.com/pscedu/cpra.mp.data/commit/2c394bf4cd8d5c27ce828cbe8b797664b7b9e79a))

## [0.33.0](https://github.com/pscedu/cpra.mp.data/compare/v0.32.1...v0.33.0) (2025-11-25)


### Features

* **structure:** convert raster legends from DAP ([39428c1](https://github.com/pscedu/cpra.mp.data/commit/39428c1b1dfa38892ef5f7eaba2d8a3b763a744c))
* **visualizations:** support multiple values per dimension in time ([32ec14c](https://github.com/pscedu/cpra.mp.data/commit/32ec14cca28bec0961876e35a70d7d542d4b97cb))

## [0.32.1](https://github.com/pscedu/cpra.mp.data/compare/v0.32.0...v0.32.1) (2025-11-06)


### Bug Fixes

* **visualizations:** handle missing negate param ([96f6d90](https://github.com/pscedu/cpra.mp.data/commit/96f6d903e4c5fef79d2224d0c7c764f0ec8c81c3))
* **visualizations:** use same model group and scenario for accretion ([0198772](https://github.com/pscedu/cpra.mp.data/commit/0198772c2e6cb7715cda813227185087a76cf735))

## [0.32.0](https://github.com/pscedu/cpra.mp.data/compare/v0.31.0...v0.32.0) (2025-10-29)


### Features

* **structure:** update data structure ([8e04816](https://github.com/pscedu/cpra.mp.data/commit/8e04816c82f4e96b94be83e40de10471c090f708))
* **visualizations:** add accretion to sediment dynamics chart ([62c7280](https://github.com/pscedu/cpra.mp.data/commit/62c728091a82d4d7cc699f186e888a3d282c546c))


### Bug Fixes

* **api:** change color/value order for discrete color maps ([eee8c00](https://github.com/pscedu/cpra.mp.data/commit/eee8c009952fee0f50a120daffd2977995e92162))


### Performance Improvements

* **api:** conditionally collect data frames in memory based on expected ([45c1dc7](https://github.com/pscedu/cpra.mp.data/commit/45c1dc761b16c1d40c510991bc7eff97195af455))

## [0.31.0](https://github.com/pscedu/cpra.mp.data/compare/v0.30.1...v0.31.0) (2025-10-15)


### Features

* **visualizations:** add difference label to time series line chart ([db66c82](https://github.com/pscedu/cpra.mp.data/commit/db66c82a2d4f36223e28ff8644062eb91767f9fe))
* **visualizations:** add zero rule to difference charts ([25b516a](https://github.com/pscedu/cpra.mp.data/commit/25b516ac2f2fc5f7edc1674d2fd73f65fea1a12d))


### Bug Fixes

* **api:** set all dimension columns to null if they are not part of the ([6c65e58](https://github.com/pscedu/cpra.mp.data/commit/6c65e587445b783d92597f638853f64b69910658))
* **api:** set non-numeric columns to null during group by ([caf62ee](https://github.com/pscedu/cpra.mp.data/commit/caf62ee2870f89ba391f0570c5fce682ecd2241c))
* **api:** treat columns starting with a negative number as value columns ([5d66b34](https://github.com/pscedu/cpra.mp.data/commit/5d66b3447185a345f169fbee56690f86a48b89ab))
* **api:** use uv lock file in building image ([2be11bf](https://github.com/pscedu/cpra.mp.data/commit/2be11bfd6d252f30334435009c7e9f84788bdbc3))

## [0.30.1](https://github.com/pscedu/cpra.mp.data/compare/v0.30.0...v0.30.1) (2025-10-13)


### Bug Fixes

* install package from source in Bridges-2 image ([f97a07a](https://github.com/pscedu/cpra.mp.data/commit/f97a07a44ced2e5da86b2388ed423279c0a27394))

## [0.30.0](https://github.com/pscedu/cpra.mp.data/compare/v0.29.1...v0.30.0) (2025-10-13)


### Features

* build unified Apptainer image ([4605fac](https://github.com/pscedu/cpra.mp.data/commit/4605fac277f6955d8c70fc109c49ef09aaea454f))
* **legends:** add support for continuous color ramps ([cf7df8e](https://github.com/pscedu/cpra.mp.data/commit/cf7df8e78a36b6f251cf213280dfd65db9557248))

## [0.29.1](https://github.com/pscedu/cpra.mp.data/compare/v0.29.0...v0.29.1) (2025-10-02)


### Bug Fixes

* **image:** copy module code before installing dependencies ([ec69874](https://github.com/pscedu/cpra.mp.data/commit/ec69874ad1996e5c888bc4b60e028366bab2fe02))

## [0.29.0](https://github.com/pscedu/cpra.mp.data/compare/v0.28.0...v0.29.0) (2025-10-02)


### Features

* **structure:** rename veg_pixel_v001 to veg_grid_cell_v001 ([dc7dd1c](https://github.com/pscedu/cpra.mp.data/commit/dc7dd1c962e82105b10498a29099ed3f3ed279d0))
* **visualizations:** add start and end date parameters ([c1e18c9](https://github.com/pscedu/cpra.mp.data/commit/c1e18c998ebdedefc237107683b6da7fb72919c3))


### Bug Fixes

* **tabular:** allow multiple values per dimension ([23fe6c8](https://github.com/pscedu/cpra.mp.data/commit/23fe6c87f615bab678d62c5a7780ce2f468ae75e))

## [0.28.0](https://github.com/pscedu/cpra.mp.data/compare/v0.27.0...v0.28.0) (2025-09-11)


### Features

* **structure:** update morph grid ([ae347d6](https://github.com/pscedu/cpra.mp.data/commit/ae347d690940bee7bcdeb124af40b413d1bafa8e))
* **variables:** add number format ([5368380](https://github.com/pscedu/cpra.mp.data/commit/53683807a3bcf4812e740a11d89ff73d7cf1fc16))
* **visualizations:** allow specifying geographies as a list ([1620224](https://github.com/pscedu/cpra.mp.data/commit/162022475814290a9e0022fd560036c947318737))
* **visualizations:** use variable number format, if specified ([f123d02](https://github.com/pscedu/cpra.mp.data/commit/f123d02df6de34d315810c1ca43f2916f66be0f9))


### Bug Fixes

* **visualizations:** ensure tooltip title is a string ([2d57df5](https://github.com/pscedu/cpra.mp.data/commit/2d57df5053aac887584a70af6727687802ed73bb))
* **visualizations:** fix tooltips for raster coordinates ([ba3d531](https://github.com/pscedu/cpra.mp.data/commit/ba3d531985aa9f95d6f1ef78ebb9b6a1997e9864))
* **visualizations:** parse dates in local time ([3c3c934](https://github.com/pscedu/cpra.mp.data/commit/3c3c934210d5c81b48ecc7e2bc7809f5eba91bc4))

## [0.27.0](https://github.com/pscedu/cpra.mp.data/compare/v0.26.0...v0.27.0) (2025-09-09)


### Features

* **structure:** update data structure ([47c43e6](https://github.com/pscedu/cpra.mp.data/commit/47c43e6a0bd58a304e81be71a4007cd80fe2776e))

## [0.26.0](https://github.com/pscedu/cpra.mp.data/compare/v0.25.1...v0.26.0) (2025-08-29)


### Features

* **api:** add endpoint for visualization metadata ([e1490a1](https://github.com/pscedu/cpra.mp.data/commit/e1490a186578686be1c4b5756e43b19b58c7ff9f))
* **visualization:** add sediment dynamics chart ([159d0ec](https://github.com/pscedu/cpra.mp.data/commit/159d0ecc532b5966bc10b307f6eaa9ea940318ec))
* **visualizations:** add support for raster grid coordinates ([5618716](https://github.com/pscedu/cpra.mp.data/commit/5618716cf7509f4b1f0686d42bc8175b712f54c1))
* **visualizations:** set sedment dynamics chart dimensions ([00c2364](https://github.com/pscedu/cpra.mp.data/commit/00c23647812f4f1918e8defeef3d3b4a7a01be64))


### Bug Fixes

* **visualization:** merge params in sediment dynamics ([695ae07](https://github.com/pscedu/cpra.mp.data/commit/695ae07dce7ff335c5639cd252ec8a5243af1956))

## [0.25.1](https://github.com/pscedu/cpra.mp.data/compare/v0.25.0...v0.25.1) (2025-08-18)


### Bug Fixes

* **raster:** include dataset dimensions in LazyFrame ([5db0f3e](https://github.com/pscedu/cpra.mp.data/commit/5db0f3e2385b3bba941e3fc173f3cc28da859025))

## [0.25.0](https://github.com/pscedu/cpra.mp.data/compare/v0.24.0...v0.25.0) (2025-08-15)


### Features

* **structure:** add test accretion variables ([2561d53](https://github.com/pscedu/cpra.mp.data/commit/2561d5335058218415c5a731a80ac28fcdad70bf))

## [0.24.0](https://github.com/pscedu/cpra.mp.data/compare/v0.23.1...v0.24.0) (2025-08-05)


### Features

* **api:** get extra dimension values from MPD ([67d879c](https://github.com/pscedu/cpra.mp.data/commit/67d879c1e56fdc3755921dbbea04e30163a4ce8d))
* **structure:** update vegetation dimensions ([1adb224](https://github.com/pscedu/cpra.mp.data/commit/1adb224b549b4dcd95cd5fa5e4c6177b0d195498))
* **visualizations:** fix extent for daily chart ([fec92ec](https://github.com/pscedu/cpra.mp.data/commit/fec92ec9dc8ee37b4d205bd047ac69dcecae1fba))
* **visualizations:** set visualization size in query params ([7af6702](https://github.com/pscedu/cpra.mp.data/commit/7af6702ead1497722a81cec0707aa2fdb254a114))
* **visualizations:** use grid label in legend ([b613f93](https://github.com/pscedu/cpra.mp.data/commit/b613f93f5bc33b9466fecb3b91267a8383ba67c6))
* **visualizations:** use selection colors for time series line ([60cf649](https://github.com/pscedu/cpra.mp.data/commit/60cf6499b0bf2948d1621504afcf4197eab8d14e))
* **visualizations:** use time dimension label in tooltip ([1b4fb7e](https://github.com/pscedu/cpra.mp.data/commit/1b4fb7e5619b391c26e041dc43006d02eddf2f77))


### Bug Fixes

* **visualizations:** cache list of geographies instead of map object ([d95bb52](https://github.com/pscedu/cpra.mp.data/commit/d95bb5227f95eea75b65b20b7d3f46ecdf3c5057))
* **visualizations:** convert geography to a number ([5eededa](https://github.com/pscedu/cpra.mp.data/commit/5eededae5d0f454e55e2552265794793106fab1e))
* **visualizations:** set min chart height and width ([f14b2d3](https://github.com/pscedu/cpra.mp.data/commit/f14b2d34456cbe0934747ea1d718ab1ebb8412b8))

## [0.23.1](https://github.com/pscedu/cpra.mp.data/compare/v0.23.0...v0.23.1) (2025-07-08)


### Bug Fixes

* **utils:** pass sort order as kwargs ([a7aca09](https://github.com/pscedu/cpra.mp.data/commit/a7aca090047a372e3db14dd4eb03cbf42416da47))

## [0.23.0](https://github.com/pscedu/cpra.mp.data/compare/v0.22.1...v0.23.0) (2025-07-08)


### Features

* **scripts:** promote ID field in tiles ([7ca3aca](https://github.com/pscedu/cpra.mp.data/commit/7ca3aca7ff8129dde833d46522a80eea8c8e45f4))
* **structure:** update data structure ([7045031](https://github.com/pscedu/cpra.mp.data/commit/704503146294a701d0ebb86a389dd852af91cab6))

## [0.22.1](https://github.com/pscedu/cpra.mp.data/compare/v0.22.0...v0.22.1) (2025-06-05)


### Bug Fixes

* **jupyter:** install package from wheel ([c8aa1af](https://github.com/pscedu/cpra.mp.data/commit/c8aa1afd71d1e9c6b0c44330b3686817311b5424))

## [0.22.0](https://github.com/pscedu/cpra.mp.data/compare/v0.21.2...v0.22.0) (2025-06-05)


### Features

* **image:** add Jupyter image ([ca50567](https://github.com/pscedu/cpra.mp.data/commit/ca50567aab336f2ae395bba1d7ac27df631939e4))
* **structure:** sort variables during import ([eea27e2](https://github.com/pscedu/cpra.mp.data/commit/eea27e2560bcea64c924a38d47bb2d45b6530b07))


### Bug Fixes

* **dimensions:** use consistent column names for dimension values ([77d8a92](https://github.com/pscedu/cpra.mp.data/commit/77d8a92467c18215be5c338fdbf079a54053c327))
* **structure:** update FFIBS legend ([4b36ec4](https://github.com/pscedu/cpra.mp.data/commit/4b36ec4199d9135de9a2a78b36bbf22fcae27944))


### Documentation

* **structure:** reorganize data structure docs ([c378e64](https://github.com/pscedu/cpra.mp.data/commit/c378e64ae13d17f58c055810b607b726d886bbd8))

## [0.21.2](https://github.com/pscedu/cpra.mp.data/compare/v0.21.1...v0.21.2) (2025-03-28)


### Bug Fixes

* **api:** update model config property name for Pydantic ([b2de1a8](https://github.com/pscedu/cpra.mp.data/commit/b2de1a8018e5b646b0a3a5b92c89fa734a587a76))

## [0.21.1](https://github.com/pscedu/cpra.mp.data/compare/v0.21.0...v0.21.1) (2025-03-28)


### Bug Fixes

* **structure:** replace veg_grid_cell with veg_pixel ([1df5ed2](https://github.com/pscedu/cpra.mp.data/commit/1df5ed233b8db6ce6e973cec416d44435c7f8847))

## [0.21.0](https://github.com/pscedu/cpra.mp.data/compare/v0.20.0...v0.21.0) (2025-03-27)


### Features

* **scripts:** add conversion of veg grid data to raster format ([0cdd562](https://github.com/pscedu/cpra.mp.data/commit/0cdd562fd2372c501f1cf85cdd0c3717efdc17f1))
* **structure:** update grids ([e8e0684](https://github.com/pscedu/cpra.mp.data/commit/e8e06843e1940130c82721358ef0fe77dadf4dea))

## [0.20.0](https://github.com/pscedu/cpra.mp.data/compare/v0.19.0...v0.20.0) (2025-03-26)


### Features

* **structure:** update data structure ([a160189](https://github.com/pscedu/cpra.mp.data/commit/a1601897299d2e76032e0fd16447f4777e1574b0))


### Bug Fixes

* **api:** allow catalog to handle decadal inconsistencies ([2f1ca55](https://github.com/pscedu/cpra.mp.data/commit/2f1ca55ebfc107db7b50e87dff4cecdadb02e2ad))
* **api:** fix monthly interval in catalog compression ([452c45f](https://github.com/pscedu/cpra.mp.data/commit/452c45fdba1c7827ebb473485e5d74c3ee489512))


### Performance Improvements

* **api:** compute catalog Parquet choices in memory ([aefac47](https://github.com/pscedu/cpra.mp.data/commit/aefac470d5741dfafb5733c656e19688eb6f9096))

## [0.19.0](https://github.com/pscedu/cpra.mp.data/compare/v0.18.0...v0.19.0) (2025-03-10)


### Features

* **structure:** update variables ([75b96e1](https://github.com/pscedu/cpra.mp.data/commit/75b96e1f7c4ecc1f9a871804d323bd58fc4f67c7))

## [0.18.0](https://github.com/pscedu/cpra.mp.data/compare/v0.17.0...v0.18.0) (2025-03-07)


### Features

* **grids:** add label field ([7340b0d](https://github.com/pscedu/cpra.mp.data/commit/7340b0d44c84bac59ea9384cfa1b8c358624a75d))
* **structure:** update grids ([62a7638](https://github.com/pscedu/cpra.mp.data/commit/62a7638e96b8f7f38d8750a8639426e4fc526a47))

## [0.17.0](https://github.com/pscedu/cpra.mp.data/compare/v0.16.0...v0.17.0) (2025-03-05)


### Features

* **api:** add select parameter to limit returned columns ([491eeed](https://github.com/pscedu/cpra.mp.data/commit/491eeed5b0b359e4ce3d28c981c07e1dcde9013b))
* **visualization:** add time series line and visualization endpoint ([1ee0154](https://github.com/pscedu/cpra.mp.data/commit/1ee01540eac0501f7baa7f6e86cbe55558868733))


### Bug Fixes

* **visualization:** melt visualization data ([5a4b138](https://github.com/pscedu/cpra.mp.data/commit/5a4b138de6afd2e9fabdecfcb05526a75ccdb025))


### Performance Improvements

* **visualizations:** limit the columns in data requests ([d6fe43b](https://github.com/pscedu/cpra.mp.data/commit/d6fe43b8f8f29c01e74d82949af3f6db940e2b2d))

## [0.16.0](https://github.com/pscedu/cpra.mp.data/compare/v0.15.0...v0.16.0) (2025-02-12)


### Features

* **grids:** validate geographies based on min, max, and count ([1ca44b8](https://github.com/pscedu/cpra.mp.data/commit/1ca44b80d940201ef3bb047fad3472ab1fe164e9))
* **structure:** update data structure ([f76f1d2](https://github.com/pscedu/cpra.mp.data/commit/f76f1d26afe6fcdfaa92a1072110c32a47344b9f))

## [0.15.0](https://github.com/pscedu/cpra.mp.data/compare/v0.14.0...v0.15.0) (2025-02-04)


### Features

* **api:** handle missing Parquet datasets ([6e8c34e](https://github.com/pscedu/cpra.mp.data/commit/6e8c34e329af2e0b1a2e6f740b4a576f624aa4c9))
* **dimensions:** only partition daily data (not monthly) ([3429cf6](https://github.com/pscedu/cpra.mp.data/commit/3429cf621049a99622bd76df3931a85210d98043))
* **structure:** update data structure ([397589c](https://github.com/pscedu/cpra.mp.data/commit/397589c4cdf26cbac05a30ba30ba7ea83215b985))


### Bug Fixes

* **api:** limit enums to string dimensions ([438300d](https://github.com/pscedu/cpra.mp.data/commit/438300d059c73b9c443875cfae335b58df1d3cd0))


### Documentation

* **readme:** update example dimensions and values ([f428ae4](https://github.com/pscedu/cpra.mp.data/commit/f428ae4bce66664dee88e0dc0bcb4fd522036ddf))
* **structure:** update data structure documentation ([223810a](https://github.com/pscedu/cpra.mp.data/commit/223810a7aad3627777ae4aa575f756b502bc20c0))

## [0.14.0](https://github.com/pscedu/cpra.mp.data/compare/v0.13.0...v0.14.0) (2025-01-10)


### Features

* **structure:** update data structure from spreadsheet ([2faa50c](https://github.com/pscedu/cpra.mp.data/commit/2faa50c6b1ab320631bf94224ae62b8107bf74b3))

## [0.13.0](https://github.com/pscedu/cpra.mp.data/compare/v0.12.0...v0.13.0) (2025-01-03)


### Features

* **api:** add validation for tile bands and output type ([11747c7](https://github.com/pscedu/cpra.mp.data/commit/11747c7956bf005cfea550af15e100ad42c54a87))


### Bug Fixes

* **api:** fix type coercion for catalog parameters ([03cfdec](https://github.com/pscedu/cpra.mp.data/commit/03cfdec87956e2c7e8d2db549a9e18093624e1cb))

## [0.12.0](https://github.com/pscedu/cpra.mp.data/compare/v0.11.0...v0.12.0) (2024-12-20)


### Features

* **structure:** update variables and dimensions ([2867284](https://github.com/pscedu/cpra.mp.data/commit/2867284b5a9165dabcbbee0383786ce21a151d8a))

## [0.11.0](https://github.com/pscedu/cpra.mp.data/compare/v0.10.0...v0.11.0) (2024-12-17)


### Features

* **scripts:** generate pmtiles from model grids ([84dd00d](https://github.com/pscedu/cpra.mp.data/commit/84dd00d11e2e9bba31e81c9b50fce542c15eefee))


### Bug Fixes

* **catalog:** fix getting choices from Parquet ([168631e](https://github.com/pscedu/cpra.mp.data/commit/168631e523afefa440fab4a403d45eff92145641))

## [0.10.0](https://github.com/pscedu/cpra.mp.data/compare/v0.9.4...v0.10.0) (2024-11-22)


### Features

* **structure:** add three CLARA variables ([1ddca18](https://github.com/pscedu/cpra.mp.data/commit/1ddca18158f1aa946ffcdf6865663fc585e617c4))


### Bug Fixes

* pin Polars to 1.13.1 to work around a panic ([cb109c2](https://github.com/pscedu/cpra.mp.data/commit/cb109c24fe6555ece202907f2212dab86aff661c))

## [0.9.4](https://github.com/pscedu/cpra.mp.data/compare/v0.9.3...v0.9.4) (2024-11-22)


### Bug Fixes

* **tabular:** fix reading of un-partitioned datasets ([7a72dd0](https://github.com/pscedu/cpra.mp.data/commit/7a72dd01ac62e94e1eb5eb8dd41a15a5dce9e0eb))

## [0.9.3](https://github.com/pscedu/cpra.mp.data/compare/v0.9.2...v0.9.3) (2024-11-22)


### Bug Fixes

* **api:** add compression parameter for datasets ([d17d0b6](https://github.com/pscedu/cpra.mp.data/commit/d17d0b6a505fab5cd47d8c72428b4b5fca9f17cb))
* **dimensions:** always partition by dataset dimensions ([dbcbaf0](https://github.com/pscedu/cpra.mp.data/commit/dbcbaf0feb4bcaaa280686b833fdbea73fd0a1b2))
* **dimensions:** use partition columns for monthly data ([fe0ee00](https://github.com/pscedu/cpra.mp.data/commit/fe0ee0034f1523b99557dd8b62826440d55368b1))
* **tabular:** read grid from data frame, if present ([ba6153b](https://github.com/pscedu/cpra.mp.data/commit/ba6153b3ebf1f6ac65d4cec72c84e0d0b7f5104a))

## [0.9.2](https://github.com/pscedu/cpra.mp.data/compare/v0.9.1...v0.9.2) (2024-11-19)


### Bug Fixes

* **structure:** fix number of hydro compartment geometries ([7a2be2f](https://github.com/pscedu/cpra.mp.data/commit/7a2be2fa4cd65858d10392e101a34513c01f538e))

## [0.9.1](https://github.com/pscedu/cpra.mp.data/compare/v0.9.0...v0.9.1) (2024-11-15)


### Bug Fixes

* **dimensions:** fix required for variable-specific dimensions ([f31a3ef](https://github.com/pscedu/cpra.mp.data/commit/f31a3efde9c1fc98beb365e18544e36cd582d0aa))
* **tabular:** fix validation message for missing/invalid columns ([1781777](https://github.com/pscedu/cpra.mp.data/commit/17817770aaf2d775f6ad53779d2392ed78ddef2e))

## [0.9.0](https://github.com/pscedu/cpra.mp.data/compare/v0.8.1...v0.9.0) (2024-11-13)


### Features

* **api:** add compression parameters to data endpoint ([17c7be5](https://github.com/pscedu/cpra.mp.data/commit/17c7be53ee920668e1887c9953bacc5ae9a638ea))
* **api:** add data structure endpoints to catalog ([92005db](https://github.com/pscedu/cpra.mp.data/commit/92005dbc9a7d27950596dafb4eafc6617185e348))
* **api:** add Parquet support and date handling to catalog ([3a21ccc](https://github.com/pscedu/cpra.mp.data/commit/3a21ccc99e0b5bcdacf9483c209b1ef560306da9))
* **dataset:** add group and aggregate parameters for raster data ([e142fb4](https://github.com/pscedu/cpra.mp.data/commit/e142fb4151e4c82b6cd2e1eb81f78269a6b356c6))
* **legend:** add function to colorize a pixel array ([9a26b6e](https://github.com/pscedu/cpra.mp.data/commit/9a26b6e68eb9e253464a5cee33317d038f85cfe0))
* **raster:** add support for JPG tile output ([222fd10](https://github.com/pscedu/cpra.mp.data/commit/222fd10767355adccf56f9eb8ff6e48d9ceb21e0))
* **structure:** remove plan_version and add grid ([9b1b361](https://github.com/pscedu/cpra.mp.data/commit/9b1b3616218f5a1d3ceab6fb7fe7c256af415dbc))
* **structure:** update grid versions ([1395f9b](https://github.com/pscedu/cpra.mp.data/commit/1395f9b2991ce04ba5a8eed85833d2f6c692ca2c))


### Bug Fixes

* **api:** allow catalog to override choices with defaults ([6c49e79](https://github.com/pscedu/cpra.mp.data/commit/6c49e79e52146e06cef02571c5dc3f5ad98311f3))
* **api:** fix race condition in streaming response ([346f98b](https://github.com/pscedu/cpra.mp.data/commit/346f98b7dcda1ed46ae063378f5941d9f39152cb))
* **raster:** allow multiple dimension values and fix filtering ([1ceb44c](https://github.com/pscedu/cpra.mp.data/commit/1ceb44c03536768a43f1115bc2b33d0f4056854a))

## [0.8.1](https://github.com/pscedu/cpra.mp.data/compare/v0.8.0...v0.8.1) (2024-10-25)


### Documentation

* **api:** update catalog API example response ([682a202](https://github.com/pscedu/cpra.mp.data/commit/682a202d5d1426613ebc6de94d0fe3d68816dae2))

## [0.8.0](https://github.com/pscedu/cpra.mp.data/compare/v0.7.0...v0.8.0) (2024-10-25)


### Features

* allow dimensions to have multiple values ([fee5cd5](https://github.com/pscedu/cpra.mp.data/commit/fee5cd524b394b9ffbc6c76b192cfd6de5908dec))


### Documentation

* **api:** update API routes ([6f50257](https://github.com/pscedu/cpra.mp.data/commit/6f502570f56dee92128060fea3d9a0b1f51e3b2d))

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
