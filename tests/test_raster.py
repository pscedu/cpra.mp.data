"""Tests for writing, locating, and virtually stacking COGs."""

import time
import xml.etree.ElementTree as ET

import numpy as np
import polars as pl
import pytest
import rasterio as rio

from cpra.mp.data import config, raster
from cpra.mp.data.exceptions import ValidationError
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.raster import cog_path_expression, compact_cog, read_cog, vrt, write_cog
from tests.factories import RASTER_KWARGS, raster_array, write_raster_dataset

DATASET_KWARGS = {k: RASTER_KWARGS[k] for k in ("variable", "grid", "time_unit")}


class TestCogPathExpression:
    def test_builds_a_dimension_directory_path(self, tmp_path):
        """The calendar dimension is never part of the path -- its value lives only in
        the band description (see write_cog) -- and neither is the file's .tif suffix,
        since the file itself is named with a UUID rather than derived from this path."""
        df = pl.DataFrame({"variable": ["vct"], "grid": ["veg_grid_cell_v001"], "calendar_year": [2030]})
        path = df.select(p=cog_path_expression(df, tmp_path))["p"].item()
        assert path == str(tmp_path / "variable=vct/grid=veg_grid_cell_v001")

    def test_metadata_dimensions_are_excluded_from_the_path(self, tmp_path):
        df = pl.DataFrame({"variable": ["vct"], "model": ["icm_v2023.0.0"]})
        path = df.select(p=cog_path_expression(df, tmp_path))["p"].item()
        assert "model=" not in path

    def test_unknown_columns_are_ignored(self, tmp_path):
        df = pl.DataFrame({"variable": ["vct"], "not_a_dimension": ["x"]})
        path = df.select(p=cog_path_expression(df, tmp_path))["p"].item()
        assert "not_a_dimension" not in path

    def test_base_path_defaults_to_the_configured_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "BASE_PATH", tmp_path)
        df = pl.DataFrame({"variable": ["vct"]})
        path = df.select(p=cog_path_expression(df))["p"].item()
        assert path.startswith(str(tmp_path))


class TestWriteCog:
    def test_writes_to_the_dimension_path(self, base_path):
        """Dataset dimensions (in variable/time_unit/grid order) and partition
        dimensions form the directory; the calendar dimension isn't part of it, and the
        file itself is named with a UUID, not a dimension value."""
        write_cog(raster_array(fill=1), base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        expected_dir = (
            base_path / "variable=vct/time_unit=annual/grid=veg_grid_cell_v001" / "model_group_id=500/scenario_id=7"
        )
        assert list(expected_dir.glob("*.tif"))

    def test_creates_parent_directories(self, base_path):
        write_cog(raster_array(fill=1), base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        assert list(base_path.rglob("*.tif"))

    def test_pixel_values_round_trip(self, base_path):
        data = raster_array()
        write_cog(data, base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        with rio.open(next(base_path.rglob("*.tif"))) as src:
            assert np.array_equal(src.read(1), data)

    def test_geometry_matches_the_grid(self, base_path):
        grid = GRIDS[RASTER_KWARGS["grid"]]
        write_cog(raster_array(fill=1), base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        with rio.open(next(base_path.rglob("*.tif"))) as src:
            assert (src.height, src.width) == (grid.height, grid.width)
            assert src.crs.to_string().lower() == grid.crs
            assert src.bounds.left == pytest.approx(grid.bounds[0])
            assert src.bounds.top == pytest.approx(grid.bounds[3])

    def test_nodata_is_set(self, base_path):
        write_cog(raster_array(fill=1), base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        with rio.open(next(base_path.rglob("*.tif"))) as src:
            assert src.nodata == -9999

    def test_elev_declares_its_scaled_nodata(self, base_path, monkeypatch):
        """elev stores Int32 millimeters, where -9999 is a plausible value (-9.999 m);
        its writes must declare `Variable.nodata` (-9999000) instead of the shared
        default. The grid is shrunk because morph_pixel is 170M pixels per band --
        nothing about the declared nodata depends on the raster's size."""
        grid = GRIDS["morph_pixel_v001"]
        monkeypatch.setattr(grid, "height", 4)
        monkeypatch.setattr(grid, "width", 8)
        elev_kwargs = {**RASTER_KWARGS, "variable": "elev", "grid": "morph_pixel_v001"}
        data = np.full((grid.height, grid.width), 1500, dtype="int32")
        write_cog(data, base_path=base_path, calendar_year=2030, **elev_kwargs)
        with rio.open(next(base_path.rglob("*.tif"))) as src:
            assert src.nodata == -9999000

    def test_single_band(self, base_path):
        write_cog(raster_array(fill=1), base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        with rio.open(next(base_path.rglob("*.tif"))) as src:
            assert src.count == 1

    def test_kwargs_are_written_as_tags(self, base_path):
        """Dimension values are the only record of what a COG contains, except the
        calendar dimension, which is never a tag -- it lives solely in the band
        description (see write_cog)."""
        write_cog(raster_array(fill=1), base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        with rio.open(next(base_path.rglob("*.tif"))) as src:
            tags = src.tags()
            description = src.descriptions[0]
        assert tags["variable"] == "vct"
        assert tags["model"] == "icm_v2023.0.0"
        assert "calendar_year" not in tags
        assert description == "calendar_year=2030"

    def test_flat_arrays_are_reshaped_to_the_grid(self, base_path):
        grid = GRIDS[RASTER_KWARGS["grid"]]
        flat = np.arange(grid.height * grid.width, dtype="int32")
        write_cog(flat, base_path=base_path, calendar_year=2030, **RASTER_KWARGS)
        with rio.open(next(base_path.rglob("*.tif"))) as src:
            assert src.shape == (grid.height, grid.width)

    def test_missing_required_dimension_raises(self, base_path):
        """calendar_year is required for annual data."""
        with pytest.raises(ValidationError, match="calendar_year"):
            write_cog(raster_array(fill=1), base_path=base_path, **RASTER_KWARGS)

    def test_invalid_dimension_value_raises(self, base_path):
        with pytest.raises(ValidationError, match="calendar_year_maximum_value"):
            write_cog(raster_array(fill=1), base_path=base_path, calendar_year=2500, **RASTER_KWARGS)

    def test_validate_false_skips_validation(self, base_path):
        write_cog(
            raster_array(fill=1),
            base_path=base_path,
            validate=False,
            calendar_year=2500,
            **RASTER_KWARGS,
        )
        assert list(base_path.rglob("*.tif"))


class TestMetadataPool:
    """`_parallel_read_cog_metadata` opens every candidate COG to find out which calendar
    values it covers -- the filenames are opaque UUIDs -- so it is the fixed cost in front
    of `read_cog`, `compact_cog`, and `catalog.raster_rows` alike.
    """

    @pytest.fixture
    def cogs(self, base_path):
        write_raster_dataset(base_path, years=tuple(range(2030, 2038)), fill_fn=lambda y: y - 2000)
        return sorted(str(p) for p in base_path.rglob("*.tif"))

    def test_the_pool_is_shared_rather_than_built_per_call(self):
        # Building one per call is what this costs: GDAL's per-thread setup is paid again by
        # every worker of every new pool, which measured ~6x a later open on the same thread
        # -- enough that a per-call pool read 81 COGs slower than a plain serial loop did.
        assert raster._metadata_pool() is raster._metadata_pool()

    def test_the_pool_width_does_not_follow_the_machine(self):
        """`ThreadPoolExecutor`'s own default is `min(32, os.cpu_count() + 4)`, which on a
        small host is 6. Width is exactly what buys the overlap this pool exists for, and an
        open's latency has nothing to do with the core count: over 81 COGs at 20 ms per open,
        4 workers measured 3.9x a serial read where 16 measured 13.9x.
        """
        assert raster._metadata_pool()._max_workers == raster._METADATA_POOL_WORKERS
        assert raster._METADATA_POOL_WORKERS >= 16

    def test_results_come_back_in_the_order_the_paths_were_given(self, cogs):
        """`_read_cog_band_rows` zips `paths` against these results positionally, so a path
        paired with another file's descriptions would mislabel its bands' calendar values
        without anything raising. Opens finish in whatever order the filesystem allows; the
        shuffle is what makes completion order unlikely to match input order.
        """
        shuffled = list(reversed(cogs))
        pooled = raster._parallel_read_cog_metadata(shuffled)

        assert pooled == [raster._read_cog_metadata(path) for path in shuffled]
        # Not vacuous only because the files genuinely differ: distinct years, distinct
        # descriptions. Identical files would pass this no matter how the results were paired.
        assert len({result[0] for result in pooled}) == len(shuffled)

    def test_the_opens_actually_overlap(self, cogs, monkeypatch):
        """The whole point is that a slow open does not block the next one. Timed rather
        than asserted structurally because "runs on a pool" is satisfiable by a pool of one
        worker, which would not be this.
        """
        delay = 0.05
        real_open = rio.open

        def slow_open(path, *args, **kwargs):
            time.sleep(delay)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(rio, "open", slow_open)
        started = time.perf_counter()
        raster._parallel_read_cog_metadata(cogs)
        elapsed = time.perf_counter() - started

        # Eight files, each stalled `delay`; serially that is 8 * delay. Half of that is a
        # wide margin around the ~1 * delay a pool this size should actually take.
        assert len(cogs) == 8
        assert elapsed < len(cogs) * delay / 2


class TestReadCog:
    def test_finds_written_cogs(self, base_path):
        write_raster_dataset(base_path, years=(2030, 2040))
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()
        assert df["calendar_year"].to_list() == [2030, 2040]

    def test_uses_the_given_base_path(self, base_path, tmp_path, monkeypatch):
        """Regression: read_cog accepted base_path but never forwarded it to
        glob_lazy_frame, so it always searched the configured BASE_PATH instead."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.setattr(config, "BASE_PATH", elsewhere)

        write_raster_dataset(base_path, years=(2030,))
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()
        assert df.height == 1

    def test_path_column_points_at_real_files(self, base_path):
        write_raster_dataset(base_path, years=(2030,))
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()
        assert all(rio.open(p) for p in df["path"])

    def test_dimension_columns_are_typed(self, base_path):
        write_raster_dataset(base_path, years=(2030,))
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()
        assert df["calendar_year"].dtype == pl.Int32
        assert df["model_group_id"].dtype == pl.Int32

    def test_scalar_filter(self, base_path):
        write_raster_dataset(base_path, years=(2030, 2040, 2050))
        df = read_cog(base_path=base_path, calendar_year=2040, **DATASET_KWARGS).collect()
        assert df["calendar_year"].to_list() == [2040]

    def test_list_filter(self, base_path):
        write_raster_dataset(base_path, years=(2030, 2040, 2050))
        df = read_cog(base_path=base_path, calendar_year=[2030, 2050], **DATASET_KWARGS).collect()
        assert df["calendar_year"].to_list() == [2030, 2050]

    def test_metadata_dimensions_are_not_required_in_the_path(self, base_path):
        """model is metadata, so it never appears in a COG path or the result columns."""
        write_raster_dataset(base_path, years=(2030,))
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()
        assert "model" not in df.columns

    def test_no_matches_yields_an_empty_frame(self, base_path):
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()
        assert df.height == 0

    def test_missing_dataset_dimension_raises(self, base_path):
        with pytest.raises(ValidationError, match="Required columns are missing"):
            read_cog(base_path=base_path, variable="vct")

    def test_include_file_paths_can_be_renamed(self, base_path):
        write_raster_dataset(base_path, years=(2030,))
        df = read_cog(base_path=base_path, include_file_paths="src", **DATASET_KWARGS).collect()
        assert "src" in df.columns


class TestVrt:
    def test_stacks_cogs_into_a_multiband_dataset(self, base_path):
        write_raster_dataset(base_path, years=(2030, 2040), fill_fn=lambda y: y - 2000)
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()

        with rio.open(vrt(df["path"])) as src:
            assert src.count == 2
            assert src.read(1)[0, 0] == 30
            assert src.read(2)[0, 0] == 40

    def test_band_numbers_are_sequential(self, base_path):
        write_raster_dataset(base_path, years=(2030, 2040, 2050))
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()

        root = ET.fromstring(vrt(df["path"]))
        bands = root.findall("VRTRasterBand")
        assert [b.get("band") for b in bands] == ["1", "2", "3"]

    def test_preserves_grid_geometry(self, base_path):
        grid = GRIDS[RASTER_KWARGS["grid"]]
        write_raster_dataset(base_path, years=(2030, 2040))
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()

        with rio.open(vrt(df["path"])) as src:
            assert (src.height, src.width) == (grid.height, grid.width)

    def test_single_path_produces_one_band(self, base_path):
        write_raster_dataset(base_path, years=(2030,))
        df = read_cog(base_path=base_path, **DATASET_KWARGS).collect()

        with rio.open(vrt(df["path"])) as src:
            assert src.count == 1

    @pytest.fixture
    def three_band_cog(self, base_path):
        """One COG carrying three bands, filled 30/31/32 -- what a compacted run looks like,
        and the only shape in which a source list can name one file more than once. The years
        are consecutive because that is what `compact_cog` merges.
        """
        write_raster_dataset(base_path, years=(2030, 2031, 2032), fill_fn=lambda y: y - 2000)
        compact_cog(base_path=base_path, catalog_path=None, group_years=None, **RASTER_KWARGS)
        tifs = list(base_path.rglob("*.tif"))
        assert len(tifs) == 1
        return str(tifs[0])

    def test_a_file_is_opened_once_however_many_of_its_bands_are_used(self, three_band_cog, monkeypatch):
        """A source list names (path, band) pairs, so a multi-band COG usually appears in it
        several times -- but one open already yields every band's XML, and re-opening per
        source re-read the same header only to discard all but one band of the result.

        Counts opens rather than asserting a duration because the saving is the opens: on a
        real 52-band request over 5 files this was 52 of them instead of 5.
        """
        opened = []
        real_open = rio.open

        def counting_open(path, *args, **kwargs):
            opened.append(str(path))
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(rio, "open", counting_open)
        out = vrt([(three_band_cog, band) for band in (1, 2, 3)])

        assert opened == [three_band_cog]
        assert [b.get("band") for b in ET.fromstring(out).findall("VRTRasterBand")] == ["1", "2", "3"]

    def test_the_same_band_can_be_used_twice(self, three_band_cog):
        """Each file is parsed once, so both of these sources resolve to the same element of
        the same tree -- and one element cannot be two output bands. Copied per source, so
        that numbering the second does not renumber the first.

        The numbers are what breaks, not the pixels: GDAL takes band order from the elements'
        document order, so an element repeated by reference still reads correctly while
        carrying whichever number was written to it last.
        """
        out = vrt([(three_band_cog, 2), (three_band_cog, 2)])
        assert [b.get("band") for b in ET.fromstring(out).findall("VRTRasterBand")] == ["1", "2"]

        with rio.open(out) as src:
            assert src.count == 2
            assert src.read(1)[0, 0] == 31
            assert src.read(2)[0, 0] == 31

    def test_bands_of_one_file_can_be_selected_in_any_order(self, three_band_cog):
        """Output band order follows the source list, not the source file's own band order,
        including when every source is the same file. The first source is also the one whose
        dataset element becomes the output root, so this is the case where that element has
        to give up its own band 1 and be renumbered to keep a later band instead.
        """
        out = vrt([(three_band_cog, 3), (three_band_cog, 1)])
        # Sequential from 1, not the numbers these bands had in their source file (3 and 1).
        assert [b.get("band") for b in ET.fromstring(out).findall("VRTRasterBand")] == ["1", "2"]

        with rio.open(out) as src:
            assert src.count == 2
            assert src.read(1)[0, 0] == 32
            assert src.read(2)[0, 0] == 30

    def test_childless_band_elements_are_not_dropped(self):
        """Regression: the band was appended under `if band_el:`, and an Element with no
        children is falsy, so a band carrying no sub-elements was silently discarded."""
        first = "<VRTDataset><VRTRasterBand band='1'><X/></VRTRasterBand></VRTDataset>"
        second = "<VRTDataset><VRTRasterBand band='1'/></VRTDataset>"

        out = ET.fromstring(first)
        band = ET.fromstring(second).find("VRTRasterBand")
        # Having no children is exactly what made `if band_el:` skip this band. Spelled
        # as len() rather than `not band`, which triggers the deprecation it demonstrates.
        assert len(band) == 0
        assert band is not None

        if band is not None:
            band.set("band", "2")
            out.append(band)
        assert len(out.findall("VRTRasterBand")) == 2


class TestCompactCogIncludeValues:
    """Writing only *some* of a partition's bands into a separate tree.

    This is what `scripts/migrate_to_long_format.py --append-missing` runs on: an output
    tree built by an earlier migration pass, and an input tree that has since grown new
    single-band COGs. Only the bands the output lacks may be written, and they must be
    written even when each one is, on its own, already sitting in a perfectly good file.
    """

    def _band_values(self, path):
        with rio.open(path) as src:
            return sorted(int(desc.split("=")[1]) for desc in src.descriptions)

    def test_writes_only_the_requested_values(self, base_path, tmp_path):
        output_path = tmp_path / "out"
        write_raster_dataset(base_path, years=(2030, 2031, 2032))

        result = compact_cog(
            base_path=base_path,
            output_base_path=output_path,
            catalog_path=None,
            include_values=[2032],
            **RASTER_KWARGS,
        )

        assert len(result["outputs"]) == 1
        assert self._band_values(result["outputs"][0]) == [2032]

    def test_an_excluded_band_does_not_extend_a_run(self, base_path, tmp_path):
        """2030 and 2032 are only contiguous *through* 2031. Excluding 2031 has to leave two
        one-band files, not one file claiming a run it doesn't have -- the same way a
        genuinely absent year would."""
        output_path = tmp_path / "out"
        write_raster_dataset(base_path, years=(2030, 2031, 2032))

        result = compact_cog(
            base_path=base_path,
            output_base_path=output_path,
            catalog_path=None,
            include_values=[2030, 2032],
            **RASTER_KWARGS,
        )

        assert sorted(self._band_values(p) for p in result["outputs"]) == [[2030], [2032]]

    def test_out_of_place_copies_a_group_that_is_already_optimal(self, base_path, tmp_path):
        """A lone year lives optimally in its own single-band file already, so an in-place
        compaction leaves it alone. Out-of-place that reasoning doesn't hold: the file isn't
        in the output tree at all, and skipping it would silently drop the year."""
        output_path = tmp_path / "out"
        write_raster_dataset(base_path, years=(2030,))

        result = compact_cog(base_path=base_path, output_base_path=output_path, catalog_path=None, **RASTER_KWARGS)

        assert len(result["outputs"]) == 1
        assert self._band_values(result["outputs"][0]) == [2030]

    def test_in_place_still_skips_a_group_that_is_already_optimal(self, base_path):
        """The converse: nothing to gain from rewriting a file into itself."""
        write_raster_dataset(base_path, years=(2030,))

        result = compact_cog(base_path=base_path, catalog_path=None, **RASTER_KWARGS)

        assert result == {"outputs": [], "sources_merged": 0}
        assert len(list(base_path.rglob("*.tif"))) == 1

    def test_refuses_to_filter_an_in_place_merge(self, base_path):
        """In place, compact_cog unlinks the sources it consumed. A source holding both an
        included and an excluded band would be unlinked with the excluded band still only
        in it -- so the combination is refused rather than silently losing data."""
        write_raster_dataset(base_path, years=(2030, 2031))

        with pytest.raises(ValueError, match="output_base_path"):
            compact_cog(base_path=base_path, catalog_path=None, include_values=[2030], **RASTER_KWARGS)

        assert len(list(base_path.rglob("*.tif"))) == 2

    def test_catalogs_only_the_values_it_wrote(self, base_path, tmp_path):
        """The catalog rows have to describe what landed in the output tree, not what
        compact_cog happened to read on the way there.

        The catalog is a directory of its own rather than the ``catalog_path`` fixture: the
        sources are written by ``write_cog``, whose own catalog hook would otherwise have
        already recorded all three years there before compact_cog ran.
        """
        output_path = tmp_path / "out"
        out_catalog = tmp_path / "out-catalog"
        write_raster_dataset(base_path, years=(2030, 2031, 2032))

        compact_cog(
            base_path=base_path,
            output_base_path=output_path,
            catalog_path=out_catalog,
            include_values=[2032],
            **RASTER_KWARGS,
        )

        rows = pl.concat([pl.read_parquet(p) for p in out_catalog.rglob("*.parquet")], how="diagonal_relaxed")
        assert rows.height == 1
        assert rows["calendar_year_min"][0] == 2032
        assert rows["calendar_year_max"][0] == 2032
