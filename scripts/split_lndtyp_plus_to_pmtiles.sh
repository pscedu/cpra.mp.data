#!/bin/bash
# Normalize a classified lnd_type_plus GeoTIFF's nodata and split it into TWO
# PMTiles archives -- one carrying only the water classes, one only the land
# classes -- so the two can be styled and toggled independently on the map.
# In each archive the other group's pixels are fully transparent.
#
# Value -> label -> color (from the CPRA land/water condition legend):
#       1  Vegetated Wetland                                    #E4E4E4   land
#       2  Water - Deep Subtidal      (Depths > 1.0 m)          #FFFFFF   water
#       3  Bare/Unvegetated Wetland                             #DDDDDD   land
#       4  Upland/Developed Areas                               #D6D6D6   land
#       5  Flotant Marsh                                        #EAEAEA   land
#       6  Water - Shallow Intertidal (0.0 m < Depths < 0.46 m) #F1F1F1   water
#       7  Water - Shallow Subtidal   (0.46 m < Depths < 1.0 m) #F8F8F8   water
#   -9999  No Data                                              transparent
#
# The split follows the legend's own labels: the three "Water - " classes are
# water, the rest are land. Flotant Marsh (5) is a vegetated mat and lands on
# the land side, even though it floats.
#
# Usage:
#   ./split_lndtyp_plus_to_pmtiles.sh /path/to/lndtyp_plus_*.tiff [output_dir]
#
# Writes <output_dir>/<basename>_water.pmtiles and <basename>_land.pmtiles
# (output_dir defaults to the current directory).
#
# Requires on PATH: gdal_calc.py, gdaldem, gdal_translate, gdaladdo, pmtiles
#   (the `pmtiles` CLI ships with the `pmtiles` PyPI package already in pyproject.toml,
#   or use the go-pmtiles binary -- both expose the same `pmtiles convert` subcommand)

set -euo pipefail

src_tif="$1"
out_dir="${2:-.}"
base="$(basename "$src_tif")"
base="${base%.*}"

mkdir -p "$out_dir"

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

nodata_tif="$work_dir/nodata9999.tif"

# 1. Fold the -1 sentinel into -9999 and declare -9999 as the band's nodata.
#    The source already uses BOTH: -9999 for ground outside the morphology grid
#    and -1 (its declared nodata) for the hydro-compartment crosswalk gap, so
#    this merges two sentinels into one and they are not distinguishable after.
#    --hideNoData is load-bearing: without it gdal_calc.py applies the source's
#    own -1 mask first and the where() never sees a -1 to act on.
gdal_calc.py \
    -A "$src_tif" \
    --outfile="$nodata_tif" \
    --calc="where(A==-1,-9999,A)" \
    --NoDataValue=-9999 \
    --type=Int16 \
    --hideNoData \
    --co COMPRESS=DEFLATE --co TILED=YES --co BIGTIFF=IF_SAFER

# 2. One color table per group, listing ONLY that group's classes. Combined with
#    -exact_color_entry below, this is what performs the split: a pixel whose
#    value is absent from the table gets no entry and comes out fully
#    transparent, so the opposite group drops out of the archive without needing
#    a separate masking pass over the raster.
cat > "$work_dir/water.txt" <<'EOF'
-9999   0   0   0   0
2     255 255 255 255
6     241 241 241 255
7     248 248 248 255
EOF

cat > "$work_dir/land.txt" <<'EOF'
-9999   0   0   0   0
1     228 228 228 255
3     221 221 221 255
4     214 214 214 255
5     234 234 234 255
EOF

# $1 = group name ("water" | "land"), matching the color table written above.
build_group() {
    local group="$1"
    local color_txt="$work_dir/${group}.txt"
    local rgba_tif="$work_dir/${group}_rgba.tif"
    local mbtiles="$work_dir/${group}.mbtiles"
    local out_pmtiles="$out_dir/${base}_${group}.pmtiles"

    # 3. Colorize in the source CRS. -exact_color_entry disables interpolation
    #    between table entries (this is categorical data, not a continuous
    #    ramp); any pixel value not listed becomes fully transparent.
    gdaldem color-relief \
        "$nodata_tif" "$color_txt" "$rgba_tif" \
        -alpha -exact_color_entry \
        -co COMPRESS=DEFLATE -co TILED=YES

    # 4. Pack into an MBTiles pyramid. The MBTiles driver implicitly warps the
    #    source to EPSG:3857 as part of CreateCopy, so no separate gdalwarp
    #    step is needed -- RESAMPLING=NEAREST governs that warp (keeping the
    #    flat category colors from blending at boundaries) as well as the
    #    overview generation below. PNG8 (paletted) is plenty for a handful of
    #    flat colors and keeps tile size down.
    gdal_translate \
        "$rgba_tif" "$mbtiles" \
        -a_srs "EPSG:26915" \
        -of MBTILES \
        -co TILE_FORMAT=PNG8 \
        -co RESAMPLING=NEAREST

    # 5. Build overview/zoom levels. Nearest keeps the categorical fills crisp
    #    instead of blending colors at zoomed-out levels. Adjust the decimation
    #    factors if you need more/fewer zoom levels for this raster's extent.
    gdaladdo -r nearest "$mbtiles" 2 4 8 16 32 64

    # 6. Convert to PMTiles.
    pmtiles convert "$mbtiles" "$out_pmtiles"

    echo "Wrote $out_pmtiles"
}

build_group water
build_group land
