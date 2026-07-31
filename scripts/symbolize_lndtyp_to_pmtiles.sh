#!/bin/bash
# Symbolize a classified npd_lndtyp_reclass GeoTIFF and package it as PMTiles.
#
# Value -> label -> color (from the CPRA land/water condition legend):
#   -9999  No Data                                          transparent
#       1  Vegetated Wetland                                 #E4E4E4
#       2  Water - Deep Subtidal    (Depths > 1.0 m)          #FFFFFF
#       3  Bare/Unvegetated Wetland                           #DDDDDD
#       4  Upland/Developed Areas                             #D6D6D6
#       5  Flotant Marsh                                      #EAEAEA
#       6  Water - Shallow Intertidal (0.0 m < Depths < 0.46 m) #F1F1F1
#       7  Water - Shallow Subtidal   (0.46 m < Depths < 1.0 m) #F8F8F8
#
# Usage:
#   ./symbolize_lndtyp_to_pmtiles.sh /path/to/MP2023_..._lndtyp_reclass.tif [output.pmtiles]
#
# Requires on PATH: gdaldem, gdal_translate, gdaladdo, pmtiles
#   (the `pmtiles` CLI ships with the `pmtiles` PyPI package already in pyproject.toml,
#   or use the go-pmtiles binary -- both expose the same `pmtiles convert` subcommand)

set -euo pipefail

src_tif="$1"
out_pmtiles="${2:-$(basename "${src_tif%.tif}").pmtiles}"

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

rgba_tif="$work_dir/rgba.tif"
color_txt="$work_dir/color_relief.txt"
mbtiles="$work_dir/tiles.mbtiles"

cat > "$color_txt" <<'EOF'
-9999   0   0   0   0
1     228 228 228 255
2     255 255 255 255
3     221 221 221 255
4     214 214 214 255
5     234 234 234 255
6     241 241 241 255
7     248 248 248 255
EOF

# 1. Colorize in the source CRS. -exact_color_entry disables interpolation
#    between table entries (this is categorical data, not a continuous
#    ramp); any pixel value not listed in color_relief.txt becomes fully
#    transparent.
gdaldem color-relief \
    "$src_tif" "$color_txt" "$rgba_tif" \
    -alpha -exact_color_entry \
    -co COMPRESS=DEFLATE -co TILED=YES

# 2. Pack into an MBTiles pyramid. The MBTiles driver implicitly warps the
#    source to EPSG:3857 as part of CreateCopy, so no separate gdalwarp
#    step is needed -- RESAMPLING=NEAREST governs that warp (keeping the
#    flat category colors from blending at boundaries) as well as the
#    overview generation below. PNG8 (paletted) is plenty for 8 flat
#    colors and keeps tile size down.
gdal_translate \
    "$rgba_tif" "$mbtiles" \
    -a_srs "EPSG:26915" \
    -of MBTILES \
    -co TILE_FORMAT=PNG8 \
    -co RESAMPLING=NEAREST

# 3. Build overview/zoom levels. Nearest keeps the categorical fills crisp
#    instead of blending colors at zoomed-out levels. Adjust the decimation
#    factors if you need more/fewer zoom levels for this raster's extent.
gdaladdo -r nearest "$mbtiles" 2 4 8 16 32 64

# 4. Convert to PMTiles.
pmtiles convert "$mbtiles" "$out_pmtiles"

echo "Wrote $out_pmtiles"
