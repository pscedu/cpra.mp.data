import json
import subprocess
import sys
from pathlib import Path
import topojson as tp

GEOMETRY_PATH = Path("cpra/mp/data/geometry")
GEOJSON_PATH = Path("cpra/mp/data/geojson")


def generate_geojson():
    for fgb_path in GEOMETRY_PATH.glob("*.fgb"):
        grid_name = fgb_path.name.split(".")[0]
        print(f"Converting {grid_name}...")
        geojson_path = GEOJSON_PATH / f"{grid_name}.geojson"

        cmd = [
            "ogr2ogr",
            "-t_srs",
            "EPSG:4326",
            "-lco",
            "ID_FIELD=id",
            (str(geojson_path.absolute())),
            (str(fgb_path.absolute())),
        ]

        result = subprocess.run(cmd, stderr=subprocess.PIPE)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    generate_geojson()
