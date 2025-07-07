import subprocess
import sys
import tempfile
from pathlib import Path

GEOMETRY_PATH = Path("cpra/mp/data/geometry")
TILES_PATH = Path("tiles")
TIPPECANOE_PATH = Path(sys.executable).parent / "tippecanoe"


def generate_tiles():
    TILES_PATH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        dest_path = Path(temp_dir)
        for fgb_path in GEOMETRY_PATH.glob("*.fgb"):
            boundary_path = dest_path / "boundary.fgb"
            label_path = dest_path / "label.fgb"
            tiles_path = TILES_PATH / f"{fgb_path.stem}.pmtiles"
            cmds = [
                [
                    "ogr2ogr",
                    "-nln",
                    "boundary",
                    "-t_srs",
                    "EPSG:4326",
                    str(boundary_path.absolute()),
                    str(fgb_path.absolute()),
                ],
                [
                    "ogr2ogr",
                    "-nln",
                    "label",
                    "-sql",
                    f"select id, label, st_pointonsurface(geometry) as geometry from {fgb_path.stem}",
                    "-dialect",
                    "sqlite",
                    "-t_srs",
                    "EPSG:4326",
                    str(label_path.absolute()),
                    str(fgb_path.absolute()),
                ],
                [
                    str(TIPPECANOE_PATH.absolute()),
                    "--force",
                    "-zg",
                    "-o",
                    str(tiles_path.absolute()),
                    "--drop-densest-as-needed",
                    "--use-attribute-for-id=id",
                    "--convert-stringified-ids-to-numbers",
                    "-L",
                    f"boundary:{str(boundary_path.absolute())}",
                    "-L",
                    f"label:{str(label_path.absolute())}",
                ],
            ]

            for cmd in cmds:
                result = subprocess.run(cmd, stderr=subprocess.PIPE)
                if result.returncode:
                    if result.stderr:
                        print(result.stderr, file=sys.stderr)
                    sys.exit(result.returncode)


if __name__ == "__main__":
    generate_tiles()
