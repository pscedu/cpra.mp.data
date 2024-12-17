import re
import subprocess
import sys
import xml.etree.cElementTree as ET
from pathlib import Path

GEOMETRY_PATH = Path("cpra/mp/data/geometry")


def crosswalk_sql(grid, crosswalk_grid):
    """Generate crosswalk SQL based on centroid intersection."""

    return re.sub(
        r"\s+",
        " ",
        f"""select src.id as src_id,
        dst.id as dst_id,
        1 as scale_factor
    from {grid} as src
        inner join {crosswalk_grid} as dst 
            on st_intersects(st_centroid(src.geometry), dst.geometry)
    order by dst.id, src.id""",
    )


def generate_crosswalks(grid):
    crosswalk_grids = []
    vrt_path = GEOMETRY_PATH / "combined.vrt"
    vrt_root = ET.Element("OGRVRTDataSource")

    for fgb_path in GEOMETRY_PATH.glob("*.fgb"):
        fgb_grid = fgb_path.name.split(".")[0]
        vrt_layer = ET.SubElement(vrt_root, "OGRVRTLayer", name=fgb_grid)
        ET.SubElement(vrt_layer, "SrcDataSource").text = str(fgb_path.absolute())
        if fgb_grid != grid:
            crosswalk_grids.append(fgb_grid)

    vrt_tree = ET.ElementTree(vrt_root)
    vrt_tree.write(str(vrt_path))

    for crosswalk_grid in crosswalk_grids:
        print(f"Generating crosswalk from {grid} to {crosswalk_grid}")
        sql = crosswalk_sql(grid, crosswalk_grid)
        parquet_name = f"{grid}__{crosswalk_grid}.parquet"
        parquet_path = Path("cpra/mp/data/crosswalk") / parquet_name
        cmd = [
            "ogr2ogr",
            "-dsco",
            "COMPRESSION=UNCOMPRESSED",
            "-sql",
            sql,
            "-dialect",
            "sqlite",
            (str(parquet_path.absolute())),
            (str(vrt_path.absolute())),
        ]

        result = subprocess.run(cmd, stderr=subprocess.PIPE)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    generate_crosswalks("hydro_compartment_v001")
