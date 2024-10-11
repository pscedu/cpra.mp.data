import re
import subprocess
import sys
import xml.etree.cElementTree as ET
from pathlib import Path


def crosswalk_sql(geographic_unit, crosswalk_unit):
    """Generate crosswalk SQL based on centroid intersection."""

    return re.sub(
        r"\s+",
        " ",
        f"""select src.id as src_id,
        dst.id as dst_id,
        1 as scale_factor
    from {geographic_unit} as src
        inner join {crosswalk_unit} as dst 
            on st_intersects(st_centroid(src.geometry), dst.geometry)
    order by dst.id, src.id""",
    )


def generate_crosswalks(geographic_unit):
    for plan_version_path in Path("cpra/mp/data/geometry").glob("*/"):
        plan_version = plan_version_path.name
        crosswalk_units = []
        vrt_path = plan_version_path / "combined.vrt"
        vrt_root = ET.Element("OGRVRTDataSource")

        for fgb_path in plan_version_path.glob("*.fgb"):
            fgb_unit = fgb_path.name.split(".")[0]
            vrt_layer = ET.SubElement(vrt_root, "OGRVRTLayer", name=fgb_unit)
            ET.SubElement(vrt_layer, "SrcDataSource").text = str(fgb_path.absolute())
            if fgb_unit != geographic_unit:
                crosswalk_units.append(fgb_unit)

        vrt_tree = ET.ElementTree(vrt_root)
        vrt_tree.write(str(vrt_path))

        for crosswalk_unit in crosswalk_units:
            print(
                f"Generating crosswalk from {geographic_unit} to {crosswalk_unit} "
                f"for {plan_version}..."
            )
            sql = crosswalk_sql(geographic_unit, crosswalk_unit)
            parquet_name = f"{plan_version}__{geographic_unit}__{plan_version}__{crosswalk_unit}.parquet"
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
    generate_crosswalks("hydro_compartment")
