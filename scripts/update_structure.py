"""Update data structure CSVs from an Excel spreadsheet."""

import polars as pl
import sys
from pathlib import Path

STRUCTURE_PATH = Path("./cpra/mp/data/structure")

for sheet_name, sheet in pl.read_excel(sys.argv[1], sheet_id=0).items():
    csv_name = f"{sheet_name.lower().replace(" ", "_")}.csv"
    print(f"Writing {csv_name}...")
    sheet.filter(status="production").select(pl.exclude("status")).write_csv(
        STRUCTURE_PATH / csv_name
    )
