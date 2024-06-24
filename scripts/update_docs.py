import json
import re
from cpra.mp.data.schema import SCHEMA


def table_row(values):
    """Create a Markdown table row from values."""

    return f"| {' | '.join(values)} |\n"


def table(header, rows):
    """Create a Markdown table."""

    return "".join(
        [
            table_row(header),
            table_row(["---"] * len(header)),
            "".join([table_row(row) for row in rows]),
        ]
    )


def level_rows(levels):
    """Convert schema levels to Markdown table rows."""

    rows = []
    for level in levels:
        cells = [f"`{level.column}`", level.name, f"`{str(level.dtype)}`"]
        if isinstance(level.required, bool):
            cells.append(f"`{str(level.required)}`")
        else:
            spec = json.loads(level.required.meta.serialize())
            input = spec["Function"]["input"][0]
            if "BinaryExpr" in input:
                inner = input["BinaryExpr"]
                cells.append(
                    f'`{inner["left"]["Column"]}` = "{inner["right"]["Literal"]["String"]}"'
                )
            else:
                inner = input["Function"]["input"]
                values = sorted(inner[1]["Literal"]["Series"]["values"])
                cells.append(
                    f'`{inner[0]["Column"]}` in ({", ".join([f'"{v}"' for v in values])})'
                )

        example_value = level.example_value
        if example_value is None and level.values:
            example_value = level.values[0]

        cells.append(str(example_value or ""))

        rows.append(cells)
    return rows


def replace_table(table_id, content, markdown):
    """Replace a dynamic table with a new version."""
    return re.sub(
        f"(<!--BEGIN TABLE {table_id}-->)(.*?)(<!--END TABLE-->)",
        f"\\1\\n{content}\\3",
        markdown,
        flags=re.DOTALL,
    )


def update_docs():
    """Update documenation Markdown."""

    with open("docs/index.md", "r") as input_file:
        markdown = input_file.read()

    # Update dimension tables.
    header = [
        "Column",
        "Description",
        "Data Type",
        "Required Condition",
        "Example Value",
    ]
    dimension_tables = {
        "dataset": [level for level in SCHEMA if level.partition is True],
        "partition": [
            level for level in SCHEMA if not isinstance(level.partition, bool)
        ],
        "nonpartition": [level for level in SCHEMA if level.partition is False],
    }
    for dimension_type, levels in dimension_tables.items():
        markdown = replace_table(
            f"{dimension_type}_dimensions", table(header, level_rows(levels)), markdown
        )

    with open("docs/index.md", "w") as output_file:
        output_file.write(markdown)


if __name__ == "__main__":
    update_docs()
