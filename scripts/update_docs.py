import json
import re
from cpra.mp.data.schema import SCHEMA
from cpra.mp.data.schema import VARIABLES


def table_row(values):
    """Create a Markdown table row from values."""

    return f"| {' | '.join([str(v) for v in values])} |\n"


def table(header, rows):
    """Create a Markdown table."""

    return "".join(
        [
            table_row(header),
            table_row(["---"] * len(header)),
            "".join([table_row(row) for row in rows]),
        ]
    )


def expression(exp):
    """Format a Polars expression for display."""

    if isinstance(exp, bool):
        return f"`{str(exp)}`"

    spec = json.loads(exp.meta.serialize())
    input = spec["Function"]["input"][0]

    if "BinaryExpr" in input:
        inner = input["BinaryExpr"]
        return f'`{inner["left"]["Column"]}` = "{inner["right"]["Literal"]["String"]}"'

    inner = input["Function"]["input"]
    values = sorted(inner[1]["Literal"]["Series"]["values"])
    return f'`{inner[0]["Column"]}` in ({", ".join([f'"{v}"' for v in values])})'


def level_rows(levels):
    """Convert schema levels to Markdown table rows."""

    rows = []
    for level in levels:
        cells = [f"`{level.column}`", level.name, f"`{str(level.dtype)}`"]

        example_value = level.example_value
        if example_value is None and level.values:
            example_value = level.values[0]

        cells.append(str(example_value or ""))

        rows.append(cells)
    return rows


def variable_row(variable):
    """Create a variable summary row from a Variable instance."""

    return [
        f"`{variable.name}`",
        variable.label,
        f"`{str(variable.dtype)}`",
        variable.unit or "",
        f"`{variable.legend.name}`" if variable.legend else "",
    ]


def replace_section(section_id, content, markdown):
    """Replace a dynamic section with a new version."""
    return re.sub(
        f"(<!--BEGIN SECTION {section_id}-->)(.*?)(<!--END SECTION-->)",
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
        dimension_content = table(header, level_rows(levels))
        dimension_content += "\n"
        for level in levels:
            dimension_content += f"#### {level.name}\n\n"
            dimension_content += f"**Column:** `{level.column}`  \n"
            dimension_content += f"**Data Type:** `{str(level.dtype)}`  \n"
            dimension_content += (
                f"**Required Condition:** {expression(level.required)}  \n"
            )
            if level.min_value:
                dimension_content += f"**Minimum Value:** {level.min_value}  \n"
            if level.max_value:
                dimension_content += f"**Maximum Value:** {level.max_value}  \n"
            if level.values:
                dimension_content += table(
                    ["Value", "Label"], zip(level.values, level.labels or level.values)
                )
            dimension_content += "\n"

        markdown = replace_section(
            f"{dimension_type}_dimensions", dimension_content, markdown
        )

    # Update variables table.
    variable_header = ["Variable", "Description", "Data Type", "Unit", "Legend"]
    variable_names = sorted(list(VARIABLES.keys()))
    variable_rows = [variable_row(VARIABLES[name]) for name in variable_names]
    markdown = replace_section(
        "variables", table(variable_header, variable_rows), markdown
    )

    with open("docs/index.md", "w") as output_file:
        output_file.write(markdown)


if __name__ == "__main__":
    update_docs()
