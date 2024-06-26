import json
import re
from cpra.mp.data.dimensions import DIMENSIONS
from cpra.mp.data.grids import GRIDS
from cpra.mp.data.variables import VARIABLES


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


def dimension_row(dim):
    """Convert a dimension to a Markdown table row."""

    cells = [f"`{dim.column}`", dim.name, f"`{str(dim.dtype)}`"]

    example_value = dim.example_value
    if example_value is None and dim.values:
        example_value = dim.values[0]

    cells.append(str(example_value or ""))
    return cells


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
        "dataset": [dim for dim in DIMENSIONS if dim.partition is True],
        "partition": [dim for dim in DIMENSIONS if not isinstance(dim.partition, bool)],
        "nonpartition": [dim for dim in DIMENSIONS if dim.partition is False],
    }
    for dimension_type, dims in dimension_tables.items():
        dimension_content = table(header, [dimension_row(d) for d in dims])
        dimension_content += "\n"
        for dim in dims:
            dimension_content += f"#### {dim.name}\n\n"
            dimension_content += f"**Column:** `{dim.column}`  \n"
            dimension_content += f"**Data Type:** `{str(dim.dtype)}`  \n"
            dimension_content += (
                f"**Required Condition:** {expression(dim.required)}  \n"
            )
            if dim.min_value:
                dimension_content += f"**Minimum Value:** {dim.min_value}  \n"
            if dim.max_value:
                dimension_content += f"**Maximum Value:** {dim.max_value}  \n"
            if dim.values:
                dimension_content += table(
                    ["Value", "Label"], zip(dim.values, dim.labels or dim.values)
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

    # Update grids tables.
    raster_header = [
        "Geographic Unit",
        "Plan Version",
        "Bounds",
        "Height",
        "Width",
        "CRS",
    ]
    raster_rows = []
    vector_header = ["Geographic Unit", "Plan Version", "CRS"]
    vector_rows = []

    for gu in GRIDS.values():
        for plan_version, grid in gu.items():
            if grid.grid_type == "raster":
                raster_rows.append(
                    [
                        grid.geographic_unit,
                        plan_version,
                        grid.bounds,
                        grid.height,
                        grid.width,
                        grid.crs.upper(),
                    ]
                )
            else:
                vector_rows.append(
                    [
                        grid.geographic_unit,
                        plan_version,
                        grid.crs.upper(),
                    ]
                )

    markdown = replace_section(
        "raster_grids", table(raster_header, raster_rows), markdown
    )

    markdown = replace_section(
        "vector_grids", table(vector_header, vector_rows), markdown
    )

    with open("docs/index.md", "w") as output_file:
        output_file.write(markdown)


if __name__ == "__main__":
    update_docs()
