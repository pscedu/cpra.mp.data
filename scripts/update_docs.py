import re
from cpra.mp.data.structure import read_structure_csv
import polars.selectors as cs


def table_row(values):
    """Create a Markdown table row from values."""

    return f"| {' | '.join([str(v) for v in values])} |\n"


def table(df, column_aliases={}, format={}):
    """Create a Markdown table."""

    return "".join(
        [
            table_row(
                [
                    column_aliases[col]
                    if col in column_aliases
                    else col.replace("_", " ").title()
                    for col in df.columns
                ]
            ),
            table_row(["---"] * len(df.columns)),
            "".join(
                [
                    table_row(
                        [
                            format[k](v) if k in format else ("" if v is None else v)
                            for k, v in row.items()
                        ]
                    )
                    for row in df.iter_rows(named=True)
                ]
            ),
        ]
    )


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

    dimensions = read_structure_csv("dimensions")
    dimension_values = read_structure_csv("dimension_values")
    grids = read_structure_csv("grids")
    legends = read_structure_csv("legends")
    variables = read_structure_csv("variables")

    markdown = replace_section(
        "variables", table(variables, column_aliases={"dtype": "Data Type"}), markdown
    )

    dimensions_content = table(dimensions)
    for dim in dimensions.iter_rows(named=True):
        values_df = dimension_values.filter(dimension=dim["name"])
        if not len(values_df):
            continue
        dimensions_content += f"\n\n ### {dim["label"]} Values"
        dimensions_content += f"\n\n{table(values_df.select("value", "label"))}"
    markdown = replace_section("dimensions", dimensions_content, markdown)

    markdown = replace_section(
        "grids", table(grids, column_aliases={"crs": "CRS"}), markdown
    )

    legends_content = "\n\n".join(
        [
            f"### `{group[0]}` Legend\n\n{table(legend_df.select(cs.exclude("name")))}"
            for group, legend_df in legends.group_by("name")
        ]
    )

    markdown = replace_section("legends", legends_content, markdown)

    with open("docs/index.md", "w") as output_file:
        output_file.write(markdown)


if __name__ == "__main__":
    update_docs()
