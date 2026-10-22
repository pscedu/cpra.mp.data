import numpy as np
import polars as pl

from cpra.mp.data.structure import read_structure_csv


class Legend:
    def __init__(self, name, df, type="discrete"):
        self.name = name
        self.break_points = df
        self.type = type

    def colorize_pixels(self, pixel_values):
        """Apply a color ramp to an array of pixels."""
        breaks = self.break_points.filter(pl.col("upper_bound").is_not_null())["upper_bound"].to_numpy()
        colors = [f"{color.lstrip('#')}{'ff' if len(color) < 8 else ''}" for color in self.break_points["color"]]
        rgba_colors = np.array([tuple(int(color[i : i + 2], 16) for i in (0, 2, 4, 6)) for color in colors])

        if self.type == "discrete":
            return np.take(rgba_colors, np.digitize(pixel_values, breaks), axis=0)[0].astype("uint8")

        return np.stack(
            [np.interp(pixel_values, breaks, rgba_colors[:, i])[0].astype(np.uint8) for i in range(4)], axis=2
        )


LEGENDS = {
    group[0]: Legend(group[0], df.select(pl.exclude("name")))
    # `sort` is empty on most rows, so the CSV reader's first-rows inference types it as a
    # string -- which sorts "10" before "2".
    for group, df in read_structure_csv("legends").with_columns(pl.col("sort").cast(pl.Int64)).group_by("name")
}
