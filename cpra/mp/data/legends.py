import numpy as np
import polars as pl

from cpra.mp.data.structure import read_structure_csv


class Legend:
    def __init__(self, name, df):
        self.name = name
        self.break_points = df

    def colorize_pixels(self, pixel_values):
        """Apply a color ramp to an array of pixels."""
        breaks = self.break_points.filter(pl.col("upper_bound").is_not_null())[
            "upper_bound"
        ].to_numpy()
        colors = [
            f"{color.lstrip("#")}{"ff" if len(color) < 8 else ""}"
            for color in self.break_points["color"]
        ]
        rgba_colors = [
            tuple(int(color[i : i + 2], 16) for i in (0, 2, 4, 6)) for color in colors
        ]
        color_idx = np.digitize(pixel_values, breaks)
        return np.take(rgba_colors, color_idx, axis=0)[0].astype("uint8")


LEGENDS = {
    group[0]: Legend(group[0], df.select(pl.exclude("name")))
    for group, df in read_structure_csv("legends").group_by("name")
}
