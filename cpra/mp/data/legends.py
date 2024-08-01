import polars as pl
from cpra.mp.data.structure import read_structure_csv


class Legend:
    def __init__(self, name, df):
        self.name = name
        self._df = df


LEGENDS = {
    group[0]: Legend(group[0], df.select(pl.exclude("name")))
    for group, df in read_structure_csv("legends").group_by("name")
}
