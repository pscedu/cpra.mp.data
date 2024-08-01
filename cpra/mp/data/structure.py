import polars as pl
from pathlib import Path


def read_structure_csv(name):
    """Read a structure CSV into a data frame."""

    return pl.read_csv(Path(__file__).parent / "structure" / f"{name}.csv")
