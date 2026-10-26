# JupyterMap is a public re-export for notebook users (interactive-only, deliberately not
# registered in VISUALIZATIONS below) -- unused within this repo by design.
from cpra.mp.data.visualizations.jupyter_map import JupyterMap  # noqa: F401
from cpra.mp.data.visualizations.map import Map
from cpra.mp.data.visualizations.sediment_dynamics import SedimentDynamics
from cpra.mp.data.visualizations.species_stacked_bar import SpeciesStackedBar
from cpra.mp.data.visualizations.stacked_bar import StackedBar
from cpra.mp.data.visualizations.time_series_line import TimeSeriesLine

VISUALIZATIONS = {viz.Meta.name: viz for viz in [SedimentDynamics, TimeSeriesLine, StackedBar, SpeciesStackedBar, Map]}
