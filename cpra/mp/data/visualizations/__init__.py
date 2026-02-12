from cpra.mp.data.visualizations.sediment_dynamics import SedimentDynamics
from cpra.mp.data.visualizations.time_series_line import TimeSeriesLine
from cpra.mp.data.visualizations.map import Map

VISUALIZATIONS = {viz.Meta.name: viz for viz in [SedimentDynamics, TimeSeriesLine, Map]}
