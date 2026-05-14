from cpra.mp.data.visualizations.sediment_dynamics import SedimentDynamics
from cpra.mp.data.visualizations.time_series_line import TimeSeriesLine
from cpra.mp.data.visualizations.map import Map
from cpra.mp.data.visualizations.jupyter_map import JupyterMap

VISUALIZATIONS = {viz.Meta.name: viz for viz in [SedimentDynamics, TimeSeriesLine, Map]}
