"""Convert stored values into the units a reader is shown.

A variable's values are stored in whatever unit the model writes and the raster/Parquet
layer keeps (``Variable.unit``), which is not always the unit a chart or legend should
label them with.  ``elev`` is the case that drives this: it is stored as Int32
millimeters -- so that ZSTD's predictor has integers to delta-encode -- but is read in
meters, the unit every legend label and axis title has always used.

``Variable.display_scale_factor`` is the **divisor** taking a stored value to a displayed
one (``1000`` for millimeters shown as meters), and ``display_unit`` /
``display_number_format`` are the label and d3 format string that go with the result.  All
three are optional: a variable stored in the unit it is shown in leaves them empty, and
every helper here is then the identity.

This module is the only place the conversion is applied.  Everything upstream -- the data
API, the raster color maps, ``legends.csv``'s break points, ``difference_legend_limit`` --
stays in storage units, so there is exactly one boundary to reason about rather than a
scale factor sprinkled through the read path.
"""


def display_unit(variable):
    """The unit to label ``variable``'s displayed values with."""

    return variable.display_unit or variable.unit


def display_number_format(variable):
    """``variable``'s d3 format string for displayed values, as Vega-Lite wants it.

    Returns ``""`` rather than ``None`` when unset -- an Altair ``format=`` of ``None``
    serializes into the spec as an explicit null, which Vega-Lite rejects.
    """

    return variable.display_number_format or ""


def to_display(variable, value):
    """Convert one stored value of ``variable`` to display units.

    For scaling numbers that are already in hand server-side (a legend's break points, a
    difference limit); use ``display_value_expr`` for values a chart only sees in the
    browser.  ``None`` passes through, since a missing bound stays missing.
    """

    if value is None or not variable.display_scale_factor:
        return value
    return value / variable.display_scale_factor


def display_value_expr(variable, field="datum.value"):
    """A Vega expression converting ``field`` from storage to display units.

    Returns ``field`` unchanged when ``variable`` needs no scaling, which lets callers
    compare the result against what they passed in to decide whether a
    ``transform_calculate`` is needed at all rather than emitting an identity transform.
    """

    if not variable or not variable.display_scale_factor:
        return field
    return f"{field} / {variable.display_scale_factor}"
