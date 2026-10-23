"""The colors the portal draws a selection with, shared by the map and the charts beside it.

`MAP_SELECTION` is the same palette, in the same order, as the UI's own `selectionColors`
(`src/helpers/config.ts`) — the map picks from it by a pin's place in the selection, so
anything drawn alongside has to pick the same way. See `selection_color`.
"""

MAP_SELECTION = [
    "#ff0000",
    "#0aefff",
    "#ffd300",
    "#be0aff",
    "#0aff99",
    "#580aff",
    "#deff0a",
    "#ff8700",
    "#a1ff0a",
    "#147df5",
]

# What a grid cell past the end of the palette is drawn with, as the UI draws it (see
# `getSelectionColor`): there are as many colors as the most grid cells any visualization
# allows, so a longer selection is already past what either has a color left to tell apart.
MAP_SELECTION_OVERFLOW = "#000000"


def selection_color(index):
    """Get the color the map draws the `index`th grid cell of a selection with.

    The map takes a pin's color — and the outline color of the grid cell the pin fell in —
    from the pin's place in the selection. A chart read beside the map has to pick by that
    same place rather than by its own ordering, which the coordinates dropped or collapsed
    on the way to a series put out of step.
    """
    return MAP_SELECTION[index] if index < len(MAP_SELECTION) else MAP_SELECTION_OVERFLOW
