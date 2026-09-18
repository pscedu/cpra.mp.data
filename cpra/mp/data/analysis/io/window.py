"""`ReadWindow` — the description of one windowed raster read: what was read, where in
the source it sits, and at what resolution.

It lives in `io/` because that is what it describes. A read window's every field is a
property of something that came off disk: the source's identity, the pixel offset and
size within it, the affine transform and CRS that place those pixels on the ground, and
the decimation level actually read. The raster scan constructs one; the op algebra
carries it along and checks it never changes; a raster sink reads it back to write the
result out. Both of those layers need the type, and the request layer -- which answers a
tile URL knowing only the declared ops and each input's grid kind -- does not need it at
all. `io` is therefore the lowest layer that serves every caller, and putting it here is
what lets the scan return a complete band-wide triple without importing the algebra that
consumes it.

`execution/frame.py` re-exports this, since a lowering already imports that module for
the rest of the band-wide contract and should not need to know which layer each half of
it came from.
"""

import dataclasses

from rasterio.crs import CRS
from rasterio.transform import Affine

__all__ = ["ReadWindow"]


@dataclasses.dataclass(frozen=True, slots=True)
class ReadWindow:
    """The frame dictionary a band-wide triple carries: everything needed to place a
    pixel back on the ground given only its row index, since row order is the only
    positional information a band-wide frame keeps (see `execution/frame.py`).

    `source` is an opaque identity for what was read -- a VRT descriptor, a tuple of
    constituent file paths, whatever the scan that built this frame decides is enough to
    tell "the same read" from "a different one" (the only property required of it: `==`
    comparison, for `frame.assert_window_unchanged`). Its concrete shape belongs to
    whatever builds the scan, not to this module.

    `col_off`/`row_off`/`width`/`height` are the read's pixel offset and size within
    `source`, plain ints rather than `rasterio.windows.Window` (whose fields are floats)
    because the read window carried here is always pixel-aligned. `transform` and `crs`
    are the read's affine transform and coordinate system; `decimation` is the
    overview/decimation level actually read, `1` meaning full resolution.

    `batch_layout` is the ORDER the read emitted its pixels in: one
    `(col_off, row_off, width, height)` tuple per batch the scan produced, in the order it
    produced them, each batch's own pixels then following row-major within it. It is NOT
    derivable from the four extent fields above. A large read is broken into batches to
    bound the read buffer, and a batch covers a sub-rectangle of a block row rather than a
    full-width strip, so "row *i* of the frame" means a different pixel depending on how
    the read happened to be chunked -- and how it is chunked depends on the BAND COUNT (a
    wider read gets fewer blocks per batch for the same memory budget), not only on the
    ground the read covers. Two reads of the same window over the same grid can therefore
    emit the same pixels in two different orders, which is why this is a field and not a
    computation: it is the only thing that can tell those two reads apart, and combining
    them positionally without noticing would silently pair up unrelated pixels. See
    `execution/frame.py`'s `assert_same_pixel_grid`, which compares it.

    Frozen, so `==` comes for free and comparing two windows is a plain comparison rather
    than a field-by-field check. No band-wide lowering may return a window differing from
    the one it was handed -- see `frame.assert_window_unchanged`.
    """

    source: object
    col_off: int
    row_off: int
    width: int
    height: int
    transform: Affine
    crs: CRS
    decimation: int = 1
    batch_layout: tuple[tuple[int, int, int, int], ...] = ()
