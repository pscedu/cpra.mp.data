"""Everything that touches the filesystem: raster/vector scans, coordinate resolution,
crosswalk reads, and output writers. No op algebra lives here -- see `execution`.

Shadows the standard library `io` module by name. Use absolute imports to disambiguate:
`import io` is always the stdlib module; `from cpra.mp.data.analysis import io` is always
this subpackage.
"""
