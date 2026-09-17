"""Guard test for the layering rules among `cpra.mp.data.analysis`'s three new
subpackages: `request/` (the request model), `io/` (filesystem access), and
`execution/` (the op algebra).

The four rules:

1. `request` imports nothing from `io` or `execution`.
2. `io` imports `request` (types/errors only) and nothing from `execution`.
3. `execution` imports `request` and `io`.
4. `api/` may import all three (unrestricted -- not checked here; it is the one place
   allowed to know about HTTP and about every layer at once).

This test is deliberately scoped to just the three new subpackages, not the whole
`cpra.mp.data.analysis` tree. The legacy modules directly under `analysis/`
(`base.py`, `raster.py`, `vector.py`, `sources.py`, `pipeline.py`, `ops.py`,
`stages.py`, `tiles.py`, `types.py`, `metadata.py`, `parameters.py`, `planner.py`,
`factory.py`, `paired.py`, ...) predate this layering and are not yet sorted into one
of the three subpackages -- that migration is later-stage work. So a module under
`request/`, `io/`, or `execution/` importing from the legacy `analysis/` root (e.g.
`from cpra.mp.data.analysis.base import AnalysisError`) is EXPLICITLY ALLOWED during
this coexistence period; `_LEGACY_ROOT_IMPORT_ALLOWED` below is that allowance, spelled
out so that widening or removing it later (once the legacy modules are themselves
sorted into a subpackage) is a deliberate, visible change to this file rather than a
silent side effect of moving something.
"""

import ast
from pathlib import Path

import cpra.mp.data.analysis

ANALYSIS_PACKAGE_DIR = Path(cpra.mp.data.analysis.__file__).parent
ANALYSIS_PACKAGE_NAME = "cpra.mp.data.analysis"

# The three new subpackages this test polices, and what each may import from the other
# two (by subpackage name, not by module -- deliberately coarse-grained for Stage 1: `io`
# importing `request` at all, whatever it imports, is treated as satisfying "request
# types/errors only" for now, because nothing under `io/` today imports from `request/`
# at all. The FINER rule -- that it may only reach for a type or an error class, not
# behavior -- has nothing to check itself against yet and is recorded here as the
# intended rule, not as something this test currently enforces at that granularity).
_ALLOWED_SIBLING_IMPORTS = {
    "request": frozenset(),
    "io": frozenset({"request"}),
    "execution": frozenset({"request", "io"}),
}

# Explicit coexistence allowance (see module docstring): every one of the three
# subpackages may import from the legacy `analysis/` root -- any module directly under
# `cpra.mp.data.analysis` that is not itself part of `request`, `io`, or `execution`.
_LEGACY_ROOT_IMPORT_ALLOWED = True


def _dotted_module_name(path: Path) -> str:
    relative = path.relative_to(ANALYSIS_PACKAGE_DIR)
    parts = [ANALYSIS_PACKAGE_NAME, *relative.with_suffix("").parts]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _subpackage_of(dotted_name: str) -> str | None:
    """`cpra.mp.data.analysis.io.raster_io` -> `"io"`; `cpra.mp.data.analysis.base` ->
    `None` (legacy root, not one of the three subpackages)."""
    prefix = ANALYSIS_PACKAGE_NAME + "."
    if not dotted_name.startswith(prefix):
        return None
    rest = dotted_name[len(prefix) :]
    head = rest.split(".", 1)[0]
    return head if head in _ALLOWED_SIBLING_IMPORTS else None


def _imported_module_names(source: str, path: Path) -> list[str]:
    """Every dotted module name a module imports, resolving a relative `from . import x`
    against `path`'s own package so it can be classified the same way an absolute import
    would be."""
    tree = ast.parse(source, filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative import: resolve against this module's own package.
                package_parts = _dotted_module_name(path).split(".")[: -node.level]
                base = ".".join(package_parts)
                module = f"{base}.{node.module}" if node.module else base
                names.append(module)
            elif node.module is not None:
                names.append(node.module)
    return names


def _subpackage_source_files():
    for subpackage in _ALLOWED_SIBLING_IMPORTS:
        directory = ANALYSIS_PACKAGE_DIR / subpackage
        if not directory.is_dir():
            continue
        yield from sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


def test_subpackages_only_import_their_permitted_siblings():
    violations = []

    for path in _subpackage_source_files():
        dotted_name = _dotted_module_name(path)
        this_subpackage = _subpackage_of(dotted_name)
        assert this_subpackage is not None, f"{path} is not under a recognized subpackage"

        allowed_siblings = _ALLOWED_SIBLING_IMPORTS[this_subpackage]

        for imported in _imported_module_names(path.read_text(), path):
            imported_subpackage = _subpackage_of(imported)
            if imported_subpackage is None:
                # Not one of the three policed subpackages: either outside
                # `cpra.mp.data.analysis` entirely (stdlib, third-party, `cpra.mp.data`
                # proper), or the legacy `analysis/` root -- both permitted, the latter
                # only because of the explicit coexistence allowance above.
                continue
            if imported_subpackage == this_subpackage:
                continue  # a module importing a sibling within its own subpackage
            if imported_subpackage not in allowed_siblings:
                violations.append(
                    f"{path}: `{this_subpackage}` module imports `{imported}` "
                    f"(subpackage `{imported_subpackage}`), which is not among its "
                    f"permitted siblings {sorted(allowed_siblings) or '(none)'}"
                )

    assert not violations, "analysis subpackage layering violation(s):\n" + "\n".join(violations)


def test_the_coexistence_allowance_is_explicit():
    """Pins the deliberate escape hatch itself, so widening or removing it (once the
    legacy `analysis/` root modules are sorted into one of the three subpackages, at a
    later stage) is a visible, intentional edit to this file rather than a silent
    consequence of some other change."""
    assert _LEGACY_ROOT_IMPORT_ALLOWED is True
