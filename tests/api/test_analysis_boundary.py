"""Guard tests for the `cpra.mp.data.analysis` package's structural boundaries.

`analysis/__init__.py`'s own module docstring states the FastAPI rule: nothing under this
package may import FastAPI. `request.errors.RequestError` and its subclasses stand in for
`HTTPException` throughout the package, and `api/boundary.py` is the one place that
translates them back into a real HTTP response. This test
is the permanent enforcement of that boundary -- it walks every `.py` module on disk
under `cpra/mp/data/analysis/` and asserts none of them contains an `import fastapi` /
`from fastapi import ...` statement.

Checking the AST of the source (rather than, say, `sys.modules` after importing the
package) is deliberate: FastAPI is a real dependency of this project and is always
importable in this environment, so the presence of `fastapi` in `sys.modules` proves
nothing about which module pulled it in. Parsing each file's own `import` statements is
the only way to pin the violation to a specific module.

This module also enforces that the package is cycle-free: every module under
`analysis/` must import cleanly as the FIRST thing a fresh interpreter imports, with no
help from import ORDER (see `test_every_module_is_independently_importable_first`'s
docstring for why order-dependence is worth a test of its own -- an alphabetizing pass
over the imports broke exactly that once already). `analysis/__init__.py` re-exports
nothing, so no module here needs an ordering trick; the test below is the permanent
proof.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import cpra.mp.data.analysis

ANALYSIS_PACKAGE_DIR = Path(cpra.mp.data.analysis.__file__).parent


def _fastapi_imports(source, path):
    """Yield a description of each top-level `fastapi` import found in `source`."""
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "fastapi" or alias.name.startswith("fastapi."):
                    yield f"{path}:{node.lineno}: import {alias.name}"
        elif isinstance(node, ast.ImportFrom) and (
            node.module is not None and (node.module == "fastapi" or node.module.startswith("fastapi."))
        ):
            yield f"{path}:{node.lineno}: from {node.module} import ..."


def test_no_module_under_analysis_imports_fastapi():
    module_paths = sorted(ANALYSIS_PACKAGE_DIR.rglob("*.py"))
    assert module_paths, "expected to find at least one module under cpra.mp.data.analysis"

    violations = []
    for path in module_paths:
        violations.extend(_fastapi_imports(path.read_text(), path))

    assert not violations, "fastapi imported under cpra.mp.data.analysis:\n" + "\n".join(violations)


def _dotted_module_name(path: Path) -> str:
    """`cpra/mp/data/analysis/io/raster_io.py` -> `cpra.mp.data.analysis.io.raster_io`;
    `cpra/mp/data/analysis/__init__.py` -> `cpra.mp.data.analysis` (the package itself).
    """
    relative = path.relative_to(ANALYSIS_PACKAGE_DIR)
    parts = ["cpra", "mp", "data", "analysis", *relative.with_suffix("").parts]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


ANALYSIS_MODULE_NAMES = sorted(
    {_dotted_module_name(p) for p in ANALYSIS_PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts}
)


@pytest.mark.parametrize("module_name", ANALYSIS_MODULE_NAMES)
def test_every_module_is_independently_importable_first(module_name):
    """Every module under `cpra.mp.data.analysis` must import cleanly as the FIRST thing
    a fresh interpreter does -- no other module of the package (least of all
    `__init__.py`) may already be sitting in `sys.modules` when it runs.

    This is the permanent regression test for the package's former `__init__.py` <->
    `planner.py` <-> `paired.py` import cycle, which used to be held together by
    documented import ORDER (a comment plus an `# isort: split` directive in
    `__init__.py`, and a mid-module import in `planner.py` placed after `paired.py`'s
    model dependencies were defined) rather than an acyclic module graph. That ordering
    broke for real once already, when a lint pass alphabetized the protected import
    block. Each module gets its own subprocess (rather than, say, clearing
    `sys.modules` and re-importing in-process) specifically so a PRIOR module's import in
    this same test session can never leave a partially-initialized package behind in
    `sys.modules` to mask the very failure this test exists to catch -- and so every
    module, not just `__init__.py`, is proven to have no hidden dependence on some other
    module of the package having already run first.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        f"import {module_name} failed when run first, alone, in a fresh interpreter:\n{result.stderr}"
    )
