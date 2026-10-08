"""Tests for the `cpra.mp.data.cli` Click group.

`cli` itself defines no commands directly -- `convert` was removed once the MP2023 dump
format went out of use, and today the command modules are `cli/catalog.py`'s `catalog`
group and `cli/crosswalk.py`'s `crosswalk` command (vector and raster crosswalk
generation). `cli` remains the registration point they attach to: importing a command
module (e.g. `cpra.mp.data.cli.catalog`, see tests/test_cli_catalog.py) mutates the
shared `cli` object as a side effect of the `@cli.group()`/`@cli.command()` decoration at
import time. That means `cli.commands` reflects whichever command modules the test
session happened to import already, not just what this file imports -- see
`test_cli_group_starts_with_no_commands_of_its_own` below for how that's tested without
depending on import order.
"""

import subprocess
import sys

from click.testing import CliRunner

from cpra.mp.data.cli import cli


def test_group_runs_and_reports_usage():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cli_group_starts_with_no_commands_of_its_own():
    """Execute the module source into a throwaway module object to get a `cli` Group
    unaffected by whatever command modules other test files (e.g. test_cli_catalog.py)
    have already imported in this session -- asserting `cli.commands == {}` on the
    shared, possibly-already-mutated object would make the result depend on test
    collection/import order.

    Deliberately not `importlib.reload`: reload rebinds `cli` inside the cached
    `cpra.mp.data.cli` module, so every later `from cpra.mp.data.cli import cli` in the
    session would get a fresh, empty Group, while the already-imported command modules
    stay attached to the old one -- and re-importing them can't re-register, since they
    are themselves still cached. That trades this test's order-dependence for a
    session-wide one.
    """
    import importlib.util

    spec = importlib.util.find_spec("cpra.mp.data.cli")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.cli.commands == {}
    assert sys.modules["cpra.mp.data.cli"] is not module


def test_crosswalk_command_is_registered():
    # Imported for the side effect the test is about -- the docstring above explains why
    # relying on another test module having already imported it would be order-dependent.
    import cpra.mp.data.cli.crosswalk  # noqa: F401

    assert "crosswalk" in cli.commands

    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "crosswalk" in result.output


def test_unknown_command_is_rejected():
    result = CliRunner().invoke(cli, ["definitely-not-a-real-cpra-command"])
    assert result.exit_code != 0


def test_commands_can_be_registered_and_invoked():
    """The group's reason for existing: command modules attach to it.

    Invoking a subcommand is also the only path that runs the group's own callback --
    `--help` short-circuits before it.
    """
    import click

    @cli.command("noop")
    def noop():
        click.echo("ran")

    try:
        result = CliRunner().invoke(cli, ["noop"])
        assert result.exit_code == 0
        assert result.output.strip() == "ran"
    finally:
        del cli.commands["noop"]

    assert "noop" not in cli.commands


def test_module_entry_point_registers_its_command_modules():
    """`python -m cpra.mp.data.cli` runs `cli/__main__.py`, which imports `cli.catalog`
    and `cli.crosswalk` for the side effect of registering their subcommands before
    invoking `cli()`.

    This regressed once: `__main__.py` imported a `convert` module that had already been
    removed, so the entry point raised `ImportError` for anyone invoking it, and nothing
    caught it -- `cli/__main__.py` calls `cli()` at import time and is in coverage's
    `omit` list specifically so that importing it under pytest doesn't run the CLI, which
    also means importing it directly can't be used to test it. Running it out-of-process
    is the only way to exercise the real entry point.
    """
    result = subprocess.run(
        [sys.executable, "-m", "cpra.mp.data.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "catalog" in result.stdout
    assert "crosswalk" in result.stdout
