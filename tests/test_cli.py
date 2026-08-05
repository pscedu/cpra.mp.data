"""Tests for the `cpra.mp.data.cli` Click group.

The group currently carries no commands -- `convert` was removed once the MP2023 dump
format went out of use -- but it remains the registration point that command modules
attach to, so it is kept working.
"""

from click.testing import CliRunner

from cpra.mp.data.cli import cli


def test_group_runs_and_reports_usage():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_group_has_no_commands():
    assert cli.commands == {}


def test_unknown_command_is_rejected():
    result = CliRunner().invoke(cli, ["convert", "some.parquet"])
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

    assert cli.commands == {}
