"""Maintain the data tree itself (as ``catalog`` maintains the catalog side-table).

Commands:

\b
  compact  merge data files, and the matching catalog files, into as few files as
           possible, in place

``compact`` covers the whole data tree and catalog by default. ``--variable``,
``--time-unit`` and ``--grid`` narrow it to the matching datasets; ``--model-group-id``
with ``--scenario-id`` narrows it to the datasets one run appears in, and within those to
the run's own files where they have directories of their own (COGs, daily Parquet).
Annual/decadal Parquet holds every run in one directory, so there that directory is
compacted whole, other runs included. What is already compact is left alone. The logic is
``cpra.mp.data.compaction``, which ``backup --compact`` also runs before archiving.

``--base-path``/``--catalog`` default to the configured ``BASE_PATH``/``CATALOG_PATH``
(``cpra.mp.data.config``), resolved at call time.

Usage:

\b
    python -m cpra.mp.data.cli data compact --dry-run
    python -m cpra.mp.data.cli data compact --variable elev --time-unit annual
    python -m cpra.mp.data.cli data compact --model-group-id 690 --scenario-id 7
"""

import time

import click

from cpra.mp.data import compaction
from cpra.mp.data.backup import inventory_run, run_name
from cpra.mp.data.cli import cli
from cpra.mp.data.compaction import find_datasets
from cpra.mp.data.config import get_base_path, get_catalog_path


@cli.group()
def data():
    """Maintain the data tree."""


@data.command()
@click.option("--variable", default=None, help="Only datasets of this variable.")
@click.option("--time-unit", default=None, help="Only datasets of this time unit.")
@click.option("--grid", default=None, help="Only datasets on this grid.")
# Also read from the environment, so a Kubernetes Job can be started from one suspended
# CronJob template with just the run set (kubectl set env MODEL_GROUP_ID=... SCENARIO_ID=...).
@click.option(
    "--model-group-id", type=int, default=None, envvar="MODEL_GROUP_ID", show_envvar=True, help="Only this run's files."
)
@click.option(
    "--scenario-id",
    type=int,
    default=None,
    envvar="SCENARIO_ID",
    show_envvar=True,
    help="Required with --model-group-id.",
)
@click.option("--base-path", default=None, help="Data root (default: configured BASE_PATH).")
@click.option("--catalog", "catalog_dir", default=None, help="Catalog root (default: configured CATALOG_PATH).")
@click.option("--dry-run", is_flag=True, help="Report what would be compacted; change nothing.")
def compact(variable, time_unit, grid, model_group_id, scenario_id, base_path, catalog_dir, dry_run):
    """Compact data and catalog files in place: everything, or the datasets (and run) selected."""
    base = get_base_path(base_path)
    catalog = get_catalog_path(catalog_dir)
    if not base.is_dir():
        raise click.UsageError(f"--base-path does not exist: {base}")
    if (model_group_id is None) != (scenario_id is None):
        raise click.UsageError("--model-group-id and --scenario-id go together")

    filters = {"variable": variable, "time_unit": time_unit, "grid": grid}
    selected = set(find_datasets(base, catalog, **filters))
    described = ", ".join(f"{name}={value}" for name, value in filters.items() if value is not None)
    if model_group_id is None:
        datasets = sorted(selected)
        scope = described or "everything"
    else:
        name = run_name(model_group_id, scenario_id)
        inventory = inventory_run(base, catalog, model_group_id, scenario_id)
        if not inventory.datasets:
            raise click.ClickException(f"no data or catalog rows found for {name} under {base} / {catalog}")
        for warning in inventory.warnings:
            click.echo(f"  WARNING: {warning}", err=True)
        datasets = [dataset for dataset in inventory.datasets if dataset in selected]
        scope = ", ".join(filter(None, [name, described]))
    if not datasets:
        raise click.ClickException(f"no datasets match {scope} under {base} / {catalog}")

    t0 = time.time()
    click.echo(f"{'would compact' if dry_run else 'compacting'} {scope} ({len(datasets)} dataset(s)):")
    compaction.compact(
        base,
        catalog,
        datasets,
        model_group_id=model_group_id,
        scenario_id=scenario_id,
        dry_run=dry_run,
        echo=click.echo,
    )
    if dry_run:
        click.echo("\nnothing changed (--dry-run)")
    else:
        click.echo(f"\nDONE: compacted {scope} in {time.time() - t0:.1f}s")
