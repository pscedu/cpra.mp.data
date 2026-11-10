"""Back up and restore one model run -- a ``model_group_id``/``scenario_id`` pair -- as a
local tar.

Commands:

\b
  backup   write every file and catalog row belonging to one run to
           <output-dir>/g<id>-s<id>-<UTC timestamp>.tar
  restore  put an archive's files and catalog rows back in place; refuses if any of
           the run is still present

The archive logic lives in ``cpra.mp.data.backup``; see its docstring for what goes into
an archive and why.

``--base-path``/``--catalog`` default to the configured ``BASE_PATH``/``CATALOG_PATH``
(``cpra.mp.data.config``), resolved at call time.

Usage:

\b
    python -m cpra.mp.data.cli backup  --model-group-id 690 --scenario-id 7 --output-dir /backups --compact
    python -m cpra.mp.data.cli restore /backups/g690-s7-20261109T201500Z.tar --dry-run
"""

import time

import click

from cpra.mp.data.backup import ArchiveError, compact_run, inventory_run, restore_archive, run_name, write_archive
from cpra.mp.data.cli import cli
from cpra.mp.data.config import get_base_path, get_catalog_path


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _report(inventory) -> None:
    by_dataset = {}
    for member in inventory.members:
        by_dataset.setdefault(member.dataset, []).append(member)
    for dataset, members in sorted(by_dataset.items()):
        parts = []
        copies = [m for m in members if m.method == "copy"]
        if copies:
            parts.append(f"{len(copies)} file(s) {_fmt_bytes(sum(m.size for m in copies))}")
        for member in members:
            if member.method == "extract":
                label = "catalog" if member.kind == "catalog" else "extracted"
                parts.append(f"{member.rows:,} {label} row(s) from {len(member.sources)} file(s)")
        click.echo(f"  {dataset}: {', '.join(parts)}")
    for warning in inventory.warnings:
        click.echo(f"  WARNING: {warning}", err=True)


@cli.command()
@click.option("--model-group-id", type=int, required=True)
@click.option("--scenario-id", type=int, required=True)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory to write the archive to.",
)
@click.option("--base-path", default=None, help="Data root (default: configured BASE_PATH).")
@click.option("--catalog", "catalog_dir", default=None, help="Catalog root (default: configured CATALOG_PATH).")
@click.option(
    "--compact",
    is_flag=True,
    help="First compact the run's data and catalog files that aren't compact already. "
    "Annual/decadal Parquet holds every run in one directory, so that directory is "
    "compacted whole.",
)
@click.option("--dry-run", is_flag=True, help="Report what would be archived (and compacted); write nothing.")
def backup(model_group_id, scenario_id, output_dir, base_path, catalog_dir, compact, dry_run):
    """Archive one model run (model group + scenario) as a local tar."""
    base = get_base_path(base_path)
    catalog = get_catalog_path(catalog_dir)
    name = run_name(model_group_id, scenario_id)
    if not base.is_dir():
        raise click.UsageError(f"--base-path does not exist: {base}")

    t0 = time.time()
    inventory = inventory_run(base, catalog, model_group_id, scenario_id)
    if not inventory.members:
        raise click.ClickException(f"no data or catalog rows found for {name} under {base} / {catalog}")

    if compact:
        if dry_run:
            click.echo(f"would compact {len(inventory.datasets)} dataset(s) first (--dry-run)")
        else:
            click.echo(f"compacting {name}:")
            compact_run(base, catalog, model_group_id, scenario_id, inventory.datasets, echo=click.echo)
            # Compaction replaces files, so find them again.
            inventory = inventory_run(base, catalog, model_group_id, scenario_id)

    click.echo(f"{name}: {len(inventory.members)} file(s) in {len(inventory.datasets)} dataset(s)")
    _report(inventory)
    if dry_run:
        click.echo("\nnothing written (--dry-run)")
        return

    click.echo("\nwriting:")
    archive = write_archive(inventory, output_dir, base, catalog, compacted=compact, echo=click.echo)
    click.echo(f"\nDONE: {archive} ({_fmt_bytes(archive.stat().st_size)}) in {time.time() - t0:.1f}s")


@cli.command()
@click.argument("archive", type=click.Path(exists=True, dir_okay=False))
@click.option("--base-path", default=None, help="Data root (default: configured BASE_PATH).")
@click.option("--catalog", "catalog_dir", default=None, help="Catalog root (default: configured CATALOG_PATH).")
@click.option("--dry-run", is_flag=True, help="Check and verify the archive against the target; write nothing.")
def restore(archive, base_path, catalog_dir, dry_run):
    """Restore a model run from an archive written by `backup`."""
    base = get_base_path(base_path)
    catalog = get_catalog_path(catalog_dir)
    t0 = time.time()
    try:
        manifest = restore_archive(archive, base, catalog, dry_run=dry_run, echo=click.echo)
    except ArchiveError as exc:
        raise click.ClickException(str(exc)) from exc
    name = run_name(manifest["model_group_id"], manifest["scenario_id"])
    if not dry_run:
        click.echo(f"\nDONE: restored {name} ({len(manifest['files'])} file(s)) in {time.time() - t0:.1f}s")
