#!/usr/bin/env python3
"""List model and dataset entries in the bundled VoxMatrix registry."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_evals.registry import registry


def print_models(console: Console) -> None:
    table = Table(title=f"Available {len(registry._model)} Models")
    table.add_column("Model", style="cyan")
    table.add_column("Class", style="green")
    table.add_column("Configured model", style="yellow")
    for name, spec in sorted(registry._model.items()):
        args = spec.get("args") or {}
        configured = args.get("model_name", args.get("path", ""))
        table.add_row(name, str(spec.get("cls", "")), str(configured))
    console.print(table)


def print_datasets(console: Console) -> None:
    table = Table(title=f"Available {len(registry._dataset)} Datasets")
    table.add_column("Dataset", style="cyan")
    table.add_column("Class", style="green")
    table.add_column("Default task", style="yellow")
    table.add_column("Source", style="magenta")
    for name, spec in sorted(registry._dataset.items()):
        args = spec.get("args") or {}
        source = args.get("name", args.get("f_name", args.get("local_path", "")))
        table.add_row(
            name,
            str(spec.get("cls", "")),
            str(args.get("default_task", "")),
            str(source),
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List available VoxMatrix models and datasets"
    )
    parser.add_argument("--models", action="store_true", help="show registered models")
    parser.add_argument(
        "--datasets", action="store_true", help="show registered datasets"
    )
    args = parser.parse_args()
    if not (args.models or args.datasets):
        args.models = True
        args.datasets = True

    console = Console()
    if args.models:
        print_models(console)
    if args.datasets:
        print_datasets(console)


if __name__ == "__main__":
    main()
