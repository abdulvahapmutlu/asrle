from __future__ import annotations

import typer

from asrle.cli.commands.analyze import analyze_cmd
from asrle.cli.commands.flamegraph import flamegraph_cmd
from asrle.cli.commands.list_backends import list_backends_cmd
from asrle.cli.commands.run_dataset import run_dataset_cmd
from asrle.cli.commands.validate_backend import validate_backend_cmd
from asrle.logging import setup_logging

app = typer.Typer(add_completion=False, help="ASR-LE: Latency + Alignment + Error Attribution Engine")


@app.callback()
def _root():
    setup_logging()


app.command("analyze")(analyze_cmd)
app.command("list-backends")(list_backends_cmd)
app.command("run-dataset")(run_dataset_cmd)
app.command("flamegraph")(flamegraph_cmd)
app.command("validate-backend")(validate_backend_cmd)


def main() -> None:
    app()
