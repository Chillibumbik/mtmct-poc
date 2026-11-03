"""Internal CLI for tests, agents and debugging. End users work through the web UI."""

from __future__ import annotations

import json
from typing import Annotated, Literal

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _print_progress(fraction: float, message: str) -> None:
    typer.echo(f"[{fraction:6.1%}] {message}")


@app.command()
def catalog() -> None:
    """List registered components."""
    from mtmct.registry import catalog as _catalog

    for category, items in _catalog().items():
        typer.echo(f"{category}: {', '.join(i.name for i in items) or '-'}")


@app.command()
def fetch(dataset: str) -> None:
    """Download raw files of a dataset."""
    from mtmct.config import paths
    from mtmct.data.base import load_adapters

    load_adapters()[dataset].fetch(paths().raw / dataset, _print_progress)


@app.command()
def prepare(dataset: str) -> None:
    """Convert raw files to the canonical layout."""
    from mtmct.config import paths
    from mtmct.data.base import load_adapters

    seqs = load_adapters()[dataset].prepare(paths().raw / dataset, paths().data, _print_progress)
    typer.echo(f"prepared: {[s.sequence for s in seqs]}")


@app.command()
def run(
    dataset: str,
    sequence: str,
    split: str = "all",
    mode: Literal["full", "oracle"] = "full",
    profile: str | None = None,
    set_: Annotated[list[str] | None, typer.Option("--set", help="dotted.key=json_value")] = None,
) -> None:
    """Run the pipeline on one sequence and print metrics."""
    from mtmct.config import load_config, paths
    from mtmct.data.base import load_sequence
    from mtmct.pipeline.run import run_pipeline

    overrides = {}
    for item in set_ or []:
        key, _, raw = item.partition("=")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    cfg = load_config(dataset, profile, overrides)
    seq = load_sequence(paths().data, dataset, sequence)
    result = run_pipeline(
        cfg,
        seq,
        mode=mode,
        split=split,
        out_dir=None,
        on_event=lambda e: typer.echo(f"[{e.stage or '-'}] {e.message}"),
        should_cancel=lambda: False,
    )
    typer.echo(json.dumps(result.metrics.get("cross_camera", {}), indent=2))
    typer.echo(f"run dir: {result.run_dir}")


@app.command()
def api(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Start the HTTP API."""
    import uvicorn

    uvicorn.run("mtmct.api.app:app", host=host, port=port, reload=reload)


@app.command()
def worker() -> None:
    """Start the background job worker."""
    from mtmct.worker.main import main as worker_main

    worker_main()
