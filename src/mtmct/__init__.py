"""MTMCT PoC: multi-target multi-camera person tracking."""

__version__ = "0.1.0"


def main() -> None:
    from mtmct.cli import app

    app()
