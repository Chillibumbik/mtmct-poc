from __future__ import annotations

import hashlib
import http.server
import threading
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import pytest

from mtmct import models
from mtmct.config import Paths


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(models, "paths", lambda: Paths(tmp_path))
    return tmp_path


@pytest.fixture
def server(tmp_path: Path) -> Iterator[str]:
    (tmp_path / "www").mkdir()
    (tmp_path / "www" / "w.bin").write_bytes(b"weights")
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path / "www"))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/w.bin"
    srv.shutdown()


def test_manifest_is_complete():
    assert {"yolox_s", "yolox_tiny", "osnet_ain_x1_0", "osnet_x0_25", "omz_0287"} <= set(models.MODELS)
    for spec in models.MODELS.values():
        assert len(spec.sha256) == 64 and spec.url.startswith("https://") and spec.licence
        assert all(len(sha) == 64 for _, sha in spec.companions)


def test_download_verifies_sha256_and_is_atomic(home: Path, server: str):
    dest = home / "m" / "w.bin"
    models._download(server, dest, hashlib.sha256(b"weights").hexdigest())
    assert dest.read_bytes() == b"weights"
    bad = home / "m" / "bad.bin"
    with pytest.raises(ValueError, match="sha256 mismatch"):
        models._download(server, bad, "0" * 64)
    assert not bad.exists() and not list(bad.parent.glob("*.part"))


def test_ensure_model_downloads_once_and_reuses(home: Path, server: str, monkeypatch: pytest.MonkeyPatch):
    spec = models.ModelSpec("toy", server, hashlib.sha256(b"weights").hexdigest(), "MIT", "toy.onnx")
    monkeypatch.setitem(models.MODELS, "toy", spec)
    path = models.ensure_model("toy")
    assert path == home / "models" / "toy.onnx" and path.read_bytes() == b"weights"
    monkeypatch.setitem(
        models.MODELS, "toy", models.ModelSpec("toy", "http://127.0.0.1:1/x", "0" * 64, "MIT", "toy.onnx")
    )
    assert models.ensure_model("toy") == path  # no new download


def test_ensure_model_export_kind_points_to_export_script(home: Path):
    with pytest.raises(FileNotFoundError, match=r"export_osnet\.py"):
        models.ensure_model("osnet_ain_x1_0")


def test_ensure_model_unknown_name():
    with pytest.raises(KeyError, match="unknown model"):
        models.ensure_model("nope")
